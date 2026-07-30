from __future__ import annotations

import asyncio
import contextlib
import re
import secrets
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from agentwire.backends.base import Backend, BackendError
from agentwire.config import Config, ConfigError, resolve_workspace
from agentwire.irc import IRCClient
from agentwire.models import BackendEvent, ChannelBinding, Question, SessionSummary
from agentwire.protocol import (
    PROTOCOL_TAG,
    Envelope,
    ProtocolError,
    Reassembler,
    TopicActivation,
    decode_envelope,
    new_envelope,
    parse_topic,
)
from agentwire.redaction import scan_secrets
from agentwire.state import QueuedPrompt, StateStore
from agentwire.text import clean_text, safe_one_line, truncate_utf8

MAX_CONTENT_BYTES = 64 * 1024
MAX_PREVIEW_BYTES = 4 * 1024
_SENSITIVE_QUESTION_RE = re.compile(
    r"\b(?:password|passphrase|secret|api[ _-]?key|access[ _-]?token|"
    r"private[ _-]?key|credential)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PendingRequest:
    token: str | int
    kind: str
    session_id: str | None = None
    questions: tuple[Question, ...] = ()
    redacted: bool = False


@dataclass(slots=True)
class ChannelRuntime:
    backend: str
    activation: TopicActivation | None = None
    binding: ChannelBinding | None = None
    busy: bool = False
    active_turn: str | None = None
    settings: dict[str, Any] = field(
        default_factory=lambda: {"delivery": "queue", "approvalReviewer": "manual"}
    )
    requests: dict[str, PendingRequest] = field(default_factory=dict)
    observed_sessions: set[str] = field(default_factory=set)


class Bridge:
    def __init__(
        self,
        config: Config,
        irc: IRCClient,
        backends: dict[str, Backend],
    ) -> None:
        self.config = config
        self.irc = irc
        self.backends = backends
        self.state = StateStore(config.bridge.state_file)
        self.channels = {
            channel: ChannelRuntime(backend=backend)
            for channel, backend in config.irc.channels.items()
        }
        self.instance = str(uuid.uuid4())
        self.epoch = secrets.token_urlsafe(24)
        self._reassemblers = {channel: Reassembler() for channel in self.channels}
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False

    async def run(self) -> None:
        try:
            await self.state.initialize()
            await asyncio.gather(
                self.irc.start(), *(backend.start() for backend in self.backends.values())
            )
            await asyncio.gather(
                self.irc.wait_ready(self.config.stack.startup_timeout),
                *(
                    backend.wait_ready(self.config.stack.startup_timeout)
                    for backend in self.backends.values()
                ),
            )
            await self._restore_bindings()
            self._tasks = [
                asyncio.create_task(self._irc_loop(), name="bridge-irc"),
                *(
                    asyncio.create_task(self._backend_loop(backend), name=f"bridge-{name}-events")
                    for name, backend in self.backends.items()
                ),
            ]
            await asyncio.gather(*self._tasks)
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        for task in self._tasks:
            if task is not current:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await asyncio.gather(
            self.irc.close(),
            *(backend.close() for backend in self.backends.values()),
            return_exceptions=True,
        )

    async def _restore_bindings(self) -> None:
        for channel, binding in (await self.state.load()).items():
            runtime = self.channels.get(channel)
            if runtime is None or binding.backend != runtime.backend:
                await self.state.set(channel, None)
                continue
            try:
                workspace = resolve_workspace(binding.cwd, self.config.bridge.allowed_roots)
                summary = await self.backends[binding.backend].attach_session(
                    binding.session_id, str(workspace)
                )
            except (ConfigError, BackendError):
                await self.state.set(channel, None)
                continue
            runtime.binding = ChannelBinding(binding.backend, summary.id, summary.cwd)
            runtime.observed_sessions.add(summary.id)
            runtime.busy = summary.busy
            runtime.active_turn = summary.active_turn_id

    async def _irc_loop(self) -> None:
        while True:
            message = await self.irc.recv()
            try:
                if message.command in {"TOPIC", "332"}:
                    await self._handle_topic(message.channel, message.text)
                    continue
                runtime = self.channels[message.channel]
                if runtime.activation is None:
                    continue
                if message.account != runtime.activation.account:
                    continue
                value = message.tags.get(PROTOCOL_TAG)
                if not isinstance(value, str):
                    continue
                if "draft/playback" in message.tags or "znc.in/playback" in message.tags:
                    continue
                envelope = self._reassemblers[message.channel].add(value)
                if envelope is not None:
                    await self._handle_action(message.channel, envelope)
            except (BackendError, ConfigError, ProtocolError, ValueError) as exc:
                await self._emit_failure(message.channel, None, str(exc))
            except Exception:
                await self._emit_failure(message.channel, None, "unexpected bridge failure")

    async def _handle_topic(self, channel: str, topic: str) -> None:
        runtime = self.channels[channel]
        # A topic change suspends the channel until the complete new marker validates.
        runtime.activation = None
        activation = parse_topic(topic)
        if activation is None:
            return
        if activation.account != self.config.bridge.owner_account:
            raise ProtocolError("topic account does not match the configured owner account")
        if activation.backend != runtime.backend:
            raise ProtocolError("topic backend does not match the configured channel backend")
        runtime.activation = activation
        if runtime.binding is not None:
            summary = await self.backends[runtime.backend].attach_session(
                runtime.binding.session_id, runtime.binding.cwd
            )
            runtime.binding = ChannelBinding(runtime.backend, summary.id, summary.cwd)
            runtime.busy = summary.busy
            runtime.active_turn = summary.active_turn_id
            await self.state.set(channel, runtime.binding)
        await self._emit_hello(channel)
        await self._emit_snapshot(channel)
        if runtime.binding is not None and not runtime.busy:
            await self._drain_queue(channel)

    async def _emit_hello(self, channel: str, reply: str | None = None) -> None:
        runtime = self.channels[channel]
        await self._emit(
            channel,
            "agent.hello",
            reply=reply,
            data={
                "protocol": "agentwire-irc-v1",
                "backend": runtime.backend,
                "epoch": self.epoch,
                "capabilities": sorted(
                    {
                        "sync",
                        "workspaces",
                        "sessions",
                        "history",
                        "settings",
                        "turns",
                        "steering",
                        "queues",
                        "requests",
                    }
                ),
                "actions": sorted(
                    {
                        "sync.request",
                        "workspace.list.request",
                        "session.list.request",
                        "history.request",
                        "session.create",
                        "session.attach",
                        "session.detach",
                        "settings.update",
                        "turn.prompt",
                        "turn.steer",
                        "turn.cancel",
                        "queue.edit",
                        "queue.move",
                        "queue.delete",
                        "queue.clear",
                        "request.respond",
                        "request.skip",
                    }
                ),
                "limits": {
                    "contentBytes": MAX_CONTENT_BYTES,
                    "queueItems": self.config.bridge.queue_limit,
                    "historyEvents": 200,
                    "historyBytes": 512 * 1024,
                    "historyDays": 30,
                },
                "settings": (
                    ["model", "effort", "collaboration", "delivery", "approvalReviewer"]
                    if runtime.backend == "codex"
                    else ["delivery"]
                ),
            },
        )

    async def _handle_action(self, channel: str, action: Envelope) -> None:
        if action.message_type != "action":
            return
        if action.history:
            await self._emit_failure(channel, action.id, "historic actions cannot be executed")
            return
        if not action.device:
            await self._emit_failure(channel, action.id, "actions require device")
            return
        if action.kind != "sync.request" and action.epoch != self.epoch:
            await self._emit_failure(channel, action.id, "stale or missing live epoch")
            return
        duplicate = await self.state.claim_action(action)
        if duplicate is not None:
            await self._emit(
                channel,
                f"action.{duplicate}" if duplicate != "accepted" else "action.accepted",
                reply=action.id,
                data={"duplicate": True},
            )
            return
        await self._emit(channel, "action.accepted", reply=action.id)
        try:
            await self._dispatch_action(channel, action)
        except (BackendError, ConfigError, ProtocolError, ValueError) as exc:
            detail = safe_one_line(str(exc), 1000)
            await self.state.finish_action(action.id, "failed", detail)
            await self._emit(channel, "action.failed", reply=action.id, data={"message": detail})
            return
        except Exception:
            detail = "backend outcome is unknown"
            await self.state.finish_action(action.id, "uncertain", detail)
            await self._emit(channel, "action.uncertain", reply=action.id, data={"message": detail})
            return
        await self.state.finish_action(action.id, "succeeded")
        await self._emit(channel, "action.succeeded", reply=action.id)

    async def _dispatch_action(self, channel: str, action: Envelope) -> None:
        handlers = {
            "sync.request": self._action_sync,
            "workspace.list.request": self._action_workspaces,
            "session.list.request": self._action_sessions,
            "history.request": self._action_history,
            "session.create": self._action_create,
            "session.attach": self._action_attach,
            "session.detach": self._action_detach,
            "settings.update": self._action_settings,
            "turn.prompt": self._action_prompt,
            "turn.steer": self._action_steer,
            "turn.cancel": self._action_cancel,
            "queue.edit": self._action_queue_edit,
            "queue.move": self._action_queue_move,
            "queue.delete": self._action_queue_delete,
            "queue.clear": self._action_queue_clear,
            "request.respond": self._action_request_respond,
            "request.skip": self._action_request_skip,
        }
        handler = handlers.get(action.kind)
        if handler is None:
            raise ProtocolError(f"action is not advertised by this Agentwire: {action.kind}")
        await handler(channel, action)

    async def _action_sync(self, channel: str, action: Envelope) -> None:
        await self._emit_hello(channel, reply=action.id)
        await self._emit_snapshot(channel, reply=action.id)

    async def _action_workspaces(self, channel: str, action: Envelope) -> None:
        items = [
            {"path": str(root), "name": root.name} for root in self.config.bridge.allowed_roots
        ]
        await self._emit(
            channel,
            "workspace.page",
            reply=action.id,
            data={"items": items, "next": None},
        )

    async def _action_sessions(self, channel: str, action: Envelope) -> None:
        cwd_value = action.data.get("cwd")
        backend = self.backends[self.channels[channel].backend]
        if cwd_value:
            cwd = str(resolve_workspace(str(cwd_value), self.config.bridge.allowed_roots))
            sessions = await backend.list_sessions(cwd)
        else:
            sessions = await backend.list_running_sessions()
        allowed = [item for item in sessions if self._workspace_allowed(item.cwd)]
        await self._emit(
            channel,
            "session.page",
            reply=action.id,
            data={"items": [self._session_data(item) for item in allowed[:100]], "next": None},
        )

    async def _action_history(self, channel: str, action: Envelope) -> None:
        before = action.data.get("beforeAt")
        if before is not None and not isinstance(before, int):
            raise ProtocolError("beforeAt must be an integer")
        limit = action.data.get("limit", 200)
        if not isinstance(limit, int):
            raise ProtocolError("limit must be an integer")
        payloads = await self.state.history(channel, before, limit)
        page_id = str(uuid.uuid4())
        await self._emit(
            channel,
            "history.begin",
            reply=action.id,
            data={"page": page_id, "count": len(payloads)},
            journal=False,
        )
        for payload in payloads:
            historic = replace(decode_envelope(payload), history=True, reply=action.id)
            await self.irc.send_protocol(channel, historic)
        await self._emit(
            channel,
            "history.end",
            reply=action.id,
            data={"page": page_id, "count": len(payloads)},
            journal=False,
        )

    async def _action_create(self, channel: str, action: Envelope) -> None:
        cwd = str(
            resolve_workspace(self._data_string(action, "cwd"), self.config.bridge.allowed_roots)
        )
        runtime = self.channels[channel]
        summary = await self.backends[runtime.backend].create_session(cwd)
        await self._set_binding(channel, summary)

    async def _action_attach(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        session_id = action.session_id or self._data_string(action, "sid")
        cwd_value = action.data.get("cwd")
        cwd = (
            str(resolve_workspace(str(cwd_value), self.config.bridge.allowed_roots))
            if cwd_value
            else None
        )
        summary = await self.backends[runtime.backend].attach_session(session_id, cwd)
        if not self._workspace_allowed(summary.cwd):
            raise ConfigError("session workspace is outside allowed roots")
        await self._set_binding(channel, summary)

    async def _action_detach(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        previous = runtime.binding
        runtime.binding = None
        runtime.busy = False
        runtime.active_turn = None
        await self.state.set(channel, None)
        await self._emit(
            channel,
            "binding.changed",
            data={"previousSid": previous.session_id if previous else None, "sid": None},
            preview="Agent session detached",
        )

    async def _action_settings(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        allowed = {"model", "effort", "collaboration", "delivery", "approvalReviewer"}
        unknown = set(action.data) - allowed
        if unknown:
            raise ProtocolError(f"unsupported settings: {', '.join(sorted(unknown))}")
        if "delivery" in action.data and action.data["delivery"] not in {"queue", "steer"}:
            raise ProtocolError("delivery must be queue or steer")
        if "approvalReviewer" in action.data and action.data["approvalReviewer"] not in {
            "manual",
            "auto_review",
        }:
            raise ProtocolError("approvalReviewer must be manual or auto_review")
        if "collaboration" in action.data and action.data["collaboration"] not in {
            "default",
            "plan",
        }:
            raise ProtocolError("collaboration must be default or plan")
        for key in {"model", "effort"} & action.data.keys():
            if not isinstance(action.data[key], str) or not action.data[key]:
                raise ProtocolError(f"{key} must be a non-empty string")
        binding = self._require_binding(runtime)
        candidate = {**runtime.settings, **action.data}
        await self.backends[runtime.backend].configure_session(binding.session_id, candidate)
        runtime.settings = candidate
        await self._emit(
            channel,
            "session.snapshot",
            session_id=binding.session_id,
            data={"settings": runtime.settings},
        )

    async def _action_prompt(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        text = self._content(action)
        if runtime.busy:
            if runtime.settings.get("delivery") == "steer":
                await self.backends[runtime.backend].steer(
                    binding.session_id, runtime.active_turn, text
                )
                return
            item = await self.state.enqueue(
                action.item_id or str(uuid.uuid4()),
                channel,
                binding.session_id,
                text,
                self.config.bridge.queue_limit,
            )
            await self._emit_queue_item(channel, "queue.item.added", item)
            return
        runtime.active_turn = await self.backends[runtime.backend].send_message(
            binding.session_id, text
        )
        runtime.busy = True

    async def _action_steer(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        if not runtime.busy:
            raise ValueError("session has no active turn")
        await self.backends[runtime.backend].steer(
            binding.session_id, runtime.active_turn, self._content(action)
        )

    async def _action_cancel(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        await self.backends[runtime.backend].cancel(binding.session_id, runtime.active_turn)

    async def _action_queue_edit(self, channel: str, action: Envelope) -> None:
        item_id = self._item_id(action)
        await self._require_queue_item(channel, item_id)
        item = await self.state.edit_queue(item_id, self._content(action))
        if not item:
            raise ValueError("unknown queue item")
        await self._emit_queue_item(channel, "queue.item.updated", item, visible=True)

    async def _action_queue_move(self, channel: str, action: Envelope) -> None:
        position = action.data.get("position")
        if not isinstance(position, int) or position < 0:
            raise ProtocolError("position must be a non-negative integer")
        item_id = self._item_id(action)
        await self._require_queue_item(channel, item_id)
        items = await self.state.move_queue(item_id, position)
        for item in items:
            await self._emit_queue_item(channel, "queue.item.moved", item)

    async def _action_queue_delete(self, channel: str, action: Envelope) -> None:
        item_id = self._item_id(action)
        await self._require_queue_item(channel, item_id)
        item = await self.state.delete_queue(item_id)
        if not item:
            raise ValueError("unknown queue item")
        await self._emit_queue_item(channel, "queue.item.removed", item, visible=True)

    async def _action_queue_clear(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        count = await self.state.clear_queue(channel, binding.session_id)
        await self._emit(
            channel,
            "queue.snapshot",
            session_id=binding.session_id,
            data={"items": [], "removed": count},
        )

    async def _action_request_respond(self, channel: str, action: Envelope) -> None:
        request_id = action.request_id or self._data_string(action, "rid")
        pending = self.channels[channel].requests.get(request_id)
        if not pending:
            raise ValueError("unknown or resolved request")
        runtime = self.channels[channel]
        if pending.session_id and (
            runtime.binding is None or runtime.binding.session_id != pending.session_id
        ):
            raise ValueError("reattach the request's session before responding")
        backend = self.backends[self.channels[channel].backend]
        if pending.kind == "approval":
            allow = action.data.get("allow")
            if not isinstance(allow, bool):
                raise ProtocolError("approval response requires boolean allow")
            await backend.resolve_approval(pending.token, allow)
        else:
            answers = action.data.get("answers")
            if not isinstance(answers, list) or not all(isinstance(item, list) for item in answers):
                raise ProtocolError("question response requires an answers array")
            await backend.resolve_question(pending.token, pending.questions, answers)
        self.channels[channel].requests.pop(request_id, None)
        await self._emit(channel, "request.resolved", request_id=request_id)

    async def _action_request_skip(self, channel: str, action: Envelope) -> None:
        request_id = action.request_id or self._data_string(action, "rid")
        pending = self.channels[channel].requests.get(request_id)
        if not pending:
            raise ValueError("unknown or resolved request")
        runtime = self.channels[channel]
        if pending.session_id and (
            runtime.binding is None or runtime.binding.session_id != pending.session_id
        ):
            raise ValueError("reattach the request's session before skipping")
        backend = self.backends[self.channels[channel].backend]
        if pending.kind == "approval":
            await backend.resolve_approval(pending.token, False)
        else:
            await backend.resolve_question(pending.token, pending.questions, None)
        self.channels[channel].requests.pop(request_id, None)
        await self._emit(channel, "request.resolved", request_id=request_id)

    async def _backend_loop(self, backend: Backend) -> None:
        async for event in backend.events():
            channel = self._channel_for(event.backend, event.session_id)
            if channel:
                with contextlib.suppress(Exception):
                    await self._handle_backend_event(channel, event)

    async def _handle_backend_event(self, channel: str, event: BackendEvent) -> None:
        runtime = self.channels[channel]
        if runtime.activation is None:
            return
        selected = runtime.binding is not None and runtime.binding.session_id == event.session_id
        if not selected:
            if event.kind in {"approval", "question"}:
                await self._open_request(channel, event, inactive=True)
            return
        if event.kind == "turn_started":
            runtime.busy = True
            runtime.active_turn = event.turn_id
            await self._emit(
                channel, "turn.started", session_id=event.session_id, turn_id=event.turn_id
            )
        elif event.kind in {"turn_done", "turn_failed"}:
            runtime.busy = False
            runtime.active_turn = None
            kind = "turn.completed" if event.kind == "turn_done" else "turn.failed"
            await self._emit(
                channel,
                kind,
                session_id=event.session_id,
                turn_id=event.turn_id,
                data={"message": event.text} if event.text else {},
                preview=f"Agent turn failed: {event.text}" if event.kind == "turn_failed" else None,
            )
            await self._drain_queue(channel)
        elif event.kind == "status_changed":
            runtime.busy = bool(event.data.get("busy"))
            await self._emit(
                channel,
                "session.status",
                session_id=event.session_id,
                data={
                    "busy": runtime.busy,
                    "flags": list(event.data.get("active_flags") or ()),
                },
            )
        elif event.kind == "progress":
            await self._emit(
                channel,
                "plan.updated",
                session_id=event.session_id,
                turn_id=event.turn_id,
                data={"summary": self._safe_content(event.text, 32 * 1024)},
            )
        elif event.kind == "assistant":
            findings = scan_secrets(event.text)
            if findings:
                data = {"omitted": True, "reason": "high-confidence secret detected"}
                preview = "Agent reply omitted from IRC because it may contain a secret"
            else:
                content = self._safe_content(event.text, MAX_CONTENT_BYTES)
                data = {"content": content}
                preview = self._preview(content)
            await self._emit(
                channel,
                "assistant.completed",
                session_id=event.session_id,
                turn_id=event.turn_id,
                item_id=event.item_id,
                data=data,
                preview=preview,
            )
        elif event.kind in {"tool_started", "tool_finished"}:
            await self._emit(
                channel,
                "tool.started" if event.kind == "tool_started" else "tool.completed",
                session_id=event.session_id,
                turn_id=event.turn_id,
                item_id=event.item_id,
                data={"kind": event.tool_kind, "success": event.success},
            )
        elif event.kind in {"approval", "question"}:
            await self._open_request(channel, event)
        elif event.kind == "request_resolved":
            match = next(
                (
                    request_id
                    for request_id, pending in runtime.requests.items()
                    if pending.token == event.request_token
                ),
                None,
            )
            if match:
                runtime.requests.pop(match, None)
                await self._emit(channel, "request.resolved", request_id=match)

    async def _open_request(
        self, channel: str, event: BackendEvent, inactive: bool = False
    ) -> None:
        runtime = self.channels[channel]
        request_id = str(uuid.uuid4())
        secret_question = event.kind == "question" and any(
            q.secret or _SENSITIVE_QUESTION_RE.search(f"{q.header} {q.prompt}")
            for q in event.questions
        )
        pending = PendingRequest(
            token=event.request_token if event.request_token is not None else request_id,
            kind=event.kind,
            session_id=event.session_id,
            questions=event.questions,
            redacted=secret_question,
        )
        runtime.requests[request_id] = pending
        if secret_question:
            data = {"type": "question", "redacted": True, "canSkip": True}
            preview = "Sensitive agent question requires the attached TUI; it may be skipped here"
        elif event.kind == "approval":
            data = {
                "type": "approval",
                "summary": event.text or "approval required",
                "redacted": not bool(event.text),
                "choices": ["allow_once", "deny"],
            }
            preview = f"Agent approval required: {event.text or 'details redacted'}"
        else:
            data = {
                "type": "question",
                "questions": [
                    {
                        "id": question.id,
                        "header": question.header,
                        "prompt": question.prompt,
                        "options": list(question.options),
                        "multiple": question.multiple,
                        "custom": question.custom,
                    }
                    for question in event.questions
                ],
                "canSkip": True,
            }
            preview = "Agent input required"
        data["inactive"] = inactive
        if inactive:
            data["sid"] = event.session_id
            preview = "Agent request waiting in an inactive session; reattach to respond"
        await self._emit(
            channel,
            "request.opened",
            session_id=event.session_id,
            turn_id=event.turn_id,
            item_id=event.item_id,
            request_id=request_id,
            data=data,
            preview=preview,
        )

    async def _drain_queue(self, channel: str) -> None:
        runtime = self.channels[channel]
        if runtime.activation is None or runtime.binding is None or runtime.busy:
            return
        items = await self.state.list_queue(channel, runtime.binding.session_id)
        if not items:
            return
        item = items[0]
        await self.state.delete_queue(item.id)
        await self._emit_queue_item(channel, "queue.item.removed", item)
        runtime.active_turn = await self.backends[runtime.backend].send_message(
            runtime.binding.session_id, item.text
        )
        runtime.busy = True

    async def _set_binding(self, channel: str, summary: SessionSummary) -> None:
        runtime = self.channels[channel]
        previous = runtime.binding
        binding = ChannelBinding(runtime.backend, summary.id, summary.cwd)
        if previous:
            runtime.observed_sessions.add(previous.session_id)
        runtime.observed_sessions.add(summary.id)
        runtime.binding = binding
        runtime.busy = summary.busy
        runtime.active_turn = summary.active_turn_id
        await self.state.set(channel, binding)
        await self._emit(
            channel,
            "binding.changed",
            session_id=summary.id,
            data={
                "previousSid": previous.session_id if previous else None,
                "session": self._session_data(summary),
            },
            preview=f"Agent session switched to {summary.title or summary.id}",
        )
        await self._emit_snapshot(channel)

    async def _emit_snapshot(self, channel: str, reply: str | None = None) -> None:
        runtime = self.channels[channel]
        binding = runtime.binding
        queue = await self.state.list_queue(channel, binding.session_id) if binding else []
        await self._emit(
            channel,
            "channel.snapshot",
            reply=reply,
            session_id=binding.session_id if binding else None,
            data={
                "active": runtime.activation is not None,
                "backend": runtime.backend,
                "binding": ({"sid": binding.session_id, "cwd": binding.cwd} if binding else None),
                "busy": runtime.busy,
                "tid": runtime.active_turn,
                "settings": runtime.settings,
                "requests": list(runtime.requests),
                "queue": [self._queue_data(item) for item in queue],
            },
        )

    async def _emit_queue_item(
        self, channel: str, kind: str, item: QueuedPrompt, visible: bool = False
    ) -> None:
        await self._emit(
            channel,
            kind,
            session_id=item.session_id,
            item_id=item.id,
            data=self._queue_data(item),
            preview=(
                f"Queued prompt {'updated' if kind == 'queue.item.updated' else 'deleted'}"
                if visible
                else None
            ),
        )

    async def _emit_failure(self, channel: str, reply: str | None, message: str) -> None:
        runtime = self.channels.get(channel)
        if runtime is None or runtime.activation is None:
            return
        detail = safe_one_line(message, 1000)
        await self._emit(
            channel,
            "action.failed",
            reply=reply,
            data={"message": detail},
            preview=f"Agentwire action failed: {detail}",
        )

    async def _emit(
        self,
        channel: str,
        kind: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        item_id: str | None = None,
        request_id: str | None = None,
        reply: str | None = None,
        data: dict[str, Any] | None = None,
        preview: str | None = None,
        journal: bool = True,
    ) -> Envelope:
        envelope = new_envelope(
            kind,
            "event",
            self.instance,
            epoch=self.epoch,
            session_id=session_id,
            turn_id=turn_id,
            item_id=item_id,
            request_id=request_id,
            reply=reply,
            data=data or {},
        )
        if journal:
            await self.state.append_event(channel, envelope)
        await self.irc.send_protocol(channel, envelope, preview)
        return envelope

    def _channel_for(self, backend: str, session_id: str | None) -> str | None:
        if not session_id:
            return None
        return next(
            (
                channel
                for channel, runtime in self.channels.items()
                if runtime.backend == backend
                and (
                    (runtime.binding is not None and runtime.binding.session_id == session_id)
                    or session_id in runtime.observed_sessions
                )
            ),
            None,
        )

    def _workspace_allowed(self, cwd: str) -> bool:
        try:
            resolve_workspace(cwd, self.config.bridge.allowed_roots)
        except ConfigError:
            return False
        return True

    @staticmethod
    def _require_binding(runtime: ChannelRuntime) -> ChannelBinding:
        if runtime.binding is None:
            raise ValueError("no agent session is attached")
        return runtime.binding

    @staticmethod
    def _data_string(action: Envelope, key: str) -> str:
        value = action.data.get(key)
        if not isinstance(value, str) or not value:
            raise ProtocolError(f"data.{key} must be a non-empty string")
        return value

    @staticmethod
    def _item_id(action: Envelope) -> str:
        if not action.item_id:
            raise ProtocolError("action requires iid")
        return action.item_id

    @staticmethod
    def _safe_content(text: str, limit: int) -> str:
        cleaned = clean_text(text)
        if len(cleaned.encode("utf-8")) > limit:
            raise ProtocolError(f"content exceeds {limit} bytes")
        return cleaned

    def _content(self, action: Envelope) -> str:
        return self._safe_content(self._data_string(action, "content"), MAX_CONTENT_BYTES)

    async def _require_queue_item(self, channel: str, item_id: str) -> QueuedPrompt:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        item = next(
            (
                queued
                for queued in await self.state.list_queue(channel, binding.session_id)
                if queued.id == item_id
            ),
            None,
        )
        if item is None:
            raise ValueError("unknown queue item for the attached session")
        return item

    @staticmethod
    def _preview(text: str) -> str:
        return truncate_utf8(text, MAX_PREVIEW_BYTES)

    @staticmethod
    def _session_data(session: SessionSummary) -> dict[str, Any]:
        return {
            "sid": session.id,
            "cwd": session.cwd,
            "title": session.title,
            "updatedAt": session.updated_at,
            "busy": session.busy,
            "flags": list(session.active_flags),
        }

    @staticmethod
    def _queue_data(item: QueuedPrompt) -> dict[str, Any]:
        return {
            "iid": item.id,
            "sid": item.session_id,
            "position": item.position,
            "content": item.text,
            "createdAt": item.created_at,
        }
