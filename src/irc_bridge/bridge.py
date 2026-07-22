from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field

from irc_bridge.backends.base import Backend, BackendError
from irc_bridge.config import Config, ConfigError, resolve_workspace
from irc_bridge.irc import IRCClient, IRCMessage
from irc_bridge.models import BackendEvent, ChannelBinding, Question, SessionSummary
from irc_bridge.paste import LitterboxClient, PasteError
from irc_bridge.state import StateStore
from irc_bridge.text import preview, safe_one_line


@dataclass(slots=True)
class PendingRequest:
    alias: str
    backend: str
    token: str | int
    questions: tuple[Question, ...] = ()


@dataclass(slots=True)
class ChannelRuntime:
    backend: str
    binding: ChannelBinding | None = None
    busy: bool = False
    active_turn: str | None = None
    active_flags: tuple[str, ...] = ()
    queue: deque[str] = field(default_factory=deque)
    session_choices: list[SessionSummary] = field(default_factory=list)
    approvals: dict[str, PendingRequest] = field(default_factory=dict)
    questions: dict[str, PendingRequest] = field(default_factory=dict)
    approval_counter: int = 0
    question_counter: int = 0
    tool_milestones: int = 0
    tool_events_suppressed: bool = False
    last_activity: str | None = None
    last_output: str | None = None
    last_reply: str | None = None


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
        self.paste = LitterboxClient(config.paste)
        self.channels = {
            channel: ChannelRuntime(backend=backend)
            for channel, backend in config.irc.channels.items()
        }
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False

    async def run(self) -> None:
        try:
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
            restore_messages = await self._restore_bindings()
            self._tasks = [
                asyncio.create_task(self._irc_loop(), name="bridge-irc"),
                *(
                    asyncio.create_task(self._backend_loop(backend), name=f"bridge-{name}-events")
                    for name, backend in self.backends.items()
                ),
            ]
            for channel, message in restore_messages:
                await self._say(channel, message)
            if self.config.bridge.notify_owner_on_start:
                channels = " and ".join(self.channels)
                await self.irc.send_privmsg(
                    self.config.bridge.owner_account,
                    f"irc-bridge is ready: {channels}. Send !help or !new <absolute-path>.",
                )
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

    async def _restore_bindings(self) -> list[tuple[str, str]]:
        messages: list[tuple[str, str]] = []
        bindings = await self.state.load()
        claimed: set[tuple[str, str]] = set()
        for channel, binding in bindings.items():
            runtime = self.channels.get(channel)
            if runtime is None or binding.backend != runtime.backend:
                await self.state.set(channel, None)
                continue
            key = (binding.backend, binding.session_id)
            if key in claimed:
                await self.state.set(channel, None)
                messages.append((channel, "duplicate saved binding was detached"))
                continue
            try:
                workspace = resolve_workspace(binding.cwd, self.config.bridge.allowed_roots)
                summary = await self.backends[binding.backend].attach_session(
                    binding.session_id, str(workspace)
                )
            except (ConfigError, BackendError) as exc:
                await self.state.set(channel, None)
                messages.append(
                    (channel, f"saved session could not be restored and was detached: {exc}")
                )
                continue
            runtime.binding = ChannelBinding(
                backend=binding.backend, session_id=summary.id, cwd=str(workspace)
            )
            claimed.add(key)
            messages.append(
                (
                    channel,
                    f"restored {binding.backend} session {self._short(summary.id)} in {workspace}",
                )
            )
        return messages

    async def _irc_loop(self) -> None:
        while True:
            message = await self.irc.recv()
            if message.account != self.config.bridge.owner_account:
                continue
            try:
                await self._handle_owner_message(message)
            except (BackendError, ConfigError, PasteError, ValueError) as exc:
                await self._say(message.channel, f"error: {safe_one_line(str(exc), 280)}")
            except Exception:
                await self._say(message.channel, "error: unexpected bridge failure")

    async def _backend_loop(self, backend: Backend) -> None:
        async for event in backend.events():
            try:
                await self._handle_backend_event(event)
            except Exception:
                channel = self._channel_for(event.backend, event.session_id)
                if channel:
                    await self._say(channel, "error: could not relay a backend event")

    async def _handle_owner_message(self, message: IRCMessage) -> None:
        text = message.text.strip()
        if not text:
            return
        if text.startswith("!"):
            command, _, argument = text.partition(" ")
            await self._command(message.channel, command.lower(), argument.strip())
            return
        runtime = self.channels[message.channel]
        if runtime.binding is None:
            raise ValueError("no session is attached; use !new <absolute-path>")
        if runtime.busy:
            if len(runtime.queue) >= self.config.bridge.queue_limit:
                raise ValueError(
                    f"queue is full ({self.config.bridge.queue_limit}); use !steer or !drop"
                )
            runtime.queue.append(text)
            await self._say(
                message.channel,
                f"queued as #{len(runtime.queue)}; use !queue, !drop, or !steer",
            )
            return
        await self._send_turn(message.channel, text)

    async def _command(self, channel: str, command: str, argument: str) -> None:
        runtime = self.channels[channel]
        if command == "!help":
            await self._say(
                channel,
                "commands: !new PATH | !sessions [PATH] | !attach N | !detach | !status | !last | "
                "!steer TEXT | !cancel | !approve [A1] | !deny [A1] | "
                "!answer Q1 ANSWER [ | ANSWER] | !reject [Q1] | !queue | "
                "!drop N|all | !paste | !paste-force",
            )
        elif command == "!new":
            await self._new_session(channel, argument)
        elif command == "!sessions":
            await self._list_sessions(channel, argument)
        elif command == "!attach":
            await self._attach(channel, argument)
        elif command == "!detach":
            if runtime.busy:
                raise ValueError("cancel the active turn before detaching")
            runtime.binding = None
            runtime.queue.clear()
            runtime.approvals.clear()
            runtime.questions.clear()
            await self.state.set(channel, None)
            await self._say(channel, "session detached")
        elif command == "!status":
            await self._status(channel)
        elif command == "!last":
            await self._last(channel)
        elif command == "!steer":
            if not argument:
                raise ValueError("usage: !steer <text>")
            binding = self._require_binding(runtime)
            if not runtime.busy:
                raise ValueError("there is no active turn to steer")
            await self.backends[runtime.backend].steer(
                binding.session_id, runtime.active_turn, argument
            )
            await self._say(channel, "steer delivered")
        elif command == "!cancel":
            binding = self._require_binding(runtime)
            if not runtime.busy:
                raise ValueError("there is no active turn to cancel")
            await self.backends[runtime.backend].cancel(binding.session_id, runtime.active_turn)
            await self._say(channel, "cancel requested")
        elif command in {"!approve", "!deny"}:
            await self._approval(channel, argument, command == "!approve")
        elif command == "!answer":
            await self._answer(channel, argument)
        elif command == "!reject":
            await self._reject(channel, argument)
        elif command == "!queue":
            if not runtime.queue:
                await self._say(channel, "queue is empty")
            else:
                entries = " | ".join(
                    f"{index}: {safe_one_line(item, 80)}"
                    for index, item in enumerate(runtime.queue, 1)
                )
                await self._say(channel, f"queue ({len(runtime.queue)}): {entries}")
        elif command == "!drop":
            await self._drop(channel, argument)
        elif command in {"!paste", "!paste-force"}:
            await self._paste(channel, force=command == "!paste-force")
        else:
            raise ValueError("unknown command; use !help")

    async def _new_session(self, channel: str, raw_path: str) -> None:
        runtime = self.channels[channel]
        if not raw_path:
            raise ValueError("usage: !new <absolute-path>")
        if runtime.busy:
            raise ValueError("cancel the active turn before changing sessions")
        workspace = resolve_workspace(raw_path, self.config.bridge.allowed_roots)
        summary = await self.backends[runtime.backend].create_session(str(workspace))
        await self._bind(channel, summary)
        await self._say(
            channel,
            f"created {runtime.backend} session {self._short(summary.id)} in {workspace}",
        )

    async def _list_sessions(self, channel: str, raw_path: str) -> None:
        runtime = self.channels[channel]
        if raw_path:
            workspace = resolve_workspace(raw_path, self.config.bridge.allowed_roots)
        elif runtime.binding:
            workspace = resolve_workspace(runtime.binding.cwd, self.config.bridge.allowed_roots)
        else:
            raise ValueError("provide an absolute path: !sessions /path/to/workspace")
        runtime.session_choices = await self.backends[runtime.backend].list_sessions(str(workspace))
        if not runtime.session_choices:
            await self._say(channel, f"no {runtime.backend} sessions found in {workspace}")
            return
        lines = [
            f"{index}. {self._short(item.id)} — {item.title}"
            for index, item in enumerate(runtime.session_choices, 1)
        ]
        await self._say(channel, "recent sessions:\n" + "\n".join(lines))

    async def _attach(self, channel: str, argument: str) -> None:
        runtime = self.channels[channel]
        if runtime.busy:
            raise ValueError("cancel the active turn before changing sessions")
        if not argument:
            raise ValueError("usage: !attach <number from !sessions>")
        try:
            index = int(argument)
        except ValueError as exc:
            raise ValueError("!attach takes the number shown by !sessions") from exc
        if index < 1 or index > len(runtime.session_choices):
            raise ValueError("that session number is not in the latest !sessions list")
        choice = runtime.session_choices[index - 1]
        workspace = resolve_workspace(choice.cwd, self.config.bridge.allowed_roots)
        summary = await self.backends[runtime.backend].attach_session(choice.id, str(workspace))
        await self._bind(channel, summary)
        await self._say(
            channel,
            f"attached {runtime.backend} session {self._short(summary.id)} in {workspace}",
        )

    async def _bind(self, channel: str, summary: SessionSummary) -> None:
        runtime = self.channels[channel]
        for other_channel, other in self.channels.items():
            if (
                other_channel != channel
                and other.binding
                and other.binding.backend == runtime.backend
                and other.binding.session_id == summary.id
            ):
                raise ValueError(f"that session is already attached to {other_channel}")
        binding = ChannelBinding(backend=runtime.backend, session_id=summary.id, cwd=summary.cwd)
        runtime.binding = binding
        runtime.busy = summary.busy
        runtime.active_turn = summary.active_turn_id
        runtime.active_flags = summary.active_flags
        runtime.queue.clear()
        runtime.approvals.clear()
        runtime.questions.clear()
        runtime.last_activity = (
            safe_one_line(summary.last_output, 180)
            if summary.last_output
            else ("working" if summary.busy else None)
        )
        runtime.last_output = summary.last_output
        runtime.last_reply = summary.last_reply
        await self.state.set(channel, binding)

    async def _status(self, channel: str) -> None:
        runtime = self.channels[channel]
        if runtime.binding is None:
            await self._say(channel, f"{runtime.backend}: detached; queue empty")
            return
        state = "busy" if runtime.busy else "idle"
        flags = ""
        if runtime.active_flags:
            labels = {
                "waitingOnApproval": "waiting on approval",
                "waitingOnUserInput": "waiting on user input",
            }
            flags = " (" + ", ".join(labels.get(flag, flag) for flag in runtime.active_flags) + ")"
        await self._say(
            channel,
            f"{runtime.backend} {state}{flags}; session {self._short(runtime.binding.session_id)}; "
            f"workspace {runtime.binding.cwd}; queued {len(runtime.queue)}; "
            f"approvals {len(runtime.approvals)}; questions {len(runtime.questions)}; "
            f"activity {runtime.last_activity or 'none observed'}",
        )

    async def _approval(self, channel: str, alias: str, allow: bool) -> None:
        runtime = self.channels[channel]
        request = self._select_request(runtime.approvals, alias, "approval")
        await self.backends[request.backend].resolve_approval(request.token, allow)
        runtime.approvals.pop(request.alias, None)
        await self._say(channel, f"{request.alias} {'approved once' if allow else 'denied'}")

    async def _answer(self, channel: str, argument: str) -> None:
        runtime = self.channels[channel]
        alias, separator, raw_answers = argument.partition(" ")
        if not separator:
            if len(runtime.questions) != 1:
                raise ValueError("usage: !answer Q1 <answer> [ | <answer>]")
            raw_answers = alias
            alias = ""
        request = self._select_request(runtime.questions, alias, "question")
        if any(question.secret for question in request.questions):
            raise ValueError("sensitive questions must be answered in the attached TUI")
        segments = [segment.strip() for segment in raw_answers.split("|")]
        if len(segments) != len(request.questions) or any(not item for item in segments):
            raise ValueError(
                f"provide {len(request.questions)} answer segment(s), separated with |"
            )
        answers = [
            self._parse_question_answer(question, segment)
            for question, segment in zip(request.questions, segments, strict=True)
        ]
        await self.backends[request.backend].resolve_question(
            request.token, request.questions, answers
        )
        runtime.questions.pop(request.alias, None)
        await self._say(channel, f"{request.alias} answered")

    async def _reject(self, channel: str, alias: str) -> None:
        runtime = self.channels[channel]
        request = self._select_request(runtime.questions, alias, "question")
        await self.backends[request.backend].resolve_question(
            request.token, request.questions, None
        )
        runtime.questions.pop(request.alias, None)
        await self._say(channel, f"{request.alias} rejected")

    @staticmethod
    def _parse_question_answer(question: Question, raw: str) -> list[str]:
        values = [item.strip() for item in raw.split(",")] if question.multiple else [raw.strip()]
        resolved: list[str] = []
        for value in values:
            if value.isdigit() and 1 <= int(value) <= len(question.options):
                resolved.append(question.options[int(value) - 1])
            elif value in question.options or question.custom:
                resolved.append(value)
            else:
                raise ValueError(f"{value!r} is not an available answer")
        return resolved

    @staticmethod
    def _select_request(
        requests: dict[str, PendingRequest], alias: str, kind: str
    ) -> PendingRequest:
        key = alias.upper()
        if not key:
            if len(requests) != 1:
                raise ValueError(f"specify a {kind} id")
            return next(iter(requests.values()))
        try:
            return requests[key]
        except KeyError as exc:
            raise ValueError(f"unknown or already resolved {kind} id {key}") from exc

    async def _drop(self, channel: str, argument: str) -> None:
        runtime = self.channels[channel]
        if not runtime.queue:
            raise ValueError("queue is empty")
        if argument.lower() == "all":
            count = len(runtime.queue)
            runtime.queue.clear()
            await self._say(channel, f"dropped all {count} queued messages")
            return
        try:
            index = int(argument)
        except ValueError as exc:
            raise ValueError("usage: !drop <queue-number>|all") from exc
        if index < 1 or index > len(runtime.queue):
            raise ValueError("queue number is out of range")
        items = list(runtime.queue)
        removed = items.pop(index - 1)
        runtime.queue = deque(items)
        await self._say(channel, f"dropped #{index}: {safe_one_line(removed, 100)}")

    async def _paste(self, channel: str, force: bool) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        text = runtime.last_reply or await self.backends[runtime.backend].get_last_reply(
            binding.session_id
        )
        if not text:
            raise ValueError("there is no assistant reply to paste")
        url = await self.paste.upload(text, force=force)
        await self._say(channel, f"1h public temporary paste: {url}")

    async def _last(self, channel: str) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        text = runtime.last_output or await self.backends[runtime.backend].get_last_reply(
            binding.session_id
        )
        if not text:
            raise ValueError("there is no assistant output in this session")
        body, _truncated = preview(
            text,
            self.config.bridge.summary_max_lines,
            self.config.bridge.summary_max_bytes,
        )
        await self._say(channel, body)

    async def _send_turn(self, channel: str, text: str) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        runtime.busy = True
        runtime.tool_milestones = 0
        runtime.tool_events_suppressed = False
        try:
            runtime.active_turn = await self.backends[runtime.backend].send_message(
                binding.session_id, text
            )
        except Exception:
            runtime.busy = False
            runtime.active_turn = None
            raise
        await self._say(channel, "turn started")

    async def _handle_backend_event(self, event: BackendEvent) -> None:
        if event.kind == "disconnected":
            for channel, runtime in self.channels.items():
                if runtime.backend == event.backend:
                    await self._say(channel, event.text or f"{event.backend} disconnected")
            return
        channel = self._channel_for(event.backend, event.session_id)
        if channel is None:
            return
        runtime = self.channels[channel]
        if event.kind == "status_changed":
            runtime.busy = bool(event.data.get("busy"))
            runtime.active_flags = tuple(str(flag) for flag in event.data.get("active_flags", ()))
            if runtime.active_flags:
                runtime.last_activity = {
                    "waitingOnApproval": "waiting on approval",
                    "waitingOnUserInput": "waiting on user input",
                }.get(runtime.active_flags[0], runtime.active_flags[0])
            elif runtime.busy and runtime.last_activity is None:
                runtime.last_activity = "working"
            return
        if event.kind == "turn_started":
            runtime.busy = True
            runtime.active_turn = event.turn_id or runtime.active_turn
            runtime.active_flags = ()
            runtime.last_activity = "working"
            runtime.tool_milestones = 0
            runtime.tool_events_suppressed = False
            return
        if event.kind == "progress":
            runtime.last_activity = safe_one_line(event.text, 180)
            runtime.last_output = event.text
            body, _truncated = preview(
                event.text,
                self.config.bridge.summary_max_lines,
                self.config.bridge.summary_max_bytes,
            )
            await self._say(channel, body)
            return
        if event.kind == "assistant":
            runtime.last_activity = safe_one_line(event.text, 180)
            runtime.last_output = event.text
            runtime.last_reply = event.text
            body, _truncated = preview(
                event.text,
                self.config.bridge.summary_max_lines,
                self.config.bridge.summary_max_bytes,
            )
            await self._say(channel, body)
            return
        if event.kind in {"tool_started", "tool_finished"}:
            if event.kind == "tool_started":
                status = "started"
            elif event.success is True:
                status = "finished successfully"
            elif event.success is False:
                status = "failed"
            else:
                status = "finished"
            runtime.last_activity = f"tool: {event.tool_kind} {status}"
            if runtime.tool_milestones < self.config.bridge.tool_milestone_limit:
                runtime.tool_milestones += 1
                await self._say(channel, f"tool: {event.tool_kind} {status}")
            elif not runtime.tool_events_suppressed:
                runtime.tool_events_suppressed = True
                await self._say(channel, "additional tool milestones suppressed for this turn")
            return
        if event.kind == "approval" and event.request_token is not None:
            runtime.approval_counter += 1
            alias = f"A{runtime.approval_counter}"
            runtime.approvals[alias] = PendingRequest(
                alias=alias, backend=event.backend, token=event.request_token
            )
            runtime.last_activity = "waiting on approval"
            await self._say(
                channel,
                f"{alias}: {event.text or 'approval needed'} — !approve {alias} or !deny {alias}",
            )
            return
        if event.kind == "question" and event.request_token is not None:
            if any(question.secret for question in event.questions):
                await self._say(
                    channel,
                    "sensitive input requested; answer it in the attached TUI (not IRC)",
                )
                return
            runtime.question_counter += 1
            alias = f"Q{runtime.question_counter}"
            runtime.questions[alias] = PendingRequest(
                alias=alias,
                backend=event.backend,
                token=event.request_token,
                questions=event.questions,
            )
            runtime.last_activity = "waiting on user input"
            details: list[str] = []
            for index, question in enumerate(event.questions, 1):
                options = ""
                if question.options:
                    options = (
                        " ["
                        + ", ".join(
                            f"{number}={label}" for number, label in enumerate(question.options, 1)
                        )
                        + "]"
                    )
                multiple = " (comma-select)" if question.multiple else ""
                details.append(f"{index}. {safe_one_line(question.prompt, 220)}{options}{multiple}")
            await self._say(
                channel,
                f"{alias}: " + " | ".join(details) + f" — !answer {alias} ... or !reject {alias}",
            )
            return
        if event.kind == "request_resolved" and event.request_token is not None:
            removed = self._remove_token(runtime, event.request_token)
            if removed:
                await self._say(channel, f"{removed} was resolved in another client")
            return
        if event.kind in {"turn_done", "turn_failed"}:
            runtime.busy = False
            runtime.active_turn = None
            runtime.active_flags = ()
            runtime.approvals.clear()
            runtime.questions.clear()
            if event.kind == "turn_failed":
                await self._say(channel, f"turn failed: {event.text or 'backend error'}")
            else:
                await self._say(channel, f"done ({runtime.tool_milestones} tool milestones)")
            if runtime.queue:
                next_message = runtime.queue.popleft()
                await self._say(channel, f"starting queued message ({len(runtime.queue)} remain)")
                await self._send_turn(channel, next_message)

    @staticmethod
    def _remove_token(runtime: ChannelRuntime, token: str | int) -> str | None:
        for requests in (runtime.approvals, runtime.questions):
            for alias, request in list(requests.items()):
                if str(request.token) == str(token):
                    del requests[alias]
                    return alias
        return None

    def _channel_for(self, backend: str, session_id: str | None) -> str | None:
        if not session_id:
            return None
        for channel, runtime in self.channels.items():
            if (
                runtime.binding
                and runtime.binding.backend == backend
                and runtime.binding.session_id == session_id
            ):
                return channel
        return None

    async def _say(self, channel: str, text: str) -> None:
        await self.irc.send_privmsg(channel, text)

    @staticmethod
    def _require_binding(runtime: ChannelRuntime) -> ChannelBinding:
        if runtime.binding is None:
            raise ValueError("no session is attached; use !new <absolute-path>")
        return runtime.binding

    @staticmethod
    def _short(session_id: str) -> str:
        return session_id if len(session_id) <= 16 else f"{session_id[:12]}…"
