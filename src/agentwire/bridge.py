from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agentwire.backends.base import Backend, BackendError
from agentwire.config import Config, ConfigError, resolve_workspace
from agentwire.irc import IRCClient, IRCMessage
from agentwire.models import BackendEvent, ChannelBinding, Question, SessionSummary
from agentwire.protocol import (
    HISTORY_EVENT_KINDS,
    PROTOCOL_TAG,
    Envelope,
    ProtocolError,
    Reassembler,
    TopicActivation,
    build_topic,
    decode_envelope,
    new_envelope,
    parse_topic,
    suggested_topic,
)
from agentwire.redaction import scan_secrets
from agentwire.state import QueuedPrompt, StateStore
from agentwire.text import clean_block, clean_text, safe_one_line, truncate_utf8

MAX_CONTENT_BYTES = 64 * 1024
MAX_PREVIEW_BYTES = 4 * 1024
MAX_REASON_BYTES = 200
# Packing history avoids one IRC command per tiny event while leaving room for
# outer envelope metadata beneath protocol's 128 KiB payload ceiling.
HISTORY_CHUNK_BYTES = 96 * 1024
# Shortest gap between two `session.status` events for the same unbound session.
OBSERVED_STATUS_SECONDS = 0.5
# A fleet of agents can churn faster than a phone can usefully render.
SUBAGENT_UPDATE_SECONDS = 0.5
# Read actions return reply-tagged data and are safe to repeat, so lifecycle
# acknowledgements would only add two IRC commands to every page request.
READ_ACTION_KINDS = frozenset(
    {"sync.request", "workspace.list.request", "session.list.request", "history.request"}
)
STATUS_ACTION_KIND = "action.status.request"
STATUS_QUEUE_LIMIT = 32
# Diagnostics name channels, accounts, kinds, and reasons. They never carry
# prompt text, tool output, tag values, or credentials.
LOGGER = logging.getLogger("agentwire.bridge")
_SESSION_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
)
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


@dataclass(slots=True, frozen=True)
class ActionDescriptor:
    """Ingress-authenticated work which may outlive the IRC receive turn."""

    envelope: Envelope
    requester_nick: str
    queued_at: float
    owner_account: str
    backend: str
    claimed: bool
    runtime_identity: object
    activation: TopicActivation


@dataclass(slots=True)
class ChannelRuntime:
    backend: str
    # A channel name can be removed and later reinstalled. Queued work belongs
    # to this exact runtime, never merely to a same-named successor.
    identity: object = field(default_factory=object, repr=False)
    activation: TopicActivation | None = None
    binding: ChannelBinding | None = None
    busy: bool = False
    active_turn: str | None = None
    closing_session: bool = False
    released_session_id: str | None = None
    settings: dict[str, Any] = field(
        default_factory=lambda: {"delivery": "queue", "approvalReviewer": "manual"}
    )
    session_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    requests: dict[str, PendingRequest] = field(default_factory=dict)
    # The reason last announced, so a repeated topic reply does not repeat it.
    suspended_reason: str | None = None
    observed_sessions: set[str] = field(default_factory=set)
    # Last `session.status` published per unbound sid: the payload and the
    # monotonic reading that coalesces repeats.
    observed_status: dict[str, tuple[dict[str, Any], float]] = field(default_factory=dict)
    # A changed payload that arrived inside the coalescing window, with the
    # flush task that will deliver it; newest payload wins, one task per sid.
    observed_status_pending: dict[str, tuple[dict[str, Any], asyncio.Task[None]]] = field(
        default_factory=dict
    )
    # `subagent.updated` is bound-session state, so one reading and one pending
    # flush per channel is enough; same newest-wins rule as observed status.
    subagents: tuple[dict[str, Any], float] | None = None
    subagents_pending: tuple[dict[str, Any], asyncio.Task[None]] | None = None
    attaching_session: str | None = None
    deferred_events: list[BackendEvent] = field(default_factory=list)


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
        self.channels: dict[str, ChannelRuntime] = {}
        self.instance = str(uuid.uuid4())
        self.epoch = secrets.token_urlsafe(24)
        self._reassemblers: dict[str, Reassembler] = {}
        self._action_queues: dict[str, asyncio.Queue[ActionDescriptor]] = {}
        self._action_workers: dict[str, asyncio.Task[None]] = {}
        self._status_queues: dict[str, asyncio.Queue[ActionDescriptor]] = {}
        self._status_workers: dict[str, asyncio.Task[None]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._managed_channels: set[str] = set()
        self._closing_channels: set[str] = set()
        self._closed = False
        for channel, backend in config.irc.channels.items():
            self._install_channel(channel, backend)
        # Configured channels whose topic reply has been evaluated at least once, and whether
        # the post-registration summary has already been announced for this process.
        self._topics_evaluated: set[str] = set()
        self._inert_announced = False

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
            for channel in self.channels:
                self._start_action_worker(channel)
                self._start_status_worker(channel)
            await asyncio.gather(*self._tasks)
        finally:
            await self.close()

    def _install_channel(self, channel: str, backend: str) -> ChannelRuntime:
        value = channel.lower()
        runtime = self.channels.get(value)
        if runtime is not None:
            return runtime
        runtime = ChannelRuntime(backend=backend)
        self.channels[value] = runtime
        self._reassemblers[value] = Reassembler()
        self._action_queues[value] = asyncio.Queue()
        self._status_queues[value] = asyncio.Queue(maxsize=STATUS_QUEUE_LIMIT)
        if self._tasks:
            self._start_action_worker(value)
            self._start_status_worker(value)
        return runtime

    def _start_action_worker(self, channel: str) -> None:
        if channel in self._action_workers:
            return
        task = asyncio.create_task(self._action_loop(channel), name=f"bridge-{channel}-actions")
        self._action_workers[channel] = task
        self._tasks.append(task)

    def _start_status_worker(self, channel: str) -> None:
        if channel in self._status_workers:
            return
        task = asyncio.create_task(self._status_loop(channel), name=f"bridge-{channel}-status")
        self._status_workers[channel] = task
        self._tasks.append(task)

    async def _remove_channel(self, channel: str) -> None:
        await self._fail_queued_mutations(channel, "channel was removed before execution")
        runtime = self.channels.pop(channel, None)
        if runtime is not None:
            for _payload, pending in runtime.observed_status_pending.values():
                pending.cancel()
            if runtime.subagents_pending is not None:
                runtime.subagents_pending[1].cancel()
        self._reassemblers.pop(channel, None)
        self._action_queues.pop(channel, None)
        self._status_queues.pop(channel, None)
        self._managed_channels.discard(channel)
        self._closing_channels.discard(channel)
        task = self._action_workers.pop(channel, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        task = self._status_workers.pop(channel, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        for runtime in self.channels.values():
            for _payload, pending in runtime.observed_status_pending.values():
                pending.cancel()
            runtime.observed_status_pending.clear()
            if runtime.subagents_pending is not None:
                runtime.subagents_pending[1].cancel()
                runtime.subagents_pending = None
        # Only descriptors still waiting in a queue are known not to have
        # reached a backend. The in-flight descriptor is left accepted so a
        # restart records it uncertain rather than claiming it failed.
        await asyncio.gather(
            *(
                self._fail_queued_mutations(channel, "bridge closed before execution")
                for channel in self.channels
            )
        )
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
        await self.state.close()

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
            value: str | None = None
            try:
                # 331 is "no topic is set". Treating it as an empty topic is
                # what turns an unset topic into a stated fact rather than an
                # absence indistinguishable from a reply that never arrived.
                if message.command in {"TOPIC", "331", "332"}:
                    if message.channel in self._managed_channels:
                        await self._handle_managed_topic(message)
                    else:
                        self._topics_evaluated.add(message.channel)
                        try:
                            await self._handle_topic(message.channel, message.text)
                        finally:
                            # Runs even when validation raised, so a channel that failed for one
                            # reason still counts toward the summary of what never came up.
                            await self._report_inert_channels()
                    continue
                value = message.tags.get(PROTOCOL_TAG)
                if not isinstance(value, str):
                    # Ordinary channel conversation. Reporting it would bury the
                    # protocol traffic that matters, so it stays silent.
                    continue
                runtime = self.channels[message.channel]
                if runtime.activation is None:
                    self._log_drop(message, "the channel has no valid activation topic")
                    continue
                if message.account != runtime.activation.account:
                    self._log_drop(
                        message,
                        f"sender account {message.account or '<none>'} is not the topic"
                        f" owner account {runtime.activation.account}",
                    )
                    continue
                if "draft/playback" in message.tags or "znc.in/playback" in message.tags:
                    self._log_drop(message, "it is history playback, which is never replayed")
                    continue
                envelope = self._reassemblers[message.channel].add(value)
                if envelope is not None:
                    await self._ingest_action(message.channel, envelope, message.nick)
                else:
                    LOGGER.debug(
                        "%s: holding a fragment until its message is complete", message.channel
                    )
            except ProtocolError as exc:
                await self._emit_failure(
                    message.channel, self._malformed_status_reply(value), str(exc)
                )
            except (BackendError, ConfigError, ValueError) as exc:
                await self._emit_failure(message.channel, None, str(exc))
            except Exception:
                await self._emit_failure(message.channel, None, "unexpected bridge failure")

    @staticmethod
    def _malformed_status_reply(value: str | None) -> str | None:
        """Recover only a validated lookup UUID from a rejected direct envelope."""
        if not value:
            return None
        try:
            raw = json.loads(value)
            action_id = raw.get("id") if isinstance(raw, dict) else None
            if (
                isinstance(raw, dict)
                and raw.get("t") == "action"
                and raw.get("k") == STATUS_ACTION_KIND
                and isinstance(action_id, str)
            ):
                uuid.UUID(action_id)
                return action_id
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass
        return None

    async def _action_loop(self, channel: str) -> None:
        queue = self._action_queues[channel]
        while True:
            descriptor = await queue.get()
            action = descriptor.envelope
            try:
                if not self._descriptor_is_trusted(channel, descriptor):
                    LOGGER.info("%s: failed queued action %s after suspension", channel, action.id)
                    if descriptor.claimed:
                        await self.state.finish_action(
                            action.id, "failed", "channel was suspended before execution"
                        )
                    await self._emit_failure(
                        channel, action.id, "channel was suspended before execution"
                    )
                    continue
                LOGGER.debug(
                    "%s: action %s (%s) waited %.1f ms",
                    channel,
                    action.kind,
                    action.id,
                    (time.monotonic() - descriptor.queued_at) * 1000,
                )
                await self._execute_action(channel, descriptor)
            except (BackendError, ConfigError, ProtocolError, ValueError) as exc:
                await self._emit_failure(channel, action.id, str(exc))
            except Exception:
                LOGGER.exception("%s: unexpected action worker failure", channel)
                await self._emit_failure(channel, action.id, "unexpected bridge failure")
            finally:
                queue.task_done()
            if channel not in self.channels:
                return

    async def _status_loop(self, channel: str) -> None:
        queue = self._status_queues[channel]
        while True:
            descriptor = await queue.get()
            action = descriptor.envelope
            try:
                if not self._descriptor_is_trusted(channel, descriptor):
                    await self._emit_failure(
                        channel, action.id, "channel was suspended before execution"
                    )
                    continue
                await self._action_status(channel, action, descriptor)
            except (BackendError, ConfigError, ProtocolError, ValueError) as exc:
                await self._emit_failure(channel, action.id, str(exc))
            except Exception:
                LOGGER.exception("%s: unexpected status worker failure", channel)
                await self._emit_failure(channel, action.id, "unexpected bridge failure")
            finally:
                queue.task_done()
            if channel not in self.channels:
                return

    def _is_own_message(self, message: IRCMessage) -> bool:
        return message.nick.lower() == self.config.irc.nickname.lower()

    def _log_drop(self, message: IRCMessage, reason: str) -> None:
        """Report a message that claimed to be protocol traffic and was dropped.

        Only the tag's presence, the sender's identity, and the reason are
        reported; the tag value is the payload and never reaches the log.
        """

        if self._is_own_message(message):
            # A server or test relay may still hand the bridge its own traffic.
            # That is expected, not a fault, so it must not warn.
            LOGGER.debug(
                "%s: ignored this bridge's own echoed %s", message.channel, message.command
            )
            return
        LOGGER.warning(
            "%s: dropped a protocol %s from %s (account %s) because %s",
            message.channel,
            message.command,
            message.nick or "<unknown>",
            message.account or "<none>",
            reason,
        )

    @property
    def _bridge_account(self) -> str:
        """The account the server confirmed, falling back to the nickname.

        A single-account deployment registers a nickname that is its account, so
        the fallback keeps that shape working before SASL has been confirmed.
        """

        return self.irc.account or self.config.irc.nickname.lower()

    async def _suspend(self, channel: str, reason: str, repair: str | None = None) -> None:
        runtime = self.channels[channel]
        await self._fail_queued_mutations(channel, "channel was suspended before execution")
        # The reason is bounded, but a repair is appended afterwards so it is
        # never truncated: a half-printed topic is not pasteable.
        detail = safe_one_line(reason, MAX_REASON_BYTES)
        if repair:
            detail = f"{detail}; set: {repair}"
        LOGGER.warning("%s: suspended: %s", channel, detail)
        if runtime.suspended_reason == detail:
            # A reconnect re-reads the same topic. Say it once per cause.
            return
        runtime.suspended_reason = detail
        # Suspension is otherwise invisible: no event can be published to a
        # channel that has no activation, so the humans in it must be told here.
        await self.irc.send_notice(channel, f"agentwire suspended: {detail}")

    async def _report_inert_channels(self) -> None:
        """Name every configured channel that answered a topic reply without activating.

        A channel named in this bridge's own configuration is meant to run an agent, so
        staying quiet about one that never came up is a defect rather than discretion. The
        per-topic rule above deliberately says nothing for a channel that was never active,
        because an ordinary topic on an unconfigured channel is not an event; that rule
        leaves a configured channel whose topic was never set, or whose topic Ergo discarded
        when an unregistered channel emptied, indistinguishable from one that is working.
        The bridge then silently drops every action a client sends it, which is only
        visible as a client that syncs forever.

        Announced once per process: the operator needs to be told, not nagged on every
        reconnect.
        """

        if self._inert_announced or not self._topics_evaluated >= set(self.config.irc.channels):
            return
        self._inert_announced = True
        inert = sorted(
            channel for channel, runtime in self.channels.items() if runtime.activation is None
        )
        if not inert:
            LOGGER.info("every configured channel activated")
            return
        LOGGER.warning(
            "configured channels that did not activate: %s; they will drop every action they "
            "receive until their topic is repaired",
            ", ".join(inert),
        )
        for channel in inert:
            runtime = self.channels[channel]
            if runtime.suspended_reason is not None:
                # Already announced with its own reason and pasteable repair line.
                continue
            await self._suspend(
                channel,
                "this channel is configured for an agent but has no activation topic",
                build_topic(
                    self.config.bridge.owner_account,
                    runtime.backend,
                    agent=self._bridge_account,
                ),
            )

    async def _handle_managed_topic(self, message: IRCMessage) -> None:
        channel = message.channel
        runtime = self.channels[channel]
        runtime.activation = None
        if channel in self._closing_channels:
            await self.irc.wait_ready()
            await self._finish_channel_close(channel)
            return
        if message.command == "331":
            topic = build_topic(
                self.config.bridge.owner_account,
                runtime.backend,
                agent=self._bridge_account,
            )
            try:
                await self.irc.wait_ready()
                await self.irc.set_private(channel)
                await self.irc.set_topic(channel, topic)
            except Exception as exc:
                LOGGER.warning(
                    "%s: private channel reconciliation failed (%s)", channel, type(exc).__name__
                )
                return
            await self._handle_topic(channel, topic)
            return
        await self._handle_topic(channel, message.text)

    async def _handle_topic(self, channel: str, topic: str) -> None:
        runtime = self.channels[channel]
        # A topic change suspends the channel until the complete new marker validates.
        was_active = runtime.activation is not None
        runtime.activation = None
        try:
            activation = parse_topic(topic)
            if activation is None:
                # No marker at all. Announce the transition out of an active
                # channel, because that is a state change operators must see,
                # but stay quiet for a channel that was already inactive: that
                # topic is ordinary channel life, and announcing it would post a
                # notice on every reconnect to a channel nobody has activated.
                if was_active:
                    await self._suspend(channel, "the activation topic was removed")
                else:
                    LOGGER.info("%s: no activation topic; the channel stays suspended", channel)
                return
            if activation.account != self.config.bridge.owner_account:
                raise ProtocolError(
                    f"topic account {activation.account} is not the configured owner account "
                    f"{self.config.bridge.owner_account}"
                )
            # Clients trust backend events only from the topic's agent account,
            # so a topic naming any other account would run the harness while
            # every event it publishes is rejected. The authority is the account
            # the server confirmed for this connection, not the configured
            # nickname: a bridge whose nickname and account differ would
            # otherwise activate into a channel where nothing it says is
            # trusted.
            if activation.agent != self._bridge_account:
                raise ProtocolError(
                    f"topic agent {activation.agent} is not this bridge's authenticated "
                    f"account {self._bridge_account}"
                )
            if activation.backend != runtime.backend:
                raise ProtocolError(
                    f"topic backend {activation.backend} is not the configured channel backend "
                    f"{runtime.backend}"
                )
        except ProtocolError as exc:
            # This topic was written to activate the channel, so the operator
            # gets the exact line that would work here. A topic predating the
            # required `agent` field is repaired by pasting one message.
            await self._suspend(
                channel,
                str(exc),
                suggested_topic(
                    topic,
                    account=self.config.bridge.owner_account,
                    agent=self._bridge_account,
                    backend=runtime.backend,
                ),
            )
            raise
        runtime.activation = activation
        runtime.suspended_reason = None
        LOGGER.info(
            "%s: activated for owner %s, agent %s, backend %s",
            channel,
            activation.account,
            activation.agent,
            activation.backend,
        )
        if runtime.binding is not None:
            summary = await self.backends[runtime.backend].attach_session(
                runtime.binding.session_id, runtime.binding.cwd
            )
            runtime.binding = ChannelBinding(runtime.backend, summary.id, summary.cwd)
            runtime.busy = summary.busy
            runtime.active_turn = summary.active_turn_id
            if channel not in self._managed_channels:
                await self.state.set(channel, runtime.binding)
        # Client sync is authoritative bootstrap. Publishing the same hello and
        # snapshot here made every activation pay for both messages twice.
        if runtime.binding is not None and not runtime.busy:
            await self._drain_queue(channel)

    async def _emit_hello(self, channel: str, reply: str | None = None) -> None:
        runtime = self.channels[channel]
        actions = {
            "sync.request",
            "workspace.list.request",
            "session.list.request",
            "history.request",
            STATUS_ACTION_KIND,
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
        if channel in self._managed_channels:
            actions.difference_update({"session.create", "session.attach", "session.detach"})
            actions.add("session.close")
        data: dict[str, Any] = {
            "protocol": "agentwire-irc-v1",
            "backend": runtime.backend,
            "epoch": self.epoch,
            "capabilities": sorted(
                {
                    "sync",
                    "workspaces",
                    "sessions",
                    "history",
                    "historyChunks",
                    "compressedFragments",
                    "settings",
                    "turns",
                    "steering",
                    "queues",
                    "requests",
                    "actionStatus",
                }
            ),
            "actions": sorted(actions),
            "limits": {
                "contentBytes": MAX_CONTENT_BYTES,
                "queueItems": self.config.bridge.queue_limit,
                "historyEvents": 200,
                "historyBytes": 512 * 1024,
                "historyChunkBytes": HISTORY_CHUNK_BYTES,
                "historyDays": 30,
            },
            "settings": (
                ["model", "effort", "collaboration", "delivery", "approvalReviewer"]
                if runtime.backend == "codex"
                else ["model", "effort", "delivery"]
                if runtime.backend in {"pi", "omp"}
                else ["delivery"]
            ),
        }
        try:
            setting_options = await self.backends[runtime.backend].setting_options()
        except BackendError:
            # Picker discovery is optional and must not prevent channel activation.
            setting_options = {}
        if setting_options:
            data["settingOptions"] = dict(setting_options)
        await self._emit(
            channel,
            "agent.hello",
            reply=reply,
            data=data,
        )

    async def _handle_action(
        self, channel: str, action: Envelope, requester_nick: str = ""
    ) -> None:
        """Synchronous test and embedding entry point.

        IRC production uses ``_ingest_action`` so a durable mutation is
        claimed before it waits behind earlier work.  Keeping this method
        immediate preserves the public test helper used by backend adapters.
        """
        descriptor = await self._prepare_action(channel, action, requester_nick)
        if descriptor is None:
            return
        if action.kind == STATUS_ACTION_KIND:
            await self._action_status(channel, action, descriptor)
        else:
            await self._execute_action(channel, descriptor)

    async def _ingest_action(
        self, channel: str, action: Envelope, requester_nick: str = ""
    ) -> None:
        """Claim mutations at authenticated ingress before serial dispatch."""
        descriptor = await self._prepare_action(channel, action, requester_nick)
        if descriptor is None:
            return
        if not self._descriptor_is_trusted(channel, descriptor):
            await self._fail_unqueued_action(descriptor, "channel changed before execution")
            return
        if action.kind == STATUS_ACTION_KIND:
            try:
                queue = self._status_queues.get(channel)
                if queue is None:
                    await self._fail_unqueued_action(descriptor, "channel changed before execution")
                    return
                queue.put_nowait(descriptor)
            except asyncio.QueueFull:
                await self._emit_failure(channel, action.id, "too many action status requests")
            return
        queue = self._action_queues.get(channel)
        if queue is None:
            await self._fail_unqueued_action(descriptor, "channel changed before execution")
            return
        await queue.put(descriptor)

    async def _prepare_action(
        self, channel: str, action: Envelope, requester_nick: str
    ) -> ActionDescriptor | None:
        if action.message_type != "action":
            # A relayed published event is never a command. Expected traffic,
            # so this stays below warning level.
            LOGGER.debug(
                "%s: ignored a %s %s, which is not an action",
                channel,
                action.message_type,
                action.kind,
            )
            return None
        if action.history:
            await self._emit_failure(channel, action.id, "historic actions cannot be executed")
            return None
        if not action.device:
            await self._emit_failure(channel, action.id, "actions require device")
            return None
        if action.kind != "sync.request" and action.epoch != self.epoch:
            await self._emit_failure(channel, action.id, "stale or missing live epoch")
            return None
        runtime = self.channels.get(channel)
        if runtime is None or runtime.activation is None:
            return None
        activation = runtime.activation
        runtime_identity = runtime.identity
        owner_account = activation.account
        backend = runtime.backend
        tracked = action.kind not in READ_ACTION_KINDS and action.kind != STATUS_ACTION_KIND
        if tracked:
            LOGGER.info("%s: journaling action %s (%s)", channel, action.kind, action.id)
            duplicate = await self.state.claim_action(action, channel, owner_account, backend)
            if duplicate is not None:
                scoped = await self.state.action_status(action.id, channel, owner_account, backend)
                if not self._captured_runtime_is_current(channel, runtime_identity, activation):
                    return None
                if scoped is None:
                    # UUIDs are global replay tombstones, including old rows
                    # without scope fields. Do not disclose their status to a
                    # different channel, owner, or backend.
                    await self._emit_failure(channel, action.id, "action UUID is already reserved")
                    return None
                await self._emit(
                    channel,
                    f"action.{duplicate}" if duplicate != "accepted" else "action.accepted",
                    reply=action.id,
                    data={"duplicate": True},
                )
                return None
            # Once SQLite accepted this UUID, a temporary IRC send failure must
            # not abandon it.  The serial worker will still execute it.
            try:
                await self._emit(channel, "action.accepted", reply=action.id)
            except Exception:
                LOGGER.warning("%s: could not publish accepted receipt for %s", channel, action.id)
            if not self._captured_runtime_is_current(channel, runtime_identity, activation):
                await self._fail_unqueued_action(
                    ActionDescriptor(
                        action,
                        requester_nick,
                        time.monotonic(),
                        owner_account,
                        backend,
                        True,
                        runtime_identity,
                        activation,
                    ),
                    "channel changed before execution",
                )
                return None
        return ActionDescriptor(
            envelope=action,
            requester_nick=requester_nick,
            queued_at=time.monotonic(),
            owner_account=owner_account,
            backend=backend,
            claimed=tracked,
            runtime_identity=runtime_identity,
            activation=activation,
        )

    def _descriptor_is_trusted(self, channel: str, descriptor: ActionDescriptor) -> bool:
        return self._captured_runtime_is_current(
            channel, descriptor.runtime_identity, descriptor.activation
        )

    def _captured_runtime_is_current(
        self, channel: str, runtime_identity: object, activation: TopicActivation
    ) -> bool:
        runtime = self.channels.get(channel)
        return bool(
            runtime is not None
            and runtime.identity is runtime_identity
            and runtime.activation is activation
            and activation.account == self.config.bridge.owner_account
            and activation.backend == runtime.backend
        )

    async def _fail_unqueued_action(self, descriptor: ActionDescriptor, detail: str) -> None:
        if descriptor.claimed:
            await self.state.finish_action(descriptor.envelope.id, "failed", detail)

    async def _execute_action(self, channel: str, descriptor: ActionDescriptor) -> None:
        action = descriptor.envelope
        started = time.monotonic()
        tracked = descriptor.claimed
        try:
            await self._dispatch_action(channel, action, descriptor.requester_nick)
        except (BackendError, ConfigError, ProtocolError, ValueError) as exc:
            detail = safe_one_line(str(exc), 1000)
            if tracked:
                await self.state.finish_action(action.id, "failed", detail)
            await self._emit(channel, "action.failed", reply=action.id, data={"message": detail})
            return
        except Exception:
            detail = "backend outcome is unknown"
            if tracked:
                await self.state.finish_action(action.id, "uncertain", detail)
                await self._emit(
                    channel, "action.uncertain", reply=action.id, data={"message": detail}
                )
            else:
                await self._emit(
                    channel, "action.failed", reply=action.id, data={"message": detail}
                )
            return
        if tracked:
            await self.state.finish_action(action.id, "succeeded")
            await self._emit(channel, "action.succeeded", reply=action.id)
        if channel in self._closing_channels:
            await self._finish_channel_close(channel)
        LOGGER.debug(
            "%s: action %s (%s) completed in %.1f ms",
            channel,
            action.kind,
            action.id,
            (time.monotonic() - started) * 1000,
        )

    async def _dispatch_action(
        self, channel: str, action: Envelope, requester_nick: str = ""
    ) -> None:
        handlers = {
            "sync.request": self._action_sync,
            "workspace.list.request": self._action_workspaces,
            "session.list.request": self._action_sessions,
            "history.request": self._action_history,
            "action.status.request": self._action_status_request,
            "session.create": self._action_create,
            "session.close": self._action_close,
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
        if action.kind == "session.create":
            await self._action_create(channel, action, requester_nick)
        else:
            await handler(channel, action)

    async def _action_sync(self, channel: str, action: Envelope) -> None:
        await self._emit_hello(channel, reply=action.id)
        await self._emit_snapshot(channel, reply=action.id)

    async def _action_status(
        self, channel: str, action: Envelope, descriptor: ActionDescriptor
    ) -> None:
        action_id = action.data.get("actionId")
        if not isinstance(action_id, str):
            raise ProtocolError("action.status.request requires data.actionId UUID")
        try:
            uuid.UUID(action_id)
        except ValueError as exc:
            raise ProtocolError("data.actionId must be a UUID") from exc
        requested_channel = action.data.get("channel")
        if "channel" in action.data and (
            not isinstance(requested_channel, str)
            or len(requested_channel) < 2
            or requested_channel[0] not in {"#", "&"}
            or any(value in requested_channel for value in " ,:\x00\r\n")
        ):
            raise ProtocolError("data.channel must be a syntactically valid IRC channel")
        target = (requested_channel or channel).lower()
        if target != channel and channel not in self.config.irc.channels:
            # A static configured channel is the authenticated control point
            # for its backend. Managed dedicated channels may have closed by
            # the time a client needs to recover a receipt.
            raise ProtocolError("cross-channel status queries require a configured control channel")
        receipt = await self.state.action_status(
            action_id, target, descriptor.owner_account, descriptor.backend
        )
        # A lookup can yield while the receiving channel loses its trust scope.
        if not self._descriptor_is_trusted(channel, descriptor):
            return
        data: dict[str, Any] = {"actionId": action_id, "status": "unknown"}
        if receipt is not None:
            data.update(
                {
                    "status": receipt.status,
                    "kind": receipt.kind,
                    "channel": receipt.channel,
                    "receivedAt": receipt.received_at,
                }
            )
            message = self._safe_receipt_detail(receipt.detail)
            if message:
                data["message"] = message
        await self._emit(channel, "action.status", reply=action.id, data=data)

    async def _action_status_request(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        activation = runtime.activation
        if activation is None:
            raise ProtocolError("channel is not active")
        await self._action_status(
            channel,
            action,
            ActionDescriptor(
                action,
                "",
                time.monotonic(),
                activation.account,
                runtime.backend,
                False,
                runtime.identity,
                activation,
            ),
        )

    @staticmethod
    def _safe_receipt_detail(detail: str) -> str | None:
        # Scan before truncation: a secret can begin beyond the visible prefix.
        if not detail or scan_secrets(detail):
            return None
        value = safe_one_line(detail, MAX_REASON_BYTES)
        return value or None

    async def _fail_queued_mutations(self, channel: str, detail: str) -> None:
        queue = self._action_queues.get(channel)
        if queue is None:
            return
        while True:
            try:
                descriptor = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if descriptor.claimed:
                    await self.state.finish_action(descriptor.envelope.id, "failed", detail)
            finally:
                queue.task_done()

    async def _action_workspaces(self, channel: str, action: Envelope) -> None:
        parent_value = action.data.get("parent")
        if parent_value is None:
            parent: Path | None = None
        elif isinstance(parent_value, str):
            parent = resolve_workspace(parent_value, self.config.bridge.allowed_roots)
        else:
            raise ProtocolError("workspace parent must be a string")
        backend = self.backends[self.channels[channel].backend]

        def build_items() -> list[dict[str, Any]]:
            # Directory listing, the per-child hasChildren probe, and session
            # counting are all filesystem work; one thread hop keeps every scan
            # off the event loop instead of only the counts.
            directories = (
                list(self.config.bridge.allowed_roots)
                if parent is None
                else self._workspace_children(parent)
            )
            items: list[dict[str, Any]] = []
            for directory in directories:
                item: dict[str, Any] = {
                    "path": str(directory),
                    "name": directory.name,
                    "hasChildren": bool(self._workspace_children(directory, limit=1)),
                }
                count = backend.count_sessions(str(directory))
                if count is not None:
                    item["sessionCount"] = count
                items.append(item)
            return items

        items = await asyncio.to_thread(build_items)
        await self._emit(
            channel,
            "workspace.page",
            reply=action.id,
            data={
                "parent": str(parent) if parent is not None else None,
                "items": items,
                "next": None,
            },
        )

    def _workspace_children(self, parent: Path, limit: int | None = None) -> list[Path]:
        children: list[Path] = []
        try:
            entries = sorted(parent.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            return children
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                resolved = entry.resolve(strict=True)
                allowed = any(
                    resolved == root or resolved.is_relative_to(root)
                    for root in self.config.bridge.allowed_roots
                )
                loops_to_ancestor = resolved == parent or parent.is_relative_to(resolved)
                if not allowed or loops_to_ancestor or not resolved.is_dir():
                    continue
            except OSError:
                continue
            if resolved not in children:
                children.append(resolved)
            if limit is not None and len(children) >= limit:
                break
        return children

    async def _action_sessions(self, channel: str, action: Envelope) -> None:
        cwd_value = action.data.get("cwd")
        scope_value = action.data.get("scope")
        if scope_value is not None and scope_value not in {"live", "workspace"}:
            raise ProtocolError("session scope must be live or workspace")
        if scope_value == "workspace" and cwd_value is None:
            raise ProtocolError("workspace session scope requires cwd")
        if scope_value == "live" and cwd_value is not None:
            raise ProtocolError("live session scope cannot include cwd")
        cursor_value = action.data.get("cursor")
        if cursor_value is None:
            offset = 0
        elif isinstance(cursor_value, str) and cursor_value.isdigit():
            offset = int(cursor_value)
        else:
            raise ProtocolError("session cursor must be a non-negative integer string")
        backend = self.backends[self.channels[channel].backend]
        if cwd_value is not None:
            if not isinstance(cwd_value, str) or not cwd_value:
                raise ProtocolError("session cwd must be a non-empty string")
            cwd = str(resolve_workspace(cwd_value, self.config.bridge.allowed_roots))
            sessions = await backend.list_sessions(cwd)
        else:
            cwd = None
            sessions = await backend.list_running_sessions()
        allowed = [item for item in sessions if self._workspace_allowed(item.cwd)]
        page = allowed[offset : offset + 100]
        next_cursor = str(offset + 100) if offset + 100 < len(allowed) else None
        await self._emit(
            channel,
            "session.page",
            reply=action.id,
            data={
                "scope": "workspace" if cwd is not None else "live",
                "cwd": cwd,
                "cursor": str(offset) if offset else None,
                "items": [self._session_data(item) for item in page],
                "next": next_cursor,
            },
        )

    async def _action_history(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime, action)
        data_sid = action.data.get("sid")
        if data_sid is not None and (not isinstance(data_sid, str) or not data_sid):
            raise ProtocolError("history sid must be a non-empty string")
        session_id = action.session_id or data_sid or binding.session_id
        if session_id != binding.session_id:
            raise ValueError("history targets a session that is no longer attached")
        cursor = action.data.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise ProtocolError("history cursor must be a string")
        before = action.data.get("beforeAt")
        if before is not None and not isinstance(before, int):
            raise ProtocolError("beforeAt must be an integer")
        limit = action.data.get("limit", 200)
        if not isinstance(limit, int):
            raise ProtocolError("limit must be an integer")
        if not 1 <= limit <= 200:
            raise ProtocolError("history limit must be between 1 and 200")
        backend_page = await self.backends[runtime.backend].list_history(
            session_id,
            cursor,
            limit,
        )
        page_id = str(uuid.uuid4())
        if backend_page is not None:
            history = [self._history_envelope(event, action.id) for event in backend_page.events]
            next_cursor = backend_page.next_cursor
        else:
            payloads = await self.state.history(channel, session_id, before, limit)
            history = [
                replace(
                    decode_envelope(payload),
                    history=True,
                    reply=action.id,
                    session_id=session_id,
                )
                for payload in payloads
            ]
            next_cursor = None
        chunks = self._history_chunks([event.to_dict() for event in history])
        page_data: dict[str, Any] = {
            "page": page_id,
            "count": len(history),
            "chunks": len(chunks),
            "cursor": cursor,
        }
        if backend_page is not None:
            page_data["next"] = next_cursor
        await self._emit(
            channel,
            "history.begin",
            session_id=session_id,
            reply=action.id,
            data=page_data,
            journal=False,
        )
        for index, chunk in enumerate(chunks):
            await self._emit(
                channel,
                "history.chunk",
                session_id=session_id,
                reply=action.id,
                data={
                    "page": page_id,
                    "index": index,
                    "events": chunk,
                },
                journal=False,
            )
        await self._emit(
            channel,
            "history.end",
            session_id=session_id,
            reply=action.id,
            data=page_data,
            journal=False,
        )

    @staticmethod
    def _history_chunks(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_bytes = 0
        for payload in history:
            # Sized on the same minified JSON the chunk envelope carries, so
            # each event is serialized once for packing rather than encoded a
            # second time just to be measured.
            size = (
                len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                + 1
            )
            if current and current_bytes + size > HISTORY_CHUNK_BYTES:
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(payload)
            current_bytes += size
        if current:
            chunks.append(current)
        return chunks

    async def _action_create(
        self, channel: str, action: Envelope, requester_nick: str = ""
    ) -> None:
        cwd = str(
            resolve_workspace(self._data_string(action, "cwd"), self.config.bridge.allowed_roots)
        )
        runtime = self.channels[channel]
        if channel in self._managed_channels:
            raise ValueError(f"managed {runtime.backend} channels cannot create another session")
        summary = await self.backends[runtime.backend].create_session(cwd)
        backend_config = (
            self.config.codex
            if runtime.backend == "codex"
            else self.config.pi
            if runtime.backend == "pi"
            else self.config.omp
            if runtime.backend == "omp"
            else None
        )
        dedicated = (
            backend_config is not None
            and backend_config.dedicated_channels
            and channel in self.config.irc.channels
        )
        if not dedicated:
            await self._set_binding(channel, summary)
            return
        if not requester_nick:
            await self.backends[runtime.backend].close_session(summary.id)
            raise ProtocolError(
                f"dedicated {runtime.backend} session creation requires requester nickname"
            )
        try:
            managed = self._managed_channel_name(summary.id, runtime.backend)
        except Exception:
            await self.backends[runtime.backend].close_session(summary.id)
            raise
        binding = ChannelBinding(runtime.backend, summary.id, summary.cwd)
        installed = False
        try:
            self.irc.register_channel(managed)
            self._managed_channels.add(managed)
            target = self._install_channel(managed, runtime.backend)
            installed = True
            target.binding = binding
            target.observed_sessions.add(summary.id)
            target.busy = summary.busy
            target.active_turn = summary.active_turn_id
            await self.irc.join_channel(managed)
            await self.irc.set_private(managed)
            topic = build_topic(
                self.config.bridge.owner_account,
                runtime.backend,
                agent=self._bridge_account,
            )
            await self.irc.set_topic(managed, topic)
            await self._handle_topic(managed, topic)
            await self.irc.invite(managed, requester_nick)
        except Exception:
            with contextlib.suppress(Exception):
                await self.backends[runtime.backend].close_session(summary.id)
            if installed:
                with contextlib.suppress(Exception):
                    await self.irc.set_topic(managed, "")
                with contextlib.suppress(Exception):
                    await self.irc.part_channel(managed)
                await self._remove_channel(managed)
            else:
                self.irc.unregister_channel(managed)
            raise

    def _managed_channel_name(self, session_id: str, backend: str) -> str:
        match = _SESSION_UUID_RE.search(session_id)
        if match is None:
            raise BackendError(f"{backend} returned a session id without a UUID")
        value = uuid.UUID(match.group()).hex
        for length in range(8, len(value) + 1, 4):
            channel = f"#{backend}-{value[:length]}"
            if channel not in self.channels:
                return channel
        raise BackendError(f"cannot allocate a unique {backend} session channel")

    async def _action_close(self, channel: str, action: Envelope) -> None:
        if channel not in self._managed_channels:
            raise ValueError("only a managed channel can close its session")
        runtime = self.channels[channel]
        if action.session_id is None:
            raise ProtocolError("session.close requires sid")
        binding = self._require_binding(runtime, action)
        if binding.backend != runtime.backend:
            raise ValueError(f"only a managed {runtime.backend} session can be closed")
        # Interrupting Codex emits turn completion while close is still pending.
        # Hold the queue until the session has been released successfully.
        runtime.closing_session = True
        try:
            await self.backends[binding.backend].close_session(binding.session_id)
        except Exception:
            runtime.closing_session = False
            # A completion may have arrived while closing held the queue.
            # Restore normal delivery if closing failed and the backend is usable.
            with contextlib.suppress(BackendError):
                await self._drain_queue(channel)
            raise
        finally:
            runtime.closing_session = False
        # Backend release has succeeded. Persistence and IRC cleanup can retry,
        # but must never restore a binding to a thread we no longer subscribe to.
        runtime.released_session_id = binding.session_id
        runtime.binding = None
        runtime.busy = False
        runtime.active_turn = None
        runtime.requests.clear()
        self._reset_subagents(runtime)
        self._closing_channels.add(channel)

    async def _finish_channel_close(self, channel: str) -> None:
        try:
            runtime = self.channels[channel]
            if runtime.released_session_id is not None:
                await self.state.set(channel, None)
                await self.state.clear_queue(channel, runtime.released_session_id)
                runtime.released_session_id = None
            await self.irc.flush()
            await self.irc.set_topic(channel, "")
            await self.irc.part_channel(channel)
        except Exception as exc:
            LOGGER.warning("%s: channel close deferred (%s)", channel, type(exc).__name__)
            return
        await self._remove_channel(channel)

    async def _action_attach(self, channel: str, action: Envelope) -> None:
        if channel in self._managed_channels:
            raise ValueError(
                f"managed {self.channels[channel].backend} channels cannot switch sessions"
            )
        runtime = self.channels[channel]
        session_id = action.session_id or self._data_string(action, "sid")
        cwd_value = action.data.get("cwd")
        cwd = (
            str(resolve_workspace(str(cwd_value), self.config.bridge.allowed_roots))
            if cwd_value
            else None
        )
        runtime.attaching_session = session_id
        runtime.deferred_events.clear()
        try:
            summary = await self.backends[runtime.backend].attach_session(session_id, cwd)
            if not self._workspace_allowed(summary.cwd):
                raise ConfigError("session workspace is outside allowed roots")
            await self._set_binding(channel, summary)
            deferred = list(runtime.deferred_events)
        except Exception:
            runtime.deferred_events.clear()
            raise
        finally:
            runtime.attaching_session = None
        runtime.deferred_events.clear()
        for event in deferred:
            await self._handle_backend_event(channel, event)

    async def _action_detach(self, channel: str, action: Envelope) -> None:
        if channel in self._managed_channels:
            raise ValueError(
                f"managed {self.channels[channel].backend} channels cannot detach their session"
            )
        runtime = self.channels[channel]
        previous = self._require_binding(runtime, action)
        runtime.binding = None
        self._reset_subagents(runtime)
        runtime.busy = False
        runtime.active_turn = None
        runtime.settings = {"delivery": "queue", "approvalReviewer": "manual"}
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
        binding = self._require_binding(runtime, action)
        candidate = {**runtime.settings, **action.data}
        await self.backends[runtime.backend].configure_session(binding.session_id, candidate)
        runtime.settings = candidate
        runtime.session_settings[binding.session_id] = dict(candidate)
        await self._emit(
            channel,
            "session.snapshot",
            session_id=binding.session_id,
            data={"settings": runtime.settings},
        )

    async def _action_prompt(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime, action)
        text = self._content(action)
        if runtime.busy:
            if runtime.settings.get("delivery") == "steer":
                await self.backends[runtime.backend].steer(
                    binding.session_id, runtime.active_turn, text
                )
                await self._emit(
                    channel,
                    "user.prompt",
                    session_id=binding.session_id,
                    turn_id=runtime.active_turn,
                    item_id=action.item_id or action.id,
                    data=self._safe_user_prompt(text),
                )
                return
            item = await self.state.enqueue(
                action.item_id or action.id,
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
        await self._emit(
            channel,
            "user.prompt",
            session_id=binding.session_id,
            turn_id=runtime.active_turn,
            item_id=action.item_id or action.id,
            data=self._safe_user_prompt(text),
        )

    async def _action_steer(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime, action)
        if not runtime.busy:
            raise ValueError("session has no active turn")
        text = self._content(action)
        await self.backends[runtime.backend].steer(binding.session_id, runtime.active_turn, text)
        await self._emit(
            channel,
            "user.prompt",
            session_id=binding.session_id,
            turn_id=runtime.active_turn,
            item_id=action.item_id or action.id,
            data=self._safe_user_prompt(text),
        )

    async def _action_cancel(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime, action)
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
        runtime = self.channels[channel]
        binding = self._require_binding(runtime, action)
        await self._emit(
            channel,
            "queue.snapshot",
            session_id=binding.session_id,
            data={"items": [self._queue_data(item) for item in items]},
        )

    async def _action_queue_delete(self, channel: str, action: Envelope) -> None:
        item_id = self._item_id(action)
        await self._require_queue_item(channel, item_id)
        item = await self.state.delete_queue(item_id)
        if not item:
            raise ValueError("unknown queue item")
        await self._emit_queue_item(channel, "queue.item.removed", item, visible=True)

    async def _action_queue_clear(self, channel: str, action: Envelope) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime, action)
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
            # A status describes one session's liveness rather than the bound
            # timeline, so it reaches every channel running that backend. Every
            # other kind stays with the channel that owns the session.
            if event.kind == "status_changed" and event.session_id:
                channels = [
                    channel
                    for channel, runtime in self.channels.items()
                    if runtime.backend == event.backend
                ]
            else:
                channels = [
                    channel
                    for channel in (self._channel_for(event.backend, event.session_id),)
                    if channel
                ]
            for channel in channels:
                with contextlib.suppress(Exception):
                    await self._handle_backend_event(channel, event)

    async def _handle_backend_event(self, channel: str, event: BackendEvent) -> None:
        runtime = self.channels[channel]
        if runtime.activation is None:
            return
        if runtime.attaching_session == event.session_id:
            runtime.deferred_events.append(event)
            return
        selected = runtime.binding is not None and runtime.binding.session_id == event.session_id
        if not selected:
            if event.kind in {"approval", "question"}:
                await self._open_request(channel, event, inactive=True)
            elif event.kind == "status_changed" and event.session_id:
                await self._emit_observed_status(channel, runtime, event)
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
        elif event.kind == "user_prompt":
            # Backends suppress echoes of IRC prompts, which the action path
            # already emitted. These prompts originated in another client.
            prompt = self._safe_user_prompt(event.text)
            content = prompt.get("content")
            await self._emit(
                channel,
                "user.prompt",
                session_id=event.session_id,
                turn_id=event.turn_id,
                item_id=event.item_id,
                data=prompt,
                # Unlike an owner prompt, the typed original is not in the
                # channel, so give the mirrored prompt a readable preview.
                preview=self._preview(content) if isinstance(content, str) else None,
            )
        elif event.kind == "progress":
            data: dict[str, Any] = {"summary": self._safe_content(event.text, 32 * 1024)}
            if event.data.get("plan") is True:
                data["plan"] = True
                data["running"] = bool(event.data.get("running"))
                status = event.data.get("status")
                if isinstance(status, str):
                    data["status"] = safe_one_line(status, 80)
                for key in ("completedSteps", "totalSteps"):
                    value = event.data.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        data[key] = value
            await self._emit(
                channel,
                "plan.updated",
                session_id=event.session_id,
                turn_id=event.turn_id,
                data=data,
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
                data=self._safe_tool_data(event),
            )
        elif event.kind == "subagent_update":
            await self._emit_subagents(channel, runtime, event)
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
        if (
            runtime.activation is None
            or runtime.binding is None
            or runtime.busy
            or runtime.closing_session
        ):
            return
        # Claim delivery before awaiting so completion and close recovery cannot
        # both dispatch the same queue head. Keep it until the backend accepts it.
        runtime.busy = True
        try:
            items = await self.state.list_queue(channel, runtime.binding.session_id)
            if not items:
                runtime.busy = False
                return
            item = items[0]
            runtime.active_turn = await self.backends[runtime.backend].send_message(
                runtime.binding.session_id, item.text
            )
        except BaseException:
            runtime.busy = False
            raise
        await self.state.delete_queue(item.id)
        await self._emit_queue_item(channel, "queue.item.removed", item)
        await self._emit(
            channel,
            "user.prompt",
            session_id=runtime.binding.session_id,
            turn_id=runtime.active_turn,
            item_id=item.id,
            data=self._safe_user_prompt(item.text),
        )

    async def _set_binding(self, channel: str, summary: SessionSummary) -> None:
        runtime = self.channels[channel]
        previous = runtime.binding
        binding = ChannelBinding(runtime.backend, summary.id, summary.cwd)
        if previous:
            runtime.observed_sessions.add(previous.session_id)
        runtime.observed_sessions.add(summary.id)
        runtime.binding = binding
        # Clients clear this list on `binding.changed`, so the coalescer must
        # forget its reading or an identical list for the new session is dropped.
        self._reset_subagents(runtime)
        runtime.settings = dict(
            runtime.session_settings.get(
                summary.id,
                {"delivery": "queue", "approvalReviewer": "manual"},
            )
        )
        runtime.busy = summary.busy
        runtime.active_turn = summary.active_turn_id
        if channel not in self._managed_channels:
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
        await self._emit(
            channel,
            "session.snapshot",
            session_id=summary.id,
            turn_id=summary.active_turn_id,
            data=self._session_snapshot_data(summary),
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
        journal: bool | None = None,
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
        if journal is not None:
            should_journal = journal
        elif kind not in HISTORY_EVENT_KINDS:
            should_journal = False
        else:
            # The journal only backs history for backends without their own
            # transcript pagination; writing it for the others is a per-event
            # database commit that is never read back.
            runtime = self.channels.get(channel)
            backend = self.backends.get(runtime.backend) if runtime else None
            should_journal = backend is None or not backend.has_authoritative_history
        if should_journal:
            await self.state.append_event(channel, envelope)
        await self.irc.send_protocol(channel, envelope, preview)
        return envelope

    async def _emit_observed_status(
        self, channel: str, runtime: ChannelRuntime, event: BackendEvent
    ) -> None:
        """Publish a status for a session this channel is not bound to.

        Feeds a client session drawer, so it carries liveness only and never
        touches the channel's binding, busy flag, or timeline.
        """
        sid = event.session_id or ""
        data: dict[str, Any] = {
            "busy": bool(event.data.get("busy")),
            "flags": [str(flag) for flag in event.data.get("active_flags") or ()],
        }
        cwd = event.data.get("cwd")
        if isinstance(cwd, str) and cwd:
            data["cwd"] = cwd
        tui = event.data.get("tui")
        if isinstance(tui, bool):
            data["tuiAttached"] = tui
        now = time.monotonic()
        last = runtime.observed_status.get(sid)
        pending = runtime.observed_status_pending.get(sid)
        # A drawer only needs the newest state, so compare against what the
        # client will see: the pending payload if one waits, else the last emit.
        visible = pending[0] if pending is not None else (last[0] if last is not None else None)
        if visible == data:
            return
        if pending is not None:
            if last is not None and last[0] == data:
                # The change cancelled itself out; the client is already right.
                pending[1].cancel()
                runtime.observed_status_pending.pop(sid, None)
            else:
                # Newest payload wins; the scheduled flush keeps its deadline.
                runtime.observed_status_pending[sid] = (data, pending[1])
            return
        if last is not None and now - last[1] < OBSERVED_STATUS_SECONDS:
            # Inside the window: hold the newest state and deliver it when the
            # window closes, so a short turn never strands a stale busy flag.
            delay = OBSERVED_STATUS_SECONDS - (now - last[1])
            task = asyncio.create_task(
                self._flush_observed_status(channel, runtime, sid, delay),
                name=f"observed-status-{channel}-{sid}",
            )
            runtime.observed_status_pending[sid] = (data, task)
            return
        runtime.observed_status[sid] = (data, now)
        await self._emit(channel, "session.status", session_id=sid, data=data)

    async def _flush_observed_status(
        self, channel: str, runtime: ChannelRuntime, sid: str, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        pending = runtime.observed_status_pending.pop(sid, None)
        if pending is None:
            return
        runtime.observed_status[sid] = (pending[0], time.monotonic())
        with contextlib.suppress(Exception):
            await self._emit(channel, "session.status", session_id=sid, data=pending[0])

    @staticmethod
    def _reset_subagents(runtime: ChannelRuntime) -> None:
        if runtime.subagents_pending is not None:
            runtime.subagents_pending[1].cancel()
            runtime.subagents_pending = None
        runtime.subagents = None

    async def _emit_subagents(
        self, channel: str, runtime: ChannelRuntime, event: BackendEvent
    ) -> None:
        """Publish the bound session's subagent list, coalesced to at most 1/s.

        The list replaces rather than merges, so only the newest one matters and
        an unchanged list is dropped. Mirrors `_emit_observed_status`, keyed by
        channel because this state only ever describes the bound session.
        """
        agents = event.data.get("agents")
        safe_agents = []
        for agent in agents if isinstance(agents, list) else []:
            if not isinstance(agent, dict):
                continue
            row = dict(agent)
            # Codex descriptions can contain delegated prompts and results.
            # Apply the same secret filter as ordinary prompts and tool output.
            for key in ("id", "type", "description"):
                value = row.get(key)
                if isinstance(value, str) and scan_secrets(value):
                    row[key] = "[omitted]"
            safe_agents.append(row)
        data: dict[str, Any] = {"agents": safe_agents}
        now = time.monotonic()
        last = runtime.subagents
        pending = runtime.subagents_pending
        # Compare against what the client will end up seeing, not what it last saw.
        visible = pending[0] if pending is not None else (last[0] if last is not None else None)
        if visible == data:
            return
        if pending is not None:
            if last is not None and last[0] == data:
                # The change cancelled itself out; the client is already right.
                pending[1].cancel()
                runtime.subagents_pending = None
            else:
                # Newest list wins; the scheduled flush keeps its deadline.
                runtime.subagents_pending = (data, pending[1])
            return
        if last is not None and now - last[1] < SUBAGENT_UPDATE_SECONDS:
            delay = SUBAGENT_UPDATE_SECONDS - (now - last[1])
            task = asyncio.create_task(
                self._flush_subagents(channel, runtime, event.session_id, delay),
                name=f"subagents-{channel}",
            )
            runtime.subagents_pending = (data, task)
            return
        runtime.subagents = (data, now)
        await self._emit(channel, "subagent.updated", session_id=event.session_id, data=data)

    async def _flush_subagents(
        self, channel: str, runtime: ChannelRuntime, sid: str | None, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        pending = runtime.subagents_pending
        if pending is None:
            return
        runtime.subagents_pending = None
        runtime.subagents = (pending[0], time.monotonic())
        with contextlib.suppress(Exception):
            await self._emit(channel, "subagent.updated", session_id=sid, data=pending[0])

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
                    or runtime.attaching_session == session_id
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
    def _require_binding(
        runtime: ChannelRuntime,
        action: Envelope | None = None,
    ) -> ChannelBinding:
        if runtime.binding is None:
            raise ValueError("no agent session is attached")
        if (
            action is not None
            and action.session_id is not None
            and (action.session_id != runtime.binding.session_id)
        ):
            raise ValueError("action targets a session that is no longer attached")
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

    @staticmethod
    def _safe_user_prompt(text: str) -> dict[str, Any]:
        if scan_secrets(text):
            return {"omitted": True, "reason": "high-confidence secret detected"}
        return {"content": truncate_utf8(clean_text(text), MAX_CONTENT_BYTES)}

    def _history_envelope(self, event: BackendEvent, reply: str) -> Envelope:
        kind = {
            "user_prompt": "user.prompt",
            "turn_started": "turn.started",
            "turn_done": "turn.completed",
            "turn_failed": "turn.failed",
            "progress": "plan.updated",
            "assistant": "assistant.completed",
            "tool_started": "tool.started",
            "tool_finished": "tool.completed",
        }.get(event.kind)
        if kind is None:
            raise ProtocolError(f"unsupported backend history event: {event.kind}")
        if event.kind == "user_prompt":
            data = self._safe_user_prompt(event.text)
        elif event.kind == "assistant":
            data = (
                {"omitted": True, "reason": "high-confidence secret detected"}
                if scan_secrets(event.text)
                else {"content": truncate_utf8(clean_text(event.text), MAX_CONTENT_BYTES)}
            )
        elif event.kind == "progress":
            data = (
                {"omitted": True, "reason": "high-confidence secret detected"}
                if scan_secrets(event.text)
                else {"summary": truncate_utf8(clean_text(event.text), 32 * 1024)}
            )
            if event.data.get("plan") is True:
                data["plan"] = True
                data["running"] = bool(event.data.get("running"))
        elif event.kind in {"tool_started", "tool_finished"}:
            data = self._safe_tool_data(event)
        elif event.kind == "turn_failed" and event.text:
            data = {"message": safe_one_line(event.text, 1000)}
        else:
            data = {}
        return new_envelope(
            kind,
            "event",
            self.instance,
            id=event.event_id or str(uuid.uuid4()),
            at=event.at or int(time.time() * 1000),
            epoch=self.epoch,
            session_id=event.session_id,
            turn_id=event.turn_id,
            item_id=event.item_id,
            reply=reply,
            history=True,
            data=data,
        )

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
            "tuiAttached": session.tui_attached,
        }

    @staticmethod
    def _session_snapshot_data(session: SessionSummary) -> dict[str, Any]:
        recent_outputs: list[dict[str, Any]] = []
        for output in session.recent_outputs[-3:]:
            if scan_secrets(output.text):
                content = "Output omitted from IRC because it may contain a secret"
                omitted = True
            else:
                content = truncate_utf8(clean_text(output.text), MAX_PREVIEW_BYTES)
                omitted = False
            item: dict[str, Any] = {
                "iid": output.id,
                "content": content,
                "omitted": omitted,
            }
            if output.turn_id:
                item["tid"] = output.turn_id
            if output.phase:
                item["phase"] = output.phase
            recent_outputs.append(item)
        recent_activity: list[dict[str, Any]] = []
        for activity in session.recent_activity[-6:]:
            event = BackendEvent(
                kind=activity.kind,
                backend="codex",
                session_id=session.id,
                turn_id=activity.turn_id,
                item_id=activity.item_id,
                tool_kind=activity.tool_kind,
                success=activity.success,
                data=activity.data,
            )
            item = {
                "kind": "tool.started" if activity.kind == "tool_started" else "tool.completed",
                "iid": activity.item_id,
                "data": Bridge._safe_tool_data(event),
            }
            if activity.turn_id:
                item["tid"] = activity.turn_id
            recent_activity.append(item)
        status = "waiting" if session.active_flags else "running" if session.busy else "ready"
        return {
            "cwd": session.cwd,
            "busy": session.busy,
            "flags": list(session.active_flags),
            "tuiAttached": session.tui_attached,
            "status": status,
            "recentOutputs": recent_outputs,
            "recentActivity": recent_activity,
        }

    @staticmethod
    def _safe_tool_data(event: BackendEvent) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": event.tool_kind, "success": event.success}
        if event.item_id:
            result["id"] = event.item_id
        limits = {"label": 200, "input": 4096, "output": 4096, "diff": 4096, "status": 80}
        for key, limit in limits.items():
            value = event.data.get(key)
            if not isinstance(value, str) or not value:
                continue
            cleaned = clean_block(value)
            if not cleaned or scan_secrets(cleaned):
                continue
            result[key] = truncate_utf8(cleaned, limit)
        for key in ("exitCode", "durationMs"):
            value = event.data.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                result[key] = value
        return result

    @staticmethod
    def _queue_data(item: QueuedPrompt) -> dict[str, Any]:
        return {
            "iid": item.id,
            "sid": item.session_id,
            "position": item.position,
            "content": item.text,
            "createdAt": item.created_at,
        }
