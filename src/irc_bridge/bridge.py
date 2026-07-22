from __future__ import annotations

import asyncio
import contextlib
import difflib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(slots=True, frozen=True)
class CommandSpec:
    name: str
    aliases: tuple[str, ...]
    usage: str
    category: str
    summary: str


COMMANDS = (
    CommandSpec("help", ("h",), "!help [topic|all]", "general", "show contextual help"),
    CommandSpec("status", ("s",), "!status", "general", "show the channel dashboard"),
    CommandSpec("last", ("l",), "!last", "output", "repeat the latest agent output"),
    CommandSpec("watch", (), "!watch [quiet|concise|verbose]", "output", "set activity detail"),
    CommandSpec("running", ("r",), "!running", "sessions", "list active sessions"),
    CommandSpec("sessions", (), "!sessions [workspace]", "sessions", "list recent sessions"),
    CommandSpec("use", ("attach",), "!use [number]", "sessions", "attach a listed session"),
    CommandSpec("new", (), "!new workspace", "sessions", "create and attach a session"),
    CommandSpec("detach", (), "!detach", "sessions", "detach the current session"),
    CommandSpec("next", (), "!next", "turn", "queue or send the held draft"),
    CommandSpec("steer", (), "!steer [text]", "turn", "redirect the active turn"),
    CommandSpec("discard", (), "!discard", "turn", "discard the held draft"),
    CommandSpec("cancel", ("stop",), "!cancel", "turn", "interrupt the active turn"),
    CommandSpec("queue", (), "!queue", "queue", "show queued turns"),
    CommandSpec("drop", (), "!drop number|all", "queue", "remove queued turns"),
    CommandSpec("yes", ("approve",), "!yes [A1]", "requests", "approve once"),
    CommandSpec("no", ("deny",), "!no [A1]", "requests", "deny an approval"),
    CommandSpec("answer", (), "!answer [Q1] answer", "requests", "answer a question"),
    CommandSpec("skip", ("reject",), "!skip [Q1]", "requests", "reject a question"),
    CommandSpec("paste", (), "!paste", "output", "upload the full final reply"),
    CommandSpec("paste-force", (), "!paste-force", "output", "override a paste scan block"),
)
COMMAND_BY_NAME = {
    alias: spec
    for spec in COMMANDS
    for alias in (spec.name, *spec.aliases)
}


@dataclass(slots=True)
class ChannelRuntime:
    backend: str
    binding: ChannelBinding | None = None
    busy: bool = False
    active_turn: str | None = None
    active_flags: tuple[str, ...] = ()
    watch_mode: str = "concise"
    held_lines: list[str] = field(default_factory=list)
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
                    f"✅ IRC bridge is ready: {channels}. Send !help or !running.",
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
            await self._bind(channel, summary)
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
                await self._say(message.channel, f"⚠️ {safe_one_line(str(exc), 280)}")
            except Exception:
                await self._say(message.channel, "❌ Unexpected bridge failure")

    async def _backend_loop(self, backend: Backend) -> None:
        async for event in backend.events():
            try:
                await self._handle_backend_event(event)
            except Exception:
                channel = self._channel_for(event.backend, event.session_id)
                if channel:
                    await self._say(channel, "❌ Could not relay a backend event")

    async def _handle_owner_message(self, message: IRCMessage) -> None:
        text = message.text.strip()
        if not text:
            return
        if text.startswith("!"):
            command, _, argument = text.partition(" ")
            await self._command(message.channel, command[1:].lower(), argument.strip())
            return
        runtime = self.channels[message.channel]
        if runtime.binding is None:
            raise ValueError("no session is attached — try !running or !new <workspace>")
        if runtime.busy or runtime.held_lines:
            if len(runtime.held_lines) >= self.config.bridge.queue_limit:
                raise ValueError(f"held draft is full ({self.config.bridge.queue_limit} messages)")
            runtime.held_lines.append(text)
            await self._say(
                message.channel,
                f"🟡 Draft held · {len(runtime.held_lines)} message(s)\n"
                "Use !next to queue it, !steer to redirect the turn, or !discard.",
            )
            return
        await self._send_turn(message.channel, text)

    async def _command(self, channel: str, command: str, argument: str) -> None:
        runtime = self.channels[channel]
        command = command.removeprefix("!")
        spec = COMMAND_BY_NAME.get(command)
        if spec is None:
            matches = difflib.get_close_matches(command, COMMAND_BY_NAME, n=1, cutoff=0.55)
            hint = f" Did you mean !{matches[0]}?" if matches else " Try !help."
            raise ValueError(f"unknown command !{command}.{hint}")
        command = spec.name
        if command == "help":
            await self._help(channel, argument)
        elif command == "new":
            await self._new_session(channel, argument)
        elif command == "sessions":
            await self._list_sessions(channel, argument)
        elif command == "running":
            await self._running_sessions(channel)
        elif command == "use":
            await self._use_session(channel, argument)
        elif command == "detach":
            self._require_safe_session_change(runtime)
            runtime.binding = None
            runtime.queue.clear()
            runtime.approvals.clear()
            runtime.questions.clear()
            await self.state.set(channel, None)
            await self._say(channel, "⚪ Session detached")
        elif command == "status":
            await self._status(channel)
        elif command == "last":
            await self._last(channel)
        elif command == "watch":
            await self._watch(channel, argument)
        elif command == "next":
            await self._next(channel)
        elif command == "steer":
            binding = self._require_binding(runtime)
            if not runtime.busy:
                raise ValueError("there is no active turn — use !next to send the held draft")
            if argument and runtime.held_lines:
                raise ValueError("a draft is already held — use !steer without text or !discard")
            text = argument or self._held_text(runtime)
            await self.backends[runtime.backend].steer(
                binding.session_id, runtime.active_turn, text
            )
            if not argument:
                runtime.held_lines.clear()
            await self._say(channel, "🧭 Steering update delivered")
        elif command == "discard":
            if not runtime.held_lines:
                raise ValueError("there is no held draft")
            count = len(runtime.held_lines)
            runtime.held_lines.clear()
            await self._say(channel, f"🗑️ Discarded held draft · {count} message(s)")
        elif command == "cancel":
            binding = self._require_binding(runtime)
            if not runtime.busy:
                raise ValueError("there is no active turn to cancel")
            await self.backends[runtime.backend].cancel(binding.session_id, runtime.active_turn)
            await self._say(channel, "⏹️ Cancellation requested")
        elif command in {"yes", "no"}:
            await self._approval(channel, argument, command == "yes")
        elif command == "answer":
            await self._answer(channel, argument)
        elif command == "skip":
            await self._reject(channel, argument)
        elif command == "queue":
            if not runtime.queue:
                await self._say(channel, "📬 Queue is empty")
            else:
                entries = "\n".join(
                    f"{index}. {safe_one_line(item, 100)}"
                    for index, item in enumerate(runtime.queue, 1)
                )
                await self._say(channel, f"📬 Queue · {len(runtime.queue)} turn(s)\n{entries}")
        elif command == "drop":
            await self._drop(channel, argument)
        elif command in {"paste", "paste-force"}:
            await self._paste(channel, force=command == "paste-force")

    async def _new_session(self, channel: str, raw_path: str) -> None:
        runtime = self.channels[channel]
        if not raw_path:
            raise ValueError("usage: !new <workspace>")
        self._require_safe_session_change(runtime)
        workspace = resolve_workspace(raw_path, self.config.bridge.allowed_roots)
        summary = await self.backends[runtime.backend].create_session(str(workspace))
        await self._bind(channel, summary)
        await self._say(
            channel,
            f"✅ Created {runtime.backend.title()} session\n"
            f"📂 {self._display_path(workspace)} · {self._short(summary.id)}",
        )

    async def _list_sessions(self, channel: str, raw_path: str) -> None:
        runtime = self.channels[channel]
        if raw_path:
            workspace = resolve_workspace(raw_path, self.config.bridge.allowed_roots)
        elif runtime.binding:
            workspace = resolve_workspace(runtime.binding.cwd, self.config.bridge.allowed_roots)
        else:
            raise ValueError("provide a workspace: !sessions <workspace>")
        runtime.session_choices = await self.backends[runtime.backend].list_sessions(str(workspace))
        if not runtime.session_choices:
            await self._say(
                channel,
                f"⚪ No {runtime.backend.title()} sessions in {self._display_path(workspace)}",
            )
            return
        await self._say(
            channel,
            self._format_session_list("Recent sessions", runtime.session_choices),
        )

    async def _running_sessions(self, channel: str) -> None:
        runtime = self.channels[channel]
        discovered = await self.backends[runtime.backend].list_running_sessions()
        allowed: dict[str, SessionSummary] = {}
        for summary in discovered:
            try:
                workspace = resolve_workspace(summary.cwd, self.config.bridge.allowed_roots)
            except ConfigError:
                continue
            allowed[summary.id] = SessionSummary(
                id=summary.id,
                cwd=str(workspace),
                title=summary.title,
                updated_at=summary.updated_at,
                busy=summary.busy,
                active_flags=summary.active_flags,
                active_turn_id=summary.active_turn_id,
                last_output=summary.last_output,
                last_reply=summary.last_reply,
            )
        runtime.session_choices = sorted(
            allowed.values(), key=lambda item: item.updated_at, reverse=True
        )[:20]
        if not runtime.session_choices:
            await self._say(channel, f"⚪ No running {runtime.backend.title()} sessions")
            return
        await self._say(
            channel,
            self._format_session_list("Running sessions", runtime.session_choices),
        )

    async def _use_session(self, channel: str, argument: str) -> None:
        runtime = self.channels[channel]
        self._require_safe_session_change(runtime)
        if not argument:
            if len(runtime.session_choices) != 1:
                raise ValueError("usage: !use <number from !sessions or !running>")
            index = 1
        else:
            try:
                index = int(argument)
            except ValueError as exc:
                raise ValueError("!use takes the number shown by !sessions or !running") from exc
        if index < 1:
            raise ValueError("session number must be positive")
        try:
            choice = runtime.session_choices[index - 1]
        except IndexError as exc:
            raise ValueError("that number is not in the latest !sessions or !running list") from exc
        workspace = resolve_workspace(choice.cwd, self.config.bridge.allowed_roots)
        summary = await self.backends[runtime.backend].attach_session(choice.id, str(workspace))
        await self._bind(channel, summary)
        await self._say(
            channel,
            f"✅ Attached {runtime.backend.title()} session\n"
            f"📂 {self._display_path(workspace)} · {self._short(summary.id)}",
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
        runtime.held_lines.clear()
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
            await self._say(
                channel,
                f"🤖 {runtime.backend.title()} · ⚪ Detached · 👁 {runtime.watch_mode}\n"
                "Try !running or !new <workspace>.",
            )
            return
        state = "🟢 Working" if runtime.busy else "⚪ Idle"
        waiting = "none"
        if runtime.active_flags:
            labels = {
                "waitingOnApproval": "waiting on approval",
                "waitingOnUserInput": "waiting on user input",
                "retry": "retrying",
            }
            waiting = ", ".join(labels.get(flag, flag) for flag in runtime.active_flags)
            state = f"🟡 {waiting.title()}"
        request_ids = [*runtime.approvals, *runtime.questions]
        held = f"{len(runtime.held_lines)} message(s)" if runtime.held_lines else "none"
        requests = ", ".join(request_ids) if request_ids else "none"
        await self._say(
            channel,
            f"🤖 {runtime.backend.title()} · {state} · 👁 {runtime.watch_mode}\n"
            f"📂 {self._display_path(Path(runtime.binding.cwd))} · "
            f"{self._short(runtime.binding.session_id)}\n"
            f"💬 {runtime.last_activity or 'No activity observed'}\n"
            f"📝 Held: {held} · 📬 Queue: {len(runtime.queue)} · ❓ Requests: {requests}",
        )

    async def _watch(self, channel: str, argument: str) -> None:
        runtime = self.channels[channel]
        if not argument:
            await self._say(channel, f"👁 Watch mode: {runtime.watch_mode}")
            return
        mode = argument.lower()
        if mode not in {"quiet", "concise", "verbose"}:
            raise ValueError("usage: !watch quiet|concise|verbose")
        runtime.watch_mode = mode
        await self._say(channel, f"👁 Watch mode set to {mode}")

    async def _help(self, channel: str, topic: str) -> None:
        runtime = self.channels[channel]
        topic = topic.lower() or "context"
        if topic == "context":
            if runtime.binding is None:
                lines = ["!running (!r) — active sessions", "!new WORKSPACE — create session"]
            elif runtime.held_lines:
                lines = [
                    "!next — queue draft",
                    "!steer — redirect turn",
                    "!discard — discard draft",
                ]
            elif runtime.approvals:
                lines = ["!yes [A1] — approve once", "!no [A1] — deny"]
            elif runtime.questions:
                lines = ["!answer [Q1] TEXT — answer", "!skip [Q1] — reject"]
            else:
                lines = [
                    "!status (!s) — dashboard",
                    "!last (!l) — latest output",
                    "!running (!r) — active sessions",
                    "!help all — every command",
                ]
            await self._say(channel, "🧭 Commands\n" + "\n".join(lines))
            return
        categories = {spec.category for spec in COMMANDS}
        if topic != "all" and topic not in categories:
            choices = ", ".join(sorted(categories))
            raise ValueError(f"unknown help topic {topic!r}; choose {choices}, or all")
        specs = COMMANDS if topic == "all" else tuple(
            spec for spec in COMMANDS if spec.category == topic
        )
        lines = [f"{spec.usage} — {spec.summary}" for spec in specs]
        await self._say(channel, f"🧭 {topic.title()} commands\n" + "\n".join(lines))

    async def _approval(self, channel: str, alias: str, allow: bool) -> None:
        runtime = self.channels[channel]
        request = self._select_request(runtime.approvals, alias, "approval")
        await self.backends[request.backend].resolve_approval(request.token, allow)
        runtime.approvals.pop(request.alias, None)
        icon = "✅" if allow else "❌"
        await self._say(channel, f"{icon} {request.alias} {'approved once' if allow else 'denied'}")

    async def _answer(self, channel: str, argument: str) -> None:
        runtime = self.channels[channel]
        first, separator, remainder = argument.partition(" ")
        if first.upper() in runtime.questions:
            if not separator or not remainder.strip():
                raise ValueError("usage: !answer [Q1] <answer> [ | <answer>]")
            alias = first
            raw_answers = remainder
        elif len(runtime.questions) == 1:
            alias = ""
            raw_answers = argument
        else:
            raise ValueError("usage: !answer Q1 <answer> [ | <answer>]")
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
        await self._say(channel, f"✅ {request.alias} answered")

    async def _reject(self, channel: str, alias: str) -> None:
        runtime = self.channels[channel]
        request = self._select_request(runtime.questions, alias, "question")
        await self.backends[request.backend].resolve_question(
            request.token, request.questions, None
        )
        runtime.questions.pop(request.alias, None)
        await self._say(channel, f"⏭️ {request.alias} skipped")

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
            await self._say(channel, f"🗑️ Dropped all {count} queued turns")
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
        await self._say(channel, f"🗑️ Dropped #{index}: {safe_one_line(removed, 100)}")

    async def _next(self, channel: str) -> None:
        runtime = self.channels[channel]
        self._require_binding(runtime)
        text = self._held_text(runtime)
        if runtime.busy:
            if len(runtime.queue) >= self.config.bridge.queue_limit:
                raise ValueError(f"queue is full ({self.config.bridge.queue_limit})")
            runtime.queue.append(text)
            runtime.held_lines.clear()
            await self._say(channel, f"📬 Draft queued as #{len(runtime.queue)}")
            return
        await self._send_turn(channel, text)
        runtime.held_lines.clear()

    async def _paste(self, channel: str, force: bool) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        text = runtime.last_reply or await self.backends[runtime.backend].get_last_reply(
            binding.session_id
        )
        if not text:
            raise ValueError("there is no assistant reply to paste")
        url = await self.paste.upload(text, force=force)
        await self._say(channel, f"🔗 {self.config.paste.expiry} public paste: {url}")

    async def _last(self, channel: str) -> None:
        runtime = self.channels[channel]
        binding = self._require_binding(runtime)
        text = runtime.last_output or await self.backends[runtime.backend].get_last_reply(
            binding.session_id
        )
        if not text:
            raise ValueError("there is no assistant output in this session")
        await self._say(channel, self._formatted_preview("💬", text))

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
        await self._say(channel, "▶️ Turn started")

    async def _handle_backend_event(self, event: BackendEvent) -> None:
        if event.kind == "disconnected":
            for channel, runtime in self.channels.items():
                if runtime.backend == event.backend:
                    await self._say(channel, f"❌ {event.text or f'{event.backend} disconnected'}")
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
            already_announced = runtime.active_turn == event.turn_id and runtime.busy
            runtime.busy = True
            runtime.active_turn = event.turn_id or runtime.active_turn
            runtime.active_flags = ()
            runtime.last_activity = "working"
            runtime.tool_milestones = 0
            runtime.tool_events_suppressed = False
            if not already_announced and self._allows(runtime, "concise"):
                await self._say(channel, "▶️ Turn started")
            return
        if event.kind == "progress":
            runtime.last_activity = safe_one_line(event.text, 180)
            runtime.last_output = event.text
            if self._allows(runtime, "concise"):
                icon = "🧭" if event.text.startswith("plan:") else "💬"
                await self._say(channel, self._formatted_preview(icon, event.text))
            return
        if event.kind == "assistant":
            runtime.last_activity = safe_one_line(event.text, 180)
            runtime.last_output = event.text
            runtime.last_reply = event.text
            await self._say(channel, self._formatted_preview("💬", event.text))
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
            if self._allows(runtime, "verbose") and (
                runtime.tool_milestones < self.config.bridge.tool_milestone_limit
            ):
                runtime.tool_milestones += 1
                await self._say(channel, f"🔧 {event.tool_kind}: {status}")
            elif self._allows(runtime, "verbose") and not runtime.tool_events_suppressed:
                runtime.tool_events_suppressed = True
                await self._say(channel, "🔧 Additional tool milestones suppressed")
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
                f"🔐 {alias}: {event.text or 'approval needed'}\n"
                f"Use !yes {alias} or !no {alias}.",
            )
            return
        if event.kind == "question" and event.request_token is not None:
            if any(question.secret for question in event.questions):
                await self._say(
                    channel,
                    "🔐 Sensitive input requested — answer it in the attached TUI, not IRC.",
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
                f"❓ {alias}: "
                + " | ".join(details)
                + f"\nUse !answer {alias} ... or !skip {alias}.",
            )
            return
        if event.kind == "request_resolved" and event.request_token is not None:
            removed = self._remove_token(runtime, event.request_token)
            if removed:
                await self._say(channel, f"✅ {removed} was resolved in another client")
            return
        if event.kind in {"turn_done", "turn_failed"}:
            runtime.busy = False
            runtime.active_turn = None
            runtime.active_flags = ()
            runtime.approvals.clear()
            runtime.questions.clear()
            if event.kind == "turn_failed":
                runtime.last_activity = f"failed: {event.text or 'backend error'}"
                await self._say(channel, f"❌ Turn failed: {event.text or 'backend error'}")
            elif self._allows(runtime, "concise"):
                await self._say(channel, "✅ Turn complete")
            if runtime.queue:
                next_message = runtime.queue.popleft()
                await self._say(channel, f"📬 Starting queued turn · {len(runtime.queue)} remain")
                await self._send_turn(channel, next_message)
            elif runtime.held_lines:
                await self._say(
                    channel,
                    "🟡 Turn finished; a draft is still held. Use !next or !discard.",
                )

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

    @staticmethod
    def _allows(runtime: ChannelRuntime, level: str) -> bool:
        ranks = {"quiet": 0, "concise": 1, "verbose": 2}
        return ranks[runtime.watch_mode] >= ranks[level]

    @staticmethod
    def _held_text(runtime: ChannelRuntime) -> str:
        if not runtime.held_lines:
            raise ValueError("there is no held draft")
        return "\n".join(runtime.held_lines)

    @staticmethod
    def _require_safe_session_change(runtime: ChannelRuntime) -> None:
        if runtime.busy:
            raise ValueError("cancel the active turn before changing sessions")
        if runtime.held_lines:
            raise ValueError("resolve the held draft with !next or !discard first")
        if runtime.queue:
            raise ValueError("clear the queue with !drop all before changing sessions")

    def _format_session_list(self, title: str, sessions: list[SessionSummary]) -> str:
        lines = [f"🟢 {title}"]
        for index, item in enumerate(sessions, 1):
            state = "🟢" if item.busy else "⚪"
            lines.append(f"{index}. {state} {item.title}")
            lines.append(
                f"   📂 {self._display_path(Path(item.cwd))} · {self._short(item.id)}"
            )
        lines.append("Use !use <number>.")
        return "\n".join(lines)

    def _formatted_preview(self, icon: str, text: str) -> str:
        prefix = f"{icon} "
        budget = max(1, self.config.bridge.summary_max_bytes - len(prefix.encode("utf-8")))
        body, _truncated = preview(
            text,
            self.config.bridge.summary_max_lines,
            budget,
        )
        return prefix + body

    @staticmethod
    def _display_path(path: Path) -> str:
        home = Path.home()
        try:
            return f"~/{path.relative_to(home)}"
        except ValueError:
            return str(path)

    async def _say(self, channel: str, text: str) -> None:
        await self.irc.send_privmsg(channel, text)

    @staticmethod
    def _require_binding(runtime: ChannelRuntime) -> ChannelBinding:
        if runtime.binding is None:
            raise ValueError("no session is attached — try !running or !new <workspace>")
        return runtime.binding

    @staticmethod
    def _short(session_id: str) -> str:
        return session_id if len(session_id) <= 16 else f"{session_id[:12]}…"
