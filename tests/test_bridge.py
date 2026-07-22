from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import MappingProxyType

import pytest

from agentwire.backends.base import Backend
from agentwire.bridge import Bridge
from agentwire.config import (
    BridgeConfig,
    CodexConfig,
    Config,
    IRCConfig,
    OpenCodeConfig,
    PasteConfig,
    SecretsConfig,
    StackConfig,
)
from agentwire.irc import IRCMessage
from agentwire.models import BackendEvent, ChannelBinding, Question, SessionSummary


class FakeIRC:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[IRCMessage] = asyncio.Queue()
        self.sent: list[tuple[str, str]] = []

    async def recv(self) -> IRCMessage:
        return await self.incoming.get()

    async def send_privmsg(self, target: str, text: str) -> None:
        self.sent.append((target, text))


class FakeBackend(Backend):
    name = "codex"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.steered: list[tuple[str, str]] = []
        self.resolved: list[tuple[str | int, bool]] = []
        self.sessions: list[SessionSummary] = []
        self.running: list[SessionSummary] = []
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def wait_ready(self, timeout: float = 30) -> None:
        pass

    async def close(self) -> None:
        pass

    def events(self) -> AsyncIterator[BackendEvent]:
        async def iterator() -> AsyncIterator[BackendEvent]:
            while True:
                yield await self._events.get()

        return iterator()

    async def list_sessions(self, cwd: str) -> list[SessionSummary]:
        return self.sessions

    async def list_running_sessions(self) -> list[SessionSummary]:
        return self.running

    async def create_session(self, cwd: str) -> SessionSummary:
        return SessionSummary("thread-1", cwd, "new")

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        return SessionSummary(session_id, cwd or "/workspace", "attached")

    async def send_message(self, session_id: str, text: str) -> str | None:
        self.sent.append((session_id, text))
        return f"turn-{len(self.sent)}"

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        self.steered.append((session_id, text))

    async def cancel(self, session_id: str, turn_id: str | None) -> None:
        pass

    async def resolve_approval(self, request_token: str | int, allow: bool) -> None:
        self.resolved.append((request_token, allow))

    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None:
        pass

    async def get_last_reply(self, session_id: str) -> str | None:
        return None


def make_bridge(tmp_path: Path) -> tuple[Bridge, FakeIRC, FakeBackend]:
    irc = FakeIRC()
    backend = FakeBackend()
    config = Config(
        path=tmp_path / "config.toml",
        bridge=BridgeConfig(
            owner_account="trev",
            allowed_roots=(tmp_path,),
            state_file=tmp_path / "state.json",
            queue_limit=2,
            summary_max_lines=2,
            summary_max_bytes=200,
            tool_milestone_limit=2,
            notify_owner_on_start=False,
        ),
        secrets=SecretsConfig(tmp_path / "secrets.env"),
        irc=IRCConfig(
            host="127.0.0.1",
            port=16698,
            server_hostname="irc.example",
            ca_file=tmp_path / "ca.pem",
            nickname="bridge",
            username="bridge",
            realname="bridge",
            password_env="IRC_PASSWORD",
            channels=MappingProxyType({"#codex": "codex"}),
        ),
        codex=CodexConfig(tmp_path / "codex.sock", "codex"),
        opencode=OpenCodeConfig(
            "http://127.0.0.1:14096", "opencode", "OPENCODE_PASSWORD", "opencode"
        ),
        paste=PasteConfig("https://example.invalid", "1h", 1024),
        stack=StackConfig("ssh", "host", 16698, "127.0.0.1", 6698, "/cert", 14096, 30),
    )
    bridge = Bridge(config, irc, {"codex": backend})  # type: ignore[arg-type]
    bridge.channels["#codex"].binding = ChannelBinding("codex", "thread-1", str(tmp_path))
    return bridge, irc, backend


@pytest.mark.asyncio
async def test_busy_chat_builds_held_draft_then_next_queues_it(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    runtime.busy = True
    await bridge._handle_owner_message(IRCMessage("#codex", "trev", "trev", "first"))
    await bridge._handle_owner_message(IRCMessage("#codex", "trev", "trev", "second"))
    assert runtime.held_lines == ["first", "second"]
    assert list(runtime.queue) == []
    await bridge._command("#codex", "next", "")
    assert list(runtime.queue) == ["first\nsecond"]
    assert runtime.held_lines == []
    assert "📬 Draft queued" in irc.sent[-1][1]


@pytest.mark.asyncio
async def test_held_draft_can_steer_or_send_after_turn_finishes(tmp_path: Path) -> None:
    bridge, _irc, backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    runtime.busy = True
    runtime.active_turn = "turn-1"
    runtime.held_lines = ["change direction"]
    await bridge._command("#codex", "steer", "")
    assert backend.steered == [("thread-1", "change direction")]
    assert runtime.held_lines == []

    runtime.busy = False
    runtime.held_lines = ["follow up"]
    await bridge._command("#codex", "next", "")
    assert backend.sent == [("thread-1", "follow up")]
    assert runtime.held_lines == []


@pytest.mark.asyncio
async def test_session_change_refuses_to_lose_draft(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    bridge.channels["#codex"].held_lines = ["keep me"]
    with pytest.raises(ValueError, match="held draft"):
        await bridge._command("#codex", "detach", "")


@pytest.mark.asyncio
async def test_running_alias_filters_outside_roots_and_feeds_use(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    outside = tmp_path.parent / "outside-running"
    outside.mkdir(exist_ok=True)
    backend.running = [
        SessionSummary("run-1", str(tmp_path), "active", updated_at=2, busy=True),
        SessionSummary("run-2", str(outside), "outside", updated_at=3, busy=True),
    ]
    await bridge._command("#codex", "r", "")
    assert [item.id for item in bridge.channels["#codex"].session_choices] == ["run-1"]
    assert "🟢 Running sessions" in irc.sent[-1][1]
    await bridge._command("#codex", "use", "")
    assert bridge.channels["#codex"].binding == ChannelBinding(
        "codex", "run-1", str(tmp_path)
    )


@pytest.mark.asyncio
async def test_core_aliases_and_compatibility_aliases(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._command("#codex", "s", "")
    assert "🤖 Codex" in irc.sent[-1][1]
    bridge.channels["#codex"].session_choices = [
        SessionSummary("thread-2", str(tmp_path), "choice")
    ]
    await bridge._command("#codex", "attach", "1")
    assert bridge.channels["#codex"].binding is not None
    assert bridge.channels["#codex"].binding.session_id == "thread-2"


@pytest.mark.asyncio
async def test_unknown_command_suggests_close_match(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    with pytest.raises(ValueError, match="Did you mean !status"):
        await bridge._command("#codex", "statsu", "")


@pytest.mark.asyncio
async def test_contextual_help_changes_with_held_draft(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    bridge.channels["#codex"].held_lines = ["draft"]
    await bridge._command("#codex", "help", "")
    assert "!next" in irc.sent[-1][1]
    assert "!discard" in irc.sent[-1][1]


@pytest.mark.asyncio
async def test_watch_modes_gate_progress_and_tools_but_not_finals(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    runtime.watch_mode = "quiet"
    await bridge._handle_backend_event(
        BackendEvent(kind="progress", backend="codex", session_id="thread-1", text="update")
    )
    await bridge._handle_backend_event(
        BackendEvent(kind="assistant", backend="codex", session_id="thread-1", text="final")
    )
    assert irc.sent == [("#codex", "💬 final")]

    runtime.watch_mode = "concise"
    await bridge._handle_backend_event(
        BackendEvent(
            kind="tool_started",
            backend="codex",
            session_id="thread-1",
            tool_kind="shell",
            data={"command": "cat /secret"},
        )
    )
    assert "cat /secret" not in " ".join(text for _channel, text in irc.sent)
    assert runtime.last_activity == "tool: shell started"

    runtime.watch_mode = "verbose"
    await bridge._handle_backend_event(
        BackendEvent(
            kind="tool_finished",
            backend="codex",
            session_id="thread-1",
            tool_kind="shell",
            success=True,
        )
    )
    assert irc.sent[-1] == ("#codex", "🔧 shell: finished successfully")


@pytest.mark.asyncio
async def test_status_is_multiline_dashboard(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    runtime.busy = True
    runtime.active_flags = ("waitingOnApproval",)
    runtime.held_lines = ["draft"]
    runtime.queue.append("later")
    await bridge._status("#codex")
    dashboard = irc.sent[-1][1]
    assert "🤖 Codex · 🟡 Waiting On Approval · 👁 concise" in dashboard
    assert "📝 Held: 1 message(s) · 📬 IRC queue: 1" in dashboard


@pytest.mark.asyncio
async def test_last_is_read_only_and_formatted(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    bridge.channels["#codex"].last_output = "Most recent commentary"
    await bridge._command("#codex", "l", "")
    assert irc.sent == [("#codex", "💬 Most recent commentary")]
    assert backend.sent == []


def test_formatted_preview_keeps_total_utf8_budget(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    rendered = bridge._formatted_preview("💬", "🙂" * 200)
    assert len(rendered.encode("utf-8")) <= bridge.config.bridge.summary_max_bytes


@pytest.mark.asyncio
async def test_request_shortcuts_are_one_shot(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    await bridge._handle_backend_event(
        BackendEvent(
            kind="approval",
            backend="codex",
            session_id="thread-1",
            request_token=7,
            text="shell approval needed",
        )
    )
    await bridge._command("#codex", "yes", "")
    assert backend.resolved == [(7, True)]
    assert runtime.approvals == {}
    assert irc.sent[-1] == ("#codex", "✅ A1 approved once")


@pytest.mark.asyncio
async def test_single_question_accepts_multiword_answer_without_id(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    question = Question(id="choice", header="Choice", prompt="What next?")
    await bridge._handle_backend_event(
        BackendEvent(
            kind="question",
            backend="codex",
            session_id="thread-1",
            request_token=8,
            questions=(question,),
        )
    )
    answers: list[Sequence[Sequence[str]] | None] = []

    async def resolve(
        request_token: str | int,
        questions: Sequence[Question],
        values: Sequence[Sequence[str]] | None,
    ) -> None:
        assert request_token == 8
        assert questions == (question,)
        answers.append(values)

    backend.resolve_question = resolve  # type: ignore[method-assign]
    await bridge._command("#codex", "answer", "a multi word answer")
    assert answers[0] == [["a multi word answer"]]
    assert irc.sent[-1] == ("#codex", "✅ Q1 answered")


@pytest.mark.asyncio
async def test_sensitive_question_is_tui_only(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_backend_event(
        BackendEvent(
            kind="question",
            backend="codex",
            session_id="thread-1",
            request_token=9,
            questions=(
                Question(
                    id="password",
                    header="Password",
                    prompt="enter the actual password",
                    secret=True,
                ),
            ),
        )
    )
    assert bridge.channels["#codex"].questions == {}
    assert "actual password" not in irc.sent[-1][1]
    assert "attached TUI" in irc.sent[-1][1]


@pytest.mark.asyncio
async def test_irc_loop_silently_ignores_wrong_account_tag(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    task = asyncio.create_task(bridge._irc_loop())
    await irc.incoming.put(IRCMessage("#codex", "mallory", "trev", "!status"))
    await asyncio.sleep(0)
    assert irc.sent == []
    await irc.incoming.put(IRCMessage("#codex", "trev", "someone", "!status"))
    await asyncio.sleep(0)
    assert len(irc.sent) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
