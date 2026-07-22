from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import MappingProxyType

import pytest

from irc_bridge.backends.base import Backend
from irc_bridge.bridge import Bridge
from irc_bridge.config import (
    BridgeConfig,
    CodexConfig,
    Config,
    IRCConfig,
    OpenCodeConfig,
    PasteConfig,
    SecretsConfig,
    StackConfig,
)
from irc_bridge.irc import IRCMessage
from irc_bridge.models import BackendEvent, ChannelBinding, Question, SessionSummary


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
        self.resolved: list[tuple[str | int, bool]] = []
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
        return []

    async def create_session(self, cwd: str) -> SessionSummary:
        return SessionSummary("thread-1", cwd, "new")

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        return SessionSummary(session_id, cwd or "/workspace", "attached")

    async def send_message(self, session_id: str, text: str) -> str | None:
        self.sent.append((session_id, text))
        return f"turn-{len(self.sent)}"

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        pass

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
async def test_busy_messages_queue_and_advance_on_completion(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    runtime.busy = True
    await bridge._handle_owner_message(IRCMessage("#codex", "trev", "trev", "second"))
    assert list(runtime.queue) == ["second"]
    await bridge._handle_backend_event(
        BackendEvent(kind="turn_done", backend="codex", session_id="thread-1")
    )
    assert backend.sent == [("thread-1", "second")]
    assert runtime.busy
    assert ("#codex", "starting queued message (0 remain)") in irc.sent


@pytest.mark.asyncio
async def test_tool_relay_does_not_include_backend_payload(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_backend_event(
        BackendEvent(
            kind="tool_started",
            backend="codex",
            session_id="thread-1",
            tool_kind="shell",
            text="cat /secret",
            data={"command": "cat /secret"},
        )
    )
    assert irc.sent == [("#codex", "tool: shell started")]


@pytest.mark.asyncio
async def test_progress_is_relayed_without_replacing_last_reply(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    runtime = bridge.channels["#codex"]
    runtime.last_reply = "previous final reply"
    await bridge._handle_backend_event(
        BackendEvent(
            kind="progress",
            backend="codex",
            session_id="thread-1",
            text="I found the cause and am checking the fix.",
        )
    )
    assert irc.sent == [("#codex", "I found the cause and am checking the fix.")]
    assert runtime.last_output == "I found the cause and am checking the fix."
    assert runtime.last_reply == "previous final reply"


@pytest.mark.asyncio
async def test_last_repeats_latest_output_without_starting_turn(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    bridge.channels["#codex"].last_output = "Most recent commentary"
    await bridge._command("#codex", "!last", "")
    assert irc.sent == [("#codex", "Most recent commentary")]
    assert backend.sent == []


@pytest.mark.asyncio
async def test_backend_status_change_updates_irc_status(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_backend_event(
        BackendEvent(
            kind="status_changed",
            backend="codex",
            session_id="thread-1",
            data={"busy": True, "active_flags": ("waitingOnApproval",)},
        )
    )
    await bridge._status("#codex")
    assert "codex busy (waiting on approval)" in irc.sent[-1][1]
    assert "activity waiting on approval" in irc.sent[-1][1]


@pytest.mark.asyncio
async def test_binding_restores_active_state_and_latest_output(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._bind(
        "#codex",
        SessionSummary(
            "thread-2",
            str(tmp_path),
            "active session",
            busy=True,
            active_turn_id="turn-2",
            last_output="Existing commentary",
        ),
    )
    await bridge._status("#codex")
    await bridge._command("#codex", "!last", "")
    assert "codex busy" in irc.sent[-2][1]
    assert "activity Existing commentary" in irc.sent[-2][1]
    assert irc.sent[-1] == ("#codex", "Existing commentary")
    assert backend.sent == []


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
