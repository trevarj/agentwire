from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from agentwire.backends.base import Backend
from agentwire.bridge import Bridge
from agentwire.config import (
    BridgeConfig,
    CodexConfig,
    Config,
    IRCConfig,
    OpenCodeConfig,
    SecretsConfig,
    StackConfig,
)
from agentwire.models import BackendEvent, ChannelBinding, Question, SessionSummary
from agentwire.protocol import new_envelope


class FakeIRC:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[tuple[str, Any, str | None]] = []

    async def start(self) -> None: ...

    async def wait_ready(self, timeout: float = 30) -> None: ...

    async def close(self) -> None: ...

    async def recv(self) -> Any:
        return await self.incoming.get()

    async def send_protocol(self, channel: str, envelope: Any, preview: str | None = None) -> None:
        self.sent.append((channel, envelope, preview))


class FakeBackend(Backend):
    name = "codex"

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.sent: list[tuple[str, str]] = []
        self.steered: list[tuple[str, str]] = []
        self.settings: dict[str, Any] = {}
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()

    async def start(self) -> None: ...

    async def wait_ready(self, timeout: float = 30) -> None: ...

    async def close(self) -> None: ...

    def events(self) -> AsyncIterator[BackendEvent]:
        async def iterate() -> AsyncIterator[BackendEvent]:
            while True:
                yield await self._events.get()

        return iterate()

    async def list_sessions(self, cwd: str) -> list[SessionSummary]:
        return [SessionSummary("s1", cwd, "session")]

    async def list_running_sessions(self) -> list[SessionSummary]:
        return [SessionSummary("s1", self.workspace, "session")]

    async def create_session(self, cwd: str) -> SessionSummary:
        return SessionSummary("s1", cwd, "session")

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        return SessionSummary(session_id, cwd or self.workspace, "session")

    async def configure_session(self, session_id: str, settings: Any) -> None:
        self.settings = dict(settings)

    async def send_message(self, session_id: str, text: str) -> str | None:
        self.sent.append((session_id, text))
        return f"turn-{len(self.sent)}"

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        self.steered.append((session_id, text))

    async def cancel(self, session_id: str, turn_id: str | None) -> None: ...

    async def resolve_approval(self, request_token: str | int, allow: bool) -> None: ...

    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None: ...

    async def get_last_reply(self, session_id: str) -> str | None:
        return None


def make_bridge(tmp_path: Path) -> tuple[Bridge, FakeIRC, FakeBackend]:
    irc = FakeIRC()
    backend = FakeBackend(str(tmp_path))
    config = Config(
        path=tmp_path / "config.toml",
        bridge=BridgeConfig("trev", (tmp_path,), tmp_path / "state.sqlite3", 2),
        secrets=SecretsConfig(tmp_path / "secrets.env"),
        irc=IRCConfig(
            "127.0.0.1",
            16698,
            "irc.example",
            tmp_path / "ca.pem",
            "bridge",
            "bridge",
            "bridge",
            "IRC_PASSWORD",
            MappingProxyType({"#codex": "codex"}),
        ),
        codex=CodexConfig(tmp_path / "codex.sock", "codex"),
        opencode=OpenCodeConfig(
            "http://127.0.0.1:14096", "opencode", "OPENCODE_PASSWORD", "opencode"
        ),
        stack=StackConfig("ssh", "host", 16698, "127.0.0.1", 6698, "/cert", 14096, 30),
    )
    return Bridge(config, irc, {"codex": backend}), irc, backend  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_topic_activates_harness_and_emits_bootstrap(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex | Workspace")
    assert bridge.channels["#codex"].activation is not None
    assert [item[1].kind for item in irc.sent] == ["agent.hello", "channel.snapshot"]
    await bridge._handle_topic("#codex", "ordinary channel")
    assert bridge.channels["#codex"].activation is None


@pytest.mark.asyncio
async def test_sync_returns_correlated_hello_and_snapshot(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    irc.sent.clear()
    action = new_envelope("sync.request", "action", "client", device="phone")
    await bridge._handle_action("#codex", action)
    correlated = [item for item in irc.sent if item[1].reply == action.id]
    assert [item[1].kind for item in correlated] == [
        "action.accepted",
        "agent.hello",
        "channel.snapshot",
        "action.succeeded",
    ]
    assert correlated[1][1].epoch == bridge.epoch


@pytest.mark.asyncio
async def test_live_prompt_is_acknowledged_and_deduplicated(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    irc.sent.clear()
    action = new_envelope(
        "turn.prompt",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"content": "hello"},
    )
    await bridge._handle_action("#codex", action)
    await bridge._handle_action("#codex", action)
    assert backend.sent == [("s1", "hello")]
    assert [item[1].kind for item in irc.sent] == [
        "action.accepted",
        "action.succeeded",
        "action.succeeded",
    ]
    assert irc.sent[-1][1].data == {"duplicate": True}


@pytest.mark.asyncio
async def test_stale_epoch_is_rejected_before_backend_dispatch(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    irc.sent.clear()
    action = new_envelope(
        "turn.prompt",
        "action",
        "client",
        epoch="stale",
        device="phone",
        data={"content": "hello"},
    )
    await bridge._handle_action("#codex", action)
    assert irc.sent[-1][1].kind == "action.failed"
    assert irc.sent[-1][1].reply == action.id
    assert backend.sent == []


@pytest.mark.asyncio
async def test_historic_action_is_explicitly_rejected(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    irc.sent.clear()
    action = new_envelope(
        "turn.prompt",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        history=True,
        data={"content": "hello"},
    )
    await bridge._handle_action("#codex", action)
    assert irc.sent[-1][1].kind == "action.failed"
    assert irc.sent[-1][1].reply == action.id
    assert backend.sent == []


@pytest.mark.asyncio
async def test_busy_prompt_queues_and_completion_drains_it(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    runtime = bridge.channels["#codex"]
    runtime.binding = ChannelBinding("codex", "s1", str(tmp_path))
    runtime.busy = True
    action = new_envelope(
        "turn.prompt",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"content": "later"},
    )
    await bridge._handle_action("#codex", action)
    assert backend.sent == []
    assert any(item[1].kind == "queue.item.added" for item in irc.sent)
    await bridge._handle_backend_event(
        "#codex", BackendEvent("turn_done", "codex", session_id="s1", turn_id="old")
    )
    assert backend.sent == [("s1", "later")]


@pytest.mark.asyncio
async def test_topic_reactivation_reconciles_idle_backend_and_drains_queue(
    tmp_path: Path,
) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge.state.initialize()
    runtime = bridge.channels["#codex"]
    runtime.binding = ChannelBinding("codex", "s1", str(tmp_path))
    runtime.busy = True
    await bridge.state.set("#codex", runtime.binding)
    await bridge.state.enqueue("queued", "#codex", "s1", "after restore", 2)

    await bridge._handle_topic("#codex", "ordinary topic")
    assert runtime.activation is None
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")

    assert runtime.busy is True
    assert backend.sent == [("s1", "after restore")]
    assert any(item[1].kind == "queue.item.removed" for item in irc.sent)


@pytest.mark.asyncio
async def test_secret_assistant_message_is_wholly_omitted(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    await bridge._handle_backend_event(
        "#codex",
        BackendEvent("assistant", "codex", session_id="s1", text="API_TOKEN=abcdefghijklmno"),
    )
    event = irc.sent[-1][1]
    assert event.kind == "assistant.completed"
    assert event.data["omitted"] is True
    assert "API_TOKEN" not in str(event.to_dict())


@pytest.mark.asyncio
async def test_sensitive_question_is_redacted_even_without_backend_flag(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))

    await bridge._handle_backend_event(
        "#codex",
        BackendEvent(
            "question",
            "codex",
            session_id="s1",
            request_token=7,
            questions=(
                Question(
                    id="database_password",
                    header="Credentials",
                    prompt="What is the database password?",
                ),
            ),
        ),
    )

    event = irc.sent[-1][1]
    assert event.kind == "request.opened"
    assert event.data["redacted"] is True
    assert "password" not in str(event.to_dict()).lower()


@pytest.mark.asyncio
async def test_request_preserves_zero_json_rpc_token(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))

    await bridge._handle_backend_event(
        "#codex",
        BackendEvent(
            "approval",
            "codex",
            session_id="s1",
            request_token=0,
            text="shell command approval needed",
        ),
    )

    pending = next(iter(bridge.channels["#codex"].requests.values()))
    assert pending.token == 0
