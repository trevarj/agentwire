from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

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
from agentwire.models import BackendEvent, ChannelBinding, Question, SessionOutput, SessionSummary
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
        self.sessions: list[SessionSummary] | None = None
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
        return (
            self.sessions
            if self.sessions is not None
            else [SessionSummary("s1", cwd, "session")]
        )

    async def list_running_sessions(self) -> list[SessionSummary]:
        return [SessionSummary("s1", self.workspace, "session")]

    async def create_session(self, cwd: str) -> SessionSummary:
        return SessionSummary("s1", cwd, "session")

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        return SessionSummary(session_id, cwd or self.workspace, "session")

    async def configure_session(self, session_id: str, settings: Any) -> None:
        self.settings = dict(settings)

    async def setting_options(self) -> Mapping[str, Any]:
        return {
            "model": [{
                "value": "gpt-test",
                "label": "GPT Test",
                "efforts": ["low", "high"],
                "defaultEffort": "high",
            }]
        }

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
    assert irc.sent[0][1].data["settingOptions"]["model"][0]["value"] == "gpt-test"
    await bridge._handle_topic("#codex", "ordinary channel")
    assert bridge.channels["#codex"].activation is None


@pytest.mark.asyncio
async def test_only_replayable_events_are_journaled(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    append_event = AsyncMock()
    bridge.state.append_event = append_event

    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    append_event.assert_not_awaited()

    await bridge._emit("#codex", "turn.started")
    append_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspace_pages_browse_allowlisted_directories(tmp_path: Path) -> None:
    (tmp_path / "project-b").mkdir()
    (tmp_path / "project-a").mkdir()
    (tmp_path / ".hidden").mkdir()
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    irc.sent.clear()

    root_action = new_envelope(
        "workspace.list.request", "action", "client", epoch=bridge.epoch, device="phone"
    )
    await bridge._handle_action("#codex", root_action)
    root_page = next(item[1] for item in irc.sent if item[1].kind == "workspace.page")
    assert root_page.data == {
        "parent": None,
        "items": [{"path": str(tmp_path), "name": tmp_path.name, "hasChildren": True}],
        "next": None,
    }

    irc.sent.clear()
    child_action = new_envelope(
        "workspace.list.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"parent": str(tmp_path)},
    )
    await bridge._handle_action("#codex", child_action)
    child_page = next(item[1] for item in irc.sent if item[1].kind == "workspace.page")
    assert [item["name"] for item in child_page.data["items"]] == ["project-a", "project-b"]
    assert child_page.data["parent"] == str(tmp_path)


@pytest.mark.asyncio
async def test_session_pages_echo_workspace_and_continue_with_cursor(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    backend.sessions = [
        SessionSummary(f"s{index}", str(tmp_path), f"Session {index}")
        for index in range(101)
    ]
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    irc.sent.clear()

    first = new_envelope(
        "session.list.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )
    await bridge._handle_action("#codex", first)
    first_page = next(item[1] for item in irc.sent if item[1].kind == "session.page")
    assert first_page.data["cwd"] == str(tmp_path)
    assert first_page.data["cursor"] is None
    assert len(first_page.data["items"]) == 100
    assert first_page.data["next"] == "100"

    irc.sent.clear()
    second = new_envelope(
        "session.list.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path), "cursor": "100"},
    )
    await bridge._handle_action("#codex", second)
    second_page = next(item[1] for item in irc.sent if item[1].kind == "session.page")
    assert second_page.data["cursor"] == "100"
    assert len(second_page.data["items"]) == 1
    assert second_page.data["next"] is None


@pytest.mark.asyncio
async def test_settings_are_isolated_per_bound_session(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    await bridge._set_binding("#codex", SessionSummary("s1", str(tmp_path), "First"))
    update = new_envelope(
        "settings.update",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"model": "gpt-test", "approvalReviewer": "auto_review"},
    )
    await bridge._handle_action("#codex", update)
    assert backend.settings["approvalReviewer"] == "auto_review"

    await bridge._set_binding("#codex", SessionSummary("s2", str(tmp_path), "Second"))
    assert bridge.channels["#codex"].settings == {
        "delivery": "queue",
        "approvalReviewer": "manual",
    }
    await bridge._set_binding("#codex", SessionSummary("s1", str(tmp_path), "First"))
    assert bridge.channels["#codex"].settings["approvalReviewer"] == "auto_review"


@pytest.mark.asyncio
async def test_binding_emits_redacted_recent_session_context(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    irc.sent.clear()

    await bridge._set_binding(
        "#codex",
        SessionSummary(
            "s1",
            str(tmp_path),
            "Existing",
            busy=True,
            active_flags=("waitingOnUserInput",),
            active_turn_id="t2",
            recent_outputs=(
                SessionOutput("i1", "t1", "First recovered output", "final"),
                SessionOutput("i2", "t2", "API_TOKEN=abcdefghijklmno", "commentary"),
            ),
        ),
    )

    assert [item[1].kind for item in irc.sent] == [
        "binding.changed",
        "session.snapshot",
        "channel.snapshot",
    ]
    snapshot = irc.sent[1][1]
    assert snapshot.turn_id == "t2"
    assert snapshot.data["status"] == "waiting"
    assert snapshot.data["recentOutputs"] == [
        {
            "iid": "i1",
            "tid": "t1",
            "phase": "final",
            "content": "First recovered output",
            "omitted": False,
        },
        {
            "iid": "i2",
            "tid": "t2",
            "phase": "commentary",
            "content": "Output omitted from IRC because it may contain a secret",
            "omitted": True,
        },
    ]


@pytest.mark.asyncio
async def test_stale_session_action_cannot_target_new_binding(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    await bridge._set_binding("#codex", SessionSummary("s2", str(tmp_path), "Second"))
    irc.sent.clear()
    stale = new_envelope(
        "turn.prompt",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"content": "wrong session"},
    )

    await bridge._handle_action("#codex", stale)

    assert backend.sent == []
    assert [item[1].kind for item in irc.sent] == ["action.accepted", "action.failed"]


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
async def test_plan_progress_preserves_completion_state(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))

    await bridge._handle_backend_event(
        "#codex",
        BackendEvent(
            "progress",
            "codex",
            session_id="s1",
            turn_id="t1",
            text="Plan completed",
            data={
                "plan": True,
                "running": False,
                "status": "completed",
                "completedSteps": 2,
                "totalSteps": 2,
            },
        ),
    )

    event = irc.sent[-1][1]
    assert event.kind == "plan.updated"
    assert event.data == {
        "summary": "Plan completed",
        "plan": True,
        "running": False,
        "status": "completed",
        "completedSteps": 2,
        "totalSteps": 2,
    }


@pytest.mark.asyncio
async def test_tool_preview_omits_only_sensitive_fields(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))

    await bridge._handle_backend_event(
        "#codex",
        BackendEvent(
            "tool_finished",
            "codex",
            session_id="s1",
            turn_id="t1",
            item_id="i1",
            tool_kind="shell",
            success=True,
            data={
                "label": "$ printenv API_TOKEN",
                "input": "API_TOKEN=abcdefghijklmno",
                "output": "working tree clean",
                "status": "completed",
                "exitCode": 0,
            },
        ),
    )

    data = irc.sent[-1][1].data
    assert data["id"] == "i1"
    assert "input" not in data
    assert data["output"] == "working tree clean"
    assert data["status"] == "completed"
    assert data["exitCode"] == 0


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
