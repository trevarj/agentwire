from __future__ import annotations

import asyncio
import contextlib
import logging
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
from agentwire.irc import IRCMessage
from agentwire.models import (
    BackendEvent,
    ChannelBinding,
    HistoryPage,
    Question,
    SessionActivity,
    SessionOutput,
    SessionSummary,
)
from agentwire.protocol import PROTOCOL_TAG, Envelope, ProtocolError, encode_envelope, new_envelope


class FakeIRC:
    def __init__(self, account: str = "") -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[tuple[str, Any, str | None]] = []
        self.notices: list[tuple[str, str]] = []
        # The account the server confirmed; empty until SASL reports one.
        self.account = account

    async def start(self) -> None: ...

    async def wait_ready(self, timeout: float = 30) -> None: ...

    async def close(self) -> None: ...

    async def recv(self) -> Any:
        return await self.incoming.get()

    async def send_protocol(self, channel: str, envelope: Any, preview: str | None = None) -> None:
        self.sent.append((channel, envelope, preview))

    async def send_notice(self, channel: str, text: str) -> None:
        self.notices.append((channel, text))


class FakeBackend(Backend):
    name = "codex"

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.sent: list[tuple[str, str]] = []
        self.steered: list[tuple[str, str]] = []
        self.settings: dict[str, Any] = {}
        self.sessions: list[SessionSummary] | None = None
        self.session_counts: dict[str, int] = {}
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()

    def count_sessions(self, cwd: str) -> int | None:
        return self.session_counts.get(cwd)

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
            self.sessions if self.sessions is not None else [SessionSummary("s1", cwd, "session")]
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
            "model": [
                {
                    "value": "gpt-test",
                    "label": "GPT Test",
                    "efforts": ["low", "high"],
                    "defaultEffort": "high",
                }
            ]
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


def make_bridge(
    tmp_path: Path,
    owner: str = "trev",
    account: str = "",
    channels: Mapping[str, str] | None = None,
) -> tuple[Bridge, FakeIRC, FakeBackend]:
    irc = FakeIRC(account)
    backend = FakeBackend(str(tmp_path))
    config = Config(
        path=tmp_path / "config.toml",
        bridge=BridgeConfig(owner, (tmp_path,), tmp_path / "state.sqlite3", 2),
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
            MappingProxyType(dict(channels or {"#codex": "codex"})),
        ),
        codex=CodexConfig(tmp_path / "codex.sock", "codex"),
        opencode=OpenCodeConfig(
            "http://127.0.0.1:14096", "opencode", "OPENCODE_PASSWORD", "opencode"
        ),
        stack=StackConfig("ssh", "host", 16698, "127.0.0.1", 6698, "/cert", 14096, 30),
    )
    return Bridge(config, irc, {"codex": backend}), irc, backend  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_action_workers_preserve_channel_independence(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path, channels={"#first": "codex", "#second": "codex"})
    for channel in bridge.channels:
        await bridge._handle_topic(channel, "agentwire:v1;account=trev;agent=bridge;backend=codex")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_done = asyncio.Event()

    async def dispatch(channel: str, _action: Envelope) -> None:
        if channel == "#first":
            first_started.set()
            await release_first.wait()
        else:
            second_done.set()

    bridge._dispatch_action = dispatch  # type: ignore[method-assign]
    workers = [asyncio.create_task(bridge._action_loop(channel)) for channel in bridge.channels]
    try:
        now = asyncio.get_running_loop().time()
        action = new_envelope("sync.request", "action", "client", device="phone")
        await bridge._action_queues["#first"].put((action, now))
        await first_started.wait()
        await bridge._action_queues["#second"].put((action, now))
        await asyncio.wait_for(second_done.wait(), 0.5)
        assert not release_first.is_set()
    finally:
        release_first.set()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_topic_activates_harness_and_waits_for_client_sync(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic(
        "#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex | Workspace"
    )
    assert bridge.channels["#codex"].activation is not None
    assert irc.sent == []
    await bridge._handle_topic("#codex", "ordinary channel")
    assert bridge.channels["#codex"].activation is None
    # Clients trust events only from the topic agent, so a topic naming another
    # account must suspend the channel instead of running an untrusted harness.
    with pytest.raises(ProtocolError, match="agent"):
        await bridge._handle_topic(
            "#codex", "agentwire:v1;account=trev;agent=intruder;backend=codex"
        )
    assert bridge.channels["#codex"].activation is None


@pytest.mark.asyncio
async def test_topic_agent_is_validated_against_the_authenticated_account(tmp_path: Path) -> None:
    # The server confirmed the account "agentwire" for a connection whose
    # nickname is "bridge". Events are attributed by account, so the account is
    # the only identity a topic may name.
    bridge, irc, _backend = make_bridge(tmp_path, account="agentwire")
    with pytest.raises(ProtocolError, match="agent bridge is not this bridge's"):
        await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    assert bridge.channels["#codex"].activation is None
    assert irc.sent == []
    assert irc.notices == [
        (
            "#codex",
            "agentwire suspended: topic agent bridge is not this bridge's "
            "authenticated account agentwire; "
            "set: agentwire:v1;account=trev;agent=agentwire;backend=codex",
        )
    ]

    irc.notices.clear()
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=agentwire;backend=codex")
    assert bridge.channels["#codex"].activation is not None
    assert irc.notices == []


@pytest.mark.asyncio
async def test_topic_written_before_agent_was_required_gets_a_pasteable_repair(
    tmp_path: Path,
) -> None:
    """The v1 upgrade made `agent=` mandatory and suspended working channels.

    The bridge must not default the field: a client authenticates events by the
    topic's agent account and ignores a bridge the topic does not name, so a
    silently activated channel publishes into a void, which is harder to
    diagnose than suspension. It refuses, and hands over the exact replacement.
    """

    bridge, irc, _backend = make_bridge(tmp_path)
    with pytest.raises(ProtocolError, match="topic is missing agent="):
        await bridge._handle_topic(
            "#codex", "agentwire:v1;account=trev;backend=codex | Project title"
        )
    assert bridge.channels["#codex"].activation is None
    assert irc.notices == [
        (
            "#codex",
            "agentwire suspended: topic is missing agent=; "
            "set: agentwire:v1;account=trev;agent=bridge;backend=codex | Project title",
        )
    ]

    # Pasting that exact line activates the channel.
    repair = irc.notices[0][1].split("set: ", 1)[1]
    await bridge._handle_topic("#codex", repair)
    assert bridge.channels["#codex"].activation is not None

    # A reconnect re-reads the same broken topic; the channel is told once per
    # cause, not once per topic reply.
    irc.notices.clear()
    for _ in range(3):
        with pytest.raises(ProtocolError):
            await bridge._handle_topic("#codex", "agentwire:v1;account=trev;backend=codex")
    assert len(irc.notices) == 1


@pytest.mark.asyncio
async def test_suspension_is_announced_for_every_silent_failure(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    # A channel that was never activated stays quiet: an ordinary topic is not
    # an event, and announcing it would post a notice on every reconnect.
    await bridge._handle_topic("#codex", "an ordinary channel topic")
    assert irc.notices == []

    # A prefixed topic whose field fails validation is announced.
    with pytest.raises(ProtocolError, match="backend"):
        await bridge._handle_topic(
            "#codex", "agentwire:v1;account=trev;agent=bridge;backend=claude"
        )
    assert irc.notices == [
        (
            "#codex",
            "agentwire suspended: topic backend claude is not the configured "
            "channel backend codex; set: agentwire:v1;account=trev;agent=bridge;backend=codex",
        )
    ]

    # So is a malformed marker, which never yields an activation to compare.
    irc.notices.clear()
    with pytest.raises(ProtocolError):
        await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=;backend=codex")
    assert irc.notices[0][0] == "#codex"
    assert irc.notices[0][1].startswith("agentwire suspended: ")

    # Losing the marker from an active channel is a state change operators must
    # see, so that transition is announced too.
    irc.notices.clear()
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    assert irc.notices == []
    await bridge._handle_topic("#codex", "an ordinary channel topic")
    assert irc.notices == [("#codex", "agentwire suspended: the activation topic was removed")]


@pytest.mark.asyncio
async def test_dropped_protocol_message_reports_its_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    action = new_envelope("sync.request", "action", "client", device="phone")
    tags = MappingProxyType({PROTOCOL_TAG: encode_envelope(action)})

    async def pump(message: IRCMessage) -> None:
        await irc.incoming.put(message)
        task = asyncio.create_task(bridge._irc_loop())
        for _ in range(200):
            if irc.incoming.empty():
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.DEBUG, logger="agentwire.bridge"):
        # Ordinary chatter carries no protocol tag and must never be reported.
        await pump(IRCMessage("#codex", "trev", "trev", "hello everyone"))
        assert caplog.records == []

        # A suspended channel silently discarded this before; now it says so.
        await pump(IRCMessage("#codex", "trev", "trev", "", tags, "TAGMSG"))
        suspended = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(suspended) == 1
        assert "no valid activation topic" in suspended[0].getMessage()

        caplog.clear()
        await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
        caplog.clear()

        # A stranger's action names the account that failed the owner check.
        await pump(IRCMessage("#codex", "mallory", "mallory", "", tags, "TAGMSG"))
        rejected = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(rejected) == 1
        assert "sender account mallory is not the topic owner account trev" in (
            rejected[0].getMessage()
        )

        # The bridge's own echoed traffic is expected, so it never warns.
        caplog.clear()
        await pump(IRCMessage("#codex", "bridge", "bridge", "", tags, "TAGMSG"))
        assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []

    # No diagnostic ever repeats a tag value or message text.
    assert all(encode_envelope(action) not in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_single_account_bridge_never_consumes_its_own_events(tmp_path: Path) -> None:
    # The supported single-account shape: owner_account equals the bridge's own
    # nickname, and the topic names that one account as both account and agent.
    bridge, irc, backend = make_bridge(tmp_path, owner="bridge")
    await bridge._handle_topic("#codex", "agentwire:v1;account=bridge;agent=bridge;backend=codex")
    assert bridge.channels["#codex"].activation is not None
    own_event = encode_envelope(new_envelope("agent.hello", "event", "bridge", epoch=bridge.epoch))

    loop_task = asyncio.create_task(bridge._irc_loop())
    action_task = asyncio.create_task(bridge._action_loop("#codex"))
    try:
        # Its own published hello, echoed back with its own account tag, must
        # never be treated as a command.
        await irc.incoming.put(
            IRCMessage(
                "#codex",
                "bridge",
                "bridge",
                "",
                MappingProxyType({PROTOCOL_TAG: own_event}),
                "TAGMSG",
            )
        )
        # A genuine action from the same shared account must still execute.
        action = new_envelope("sync.request", "action", "client", device="phone")
        await irc.incoming.put(
            IRCMessage(
                "#codex",
                "bridge",
                "bridge",
                "",
                MappingProxyType({PROTOCOL_TAG: encode_envelope(action)}),
                "TAGMSG",
            )
        )
        for _ in range(200):
            if any(item[1].kind == "channel.snapshot" for item in irc.sent):
                break
            await asyncio.sleep(0.005)
    finally:
        loop_task.cancel()
        action_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task
        with contextlib.suppress(asyncio.CancelledError):
            await action_task
    kinds = [item[1].kind for item in irc.sent]
    # Echoed event produced no response; read-only sync emits only its data.
    assert "action.accepted" not in kinds and "action.succeeded" not in kinds
    assert "action.failed" not in kinds and "action.uncertain" not in kinds
    assert "agent.hello" in kinds
    assert backend.sent == []


@pytest.mark.asyncio
async def test_only_replayable_events_are_journaled(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    append_event = AsyncMock()
    bridge.state.append_event = append_event

    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    append_event.assert_not_awaited()

    await bridge._emit("#codex", "turn.started")
    append_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspace_pages_browse_allowlisted_directories(tmp_path: Path) -> None:
    (tmp_path / "project-b").mkdir()
    (tmp_path / "project-a").mkdir()
    (tmp_path / ".hidden").mkdir()
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    assert all("sessionCount" not in item for item in child_page.data["items"])


@pytest.mark.asyncio
async def test_workspace_pages_carry_backend_session_counts(tmp_path: Path) -> None:
    (tmp_path / "project-a").mkdir()
    (tmp_path / "project-b").mkdir()
    bridge, irc, backend = make_bridge(tmp_path)
    backend.session_counts = {str(tmp_path / "project-a"): 3}
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    irc.sent.clear()

    action = new_envelope(
        "workspace.list.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"parent": str(tmp_path)},
    )
    await bridge._handle_action("#codex", action)
    page = next(item[1] for item in irc.sent if item[1].kind == "workspace.page")
    assert [item.get("sessionCount") for item in page.data["items"]] == [3, None]
    assert "sessionCount" not in page.data["items"][1]


@pytest.mark.asyncio
async def test_session_pages_echo_workspace_and_continue_with_cursor(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    backend.sessions = [
        SessionSummary(f"s{index}", str(tmp_path), f"Session {index}") for index in range(101)
    ]
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    assert first_page.data["scope"] == "workspace"
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
async def test_live_session_page_is_explicitly_scoped(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    irc.sent.clear()

    action = new_envelope(
        "session.list.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"scope": "live"},
    )
    await bridge._handle_action("#codex", action)

    page = next(item[1] for item in irc.sent if item[1].kind == "session.page")
    assert page.data["scope"] == "live"
    assert page.data["cwd"] is None


@pytest.mark.asyncio
async def test_history_is_scoped_to_binding_and_echoes_backend_cursor(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    await bridge._set_binding("#codex", SessionSummary("s1", str(tmp_path), "session"))
    backend.list_history = AsyncMock(  # type: ignore[method-assign]
        return_value=HistoryPage(
            (
                BackendEvent(
                    "user_prompt",
                    "codex",
                    session_id="s1",
                    turn_id="t1",
                    item_id="i1",
                    text="hello",
                    at=100,
                    event_id="8eaf7815-cc5c-50de-836f-a2f2c70ca283",
                ),
            ),
            "older",
        )
    )
    irc.sent.clear()

    action = new_envelope(
        "history.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"cursor": "current", "limit": 20},
    )
    await bridge._handle_action("#codex", action)

    backend.list_history.assert_awaited_once_with("s1", "current", 20)
    begin = next(item[1] for item in irc.sent if item[1].kind == "history.begin")
    chunk = next(item[1] for item in irc.sent if item[1].kind == "history.chunk")
    prompt = Envelope.from_dict(chunk.data["events"][0])
    end = next(item[1] for item in irc.sent if item[1].kind == "history.end")
    assert begin.session_id == end.session_id == chunk.session_id == prompt.session_id == "s1"
    assert begin.reply == end.reply == chunk.reply == prompt.reply == action.id
    assert begin.data["next"] == end.data["next"] == "older"
    assert begin.data["chunks"] == end.data["chunks"] == 1
    assert prompt.history is True
    assert prompt.data == {"content": "hello"}

    bad = new_envelope(
        "history.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="other",
    )
    irc.sent.clear()
    await bridge._handle_action("#codex", bad)
    failed = next(item[1] for item in irc.sent if item[1].kind == "action.failed")
    assert "no longer attached" in failed.data["message"]


@pytest.mark.asyncio
async def test_history_events_are_packed_into_chunks(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    backend.list_history = AsyncMock(  # type: ignore[method-assign]
        return_value=HistoryPage(
            tuple(
                BackendEvent(
                    "assistant",
                    "codex",
                    session_id="s1",
                    turn_id="t1",
                    item_id=f"i{index}",
                    text=f"reply {index}",
                )
                for index in range(100)
            ),
            None,
        )
    )

    action = new_envelope(
        "history.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
    )
    await bridge._handle_action("#codex", action)

    assert [event.kind for _channel, event, _preview in irc.sent] == [
        "history.begin",
        "history.chunk",
        "history.end",
    ]
    assert len(irc.sent[1][1].data["events"]) == 100


@pytest.mark.asyncio
async def test_settings_are_isolated_per_bound_session(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
            recent_activity=(
                SessionActivity(
                    "tool_started",
                    "tool-1",
                    "t2",
                    "shell",
                    data={"label": "$ git status --short", "input": "git status --short"},
                ),
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
    assert snapshot.data["recentActivity"] == [
        {
            "kind": "tool.started",
            "iid": "tool-1",
            "tid": "t2",
            "data": {
                "id": "tool-1",
                "kind": "shell",
                "success": None,
                "label": "$ git status --short",
                "input": "git status --short",
            },
        }
    ]


@pytest.mark.asyncio
async def test_attach_defers_live_events_until_after_the_session_snapshot(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    irc.sent.clear()

    async def attach(session_id: str, cwd: str | None = None) -> SessionSummary:
        await bridge._handle_backend_event(
            "#codex",
            BackendEvent(
                kind="tool_started",
                backend="codex",
                session_id=session_id,
                turn_id="turn-1",
                item_id="tool-1",
                tool_kind="shell",
                data={"label": "$ make test"},
            ),
        )
        return SessionSummary(session_id, cwd or str(tmp_path), "Running", busy=True)

    backend.attach_session = attach  # type: ignore[method-assign]
    action = new_envelope(
        "session.attach",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"cwd": str(tmp_path)},
    )
    await bridge._action_attach("#codex", action)

    assert [item[1].kind for item in irc.sent] == [
        "binding.changed",
        "session.snapshot",
        "channel.snapshot",
        "tool.started",
    ]


@pytest.mark.asyncio
async def test_stale_session_action_cannot_target_new_binding(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    irc.sent.clear()
    action = new_envelope("sync.request", "action", "client", device="phone")
    await bridge._handle_action("#codex", action)
    correlated = [item for item in irc.sent if item[1].reply == action.id]
    assert [item[1].kind for item in correlated] == ["agent.hello", "channel.snapshot"]
    assert correlated[0][1].epoch == bridge.epoch


@pytest.mark.asyncio
async def test_live_prompt_is_acknowledged_and_deduplicated(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    prompt = next(item[1] for item in irc.sent if item[1].kind == "user.prompt")
    assert prompt.item_id == action.id
    assert [item[1].kind for item in irc.sent] == [
        "action.accepted",
        "user.prompt",
        "action.succeeded",
        "action.succeeded",
    ]
    assert irc.sent[-1][1].data == {"duplicate": True}


@pytest.mark.asyncio
async def test_stale_epoch_is_rejected_before_backend_dispatch(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
async def test_queue_move_emits_one_snapshot(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    await bridge.state.enqueue("one", "#codex", "s1", "first", 2)
    await bridge.state.enqueue("two", "#codex", "s1", "second", 2)
    irc.sent.clear()

    action = new_envelope(
        "queue.move",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        item_id="two",
        data={"position": 0},
    )
    await bridge._handle_action("#codex", action)

    snapshots = [event for _channel, event, _preview in irc.sent if event.kind == "queue.snapshot"]
    assert len(snapshots) == 1
    assert [item["iid"] for item in snapshots[0].data["items"]] == ["two", "one"]
    assert not any(event.kind == "queue.item.moved" for _channel, event, _preview in irc.sent)


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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")

    assert runtime.busy is True
    assert backend.sent == [("s1", "after restore")]
    assert any(item[1].kind == "queue.item.removed" for item in irc.sent)


@pytest.mark.asyncio
async def test_secret_assistant_message_is_wholly_omitted(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
async def test_followed_user_prompt_is_relayed_through_the_same_redaction(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    await bridge._handle_backend_event(
        "#codex",
        BackendEvent(
            "user_prompt",
            "codex",
            session_id="s1",
            turn_id="t1",
            item_id="u1",
            text="please fix the bug",
        ),
    )
    channel, event, preview = irc.sent[-1]
    assert event.kind == "user.prompt"
    assert event.data["content"] == "please fix the bug"
    # The typed original never hit IRC, so the mirrored prompt is readable.
    assert preview == "please fix the bug"

    await bridge._handle_backend_event(
        "#codex",
        BackendEvent("user_prompt", "codex", session_id="s1", text="API_TOKEN=abcdefghijklmno"),
    )
    event = irc.sent[-1][1]
    assert event.kind == "user.prompt"
    assert event.data["omitted"] is True
    assert "API_TOKEN" not in str(event.to_dict())


@pytest.mark.asyncio
async def test_plan_progress_preserves_completion_state(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
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


@pytest.mark.asyncio
async def test_a_configured_channel_that_never_activates_is_announced(tmp_path: Path) -> None:
    """The silence that makes a topicless channel look like a working one."""

    bridge, irc, _backend = make_bridge(tmp_path)
    # An unregistered channel that emptied comes back with no topic at all, which is
    # exactly the ordinary-topic case the per-topic rule stays quiet about.
    bridge._topics_evaluated.add("#codex")
    await bridge._handle_topic("#codex", "")
    assert irc.notices == []

    await bridge._report_inert_channels()

    channel, text = irc.notices[0]
    assert channel == "#codex"
    assert "no activation topic" in text
    # The repair is pasteable and built from this deployment's own configuration.
    assert "set: agentwire:v1;account=trev;agent=bridge;backend=codex" in text

    # Announced once per process, not on every reconnect.
    irc.notices.clear()
    await bridge._report_inert_channels()
    assert irc.notices == []


@pytest.mark.asyncio
async def test_an_activated_channel_is_not_announced_as_inert(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge._topics_evaluated.add("#codex")
    irc.notices.clear()

    await bridge._report_inert_channels()

    assert irc.notices == []


@pytest.mark.asyncio
async def test_status_of_an_unbound_session_reaches_every_channel_on_that_backend(
    tmp_path: Path,
) -> None:
    """A session drawer renders sessions no channel is bound to.

    Routing by ownership would drop those events, so a status fans out across
    the backend while the bound timeline stays untouched.
    """

    bridge, irc, backend = make_bridge(tmp_path, channels={"#codex": "codex", "#second": "codex"})
    for channel in ("#codex", "#second"):
        await bridge._handle_topic(channel, "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    irc.sent.clear()

    task = asyncio.create_task(bridge._backend_loop(backend))
    await backend._events.put(
        BackendEvent(
            "status_changed",
            "codex",
            session_id="s9",
            data={"busy": True, "active_flags": ["waiting"], "cwd": "/w", "tui": True},
        )
    )
    for _ in range(200):
        if len(irc.sent) >= 2:
            break
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert [(channel, event.kind, event.session_id) for channel, event, _ in irc.sent] == [
        ("#codex", "session.status", "s9"),
        ("#second", "session.status", "s9"),
    ]
    assert irc.sent[0][1].data == {
        "busy": True,
        "flags": ["waiting"],
        "cwd": "/w",
        "tuiAttached": True,
    }
    # The bound session is untouched: no busy flip, no binding change.
    assert bridge.channels["#codex"].busy is False
    assert bridge.channels["#codex"].binding == ChannelBinding("codex", "s1", str(tmp_path))


@pytest.mark.asyncio
async def test_observed_session_status_is_coalesced_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    irc.sent.clear()
    # A short real window keeps the trailing flush observable without slow tests.
    monkeypatch.setattr("agentwire.bridge.OBSERVED_STATUS_SECONDS", 0.2)

    async def status(sid: str, busy: bool, flags: list[str] | None = None) -> None:
        await bridge._handle_backend_event(
            "#codex",
            BackendEvent(
                "status_changed",
                "codex",
                session_id=sid,
                data={"busy": busy, "active_flags": flags or []},
            ),
        )

    def sent() -> list[tuple[str | None, bool, list[str]]]:
        return [
            (event.session_id, event.data["busy"], event.data["flags"]) for _, event, _ in irc.sent
        ]

    await status("s9", True)
    assert sent() == [("s9", True, [])]

    # Repeating what the client already knows says nothing.
    await status("s9", True)
    assert len(irc.sent) == 1

    # A change inside the window is held, then delivered when the window closes:
    # the newest state always reaches the drawer, never a stranded busy flag.
    await status("s9", False)
    assert len(irc.sent) == 1
    await asyncio.sleep(0.3)
    assert sent() == [("s9", True, []), ("s9", False, [])]

    # Two changes inside one window cancel out to the already-visible state.
    await status("s9", True)
    await status("s9", False)
    await asyncio.sleep(0.3)
    assert len(irc.sent) == 2

    # Newest pending payload wins when the flush fires.
    await status("s9", True)
    assert len(irc.sent) == 3
    await status("s9", False)
    await status("s9", False, ["waiting"])
    await asyncio.sleep(0.3)
    assert sent()[-1] == ("s9", False, ["waiting"])
    assert len(irc.sent) == 4

    # Coalescing is per session: another session is not held back by the first.
    await status("s8", True)
    assert sent()[-1] == ("s8", True, [])


@pytest.mark.asyncio
async def test_subagent_updates_are_bound_session_state_and_coalesced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    irc.sent.clear()
    # A short real window keeps the trailing flush observable without slow tests.
    monkeypatch.setattr("agentwire.bridge.SUBAGENT_UPDATE_SECONDS", 0.2)

    async def update(sid: str, *ids: str) -> None:
        await bridge._handle_backend_event(
            "#codex",
            BackendEvent(
                "subagent_update",
                "codex",
                session_id=sid,
                data={
                    "agents": [
                        {
                            "id": agent_id,
                            "type": "Terra",
                            "description": "d",
                            "status": "running",
                            "isBackground": False,
                        }
                        for agent_id in ids
                    ]
                },
            ),
        )

    def sent() -> list[tuple[str, list[str]]]:
        return [
            (event.kind, [agent["id"] for agent in event.data["agents"]])
            for _, event, _ in irc.sent
        ]

    await update("s1", "a1")
    assert sent() == [("subagent.updated", ["a1"])]

    # Repeating the visible list says nothing.
    await update("s1", "a1")
    assert len(irc.sent) == 1

    # A change inside the window is held and delivered when the window closes.
    await update("s1", "a1", "a2")
    assert len(irc.sent) == 1
    await asyncio.sleep(0.3)
    assert sent() == [("subagent.updated", ["a1"]), ("subagent.updated", ["a1", "a2"])]

    # Two changes inside one window cancel out to the already-visible list.
    await update("s1", "a3")
    await update("s1", "a1", "a2")
    await asyncio.sleep(0.3)
    assert len(irc.sent) == 2

    # Newest list wins when the flush fires.
    await update("s1")
    assert len(irc.sent) == 3
    await update("s1", "a4")
    await update("s1", "a5")
    await asyncio.sleep(0.3)
    assert sent()[-1] == ("subagent.updated", ["a5"])
    assert len(irc.sent) == 4

    # Session-owned: another session's agents never reach this channel.
    await update("s9", "other")
    assert len(irc.sent) == 4
    assert bridge.channels["#codex"].subagents is not None

    # Rebinding forgets the reading, so the new session's list is never suppressed.
    bridge._reset_subagents(bridge.channels["#codex"])
    assert bridge.channels["#codex"].subagents is None
    await update("s1", "a5")
    assert sent()[-1] == ("subagent.updated", ["a5"])
    assert len(irc.sent) == 5
