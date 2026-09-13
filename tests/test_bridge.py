from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentwire import cli
from agentwire.backends.base import Backend, BackendError
from agentwire.bridge import AUDIO_TAG, FAST_READ_ACTION_KINDS, MAX_CONTENT_BYTES, Bridge
from agentwire.config import (
    BridgeConfig,
    CodexConfig,
    Config,
    IRCConfig,
    OmpConfig,
    OpenCodeConfig,
    PiConfig,
    PMConfig,
    SecretsConfig,
    StackConfig,
    VoiceConfig,
)
from agentwire.control import ControlError, DelegateRequest, ReportRequest, send_control_request
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
from agentwire.protocol import (
    ACTION_KINDS,
    PROTOCOL_TAG,
    Envelope,
    ProtocolError,
    encode_envelope,
    new_envelope,
)
from agentwire.reference_client import HarnessState
from agentwire.state import StateStore
from agentwire.voice import VoiceError, VoiceMessage, voice_action_id


class FakeIRC:
    def __init__(self, account: str = "") -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[tuple[str, Any, str | None]] = []
        self.notices: list[tuple[str, str]] = []
        self.operations: list[tuple[str, ...]] = []
        self.accepted: set[str] = set()
        # The account the server confirmed; empty until SASL reports one.
        self.account = account

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def recv(self) -> Any:
        return await self.incoming.get()

    async def send_protocol(self, channel: str, envelope: Any, preview: str | None = None) -> None:
        self.sent.append((channel, envelope, preview))
        self.operations.append(("event", channel, envelope.kind))

    async def send_notice(self, channel: str, text: str) -> None:
        self.notices.append((channel, text))

    def register_channel(self, channel: str) -> None:
        self.accepted.add(channel)
        self.operations.append(("register", channel))

    def unregister_channel(self, channel: str) -> None:
        self.accepted.discard(channel)
        self.operations.append(("unregister", channel))

    async def wait_ready(self, timeout: float = 30) -> None:
        self.operations.append(("ready",))

    async def join_channel(self, channel: str) -> None:
        self.operations.append(("join", channel))

    async def set_private(self, channel: str) -> None:
        self.operations.append(("mode", channel, "+is"))

    async def set_topic(self, channel: str, topic: str) -> None:
        self.operations.append(("topic", channel, topic))

    async def invite(self, channel: str, nick: str) -> None:
        self.operations.append(("invite", channel, nick))

    async def flush(self) -> None:
        self.operations.append(("flush",))

    async def part_channel(self, channel: str) -> None:
        self.operations.append(("part", channel))
        self.accepted.discard(channel)


class FakeBackend(Backend):
    name = "codex"

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.sent: list[tuple[str, str]] = []
        self.steered: list[tuple[str, str]] = []
        self.settings: dict[str, Any] = {}
        self.sessions: list[SessionSummary] | None = None
        self.session_counts: dict[str, int] = {}
        self.created_session_id = "s1"
        self.closed_sessions: list[str] = []
        self.attached_sessions: list[str] = []
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
        return SessionSummary(self.created_session_id, cwd, "session")

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        self.attached_sessions.append(session_id)
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
    backend_name: str = "codex",
    dedicated_channels: bool = False,
    voice: VoiceConfig | None = None,
    pm: PMConfig | None = None,
) -> tuple[Bridge, FakeIRC, FakeBackend]:
    irc = FakeIRC(account)
    backend = FakeBackend(str(tmp_path))
    backend.name = backend_name
    configured_channels = dict(channels or {f"#{backend_name}": backend_name})
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
            MappingProxyType(configured_channels),
        ),
        codex=CodexConfig(tmp_path / "codex.sock", "codex", dedicated_channels),
        opencode=OpenCodeConfig(
            "http://127.0.0.1:14096", "opencode", "OPENCODE_PASSWORD", "opencode"
        ),
        stack=StackConfig("ssh", "host", 16698, "127.0.0.1", 6698, "/cert", 14096, 30),
        pi=(
            PiConfig("pi", tmp_path / "sockets", tmp_path / "sessions", dedicated_channels)
            if backend_name == "pi"
            else None
        ),
        omp=(
            OmpConfig("omp", tmp_path / "sockets", tmp_path / "sessions", dedicated_channels)
            if backend_name == "omp"
            else None
        ),
        voice=voice,
        pm=pm,
    )
    return Bridge(config, irc, {backend_name: backend}), irc, backend  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_name", "expected_settings"),
    [
        ("codex", ["model", "effort", "collaboration", "delivery", "approvalReviewer"]),
        ("pi", ["model", "effort", "delivery"]),
        ("omp", ["model", "effort", "delivery"]),
        ("opencode", ["delivery"]),
        ("claude", ["delivery"]),
    ],
)
async def test_hello_advertisements_match_dispatch_and_backend_settings(
    tmp_path: Path, backend_name: str, expected_settings: list[str]
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path, backend_name=backend_name)
    channel = f"#{backend_name}"
    await bridge._emit_hello(channel)
    hello = next(event for target, event, _ in irc.sent if target == channel)

    handler_kinds = set(
        re.findall(r'"([^\"]+)": self\._action_', inspect.getsource(Bridge._dispatch_action))
    )
    assert set(hello.data["actions"]) <= handler_kinds | FAST_READ_ACTION_KINDS <= ACTION_KINDS
    assert ACTION_KINDS - (handler_kinds | FAST_READ_ACTION_KINDS) == {
        "session.rename",
        "session.fork",
        "session.archive",
        "session.unarchive",
    }
    assert hello.data["settings"] == expected_settings


@pytest.mark.asyncio
async def test_action_workers_preserve_channel_independence(tmp_path: Path) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path, channels={"#first": "codex", "#second": "codex"})
    for channel in bridge.channels:
        await bridge._handle_topic(channel, "agentwire:v1;account=trev;agent=bridge;backend=codex")
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_done = asyncio.Event()

    async def dispatch(channel: str, _action: Envelope, _requester_nick: str = "") -> None:
        if channel == "#first":
            first_started.set()
            await release_first.wait()
        else:
            second_done.set()

    bridge._dispatch_action = dispatch  # type: ignore[method-assign]
    workers = [asyncio.create_task(bridge._action_loop(channel)) for channel in bridge.channels]
    try:
        action = new_envelope("sync.request", "action", "client", device="phone")
        await bridge._ingest_action("#first", action, "trev")
        await first_started.wait()
        await bridge._ingest_action("#second", action, "trev")
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
async def test_pi_dedicated_create_preserves_source_and_invites_requester(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=True,
    )
    backend.created_session_id = "2026-08-24T12-00-00-000Z_11111111-1111-7111-8111-111111111111"
    await bridge._handle_topic("#pi", "agentwire:v1;account=trev;agent=bridge;backend=pi")
    source = ChannelBinding("pi", "existing", str(tmp_path))
    bridge.channels["#pi"].binding = source
    await bridge.state.set("#pi", source)
    action = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )

    await bridge._handle_action("#pi", action, "Alice")

    managed = "#pi-11111111"
    assert bridge.channels["#pi"].binding == source
    assert bridge.channels[managed].binding == ChannelBinding(
        "pi", backend.created_session_id, str(tmp_path)
    )
    assert ("invite", managed, "Alice") in irc.operations
    assert [
        operation[0]
        for operation in irc.operations
        if operation[0] in {"join", "mode", "topic", "invite"}
    ] == ["join", "mode", "topic", "invite"]
    assert managed not in await bridge.state.load()

    irc.sent.clear()
    await bridge._emit_hello(managed)
    actions = next(event.data["actions"] for _, event, _ in irc.sent if event.kind == "agent.hello")
    assert "session.close" in actions
    assert "session.create" not in actions

    restored, restored_irc, restored_backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=True,
    )
    await restored._restore_bindings()
    assert managed not in restored_irc.accepted
    assert managed not in restored.channels
    assert restored_backend.attached_sessions == ["existing"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["omp", "codex"])
async def test_dedicated_create_routes_channel_and_close(tmp_path: Path, backend_name: str) -> None:
    channel = f"#{backend_name}"
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={channel: backend_name},
        backend_name=backend_name,
        dedicated_channels=True,
    )
    backend.created_session_id = "55555555-5555-7555-8555-555555555555"
    await bridge._handle_topic(
        channel, f"agentwire:v1;account=trev;agent=bridge;backend={backend_name}"
    )
    source = ChannelBinding(backend_name, "existing", str(tmp_path))
    bridge.channels[channel].binding = source
    create = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )

    await bridge._handle_action(channel, create, "Alice")

    managed = f"#{backend_name}-55555555"
    assert bridge.channels[channel].binding == source
    assert bridge.channels[managed].binding == ChannelBinding(
        backend_name, backend.created_session_id, str(tmp_path)
    )
    assert ("invite", managed, "Alice") in irc.operations
    await bridge._emit_hello(managed)
    hello = next(event for _, event, _ in irc.sent if event.kind == "agent.hello")
    assert hello.data["backend"] == backend_name
    assert "model" in hello.data["settings"]
    managed_hello = next(
        event for target, event, _ in irc.sent if target == managed and event.kind == "agent.hello"
    )
    assert "session.close" in managed_hello.data["actions"]

    close = new_envelope(
        "session.close",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id=backend.created_session_id,
    )
    await bridge._handle_action(managed, close, "Alice")
    assert backend.closed_sessions == [backend.created_session_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_close, send_fails", [(True, False), (True, True), (False, False)])
async def test_codex_close_holds_queue_until_backend_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reject_close: bool, send_fails: bool
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, dedicated_channels=True)
    backend.created_session_id = "66666666-6666-7666-8666-666666666666"
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    create = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )
    await bridge._handle_action("#codex", create, "Alice")
    managed = "#codex-66666666"
    sid = backend.created_session_id
    runtime = bridge.channels[managed]
    binding = runtime.binding
    await bridge.state.enqueue("pending", managed, sid, "keep this prompt", 2)

    async def reject(session_id: str) -> None:
        await bridge._handle_backend_event(
            managed, BackendEvent("turn_done", "codex", session_id=sid)
        )
        assert backend.sent == []
        if reject_close:
            raise BackendError("Codex unsubscribe failed")

    monkeypatch.setattr(backend, "close_session", reject)
    if send_fails:

        async def fail_send(session_id: str, text: str) -> str | None:
            raise BackendError("Codex disconnected")

        monkeypatch.setattr(backend, "send_message", fail_send)
    close = new_envelope(
        "session.close",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id=sid,
    )
    await bridge._handle_action(managed, close, "Alice")
    assert irc.sent[-1][1].kind == ("action.failed" if reject_close else "action.succeeded")
    assert runtime.binding == (binding if reject_close else None)
    assert (managed in bridge.channels) is reject_close
    assert [item.id for item in await bridge.state.list_queue(managed, sid)] == (
        ["pending"] if send_fails else []
    )
    assert backend.sent == ([(sid, "keep this prompt")] if reject_close and not send_fails else [])
    assert not runtime.closing_session


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_method", ["set", "clear_queue"])
async def test_codex_close_retries_state_cleanup_without_releasing_backend_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_method: str
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, dedicated_channels=True)
    backend.created_session_id = "77777777-7777-7777-8777-777777777777"
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    await bridge._handle_action(
        "#codex",
        new_envelope(
            "session.create",
            "action",
            "client",
            epoch=bridge.epoch,
            device="phone",
            data={"cwd": str(tmp_path)},
        ),
        "Alice",
    )
    managed = "#codex-77777777"
    sid = backend.created_session_id
    await bridge.state.enqueue("pending", managed, sid, "queued", 2)
    original = getattr(bridge.state, cleanup_method)

    async def unavailable(*args: Any) -> None:
        raise OSError("state store unavailable")

    monkeypatch.setattr(bridge.state, cleanup_method, unavailable)
    await bridge._handle_action(
        managed,
        new_envelope(
            "session.close", "action", "client", epoch=bridge.epoch, device="phone", session_id=sid
        ),
        "Alice",
    )
    assert irc.sent[-1][1].kind == "action.succeeded"
    assert backend.closed_sessions == [sid]
    assert bridge.channels[managed].binding is None
    assert bridge.channels[managed].released_session_id == sid
    assert managed in bridge._closing_channels

    monkeypatch.setattr(bridge.state, cleanup_method, original)
    await bridge._finish_channel_close(managed)
    assert managed not in bridge.channels
    assert await bridge.state.list_queue(managed, sid) == []
    assert backend.closed_sessions == [sid]


@pytest.mark.asyncio
async def test_pi_dedicated_create_rolls_back_channel_and_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=True,
    )
    backend.created_session_id = "2026-08-24T12-00-00-000Z_aaaaaaaa-aaaa-7aaa-8aaa-aaaaaaaaaaaa"
    await bridge._handle_topic("#pi", "agentwire:v1;account=trev;agent=bridge;backend=pi")

    async def reject_invite(_channel: str, _nick: str) -> None:
        raise RuntimeError("invite failed")

    monkeypatch.setattr(irc, "invite", reject_invite)
    action = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )

    with pytest.raises(RuntimeError, match="invite failed"):
        await bridge._action_create("#pi", action, "Alice")

    assert backend.closed_sessions == [backend.created_session_id]
    assert "#pi-aaaaaaaa" not in bridge.channels
    assert "#pi-aaaaaaaa" not in irc.accepted
    assert "#pi-aaaaaaaa" not in await bridge.state.load()


@pytest.mark.asyncio
async def test_managed_pi_reconnect_recreates_private_channel_only_when_topic_is_missing(
    tmp_path: Path,
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=True,
    )
    backend.created_session_id = "2026-08-24T12-00-00-000Z_bbbbbbbb-bbbb-7bbb-8bbb-bbbbbbbbbbbb"
    await bridge._handle_topic("#pi", "agentwire:v1;account=trev;agent=bridge;backend=pi")
    await bridge._action_create(
        "#pi",
        new_envelope(
            "session.create",
            "action",
            "client",
            epoch=bridge.epoch,
            device="phone",
            data={"cwd": str(tmp_path)},
        ),
        "Alice",
    )
    managed = "#pi-bbbbbbbb"
    topic = "agentwire:v1;account=trev;agent=bridge;backend=pi"

    irc.operations.clear()
    await bridge._handle_managed_topic(IRCMessage(managed, "", "server", "", command="331"))
    assert [operation[0] for operation in irc.operations] == ["ready", "mode", "topic"]
    assert bridge.channels[managed].activation is not None

    irc.operations.clear()
    await bridge._handle_managed_topic(IRCMessage(managed, "", "server", topic, command="332"))
    assert irc.operations == []
    assert bridge.channels[managed].activation is not None


@pytest.mark.asyncio
async def test_pi_dedicated_create_disabled_keeps_static_binding(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=False,
    )
    backend.created_session_id = "2026-08-24T12-00-00-000Z_22222222-2222-7222-8222-222222222222"
    await bridge._handle_topic("#pi", "agentwire:v1;account=trev;agent=bridge;backend=pi")
    action = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )

    await bridge._handle_action("#pi", action, "Alice")

    assert bridge.channels["#pi"].binding is not None
    assert bridge.channels["#pi"].binding.session_id == backend.created_session_id
    assert not any(operation[0] == "join" for operation in irc.operations)


@pytest.mark.asyncio
async def test_managed_pi_close_is_safe_and_succeeds_before_part(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=True,
    )
    backend.created_session_id = "2026-08-24T12-00-00-000Z_33333333-3333-7333-8333-333333333333"
    await bridge._handle_topic("#pi", "agentwire:v1;account=trev;agent=bridge;backend=pi")
    create = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )
    await bridge._handle_action("#pi", create, "Alice")
    managed = "#pi-33333333"
    irc.operations.clear()

    recursive = new_envelope(
        "session.create",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"cwd": str(tmp_path)},
    )
    await bridge._handle_action(managed, recursive, "Alice")
    assert irc.sent[-1][1].kind == "action.failed"

    wrong = new_envelope(
        "session.close",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="wrong",
    )
    await bridge._handle_action(managed, wrong, "Alice")
    assert irc.sent[-1][1].kind == "action.failed"
    assert managed in bridge.channels

    close = new_envelope(
        "session.close",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id=backend.created_session_id,
    )
    queue = bridge._action_queues[managed]
    bridge._start_action_worker(managed)
    worker = bridge._action_workers[managed]
    await bridge._ingest_action(managed, close, "Alice")
    await asyncio.wait_for(queue.join(), 1)
    await asyncio.wait_for(worker, 1)

    assert backend.closed_sessions == [backend.created_session_id]
    assert managed not in bridge.channels
    assert managed not in bridge._action_workers
    assert managed not in await bridge.state.load()
    succeeded = irc.operations.index(("event", managed, "action.succeeded"))
    parted = irc.operations.index(("part", managed))
    assert succeeded < parted
    assert ("topic", managed, "") in irc.operations

    static_close = new_envelope(
        "session.close",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="static",
    )
    bridge.channels["#pi"].binding = ChannelBinding("pi", "static", str(tmp_path))
    await bridge._handle_action("#pi", static_close, "Alice")
    assert backend.closed_sessions == [backend.created_session_id]
    assert irc.sent[-1][1].kind == "action.failed"


@pytest.mark.asyncio
async def test_managed_pi_close_retries_after_reconnect_without_respawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#pi": "pi"},
        backend_name="pi",
        dedicated_channels=True,
    )
    backend.created_session_id = "2026-08-24T12-00-00-000Z_44444444-4444-7444-8444-444444444444"
    await bridge._handle_topic("#pi", "agentwire:v1;account=trev;agent=bridge;backend=pi")
    await bridge._action_create(
        "#pi",
        new_envelope(
            "session.create",
            "action",
            "client",
            epoch=bridge.epoch,
            device="phone",
            data={"cwd": str(tmp_path)},
        ),
        "Alice",
    )
    managed = "#pi-44444444"
    attempts = 0

    async def flaky_flush() -> None:
        nonlocal attempts
        attempts += 1
        irc.operations.append(("flush-failed" if attempts == 1 else "flush",))
        if attempts == 1:
            raise RuntimeError("disconnected")

    monkeypatch.setattr(irc, "flush", flaky_flush)
    irc.operations.clear()
    close = new_envelope(
        "session.close",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id=backend.created_session_id,
    )
    await bridge._handle_action(managed, close, "Alice")

    assert [event.kind for channel, event, _ in irc.sent if channel == managed][-1] == (
        "action.succeeded"
    )
    assert managed in bridge.channels
    assert managed in bridge._closing_channels
    assert bridge.channels[managed].binding is None
    assert not any(operation[0] == "part" for operation in irc.operations)

    await bridge._handle_managed_topic(
        IRCMessage(
            managed,
            "",
            "server",
            "agentwire:v1;account=trev;agent=bridge;backend=pi",
            command="332",
        )
    )

    assert backend.closed_sessions == [backend.created_session_id]
    assert managed not in bridge.channels
    assert [operation[0] for operation in irc.operations[-4:]] == [
        "ready",
        "flush",
        "topic",
        "part",
    ]


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
@pytest.mark.parametrize("kind", ["approval", "question"])
async def test_sync_recovers_pending_request_until_resolved(tmp_path: Path, kind: str) -> None:
    bridge, irc, backend = make_bridge(tmp_path, backend_name="omp")
    await bridge._handle_topic("#omp", "agentwire:v1;account=trev;agent=bridge;backend=omp")
    runtime = bridge.channels["#omp"]
    runtime.binding = ChannelBinding("omp", "s1", str(tmp_path))
    runtime.busy = True
    runtime.active_turn = "t1"
    question = Question("target", "Target", "Which target?", ("preview", "production"))
    decision = AsyncMock()
    if kind == "approval":
        backend.resolve_approval = decision  # type: ignore[method-assign]
        response = {"allow": False}
        expected_decision = ("backend-request", False)
    else:
        backend.resolve_question = decision  # type: ignore[method-assign]
        response = {"answers": [["preview"]]}
        expected_decision = ("backend-request", (question,), [["preview"]])
    await bridge._handle_backend_event(
        "#omp",
        BackendEvent(
            "approval" if kind == "approval" else "question",
            "omp",
            session_id="s1",
            turn_id="t1",
            item_id="tool-1",
            request_token="backend-request",
            text="Run the deployment command?",
            questions=(question,) if kind == "question" else (),
        ),
    )
    opened = irc.sent[-1][1]
    connected = HarnessState(epoch=bridge.epoch, session_id="s1")
    connected.apply(opened)
    irc.sent.clear()

    sync = new_envelope("sync.request", "action", "client", device="phone")
    await bridge._handle_action("#omp", sync)

    assert [(event.kind, event.reply) for _, event, _ in irc.sent[:2]] == [
        ("agent.hello", sync.id),
        ("channel.snapshot", sync.id),
    ]
    reopened = HarnessState()
    for _, event, _ in irc.sent:
        reopened.apply(Envelope.from_dict(event.to_dict()))
        connected.apply(event)
    request_id = next(iter(reopened.requests))
    assert request_id == opened.request_id
    assert reopened.requests == connected.requests
    assert (reopened.session_id, reopened.turn_id, reopened.busy) == ("s1", "t1", True)
    recovered = reopened.requests[request_id]
    if kind == "approval":
        assert recovered["summary"] == "Run the deployment command?"
        assert recovered["choices"] == ["allow_once", "deny"]
    else:
        assert recovered["questions"][0]["prompt"] == "Which target?"
        assert recovered["questions"][0]["options"] == ["preview", "production"]
    replay = irc.sent[-1][1]
    assert (replay.session_id, replay.turn_id, replay.item_id) == ("s1", "t1", "tool-1")
    assert replay.id != opened.id
    assert irc.sent[-1][2] is None
    decision.assert_not_awaited()
    irc.sent.clear()

    answer = new_envelope(
        "request.respond",
        "action",
        "client",
        epoch=reopened.epoch,
        device="phone",
        session_id=reopened.session_id,
        request_id=request_id,
        data=response,
    )
    await bridge._handle_action("#omp", answer)
    for _, event, _ in irc.sent:
        reopened.apply(event)
        connected.apply(event)
    assert reopened.requests == connected.requests == {}
    irc.sent.clear()

    await bridge._handle_action(
        "#omp", new_envelope("sync.request", "action", "client", device="phone")
    )
    for _, event, _ in irc.sent:
        reopened.apply(event)
        connected.apply(event)
    assert reopened.requests == connected.requests == {}
    decision.assert_awaited_once_with(*expected_decision)
    assert backend.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["resolved", "binding", "activation"])
async def test_sync_does_not_replay_requests_after_state_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    runtime = bridge.channels["#codex"]
    runtime.binding = ChannelBinding("codex", "s1", str(tmp_path))
    await bridge._handle_backend_event(
        "#codex",
        BackendEvent("approval", "codex", session_id="s1", request_token=7, text="Run command?"),
    )
    irc.sent.clear()
    sync = new_envelope("sync.request", "action", "client", device="phone")
    send_protocol = irc.send_protocol

    async def change_after_snapshot(
        channel: str, envelope: Envelope, preview: str | None = None
    ) -> None:
        await send_protocol(channel, envelope, preview)
        if envelope.kind == "channel.snapshot" and envelope.reply == sync.id:
            if change == "resolved":
                await bridge._handle_backend_event(
                    channel,
                    BackendEvent("request_resolved", "codex", session_id="s1", request_token=7),
                )
            elif change == "binding":
                await bridge._set_binding(channel, SessionSummary("s2", str(tmp_path), "Other"))
            else:
                runtime.activation = None

    monkeypatch.setattr(irc, "send_protocol", change_after_snapshot)
    await bridge._handle_action("#codex", sync)

    consumer = HarnessState()
    for _, event, _ in irc.sent:
        consumer.apply(event)
    assert consumer.requests == {}
    assert "request.opened" not in [event.kind for _, event, _ in irc.sent]


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
@pytest.mark.parametrize("action_kind", ["turn.steer", "turn.prompt"])
async def test_steering_announces_user_prompt_once(tmp_path: Path, action_kind: str) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    runtime = bridge.channels["#codex"]
    runtime.binding = ChannelBinding("codex", "s1", str(tmp_path))
    runtime.busy = True
    runtime.active_turn = "t1"
    runtime.settings["delivery"] = "steer"
    action = new_envelope(
        action_kind,
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        session_id="s1",
        data={"content": "focus on the failing test"},
    )
    await bridge._handle_action("#codex", action, "Alice")
    assert backend.steered == [("s1", "focus on the failing test")]
    prompts = [event for _, event, _ in irc.sent if event.kind == "user.prompt"]
    assert len(prompts) == 1
    assert prompts[0].turn_id == "t1"
    assert prompts[0].item_id == action.id
    assert prompts[0].data == {"content": "focus on the failing test"}


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
async def test_subagent_descriptions_apply_secret_filter(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.channels["#codex"].binding = ChannelBinding("codex", "s1", str(tmp_path))
    agents = [
        {
            "id": "child-1",
            "type": "agent",
            "status": "running",
            "description": "API_TOKEN=abcdefghijklmno",
            "isBackground": True,
        }
    ]
    await bridge._handle_backend_event(
        "#codex", BackendEvent("subagent_update", "codex", session_id="s1", data={"agents": agents})
    )
    event = irc.sent[-1][1]
    assert event.kind == "subagent.updated"
    assert event.data["agents"][0]["description"] == "[omitted]"
    assert "API_TOKEN" not in str(event.to_dict())
    assert agents[0]["description"].startswith("API_TOKEN")


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


@pytest.mark.asyncio
async def test_status_query_reads_accepted_queued_mutation_without_backend_work(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    started = asyncio.Event()
    release = asyncio.Event()

    async def stalled(_channel: str, _action: Envelope, _nick: str = "") -> None:
        started.set()
        await release.wait()

    bridge._dispatch_action = stalled  # type: ignore[method-assign]
    action_worker = asyncio.create_task(bridge._action_loop("#codex"))
    status_worker = asyncio.create_task(bridge._status_loop("#codex"))
    irc_worker = asyncio.create_task(bridge._irc_loop())
    try:
        mutation = new_envelope(
            "turn.prompt", "action", "client", epoch=bridge.epoch, device="phone"
        )
        await irc.incoming.put(
            IRCMessage(
                "#codex",
                "trev",
                "trev",
                "",
                MappingProxyType({PROTOCOL_TAG: encode_envelope(mutation)}),
                "TAGMSG",
            )
        )
        await started.wait()
        query = new_envelope(
            "action.status.request",
            "action",
            "client",
            epoch=bridge.epoch,
            device="phone",
            data={"actionId": mutation.id},
        )
        await irc.incoming.put(
            IRCMessage(
                "#codex",
                "trev",
                "trev",
                "",
                MappingProxyType({PROTOCOL_TAG: encode_envelope(query)}),
                "TAGMSG",
            )
        )
        status_events: list[Envelope] = []
        for _ in range(50):
            status_events = [
                event for _target, event, _preview in irc.sent if event.kind == "action.status"
            ]
            if status_events:
                break
            await asyncio.sleep(0.01)
        status = status_events[-1]
        assert status.kind == "action.status"
        assert status.data["status"] == "accepted"
        assert status.data["kind"] == "turn.prompt"
        release.set()
    finally:
        release.set()
        for worker in (action_worker, status_worker, irc_worker):
            worker.cancel()
        await asyncio.gather(action_worker, status_worker, irc_worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_status_query_cross_channel_is_scoped_to_configured_control_channel(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    mutation = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
    assert await bridge.state.claim_action(mutation, "#closed", "trev", "codex") is None
    await bridge.state.finish_action(mutation.id, "succeeded")
    query = new_envelope(
        "action.status.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"actionId": mutation.id, "channel": "#closed"},
    )
    await bridge._handle_action("#codex", query)
    assert irc.sent[-1][1].data["status"] == "succeeded"

    unknown = new_envelope(
        "action.status.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"actionId": mutation.id, "channel": "#wrong"},
    )
    await bridge._handle_action("#codex", unknown)
    assert irc.sent[-1][1].data == {"actionId": mutation.id, "status": "unknown"}


@pytest.mark.asyncio
async def test_status_query_does_not_publish_after_scope_changes(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    mutation = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
    await bridge.state.claim_action(mutation, "#codex", "trev", "codex")
    original_lookup = bridge.state.action_status

    async def lose_scope(*args: Any) -> Any:
        receipt = await original_lookup(*args)
        bridge.channels["#codex"].activation = None
        return receipt

    bridge.state.action_status = lose_scope  # type: ignore[method-assign]
    query = new_envelope(
        "action.status.request",
        "action",
        "client",
        epoch=bridge.epoch,
        device="phone",
        data={"actionId": mutation.id},
    )
    await bridge._handle_action("#codex", query)
    assert not any(event.reply == query.id for _channel, event, _preview in irc.sent)


@pytest.mark.asyncio
async def test_ingress_collision_is_suppressed_without_cross_scope_status_leak(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    action = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
    assert await bridge.state.claim_action(action, "#other", "trev", "codex") is None
    await bridge.state.finish_action(action.id, "succeeded")
    await bridge._ingest_action("#codex", action, "trev")
    event = irc.sent[-1][1]
    assert event.kind == "action.failed"
    assert event.reply == action.id
    assert event.data == {"message": "action UUID is already reserved"}
    assert bridge._action_queues["#codex"].empty()


@pytest.mark.asyncio
async def test_irc_ingress_correlates_malformed_status_lookup_failure(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    action_id = "00000000-0000-4000-8000-00000000a201"
    malformed = {
        "v": 1,
        "k": "action.status.request",
        "t": "action",
        "id": action_id,
        "at": 1,
        "inst": "client",
        "epoch": bridge.epoch,
        "device": "phone",
        "data": {"channel": "#codex"},
    }
    task = asyncio.create_task(bridge._irc_loop())
    try:
        await irc.incoming.put(
            IRCMessage(
                "#codex",
                "trev",
                "trev",
                "",
                MappingProxyType({PROTOCOL_TAG: json.dumps(malformed)}),
                "TAGMSG",
            )
        )
        for _ in range(20):
            if irc.sent:
                break
            await asyncio.sleep(0.01)
        event = irc.sent[-1][1]
        assert (event.kind, event.reply) == ("action.failed", action_id)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _send_irc_action(irc: FakeIRC, channel: str, action: Envelope) -> None:
    await irc.incoming.put(
        IRCMessage(
            channel,
            "trev",
            "trev",
            "",
            MappingProxyType({PROTOCOL_TAG: encode_envelope(action)}),
            "TAGMSG",
        )
    )


@pytest.mark.asyncio
async def test_irc_status_queue_is_bounded_without_blocking_mutation_worker(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    worker = asyncio.create_task(bridge._irc_loop())
    try:
        for number in range(33):
            await _send_irc_action(
                irc,
                "#codex",
                new_envelope(
                    "action.status.request",
                    "action",
                    "client",
                    epoch=bridge.epoch,
                    device="phone",
                    data={"actionId": f"00000000-0000-4000-8000-{number:012d}"},
                ),
            )
        for _ in range(50):
            if bridge._status_queues["#codex"].qsize() == 32 and irc.sent:
                break
            await asyncio.sleep(0.01)
        assert bridge._status_queues["#codex"].qsize() == 32
        overload = irc.sent[-1][1]
        assert overload.kind == "action.failed"
        assert overload.data == {"message": "too many action status requests"}
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_irc_ingress_terminalizes_claim_when_channel_is_removed_mid_claim(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    entered = asyncio.Event()
    release = asyncio.Event()
    original_claim = bridge.state.claim_action

    async def stalled_claim(*args: object, **kwargs: object) -> str | None:
        entered.set()
        await release.wait()
        return await original_claim(*args, **kwargs)  # type: ignore[arg-type]

    bridge.state.claim_action = stalled_claim  # type: ignore[method-assign]
    action = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
    worker = asyncio.create_task(bridge._irc_loop())
    try:
        await _send_irc_action(irc, "#codex", action)
        await entered.wait()
        await bridge._remove_channel("#codex")
        release.set()
        for _ in range(50):
            receipt = await bridge.state.action_status(action.id, "#codex", "trev", "codex")
            if receipt is not None:
                break
            await asyncio.sleep(0.01)
        assert receipt is not None and receipt.status == "failed"
    finally:
        release.set()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_irc_ingress_executes_after_accepted_send_failure(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    executed = asyncio.Event()

    async def dispatch(_channel: str, _action: Envelope, _nick: str = "") -> None:
        executed.set()

    original_send = irc.send_protocol

    async def fail_accepted(channel: str, envelope: Envelope, preview: str | None = None) -> None:
        if envelope.kind == "action.accepted":
            raise OSError("network write failed")
        await original_send(channel, envelope, preview)

    bridge._dispatch_action = dispatch  # type: ignore[method-assign]
    irc.send_protocol = fail_accepted  # type: ignore[method-assign]
    action_worker = asyncio.create_task(bridge._action_loop("#codex"))
    ingress_worker = asyncio.create_task(bridge._irc_loop())
    try:
        action = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
        await _send_irc_action(irc, "#codex", action)
        await asyncio.wait_for(executed.wait(), 1)
        await asyncio.wait_for(bridge._action_queues["#codex"].join(), 1)
        assert any(event.kind == "action.succeeded" for _target, event, _preview in irc.sent)
    finally:
        action_worker.cancel()
        ingress_worker.cancel()
        await asyncio.gather(action_worker, ingress_worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_marks_queued_receipt_failed_but_recovers_inflight_as_uncertain(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    started = asyncio.Event()

    async def stalled(_channel: str, _action: Envelope, _nick: str = "") -> None:
        started.set()
        await asyncio.Event().wait()

    bridge._dispatch_action = stalled  # type: ignore[method-assign]
    bridge._start_action_worker("#codex")
    ingress = asyncio.create_task(bridge._irc_loop())
    inflight = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
    queued = new_envelope("turn.steer", "action", "client", epoch=bridge.epoch, device="phone")
    try:
        await _send_irc_action(irc, "#codex", inflight)
        await started.wait()
        await _send_irc_action(irc, "#codex", queued)
        for _ in range(50):
            if bridge._action_queues["#codex"].qsize() == 1:
                break
            await asyncio.sleep(0.01)
        await bridge.close()
        recovered = StateStore(bridge.config.bridge.state_file)
        queued_receipt = await recovered.action_status(queued.id, "#codex", "trev", "codex")
        assert queued_receipt is not None and queued_receipt.status == "failed"
        assert await recovered.claim_action(inflight, "#codex", "trev", "codex") == "uncertain"
    finally:
        ingress.cancel()
        await asyncio.gather(ingress, return_exceptions=True)


@pytest.mark.asyncio
async def test_irc_ingress_terminalizes_claim_when_suspended_during_accepted_emit(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    entered = asyncio.Event()
    release = asyncio.Event()
    original_send = irc.send_protocol

    async def stalled_accept(channel: str, envelope: Envelope, preview: str | None = None) -> None:
        if envelope.kind == "action.accepted":
            entered.set()
            await release.wait()
        await original_send(channel, envelope, preview)

    irc.send_protocol = stalled_accept  # type: ignore[method-assign]
    action = new_envelope("turn.prompt", "action", "client", epoch=bridge.epoch, device="phone")
    worker = asyncio.create_task(bridge._irc_loop())
    try:
        await _send_irc_action(irc, "#codex", action)
        await entered.wait()
        await bridge._handle_topic("#codex", "ordinary topic")
        release.set()
        await asyncio.sleep(0.05)
        for _ in range(50):
            receipt = await bridge.state.action_status(action.id, "#codex", "trev", "codex")
            if receipt is not None:
                break
            await asyncio.sleep(0.01)
        assert receipt is not None and receipt.status == "failed"
        assert bridge._action_queues["#codex"].empty()
    finally:
        release.set()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


VOICE_URL = "https://files.invalid/note?signature=private-value"
VOICE_BODY = f"[voice 0:03 audio/ogg expires=2099-01-01T00:00:00Z] {VOICE_URL}#waveform=123"


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@contextlib.asynccontextmanager
async def _running_bridge(bridge: Bridge) -> AsyncIterator[None]:
    runner = asyncio.create_task(bridge.run())
    try:
        await _wait_until(
            lambda: (
                runner.done()
                or (
                    len(bridge._action_workers) == len(bridge.channels)
                    and (bridge.config.pm is None or bridge.config.pm.control_socket.exists())
                )
            )
        )
        if runner.done():
            await runner
            raise AssertionError("bridge exited before serving")
        yield
    finally:
        await bridge.close()
        await asyncio.gather(runner, return_exceptions=True)


async def _activate_bound(bridge: Bridge, channel: str, sid: str = "s1") -> None:
    backend = bridge.channels[channel].backend
    await bridge._handle_topic(channel, f"agentwire:v1;account=trev;agent=bridge;backend={backend}")
    bridge.channels[channel].binding = ChannelBinding(
        backend, sid, str(bridge.config.bridge.allowed_roots[0])
    )


def _voice_message(
    channel: str = "#codex",
    *,
    body: str = VOICE_BODY,
    account: str = "trev",
    tags: Mapping[str, str | None] | None = None,
    command: str = "PRIVMSG",
) -> IRCMessage:
    return IRCMessage(
        channel,
        account,
        "phone",
        body,
        MappingProxyType(dict(tags) if tags is not None else {AUDIO_TAG: "1"}),
        command,
    )


async def _irc_barrier(bridge: Bridge, irc: FakeIRC, channel: str = "#codex") -> None:
    action = new_envelope("sync.request", "action", "phone", device="phone")
    await _send_irc_action(irc, channel, action)
    await _wait_until(lambda: any(event.reply == action.id for _, event, _ in irc.sent))


async def _finish_voice(bridge: Bridge, irc: FakeIRC) -> None:
    await _irc_barrier(bridge, irc)
    assert bridge._voice_queue is not None
    await asyncio.wait_for(bridge._voice_queue.join(), 3)
    for queue in bridge._action_queues.values():
        await asyncio.wait_for(queue.join(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["idle", "steer", "queue"])
async def test_live_voice_uses_normal_prompt_delivery_and_visible_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delivery: str
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    transcribe = AsyncMock(return_value="Please explain the failing check.")
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        runtime = bridge.channels["#codex"]
        runtime.busy = delivery != "idle"
        runtime.settings["delivery"] = delivery
        irc.sent.clear()
        await irc.incoming.put(_voice_message())
        await _finish_voice(bridge, irc)
        content = "[voice] Please explain the failing check."
        action_id = voice_action_id("trev", "#codex", "codex", "s1", VOICE_URL)
        receipt = await bridge.state.action_status(action_id, "#codex", "trev", "codex")
        assert receipt is not None and receipt.status == "succeeded"
        initial_kind = "queue.item.added" if delivery == "queue" else "user.prompt"
        visible = [(event, preview) for _, event, preview in irc.sent if event.kind == initial_kind]
        assert [(event.data["content"], preview) for event, preview in visible] == [
            (content, content)
        ]
        assert visible[0][0].item_id == action_id
        assert visible[0][0].session_id == "s1"
        transcribe.assert_awaited_once_with(
            VoiceMessage(3, "audio/ogg", VOICE_URL), bridge.config.voice.model_path
        )
        if delivery == "steer":
            assert backend.steered == [("s1", content)]
            assert backend.sent == []
        elif delivery == "queue":
            assert backend.sent == []
            assert not any(event.kind == "user.prompt" for _, event, _ in irc.sent)
            await bridge._handle_backend_event(
                "#codex", BackendEvent("turn_done", "codex", session_id="s1")
            )
            assert backend.sent == [("s1", content)]
            assert await bridge.state.list_queue("#codex", "s1") == []
        else:
            assert backend.sent == [("s1", content)]


@pytest.mark.asyncio
async def test_voice_requires_marker_command_owner_and_static_active_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#codex": "codex", "#inactive": "codex", "#unbound": "codex"},
        voice=VoiceConfig(tmp_path / "model.bin"),
    )
    transcribe = AsyncMock(return_value="Allowed")
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        await bridge._handle_topic(
            "#unbound", "agentwire:v1;account=trev;agent=bridge;backend=codex"
        )
        bridge.channels["#inactive"].binding = ChannelBinding("codex", "s1", str(tmp_path))
        bridge._install_channel("#dynamic", "codex")
        await _activate_bound(bridge, "#dynamic")
        with caplog.at_level(logging.WARNING, logger="agentwire.bridge"):
            for message in (
                _voice_message(tags={}),
                _voice_message(tags={AUDIO_TAG: "0"}),
                _voice_message(tags={AUDIO_TAG: ""}),
                _voice_message(tags={AUDIO_TAG: "true"}),
                _voice_message(command="NOTICE"),
                _voice_message(command="TAGMSG"),
                _voice_message(account=""),
                _voice_message(account="mallory"),
                _voice_message(tags={AUDIO_TAG: "1", "draft/playback": "private-tag"}),
                _voice_message(tags={AUDIO_TAG: "1", "znc.in/playback": "private-tag"}),
                _voice_message("#inactive"),
                _voice_message("#unbound"),
                _voice_message("#dynamic"),
                _voice_message("#unknown"),
                _voice_message(body="ordinary channel conversation"),
            ):
                await irc.incoming.put(message)
            await _finish_voice(bridge, irc)
        transcribe.assert_not_awaited()
        assert backend.sent == []
        assert backend.steered == []
        assert not any(event.kind == "action.accepted" for _, event, _ in irc.sent)
        logs = caplog.text
        assert "authenticated channel owner" in logs and "history playback" in logs
        assert VOICE_URL not in logs and "private-tag" not in logs
        runtime = bridge.channels["#codex"]
        activation = runtime.activation
        assert activation is not None
        runtime.activation = replace(activation, account="mallory")
        await irc.incoming.put(_voice_message(account="mallory"))
        await _irc_barrier(bridge, irc, "#unbound")
        assert bridge._voice_queue is not None
        await asyncio.wait_for(bridge._voice_queue.join(), 3)
        transcribe.assert_not_awaited()
        runtime.activation = activation
        await irc.incoming.put(_voice_message())
        await _finish_voice(bridge, irc)
        assert backend.sent == [("s1", "[voice] Allowed")]


@pytest.mark.asyncio
@pytest.mark.parametrize("playback", ["draft/playback", "znc.in/playback"])
async def test_playback_is_rejected_before_voice_or_protocol_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, playback: str
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    transcribe = AsyncMock()
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        irc.sent.clear()
        for command in ("PRIVMSG", "TAGMSG"):
            await irc.incoming.put(
                _voice_message(
                    command=command,
                    body=VOICE_BODY.replace("[voice ", "[voice encrypted "),
                    tags={AUDIO_TAG: "1", PROTOCOL_TAG: "malformed private payload", playback: ""},
                )
            )
        await _finish_voice(bridge, irc)
        assert not any(event.kind == "action.failed" for _, event, _ in irc.sent)
        transcribe.assert_not_awaited()
        assert backend.sent == []


@pytest.mark.asyncio
async def test_absent_voice_config_never_transcribes_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    transcribe = AsyncMock()
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        await irc.incoming.put(_voice_message())
        await _irc_barrier(bridge, irc)
        transcribe.assert_not_awaited()
        assert backend.sent == []


@pytest.mark.asyncio
async def test_voice_errors_are_safe_visible_and_preclaim_failures_remain_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    transcribe = AsyncMock(
        side_effect=[VoiceError("audio download failed"), RuntimeError(VOICE_URL), "Recovered"]
    )
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        irc.sent.clear()
        await irc.incoming.put(
            _voice_message(body=VOICE_BODY.replace("[voice ", "[voice encrypted "))
        )
        await _finish_voice(bridge, irc)
        transcribe.assert_not_awaited()
        action_id = voice_action_id("trev", "#codex", "codex", "s1", VOICE_URL)
        for category in ("audio download failed", "unexpected voice failure"):
            await irc.incoming.put(_voice_message())
            await _finish_voice(bridge, irc)
            failures = [event for _, event, _ in irc.sent if event.kind == "action.failed"]
            assert failures[-1].reply is None
            assert failures[-1].data == {"message": f"Voice transcription failed: {category}"}
            assert await bridge.state.action_status(action_id, "#codex", "trev", "codex") is None
        await irc.incoming.put(_voice_message())
        await _finish_voice(bridge, irc)
        assert backend.sent == [("s1", "[voice] Recovered")]
        assert transcribe.await_count == 3
        assert VOICE_URL not in repr(irc.sent) + repr(irc.notices) + caplog.text


@pytest.mark.asyncio
async def test_voice_retry_suppresses_inflight_cpu_and_durable_duplicate_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(_message: VoiceMessage, _model: Path) -> str:
        entered.set()
        await release.wait()
        return "Once"

    transcription = AsyncMock(side_effect=transcribe)
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcription)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        await irc.incoming.put(_voice_message())
        await asyncio.wait_for(entered.wait(), 3)
        await irc.incoming.put(_voice_message(body=VOICE_BODY.replace("waveform=123", "key=other")))
        await _irc_barrier(bridge, irc)
        assert bridge._voice_queue is not None and bridge._voice_queue.empty()
        bridge.epoch = "rotated-live-epoch"
        release.set()
        await _finish_voice(bridge, irc)
        await irc.incoming.put(_voice_message())
        await _finish_voice(bridge, irc)
        transcription.assert_awaited_once()
        assert backend.sent == [("s1", "[voice] Once")]


@pytest.mark.asyncio
async def test_voice_queue_full_does_not_block_irc_and_rejected_note_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(message: VoiceMessage, _model: Path) -> str:
        entered.set()
        await release.wait()
        return message.fetch_url.rsplit("/", 1)[-1]

    transcription = AsyncMock(side_effect=transcribe)
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcription)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        runtime = bridge.channels["#codex"]
        runtime.busy = True
        runtime.settings["delivery"] = "steer"
        await irc.incoming.put(
            _voice_message(body="[voice 0:01 audio/ogg] https://files.invalid/0")
        )
        await asyncio.wait_for(entered.wait(), 3)
        for number in range(1, 6):
            await irc.incoming.put(
                _voice_message(body=f"[voice 0:01 audio/ogg] https://files.invalid/{number}")
            )
        await _irc_barrier(bridge, irc)
        assert bridge._voice_queue is not None and bridge._voice_queue.qsize() == 4
        failures = [event for _, event, _ in irc.sent if event.kind == "action.failed"]
        assert [event.data for event in failures] == [
            {"message": "Voice transcription failed: voice queue full"}
        ]
        rejected = voice_action_id("trev", "#codex", "codex", "s1", "https://files.invalid/5")
        assert await bridge.state.action_status(rejected, "#codex", "trev", "codex") is None
        release.set()
        await _finish_voice(bridge, irc)
        assert backend.steered == [("s1", f"[voice] {number}") for number in range(5)]
        await irc.incoming.put(
            _voice_message(body="[voice 0:01 audio/ogg] https://files.invalid/5")
        )
        await _finish_voice(bridge, irc)
        assert backend.steered[-1] == ("s1", "[voice] 5")
        assert transcription.await_count == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["accepted", "succeeded", "failed", "uncertain"])
async def test_voice_restart_uses_scoped_retained_action_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    first, _, _ = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    action_id = voice_action_id("trev", "#codex", "codex", "s1", VOICE_URL)
    action = new_envelope(
        "turn.prompt",
        "action",
        first.instance,
        id=action_id,
        epoch=first.epoch,
        device="bridge-voice",
        session_id="s1",
        data={"content": "[voice] Already accepted"},
    )
    await first.state.set("#codex", ChannelBinding("codex", "s1", str(tmp_path)))
    await first.state.claim_action(action, "#codex", "trev", "codex")
    if status != "accepted":
        await first.state.finish_action(action_id, status)
    await first.close()
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    transcribe = AsyncMock(return_value="Different recording")
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        assert backend.attached_sessions == ["s1"]
        await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
        await irc.incoming.put(_voice_message())
        await _finish_voice(bridge, irc)
        transcribe.assert_not_awaited()
        assert backend.sent == []
        receipt = await bridge.state.action_status(action_id, "#codex", "trev", "codex")
        assert receipt is not None
        assert receipt.status == ("uncertain" if status == "accepted" else status)
        await irc.incoming.put(_voice_message(body=VOICE_BODY.replace("/note?", "/different?")))
        await _finish_voice(bridge, irc)
        assert backend.sent == [("s1", "[voice] Different recording")]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["sid", "activation", "runtime"])
async def test_voice_revalidates_capture_after_transcription_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(_message: VoiceMessage, _model: Path) -> str:
        entered.set()
        await release.wait()
        return "Pinned"

    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        await irc.incoming.put(_voice_message())
        await asyncio.wait_for(entered.wait(), 3)
        sid = "s2" if change == "sid" else "s1"
        if change == "runtime":
            await bridge._remove_channel("#codex")
            bridge._install_channel("#codex", "codex")
        await _activate_bound(bridge, "#codex", sid)
        release.set()
        await _finish_voice(bridge, irc)
        assert backend.sent == []
        action_id = voice_action_id("trev", "#codex", "codex", "s1", VOICE_URL)
        assert await bridge.state.action_status(action_id, "#codex", "trev", "codex") is None
        failures = [event for _, event, _ in irc.sent if event.kind == "action.failed"]
        assert failures[-1].data == {"message": "Voice transcription failed: binding changed"}
        await irc.incoming.put(_voice_message())
        await _finish_voice(bridge, irc)
        assert backend.sent == [(sid, "[voice] Pinned")]


@pytest.mark.asyncio
async def test_bridge_shutdown_reaps_transcription_before_state_close_and_leaves_no_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(tmp_path, voice=VoiceConfig(tmp_path / "model.bin"))
    entered, reaped = asyncio.Event(), asyncio.Event()

    async def transcribe(_message: VoiceMessage, _model: Path) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            reaped.set()
        raise AssertionError("cancelled transcription returned")

    original_close = bridge.state.close

    async def close_state() -> None:
        assert reaped.is_set()
        await original_close()

    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcribe)
    monkeypatch.setattr(bridge.state, "close", close_state)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex")
        await irc.incoming.put(_voice_message())
        await asyncio.wait_for(entered.wait(), 3)
        await irc.incoming.put(_voice_message(body=VOICE_BODY.replace("/note?", "/queued?")))
        await _irc_barrier(bridge, irc)
        await bridge.close()
        assert reaped.is_set()
        assert backend.sent == []
        assert bridge._voice_queue is not None and bridge._voice_queue.empty()
        assert bridge._voice_inflight == set()
    recovered = StateStore(bridge.config.bridge.state_file)
    try:
        action_id = voice_action_id("trev", "#codex", "codex", "s1", VOICE_URL)
        assert await recovered.action_status(action_id, "#codex", "trev", "codex") is None
    finally:
        await recovered.close()


def _pm_bridge(tmp_path: Path) -> tuple[Bridge, FakeIRC, FakeBackend]:
    return make_bridge(
        tmp_path,
        channels={"#pm": "codex", "#worker": "codex", "#other": "codex"},
        pm=PMConfig(
            tmp_path / "c.sock",
            "#pm",
            MappingProxyType({"touch-hockey": "#worker", "motd-dev": "#other"}),
        ),
    )


@pytest.mark.parametrize("command", ["run", "stack"])
@pytest.mark.parametrize("manual", [False, True])
def test_runtime_cli_manual_approval_controls_pm_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, manual: bool
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    decision = AsyncMock()
    backend.resolve_approval = decision  # type: ignore[method-assign]

    async def launch(config: Config) -> None:
        bridge.config = config
        try:
            await _activate_bound(bridge, "#pm")
            await bridge._handle_backend_event(
                "#pm",
                BackendEvent("approval", "codex", session_id="s1", request_token=0),
            )
            await bridge._handle_action(
                "#pm", new_envelope("sync.request", "action", "client", device="phone")
            )
            client = HarnessState()
            for _, event, _ in irc.sent:
                client.apply(event)
            if manual:
                request = next(iter(client.requests.values()))
                assert request["choices"] == ["allow_once", "deny"]
                decision.assert_not_awaited()
            else:
                assert client.requests == {}
                assert bridge.channels["#pm"].requests == {}
                decision.assert_awaited_once_with(0, True)
        finally:
            await bridge.close()

    monkeypatch.setattr(cli, "load_config", lambda _path: bridge.config)
    monkeypatch.setattr(cli, "run_bridge", launch)
    monkeypatch.setattr(cli, "run_stack", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["agentwire", "--config", str(bridge.config.path), command]
        + (["--manual-approval"] if manual else []),
    )
    cli.main()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_name", "boundary"),
    [
        ("codex", "no-pm"),
        ("codex", "question"),
        ("omp", "inactive"),
        ("pi", "unbound"),
        ("claude", "non-pm"),
        ("pi", "non-pm"),
        ("omp", "non-pm"),
        ("codex", "closing"),
    ],
)
async def test_pm_auto_approval_leaves_other_requests_manual(
    tmp_path: Path, backend_name: str, boundary: str
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        backend_name=backend_name,
        channels={
            "#pm": backend_name,
            "#worker": backend_name,
            f"#{backend_name}": backend_name,
        },
        pm=(
            PMConfig(tmp_path / "c.sock", "#pm", MappingProxyType({"project": "#worker"}))
            if boundary != "no-pm"
            else None
        ),
    )
    decision = AsyncMock()
    answer = AsyncMock()
    backend.resolve_approval = decision  # type: ignore[method-assign]
    backend.resolve_question = answer  # type: ignore[method-assign]
    channel = f"#{backend_name}" if boundary == "non-pm" else "#worker"
    try:
        await _activate_bound(bridge, channel)
        runtime = bridge.channels[channel]
        if boundary == "unbound":
            runtime.binding = None
        elif boundary == "closing":
            runtime.closing_session = True
        kind = "question" if boundary == "question" else "approval"
        await bridge._handle_backend_event(
            channel,
            BackendEvent(
                kind,
                backend_name,
                session_id="other" if boundary == "inactive" else "s1",
                request_token=7,
                questions=(Question("target", "Target", "Which target?", ("preview",)),)
                if kind == "question"
                else (),
            ),
        )
        opened = irc.sent[-1][1]
        assert opened.kind == "request.opened"
        assert opened.request_id in runtime.requests
        assert opened.data["type"] == kind
        assert opened.data["inactive"] is (boundary in {"inactive", "unbound"})
        decision.assert_not_awaited()
        answer.assert_not_awaited()
        assert backend.sent == []
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_pm_auto_approval_resolves_once_across_sync_and_manual_races(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        backend_name="omp",
        channels={"#pm": "omp", "#worker": "omp"},
        pm=PMConfig(tmp_path / "c.sock", "#pm", MappingProxyType({"project": "#worker"})),
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def resolve(_token: str | int, _allow: bool) -> None:
        entered.set()
        await release.wait()

    decision = AsyncMock(side_effect=resolve)
    backend.resolve_approval = decision  # type: ignore[method-assign]
    opening: asyncio.Task[None] | None = None
    try:
        await _activate_bound(bridge, "#worker")
        runtime = bridge.channels["#worker"]
        runtime.busy, runtime.active_turn = True, "t1"
        queued = await bridge.state.enqueue("next", "#worker", "s1", "Next prompt", 2)
        opening = asyncio.create_task(
            bridge._handle_backend_event(
                "#worker",
                BackendEvent("approval", "omp", session_id="s1", turn_id="t1", request_token=0),
            )
        )
        await asyncio.wait_for(entered.wait(), 3)
        irc.sent.clear()
        await bridge._handle_action(
            "#worker", new_envelope("sync.request", "action", "client", device="phone")
        )
        client = HarnessState()
        for _, event, _ in irc.sent:
            client.apply(event)
        request_id = next(iter(client.requests))
        irc.sent.clear()
        for kind in ("request.respond", "request.skip"):
            action = new_envelope(
                kind,
                "action",
                "client",
                epoch=bridge.epoch,
                device="phone",
                session_id="s1",
                request_id=request_id,
                data={"allow": False} if kind == "request.respond" else {},
            )
            await bridge._handle_action("#worker", action)
            assert [event.kind for _, event, _ in irc.sent if event.reply == action.id] == [
                "action.accepted",
                "action.failed",
            ]
        assert request_id in runtime.requests
        decision.assert_awaited_once_with(0, True)
        release.set()
        await asyncio.wait_for(opening, 3)
        await bridge._handle_backend_event(
            "#worker",
            BackendEvent("request_resolved", "omp", session_id="s1", request_token=0),
        )
        for _, event, _ in irc.sent:
            client.apply(event)
        assert client.requests == runtime.requests == {}
        assert [
            event.request_id for _, event, _ in irc.sent if event.kind == "request.resolved"
        ] == [request_id]
        assert (runtime.busy, runtime.active_turn) == (True, "t1")
        assert await bridge.state.list_queue("#worker", "s1") == [queued]
        assert backend.sent == []

        irc.sent.clear()
        await bridge._handle_action(
            "#worker", new_envelope("sync.request", "action", "client", device="phone")
        )
        reopened = HarnessState()
        for _, event, _ in irc.sent:
            reopened.apply(event)
        assert reopened.requests == {}
        decision.assert_awaited_once_with(0, True)
        await bridge._handle_backend_event(
            "#worker", BackendEvent("turn_done", "omp", session_id="s1", turn_id="t1")
        )
        assert backend.sent == [("s1", "Next prompt")]
    finally:
        release.set()
        if opening is not None:
            await asyncio.gather(opening, return_exceptions=True)
        await bridge.close()


@pytest.mark.asyncio
async def test_pm_auto_approval_failure_remains_recoverable_and_safe(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    sensitive = "PRIVATE backend transport details"
    decision = AsyncMock(side_effect=RuntimeError(sensitive))
    backend.resolve_approval = decision  # type: ignore[method-assign]
    try:
        await _activate_bound(bridge, "#worker")
        irc.sent.clear()
        with caplog.at_level(logging.WARNING, logger="agentwire.bridge"):
            await bridge._handle_backend_event(
                "#worker",
                BackendEvent(
                    "approval", "codex", session_id="s1", request_token=9, text="Run command?"
                ),
            )
        opened = irc.sent[-1][1]
        request_id = opened.request_id
        assert opened.kind == "request.opened"
        assert "manual" in irc.sent[-1][2].lower()
        assert any(
            record.levelno == logging.WARNING and record.name == "agentwire.bridge"
            for record in caplog.records
        )
        assert sensitive not in caplog.text + repr(irc.sent) + repr(irc.notices)
        assert not any(event.kind == "request.resolved" for _, event, _ in irc.sent)
        irc.sent.clear()
        await bridge._handle_action(
            "#worker", new_envelope("sync.request", "action", "client", device="phone")
        )
        client = HarnessState()
        for _, event, _ in irc.sent:
            client.apply(event)
        assert client.requests[request_id]["choices"] == ["allow_once", "deny"]
        assert irc.sent[-1][1].id != opened.id
        decision.assert_awaited_once_with(9, True)
        decision.side_effect = None
        irc.sent.clear()
        await bridge._handle_action(
            "#worker",
            new_envelope(
                "request.respond",
                "action",
                "client",
                epoch=bridge.epoch,
                device="phone",
                session_id="s1",
                request_id=request_id,
                data={"allow": False},
            ),
        )
        for _, event, _ in irc.sent:
            client.apply(event)
        assert client.requests == bridge.channels["#worker"].requests == {}
        assert decision.await_count == 2
        decision.assert_awaited_with(9, False)
        assert backend.sent == []
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_control_socket_routes_only_scoped_delegations_and_worker_reports(
    tmp_path: Path,
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    async with _running_bridge(bridge):
        for channel, sid in (("#pm", "coordinator"), ("#worker", "worker"), ("#other", "other")):
            await _activate_bound(bridge, channel, sid)
        irc.sent.clear()
        path = bridge.config.pm.control_socket
        delegated = await send_control_request(
            path, DelegateRequest("touch-hockey", "TH-0001", "Read the project package name.")
        )
        await asyncio.wait_for(bridge._action_queues["#worker"].join(), 3)
        reported = await send_control_request(
            path, ReportRequest("touch-hockey", "TH-0001", "done", "Package is touch-hockey.")
        )
        await asyncio.wait_for(bridge._action_queues["#pm"].join(), 3)
        delegation = (
            "[pm delegation project=touch-hockey task=TH-0001]\nRead the project package name.\n\n"
            "Do not edit the PM board. Report exactly one done or blocked result "
            "through the Agentwire PM report operation."
        )
        report_text = (
            "[worker report project=touch-hockey task=TH-0001 status=done]\n"
            "Package is touch-hockey."
        )
        assert backend.sent == [("worker", delegation), ("coordinator", report_text)]
        prompts = [
            (channel, event, preview)
            for channel, event, preview in irc.sent
            if event.kind == "user.prompt"
        ]
        assert [(channel, event.item_id, preview) for channel, event, preview in prompts] == [
            ("#worker", delegated, delegation),
            ("#pm", reported, report_text),
        ]
        for action_id, channel in ((delegated, "#worker"), (reported, "#pm")):
            receipt = await bridge.state.action_status(action_id, channel, "trev", "codex")
            assert receipt is not None and receipt.status == "succeeded"
            assert await bridge.state.action_status(action_id, "#other", "trev", "codex") is None
        bridge.channels["#pm"].busy = False
        unknown_task = await send_control_request(
            path, ReportRequest("touch-hockey", "TH-9999", "blocked", "Needs PM reconciliation.")
        )
        await asyncio.wait_for(bridge._action_queues["#pm"].join(), 3)
        assert backend.sent[-1] == (
            "coordinator",
            "[worker report project=touch-hockey task=TH-9999 status=blocked]\n"
            "Needs PM reconciliation.",
        )
        assert unknown_task not in {delegated, reported}


@pytest.mark.asyncio
async def test_control_policy_rejections_do_not_claim_or_prompt_and_listener_survives(
    tmp_path: Path,
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    async with _running_bridge(bridge):
        path = bridge.config.pm.control_socket
        irc.sent.clear()
        for request, category in (
            (DelegateRequest("unmapped", "TH-0001", "Denied"), "unknown project"),
            (ReportRequest("unmapped", "TH-0001", "done", "Denied"), "unknown project"),
            (DelegateRequest("touch-hockey", "TH-0001", "Denied"), "channel unavailable"),
            (ReportRequest("touch-hockey", "TH-0001", "blocked", "Denied"), "channel unavailable"),
        ):
            with pytest.raises(ControlError, match=f"^{category}$"):
                await send_control_request(path, request)
        await bridge._handle_topic(
            "#worker", "agentwire:v1;account=trev;agent=bridge;backend=codex"
        )
        with pytest.raises(ControlError, match="^channel unavailable$"):
            await send_control_request(path, DelegateRequest("touch-hockey", "TH-0001", "Unbound"))
        assert backend.sent == []
        assert not any(event.kind == "action.accepted" for _, event, _ in irc.sent)
        await _activate_bound(bridge, "#worker", "worker")
        await send_control_request(path, DelegateRequest("touch-hockey", "TH-0001", "Allowed"))
        await asyncio.wait_for(bridge._action_queues["#worker"].join(), 3)
        assert backend.sent[0][0] == "worker"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delegate", "report"])
async def test_control_bounds_final_wrapped_utf8_content_before_durable_ingress(
    tmp_path: Path, operation: str
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    if operation == "delegate":
        prefix = "[pm delegation project=touch-hockey task=TH-0001]\n"
        suffix = (
            "\n\nDo not edit the PM board. Report exactly one done or blocked result "
            "through the Agentwire PM report operation."
        )
        channel, sid = "#worker", "worker"
    else:
        prefix = "[worker report project=touch-hockey task=TH-0001 status=done]\n"
        suffix = ""
        channel, sid = "#pm", "coordinator"
    text_bytes = MAX_CONTENT_BYTES - len((prefix + suffix).encode())
    text = "é" * (text_bytes // 2) + "x" * (text_bytes % 2)

    def request(content: str) -> DelegateRequest | ReportRequest:
        if operation == "delegate":
            return DelegateRequest("touch-hockey", "TH-0001", content)
        return ReportRequest("touch-hockey", "TH-0001", "done", content)

    async with _running_bridge(bridge):
        await _activate_bound(bridge, channel, sid)
        irc.sent.clear()
        with pytest.raises(ControlError, match="^content too large$"):
            await send_control_request(bridge.config.pm.control_socket, request(text + "x"))
        assert backend.sent == []
        assert not any(event.kind == "action.accepted" for _, event, _ in irc.sent)
        await send_control_request(bridge.config.pm.control_socket, request(text))
        await asyncio.wait_for(bridge._action_queues[channel].join(), 3)
        assert backend.sent == [(sid, prefix + text + suffix)]
        assert len(backend.sent[0][1].encode()) == MAX_CONTENT_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["claim", "dispatch"])
async def test_control_binding_switch_fails_the_pinned_action_not_the_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    if phase == "claim":
        original_claim = bridge.state.claim_action

        async def claim(*args: Any, **kwargs: Any) -> str | None:
            entered.set()
            await release.wait()
            return await original_claim(*args, **kwargs)

        monkeypatch.setattr(bridge.state, "claim_action", claim)
    else:
        original_dispatch = bridge._dispatch_action

        async def dispatch(channel: str, action: Envelope, nick: str = "") -> None:
            entered.set()
            await release.wait()
            await original_dispatch(channel, action, nick)

        monkeypatch.setattr(bridge, "_dispatch_action", dispatch)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#worker", "original")
        sending = asyncio.create_task(
            send_control_request(
                bridge.config.pm.control_socket,
                DelegateRequest("touch-hockey", "TH-0001", "Stay with the original session."),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if phase == "dispatch":
                # A receipt acknowledges ingress, not successful task execution.
                action_id = await asyncio.wait_for(sending, 3)
                receipt = await bridge.state.action_status(action_id, "#worker", "trev", "codex")
                assert receipt is not None and receipt.status == "accepted"
            bridge.channels["#worker"].binding = ChannelBinding(
                "codex", "replacement", str(tmp_path)
            )
            release.set()
            action_id = await asyncio.wait_for(sending, 3)
            await asyncio.wait_for(bridge._action_queues["#worker"].join(), 3)
            receipt = await bridge.state.action_status(action_id, "#worker", "trev", "codex")
            assert receipt is not None and receipt.status == "failed"
            assert backend.sent == []
            assert any(
                event.kind == "action.failed" and event.reply == action_id
                for _, event, _ in irc.sent
            )
        finally:
            release.set()
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_activation_race_does_not_acknowledge_unqueued_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    original_send = irc.send_protocol

    async def send(channel: str, envelope: Envelope, preview: str | None = None) -> None:
        if envelope.kind == "action.accepted":
            entered.set()
            await release.wait()
        await original_send(channel, envelope, preview)

    monkeypatch.setattr(irc, "send_protocol", send)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#worker", "worker")
        sending = asyncio.create_task(
            send_control_request(
                bridge.config.pm.control_socket,
                DelegateRequest("touch-hockey", "TH-0001", "Do not cross activation."),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await bridge._handle_topic("#worker", "ordinary topic")
            release.set()
            with pytest.raises(ControlError, match="^binding changed$"):
                await asyncio.wait_for(sending, 3)
            action_id = next(
                event.reply for _, event, _ in irc.sent if event.kind == "action.accepted"
            )
            receipt = await bridge.state.action_status(action_id, "#worker", "trev", "codex")
            assert receipt is not None and receipt.status == "failed"
            assert backend.sent == []
            assert bridge._action_queues["#worker"].empty()
        finally:
            release.set()
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.asyncio
async def test_bridge_control_starts_after_restore_and_workers_and_closes_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, _, backend = _pm_bridge(tmp_path)
    await bridge.state.set("#worker", ChannelBinding("codex", "restored", str(tmp_path)))
    order: list[str] = []
    original_restore = bridge._restore_bindings
    original_start = bridge._control_server.start
    original_close = bridge._control_server.close
    original_state_close = bridge.state.close

    async def irc_ready(_timeout: float = 30) -> None:
        order.append("irc-ready")

    async def backend_ready(_timeout: float = 30) -> None:
        order.append("backend-ready")

    async def restore() -> None:
        await original_restore()
        order.append("restored")

    async def control_start() -> None:
        assert {"irc-ready", "backend-ready", "restored"} <= set(order)
        assert backend.attached_sessions == ["restored"]
        assert set(bridge._action_workers) == set(bridge.channels)
        assert set(bridge._status_workers) == set(bridge.channels)
        await original_start()
        order.append("control-start")

    async def control_close() -> None:
        order.append("control-close")
        assert all(not task.done() for task in bridge._action_workers.values())
        await original_close()

    async def backend_close() -> None:
        assert "control-close" in order
        order.append("backend-close")

    async def state_close() -> None:
        assert "control-close" in order
        assert all(task.done() for task in bridge._action_workers.values())
        order.append("state-close")
        await original_state_close()

    monkeypatch.setattr(bridge.irc, "wait_ready", irc_ready)
    monkeypatch.setattr(backend, "wait_ready", backend_ready)
    monkeypatch.setattr(bridge, "_restore_bindings", restore)
    monkeypatch.setattr(bridge._control_server, "start", control_start)
    monkeypatch.setattr(bridge._control_server, "close", control_close)
    monkeypatch.setattr(backend, "close", backend_close)
    monkeypatch.setattr(bridge.state, "close", state_close)
    async with _running_bridge(bridge):
        await _wait_until(lambda: "control-start" in order)
        await bridge.close()
        assert not bridge.config.pm.control_socket.exists()
        assert (
            order.index("control-close") < order.index("backend-close") < order.index("state-close")
        )


@pytest.mark.asyncio
async def test_ingest_boolean_means_descriptor_was_handed_to_a_live_queue(tmp_path: Path) -> None:
    bridge, _, _ = make_bridge(tmp_path)
    try:
        await _activate_bound(bridge, "#codex")
        mutation = new_envelope(
            "turn.prompt",
            "action",
            "phone",
            device="phone",
            epoch=bridge.epoch,
            session_id="s1",
            data={"content": "Queued"},
        )
        assert await bridge._ingest_action("#codex", mutation) is True
        assert await bridge._ingest_action("#codex", mutation) is False
        assert bridge._action_queues["#codex"].qsize() == 1
        for _ in range(32):
            query = new_envelope(
                "action.status.request",
                "action",
                "phone",
                device="phone",
                epoch=bridge.epoch,
                data={"actionId": mutation.id},
            )
            assert await bridge._ingest_action("#codex", query) is True
        assert await bridge._ingest_action("#codex", query) is False
        await bridge.close()
        assert await bridge._ingest_action("#codex", mutation) is False
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_queued_voice_checks_binding_before_spending_more_transcription_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#codex": "codex", "#second": "codex"},
        voice=VoiceConfig(tmp_path / "model.bin"),
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def transcribe(_message: VoiceMessage, _model: Path) -> str:
        entered.set()
        await release.wait()
        return "First only"

    transcription = AsyncMock(side_effect=transcribe)
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", transcription)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex", "first")
        await _activate_bound(bridge, "#second", "second")
        await irc.incoming.put(_voice_message())
        await asyncio.wait_for(entered.wait(), 3)
        await irc.incoming.put(_voice_message("#second"))
        await _irc_barrier(bridge, irc)
        bridge.channels["#second"].binding = ChannelBinding("codex", "replacement", str(tmp_path))
        release.set()
        await _finish_voice(bridge, irc)
        transcription.assert_awaited_once()
        assert backend.sent == [("first", "[voice] First only")]
        second_id = voice_action_id("trev", "#second", "codex", "second", VOICE_URL)
        assert await bridge.state.action_status(second_id, "#second", "trev", "codex") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["steer", "queue"])
async def test_busy_local_control_prompts_have_one_readable_initial_event(
    tmp_path: Path, delivery: str
) -> None:
    bridge, irc, backend = _pm_bridge(tmp_path)
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#pm", "coordinator")
        runtime = bridge.channels["#pm"]
        runtime.busy = True
        runtime.settings["delivery"] = delivery
        irc.sent.clear()
        action_id = await send_control_request(
            bridge.config.pm.control_socket,
            ReportRequest("touch-hockey", "TH-0001", "blocked", "Need a decision."),
        )
        await asyncio.wait_for(bridge._action_queues["#pm"].join(), 3)
        content = (
            "[worker report project=touch-hockey task=TH-0001 status=blocked]\nNeed a decision."
        )
        events = [
            (event, preview)
            for _, event, preview in irc.sent
            if event.kind in {"user.prompt", "queue.item.added"}
        ]
        assert [(event.kind, event.item_id, preview) for event, preview in events] == [
            ("user.prompt" if delivery == "steer" else "queue.item.added", action_id, content)
        ]
        if delivery == "steer":
            assert backend.steered == [("coordinator", content)]
        else:
            assert backend.sent == []
            assert [item.text for item in await bridge.state.list_queue("#pm", "coordinator")] == [
                content
            ]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["voice", "control"])
async def test_queued_local_prompt_cannot_drain_into_a_replacement_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    bridge, irc, backend = make_bridge(
        tmp_path,
        channels={"#codex": "codex", "#pm": "codex"},
        voice=VoiceConfig(tmp_path / "model.bin"),
        pm=PMConfig(tmp_path / "c.sock", "#pm", MappingProxyType({"touch-hockey": "#codex"})),
    )
    monkeypatch.setattr("agentwire.bridge.transcribe_voice", AsyncMock(return_value="Pinned queue"))
    async with _running_bridge(bridge):
        await _activate_bound(bridge, "#codex", "original")
        await _activate_bound(bridge, "#pm", "original")
        channel = "#codex" if source == "voice" else "#pm"
        runtime = bridge.channels[channel]
        runtime.busy = True
        if source == "voice":
            await irc.incoming.put(_voice_message())
            await _finish_voice(bridge, irc)
        else:
            await send_control_request(
                bridge.config.pm.control_socket,
                ReportRequest("touch-hockey", "TH-0001", "done", "Pinned queue"),
            )
            await asyncio.wait_for(bridge._action_queues[channel].join(), 3)
        original_queue = await bridge.state.list_queue(channel, "original")
        assert len(original_queue) == 1
        entered, release = asyncio.Event(), asyncio.Event()
        original_list = bridge.state.list_queue

        async def list_queue(target: str, sid: str) -> list[Any]:
            items = await original_list(target, sid)
            if target == channel and sid == "original":
                entered.set()
                await release.wait()
            return items

        monkeypatch.setattr(bridge.state, "list_queue", list_queue)
        runtime.busy = False
        draining = asyncio.create_task(bridge._drain_queue(channel))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await bridge._set_binding(
                channel, SessionSummary("replacement", str(tmp_path), "replacement", busy=True)
            )
            release.set()
            await asyncio.wait_for(draining, 3)
            assert backend.sent == []
            assert runtime.busy is True
            assert await original_list(channel, "original") == original_queue
            assert await original_list(channel, "replacement") == []
        finally:
            release.set()
            draining.cancel()
            await asyncio.gather(draining, return_exceptions=True)
