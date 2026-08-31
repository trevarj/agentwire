from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, call

import pytest

from agentwire.backends.base import BackendError
from agentwire.backends.pi import PiBackend, _cwd_dir_name, _Session, _session_stem, _Transport
from agentwire.config import PiConfig
from agentwire.models import BackendEvent

STEM = "2026-08-24T12-00-00-000Z_11111111-1111-7111-8111-111111111111"
CWD = "/workspace/project"


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        for line in data.decode("utf-8").splitlines():
            if line:
                self.frames.append(json.loads(line))

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def backend(tmp_path: Path, binary: str = "pi") -> PiBackend:
    return PiBackend(
        PiConfig(
            binary=binary,
            socket_dir=tmp_path / "sockets",
            session_root=tmp_path / "sessions",
        )
    )


def attached(harness: PiBackend, busy: bool = False, tui: bool = True) -> _Session:
    transport = _Transport(asyncio.StreamReader(), FakeWriter())
    session = _Session(
        id=STEM,
        cwd=CWD,
        transport=transport,
        tui=tui,
        session_file=f"/sessions/{_cwd_dir_name(CWD)}/{STEM}.jsonl",
    )
    session.busy = busy
    if busy:
        session.turn_id = "turn-1"
    harness._sessions[STEM] = session
    return session


def drain(harness: PiBackend) -> list[BackendEvent]:
    return [harness._events.get_nowait() for _ in range(harness._events.qsize())]


def write_session_file(root: Path, cwd: str, stem: str, entries: list[dict[str, Any]]) -> Path:
    directory = root / _cwd_dir_name(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jsonl"
    header = {"type": "session", "version": 3, "id": "abc", "cwd": cwd}
    lines = [json.dumps(header)] + [json.dumps(entry) for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def message_entry(
    entry_id: str,
    parent: str | None,
    role: str,
    stamp: str = "2026-08-24T12:00:00.000Z",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent,
        "timestamp": stamp,
        "message": {"role": role, **extra},
    }


def test_session_identity_helpers() -> None:
    assert _session_stem(f"/x/{STEM}.jsonl") == STEM
    assert _session_stem(None) is None
    assert _cwd_dir_name("/home/trev/Workspace/motd") == "--home-trev-Workspace-motd--"


@pytest.mark.asyncio
async def test_external_prompt_opens_turn_and_settle_closes_it(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)

    await harness._handle_frame(session, {"type": "agent_start"})
    await harness._handle_frame(
        session, {"type": "message_end", "message": {"role": "user", "text": "fix the bug"}}
    )
    await harness._handle_frame(
        session,
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "text": "Looking at the test first.",
                "toolCalls": [{"id": "t1", "name": "bash"}],
                "stopReason": "toolUse",
            },
        },
    )
    await harness._handle_frame(
        session,
        {
            "type": "tool_execution_start",
            "toolCallId": "t1",
            "toolName": "bash",
            "args": {"command": "pytest -q"},
        },
    )
    await harness._handle_frame(
        session,
        {
            "type": "tool_execution_end",
            "toolCallId": "t1",
            "toolName": "bash",
            "isError": False,
            "output": "1 passed",
        },
    )
    await harness._handle_frame(
        session,
        {
            "type": "message_end",
            "message": {"role": "assistant", "text": "Fixed.", "stopReason": "stop"},
        },
    )
    await harness._handle_frame(session, {"type": "agent_settled"})

    events = drain(harness)
    # Status brackets the turn for the session drawer; the turn itself is the rest.
    assert [event.data["busy"] for event in events if event.kind == "status_changed"] == [
        True,
        False,
    ]
    events = [event for event in events if event.kind != "status_changed"]
    kinds = [event.kind for event in events]
    assert kinds == [
        "turn_started",
        "user_prompt",
        "progress",
        "tool_started",
        "tool_finished",
        "assistant",
        "turn_done",
    ]
    turn = events[0].turn_id
    assert all(event.turn_id == turn for event in events)
    assert events[1].text == "fix the bug"
    assert events[3].tool_kind == "shell"
    assert events[3].data["label"] == "$ pytest -q"
    assert events[4].data["output"] == "1 passed"
    assert events[5].text == "Fixed."
    assert session.last_reply == "Fixed."
    assert not session.busy


@pytest.mark.asyncio
async def test_bridge_prompt_echo_is_suppressed(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)

    async def respond() -> None:
        # Answer the prompt command the moment it is written.
        while not session.transport._pending:
            await asyncio.sleep(0)
        token = next(iter(session.transport._pending))
        session.transport.dispatch_response(
            {"id": token, "type": "response", "command": "prompt", "success": True}
        )

    responder = asyncio.create_task(respond())
    turn_id = await harness.send_message(STEM, "do the thing")
    await responder
    assert turn_id is not None
    await harness._handle_frame(
        session, {"type": "message_end", "message": {"role": "user", "text": "do the thing"}}
    )
    events = drain(harness)
    assert [event.kind for event in events] == ["turn_started"]
    assert session.expected_prompts == 0

    # An external prompt after the echo still surfaces.
    await harness._handle_frame(
        session, {"type": "message_end", "message": {"role": "user", "text": "typed in the TUI"}}
    )
    assert [event.kind for event in drain(harness)] == ["user_prompt"]


@pytest.mark.asyncio
async def test_raw_rpc_messages_and_error_turns_normalize(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)

    await harness._handle_frame(
        session,
        {
            "type": "message_end",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "raw prompt"}],
                "timestamp": 1000,
            },
        },
    )
    await harness._handle_frame(
        session,
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "boom"}],
                "stopReason": "error",
                "errorMessage": "provider exploded",
            },
        },
    )
    events = drain(harness)
    assert [event.kind for event in events] == ["turn_started", "user_prompt", "turn_failed"]
    assert events[1].text == "raw prompt"
    assert "provider exploded" in events[2].text
    assert session.turn_id is None


@pytest.mark.asyncio
async def test_live_socket_discovery_and_disconnect(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    socket_dir = harness.config.socket_dir
    socket_dir.mkdir(parents=True)
    path = socket_dir / "123.sock"
    connections: list[asyncio.StreamWriter] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connections.append(writer)
        hello = {
            "type": "hello",
            "pv": 1,
            "sessionId": "ignored",
            "sessionFile": f"/s/{STEM}.jsonl",
            "cwd": CWD,
            "sessionName": "tui work",
            "busy": False,
            "pid": 123,
        }
        writer.write((json.dumps(hello) + "\n").encode())
        await writer.drain()

    server = await asyncio.start_unix_server(serve, path=str(path))
    try:
        await harness.start()
        for _ in range(50):
            if harness._sessions:
                break
            await asyncio.sleep(0.02)
        running = await harness.list_running_sessions()
        assert [summary.id for summary in running] == [STEM]
        assert running[0].tui_attached is True
        assert running[0].title == "tui work"
        assert await harness.session_busy(STEM) is False

        # Attach prefers the live process over spawning.
        summary = await harness.attach_session(STEM, CWD)
        assert summary.tui_attached is True

        drain(harness)
        connections[0].close()
        for _ in range(50):
            if not harness._sessions:
                break
            await asyncio.sleep(0.02)
        assert STEM not in harness._sessions
        assert any(event.kind == "disconnected" for event in drain(harness))
    finally:
        server.close()
        await harness.close()


@pytest.mark.asyncio
async def test_attach_refuses_second_writer_for_live_session_file(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    path = write_session_file(
        harness.config.session_root, CWD, STEM, [message_entry("u1", None, "user", content="hi")]
    )
    session.session_file = str(path)
    # Rekeyed live session (renamed) still owns the file: attach by the stem
    # must return the live session, not spawn a second writer.
    harness._sessions["other-key"] = harness._sessions.pop(STEM)
    session.id = "other-key"
    summary = await harness.attach_session(STEM, CWD)
    assert summary.id == "other-key"
    assert summary.tui_attached is True


FAKE_PI = """import json, os, sys

state = {
    "sessionId": "spawned",
    "sessionFile": os.environ.get("FAKE_PI_SESSION_FILE"),
    "cwd": os.getcwd(),
    "sessionName": None,
    "model": {"provider": "anthropic", "id": "claude-1", "name": "Claude"},
    "thinkingLevel": "medium",
    "busy": False,
    "pid": os.getpid(),
}
assert os.environ.get("AGENTWIRE_SPAWNED") == "1"
for line in sys.stdin:
    command = json.loads(line)
    kind = command.get("type")
    response = {"id": command.get("id"), "type": "response", "command": kind, "success": True}
    if kind == "get_state":
        response["data"] = state
    elif kind == "switch_session":
        state["sessionFile"] = command["sessionPath"]
    elif kind == "prompt":
        pass
    elif kind == "get_entries":
        response["data"] = {"entries": [
            {"type": "message", "id": "e1", "timestamp": "2026-08-24T12:00:00.000Z",
             "message": {"role": "user", "content": "from rpc", "timestamp": 500}},
            {"type": "message", "id": "e2", "timestamp": "2026-08-24T12:00:01.000Z",
             "message": {"role": "assistant", "stopReason": "stop",
                          "content": [{"type": "text", "text": "rpc reply"}], "timestamp": 1500}},
        ], "leafId": "e2"}
    else:
        response["success"] = False
        response["error"] = "unsupported"
    sys.stdout.write(json.dumps(response) + "\\n")
    sys.stdout.flush()
"""


def fake_pi(tmp_path: Path, session_file: Path) -> str:
    # No /usr/bin/env in the nix build sandbox: pin the interpreter directly.
    payload = tmp_path / "fake-pi.py"
    payload.write_text(FAKE_PI, encoding="utf-8")
    script = tmp_path / "fake-pi"
    # Resolve sh from PATH: sandboxed environments may lack /bin entirely.
    sh = shutil.which("sh") or "/bin/sh"
    script.write_text(f'#!{sh}\nexec {sys.executable} {payload} "$@"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    os.environ["FAKE_PI_SESSION_FILE"] = str(session_file)
    return str(script)


@pytest.mark.asyncio
async def test_spawned_rpc_session_resumes_prompts_and_reads_history(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    path = write_session_file(
        harness.config.session_root,
        str(workdir),
        STEM,
        [message_entry("u1", None, "user", content="hello")],
    )
    harness.config = PiConfig(
        binary=fake_pi(tmp_path, path),
        socket_dir=harness.config.socket_dir,
        session_root=harness.config.session_root,
    )
    # No cwd hint: the backend locates the file by stem and reads the
    # workspace from the session header.
    summary = await harness.attach_session(STEM, None)
    try:
        assert summary.id == STEM
        assert summary.tui_attached is False
        session = harness._sessions[STEM]
        assert session.transport.process is not None

        turn_id = await harness.send_message(STEM, "continue please")
        assert turn_id is not None
        # Registration reports the spawned session's liveness before the turn opens.
        assert [event.kind for event in drain(harness)] == ["status_changed", "turn_started"]

        history = await harness.list_history(STEM, None, 10)
        kinds = [event.kind for event in history.events]
        # The bridge-driven turn is still open (send_message above), so the
        # newest history turn is not closed, matching the other backends.
        assert kinds == ["turn_started", "user_prompt", "assistant"]
        assert history.events[2].text == "rpc reply"

        reply = await harness.get_last_reply(STEM)
        assert reply == "rpc reply"
    finally:
        await harness.close()
    assert session.transport.process.returncode is not None


@pytest.mark.asyncio
async def test_create_session_requires_start_and_spawn_failure_is_safe(tmp_path: Path) -> None:
    harness = backend(tmp_path, binary=str(tmp_path / "missing-binary"))
    with pytest.raises(BackendError, match="cannot start pi"):
        await harness.create_session(str(tmp_path))
    assert harness._sessions == {}


@pytest.mark.asyncio
async def test_extension_ui_dialogs_relay_and_resolve(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness, busy=True, tui=False)
    writer = session.transport.writer

    await harness._handle_frame(
        session,
        {
            "type": "extension_ui_request",
            "id": "q1",
            "method": "select",
            "title": "Pick one",
            "options": ["a", "b"],
        },
    )
    await harness._handle_frame(
        session,
        {
            "type": "extension_ui_request",
            "id": "c1",
            "method": "confirm",
            "title": "Push?",
            "message": "git push origin main",
        },
    )
    await harness._handle_frame(
        session,
        {"type": "extension_ui_request", "id": "n1", "method": "notify", "message": "ignored"},
    )
    events = drain(harness)
    assert [event.kind for event in events] == [
        "status_changed",
        "question",
        "status_changed",
        "approval",
    ]
    assert events[0].data["active_flags"] == ["waiting"]
    question = events[1]
    assert question.questions[0].options == ("a", "b")
    assert question.questions[0].custom is False

    await harness.resolve_question("q1", question.questions, [["b"]])
    await harness.resolve_approval("c1", False)
    assert writer.frames == [
        {"type": "extension_ui_response", "id": "q1", "value": "b"},
        {"type": "extension_ui_response", "id": "c1", "confirmed": False},
    ]
    with pytest.raises(BackendError):
        await harness.resolve_approval("c1", True)

    await harness._handle_frame(
        session,
        {"type": "extension_ui_request", "id": "q2", "method": "input", "title": "Name?"},
    )
    drain(harness)
    await harness.resolve_question("q2", (), None)
    assert writer.frames[-1] == {"type": "extension_ui_response", "id": "q2", "cancelled": True}


@pytest.mark.asyncio
async def test_plan_mode_questionnaire_round_trip(tmp_path: Path) -> None:
    """Replay the frames pi-workflow's plan_mode_question emits through pi-tui-kit."""
    harness = backend(tmp_path)
    session = attached(harness, busy=True, tui=False)
    writer = session.transport.writer
    select_options = [
        "1. Radio rows \u2014 match the settings picker",
        "2. Filter chips \u2014 keep the current look",
        "3. Other (free-form)",
    ]

    async def ask(token: str, method: str, **extra: Any) -> Any:
        await harness._handle_frame(
            session,
            {"type": "extension_ui_request", "id": token, "method": method, **extra},
        )
        events = drain(harness)
        assert [event.kind for event in events][-2:] == ["status_changed", "question"]
        return events[-1]

    # (a) the questionnaire select splits its title and hides the free-form row.
    event = await ask(
        "s1", "select", title="Option UI: How should options render?", options=select_options
    )
    question = event.questions[0]
    assert (question.header, question.prompt) == ("Option UI", "How should options render?")
    assert question.options == tuple(select_options[:2])
    assert question.custom is True

    # (b) a listed option answers verbatim.
    await harness.resolve_question("s1", event.questions, [[select_options[1]]])
    assert writer.frames[-1] == {
        "type": "extension_ui_response",
        "id": "s1",
        "value": select_options[1],
    }

    # (c) typed text answers through the free-form row, then the editor follow-up
    #     carries the real text.
    event = await ask(
        "s2", "select", title="Option UI: How should options render?", options=select_options
    )
    await harness.resolve_question("s2", event.questions, [["segmented buttons"]])
    assert writer.frames[-1] == {
        "type": "extension_ui_response",
        "id": "s2",
        "value": select_options[2],
    }
    event = await ask("e1", "editor", title="How should options render?")
    question = event.questions[0]
    assert question.options == ()
    assert question.custom is True
    assert (question.header, question.prompt) == (
        "How should options render?",
        "How should options render?",
    )
    await harness.resolve_question("e1", event.questions, [["segmented buttons"]])
    assert writer.frames[-1] == {
        "type": "extension_ui_response",
        "id": "e1",
        "value": "segmented buttons",
    }

    # (d) a title without a short header prefix stays whole, custom row absent.
    event = await ask("s3", "select", title="Pick one", options=["a", "b"])
    question = event.questions[0]
    assert (question.header, question.prompt) == ("Pick one", "Pick one")
    assert question.options == ("a", "b")
    assert question.custom is False
    await harness.resolve_question("s3", event.questions, [["typed instead"]])
    assert writer.frames[-1] == {
        "type": "extension_ui_response",
        "id": "s3",
        "value": "typed instead",
    }

    # (e) cancelling closes the questionnaire.
    event = await ask(
        "s4", "select", title="Option UI: How should options render?", options=select_options
    )
    await harness.resolve_question("s4", event.questions, None)
    assert writer.frames[-1] == {"type": "extension_ui_response", "id": "s4", "cancelled": True}


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Option UI: How should options render?", ("Option UI", "How should options render?")),
        ("Pick one", ("Pick one", "Pick one")),
        # A long leading segment is prose, not a header.
        (
            "A header segment far past the limit: tail",
            ("A header segment far past the limit: tail",) * 2,
        ),
        (": nothing", (": nothing", ": nothing")),
    ],
)
def test_split_title(title: str, expected: tuple[str, str]) -> None:
    assert PiBackend._split_title(title) == expected


@pytest.mark.asyncio
async def test_history_pages_from_disk_with_cursor(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    write_session_file(
        harness.config.session_root,
        CWD,
        STEM,
        [
            message_entry("u1", None, "user", content="first", stamp="2026-08-24T12:00:00Z"),
            message_entry(
                "a1",
                "u1",
                "assistant",
                stamp="2026-08-24T12:00:01Z",
                content=[{"type": "text", "text": "one"}],
                stopReason="stop",
            ),
            message_entry("u2", "a1", "user", content="second", stamp="2026-08-24T12:00:02Z"),
            message_entry(
                "a2",
                "u2",
                "assistant",
                stamp="2026-08-24T12:00:03Z",
                content=[
                    {"type": "text", "text": "checking"},
                    {
                        "type": "toolCall",
                        "id": "t1",
                        "name": "read",
                        "arguments": {"path": "/tmp/x"},
                    },
                ],
                stopReason="toolUse",
            ),
            message_entry(
                "r1",
                "a2",
                "toolResult",
                stamp="2026-08-24T12:00:04Z",
                toolCallId="t1",
                toolName="read",
                content=[{"type": "text", "text": "contents"}],
                isError=False,
            ),
            message_entry(
                "a3",
                "r1",
                "assistant",
                stamp="2026-08-24T12:00:05Z",
                content=[{"type": "text", "text": "two"}],
                stopReason="stop",
            ),
        ],
    )
    page = await harness.list_history(STEM, None, 4)
    kinds = [event.kind for event in page.events]
    assert kinds == [
        "turn_started",
        "user_prompt",
        "progress",
        "tool_started",
        "tool_finished",
        "assistant",
        "turn_done",
    ]
    assert page.next_cursor == "4"
    assert page.events[3].tool_kind == "file read"
    assert page.events[3].data["label"] == "Read /tmp/x"
    at_values = [event.at for event in page.events]
    assert at_values == sorted(at_values)

    older = await harness.list_history(STEM, page.next_cursor, 10)
    older_kinds = [event.kind for event in older.events]
    assert older_kinds == ["turn_started", "user_prompt", "assistant", "turn_done"]
    assert older.next_cursor is None

    with pytest.raises(BackendError, match="cursor"):
        await harness.list_history(STEM, "not-a-number", 5)


@pytest.mark.asyncio
async def test_list_sessions_reads_titles_and_marks_live(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    write_session_file(
        harness.config.session_root,
        CWD,
        STEM,
        [message_entry("u1", None, "user", content="rename the module and fix imports")],
    )
    other_stem = "2026-08-23T10-00-00-000Z_22222222-2222-7222-8222-222222222222"
    write_session_file(harness.config.session_root, CWD, other_stem, [])
    old = harness.config.session_root / _cwd_dir_name(CWD) / f"{other_stem}.jsonl"
    os.utime(old, (time.time() - 3600, time.time() - 3600))

    session = attached(harness, busy=True)
    summaries = await harness.list_sessions(CWD)
    assert [summary.id for summary in summaries] == [STEM, other_stem]
    assert summaries[0].title == "rename the module and fix imports"
    assert summaries[0].busy is True
    assert summaries[0].tui_attached is True
    assert summaries[1].title == "untitled"
    assert session.id == STEM


@pytest.mark.asyncio
async def test_live_history_pages_past_extension_limit_and_stops_on_no_progress(
    tmp_path: Path,
) -> None:
    harness = backend(tmp_path)
    session = attached(harness)

    def entry(index: int) -> dict[str, Any]:
        return {
            "id": f"e{index}",
            "timestamp": "2026-08-24T12:00:00Z",
            "message": {"role": "user", "text": f"message {index}"},
        }

    first = [entry(index) for index in range(500)]
    second = [entry(index) for index in range(500, 501)]
    session.transport.request = AsyncMock(
        side_effect=[
            {"entries": first, "leafId": "e500"},
            {"entries": second, "leafId": "e500"},
        ]
    )

    entries = await harness._entries(STEM)

    assert len(entries) == 501
    assert session.transport.request.await_args_list == [
        call({"type": "get_entries", "limit": 500}),
        call({"type": "get_entries", "limit": 500, "since": "e499"}),
    ]

    # A new message frame would clear the paging cache; simulate that so the
    # second fetch exercises the leaf-stop logic instead of the cache.
    harness._entries_cache = None
    session.transport.request = AsyncMock(
        side_effect=[
            {"entries": first, "leafId": "later"},
            {"entries": first, "leafId": "later"},
        ]
    )
    assert len(await harness._entries(STEM)) == 500
    assert session.transport.request.await_count == 2


@pytest.mark.asyncio
async def test_waiting_state_comes_from_tui_frames_and_open_rpc_requests(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = harness._register(
        {
            "sessionFile": f"/s/{STEM}.jsonl",
            "cwd": CWD,
            "busy": True,
            "waiting": True,
        },
        _Transport(asyncio.StreamReader(), FakeWriter()),
        tui=True,
    )
    assert session is not None
    session.pump.cancel()
    assert drain(harness)[0].data["active_flags"] == ["waiting"]

    await harness._handle_frame(
        session, {"type": "session_changed", "waiting": False, "sessionName": "live"}
    )
    assert drain(harness)[0].data["active_flags"] == []

    rpc = attached(harness, tui=False)
    await harness._handle_frame(
        rpc,
        {"type": "extension_ui_request", "id": "q", "method": "input", "title": "Value"},
    )
    assert drain(harness)[0].data["active_flags"] == ["waiting"]


@pytest.mark.asyncio
async def test_close_session_only_stops_bridge_owned_rpc(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness, tui=True)
    with pytest.raises(BackendError, match="live pi TUI"):
        await harness.close_session(STEM)

    session.tui = False
    with pytest.raises(BackendError, match="not owned"):
        await harness.close_session(STEM)

    session.transport.process = object()  # type: ignore[assignment]
    session.transport.close = AsyncMock()
    harness._ui_requests["q"] = (session, {}, None)
    await harness.close_session(STEM)

    assert STEM not in harness._sessions
    assert harness._ui_requests == {}
    session.transport.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_session_keeps_registry_until_cancelled_termination_finishes(
    tmp_path: Path,
) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    session.tui = False
    session.transport.process = object()  # type: ignore[assignment]
    started = asyncio.Event()
    release = asyncio.Event()

    async def block_close() -> None:
        started.set()
        await release.wait()

    session.transport.close = block_close  # type: ignore[method-assign]
    harness._ui_requests["q"] = (session, {}, None)
    closing = asyncio.create_task(harness.close_session(STEM))
    await started.wait()
    assert harness._sessions[STEM] is session
    assert "q" in harness._ui_requests

    closing.cancel()
    await asyncio.sleep(0)
    assert harness._sessions[STEM] is session
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert STEM not in harness._sessions
    assert harness._ui_requests == {}


def test_count_sessions_counts_jsonl_files(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    assert harness.count_sessions(CWD) == 0

    directory = harness.config.session_root / _cwd_dir_name(CWD)
    directory.mkdir(parents=True)
    assert harness.count_sessions(CWD) == 0

    write_session_file(harness.config.session_root, CWD, STEM, [])
    other_stem = "2026-08-23T10-00-00-000Z_22222222-2222-7222-8222-222222222222"
    write_session_file(harness.config.session_root, CWD, other_stem, [])
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")
    assert harness.count_sessions(CWD) == 2


@pytest.mark.asyncio
async def test_steer_cancel_and_settings_round_trip(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness, busy=True)

    async def respond_all() -> None:
        answered: set[str] = set()
        for _ in range(5):
            token: str | None = None
            while token is None:
                token = next(
                    (
                        key
                        for key, future in session.transport._pending.items()
                        if key not in answered and not future.done()
                    ),
                    None,
                )
                if token is None:
                    await asyncio.sleep(0)
            answered.add(token)
            payload: dict[str, Any] = {"id": token, "type": "response", "success": True}
            command = session.transport.writer.frames[-1]
            payload["command"] = command["type"]
            if command["type"] == "get_available_models":
                payload["data"] = {
                    "models": [
                        {
                            "provider": "anthropic",
                            "id": "claude-1",
                            "name": "Claude",
                            "reasoning": True,
                        },
                        {"provider": "ollama", "id": "small", "name": "Small", "reasoning": False},
                    ]
                }
            session.transport.dispatch_response(payload)

    responder = asyncio.create_task(respond_all())
    await harness.steer(STEM, None, "focus on tests")
    await harness.cancel(STEM, None)
    options = await harness.setting_options()
    await harness.configure_session(STEM, {"model": "anthropic/claude-1", "effort": "high"})
    await responder

    commands = [frame["type"] for frame in session.transport.writer.frames]
    assert commands == [
        "steer",
        "abort",
        "get_available_models",
        "set_model",
        "set_thinking_level",
    ]
    assert options["model"][0] == {
        "value": "anthropic/claude-1",
        "label": "Claude",
        "efforts": ["off", "minimal", "low", "medium", "high"],
    }
    assert options["model"][1]["efforts"] == ["off"]

    session.busy = False
    with pytest.raises(BackendError, match="no active turn"):
        await harness.steer(STEM, None, "too late")
    with pytest.raises(BackendError, match="no active turn"):
        await harness.cancel(STEM, None)
    with pytest.raises(BackendError, match="unsupported pi settings"):
        await harness.configure_session(STEM, {"temperature": 1})


@pytest.mark.asyncio
async def test_session_changed_rekeys_live_session(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness, busy=True)
    new_stem = "2026-08-24T13-00-00-000Z_33333333-3333-7333-8333-333333333333"
    await harness._handle_frame(
        session,
        {
            "type": "session_changed",
            "sessionFile": f"/s/{new_stem}.jsonl",
            "cwd": CWD,
            "sessionName": "fresh",
        },
    )
    assert STEM not in harness._sessions
    assert harness._sessions[new_stem] is session
    assert session.title == "fresh"
    # The dangling turn of the old session closed, and the rekeyed session
    # reports its own status under the new id.
    events = drain(harness)
    assert [event.kind for event in events] == ["status_changed", "turn_done", "status_changed"]
    assert events[-1].session_id == new_stem


@pytest.mark.asyncio
async def test_registration_and_turn_edges_report_session_status(tmp_path: Path) -> None:
    """pi pushes state, so the drawer is fed at registration and turn edges."""

    harness = backend(tmp_path)
    session = harness._register(
        {"sessionFile": f"/s/{STEM}.jsonl", "cwd": CWD, "sessionName": "live"},
        _Transport(asyncio.StreamReader(), FakeWriter()),
        tui=True,
    )
    assert session is not None
    session.pump.cancel()

    hello = drain(harness)
    assert [event.kind for event in hello] == ["status_changed"]
    assert hello[0].session_id == STEM
    assert hello[0].data == {"busy": False, "active_flags": [], "cwd": CWD, "tui": True}

    await harness._handle_frame(session, {"type": "agent_start"})
    started = drain(harness)
    assert [event.kind for event in started] == ["status_changed"]
    assert started[0].data["busy"] is True

    # An open extension dialog is the one flag pi can state today.
    await harness._handle_frame(
        session,
        {"type": "extension_ui_request", "id": "u1", "method": "confirm", "title": "Run it?"},
    )
    drain(harness)
    await harness._handle_frame(session, {"type": "agent_settled"})
    settled = drain(harness)
    assert [event.kind for event in settled] == ["status_changed"]
    assert settled[0].data["busy"] is False
    assert settled[0].data["active_flags"] == ["waiting"]

    await harness.resolve_approval("u1", True)
    resolved = drain(harness)
    assert [event.kind for event in resolved] == ["status_changed"]
    assert resolved[0].data["active_flags"] == []
    await harness._handle_frame(session, {"type": "session_changed", "sessionName": "renamed"})
    changed = drain(harness)
    assert [event.kind for event in changed] == ["status_changed"]
    assert changed[0].data["active_flags"] == []


@pytest.mark.asyncio
async def test_subagent_frames_and_state_payloads_are_sanitized(tmp_path: Path) -> None:
    """The extension bounds the registry; the backend still allowlists the shape."""

    harness = backend(tmp_path)
    session = attached(harness)
    await harness._handle_frame(
        session,
        {
            "type": "subagent_update",
            "agents": [
                {
                    "id": "a1",
                    "type": "Explore",
                    "description": "map  the\nrepository",
                    "status": "running",
                    "isBackground": True,
                    "secret": "must not survive",
                },
                {
                    "id": "a2",
                    "type": "Terra",
                    "description": "x" * 400,
                    "status": "completed",
                    "isBackground": False,
                    "toolUses": 7,
                    "durationMs": 4200,
                    "tokens": {"input": 5},
                },
                # Dropped: no id, and an unknown status.
                {"type": "Terra", "status": "running"},
                {"id": "a3", "status": "cancelled"},
            ],
        },
    )
    events = drain(harness)
    assert [event.kind for event in events] == ["subagent_update"]
    agents = events[0].data["agents"]
    assert events[0].session_id == STEM
    assert agents[0] == {
        "id": "a1",
        "type": "Explore",
        "description": "map the repository",
        "status": "running",
        "isBackground": True,
    }
    assert len(agents) == 2
    assert agents[1]["toolUses"] == 7
    assert agents[1]["durationMs"] == 4200
    # A non-numeric `tokens` is dropped rather than forwarded.
    assert "tokens" not in agents[1]
    assert len(agents[1]["description"].encode("utf-8")) <= 200

    # A frame without a usable list produces nothing at all.
    await harness._handle_frame(session, {"type": "subagent_update", "agents": "nope"})
    assert drain(harness) == []


@pytest.mark.asyncio
async def test_hello_and_session_changed_carry_the_current_subagents(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    hello = {
        "sessionFile": f"/s/{STEM}.jsonl",
        "cwd": CWD,
        "sessionName": "live",
        "subagents": [{"id": "a1", "type": "Terra", "description": "d", "status": "queued"}],
    }
    session = harness._register(hello, _Transport(asyncio.StreamReader(), FakeWriter()), tui=True)
    assert session is not None
    session.pump.cancel()
    # `_register` reports liveness; the hello list follows it on connect, so a
    # client that attaches mid-run does not wait for the next lifecycle event.
    harness._queue_subagents(session, hello)
    registered = drain(harness)
    assert [event.kind for event in registered] == ["status_changed", "subagent_update"]
    assert registered[1].data["agents"][0]["id"] == "a1"

    await harness._handle_frame(
        session, {"type": "session_changed", "sessionName": "renamed", "subagents": []}
    )
    changed = drain(harness)
    assert [event.kind for event in changed] == ["status_changed", "subagent_update"]
    assert changed[1].data == {"agents": []}

    # A state payload with no `subagents` member says nothing about the list.
    await harness._handle_frame(session, {"type": "session_changed", "sessionName": "again"})
    assert [event.kind for event in drain(harness)] == ["status_changed"]
