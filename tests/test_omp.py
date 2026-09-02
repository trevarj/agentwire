from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentwire.backends.base import BackendError
from agentwire.backends.omp import (
    _MAX_FRAME_BYTES,
    OmpBackend,
    _OmpFrameDecoder,
)
from agentwire.backends.pi import PiBackend, _ResponseError, _Session, _Transport
from agentwire.config import OmpConfig


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.frames.extend(json.loads(line) for line in data.splitlines() if line)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def backend(tmp_path: Path) -> OmpBackend:
    return OmpBackend(OmpConfig("omp", tmp_path / "sockets", tmp_path / "sessions"))


def attached(harness: OmpBackend, *, busy: bool = False) -> _Session:
    session = _Session(
        id="session-1",
        cwd="/workspace/project",
        transport=_Transport(asyncio.StreamReader(), FakeWriter(), backend="omp"),
        tui=False,
        busy=busy,
        turn_id="turn-1" if busy else None,
    )
    harness._sessions[session.id] = session
    return session


def drain(harness: OmpBackend) -> list[str]:
    return [harness._events.get_nowait().kind for _ in range(harness._events.qsize())]


def test_omp_busy_state() -> None:
    harness = OmpBackend(OmpConfig("omp", Path("/sockets"), Path("/sessions")))
    assert harness._state_busy({"isStreaming": True})
    assert harness._state_busy({"isCompacting": True})
    assert not harness._state_busy({"busy": True})


@pytest.mark.asyncio
async def test_omp_only_terminal_agent_end_settles(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness, busy=True)

    await harness._handle_frame(session, {"type": "agent_end", "isTerminal": False})
    assert session.busy
    assert drain(harness) == []

    await harness._handle_frame(session, {"type": "agent_end", "isTerminal": True})
    assert not session.busy
    assert drain(harness) == ["status_changed", "turn_done"]


@pytest.mark.asyncio
async def test_omp_command_output_and_local_prompt_settle(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)

    await harness._handle_frame(session, {"type": "command_output", "text": " local output \n"})
    assert session.last_reply == "local output"
    await harness._handle_frame(session, {"type": "prompt_result", "agentInvoked": False})
    await asyncio.sleep(0)

    assert not session.busy
    assert drain(harness) == ["turn_started", "assistant", "status_changed", "turn_done"]


@pytest.mark.asyncio
async def test_omp_late_prompt_failure_fails_current_turn(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness, busy=True)
    session.transport._completed["aw-1"] = "prompt"

    await harness._handle_frame(
        session,
        {"id": "aw-1", "type": "response", "success": False, "error": "schedule failed"},
    )
    await asyncio.sleep(0)
    assert not session.busy
    assert drain(harness) == ["turn_failed", "status_changed"]


def test_omp_session_bucket_and_title_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    temp = tmp_path / "tmp"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("tempfile.tempdir", str(temp))
    harness = backend(tmp_path)

    assert harness._session_directory(str(home)) == tmp_path / "sessions" / "-"
    assert harness._session_directory(str(home / "work" / "repo")) == (
        tmp_path / "sessions" / "-work-repo"
    )
    assert harness._session_directory(str(temp)) == tmp_path / "sessions" / "-tmp"
    assert harness._session_directory(str(temp / "repo")) == (tmp_path / "sessions" / "-tmp-repo")
    assert harness._session_directory("/opt/work") == (tmp_path / "sessions" / "--opt-work--")

    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        "\n".join(
            (
                json.dumps({"type": "title", "title": "Example"}),
                json.dumps({"type": "session", "sessionId": "internal", "cwd": "/work"}),
            )
        )
        + "\n"
    )
    assert PiBackend._file_header(session_file)["sessionId"] == "internal"


@pytest.mark.asyncio
async def test_omp_pages_spawned_history_chronologically(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    session.transport.request = AsyncMock(
        side_effect=[
            {
                "messages": [{"role": "user", "content": "one", "timestamp": 1}],
                "totalMessages": 2,
                "nextCursor": "opaque",
            },
            {
                "messages": [{"role": "assistant", "content": "two", "timestamp": 2}],
                "totalMessages": 2,
                "nextCursor": None,
            },
        ]
    )

    entries = await harness._transport_entries(session)

    assert [entry["id"] for entry in entries] == ["0", "1"]
    assert [entry["message"]["text"] for entry in entries] == ["one", "two"]
    assert session.transport.request.await_args_list[1].args[0]["cursor"] == "opaque"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["session_busy", "stale_cursor"])
async def test_omp_restarts_failed_page_as_monolithic_snapshot(tmp_path: Path, code: str) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    session.transport.request = AsyncMock(
        side_effect=[
            {
                "messages": [{"role": "user", "content": "discard me"}],
                "totalMessages": 2,
                "nextCursor": "opaque",
            },
            _ResponseError("retry", code),
            {
                "messages": [
                    {"role": "user", "content": "fresh"},
                    {"role": "assistant", "content": "snapshot"},
                ]
            },
        ]
    )

    entries = await harness._transport_entries(session)

    assert [entry["message"]["text"] for entry in entries] == ["fresh", "snapshot"]
    assert session.transport.request.await_args_list[-1].args[0] == {"type": "get_messages"}


@pytest.mark.asyncio
async def test_omp_approval_question_and_cancel_responses(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    approval = {
        "type": "extension_ui_request",
        "id": "approve-1",
        "method": "select",
        "title": "  Allow tool: write\n/path",
        "options": ["Approve", "Deny"],
    }
    await harness._handle_frame(session, approval)
    events = [harness._events.get_nowait() for _ in range(harness._events.qsize())]
    assert [event.kind for event in events] == ["status_changed", "approval"]
    assert events[-1].text == "Allow tool: write\n/path"

    await harness.resolve_approval("approve-1", True)
    assert session.transport.writer.frames[-1] == {
        "type": "extension_ui_response",
        "id": "approve-1",
        "value": "Approve",
    }
    drain(harness)

    question = {
        "type": "extension_ui_request",
        "id": "question-1",
        "method": "select",
        "title": "Choose",
        "options": ["One", "Other (type your own)"],
    }
    await harness._handle_frame(session, question)
    events = [harness._events.get_nowait() for _ in range(harness._events.qsize())]
    assert events[-1].kind == "question"
    assert events[-1].questions[0].options == ("One",)
    assert events[-1].questions[0].custom

    await harness._handle_frame(
        session,
        {
            "type": "extension_ui_request",
            "id": "cancel-1",
            "method": "cancel",
            "targetId": "question-1",
        },
    )
    assert "question-1" not in harness._ui_requests
    assert drain(harness) == ["request_resolved", "status_changed"]

    await harness._handle_frame(
        session,
        {
            "type": "extension_ui_request",
            "id": "cancel-2",
            "method": "cancel",
            "targetId": "question-1",
        },
    )
    assert drain(harness) == []


@pytest.mark.asyncio
async def test_omp_efforts_use_ordered_model_metadata(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    session.transport.request = AsyncMock(
        return_value={
            "models": [
                {
                    "provider": "anthropic",
                    "id": "large",
                    "name": "Large",
                    "reasoning": True,
                    "thinking": {"efforts": ["low", "high", "max"]},
                },
                {
                    "provider": "local",
                    "id": "small",
                    "name": "Small",
                    "reasoning": False,
                },
            ]
        }
    )

    options = await harness.setting_options()

    assert options["model"][0]["efforts"] == ["off", "low", "high", "max"]
    assert options["model"][1]["efforts"] == ["off"]


def chunk_frames(raw: bytes, *, chunk_id: str = "rpc-1") -> list[dict[str, Any]]:
    parts = [raw[index : index + 256 * 1024] for index in range(0, len(raw), 256 * 1024)]
    return [
        {
            "type": "rpc_chunk",
            "chunkId": chunk_id,
            "index": index,
            "count": len(parts),
            "byteLength": len(raw),
            "data": base64.b64encode(part).decode(),
        }
        for index, part in enumerate(parts)
    ]


def push(decoder: _OmpFrameDecoder, frame: Any) -> dict[str, Any] | None:
    return decoder((json.dumps(frame) + "\n").encode())


def test_omp_v1_and_v2_frame_decoding() -> None:
    decoder = _OmpFrameDecoder()
    assert push(decoder, {"type": "response"}) == {"type": "response"}

    raw = json.dumps({"type": "response", "text": "x" * _MAX_FRAME_BYTES}).encode()
    decoder.protocol2 = True
    result = None
    for frame in chunk_frames(raw):
        result = push(decoder, frame)
    assert result == json.loads(raw)


@pytest.mark.parametrize(
    "case",
    [
        "oversized",
        "base64",
        "interleaved",
        "chunk_id",
        "count",
        "length",
        "index",
        "overflow",
        "utf8",
        "non_object",
    ],
)
def test_omp_rejects_malformed_chunk_sequences(case: str) -> None:
    decoder = _OmpFrameDecoder()
    decoder.protocol2 = True
    if case == "oversized":
        with pytest.raises(BackendError):
            decoder(b" " * _MAX_FRAME_BYTES + b"\n")
        return
    if case == "overflow":
        frames = [
            {
                "type": "rpc_chunk",
                "chunkId": "rpc-1",
                "index": index,
                "count": 5,
                "byteLength": _MAX_FRAME_BYTES,
                "data": base64.b64encode(b"x" * (256 * 1024)).decode(),
            }
            for index in range(5)
        ]
    else:
        raw = (
            b"\xff" * _MAX_FRAME_BYTES
            if case == "utf8"
            else b"[]" + b" " * _MAX_FRAME_BYTES
            if case == "non_object"
            else json.dumps({"type": "response", "text": "x" * _MAX_FRAME_BYTES}).encode()
        )
        frames = chunk_frames(raw)
    if case == "base64":
        frames[0]["data"] = "eA="
        with pytest.raises(BackendError):
            push(decoder, frames[0])
        return
    push(decoder, frames[0])
    if case == "interleaved":
        with pytest.raises(BackendError):
            push(decoder, {"type": "response"})
        return
    if case == "chunk_id":
        frames[1]["chunkId"] = "rpc-2"
    elif case == "count":
        frames[1]["count"] += 1
    elif case == "length":
        frames[1]["byteLength"] += 1
    elif case == "index":
        frames[1]["index"] += 1
    with pytest.raises(BackendError):
        for frame in frames[1:]:
            push(decoder, frame)


@pytest.mark.asyncio
async def test_omp_requires_ready_and_negotiates_bounded_v2(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    v1_reader = asyncio.StreamReader()
    v1_reader.feed_data(
        b'{"type":"ready","supportedProtocolVersions":[1],"maxFrameBytes":1048576}\n'
    )
    v1_transport = harness._make_transport(v1_reader, FakeWriter())
    await harness._initialize_transport(v1_transport)
    assert isinstance(v1_transport.decoder, _OmpFrameDecoder)
    assert not v1_transport.decoder.protocol2
    assert v1_transport.writer.frames == []

    reader = asyncio.StreamReader()

    class RespondingWriter(FakeWriter):
        def write(self, data: bytes) -> None:
            super().write(data)
            command = self.frames[-1]
            reader.feed_data(
                (
                    json.dumps(
                        {
                            "id": command["id"],
                            "type": "response",
                            "command": command["type"],
                            "success": True,
                            "data": {"protocolVersion": 2},
                        }
                    )
                    + "\n"
                ).encode()
            )

    reader.feed_data(
        (
            json.dumps(
                {
                    "type": "ready",
                    "supportedProtocolVersions": [1, 2],
                    "maxFrameBytes": _MAX_FRAME_BYTES * 2,
                    "maxReassembledFrameBytes": 128 * 1024 * 1024,
                }
            )
            + "\n"
        ).encode()
    )
    transport = harness._make_transport(reader, RespondingWriter())
    await harness._initialize_transport(transport)
    decoder = transport.decoder
    assert isinstance(decoder, _OmpFrameDecoder)
    assert decoder.protocol2
    assert decoder.max_frame_bytes == _MAX_FRAME_BYTES
    assert transport.writer.frames[0]["type"] == "negotiate_protocol"

    wrong_reader = asyncio.StreamReader()
    wrong_reader.feed_data(b'{"type":"hello"}\n')
    with pytest.raises(BackendError, match="required ready"):
        await harness._initialize_transport(harness._make_transport(wrong_reader, FakeWriter()))


FAKE_OMP = """import json, os, sys, time

session_file = os.environ["FAKE_OMP_SESSION_FILE"]
assert os.environ.get("AGENTWIRE_SPAWNED") == "1"
expected = ["--mode", "rpc", "--approval-mode", "always-ask", "--session-dir"]
assert sys.argv[1:6] == expected
assert sys.argv[6] == os.path.dirname(session_file)

def emit(frame):
    sys.stdout.write(json.dumps(frame) + "\\n")
    sys.stdout.flush()

emit({
    "type": "ready",
    "protocolVersion": 1,
    "supportedProtocolVersions": [1, 2],
    "maxFrameBytes": 1048576,
    "maxReassembledFrameBytes": 67108864,
})
for line in sys.stdin:
    command = json.loads(line)
    kind = command["type"]
    response = {
        "id": command.get("id"),
        "type": "response",
        "command": kind,
        "success": True,
    }
    if kind == "negotiate_protocol":
        response["data"] = {"protocolVersion": 2}
    elif kind == "get_state":
        response["data"] = {
            "sessionId": "internal-id",
            "sessionFile": session_file,
            "sessionName": "Spawned OMP",
            "isStreaming": False,
            "isCompacting": False,
        }
    elif kind == "prompt":
        response["data"] = {"agentInvoked": True}
        emit(response)
        emit({"type": "agent_start"})
        emit({"type": "message_end", "message": {"role": "user", "content": command["message"]}})
        emit({
            "type": "tool_execution_start",
            "toolCallId": "tool-1",
            "toolName": "read",
            "args": {"path": "x"},
        })
        emit({
            "type": "tool_execution_end",
            "toolCallId": "tool-1",
            "toolName": "read",
            "isError": False,
            "output": "ok",
        })
        emit({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": "done",
                "stopReason": "stop",
            },
        })
        emit({"type": "agent_end", "messages": [], "isTerminal": False})
        time.sleep(0.1)
        emit({"type": "agent_end", "messages": [], "isTerminal": True})
        continue
    elif kind == "get_messages_page":
        response["data"] = {
            "messages": [
                {"role": "user", "content": "run"},
                {"role": "toolResult", "toolCallId": "tool-1", "toolName": "read", "content": "ok"},
                {"role": "assistant", "content": "done", "stopReason": "stop"},
            ],
            "totalMessages": 3,
            "nextCursor": None,
        }
    else:
        response["success"] = False
        response["error"] = "unsupported"
    emit(response)
"""


def fake_omp(tmp_path: Path, session_file: Path) -> str:
    payload = tmp_path / "fake-omp.py"
    payload.write_text(FAKE_OMP)
    script = tmp_path / "fake-omp"
    sh = shutil.which("sh") or "/bin/sh"
    script.write_text(f'#!{sh}\nexec {sys.executable} {payload} "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    os.environ["FAKE_OMP_SESSION_FILE"] = str(session_file)
    return str(script)


@pytest.mark.asyncio
async def test_spawned_omp_launch_lifecycle_and_history(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    harness = backend(tmp_path)
    session_file = harness._session_directory(str(workdir)) / (
        "2026-09-02T12-00-00-000Z_11111111-1111-7111-8111-111111111111.jsonl"
    )
    harness.config = OmpConfig(
        fake_omp(tmp_path, session_file),
        harness.config.socket_dir,
        harness.config.session_root,
    )

    summary = await harness.create_session(str(workdir))
    try:
        assert summary.id == session_file.stem
        assert summary.cwd == str(workdir)
        assert not summary.busy
        turn_id = await harness.send_message(summary.id, "run")
        assert turn_id is not None
        await asyncio.sleep(0.03)
        assert await harness.session_busy(summary.id)
        await asyncio.sleep(0.12)
        assert not await harness.session_busy(summary.id)
        kinds = drain(harness)
        assert kinds.count("turn_done") == 1
        assert "tool_started" in kinds
        assert "tool_finished" in kinds
        history = await harness.list_history(summary.id, None, 10)
        assert [event.kind for event in history.events] == [
            "turn_started",
            "user_prompt",
            "tool_finished",
            "assistant",
            "turn_done",
        ]
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_omp_live_socket_requires_exact_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = backend(tmp_path)
    reader = asyncio.StreamReader()
    writer = FakeWriter()
    reader.feed_data(b'{"type":"hello","backend":"pi","sessionId":"wrong","cwd":"/work"}\n')

    async def misplaced(_path: str, **_kwargs: Any) -> tuple[asyncio.StreamReader, FakeWriter]:
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", misplaced)
    with pytest.raises(BackendError, match="identified itself as pi"):
        await harness._connect_socket(tmp_path / "wrong.sock")
    assert writer.closed
    assert harness._sessions == {}

    good_reader = asyncio.StreamReader()
    good_writer = FakeWriter()
    stem = "2026-09-02T12-00-00-000Z_22222222-2222-7222-8222-222222222222"
    good_reader.feed_data(
        (
            json.dumps(
                {
                    "type": "hello",
                    "backend": "omp",
                    "sessionId": "internal-id",
                    "sessionFile": f"/sessions/{stem}.jsonl",
                    "cwd": "/work",
                }
            )
            + "\n"
        ).encode()
    )

    async def correct(_path: str, **_kwargs: Any) -> tuple[asyncio.StreamReader, FakeWriter]:
        return good_reader, good_writer

    monkeypatch.setattr(asyncio, "open_unix_connection", correct)
    await harness._connect_socket(tmp_path / "good.sock")
    assert list(harness._sessions) == [stem]
    await harness.close()


@pytest.mark.asyncio
async def test_omp_immediate_local_prompt_response_settles(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    session = attached(harness)
    session.transport.request = AsyncMock(return_value={"agentInvoked": False})

    turn_id = await harness.send_message(session.id, "/name local")
    await asyncio.sleep(0)

    assert turn_id is not None
    assert not session.busy
    assert drain(harness) == ["turn_started", "status_changed", "turn_done"]


@pytest.mark.asyncio
async def test_omp_cancelled_switch_is_a_failed_attach(tmp_path: Path) -> None:
    harness = backend(tmp_path)
    reader = asyncio.StreamReader()

    class CancelWriter(FakeWriter):
        def write(self, data: bytes) -> None:
            super().write(data)
            command = self.frames[-1]
            reader.feed_data(
                (
                    json.dumps(
                        {
                            "id": command["id"],
                            "type": "response",
                            "command": "switch_session",
                            "success": True,
                            "data": {"cancelled": True},
                        }
                    )
                    + "\n"
                ).encode()
            )

    transport = harness._make_transport(reader, CancelWriter())
    with pytest.raises(BackendError, match="switch was cancelled"):
        await harness._request_direct(
            transport, {"type": "switch_session", "sessionPath": "/session.jsonl"}
        )
