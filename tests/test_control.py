from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import socket
import stat
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from agentwire import control
from agentwire.control import (
    ControlError,
    ControlServer,
    DelegateRequest,
    ReportRequest,
    send_control_request,
)

ACTION_ID = "10000000-0000-4000-8000-000000000001"
DELEGATE = {"op": "delegate", "project": "touch-hockey", "task": "TH-0001", "text": "Do work"}
REPORT = {
    "op": "report",
    "project": "touch-hockey",
    "task": "TH-0001",
    "status": "done",
    "text": "Done",
}
SECRET = "https://private.invalid/signed?token=do-not-disclose"


def wire(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


async def accepted(_request: DelegateRequest | ReportRequest) -> str:
    return ACTION_ID


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


@contextlib.asynccontextmanager
async def serving(
    path: Path,
    on_request: Callable[[DelegateRequest | ReportRequest], Awaitable[str]] = accepted,
) -> AsyncIterator[ControlServer]:
    server = ControlServer(path, on_request)
    await server.start()
    try:
        yield server
    finally:
        await server.close()


async def exchange(path: Path, *parts: bytes) -> dict[str, object]:
    reader, writer = await asyncio.open_unix_connection(str(path))
    try:
        for part in parts:
            writer.write(part)
            await writer.drain()
            await asyncio.sleep(0)
        writer.write_eof()
        return json.loads(await asyncio.wait_for(reader.read(), 2))
    finally:
        await close_writer(writer)


@contextlib.asynccontextmanager
async def replying(path: Path, response: bytes) -> AsyncIterator[list[bytes]]:
    received: list[bytes] = []
    tasks: set[asyncio.Task[None]] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            received.append(await reader.read())
            writer.write(response)
            with contextlib.suppress(OSError):
                await writer.drain()
        finally:
            await close_writer(writer)

    def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(handle(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_unix_server(connected, path=str(path))
    try:
        yield received
    finally:
        server.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.wait_closed()


@pytest.mark.asyncio
async def test_control_creates_private_socket_and_round_trips_typed_requests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "c.sock"
    requests: list[DelegateRequest | ReportRequest] = []

    async def receive(request: DelegateRequest | ReportRequest) -> str:
        requests.append(request)
        return ACTION_ID

    delegate = DelegateRequest("touch-hockey", "TH-0001", "Inspect\nthe café")
    done = ReportRequest("touch-hockey", "TH-0001", "done", "Checked the café")
    blocked = ReportRequest("touch-hockey", "TH-0001", "blocked", "Need a decision")
    async with serving(path, receive):
        assert stat.S_IMODE(path.parent.lstat().st_mode) == 0o700
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600
        assert path.lstat().st_uid == os.getuid()
        for request in (delegate, done, blocked):
            assert await send_control_request(path, request) == ACTION_ID
        assert requests == [delegate, done, blocked]
    assert not path.exists()
    assert path.parent.is_dir()


@pytest.mark.asyncio
async def test_control_accepts_split_utf8_and_exact_content_and_frame_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "c.sock"
    requests: list[DelegateRequest | ReportRequest] = []

    async def receive(request: DelegateRequest | ReportRequest) -> str:
        requests.append(request)
        return ACTION_ID

    async with serving(path, receive):
        body = wire({**DELEGATE, "text": "é" * (64 * 1024 // 2)})
        split = body.index("é".encode()) + 1
        assert await exchange(path, body[:split], body[split:]) == {
            "ok": True,
            "actionId": ACTION_ID,
        }
        body = wire(DELEGATE)[:-1]
        frame = body + b" " * (128 * 1024 - len(body) - 1) + b"\n"
        assert (await exchange(path, frame))["ok"] is True
        assert requests[0].text == "é" * (64 * 1024 // 2)
        assert requests[1].text == DELEGATE["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        b"not json\n",
        b"[]\n",
        b'{"op":"delegate","op":"report"}\n',
        b'{"op":NaN}\n',
        b'{"op":"\xff"}\n',
        wire({**DELEGATE, "channel": "#pm"}),
        wire({**DELEGATE, "op": "prompt"}),
        wire({key: value for key, value in DELEGATE.items() if key != "project"}),
        wire({**DELEGATE, "project": " "}),
        wire({**DELEGATE, "project": 1}),
        wire({**DELEGATE, "task": "th-0001"}),
        wire({**DELEGATE, "task": "1TH-0001"}),
        wire({**DELEGATE, "task": "TH-001"}),
        wire({**DELEGATE, "task": "TH-0001\n"}),
        wire({**DELEGATE, "task": True}),
        wire({**DELEGATE, "text": "\t\n "}),
        wire({**DELEGATE, "text": []}),
        wire(DELEGATE)[:-1],
        wire({**REPORT, "status": "running"}),
        wire({**REPORT, "status": True}),
        wire({**REPORT, "text": "Done\nMore"}),
        wire({**REPORT, "text": "Done\r"}),
        wire({**REPORT, "text": "Done\u2028More"}),
        wire({**DELEGATE, "status": "done"}),
        wire(DELEGATE) + wire(REPORT),
        wire(DELEGATE) + b" ",
        b'{"op":"delegate","project":"touch-hockey","task":"TH-0001","text":"\\ud800"}\n',
    ],
)
async def test_control_rejects_bad_frames_without_ingress_and_listener_survives(
    tmp_path: Path, frame: bytes
) -> None:
    path = tmp_path / "c.sock"
    requests: list[DelegateRequest | ReportRequest] = []

    async def receive(request: DelegateRequest | ReportRequest) -> str:
        requests.append(request)
        return ACTION_ID

    async with serving(path, receive):
        assert await exchange(path, frame) == {"ok": False, "error": "invalid request"}
        assert requests == []
        assert await send_control_request(
            path, DelegateRequest("touch-hockey", "TH-0001", "OK")
        ) == (ACTION_ID)
        assert [request.text for request in requests] == ["OK"]


@pytest.mark.asyncio
async def test_control_rejects_delayed_second_frame_before_any_ingress(tmp_path: Path) -> None:
    requests: list[DelegateRequest | ReportRequest] = []

    async def receive(request: DelegateRequest | ReportRequest) -> str:
        requests.append(request)
        return ACTION_ID

    async with serving(tmp_path / "c.sock", receive):
        response = await exchange(tmp_path / "c.sock", wire(DELEGATE), wire(REPORT))
        assert response == {"ok": False, "error": "invalid request"}
        assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (wire({**DELEGATE, "text": "é" * (64 * 1024 // 2 + 1)}), "content too large"),
        (b"x" * (128 * 1024) + b"\n", "request too large"),
        (b"x" * (128 * 1024 + 1), "request too large"),
    ],
    ids=["utf8-content", "frame-with-lf", "frame-without-lf"],
)
async def test_control_rejects_oversized_frames_and_content(
    tmp_path: Path, frame: bytes, error: str
) -> None:
    calls = 0

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        nonlocal calls
        calls += 1
        return ACTION_ID

    async with serving(tmp_path / "c.sock", receive):
        assert await exchange(tmp_path / "c.sock", frame) == {"ok": False, "error": error}
        assert calls == 0
        assert (await exchange(tmp_path / "c.sock", wire(DELEGATE)))["ok"] is True
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ControlError("unknown project"), "unknown project"),
        (ControlError(SECRET), "control request failed"),
        (RuntimeError(SECRET), "delegation outcome unknown"),
    ],
)
async def test_control_callback_failures_are_safe_and_listener_survives(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, error: Exception, category: str
) -> None:
    calls = 0

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return ACTION_ID

    async with serving(tmp_path / "c.sock", receive):
        response = await exchange(tmp_path / "c.sock", wire({**DELEGATE, "text": SECRET}))
        assert response == {"ok": False, "error": category}
        assert SECRET not in json.dumps(response) + caplog.text
        assert (await exchange(tmp_path / "c.sock", wire(DELEGATE)))["ok"] is True
        assert calls == 2


@pytest.mark.asyncio
async def test_control_does_not_acknowledge_invalid_callback_action_id(tmp_path: Path) -> None:
    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        return SECRET

    async with serving(tmp_path / "c.sock", receive):
        assert await exchange(tmp_path / "c.sock", wire(DELEGATE)) == {
            "ok": False,
            "error": "delegation outcome unknown",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error"),
    [
        (b"", "delegation outcome unknown"),
        (b"{", "delegation outcome unknown"),
        (b"\xff\n", "delegation outcome unknown"),
        (wire({"ok": True, "actionId": SECRET}), "delegation outcome unknown"),
        (wire({"ok": 1, "actionId": ACTION_ID}), "delegation outcome unknown"),
        (wire({"ok": True, "actionId": ACTION_ID, "extra": SECRET}), "delegation outcome unknown"),
        (wire({"ok": True, "actionId": ACTION_ID})[:-1], "delegation outcome unknown"),
        (wire({"ok": True, "actionId": ACTION_ID}) * 2, "delegation outcome unknown"),
        (b"x" * (128 * 1024 + 1), "delegation outcome unknown"),
        (wire({"ok": False, "error": "unknown project"}), "unknown project"),
        (wire({"ok": False, "error": SECRET}), "control request failed"),
        (wire({"ok": False, "error": {"detail": SECRET}}), "delegation outcome unknown"),
        (b'{"ok":false,"ok":false,"error":"unknown project"}\n', "delegation outcome unknown"),
    ],
    ids=[
        "eof",
        "malformed",
        "non-utf8",
        "invalid-uuid",
        "non-boolean-ok",
        "extra-key",
        "missing-lf",
        "multiple",
        "oversized",
        "safe-error",
        "unsafe-error",
        "non-string-error",
        "duplicate-key",
    ],
)
async def test_control_client_safe_response_failures_never_retry(
    tmp_path: Path, response: bytes, error: str
) -> None:
    async with replying(tmp_path / "c.sock", response) as received:
        with pytest.raises(ControlError) as caught:
            await send_control_request(
                tmp_path / "c.sock", DelegateRequest("touch-hockey", "TH-0001", SECRET)
            )
        assert str(caught.value) == error
        assert SECRET not in str(caught.value)
        assert len(received) == 1
        assert json.loads(received[0]) == {**DELEGATE, "text": SECRET}


@pytest.mark.asyncio
async def test_control_client_connection_failure_is_safe(tmp_path: Path) -> None:
    with pytest.raises(ControlError) as caught:
        await send_control_request(
            tmp_path / "private-secret-path.sock", DelegateRequest("touch-hockey", "TH-0001", "OK")
        )
    assert str(caught.value) == "control unavailable"
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control_request", "error"),
    [
        (DelegateRequest("touch-hockey", "TH-1", "OK"), "invalid request"),
        (DelegateRequest("touch-hockey", "TH-0001", ""), "invalid request"),
        (DelegateRequest("touch-hockey", "TH-0001", "é" * 32769), "content too large"),
        (DelegateRequest("touch-hockey", "TH-0001", "\x00" * 32768), "request too large"),
        (DelegateRequest("touch-hockey", "TH-0001", "\ud800"), "invalid request"),
        (ReportRequest("touch-hockey", "TH-0001", "done", "Done\n"), "invalid request"),
    ],
)
async def test_control_client_validates_before_connecting(
    tmp_path: Path, control_request: DelegateRequest | ReportRequest, error: str
) -> None:
    async with replying(tmp_path / "c.sock", wire({"ok": True, "actionId": ACTION_ID})) as received:
        with pytest.raises(ControlError) as caught:
            await send_control_request(tmp_path / "c.sock", control_request)
        assert str(caught.value) == error
        assert received == []


@pytest.mark.asyncio
async def test_control_request_deadline_closes_idle_peer_without_ingress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "_IO_TIMEOUT_SECONDS", 0.1)
    calls = 0

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        nonlocal calls
        calls += 1
        return ACTION_ID

    async with serving(tmp_path / "c.sock", receive):
        reader, writer = await asyncio.open_unix_connection(str(tmp_path / "c.sock"))
        try:
            writer.write(wire(DELEGATE))
            await writer.drain()
            # LF without a write-half-close is incomplete, not permission to dispatch.
            response = json.loads(await asyncio.wait_for(reader.read(), 2))
            assert response == {"ok": False, "error": "request timed out"}
            assert calls == 0
        finally:
            await close_writer(writer)
        assert (await exchange(tmp_path / "c.sock", wire(DELEGATE)))["ok"] is True


@pytest.mark.asyncio
async def test_control_callback_timeout_reaps_callback_and_reports_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "_IO_TIMEOUT_SECONDS", 0.1)
    cancelled = asyncio.Event()
    calls = 0

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        nonlocal calls
        calls += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ACTION_ID

    async with serving(tmp_path / "c.sock", receive):
        with pytest.raises(ControlError, match="delegation outcome unknown"):
            await send_control_request(
                tmp_path / "c.sock", DelegateRequest("touch-hockey", "TH-0001", "OK")
            )
        await asyncio.wait_for(cancelled.wait(), 2)
        assert calls == 1


@pytest.mark.asyncio
async def test_control_close_cancels_and_reaps_idle_and_dispatched_handlers(tmp_path: Path) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ACTION_ID

    path = tmp_path / "c.sock"
    async with serving(path, receive) as server:
        reader, writer = await asyncio.open_unix_connection(str(path))
        pending = asyncio.create_task(
            send_control_request(path, DelegateRequest("touch-hockey", "TH-0001", "OK"))
        )
        try:
            writer.write(b"{")
            await writer.drain()
            await asyncio.wait_for(entered.wait(), 2)
            await server.close()
            assert cancelled.is_set()
            assert await asyncio.wait_for(reader.read(), 2) == b""
            with pytest.raises(ControlError, match="delegation outcome unknown"):
                await pending
            assert not path.exists()
        finally:
            await close_writer(writer)
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_closes_accepted_writer_when_handler_never_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ControlServer(tmp_path / "c.sock", accepted)
    accepted_connection = asyncio.Event()
    handler_started = asyncio.Event()
    writers: list[asyncio.StreamWriter] = []
    original_accept = server._accept
    original_handle = server._handle

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        handler_started.set()
        await original_handle(reader, writer)

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Retaining the writer makes cleanup independent of garbage collection.
        writers.append(writer)
        original_accept(reader, writer)
        for task in server._handlers:
            task.cancel()
        accepted_connection.set()

    monkeypatch.setattr(server, "_handle", handle)
    monkeypatch.setattr(server, "_accept", accept)
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(tmp_path / "c.sock"))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(accepted_connection.wait(), 2)
        closing = asyncio.create_task(server.close())
        await asyncio.wait_for(asyncio.shield(closing), 2)
        assert not handler_started.is_set()
        assert writers[0].is_closing()
        assert await asyncio.wait_for(reader.read(), 2) == b""
    finally:
        # A pre-fix failure must not strand the test inside Server.wait_closed().
        for accepted_writer in writers:
            accepted_writer.close()
        await close_writer(writer)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await server.close()



@pytest.mark.asyncio
async def test_control_close_makes_progress_with_a_completed_handler(tmp_path: Path) -> None:
    handled: list[asyncio.Task[None]] = []

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        task = asyncio.current_task()
        assert task is not None
        handled.append(task)
        return ACTION_ID

    server = ControlServer(tmp_path / "c.sock", receive)
    await server.start()
    closing: asyncio.Task[None] | None = None
    try:
        assert await send_control_request(
            tmp_path / "c.sock", DelegateRequest("touch-hockey", "TH-0001", "Complete")
        ) == ACTION_ID
        await handled[0]
        # Model the interval before a completed handler's scheduled discard runs.
        server._handlers.add(handled[0])
        closing = asyncio.create_task(server.close())
        await asyncio.wait_for(asyncio.shield(closing), 2)
        assert not (tmp_path / "c.sock").exists()
    finally:
        server._handlers.difference_update(handled)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await server.close()

@pytest.mark.asyncio
async def test_control_close_finishes_cleanup_when_its_caller_is_cancelled(tmp_path: Path) -> None:
    entered = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelling.set()
            await release.wait()
        return ACTION_ID

    path = tmp_path / "c.sock"
    async with serving(path, receive) as server:
        pending = asyncio.create_task(
            send_control_request(path, DelegateRequest("touch-hockey", "TH-0001", "OK"))
        )
        closing: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            closing = asyncio.create_task(server.close())
            await asyncio.wait_for(cancelling.wait(), 2)
            closing.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert not path.exists()
            with pytest.raises(ControlError, match="delegation outcome unknown"):
                await pending
        finally:
            release.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_client_cancellation_closes_its_socket_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    opened: list[asyncio.StreamWriter] = []
    original_open = asyncio.open_unix_connection
    calls = 0

    async def capture(*args: object, **kwargs: object):
        reader, writer = await original_open(*args, **kwargs)
        opened.append(writer)
        return reader, writer

    async def receive(_request: DelegateRequest | ReportRequest) -> str:
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()
        return ACTION_ID

    monkeypatch.setattr(asyncio, "open_unix_connection", capture)
    async with serving(tmp_path / "c.sock", receive):
        pending = asyncio.create_task(
            send_control_request(
                tmp_path / "c.sock", DelegateRequest("touch-hockey", "TH-0001", "OK")
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert calls == 1
        assert len(opened) == 1
        assert opened[0].is_closing()
        await asyncio.wait_for(opened[0].wait_closed(), 2)


@pytest.mark.asyncio
async def test_control_replaces_stale_socket_but_refuses_live_listener(tmp_path: Path) -> None:
    path = tmp_path / "c.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    assert stat.S_ISSOCK(path.lstat().st_mode)
    async with serving(path):
        inode = path.lstat().st_ino
        contender = ControlServer(path, accepted)
        with pytest.raises(ControlError, match="control socket already in use"):
            await contender.start()
        await contender.close()
        assert path.lstat().st_ino == inode
        assert await send_control_request(
            path, DelegateRequest("touch-hockey", "TH-0001", "OK")
        ) == (ACTION_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [0o755, 0o770, 0o1700])
async def test_control_refuses_nonprivate_directory_without_chmod(
    tmp_path: Path, mode: int
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=mode)
    parent.chmod(mode)
    with pytest.raises(ControlError, match="unsafe control directory"):
        await ControlServer(parent / "c.sock", accepted).start()
    assert stat.S_IMODE(parent.lstat().st_mode) == mode
    assert not (parent / "c.sock").exists()


@pytest.mark.asyncio
async def test_control_refuses_symlink_and_wrong_owner_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ControlError, match="unsafe control directory"):
        await ControlServer(linked / "c.sock", accepted).start()
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.raises(ControlError, match="unsafe control directory"):
        await ControlServer(parent / "c.sock", accepted).start()
    assert linked.is_symlink()
    assert not (parent / "c.sock").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "symlink", "directory", "fifo"])
async def test_control_refuses_nonsockets_without_replacing_them(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "c.sock"
    target = tmp_path / "keep"
    target.write_text(SECRET)
    if kind == "file":
        path.write_text(SECRET)
    elif kind == "symlink":
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path, 0o600)
    before = path.lstat()
    with pytest.raises(ControlError, match="unsafe control socket"):
        await ControlServer(path, accepted).start()
    assert path.lstat().st_ino == before.st_ino
    assert path.lstat().st_mode == before.st_mode
    assert target.read_text() == SECRET


@pytest.mark.asyncio
async def test_control_refuses_wrong_owner_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    inode = path.lstat().st_ino
    original_stat = os.stat

    def foreign_stat(name: object, **kwargs: object) -> os.stat_result:
        result = original_stat(name, **kwargs)
        if name == path.name and kwargs.get("dir_fd") is not None:
            fields = list(result)
            fields[4] = result.st_uid + 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(os, "stat", foreign_stat)
    with pytest.raises(ControlError, match="unsafe control socket"):
        await ControlServer(path, accepted).start()
    assert path.lstat().st_ino == inode


@pytest.mark.asyncio
async def test_control_does_not_remove_path_replaced_during_stale_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    async def replace(_sock: socket.socket, _address: str) -> None:
        path.unlink()
        path.write_text(SECRET)
        raise ConnectionRefusedError(errno.ECONNREFUSED, SECRET)

    monkeypatch.setattr(asyncio.get_running_loop(), "sock_connect", replace)
    with pytest.raises(ControlError, match="control socket changed"):
        await ControlServer(path, accepted).start()
    assert path.read_text() == SECRET


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_start", [False, True])
async def test_control_probe_timeout_and_cancellation_close_probe_without_unlinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_start: bool
) -> None:
    path = tmp_path / "c.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()
    inode = path.lstat().st_ino
    entered = asyncio.Event()
    probes: list[socket.socket] = []

    async def hang(sock: socket.socket, _address: str) -> None:
        probes.append(sock)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio.get_running_loop(), "sock_connect", hang)
    monkeypatch.setattr(control, "_PROBE_TIMEOUT_SECONDS", 0.1)
    server = ControlServer(path, accepted)
    starting = asyncio.create_task(server.start())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel_start:
            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting
        else:
            with pytest.raises(ControlError, match="control socket already in use"):
                await starting
        assert path.lstat().st_ino == inode
        assert len(probes) == 1
        assert probes[0].fileno() == -1
    finally:
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
        await server.close()


@pytest.mark.asyncio
async def test_control_does_not_steal_listener_created_just_before_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.sock"
    original_bind = socket.socket.bind
    rival = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def raced_bind(sock: socket.socket, address: str) -> None:
        if address == str(path):
            original_bind(rival, address)
            rival.listen()
        original_bind(sock, address)

    monkeypatch.setattr(socket.socket, "bind", raced_bind)
    try:
        with pytest.raises(ControlError, match="control unavailable"):
            await ControlServer(path, accepted).start()
        assert stat.S_ISSOCK(path.lstat().st_mode)
        _reader, writer = await asyncio.open_unix_connection(str(path))
        await close_writer(writer)
    finally:
        rival.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_socket", [False, True])
async def test_control_close_only_unlinks_its_bound_inode(
    tmp_path: Path, replacement_socket: bool
) -> None:
    path = tmp_path / "c.sock"
    rival: socket.socket | None = None
    async with serving(path) as server:
        path.unlink()
        if replacement_socket:
            rival = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            rival.bind(str(path))
            rival.listen()
        else:
            path.write_text(SECRET)
        try:
            inode = path.lstat().st_ino
            await server.close()
            assert path.lstat().st_ino == inode
            if replacement_socket:
                _reader, writer = await asyncio.open_unix_connection(str(path))
                await close_writer(writer)
            else:
                assert path.read_text() == SECRET
        finally:
            if rival is not None:
                rival.close()


@pytest.mark.asyncio
async def test_control_close_preserves_symlink_to_renamed_bound_socket(tmp_path: Path) -> None:
    path = tmp_path / "c.sock"
    moved = tmp_path / "original.sock"
    async with serving(path) as server:
        bound_inode = path.lstat().st_ino
        path.rename(moved)
        path.symlink_to(moved.name)
        link_inode = path.lstat().st_ino
        assert (
            await send_control_request(
                path, DelegateRequest("touch-hockey", "TH-0001", "Before close")
            )
            == ACTION_ID
        )
        await server.close()
        assert path.is_symlink()
        assert path.lstat().st_ino == link_inode
        assert path.readlink() == Path("original.sock")
        assert moved.is_socket() and moved.lstat().st_ino == bound_inode


@pytest.mark.asyncio
async def test_control_close_keeps_replacement_directory_and_cleans_original(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "private"
    path = parent / "c.sock"
    async with serving(path) as server:
        moved = tmp_path / "moved"
        parent.rename(moved)
        parent.mkdir(mode=0o700)
        path.write_text(SECRET)
        await server.close()
        assert path.read_text() == SECRET
        assert not (moved / "c.sock").exists()


@pytest.mark.asyncio
async def test_control_server_restarts_at_the_same_workspace_visible_path(tmp_path: Path) -> None:
    path = tmp_path / "private" / "c.sock"
    server = ControlServer(path, accepted)
    try:
        await server.start()
        assert (
            await send_control_request(
                path, DelegateRequest("touch-hockey", "TH-0001", "Before restart")
            )
            == ACTION_ID
        )
        await server.close()
        assert not path.exists()
        await server.start()
        assert (
            await send_control_request(
                path, ReportRequest("touch-hockey", "TH-0001", "done", "After restart")
            )
            == ACTION_ID
        )
    finally:
        await server.close()


async def cli_process(tmp_path: Path, *arguments: str) -> tuple[int, str, str]:
    # Run the real argparse/main/client path, making either private-config
    # entry point fatal even when a caller has no readable config.
    bootstrap = (
        "from agentwire import cli\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('private config access')\n"
        "cli.load_config = forbidden\n"
        "cli.default_config_path = forbidden\n"
        "cli.main()\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        bootstrap,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "AGENTWIRE_CONFIG": str(tmp_path / "unreadable-private-config.toml"),
        },
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    assert process.returncode is not None
    return process.returncode, stdout.decode(), stderr.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delegate", "report"])
async def test_cli_scoped_success_uses_no_private_config_and_prints_only_receipt(
    tmp_path: Path, operation: str
) -> None:
    received: list[DelegateRequest | ReportRequest] = []

    async def accept(request: DelegateRequest | ReportRequest) -> str:
        received.append(request)
        return ACTION_ID

    arguments = [
        operation,
        "--socket",
        str(tmp_path / "c.sock"),
        "--project",
        "touch-hockey",
        "--task",
        "TH-0001",
        "--text",
        "Read the package name.",
    ]
    if operation == "report":
        arguments.extend(("--status", "blocked"))
    async with serving(tmp_path / "c.sock", accept):
        assert await cli_process(tmp_path, *arguments) == (0, f"{ACTION_ID}\n", "")
    assert received == [
        DelegateRequest("touch-hockey", "TH-0001", "Read the package name.")
        if operation == "delegate"
        else ReportRequest("touch-hockey", "TH-0001", "blocked", "Read the package name.")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "category"),
    [
        (wire({"ok": False, "error": "unknown project"}), "unknown project"),
        (b"", "delegation outcome unknown"),
        (wire({"ok": False, "error": SECRET}), "control request failed"),
    ],
)
async def test_cli_safe_failure_and_unknown_outcome_exit_once_without_private_config(
    tmp_path: Path, response: bytes, category: str
) -> None:
    async with replying(tmp_path / "c.sock", response) as received:
        result = await cli_process(
            tmp_path,
            "delegate",
            "--socket",
            str(tmp_path / "c.sock"),
            "--project",
            "touch-hockey",
            "--task",
            "TH-0001",
            "--text",
            SECRET,
        )
    assert result == (1, "", f"agentwire: {category}\n")
    assert len(received) == 1
    assert json.loads(received[0]) == {**DELEGATE, "text": SECRET}


@pytest.mark.asyncio
async def test_cli_connection_failure_never_discloses_socket_or_config_paths(
    tmp_path: Path,
) -> None:
    assert await cli_process(
        tmp_path,
        "report",
        "--socket",
        str(tmp_path / "missing-private-socket"),
        "--project",
        "touch-hockey",
        "--task",
        "TH-0001",
        "--status",
        "done",
        "--text",
        SECRET,
    ) == (1, "", "agentwire: control unavailable\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delegate", "report"])
async def test_cli_requires_every_scoped_option_and_exposes_no_channel_prompt(
    tmp_path: Path, operation: str
) -> None:
    options = {
        "--socket": str(tmp_path / "missing.sock"),
        "--project": "touch-hockey",
        "--task": "TH-0001",
        "--text": "Task",
    }
    if operation == "report":
        options["--status"] = "done"
    for missing in options:
        arguments = [operation]
        for key, value in options.items():
            if key != missing:
                arguments.extend((key, value))
        code, stdout, stderr = await cli_process(tmp_path, *arguments)
        assert code == 2 and stdout == ""
        assert missing in stderr
    complete = [operation, *(part for pair in options.items() for part in pair)]
    code, stdout, _stderr = await cli_process(tmp_path, *complete, "--channel", "#pm")
    assert code == 2 and stdout == ""
    if operation == "report":
        complete[complete.index("done")] = "running"
        code, stdout, _stderr = await cli_process(tmp_path, *complete)
        assert code == 2 and stdout == ""
    code, stdout, _stderr = await cli_process(
        tmp_path, "prompt", "--channel", "#pm", "--text", "Not a supported operation"
    )
    assert code == 2 and stdout == ""
