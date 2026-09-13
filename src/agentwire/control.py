from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import re
import socket
import stat
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_MAX_FRAME_BYTES = 128 * 1024
_MAX_CONTENT_BYTES = 64 * 1024
_IO_TIMEOUT_SECONDS = 30.0
_PROBE_TIMEOUT_SECONDS = 1.0
_CLOSE_TIMEOUT_SECONDS = 5.0
_TASK_ID = re.compile(r"[A-Z][A-Z0-9]*-[0-9]{4}")
_SAFE_ERRORS = frozenset(
    {
        "invalid request",
        "request too large",
        "content too large",
        "request timed out",
        "control request failed",
        "control unavailable",
        "unsafe control directory",
        "unsafe control socket",
        "control socket already in use",
        "control socket changed",
        "unknown project",
        "channel unavailable",
        "binding changed",
        "request not accepted",
        "delegation outcome unknown",
    }
)


class ControlError(RuntimeError):
    """A fixed, safe category suitable for a control response or CLI diagnostic."""

    def __init__(self, category: str) -> None:
        super().__init__(
            category
            if isinstance(category, str) and category in _SAFE_ERRORS
            else "control request failed"
        )


@dataclass(slots=True, frozen=True)
class DelegateRequest:
    project: str
    task: str
    text: str


@dataclass(slots=True, frozen=True)
class ReportRequest:
    project: str
    task: str
    status: Literal["done", "blocked"]
    text: str


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ControlError("invalid request")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> None:
    raise ControlError("invalid request")


def _decode_json(frame: bytes) -> object:
    try:
        return json.loads(
            frame.decode("utf-8"), object_pairs_hook=_object, parse_constant=_invalid_constant
        )
    except (ValueError, RecursionError):
        raise ControlError("invalid request") from None


def _request(value: object) -> DelegateRequest | ReportRequest:
    if not isinstance(value, dict):
        raise ControlError("invalid request")
    op = value.get("op")
    keys = {"op", "project", "task", "text"}
    if op == "report":
        keys.add("status")
    if op not in ("delegate", "report") or value.keys() != keys:
        raise ControlError("invalid request")
    project, task, text = value["project"], value["task"], value["text"]
    if (
        not isinstance(project, str)
        or not project.strip()
        or not isinstance(task, str)
        or _TASK_ID.fullmatch(task) is None
        or not isinstance(text, str)
        or not text.strip()
    ):
        raise ControlError("invalid request")
    try:
        project.encode("utf-8")
        content_size = len(text.encode("utf-8"))
    except UnicodeError:
        raise ControlError("invalid request") from None
    if content_size > _MAX_CONTENT_BYTES:
        raise ControlError("content too large")
    if op == "report":
        status = value["status"]
        if status not in ("done", "blocked") or text.splitlines() != [text]:
            raise ControlError("invalid request")
        return ReportRequest(project, task, status, text)
    return DelegateRequest(project, task, text)


def _encode_request(request: DelegateRequest | ReportRequest) -> bytes:
    if type(request) not in (DelegateRequest, ReportRequest):
        raise ControlError("invalid request")
    value = {
        "op": "report" if isinstance(request, ReportRequest) else "delegate",
        "project": request.project,
        "task": request.task,
        "text": request.text,
    }
    if isinstance(request, ReportRequest):
        value["status"] = request.status
    _request(value)
    frame = _encode_json(value)
    if len(frame) > _MAX_FRAME_BYTES:
        raise ControlError("request too large")
    return frame


def _encode_json(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _action_id(value: object) -> str:
    if isinstance(value, str) and len(value) == 36:
        with contextlib.suppress(ValueError):
            if str(uuid.UUID(value)) == value:
                return value
    raise ControlError("delegation outcome unknown")


async def _read_frame(reader: asyncio.StreamReader) -> bytes:
    try:
        frame = await reader.readuntil(b"\n")
    except asyncio.LimitOverrunError:
        raise ControlError("request too large") from None
    except asyncio.IncompleteReadError:
        raise ControlError("invalid request") from None
    if len(frame) > _MAX_FRAME_BYTES:
        raise ControlError("request too large")
    # A write-half-close makes delayed trailing frames rejectable before ingress.
    # Responses likewise end with EOF, since each connection has just one exchange.
    if await reader.read(1):
        raise ControlError("invalid request")
    return frame


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError, TimeoutError):
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await writer.wait_closed()


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


class ControlServer:
    def __init__(
        self,
        path: Path,
        on_request: Callable[[DelegateRequest | ReportRequest], Awaitable[str]],
    ) -> None:
        self.path = path
        self.on_request = on_request
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task[None]] = set()
        self._directory_fd: int | None = None
        self._directory_identity: tuple[int, int] | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._closing = True
        self._lifecycle_lock = asyncio.Lock()

    def _check_directory(self) -> None:
        info = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or (
                self._directory_identity is not None and _identity(info) != self._directory_identity
            )
        ):
            raise ControlError("unsafe control directory")

    def _entry(self) -> os.stat_result | None:
        assert self._directory_fd is not None
        try:
            return os.stat(self.path.name, dir_fd=self._directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._server is not None:
                return
            sock: socket.socket | None = None
            try:
                with contextlib.suppress(FileExistsError):
                    self.path.parent.mkdir(mode=0o700)
                self._check_directory()
                self._directory_fd = os.open(
                    self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
                )
                self._directory_identity = _identity(os.fstat(self._directory_fd))
                self._check_directory()
                existing = self._entry()
                if existing is not None:
                    if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != os.getuid():
                        raise ControlError("unsafe control socket")
                    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    probe.setblocking(False)
                    try:
                        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
                            await asyncio.get_running_loop().sock_connect(probe, str(self.path))
                    except TimeoutError:
                        raise ControlError("control socket already in use") from None
                    except OSError as exc:
                        if exc.errno != errno.ECONNREFUSED:
                            raise ControlError("control socket changed") from None
                    else:
                        raise ControlError("control socket already in use")
                    finally:
                        probe.close()
                    self._check_directory()
                    current = self._entry()
                    if (
                        current is None
                        or _identity(current) != _identity(existing)
                        or not stat.S_ISSOCK(current.st_mode)
                        or current.st_uid != os.getuid()
                    ):
                        raise ControlError("control socket changed")
                    os.unlink(self.path.name, dir_fd=self._directory_fd)
                self._check_directory()
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.setblocking(False)
                # Supplying a bound socket avoids asyncio's automatic stale-path unlink.
                # A listener arriving after our probe must win, not be silently removed.
                sock.bind(str(self.path))
                current = self._entry()
                if (
                    current is None
                    or not stat.S_ISSOCK(current.st_mode)
                    or current.st_uid != os.getuid()
                ):
                    raise ControlError("control socket changed")
                self._socket_identity = _identity(current)
                os.chmod(self.path.name, 0o600, dir_fd=self._directory_fd, follow_symlinks=False)
                self._check_directory()
                sock.listen()
                # asyncio 3.13+ follows symlinks when cleaning up its socket path.
                # Only our no-follow, directory-relative inode check may unlink it.
                self._server = await asyncio.start_unix_server(
                    self._accept,
                    sock=sock,
                    limit=_MAX_FRAME_BYTES,
                    start_serving=False,
                    **({"cleanup_socket": False} if sys.version_info >= (3, 13) else {}),
                )
                sock = None
                self._closing = False
                await self._server.start_serving()
            except BaseException as exc:
                if sock is not None:
                    sock.close()
                await self._shutdown()
                if isinstance(exc, ControlError) or not isinstance(exc, Exception):
                    raise
                raise ControlError("control unavailable") from None

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._handle(reader, writer))
        self._handlers.add(task)

        def finished(done: asyncio.Task[None]) -> None:
            self._handlers.discard(done)
            # Cancellation before the coroutine's first step skips its finally.
            # The accept callback owns the writer until this task is finished.
            writer.close()

        task.add_done_callback(finished)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        dispatched = False
        try:
            if self._closing:
                return
            try:
                async with asyncio.timeout(_IO_TIMEOUT_SECONDS):
                    request = _request(_decode_json(await _read_frame(reader)))
                    dispatched = True
                    action_id = _action_id(await self.on_request(request))
                    response = {"ok": True, "actionId": action_id}
            except ControlError as exc:
                response = {"ok": False, "error": str(exc)}
            except TimeoutError:
                response = {
                    "ok": False,
                    "error": "delegation outcome unknown" if dispatched else "request timed out",
                }
            except Exception:
                response = {
                    "ok": False,
                    "error": "delegation outcome unknown" if dispatched else "invalid request",
                }
            with contextlib.suppress(OSError, TimeoutError):
                async with asyncio.timeout(_IO_TIMEOUT_SECONDS):
                    writer.write(_encode_json(response))
                    await writer.drain()
        finally:
            await _close_writer(writer)

    async def _shutdown(self) -> None:
        self._closing = True
        if self._server is not None:
            self._server.close()
        while self._handlers:
            tasks = tuple(self._handlers)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # gather may complete synchronously for already-finished tasks.
            # Do not spin waiting for their scheduled done callbacks to run.
            self._handlers.difference_update(tasks)
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        if self._directory_fd is not None:
            try:
                current = self._entry()
                if (
                    current is not None
                    and _identity(current) == self._socket_identity
                    and stat.S_ISSOCK(current.st_mode)
                    and current.st_uid == os.getuid()
                ):
                    os.unlink(self.path.name, dir_fd=self._directory_fd)
            except OSError:
                pass
            finally:
                os.close(self._directory_fd)
                self._directory_fd = None
                self._directory_identity = None
                self._socket_identity = None

    async def _close(self) -> None:
        async with self._lifecycle_lock:
            await self._shutdown()

    async def close(self) -> None:
        task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise


async def send_control_request(path: Path, request: DelegateRequest | ReportRequest) -> str:
    """Send once, half-close the request, and return only a confirmed action UUID."""
    frame = _encode_request(request)
    try:
        async with asyncio.timeout(_IO_TIMEOUT_SECONDS):
            reader, writer = await asyncio.open_unix_connection(str(path), limit=_MAX_FRAME_BYTES)
    except (OSError, ValueError, TimeoutError):
        raise ControlError("control unavailable") from None
    try:
        try:
            async with asyncio.timeout(_IO_TIMEOUT_SECONDS):
                writer.write(frame)
                writer.write_eof()
                await writer.drain()
                response = _decode_json(await _read_frame(reader))
        except (OSError, ValueError, TimeoutError, ControlError):
            raise ControlError("delegation outcome unknown") from None
        if not isinstance(response, dict):
            raise ControlError("delegation outcome unknown")
        if response.get("ok") is True and response.keys() == {"ok", "actionId"}:
            return _action_id(response["actionId"])
        if (
            response.get("ok") is False
            and response.keys() == {"ok", "error"}
            and isinstance(response["error"], str)
        ):
            raise ControlError(response["error"])
        raise ControlError("delegation outcome unknown")
    finally:
        await _close_writer(writer)
