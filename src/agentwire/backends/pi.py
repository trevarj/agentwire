from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentwire.backends.base import Backend, BackendError
from agentwire.config import PiConfig
from agentwire.models import (
    BackendEvent,
    HistoryPage,
    Question,
    SessionSummary,
)
from agentwire.text import clean_block, safe_one_line, truncate_utf8

_MAX_TOOL_PAYLOAD_BYTES = 32 * 1024
_MAX_LINE_BYTES = 8 * 1024 * 1024
_COMMAND_TIMEOUT = 30.0
# Socket discovery cadence: directory scan is tiny, and a new TUI should appear
# before a second phone tap.
_DISCOVER_SECONDS = 0.5
_SESSION_LIST_LIMIT = 20
# pi-tui-kit renders questionnaire titles as "<header>: <prompt>" and appends a
# synthetic free-form row to every select; both are undone for the clients.
_UI_HEADER_MAX = 24
_UI_CUSTOM_ROW = re.compile(r"^\d+\. Other \(free-form\)$")
# Subagent rows the extension reports; the registry is already bounded there, so
# the bridge only has to keep the shape safe.
_SUBAGENT_STATUSES = frozenset({"queued", "running", "completed", "failed"})
_SUBAGENT_METRICS = ("toolUses", "durationMs", "tokens")

# pi's built-in tool names, mapped onto the vocabulary the bridge and its
# clients already render for the other backends.
_TOOL_KINDS = {
    "bash": "shell",
    "edit": "file edit",
    "write": "file edit",
    "read": "file read",
    "grep": "file read",
    "find": "file read",
    "ls": "file read",
    "web_search": "web search",
    "fetch_content": "web",
    "agent": "agent",
    "task": "agent",
}


def _session_stem(session_file: str | None) -> str | None:
    """A pi session's durable identity is its JSONL file name."""
    if not session_file:
        return None
    name = Path(session_file).name
    return name[: -len(".jsonl")] if name.endswith(".jsonl") else name


def _cwd_dir_name(cwd: str) -> str:
    """Map a workspace path onto pi's per-directory session folder name."""
    return f"--{cwd.strip('/').replace('/', '-')}--"


def _subagent_event(backend: str, session_id: str, frame: Mapping[str, Any]) -> BackendEvent | None:
    """Translate one extension `subagent_update` frame into a backend event.

    The extension is trusted for bounds but not for shape: only the allowlisted
    members survive, so a future field cannot reach a client unreviewed.
    """
    raw = frame.get("agents")
    if not isinstance(raw, list):
        return None
    agents: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        identifier = str(item.get("id") or "")
        status = str(item.get("status") or "")
        if not identifier or status not in _SUBAGENT_STATUSES:
            continue
        agent: dict[str, Any] = {
            "id": safe_one_line(identifier, 200),
            "type": safe_one_line(str(item.get("type") or "agent"), 200),
            "description": safe_one_line(str(item.get("description") or ""), 200),
            "status": status,
            "isBackground": bool(item.get("isBackground")),
        }
        for key in _SUBAGENT_METRICS:
            value = item.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                agent[key] = value
        agents.append(agent)
    return BackendEvent(
        kind="subagent_update",
        backend=backend,
        session_id=session_id,
        data={"agents": agents},
    )


def _entry_millis(entry: Mapping[str, Any]) -> int:
    message = entry.get("message")
    if isinstance(message, Mapping):
        stamp = message.get("timestamp")
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
            return int(stamp)
    raw = entry.get("timestamp")
    if isinstance(raw, str):
        try:
            return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return 0
    return 0


class _Transport:
    """One JSONL command/event stream: an extension socket or an RPC stdio pair.

    Both sides speak the same protocol (the extension deliberately mirrors pi's
    RPC mode), so the backend needs exactly one pump and one command path.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter | Any,
        process: asyncio.subprocess.Process | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.process = process
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._sequence = 0
        self.closed = False

    async def request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise BackendError("pi session connection is closed")
        self._sequence += 1
        token = f"aw-{self._sequence}"
        command = {**command, "id": token}
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[token] = future
        try:
            await self.write(command)
            response = await asyncio.wait_for(future, _COMMAND_TIMEOUT)
        except TimeoutError as exc:
            raise BackendError("pi did not answer in time") from exc
        finally:
            self._pending.pop(token, None)
        if not response.get("success"):
            raise BackendError(
                safe_one_line(str(response.get("error") or "pi rejected the command"), 180)
            )
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    async def write(self, frame: dict[str, Any]) -> None:
        try:
            self.writer.write(json.dumps(frame).encode("utf-8") + b"\n")
            await self.writer.drain()
        except (OSError, ConnectionError) as exc:
            raise BackendError(f"pi session connection failed: {exc}") from exc

    def dispatch_response(self, frame: dict[str, Any]) -> bool:
        token = frame.get("id")
        future = self._pending.get(token) if isinstance(token, str) else None
        if future is None or future.done():
            return False
        future.set_result(frame)
        return True

    async def close(self) -> None:
        self.closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(BackendError("pi session connection closed"))
        self._pending.clear()
        with contextlib.suppress(Exception):
            self.writer.close()
        if self.process is not None and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()


@dataclass(slots=True)
class _Session:
    """One pi session: a live TUI reached over its extension socket, or a
    bridge-owned ``pi --mode rpc`` subprocess."""

    id: str
    cwd: str
    transport: _Transport
    tui: bool
    session_file: str | None = None
    title: str = "untitled"
    updated_at: float = 0.0
    pump: asyncio.Task[None] | None = None
    busy: bool = False
    turn_id: str | None = None
    last_reply: str | None = None
    # Prompts this bridge injected, whose user-message echoes must not open a
    # second turn or repeat on the wire.
    expected_prompts: int = 0
    tools: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    socket_path: Path | None = None


class PiBackend(Backend):
    """Expose pi sessions through the Agentwire backend contract.

    pi has no shared server. A running TUI is reached through the socket its
    Agentwire extension serves; a session nobody is running is resumed (and a
    fresh one created) by spawning ``pi --mode rpc``, which speaks the same
    JSONL protocol over stdio. ``AGENTWIRE_SPAWNED=1`` stops the extension in
    spawned processes from registering a duplicate socket.
    """

    name = "pi"

    def __init__(self, config: PiConfig) -> None:
        self.config = config
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._sessions: dict[str, _Session] = {}
        self._known_sockets: set[Path] = set()
        self._discover: asyncio.Task[None] | None = None
        # Extension UI dialogs from spawned sessions relayed as questions and
        # approvals; the value carries what the JSONL response needs.
        # The third slot keeps the stripped "Other (free-form)" row so a typed
        # answer can be routed back through it.
        self._ui_requests: dict[str, tuple[_Session, dict[str, Any], str | None]] = {}
        self._setting_options: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._ready.is_set():
            return
        self._closed = False
        self._ready.set()
        await self._discover_sockets()
        self._discover = asyncio.create_task(self._discover_loop(), name="pi-discover")
        await self._events.put(BackendEvent(kind="connected", backend=self.name))

    async def wait_ready(self, timeout: float = 30) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except TimeoutError as exc:
            raise BackendError("timed out waiting for pi") from exc

    async def close(self) -> None:
        self._closed = True
        self._ready.clear()
        if self._discover is not None:
            self._discover.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._discover
            self._discover = None
        self._ui_requests.clear()
        for session in list(self._sessions.values()):
            await self._close_session(session)
        self._sessions.clear()
        self._known_sockets.clear()

    async def _close_session(self, session: _Session) -> None:
        if session.pump is not None and session.pump is not asyncio.current_task():
            session.pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.pump
        session.pump = None
        await session.transport.close()

    def events(self) -> AsyncIterator[BackendEvent]:
        async def iterate() -> AsyncIterator[BackendEvent]:
            while True:
                yield await self._events.get()

        return iterate()

    # ------------------------------------------------------------------
    # live socket discovery
    # ------------------------------------------------------------------

    async def _discover_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(_DISCOVER_SECONDS)
            with contextlib.suppress(Exception):
                await self._discover_sockets()

    async def _discover_sockets(self) -> None:
        try:
            paths = sorted(self.config.socket_dir.glob("*.sock"))
        except OSError:
            return
        for path in paths:
            if path in self._known_sockets:
                continue
            self._known_sockets.add(path)
            try:
                await self._connect_socket(path)
            except (OSError, BackendError, json.JSONDecodeError):
                # A stale socket from a dead pi; forget it so a reused pid
                # with a fresh socket file is retried.
                self._known_sockets.discard(path)
                with contextlib.suppress(OSError):
                    if not await self._socket_alive(path):
                        path.unlink()

    @staticmethod
    async def _socket_alive(path: Path) -> bool:
        try:
            _reader, writer = await asyncio.open_unix_connection(str(path))
        except OSError:
            return False
        writer.close()
        return True

    async def _connect_socket(self, path: Path) -> None:
        reader, writer = await asyncio.open_unix_connection(str(path), limit=_MAX_LINE_BYTES)
        transport = _Transport(reader, writer)
        line = await asyncio.wait_for(reader.readline(), 10)
        hello = json.loads(line.decode("utf-8"))
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            await transport.close()
            raise BackendError(f"unexpected first frame from {path}")
        session = self._register(hello, transport, tui=True)
        if session is not None:
            # `hello` carries the current list, so a reconnect does not have to
            # wait for the next lifecycle event to repopulate the client.
            self._queue_subagents(session, hello)
        if session is None:
            # Another connection already serves this session; keep the path
            # marked known so discovery does not reconnect every sweep.
            await transport.close()
            return
        session.socket_path = path

    def _register(
        self, state: Mapping[str, Any], transport: _Transport, tui: bool
    ) -> _Session | None:
        session_file = state.get("sessionFile")
        session_id = (
            _session_stem(session_file if isinstance(session_file, str) else None)
            or str(state.get("sessionId") or "")
            or str(uuid.uuid4())
        )
        existing = self._sessions.get(session_id)
        if existing is not None and not existing.transport.closed:
            return None
        session = _Session(
            id=session_id,
            cwd=str(state.get("cwd") or ""),
            transport=transport,
            tui=tui,
            session_file=session_file if isinstance(session_file, str) else None,
            title=safe_one_line(str(state.get("sessionName") or "") or "untitled", 100),
            updated_at=time.time(),
            busy=bool(state.get("busy")),
        )
        self._sessions[session_id] = session
        session.pump = asyncio.create_task(self._pump(session), name=f"pi-{session_id}")
        # The first status a client sees for this session; the queue is unbounded,
        # so a synchronous registration never blocks on it.
        self._events.put_nowait(self._status_event(session))
        return session

    def _status_event(self, session: _Session) -> BackendEvent:
        """One session's liveness, for the client session drawer.

        pi pushes state changes, so status is reported where those arrive rather
        than polled.
        """
        waiting = any(owner is session for owner, *_ in self._ui_requests.values())
        return BackendEvent(
            kind="status_changed",
            backend=self.name,
            session_id=session.id,
            data={
                "busy": session.busy,
                "active_flags": ["waiting"] if waiting else [],
                "cwd": session.cwd,
                "tui": session.tui,
            },
        )

    # ------------------------------------------------------------------
    # event pump and translation
    # ------------------------------------------------------------------

    async def _pump(self, session: _Session) -> None:
        try:
            while True:
                line = await session.transport.reader.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(frame, dict):
                    continue
                if frame.get("type") == "response":
                    session.transport.dispatch_response(frame)
                    continue
                await self._handle_frame(session, frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                await self._drop_session(session, f"pi session stream failed: {exc}")
            return
        if not self._closed:
            await self._drop_session(session, "pi session ended")

    async def _drop_session(self, session: _Session, reason: str) -> None:
        self._sessions.pop(session.id, None)
        if session.socket_path is not None:
            self._known_sockets.discard(session.socket_path)
        await session.transport.close()
        if session.busy and session.turn_id is not None:
            await self._events.put(
                BackendEvent(
                    kind="turn_failed",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=session.turn_id,
                    text=safe_one_line(reason, 180),
                )
            )
        await self._events.put(
            BackendEvent(
                kind="disconnected",
                backend=self.name,
                session_id=session.id,
                text=safe_one_line(reason, 180),
            )
        )

    async def _handle_frame(self, session: _Session, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "session_changed":
            await self._session_changed(session, frame)
        elif kind == "agent_start":
            session.busy = True
            session.updated_at = time.time()
            await self._events.put(self._status_event(session))
        elif kind == "agent_settled":
            await self._settle_turn(session)
        elif kind == "message_end":
            await self._handle_message(session, frame.get("message"))
        elif kind == "tool_execution_start":
            await self._handle_tool_start(session, frame)
        elif kind == "tool_execution_end":
            await self._handle_tool_end(session, frame)
        elif kind == "extension_ui_request":
            await self._handle_ui_request(session, frame)
        elif kind == "subagent_update":
            event = _subagent_event(self.name, session.id, frame)
            if event is not None:
                await self._events.put(event)
        # message_start/message_update (RPC streaming) and fire-and-forget UI
        # methods carry nothing the bridge renders; they are dropped here.

    async def _session_changed(self, session: _Session, frame: Mapping[str, Any]) -> None:
        """The TUI switched, forked, or renamed its session in place."""
        name = str(frame.get("sessionName") or "")
        if name:
            session.title = safe_one_line(name, 100)
        session_file = frame.get("sessionFile")
        stem = _session_stem(session_file if isinstance(session_file, str) else None)
        if stem and stem != session.id:
            self._sessions.pop(session.id, None)
            if session.busy and session.turn_id is not None:
                await self._settle_turn(session)
            session.id = stem
            session.session_file = str(session_file)
            session.cwd = str(frame.get("cwd") or session.cwd)
            session.tools.clear()
            session.expected_prompts = 0
            self._sessions[stem] = session
        session.updated_at = time.time()
        await self._events.put(self._status_event(session))
        self._queue_subagents(session, frame)

    def _queue_subagents(self, session: _Session, state: Mapping[str, Any]) -> None:
        """Publish the subagent list a state payload carries, when it carries one."""
        if "subagents" not in state:
            return
        event = _subagent_event(self.name, session.id, {"agents": state["subagents"]})
        if event is not None:
            self._events.put_nowait(event)

    def _open_turn(self, session: _Session) -> tuple[str, list[BackendEvent]]:
        if session.turn_id is not None:
            return session.turn_id, []
        turn_id = str(uuid.uuid4())
        session.turn_id = turn_id
        session.busy = True
        return turn_id, [
            BackendEvent(
                kind="turn_started",
                backend=self.name,
                session_id=session.id,
                turn_id=turn_id,
            )
        ]

    async def _settle_turn(self, session: _Session) -> None:
        session.busy = False
        session.updated_at = time.time()
        # Reported before the early return: settling with no open turn is still
        # the moment this session went idle.
        await self._events.put(self._status_event(session))
        turn_id = session.turn_id
        if turn_id is None:
            return
        session.turn_id = None
        session.tools.clear()
        await self._events.put(
            BackendEvent(
                kind="turn_done",
                backend=self.name,
                session_id=session.id,
                turn_id=turn_id,
            )
        )

    def _normalize_message(self, message: Any) -> dict[str, Any] | None:
        """Accept both wire shapes: the extension's condensed messages and the
        raw AgentMessage payloads pi's RPC mode streams."""
        if not isinstance(message, dict):
            return None
        if "text" in message:
            return message
        return self._condense_file_message(message)

    async def _handle_message(self, session: _Session, raw: Any) -> None:
        message = self._normalize_message(raw)
        if message is None:
            return
        role = message.get("role")
        text = str(message.get("text") or "").strip()
        session.updated_at = time.time()
        if role == "user":
            if session.expected_prompts > 0:
                # The echo of a prompt this bridge already announced.
                session.expected_prompts -= 1
                return
            turn_id, events = self._open_turn(session)
            for event in events:
                await self._events.put(event)
            if text:
                await self._events.put(
                    BackendEvent(
                        kind="user_prompt",
                        backend=self.name,
                        session_id=session.id,
                        turn_id=turn_id,
                        text=text,
                    )
                )
            return
        if role != "assistant":
            return
        turn_id, events = self._open_turn(session)
        for event in events:
            await self._events.put(event)
        stop_reason = message.get("stopReason")
        tool_calls = message.get("toolCalls")
        narration = stop_reason == "toolUse" or bool(tool_calls)
        if stop_reason == "aborted":
            # The interrupted turn closes on agent_settled; partial text is
            # not a reply.
            return
        if stop_reason == "error":
            detail = str(message.get("errorMessage") or text or "pi turn failed")
            session.turn_id = None
            session.tools.clear()
            await self._events.put(
                BackendEvent(
                    kind="turn_failed",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=turn_id,
                    text=safe_one_line(detail, 180),
                )
            )
            return
        if not text:
            return
        if narration:
            await self._events.put(
                BackendEvent(
                    kind="progress",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=turn_id,
                    text=text,
                )
            )
            return
        session.last_reply = text
        await self._events.put(
            BackendEvent(
                kind="assistant",
                backend=self.name,
                session_id=session.id,
                turn_id=turn_id,
                text=text,
            )
        )

    async def _handle_tool_start(self, session: _Session, frame: Mapping[str, Any]) -> None:
        item_id = str(frame.get("toolCallId") or "")
        name = str(frame.get("toolName") or "tool")
        turn_id, events = self._open_turn(session)
        for event in events:
            await self._events.put(event)
        kind = self._tool_kind(name)
        data = self._tool_metadata(name, frame.get("args"))
        session.tools[item_id] = (kind, data)
        await self._events.put(
            BackendEvent(
                kind="tool_started",
                backend=self.name,
                session_id=session.id,
                turn_id=turn_id,
                item_id=item_id,
                tool_kind=kind,
                data=data,
            )
        )

    async def _handle_tool_end(self, session: _Session, frame: Mapping[str, Any]) -> None:
        item_id = str(frame.get("toolCallId") or "")
        entry = session.tools.pop(item_id, None)
        if entry is None:
            kind = self._tool_kind(str(frame.get("toolName") or "tool"))
            data: dict[str, Any] = {}
        else:
            kind, data = entry
        failed = bool(frame.get("isError"))
        payload = dict(data)
        payload["status"] = "error" if failed else "completed"
        output = frame.get("output")
        if isinstance(output, str) and output:
            payload["output"] = truncate_utf8(clean_block(output), _MAX_TOOL_PAYLOAD_BYTES)
        elif isinstance(frame.get("result"), dict):
            text = self._result_text(frame["result"].get("content"))
            if text:
                payload["output"] = truncate_utf8(clean_block(text), _MAX_TOOL_PAYLOAD_BYTES)
        await self._events.put(
            BackendEvent(
                kind="tool_finished",
                backend=self.name,
                session_id=session.id,
                turn_id=session.turn_id,
                item_id=item_id,
                tool_kind=kind,
                success=not failed,
                data=payload,
            )
        )

    # ------------------------------------------------------------------
    # extension UI relay (spawned sessions)
    # ------------------------------------------------------------------

    async def _handle_ui_request(self, session: _Session, frame: dict[str, Any]) -> None:
        method = str(frame.get("method") or "")
        token = str(frame.get("id") or "")
        if not token or method not in {"select", "confirm", "input", "editor"}:
            # Fire-and-forget methods (notify, setStatus, ...) render nothing.
            return
        title = safe_one_line(str(frame.get("title") or "pi extension request"), 160)
        if method == "confirm":
            self._ui_requests[token] = (session, frame, None)
            message = safe_one_line(str(frame.get("message") or ""), 160)
            await self._events.put(
                BackendEvent(
                    kind="approval",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=session.turn_id,
                    request_token=token,
                    text=f"{title}: {message}" if message else title,
                )
            )
            return
        options = tuple(str(option) for option in frame.get("options") or () if str(option).strip())
        custom_row: str | None = None
        if method == "select" and options and _UI_CUSTOM_ROW.match(options[-1]):
            # The free-form row is an input affordance, not a real choice.
            custom_row = options[-1]
            options = options[:-1]
        header, prompt = self._split_title(title)
        self._ui_requests[token] = (session, frame, custom_row)
        await self._events.put(
            BackendEvent(
                kind="question",
                backend=self.name,
                session_id=session.id,
                turn_id=session.turn_id,
                request_token=token,
                questions=(
                    Question(
                        id="1",
                        header=header,
                        prompt=prompt,
                        options=options,
                        custom=method != "select" or custom_row is not None,
                    ),
                ),
            )
        )

    async def resolve_approval(self, request_token: str | int, allow: bool) -> None:
        entry = self._ui_requests.pop(str(request_token), None)
        if entry is None:
            raise BackendError("approval was already resolved")
        session, frame, _ = entry
        await session.transport.write(
            {"type": "extension_ui_response", "id": frame.get("id"), "confirmed": allow}
        )

    @staticmethod
    def _split_title(title: str) -> tuple[str, str]:
        """Split a pi-tui-kit "<header>: <prompt>" title, else reuse the title."""
        header, sep, prompt = title.partition(": ")
        if sep and prompt.strip() and header.strip() and len(header) <= _UI_HEADER_MAX:
            return header, prompt
        return title, title

    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None:
        entry = self._ui_requests.pop(str(request_token), None)
        if entry is None:
            raise BackendError("question was already resolved")
        session, frame, custom_row = entry
        response: dict[str, Any] = {"type": "extension_ui_response", "id": frame.get("id")}
        first = next(iter(answers or ()), ())
        value = next(iter(first), None)
        if answers is None or value is None:
            response["cancelled"] = True
        else:
            if custom_row is not None and value not in {
                str(option) for option in frame.get("options") or ()
            }:
                # Typed text: pick the free-form row so pi-tui-kit follows up
                # with the editor request that carries the real answer.
                value = custom_row
            response["value"] = value
        await session.transport.write(response)

    # ------------------------------------------------------------------
    # tool presentation
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_kind(name: str) -> str:
        return _TOOL_KINDS.get(name.lower(), "tool")

    @staticmethod
    def _tool_metadata(name: str, payload: Any) -> dict[str, Any]:
        values = payload if isinstance(payload, dict) else {}

        def field_text(*keys: str) -> str:
            for key in keys:
                value = values.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        data: dict[str, Any] = {}
        lowered = name.lower()
        if lowered == "bash":
            command = field_text("command")
            if command:
                data["label"] = f"$ {safe_one_line(command, 160)}"
                data["input"] = truncate_utf8(clean_block(command), _MAX_TOOL_PAYLOAD_BYTES)
        elif lowered in {"read", "write", "edit"}:
            path = field_text("path", "file_path")
            verb = "Read" if lowered == "read" else "Edit"
            data["label"] = safe_one_line(f"{verb} {path}" if path else name, 160)
        elif lowered in {"grep", "find", "glob"}:
            pattern = field_text("pattern", "query")
            data["label"] = safe_one_line(f"Search: {pattern}" if pattern else name, 160)
        elif lowered == "web_search":
            query = field_text("query")
            data["label"] = safe_one_line(f"Search: {query}" if query else name, 160)
        elif lowered == "fetch_content":
            url = field_text("url")
            data["label"] = safe_one_line(f"Fetch: {url}" if url else name, 160)
        elif lowered in {"agent", "task"}:
            description = field_text("description", "subagent_type")
            data["label"] = safe_one_line(description or name, 160)
        else:
            data["label"] = safe_one_line(name, 160)
        return data

    @staticmethod
    def _result_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n\n".join(part for part in parts if part.strip()).strip()

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    async def list_sessions(self, cwd: str) -> list[SessionSummary]:
        directory = self.config.session_root / _cwd_dir_name(cwd)
        try:
            files = [path for path in directory.iterdir() if path.suffix == ".jsonl"]
        except OSError:
            files = []
        files.sort(key=self._mtime, reverse=True)
        summaries = []
        for path in files[:_SESSION_LIST_LIMIT]:
            summaries.append(await asyncio.to_thread(self._file_summary, path, cwd))
        return summaries

    def count_sessions(self, cwd: str) -> int | None:
        directory = self.config.session_root / _cwd_dir_name(cwd)
        try:
            return sum(1 for path in directory.iterdir() if path.suffix == ".jsonl")
        except OSError:
            return 0

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _file_summary(self, path: Path, cwd: str) -> SessionSummary:
        stem = path.name[: -len(".jsonl")]
        live = self._sessions.get(stem)
        title = "untitled"
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict) or entry.get("type") != "message":
                        continue
                    message = entry.get("message")
                    if isinstance(message, Mapping) and message.get("role") == "user":
                        text = self._condensed_text(message)
                        if text:
                            title = safe_one_line(text, 100)
                            break
        except OSError:
            pass
        if live is not None and live.title != "untitled":
            title = live.title
        return SessionSummary(
            id=stem,
            cwd=cwd,
            title=title,
            updated_at=self._mtime(path),
            busy=bool(live and live.busy),
            tui_attached=bool(live and live.tui),
            active_turn_id=live.turn_id if live else None,
        )

    async def list_running_sessions(self) -> list[SessionSummary]:
        return [self._live_summary(session) for session in self._sessions.values()]

    def _live_summary(self, session: _Session) -> SessionSummary:
        return SessionSummary(
            id=session.id,
            cwd=session.cwd,
            title=session.title,
            updated_at=session.updated_at,
            busy=session.busy,
            tui_attached=session.tui,
            active_turn_id=session.turn_id,
            last_reply=session.last_reply,
        )

    async def create_session(self, cwd: str) -> SessionSummary:
        session = await self._spawn(cwd, resume_path=None)
        return self._live_summary(session)

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        existing = self._sessions.get(session_id)
        if existing is not None:
            return self._live_summary(existing)
        path = self._session_file(session_id, cwd)
        if path is None:
            raise BackendError(f"pi session {session_id} was not found on disk")
        for session in self._sessions.values():
            if session.session_file and Path(session.session_file) == path:
                # The live process renamed or rekeyed itself; never race it
                # with a second writer on the same JSONL.
                return self._live_summary(session)
        header = await asyncio.to_thread(self._file_header, path)
        workspace = cwd or str(header.get("cwd") or "")
        if not workspace:
            raise BackendError(f"pi session {session_id} has no recorded workspace")
        session = await self._spawn(workspace, resume_path=path)
        return self._live_summary(session)

    def _session_file(self, session_id: str, cwd: str | None) -> Path | None:
        if not session_id or "/" in session_id or session_id.startswith("."):
            return None
        if cwd:
            candidate = self.config.session_root / _cwd_dir_name(cwd) / f"{session_id}.jsonl"
            if candidate.is_file():
                return candidate
        matches = list(self.config.session_root.glob(f"*/{session_id}.jsonl"))
        return max(matches, key=self._mtime) if matches else None

    @staticmethod
    def _file_header(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as handle:
                line = handle.readline()
            header = json.loads(line)
        except (OSError, json.JSONDecodeError):
            return {}
        return header if isinstance(header, dict) else {}

    async def _spawn(self, cwd: str, resume_path: Path | None) -> _Session:
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.binary,
                "--mode",
                "rpc",
                cwd=cwd,
                env={**os.environ, "AGENTWIRE_SPAWNED": "1"},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=_MAX_LINE_BYTES,
            )
        except OSError as exc:
            raise BackendError(f"cannot start pi: {exc}") from exc
        assert process.stdout is not None and process.stdin is not None
        transport = _Transport(process.stdout, process.stdin, process)
        # The pump cannot run yet: it would consume command responses that
        # request() awaits are not registered for. Requests here read their
        # responses through the same reader sequentially.
        try:
            if resume_path is not None:
                await self._request_direct(
                    transport, {"type": "switch_session", "sessionPath": str(resume_path)}
                )
            state = await self._request_direct(transport, {"type": "get_state"})
        except BackendError:
            await transport.close()
            raise
        session = self._register({**state, "cwd": cwd}, transport, tui=False)
        if session is None:
            await transport.close()
            raise BackendError("pi session is already attached through a live process")
        return session

    @staticmethod
    async def _request_direct(transport: _Transport, command: dict[str, Any]) -> dict[str, Any]:
        """Issue one command before the pump owns the reader, skipping events."""
        token = f"aw-setup-{uuid.uuid4().hex[:8]}"
        await transport.write({**command, "id": token})
        deadline = asyncio.get_running_loop().time() + _COMMAND_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise BackendError("pi did not answer during session setup")
            try:
                line = await asyncio.wait_for(transport.reader.readline(), remaining)
            except TimeoutError as exc:
                raise BackendError("pi did not answer during session setup") from exc
            if not line:
                raise BackendError("pi exited during session setup")
            try:
                frame = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(frame, dict) or frame.get("id") != token:
                continue
            if not frame.get("success"):
                raise BackendError(
                    safe_one_line(str(frame.get("error") or "pi rejected the command"), 180)
                )
            data = frame.get("data")
            return data if isinstance(data, dict) else {}

    async def session_busy(self, session_id: str) -> bool | None:
        session = self._sessions.get(session_id)
        return session.busy if session is not None else None

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------

    def _require(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise BackendError(f"pi session {session_id} is not attached")
        return session

    async def send_message(self, session_id: str, text: str) -> str | None:
        session = self._require(session_id)
        command: dict[str, Any] = {"type": "prompt", "message": text}
        if session.busy:
            # pi's native RPC rejects a mid-stream prompt without an explicit
            # delivery mode; the extension ignores the extra field and steers
            # on its own.
            command["streamingBehavior"] = "steer"
        session.expected_prompts += 1
        try:
            await session.transport.request(command)
        except BackendError:
            session.expected_prompts = max(0, session.expected_prompts - 1)
            raise
        turn_id, events = self._open_turn(session)
        for event in events:
            await self._events.put(event)
        return turn_id

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        session = self._require(session_id)
        if not session.busy:
            raise BackendError("pi has no active turn to steer")
        session.expected_prompts += 1
        try:
            await session.transport.request({"type": "steer", "message": text})
        except BackendError:
            session.expected_prompts = max(0, session.expected_prompts - 1)
            raise

    async def cancel(self, session_id: str, turn_id: str | None) -> None:
        session = self._require(session_id)
        if not session.busy:
            raise BackendError("pi has no active turn to cancel")
        await session.transport.request({"type": "abort"})

    async def get_last_reply(self, session_id: str) -> str | None:
        session = self._sessions.get(session_id)
        if session is not None and session.last_reply:
            return session.last_reply
        entries = await self._entries(session_id)
        for entry in reversed(entries):
            message = entry.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if message.get("stopReason") in {"toolUse", "error", "aborted"}:
                continue
            text = self._condensed_text(message)
            if text:
                if session is not None:
                    session.last_reply = text
                return text
        return None

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------

    async def setting_options(self) -> Mapping[str, Any]:
        if self._setting_options is not None:
            return self._setting_options
        session = next(iter(self._sessions.values()), None)
        if session is None:
            # Picker discovery is optional; a catalog needs a reachable pi.
            return {}
        data = await session.transport.request({"type": "get_available_models"})
        models = []
        for item in data.get("models") or ():
            if not isinstance(item, Mapping):
                continue
            provider = str(item.get("provider") or "")
            model_id = str(item.get("id") or "")
            if not provider or not model_id:
                continue
            models.append(
                {
                    "value": f"{provider}/{model_id}",
                    "label": str(item.get("name") or model_id),
                    "efforts": (
                        ["off", "minimal", "low", "medium", "high"]
                        if item.get("reasoning")
                        else ["off"]
                    ),
                }
            )
        self._setting_options = {"model": models}
        return self._setting_options

    async def configure_session(self, session_id: str, settings: Mapping[str, Any]) -> None:
        allowed = {"model", "effort", "delivery"}
        unsupported = set(settings) - allowed
        if unsupported:
            raise BackendError(f"unsupported pi settings: {', '.join(sorted(unsupported))}")
        session = self._require(session_id)
        model = settings.get("model")
        if model:
            provider, separator, model_id = str(model).partition("/")
            if not separator:
                raise BackendError("pi model values use provider/model form")
            await session.transport.request(
                {"type": "set_model", "provider": provider, "modelId": model_id}
            )
        if effort := settings.get("effort"):
            await session.transport.request({"type": "set_thinking_level", "level": str(effort)})

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    async def _entries(self, session_id: str) -> list[dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session is not None and not session.transport.closed:
            data = await session.transport.request({"type": "get_entries"})
            entries: list[dict[str, Any]] = []
            for entry in data.get("entries") or ():
                if not isinstance(entry, dict):
                    continue
                # pi's native RPC returns every entry type with raw messages;
                # the extension pre-filters and condenses. Normalize both.
                if entry.get("type") not in (None, "message"):
                    continue
                message = self._normalize_message(entry.get("message"))
                if message is None:
                    continue
                entries.append(
                    {
                        "id": str(entry.get("id") or ""),
                        "timestamp": entry.get("timestamp"),
                        "message": message,
                    }
                )
            return entries
        path = self._session_file(session_id, session.cwd if session else None)
        if path is None:
            raise BackendError(f"pi session {session_id} was not found on disk")
        return await asyncio.to_thread(self._file_entries, path)

    @staticmethod
    def _prune_args(args: Any) -> dict[str, Any]:
        """Keep short scalar argument fields, mirroring the extension's pruning."""
        if not isinstance(args, Mapping):
            return {}
        pruned: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                pruned[key] = truncate_utf8(value, 2048)
            elif isinstance(value, (int, float, bool)):
                pruned[key] = value
        return pruned

    @staticmethod
    def _condensed_text(message: Mapping[str, Any]) -> str:
        text = message.get("text")
        if isinstance(text, str):
            return text.strip()
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "\n\n".join(part for part in parts if part.strip()).strip()

    def _file_entries(self, path: Path) -> list[dict[str, Any]]:
        """Condense the active branch of a session JSONL into wire entries."""
        entries: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            raise BackendError(f"cannot read pi session {path.name}: {exc}") from exc
        parsed: list[dict[str, Any]] = []
        for line in lines[1:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                parsed.append(entry)
        if not parsed:
            return []
        by_id = {str(entry.get("id")): entry for entry in parsed if entry.get("id")}
        branch: list[dict[str, Any]] = []
        cursor: dict[str, Any] | None = parsed[-1]
        while cursor is not None:
            branch.append(cursor)
            parent = cursor.get("parentId")
            cursor = by_id.get(str(parent)) if parent else None
        for entry in reversed(branch):
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            condensed = self._condense_file_message(message)
            if condensed is None:
                continue
            entries.append(
                {
                    "id": str(entry.get("id") or ""),
                    "timestamp": entry.get("timestamp"),
                    "message": condensed,
                }
            )
        return entries

    def _condense_file_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        role = message.get("role")
        timestamp = message.get("timestamp")
        if role == "user":
            text = self._condensed_text(message)
            return {"role": role, "text": text, "timestamp": timestamp} if text else None
        if role == "assistant":
            content = message.get("content")
            items = content if isinstance(content, list) else []
            tool_calls = [
                {
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or "tool"),
                    "arguments": self._prune_args(item.get("arguments")),
                }
                for item in items
                if isinstance(item, Mapping) and item.get("type") == "toolCall"
            ]
            text = self._condensed_text(message)
            if not text and not tool_calls:
                return None
            condensed: dict[str, Any] = {
                "role": role,
                "text": text,
                "toolCalls": tool_calls,
                "stopReason": message.get("stopReason"),
                "timestamp": timestamp,
            }
            if isinstance(message.get("errorMessage"), str):
                condensed["errorMessage"] = message["errorMessage"]
            return condensed
        if role == "toolResult":
            return {
                "role": role,
                "toolCallId": str(message.get("toolCallId") or ""),
                "toolName": str(message.get("toolName") or "tool"),
                "text": self._condensed_text(message),
                "isError": bool(message.get("isError")),
                "timestamp": timestamp,
            }
        return None

    async def list_history(
        self,
        session_id: str,
        cursor: str | None,
        limit: int,
    ) -> HistoryPage:
        if cursor is None:
            offset = 0
        elif cursor.isdigit():
            offset = int(cursor)
        else:
            raise BackendError("pi history cursor must be a non-negative integer string")
        entries = await self._entries(session_id)
        end = max(0, len(entries) - offset)
        start = max(0, end - limit)
        session = self._sessions.get(session_id)
        turn_ids = self._turn_ids(session_id, entries)
        last_of_turn = {turn: index for index, turn in enumerate(turn_ids) if turn}
        times = self._entry_times(entries)
        events: list[BackendEvent] = []
        for index in range(start, end):
            cap = times[index + 1] if index + 1 < len(entries) else times[index] + 1000
            for event in self._history_entry(
                session_id,
                entries[index],
                turn_ids[index],
                times[index],
                starts_turn=index == 0 or turn_ids[index] != turn_ids[index - 1],
                ends_turn=(
                    last_of_turn.get(turn_ids[index]) == index
                    and (index < len(entries) - 1 or not (session is not None and session.busy))
                ),
            ):
                event.at = min(event.at or times[index], cap - 1)
                events.append(event)
        return HistoryPage(
            events=tuple(events),
            next_cursor=str(offset + (end - start)) if start > 0 else None,
        )

    def _turn_ids(self, session_id: str, entries: Sequence[Mapping[str, Any]]) -> list[str | None]:
        turn_ids: list[str | None] = []
        current: str | None = None
        for entry in entries:
            message = entry.get("message")
            if isinstance(message, Mapping) and message.get("role") == "user":
                current = self._identifier(f"{session_id}\0{entry.get('id')}", "turn")
            turn_ids.append(current)
        return turn_ids

    @staticmethod
    def _entry_times(entries: Sequence[Mapping[str, Any]]) -> list[int]:
        anchor = int(time.time() * 1000) - max(0, len(entries) - 1) * 1000
        times: list[int] = []
        previous: int | None = None
        for index, entry in enumerate(entries):
            value = _entry_millis(entry) or anchor + index * 1000
            if previous is not None and value <= previous:
                value = previous + 1
            times.append(value)
            previous = value
        return times

    def _history_entry(
        self,
        session_id: str,
        entry: Mapping[str, Any],
        turn_id: str | None,
        base: int,
        starts_turn: bool,
        ends_turn: bool,
    ) -> list[BackendEvent]:
        message = entry.get("message")
        payload = message if isinstance(message, Mapping) else {}
        entry_id = str(entry.get("id") or "") or None
        events: list[BackendEvent] = []
        role = payload.get("role")
        if role == "user":
            if starts_turn and turn_id is not None:
                events.append(
                    self._history_event(
                        "turn_started", session_id, turn_id, None, base, "turn.started"
                    )
                )
            events.append(
                self._history_event(
                    "user_prompt",
                    session_id,
                    turn_id,
                    entry_id,
                    base + 1,
                    "user.prompt",
                    text=self._condensed_text(payload),
                )
            )
        elif role == "assistant":
            tool_calls = [
                item for item in payload.get("toolCalls") or () if isinstance(item, Mapping)
            ]
            text = self._condensed_text(payload)
            if text:
                events.append(
                    self._history_event(
                        "progress" if tool_calls else "assistant",
                        session_id,
                        turn_id,
                        entry_id,
                        base,
                        "plan.updated" if tool_calls else "assistant.completed",
                        text=text,
                        data={"plan": False} if tool_calls else {},
                    )
                )
            for offset, item in enumerate(tool_calls, 1):
                name = str(item.get("name") or "tool")
                events.append(
                    self._history_event(
                        "tool_started",
                        session_id,
                        turn_id,
                        str(item.get("id") or ""),
                        base + offset,
                        "tool.started",
                        tool_kind=self._tool_kind(name),
                        data=self._tool_metadata(name, item.get("arguments")),
                    )
                )
        elif role == "toolResult":
            failed = bool(payload.get("isError"))
            data: dict[str, Any] = {"status": "error" if failed else "completed"}
            output = self._condensed_text(payload)
            if output:
                data["output"] = truncate_utf8(clean_block(output), _MAX_TOOL_PAYLOAD_BYTES)
            events.append(
                self._history_event(
                    "tool_finished",
                    session_id,
                    turn_id,
                    str(payload.get("toolCallId") or "") or None,
                    base,
                    "tool.completed",
                    success=not failed,
                    data=data,
                )
            )
        if ends_turn and turn_id is not None:
            events.append(
                self._history_event(
                    "turn_done", session_id, turn_id, None, base + 999, "turn.completed"
                )
            )
        return events

    def _history_event(
        self,
        kind: str,
        session_id: str,
        turn_id: str | None,
        item_id: str | None,
        at: int,
        seed: str,
        **values: Any,
    ) -> BackendEvent:
        return BackendEvent(
            kind=kind,  # type: ignore[arg-type]
            backend=self.name,
            session_id=session_id,
            turn_id=turn_id,
            item_id=item_id,
            at=at,
            event_id=self._identifier(f"{session_id}\0{turn_id or ''}\0{item_id or ''}", seed),
            **values,
        )

    @staticmethod
    def _identifier(seed: str, kind: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentwire-pi:{kind}:{seed}"))
