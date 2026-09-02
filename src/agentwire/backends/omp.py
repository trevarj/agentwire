from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentwire.backends.base import BackendError
from agentwire.backends.pi import (
    _MAX_TOOL_PAYLOAD_BYTES,
    PiBackend,
    _ResponseError,
    _Session,
    _Transport,
)
from agentwire.config import OmpConfig
from agentwire.models import BackendEvent
from agentwire.text import clean_block, safe_one_line, truncate_utf8

_MAX_FRAME_BYTES = 1024 * 1024
_MAX_REASSEMBLED_BYTES = 64 * 1024 * 1024
_HISTORY_PAGE = 256
_CHUNK_BYTES = 256 * 1024


class _OmpFrameDecoder:
    def __init__(self) -> None:
        self.max_frame_bytes = _MAX_FRAME_BYTES
        self.max_reassembled_bytes = _MAX_REASSEMBLED_BYTES
        self.protocol2 = False
        self._pending: tuple[str, int, int, int, list[bytes], int] | None = None

    def __call__(self, line: bytes) -> dict[str, Any] | None:
        if len(line) > self.max_frame_bytes:
            raise BackendError("omp RPC frame exceeds the transport limit")
        try:
            frame = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("omp RPC frame is not valid JSON") from exc
        if not isinstance(frame, dict):
            raise BackendError("omp RPC frame must be an object")
        if frame.get("type") != "rpc_chunk":
            if self._pending is not None:
                raise BackendError("omp RPC chunk sequence was interrupted")
            return frame
        if not self.protocol2:
            raise BackendError("omp sent an RPC chunk before protocol negotiation")
        return self._chunk(frame)

    def _chunk(self, frame: Mapping[str, Any]) -> dict[str, Any] | None:
        chunk_id = frame.get("chunkId")
        index = frame.get("index")
        count = frame.get("count")
        byte_length = frame.get("byteLength")
        integers = (index, count, byte_length)
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or len(chunk_id) > 128
            or any(not isinstance(value, int) or isinstance(value, bool) for value in integers)
            or index < 0
            or count < 2
            or count > math.ceil(self.max_reassembled_bytes / _CHUNK_BYTES)
            or index >= count
            or byte_length < self.max_frame_bytes
            or byte_length > self.max_reassembled_bytes
        ):
            raise BackendError("invalid omp RPC chunk metadata")
        data = frame.get("data")
        if not isinstance(data, str) or not data:
            raise BackendError("invalid omp RPC chunk data")
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BackendError("invalid omp RPC chunk data") from exc
        if base64.b64encode(decoded).decode("ascii") != data:
            raise BackendError("invalid omp RPC chunk data")
        if len(decoded) > _CHUNK_BYTES:
            raise BackendError("omp RPC chunk payload exceeds the transport limit")

        if self._pending is None:
            if index != 0:
                raise BackendError("omp RPC chunk sequence must start at index 0")
            self._pending = (chunk_id, count, byte_length, 0, [], 0)
        pending_id, pending_count, pending_length, next_index, chunks, received = self._pending
        if (
            pending_id != chunk_id
            or pending_count != count
            or pending_length != byte_length
            or next_index != index
        ):
            raise BackendError("omp RPC chunk sequence mismatch")
        chunks.append(decoded)
        received += len(decoded)
        if received > byte_length or received > self.max_reassembled_bytes:
            raise BackendError("omp RPC chunk sequence exceeds its declared length")
        next_index += 1
        self._pending = (chunk_id, count, byte_length, next_index, chunks, received)
        if next_index < count:
            return None
        self._pending = None
        if received != byte_length:
            raise BackendError("omp RPC chunk sequence length mismatch")
        try:
            value = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("reassembled omp RPC frame is invalid") from exc
        if not isinstance(value, dict):
            raise BackendError("omp RPC frame must be an object")
        return value


class OmpBackend(PiBackend):
    name = "omp"

    def __init__(self, config: OmpConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.config = config

    def _make_transport(self, reader: Any, writer: Any, process: Any = None) -> _Transport:
        return _Transport(
            reader,
            writer,
            process,
            backend=self.name,
            decoder=_OmpFrameDecoder(),
        )

    def _state_busy(self, state: Mapping[str, Any]) -> bool:
        return bool(state.get("isStreaming") or state.get("isCompacting"))

    def _is_settled_frame(self, frame: Mapping[str, Any]) -> bool:
        kind = frame.get("type")
        return kind == "agent_settled" or (
            kind == "agent_end" and frame.get("isTerminal") is not False
        )

    def _is_approval_select(self, frame: Mapping[str, Any]) -> bool:
        return frame.get("options") == ["Approve", "Deny"] and str(
            frame.get("title") or ""
        ).lstrip().startswith("Allow tool:")

    def _custom_option(self, options: tuple[str, ...]) -> str | None:
        sentinel = "Other (type your own)"
        return sentinel if options and options[-1] == sentinel else None

    def _approval_response(self, frame: Mapping[str, Any], allow: bool) -> dict[str, Any]:
        return {
            "type": "extension_ui_response",
            "id": frame.get("id"),
            "value": "Approve" if allow else "Deny",
        }

    async def setting_options(self) -> Mapping[str, Any]:
        if self._setting_options is not None:
            return self._setting_options
        session = next(iter(self._sessions.values()), None)
        if session is None:
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
            thinking = item.get("thinking")
            raw_efforts = thinking.get("efforts") if isinstance(thinking, Mapping) else None
            if isinstance(raw_efforts, list) and all(
                isinstance(effort, str) and effort for effort in raw_efforts
            ):
                efforts = list(dict.fromkeys(raw_efforts))
                if "off" not in efforts:
                    efforts.insert(0, "off")
            else:
                efforts = (
                    ["off", "minimal", "low", "medium", "high"]
                    if item.get("reasoning")
                    else ["off"]
                )
            models.append(
                {
                    "value": f"{provider}/{model_id}",
                    "label": str(item.get("name") or model_id),
                    "efforts": efforts,
                }
            )
        self._setting_options = {"model": models}
        return self._setting_options

    async def send_message(self, session_id: str, text: str) -> str | None:
        session = self._require(session_id)
        command: dict[str, Any] = {"type": "prompt", "message": text}
        if session.busy:
            command["streamingBehavior"] = "steer"
        session.expected_prompts += 1
        try:
            data = await session.transport.request(command)
        except BackendError:
            session.expected_prompts = max(0, session.expected_prompts - 1)
            raise
        turn_id, events = self._open_turn(session)
        for event in events:
            await self._events.put(event)
        if data.get("agentInvoked") is False:
            self._schedule_settlement(session)
        return turn_id

    async def _handle_frame(self, session: _Session, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "extension_ui_request" and frame.get("method") == "cancel":
            target = str(frame.get("targetId") or "")
            if not target or self._ui_requests.pop(target, None) is None:
                return
            await self._events.put(
                BackendEvent(
                    kind="request_resolved",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=session.turn_id,
                    request_token=target,
                )
            )
            await self._events.put(self._status_event(session))
            return
        if kind == "command_output":
            text = frame.get("text")
            if isinstance(text, str) and text:
                turn_id, events = self._open_turn(session)
                for event in events:
                    await self._events.put(event)
                cleaned = truncate_utf8(clean_block(text).strip(), _MAX_TOOL_PAYLOAD_BYTES)
                session.last_reply = cleaned
                await self._events.put(
                    BackendEvent(
                        kind="assistant",
                        backend=self.name,
                        session_id=session.id,
                        turn_id=turn_id,
                        text=cleaned,
                    )
                )
            return
        if kind == "prompt_result" and frame.get("agentInvoked") is False:
            self._schedule_settlement(session)
            return
        if kind == "response" and not frame.get("success"):
            command = session.transport.take_completed_command(frame.get("id"))
            if command == "prompt":
                asyncio.create_task(
                    self._fail_turn(
                        session,
                        safe_one_line(str(frame.get("error") or "omp prompt failed"), 180),
                    ),
                    name=f"{self.name}-{session.id}-fail",
                )
            return
        if kind == "session_info_update":
            await self._session_info_update(session, frame)
            return
        await super()._handle_frame(session, frame)

    def _schedule_settlement(self, session: _Session) -> None:
        asyncio.create_task(self._settle_turn(session), name=f"{self.name}-{session.id}-settle")

    async def _fail_turn(self, session: _Session, reason: str) -> None:
        turn_id = session.turn_id
        session.turn_id = None
        session.busy = False
        session.waiting = False
        session.tools.clear()
        if turn_id is not None:
            await self._events.put(
                BackendEvent(
                    kind="turn_failed",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=turn_id,
                    text=reason,
                )
            )
        await self._events.put(self._status_event(session))

    async def _session_info_update(self, session: _Session, frame: Mapping[str, Any]) -> None:
        nested = frame.get("data")
        state = nested if isinstance(nested, Mapping) else frame
        update: dict[str, Any] = {}
        name = state.get("sessionName") or state.get("name") or state.get("title")
        if isinstance(name, str) and name:
            update["sessionName"] = name
        session_file = state.get("sessionFile")
        if isinstance(session_file, str) and session_file:
            update["sessionFile"] = session_file
        if update:
            await self._session_changed(session, update)

    async def _request_direct(
        self, transport: _Transport, command: dict[str, Any]
    ) -> dict[str, Any]:
        data = await super()._request_direct(transport, command)
        if command.get("type") == "switch_session" and data.get("cancelled") is True:
            raise BackendError("omp session switch was cancelled")
        return data

    async def _transport_entries(self, session: _Session) -> list[dict[str, Any]]:
        if session.tui:
            return await super()._transport_entries(session)
        messages: list[Any] = []
        cursor: str | None = None
        seen: set[str] = set()
        total: int | None = None
        try:
            while True:
                command: dict[str, Any] = {
                    "type": "get_messages_page",
                    "limit": _HISTORY_PAGE,
                }
                if cursor is not None:
                    command["cursor"] = cursor
                data = await session.transport.request(command)
                page = data.get("messages")
                page_total = data.get("totalMessages")
                if (
                    not isinstance(page, list)
                    or not isinstance(page_total, int)
                    or isinstance(page_total, bool)
                    or page_total < 0
                    or (total is not None and page_total != total)
                ):
                    raise BackendError("omp returned an invalid history page")
                total = page_total
                messages.extend(page)
                next_cursor = data.get("nextCursor")
                if next_cursor is None:
                    if len(messages) != total:
                        raise BackendError("omp returned an incomplete history snapshot")
                    break
                if (
                    not isinstance(next_cursor, str)
                    or not next_cursor
                    or next_cursor in seen
                    or len(messages) >= total
                ):
                    raise BackendError("omp returned an invalid history cursor")
                seen.add(next_cursor)
                cursor = next_cursor
        except _ResponseError as exc:
            if exc.code not in {"session_busy", "stale_cursor"}:
                raise
            data = await session.transport.request({"type": "get_messages"})
            snapshot = data.get("messages")
            if not isinstance(snapshot, list):
                raise BackendError("omp returned invalid session messages") from exc
            messages = snapshot
        return self._message_entries(messages)

    def _message_entries(self, messages: list[Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for index, raw in enumerate(messages):
            message = self._normalize_message(raw)
            if message is None:
                continue
            entries.append(
                {
                    "id": str(index),
                    "timestamp": raw.get("timestamp") if isinstance(raw, Mapping) else None,
                    "message": message,
                }
            )
        return entries

    async def _initialize_transport(self, transport: _Transport) -> None:
        ready = await transport.read_frame(10)
        if ready is None or ready.get("type") != "ready":
            raise BackendError("omp did not send the required ready frame")
        decoder = transport.decoder
        assert isinstance(decoder, _OmpFrameDecoder)
        decoder.max_frame_bytes = self._advertised_limit(
            ready.get("maxFrameBytes"), _MAX_FRAME_BYTES
        )
        decoder.max_reassembled_bytes = self._advertised_limit(
            ready.get("maxReassembledFrameBytes"), _MAX_REASSEMBLED_BYTES
        )
        versions = ready.get("supportedProtocolVersions")
        if isinstance(versions, list) and 2 in versions:
            negotiated = await self._request_direct(
                transport, {"type": "negotiate_protocol", "protocolVersion": 2}
            )
            if negotiated.get("protocolVersion") != 2:
                raise BackendError("omp did not confirm RPC protocol 2")
            decoder.protocol2 = True

    @staticmethod
    def _advertised_limit(value: Any, cap: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return min(value, cap)
        return cap

    def _spawn_args(self, cwd: str) -> list[str]:
        return [
            "--mode",
            "rpc",
            "--approval-mode",
            "always-ask",
            "--session-dir",
            str(self._session_directory(cwd)),
        ]

    def _session_directory(self, cwd: str) -> Path:
        path = Path(cwd).resolve()
        home = Path.home().resolve()
        temp = Path(tempfile.gettempdir()).resolve()
        if path == home:
            bucket = "-"
        elif path.is_relative_to(home):
            bucket = f"-{path.relative_to(home)}"
        elif path == temp:
            bucket = "-tmp"
        elif path.is_relative_to(temp):
            bucket = f"-tmp-{path.relative_to(temp)}"
        else:
            bucket = "--" + str(path).lstrip("/\\") + "--"
        return self.config.session_root / (
            bucket.replace("/", "-").replace("\\", "-").replace(":", "-")
        )
