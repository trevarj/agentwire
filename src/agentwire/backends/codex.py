from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import re
import secrets
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiohttp

from agentwire.backends.base import Backend, BackendError
from agentwire.config import CodexConfig
from agentwire.models import (
    BackendEvent,
    HistoryPage,
    Question,
    SessionActivity,
    SessionOutput,
    SessionSummary,
)
from agentwire.text import safe_one_line, truncate_utf8

_MAX_WEBSOCKET_MESSAGE_BYTES = 8 * 1024 * 1024
_OVERLOAD_ERROR_CODE = -32001
_OVERLOAD_RETRIES = 3


class _CodexRPCError(BackendError):
    """Preserve JSON-RPC error metadata needed for safe retry decisions."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


def _process_start_time(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    """Return Linux's stable process start tick without trusting a reused PID."""
    try:
        fields = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1]
        return fields.split()[19]
    except (IndexError, OSError):
        return None


class CodexTuiSessionPresence:
    """Publish one Agentwire-launched TUI's exact current Codex thread."""

    def __init__(
        self,
        socket_path: Path,
        pid: int | None = None,
        proc_root: Path = Path("/proc"),
    ) -> None:
        self.pid = pid if pid is not None else os.getpid()
        self.proc_root = proc_root
        self.directory = socket_path.parent / "tui-presence"
        self.path = self.directory / f"{self.pid}-{secrets.token_hex(8)}.json"
        self.start_time = _process_start_time(self.pid, proc_root)

    def update(self, session_id: str | None) -> None:
        if not session_id:
            self.clear()
            return
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        payload = {
            "pid": self.pid,
            "startTime": self.start_time,
            "sessionId": session_id,
        }
        descriptor, temp_name = tempfile.mkstemp(prefix=".presence-", dir=self.directory)
        temp = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.write("\n")
            os.chmod(temp, 0o600)
            os.replace(temp, self.path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    @staticmethod
    def sessions(
        socket_path: Path,
        proc_root: Path = Path("/proc"),
    ) -> set[str]:
        directory = socket_path.parent / "tui-presence"
        sessions: set[str] = set()
        try:
            records = tuple(directory.glob("*.json"))
        except OSError:
            return sessions
        for record in records:
            valid = False
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
                pid = int(payload["pid"])
                session_id = str(payload["sessionId"])
                current_start = _process_start_time(pid, proc_root)
                recorded_start = payload.get("startTime")
                valid = bool(
                    session_id
                    and current_start is not None
                    and (recorded_start is None or current_start == str(recorded_start))
                )
                if valid:
                    sessions.add(session_id)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            if not valid:
                with contextlib.suppress(OSError):
                    record.unlink()
        return sessions


class CodexBackend(Backend):
    name = "codex"
    has_authoritative_history = True

    def __init__(self, config: CodexConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._transport_closing = False
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._server_requests: dict[str | int, tuple[str, dict[str, Any]]] = {}
        self._active_turns: dict[str, str] = {}
        self._turn_messages: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._last_replies: dict[str, str] = {}
        self._last_plan_updates: dict[tuple[str, str], tuple[str, str, int, int]] = {}
        self._session_settings: dict[str, dict[str, Any]] = {}
        self._setting_options: dict[str, Any] | None = None

    async def start(self) -> None:
        async with self._connect_lock:
            if self._reader_task is not None and not self._reader_task.done():
                return
            self._closed = False
            await self._close_transport()
            deadline = asyncio.get_running_loop().time() + 30
            last_error: BaseException | None = None
            while asyncio.get_running_loop().time() < deadline:
                try:
                    connector = aiohttp.UnixConnector(path=str(self.config.socket_path))
                    timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
                    self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
                    self._ws = await self._session.ws_connect(
                        "http://localhost/",
                        timeout=aiohttp.ClientWSTimeout(ws_close=5),
                        heartbeat=20,
                        compress=0,
                        max_msg_size=_MAX_WEBSOCKET_MESSAGE_BYTES,
                    )
                    self._reader_task = asyncio.create_task(self._reader(), name="codex-reader")
                    await self._request(
                        "initialize",
                        {
                            "clientInfo": {
                                "name": "agentwire",
                                "title": "IRC agent bridge",
                                "version": "0.1.0",
                            },
                            "capabilities": {
                                "experimentalApi": True,
                                "requestAttestation": False,
                                "optOutNotificationMethods": [
                                    "item/agentMessage/delta",
                                    "item/reasoning/summaryTextDelta",
                                    "item/reasoning/textDelta",
                                    "item/commandExecution/outputDelta",
                                    "item/fileChange/outputDelta",
                                ],
                            },
                        },
                    )
                    await self._send({"method": "initialized"})
                    self._ready.set()
                    await self._events.put(BackendEvent(kind="connected", backend=self.name))
                    return
                except (OSError, aiohttp.ClientError, TimeoutError, BackendError) as exc:
                    last_error = exc
                    await self._close_transport()
                    await asyncio.sleep(0.25)
            raise BackendError(f"Codex app-server did not become ready: {last_error}")

    async def wait_ready(self, timeout: float = 30) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except TimeoutError as exc:
            raise BackendError("timed out waiting for Codex app-server") from exc

    async def close(self) -> None:
        self._closed = True
        self._ready.clear()
        await self._close_transport()

    async def _close_transport(self) -> None:
        current = asyncio.current_task()
        self._setting_options = None
        self._transport_closing = True
        try:
            if self._reader_task is not None and self._reader_task is not current:
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader_task
            self._reader_task = None
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.close()
            self._ws = None
            if self._session is not None:
                with contextlib.suppress(Exception):
                    await self._session.close()
            self._session = None
            self._fail_pending("Codex connection closed")
        finally:
            self._transport_closing = False

    def _fail_pending(self, message: str) -> None:
        """Wake every caller when the transport fails instead of waiting for RPC timeouts."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(BackendError(message))
        self._pending.clear()

    async def _reader(self) -> None:
        assert self._ws is not None
        try:
            async for frame in self._ws:
                if frame.type == aiohttp.WSMsgType.TEXT:
                    try:
                        message = json.loads(frame.data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(message, dict):
                        await self._handle_message(message)
                elif frame.type in {
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            if not self._closed and not self._transport_closing:
                self._ready.clear()
                self._fail_pending("Codex app-server disconnected")
                await self._events.put(
                    BackendEvent(
                        kind="disconnected",
                        backend=self.name,
                        text="Codex app-server disconnected",
                    )
                )

    async def _handle_message(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if method is None and isinstance(request_id, int):
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message.get("error") or {}
                if isinstance(error, dict):
                    raw_code = error.get("code")
                    code = raw_code if isinstance(raw_code, int) else None
                    detail = str(error.get("message", "Codex request failed"))
                else:
                    code = None
                    detail = "Codex request failed"
                future.set_exception(_CodexRPCError(detail, code))
            else:
                future.set_result(message.get("result"))
            return
        if isinstance(method, str) and request_id is not None:
            params = message.get("params")
            await self._handle_server_request(
                request_id, method, params if isinstance(params, dict) else {}
            )
            return
        if isinstance(method, str):
            params = message.get("params")
            await self._handle_notification(method, params if isinstance(params, dict) else {})

    async def _handle_server_request(
        self, request_id: str | int, method: str, params: dict[str, Any]
    ) -> None:
        self._server_requests[request_id] = (method, params)
        thread_id = str(params.get("threadId") or params.get("conversationId") or "") or None
        turn_id = str(params.get("turnId") or "") or None
        item_id = str(params.get("itemId") or params.get("callId") or "") or None
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
            "execCommandApproval",
            "applyPatchApproval",
        }:
            kind = {
                "item/commandExecution/requestApproval": "shell command",
                "item/fileChange/requestApproval": "file change",
                "item/permissions/requestApproval": "additional permissions",
                "execCommandApproval": "shell command",
                "applyPatchApproval": "file change",
            }[method]
            text = f"{kind} approval needed"
            await self._events.put(
                BackendEvent(
                    kind="approval",
                    backend=self.name,
                    session_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                    request_token=request_id,
                    text=text,
                    data={"method": method},
                )
            )
            return
        if method == "item/tool/requestUserInput":
            questions: list[Question] = []
            for index, raw in enumerate(params.get("questions") or [], 1):
                if not isinstance(raw, dict):
                    continue
                options = tuple(
                    str(option.get("label", ""))
                    for option in raw.get("options") or []
                    if isinstance(option, dict) and option.get("label")
                )
                questions.append(
                    Question(
                        id=str(raw.get("id") or index),
                        header=str(raw.get("header") or f"Question {index}"),
                        prompt=str(raw.get("question") or "Input requested"),
                        options=options,
                        multiple=False,
                        custom=bool(raw.get("isOther", True)),
                        secret=bool(raw.get("isSecret", False)),
                    )
                )
            await self._events.put(
                BackendEvent(
                    kind="question",
                    backend=self.name,
                    session_id=thread_id,
                    turn_id=turn_id,
                    item_id=item_id,
                    request_token=request_id,
                    questions=tuple(questions),
                )
            )
            return
        if method == "mcpServer/elicitation/request":
            await self._events.put(
                BackendEvent(
                    kind="question",
                    backend=self.name,
                    session_id=thread_id,
                    turn_id=turn_id,
                    request_token=request_id,
                    questions=(
                        Question(
                            id="mcp",
                            header="MCP input",
                            prompt="MCP elicitation requires the attached TUI",
                            secret=True,
                        ),
                    ),
                )
            )

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        thread_id = str(params.get("threadId") or "") or None
        if method == "thread/status/changed" and thread_id:
            status = params.get("status") or {}
            status_type = str(status.get("type") if isinstance(status, dict) else status)
            active_flags = (
                tuple(str(flag) for flag in status.get("activeFlags") or ())
                if isinstance(status, dict)
                else ()
            )
            await self._events.put(
                BackendEvent(
                    kind="status_changed",
                    backend=self.name,
                    session_id=thread_id,
                    data={"busy": status_type == "active", "active_flags": active_flags},
                )
            )
            return
        if method == "turn/plan/updated" and thread_id:
            plan = [item for item in params.get("plan") or [] if isinstance(item, dict)]
            active_step = next(
                (
                    str(item.get("step") or "").strip()
                    for item in plan
                    if item.get("status") == "inProgress"
                ),
                "",
            )
            completed_steps = sum(item.get("status") == "completed" for item in plan)
            complete = bool(plan) and completed_steps == len(plan)
            status = "completed" if complete else "inProgress" if active_step else "pending"
            explanation = str(params.get("explanation") or "").strip()
            text = explanation or (
                "Plan completed"
                if complete
                else f"plan: {active_step}"
                if active_step
                else "Plan updated"
            )
            turn_id = str(params.get("turnId") or "")
            key = (thread_id, turn_id)
            signature = (text, status, completed_steps, len(plan))
            if self._last_plan_updates.get(key) != signature:
                self._last_plan_updates[key] = signature
                await self._events.put(
                    BackendEvent(
                        kind="progress",
                        backend=self.name,
                        session_id=thread_id,
                        turn_id=turn_id or None,
                        text=text,
                        data={
                            "plan": True,
                            "running": bool(plan) and not complete,
                            "status": status,
                            "completedSteps": completed_steps,
                            "totalSteps": len(plan),
                        },
                    )
                )
            return
        if method == "serverRequest/resolved":
            token = params.get("requestId")
            stored = self._server_requests.pop(token, None)
            if thread_id is None and stored is not None:
                stored_params = stored[1]
                thread_id = (
                    str(stored_params.get("threadId") or stored_params.get("conversationId") or "")
                    or None
                )
            await self._events.put(
                BackendEvent(
                    kind="request_resolved",
                    backend=self.name,
                    session_id=thread_id,
                    request_token=token,
                )
            )
            return
        if method == "turn/started" and thread_id:
            turn = params.get("turn") or {}
            turn_id = str(turn.get("id") or "") or None
            if turn_id:
                self._active_turns[thread_id] = turn_id
            await self._events.put(
                BackendEvent(
                    kind="turn_started",
                    backend=self.name,
                    session_id=thread_id,
                    turn_id=turn_id,
                )
            )
            return
        if method in {"item/started", "item/completed"} and thread_id:
            item = params.get("item") or {}
            if not isinstance(item, dict):
                return
            turn_id = str(params.get("turnId") or "") or None
            item_type = item.get("type")
            if item_type == "agentMessage" and method == "item/completed" and turn_id:
                self._turn_messages.setdefault((thread_id, turn_id), []).append(item)
                if item.get("phase") == "commentary":
                    text = str(item.get("text") or "").strip()
                    if text:
                        await self._events.put(
                            BackendEvent(
                                kind="progress",
                                backend=self.name,
                                session_id=thread_id,
                                turn_id=turn_id,
                                item_id=str(item.get("id") or "") or None,
                                text=text,
                            )
                        )
                return
            tool_kind = self._tool_kind(str(item_type))
            if tool_kind:
                success = self._tool_success(item) if method == "item/completed" else None
                await self._events.put(
                    BackendEvent(
                        kind="tool_started" if method == "item/started" else "tool_finished",
                        backend=self.name,
                        session_id=thread_id,
                        turn_id=turn_id,
                        item_id=str(item.get("id") or "") or None,
                        tool_kind=tool_kind,
                        success=success,
                        data=self._tool_metadata(item),
                    )
                )
            return
        if method == "turn/completed" and thread_id:
            turn = params.get("turn") or {}
            turn_id = str(turn.get("id") or self._active_turns.get(thread_id) or "") or None
            messages = list(self._turn_messages.pop((thread_id, turn_id or ""), []))
            if not messages:
                messages = [
                    item
                    for item in turn.get("items") or []
                    if isinstance(item, dict) and item.get("type") == "agentMessage"
                ]
            final = self._select_final_message(messages)
            if final:
                self._last_replies[thread_id] = final
                await self._events.put(
                    BackendEvent(
                        kind="assistant",
                        backend=self.name,
                        session_id=thread_id,
                        turn_id=turn_id,
                        text=final,
                    )
                )
            status = str(turn.get("status") or "completed")
            kind = "turn_done" if status in {"completed", "interrupted"} else "turn_failed"
            error = turn.get("error") or {}
            text = (
                safe_one_line(str(error.get("message") or status)) if kind == "turn_failed" else ""
            )
            self._active_turns.pop(thread_id, None)
            if turn_id:
                self._last_plan_updates.pop((thread_id, turn_id), None)
            await self._events.put(
                BackendEvent(
                    kind=kind,
                    backend=self.name,
                    session_id=thread_id,
                    turn_id=turn_id,
                    text=text,
                )
            )
            return
        if method == "error" and thread_id:
            error = params.get("error") or {}
            if not params.get("willRetry", False):
                await self._events.put(
                    BackendEvent(
                        kind="turn_failed",
                        backend=self.name,
                        session_id=thread_id,
                        turn_id=str(params.get("turnId") or "") or None,
                        text=safe_one_line(str(error.get("message") or "Codex turn failed")),
                    )
                )

    @staticmethod
    def _tool_kind(item_type: str) -> str | None:
        return {
            "commandExecution": "shell",
            "fileChange": "file edit",
            "mcpToolCall": "MCP tool",
            "dynamicToolCall": "tool",
            "collabAgentToolCall": "agent",
            "subAgentActivity": "agent",
            "webSearch": "web search",
            "imageView": "image view",
            "imageGeneration": "image generation",
        }.get(item_type)

    @staticmethod
    def _tool_success(item: dict[str, Any]) -> bool | None:
        item_type = item.get("type")
        status = str(item.get("status") or "").lower()
        if item_type == "commandExecution":
            code = item.get("exitCode")
            return code == 0 if isinstance(code, int) else status in {"completed", "success"}
        if item_type == "dynamicToolCall" and isinstance(item.get("success"), bool):
            return bool(item["success"])
        if status:
            return status not in {"failed", "error", "declined"}
        return None

    @staticmethod
    def _tool_metadata(item: dict[str, Any]) -> dict[str, Any]:
        item_type = str(item.get("type") or "")
        data: dict[str, Any] = {}
        status = item.get("status")
        if isinstance(status, str) and status:
            data["status"] = status
        for key in ("exitCode", "durationMs"):
            value = item.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                data[key] = value
        if item_type == "commandExecution":
            command = str(item.get("command") or "").strip()
            if command:
                data["label"] = f"$ {safe_one_line(command, 160)}"
                data["input"] = truncate_utf8(command, 32 * 1024)
            output = str(item.get("aggregatedOutput") or "").strip("\r\n")
            if output:
                data["output"] = truncate_utf8(output, 32 * 1024)
        elif item_type == "fileChange":
            changes = [change for change in item.get("changes") or [] if isinstance(change, dict)]
            paths = [str(change.get("path") or "").strip() for change in changes]
            paths = [path for path in paths if path]
            data["label"] = safe_one_line(
                "Edit " + ", ".join(paths[:3]) if paths else "File changes",
                160,
            )
            sections = [CodexBackend._git_diff(change) for change in changes]
            diff = "\n\n".join(section for section in sections if section)
            if diff:
                data["diff"] = truncate_utf8(diff, 32 * 1024)
        elif item_type == "mcpToolCall":
            server = str(item.get("server") or "").strip()
            tool = str(item.get("tool") or "").strip()
            if server or tool:
                data["label"] = safe_one_line(" / ".join(part for part in (server, tool) if part))
        elif item_type == "dynamicToolCall":
            namespace = str(item.get("namespace") or "").strip()
            tool = str(item.get("tool") or "").strip()
            if namespace or tool:
                data["label"] = safe_one_line(
                    " / ".join(part for part in (namespace, tool) if part)
                )
        elif item_type == "webSearch":
            query = str(item.get("query") or "").strip()
            if query:
                data["label"] = safe_one_line(f"Search: {query}", 160)
        return data

    @staticmethod
    def _git_diff(change: dict[str, Any]) -> str:
        path = str(change.get("path") or "").strip()
        raw_diff = str(change.get("diff") or "").replace("\r\n", "\n").replace("\r", "\n")
        has_final_newline = raw_diff.endswith("\n")
        diff = raw_diff.strip("\n")
        if not path:
            return diff
        if diff.startswith("diff --git "):
            return diff
        display_path = path.lstrip("/") or path
        kind_value = change.get("kind")
        kind = (
            str(kind_value.get("type") or "update")
            if isinstance(kind_value, Mapping)
            else str(kind_value or "update")
        ).lower()
        added = kind in {"add", "added", "create", "created"}
        deleted = kind in {"delete", "deleted", "remove", "removed"}
        old_path = "/dev/null" if added else f"a/{display_path}"
        new_path = "/dev/null" if deleted else f"b/{display_path}"
        header = f"diff --git a/{display_path} b/{display_path}"
        if diff.startswith("--- "):
            return f"{header}\n{diff}"
        file_headers = f"{header}\n--- {old_path}\n+++ {new_path}"
        if added:
            return CodexBackend._content_diff(file_headers, diff, "+", has_final_newline)
        if deleted:
            return CodexBackend._content_diff(file_headers, diff, "-", has_final_newline)
        return f"{file_headers}\n{diff}".rstrip()

    @staticmethod
    def _content_diff(
        file_headers: str,
        content: str,
        prefix: str,
        has_final_newline: bool,
    ) -> str:
        if not content:
            return file_headers
        lines = content.split("\n")
        count = len(lines)
        hunk = f"@@ -0,0 +1,{count} @@" if prefix == "+" else f"@@ -1,{count} +0,0 @@"
        body = "\n".join(f"{prefix}{line}" for line in lines)
        if not has_final_newline:
            body += "\n\\ No newline at end of file"
        return f"{file_headers}\n{hunk}\n{body}"

    @staticmethod
    def _select_final_message(messages: list[dict[str, Any]]) -> str:
        finals = [
            str(item.get("text") or "") for item in messages if item.get("phase") == "final_answer"
        ]
        if not finals:
            finals = [
                str(item.get("text") or "") for item in messages if item.get("phase") in {None, ""}
            ]
        return "\n\n".join(text for text in finals if text.strip()).strip()

    async def _send(self, message: dict[str, Any]) -> None:
        if self._ws is None or self._ws.closed:
            raise BackendError("Codex app-server is not connected")
        await self._ws.send_str(json.dumps(message, separators=(",", ":")))

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(_OVERLOAD_RETRIES + 1):
            try:
                return await self._request_once(method, params)
            except _CodexRPCError as exc:
                if exc.code != _OVERLOAD_ERROR_CODE or attempt == _OVERLOAD_RETRIES:
                    raise
                # App-server guarantees -32001 means ingress rejection, so replay is safe.
                delay = 0.1 * (2**attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _request_once(self, method: str, params: dict[str, Any] | None = None) -> Any:
        reader_stopped = self._reader_task is None or self._reader_task.done()
        if self._ws is None or self._ws.closed or reader_stopped:
            if method == "initialize":
                raise BackendError("Codex app-server is not connected")
            await self.start()
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            await self._send(message)
            return await asyncio.wait_for(future, 30)
        except TimeoutError as exc:
            raise BackendError(f"Codex request {method} timed out") from exc
        finally:
            pending = self._pending.pop(request_id, None)
            if pending is not None and not pending.done():
                pending.cancel()

    def events(self) -> AsyncIterator[BackendEvent]:
        async def iterate() -> AsyncIterator[BackendEvent]:
            while True:
                yield await self._events.get()

        return iterate()

    async def list_sessions(self, cwd: str) -> list[SessionSummary]:
        result = await self._request(
            "thread/list",
            {
                "limit": 20,
                "sortKey": "updated_at",
                "sortDirection": "desc",
                "cwd": cwd,
            },
        )
        sessions: list[SessionSummary] = []
        for item in (result or {}).get("data") or []:
            if not isinstance(item, dict) or not self._top_level_thread(item):
                continue
            sessions.append(self._summary(item))
        tui_sessions = self._tui_session_ids()
        return [replace(session, tui_attached=session.id in tui_sessions) for session in sessions]

    async def list_running_sessions(self) -> list[SessionSummary]:
        sessions: list[SessionSummary] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "limit": 100,
                "sortKey": "updated_at",
                "sortDirection": "desc",
            }
            if cursor:
                params["cursor"] = cursor
            result = await self._request("thread/list", params)
            for item in (result or {}).get("data") or []:
                if isinstance(item, dict) and self._top_level_thread(item):
                    sessions.append(self._summary(item))
            cursor = str((result or {}).get("nextCursor") or "") or None
            if not cursor:
                break

        explicit_ids = self._tui_session_ids()
        selected = {session.id for session in sessions if session.busy}
        selected.update(explicit_ids)
        return [
            replace(session, tui_attached=session.id in explicit_ids)
            for session in sessions
            if session.id in selected
        ]

    async def list_history(
        self,
        session_id: str,
        cursor: str | None,
        limit: int,
    ) -> HistoryPage:
        params: dict[str, Any] = {
            "threadId": session_id,
            "limit": limit,
            "sortDirection": "desc",
            "itemsView": "full",
        }
        if cursor:
            params["cursor"] = cursor
        result = await self._request("thread/turns/list", params)
        turns = [turn for turn in (result or {}).get("data") or [] if isinstance(turn, dict)]
        events: list[BackendEvent] = []
        fallback_at = int(time.time() * 1000) - len(turns) * 1000
        for turn_index, turn in enumerate(reversed(turns)):
            turn_id = str(turn.get("id") or "") or None
            turn_at = self._timestamp_ms(
                turn.get("createdAt") or turn.get("startedAt") or turn.get("updatedAt"),
                fallback_at + turn_index * 1000,
            )
            events.append(
                BackendEvent(
                    kind="turn_started",
                    backend=self.name,
                    session_id=session_id,
                    turn_id=turn_id,
                    at=turn_at,
                    event_id=self._history_event_id(session_id, turn_id, None, "turn.started"),
                )
            )
            for item_index, item in enumerate(turn.get("items") or []):
                if not isinstance(item, dict):
                    continue
                event = self._history_item(
                    session_id,
                    turn_id,
                    item,
                    turn_at + item_index + 1,
                )
                if event is not None:
                    events.append(event)
            status = str(turn.get("status") or "completed")
            if status in {"inProgress", "in_progress", "running"}:
                continue
            kind = "turn_failed" if status in {"failed", "error"} else "turn_done"
            error = turn.get("error")
            message = (
                safe_one_line(str(error.get("message") or status))
                if kind == "turn_failed" and isinstance(error, dict)
                else ""
            )
            events.append(
                BackendEvent(
                    kind=kind,
                    backend=self.name,
                    session_id=session_id,
                    turn_id=turn_id,
                    text=message,
                    at=turn_at + len(turn.get("items") or []) + 1,
                    event_id=self._history_event_id(
                        session_id,
                        turn_id,
                        None,
                        "turn.failed" if kind == "turn_failed" else "turn.completed",
                    ),
                )
            )
        return HistoryPage(
            events=tuple(events),
            next_cursor=str((result or {}).get("nextCursor") or "") or None,
        )

    def _history_item(
        self,
        session_id: str,
        turn_id: str | None,
        item: dict[str, Any],
        at: int,
    ) -> BackendEvent | None:
        item_type = str(item.get("type") or "")
        item_id = str(item.get("id") or "") or None
        common = {
            "backend": self.name,
            "session_id": session_id,
            "turn_id": turn_id,
            "item_id": item_id,
            "at": at,
        }
        if item_type == "userMessage":
            text = self._user_message_text(item)
            if not text:
                return None
            return BackendEvent(
                kind="user_prompt",
                text=text,
                event_id=self._history_event_id(session_id, turn_id, item_id, "user.prompt"),
                **common,
            )
        if item_type == "agentMessage":
            text = str(item.get("text") or "").strip()
            if not text:
                return None
            progress = item.get("phase") == "commentary"
            return BackendEvent(
                kind="progress" if progress else "assistant",
                text=text,
                data={"plan": False} if progress else {},
                event_id=self._history_event_id(
                    session_id,
                    turn_id,
                    item_id,
                    "plan.updated" if progress else "assistant.completed",
                ),
                **common,
            )
        if item_type == "plan":
            text = str(item.get("text") or item.get("explanation") or "Plan updated").strip()
            return BackendEvent(
                kind="progress",
                text=text,
                data={"plan": True, "running": False},
                event_id=self._history_event_id(session_id, turn_id, item_id, "plan.updated"),
                **common,
            )
        tool_kind = self._tool_kind(item_type)
        if tool_kind:
            status = str(item.get("status") or "").lower()
            finished = status not in {"", "inprogress", "in_progress", "running", "pending"}
            return BackendEvent(
                kind="tool_finished" if finished else "tool_started",
                tool_kind=tool_kind,
                success=self._tool_success(item) if finished else None,
                data=self._tool_metadata(item),
                event_id=self._history_event_id(
                    session_id,
                    turn_id,
                    item_id,
                    "tool.completed" if finished else "tool.started",
                ),
                **common,
            )
        # Reasoning and unsupported attachment types are intentionally omitted.
        return None

    @staticmethod
    def _user_message_text(item: Mapping[str, Any]) -> str:
        direct = str(item.get("text") or "").strip()
        if direct:
            return direct
        parts: list[str] = []
        for content in item.get("content") or ():
            if not isinstance(content, Mapping):
                continue
            if content.get("type") in {"text", "inputText"}:
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _timestamp_ms(value: Any, fallback: int) -> int:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timestamp = float(value)
            return int(timestamp if timestamp > 10_000_000_000 else timestamp * 1000)
        return fallback

    @staticmethod
    def _history_event_id(
        session_id: str,
        turn_id: str | None,
        item_id: str | None,
        kind: str,
    ) -> str:
        seed = "\0".join((session_id, turn_id or "", item_id or "", kind))
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentwire-history:{seed}"))

    def _tui_session_ids(self) -> set[str]:
        sessions = self._explicit_codex_sessions()
        sessions.update(CodexTuiSessionPresence.sessions(self.config.socket_path))
        return sessions

    async def session_busy(self, session_id: str) -> bool | None:
        return self._rollout_busy(session_id)

    @staticmethod
    def _rollout_busy(
        session_id: str,
        sessions_root: Path | None = None,
    ) -> bool | None:
        if re.fullmatch(r"[0-9a-f-]+", session_id) is None:
            return None
        if sessions_root is None:
            codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
            sessions_root = codex_home / "sessions"
        try:
            candidates = tuple(sessions_root.rglob(f"*{session_id}.jsonl"))
            rollout = max(candidates, key=lambda path: path.stat().st_mtime)
            with rollout.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 262_144))
                data = handle.read()
        except (OSError, ValueError):
            return None
        for raw in reversed(data.splitlines()):
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if record.get("type") != "event_msg":
                continue
            payload = record.get("payload") or {}
            event_type = payload.get("type") if isinstance(payload, dict) else None
            if event_type == "task_started":
                return True
            if event_type in {"task_complete", "task_failed", "turn_aborted"}:
                return False
        return None

    @staticmethod
    def _explicit_codex_sessions(
        proc_root: Path = Path("/proc"),
    ) -> set[str]:
        explicit_ids: set[str] = set()
        try:
            processes = tuple(proc_root.iterdir())
        except OSError:
            return explicit_ids
        for process in processes:
            if not process.name.isdigit():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                arguments = [
                    part.decode(errors="replace")
                    for part in (process / "cmdline").read_bytes().split(b"\0")
                    if part
                ]
                if not arguments or Path(arguments[0]).name != "codex":
                    continue
                if "app-server" in arguments[1:]:
                    continue
                if "resume" in arguments[1:]:
                    position = arguments.index("resume")
                    if position + 1 < len(arguments):
                        explicit_ids.add(arguments[position + 1])
                rollout = CodexBackend._process_rollout_session(process)
                if rollout:
                    explicit_ids.add(rollout)
            except (OSError, ValueError):
                continue
        return explicit_ids

    @staticmethod
    def _process_rollout_session(process: Path) -> str | None:
        """Return the newest top-level Codex TUI rollout held open by a process."""
        candidates: list[tuple[int, str]] = []
        try:
            descriptors = tuple((process / "fd").iterdir())
        except OSError:
            return None
        for descriptor in descriptors:
            try:
                target = descriptor.resolve(strict=True)
                if target.suffix != ".jsonl":
                    continue
                with target.open(encoding="utf-8") as handle:
                    first = json.loads(handle.readline())
                if first.get("type") != "session_meta":
                    continue
                payload = first.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("originator") != "codex-tui" or payload.get("source") != "cli":
                    continue
                session_id = str(payload.get("id") or "")
                if not session_id:
                    continue
                candidates.append((target.stat().st_mtime_ns, session_id))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
        return max(candidates)[1] if candidates else None

    @staticmethod
    def _top_level_thread(thread: Mapping[str, Any]) -> bool:
        return not bool(thread.get("parentThreadId"))

    async def create_session(self, cwd: str) -> SessionSummary:
        result = await self._request(
            "thread/start",
            {"cwd": cwd, "serviceName": "agentwire"},
        )
        thread = (result or {}).get("thread") or {}
        if not thread.get("id"):
            raise BackendError("Codex thread/start returned no thread id")
        return self._summary(thread)

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        params: dict[str, Any] = {
            "threadId": session_id,
            "excludeTurns": True,
            "initialTurnsPage": {
                "limit": 3,
                "sortDirection": "desc",
                "itemsView": "full",
            },
        }
        if cwd:
            params["cwd"] = cwd
        result = await self._request("thread/resume", params)
        thread = (result or {}).get("thread") or {}
        if not thread.get("id"):
            raise BackendError("Codex thread/resume returned no thread")
        summary = self._summary(thread)
        turns = ((result or {}).get("initialTurnsPage") or {}).get("data") or []
        active_turn_id = next(
            (
                str(turn.get("id") or "") or None
                for turn in turns
                if isinstance(turn, dict) and turn.get("status") == "inProgress"
            ),
            None,
        )
        if active_turn_id:
            self._active_turns[session_id] = active_turn_id
        recent_activity = await self._active_turn_activity(
            session_id,
            active_turn_id,
            turns,
        )
        recent_outputs: list[SessionOutput] = []
        for turn in reversed(turns):
            if not isinstance(turn, dict):
                continue
            turn_id = str(turn.get("id") or "") or None
            for index, item in enumerate(turn.get("items") or []):
                if not isinstance(item, dict) or item.get("type") != "agentMessage":
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                recent_outputs.append(
                    SessionOutput(
                        id=str(item.get("id") or f"{turn_id or 'turn'}-output-{index}"),
                        turn_id=turn_id,
                        text=text,
                        phase=str(item.get("phase") or "") or None,
                    )
                )
        recent_outputs = recent_outputs[-3:]
        latest_turn = next((turn for turn in turns if isinstance(turn, dict)), {})
        messages = [
            item
            for item in latest_turn.get("items") or []
            if isinstance(item, dict) and item.get("type") == "agentMessage"
        ]
        last_output = (
            next(
                (
                    str(item.get("text") or "").strip()
                    for item in reversed(messages)
                    if str(item.get("text") or "").strip()
                ),
                "",
            )
            or None
        )
        last_reply = self._select_final_message(messages) or None
        if last_reply:
            self._last_replies[session_id] = last_reply
        return SessionSummary(
            id=summary.id,
            cwd=summary.cwd,
            title=summary.title,
            updated_at=summary.updated_at,
            busy=summary.busy,
            active_flags=summary.active_flags,
            tui_attached=session_id in self._tui_session_ids(),
            active_turn_id=active_turn_id,
            last_output=last_output,
            last_reply=last_reply,
            recent_outputs=tuple(recent_outputs),
            recent_activity=tuple(recent_activity),
        )

    async def _active_turn_activity(
        self,
        session_id: str,
        active_turn_id: str | None,
        resumed_turns: list[Any],
    ) -> list[SessionActivity]:
        if not active_turn_id:
            return []
        items: list[dict[str, Any]] = []
        try:
            result = await self._request(
                "thread/items/list",
                {
                    "threadId": session_id,
                    "turnId": active_turn_id,
                    "limit": 50,
                    "sortDirection": "desc",
                },
            )
            for entry in reversed((result or {}).get("data") or []):
                item = entry.get("item") if isinstance(entry, dict) else None
                if isinstance(item, dict):
                    items.append(item)
        except BackendError:
            # Older app-server builds can still return current in-memory items on resume.
            active_turn = next(
                (
                    turn
                    for turn in resumed_turns
                    if isinstance(turn, dict) and str(turn.get("id") or "") == active_turn_id
                ),
                {},
            )
            items = [item for item in active_turn.get("items") or [] if isinstance(item, dict)]

        activity: list[SessionActivity] = []
        for item in items:
            item_id = str(item.get("id") or "")
            tool_kind = self._tool_kind(str(item.get("type") or ""))
            if not item_id or not tool_kind:
                continue
            status = str(item.get("status") or "").lower()
            finished = status in {
                "completed",
                "success",
                "failed",
                "error",
                "declined",
                "interrupted",
            } or isinstance(item.get("exitCode"), int)
            activity.append(
                SessionActivity(
                    kind="tool_finished" if finished else "tool_started",
                    item_id=item_id,
                    turn_id=active_turn_id,
                    tool_kind=tool_kind,
                    success=self._tool_success(item) if finished else None,
                    data=self._tool_metadata(item),
                )
            )
        return activity[-6:]

    @staticmethod
    def _summary(thread: dict[str, Any]) -> SessionSummary:
        status = thread.get("status") or {}
        status_type = str(status.get("type") if isinstance(status, dict) else status)
        active_flags = (
            tuple(str(flag) for flag in status.get("activeFlags") or ())
            if isinstance(status, dict)
            else ()
        )
        return SessionSummary(
            id=str(thread.get("id") or ""),
            cwd=str(thread.get("cwd") or ""),
            title=safe_one_line(
                str(thread.get("name") or thread.get("preview") or "untitled"), 100
            ),
            updated_at=float(thread.get("updatedAt") or thread.get("createdAt") or time.time()),
            busy=status_type == "active",
            active_flags=active_flags,
        )

    @staticmethod
    def _input(text: str) -> list[dict[str, Any]]:
        return [{"type": "text", "text": text, "text_elements": []}]

    async def send_message(self, session_id: str, text: str) -> str | None:
        params: dict[str, Any] = {"threadId": session_id, "input": self._input(text)}
        settings = self._session_settings.get(session_id, {})
        params.update(self._codex_settings(settings))
        result = await self._request("turn/start", params)
        turn = (result or {}).get("turn") or {}
        turn_id = str(turn.get("id") or "") or None
        if turn_id:
            self._active_turns[session_id] = turn_id
        return turn_id

    async def configure_session(self, session_id: str, settings: Mapping[str, Any]) -> None:
        allowed = {"model", "effort", "collaboration", "delivery", "approvalReviewer"}
        unsupported = set(settings) - allowed
        if unsupported:
            raise BackendError(f"unsupported Codex settings: {', '.join(sorted(unsupported))}")
        if settings.get("collaboration") and not settings.get("model"):
            raise BackendError("Codex collaboration mode requires an explicit model")
        params = {"threadId": session_id, **self._codex_settings(settings)}
        await self._request("thread/settings/update", params)
        self._session_settings[session_id] = dict(settings)

    async def setting_options(self) -> Mapping[str, Any]:
        if self._setting_options is not None:
            return self._setting_options
        models: list[dict[str, Any]] = []
        model_ids: set[str] = set()
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 100, "includeHidden": False}
            if cursor is not None:
                params["cursor"] = cursor
            result = await self._request("model/list", params)
            if not isinstance(result, dict):
                raise BackendError("Codex returned an invalid model catalog")
            entries = result.get("data")
            if not isinstance(entries, list):
                raise BackendError("Codex returned an invalid model catalog")
            for item in entries:
                if not isinstance(item, dict):
                    continue
                model_id = str(item.get("model") or item.get("id") or "")
                if not model_id:
                    continue
                efforts = [
                    str(option["reasoningEffort"])
                    for option in item.get("supportedReasoningEfforts") or ()
                    if isinstance(option, dict) and option.get("reasoningEffort")
                ]
                model: dict[str, Any] = {
                    "value": model_id,
                    "label": str(item.get("displayName") or model_id),
                    "efforts": efforts,
                }
                if default_effort := item.get("defaultReasoningEffort"):
                    model["defaultEffort"] = str(default_effort)
                if item.get("isDefault") is True:
                    model["default"] = True
                if model_id not in model_ids:
                    models.append(model)
                    model_ids.add(model_id)
            next_cursor = result.get("nextCursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)
        self._setting_options = {"model": models}
        return self._setting_options

    @staticmethod
    def _codex_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
        params = {key: settings[key] for key in ("model", "effort") if key in settings}
        if collaboration := settings.get("collaboration"):
            params["collaborationMode"] = {
                "mode": collaboration,
                "settings": {
                    "model": settings["model"],
                    "reasoning_effort": settings.get("effort"),
                    "developer_instructions": None,
                },
            }
        if reviewer := settings.get("approvalReviewer"):
            params["approvalsReviewer"] = "auto_review" if reviewer == "auto_review" else "user"
        return params

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        expected = turn_id or self._active_turns.get(session_id)
        if not expected:
            raise BackendError("Codex has no active turn to steer")
        await self._request(
            "turn/steer",
            {
                "threadId": session_id,
                "expectedTurnId": expected,
                "input": self._input(text),
            },
        )

    async def cancel(self, session_id: str, turn_id: str | None) -> None:
        active = turn_id or self._active_turns.get(session_id)
        if not active:
            raise BackendError("Codex has no active turn to cancel")
        await self._request("turn/interrupt", {"threadId": session_id, "turnId": active})

    async def resolve_approval(self, request_token: str | int, allow: bool) -> None:
        try:
            method, params = self._server_requests.pop(request_token)
        except KeyError as exc:
            raise BackendError("approval was already resolved") from exc
        if (
            method == "item/commandExecution/requestApproval"
            or method == "item/fileChange/requestApproval"
        ):
            result = {"decision": "accept" if allow else "decline"}
        elif method in {"execCommandApproval", "applyPatchApproval"}:
            result = {"decision": "approved" if allow else "denied"}
        elif method == "item/permissions/requestApproval":
            requested = params.get("permissions") if allow else {}
            result = {"permissions": requested or {}, "scope": "turn"}
        else:
            raise BackendError("unsupported Codex approval type")
        await self._send({"id": request_token, "result": result})

    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None:
        try:
            method, _params = self._server_requests.pop(request_token)
        except KeyError as exc:
            raise BackendError("question was already resolved") from exc
        if method != "item/tool/requestUserInput":
            raise BackendError("this input request must be resolved in the TUI")
        mapped: dict[str, dict[str, list[str]]] = {}
        for question, values in zip(questions, answers or (), strict=False):
            mapped[question.id] = {"answers": list(values)}
        await self._send({"id": request_token, "result": {"answers": mapped}})

    async def get_last_reply(self, session_id: str) -> str | None:
        if cached := self._last_replies.get(session_id):
            return cached
        result = await self._request("thread/read", {"threadId": session_id, "includeTurns": True})
        thread = (result or {}).get("thread") or {}
        for turn in reversed(thread.get("turns") or []):
            if not isinstance(turn, dict):
                continue
            messages = [
                item
                for item in turn.get("items") or []
                if isinstance(item, dict) and item.get("type") == "agentMessage"
            ]
            if text := self._select_final_message(messages):
                self._last_replies[session_id] = text
                return text
        return None
