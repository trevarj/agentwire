from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import quote

import aiohttp

from agentwire.backends.base import Backend, BackendError
from agentwire.config import OpenCodeConfig
from agentwire.models import BackendEvent, Question, SessionSummary
from agentwire.text import safe_one_line


class OpenCodeBackend(Backend):
    name = "opencode"

    def __init__(self, config: OpenCodeConfig, password: str) -> None:
        self.config = config
        self.password = password
        self._session: aiohttp.ClientSession | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._session_cwds: dict[str, str] = {}
        self._busy: set[str] = set()
        self._active_messages: dict[str, str] = {}
        self._last_message_ids: dict[str, str] = {}
        self._last_replies: dict[str, str] = {}
        self._tool_states: dict[str, str] = {}
        self._requests: dict[str, tuple[str, str, tuple[Question, ...]]] = {}

    async def start(self) -> None:
        if self._reader_task is not None:
            return
        auth = aiohttp.BasicAuth(self.config.username, self.password)
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(auth=auth, timeout=timeout)
        deadline = asyncio.get_running_loop().time() + 30
        last_error: BaseException | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                health = await self._json("GET", "/global/health", timeout=2)
                if not isinstance(health, dict) or health.get("healthy") is not True:
                    raise BackendError("OpenCode health check was not healthy")
                self._reader_task = asyncio.create_task(self._event_loop(), name="opencode-events")
                self._ready.set()
                await self._events.put(BackendEvent(kind="connected", backend=self.name))
                return
            except (aiohttp.ClientError, OSError, TimeoutError, BackendError) as exc:
                last_error = exc
                await asyncio.sleep(0.25)
        await self.close()
        raise BackendError(f"OpenCode server did not become ready: {last_error}")

    async def wait_ready(self, timeout: float = 30) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except TimeoutError as exc:
            raise BackendError("timed out waiting for OpenCode") from exc

    async def close(self) -> None:
        self._closed = True
        self._ready.clear()
        current = asyncio.current_task()
        if self._reader_task is not None and self._reader_task is not current:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        self._reader_task = None
        if self._session is not None:
            await self._session.close()
        self._session = None

    async def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        body: Any = None,
        expected: tuple[int, ...] = (200,),
        timeout: float = 30,
    ) -> Any:
        if self._session is None:
            raise BackendError("OpenCode is not connected")
        kwargs: dict[str, Any] = {"params": params}
        kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)
        if body is not None:
            kwargs["json"] = body
        try:
            async with self._session.request(
                method, f"{self.config.url}{path}", **kwargs
            ) as response:
                raw = await response.text()
                if response.status not in expected:
                    detail = ""
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(raw)
                        detail = safe_one_line(
                            str(
                                parsed.get("data", {}).get("message") or parsed.get("message") or ""
                            ),
                            160,
                        )
                    suffix = f": {detail}" if detail else ""
                    raise BackendError(
                        f"OpenCode {method} {path} returned HTTP {response.status}{suffix}"
                    )
                if not raw.strip():
                    return None
                return json.loads(raw)
        except (aiohttp.ClientError, json.JSONDecodeError, TimeoutError) as exc:
            raise BackendError(f"OpenCode request failed: {exc}") from exc

    async def _event_loop(self) -> None:
        delay = 0.25
        while not self._closed:
            try:
                await self._consume_events()
                if not self._closed:
                    raise BackendError("OpenCode event stream ended")
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._closed:
                    self._ready.clear()
                    await self._events.put(
                        BackendEvent(
                            kind="disconnected",
                            backend=self.name,
                            text="OpenCode event stream disconnected; reconnecting",
                        )
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10)
                    continue
            delay = 0.25

    async def _consume_events(self) -> None:
        assert self._session is not None
        stream_timeout = aiohttp.ClientTimeout(
            total=None, connect=30, sock_connect=30, sock_read=None
        )
        async with self._session.get(
            f"{self.config.url}/global/event", timeout=stream_timeout
        ) as response:
            if response.status != 200:
                raise BackendError(f"OpenCode event stream returned HTTP {response.status}")
            self._ready.set()
            buffer: list[str] = []
            async for raw in response.content:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if buffer:
                        payload = "\n".join(buffer)
                        buffer.clear()
                        with contextlib.suppress(json.JSONDecodeError):
                            await self._handle_global_event(json.loads(payload))
                    continue
                if line.startswith("data:"):
                    buffer.append(line[5:].lstrip())

    async def _handle_global_event(self, wrapper: Any) -> None:
        if not isinstance(wrapper, dict):
            return
        directory = str(wrapper.get("directory") or "")
        event = wrapper.get("payload", wrapper)
        if not isinstance(event, dict):
            return
        kind = str(event.get("type") or "")
        props = event.get("properties") or {}
        if not isinstance(props, dict):
            return
        session_id = str(props.get("sessionID") or "") or None
        if session_id and directory:
            self._session_cwds[session_id] = directory

        if kind == "session.status" and session_id:
            status = props.get("status") or {}
            status_type = str(status.get("type") if isinstance(status, dict) else status)
            if status_type in {"busy", "retry"}:
                if session_id not in self._busy:
                    self._busy.add(session_id)
                    await self._events.put(
                        BackendEvent(kind="turn_started", backend=self.name, session_id=session_id)
                    )
            elif status_type == "idle":
                await self._finish_turn(session_id)
            return
        if kind == "session.idle" and session_id:
            await self._finish_turn(session_id)
            return
        if kind == "session.error" and session_id:
            error = props.get("error") or {}
            message = "OpenCode turn failed"
            if isinstance(error, dict):
                data = error.get("data") or {}
                if isinstance(data, dict) and data.get("message"):
                    message = safe_one_line(str(data["message"]), 180)
            self._busy.discard(session_id)
            await self._events.put(
                BackendEvent(
                    kind="turn_failed",
                    backend=self.name,
                    session_id=session_id,
                    text=message,
                )
            )
            return
        if kind == "message.part.updated":
            part = props.get("part") or {}
            if isinstance(part, dict) and part.get("type") == "tool":
                await self._handle_tool(part)
            return
        if kind in {"permission.asked", "permission.v2.asked"} and session_id:
            token = str(props.get("id") or "")
            if not token:
                return
            action = str(props.get("permission") or props.get("action") or "tool")
            version = "v2" if ".v2." in kind else "v1"
            self._requests[token] = (version, session_id, ())
            await self._events.put(
                BackendEvent(
                    kind="approval",
                    backend=self.name,
                    session_id=session_id,
                    request_token=token,
                    text=f"{safe_one_line(action, 80)} approval needed",
                )
            )
            return
        if kind in {"question.asked", "question.v2.asked"} and session_id:
            token = str(props.get("id") or "")
            if not token:
                return
            questions = tuple(
                self._question(raw, index)
                for index, raw in enumerate(props.get("questions") or [], 1)
                if isinstance(raw, dict)
            )
            version = "v2" if ".v2." in kind else "v1"
            self._requests[token] = (version, session_id, questions)
            await self._events.put(
                BackendEvent(
                    kind="question",
                    backend=self.name,
                    session_id=session_id,
                    request_token=token,
                    questions=questions,
                )
            )
            return
        if kind in {
            "permission.replied",
            "permission.v2.replied",
            "question.replied",
            "question.v2.replied",
            "question.rejected",
            "question.v2.rejected",
        }:
            token = str(props.get("requestID") or "")
            self._requests.pop(token, None)
            await self._events.put(
                BackendEvent(
                    kind="request_resolved",
                    backend=self.name,
                    session_id=session_id,
                    request_token=token,
                )
            )

    async def _handle_tool(self, part: dict[str, Any]) -> None:
        part_id = str(part.get("id") or "")
        session_id = str(part.get("sessionID") or "") or None
        state = part.get("state") or {}
        status = str(state.get("status") if isinstance(state, dict) else "")
        previous = self._tool_states.get(part_id)
        if not part_id or not session_id or status == previous:
            return
        self._tool_states[part_id] = status
        tool_kind = self._tool_kind(str(part.get("tool") or ""))
        if status in {"pending", "running"} and previous not in {"pending", "running"}:
            await self._events.put(
                BackendEvent(
                    kind="tool_started",
                    backend=self.name,
                    session_id=session_id,
                    item_id=part_id,
                    tool_kind=tool_kind,
                )
            )
        elif status in {"completed", "error"}:
            await self._events.put(
                BackendEvent(
                    kind="tool_finished",
                    backend=self.name,
                    session_id=session_id,
                    item_id=part_id,
                    tool_kind=tool_kind,
                    success=status == "completed",
                )
            )

    @staticmethod
    def _tool_kind(tool: str) -> str:
        lowered = tool.lower()
        if any(part in lowered for part in ("bash", "shell", "command")):
            return "shell"
        if any(part in lowered for part in ("edit", "write", "patch")):
            return "file edit"
        if any(part in lowered for part in ("read", "glob", "grep")):
            return "file read"
        if any(part in lowered for part in ("web", "fetch", "search")):
            return "web"
        return "tool"

    @staticmethod
    def _question(raw: dict[str, Any], index: int) -> Question:
        prompt = str(raw.get("question") or "Input requested")
        header = str(raw.get("header") or f"Question {index}")
        secret_words = (
            "password",
            "passphrase",
            "secret",
            "token",
            "api key",
            "access key",
            "private key",
            "credential",
            "one-time code",
            "otp",
            "2fa",
        )
        secret = any(word in f"{header} {prompt}".lower() for word in secret_words)
        return Question(
            id=str(index),
            header=header,
            prompt=prompt,
            options=tuple(
                str(option.get("label"))
                for option in raw.get("options") or []
                if isinstance(option, dict) and option.get("label")
            ),
            multiple=bool(raw.get("multiple", False)),
            custom=bool(raw.get("custom", True)),
            secret=secret,
        )

    async def _finish_turn(self, session_id: str) -> None:
        was_busy = session_id in self._busy
        self._busy.discard(session_id)
        reply = await self._latest_reply(session_id)
        if reply is not None:
            message_id, text = reply
            if message_id != self._last_message_ids.get(session_id):
                self._last_message_ids[session_id] = message_id
                self._last_replies[session_id] = text
                await self._events.put(
                    BackendEvent(
                        kind="assistant",
                        backend=self.name,
                        session_id=session_id,
                        turn_id=self._active_messages.get(session_id),
                        text=text,
                    )
                )
        if was_busy:
            await self._events.put(
                BackendEvent(
                    kind="turn_done",
                    backend=self.name,
                    session_id=session_id,
                    turn_id=self._active_messages.pop(session_id, None),
                )
            )

    async def _latest_reply(self, session_id: str) -> tuple[str, str] | None:
        cwd = self._session_cwds.get(session_id)
        params: dict[str, str | int] = {"limit": 10}
        if cwd:
            params["directory"] = cwd
        messages = await self._json("GET", f"/session/{quote(session_id)}/message", params=params)
        for message in reversed(messages or []):
            if not isinstance(message, dict):
                continue
            info = message.get("info") or {}
            if not isinstance(info, dict) or info.get("role") != "assistant":
                continue
            texts: list[str] = []
            for part in message.get("parts") or []:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and not part.get("synthetic", False)
                    and not part.get("ignored", False)
                    and str(part.get("text") or "").strip()
                ):
                    texts.append(str(part["text"]))
            text = "\n\n".join(texts).strip()
            if text:
                return str(info.get("id") or hash(text)), text
        return None

    def events(self) -> AsyncIterator[BackendEvent]:
        async def iterate() -> AsyncIterator[BackendEvent]:
            while True:
                yield await self._events.get()

        return iterate()

    async def list_sessions(self, cwd: str) -> list[SessionSummary]:
        raw = await self._json("GET", "/session", params={"directory": cwd, "limit": 20})
        statuses = await self._json("GET", "/session/status")
        if not isinstance(statuses, dict):
            raise BackendError("OpenCode session/status returned an invalid response")
        sessions = [
            self._summary(item, self._status_type(statuses, str(item.get("id") or "")))
            for item in raw or []
            if isinstance(item, dict)
        ]
        for summary in sessions:
            self._session_cwds[summary.id] = summary.cwd
        return sessions

    async def list_running_sessions(self) -> list[SessionSummary]:
        statuses = await self._json("GET", "/session/status")
        if not isinstance(statuses, dict):
            raise BackendError("OpenCode session/status returned an invalid response")
        raw = await self._json("GET", "/session", params={"limit": 100})
        sessions: list[SessionSummary] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("id") or "")
            status = statuses.get(session_id) or {}
            status_type = str(status.get("type") if isinstance(status, dict) else status)
            if status_type not in {"busy", "retry", "active"}:
                continue
            summary = self._summary(item, status_type)
            sessions.append(summary)
            self._busy.add(summary.id)
            self._session_cwds[summary.id] = summary.cwd
        return sessions

    async def create_session(self, cwd: str) -> SessionSummary:
        raw = await self._json("POST", "/session", params={"directory": cwd}, body={})
        if not isinstance(raw, dict) or not raw.get("id"):
            raise BackendError("OpenCode session/create returned no session")
        summary = self._summary(raw)
        self._session_cwds[summary.id] = summary.cwd
        return summary

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        params = {"directory": cwd} if cwd else None
        raw = await self._json("GET", f"/session/{quote(session_id)}", params=params)
        if not isinstance(raw, dict) or not raw.get("id"):
            raise BackendError("OpenCode session/get returned no session")
        statuses = await self._json("GET", "/session/status")
        if not isinstance(statuses, dict):
            raise BackendError("OpenCode session/status returned an invalid response")
        status_type = self._status_type(statuses, session_id)
        summary = self._summary(raw, status_type)
        if summary.busy:
            self._busy.add(summary.id)
        self._session_cwds[summary.id] = summary.cwd
        return summary

    @staticmethod
    def _status_type(statuses: dict[str, Any], session_id: str) -> str:
        status = statuses.get(session_id) or {}
        return str(status.get("type") if isinstance(status, dict) else status)

    @staticmethod
    def _summary(raw: dict[str, Any], status_type: str = "") -> SessionSummary:
        times = raw.get("time") or {}
        updated = float(times.get("updated") or times.get("created") or time.time())
        if updated > 10_000_000_000:
            updated /= 1000
        busy = status_type in {"busy", "retry", "active"}
        return SessionSummary(
            id=str(raw.get("id") or ""),
            cwd=str(raw.get("directory") or raw.get("path") or ""),
            title=safe_one_line(str(raw.get("title") or "untitled"), 100),
            updated_at=updated,
            busy=busy,
            active_flags=("retry",) if status_type == "retry" else (),
        )

    async def send_message(self, session_id: str, text: str) -> str | None:
        params = None
        if cwd := self._session_cwds.get(session_id):
            params = {"directory": cwd}
        await self._json(
            "POST",
            f"/session/{quote(session_id)}/prompt_async",
            params=params,
            body={"parts": [{"type": "text", "text": text}]},
            expected=(204,),
        )
        self._busy.add(session_id)
        return None

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        await self._json(
            "POST",
            f"/api/session/{quote(session_id)}/prompt",
            body={"prompt": {"text": text}, "delivery": "steer"},
        )

    async def cancel(self, session_id: str, turn_id: str | None) -> None:
        await self._json("POST", f"/api/session/{quote(session_id)}/interrupt", expected=(204,))

    async def resolve_approval(self, request_token: str | int, allow: bool) -> None:
        token = str(request_token)
        try:
            version, session_id, _ = self._requests.pop(token)
        except KeyError as exc:
            raise BackendError("approval was already resolved") from exc
        if version == "v2":
            path = f"/api/session/{quote(session_id)}/permission/{quote(token)}/reply"
            expected = (204,)
        else:
            path = f"/permission/{quote(token)}/reply"
            expected = (200,)
        params = None
        if cwd := self._session_cwds.get(session_id):
            params = {"directory": cwd}
        await self._json(
            "POST",
            path,
            params=params,
            body={"reply": "once" if allow else "reject"},
            expected=expected,
        )

    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None:
        token = str(request_token)
        try:
            version, session_id, _ = self._requests.pop(token)
        except KeyError as exc:
            raise BackendError("question was already resolved") from exc
        if version == "v2":
            base = f"/api/session/{quote(session_id)}/question/{quote(token)}"
            expected = (204,)
        else:
            base = f"/question/{quote(token)}"
            expected = (200,)
        params = None
        if cwd := self._session_cwds.get(session_id):
            params = {"directory": cwd}
        if answers is None:
            await self._json("POST", f"{base}/reject", params=params, expected=expected)
        else:
            await self._json(
                "POST",
                f"{base}/reply",
                params=params,
                body={"answers": [list(values) for values in answers]},
                expected=expected,
            )

    async def get_last_reply(self, session_id: str) -> str | None:
        if cached := self._last_replies.get(session_id):
            return cached
        latest = await self._latest_reply(session_id)
        if latest is None:
            return None
        message_id, text = latest
        self._last_message_ids[session_id] = message_id
        self._last_replies[session_id] = text
        return text
