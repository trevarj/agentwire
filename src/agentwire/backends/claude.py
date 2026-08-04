from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    get_session_info,
    get_session_messages,
    list_sessions,
)

from agentwire.backends.base import Backend, BackendError
from agentwire.config import ClaudeConfig
from agentwire.models import (
    BackendEvent,
    HistoryPage,
    Question,
    SessionOutput,
    SessionSummary,
)
from agentwire.text import clean_block, safe_one_line, truncate_utf8

_MAX_TOOL_PAYLOAD_BYTES = 32 * 1024
_RECENT_OUTPUTS = 3

# Claude Code's built-in tool names, mapped onto the vocabulary the bridge and
# its clients already render for Codex and OpenCode.
_TOOL_KINDS = {
    "bash": "shell",
    "bashoutput": "shell",
    "killshell": "shell",
    "edit": "file edit",
    "multiedit": "file edit",
    "notebookedit": "file edit",
    "write": "file edit",
    "glob": "file read",
    "grep": "file read",
    "read": "file read",
    "webfetch": "web",
    "websearch": "web search",
    "agent": "agent",
    "task": "agent",
}


@dataclass(slots=True)
class _Session:
    """One Claude Code session and the CLI subprocess that currently owns it."""

    id: str
    cwd: str
    client: Any
    title: str = "untitled"
    updated_at: float = 0.0
    pump: asyncio.Task[None] | None = None
    busy: bool = False
    turn_id: str | None = None
    final_text: str = ""
    last_reply: str | None = None
    tools: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    # Tool calls rendered through a richer channel (plan updates, question
    # requests) whose raw tool cards and results must stay off the wire.
    hidden_tools: set[str] = field(default_factory=set)
    plan_signature: tuple[Any, ...] | None = None


class ClaudeBackend(Backend):
    """Expose Claude Agent SDK sessions through the Agentwire backend contract.

    Unlike Codex and OpenCode there is no shared long-running server: the SDK
    owns one ``claude`` CLI subprocess per session, so this backend keeps a pool
    of clients and folds every session's message stream into a single queue.
    """

    name = "claude"

    def __init__(self, config: ClaudeConfig, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = api_key
        self._events: asyncio.Queue[BackendEvent] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._sessions: dict[str, _Session] = {}
        self._approvals: dict[str, asyncio.Future[bool]] = {}
        # A resolved question carries per-question answer lists; None is a skip.
        self._questions: dict[str, asyncio.Future[list[list[str]] | None]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._ready.is_set():
            return
        self._closed = False
        self._ready.set()
        await self._events.put(BackendEvent(kind="connected", backend=self.name))

    async def wait_ready(self, timeout: float = 30) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except TimeoutError as exc:
            raise BackendError("timed out waiting for Claude") from exc

    async def close(self) -> None:
        self._closed = True
        self._ready.clear()
        for future in (*self._approvals.values(), *self._questions.values()):
            if not future.done():
                future.set_exception(BackendError("Claude backend is shutting down"))
        self._approvals.clear()
        self._questions.clear()
        for session in list(self._sessions.values()):
            await self._close_session(session)
        self._sessions.clear()

    async def _close_session(self, session: _Session) -> None:
        current = asyncio.current_task()
        if session.pump is not None and session.pump is not current:
            session.pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.pump
        session.pump = None
        with contextlib.suppress(Exception):
            await session.client.disconnect()

    def events(self) -> AsyncIterator[BackendEvent]:
        async def iterate() -> AsyncIterator[BackendEvent]:
            while True:
                yield await self._events.get()

        return iterate()

    # ------------------------------------------------------------------
    # session pool
    # ------------------------------------------------------------------

    def _options(self, session_id: str, cwd: str, resume: bool) -> ClaudeAgentOptions:
        # The transport already inherits this process's environment, so only the
        # one configured credential variable is added on top of it.
        environment: dict[str, str] = {}
        if self.config.api_key_env and self.api_key:
            environment[self.config.api_key_env] = self.api_key
        values: dict[str, Any] = {
            "cwd": cwd,
            "env": environment,
            "permission_mode": self.config.permission_mode,
            "can_use_tool": self._permission_handler(session_id),
            # Project instructions belong to the workspace the owner picked, but
            # user-global settings could silently pre-approve tools that this
            # bridge exists to route through IRC approvals.
            "setting_sources": ["project"],
            "cli_path": self.config.binary,
        }
        if self.config.model:
            values["model"] = self.config.model
        values["resume" if resume else "session_id"] = session_id
        return ClaudeAgentOptions(**values)

    def _permission_handler(self, session_id: str) -> Any:
        async def can_use_tool(
            tool_name: str,
            input_data: dict[str, Any],
            context: ToolPermissionContext,
        ) -> PermissionResultAllow | PermissionResultDeny:
            if tool_name == "AskUserQuestion":
                return await self._request_question(session_id, input_data, context)
            allowed = await self._request_approval(session_id, tool_name, input_data, context)
            if allowed:
                return PermissionResultAllow(updated_input=input_data)
            return PermissionResultDeny(message="denied by the Agentwire owner")

        return can_use_tool

    async def _request_approval(
        self,
        session_id: str,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> bool:
        token = str(context.tool_use_id or uuid.uuid4())
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._approvals[token] = future
        metadata = self._tool_metadata(tool_name, input_data)
        summary = context.title or metadata.get("label") or self._tool_kind(tool_name)
        await self._events.put(
            BackendEvent(
                kind="approval",
                backend=self.name,
                session_id=session_id,
                turn_id=self._turn_for(session_id),
                item_id=context.tool_use_id,
                request_token=token,
                text=f"{safe_one_line(str(summary), 120)} approval needed",
                data={"tool": tool_name},
            )
        )
        try:
            return await future
        finally:
            self._approvals.pop(token, None)

    async def _request_question(
        self,
        session_id: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Route Claude's AskUserQuestion tool through the Agentwire question flow.

        The CLI answers the tool itself when the permission callback allows it
        with ``{"questions": ..., "answers": {question text: label}}``, so the
        owner's answers travel back inside ``updated_input`` (verified against
        Claude Code 2.1.220).
        """
        raw_questions = [
            item for item in input_data.get("questions") or () if isinstance(item, dict)
        ]
        questions = tuple(
            self._parse_question(raw, index) for index, raw in enumerate(raw_questions, 1)
        )
        if not questions:
            return PermissionResultDeny(message="the question could not be relayed")
        token = str(context.tool_use_id or uuid.uuid4())
        future: asyncio.Future[list[list[str]] | None] = asyncio.get_running_loop().create_future()
        self._questions[token] = future
        await self._events.put(
            BackendEvent(
                kind="question",
                backend=self.name,
                session_id=session_id,
                turn_id=self._turn_for(session_id),
                item_id=context.tool_use_id,
                request_token=token,
                questions=questions,
            )
        )
        try:
            answers = await future
        finally:
            self._questions.pop(token, None)
        if answers is None:
            return PermissionResultDeny(message="the Agentwire owner skipped the question")
        selected: dict[str, Any] = {}
        for question, raw, values in zip(questions, raw_questions, answers, strict=False):
            key = str(raw.get("question") or question.prompt)
            # Single-select answers are a bare label; multi-select is a list.
            selected[key] = list(values) if question.multiple else next(iter(values), "")
        return PermissionResultAllow(
            updated_input={"questions": raw_questions, "answers": selected}
        )

    @staticmethod
    def _parse_question(raw: dict[str, Any], index: int) -> Question:
        return Question(
            id=str(index),
            header=str(raw.get("header") or f"Question {index}"),
            prompt=str(raw.get("question") or "Input requested"),
            options=tuple(
                str(option.get("label"))
                for option in raw.get("options") or ()
                if isinstance(option, dict) and option.get("label")
            ),
            multiple=bool(raw.get("multiSelect", False)),
        )

    def _turn_for(self, session_id: str) -> str | None:
        session = self._sessions.get(session_id)
        return session.turn_id if session else None

    def _require(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise BackendError(f"Claude session {session_id} is not attached")
        return session

    async def _connect(self, session_id: str, cwd: str, resume: bool) -> _Session:
        client = ClaudeSDKClient(options=self._options(session_id, cwd, resume))
        try:
            await client.connect()
        except (ClaudeSDKError, OSError, TimeoutError) as exc:
            with contextlib.suppress(Exception):
                await client.disconnect()
            raise BackendError(f"Claude session could not start: {exc}") from exc
        session = _Session(id=session_id, cwd=cwd, client=client, updated_at=time.time())
        self._sessions[session_id] = session
        session.pump = asyncio.create_task(self._pump(session), name=f"claude-{session_id}")
        return session

    async def _pump(self, session: _Session) -> None:
        try:
            async for message in session.client.receive_messages():
                await self._handle_message(session, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._closed:
                return
            session.busy = False
            await self._events.put(
                BackendEvent(
                    kind="disconnected",
                    backend=self.name,
                    session_id=session.id,
                    text=safe_one_line(f"Claude session stream ended: {exc}", 180),
                )
            )

    # ------------------------------------------------------------------
    # live message translation
    # ------------------------------------------------------------------

    async def _handle_message(self, session: _Session, message: Any) -> None:
        if isinstance(message, AssistantMessage):
            await self._handle_assistant(session, message)
        elif isinstance(message, UserMessage):
            await self._handle_tool_results(session, message)
        elif isinstance(message, ResultMessage):
            await self._finish_turn(session, message)

    async def _handle_assistant(self, session: _Session, message: AssistantMessage) -> None:
        narration: list[str] = []
        tools: list[ToolUseBlock] = []
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                narration.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tools.append(block)
        text = "\n\n".join(narration).strip()
        if tools:
            # Text emitted alongside a tool call is narration, never the answer.
            if text:
                await self._events.put(
                    BackendEvent(
                        kind="progress",
                        backend=self.name,
                        session_id=session.id,
                        turn_id=session.turn_id,
                        item_id=message.message_id,
                        text=text,
                    )
                )
            for block in tools:
                await self._start_tool(session, block)
        elif text:
            session.final_text = text

    async def _start_tool(self, session: _Session, block: ToolUseBlock) -> None:
        if block.name == "TodoWrite":
            session.hidden_tools.add(block.id)
            await self._emit_plan(session, block.input)
            return
        if block.name == "AskUserQuestion":
            # The question request card comes from the permission callback; a
            # duplicate raw tool card would only confuse the timeline.
            session.hidden_tools.add(block.id)
            return
        kind = self._tool_kind(block.name)
        data = self._tool_metadata(block.name, block.input)
        session.tools[block.id] = (kind, data)
        await self._events.put(
            BackendEvent(
                kind="tool_started",
                backend=self.name,
                session_id=session.id,
                turn_id=session.turn_id,
                item_id=block.id,
                tool_kind=kind,
                data=data,
            )
        )

    async def _emit_plan(self, session: _Session, payload: Any) -> None:
        summary, data = self._plan_update(payload)
        signature = (
            summary,
            data["status"],
            data["completedSteps"],
            data["totalSteps"],
        )
        if session.plan_signature == signature:
            return
        session.plan_signature = signature
        await self._events.put(
            BackendEvent(
                kind="progress",
                backend=self.name,
                session_id=session.id,
                turn_id=session.turn_id,
                text=summary,
                data=data,
            )
        )

    @staticmethod
    def _plan_update(payload: Any) -> tuple[str, dict[str, Any]]:
        """Map Claude's TodoWrite input onto the bridge's plan progress shape."""
        raw = payload.get("todos") if isinstance(payload, dict) else None
        todos = [item for item in raw or () if isinstance(item, dict)]
        active = next(
            (
                str(item.get("activeForm") or item.get("content") or "").strip()
                for item in todos
                if item.get("status") == "in_progress"
            ),
            "",
        )
        completed = sum(item.get("status") == "completed" for item in todos)
        complete = bool(todos) and completed == len(todos)
        status = "completed" if complete else "inProgress" if active else "pending"
        summary = "Plan completed" if complete else f"plan: {active}" if active else "Plan updated"
        return summary, {
            "plan": True,
            "running": bool(todos) and not complete,
            "status": status,
            "completedSteps": completed,
            "totalSteps": len(todos),
        }

    async def _handle_tool_results(self, session: _Session, message: UserMessage) -> None:
        content = message.content
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, ToolResultBlock):
                continue
            if block.tool_use_id in session.hidden_tools:
                session.hidden_tools.discard(block.tool_use_id)
                continue
            entry = session.tools.pop(block.tool_use_id, None)
            if entry is None:
                continue
            kind, data = entry
            payload = dict(data)
            payload["status"] = "error" if block.is_error else "completed"
            output = self._result_text(block.content)
            if output:
                payload["output"] = truncate_utf8(clean_block(output), _MAX_TOOL_PAYLOAD_BYTES)
            await self._events.put(
                BackendEvent(
                    kind="tool_finished",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=session.turn_id,
                    item_id=block.tool_use_id,
                    tool_kind=kind,
                    success=not bool(block.is_error),
                    data=payload,
                )
            )

    async def _finish_turn(self, session: _Session, message: ResultMessage) -> None:
        turn_id = session.turn_id
        text = (message.result or "").strip() or session.final_text
        session.busy = False
        session.turn_id = None
        session.final_text = ""
        session.plan_signature = None
        session.updated_at = time.time()
        if message.is_error:
            detail = safe_one_line(
                str(message.result or message.subtype or "Claude turn failed"), 180
            )
            await self._events.put(
                BackendEvent(
                    kind="turn_failed",
                    backend=self.name,
                    session_id=session.id,
                    turn_id=turn_id,
                    text=detail,
                )
            )
            return
        if text:
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
        await self._events.put(
            BackendEvent(
                kind="turn_done",
                backend=self.name,
                session_id=session.id,
                turn_id=turn_id,
            )
        )

    # ------------------------------------------------------------------
    # tool presentation
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_kind(name: str) -> str:
        if name.startswith("mcp__"):
            return "MCP tool"
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
        if name == "Bash":
            command = field_text("command")
            if command:
                data["label"] = f"$ {safe_one_line(command, 160)}"
                data["input"] = truncate_utf8(clean_block(command), _MAX_TOOL_PAYLOAD_BYTES)
        elif name in {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit"}:
            path = field_text("file_path", "notebook_path", "path")
            verb = "Read" if name == "Read" else "Edit"
            data["label"] = safe_one_line(f"{verb} {path}" if path else name, 160)
        elif name in {"Glob", "Grep"}:
            pattern = field_text("pattern", "query")
            data["label"] = safe_one_line(f"Search: {pattern}" if pattern else name, 160)
        elif name == "WebSearch":
            query = field_text("query")
            data["label"] = safe_one_line(f"Search: {query}" if query else name, 160)
        elif name == "WebFetch":
            url = field_text("url")
            data["label"] = safe_one_line(f"Fetch: {url}" if url else name, 160)
        elif name == "Task":
            description = field_text("description", "subagent_type")
            data["label"] = safe_one_line(description or name, 160)
        elif name == "AskUserQuestion":
            # Only history renders this as a tool card; live turns raise a request.
            first = next(
                (
                    str(item.get("question") or "").strip()
                    for item in values.get("questions") or ()
                    if isinstance(item, dict)
                ),
                "",
            )
            data["label"] = safe_one_line(f"Question: {first}" if first else "Question", 160)
        elif name.startswith("mcp__"):
            parts = [part for part in name.split("__")[1:] if part]
            data["label"] = safe_one_line(" / ".join(parts) or name, 160)
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
        try:
            infos = await asyncio.to_thread(list_sessions, directory=cwd, limit=20)
        except OSError as exc:
            raise BackendError(f"cannot read Claude sessions for {cwd}: {exc}") from exc
        return [self._summary(info, cwd) for info in infos]

    async def list_running_sessions(self) -> list[SessionSummary]:
        # Claude has no shared daemon, so only sessions this bridge started can
        # still be running a turn.
        return [self._live_summary(session) for session in self._sessions.values() if session.busy]

    async def create_session(self, cwd: str) -> SessionSummary:
        session = await self._connect(str(uuid.uuid4()), cwd, resume=False)
        return self._live_summary(session)

    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary:
        existing = self._sessions.get(session_id)
        if existing is not None:
            return await self._resumed_summary(existing)
        try:
            info = await asyncio.to_thread(get_session_info, session_id, cwd)
        except OSError as exc:
            raise BackendError(f"cannot read Claude session {session_id}: {exc}") from exc
        workspace = cwd or (info.cwd if info is not None else None)
        if not workspace:
            raise BackendError(f"Claude session {session_id} has no recorded workspace")
        session = await self._connect(session_id, workspace, resume=True)
        if info is not None:
            session.title = self._title(info)
            session.updated_at = self._seconds(info.last_modified)
        return await self._resumed_summary(session)

    async def _resumed_summary(self, session: _Session) -> SessionSummary:
        outputs = await self._recent_outputs(session)
        if outputs and session.last_reply is None:
            session.last_reply = outputs[-1].text
        return SessionSummary(
            id=session.id,
            cwd=session.cwd,
            title=session.title,
            updated_at=session.updated_at,
            busy=session.busy,
            active_turn_id=session.turn_id,
            last_output=outputs[-1].text if outputs else None,
            last_reply=session.last_reply,
            recent_outputs=tuple(outputs),
        )

    async def _recent_outputs(self, session: _Session) -> list[SessionOutput]:
        messages = await self._transcript(session.id, session.cwd)
        outputs: list[SessionOutput] = []
        for message in messages:
            if message.type != "assistant":
                continue
            text = self._message_text(message.message)
            if text:
                outputs.append(SessionOutput(id=message.uuid, turn_id=None, text=text))
        return outputs[-_RECENT_OUTPUTS:]

    async def _transcript(self, session_id: str, cwd: str | None) -> list[Any]:
        try:
            return await asyncio.to_thread(get_session_messages, session_id, cwd)
        except OSError as exc:
            raise BackendError(f"cannot read Claude transcript {session_id}: {exc}") from exc

    async def session_busy(self, session_id: str) -> bool | None:
        session = self._sessions.get(session_id)
        return session.busy if session is not None else None

    def _live_summary(self, session: _Session) -> SessionSummary:
        return SessionSummary(
            id=session.id,
            cwd=session.cwd,
            title=session.title,
            updated_at=session.updated_at,
            busy=session.busy,
            active_turn_id=session.turn_id,
            last_reply=session.last_reply,
        )

    def _summary(self, info: Any, cwd: str) -> SessionSummary:
        live = self._sessions.get(info.session_id)
        return SessionSummary(
            id=info.session_id,
            cwd=info.cwd or cwd,
            title=self._title(info),
            updated_at=self._seconds(info.last_modified),
            busy=bool(live and live.busy),
            active_turn_id=live.turn_id if live else None,
        )

    @staticmethod
    def _title(info: Any) -> str:
        raw = info.custom_title or info.summary or info.first_prompt or "untitled"
        return safe_one_line(str(raw), 100)

    @staticmethod
    def _seconds(value: Any) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return time.time()
        timestamp = float(value)
        # SDKSessionInfo reports milliseconds; SessionSummary carries seconds.
        return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------

    async def send_message(self, session_id: str, text: str) -> str | None:
        session = self._require(session_id)
        turn_id = str(uuid.uuid4())
        session.turn_id = turn_id
        session.busy = True
        session.final_text = ""
        session.plan_signature = None
        try:
            await session.client.query(text)
        except (ClaudeSDKError, OSError, TimeoutError) as exc:
            session.busy = False
            session.turn_id = None
            raise BackendError(f"Claude prompt failed: {exc}") from exc
        # Announced only once the prompt is accepted: an announced turn that never
        # starts would leave the channel busy with no completion to clear it.
        await self._events.put(
            BackendEvent(
                kind="turn_started",
                backend=self.name,
                session_id=session_id,
                turn_id=turn_id,
            )
        )
        return turn_id

    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None:
        session = self._require(session_id)
        if not session.busy:
            raise BackendError("Claude has no active turn to steer")
        try:
            # The CLI folds an extra streaming-input message into the running turn.
            await session.client.query(text)
        except (ClaudeSDKError, OSError, TimeoutError) as exc:
            raise BackendError(f"Claude steer failed: {exc}") from exc

    async def cancel(self, session_id: str, turn_id: str | None) -> None:
        session = self._require(session_id)
        if not session.busy:
            raise BackendError("Claude has no active turn to cancel")
        try:
            await session.client.interrupt()
        except (ClaudeSDKError, OSError, TimeoutError) as exc:
            raise BackendError(f"Claude interrupt failed: {exc}") from exc

    async def resolve_approval(self, request_token: str | int, allow: bool) -> None:
        future = self._approvals.pop(str(request_token), None)
        if future is None or future.done():
            raise BackendError("approval was already resolved")
        # No request_resolved event: this is the only path that can answer a Claude
        # approval, and the bridge already emits request.resolved for the client.
        future.set_result(allow)

    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None:
        future = self._questions.pop(str(request_token), None)
        if future is None or future.done():
            raise BackendError("question was already resolved")
        # None (a skip) denies the AskUserQuestion call; the model sees the
        # denial message and continues without an answer.
        future.set_result([list(values) for values in answers] if answers is not None else None)

    async def get_last_reply(self, session_id: str) -> str | None:
        session = self._sessions.get(session_id)
        if session is not None and session.last_reply:
            return session.last_reply
        cwd = session.cwd if session is not None else None
        for message in reversed(await self._transcript(session_id, cwd)):
            if message.type != "assistant":
                continue
            text = self._message_text(message.message)
            if text:
                if session is not None:
                    session.last_reply = text
                return text
        return None

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

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
            raise BackendError("Claude history cursor must be a non-negative integer string")
        session = self._sessions.get(session_id)
        cwd = session.cwd if session else None
        messages = await self._transcript(session_id, cwd)
        end = max(0, len(messages) - offset)
        start = max(0, end - limit)
        # Turn grouping and timestamps are derived from the whole transcript so a
        # turn split across two pages keeps one identity and one ordering.
        turn_ids = self._turn_ids(messages)
        last_of_turn = {turn: index for index, turn in enumerate(turn_ids) if turn}
        times = await self._message_times(session_id, cwd, messages)
        events: list[BackendEvent] = []
        for index in range(start, end):
            # Events derived from one message stay inside its window so pages
            # remain chronologically disjoint whatever the offsets inside are.
            cap = times[index + 1] if index + 1 < len(messages) else times[index] + 1000
            for event in self._history_message(
                session_id,
                messages[index],
                turn_ids[index],
                times[index],
                starts_turn=index == 0 or turn_ids[index] != turn_ids[index - 1],
                ends_turn=(
                    last_of_turn.get(turn_ids[index]) == index
                    and (index < len(messages) - 1 or not (session is not None and session.busy))
                ),
            ):
                event.at = min(event.at or times[index], cap - 1)
                events.append(event)
        return HistoryPage(
            events=tuple(events),
            next_cursor=str(offset + (end - start)) if start > 0 else None,
        )

    def _turn_ids(self, messages: Sequence[Any]) -> list[str | None]:
        """Assign one synthesized turn id per user prompt across the transcript."""
        turn_ids: list[str | None] = []
        current: str | None = None
        for message in messages:
            payload = message.message if isinstance(message.message, dict) else {}
            if message.type == "user" and self._message_text(payload):
                current = self._identifier(message.uuid, "turn")
            turn_ids.append(current)
        return turn_ids

    async def _message_times(
        self,
        session_id: str,
        cwd: str | None,
        messages: Sequence[Any],
    ) -> list[int]:
        """Return one strictly increasing millisecond timestamp per message.

        The session transcript on disk records an ISO timestamp for every entry
        even though ``SessionMessage`` drops it, so real times are read straight
        from the JSONL. Messages without a recorded time fall back to one-second
        spacing from the session's last-modified anchor, and any non-monotonic
        value is clamped forward so pages stay chronologically ordered.
        """
        recorded = await asyncio.to_thread(self._transcript_times, session_id, cwd)
        anchor = await self._history_anchor(session_id, cwd, len(messages))
        times: list[int] = []
        previous: int | None = None
        for index, message in enumerate(messages):
            value = recorded.get(message.uuid, anchor + index * 1000)
            if previous is not None and value <= previous:
                value = previous + 1
            times.append(value)
            previous = value
        return times

    @staticmethod
    def _transcript_times(session_id: str, cwd: str | None) -> dict[str, int]:
        """Map transcript entry uuid to epoch milliseconds from the on-disk JSONL."""
        if not session_id or not all(c.isalnum() or c == "-" for c in session_id):
            return {}
        config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
        candidates = sorted(
            (Path(config_dir) / "projects").glob(f"*/{session_id}.jsonl"),
            key=lambda path: path.name,
        )
        for path in candidates:
            times: dict[str, int] = {}
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(entry, dict):
                            continue
                        identifier = entry.get("uuid")
                        stamp = entry.get("timestamp")
                        if not isinstance(identifier, str) or not isinstance(stamp, str):
                            continue
                        try:
                            moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        times[identifier] = int(moment.timestamp() * 1000)
            except OSError:
                continue
            if times:
                return times
        return {}

    async def _history_anchor(self, session_id: str, cwd: str | None, count: int) -> int:
        """Return the fallback millisecond timestamp for the transcript's start.

        Used only for messages whose on-disk entry lacks a readable timestamp:
        those are spaced one second apart backwards from the session's
        last-modified time. Anchoring on a stored value rather than the clock
        keeps `at` identical across pages.
        """
        try:
            info = await asyncio.to_thread(get_session_info, session_id, cwd)
        except OSError:
            info = None
        latest = (
            info.last_modified
            if info is not None and isinstance(info.last_modified, int)
            else int(time.time() * 1000)
        )
        return latest - max(0, count - 1) * 1000

    def _history_message(
        self,
        session_id: str,
        message: Any,
        turn_id: str | None,
        base: int,
        starts_turn: bool,
        ends_turn: bool,
    ) -> list[BackendEvent]:
        payload = message.message if isinstance(message.message, dict) else {}
        blocks = payload.get("content")
        events: list[BackendEvent] = []
        if message.type == "user":
            prompt = self._message_text(payload)
            if prompt:
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
                        message.uuid,
                        base + 1,
                        "user.prompt",
                        text=prompt,
                    )
                )
            else:
                events.extend(self._history_tool_results(session_id, turn_id, blocks, base))
        else:
            events.extend(self._history_assistant(session_id, turn_id, message, blocks, base))
        if ends_turn and turn_id is not None:
            events.append(
                self._history_event(
                    "turn_done", session_id, turn_id, None, base + 999, "turn.completed"
                )
            )
        return events

    def _history_assistant(
        self,
        session_id: str,
        turn_id: str | None,
        message: Any,
        blocks: Any,
        base: int,
    ) -> list[BackendEvent]:
        events: list[BackendEvent] = []
        items = blocks if isinstance(blocks, list) else []
        tools = [
            item
            for item in items
            if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("id")
        ]
        text = self._message_text({"content": items})
        if text:
            events.append(
                self._history_event(
                    "progress" if tools else "assistant",
                    session_id,
                    turn_id,
                    message.uuid,
                    base,
                    "plan.updated" if tools else "assistant.completed",
                    text=text,
                    data={"plan": False} if tools else {},
                )
            )
        for offset, item in enumerate(tools, 1):
            item_id = str(item["id"])
            if item.get("name") == "TodoWrite":
                summary, data = self._plan_update(item.get("input"))
                events.append(
                    self._history_event(
                        "progress",
                        session_id,
                        turn_id,
                        item_id,
                        base + offset,
                        "plan.updated",
                        text=summary,
                        data={**data, "running": False},
                    )
                )
                continue
            name = str(item.get("name") or "tool")
            events.append(
                self._history_event(
                    "tool_started",
                    session_id,
                    turn_id,
                    item_id,
                    base + offset,
                    "tool.started",
                    tool_kind=self._tool_kind(name),
                    data=self._tool_metadata(name, item.get("input")),
                )
            )
        return events

    def _history_tool_results(
        self,
        session_id: str,
        turn_id: str | None,
        blocks: Any,
        base: int,
    ) -> list[BackendEvent]:
        events: list[BackendEvent] = []
        items = blocks if isinstance(blocks, list) else []
        for offset, item in enumerate(items):
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            item_id = str(item.get("tool_use_id") or "")
            if not item_id:
                continue
            failed = bool(item.get("is_error"))
            data: dict[str, Any] = {"status": "error" if failed else "completed"}
            output = self._result_text(item.get("content"))
            if output:
                data["output"] = truncate_utf8(clean_block(output), _MAX_TOOL_PAYLOAD_BYTES)
            events.append(
                self._history_event(
                    "tool_finished",
                    session_id,
                    turn_id,
                    item_id,
                    base + offset,
                    "tool.completed",
                    success=not failed,
                    data=data,
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
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentwire-claude:{kind}:{seed}"))

    @staticmethod
    def _message_text(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        content = payload.get("content")
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
