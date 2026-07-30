from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal[
    "connected",
    "disconnected",
    "turn_started",
    "turn_done",
    "turn_failed",
    "status_changed",
    "progress",
    "assistant",
    "tool_started",
    "tool_finished",
    "approval",
    "question",
    "request_resolved",
]


@dataclass(slots=True, frozen=True)
class SessionOutput:
    id: str
    turn_id: str | None
    text: str
    phase: str | None = None


@dataclass(slots=True, frozen=True)
class SessionActivity:
    kind: Literal["tool_started", "tool_finished"]
    item_id: str
    turn_id: str | None
    tool_kind: str
    success: bool | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SessionSummary:
    id: str
    cwd: str
    title: str
    updated_at: float = 0
    busy: bool = False
    active_flags: tuple[str, ...] = ()
    tui_attached: bool = False
    active_turn_id: str | None = None
    last_output: str | None = None
    last_reply: str | None = None
    recent_outputs: tuple[SessionOutput, ...] = ()
    recent_activity: tuple[SessionActivity, ...] = ()


@dataclass(slots=True, frozen=True)
class Question:
    id: str
    header: str
    prompt: str
    options: tuple[str, ...] = ()
    multiple: bool = False
    custom: bool = True
    secret: bool = False


@dataclass(slots=True)
class BackendEvent:
    kind: EventKind
    backend: str
    session_id: str | None = None
    turn_id: str | None = None
    item_id: str | None = None
    request_token: str | int | None = None
    text: str = ""
    tool_kind: str = "tool"
    success: bool | None = None
    questions: tuple[Question, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChannelBinding:
    backend: str
    session_id: str
    cwd: str
