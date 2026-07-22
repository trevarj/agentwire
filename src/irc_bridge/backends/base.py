from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence

from irc_bridge.models import BackendEvent, Question, SessionSummary


class BackendError(RuntimeError):
    """A safe, user-presentable backend failure."""


class Backend(abc.ABC):
    name: str

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def wait_ready(self, timeout: float = 30) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[BackendEvent]: ...

    @abc.abstractmethod
    async def list_sessions(self, cwd: str) -> list[SessionSummary]: ...

    @abc.abstractmethod
    async def list_running_sessions(self) -> list[SessionSummary]: ...

    @abc.abstractmethod
    async def create_session(self, cwd: str) -> SessionSummary: ...

    @abc.abstractmethod
    async def attach_session(self, session_id: str, cwd: str | None = None) -> SessionSummary: ...

    @abc.abstractmethod
    async def send_message(self, session_id: str, text: str) -> str | None: ...

    @abc.abstractmethod
    async def steer(self, session_id: str, turn_id: str | None, text: str) -> None: ...

    @abc.abstractmethod
    async def cancel(self, session_id: str, turn_id: str | None) -> None: ...

    @abc.abstractmethod
    async def resolve_approval(self, request_token: str | int, allow: bool) -> None: ...

    @abc.abstractmethod
    async def resolve_question(
        self,
        request_token: str | int,
        questions: Sequence[Question],
        answers: Sequence[Sequence[str]] | None,
    ) -> None: ...

    @abc.abstractmethod
    async def get_last_reply(self, session_id: str) -> str | None: ...
