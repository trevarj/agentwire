from __future__ import annotations

from typing import Any

import pytest

from irc_bridge.backends.opencode import OpenCodeBackend
from irc_bridge.config import OpenCodeConfig
from irc_bridge.models import Question


class StubOpenCode(OpenCodeBackend):
    def __init__(self) -> None:
        super().__init__(
            OpenCodeConfig(
                url="http://127.0.0.1:14096",
                username="opencode",
                password_env="PASSWORD",
                binary="opencode",
            ),
            "password",
        )
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.messages: list[dict[str, Any]] = []

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append((method, path, kwargs))
        if path.endswith("/message"):
            return self.messages
        return None


@pytest.mark.asyncio
async def test_idle_emits_complete_reply_without_tool_payload() -> None:
    backend = StubOpenCode()
    backend.messages = [
        {
            "info": {"id": "msg_1", "role": "assistant"},
            "parts": [
                {"type": "text", "text": "complete reply"},
                {"type": "tool", "state": {"output": "secret output"}},
            ],
        }
    ]
    await backend._handle_global_event(
        {
            "directory": "/workspace",
            "payload": {
                "type": "session.status",
                "properties": {"sessionID": "ses_1", "status": {"type": "busy"}},
            },
        }
    )
    await backend._handle_global_event(
        {
            "directory": "/workspace",
            "payload": {
                "type": "session.idle",
                "properties": {"sessionID": "ses_1"},
            },
        }
    )
    events = [await backend._events.get() for _ in range(3)]
    assert [event.kind for event in events] == ["turn_started", "assistant", "turn_done"]
    assert events[1].text == "complete reply"


@pytest.mark.asyncio
async def test_permission_reply_is_one_shot() -> None:
    backend = StubOpenCode()
    backend._requests["per_1"] = ("v2", "ses_1", ())
    await backend.resolve_approval("per_1", True)
    method, path, kwargs = backend.calls[-1]
    assert method == "POST"
    assert path == "/api/session/ses_1/permission/per_1/reply"
    assert kwargs["body"] == {"reply": "once"}


@pytest.mark.asyncio
async def test_normal_prompt_uses_stable_async_session_route() -> None:
    backend = StubOpenCode()
    backend._session_cwds["ses_1"] = "/workspace"
    await backend.send_message("ses_1", "hello")
    method, path, kwargs = backend.calls[-1]
    assert method == "POST"
    assert path == "/session/ses_1/prompt_async"
    assert kwargs["params"] == {"directory": "/workspace"}
    assert kwargs["body"] == {"parts": [{"type": "text", "text": "hello"}]}
    assert kwargs["expected"] == (204,)


@pytest.mark.asyncio
async def test_question_reject_uses_session_owned_v2_route() -> None:
    backend = StubOpenCode()
    question = Question(id="1", header="Choice", prompt="Choose")
    backend._requests["que_1"] = ("v2", "ses_1", (question,))
    await backend.resolve_question("que_1", (question,), None)
    method, path, _kwargs = backend.calls[-1]
    assert method == "POST"
    assert path == "/api/session/ses_1/question/que_1/reject"
