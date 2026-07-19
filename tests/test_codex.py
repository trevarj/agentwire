from __future__ import annotations

from pathlib import Path

import pytest

from irc_bridge.backends.codex import CodexBackend
from irc_bridge.config import CodexConfig


@pytest.mark.asyncio
async def test_tool_event_is_sanitized() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "item/started",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "item-1",
                "type": "commandExecution",
                "command": "cat /secret/file",
            },
        },
    )
    event = await backend._events.get()
    assert event.kind == "tool_started"
    assert event.tool_kind == "shell"
    assert event.text == ""
    assert event.data == {}


@pytest.mark.asyncio
async def test_approval_omits_reason_and_command() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_server_request(
        7,
        "item/commandExecution/requestApproval",
        {
            "threadId": "thread-1",
            "command": "cat /secret/file",
            "reason": "contains a private path",
        },
    )
    event = await backend._events.get()
    assert event.text == "shell command approval needed"
