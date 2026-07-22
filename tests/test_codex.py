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


@pytest.mark.asyncio
async def test_completed_commentary_is_relayed_as_progress() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "item-1",
                "type": "agentMessage",
                "phase": "commentary",
                "text": "I found the cause and am checking the fix.",
            },
        },
    )
    event = await backend._events.get()
    assert event.kind == "progress"
    assert event.text == "I found the cause and am checking the fix."


@pytest.mark.asyncio
async def test_thread_status_change_is_relayed() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "thread/status/changed",
        {"threadId": "thread-1", "status": {"type": "active", "activeFlags": []}},
    )
    event = await backend._events.get()
    assert event.kind == "status_changed"
    assert event.data == {"busy": True, "active_flags": ()}


@pytest.mark.asyncio
async def test_plan_update_relays_current_step() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "turn/plan/updated",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "plan": [
                {"step": "Inspect the protocol", "status": "completed"},
                {"step": "Fix status tracking", "status": "inProgress"},
            ],
        },
    )
    event = await backend._events.get()
    assert event.kind == "progress"
    assert event.text == "plan: Fix status tracking"


@pytest.mark.asyncio
async def test_attach_recovers_active_turn() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(method: str, params: dict[str, object]) -> object:
        assert method == "thread/resume"
        assert params["initialTurnsPage"] == {
            "limit": 1,
            "sortDirection": "desc",
            "itemsView": "full",
        }
        return {
            "thread": {
                "id": "thread-1",
                "cwd": "/workspace",
                "status": {
                    "type": "active",
                    "activeFlags": ["waitingOnUserInput"],
                },
            },
            "initialTurnsPage": {
                "data": [
                    {
                        "id": "turn-1",
                        "status": "inProgress",
                        "items": [
                            {
                                "id": "item-1",
                                "type": "agentMessage",
                                "phase": "commentary",
                                "text": "Checking the remaining protocol events.",
                            }
                        ],
                    }
                ]
            },
        }

    backend._request = request  # type: ignore[method-assign]
    summary = await backend.attach_session("thread-1", "/workspace")
    assert summary.busy
    assert summary.active_flags == ("waitingOnUserInput",)
    assert summary.active_turn_id == "turn-1"
    assert summary.last_output == "Checking the remaining protocol events."
    assert summary.last_reply is None
    assert backend._active_turns == {"thread-1": "turn-1"}


def test_session_summary_includes_current_thread_status() -> None:
    summary = CodexBackend._summary(
        {
            "id": "thread-1",
            "cwd": "/workspace",
            "status": {"type": "active", "activeFlags": []},
        }
    )
    assert summary.busy
