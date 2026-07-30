from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentwire.backends.codex import CodexBackend
from agentwire.config import CodexConfig


@pytest.mark.asyncio
async def test_setting_options_are_sourced_from_paginated_model_catalog() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    calls: list[tuple[str, dict[str, object]]] = []

    async def request(method: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((method, params))
        if len(calls) == 1:
            return {
                "data": [{
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "displayName": "GPT-5.6 Sol",
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium", "description": "Fast"},
                        {"reasoningEffort": "high", "description": "Deep"},
                    ],
                }],
                "nextCursor": "next",
            }
        return {"data": [], "nextCursor": None}

    backend._request = request  # type: ignore[method-assign]
    options = await backend.setting_options()

    assert calls == [
        ("model/list", {"limit": 100, "includeHidden": False}),
        ("model/list", {"limit": 100, "includeHidden": False, "cursor": "next"}),
    ]
    assert options == {"model": [{
        "value": "gpt-5.6-sol",
        "label": "GPT-5.6 Sol",
        "efforts": ["medium", "high"],
        "defaultEffort": "high",
        "default": True,
    }]}
    assert await backend.setting_options() is options


@pytest.mark.asyncio
async def test_protocol_settings_map_auto_review_without_disabling_approvals() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    calls: list[tuple[str, dict[str, object]]] = []

    async def request(method: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((method, params))
        return {"turn": {"id": "turn-1"}}

    backend._request = request  # type: ignore[method-assign]
    await backend.configure_session(
        "thread-1",
        {
            "model": "gpt-5.6",
            "effort": "high",
            "collaboration": "default",
            "approvalReviewer": "auto_review",
            "delivery": "queue",
        },
    )
    await backend.send_message("thread-1", "hello")
    assert calls[0][0] == "thread/settings/update"
    assert calls[1][0] == "turn/start"
    params = calls[1][1]
    assert params["approvalsReviewer"] == "auto_review"
    assert params["effort"] == "high"
    assert params["collaborationMode"] == {
        "mode": "default",
        "settings": {
            "model": "gpt-5.6",
            "reasoning_effort": "high",
            "developer_instructions": None,
        },
    }
    assert "approvalPolicy" not in params


def _fake_process(proc_root: Path, pid: int, cwd: Path, *arguments: str) -> None:
    process = proc_root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in arguments) + b"\0")
    os.symlink(cwd, process / "cwd")


def test_live_codex_processes_find_new_and_resumed_sessions(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _fake_process(proc_root, 10, first, "/bin/codex")
    _fake_process(proc_root, 11, first, "/bin/codex")
    _fake_process(proc_root, 12, second, "/bin/codex", "resume", "thread-12")
    _fake_process(proc_root, 13, second, "/bin/codex", "app-server")
    _fake_process(proc_root, 14, second, "/bin/not-codex")

    explicit, workspaces = CodexBackend._live_codex_processes(proc_root)

    assert explicit == {"thread-12"}
    assert workspaces == {str(first): 2}


def test_rollout_busy_tracks_latest_task_boundary(tmp_path: Path) -> None:
    session_id = "019f887c-9632-7e61-a50c-26d4125854ce"
    rollout = tmp_path / f"rollout-{session_id}.jsonl"
    rollout.write_text(
        '{"type":"event_msg","payload":{"type":"task_complete"}}\n'
        '{"type":"event_msg","payload":{"type":"task_started"}}\n',
        encoding="utf-8",
    )
    assert CodexBackend._rollout_busy(session_id, tmp_path) is True
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"event_msg","payload":{"type":"task_complete"}}\n')
    assert CodexBackend._rollout_busy(session_id, tmp_path) is False


def test_rollout_busy_rejects_unsafe_session_id(tmp_path: Path) -> None:
    assert CodexBackend._rollout_busy("../thread", tmp_path) is None


@pytest.mark.asyncio
async def test_tool_event_extracts_safe_display_metadata() -> None:
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
    assert event.data == {"label": "$ cat /secret/file", "input": "cat /secret/file"}


@pytest.mark.asyncio
async def test_completed_command_includes_small_output_metadata() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "item-1",
                "type": "commandExecution",
                "command": "git status --short",
                "aggregatedOutput": " M src/agentwire/bridge.py",
                "status": "completed",
                "exitCode": 0,
            },
        },
    )

    event = await backend._events.get()
    assert event.kind == "tool_finished"
    assert event.data == {
        "status": "completed",
        "exitCode": 0,
        "label": "$ git status --short",
        "input": "git status --short",
        "output": " M src/agentwire/bridge.py",
    }


@pytest.mark.asyncio
async def test_completed_file_change_includes_diff_metadata() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "item-1",
                "type": "fileChange",
                "status": "completed",
                "changes": [
                    {
                        "path": "src/agentwire/bridge.py",
                        "kind": "update",
                        "diff": "@@ -1 +1 @@\n-old\n+new",
                    }
                ],
            },
        },
    )

    event = await backend._events.get()
    assert event.kind == "tool_finished"
    assert event.data == {
        "status": "completed",
        "label": "Edit src/agentwire/bridge.py",
        "diff": (
            "diff --git a/src/agentwire/bridge.py b/src/agentwire/bridge.py\n"
            "--- a/src/agentwire/bridge.py\n"
            "+++ b/src/agentwire/bridge.py\n"
            "@@ -1 +1 @@\n-old\n+new"
        ),
    }


def test_file_change_does_not_duplicate_existing_unified_headers() -> None:
    diff = CodexBackend._git_diff(
        {
            "path": "changed.py",
            "kind": "update",
            "diff": "--- a/changed.py\n+++ b/changed.py\n@@ -1 +1 @@\n-old\n+new",
        }
    )

    assert diff.count("--- a/changed.py") == 1
    assert diff.startswith("diff --git a/changed.py b/changed.py\n")


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
async def test_resolved_request_recovers_missing_thread_context() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    backend._server_requests[7] = (
        "item/commandExecution/requestApproval",
        {"threadId": "thread-1"},
    )

    await backend._handle_notification("serverRequest/resolved", {"requestId": 7})

    event = await backend._events.get()
    assert event.kind == "request_resolved"
    assert event.session_id == "thread-1"
    assert event.request_token == 7
    assert 7 not in backend._server_requests


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
    assert event.data == {
        "plan": True,
        "running": True,
        "status": "inProgress",
        "completedSteps": 1,
        "totalSteps": 2,
    }
    await backend._handle_notification(
        "turn/plan/updated",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "plan": [{"step": "Fix status tracking", "status": "inProgress"}],
        },
    )
    second = await backend._events.get()
    assert second.data["completedSteps"] == 0
    assert second.data["totalSteps"] == 1
    await backend._handle_notification(
        "turn/plan/updated",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "plan": [{"step": "Fix status tracking", "status": "inProgress"}],
        },
    )
    assert backend._events.empty()

    await backend._handle_notification(
        "turn/plan/updated",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "plan": [{"step": "Fix status tracking", "status": "completed"}],
        },
    )
    completed = await backend._events.get()
    assert completed.text == "Plan completed"
    assert completed.data == {
        "plan": True,
        "running": False,
        "status": "completed",
        "completedSteps": 1,
        "totalSteps": 1,
    }


@pytest.mark.asyncio
async def test_attach_recovers_active_turn() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(method: str, params: dict[str, object]) -> object:
        assert method == "thread/resume"
        assert params["initialTurnsPage"] == {
            "limit": 3,
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
    assert [output.text for output in summary.recent_outputs] == [
        "Checking the remaining protocol events."
    ]
    assert backend._active_turns == {"thread-1": "turn-1"}


@pytest.mark.asyncio
async def test_attach_restores_the_last_three_outputs_in_chronological_order() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(_method: str, _params: dict[str, object]) -> object:
        return {
            "thread": {"id": "thread-1", "cwd": "/workspace", "status": {"type": "idle"}},
            "initialTurnsPage": {
                "data": [
                    {
                        "id": "newest",
                        "status": "completed",
                        "items": [{"id": "i4", "type": "agentMessage", "text": "Newest"}],
                    },
                    {
                        "id": "middle",
                        "status": "completed",
                        "items": [{"id": "i3", "type": "agentMessage", "text": "Middle"}],
                    },
                    {
                        "id": "oldest",
                        "status": "completed",
                        "items": [
                            {"id": "i1", "type": "agentMessage", "text": "Old one"},
                            {"id": "i2", "type": "agentMessage", "text": "Old two"},
                        ],
                    },
                ]
            },
        }

    backend._request = request  # type: ignore[method-assign]
    summary = await backend.attach_session("thread-1", "/workspace")

    assert [output.text for output in summary.recent_outputs] == ["Old two", "Middle", "Newest"]


def test_session_summary_includes_current_thread_status() -> None:
    summary = CodexBackend._summary(
        {
            "id": "thread-1",
            "cwd": "/workspace",
            "status": {"type": "active", "activeFlags": []},
        }
    )
    assert summary.busy


@pytest.mark.asyncio
async def test_running_sessions_paginates_and_filters_active_threads() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    calls: list[dict[str, object]] = []

    async def request(method: str, params: dict[str, object]) -> object:
        assert method == "thread/list"
        calls.append(params)
        if "cursor" not in params:
            return {
                "data": [
                    {
                        "id": "active-1",
                        "cwd": "/workspace/one",
                        "status": {"type": "active", "activeFlags": []},
                    },
                    {
                        "id": "idle-1",
                        "cwd": "/workspace/one",
                        "status": {"type": "idle"},
                    },
                ],
                "nextCursor": "page-2",
            }
        return {
            "data": [
                {
                    "id": "active-2",
                    "cwd": "/workspace/two",
                    "status": {"type": "active", "activeFlags": []},
                }
            ]
        }

    backend._request = request  # type: ignore[method-assign]
    sessions = await backend.list_running_sessions()
    assert [session.id for session in sessions] == ["active-1", "active-2"]
    assert calls[1]["cursor"] == "page-2"


@pytest.mark.asyncio
async def test_running_sessions_include_threads_with_live_codex_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(method: str, params: dict[str, object]) -> object:
        assert method == "thread/list"
        return {
            "data": [
                {"id": "newest", "cwd": "/workspace", "status": {"type": "idle"}},
                {"id": "resumed", "cwd": "/other", "status": {"type": "idle"}},
                {"id": "old", "cwd": "/workspace", "status": {"type": "idle"}},
            ]
        }

    monkeypatch.setattr(
        CodexBackend,
        "_live_codex_processes",
        staticmethod(lambda: ({"resumed"}, {"/workspace": 1})),
    )
    backend._request = request  # type: ignore[method-assign]

    sessions = await backend.list_running_sessions()

    assert [session.id for session in sessions] == ["newest", "resumed"]
