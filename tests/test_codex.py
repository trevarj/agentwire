from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentwire.backends.base import BackendError
from agentwire.backends.codex import CodexBackend, CodexTuiSessionPresence, _CodexRPCError
from agentwire.config import CodexConfig
from agentwire.stack import _CodexTuiRelayTracker


@pytest.mark.asyncio
async def test_overloaded_requests_retry_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    attempts = 0
    delays: list[float] = []

    async def request_once(method: str, params: object) -> object:
        nonlocal attempts
        attempts += 1
        assert method == "thread/list"
        assert params == {"limit": 20}
        if attempts < 3:
            raise _CodexRPCError("Server overloaded; retry later.", -32001)
        return {"data": []}

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(backend, "_request_once", request_once)
    monkeypatch.setattr("agentwire.backends.codex.random.uniform", lambda _a, _b: 0.05)
    monkeypatch.setattr("agentwire.backends.codex.asyncio.sleep", sleep)

    assert await backend._request("thread/list", {"limit": 20}) == {"data": []}
    assert attempts == 3
    assert delays == pytest.approx([0.15, 0.25])


@pytest.mark.asyncio
async def test_non_overload_rpc_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    attempts = 0

    async def request_once(_method: str, _params: object) -> object:
        nonlocal attempts
        attempts += 1
        raise _CodexRPCError("invalid request", -32600)

    monkeypatch.setattr(backend, "_request_once", request_once)

    with pytest.raises(BackendError, match="invalid request"):
        await backend._request("thread/list")
    assert attempts == 1


@pytest.mark.asyncio
async def test_reader_disconnect_fails_pending_requests_immediately() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    pending = asyncio.get_running_loop().create_future()
    backend._pending[1] = pending

    class ClosedWebSocket:
        def __aiter__(self) -> ClosedWebSocket:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

    backend._ws = ClosedWebSocket()  # type: ignore[assignment]
    await backend._reader()

    with pytest.raises(BackendError, match="disconnected"):
        await pending
    event = await backend._events.get()
    assert event.kind == "disconnected"


@pytest.mark.asyncio
async def test_request_reconnects_when_reader_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    stale_reader = asyncio.create_task(asyncio.sleep(0))
    await stale_reader
    backend._reader_task = stale_reader

    class StaleWebSocket:
        closed = False

    backend._ws = StaleWebSocket()  # type: ignore[assignment]
    reconnected = False

    async def start() -> None:
        nonlocal reconnected
        reconnected = True

    async def send(message: dict[str, object]) -> None:
        await backend._handle_message({"id": message["id"], "result": {"data": []}})

    monkeypatch.setattr(backend, "start", start)
    monkeypatch.setattr(backend, "_send", send)

    assert await backend._request_once("thread/list") == {"data": []}
    assert reconnected


@pytest.mark.asyncio
async def test_send_failure_does_not_leak_pending_request(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    class OpenWebSocket:
        closed = False

    backend._ws = OpenWebSocket()  # type: ignore[assignment]
    backend._reader_task = asyncio.create_task(asyncio.sleep(30))

    async def send(_message: dict[str, object]) -> None:
        raise BackendError("send failed")

    monkeypatch.setattr(backend, "_send", send)

    try:
        with pytest.raises(BackendError, match="send failed"):
            await backend._request_once("thread/list")
        assert backend._pending == {}
    finally:
        backend._reader_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await backend._reader_task


@pytest.mark.asyncio
async def test_setting_options_are_sourced_from_paginated_model_catalog() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    calls: list[tuple[str, dict[str, object]]] = []

    async def request(method: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((method, params))
        if len(calls) == 1:
            return {
                "data": [
                    {
                        "id": "gpt-5.6-sol",
                        "model": "gpt-5.6-sol",
                        "displayName": "GPT-5.6 Sol",
                        "isDefault": True,
                        "defaultReasoningEffort": "high",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "medium", "description": "Fast"},
                            {"reasoningEffort": "high", "description": "Deep"},
                        ],
                    }
                ],
                "nextCursor": "next",
            }
        return {"data": [], "nextCursor": None}

    backend._request = request  # type: ignore[method-assign]
    options = await backend.setting_options()

    assert calls == [
        ("model/list", {"limit": 100, "includeHidden": False}),
        ("model/list", {"limit": 100, "includeHidden": False, "cursor": "next"}),
    ]
    assert options == {
        "model": [
            {
                "value": "gpt-5.6-sol",
                "label": "GPT-5.6 Sol",
                "efforts": ["medium", "high"],
                "defaultEffort": "high",
                "default": True,
            }
        ]
    }
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


def _fake_process(proc_root: Path, pid: int, *arguments: str) -> None:
    process = proc_root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in arguments) + b"\0")


def _fake_process_stat(proc_root: Path, pid: int, start_time: str) -> None:
    process = proc_root / str(pid)
    process.mkdir(exist_ok=True)
    fields = ["S", *("0" for _ in range(18)), start_time]
    (process / "stat").write_text(f"{pid} (agentwire) {' '.join(fields)}\n", encoding="utf-8")


def _fake_open_rollout(
    proc_root: Path,
    pid: int,
    rollout: Path,
    session_id: str,
    source: object = "cli",
) -> None:
    rollout.write_text(
        '{"type":"session_meta","payload":'
        f'{{"id":"{session_id}","originator":"codex-tui","source":{source!r}}}}}\n'.replace(
            "'", '"'
        ),
        encoding="utf-8",
    )
    descriptors = proc_root / str(pid) / "fd"
    descriptors.mkdir()
    (descriptors / "7").symlink_to(rollout)


def test_explicit_codex_sessions_only_accept_resumed_session_ids(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _fake_process(proc_root, 10, "/bin/codex")
    _fake_process(proc_root, 12, "/bin/codex", "resume", "thread-12")
    _fake_process(proc_root, 13, "/bin/codex", "app-server")
    _fake_process(proc_root, 14, "/bin/not-codex", "resume", "thread-14")

    explicit = CodexBackend._explicit_codex_sessions(proc_root)

    assert explicit == {"thread-12"}


def test_explicit_codex_sessions_reads_top_level_rollout_from_plain_tui(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _fake_process(proc_root, 10, "/bin/codex")
    root = tmp_path / "root.jsonl"
    guardian = tmp_path / "guardian.jsonl"
    _fake_open_rollout(proc_root, 10, root, "thread-root")
    guardian.write_text(
        '{"type":"session_meta","payload":{"id":"thread-guardian",'
        '"originator":"codex-tui","source":{"subagent":{"thread_spawn":{"parent_thread_id":'
        '"thread-root"}}}}}\n',
        encoding="utf-8",
    )
    (proc_root / "10" / "fd" / "8").symlink_to(guardian)

    assert CodexBackend._explicit_codex_sessions(proc_root) == {"thread-root"}


def test_tui_presence_tracks_exact_session_and_removes_stale_process(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _fake_process_stat(proc_root, 42, "100")
    socket_path = tmp_path / "runtime" / "codex.sock"
    presence = CodexTuiSessionPresence(socket_path, pid=42, proc_root=proc_root)

    presence.update("thread-42")
    assert CodexTuiSessionPresence.sessions(socket_path, proc_root) == {"thread-42"}

    _fake_process_stat(proc_root, 42, "101")
    assert CodexTuiSessionPresence.sessions(socket_path, proc_root) == set()
    assert not presence.path.exists()


def test_tui_relay_tracks_thread_switches_and_disconnect() -> None:
    updates: list[str | None] = []

    class Presence:
        def update(self, session_id: str | None) -> None:
            updates.append(session_id)

        def clear(self) -> None:
            updates.append(None)

    tracker = _CodexTuiRelayTracker(Presence())  # type: ignore[arg-type]
    tracker.client_message('{"id":7,"method":"thread/resume","params":{"threadId":"thread-old"}}')
    tracker.server_message('{"id":7,"result":{"thread":{"id":"thread-current"}}}')
    assert tracker.current_session == "thread-current"

    tracker.client_message(
        '{"id":8,"method":"thread/unsubscribe","params":{"threadId":"thread-current"}}'
    )
    tracker.server_message('{"id":8,"result":{"status":"unsubscribed"}}')
    tracker.server_message('{"method":"thread/started","params":{"thread":{"id":"thread-next"}}}')
    tracker.close()

    assert updates == ["thread-current", None, "thread-next", None]


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
async def test_live_user_messages_relay_tui_prompts_and_suppress_bridge_echoes() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(method: str, params: dict[str, object]) -> object:
        text = params["input"][0]["text"]  # type: ignore[index]
        await backend._handle_notification(
            "item/started",
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "id": f"bridge-{method}",
                    "type": "userMessage",
                    "content": [{"type": "text", "text": text}],
                },
            },
        )
        return {"turn": {"id": "turn-1"}}

    backend._request = request  # type: ignore[method-assign]
    await backend.send_message("thread-1", "bridge prompt")
    await backend.steer("thread-1", "turn-1", "bridge steer")
    assert backend._events.empty()

    await backend._handle_notification(
        "item/started",
        {
            "threadId": "thread-1",
            "turnId": "turn-2",
            "item": {
                "id": "tui-1",
                "type": "userMessage",
                "content": [{"type": "text", "text": "a real TUI prompt"}],
            },
        },
    )
    event = await backend._events.get()
    assert (event.kind, event.turn_id, event.text) == (
        "user_prompt",
        "turn-2",
        "a real TUI prompt",
    )

    # Item lifecycle completion repeats the same item but must not duplicate IRC text.
    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-2",
            "item": {
                "id": "tui-1",
                "type": "userMessage",
                "content": [{"type": "text", "text": "a real TUI prompt"}],
            },
        },
    )
    assert backend._events.empty()


@pytest.mark.asyncio
async def test_rejected_prompt_does_not_suppress_identical_tui_message() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def reject(_method: str, _params: dict[str, object]) -> object:
        raise BackendError("rejected")

    backend._request = reject  # type: ignore[method-assign]
    with pytest.raises(BackendError, match="rejected"):
        await backend.send_message("thread-1", "same text")

    await backend._handle_notification(
        "item/started",
        {
            "threadId": "thread-1",
            "turnId": "turn-tui",
            "item": {
                "id": "tui-1",
                "type": "userMessage",
                "content": [{"type": "text", "text": "same text"}],
            },
        },
    )
    assert (await backend._events.get()).kind == "user_prompt"


@pytest.mark.asyncio
async def test_mcp_and_dynamic_tool_events_include_bounded_arguments_and_text_output() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "mcp-1",
                "type": "mcpToolCall",
                "server": "github",
                "tool": "search_issues",
                "arguments": {"query": "agentwire"},
                "result": {"content": [{"type": "text", "text": "Found 2 issues"}]},
                "status": "completed",
            },
        },
    )
    mcp = await backend._events.get()
    assert mcp.data == {
        "status": "completed",
        "label": "github / search_issues",
        "input": '{"query":"agentwire"}',
        "output": "Found 2 issues",
    }

    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {
                "id": "dynamic-1",
                "type": "dynamicToolCall",
                "namespace": "functions",
                "tool": "search",
                "arguments": {"q": "schema"},
                "contentItems": [{"type": "inputText", "text": "one result"}],
                "success": True,
                "status": "completed",
            },
        },
    )
    dynamic = await backend._events.get()
    assert dynamic.data == {
        "status": "completed",
        "label": "functions / search",
        "input": '{"q":"schema"}',
        "output": "one result",
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
                        "kind": {"type": "update"},
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


def test_added_file_content_becomes_a_unified_diff() -> None:
    diff = CodexBackend._git_diff(
        {
            "path": "/tmp/hello-world.sh",
            "kind": {"type": "add"},
            "diff": '#!/bin/sh\n\necho "hello world"\n',
        }
    )

    assert diff == (
        "diff --git a/tmp/hello-world.sh b/tmp/hello-world.sh\n"
        "--- /dev/null\n"
        "+++ b/tmp/hello-world.sh\n"
        "@@ -0,0 +1,3 @@\n"
        "+#!/bin/sh\n"
        "+\n"
        '+echo "hello world"'
    )


def test_deleted_file_content_becomes_a_unified_diff() -> None:
    diff = CodexBackend._git_diff(
        {
            "path": "obsolete.txt",
            "kind": {"type": "delete"},
            "diff": "first\nsecond",
        }
    )

    assert diff == (
        "diff --git a/obsolete.txt b/obsolete.txt\n"
        "--- a/obsolete.txt\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-first\n"
        "-second\n"
        "\\ No newline at end of file"
    )


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
async def test_collaboration_items_relay_normalized_subagent_statuses() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    await backend._handle_notification(
        "item/started",
        {
            "threadId": "parent",
            "turnId": "turn-1",
            "item": {
                "id": "collab-1",
                "type": "collabAgentToolCall",
                "tool": "spawnAgent",
                "model": "test-model",
                "prompt": "map the repository",
                "receiverThreadIds": ["child-a", "child-b", "child-c"],
                "agentsStates": {
                    "child-a": {"status": "pendingInit"},
                    "child-b": {"status": "running", "message": "reading tests"},
                },
                "status": "inProgress",
            },
        },
    )
    update = await backend._events.get()
    assert update.kind == "subagent_update"
    assert update.data == {
        "agents": [
            {
                "id": "child-a",
                "type": "test-model",
                "description": "map the repository",
                "status": "queued",
                "isBackground": True,
            },
            {
                "id": "child-b",
                "type": "test-model",
                "description": "reading tests",
                "status": "running",
                "isBackground": True,
            },
            {
                "id": "child-c",
                "type": "test-model",
                "description": "map the repository",
                "status": "queued",
                "isBackground": True,
            },
        ]
    }
    # The original collaboration tool event remains available to existing clients.
    assert (await backend._events.get()).kind == "tool_started"

    await backend._handle_notification(
        "item/completed",
        {
            "threadId": "parent",
            "turnId": "turn-1",
            "item": {
                "id": "activity-1",
                "type": "subAgentActivity",
                "agentThreadId": "child-b",
                "agentPath": "agents/review",
                "kind": "completed",
            },
        },
    )
    complete = await backend._events.get()
    assert complete.kind == "subagent_update"
    assert complete.data["agents"][1]["status"] == "completed"
    assert complete.data["agents"][1]["description"] == "agents/review"
    assert complete.data["agents"][1]["type"] == "test-model"


@pytest.mark.asyncio
async def test_attach_hydrates_subagents_without_resuming_child_threads() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    calls: list[tuple[str, dict[str, object]]] = []

    async def request(method: str, params: dict[str, object]) -> object:
        calls.append((method, params))
        assert method == "thread/resume"
        return {
            "thread": {"id": "parent", "cwd": "/workspace", "status": {"type": "idle"}},
            "initialTurnsPage": {
                "data": [
                    {
                        "id": "turn-new",
                        "status": "completed",
                        "items": [
                            {
                                "id": "collab-new",
                                "type": "collabAgentToolCall",
                                "tool": "spawnAgent",
                                "receiverThreadIds": ["child-a"],
                                "agentsStates": {"child-a": {"status": "completed"}},
                                "status": "completed",
                            }
                        ],
                    },
                    {
                        "id": "turn-old",
                        "status": "completed",
                        "items": [
                            {
                                "id": "collab-old",
                                "type": "collabAgentToolCall",
                                "tool": "spawnAgent",
                                "receiverThreadIds": ["child-a"],
                                "agentsStates": {"child-a": {"status": "running"}},
                                "status": "inProgress",
                            }
                        ],
                    },
                ]
            },
        }

    backend._request = request  # type: ignore[method-assign]
    await backend.attach_session("parent", "/workspace")
    event = await backend._events.get()
    assert event.kind == "subagent_update"
    assert event.data["agents"][0]["id"] == "child-a"
    assert event.data["agents"][0]["status"] == "completed"
    assert [method for method, _ in calls] == ["thread/resume"]


@pytest.mark.asyncio
async def test_idle_attach_clears_stale_active_turn_after_reconnect() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    backend._active_turns["thread-1"] = "stale-turn"

    async def request(method: str, _params: dict[str, object]) -> object:
        assert method == "thread/resume"
        return {
            "thread": {"id": "thread-1", "cwd": "/workspace", "status": {"type": "idle"}},
            "initialTurnsPage": {"data": []},
        }

    backend._request = request  # type: ignore[method-assign]
    summary = await backend.attach_session("thread-1", "/workspace")
    assert summary.active_turn_id is None
    assert "thread-1" not in backend._active_turns


@pytest.mark.asyncio
async def test_close_bridge_owned_thread_interrupts_then_unsubscribes_and_preserves_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    backend._created_threads.add("thread-1")
    backend._active_turns["thread-1"] = "turn-1"
    monkeypatch.setattr(backend, "_tui_session_ids", lambda: set())
    calls: list[tuple[str, dict[str, object]]] = []

    async def request(method: str, params: dict[str, object]) -> object:
        calls.append((method, params))
        if method == "thread/unsubscribe":
            return {"status": "unsubscribed"}
        return {}

    backend._request = request  # type: ignore[method-assign]
    await backend.close_session("thread-1")
    assert calls == [
        ("turn/interrupt", {"threadId": "thread-1", "turnId": "turn-1"}),
        ("thread/unsubscribe", {"threadId": "thread-1"}),
    ]
    assert "thread-1" not in backend._created_threads
    assert "thread-1" not in backend._active_turns

    backend._created_threads.add("thread-1")
    backend._active_turns["thread-1"] = "turn-2"
    failed_calls: list[str] = []

    async def unavailable(method: str, _params: dict[str, object]) -> object:
        failed_calls.append(method)
        if method == "thread/unsubscribe":
            raise BackendError("unavailable")
        return {}

    backend._request = unavailable  # type: ignore[method-assign]
    with pytest.raises(BackendError, match="unavailable"):
        await backend.close_session("thread-1")
    assert "thread-1" in backend._created_threads
    assert "thread-1" not in backend._active_turns
    assert failed_calls == ["turn/interrupt", "thread/unsubscribe"]

    async def retry(method: str, _params: dict[str, object]) -> object:
        assert method == "thread/unsubscribe"
        return {"status": "unsubscribed"}

    backend._request = retry  # type: ignore[method-assign]
    await backend.close_session("thread-1")
    assert "thread-1" not in backend._created_threads

    backend._created_threads.add("thread-1")
    backend._active_turns["thread-1"] = "turn-old"

    async def new_turn(method: str, _params: dict[str, object]) -> object:
        if method == "turn/interrupt":
            backend._active_turns["thread-1"] = "turn-new"
            return {}
        pytest.fail("a newer turn must prevent unsubscribe")

    backend._request = new_turn  # type: ignore[method-assign]
    with pytest.raises(BackendError, match="started a new turn"):
        await backend.close_session("thread-1")
    assert "thread-1" in backend._created_threads
    assert backend._active_turns["thread-1"] == "turn-new"

    monkeypatch.setattr(backend, "_tui_session_ids", lambda: {"thread-1"})
    with pytest.raises(BackendError, match="attached Codex TUI"):
        await backend.close_session("thread-1")


@pytest.mark.asyncio
async def test_successful_unsubscribe_finishes_close_despite_external_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    backend._created_threads.add("thread-1")
    monkeypatch.setattr(backend, "_tui_session_ids", lambda: set())

    async def request(method: str, _params: dict[str, object]) -> object:
        assert method == "thread/unsubscribe"
        await backend._handle_notification(
            "turn/started", {"threadId": "thread-1", "turn": {"id": "external-turn"}}
        )
        return {"status": "unsubscribed"}

    backend._request = request  # type: ignore[method-assign]
    await backend.close_session("thread-1")
    assert "thread-1" not in backend._created_threads
    assert "thread-1" not in backend._active_turns


@pytest.mark.asyncio
async def test_completed_older_turn_does_not_clear_a_newer_active_turn() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    backend._active_turns["thread-1"] = "turn-new"
    await backend._handle_notification(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {"id": "turn-old", "status": "completed"},
        },
    )
    assert backend._active_turns["thread-1"] == "turn-new"


@pytest.mark.asyncio
async def test_attach_recovers_active_turn() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(method: str, params: dict[str, object]) -> object:
        if method == "thread/items/list":
            assert params == {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "limit": 50,
                "sortDirection": "desc",
            }
            return {
                "data": [
                    {
                        "turnId": "turn-1",
                        "item": {
                            "id": "tool-1",
                            "type": "commandExecution",
                            "command": "git status --short",
                            "status": "inProgress",
                        },
                    }
                ]
            }
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
    assert [(item.kind, item.item_id, item.tool_kind) for item in summary.recent_activity] == [
        ("tool_started", "tool-1", "shell")
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
                    {
                        "id": "subagent",
                        "cwd": "/workspace/one",
                        "parentThreadId": "active-1",
                        "status": {"type": "active"},
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
async def test_history_reads_exact_thread_and_omits_reasoning() -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))
    calls: list[tuple[str, dict[str, object]]] = []

    async def request(method: str, params: dict[str, object]) -> object:
        calls.append((method, params))
        return {
            "data": [
                {
                    "id": "turn-1",
                    "status": "completed",
                    "createdAt": 10,
                    "items": [
                        {
                            "id": "user-1",
                            "type": "userMessage",
                            "content": [{"type": "text", "text": "Fix the session list"}],
                        },
                        {"id": "reasoning-1", "type": "reasoning", "text": "private chain"},
                        {
                            "id": "assistant-1",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "Done",
                        },
                    ],
                }
            ],
            "nextCursor": "older",
        }

    backend._request = request  # type: ignore[method-assign]
    page = await backend.list_history("thread-1", "cursor-1", 25)

    assert calls == [
        (
            "thread/turns/list",
            {
                "threadId": "thread-1",
                "limit": 25,
                "sortDirection": "desc",
                "itemsView": "full",
                "cursor": "cursor-1",
            },
        )
    ]
    assert page.next_cursor == "older"
    assert [event.kind for event in page.events] == [
        "turn_started",
        "user_prompt",
        "assistant",
        "turn_done",
    ]
    assert page.events[1].text == "Fix the session list"
    assert all(event.event_id for event in page.events)


@pytest.mark.asyncio
async def test_running_sessions_include_exact_tui_presence_without_workspace_guessing(
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
        "_explicit_codex_sessions",
        staticmethod(lambda: {"resumed"}),
    )
    monkeypatch.setattr(
        CodexTuiSessionPresence,
        "sessions",
        staticmethod(lambda _socket_path: {"old"}),
    )
    backend._request = request  # type: ignore[method-assign]

    sessions = await backend.list_running_sessions()

    assert [session.id for session in sessions] == ["resumed", "old"]
    assert [session.tui_attached for session in sessions] == [True, True]


@pytest.mark.asyncio
async def test_workspace_session_page_preserves_exact_tui_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = CodexBackend(CodexConfig(Path("/tmp/codex.sock"), "codex"))

    async def request(method: str, params: dict[str, object]) -> object:
        assert method == "thread/list"
        assert params["cwd"] == "/workspace"
        return {
            "data": [
                {"id": "tui", "cwd": "/workspace", "status": {"type": "idle"}},
                {"id": "stored", "cwd": "/workspace", "status": {"type": "idle"}},
            ]
        }

    monkeypatch.setattr(backend, "_tui_session_ids", lambda: {"tui"})
    backend._request = request  # type: ignore[method-assign]

    sessions = await backend.list_sessions("/workspace")

    assert [session.tui_attached for session in sessions] == [True, False]
