from __future__ import annotations

import asyncio
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SessionMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agentwire.backends import claude as claude_module
from agentwire.backends.base import BackendError
from agentwire.backends.claude import ClaudeBackend, _Session
from agentwire.config import ClaudeConfig, ConfigError, load_config
from agentwire.models import BackendEvent

SESSION = "11111111-1111-4111-8111-111111111111"


class StubClient:
    """Stand in for ClaudeSDKClient so no `claude` CLI subprocess is spawned."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.interrupts = 0
        self.disconnected = False

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.prompts.append(prompt)

    async def interrupt(self) -> None:
        self.interrupts += 1

    async def disconnect(self) -> None:
        self.disconnected = True


def backend() -> ClaudeBackend:
    return ClaudeBackend(
        ClaudeConfig(binary="claude", model=None, permission_mode="default", api_key_env=None)
    )


def attached(harness: ClaudeBackend, busy: bool = False) -> _Session:
    session = _Session(id=SESSION, cwd="/workspace", client=StubClient())
    session.busy = busy
    session.turn_id = "turn-1" if busy else None
    harness._sessions[SESSION] = session
    return session


def drain(harness: ClaudeBackend) -> list[BackendEvent]:
    return [harness._events.get_nowait() for _ in range(harness._events.qsize())]


def assistant(*blocks: Any) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-opus-5", message_id="msg_1")


def result(is_error: bool = False, text: str | None = None) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id=SESSION,
        result=text,
    )


@pytest.mark.asyncio
async def test_narration_tools_and_final_answer_map_to_bridge_events() -> None:
    harness = backend()
    session = attached(harness, busy=True)

    await harness._handle_message(
        session,
        assistant(
            TextBlock(text="Checking the failing test first."),
            ToolUseBlock(id="toolu_1", name="Bash", input={"command": "pytest -q"}),
        ),
    )
    await harness._handle_message(
        session,
        UserMessage(
            content=[ToolResultBlock(tool_use_id="toolu_1", content="1 failed", is_error=False)]
        ),
    )
    await harness._handle_message(session, assistant(TextBlock(text="The assertion is inverted.")))
    await harness._handle_message(session, result(text="The assertion is inverted."))

    events = drain(harness)
    assert [event.kind for event in events] == [
        "progress",
        "tool_started",
        "tool_finished",
        "assistant",
        "turn_done",
    ]
    progress, started, finished, reply, done = events
    # Text emitted next to a tool call is commentary, never the turn's answer.
    assert progress.text == "Checking the failing test first."
    assert progress.data == {}
    assert (started.tool_kind, started.item_id) == ("shell", "toolu_1")
    assert started.data["label"] == "$ pytest -q"
    assert started.data["input"] == "pytest -q"
    assert finished.success is True
    assert finished.data["output"] == "1 failed"
    assert finished.data["status"] == "completed"
    assert reply.text == "The assertion is inverted."
    assert all(event.turn_id == "turn-1" for event in events)
    assert done.turn_id == "turn-1"
    assert session.busy is False and session.turn_id is None
    assert session.last_reply == "The assertion is inverted."


@pytest.mark.asyncio
async def test_todo_write_becomes_plan_progress_and_repeats_are_suppressed() -> None:
    harness = backend()
    session = attached(harness, busy=True)
    todos = {
        "todos": [
            {"content": "Read the failure", "status": "completed"},
            {"content": "Fix it", "activeForm": "Fixing it", "status": "in_progress"},
        ]
    }

    await harness._handle_message(
        session, assistant(ToolUseBlock(id="toolu_plan", name="TodoWrite", input=todos))
    )
    # The tool result for a plan write must not surface as a separate tool card.
    await harness._handle_message(
        session, UserMessage(content=[ToolResultBlock(tool_use_id="toolu_plan", content="ok")])
    )
    await harness._handle_message(
        session, assistant(ToolUseBlock(id="toolu_plan2", name="TodoWrite", input=todos))
    )

    events = drain(harness)
    assert [event.kind for event in events] == ["progress"]
    assert events[0].text == "plan: Fixing it"
    assert events[0].data == {
        "plan": True,
        "running": True,
        "status": "inProgress",
        "completedSteps": 1,
        "totalSteps": 2,
    }

    await harness._handle_message(
        session,
        assistant(
            ToolUseBlock(
                id="toolu_plan3",
                name="TodoWrite",
                input={"todos": [{"content": "Fix it", "status": "completed"}]},
            )
        ),
    )
    completion = drain(harness)
    assert completion[0].text == "Plan completed"
    assert completion[0].data["running"] is False
    assert completion[0].data["status"] == "completed"


@pytest.mark.asyncio
async def test_failed_result_reports_the_failure_instead_of_an_assistant_reply() -> None:
    harness = backend()
    session = attached(harness, busy=True)
    session.final_text = "partial work"

    await harness._handle_message(session, result(is_error=True, text="usage limit reached"))

    events = drain(harness)
    assert [event.kind for event in events] == ["turn_failed"]
    assert events[0].text == "usage limit reached"
    assert session.busy is False
    assert session.last_reply is None


@pytest.mark.asyncio
async def test_errored_tool_result_marks_the_tool_unsuccessful() -> None:
    harness = backend()
    session = attached(harness, busy=True)
    await harness._handle_message(
        session,
        assistant(ToolUseBlock(id="toolu_2", name="Read", input={"file_path": "/workspace/a.py"})),
    )
    await harness._handle_message(
        session,
        UserMessage(
            content=[ToolResultBlock(tool_use_id="toolu_2", content="no such file", is_error=True)]
        ),
    )

    started, finished = drain(harness)
    assert started.data["label"] == "Read /workspace/a.py"
    assert started.tool_kind == "file read"
    assert finished.success is False
    assert finished.data["status"] == "error"


@pytest.mark.asyncio
async def test_permission_request_opens_an_approval_and_resolves_once() -> None:
    harness = backend()
    attached(harness, busy=True)
    handler = harness._permission_handler(SESSION)
    pending = asyncio.create_task(
        handler(
            "Bash",
            {"command": "rm -rf build"},
            ToolPermissionContext(tool_use_id="toolu_9", title="Run rm -rf build"),
        )
    )

    opened = await harness._events.get()
    assert opened.kind == "approval"
    assert opened.request_token == "toolu_9"
    assert opened.session_id == SESSION
    assert opened.text == "Run rm -rf build approval needed"

    await harness.resolve_approval("toolu_9", True)
    assert isinstance(await pending, PermissionResultAllow)
    # The bridge emits request.resolved itself, so the backend must not double it.
    assert harness._events.empty()

    with pytest.raises(BackendError, match="already resolved"):
        await harness.resolve_approval("toolu_9", True)


@pytest.mark.asyncio
async def test_denied_permission_returns_a_deny_result() -> None:
    harness = backend()
    attached(harness, busy=True)
    handler = harness._permission_handler(SESSION)
    pending = asyncio.create_task(
        handler("Write", {"file_path": "/etc/passwd"}, ToolPermissionContext(tool_use_id="toolu_x"))
    )
    await harness._events.get()
    await harness.resolve_approval("toolu_x", False)
    assert isinstance(await pending, PermissionResultDeny)


@pytest.mark.asyncio
async def test_prompt_opens_a_turn_and_steering_requires_an_active_one() -> None:
    harness = backend()
    session = attached(harness)

    turn_id = await harness.send_message(SESSION, "explain the failure")
    assert turn_id is not None
    assert session.busy is True and session.turn_id == turn_id
    assert session.client.prompts == ["explain the failure"]
    started = drain(harness)
    assert [event.kind for event in started] == ["turn_started"]
    assert started[0].turn_id == turn_id

    # A prompt the CLI refuses must not leave an announced turn hanging open.
    async def refuse(prompt: str, session_id: str = "default") -> None:
        raise OSError("broken pipe")

    stalled = _Session(id="other", cwd="/workspace", client=StubClient())
    stalled.client.query = refuse
    harness._sessions["other"] = stalled
    with pytest.raises(BackendError, match="prompt failed"):
        await harness.send_message("other", "hello")
    assert stalled.busy is False and stalled.turn_id is None
    assert harness._events.empty()

    await harness.steer(SESSION, turn_id, "actually check the other file")
    assert session.client.prompts[-1] == "actually check the other file"
    await harness.cancel(SESSION, turn_id)
    assert session.client.interrupts == 1

    await harness._handle_message(session, result(text="done"))
    with pytest.raises(BackendError, match="no active turn to steer"):
        await harness.steer(SESSION, None, "too late")
    with pytest.raises(BackendError, match="no active turn to cancel"):
        await harness.cancel(SESSION, None)


@pytest.mark.asyncio
async def test_unknown_session_and_questions_are_refused() -> None:
    harness = backend()
    with pytest.raises(BackendError, match="is not attached"):
        await harness.send_message(SESSION, "hello")
    with pytest.raises(BackendError, match="does not raise answerable questions"):
        await harness.resolve_question("token", (), None)


def transcript() -> list[SessionMessage]:
    return [
        SessionMessage(
            type="user",
            uuid="u1",
            session_id=SESSION,
            message={"role": "user", "content": "first prompt"},
        ),
        SessionMessage(
            type="assistant",
            uuid="a1",
            session_id=SESSION,
            message={
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "looking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Grep",
                        "input": {"pattern": "x"},
                    },
                ],
            },
        ),
        SessionMessage(
            type="user",
            uuid="u2",
            session_id=SESSION,
            message={
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "1 match"}
                ],
            },
        ),
        SessionMessage(
            type="assistant",
            uuid="a2",
            session_id=SESSION,
            message={"role": "assistant", "content": [{"type": "text", "text": "found it"}]},
        ),
        SessionMessage(
            type="user",
            uuid="u3",
            session_id=SESSION,
            message={"role": "user", "content": "second prompt"},
        ),
        SessionMessage(
            type="assistant",
            uuid="a3",
            session_id=SESSION,
            message={"role": "assistant", "content": [{"type": "text", "text": "all done"}]},
        ),
    ]


class StubInfo:
    session_id = SESSION
    summary = "Fix the parser"
    custom_title = None
    first_prompt = "Fix the parser"
    cwd = "/workspace"
    last_modified = 1_785_400_000_000


def stub_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        claude_module,
        "get_session_messages",
        lambda session_id, directory=None, limit=None, offset=0: transcript(),
    )
    monkeypatch.setattr(
        claude_module, "get_session_info", lambda session_id, directory=None: StubInfo()
    )


@pytest.mark.asyncio
async def test_history_is_authoritative_and_pages_backwards_by_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_transcript(monkeypatch)
    harness = backend()

    newest = await harness.list_history(SESSION, None, 2)
    assert [event.kind for event in newest.events] == [
        "turn_started",
        "user_prompt",
        "assistant",
        "turn_done",
    ]
    assert newest.events[1].text == "second prompt"
    assert newest.events[2].text == "all done"
    assert newest.next_cursor == "2"
    # Every event in a turn shares one synthesized turn id so clients can group them.
    turn_ids = {event.turn_id for event in newest.events}
    assert len(turn_ids) == 1 and None not in turn_ids
    assert all(event.at is not None and event.event_id for event in newest.events)

    older = await harness.list_history(SESSION, newest.next_cursor, 4)
    assert [event.kind for event in older.events] == [
        "turn_started",
        "user_prompt",
        "progress",
        "tool_started",
        "tool_finished",
        "assistant",
        "turn_done",
    ]
    assert older.events[3].tool_kind == "file read"
    assert older.events[4].success is True
    assert older.next_cursor is None

    with pytest.raises(BackendError, match="cursor"):
        await harness.list_history(SESSION, "not-a-number", 10)


@pytest.mark.asyncio
async def test_history_pages_are_stable_and_ordered_against_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_transcript(monkeypatch)
    harness = backend()
    newest = await harness.list_history(SESSION, None, 2)
    again = await harness.list_history(SESSION, None, 2)
    identity = [(event.event_id, event.at, event.turn_id) for event in newest.events]
    assert identity == [(event.event_id, event.at, event.turn_id) for event in again.events]

    older = await harness.list_history(SESSION, newest.next_cursor, 4)
    # An older page must never claim timestamps newer than the page after it.
    assert max(event.at or 0 for event in older.events) < min(
        event.at or 0 for event in newest.events
    )
    assert len({event.event_id for event in (*older.events, *newest.events)}) == len(
        older.events
    ) + len(newest.events)


@pytest.mark.asyncio
async def test_a_turn_split_across_pages_keeps_one_turn_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_transcript(monkeypatch)
    harness = backend()
    # Cut the transcript so the second prompt's turn straddles the page boundary.
    tail = await harness.list_history(SESSION, None, 1)
    head = await harness.list_history(SESSION, "1", 1)
    tail_turns = {event.turn_id for event in tail.events}
    head_turns = {event.turn_id for event in head.events}
    assert tail_turns == head_turns
    assert None not in tail_turns
    # The prompt page opens the turn; the reply page must not re-open it.
    assert [event.kind for event in head.events] == ["turn_started", "user_prompt"]
    assert [event.kind for event in tail.events] == ["assistant", "turn_done"]


@pytest.mark.asyncio
async def test_sessions_and_last_reply_come_from_the_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_transcript(monkeypatch)
    monkeypatch.setattr(
        claude_module,
        "list_sessions",
        lambda directory=None, limit=None, offset=0, include_worktrees=True: [StubInfo()],
    )
    harness = backend()
    sessions = await harness.list_sessions("/workspace")
    assert len(sessions) == 1
    assert sessions[0].id == SESSION
    assert sessions[0].title == "Fix the parser"
    assert sessions[0].cwd == "/workspace"
    # SDKSessionInfo reports milliseconds while SessionSummary carries seconds.
    assert sessions[0].updated_at == 1_785_400_000.0
    assert sessions[0].busy is False

    assert await harness.list_running_sessions() == []
    assert await harness.get_last_reply(SESSION) == "all done"


def test_tool_labels_cover_claude_code_built_ins() -> None:
    kind = ClaudeBackend._tool_kind
    assert kind("Bash") == "shell"
    assert kind("Edit") == "file edit"
    assert kind("Glob") == "file read"
    assert kind("WebSearch") == "web search"
    assert kind("Task") == "agent"
    assert kind("mcp__github__list_prs") == "MCP tool"
    assert kind("SomethingNew") == "tool"

    metadata = ClaudeBackend._tool_metadata
    assert metadata("WebFetch", {"url": "https://example.test"})["label"] == (
        "Fetch: https://example.test"
    )
    assert metadata("mcp__github__list_prs", {})["label"] == "github / list_prs"
    assert metadata("Task", {"description": "review the diff"})["label"] == "review the diff"


def test_claude_channel_requires_its_section_and_a_known_permission_mode(tmp_path) -> None:
    def write(extra: str, backend_name: str = "claude") -> str:
        path = tmp_path / "config.toml"
        path.write_text(
            f"""
[bridge]
owner_account = "trev"
allowed_roots = ["{tmp_path}"]
state_file = "{tmp_path}/state.sqlite3"

[secrets]
env_file = "{tmp_path}/secrets.env"

[irc]
host = "127.0.0.1"
port = 16698
server_hostname = "irc.example.com"
ca_file = "{tmp_path}/ca.pem"
nickname = "agentwire"
username = "agentwire"
realname = "Agentwire"
password_env = "AGENTWIRE_IRC_PASSWORD"
channels = {{ "#agent" = "{backend_name}" }}

[codex]
socket_path = "{tmp_path}/codex.sock"
binary = "codex"
{extra}

[stack]
ssh_binary = "ssh"
ssh_host = "host"
local_port = 16698
remote_host = "127.0.0.1"
remote_port = 6698
remote_cert_path = "/cert"
""",
            encoding="utf-8",
        )
        return str(path)

    with pytest.raises(ConfigError, match=r"missing \[claude\] table"):
        load_config(write(""))

    with pytest.raises(ConfigError, match="permission_mode"):
        load_config(write('\n[claude]\nbinary = "claude"\npermission_mode = "yolo"\n'))

    config = load_config(
        write('\n[claude]\nbinary = "claude"\nmodel = "claude-opus-5"\napi_key_env = "KEY"\n')
    )
    assert config.claude is not None
    assert config.claude.binary == "claude"
    assert config.claude.model == "claude-opus-5"
    assert config.claude.permission_mode == "default"
    assert config.claude.api_key_env == "KEY"

    # A section left in the file stays inert until a channel selects the backend.
    codex_only = load_config(write('\n[claude]\nbinary = "claude"\n', backend_name="codex"))
    assert codex_only.claude is None

    with pytest.raises(ConfigError, match="unsupported backend"):
        load_config(write("", backend_name="gemini"))
