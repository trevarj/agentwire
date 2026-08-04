from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ResultMessage

from agentwire.backends import claude as claude_module
from agentwire.backends.base import BackendError
from agentwire.backends.claude import ClaudeBackend, _Session
from agentwire.backends.claude_follow import TranscriptTailer
from agentwire.config import ClaudeConfig
from agentwire.models import BackendEvent

SESSION = "22222222-2222-4222-8222-222222222222"


def stamp(seconds_ago: float = 0.0) -> str:
    moment = datetime.fromtimestamp(time.time() - seconds_ago, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def entry(kind: str, uid: str, message: Any = None, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"type": kind, "uuid": uid, "timestamp": stamp(), **extra}
    if message is not None:
        data["message"] = message
    return data


def prompt(uid: str, text: str, **extra: Any) -> dict[str, Any]:
    return entry("user", uid, {"role": "user", "content": text}, **extra)


def tool_result(uid: str, tool_id: str, output: str, is_error: bool = False) -> dict[str, Any]:
    blocks = [{"type": "tool_result", "tool_use_id": tool_id, "content": output}]
    if is_error:
        blocks[0]["is_error"] = True
    return entry("user", uid, {"role": "user", "content": blocks})


def reply(uid: str, text: str, stop: str = "end_turn") -> dict[str, Any]:
    content = [{"type": "text", "text": text}]
    return entry("assistant", uid, {"role": "assistant", "stop_reason": stop, "content": content})


def tool_use(uid: str, tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    content = [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]
    return entry(
        "assistant", uid, {"role": "assistant", "stop_reason": "tool_use", "content": content}
    )


def boundary(uid: str) -> dict[str, Any]:
    return entry("system", uid, subtype="turn_duration", durationMs=100)


def append(path: Path, *entries: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for item in entries:
            handle.write(json.dumps(item) + "\n")


def backend() -> ClaudeBackend:
    return ClaudeBackend(
        ClaudeConfig(binary="claude", model=None, permission_mode="default", api_key_env=None)
    )


def drain(harness: ClaudeBackend) -> list[BackendEvent]:
    return [harness._events.get_nowait() for _ in range(harness._events.qsize())]


# ----------------------------------------------------------------------
# TranscriptTailer file mechanics
# ----------------------------------------------------------------------


def test_tailer_never_parses_a_partial_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    tailer = TranscriptTailer(path)
    assert tailer.poll() == []  # the file does not exist yet

    append(path, prompt("u1", "hello"))
    partial = json.dumps(prompt("u2", "still being written"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(partial[:25])
    assert [item["uuid"] for item in tailer.poll()] == ["u1"]
    # The unterminated tail must not be consumed, half-parsed, or skipped.
    assert tailer.poll() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write(partial[25:] + "\n")
    assert [item["uuid"] for item in tailer.poll()] == ["u2"]


def test_tailer_prime_consumes_existing_entries_without_replaying_them(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    append(path, prompt("u1", "old"), boundary("s1"))
    tailer = TranscriptTailer(path)
    assert [item["uuid"] for item in tailer.prime()] == ["u1", "s1"]
    assert tailer.poll() == []
    append(path, prompt("u2", "new"))
    assert [item["uuid"] for item in tailer.poll()] == ["u2"]


def test_tailer_truncation_rescans_without_double_emitting(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    append(path, prompt("u1", "one"), prompt("u2", "two"))
    tailer = TranscriptTailer(path)
    assert len(tailer.poll()) == 2
    path.write_text(json.dumps(prompt("u3", "three")) + "\n", encoding="utf-8")
    assert [item["uuid"] for item in tailer.poll()] == ["u3"]


def test_tailer_replacement_with_a_new_inode_dedupes_by_entry_uuid(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    append(path, prompt("u1", "one"), prompt("u2", "two"))
    tailer = TranscriptTailer(path)
    assert len(tailer.poll()) == 2
    replacement = tmp_path / "replacement.jsonl"
    append(replacement, prompt("u1", "one"), prompt("u2", "two"), prompt("u3", "three"))
    os.replace(replacement, path)
    assert [item["uuid"] for item in tailer.poll()] == ["u3"]
    assert tailer.poll() == []


def test_tailer_survives_deletion_and_recreation(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    append(path, prompt("u1", "one"))
    tailer = TranscriptTailer(path)
    assert len(tailer.poll()) == 1
    path.unlink()
    assert tailer.poll() == []
    append(path, prompt("u1", "one"), prompt("u2", "two"))
    assert [item["uuid"] for item in tailer.poll()] == ["u2"]


def test_tailer_skips_malformed_lines_without_stalling(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json at all\n")
        handle.write('"a bare string"\n')
    append(path, prompt("u1", "fine"))
    tailer = TranscriptTailer(path)
    assert [item["uuid"] for item in tailer.poll()] == ["u1"]


# ----------------------------------------------------------------------
# transcript entry mapping
# ----------------------------------------------------------------------


def follow_session() -> _Session:
    return _Session(id=SESSION, cwd="/workspace")


def test_follow_maps_a_full_external_turn_onto_the_driven_vocabulary() -> None:
    harness = backend()
    session = follow_session()

    opened = harness._follow_events(session, prompt("u1", "fix the flaky test"))
    assert [event.kind for event in opened] == ["turn_started", "user_prompt"]
    assert opened[1].text == "fix the flaky test"
    assert session.busy is True
    turn_id = opened[0].turn_id
    # The synthesized turn id matches what a later history replay will use.
    assert turn_id == harness._identifier("u1", "turn")

    narration = harness._follow_events(session, reply("a1", "Looking now.", stop="tool_use"))
    assert [event.kind for event in narration] == ["progress"]

    started = harness._follow_events(
        session, tool_use("a2", "toolu_1", "Bash", {"command": "pytest -q"})
    )
    assert [event.kind for event in started] == ["tool_started"]
    assert started[0].tool_kind == "shell"
    assert started[0].data["label"] == "$ pytest -q"

    finished = harness._follow_events(session, tool_result("u2", "toolu_1", "1 failed"))
    assert [event.kind for event in finished] == ["tool_finished"]
    assert finished[0].success is True
    # The result card keeps the metadata captured when the tool started.
    assert finished[0].data["label"] == "$ pytest -q"
    assert finished[0].data["output"] == "1 failed"

    # The final reply carries stop_reason end_turn and closes the turn itself:
    # print-mode resumes never write the turn_duration marker.
    answer = harness._follow_events(session, reply("a3", "The assertion was inverted."))
    assert [event.kind for event in answer] == ["assistant", "turn_done"]
    assert session.last_reply == "The assertion was inverted."
    assert session.busy is False

    # Interactive sessions write turn_duration afterwards; it must not
    # double-complete the already closed turn.
    assert harness._follow_events(session, boundary("s1")) == []
    events = [*opened, *narration, *started, *finished, *answer]
    assert {event.turn_id for event in events} == {turn_id}


def test_follow_drops_entries_with_no_protocol_shape() -> None:
    harness = backend()
    session = follow_session()
    dropped = [
        prompt("u1", "hidden", isMeta=True),
        prompt("u2", "compacted", isCompactSummary=True),
        reply("a1", "subagent text") | {"isSidechain": True},
        entry("system", "s1", subtype="local_command"),
        entry("queue-operation", "q1"),
        entry("mode", "m1"),
        entry("file-history-snapshot", "f1"),
        prompt("u3", "<command-name>/model</command-name>"),
        prompt("u4", "<local-command-stdout>ok</local-command-stdout>"),
        boundary("s2"),  # no open turn: nothing to complete
    ]
    for item in dropped:
        assert harness._follow_events(session, item) == []
    assert session.busy is False


def test_follow_interrupt_marker_closes_the_turn_without_a_prompt() -> None:
    harness = backend()
    session = follow_session()
    harness._follow_events(session, prompt("u1", "long task"))
    closed = harness._follow_events(session, prompt("u2", "[Request interrupted by user]"))
    assert [event.kind for event in closed] == ["turn_done"]
    assert session.busy is False and session.follow_turn_id is None


def test_follow_todowrite_becomes_deduped_plan_progress() -> None:
    harness = backend()
    session = follow_session()
    harness._follow_events(session, prompt("u1", "plan the work"))
    todos = {"todos": [{"content": "Fix it", "activeForm": "Fixing it", "status": "in_progress"}]}
    first = harness._follow_events(session, tool_use("a1", "todo_1", "TodoWrite", todos))
    assert [event.kind for event in first] == ["progress"]
    assert first[0].data["plan"] is True
    # The raw tool result for a plan write stays hidden, as in driven mode.
    assert harness._follow_events(session, tool_result("u2", "todo_1", "ok")) == []
    assert harness._follow_events(session, tool_use("a2", "todo_2", "TodoWrite", todos)) == []


def test_follow_question_tool_renders_a_card_like_history_replay() -> None:
    harness = backend()
    session = follow_session()
    question = {"questions": [{"question": "Ship it?", "options": [{"label": "Yes"}]}]}
    events = harness._follow_events(session, tool_use("a1", "ask_1", "AskUserQuestion", question))
    assert [event.kind for event in events] == ["tool_started"]
    assert events[0].data["label"] == "Question: Ship it?"


# ----------------------------------------------------------------------
# observe-only attach and promotion
# ----------------------------------------------------------------------


class StubInfo:
    session_id = SESSION
    summary = "External session"
    custom_title = None
    first_prompt = "External session"
    cwd = "/workspace"
    last_modified = 1_785_400_000_000


class StubClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.disconnected = False

    async def connect(self) -> None: ...

    async def query(self, text: str, session_id: str = "default") -> None:
        self.prompts.append(text)

    async def receive_messages(self):  # noqa: ANN201 - mirrors the SDK's untyped iterator
        return
        yield

    async def disconnect(self) -> None:
        self.disconnected = True


def stub_transcript(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    project = tmp_path / "projects" / "workspace"
    project.mkdir(parents=True)
    path = project / f"{SESSION}.jsonl"
    path.touch()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        claude_module, "get_session_info", lambda session_id, directory=None: StubInfo()
    )
    monkeypatch.setattr(
        claude_module,
        "get_session_messages",
        lambda session_id, directory=None, limit=None, offset=0: [],
    )
    return path


@pytest.mark.asyncio
async def test_attach_observes_without_spawning_a_cli_and_never_replays_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = stub_transcript(monkeypatch, tmp_path)
    append(path, prompt("u1", "earlier work"), reply("a1", "done earlier"), boundary("s1"))
    harness = backend()
    try:
        summary = await harness.attach_session(SESSION)
        session = harness._sessions[SESSION]
        assert session.client is None
        assert session.follower is not None
        assert summary.busy is False
        # Pre-attach transcript is history: it must not be emitted live.
        assert drain(harness) == []
        # last_reply is still recoverable from the primed transcript state.
        assert session.last_reply == "done earlier"

        append(path, prompt("u2", "now do this"), reply("a2", "on it", stop="tool_use"))
        await harness._follow_emit(session)
        kinds = [event.kind for event in drain(harness)]
        assert kinds == ["turn_started", "user_prompt", "progress"]
        assert session.busy is True
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_attach_reports_a_fresh_open_turn_busy_and_a_stale_one_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = stub_transcript(monkeypatch, tmp_path)
    append(path, prompt("u1", "running right now"))
    harness = backend()
    try:
        summary = await harness.attach_session(SESSION)
        assert summary.busy is True
        assert summary.active_turn_id == harness._identifier("u1", "turn")
    finally:
        await harness.close()

    stale = stub_transcript(monkeypatch, tmp_path / "stale")
    old = prompt("u1", "abandoned")
    old["timestamp"] = stamp(seconds_ago=3600)
    append(stale, old)
    harness = backend()
    try:
        summary = await harness.attach_session(SESSION)
        # An interrupted or crashed turn must not report the session busy forever.
        assert summary.busy is False
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_prompt_promotes_to_a_driven_client_without_double_emitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = stub_transcript(monkeypatch, tmp_path)
    append(path, prompt("u1", "earlier"), boundary("s1"))
    client = StubClient()
    monkeypatch.setattr(claude_module, "ClaudeSDKClient", lambda options: client)
    harness = backend()
    try:
        await harness.attach_session(SESSION)
        session = harness._sessions[SESSION]
        # A whole external turn lands between attach and the owner's prompt.
        append(
            path, prompt("u2", "terminal prompt"), reply("a2", "terminal answer"), boundary("s2")
        )

        turn_id = await harness.send_message(SESSION, "take over from the phone")
        assert session.client is client
        assert client.prompts == ["take over from the phone"]
        kinds = [event.kind for event in drain(harness)]
        # The external turn is relayed exactly once, before the driven turn opens.
        assert kinds == ["turn_started", "user_prompt", "assistant", "turn_done", "turn_started"]

        # The CLI echoes the driven turn into the transcript; finishing the
        # driven turn discards that echo so the follower cannot replay it.
        append(path, prompt("u3", "take over from the phone"), reply("a3", "driven answer"))
        await harness._handle_message(
            session,
            ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id=SESSION,
                result="driven answer",
            ),
        )
        finished = [event.kind for event in drain(harness)]
        assert finished == ["assistant", "turn_done"]
        assert session.turn_id is None and turn_id is not None
        await harness._follow_emit(session)
        assert drain(harness) == []
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_prompt_refuses_to_promote_into_a_running_external_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = stub_transcript(monkeypatch, tmp_path)
    harness = backend()
    try:
        await harness.attach_session(SESSION)
        append(path, prompt("u1", "terminal turn in flight"))
        with pytest.raises(BackendError, match="another process"):
            await harness.send_message(SESSION, "collide")
        # The in-flight prompt was still relayed for the timeline.
        assert [event.kind for event in drain(harness)] == ["turn_started", "user_prompt"]
        assert harness._sessions[SESSION].client is None
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_steer_is_refused_and_cancel_clears_an_observed_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = stub_transcript(monkeypatch, tmp_path)
    append(path, prompt("u1", "running right now"))
    harness = backend()
    try:
        await harness.attach_session(SESSION)
        session = harness._sessions[SESSION]
        with pytest.raises(BackendError, match="observed only"):
            await harness.steer(SESSION, None, "nudge")
        # Cancel cannot interrupt the external process, but it unjams the channel.
        await harness.cancel(SESSION, None)
        assert [event.kind for event in drain(harness)] == ["turn_done"]
        assert session.busy is False
        with pytest.raises(BackendError, match="no active turn"):
            await harness.cancel(SESSION, None)
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_follow_loop_emits_appended_entries_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_module, "_FOLLOW_POLL_SECONDS", 0.01)
    path = stub_transcript(monkeypatch, tmp_path)
    harness = backend()
    try:
        await harness.attach_session(SESSION)
        append(path, prompt("u1", "typed in the terminal"))
        first = await asyncio.wait_for(harness._events.get(), timeout=2)
        second = await asyncio.wait_for(harness._events.get(), timeout=2)
        assert (first.kind, second.kind) == ("turn_started", "user_prompt")
        assert second.text == "typed in the terminal"
    finally:
        await harness.close()
