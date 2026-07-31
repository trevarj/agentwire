from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

import pytest

from agentwire.models import ChannelBinding
from agentwire.protocol import decode_envelope, new_envelope
from agentwire.state import StateStore


@pytest.mark.asyncio
async def test_sqlite_state_persists_private_bindings_and_action_deduplication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state" / "state.sqlite3"
    store = StateStore(path)
    binding = ChannelBinding("codex", "thread-1", "/workspace")
    await store.set("#Codex", binding)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert await StateStore(path).load() == {"#codex": binding}

    action = new_envelope(
        "sync.request",
        "action",
        "client",
        id=str(uuid.uuid4()),
        device="phone",
    )
    assert await store.claim_action(action) is None
    assert await store.claim_action(action) == "accepted"
    await store.finish_action(action.id, "succeeded")
    assert await StateStore(path).claim_action(action) == "succeeded"


@pytest.mark.asyncio
async def test_queue_is_ordered_editable_and_durable(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    first = await store.enqueue("one", "#c", "s", "first", 2)
    second = await store.enqueue("two", "#c", "s", "second", 2)
    assert [item.id for item in await store.list_queue("#c", "s")] == ["one", "two"]
    await store.move_queue(second.id, 0)
    edited = await store.edit_queue(first.id, "changed")
    assert edited is not None and edited.text == "changed"
    assert [item.id for item in await store.list_queue("#c", "s")] == ["two", "one"]
    with pytest.raises(ValueError, match="queue limit"):
        await store.enqueue("three", "#c", "s", "third", 2)
    assert await store.clear_queue("#c", "s") == 2


@pytest.mark.asyncio
async def test_interrupted_accepted_action_becomes_uncertain_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    action = new_envelope("sync.request", "action", "client", device="phone")
    assert await StateStore(path).claim_action(action) is None
    assert await StateStore(path).claim_action(action) == "uncertain"


@pytest.mark.asyncio
async def test_event_history_is_oldest_first_and_marked_by_caller(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    now = int(time.time() * 1000)
    first = new_envelope("turn.started", "event", "agent", at=now - 2, session_id="s1")
    second = new_envelope("turn.completed", "event", "agent", at=now - 1, session_id="s1")
    other = new_envelope("turn.completed", "event", "agent", at=now, session_id="s2")
    control = new_envelope("channel.snapshot", "event", "agent", at=now)
    await store.append_event("#c", first)
    await store.append_event("#c", second)
    await store.append_event("#c", other)
    await store.append_event("#c", control)
    payloads = await store.history("#c", "s1", before_at=now + 1)
    assert [decode_envelope(item).kind for item in payloads] == [
        "turn.started",
        "turn.completed",
    ]


@pytest.mark.asyncio
async def test_event_history_migration_backfills_session_id(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    event = new_envelope("assistant.completed", "event", "agent", session_id="session-old")
    with sqlite3.connect(path) as database:
        database.execute(
            """CREATE TABLE events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                channel TEXT NOT NULL,
                at INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        database.execute(
            "INSERT INTO events(id, channel, at, kind, payload) VALUES (?, ?, ?, ?, ?)",
            (event.id, "#c", event.at, event.kind, json.dumps(event.to_dict())),
        )

    payloads = await StateStore(path).history("#c", "session-old")

    assert [decode_envelope(item).id for item in payloads] == [event.id]
    with sqlite3.connect(path) as database:
        assert database.execute(
            "SELECT session_id FROM events WHERE id = ?", (event.id,)
        ).fetchone() == ("session-old",)


@pytest.mark.asyncio
async def test_legacy_json_binding_is_migrated_with_private_backup(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "channels": {"#Codex": {"backend": "codex", "session_id": "s1", "cwd": "/work"}},
            }
        ),
        encoding="utf-8",
    )
    store = StateStore(path)
    assert await store.load() == {"#codex": ChannelBinding("codex", "s1", "/work")}
    backup = tmp_path / "state.json.legacy-json"
    assert backup.exists()
    assert os.stat(backup).st_mode & 0o777 == 0o600
