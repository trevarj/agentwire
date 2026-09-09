from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentwire.models import ChannelBinding
from agentwire.protocol import HISTORY_EVENT_KINDS, Envelope, encode_envelope


@dataclass(slots=True, frozen=True)
class QueuedPrompt:
    id: str
    channel: str
    session_id: str
    position: int
    text: str
    created_at: int


@dataclass(slots=True, frozen=True)
class ActionReceipt:
    """The scoped, non-sensitive part of a durable mutation receipt."""

    id: str
    kind: str
    channel: str
    received_at: int
    status: str
    detail: str


class StateStore:
    """Private SQLite journal for bindings, actions, events, and prompt queues."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._initialized = False
        self._db: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize)
            self._initialized = True

    async def load(self) -> dict[str, ChannelBinding]:
        await self.initialize()
        async with self._lock:
            rows = await asyncio.to_thread(self._load_bindings)
        return {
            row[0]: ChannelBinding(backend=row[1], session_id=row[2], cwd=row[3]) for row in rows
        }

    async def set(self, channel: str, binding: ChannelBinding | None) -> None:
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._set_binding, channel.lower(), binding)

    async def claim_action(
        self,
        envelope: Envelope,
        channel: str,
        owner_account: str | None = None,
        backend: str | None = None,
    ) -> str | None:
        """Insert an action, returning its existing status when it is a duplicate."""
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_action, envelope, channel.lower(), owner_account, backend
            )

    async def finish_action(self, action_id: str, status: str, detail: str = "") -> None:
        if status not in {"succeeded", "failed", "uncertain"}:
            raise ValueError("invalid terminal action status")
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._finish_action, action_id, status, detail)

    async def action_status(
        self, action_id: str, channel: str, owner_account: str, backend: str
    ) -> ActionReceipt | None:
        """Return a receipt only when it belongs to this trusted scope."""
        await self.initialize()
        async with self._lock:
            row = await asyncio.to_thread(
                self._action_status, action_id, channel.lower(), owner_account, backend
            )
        return ActionReceipt(*row) if row else None

    async def append_event(self, channel: str, envelope: Envelope) -> None:
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._append_event, channel.lower(), envelope)

    async def history(
        self,
        channel: str,
        session_id: str,
        before_at: int | None = None,
        limit: int = 200,
    ) -> list[str]:
        if not 1 <= limit <= 200:
            raise ValueError("history limit must be between 1 and 200")
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(
                self._history,
                channel.lower(),
                session_id,
                before_at,
                limit,
            )

    async def list_queue(self, channel: str, session_id: str) -> list[QueuedPrompt]:
        await self.initialize()
        async with self._lock:
            rows = await asyncio.to_thread(self._list_queue, channel.lower(), session_id)
        return [QueuedPrompt(*row) for row in rows]

    async def enqueue(
        self, item_id: str, channel: str, session_id: str, text: str, limit: int
    ) -> QueuedPrompt:
        await self.initialize()
        async with self._lock:
            row = await asyncio.to_thread(
                self._enqueue, item_id, channel.lower(), session_id, text, limit
            )
        return QueuedPrompt(*row)

    async def edit_queue(self, item_id: str, text: str) -> QueuedPrompt | None:
        await self.initialize()
        async with self._lock:
            row = await asyncio.to_thread(self._edit_queue, item_id, text)
        return QueuedPrompt(*row) if row else None

    async def move_queue(self, item_id: str, position: int) -> list[QueuedPrompt]:
        await self.initialize()
        async with self._lock:
            rows = await asyncio.to_thread(self._move_queue, item_id, position)
        return [QueuedPrompt(*row) for row in rows]

    async def delete_queue(self, item_id: str) -> QueuedPrompt | None:
        await self.initialize()
        async with self._lock:
            row = await asyncio.to_thread(self._delete_queue, item_id)
        return QueuedPrompt(*row) if row else None

    async def clear_queue(self, channel: str, session_id: str) -> int:
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(self._clear_queue, channel.lower(), session_id)

    async def close(self) -> None:
        async with self._lock:
            if self._db is not None:
                await asyncio.to_thread(self._db.close)
                self._db = None
                self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            # One persistent WAL connection. Per-operation connections in
            # DELETE journal mode paid connection setup plus a journal-file
            # create/fsync/delete cycle on every commit, which dominated the
            # event write path. All access is serialized by self._lock, so
            # sharing one handle across to_thread workers is safe.
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._db = connection
        return self._db

    def _initialize(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        legacy_bindings = self._move_legacy_json()
        with self._connect() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS bindings (
                    channel TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    channel TEXT,
                    received_at INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL,
                    owner_account TEXT,
                    backend TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL,
                    channel TEXT NOT NULL,
                    session_id TEXT,
                    at INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_channel_at
                    ON events(channel, at DESC, sequence DESC);
                CREATE TABLE IF NOT EXISTS queue (
                    id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(channel, session_id, position)
                );
                """
            )
            # Dynamic Pi channels became process-local. Remove state written by
            # older builds; orphaned bindings are cleared during bridge restore.
            database.execute("DROP TABLE IF EXISTS managed_channels")
            event_columns = {
                str(row[1]) for row in database.execute("PRAGMA table_info(events)").fetchall()
            }
            if "session_id" not in event_columns:
                database.execute("ALTER TABLE events ADD COLUMN session_id TEXT")
            action_columns = {
                str(row[1]) for row in database.execute("PRAGMA table_info(actions)").fetchall()
            }
            # Rows written before scoped status were not authenticated against
            # an owner/backend pair.  They remain replay tombstones but cannot
            # disclose status through the new query API.
            if "owner_account" not in action_columns:
                database.execute("ALTER TABLE actions ADD COLUMN owner_account TEXT")
            if "backend" not in action_columns:
                database.execute("ALTER TABLE actions ADD COLUMN backend TEXT")
            database.execute(
                """CREATE INDEX IF NOT EXISTS actions_scoped_status
                   ON actions(id, channel, owner_account, backend, received_at)"""
            )
            for sequence, payload in database.execute(
                "SELECT sequence, payload FROM events WHERE session_id IS NULL"
            ).fetchall():
                try:
                    session_id = json.loads(str(payload)).get("sid")
                except (AttributeError, json.JSONDecodeError, TypeError):
                    session_id = None
                if isinstance(session_id, str) and session_id:
                    database.execute(
                        "UPDATE events SET session_id = ? WHERE sequence = ?",
                        (session_id, sequence),
                    )
            database.execute(
                """CREATE INDEX IF NOT EXISTS events_channel_session_at
                   ON events(channel, session_id, at DESC, sequence DESC)"""
            )
            database.execute(
                """UPDATE actions SET status = 'uncertain',
                   detail = 'Agentwire restarted before recording the backend outcome'
                WHERE status = 'accepted'"""
            )
            cutoff = _now_ms() - 30 * 24 * 60 * 60 * 1000
            database.execute("DELETE FROM events WHERE at < ?", (cutoff,))
            database.execute("DELETE FROM actions WHERE received_at < ?", (cutoff,))
            for channel, binding in legacy_bindings.items():
                database.execute(
                    """INSERT OR IGNORE INTO bindings
                       (channel, backend, session_id, cwd, updated_at) VALUES (?, ?, ?, ?, ?)""",
                    (
                        channel,
                        binding.backend,
                        binding.session_id,
                        binding.cwd,
                        _now_ms(),
                    ),
                )
        os.chmod(self.path, 0o600)
        # WAL sidecar files inherit creation-time permissions, not the chmod
        # above; keep the documented 0600 posture for everything on disk.
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(OSError):
                os.chmod(f"{self.path}{suffix}", 0o600)

    def _move_legacy_json(self) -> dict[str, ChannelBinding]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("rb") as handle:
                if handle.read(16) == b"SQLite format 3\0":
                    return {}
        except OSError as exc:
            raise RuntimeError(f"cannot inspect bridge state {self.path}: {exc}") from exc
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read legacy bridge state {self.path}: {exc}") from exc
        if raw.get("version") != 1 or not isinstance(raw.get("channels"), dict):
            raise RuntimeError(f"unsupported bridge state format in {self.path}")
        bindings: dict[str, ChannelBinding] = {}
        for channel, value in raw["channels"].items():
            if not isinstance(value, dict):
                continue
            try:
                bindings[str(channel).lower()] = ChannelBinding(
                    str(value["backend"]), str(value["session_id"]), str(value["cwd"])
                )
            except KeyError:
                continue
        backup = self.path.with_name(f"{self.path.name}.legacy-json")
        if backup.exists():
            raise RuntimeError(f"legacy state backup already exists: {backup}")
        os.replace(self.path, backup)
        os.chmod(backup, 0o600)
        return bindings

    def _load_bindings(self) -> list[tuple[str, str, str, str]]:
        with self._connect() as database:
            return database.execute(
                "SELECT channel, backend, session_id, cwd FROM bindings"
            ).fetchall()

    def _set_binding(self, channel: str, binding: ChannelBinding | None) -> None:
        with self._connect() as database:
            if binding is None:
                database.execute("DELETE FROM bindings WHERE channel = ?", (channel,))
            else:
                database.execute(
                    """INSERT INTO bindings(channel, backend, session_id, cwd, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(channel) DO UPDATE SET backend=excluded.backend,
                       session_id=excluded.session_id, cwd=excluded.cwd,
                       updated_at=excluded.updated_at""",
                    (channel, binding.backend, binding.session_id, binding.cwd, _now_ms()),
                )

    def _claim_action(
        self, envelope: Envelope, channel: str, owner_account: str | None, backend: str | None
    ) -> str | None:
        payload = encode_envelope(envelope)
        # The channel is the receiving channel, never a client-supplied field:
        # forensics need to know where an action actually arrived.
        with self._connect() as database:
            row = database.execute(
                "SELECT status FROM actions WHERE id = ?", (envelope.id,)
            ).fetchone()
            if row:
                return str(row[0])
            database.execute(
                """INSERT INTO actions
                   (id, kind, channel, received_at, status, payload, owner_account, backend)
                   VALUES (?, ?, ?, ?, 'accepted', ?, ?, ?)""",
                (envelope.id, envelope.kind, channel, _now_ms(), payload, owner_account, backend),
            )
        return None

    def _finish_action(self, action_id: str, status: str, detail: str) -> None:
        with self._connect() as database:
            database.execute(
                "UPDATE actions SET status = ?, detail = ? WHERE id = ?",
                (status, detail, action_id),
            )

    def _action_status(
        self, action_id: str, channel: str, owner_account: str, backend: str
    ) -> tuple[str, str, str, int, str, str] | None:
        cutoff = _now_ms() - 30 * 24 * 60 * 60 * 1000
        with self._connect() as database:
            return database.execute(
                """SELECT id, kind, channel, received_at, status, detail FROM actions
                   WHERE id = ? AND channel = ? AND owner_account = ? AND backend = ?
                   AND received_at >= ?""",
                (action_id, channel, owner_account, backend, cutoff),
            ).fetchone()

    def _append_event(self, channel: str, envelope: Envelope) -> None:
        with self._connect() as database:
            database.execute(
                """INSERT OR IGNORE INTO events
                   (id, channel, session_id, at, kind, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    envelope.id,
                    channel,
                    envelope.session_id,
                    envelope.at,
                    envelope.kind,
                    encode_envelope(envelope),
                ),
            )

    def _history(
        self,
        channel: str,
        session_id: str,
        before_at: int | None,
        limit: int,
    ) -> list[str]:
        cutoff = before_at if before_at is not None else _now_ms() + 1
        history_kinds = tuple(sorted(HISTORY_EVENT_KINDS))
        placeholders = ", ".join("?" for _ in history_kinds)
        with self._connect() as database:
            rows = database.execute(
                f"""SELECT payload FROM events
                   WHERE channel = ? AND session_id = ? AND kind IN ({placeholders})
                   AND at < ? AND at >= ?
                   ORDER BY at DESC, sequence DESC LIMIT ?""",
                (
                    channel,
                    session_id,
                    *history_kinds,
                    cutoff,
                    _now_ms() - 30 * 24 * 60 * 60 * 1000,
                    limit,
                ),
            ).fetchall()
        payloads = [str(row[0]) for row in reversed(rows)]
        total = 0
        result: list[str] = []
        for payload in payloads:
            size = len(payload.encode("utf-8"))
            if result and total + size > 512 * 1024:
                break
            result.append(payload)
            total += size
        return result

    def _list_queue(self, channel: str, session_id: str) -> list[tuple[Any, ...]]:
        with self._connect() as database:
            return database.execute(
                """SELECT id, channel, session_id, position, text, created_at FROM queue
                   WHERE channel = ? AND session_id = ? ORDER BY position""",
                (channel, session_id),
            ).fetchall()

    def _enqueue(
        self, item_id: str, channel: str, session_id: str, text: str, limit: int
    ) -> tuple[Any, ...]:
        with self._connect() as database:
            count = database.execute(
                "SELECT COUNT(*) FROM queue WHERE channel = ? AND session_id = ?",
                (channel, session_id),
            ).fetchone()[0]
            if count >= limit:
                raise ValueError(f"queue limit of {limit} reached")
            created_at = _now_ms()
            database.execute(
                "INSERT INTO queue VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, channel, session_id, count, text, created_at),
            )
        return item_id, channel, session_id, count, text, created_at

    def _edit_queue(self, item_id: str, text: str) -> tuple[Any, ...] | None:
        with self._connect() as database:
            database.execute("UPDATE queue SET text = ? WHERE id = ?", (text, item_id))
            return database.execute(
                """SELECT id, channel, session_id, position, text, created_at
                   FROM queue WHERE id = ?""",
                (item_id,),
            ).fetchone()

    def _move_queue(self, item_id: str, position: int) -> list[tuple[Any, ...]]:
        with self._connect() as database:
            row = database.execute(
                "SELECT channel, session_id FROM queue WHERE id = ?", (item_id,)
            ).fetchone()
            if not row:
                raise ValueError("unknown queue item")
            channel, session_id = row
            items = database.execute(
                "SELECT id FROM queue WHERE channel = ? AND session_id = ? ORDER BY position",
                (channel, session_id),
            ).fetchall()
            ids = [value[0] for value in items if value[0] != item_id]
            position = max(0, min(position, len(ids)))
            ids.insert(position, item_id)
            database.execute(
                """UPDATE queue SET position = position + 1000000
                   WHERE channel = ? AND session_id = ?""",
                (channel, session_id),
            )
            for index, queued_id in enumerate(ids):
                database.execute("UPDATE queue SET position = ? WHERE id = ?", (index, queued_id))
        return self._list_queue(channel, session_id)

    def _delete_queue(self, item_id: str) -> tuple[Any, ...] | None:
        with self._connect() as database:
            row = database.execute(
                """SELECT id, channel, session_id, position, text, created_at
                   FROM queue WHERE id = ?""",
                (item_id,),
            ).fetchone()
            if not row:
                return None
            database.execute("DELETE FROM queue WHERE id = ?", (item_id,))
            database.execute(
                """UPDATE queue SET position = position - 1
                   WHERE channel = ? AND session_id = ? AND position > ?""",
                (row[1], row[2], row[3]),
            )
            return row

    def _clear_queue(self, channel: str, session_id: str) -> int:
        with self._connect() as database:
            cursor = database.execute(
                "DELETE FROM queue WHERE channel = ? AND session_id = ?", (channel, session_id)
            )
            return cursor.rowcount


def _now_ms() -> int:
    return int(time.time() * 1000)
