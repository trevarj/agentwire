from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class TranscriptTailer:
    """Incrementally read complete JSONL entries from a Claude Code transcript.

    Claude Code appends one JSON object per line while a session runs, so a
    reader can observe a half-written trailing line at any moment. The tailer
    therefore consumes only newline-terminated lines and resumes from a byte
    offset, which means a partial line is never parsed and is picked up whole
    on a later poll.

    Truncation and replacement are detected by a shrinking size or a changed
    inode; both restart the scan from the top of the file. The per-entry
    ``uuid`` set then suppresses entries that were already delivered, so a
    rescan cannot double-emit. Entries without a ``uuid`` (bookkeeping types
    such as ``mode`` or ``last-prompt``) can reappear after a rescan; callers
    drop those kinds anyway. An in-place rewrite that keeps the inode and does
    not shrink the file is indistinguishable from an append and is not
    handled; Claude Code transcripts are append-only in practice.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._inode: int | None = None
        self._seen: set[str] = set()

    def prime(self) -> list[dict[str, Any]]:
        """Consume everything already on disk without treating it as new.

        Returns the existing complete entries so the caller can rebuild
        in-flight turn state; later ``poll`` calls yield only entries appended
        after this point.
        """
        return self._read()

    def poll(self) -> list[dict[str, Any]]:
        """Return complete, previously unseen entries appended since last read."""
        return self._read()

    def discard_pending(self) -> None:
        """Consume pending entries without delivering them.

        Used after a bridge-driven turn: the SDK message stream already
        relayed that turn live, so its transcript echo must not be emitted a
        second time.
        """
        self._read()

    def _read(self) -> list[dict[str, Any]]:
        try:
            stat = os.stat(self.path)
        except OSError:
            # Not existing yet, or deleted mid-follow: keep state and retry.
            return []
        if self._inode is not None and stat.st_ino != self._inode:
            self._offset = 0
        if stat.st_size < self._offset:
            self._offset = 0
        self._inode = stat.st_ino
        if stat.st_size == self._offset:
            return []
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return []
        end = chunk.rfind(b"\n")
        if end < 0:
            # Only a partial trailing line so far; wait for the newline.
            return []
        complete = chunk[: end + 1]
        self._offset += end + 1
        entries: list[dict[str, Any]] = []
        for line in complete.split(b"\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(entry, dict):
                continue
            identifier = entry.get("uuid")
            if isinstance(identifier, str) and identifier:
                if identifier in self._seen:
                    continue
                self._seen.add(identifier)
            entries.append(entry)
        return entries
