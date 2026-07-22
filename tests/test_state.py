from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentwire.models import ChannelBinding
from agentwire.state import StateStore


@pytest.mark.asyncio
async def test_state_persists_only_binding_fields(tmp_path: Path) -> None:
    path = tmp_path / "state" / "state.json"
    store = StateStore(path)
    binding = ChannelBinding("codex", "thread-1", "/workspace")
    await store.set("#Codex", binding)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {
        "version": 1,
        "channels": {
            "#codex": {
                "backend": "codex",
                "session_id": "thread-1",
                "cwd": "/workspace",
            }
        },
    }
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert await StateStore(path).load() == {"#codex": binding}
