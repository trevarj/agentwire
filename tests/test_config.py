from __future__ import annotations

import os
from pathlib import Path

import pytest

from irc_bridge.config import ConfigError, _path, load_secret_env, resolve_workspace


def test_workspace_requires_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="absolute"):
        resolve_workspace("relative/path", (tmp_path,))


def test_path_rejects_unresolved_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_BRIDGE_RUNTIME", raising=False)
    with pytest.raises(ConfigError, match="unresolved"):
        _path("$MISSING_BRIDGE_RUNTIME/codex.sock")


def test_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigError, match="outside"):
        resolve_workspace(str(root / "escape"), (root,))


def test_secret_file_requires_private_permissions(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("PASSWORD=correct-horse-battery-staple\n", encoding="utf-8")
    os.chmod(secrets, 0o644)
    with pytest.raises(ConfigError, match="group/world"):
        load_secret_env(secrets)
    os.chmod(secrets, 0o600)
    assert load_secret_env(secrets) == {"PASSWORD": "correct-horse-battery-staple"}
