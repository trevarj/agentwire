from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentwire.cli import default_config_path
from agentwire.config import ConfigError, _path, load_secret_env, resolve_workspace


def test_default_config_path_prefers_agentwire_and_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTWIRE_CONFIG", raising=False)
    monkeypatch.delenv("IRC_BRIDGE_CONFIG", raising=False)
    legacy = tmp_path / "irc-bridge" / "config.toml"
    legacy.parent.mkdir()
    legacy.touch()
    assert default_config_path() == legacy
    current = tmp_path / "agentwire" / "config.toml"
    current.parent.mkdir()
    current.touch()
    assert default_config_path() == current


def test_default_config_path_honors_agentwire_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "custom.toml"
    monkeypatch.setenv("AGENTWIRE_CONFIG", str(configured))
    assert default_config_path() == configured


def test_workspace_accepts_path_relative_to_allowed_root(tmp_path: Path) -> None:
    workspace = tmp_path / "relative" / "path"
    workspace.mkdir(parents=True)
    assert resolve_workspace("relative/path", (tmp_path,)) == workspace


def test_relative_workspace_rejects_ambiguous_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "project").mkdir(parents=True)
    (second / "project").mkdir(parents=True)
    with pytest.raises(ConfigError, match="ambiguous"):
        resolve_workspace("project", (first, second))


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
