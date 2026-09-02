from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentwire.cli import default_config_path
from agentwire.config import (
    ConfigError,
    _path,
    install_secret_env,
    load_config,
    load_secret_env,
    resolve_workspace,
)


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


def _write_config(tmp_path: Path, channels: str) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[bridge]
owner_account = "owner"
allowed_roots = ["{tmp_path}"]
state_file = "{tmp_path / "state.sqlite3"}"
queue_limit = 10

[secrets]
env_file = "{tmp_path / "secrets.env"}"

[irc]
host = "127.0.0.1"
port = 16698
server_hostname = "irc.example"
ca_file = "{tmp_path / "ca.pem"}"
nickname = "agentwire"
username = "agentwire"
realname = "Agentwire"
password_env = "IRC_PASSWORD"
channels = {channels}

[codex]
socket_path = "{tmp_path / "codex.sock"}"
binary = "codex"

[stack]
ssh_binary = "ssh"
ssh_host = "example"
local_port = 16698
remote_host = "127.0.0.1"
remote_port = 6698
remote_cert_path = "/tmp/cert.pem"
startup_timeout = 30
""",
        encoding="utf-8",
    )
    return config


def test_codex_only_config_does_not_require_opencode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, '{ "#codex" = "codex" }')
    secrets = tmp_path / "secrets.env"
    secrets.write_text("IRC_PASSWORD=test-only\n", encoding="utf-8")
    secrets.chmod(0o600)

    monkeypatch.setenv("IRC_PASSWORD", "placeholder")
    config = load_config(config_path)
    assert config.opencode is None
    install_secret_env(config)
    assert os.environ["IRC_PASSWORD"] == "test-only"


def test_opencode_channel_still_requires_opencode_table(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, '{ "#opencode" = "opencode" }')
    with pytest.raises(ConfigError, match=r"missing \[opencode\] table"):
        load_config(config_path)


def test_pi_channel_requires_pi_table_and_loads_defaults(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, '{ "#pi" = "pi" }')
    with pytest.raises(ConfigError, match=r"missing \[pi\] table"):
        load_config(config_path)

    with config_path.open("a", encoding="utf-8") as handle:
        handle.write('\n[pi]\nbinary = "pi"\n')
    config = load_config(config_path)
    assert config.codex is None
    assert config.pi is not None
    assert config.pi.binary == "pi"
    assert config.pi.socket_dir.name == "pi"
    assert config.pi.socket_dir.parent.name == "agentwire"
    assert config.pi.session_root == Path("~/.pi/agent/sessions").expanduser().resolve(strict=False)
    assert config.pi.dedicated_channels is False

    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("dedicated_channels = true\n")
    assert load_config(config_path).pi.dedicated_channels is True  # type: ignore[union-attr]


def test_omp_channel_requires_table_and_loads_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, '{ "#omp" = "omp" }')
    with pytest.raises(ConfigError, match=r"missing \[omp\] table"):
        load_config(config_path)

    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[omp]\n")
    config = load_config(config_path)
    assert config.codex is None
    assert config.pi is None
    assert config.omp is not None
    assert config.omp.binary == "omp"
    assert config.omp.socket_dir == runtime / "agentwire" / "omp"
    assert config.omp.session_root == Path("~/.omp/agent/sessions").expanduser().resolve(
        strict=False
    )
    assert config.omp.dedicated_channels is False

    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("dedicated_channels = true\n")
    assert load_config(config_path).omp.dedicated_channels is True  # type: ignore[union-attr]


def test_unselected_omp_table_is_ignored(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, '{ "#codex" = "codex" }')
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write('\n[omp]\nbinary = ""\n')
    assert load_config(config_path).omp is None


def test_pi_dedicated_channels_must_be_boolean(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, '{ "#pi" = "pi" }')
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write('\n[pi]\nbinary = "pi"\ndedicated_channels = "yes"\n')
    with pytest.raises(ConfigError, match="dedicated_channels.*boolean"):
        load_config(config_path)


def test_codex_only_config_ignores_pi_table_and_rejects_unknown_backend(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, '{ "#codex" = "codex" }')
    config = load_config(config_path)
    assert config.pi is None

    bad = _write_config(tmp_path, '{ "#other" = "gemini" }')
    with pytest.raises(ConfigError, match="unsupported backend"):
        load_config(bad)
