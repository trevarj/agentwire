from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentwire.cli import default_config_path
from agentwire.config import (
    Config,
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


@pytest.mark.parametrize("value", [None, "true", "false", '"yes"'])
def test_codex_dedicated_channels_config(tmp_path: Path, value: str | None) -> None:
    config_path = _write_config(tmp_path, '{ "#codex" = "codex" }')
    if value is not None:
        text = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            text.replace("[codex]\n", f"[codex]\ndedicated_channels = {value}\n"),
            encoding="utf-8",
        )
    if value == '"yes"':
        with pytest.raises(ConfigError, match="dedicated_channels.*boolean"):
            load_config(config_path)
    else:
        config = load_config(config_path)
        assert config.codex is not None
        assert config.codex.dedicated_channels is (value == "true")


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


def test_voice_and_pm_are_optional_for_existing_positional_callers(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, '{ "#codex" = "codex", "#omp" = "omp" }')
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[omp]\n")
    config = load_config(config_path)
    legacy = Config(
        config.path,
        config.bridge,
        config.secrets,
        config.irc,
        config.codex,
        config.opencode,
        config.stack,
        config.claude,
        config.pi,
        config.omp,
    )
    assert legacy == config
    assert config.voice is None
    assert config.pm is None


def test_voice_and_pm_expand_paths_and_normalize_static_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTWIRE_TEST_WORKSPACE", str(tmp_path / "Workspace"))
    config_path = _write_config(
        tmp_path, '{ "#PM" = "codex", "#Touch-Hockey" = "codex", "#motd-dev" = "codex" }'
    )
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write('\n[voice]\nmodel_path = "~/.local/share/whisper/model.bin"\n')
    voice_only = load_config(config_path)
    assert voice_only.pm is None
    assert voice_only.voice is not None
    assert voice_only.voice.model_path == tmp_path / ".local/share/whisper/model.bin"
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            '\n[pm]\ncontrol_socket = "$AGENTWIRE_TEST_WORKSPACE/.agentwire/control.sock"\n'
            'coordinator_channel = " #PM "\n'
            'projects = { " touch-hockey " = " #TOUCH-HOCKEY ", "motd-dev" = "#motd-dev" }\n'
        )
    config = load_config(config_path)
    assert config.voice == voice_only.voice
    assert config.pm is not None
    assert config.pm.control_socket == tmp_path / "Workspace/.agentwire/control.sock"
    assert config.pm.coordinator_channel == "#pm"
    assert config.pm.projects == {"touch-hockey": "#touch-hockey", "motd-dev": "#motd-dev"}
    with pytest.raises(TypeError):
        config.pm.projects["other"] = "#pm"  # type: ignore[index]
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace('[voice]\nmodel_path = "~/.local/share/whisper/model.bin"\n', ""),
        encoding="utf-8",
    )
    pm_only = load_config(config_path)
    assert pm_only.voice is None
    assert pm_only.pm == config.pm


@pytest.mark.parametrize(
    "table",
    [
        "voice = false\n",
        "pm = false\n",
        '[voice]\nmodel_path = ""\n',
        "[voice]\nmodel_path = 123\n",
        "[voice]\n",
        '[pm]\ncoordinator_channel = "#pm"\nprojects = { worker = "#worker" }\n',
        '[pm]\ncontrol_socket = "/control.sock"\nprojects = { worker = "#worker" }\n',
    ],
)
def test_optional_feature_tables_require_typed_fields(tmp_path: Path, table: str) -> None:
    config_path = _write_config(tmp_path, '{ "#pm" = "codex", "#worker" = "codex" }')
    config_path.write_text(table + config_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)


@pytest.mark.parametrize(
    ("coordinator", "projects"),
    [
        ("#pm", "[]"),
        ("#pm", "{}"),
        ("#pm", '{ " " = "#worker" }'),
        ("#pm", '{ worker = "" }'),
        ("#pm", "{ worker = 123 }"),
        ("#pm", '{ worker = "#worker", " worker " = "#other" }'),
        ("#pm", '{ worker = "#worker", worker = "#other" }'),
        ("#pm", '{ worker = "#worker", other = "#WORKER" }'),
        ("#pm", '{ worker = "#PM" }'),
        ("#pm", '{ worker = "#unconfigured" }'),
        ("#unconfigured", '{ worker = "#worker" }'),
    ],
)
def test_pm_rejects_empty_duplicate_or_unconfigured_routes(
    tmp_path: Path, coordinator: str, projects: str
) -> None:
    config_path = _write_config(
        tmp_path, '{ "#pm" = "codex", "#worker" = "codex", "#other" = "codex" }'
    )
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            '\n[pm]\ncontrol_socket = "~/Workspace/.agentwire/control.sock"\n'
            f'coordinator_channel = "{coordinator}"\nprojects = {projects}\n'
        )
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_static_channels_reject_case_insensitive_duplicates(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, '{ "#worker" = "codex", "#WORKER" = "codex" }')
    with pytest.raises(ConfigError):
        load_config(config_path)
