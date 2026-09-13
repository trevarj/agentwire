from __future__ import annotations

import os
import stat
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class ConfigError(RuntimeError):
    pass


def _path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ConfigError(f"path contains an unresolved environment variable: {value}")
    return Path(expanded).resolve(strict=False)


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _required_str(data: Mapping[str, Any], key: str, section: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"[{section}].{key} must be a non-empty string")
    return value.strip()


def _optional_str(data: Mapping[str, Any], key: str, section: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"[{section}].{key} must be a non-empty string when present")
    return value.strip()


def _positive_int(data: Mapping[str, Any], key: str, default: int, section: str) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"[{section}].{key} must be a positive integer")
    return value


def _boolean(data: Mapping[str, Any], key: str, default: bool, section: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"[{section}].{key} must be a boolean")
    return value


@dataclass(slots=True, frozen=True)
class BridgeConfig:
    owner_account: str
    allowed_roots: tuple[Path, ...]
    state_file: Path
    queue_limit: int


@dataclass(slots=True, frozen=True)
class SecretsConfig:
    env_file: Path


@dataclass(slots=True, frozen=True)
class IRCConfig:
    host: str
    port: int
    server_hostname: str
    ca_file: Path
    nickname: str
    username: str
    realname: str
    password_env: str
    channels: Mapping[str, str]


@dataclass(slots=True, frozen=True)
class CodexConfig:
    socket_path: Path
    binary: str
    dedicated_channels: bool = False


@dataclass(slots=True, frozen=True)
class OpenCodeConfig:
    url: str
    username: str
    password_env: str
    binary: str


@dataclass(slots=True, frozen=True)
class ClaudeConfig:
    binary: str
    model: str | None
    permission_mode: str
    api_key_env: str | None


@dataclass(slots=True, frozen=True)
class PiConfig:
    binary: str
    socket_dir: Path
    session_root: Path
    dedicated_channels: bool = False


@dataclass(slots=True, frozen=True)
class OmpConfig:
    binary: str
    socket_dir: Path
    session_root: Path
    dedicated_channels: bool = False


@dataclass(slots=True, frozen=True)
class StackConfig:
    ssh_binary: str
    ssh_host: str
    local_port: int
    remote_host: str
    remote_port: int
    remote_cert_path: str
    opencode_port: int
    startup_timeout: int


@dataclass(slots=True, frozen=True)
class VoiceConfig:
    model_path: Path


@dataclass(slots=True, frozen=True)
class PMConfig:
    control_socket: Path
    coordinator_channel: str
    projects: Mapping[str, str]


@dataclass(slots=True, frozen=True)
class Config:
    path: Path
    bridge: BridgeConfig
    secrets: SecretsConfig
    irc: IRCConfig
    codex: CodexConfig | None
    opencode: OpenCodeConfig | None
    stack: StackConfig
    # Optional sections trail the required ones to preserve positional callers.
    claude: ClaudeConfig | None = None
    pi: PiConfig | None = None
    omp: OmpConfig | None = None
    voice: VoiceConfig | None = None
    pm: PMConfig | None = None
    # Runtime-only policy; the CLI can require manual approval for PM channels.
    auto_approve: bool = True


# Only the modes that keep Agentwire's approval routing meaningful are accepted.
CLAUDE_PERMISSION_MODES = frozenset({"default", "acceptEdits", "plan", "bypassPermissions"})


def _default_pi_socket_dir() -> str:
    """Match the pi extension's socket location: runtime dir, tmp fallback."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = runtime if runtime and runtime.startswith("/") else tempfile.gettempdir()
    return str(Path(base) / "agentwire" / "pi")


def _default_omp_socket_dir() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = runtime if runtime and runtime.startswith("/") else tempfile.gettempdir()
    return str(Path(base) / "agentwire" / "omp")


def load_config(path: str | Path) -> Config:
    config_path = _path(str(path))
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    bridge = _table(raw, "bridge")
    roots_raw = bridge.get("allowed_roots")
    if (
        not isinstance(roots_raw, list)
        or not roots_raw
        or not all(isinstance(item, str) and item for item in roots_raw)
    ):
        raise ConfigError("[bridge].allowed_roots must be a non-empty string array")
    roots = tuple(_path(item) for item in roots_raw)

    secrets = _table(raw, "secrets")
    irc = _table(raw, "irc")
    channels_raw = irc.get("channels")
    if not isinstance(channels_raw, dict) or not channels_raw:
        raise ConfigError("[irc].channels must map channel names to backend names")
    channels: dict[str, str] = {}
    for channel, backend in channels_raw.items():
        if not isinstance(channel, str) or not channel.startswith("#"):
            raise ConfigError(f"invalid IRC channel: {channel!r}")
        if backend not in {"codex", "opencode", "claude", "pi", "omp"}:
            raise ConfigError(f"unsupported backend {backend!r} for {channel}")
        if channel.lower() in channels:
            raise ConfigError("[irc].channels must have unique channel names")
        channels[channel.lower()] = backend

    codex = _table(raw, "codex") if "codex" in channels.values() else None
    opencode = _table(raw, "opencode") if "opencode" in channels.values() else None
    claude = _table(raw, "claude") if "claude" in channels.values() else None
    pi = _table(raw, "pi") if "pi" in channels.values() else None
    omp = _table(raw, "omp") if "omp" in channels.values() else None
    stack = _table(raw, "stack")
    voice = _table(raw, "voice") if "voice" in raw else None
    pm = _table(raw, "pm") if "pm" in raw else None
    projects: dict[str, str] = {}
    if pm is not None:
        projects_raw = _table(pm, "projects")
        if not projects_raw:
            raise ConfigError("[pm].projects must be a non-empty table")
        for project, channel in projects_raw.items():
            if (
                not isinstance(project, str)
                or not project.strip()
                or not isinstance(channel, str)
                or not channel.strip()
            ):
                raise ConfigError("[pm].projects must map non-empty project names to channel names")
            project = project.strip()
            if project in projects:
                raise ConfigError("[pm].projects must have unique project names")
            projects[project] = channel.strip().lower()
    permission_mode = "default"
    if claude is not None:
        permission_mode = _optional_str(claude, "permission_mode", "claude") or "default"
        if permission_mode not in CLAUDE_PERMISSION_MODES:
            allowed = ", ".join(sorted(CLAUDE_PERMISSION_MODES))
            raise ConfigError(f"[claude].permission_mode must be one of: {allowed}")

    result = Config(
        path=config_path,
        bridge=BridgeConfig(
            owner_account=_required_str(bridge, "owner_account", "bridge").lower(),
            allowed_roots=roots,
            state_file=_path(_required_str(bridge, "state_file", "bridge")),
            queue_limit=_positive_int(bridge, "queue_limit", 10, "bridge"),
        ),
        secrets=SecretsConfig(
            env_file=_path(_required_str(secrets, "env_file", "secrets")),
        ),
        irc=IRCConfig(
            host=_required_str(irc, "host", "irc"),
            port=_positive_int(irc, "port", 6697, "irc"),
            server_hostname=_required_str(irc, "server_hostname", "irc"),
            ca_file=_path(_required_str(irc, "ca_file", "irc")),
            nickname=_required_str(irc, "nickname", "irc"),
            username=_required_str(irc, "username", "irc"),
            realname=_required_str(irc, "realname", "irc"),
            password_env=_required_str(irc, "password_env", "irc"),
            channels=MappingProxyType(channels),
        ),
        codex=(
            CodexConfig(
                socket_path=_path(_required_str(codex, "socket_path", "codex")),
                binary=_required_str(codex, "binary", "codex"),
                dedicated_channels=_boolean(codex, "dedicated_channels", False, "codex"),
            )
            if codex is not None
            else None
        ),
        opencode=(
            OpenCodeConfig(
                url=_required_str(opencode, "url", "opencode").rstrip("/"),
                username=_required_str(opencode, "username", "opencode"),
                password_env=_required_str(opencode, "password_env", "opencode"),
                binary=_required_str(opencode, "binary", "opencode"),
            )
            if opencode is not None
            else None
        ),
        claude=(
            ClaudeConfig(
                binary=_required_str(claude, "binary", "claude"),
                model=_optional_str(claude, "model", "claude"),
                permission_mode=permission_mode,
                api_key_env=_optional_str(claude, "api_key_env", "claude"),
            )
            if claude is not None
            else None
        ),
        pi=(
            PiConfig(
                binary=_required_str(pi, "binary", "pi"),
                socket_dir=_path(_optional_str(pi, "socket_dir", "pi") or _default_pi_socket_dir()),
                session_root=_path(
                    _optional_str(pi, "session_root", "pi") or "~/.pi/agent/sessions"
                ),
                dedicated_channels=_boolean(pi, "dedicated_channels", False, "pi"),
            )
            if pi is not None
            else None
        ),
        stack=StackConfig(
            ssh_binary=_required_str(stack, "ssh_binary", "stack"),
            ssh_host=_required_str(stack, "ssh_host", "stack"),
            local_port=_positive_int(stack, "local_port", 16698, "stack"),
            remote_host=_required_str(stack, "remote_host", "stack"),
            remote_port=_positive_int(stack, "remote_port", 6698, "stack"),
            remote_cert_path=_required_str(stack, "remote_cert_path", "stack"),
            opencode_port=_positive_int(stack, "opencode_port", 14096, "stack"),
            startup_timeout=_positive_int(stack, "startup_timeout", 30, "stack"),
        ),
        omp=(
            OmpConfig(
                binary=_optional_str(omp, "binary", "omp") or "omp",
                socket_dir=_path(
                    _optional_str(omp, "socket_dir", "omp") or _default_omp_socket_dir()
                ),
                session_root=_path(
                    _optional_str(omp, "session_root", "omp") or "~/.omp/agent/sessions"
                ),
                dedicated_channels=_boolean(omp, "dedicated_channels", False, "omp"),
            )
            if omp is not None
            else None
        ),
        voice=(
            VoiceConfig(model_path=_path(_required_str(voice, "model_path", "voice")))
            if voice is not None
            else None
        ),
        pm=(
            PMConfig(
                control_socket=_path(_required_str(pm, "control_socket", "pm")),
                coordinator_channel=_required_str(pm, "coordinator_channel", "pm").lower(),
                projects=MappingProxyType(projects),
            )
            if pm is not None
            else None
        ),
    )
    _validate_cross_fields(result)
    return result


def _validate_cross_fields(config: Config) -> None:
    if config.irc.port != config.stack.local_port:
        raise ConfigError("[irc].port and [stack].local_port must match")
    expected_url = f"http://127.0.0.1:{config.stack.opencode_port}"
    if config.opencode is not None and config.opencode.url != expected_url:
        raise ConfigError(f"[opencode].url must be {expected_url}")
    for root in config.bridge.allowed_roots:
        if not root.is_absolute():
            raise ConfigError(f"allowlisted root is not absolute: {root}")
    if config.pm is not None:
        targets = (config.pm.coordinator_channel, *config.pm.projects.values())
        if any(channel not in config.irc.channels for channel in targets):
            raise ConfigError("[pm] channels must be statically configured in [irc].channels")
        if len(set(targets)) != len(targets):
            raise ConfigError("[pm] coordinator and project channels must be distinct")


def load_secret_env(path: Path) -> dict[str, str]:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise ConfigError(f"cannot stat secrets file {path}: {exc}") from exc
    if mode & 0o077:
        raise ConfigError(f"secrets file {path} must not be group/world accessible")
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read secrets file {path}: {exc}") from exc
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"invalid secrets line {lineno}: expected NAME=value")
        name, value = line.split("=", 1)
        name = name.strip()
        if not name.replace("_", "A").isalnum() or not name[0].isalpha():
            raise ConfigError(f"invalid environment name on secrets line {lineno}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise ConfigError(f"empty secret for {name}")
        result[name] = value
    return result


def install_secret_env(config: Config) -> None:
    secrets = load_secret_env(config.secrets.env_file)
    required = {config.irc.password_env}
    if config.opencode is not None:
        required.add(config.opencode.password_env)
    # Claude may authenticate through the CLI's own stored credentials, so an
    # API key is only mandatory once the config names the variable holding it.
    if config.claude is not None and config.claude.api_key_env is not None:
        required.add(config.claude.api_key_env)
    missing = sorted(required - secrets.keys())
    if missing:
        raise ConfigError(f"missing required secret variables: {', '.join(missing)}")
    os.environ.update(secrets)


def resolve_workspace(raw: str, allowed_roots: tuple[Path, ...]) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if "$" in str(expanded):
        raise ConfigError(f"workspace contains an unresolved environment variable: {raw}")
    candidates = (
        [expanded] if expanded.is_absolute() else [root / expanded for root in allowed_roots]
    )
    resolved_candidates: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if (
            any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots)
            and resolved not in resolved_candidates
        ):
            resolved_candidates.append(resolved)
    if not resolved_candidates:
        if expanded.is_absolute():
            candidate = expanded.resolve(strict=False)
            if candidate.exists() and not candidate.is_dir():
                raise ConfigError(f"workspace is not a directory: {candidate}")
            if candidate.exists():
                raise ConfigError("workspace is outside the configured allowlisted roots")
            raise ConfigError(f"workspace does not exist: {candidate}")
        raise ConfigError(f"workspace does not exist under an allowed root: {raw}")
    if len(resolved_candidates) > 1:
        choices = ", ".join(str(path) for path in resolved_candidates)
        raise ConfigError(f"workspace is ambiguous; use an absolute path: {choices}")
    return resolved_candidates[0]
