from __future__ import annotations

import os
import stat
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


def _positive_int(data: Mapping[str, Any], key: str, default: int, section: str) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"[{section}].{key} must be a positive integer")
    return value


@dataclass(slots=True, frozen=True)
class BridgeConfig:
    owner_account: str
    allowed_roots: tuple[Path, ...]
    state_file: Path
    queue_limit: int
    summary_max_lines: int
    summary_max_bytes: int
    tool_milestone_limit: int
    notify_owner_on_start: bool


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


@dataclass(slots=True, frozen=True)
class OpenCodeConfig:
    url: str
    username: str
    password_env: str
    binary: str


@dataclass(slots=True, frozen=True)
class PasteConfig:
    url: str
    expiry: str
    max_bytes: int


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
class Config:
    path: Path
    bridge: BridgeConfig
    secrets: SecretsConfig
    irc: IRCConfig
    codex: CodexConfig
    opencode: OpenCodeConfig
    paste: PasteConfig
    stack: StackConfig


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
        if backend not in {"codex", "opencode"}:
            raise ConfigError(f"unsupported backend {backend!r} for {channel}")
        channels[channel.lower()] = backend

    codex = _table(raw, "codex")
    opencode = _table(raw, "opencode")
    paste = _table(raw, "paste")
    stack = _table(raw, "stack")
    expiry = _required_str(paste, "expiry", "paste")
    if expiry not in {"1h", "12h", "24h", "72h"}:
        raise ConfigError("[paste].expiry must be one of 1h, 12h, 24h, or 72h")

    result = Config(
        path=config_path,
        bridge=BridgeConfig(
            owner_account=_required_str(bridge, "owner_account", "bridge").lower(),
            allowed_roots=roots,
            state_file=_path(_required_str(bridge, "state_file", "bridge")),
            queue_limit=_positive_int(bridge, "queue_limit", 10, "bridge"),
            summary_max_lines=_positive_int(bridge, "summary_max_lines", 8, "bridge"),
            summary_max_bytes=_positive_int(bridge, "summary_max_bytes", 1200, "bridge"),
            tool_milestone_limit=_positive_int(bridge, "tool_milestone_limit", 12, "bridge"),
            notify_owner_on_start=bool(bridge.get("notify_owner_on_start", True)),
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
        codex=CodexConfig(
            socket_path=_path(_required_str(codex, "socket_path", "codex")),
            binary=_required_str(codex, "binary", "codex"),
        ),
        opencode=OpenCodeConfig(
            url=_required_str(opencode, "url", "opencode").rstrip("/"),
            username=_required_str(opencode, "username", "opencode"),
            password_env=_required_str(opencode, "password_env", "opencode"),
            binary=_required_str(opencode, "binary", "opencode"),
        ),
        paste=PasteConfig(
            url=_required_str(paste, "url", "paste"),
            expiry=expiry,
            max_bytes=_positive_int(paste, "max_bytes", 1_048_576, "paste"),
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
    )
    _validate_cross_fields(result)
    return result


def _validate_cross_fields(config: Config) -> None:
    if config.irc.port != config.stack.local_port:
        raise ConfigError("[irc].port and [stack].local_port must match")
    expected_url = f"http://127.0.0.1:{config.stack.opencode_port}"
    if config.opencode.url != expected_url:
        raise ConfigError(f"[opencode].url must be {expected_url}")
    for root in config.bridge.allowed_roots:
        if not root.is_absolute():
            raise ConfigError(f"allowlisted root is not absolute: {root}")


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
    required = {config.irc.password_env, config.opencode.password_env}
    missing = sorted(required - secrets.keys())
    if missing:
        raise ConfigError(f"missing required secret variables: {', '.join(missing)}")
    os.environ.update(secrets)


def resolve_workspace(raw: str, allowed_roots: tuple[Path, ...]) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not expanded.is_absolute():
        raise ConfigError("workspace path must be absolute")
    candidate = expanded.resolve(strict=False)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"workspace does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise ConfigError(f"workspace is not a directory: {resolved}")
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise ConfigError("workspace is outside the configured allowlisted roots")
    return resolved
