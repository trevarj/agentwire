from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from irc_bridge.config import ConfigError, load_config
from irc_bridge.stack import (
    StackError,
    codex_tui,
    doctor,
    opencode_tui,
    run_bridge,
    run_stack,
    sync_certificate,
)


def default_config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return base / "irc-bridge" / "config.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irc-bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="live private config file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="run only the bridge")
    subparsers.add_parser("stack", help="run the tunnel, backends, and bridge")
    subparsers.add_parser("doctor", help="validate the live setup")
    subparsers.add_parser("sync-cert", help="refresh the trusted Ergo certificate")
    subparsers.add_parser("codex-tui", help="attach the Codex TUI")
    opencode = subparsers.add_parser("opencode-tui", help="attach the OpenCode TUI")
    opencode.add_argument("cwd", nargs="?", default=os.getcwd())
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        config = load_config(arguments.config)
        if arguments.command == "run":
            asyncio.run(run_bridge(config))
        elif arguments.command == "stack":
            asyncio.run(run_stack(config))
        elif arguments.command == "doctor":
            for check in doctor(config):
                print(check)
            print("configuration is valid")
        elif arguments.command == "sync-cert":
            sync_certificate(config)
            print(f"updated {config.irc.ca_file}")
        elif arguments.command == "codex-tui":
            codex_tui(config)
        elif arguments.command == "opencode-tui":
            opencode_tui(config, arguments.cwd)
    except KeyboardInterrupt:
        return
    except (ConfigError, StackError, RuntimeError) as exc:
        print(f"irc-bridge: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
