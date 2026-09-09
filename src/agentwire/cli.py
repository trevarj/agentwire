from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from agentwire.config import ConfigError, load_config
from agentwire.stack import (
    StackError,
    codex_tui,
    doctor,
    doctor_report,
    opencode_tui,
    run_bridge,
    run_stack,
    sync_certificate,
)


def default_config_path() -> Path:
    configured = os.environ.get("AGENTWIRE_CONFIG") or os.environ.get("IRC_BRIDGE_CONFIG")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    current = base / "agentwire" / "config.toml"
    legacy = base / "irc-bridge" / "config.toml"
    return legacy if not current.exists() and legacy.exists() else current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentwire")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="live private config file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="run only the bridge")
    subparsers.add_parser("stack", help="run the tunnel, backends, and bridge")
    doctor_parser = subparsers.add_parser("doctor", help="validate the live setup")
    doctor_parser.add_argument(
        "--json", action="store_true", help="emit a structured diagnostic report"
    )
    subparsers.add_parser("sync-cert", help="refresh the trusted Ergo certificate")
    subparsers.add_parser("codex-tui", help="attach the Codex TUI")
    opencode = subparsers.add_parser("opencode-tui", help="attach the OpenCode TUI")
    opencode.add_argument("cwd", nargs="?", default=os.getcwd())
    return parser


def configure_logging() -> None:
    """Send bridge diagnostics to stderr.

    ``AGENTWIRE_LOG_LEVEL`` overrides the default; ``DEBUG`` adds the
    per-message classifications that are too chatty for normal operation.
    """

    level = os.environ.get("AGENTWIRE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    arguments = _parser().parse_args()
    configure_logging()
    if arguments.command == "doctor" and arguments.json:
        result = doctor_report(arguments.config)
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        if any(check["status"] == "error" for check in result["checks"]):
            raise SystemExit(1)
        return
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
        print(f"agentwire: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
