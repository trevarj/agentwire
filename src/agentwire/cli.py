from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from agentwire.config import ConfigError, load_config
from agentwire.control import ControlError, DelegateRequest, ReportRequest, send_control_request
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
        help="live private config file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("run", "run only the bridge"),
        ("stack", "run the tunnel, backends, and bridge"),
    ):
        runtime = subparsers.add_parser(command, help=help_text)
        runtime.add_argument(
            "--manual-approval",
            action="store_true",
            help="require manual approval in PM coordinator and project channels",
        )
    doctor_parser = subparsers.add_parser("doctor", help="validate the live setup")
    doctor_parser.add_argument(
        "--json", action="store_true", help="emit a structured diagnostic report"
    )
    subparsers.add_parser("sync-cert", help="refresh the trusted Ergo certificate")
    subparsers.add_parser("codex-tui", help="attach the Codex TUI")
    opencode = subparsers.add_parser("opencode-tui", help="attach the OpenCode TUI")
    opencode.add_argument("cwd", nargs="?", default=os.getcwd())
    for command in ("delegate", "report"):
        control = subparsers.add_parser(command, help=f"{command} a scoped PM task")
        control.add_argument("--socket", type=Path, required=True)
        control.add_argument("--project", required=True)
        control.add_argument("--task", required=True)
        control.add_argument("--text", required=True)
        if command == "report":
            control.add_argument("--status", choices=("done", "blocked"), required=True)
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
    if arguments.command in {"delegate", "report"}:
        request = (
            DelegateRequest(arguments.project, arguments.task, arguments.text)
            if arguments.command == "delegate"
            else ReportRequest(arguments.project, arguments.task, arguments.status, arguments.text)
        )
        try:
            action_id = asyncio.run(send_control_request(arguments.socket.expanduser(), request))
        except ControlError as exc:
            print(f"agentwire: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print(action_id)
        return
    if arguments.config is None:
        arguments.config = default_config_path()
    if arguments.command == "doctor" and arguments.json:
        result = doctor_report(arguments.config)
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        if any(check["status"] == "error" for check in result["checks"]):
            raise SystemExit(1)
        return
    try:
        config = load_config(arguments.config)
        if arguments.command in {"run", "stack"}:
            config = replace(config, auto_approve=not arguments.manual_approval)
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
