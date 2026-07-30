from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from agentwire.backends.codex import CodexBackend, CodexTuiSessionPresence
from agentwire.backends.opencode import OpenCodeBackend
from agentwire.bridge import Bridge
from agentwire.config import Config, install_secret_env
from agentwire.irc import IRCClient


class StackError(RuntimeError):
    pass


class _CodexTuiRelayTracker:
    """Track the thread selected by one TUI connection from JSON-RPC lifecycle traffic."""

    _THREAD_METHODS = {"thread/start", "thread/resume", "thread/fork"}

    def __init__(self, presence: CodexTuiSessionPresence) -> None:
        self.presence = presence
        self.current_session: str | None = None
        self.pending: dict[str | int, tuple[str, str | None]] = {}

    def client_message(self, text: str) -> None:
        message = self._object(text)
        request_id = message.get("id")
        method = message.get("method")
        if not isinstance(request_id, (str, int)) or not isinstance(method, str):
            return
        if method not in self._THREAD_METHODS | {"thread/unsubscribe"}:
            return
        params = message.get("params")
        thread_id = str(params.get("threadId") or "") if isinstance(params, dict) else ""
        self.pending[request_id] = (method, thread_id or None)

    def server_message(self, text: str) -> None:
        message = self._object(text)
        method = message.get("method")
        params = message.get("params")
        if method == "thread/started" and isinstance(params, dict):
            thread = params.get("thread")
            if isinstance(thread, dict):
                self._select(str(thread.get("id") or "") or None)
            return
        request_id = message.get("id")
        if not isinstance(request_id, (str, int)):
            return
        pending = self.pending.pop(request_id, None)
        if pending is None or "error" in message:
            return
        pending_method, pending_thread = pending
        if pending_method in self._THREAD_METHODS:
            result = message.get("result")
            thread = result.get("thread") if isinstance(result, dict) else None
            if isinstance(thread, dict):
                self._select(str(thread.get("id") or "") or None)
        elif pending_method == "thread/unsubscribe" and pending_thread == self.current_session:
            self._select(None)

    def close(self) -> None:
        self.current_session = None
        self.pending.clear()
        self.presence.clear()

    def _select(self, session_id: str | None) -> None:
        self.current_session = session_id
        self.presence.update(session_id)

    @staticmethod
    def _object(text: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


def _binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise StackError(f"required binary is not on PATH: {name}")
    return resolved


def _private_file(path: Path, description: str) -> None:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        raise StackError(f"cannot stat {description} {path}: {exc}") from exc
    if not path.is_file():
        raise StackError(f"{description} is not a regular file: {path}")
    if mode & 0o077:
        raise StackError(f"{description} must not be group/world accessible: {path}")


def doctor(config: Config) -> list[str]:
    _private_file(config.path, "live config")
    _private_file(config.secrets.env_file, "secrets file")
    install_secret_env(config)
    try:
        ssl.create_default_context(cafile=str(config.irc.ca_file))
    except (OSError, ssl.SSLError) as exc:
        raise StackError(f"invalid IRC CA file {config.irc.ca_file}: {exc}") from exc
    checks = {
        "ssh": config.stack.ssh_binary,
        "codex": config.codex.binary,
    }
    if config.opencode is not None:
        checks["opencode"] = config.opencode.binary
    return [f"{label}: {_binary(binary)}" for label, binary in checks.items()]


def _prepare_runtime(config: Config) -> None:
    socket_path = config.codex.socket_path
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(socket_path.parent, 0o700)
    try:
        mode = socket_path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise StackError(f"refusing to replace non-socket path: {socket_path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(socket_path))
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError as exc:
        raise StackError(f"cannot safely inspect existing Codex socket: {exc}") from exc
    else:
        raise StackError(f"Codex app-server is already listening at {socket_path}")
    finally:
        probe.close()
    socket_path.unlink()


async def run_bridge(config: Config) -> None:
    install_secret_env(config)
    irc_password = os.environ[config.irc.password_env]
    backends = {"codex": CodexBackend(config.codex)}
    if config.opencode is not None:
        opencode_password = os.environ[config.opencode.password_env]
        backends["opencode"] = OpenCodeBackend(config.opencode, opencode_password)
    bridge = Bridge(config, IRCClient(config.irc, irc_password), backends)
    await bridge.run()


async def run_stack(config: Config) -> None:
    checks = doctor(config)
    for check in checks:
        print(f"doctor: {check}", flush=True)
    _prepare_runtime(config)
    ssh = _binary(config.stack.ssh_binary)
    codex = _binary(config.codex.binary)
    commands = [
        (
            "ssh tunnel",
            [
                ssh,
                "-N",
                "-T",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-L",
                f"127.0.0.1:{config.stack.local_port}:{config.stack.remote_host}:{config.stack.remote_port}",
                config.stack.ssh_host,
            ],
        ),
        (
            "Codex app-server",
            [
                codex,
                "app-server",
                "--listen",
                f"unix://{config.codex.socket_path}",
            ],
        ),
    ]
    if config.opencode is not None:
        commands.append(
            (
                "OpenCode server",
                [
                    _binary(config.opencode.binary),
                    "serve",
                    "--hostname",
                    "127.0.0.1",
                    "--port",
                    str(config.stack.opencode_port),
                ],
            )
        )
    processes: list[tuple[str, asyncio.subprocess.Process]] = []
    try:
        for name, command in commands:
            process = await asyncio.create_subprocess_exec(
                *command,
                start_new_session=True,
            )
            processes.append((name, process))
            await asyncio.sleep(0)
            if process.returncode is not None:
                raise StackError(f"{name} exited immediately with {process.returncode}")
            print(f"started: {name} (pid {process.pid})", flush=True)
        bridge_task = asyncio.create_task(run_bridge(config), name="bridge")
        watchers: dict[asyncio.Task[int], str] = {
            asyncio.create_task(process.wait(), name=f"wait-{name}"): name
            for name, process in processes
        }
        done, pending = await asyncio.wait(
            [bridge_task, *watchers], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if bridge_task in done:
            exception = bridge_task.exception()
            if exception:
                raise exception
            raise StackError("bridge stopped unexpectedly")
        finished = next(task for task in done if task in watchers)
        raise StackError(f"{watchers[finished]} exited with {finished.result()}")
    finally:
        if "bridge_task" in locals() and not bridge_task.done():
            bridge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bridge_task
        await _stop_processes(processes)


async def _stop_processes(
    processes: list[tuple[str, asyncio.subprocess.Process]],
) -> None:
    for _name, process in reversed(processes):
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    for _name, process in reversed(processes):
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


def sync_certificate(config: Config) -> None:
    destination = config.irc.ca_file
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        result = subprocess.run(
            [
                _binary("scp"),
                f"{config.stack.ssh_host}:{config.stack.remote_cert_path}",
                str(temp),
            ],
            check=False,
        )
        if result.returncode != 0:
            raise StackError(f"scp failed with exit status {result.returncode}")
        try:
            ssl.create_default_context(cafile=str(temp))
        except (OSError, ssl.SSLError) as exc:
            raise StackError(f"downloaded certificate is invalid: {exc}") from exc
        os.chmod(temp, 0o600)
        os.replace(temp, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


async def _relay_websocket_frames(
    source: aiohttp.ClientWebSocketResponse | web.WebSocketResponse,
    destination: aiohttp.ClientWebSocketResponse | web.WebSocketResponse,
    observe: Any,
) -> None:
    async for frame in source:
        if frame.type == aiohttp.WSMsgType.TEXT:
            observe(frame.data)
            await destination.send_str(frame.data)
        elif frame.type == aiohttp.WSMsgType.BINARY:
            await destination.send_bytes(frame.data)
        elif frame.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        }:
            return


async def _run_codex_tui(config: Config, binary: str) -> int:
    runtime = config.codex.socket_path.parent
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(runtime, 0o700)
    # Keep the name no longer than the configured socket so AF_UNIX path limits still hold.
    relay_path = runtime / f".t{secrets.token_hex(4)}"
    presence = CodexTuiSessionPresence(config.codex.socket_path)
    tracker = _CodexTuiRelayTracker(presence)
    connected = asyncio.Lock()

    async def relay(request: web.Request) -> web.StreamResponse:
        if connected.locked():
            raise web.HTTPServiceUnavailable(text="this Agentwire TUI relay is already in use")
        async with connected:
            downstream = web.WebSocketResponse()
            await downstream.prepare(request)
            connector = aiohttp.UnixConnector(path=str(config.codex.socket_path))
            try:
                async with (
                    aiohttp.ClientSession(connector=connector) as session,
                    session.ws_connect(
                        "http://localhost/",
                        timeout=aiohttp.ClientWSTimeout(ws_close=5),
                        heartbeat=20,
                    ) as upstream,
                ):
                    client_to_server = asyncio.create_task(
                        _relay_websocket_frames(
                            downstream,
                            upstream,
                            tracker.client_message,
                        )
                    )
                    server_to_client = asyncio.create_task(
                        _relay_websocket_frames(
                            upstream,
                            downstream,
                            tracker.server_message,
                        )
                    )
                    done, pending = await asyncio.wait(
                        {client_to_server, server_to_client},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        task.result()
            except (OSError, aiohttp.ClientError):
                await downstream.close(
                    code=aiohttp.WSCloseCode.INTERNAL_ERROR,
                    message=b"Codex app-server relay failed",
                )
            finally:
                tracker.close()
                await downstream.close()
            return downstream

    application = web.Application()
    application.router.add_get("/", relay)
    runner = web.AppRunner(application, access_log=None)
    process: asyncio.subprocess.Process | None = None
    try:
        await runner.setup()
        site = web.UnixSite(runner, str(relay_path))
        await site.start()
        os.chmod(relay_path, 0o600)
        process = await asyncio.create_subprocess_exec(
            binary,
            "--remote",
            f"unix://{relay_path}",
        )
        return await process.wait()
    finally:
        tracker.close()
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        await runner.cleanup()
        with contextlib.suppress(FileNotFoundError):
            relay_path.unlink()


def codex_tui(config: Config) -> None:
    install_secret_env(config)
    binary = _binary(config.codex.binary)
    return_code = asyncio.run(_run_codex_tui(config, binary))
    if return_code:
        raise StackError(f"Codex TUI exited with status {return_code}")


def opencode_tui(config: Config, cwd: str) -> None:
    install_secret_env(config)
    if config.opencode is None:
        raise StackError("OpenCode is not enabled by any configured IRC channel")
    binary = _binary(config.opencode.binary)
    os.execv(
        binary,
        [binary, "attach", config.opencode.url, "--dir", cwd],
    )
