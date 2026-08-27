from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import secrets
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentwire.bridge import Bridge
from agentwire.config import ClaudeConfig, Config, install_secret_env
from agentwire.irc import IRCClient

LOGGER = logging.getLogger("agentwire.stack")


class StackError(RuntimeError):
    pass


_MAX_WEBSOCKET_MESSAGE_BYTES = 8 * 1024 * 1024
_TRANSPORT_CLOSE_TIMEOUT = 5
_RESTART_MIN_DELAY = 0.5
_RESTART_MAX_DELAY = 30.0
# A helper that stayed up this long counts as healthy again, so an outage hours
# after the last one starts from a short delay instead of the capped one.
_HEALTHY_RUNTIME = 60.0


class _CodexTuiRelayTracker:
    """Track the thread selected by one TUI connection from JSON-RPC lifecycle traffic."""

    _THREAD_METHODS = {"thread/start", "thread/resume", "thread/fork"}

    def __init__(self, presence: Any) -> None:
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
    checks = {"ssh": config.stack.ssh_binary}
    if config.codex is not None:
        checks["codex"] = config.codex.binary
    if config.opencode is not None:
        checks["opencode"] = config.opencode.binary
    if config.claude is not None:
        checks["claude"] = config.claude.binary
    if config.pi is not None:
        checks["pi"] = config.pi.binary
    results = [f"{label}: {_binary(binary)}" for label, binary in checks.items()]
    results.append("ergo fakelag: disable privately or exempt bot with nofakelag-only oper class")
    if config.claude is not None:
        results.append(f"claude auth: {_claude_credentials(config.claude)}")
    if config.pi is not None:
        sockets = len(list(config.pi.socket_dir.glob("*.sock")))
        results.append(f"pi sockets: {sockets} live in {config.pi.socket_dir}")
    return results


def _claude_credentials(claude: ClaudeConfig) -> str:
    """Validate that the `claude` CLI will be able to authenticate."""
    if claude.api_key_env is not None:
        # install_secret_env already proved the variable exists in the secrets
        # file; the value itself must never appear in doctor output.
        return f"api key from {claude.api_key_env}"
    try:
        result = subprocess.run(
            [_binary(claude.binary), "auth", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StackError(f"cannot check claude credentials: {exc}") from exc
    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError:
        status = {}
    if not isinstance(status, dict) or status.get("loggedIn") is not True:
        raise StackError(
            "claude CLI has no stored credentials; run `claude auth login` "
            "or set [claude].api_key_env"
        )
    return f"logged in via {status.get('authMethod') or 'stored credentials'}"


def _prepare_runtime(config: Config) -> None:
    if config.codex is None:
        return
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
    backends = {}
    if config.codex is not None:
        from agentwire.backends.codex import CodexBackend

        backends["codex"] = CodexBackend(config.codex)
    if config.opencode is not None:
        from agentwire.backends.opencode import OpenCodeBackend

        opencode_password = os.environ[config.opencode.password_env]
        backends["opencode"] = OpenCodeBackend(config.opencode, opencode_password)
    if config.claude is not None:
        from agentwire.backends.claude import ClaudeBackend

        api_key = (
            os.environ[config.claude.api_key_env] if config.claude.api_key_env is not None else None
        )
        backends["claude"] = ClaudeBackend(config.claude, api_key)
    if config.pi is not None:
        from agentwire.backends.pi import PiBackend

        backends["pi"] = PiBackend(config.pi)
    bridge = Bridge(config, IRCClient(config.irc, irc_password), backends)
    await bridge.run()


@dataclass(slots=True)
class _Helper:
    name: str
    command: list[str]
    # Whether the rest of the stack can ride out this process dying. Only the
    # peers that reconnect on their own may be restarted underneath the bridge;
    # for the others a restart would leave a connection nothing re-establishes.
    restart: bool


class _Supervisor:
    """Own one helper process for the lifetime of the stack.

    An ssh tunnel dies whenever the network or the far end blinks, and the
    bridge is built to ride that out: the IRC client reconnects forever, and so
    does the OpenCode event stream. Failing the whole stack for a blip turned a
    recoverable outage into a manual restart, so a helper whose peers reconnect
    is restarted in place instead, with the same backoff shape the IRC client
    uses. A helper that is not restartable still stops the stack, loudly.
    """

    def __init__(self, helper: _Helper) -> None:
        self.helper = helper
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start the process once. A failure here is a setup error, not an outage."""
        process = await self._spawn()
        await asyncio.sleep(0)
        if process.returncode is not None:
            raise StackError(f"{self.helper.name} exited immediately with {process.returncode}")
        print(f"started: {self.helper.name} (pid {process.pid})", flush=True)

    async def run(self) -> int:
        """Keep the process alive, returning its status once it may not be restarted."""
        delay = _RESTART_MIN_DELAY
        loop = asyncio.get_running_loop()
        process = self.process
        while True:
            if process is None:
                try:
                    process = await self._spawn()
                except OSError as exc:
                    LOGGER.warning(
                        "cannot restart %s (%s); retrying in about %.1fs",
                        self.helper.name,
                        type(exc).__name__,
                        delay,
                    )
                    await self._backoff(delay)
                    delay = min(delay * 2, _RESTART_MAX_DELAY)
                    continue
                LOGGER.warning("restarted %s (pid %d)", self.helper.name, process.pid)
            started = loop.time()
            status = await process.wait()
            # Forget the exited process so shutdown has nothing stale to signal.
            process = self.process = None
            if not self.helper.restart:
                return status
            if loop.time() - started >= _HEALTHY_RUNTIME:
                delay = _RESTART_MIN_DELAY
            LOGGER.warning(
                "%s exited with %d; restarting in about %.1fs", self.helper.name, status, delay
            )
            await self._backoff(delay)
            delay = min(delay * 2, _RESTART_MAX_DELAY)

    async def _spawn(self) -> asyncio.subprocess.Process:
        self.process = await asyncio.create_subprocess_exec(
            *self.helper.command,
            start_new_session=True,
        )
        return self.process

    @staticmethod
    async def _backoff(delay: float) -> None:
        await asyncio.sleep(delay + random.random() * min(delay, 1))


async def run_stack(config: Config) -> None:
    checks = doctor(config)
    for check in checks:
        print(f"doctor: {check}", flush=True)
    _prepare_runtime(config)
    ssh = _binary(config.stack.ssh_binary)
    helpers = [
        _Helper(
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
            # The IRC client reconnects for as long as the bridge runs, so a
            # tunnel that dies with the network is an outage, not a shutdown.
            restart=True,
        )
    ]
    if config.codex is not None:
        helpers.append(
            _Helper(
                "Codex app-server",
                [
                    _binary(config.codex.binary),
                    "app-server",
                    "--listen",
                    f"unix://{config.codex.socket_path}",
                ],
                # CodexBackend holds one JSON-RPC websocket it never re-establishes,
                # so this exit still stops the stack.
                restart=False,
            )
        )
    if config.opencode is not None:
        helpers.append(
            _Helper(
                "OpenCode server",
                [
                    _binary(config.opencode.binary),
                    "serve",
                    "--hostname",
                    "127.0.0.1",
                    "--port",
                    str(config.stack.opencode_port),
                ],
                # The OpenCode backend reconnects its event stream on its own and
                # its requests carry no server-side connection state.
                restart=True,
            )
        )
    # Claude deliberately starts no process here: the Agent SDK owns one `claude`
    # CLI subprocess per session, created and torn down by ClaudeBackend itself.
    # pi likewise: live TUI sessions serve their own extension sockets, and
    # PiBackend owns any `pi --mode rpc` subprocess it spawns.
    supervisors = [_Supervisor(helper) for helper in helpers]
    bridge_task: asyncio.Task[None] | None = None
    watchers: dict[asyncio.Task[int], _Supervisor] = {}
    try:
        for supervisor in supervisors:
            await supervisor.start()
        bridge_task = asyncio.create_task(run_bridge(config), name="bridge")
        watchers = {
            asyncio.create_task(supervisor.run(), name=f"supervise-{supervisor.helper.name}"): (
                supervisor
            )
            for supervisor in supervisors
        }
        done, pending = await asyncio.wait(
            [bridge_task, *watchers], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if bridge_task in done:
            exception = bridge_task.exception()
            if exception:
                raise exception
            raise StackError("bridge stopped unexpectedly")
        finished = next(task for task in done if task in watchers)
        raise StackError(f"{watchers[finished].helper.name} exited with {finished.result()}")
    finally:
        running = [
            task for task in (bridge_task, *watchers) if task is not None and not task.done()
        ]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await _stop_processes(
            [
                (supervisor.helper.name, supervisor.process)
                for supervisor in supervisors
                if supervisor.process is not None
            ]
        )


async def _stop_processes(
    processes: list[tuple[str, asyncio.subprocess.Process]],
) -> None:
    for _name, process in reversed(processes):
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                # Let the parent reap its own workers before escalating to its process group.
                process.terminate()
    for _name, process in reversed(processes):
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), _TRANSPORT_CLOSE_TIMEOUT)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                try:
                    await asyncio.wait_for(process.wait(), _TRANSPORT_CLOSE_TIMEOUT)
                except TimeoutError as exc:
                    raise StackError(f"process {process.pid} did not exit after SIGKILL") from exc


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


async def _relay_websocket_frames(source: Any, destination: Any, observe: Any) -> None:
    import aiohttp

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
    import aiohttp
    from aiohttp import web

    from agentwire.backends.codex import CodexTuiSessionPresence

    if config.codex is None:
        raise StackError("Codex is not enabled by any configured IRC channel")
    runtime = config.codex.socket_path.parent
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(runtime, 0o700)
    # Keep the name no longer than the configured socket so AF_UNIX path limits still hold.
    relay_path = runtime / f".t{secrets.token_hex(4)}"
    presence = CodexTuiSessionPresence(config.codex.socket_path)
    tracker = _CodexTuiRelayTracker(presence)
    connected = asyncio.Lock()

    async def relay(request: web.Request) -> web.StreamResponse:
        if "Origin" in request.headers:
            raise web.HTTPForbidden(text="Origin-bearing requests are not allowed")
        probe = web.WebSocketResponse().can_prepare(request)
        if not probe.ok:
            raise web.HTTPUpgradeRequired(text="WebSocket upgrade required")
        if connected.locked():
            raise web.HTTPServiceUnavailable(text="this Agentwire TUI relay is already in use")
        async with connected:
            connector = aiohttp.UnixConnector(path=str(config.codex.socket_path))
            timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
            downstream = web.WebSocketResponse(
                heartbeat=20,
                compress=False,
                max_msg_size=_MAX_WEBSOCKET_MESSAGE_BYTES,
            )
            try:
                async with (
                    aiohttp.ClientSession(connector=connector, timeout=timeout) as session,
                    session.ws_connect(
                        "http://localhost/",
                        timeout=aiohttp.ClientWSTimeout(ws_close=5),
                        heartbeat=20,
                        compress=0,
                        max_msg_size=_MAX_WEBSOCKET_MESSAGE_BYTES,
                    ) as upstream,
                ):
                    await downstream.prepare(request)
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
            except asyncio.CancelledError:
                raise
            except (OSError, aiohttp.ClientError, TimeoutError):
                if downstream.prepared:
                    await downstream.close(
                        code=aiohttp.WSCloseCode.INTERNAL_ERROR,
                        message=b"Codex app-server relay failed",
                    )
                else:
                    raise web.HTTPBadGateway(text="Codex app-server is unavailable") from None
            finally:
                tracker.close()
                if downstream.prepared:
                    await downstream.close()
            return downstream

    application = web.Application(client_max_size=64 * 1024)
    application.router.add_get("/", relay)
    runner = web.AppRunner(
        application,
        access_log=None,
        handler_cancellation=True,
        shutdown_timeout=_TRANSPORT_CLOSE_TIMEOUT,
    )
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
                await asyncio.wait_for(process.wait(), _TRANSPORT_CLOSE_TIMEOUT)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), _TRANSPORT_CLOSE_TIMEOUT)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(runner.cleanup(), _TRANSPORT_CLOSE_TIMEOUT)
        with contextlib.suppress(FileNotFoundError):
            relay_path.unlink()


def codex_tui(config: Config) -> None:
    install_secret_env(config)
    if config.codex is None:
        raise StackError("Codex is not enabled by any configured IRC channel")
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
