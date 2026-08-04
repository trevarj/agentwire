from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from agentwire.backends.codex import CodexTuiSessionPresence
from agentwire.config import CodexConfig
from agentwire.stack import _run_codex_tui, _stop_processes


@pytest.mark.asyncio
async def test_codex_tui_relay_forwards_lifecycle_and_publishes_presence(
    tmp_path: Path,
) -> None:
    upstream_path = tmp_path / "codex.sock"
    observed_sessions: list[set[str]] = []

    async def upstream(request: web.Request) -> web.StreamResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        async for frame in websocket:
            if frame.type != aiohttp.WSMsgType.TEXT:
                continue
            message = json.loads(frame.data)
            assert message["method"] == "thread/resume"
            await websocket.send_json(
                {"id": message["id"], "result": {"thread": {"id": "thread-live"}}}
            )
            for _ in range(50):
                sessions = CodexTuiSessionPresence.sessions(upstream_path)
                if sessions:
                    observed_sessions.append(sessions)
                    break
                await asyncio.sleep(0.01)
            await websocket.close()
        return websocket

    application = web.Application()
    application.router.add_get("/", upstream)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.UnixSite(runner, str(upstream_path))
    await site.start()

    fake_codex = tmp_path / "fake-codex"
    status_path = tmp_path / "relay-status"
    fake_codex.write_text(
        f"""#!{sys.executable}
import asyncio
import sys
import aiohttp

async def main():
    socket_path = sys.argv[2].removeprefix("unix://")
    connector = aiohttp.UnixConnector(path=socket_path)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get("http://localhost/") as response:
            plain_status = response.status
        async with session.get(
            "http://localhost/",
            headers={{"Origin": "https://untrusted.example"}},
        ) as response:
            origin_status = response.status
        with open({str(status_path)!r}, "w", encoding="utf-8") as handle:
            handle.write(f"{{plain_status}} {{origin_status}}")
        async with session.ws_connect("http://localhost/") as websocket:
            await websocket.send_json({{
                "id": 1,
                "method": "thread/resume",
                "params": {{"threadId": "thread-live"}},
            }})
            await websocket.receive()
            await asyncio.sleep(0.1)

asyncio.run(main())
""",
        encoding="utf-8",
    )
    os.chmod(fake_codex, 0o700)
    config = SimpleNamespace(codex=CodexConfig(upstream_path, "codex"))

    try:
        result = await _run_codex_tui(config, str(fake_codex))  # type: ignore[arg-type]
    finally:
        await runner.cleanup()

    assert result == 0
    assert status_path.read_text(encoding="utf-8") == "426 403"
    assert observed_sessions == [{"thread-live"}]
    assert CodexTuiSessionPresence.sessions(upstream_path) == set()


@pytest.mark.asyncio
async def test_stop_processes_lets_parent_reap_children() -> None:
    class Process:
        pid = 42
        returncode: int | None = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    process = Process()

    await _stop_processes([("Codex app-server", process)])  # type: ignore[list-item]

    assert process.terminated
    assert process.returncode == 0


def test_claude_credentials_check_reads_auth_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentwire import stack
    from agentwire.config import ClaudeConfig
    from agentwire.stack import StackError, _claude_credentials

    keyed = ClaudeConfig(binary="claude", model=None, permission_mode="default", api_key_env="KEY")
    # A configured api_key_env is validated by install_secret_env; doctor must
    # name the variable without ever touching its value.
    assert _claude_credentials(keyed) == "api key from KEY"

    stored = ClaudeConfig(binary="claude", model=None, permission_mode="default", api_key_env=None)
    monkeypatch.setattr(stack, "_binary", lambda name: f"/bin/{name}")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout='{"loggedIn": true, "authMethod": "claude.ai"}', returncode=0)

    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    assert _claude_credentials(stored) == "logged in via claude.ai"
    assert calls == [["/bin/claude", "auth", "status"]]

    monkeypatch.setattr(
        stack.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout='{"loggedIn": false}', returncode=1),
    )
    with pytest.raises(StackError, match="no stored credentials"):
        _claude_credentials(stored)

    monkeypatch.setattr(
        stack.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="not json", returncode=0),
    )
    with pytest.raises(StackError, match="no stored credentials"):
        _claude_credentials(stored)
