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
from agentwire.stack import _run_codex_tui


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
    fake_codex.write_text(
        f"""#!{sys.executable}
import asyncio
import sys
import aiohttp

async def main():
    socket_path = sys.argv[2].removeprefix("unix://")
    connector = aiohttp.UnixConnector(path=socket_path)
    async with aiohttp.ClientSession(connector=connector) as session:
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
    assert observed_sessions == [{"thread-live"}]
    assert CodexTuiSessionPresence.sessions(upstream_path) == set()
