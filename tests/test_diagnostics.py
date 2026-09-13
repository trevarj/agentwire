from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest
from test_bridge import make_bridge
from test_irc import make_client
from test_protocol import _schema_validator

from agentwire import cli, stack
from agentwire.backends.claude import ClaudeBackend
from agentwire.backends.codex import CodexBackend
from agentwire.backends.omp import OmpBackend
from agentwire.backends.opencode import OpenCodeBackend
from agentwire.backends.pi import PiBackend
from agentwire.config import ClaudeConfig, OmpConfig, PiConfig, VoiceConfig
from agentwire.diagnostics import DiagnosticCheck, report, validate_report
from agentwire.irc import IRCMessage, _OutgoingMessage
from agentwire.protocol import (
    PROTOCOL_TAG,
    ProtocolError,
    decode_envelope,
    encode_envelope,
    new_envelope,
)


def test_all_builtin_snapshots_are_cached_and_sessions_are_not_readiness(tmp_path: Path) -> None:
    bridge, _irc, fake = make_bridge(tmp_path)
    config = bridge.config
    assert fake.diagnostic_snapshot()[0].status == "unknown"
    adapters = [
        ClaudeBackend(ClaudeConfig("claude", None, "default", None)),
        CodexBackend(config.codex),
        OpenCodeBackend(config.opencode, "private-secret"),
        PiBackend(PiConfig("pi", tmp_path, tmp_path)),
        OmpBackend(OmpConfig("omp", tmp_path, tmp_path)),
    ]
    for adapter in adapters:
        # No start, subprocess, session discovery or transport is needed.
        assert adapter.diagnostic_snapshot()[0].status == "warning"
        adapter._ready.set()
        snapshot = report("bridge", adapter.diagnostic_snapshot())
        assert snapshot["checks"][0]["status"] == "ok"
        assert snapshot["checks"][0]["facts"]["ready"] is True
        assert "private-secret" not in json.dumps(snapshot)
        if isinstance(adapter, PiBackend):
            assert snapshot["checks"][0]["facts"]["sessionCount"] == 0
        adapter._closed = True
        assert adapter.diagnostic_snapshot()[0].facts["ready"] is False


def test_irc_snapshot_measures_queue_without_consuming_payload(tmp_path: Path) -> None:
    irc = make_client(tmp_path)
    item = _OutgoingMessage("#private", "private-prompt", queued_at=time.monotonic() - 20)
    irc._outgoing.put_nowait(item)
    checks = {check.code: check for check in irc.diagnostic_snapshot()}
    assert checks["irc.connection"].facts == {"connected": False}
    assert checks["irc.outgoing"].status == "warning"
    assert checks["irc.outgoing"].facts["oldestQueuedMs"] >= 20000
    assert checks["irc.outgoing"].facts["queueDepth"] == 1
    assert irc._outgoing.get_nowait() is item
    assert "private" not in json.dumps([check.to_dict() for check in checks.values()])


@pytest.mark.parametrize(
    "invalid",
    [
        {"path": "/private"},
        {"queueDepth": True},
        {"ready": "yes"},
        {"queueDepth": -1},
        {"queueDepth": {"payload": "private"}},
    ],
)
def test_diagnostics_codec_and_schema_reject_unsafe_facts(invalid: dict) -> None:
    fixture = Path(__file__).parents[1] / "protocol/fixtures/diagnostics-snapshot.json"
    raw = json.loads(fixture.read_text())
    raw["data"]["checks"][0]["facts"] = invalid
    assert not _schema_validator().is_valid(raw)
    with pytest.raises(ProtocolError):
        decode_envelope(json.dumps(raw))


def test_doctor_json_failure_is_one_safe_report(tmp_path: Path, monkeypatch, capsys) -> None:
    path = tmp_path / "private-name.toml"
    path.write_text('secret = "private-secret"\n[invalid')
    monkeypatch.setattr(sys, "argv", ["agentwire", "--config", str(path), "doctor", "--json"])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 1
    captured = capsys.readouterr()
    value = json.loads(captured.out)
    validate_report(value)
    assert value["source"] == "doctor"
    assert any(
        check["code"] == "config.load" and check["status"] == "error" for check in value["checks"]
    )
    assert "private-" not in captured.out + captured.err
    assert captured.err == ""


def test_doctor_collects_independent_failures_without_secret_values(
    tmp_path: Path, monkeypatch
) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    config = bridge.config
    config.path.write_text("placeholder")
    config.path.chmod(0o600)
    config.secrets.env_file.write_text(
        "IRC_PASSWORD=private-secret\nOPENCODE_PASSWORD=private-secret\n"
    )
    config.secrets.env_file.chmod(0o600)
    monkeypatch.setattr(stack, "load_config", lambda _path: config)
    monkeypatch.setattr(
        stack.shutil, "which", lambda binary: None if binary == "codex" else "/private/bin"
    )
    value = stack.doctor_report(config.path)
    by_code = {check["code"]: check for check in value["checks"]}
    assert by_code["secrets.required"]["status"] == "ok"
    assert by_code["irc.ca"]["status"] == "error"
    assert by_code["binary.codex"]["status"] == "error"
    assert by_code["binary.opencode"]["status"] == "ok"
    assert by_code["runtime.live"]["status"] == "unknown"
    assert "private-secret" not in json.dumps(value)
    assert "/private" not in json.dumps(value)
    assert not {"binary.ffmpeg", "binary.whisper", "voice.model"} & by_code.keys()


@pytest.mark.parametrize(
    ("missing_binary", "model_kind"),
    [(None, "regular"), ("ffmpeg", "missing"), ("whisper-cli", "directory"), (None, "unreadable")],
)
def test_doctor_voice_checks_are_independent_and_do_not_disclose_paths(
    tmp_path: Path, monkeypatch, capsys, missing_binary: str | None, model_kind: str
) -> None:
    bridge, _irc, _backend = make_bridge(tmp_path)
    model = tmp_path / "private-model-name.bin"
    if model_kind == "directory":
        model.mkdir()
    elif model_kind != "missing":
        model.write_bytes(b"private-model-content")
        if model_kind == "unreadable":
            model.chmod(0o000)
            monkeypatch.setattr(stack.os, "access", lambda _path, _mode: False)
    config = replace(bridge.config, voice=VoiceConfig(model))
    config.path.write_text("placeholder")
    config.path.chmod(0o600)
    monkeypatch.setattr(stack, "load_config", lambda _path: config)
    monkeypatch.setattr(
        stack.shutil, "which", lambda binary: None if binary == missing_binary else "/private/bin"
    )
    value = stack.doctor_report(config.path)
    validate_report(value)
    by_code = {check["code"]: check for check in value["checks"]}
    assert by_code["binary.ffmpeg"]["status"] == ("error" if missing_binary == "ffmpeg" else "ok")
    assert by_code["binary.whisper"]["status"] == (
        "error" if missing_binary == "whisper-cli" else "ok"
    )
    assert by_code["voice.model"]["status"] == ("ok" if model_kind == "regular" else "error")
    captured = capsys.readouterr()
    serialized = json.dumps(value) + captured.out + captured.err
    assert "/private" not in serialized
    assert "private-model" not in serialized
    assert str(tmp_path) not in serialized


@pytest.mark.asyncio
async def test_irc_diagnostics_bypasses_stalled_mutations_and_is_not_journaled(
    tmp_path: Path,
) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    bridge.state.append_event = AsyncMock(side_effect=AssertionError("diagnostics entered history"))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def stalled(*_args) -> None:
        entered.set()
        await release.wait()

    bridge._dispatch_action = stalled
    workers = [
        asyncio.create_task(bridge._action_loop("#codex")),
        asyncio.create_task(bridge._status_loop("#codex")),
        asyncio.create_task(bridge._irc_loop()),
    ]

    async def send(kind: str) -> str:
        action = new_envelope(kind, "action", "client", epoch=bridge.epoch, device="phone")
        await irc.incoming.put(
            IRCMessage(
                "#codex",
                "trev",
                "trev",
                "",
                MappingProxyType({PROTOCOL_TAG: encode_envelope(action)}),
                "TAGMSG",
            )
        )
        return action.id

    try:
        await send("turn.prompt")
        await asyncio.wait_for(entered.wait(), 1)
        query_id = await send("diagnostics.request")
        for _ in range(100):
            replies = [event for _, event, _ in irc.sent if event.reply == query_id]
            if replies:
                break
            await asyncio.sleep(0.01)
        assert len(replies) == 1
        assert replies[0].kind == "diagnostics.snapshot"
        validate_report(replies[0].data)
        assert await bridge.state.action_status(query_id, "#codex", "trev", "codex") is None
        bridge.state.append_event.assert_not_awaited()
    finally:
        release.set()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_bridge_reserves_space_for_its_checks(tmp_path: Path) -> None:
    bridge, irc, backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    backend.diagnostic_snapshot = lambda: [
        DiagnosticCheck(f"backend.check{i}", "unknown", "No cached observation.") for i in range(64)
    ]
    query = new_envelope(
        "diagnostics.request", "action", "client", epoch=bridge.epoch, device="phone"
    )
    await bridge._handle_action("#codex", query)
    result = irc.sent[-1][1]
    assert result.kind == "diagnostics.snapshot"
    assert result.reply == query.id
    validate_report(result.data)
    assert len(result.data["checks"]) == 64
    assert any(check["code"] == "backend.truncated" for check in result.data["checks"])
    assert any(check["code"] == "channel.activation" for check in result.data["checks"])


@pytest.mark.asyncio
async def test_diagnostics_read_queue_is_bounded_and_live_gated(tmp_path: Path) -> None:
    bridge, irc, _backend = make_bridge(tmp_path)
    await bridge._handle_topic("#codex", "agentwire:v1;account=trev;agent=bridge;backend=codex")
    for _ in range(33):
        action = new_envelope(
            "diagnostics.request", "action", "client", epoch=bridge.epoch, device="phone"
        )
        await bridge._ingest_action("#codex", action)
    assert bridge._status_queues["#codex"].qsize() == 32
    assert irc.sent[-1][1].reply == action.id
    assert irc.sent[-1][1].kind == "action.failed"
    stale = new_envelope("diagnostics.request", "action", "client", epoch="old", device="phone")
    await bridge._handle_action("#codex", stale)
    assert irc.sent[-1][1].reply == stale.id
    assert irc.sent[-1][1].kind == "action.failed"
    historic = replace(stale, epoch=bridge.epoch, history=True)
    await bridge._handle_action("#codex", historic)
    assert irc.sent[-1][1].data["message"] == "historic actions cannot be executed"
    assert not any(event.kind == "diagnostics.snapshot" for _, event, _ in irc.sent)
