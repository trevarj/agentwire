from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType

import pytest

from agentwire.config import IRCConfig
from agentwire.irc import IRCClient, parse_irc_line
from agentwire.protocol import PROTOCOL_TAG, new_envelope


def test_parse_ircv3_account_tag() -> None:
    line = parse_irc_line("@account=trev;time=2026-07-19T00:00:00Z :trev!u@h PRIVMSG #codex :hello")
    assert line.command == "PRIVMSG"
    assert line.tags["account"] == "trev"
    assert line.params == ("#codex", "hello")


def test_parse_ircv3_tag_escaping() -> None:
    line = parse_irc_line("@example=hello\\sworld\\:ok :n!u@h TAGMSG #c")
    assert line.tags["example"] == "hello world;ok"


def make_client(tmp_path: Path) -> IRCClient:
    return IRCClient(
        IRCConfig(
            "localhost",
            6697,
            "localhost",
            tmp_path / "ca.pem",
            "agentwire",
            "agentwire",
            "Agentwire",
            "PASSWORD",
            MappingProxyType({"#c": "codex"}),
        ),
        "secret",
    )


@pytest.mark.asyncio
async def test_protocol_preview_tags_multiline_batch_opening_only(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client._caps = {"batch", "draft/multiline"}
    lines: list[str] = []

    async def capture(line: str) -> None:
        lines.append(line)

    client._write_line = capture  # type: ignore[method-assign]
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": "body"})
    await client.send_protocol("#c", envelope, "first\nsecond")
    task = asyncio.create_task(client._write_messages())
    await asyncio.sleep(0.35)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lines[0].startswith(f"@{PROTOCOL_TAG}=")
    assert " BATCH +" in lines[0]
    assert all(PROTOCOL_TAG not in line for line in lines[1:])


@pytest.mark.asyncio
async def test_fragment_tail_uses_tagmsg(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": "x" * 10000})
    await client.send_protocol("#c", envelope, "preview")
    messages = []
    while not client._outgoing.empty():
        messages.append(client._outgoing.get_nowait())
    assert messages[0].command == "PRIVMSG"
    assert len(messages) > 1
    assert all(message.command == "TAGMSG" for message in messages[1:])
