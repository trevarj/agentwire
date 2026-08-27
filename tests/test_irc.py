from __future__ import annotations

import asyncio
import random
from pathlib import Path
from types import MappingProxyType

import pytest

from agentwire.config import IRCConfig
from agentwire.irc import IRCClient, IRCError, parse_irc_line
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
    completed = asyncio.Event()

    async def capture(line: str) -> None:
        lines.append(line)
        if line.startswith("BATCH -"):
            completed.set()

    client._write_line = capture  # type: ignore[method-assign]
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": "body"})
    await client.send_protocol("#c", envelope, "first\nsecond")
    task = asyncio.create_task(client._write_messages())
    await asyncio.wait_for(completed.wait(), 0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lines[0].startswith(f"@{PROTOCOL_TAG}=")
    assert " BATCH +" in lines[0]
    assert all(PROTOCOL_TAG not in line for line in lines[1:])


@pytest.mark.asyncio
async def test_fragment_tail_uses_tagmsg(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    content = random.Random(0).randbytes(10000).hex()
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": content})
    await client.send_protocol("#c", envelope, "preview")
    messages = []
    while not client._outgoing.empty():
        messages.append(client._outgoing.get_nowait())
    assert messages[0].command == "PRIVMSG"
    assert len(messages) > 1
    assert all(message.command == "TAGMSG" for message in messages[1:])


@pytest.mark.asyncio
async def test_notice_is_sent_as_a_notice(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    lines: list[str] = []

    async def capture(line: str) -> None:
        lines.append(line)

    client._write_line = capture  # type: ignore[method-assign]
    await client.send_notice("#c", "agentwire suspended: reason")
    task = asyncio.create_task(client._write_messages())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lines == ["NOTICE #c :agentwire suspended: reason"]


@pytest.mark.asyncio
async def test_writer_failure_interrupts_reader_and_reconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    reader = asyncio.StreamReader()

    class Writer:
        def is_closing(self) -> bool:
            return False

        def write(self, _data: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    async def open_connection(
        *_args: object, **_kwargs: object
    ) -> tuple[asyncio.StreamReader, Writer]:
        return reader, Writer()

    async def negotiate(_reader: asyncio.StreamReader) -> None:
        return None

    async def fail_writer() -> None:
        raise IRCError("writer failed")

    monkeypatch.setattr("agentwire.irc.ssl.create_default_context", lambda **_kwargs: object())
    monkeypatch.setattr("agentwire.irc.asyncio.open_connection", open_connection)
    client._negotiate = negotiate  # type: ignore[method-assign]
    client._write_messages = fail_writer  # type: ignore[method-assign]

    with pytest.raises(IRCError, match="writer failed"):
        await asyncio.wait_for(client._connect_once(), 0.5)


CAPABILITIES = (
    "sasl account-tag message-tags server-time batch draft/multiline labeled-response "
    "echo-message standard-replies draft/chathistory draft/event-playback extended-join"
)


def make_two_channel_client(tmp_path: Path, written: list[str]) -> IRCClient:
    client = IRCClient(
        IRCConfig(
            "localhost",
            6697,
            "localhost",
            tmp_path / "ca.pem",
            "agentwire",
            "agentwire",
            "Agentwire",
            "PASSWORD",
            MappingProxyType({"#codex": "codex", "#claude": "claude"}),
        ),
        "secret",
    )

    async def capture(line: str) -> None:
        written.append(line)

    client._write_line = capture  # type: ignore[method-assign]
    return client


def feed(reader: asyncio.StreamReader, lines: tuple[str, ...]) -> None:
    for line in lines:
        reader.feed_data(f"{line}\r\n".encode())
    reader.feed_eof()


REGISTRATION = (
    f":s CAP * LS :{CAPABILITIES}",
    f":s CAP * ACK :{CAPABILITIES}",
    "AUTHENTICATE +",
    ":s 900 agentwire agentwire!u@h agentwire :You are now logged in as agentwire",
    ":s 903 agentwire :SASL authentication successful",
    ":s 001 agentwire :Welcome",
)


@pytest.mark.asyncio
async def test_every_channel_topic_reaches_the_bridge(tmp_path: Path) -> None:
    """Both channels' topics reach the bridge, not just one of them.

    Two ways to lose a topic meet here. A server answers each JOIN with that
    channel's topic before it echoes the next JOIN, so a registration loop that
    consumed lines without dispatching them discarded the earlier channel's
    topic; and a readiness condition satisfied by the last JOIN echo returned
    before the last channel's topic had been read at all.
    """

    written: list[str] = []
    client = make_two_channel_client(tmp_path, written)
    reader = asyncio.StreamReader()
    feed(
        reader,
        REGISTRATION
        + (
            ":agentwire!u@h JOIN #codex",
            ":s 332 agentwire #codex :agentwire:v1;account=trev;agent=agentwire;backend=codex",
            ":s 353 agentwire = #codex :@agentwire trev",
            ":s 366 agentwire #codex :End of NAMES",
            ":agentwire!u@h JOIN #claude",
            ":s 353 agentwire = #claude :@agentwire trev",
            ":s 366 agentwire #claude :End of NAMES",
            ":s 332 agentwire #claude :agentwire:v1;account=trev;agent=agentwire;backend=claude",
        ),
    )

    await client._negotiate(reader)

    cap_request = next(line for line in written if line.startswith("CAP REQ"))
    assert "echo-message" not in cap_request

    # The account the server granted, not the configured nickname, is what
    # activation validates a topic's agent against.
    assert client.account == "agentwire"
    topics = {}
    while not client._messages.empty():
        message = client._messages.get_nowait()
        topics[message.channel] = (message.command, message.text)
    assert topics == {
        "#codex": ("332", "agentwire:v1;account=trev;agent=agentwire;backend=codex"),
        "#claude": ("332", "agentwire:v1;account=trev;agent=agentwire;backend=claude"),
    }


@pytest.mark.asyncio
async def test_channel_without_a_topic_is_reported_not_awaited(tmp_path: Path) -> None:
    """A channel with no topic answers 331, and only because the bridge asks.

    A server sends RPL_TOPIC unsolicited only when a topic exists, and sends
    nothing at all when it does not, so a bridge that waits for the join burst
    cannot tell "no topic" from "topic still coming": both are silence.
    """

    written: list[str] = []
    client = make_two_channel_client(tmp_path, written)
    reader = asyncio.StreamReader()
    feed(
        reader,
        REGISTRATION
        + (
            ":agentwire!u@h JOIN #codex",
            ":s 332 agentwire #codex :agentwire:v1;account=trev;agent=agentwire;backend=codex",
            ":agentwire!u@h JOIN #claude",
            ":s 331 agentwire #claude :No topic is set",
        ),
    )

    await client._negotiate(reader)

    assert [line for line in written if line.startswith("TOPIC ")] == [
        "TOPIC #codex",
        "TOPIC #claude",
    ]
    replies = [client._messages.get_nowait() for _ in range(client._messages.qsize())]
    assert [(item.channel, item.command, item.text) for item in replies] == [
        ("#codex", "332", "agentwire:v1;account=trev;agent=agentwire;backend=codex"),
        ("#claude", "331", ""),
    ]
