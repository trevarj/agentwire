from __future__ import annotations

import asyncio
import base64
import contextlib
import random
import secrets
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from agentwire.config import IRCConfig
from agentwire.protocol import PROTOCOL_TAG, Envelope, fragment_envelope
from agentwire.text import clean_text, truncate_utf8


class IRCError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class IRCMessage:
    channel: str
    account: str
    nick: str
    text: str
    tags: Mapping[str, str | None] = field(default_factory=lambda: MappingProxyType({}))
    command: str = "PRIVMSG"


@dataclass(slots=True, frozen=True)
class IRCLine:
    tags: dict[str, str | None]
    prefix: str | None
    command: str
    params: tuple[str, ...]


@dataclass(slots=True)
class _IncomingBatch:
    channel: str = ""
    account: str = ""
    nick: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class _OutgoingMessage:
    target: str
    text: str = ""
    tags: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    command: str = "PRIVMSG"


def _unescape_tag(value: str) -> str:
    result: list[str] = []
    escaped = False
    replacements = {":": ";", "s": " ", "\\": "\\", "r": "\r", "n": "\n"}
    for char in value:
        if escaped:
            result.append(replacements.get(char, char))
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            result.append(char)
    if escaped:
        result.append("\\")
    return "".join(result)


def _escape_tag(value: str) -> str:
    replacements = {";": r"\:", " ": r"\s", "\\": r"\\", "\r": r"\r", "\n": r"\n"}
    return "".join(replacements.get(char, char) for char in value)


def _format_tags(tags: Mapping[str, str]) -> str:
    return "@" + ";".join(f"{key}={_escape_tag(value)}" for key, value in tags.items()) + " "


def parse_irc_line(raw: str) -> IRCLine:
    rest = raw.rstrip("\r\n")
    tags: dict[str, str | None] = {}
    prefix: str | None = None
    if rest.startswith("@"):
        tag_block, _, rest = rest.partition(" ")
        for pair in tag_block[1:].split(";"):
            key, separator, value = pair.partition("=")
            tags[key] = _unescape_tag(value) if separator else None
    if rest.startswith(":"):
        prefix, _, rest = rest[1:].partition(" ")
    middle, separator, trailing = rest.partition(" :")
    pieces = middle.split()
    if not pieces:
        raise IRCError("received an empty IRC line")
    params = pieces[1:]
    if separator:
        params.append(trailing)
    return IRCLine(tags=tags, prefix=prefix, command=pieces[0].upper(), params=tuple(params))


class IRCClient:
    def __init__(self, config: IRCConfig, password: str) -> None:
        self.config = config
        self.password = password
        self._messages: asyncio.Queue[IRCMessage] = asyncio.Queue()
        self._outgoing: asyncio.Queue[_OutgoingMessage] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._runner: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._caps: set[str] = set()
        self._joined: set[str] = set()
        self._batches: dict[str, _IncomingBatch] = {}

    async def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self._run(), name="irc-connection")

    async def wait_ready(self, timeout: float = 30) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except TimeoutError as exc:
            raise IRCError("timed out connecting and joining IRC") from exc

    async def close(self) -> None:
        self._closed = True
        self._ready.clear()
        current = asyncio.current_task()
        if self._runner is not None and self._runner is not current:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
        self._runner = None
        await self._close_connection()

    async def recv(self) -> IRCMessage:
        return await self._messages.get()

    async def send_privmsg(self, target: str, text: str) -> None:
        cleaned = clean_text(text)
        if not cleaned:
            return
        await self._outgoing.put(_OutgoingMessage(target, cleaned))

    async def send_tagmsg(self, target: str, tags: Mapping[str, str]) -> None:
        await self._outgoing.put(_OutgoingMessage(target, tags=tags, command="TAGMSG"))

    async def send_protocol(
        self, target: str, envelope: Envelope, preview: str | None = None
    ) -> None:
        fragments = fragment_envelope(envelope)
        first_tags = MappingProxyType({PROTOCOL_TAG: fragments[0]})
        if preview:
            await self._outgoing.put(_OutgoingMessage(target, clean_text(preview), first_tags))
        else:
            await self.send_tagmsg(target, first_tags)
        for fragment in fragments[1:]:
            await self.send_tagmsg(target, MappingProxyType({PROTOCOL_TAG: fragment}))

    async def _run(self) -> None:
        delay = 0.5
        while not self._closed:
            try:
                await self._connect_once()
                delay = 0.5
            except asyncio.CancelledError:
                raise
            except Exception:
                self._ready.clear()
                await self._close_connection()
                if self._closed:
                    return
                await asyncio.sleep(delay + random.random() * min(delay, 1))
                delay = min(delay * 2, 30)

    async def _connect_once(self) -> None:
        context = ssl.create_default_context(cafile=str(self.config.ca_file))
        reader, writer = await asyncio.open_connection(
            self.config.host,
            self.config.port,
            ssl=context,
            server_hostname=self.config.server_hostname,
        )
        self._writer = writer
        self._caps.clear()
        self._joined.clear()
        self._batches.clear()
        await self._write_line("CAP LS 302")
        await self._write_line(f"NICK {self.config.nickname}")
        await self._write_line(f"USER {self.config.username} 0 * :{self.config.realname}")
        await self._negotiate(reader)
        self._writer_task = asyncio.create_task(self._write_messages(), name="irc-writer")
        while not self._closed:
            raw = await reader.readline()
            if not raw:
                raise IRCError("IRC connection closed")
            line = parse_irc_line(raw.decode("utf-8", errors="replace"))
            await self._handle_line(line)

    async def _negotiate(self, reader: asyncio.StreamReader) -> None:
        offered: set[str] = set()
        cap_finished = False
        sasl_started = False
        sasl_complete = False
        welcome = False
        required = {
            "sasl",
            "account-tag",
            "message-tags",
            "server-time",
            "batch",
            "draft/multiline",
            "labeled-response",
            "echo-message",
            "standard-replies",
            "draft/chathistory",
            "draft/event-playback",
        }
        wanted = required | {"extended-join"}
        while not (welcome and cap_finished):
            raw = await reader.readline()
            if not raw:
                raise IRCError("IRC disconnected during registration")
            line = parse_irc_line(raw.decode("utf-8", errors="replace"))
            if line.command == "PING":
                await self._write_line(f"PONG :{line.params[-1]}")
                continue
            if line.command == "CAP" and len(line.params) >= 2:
                subcommand = line.params[1].upper()
                if subcommand == "LS":
                    offered.update(cap.split("=", 1)[0] for cap in line.params[-1].split())
                    multiline = len(line.params) >= 3 and line.params[-2] == "*"
                    if not multiline:
                        missing = required - offered
                        if missing:
                            names = ", ".join(sorted(missing))
                            raise IRCError(f"IRC server lacks required capabilities: {names}")
                        request = sorted(wanted & offered)
                        await self._write_line(f"CAP REQ :{' '.join(request)}")
                    continue
                if subcommand == "NAK":
                    raise IRCError("IRC server rejected required capabilities")
                if subcommand == "ACK":
                    self._caps.update(
                        cap.lstrip("-").split("=", 1)[0]
                        for cap in line.params[-1].split()
                        if not cap.startswith("-")
                    )
                    if required - self._caps:
                        raise IRCError("IRC server did not acknowledge required capabilities")
                    await self._write_line("AUTHENTICATE PLAIN")
                    sasl_started = True
                    continue
            if line.command == "AUTHENTICATE" and sasl_started and line.params[-1] == "+":
                payload = base64.b64encode(
                    f"{self.config.nickname}\0{self.config.nickname}\0{self.password}".encode()
                ).decode()
                for offset in range(0, len(payload), 400):
                    await self._write_line(f"AUTHENTICATE {payload[offset : offset + 400]}")
                if len(payload) % 400 == 0:
                    await self._write_line("AUTHENTICATE +")
                continue
            if line.command == "903":
                sasl_complete = True
                await self._write_line("CAP END")
                cap_finished = True
                continue
            if line.command in {"904", "905", "906", "907"}:
                raise IRCError("IRC SASL authentication failed")
            if line.command == "001":
                welcome = True
        if not sasl_complete:
            raise IRCError("IRC registration completed without SASL")
        await self._write_line(f"MODE {self.config.nickname} +B")
        for channel in self.config.channels:
            await self._write_line(f"JOIN {channel}")
        while self._joined != set(self.config.channels):
            raw = await reader.readline()
            if not raw:
                raise IRCError("IRC disconnected while joining channels")
            line = parse_irc_line(raw.decode("utf-8", errors="replace"))
            if line.command == "PING":
                await self._write_line(f"PONG :{line.params[-1]}")
            elif line.command == "JOIN" and self._nick(line.prefix) == self.config.nickname:
                channel = line.params[0].lower()
                if channel in self.config.channels:
                    self._joined.add(channel)
                else:
                    await self._write_line(f"PART {channel} :not a bridge channel")
            elif line.command in {"403", "405", "471", "473", "474", "475", "477"}:
                raise IRCError(f"IRC could not join a configured channel ({line.command})")
        self._ready.set()

    async def _handle_line(self, line: IRCLine) -> None:
        if line.command == "PING":
            await self._write_line(f"PONG :{line.params[-1]}")
            return
        nick = self._nick(line.prefix)
        if line.command == "KICK" and len(line.params) >= 2:
            channel, target = line.params[:2]
            if target.lower() == self.config.nickname.lower():
                self._joined.discard(channel.lower())
                await self._write_line(f"JOIN {channel}")
                self._ready.clear()
            return
        if line.command == "JOIN" and nick.lower() == self.config.nickname.lower():
            channel = line.params[0].lower()
            if channel in self.config.channels:
                self._joined.add(channel)
                if self._joined == set(self.config.channels):
                    self._ready.set()
            else:
                await self._write_line(f"PART {channel} :not a bridge channel")
            return
        if line.command == "BATCH" and line.params:
            batch_id = line.params[0]
            if PROTOCOL_TAG in line.tags and len(line.params) >= 3 and batch_id.startswith("+"):
                channel = line.params[2].lower()
                if channel in self.config.channels:
                    await self._messages.put(
                        IRCMessage(
                            channel=channel,
                            account=str(line.tags.get("account") or "").lower(),
                            nick=nick,
                            text="",
                            tags=MappingProxyType(dict(line.tags)),
                            command="BATCH",
                        )
                    )
            if (
                batch_id.startswith("+")
                and len(line.params) >= 2
                and line.params[1] == "draft/multiline"
            ):
                self._batches[batch_id[1:]] = _IncomingBatch()
            elif batch_id.startswith("-"):
                batch = self._batches.pop(batch_id[1:], None)
                if batch and batch.channel and batch.lines:
                    await self._messages.put(
                        IRCMessage(
                            channel=batch.channel,
                            account=batch.account,
                            nick=batch.nick,
                            text="\n".join(batch.lines),
                        )
                    )
            return
        if line.command in {"TOPIC", "331", "332"}:
            if line.command == "332" and len(line.params) >= 3:
                channel, text = line.params[-2].lower(), line.params[-1]
            elif line.command == "331" and len(line.params) >= 2:
                channel = next(
                    (param.lower() for param in line.params if param.startswith("#")), ""
                )
                text = ""
            elif len(line.params) >= 2:
                channel, text = line.params[0].lower(), line.params[1]
            else:
                return
            if channel in self.config.channels:
                await self._messages.put(
                    IRCMessage(
                        channel,
                        str(line.tags.get("account") or "").lower(),
                        nick,
                        text,
                        MappingProxyType(dict(line.tags)),
                        line.command,
                    )
                )
            return
        if line.command not in {"PRIVMSG", "TAGMSG"}:
            return
        if len(line.params) < (2 if line.command == "PRIVMSG" else 1):
            return
        channel = line.params[0].lower()
        if channel not in self.config.channels:
            return
        account = str(line.tags.get("account") or "").lower()
        message = IRCMessage(
            channel=channel,
            account=account,
            nick=nick,
            text=line.params[1] if line.command == "PRIVMSG" else "",
            tags=MappingProxyType(dict(line.tags)),
            command=line.command,
        )
        batch_id = line.tags.get("batch")
        if isinstance(batch_id, str) and batch_id in self._batches:
            batch = self._batches[batch_id]
            if not batch.channel:
                batch.channel, batch.account, batch.nick = channel, account, nick
            if (batch.channel, batch.account, batch.nick) == (channel, account, nick):
                batch.lines.append(line.params[1])
            return
        await self._messages.put(message)

    async def _write_messages(self) -> None:
        while not self._closed:
            message = await self._outgoing.get()
            target, text = message.target, message.text
            if message.command == "TAGMSG":
                await self._write_line(f"{_format_tags(message.tags)}TAGMSG {target}")
                continue
            lines: list[str] = []
            for logical in text.splitlines() or [text]:
                remaining = logical
                while len(remaining.encode("utf-8")) > 350:
                    piece = truncate_utf8(remaining, 350)
                    lines.append(piece)
                    remaining = remaining[len(piece) :]
                lines.append(remaining or " ")
            if len(lines) > 1 and {"batch", "draft/multiline"} <= self._caps:
                batch_id = f"bridge-{secrets.token_hex(4)}"
                tag_prefix = _format_tags(message.tags) if message.tags else ""
                await self._write_line(f"{tag_prefix}BATCH +{batch_id} draft/multiline {target}")
                for item in lines:
                    await self._write_line(f"@batch={batch_id} PRIVMSG {target} :{item}")
                    await asyncio.sleep(0.15)
                await self._write_line(f"BATCH -{batch_id}")
            else:
                for item in lines:
                    tag_prefix = _format_tags(message.tags) if message.tags else ""
                    await self._write_line(f"{tag_prefix}PRIVMSG {target} :{item}")
                    await asyncio.sleep(0.15)

    async def _write_line(self, line: str) -> None:
        if self._writer is None or self._writer.is_closing():
            raise IRCError("IRC is not connected")
        self._writer.write(f"{line}\r\n".encode())
        await self._writer.drain()

    async def _close_connection(self) -> None:
        if self._writer_task is not None:
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task
        self._writer_task = None
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._writer = None

    @staticmethod
    def _nick(prefix: str | None) -> str:
        return (prefix or "").split("!", 1)[0]
