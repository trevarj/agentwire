from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import random
import secrets
import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from agentwire.config import IRCConfig
from agentwire.diagnostics import DiagnosticCheck
from agentwire.protocol import PROTOCOL_TAG, Envelope, fragment_envelope
from agentwire.text import clean_text

# Diagnostics describe identities, classifications, and lifecycle transitions.
# They never carry message text, tag values, or credentials: IRC traffic is the
# payload this bridge is trusted with, and a log file is not a secret store.
LOGGER = logging.getLogger("agentwire.irc")
REQUIRED_CAPABILITIES = frozenset(
    {
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
)


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


@dataclass(slots=True)
class _ControlWaiter:
    kind: str
    channel: str
    future: asyncio.Future[None]
    topic: str = ""
    nick: str = ""


@dataclass(slots=True, frozen=True)
class _OutgoingMessage:
    target: str
    text: str = ""
    tags: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    command: str = "PRIVMSG"
    queued_at: float = field(default_factory=time.monotonic)
    done: asyncio.Future[None] | None = None


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
        # The account the server confirms for this connection, which is what
        # tags every message the bridge publishes. It is not necessarily the
        # configured nickname, so activation validates against this value.
        self.account = ""
        self._messages: asyncio.Queue[IRCMessage] = asyncio.Queue()
        self._outgoing: asyncio.Queue[_OutgoingMessage] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._runner: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._caps: set[str] = set()
        self._channel_order = [self._valid_channel(channel) for channel in config.channels]
        self._accepted_channels = set(self._channel_order)
        self._joined: set[str] = set()
        self._control_lock = asyncio.Lock()
        self._control_waiter: _ControlWaiter | None = None
        self._write_lock = asyncio.Lock()
        self._flush_waiters: set[asyncio.Future[None]] = set()
        # Channels whose topic the server has answered, with 331 or 332.
        self._topics: set[str] = set()
        self._batches: dict[str, _IncomingBatch] = {}

    def diagnostic_snapshot(self) -> list[DiagnosticCheck]:
        connected = self._writer is not None and not self._writer.is_closing()
        authenticated = connected and bool(self.account)
        capabilities_ready = REQUIRED_CAPABILITIES - {"echo-message"} <= self._caps
        # asyncio.Queue has no public peek. Read only its oldest item on this
        # event loop, without dequeueing or affecting transport ordering.
        oldest = self._outgoing._queue[0] if self._outgoing.qsize() else None
        age = max(0, int((time.monotonic() - oldest.queued_at) * 1000)) if oldest else 0
        return [
            DiagnosticCheck(
                "irc.connection",
                "ok" if connected else "warning",
                "IRC is connected." if connected else "IRC is disconnected.",
                {"connected": connected},
            ),
            DiagnosticCheck(
                "irc.authentication",
                "ok" if authenticated else "warning",
                "IRC authentication is confirmed."
                if authenticated
                else "IRC authentication is unconfirmed.",
                {"authenticated": authenticated},
            ),
            DiagnosticCheck(
                "irc.capabilities",
                "ok" if connected and capabilities_ready else "warning",
                "Required IRC capabilities are ready."
                if connected and capabilities_ready
                else "Required IRC capabilities are unavailable.",
                {
                    "capabilitiesReady": connected and capabilities_ready,
                    "ready": self._ready.is_set(),
                },
            ),
            DiagnosticCheck(
                "irc.outgoing",
                "warning" if age >= 10000 else "ok",
                "Outgoing messages are queued." if oldest else "Outgoing queue is empty.",
                {"queueDepth": self._outgoing.qsize(), "oldestQueuedMs": age},
            ),
        ]

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

    async def send_notice(self, target: str, text: str) -> None:
        cleaned = clean_text(text)
        if not cleaned:
            return
        await self._outgoing.put(_OutgoingMessage(target, cleaned, command="NOTICE"))

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

    def register_channel(self, channel: str) -> None:
        value = self._valid_channel(channel)
        if value not in self._accepted_channels:
            self._channel_order.append(value)
            self._accepted_channels.add(value)
            self._ready.clear()

    def unregister_channel(self, channel: str) -> None:
        value = self._valid_channel(channel)
        self._accepted_channels.discard(value)
        self._channel_order = [channel for channel in self._channel_order if channel != value]
        self._joined.discard(value)
        self._topics.discard(value)
        if (
            self._writer_task is not None
            and not self._writer_task.done()
            and self._joined == self._topics == self._accepted_channels
        ):
            self._ready.set()

    async def join_channel(self, channel: str, timeout: float = 10) -> None:
        value = self._registered_channel(channel)
        if value in self._joined:
            return
        await self._control("join", value, (f"JOIN {value}",), timeout)

    async def set_private(self, channel: str, timeout: float = 10) -> None:
        value = self._registered_channel(channel)
        await self._control(
            "mode",
            value,
            (f"MODE {value} +is", f"MODE {value}"),
            timeout,
        )

    async def set_topic(self, channel: str, topic: str, timeout: float = 10) -> None:
        value = self._registered_channel(channel)
        if "\r" in topic or "\n" in topic:
            raise IRCError("IRC topic contains a line break")
        await self._control("topic", value, (f"TOPIC {value} :{topic}",), timeout, topic=topic)
        if self._joined == self._topics == self._accepted_channels:
            self._ready.set()

    async def invite(self, channel: str, nick: str, timeout: float = 10) -> None:
        value = self._registered_channel(channel)
        target = self._valid_nick(nick)
        await self._control(
            "invite",
            value,
            (f"INVITE {target} {value}",),
            timeout,
            nick=target,
        )

    async def _control(
        self,
        kind: str,
        channel: str,
        commands: tuple[str, ...],
        timeout: float,
        *,
        topic: str = "",
        nick: str = "",
    ) -> None:
        async with self._control_lock:
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._control_waiter = _ControlWaiter(kind, channel, future, topic, nick)
            try:
                for command in commands:
                    await self._write_line(command)
                await asyncio.wait_for(future, timeout)
            except TimeoutError as exc:
                raise IRCError(f"timed out waiting for {kind} confirmation in {channel}") from exc
            finally:
                self._control_waiter = None

    async def flush(self) -> None:
        if self._writer_task is None or self._writer_task.done():
            raise IRCError("IRC writer is not running")
        future = asyncio.get_running_loop().create_future()
        self._flush_waiters.add(future)
        try:
            await self._outgoing.put(_OutgoingMessage("", command="FLUSH", done=future))
            await future
        finally:
            self._flush_waiters.discard(future)

    async def part_channel(self, channel: str, timeout: float = 10) -> None:
        value = self._registered_channel(channel)
        await self._control("part", value, (f"PART {value} :session closed",), timeout)
        self.unregister_channel(value)

    async def _run(self) -> None:
        delay = 0.5
        while not self._closed:
            try:
                await self._connect_once()
                delay = 0.5
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._ready.clear()
                await self._close_connection()
                if self._closed:
                    return
                # Only IRCError text is this module's own wording; any other
                # exception is reported by class alone so no server or socket
                # detail is copied into the log.
                reason = str(exc) if isinstance(exc, IRCError) else ""
                LOGGER.warning(
                    "IRC connection failed (%s%s); reconnecting in about %.1fs",
                    type(exc).__name__,
                    f": {reason}" if reason else "",
                    delay,
                )
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
        self._topics.clear()
        self._batches.clear()
        # The account is re-confirmed by SASL on every connection.
        self.account = ""
        await self._write_line("CAP LS 302")
        await self._write_line(f"NICK {self.config.nickname}")
        await self._write_line(f"USER {self.config.username} 0 * :{self.config.realname}")
        await self._negotiate(reader)
        reader_task = asyncio.create_task(self._read_messages(reader), name="irc-reader")
        writer_task = asyncio.create_task(self._write_messages(), name="irc-writer")
        self._writer_task = writer_task
        if self._joined == self._topics == self._accepted_channels:
            self._ready.set()
        try:
            done, _pending = await asyncio.wait(
                {reader_task, writer_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            reader_task.cancel()
            writer_task.cancel()
            await asyncio.gather(reader_task, writer_task, return_exceptions=True)

    async def _read_messages(self, reader: asyncio.StreamReader) -> None:
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
        required = REQUIRED_CAPABILITIES
        # Bridge must prove clients can use echo-message, but enabling it on
        # this connection only sends every published event back to be discarded.
        enabled_required = required - {"echo-message"}
        wanted = enabled_required | {"extended-join"}
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
                    if enabled_required - self._caps:
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
            if line.command == "900" and len(line.params) >= 3:
                # RPL_LOGGEDIN names the account the server actually granted,
                # which is the account tag every message of ours will carry.
                self._note_account(line.params[2], "SASL login")
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
        LOGGER.info(
            "IRC registered as nick %s, account %s",
            self.config.nickname,
            self.account or "<unconfirmed>",
        )
        await self._write_line(f"MODE {self.config.nickname} +B")
        for channel in self._channel_order:
            await self._write_line(f"JOIN {channel}")
            # A server sends RPL_TOPIC unsolicited only when a topic is set, and
            # sends nothing at all for a channel that has none. Activation is
            # driven entirely by the topic, so waiting for a line the server was
            # never obliged to send left such a channel indistinguishable from
            # one whose topic had not arrived yet: silent, suspended, forever.
            # Asking explicitly guarantees exactly one 331 or 332 per channel.
            await self._write_line(f"TOPIC {channel}")
        expected = set(self._accepted_channels)
        while self._joined != expected or self._topics != expected:
            raw = await reader.readline()
            if not raw:
                raise IRCError("IRC disconnected while joining channels")
            line = parse_irc_line(raw.decode("utf-8", errors="replace"))
            if line.command in {"403", "405", "471", "473", "474", "475", "477"}:
                raise IRCError(f"IRC could not join a configured channel ({line.command})")
            # Every other line goes through the normal dispatcher, so a topic
            # reply that arrives while a later channel is still joining reaches
            # the bridge instead of being consumed and discarded here. Readiness
            # waits for the topic of every channel, not merely the last JOIN
            # echo, so no reply is left unread behind the loop either.
            await self._handle_line(line)

    def _note_account(self, account: str, source: str) -> None:
        value = account.strip().lower()
        if not value or value == self.account:
            return
        if self.account:
            LOGGER.warning("IRC account changed from %s to %s (%s)", self.account, value, source)
        else:
            LOGGER.info("IRC authenticated account is %s (%s)", value, source)
        self.account = value

    async def _handle_line(self, line: IRCLine) -> None:
        if line.command == "PING":
            await self._write_line(f"PONG :{line.params[-1]}")
            return
        nick = self._nick(line.prefix)
        if line.command == "900" and len(line.params) >= 3:
            self._note_account(line.params[2], "SASL login")
            return
        if nick.lower() == self.config.nickname.lower():
            # echo-message returns the bridge's own traffic carrying the account
            # tag the server attributed it to, which is the identity clients
            # authenticate events against.
            self._note_account(str(line.tags.get("account") or ""), "own echoed message")
        if self._handle_control_reply(line, nick):
            return
        if line.command == "KICK" and len(line.params) >= 2:
            channel, target = line.params[:2]
            value = channel.lower()
            if target.lower() == self.config.nickname.lower() and value in self._accepted_channels:
                LOGGER.warning("kicked from %s; rejoining", value)
                self._joined.discard(value)
                self._topics.discard(value)
                await self._write_line(f"JOIN {value}")
                # Re-ask, for the same reason the join sequence asks.
                await self._write_line(f"TOPIC {value}")
                self._ready.clear()
            return
        if line.command == "JOIN" and nick.lower() == self.config.nickname.lower():
            channel = line.params[0].lower()
            if channel in self._accepted_channels:
                LOGGER.info("joined %s", channel)
                self._joined.add(channel)
                if (
                    self._writer_task is not None
                    and not self._writer_task.done()
                    and self._joined == self._topics == self._accepted_channels
                ):
                    self._ready.set()
            else:
                LOGGER.info("parting %s: not a configured bridge channel", channel)
                await self._write_line(f"PART {channel} :not a bridge channel")
            return
        if line.command == "BATCH" and line.params:
            batch_id = line.params[0]
            if PROTOCOL_TAG in line.tags and len(line.params) >= 3 and batch_id.startswith("+"):
                channel = line.params[2].lower()
                if channel in self._accepted_channels:
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
            if channel in self._accepted_channels:
                LOGGER.info(
                    "%s: topic reply %s received (%s)",
                    channel,
                    line.command,
                    "no topic is set" if line.command == "331" else "topic present",
                )
                self._topics.add(channel)
                if (
                    self._writer_task is not None
                    and not self._writer_task.done()
                    and self._joined == self._topics == self._accepted_channels
                ):
                    self._ready.set()
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
        if channel not in self._accepted_channels:
            if PROTOCOL_TAG in line.tags:
                LOGGER.warning(
                    "dropped a protocol %s from %s: %s is not a configured bridge channel",
                    line.command,
                    nick or "<server>",
                    channel,
                )
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
            try:
                if message.command == "FLUSH":
                    if message.done is not None and not message.done.done():
                        message.done.set_result(None)
                    continue
                target, text = message.target, message.text
                LOGGER.debug(
                    "outgoing %s to %s waited %.1f ms (queue depth %d)",
                    message.command,
                    target,
                    (time.monotonic() - message.queued_at) * 1000,
                    self._outgoing.qsize(),
                )
                if message.command == "TAGMSG":
                    await self._write_line(f"{_format_tags(message.tags)}TAGMSG {target}")
                    continue
                lines: list[str] = []
                for logical in text.splitlines() or [text]:
                    # Split on the encoded bytes so a long line costs one
                    # encode instead of re-encoding the remainder per piece.
                    encoded = logical.encode("utf-8")
                    while len(encoded) > 350:
                        piece = encoded[:350].decode("utf-8", errors="ignore")
                        lines.append(piece)
                        encoded = encoded[len(piece.encode("utf-8")) :]
                    lines.append(encoded.decode("utf-8") or " ")
                # A multiline batch carries PRIVMSG lines, so only a PRIVMSG may use
                # it; a NOTICE is always sent as discrete lines.
                if (
                    len(lines) > 1
                    and message.command == "PRIVMSG"
                    and {"batch", "draft/multiline"} <= self._caps
                ):
                    batch_id = f"bridge-{secrets.token_hex(4)}"
                    tag_prefix = _format_tags(message.tags) if message.tags else ""
                    await self._write_line(
                        f"{tag_prefix}BATCH +{batch_id} draft/multiline {target}"
                    )
                    for item in lines:
                        await self._write_line(f"@batch={batch_id} PRIVMSG {target} :{item}")
                    await self._write_line(f"BATCH -{batch_id}")
                else:
                    for item in lines:
                        tag_prefix = _format_tags(message.tags) if message.tags else ""
                        await self._write_line(f"{tag_prefix}{message.command} {target} :{item}")
            finally:
                self._outgoing.task_done()

    async def _write_line(self, line: str) -> None:
        async with self._write_lock:
            if self._writer is None or self._writer.is_closing():
                raise IRCError("IRC is not connected")
            self._writer.write(f"{line}\r\n".encode())
            await self._writer.drain()

    def _handle_control_reply(self, line: IRCLine, nick: str) -> bool:
        waiter = self._control_waiter
        if waiter is None or waiter.future.done():
            return False
        channel = next((param.lower() for param in line.params if param.startswith("#")), "")
        if waiter.kind == "join" and line.command == "JOIN":
            if nick.lower() == self.config.nickname.lower() and channel == waiter.channel:
                self._joined.add(channel)
                waiter.future.set_result(None)
                return True
        elif waiter.kind == "mode" and line.command == "324" and channel == waiter.channel:
            index = next(i for i, param in enumerate(line.params) if param.lower() == channel)
            modes = line.params[index + 1] if index + 1 < len(line.params) else ""
            enabled: set[str] = set()
            adding = True
            for char in modes:
                if char == "+":
                    adding = True
                elif char == "-":
                    adding = False
                elif char in {"i", "s"}:
                    (enabled.add if adding else enabled.discard)(char)
            if enabled == {"i", "s"}:
                waiter.future.set_result(None)
            else:
                waiter.future.set_exception(IRCError("IRC did not apply private channel modes"))
            return True
        elif waiter.kind == "topic" and line.command == "TOPIC":
            if (
                nick.lower() == self.config.nickname.lower()
                and channel == waiter.channel
                and len(line.params) >= 2
                and line.params[-1] == waiter.topic
            ):
                self._topics.add(channel)
                waiter.future.set_result(None)
                return True
        elif waiter.kind == "invite" and line.command == "341":
            if channel == waiter.channel and any(
                param.lower() == waiter.nick.lower() for param in line.params
            ):
                waiter.future.set_result(None)
                return True
        elif (
            waiter.kind == "part"
            and line.command == "PART"
            and nick.lower() == self.config.nickname.lower()
            and channel == waiter.channel
        ):
            self._joined.discard(channel)
            self._topics.discard(channel)
            waiter.future.set_result(None)
            return True

        errors = {
            "join": {"403", "405", "471", "473", "474", "475", "477"},
            "mode": {"403", "442", "472", "482"},
            "topic": {"403", "442", "482"},
            "invite": {"401", "403", "442", "443", "482"},
            "part": {"403", "442"},
        }
        if line.command not in errors[waiter.kind]:
            return False
        if channel and channel != waiter.channel:
            return False
        if (
            waiter.kind == "invite"
            and line.command == "401"
            and not any(param.lower() == waiter.nick.lower() for param in line.params)
        ):
            return False
        waiter.future.set_exception(IRCError(f"IRC rejected {waiter.kind} ({line.command})"))
        return True

    async def _close_connection(self) -> None:
        waiter = self._control_waiter
        if waiter is not None and not waiter.future.done():
            waiter.future.set_exception(IRCError("IRC connection closed before control reply"))
        for flush_waiter in self._flush_waiters:
            if not flush_waiter.done():
                flush_waiter.set_exception(
                    IRCError("IRC connection closed before outgoing messages flushed")
                )
        if self._writer_task is not None:
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
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

    @staticmethod
    def _valid_channel(channel: str) -> str:
        value = channel.lower()
        if (
            len(value.encode("utf-8")) > 50
            or not value.startswith("#")
            or len(value) == 1
            or any(ord(char) < 0x21 or char in ",:" for char in value)
        ):
            raise IRCError("invalid IRC channel")
        return value

    def _registered_channel(self, channel: str) -> str:
        value = self._valid_channel(channel)
        if value not in self._accepted_channels:
            raise IRCError(f"IRC channel is not registered: {value}")
        return value

    @staticmethod
    def _valid_nick(nick: str) -> str:
        if (
            not nick
            or len(nick.encode("utf-8")) > 30
            or nick[0].isdigit()
            or any(ord(char) < 0x21 or char in " ,:*?!@." for char in nick)
        ):
            raise IRCError("invalid IRC nickname")
        return nick
