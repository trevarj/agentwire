from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = 1
PROTOCOL_TAG = "+trevarj.github.io/agentwire"
TOPIC_PREFIX = "agentwire:v1;"
MAX_TAG_SECTION_BYTES = 4094
MAX_TAG_VALUE_BYTES = MAX_TAG_SECTION_BYTES - len(PROTOCOL_TAG.encode("utf-8")) - 1
MAX_PAYLOAD_BYTES = 128 * 1024
MAX_FRAGMENTS = 64
MAX_INFLIGHT_MESSAGES = 16
MAX_INFLIGHT_BYTES = 2 * 1024 * 1024
FRAGMENT_TIMEOUT_SECONDS = 30.0

ACTION_KINDS = frozenset(
    {
        "sync.request",
        "workspace.list.request",
        "session.list.request",
        "history.request",
        "session.create",
        "session.attach",
        "session.detach",
        "session.rename",
        "session.fork",
        "session.archive",
        "session.unarchive",
        "settings.update",
        "turn.prompt",
        "turn.steer",
        "turn.cancel",
        "queue.edit",
        "queue.move",
        "queue.delete",
        "queue.clear",
        "request.respond",
        "request.skip",
    }
)

EVENT_KINDS = frozenset(
    {
        "agent.hello",
        "channel.snapshot",
        "binding.changed",
        "session.snapshot",
        "session.status",
        "workspace.page",
        "session.page",
        "history.begin",
        "history.end",
        "action.accepted",
        "action.succeeded",
        "action.failed",
        "action.uncertain",
        "queue.snapshot",
        "queue.item.added",
        "queue.item.updated",
        "queue.item.moved",
        "queue.item.removed",
        "user.prompt",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "assistant.delta",
        "assistant.completed",
        "plan.updated",
        "tool.started",
        "tool.updated",
        "tool.completed",
        "usage.updated",
        "request.opened",
        "request.resolved",
        "approval.review.started",
        "approval.review.completed",
    }
)

# Only durable transcript and request lifecycle events belong in history replay.
# Discovery, snapshots, acknowledgements, and queue state are rebuilt by sync.
HISTORY_EVENT_KINDS = frozenset(
    {
        "turn.started",
        "user.prompt",
        "turn.completed",
        "turn.failed",
        "assistant.delta",
        "assistant.completed",
        "plan.updated",
        "tool.started",
        "tool.updated",
        "tool.completed",
        "usage.updated",
        "request.opened",
        "request.resolved",
        "approval.review.started",
        "approval.review.completed",
    }
)

VISIBLE_EVENT_KINDS = frozenset(
    {
        "binding.changed",
        "action.failed",
        "action.uncertain",
        "turn.failed",
        "assistant.completed",
        "request.opened",
        "request.resolved",
        "queue.item.updated",
        "queue.item.removed",
    }
)


class ProtocolError(ValueError):
    """A malformed or out-of-policy Agentwire protocol message."""


@dataclass(slots=True, frozen=True)
class TopicActivation:
    account: str
    # The controller (`account`) authorizes actions; the agent account is the
    # separate identity trusted to publish backend state. Clients authenticate
    # events against `agent`, so a topic without it cannot activate a harness.
    agent: str
    backend: str
    title: str = ""
    options: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Envelope:
    kind: str
    message_type: Literal["action", "event"]
    id: str
    at: int
    instance: str
    epoch: str | None = None
    device: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    item_id: str | None = None
    request_id: str | None = None
    revision: int | None = None
    reply: str | None = None
    history: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "k": self.kind,
            "t": self.message_type,
            "id": self.id,
            "at": self.at,
            "inst": self.instance,
        }
        optional = {
            "epoch": self.epoch,
            "device": self.device,
            "sid": self.session_id,
            "tid": self.turn_id,
            "iid": self.item_id,
            "rid": self.request_id,
            "rev": self.revision,
            "reply": self.reply,
        }
        result.update({key: value for key, value in optional.items() if value is not None})
        if self.history:
            result["hist"] = True
        if self.data:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Envelope:
        if value.get("v") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported protocol version")
        kind = _required_string(value, "k")
        message_type = _required_string(value, "t")
        if message_type not in {"action", "event"}:
            raise ProtocolError("t must be action or event")
        allowed = ACTION_KINDS if message_type == "action" else EVENT_KINDS
        if kind not in allowed:
            raise ProtocolError(f"unsupported {message_type} kind: {kind}")
        message_id = _required_string(value, "id")
        try:
            uuid.UUID(message_id)
        except ValueError as exc:
            raise ProtocolError("id must be a UUID") from exc
        at = value.get("at")
        if not isinstance(at, int) or isinstance(at, bool) or at < 0:
            raise ProtocolError("at must be a non-negative integer timestamp")
        instance = _required_string(value, "inst")
        data = value.get("data", {})
        if not isinstance(data, dict):
            raise ProtocolError("data must be an object")
        revision = value.get("rev")
        if revision is not None and (
            not isinstance(revision, int) or isinstance(revision, bool) or revision < 0
        ):
            raise ProtocolError("rev must be a non-negative integer")
        history = value.get("hist", False)
        if not isinstance(history, bool):
            raise ProtocolError("hist must be a boolean")
        return cls(
            kind=kind,
            message_type=message_type,  # type: ignore[arg-type]
            id=message_id,
            at=at,
            instance=instance,
            epoch=_optional_string(value, "epoch"),
            device=_optional_string(value, "device"),
            session_id=_optional_string(value, "sid"),
            turn_id=_optional_string(value, "tid"),
            item_id=_optional_string(value, "iid"),
            request_id=_optional_string(value, "rid"),
            revision=revision,
            reply=_optional_string(value, "reply"),
            history=history,
            data=data,
        )


def new_envelope(
    kind: str,
    message_type: Literal["action", "event"],
    instance: str,
    **values: Any,
) -> Envelope:
    return Envelope(
        kind=kind,
        message_type=message_type,
        id=values.pop("id", str(uuid.uuid4())),
        at=values.pop("at", int(time.time() * 1000)),
        instance=instance,
        **values,
    )


def encode_envelope(envelope: Envelope) -> str:
    raw = json.dumps(
        envelope.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    return raw.decode("utf-8")


def decode_envelope(value: str) -> Envelope:
    if len(value.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError("protocol value must be a JSON object")
    return Envelope.from_dict(decoded)


def tag_wire_size(value: str) -> int:
    """Return the UTF-8 byte size after IRCv3 message-tag escaping."""
    escaped = value.replace("\\", "\\\\").replace(";", "\\:").replace(" ", "\\s")
    escaped = escaped.replace("\r", "\\r").replace("\n", "\\n")
    return len(escaped.encode("utf-8"))


def parse_topic(topic: str) -> TopicActivation | None:
    if not topic.startswith(TOPIC_PREFIX):
        return None
    header, separator, title = topic.partition(" | ")
    pairs = header[len(TOPIC_PREFIX) :].split(";")
    options: dict[str, str] = {}
    for pair in pairs:
        key, equals, value = pair.partition("=")
        if not equals or not key or not value:
            raise ProtocolError("invalid Agentwire topic parameter")
        if key in options:
            raise ProtocolError(f"duplicate Agentwire topic parameter: {key}")
        options[key] = urllib.parse.unquote(value)
    account = options.get("account", "").lower()
    agent = options.get("agent", "").lower()
    # Backend names are a closed lowercase set and the bridge compares this
    # against the configured channel backend with ==. Folding here keeps a
    # topic reading "backend=Claude" from suspending the channel: activation
    # would stay None, every action would be dropped before reaching the
    # journal, and the client would see nothing but an endless sync.
    backend = options.get("backend", "").lower()
    # Naming the absent fields is what makes the v1 upgrade actionable. A topic
    # written before `agent` became required fails here, and its operator needs
    # to be told which field to add, not that three fields are required.
    missing = [
        name
        for name, value in (("account", account), ("agent", agent), ("backend", backend))
        if not value
    ]
    if missing:
        raise ProtocolError("topic is missing " + ", ".join(f"{name}=" for name in missing))
    return TopicActivation(account, agent, backend, title if separator else "", options)


def suggested_topic(topic: str, *, account: str, agent: str, backend: str) -> str:
    """Rebuild a correct activation topic, preserving any human title.

    The result is what the deployment's own configuration says the topic should
    be, so an operator can paste it verbatim to repair a rejected one.
    """

    return build_topic(account, backend, topic.partition(" | ")[2], agent=agent)


def build_topic(account: str, backend: str, title: str = "", *, agent: str | None = None) -> str:
    """Build an activation topic.

    ``account`` and ``agent`` are IRC account names, never engine names:
    ``account`` is the owner whose commands the bridge obeys and ``agent`` is
    the bot account whose messages clients trust as backend state. Only
    ``backend`` names the engine (``codex``, ``opencode``, ``claude``, or ``pi``);
    ``agent="claude"`` would mean an IRC account literally named claude.
    ``agent`` defaults to ``account``: the supported single-account deployment
    shape, where one SASL identity both issues commands and publishes state.
    """

    def quote(value: str) -> str:
        return urllib.parse.quote(value, safe="-._~")

    topic = (
        f"{TOPIC_PREFIX}account={quote(account.lower())};"
        f"agent={quote((agent or account).lower())};backend={quote(backend)}"
    )
    return f"{topic} | {title}" if title else topic


def fragment_envelope(envelope: Envelope, max_tag_bytes: int = MAX_TAG_VALUE_BYTES) -> list[str]:
    encoded = encode_envelope(envelope)
    if tag_wire_size(encoded) <= max_tag_bytes:
        return [encoded]
    raw = encoded.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    encoded_payload = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    # Reserve enough room for metadata and varying integer widths.
    chunk_size = max_tag_bytes - 360
    if chunk_size <= 0:
        raise ProtocolError("fragment tag budget is too small")
    chunks = [
        encoded_payload[i : i + chunk_size] for i in range(0, len(encoded_payload), chunk_size)
    ]
    if len(chunks) > MAX_FRAGMENTS:
        raise ProtocolError(f"payload requires more than {MAX_FRAGMENTS} fragments")
    result: list[str] = []
    for part, chunk in enumerate(chunks):
        fragment = {
            "v": PROTOCOL_VERSION,
            "k": "fragment",
            "id": envelope.id,
            "of": envelope.kind,
            "t": envelope.message_type,
            "epoch": envelope.epoch,
            "sid": envelope.session_id,
            "part": part,
            "parts": len(chunks),
            "bytes": len(raw),
            "sha256": digest,
            "b64": chunk,
        }
        value = json.dumps(
            {key: value for key, value in fragment.items() if value is not None},
            sort_keys=True,
            separators=(",", ":"),
        )
        if tag_wire_size(value) > max_tag_bytes:
            raise ProtocolError("fragment exceeds tag budget")
        result.append(value)
    return result


@dataclass(slots=True)
class _PartialMessage:
    created: float
    parts: int
    size: int
    sha256: str
    kind: str
    message_type: str
    epoch: str | None
    session_id: str | None
    chunks: dict[int, str] = field(default_factory=dict)


class Reassembler:
    def __init__(self) -> None:
        self._messages: dict[str, _PartialMessage] = {}

    def add(self, value: str, now: float | None = None) -> Envelope | None:
        current = time.monotonic() if now is None else now
        self.expire(current)
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict) or raw.get("k") != "fragment":
            return decode_envelope(value)
        if raw.get("v") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported fragment version")
        message_id = _required_string(raw, "id")
        parts = raw.get("parts")
        part = raw.get("part")
        size = raw.get("bytes")
        digest = _required_string(raw, "sha256")
        kind = _required_string(raw, "of")
        message_type = _required_string(raw, "t")
        epoch = _optional_string(raw, "epoch")
        session_id = _optional_string(raw, "sid")
        chunk = _required_string(raw, "b64")
        if not isinstance(parts, int) or not 2 <= parts <= MAX_FRAGMENTS:
            raise ProtocolError("invalid fragment count")
        if not isinstance(part, int) or not 0 <= part < parts:
            raise ProtocolError("invalid fragment index")
        if not isinstance(size, int) or not 1 <= size <= MAX_PAYLOAD_BYTES:
            raise ProtocolError("invalid reconstructed byte count")
        existing = self._messages.get(message_id)
        if existing is None:
            if len(self._messages) >= MAX_INFLIGHT_MESSAGES:
                raise ProtocolError("too many in-flight fragmented messages")
            if sum(item.size for item in self._messages.values()) + size > MAX_INFLIGHT_BYTES:
                raise ProtocolError("fragment memory budget exceeded")
            existing = _PartialMessage(
                current, parts, size, digest, kind, message_type, epoch, session_id
            )
            self._messages[message_id] = existing
        elif (
            existing.parts,
            existing.size,
            existing.sha256,
            existing.kind,
            existing.message_type,
            existing.epoch,
            existing.session_id,
        ) != (parts, size, digest, kind, message_type, epoch, session_id):
            self._messages.pop(message_id, None)
            raise ProtocolError("inconsistent fragment metadata")
        previous = existing.chunks.get(part)
        if previous is not None and previous != chunk:
            self._messages.pop(message_id, None)
            raise ProtocolError("conflicting duplicate fragment")
        existing.chunks[part] = chunk
        if len(existing.chunks) != parts:
            return None
        joined = "".join(existing.chunks[index] for index in range(parts))
        padding = "=" * (-len(joined) % 4)
        try:
            payload = base64.urlsafe_b64decode(joined + padding)
        except ValueError as exc:
            self._messages.pop(message_id, None)
            raise ProtocolError("invalid base64url fragment data") from exc
        self._messages.pop(message_id, None)
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ProtocolError("fragment checksum or byte count mismatch")
        try:
            envelope = decode_envelope(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ProtocolError("fragment payload is not UTF-8") from exc
        if (
            envelope.id,
            envelope.kind,
            envelope.message_type,
            envelope.epoch,
            envelope.session_id,
        ) != (message_id, kind, message_type, epoch, session_id):
            raise ProtocolError("fragment metadata does not match reconstructed envelope")
        return envelope

    def expire(self, now: float | None = None) -> tuple[str, ...]:
        current = time.monotonic() if now is None else now
        expired = tuple(
            message_id
            for message_id, item in self._messages.items()
            if current - item.created >= FRAGMENT_TIMEOUT_SECONDS
        )
        for message_id in expired:
            self._messages.pop(message_id, None)
        return expired


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ProtocolError(f"{key} must be a non-empty string")
    return item


def _optional_string(value: dict[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item:
        raise ProtocolError(f"{key} must be a non-empty string when present")
    return item
