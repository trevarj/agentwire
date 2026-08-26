from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from agentwire.protocol import (
    MAX_PAYLOAD_BYTES,
    MAX_TAG_SECTION_BYTES,
    PROTOCOL_TAG,
    ProtocolError,
    Reassembler,
    build_topic,
    decode_envelope,
    encode_envelope,
    fragment_envelope,
    new_envelope,
    parse_topic,
    tag_wire_size,
)


def test_topic_activation_is_exact_and_percent_decoded() -> None:
    topic = build_topic("Trev", "codex local", "Agent workspace", agent="AgentWire")
    assert topic == (
        "agentwire:v1;account=trev;agent=agentwire;backend=codex%20local | Agent workspace"
    )
    activation = parse_topic(topic)
    assert activation is not None
    assert (activation.account, activation.agent, activation.backend, activation.title) == (
        "trev",
        "agentwire",
        "codex local",
        "Agent workspace",
    )
    assert parse_topic("chat agentwire:v1;account=trev;agent=agentwire;backend=codex") is None
    # The agent account is the client's trust root for events, so a topic
    # without it must not activate anything.
    with pytest.raises(ProtocolError, match="agent"):
        parse_topic("agentwire:v1;account=trev;backend=codex")


def test_topic_backend_is_case_folded() -> None:
    # The bridge compares the parsed backend against the configured channel
    # backend with ==, and a mismatch leaves activation None so every action is
    # dropped before it is journaled — a silent, endless sync on the client.
    # Backends are a closed lowercase set, so fold rather than reject.
    for written in ("claude", "Claude", "CLAUDE"):
        activation = parse_topic(f"agentwire:v1;account=trev;agent=agentwire;backend={written}")
        assert activation is not None
        assert activation.backend == "claude"


def test_single_account_topics_are_first_class() -> None:
    # One SASL account for controller and bot, separated by channel, is a
    # supported deployment: agent defaults to account and parses back equal.
    topic = build_topic("AgentWire", "claude")
    assert topic == "agentwire:v1;account=agentwire;agent=agentwire;backend=claude"
    activation = parse_topic(topic)
    assert activation is not None
    assert activation.account == activation.agent == "agentwire"
    explicit = parse_topic("agentwire:v1;account=agentwire;agent=agentwire;backend=claude")
    assert explicit is not None
    assert explicit.account == explicit.agent == "agentwire"


def test_envelope_round_trip_is_minified_and_validated() -> None:
    envelope = new_envelope(
        "turn.prompt",
        "action",
        "client",
        id=str(uuid.uuid4()),
        epoch="live",
        device="phone",
        session_id="session",
        data={"content": "hello"},
    )
    encoded = encode_envelope(envelope)
    assert " " not in encoded
    assert decode_envelope(encoded) == envelope
    invalid = json.loads(encoded)
    invalid["v"] = 2
    with pytest.raises(ProtocolError, match="version"):
        decode_envelope(json.dumps(invalid))


def test_checked_fragment_round_trip_and_conflict_rejection() -> None:
    envelope = new_envelope(
        "assistant.completed",
        "event",
        "agent",
        id=str(uuid.uuid4()),
        epoch="live",
        data={"content": "🙂" * 3000},
    )
    fragments = fragment_envelope(envelope, max_tag_bytes=700)
    assert 1 < len(fragments) <= 64
    reassembler = Reassembler()
    decoded = None
    for fragment in reversed(fragments):
        decoded = reassembler.add(fragment) or decoded
    assert decoded == envelope

    reassembler = Reassembler()
    reassembler.add(fragments[0])
    conflicting = json.loads(fragments[0])
    conflicting["b64"] += "A"
    with pytest.raises(ProtocolError, match="conflicting"):
        reassembler.add(json.dumps(conflicting, separators=(",", ":")))


def test_fragmentation_uses_escaped_irc_tag_size() -> None:
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": " " * 300})
    fragments = fragment_envelope(envelope, max_tag_bytes=500)
    assert len(fragments) > 1
    assert all(tag_wire_size(fragment) <= 500 for fragment in fragments)


def test_default_fragment_budget_includes_tag_name() -> None:
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": "; " * 5000})
    fragments = fragment_envelope(envelope)
    assert all(
        len(PROTOCOL_TAG.encode("utf-8")) + 1 + tag_wire_size(fragment) <= MAX_TAG_SECTION_BYTES
        for fragment in fragments
    )


def test_payload_limit_is_enforced_before_fragmentation() -> None:
    envelope = new_envelope(
        "assistant.completed",
        "event",
        "agent",
        data={"content": "x" * MAX_PAYLOAD_BYTES},
    )
    with pytest.raises(ProtocolError, match="exceeds"):
        fragment_envelope(envelope)


def test_committed_fixtures_decode_with_reference_codec() -> None:
    protocol_dir = Path(__file__).parents[1] / "protocol"
    json.loads((protocol_dir / "agentwire-v1.schema.json").read_text(encoding="utf-8"))
    fixtures = protocol_dir / "fixtures"
    assert parse_topic((fixtures / "topic.txt").read_text(encoding="utf-8").strip()) is not None
    hello = decode_envelope((fixtures / "hello.json").read_text(encoding="utf-8"))
    prompt = decode_envelope((fixtures / "prompt-action.json").read_text(encoding="utf-8"))
    assert hello.kind == "agent.hello"
    assert prompt.kind == "turn.prompt"

    claude_topic = parse_topic((fixtures / "claude-topic.txt").read_text(encoding="utf-8").strip())
    assert claude_topic is not None
    assert claude_topic.backend == "claude"
    claude_hello = decode_envelope((fixtures / "claude-hello.json").read_text(encoding="utf-8"))
    assert claude_hello.kind == "agent.hello"
    assert claude_hello.data["backend"] == "claude"
    # Claude advertises no model picker, so delivery is its only safe setting.
    assert claude_hello.data["settings"] == ["delivery"]

    pi_topic = parse_topic((fixtures / "pi-topic.txt").read_text(encoding="utf-8").strip())
    assert pi_topic is not None
    assert pi_topic.backend == "pi"
    pi_hello = decode_envelope((fixtures / "pi-hello.json").read_text(encoding="utf-8"))
    assert pi_hello.kind == "agent.hello"
    assert pi_hello.data["backend"] == "pi"
    # pi exposes its model catalog and thinking level through the bridge.
    assert pi_hello.data["settings"] == ["model", "effort", "delivery"]

    # A status for a session no channel is bound to: liveness only, no timeline.
    observed = decode_envelope((fixtures / "observed-status.json").read_text(encoding="utf-8"))
    assert observed.kind == "session.status"
    assert observed.session_id == "observed-example"
    assert observed.data == {
        "busy": True,
        "cwd": "/home/example/project",
        "flags": ["waiting"],
        "tuiAttached": True,
    }


def test_every_committed_envelope_fixture_re_encodes_byte_for_byte() -> None:
    fixtures = Path(__file__).parents[1] / "protocol" / "fixtures"
    for name in (
        "hello.json",
        "prompt-action.json",
        "claude-hello.json",
        "pi-hello.json",
        "observed-status.json",
    ):
        raw = (fixtures / name).read_text(encoding="utf-8").strip()
        assert encode_envelope(decode_envelope(raw)) == raw
