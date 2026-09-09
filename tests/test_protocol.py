from __future__ import annotations

import base64
import json
import random
import uuid
import zlib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from agentwire.protocol import (
    ACTION_KINDS,
    EVENT_KINDS,
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


def _schema_validator() -> Draft202012Validator:
    schema_path = Path(__file__).parents[1] / "protocol" / "agentwire-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


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


def test_omp_topic_and_hello_fixtures() -> None:
    topic_fixture = "agentwire:v1;account=trev;agent=agentwire;backend=omp | OMP workspace"
    assert build_topic("trev", "omp", "OMP workspace", agent="agentwire") == topic_fixture
    topic = parse_topic(topic_fixture)
    assert topic is not None
    assert topic.backend == "omp"

    hello_fixture = {
        "at": 1785400003000,
        "data": {
            "backend": "omp",
            "epoch": "epoch-example",
            "protocol": "agentwire-irc-v1",
            "settings": ["model", "effort", "delivery"],
        },
        "epoch": "epoch-example",
        "id": "77777777-7777-4777-8777-777777777777",
        "inst": "66666666-6666-4666-8666-666666666666",
        "k": "agent.hello",
        "t": "event",
        "v": 1,
    }
    hello = decode_envelope(json.dumps(hello_fixture))
    assert hello.kind == "agent.hello"
    assert hello.data["backend"] == "omp"
    assert hello.data["settings"] == ["model", "effort", "delivery"]


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


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda envelope: envelope.update({"unexpected": True}), "unsupported envelope fields"),
        (lambda envelope: envelope.pop("device"), "actions require device"),
        (lambda envelope: envelope.update({"t": "event"}), "unsupported event kind"),
        (lambda envelope: envelope.update({"v": True}), "unsupported protocol version"),
        (lambda envelope: envelope.update({"device": None}), "device must be a non-empty string"),
    ],
)
def test_reference_decoder_matches_strict_v1_envelope_rules(change, error: str) -> None:
    raw = new_envelope("sync.request", "action", "client", device="phone").to_dict()
    change(raw)
    with pytest.raises(ProtocolError, match=error):
        decode_envelope(json.dumps(raw))


def test_checked_fragment_round_trip_and_conflict_rejection() -> None:
    compressed = new_envelope(
        "assistant.completed",
        "event",
        "agent",
        id=str(uuid.uuid4()),
        epoch="live",
        data={"content": "🙂" * 3000},
    )
    compressed_fragments = fragment_envelope(compressed, max_tag_bytes=700)
    assert len(compressed_fragments) == 1
    assert json.loads(compressed_fragments[0])["encoding"] == "zlib"
    assert Reassembler().add(compressed_fragments[0]) == compressed

    random_text = random.Random(0).randbytes(6000).hex()
    envelope = new_envelope(
        "assistant.completed",
        "event",
        "agent",
        id=str(uuid.uuid4()),
        epoch="live",
        data={"content": random_text},
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


def test_fragments_reject_unknown_fields_and_invalid_metadata_types() -> None:
    envelope = new_envelope("assistant.completed", "event", "agent", data={"content": "x" * 2_000})
    fragment = json.loads(fragment_envelope(envelope, max_tag_bytes=500)[0])
    fragment["extra"] = True
    with pytest.raises(ProtocolError, match="unsupported fragment fields"):
        Reassembler().add(json.dumps(fragment))

    fragment.pop("extra")
    fragment["parts"] = True
    with pytest.raises(ProtocolError, match="invalid fragment count"):
        Reassembler().add(json.dumps(fragment))


def test_compressed_fragments_reject_oversized_output() -> None:
    envelope = new_envelope(
        "assistant.completed", "event", "agent", data={"content": "compress me" * 1000}
    )
    fragment = json.loads(fragment_envelope(envelope, max_tag_bytes=700)[0])
    fragment["bytes"] = 1
    fragment["b64"] = base64.urlsafe_b64encode(zlib.compress(b"too large")).decode().rstrip("=")
    with pytest.raises(ProtocolError, match="compressed fragment size"):
        Reassembler().add(json.dumps(fragment, separators=(",", ":")))


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
    validator = _schema_validator()
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

    # Subagent state is session-owned and replaces the client's whole list.
    subagents = decode_envelope((fixtures / "subagent-update.json").read_text(encoding="utf-8"))
    assert subagents.kind == "subagent.updated"
    assert subagents.session_id == "session-example"
    agents = subagents.data["agents"]
    assert [agent["status"] for agent in agents] == ["running", "completed"]
    assert agents[0] == {
        "id": "agent-1",
        "type": "Explore",
        "description": "map the repository",
        "status": "running",
        "isBackground": True,
    }
    assert (agents[1]["toolUses"], agents[1]["durationMs"], agents[1]["tokens"]) == (7, 4200, 1234)

    for path in sorted(fixtures.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert not list(validator.iter_errors(raw)), path.name
        decode_envelope(json.dumps(raw))

    for path in sorted((fixtures / "invalid").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert list(validator.iter_errors(raw)), path.name
        with pytest.raises(ProtocolError):
            Reassembler().add(json.dumps(raw))


def test_schema_kind_sets_match_the_reference_codec() -> None:
    schema = json.loads(
        (Path(__file__).parents[1] / "protocol" / "agentwire-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    definitions = schema["$defs"]
    assert set(definitions["actionKind"]["enum"]) == ACTION_KINDS
    assert set(definitions["eventKind"]["enum"]) == EVENT_KINDS


def test_integral_json_numbers_match_schema_and_jvm_clients() -> None:
    raw = new_envelope("assistant.completed", "event", "agent", at=1).to_dict()
    raw.update(v=1.0, at=1.0, rev=2.0)
    assert not list(_schema_validator().iter_errors(raw))
    decoded = decode_envelope(json.dumps(raw))
    assert decoded.at == 1 and type(decoded.at) is int
    assert decoded.revision == 2 and type(decoded.revision) is int
    for field in ("at", "rev"):
        invalid = {**raw, field: 2**63}
        assert list(_schema_validator().iter_errors(invalid))
        with pytest.raises(ProtocolError):
            decode_envelope(json.dumps(invalid))


def test_conformance_corpus_is_current_and_schema_valid() -> None:
    protocol_dir = Path(__file__).parents[1] / "protocol"
    generator = protocol_dir / "conformance" / "generate.py"
    # Importing the generator directly avoids a subprocess and keeps this test
    # usable by downstream packagers that run pytest without a console script.
    import importlib.util

    spec = importlib.util.spec_from_file_location("agentwire_conformance", generator)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0

    validator = _schema_validator()
    for path in sorted((protocol_dir / "conformance").glob("*.json")):
        if path.name == "manifest.json":
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if "envelope" in document:
            assert not list(validator.iter_errors(json.loads(document["envelope"]))), path.name
            for fragment in document["fragments"]:
                assert not list(validator.iter_errors(json.loads(fragment))), path.name
            continue
        for step in document["steps"]:
            assert not list(validator.iter_errors(json.loads(step["tag"]))), path.name


def test_replay_corpus_asserts_live_state_isolation_and_full_tool_identity() -> None:
    document = json.loads(
        (
            Path(__file__).parents[1] / "protocol" / "conformance" / "replay-and-isolation.json"
        ).read_text(encoding="utf-8")
    )
    history_state = document["steps"][4]["state"]
    assert (history_state["busy"], history_state["tid"]) == (True, "turn-1")
    final_state = document["steps"][-1]["state"]
    assert (final_state["busy"], final_state["tid"]) == (True, "turn-1")
    assert set(final_state["tools"]) == {
        '["sess-conformance",null,"shared-tool"]',
        '["sess-conformance","another-turn","shared-tool"]',
        '["sess-conformance","turn-1","call-1"]',
        '["sess-conformance","turn-1","preserved"]',
    }
    preserved = final_state["tools"]['["sess-conformance","turn-1","preserved"]']
    assert preserved == {
        "event": "tool.completed",
        "iid": "preserved",
        "input": "pytest -q",
        "kind": "shell",
        "label": "run tests",
        "output": "passed",
        "sid": "sess-conformance",
        "success": True,
        "tid": "turn-1",
    }


def test_every_committed_envelope_fixture_re_encodes_byte_for_byte() -> None:
    fixtures = Path(__file__).parents[1] / "protocol" / "fixtures"
    for name in (
        "hello.json",
        "prompt-action.json",
        "claude-hello.json",
        "pi-hello.json",
        "observed-status.json",
        "subagent-update.json",
    ):
        raw = (fixtures / name).read_text(encoding="utf-8").strip()
        assert encode_envelope(decode_envelope(raw)) == raw
