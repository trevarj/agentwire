from __future__ import annotations

import io
import json

import pytest

from agentwire.protocol import ProtocolError, fragment_envelope, new_envelope
from agentwire.reference_client import ProtocolClient, run_jsonl


def test_reference_client_reduces_events_to_render_state() -> None:
    client = ProtocolClient(device="phone", instance="client")
    client.set_topic("agentwire:v1;account=trev;agent=agentwire;backend=codex | Test")
    hello = new_envelope("agent.hello", "event", "agent", epoch="live", data={"backend": "codex"})
    snapshot = new_envelope(
        "channel.snapshot",
        "event",
        "agent",
        epoch="live",
        session_id="s1",
        data={"active": True, "backend": "codex", "binding": {"sid": "s1"}, "busy": False},
    )
    for event in (hello, snapshot):
        for fragment in fragment_envelope(event):
            client.ingest(fragment)
    assert client.state.epoch == "live"
    assert client.state.session_id == "s1"
    action, _ = client.action("turn.prompt", session_id="s1", data={"content": "hello"})
    assert (action.epoch, action.device) == ("live", "phone")


def test_reference_client_replaces_timeline_context_on_binding_change() -> None:
    client = ProtocolClient(device="phone", instance="client")
    client.state.assistant = [{"content": "old"}]
    client.state.apply(
        new_envelope(
            "binding.changed",
            "event",
            "agent",
            session_id="s2",
            data={"session": {"sid": "s2"}},
        )
    )
    assert client.state.assistant == []

    client.state.apply(
        new_envelope(
            "session.snapshot",
            "event",
            "agent",
            session_id="s2",
            data={"status": "ready", "recentOutputs": [{"iid": "i1", "content": "new"}]},
        )
    )
    assert client.state.assistant == [{"iid": "i1", "content": "new"}]


def test_reference_client_applies_packed_history_events() -> None:
    client = ProtocolClient(device="phone", instance="client")
    historic = new_envelope(
        "assistant.completed",
        "event",
        "agent",
        session_id="s1",
        item_id="i1",
        reply="request-1",
        history=True,
        data={"content": "restored"},
    )
    client.state.apply(
        new_envelope(
            "history.chunk",
            "event",
            "agent",
            session_id="s1",
            reply="request-1",
            data={"page": "p1", "index": 0, "events": [historic.to_dict()]},
        )
    )
    assert client.state.assistant == [{"iid": "i1", "content": "restored"}]

    invalid = new_envelope(
        "history.chunk",
        "event",
        "agent",
        session_id="other",
        reply="request-1",
        data={"events": [historic.to_dict()]},
    )
    with pytest.raises(ProtocolError, match="metadata does not match"):
        client.state.apply(invalid)


def test_jsonl_cli_emits_action_wire_messages() -> None:
    input_stream = io.StringIO(
        '{"op":"topic","topic":"agentwire:v1;account=trev;agent=agentwire;backend=codex"}\n'
        '{"op":"action","kind":"sync.request"}\n'
    )
    output_stream = io.StringIO()
    assert run_jsonl(input_stream, output_stream) == 0
    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[0]["activation"]["backend"] == "codex"
    assert responses[0]["activation"]["agent"] == "agentwire"
    assert responses[1]["messages"][0]["command"] == "TAGMSG"


def test_jsonl_prompt_defaults_to_readable_privmsg() -> None:
    input_stream = io.StringIO('{"op":"action","kind":"turn.prompt","data":{"content":"hello"}}\n')
    output_stream = io.StringIO()
    run_jsonl(input_stream, output_stream)
    message = json.loads(output_stream.getvalue())["messages"][0]
    assert message["command"] == "PRIVMSG"
    assert message["body"] == "hello"
