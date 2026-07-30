from __future__ import annotations

import io
import json

from agentwire.protocol import fragment_envelope, new_envelope
from agentwire.reference_client import ProtocolClient, run_jsonl


def test_reference_client_reduces_events_to_render_state() -> None:
    client = ProtocolClient(device="phone", instance="client")
    client.set_topic("agentwire:v1;account=trev;backend=codex | Test")
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


def test_jsonl_cli_emits_action_wire_messages() -> None:
    input_stream = io.StringIO(
        '{"op":"topic","topic":"agentwire:v1;account=trev;backend=codex"}\n'
        '{"op":"action","kind":"sync.request"}\n'
    )
    output_stream = io.StringIO()
    assert run_jsonl(input_stream, output_stream) == 0
    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[0]["activation"]["backend"] == "codex"
    assert responses[1]["messages"][0]["command"] == "TAGMSG"


def test_jsonl_prompt_defaults_to_readable_privmsg() -> None:
    input_stream = io.StringIO('{"op":"action","kind":"turn.prompt","data":{"content":"hello"}}\n')
    output_stream = io.StringIO()
    run_jsonl(input_stream, output_stream)
    message = json.loads(output_stream.getvalue())["messages"][0]
    assert message["command"] == "PRIVMSG"
    assert message["body"] == "hello"
