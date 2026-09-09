#!/usr/bin/env python3
"""Generate the checked-in Agentwire cross-client conformance corpus.

The corpus has no runtime dependency: downstream clients copy its JSON files
and record the manifest hashes they imported.  Run ``--check`` in CI to prove
the committed output still follows the protocol reference implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentwire.protocol import Envelope, encode_envelope, fragment_envelope
from agentwire.reference_client import ProtocolClient

TOPIC = "agentwire:v1;account=trev;agent=agentwire;backend=claude | conformance"
EPOCH = "epoch-conformance"
INSTANCE = "11111111-1111-4111-8111-111111111111"
SESSION = "sess-conformance"
TURN = "turn-1"


class Envelopes:
    """Build deterministic events with stable UUIDs and timestamps."""

    def __init__(self) -> None:
        self.number = 0

    def event(self, kind: str, **kwargs: Any) -> Envelope:
        self.number += 1
        return Envelope(
            kind=kind,
            message_type="event",
            id=f"{self.number:08d}-0000-4000-8000-000000000000",
            at=1_785_400_000_000 + self.number,
            instance=INSTANCE,
            epoch=EPOCH,
            **kwargs,
        )


def _hello(events: Envelopes) -> Envelope:
    return events.event(
        "agent.hello",
        data={
            "protocol": "agentwire-irc-v1",
            "backend": "claude",
            "epoch": EPOCH,
            "capabilities": [
                "actionStatus",
                "compressedFragments",
                "diagnostics",
                "history",
                "historyChunks",
                "queues",
                "requests",
                "sessions",
                "settings",
                "steering",
                "sync",
                "turns",
                "workspaces",
            ],
            "actions": [
                "sync.request",
                "workspace.list.request",
                "session.list.request",
                "history.request",
                "action.status.request",
                "diagnostics.request",
                "session.create",
                "session.attach",
                "session.detach",
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
            ],
            "limits": {
                "contentBytes": 65_536,
                "queueItems": 10,
                "historyEvents": 200,
                "historyBytes": 524_288,
                "historyChunkBytes": 98_304,
                "historyDays": 30,
            },
            "settings": ["delivery"],
        },
    )


def claude_session(events: Envelopes) -> list[Envelope]:
    return [
        _hello(events),
        events.event(
            "channel.snapshot",
            data={
                "active": True,
                "backend": "claude",
                "binding": {"sid": SESSION, "cwd": "/workspace"},
                "busy": False,
                "tid": None,
                "settings": {"delivery": "queue"},
                "requests": [],
                "queue": [],
            },
        ),
        events.event(
            "binding.changed", session_id=SESSION, data={"sid": SESSION, "cwd": "/workspace"}
        ),
        events.event(
            "session.snapshot",
            session_id=SESSION,
            data={"busy": False, "status": "ready", "recentOutputs": []},
        ),
        events.event("turn.started", session_id=SESSION, turn_id=TURN, data={}),
        events.event(
            "user.prompt",
            session_id=SESSION,
            turn_id=TURN,
            item_id="prompt-1",
            data={"content": "why does the test fail?"},
        ),
        events.event(
            "tool.started",
            session_id=SESSION,
            turn_id=TURN,
            item_id="call-1",
            data={"kind": "file read", "label": "file read", "input": "StateTest.kt"},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="call-1",
            data={"kind": "file read", "label": "file read", "success": True, "durationMs": 12},
        ),
        events.event(
            "request.opened",
            session_id=SESSION,
            turn_id=TURN,
            request_id="req-1",
            data={"type": "question", "canSkip": True, "questions": []},
        ),
        events.event(
            "request.resolved", session_id=SESSION, turn_id=TURN, request_id="req-1", data={}
        ),
        events.event(
            "assistant.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="msg-1",
            data={"content": "The assertion arguments are inverted."},
        ),
        events.event("turn.completed", session_id=SESSION, turn_id=TURN, data={}),
    ]


def queue_and_acks(events: Envelopes) -> list[Envelope]:
    return [
        events.event(
            "queue.snapshot",
            session_id=SESSION,
            data={"items": [{"iid": "q-1", "content": "run tests", "position": 0}]},
        ),
        events.event(
            "queue.item.added",
            session_id=SESSION,
            data={"iid": "q-2", "content": "then lint", "position": 1},
        ),
        events.event(
            "queue.item.updated",
            session_id=SESSION,
            data={"iid": "q-2", "content": "then lint all", "position": 1},
        ),
        events.event("queue.item.moved", session_id=SESSION, data={"iid": "q-2", "position": 0}),
        events.event("queue.item.removed", session_id=SESSION, data={"iid": "q-2"}),
        events.event("action.accepted", reply="00000000-0000-4000-8000-00000000cafe", data={}),
        events.event("action.succeeded", reply="00000000-0000-4000-8000-00000000cafe", data={}),
        events.event(
            "action.failed",
            reply="00000000-0000-4000-8000-00000000beef",
            data={"message": "safe failure"},
        ),
        events.event("action.uncertain", reply="00000000-0000-4000-8000-00000000f00d", data={}),
    ]


def action_status(events: Envelopes) -> list[Envelope]:
    action_id = "00000000-0000-4000-8000-00000000cafe"
    return [
        _hello(events),
        events.event(
            "action.status",
            reply="00000000-0000-4000-8000-00000000a001",
            data={
                "actionId": action_id,
                "status": "accepted",
                "kind": "turn.prompt",
                "channel": "#claude",
                "receivedAt": 1_785_400_000_100,
            },
        ),
        # A delayed accepted acknowledgement and a stale unknown lookup must
        # never hide the terminal receipt already rendered by a client.
        events.event("action.accepted", reply=action_id, data={}),
        events.event(
            "action.status",
            reply="00000000-0000-4000-8000-00000000a004",
            data={"actionId": action_id, "status": "unknown"},
        ),
        events.event(
            "action.status",
            reply="00000000-0000-4000-8000-00000000a002",
            data={
                "actionId": action_id,
                "status": "succeeded",
                "kind": "turn.prompt",
                "channel": "#claude",
                "receivedAt": 1_785_400_000_100,
            },
        ),
        events.event(
            "action.status",
            reply="00000000-0000-4000-8000-00000000a003",
            data={
                "actionId": "00000000-0000-4000-8000-00000000dead",
                "status": "unknown",
            },
        ),
    ]


def replay_and_isolation(events: Envelopes) -> list[Envelope]:
    live_start = events.event(
        "tool.started",
        session_id=SESSION,
        turn_id=TURN,
        item_id="preserved",
        data={"kind": "shell", "label": "run tests", "input": "pytest -q"},
    )
    live_completed = events.event(
        "tool.completed",
        session_id=SESSION,
        turn_id=TURN,
        item_id="preserved",
        data={"success": True, "output": "passed"},
    )
    historic = events.event(
        "turn.completed",
        session_id=SESSION,
        turn_id=TURN,
        reply="history-1",
        history=True,
        data={},
    )
    historic_tool = events.event(
        "tool.completed",
        session_id=SESSION,
        turn_id=TURN,
        item_id="call-1",
        reply="history-1",
        history=True,
        data={"kind": "file read", "success": True},
    )
    historic_start = events.event(
        "tool.started",
        session_id=SESSION,
        turn_id=TURN,
        item_id="preserved",
        reply="history-1",
        history=True,
        data={"kind": "shell", "label": "old replay", "input": "pytest -q"},
    )
    return [
        _hello(events),
        events.event(
            "channel.snapshot",
            session_id=SESSION,
            data={
                "active": True,
                "backend": "claude",
                "binding": {"sid": SESSION},
                "busy": True,
                "tid": TURN,
            },
        ),
        live_start,
        live_completed,
        events.event(
            "history.chunk",
            session_id=SESSION,
            reply="history-1",
            data={
                "page": "p1",
                "index": 0,
                "events": [historic.to_dict(), historic_tool.to_dict(), historic_start.to_dict()],
            },
        ),
        # The duplicate repeats a previously observed envelope.  It must have
        # no additional effect when an IRC playback overlaps a history page.
        historic_tool,
        events.event("turn.completed", session_id="other-session", turn_id="other-turn", data={}),
        events.event(
            "tool.started",
            session_id=SESSION,
            item_id="shared-tool",
            data={"kind": "shell", "label": "without a turn id"},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            item_id="shared-tool",
            data={"kind": "shell", "success": True},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id="another-turn",
            item_id="shared-tool",
            data={"kind": "file edit", "success": True},
        ),
        events.event(
            "assistant.completed",
            session_id=SESSION,
            turn_id=TURN,
            data={"content": "First output without an item ID"},
        ),
        events.event(
            "assistant.completed",
            session_id=SESSION,
            turn_id=TURN,
            data={"content": "Second output without an item ID"},
        ),
    ]


def _corpus(events: list[Envelope]) -> dict[str, Any]:
    client = ProtocolClient(device="device-conformance", instance=INSTANCE)
    assert client.set_topic(TOPIC) is not None
    steps = []
    for event in events:
        tag = encode_envelope(event)
        assert client.ingest(tag) is not None
        steps.append(
            {"kind": event.kind, "tag": tag, "state": copy.deepcopy(client.state.to_dict())}
        )
    return {"topic": TOPIC, "steps": steps}


def tool_activity(events: Envelopes) -> list[Envelope]:
    started = events.event(
        "tool.started",
        session_id=SESSION,
        turn_id=TURN,
        item_id="reused",
        data={"kind": "shell", "input": "pytest -q", "label": "Run checks"},
    )
    completed = events.event(
        "tool.completed",
        session_id=SESSION,
        turn_id=TURN,
        item_id="reused",
        data={"success": False, "output": "failed", "kind": "shell"},
    )
    return [
        _hello(events),
        events.event(
            "channel.snapshot", data={"binding": {"sid": SESSION}, "busy": True, "tid": TURN}
        ),
        events.event(
            "user.prompt", session_id=SESSION, turn_id=TURN, data={"content": "Inspect the code"}
        ),
        started,
        completed,
        completed,
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="edit",
            data={"kind": "file edit", "diff": "+replacement", "success": True},
        ),
        events.event(
            "assistant.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="narration",
            data={"content": "The first check failed. I will inspect the source."},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="read",
            data={"kind": "file read", "input": "source.py", "output": "source", "success": True},
        ),
        events.event(
            "request.opened",
            session_id=SESSION,
            turn_id=TURN,
            request_id="question",
            data={
                "type": "question",
                "title": "Continue?",
                "questions": [{"id": "continue", "prompt": "Continue?", "options": []}],
            },
        ),
        events.event(
            "tool.updated",
            session_id=SESSION,
            turn_id=TURN,
            item_id="web",
            data={"kind": "web search", "label": "Search reference"},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="agent",
            data={"kind": "agent", "success": True},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id=TURN,
            item_id="other",
            data={"kind": "MCP tool", "label": "shell", "output": "tests passed"},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            item_id="no-turn",
            data={"kind": "shell", "success": False},
        ),
        events.event(
            "tool.completed",
            session_id=SESSION,
            turn_id="turn-2",
            item_id="reused",
            data={"kind": "file edit", "success": True},
        ),
        events.event("turn.completed", session_id=SESSION, turn_id=TURN),
    ]


def diagnostics(events: Envelopes) -> list[Envelope]:
    return [
        _hello(events),
        events.event(
            "diagnostics.snapshot",
            reply="diagnostics-query",
            data={
                "schemaVersion": 1,
                "generatedAt": 1_785_400_000_000,
                "source": "bridge",
                "checks": [
                    {
                        "code": "backend.ready",
                        "status": "ok",
                        "explanation": "Backend is ready.",
                        "facts": {"ready": True, "sessionCount": 0},
                    },
                    {
                        "code": "irc.outgoing",
                        "status": "warning",
                        "explanation": "Outgoing messages are queued.",
                        "facts": {"queueDepth": 2, "oldestQueuedMs": 15000},
                        "nextStep": "Check IRC connectivity.",
                    },
                ],
            },
        ),
    ]


def _documents() -> dict[str, str]:
    scenarios: tuple[tuple[str, Callable[[Envelopes], list[Envelope]]], ...] = (
        ("claude-session", claude_session),
        ("queue-and-acks", queue_and_acks),
        ("action-status", action_status),
        ("diagnostics", diagnostics),
        ("tool-activity", tool_activity),
        ("replay-and-isolation", replay_and_isolation),
    )
    events = Envelopes()
    documents = {
        f"{name}.json": json.dumps(_corpus(builder(events)), indent=2, sort_keys=True) + "\n"
        for name, builder in scenarios
    }
    big = events.event(
        "assistant.completed",
        session_id=SESSION,
        turn_id=TURN,
        item_id="msg-big",
        data={"content": random.Random(0).randbytes(10_000).hex()},
    )
    fragments = fragment_envelope(big)
    assert len(fragments) > 1
    documents["fragmented.json"] = (
        json.dumps(
            {"envelope": encode_envelope(big), "fragments": fragments}, indent=2, sort_keys=True
        )
        + "\n"
    )
    manifest = {
        "format": 1,
        "generatedBy": "protocol/conformance/generate.py",
        "protocol": "agentwire-irc-v1",
        "files": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in sorted(documents.items())
        },
    }
    documents["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed corpus differs")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    documents = _documents()
    mismatches = []
    for name, text in documents.items():
        path = args.output / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                mismatches.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if mismatches:
        print("conformance corpus is stale: " + ", ".join(map(str, mismatches)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
