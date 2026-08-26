from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, TextIO

from agentwire.protocol import (
    PROTOCOL_TAG,
    Envelope,
    ProtocolError,
    Reassembler,
    TopicActivation,
    fragment_envelope,
    new_envelope,
    parse_topic,
)
from agentwire.text import truncate_utf8


@dataclass(slots=True)
class HarnessState:
    active: bool = False
    backend: str | None = None
    epoch: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    busy: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    queue: list[dict[str, Any]] = field(default_factory=list)
    requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    assistant: list[dict[str, Any]] = field(default_factory=list)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan: dict[str, Any] | None = None
    # Replace-not-merge: every `subagent.updated` carries the full current list.
    subagents: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, event: Envelope) -> None:
        if event.message_type != "event":
            return
        if event.kind == "agent.hello":
            self.active = True
            self.backend = str(event.data.get("backend") or "") or None
            self.epoch = event.epoch or str(event.data.get("epoch") or "") or None
        elif event.kind == "channel.snapshot":
            self.active = bool(event.data.get("active", True))
            self.backend = str(event.data.get("backend") or "") or self.backend
            binding = event.data.get("binding")
            self.session_id = (
                str(binding.get("sid"))
                if isinstance(binding, dict) and binding.get("sid")
                else None
            )
            self.turn_id = str(event.data.get("tid") or "") or None
            self.busy = bool(event.data.get("busy"))
            self.settings = dict(event.data.get("settings") or {})
            self.queue = list(event.data.get("queue") or [])
        elif event.kind == "binding.changed":
            session = event.data.get("session")
            self.session_id = (
                str(session.get("sid"))
                if isinstance(session, dict) and session.get("sid")
                else event.session_id
            )
            self.turn_id = None
            self.busy = False
            self.assistant.clear()
            self.tools.clear()
            self.plan = None
            self.subagents = []
        elif event.kind in {"session.snapshot", "session.status"}:
            self.settings.update(event.data.get("settings") or {})
            if "busy" in event.data:
                self.busy = bool(event.data["busy"])
            if event.kind == "session.snapshot" and "recentOutputs" in event.data:
                self.assistant = [
                    dict(item)
                    for item in event.data.get("recentOutputs") or []
                    if isinstance(item, dict)
                ]
        elif event.kind == "turn.started":
            self.busy = True
            self.turn_id = event.turn_id
        elif event.kind in {"turn.completed", "turn.failed"}:
            self.busy = False
            self.turn_id = None
        elif event.kind == "assistant.completed":
            self.assistant.append({"iid": event.item_id, **event.data})
        elif event.kind == "plan.updated":
            self.plan = dict(event.data)
        elif event.kind == "subagent.updated":
            self.subagents = [
                dict(agent) for agent in event.data.get("agents") or [] if isinstance(agent, dict)
            ]
        elif event.kind.startswith("tool.") and event.item_id:
            self.tools[event.item_id] = {"kind": event.kind, **event.data}
        elif event.kind == "request.opened" and event.request_id:
            self.requests[event.request_id] = dict(event.data)
        elif event.kind == "request.resolved" and event.request_id:
            self.requests.pop(event.request_id, None)
        elif event.kind.startswith("queue.item."):
            self._apply_queue(event)
        elif event.kind == "queue.snapshot":
            self.queue = list(event.data.get("items") or [])

    def _apply_queue(self, event: Envelope) -> None:
        item_id = event.item_id or str(event.data.get("iid") or "")
        self.queue = [item for item in self.queue if str(item.get("iid")) != item_id]
        if event.kind != "queue.item.removed":
            self.queue.append(dict(event.data))
            self.queue.sort(key=lambda item: int(item.get("position", 0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "backend": self.backend,
            "epoch": self.epoch,
            "sid": self.session_id,
            "tid": self.turn_id,
            "busy": self.busy,
            "settings": self.settings,
            "queue": self.queue,
            "requests": self.requests,
            "assistant": self.assistant,
            "tools": self.tools,
            "plan": self.plan,
            "subagents": self.subagents,
        }


class ProtocolClient:
    """Small dependency-free v1 implementation for client tests and prototypes."""

    def __init__(self, device: str | None = None, instance: str | None = None) -> None:
        self.device = device or str(uuid.uuid4())
        self.instance = instance or str(uuid.uuid4())
        self.activation: TopicActivation | None = None
        self.reassembler = Reassembler()
        self.state = HarnessState()

    def set_topic(self, topic: str) -> TopicActivation | None:
        self.activation = parse_topic(topic)
        if self.activation is None:
            self.state.active = False
        return self.activation

    def ingest(self, tag_value: str) -> Envelope | None:
        event = self.reassembler.add(tag_value)
        if event is not None:
            self.state.apply(event)
        return event

    def action(
        self,
        kind: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        item_id: str | None = None,
        request_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> tuple[Envelope, list[str]]:
        envelope = new_envelope(
            kind,
            "action",
            self.instance,
            epoch=self.state.epoch,
            device=self.device,
            session_id=session_id,
            turn_id=turn_id,
            item_id=item_id,
            request_id=request_id,
            data=data or {},
        )
        return envelope, fragment_envelope(envelope)


def run_jsonl(input_stream: TextIO, output_stream: TextIO) -> int:
    client = ProtocolClient()
    for number, line in enumerate(input_stream, 1):
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ProtocolError("JSONL request must be an object")
            operation = request.get("op")
            if operation == "topic":
                activation = client.set_topic(str(request.get("topic") or ""))
                response = {
                    "ok": True,
                    "activation": (
                        {
                            "account": activation.account,
                            "agent": activation.agent,
                            "backend": activation.backend,
                            "title": activation.title,
                        }
                        if activation
                        else None
                    ),
                    "state": client.state.to_dict(),
                }
            elif operation == "ingest":
                value = request.get("tag")
                if not isinstance(value, str):
                    raise ProtocolError("ingest requires a string tag")
                event = client.ingest(value)
                response = {
                    "ok": True,
                    "event": event.to_dict() if event else None,
                    "state": client.state.to_dict(),
                }
            elif operation == "action":
                kind = request.get("kind")
                if not isinstance(kind, str):
                    raise ProtocolError("action requires kind")
                action_data = request.get("data") if isinstance(request.get("data"), dict) else {}
                envelope, fragments = client.action(
                    kind,
                    session_id=request.get("sid"),
                    turn_id=request.get("tid"),
                    item_id=request.get("iid"),
                    request_id=request.get("rid"),
                    data=action_data,
                )
                preview = ""
                if kind in {"turn.prompt", "turn.steer"}:
                    explicit = request.get("preview")
                    content = action_data.get("content")
                    preview = truncate_utf8(
                        str(explicit if explicit is not None else content or ""), 4096
                    )
                response = {
                    "ok": True,
                    "id": envelope.id,
                    "messages": [
                        {
                            "command": "PRIVMSG" if index == 0 and preview else "TAGMSG",
                            "body": preview if index == 0 else "",
                            "tags": {PROTOCOL_TAG: fragment},
                        }
                        for index, fragment in enumerate(fragments)
                    ],
                }
            elif operation == "state":
                response = {"ok": True, "state": client.state.to_dict()}
            else:
                raise ProtocolError("op must be topic, ingest, action, or state")
        except (json.JSONDecodeError, ProtocolError, ValueError) as exc:
            response = {"ok": False, "line": number, "error": str(exc)}
        output_stream.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
        output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentwire IRC v1 JSONL reference client")
    parser.parse_args(argv)
    return run_jsonl(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
