from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, TextIO

from agentwire.protocol import (
    HISTORY_EVENT_KINDS,
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
    action_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan: dict[str, Any] | None = None
    # Replace-not-merge: every `subagent.updated` carries the full current list.
    subagents: list[dict[str, Any]] = field(default_factory=list)
    _seen_event_ids: set[str] = field(default_factory=set, repr=False)
    _assistant_positions: dict[str, int] = field(default_factory=dict, repr=False)

    def apply(self, event: Envelope) -> None:
        if event.message_type != "event":
            return
        if event.kind == "agent.hello":
            if event.epoch != self.epoch:
                self._seen_event_ids.clear()
            self.active = True
            self.backend = str(event.data.get("backend") or "") or None
            self.epoch = event.epoch or str(event.data.get("epoch") or "") or None
        elif event.kind == "channel.snapshot":
            self.active = bool(event.data.get("active", True))
            self.backend = str(event.data.get("backend") or "") or self.backend
            binding = event.data.get("binding")
            session_id = (
                str(binding.get("sid"))
                if isinstance(binding, dict) and binding.get("sid")
                else None
            )
            if session_id != self.session_id:
                self._reset_timeline()
            self.session_id = session_id
            self.turn_id = str(event.data.get("tid") or "") or None
            self.busy = bool(event.data.get("busy"))
            self.settings = dict(event.data.get("settings") or {})
            self.queue = list(event.data.get("queue") or [])
        elif event.kind == "history.chunk":
            for raw in event.data.get("events") or ():
                if not isinstance(raw, dict):
                    raise ProtocolError("history chunk events must be objects")
                historic = Envelope.from_dict(raw)
                if (
                    historic.message_type != "event"
                    or historic.kind not in HISTORY_EVENT_KINDS
                    or not historic.history
                    or historic.session_id != event.session_id
                    or historic.reply != event.reply
                ):
                    raise ProtocolError("history chunk event metadata does not match its page")
                self.apply(historic)
        elif event.kind == "binding.changed":
            session = event.data.get("session")
            self.session_id = (
                str(session.get("sid"))
                if isinstance(session, dict) and session.get("sid")
                else event.session_id
            )
            self._reset_timeline()
        elif event.kind in {
            "action.accepted",
            "action.succeeded",
            "action.failed",
            "action.uncertain",
        }:
            if event.reply and not event.history:
                receipt: dict[str, Any] = {"status": event.kind.removeprefix("action.")}
                if isinstance(event.data.get("message"), str):
                    receipt["message"] = event.data["message"]
                self._merge_action_status(event.reply, receipt)
        elif event.kind == "action.status":
            action_id = event.data.get("actionId")
            if isinstance(action_id, str) and not event.history:
                self._merge_action_status(action_id, dict(event.data))
        elif event.kind in {"session.snapshot", "session.status"} and self._is_bound(event):
            self.settings.update(event.data.get("settings") or {})
            if "busy" in event.data:
                self.busy = bool(event.data["busy"])
            if event.kind == "session.snapshot" and "recentOutputs" in event.data:
                self.assistant = [
                    dict(item)
                    for item in event.data.get("recentOutputs") or []
                    if isinstance(item, dict)
                ]
                self._assistant_positions = {
                    self._output_key(event, item): index
                    for index, item in enumerate(self.assistant)
                    if item.get("iid")
                }
        elif event.kind == "turn.started" and not event.history and self._is_bound(event):
            self.busy = True
            self.turn_id = event.turn_id
        elif (
            event.kind in {"turn.completed", "turn.failed"}
            and not event.history
            and self._is_bound(event)
        ):
            self.busy = False
            self.turn_id = None
        elif event.kind == "assistant.completed" and self._is_bound(event):
            if not self._mark_once(event):
                return
            item = {"iid": event.item_id, **event.data}
            key = self._item_key(event) if event.item_id else event.id
            position = self._assistant_positions.get(key)
            if position is None:
                self._assistant_positions[key] = len(self.assistant)
                self.assistant.append(item)
            else:
                self.assistant[position] = item
        elif event.kind == "plan.updated" and not event.history and self._is_bound(event):
            self.plan = dict(event.data)
        elif event.kind == "subagent.updated" and not event.history and self._is_bound(event):
            self.subagents = [
                dict(agent) for agent in event.data.get("agents") or [] if isinstance(agent, dict)
            ]
        elif event.kind.startswith("tool.") and event.item_id and self._is_bound(event):
            if not self._mark_once(event):
                return
            key = self._tool_key(event)
            previous = self.tools.get(key, {})
            if event.history and self._tool_rank(event.kind) < self._tool_rank(
                previous.get("event")
            ):
                return
            self.tools[key] = {
                **previous,
                **event.data,
                "sid": event.session_id,
                "tid": event.turn_id,
                "iid": event.item_id,
                "event": event.kind,
            }
        elif (
            event.kind == "request.opened"
            and event.request_id
            and not event.history
            and self._is_bound(event)
        ):
            self.requests[event.request_id] = dict(event.data)
        elif (
            event.kind == "request.resolved"
            and event.request_id
            and not event.history
            and self._is_bound(event)
        ):
            self.requests.pop(event.request_id, None)
        elif event.kind.startswith("queue.item.") and self._is_queue_relevant(event):
            self._apply_queue(event)
        elif event.kind == "queue.snapshot" and self._is_queue_relevant(event):
            self.queue = list(event.data.get("items") or [])

    def _is_bound(self, event: Envelope) -> bool:
        """Whether an event may affect the channel's selected-session state."""

        return event.session_id is None or (
            self.session_id is not None and event.session_id == self.session_id
        )

    def _reset_timeline(self) -> None:
        self.turn_id = None
        self.busy = False
        self.assistant.clear()
        self._assistant_positions.clear()
        self.tools.clear()
        self.plan = None
        self.subagents = []
        self._seen_event_ids.clear()

    def _merge_action_status(self, action_id: str, incoming: dict[str, Any]) -> None:
        """Never let delayed acceptance or unknown lookup hide a terminal receipt."""
        previous = self.action_status.get(action_id)
        if previous is not None and self._receipt_rank(incoming) < self._receipt_rank(previous):
            return
        self.action_status[action_id] = incoming

    @staticmethod
    def _receipt_rank(receipt: dict[str, Any]) -> int:
        return {
            "unknown": 0,
            "accepted": 1,
            "succeeded": 2,
            "failed": 2,
            "uncertain": 2,
        }.get(receipt.get("status"), -1)

    def _is_queue_relevant(self, event: Envelope) -> bool:
        # Queue operations remain useful before the first binding snapshot.
        # Once a channel is bound, a delayed queue event from another session
        # must not replace the selected session's pending work.
        return self.session_id is None or self._is_bound(event)

    def _mark_once(self, event: Envelope) -> bool:
        # Ignore only events we actually applied.  An observed-session event
        # can arrive before a binding, then be received again after a user
        # attaches it; the early ignored copy must not suppress that update.
        if event.id in self._seen_event_ids:
            return False
        self._seen_event_ids.add(event.id)
        return True

    @staticmethod
    def _item_key(event: Envelope) -> str:
        # JSON is a stable, reversible representation of the full identity.
        # ``iid`` alone collides when backends reuse tool ids across turns.
        return json.dumps([event.session_id, event.turn_id, event.item_id], separators=(",", ":"))

    @classmethod
    def _output_key(cls, event: Envelope, item: dict[str, Any]) -> str:
        return json.dumps(
            [event.session_id, item.get("tid") or event.turn_id, item.get("iid")],
            separators=(",", ":"),
        )

    @classmethod
    def _tool_key(cls, event: Envelope) -> str:
        return cls._item_key(event)

    @staticmethod
    def _tool_rank(kind: object) -> int:
        return {"tool.started": 0, "tool.updated": 1, "tool.completed": 2}.get(kind, -1)

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
            "actionStatus": self.action_status,
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
