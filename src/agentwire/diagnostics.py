"""Bounded diagnostic reports shared by local checks and live IRC reads."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

STATUSES = frozenset({"ok", "warning", "error", "unknown"})
BOOLEAN_FACTS = frozenset(
    {
        "ready",
        "closed",
        "connected",
        "authenticated",
        "capabilitiesReady",
        "active",
        "bound",
        "busy",
        "available",
        "private",
        "valid",
    }
)
COUNT_FACTS = frozenset(
    {"pendingRequests", "sessionCount", "socketCount", "queueDepth", "oldestQueuedMs"}
)


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    code: str
    status: str
    explanation: str
    facts: dict[str, bool | int | float] = field(default_factory=dict)
    next_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "status": self.status,
            "explanation": self.explanation,
            "facts": dict(self.facts),
        }
        if self.next_step is not None:
            value["nextStep"] = self.next_step
        return value


def report(source: str, checks: list[DiagnosticCheck]) -> dict[str, Any]:
    value = {
        "schemaVersion": 1,
        "generatedAt": int(time.time() * 1000),
        "source": source,
        "checks": [check.to_dict() for check in checks],
    }
    validate_report(value)
    return value


def validate_report(value: dict[str, Any]) -> None:
    """Reject arbitrary facts and nested payloads at the protocol boundary."""
    if set(value) != {"schemaVersion", "generatedAt", "source", "checks"}:
        raise ValueError("diagnostics report has invalid fields")
    if isinstance(value["schemaVersion"], bool) or value["schemaVersion"] != 1:
        raise ValueError("unsupported diagnostics schema version")
    generated = value["generatedAt"]
    if (
        isinstance(generated, bool)
        or not isinstance(generated, (int, float))
        or not 0 <= generated <= 2**63 - 1
        or int(generated) != generated
    ):
        raise ValueError("diagnostics generatedAt must be a timestamp")
    if value["source"] not in ("doctor", "bridge"):
        raise ValueError("diagnostics source is invalid")
    checks = value["checks"]
    if not isinstance(checks, list) or not 1 <= len(checks) <= 64:
        raise ValueError("diagnostics requires bounded checks")
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) - {"code", "status", "explanation", "facts", "nextStep"}
            or not {"code", "status", "explanation", "facts"} <= check.keys()
        ):
            raise ValueError("diagnostic check has invalid fields")
        code = check["code"]
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9.]{0,79}", code):
            raise ValueError("diagnostic code is invalid")
        if not isinstance(check["status"], str) or check["status"] not in STATUSES:
            raise ValueError("diagnostic status is invalid")
        for key in ("explanation", "nextStep"):
            if key in check and (
                not isinstance(check[key], str) or not 1 <= len(check[key]) <= 300
            ):
                raise ValueError("diagnostic text must be bounded")
        facts = check["facts"]
        if not isinstance(facts, dict) or set(facts) - (BOOLEAN_FACTS | COUNT_FACTS):
            raise ValueError("diagnostic facts are not allowlisted")
        for key, fact in facts.items():
            if key in BOOLEAN_FACTS:
                if not isinstance(fact, bool):
                    raise ValueError("diagnostic boolean fact is invalid")
            elif (
                isinstance(fact, bool)
                or not isinstance(fact, (int, float))
                or not 0 <= fact <= 2**63 - 1
            ):
                raise ValueError("diagnostic count fact is invalid")


def backend_readiness(ready: bool, closed: bool, **counts: int) -> DiagnosticCheck:
    return DiagnosticCheck(
        "backend.ready",
        "ok" if ready and not closed else "warning",
        "Backend is ready." if ready and not closed else "Backend is not ready.",
        {"ready": ready and not closed, "closed": closed, **counts},
    )
