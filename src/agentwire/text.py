from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub("", text).strip()


def truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def preview(text: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    cleaned = clean_text(text)
    lines = cleaned.splitlines() or [""]
    truncated = len(lines) > max_lines or len(cleaned.encode("utf-8")) > max_bytes
    if not truncated:
        return cleaned, False
    marker = "… [truncated — use !paste for a 1h full-reply link]"
    budget = max(1, max_bytes - len(("\n" + marker).encode("utf-8")))
    body = "\n".join(lines[:max_lines])
    body = truncate_utf8(body, budget).rstrip()
    return f"{body}\n{marker}" if body else marker, True


def safe_one_line(text: str, max_bytes: int = 240) -> str:
    value = " ".join(clean_text(text).split())
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    return truncate_utf8(value, max(1, max_bytes - 3)).rstrip() + "..."
