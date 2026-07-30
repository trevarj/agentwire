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


def safe_one_line(text: str, max_bytes: int = 240) -> str:
    value = " ".join(clean_text(text).split())
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    return truncate_utf8(value, max(1, max_bytes - 3)).rstrip() + "..."
