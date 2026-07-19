from __future__ import annotations

import re
from dataclasses import dataclass

import aiohttp

from irc_bridge.config import PasteConfig
from irc_bridge.text import clean_text


class PasteError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class SecretFinding:
    rule: str


_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("credential URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@", re.I)),
    (
        "secret assignment",
        re.compile(
            r"(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|TOKEN|SECRET|API_KEY)"
            r"[A-Z0-9_]*\s*[:=]\s*['\"]?[^\s'\"]{8,}"
        ),
    ),
)


def scan_secrets(text: str) -> list[SecretFinding]:
    return [SecretFinding(name) for name, pattern in _SECRET_RULES if pattern.search(text)]


class LitterboxClient:
    def __init__(self, config: PasteConfig) -> None:
        self.config = config

    async def upload(self, text: str, force: bool = False) -> str:
        cleaned = clean_text(text)
        encoded = cleaned.encode("utf-8")
        if not encoded:
            raise PasteError("there is no reply to upload")
        if len(encoded) > self.config.max_bytes:
            raise PasteError(f"reply exceeds the {self.config.max_bytes}-byte paste limit")
        findings = scan_secrets(cleaned)
        if findings and not force:
            rules = ", ".join(sorted({finding.rule for finding in findings}))
            raise PasteError(
                f"secret scan blocked the upload ({rules}); use !paste-force to override"
            )
        form = aiohttp.FormData()
        form.add_field("reqtype", "fileupload")
        form.add_field("time", self.config.expiry)
        form.add_field(
            "fileToUpload",
            encoded,
            filename="agent-reply.txt",
            content_type="text/plain; charset=utf-8",
        )
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(self.config.url, data=form) as response,
            ):
                body = (await response.text()).strip()
                if response.status != 200:
                    raise PasteError(f"temporary paste host returned HTTP {response.status}")
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise PasteError(f"temporary paste upload failed: {exc}") from exc
        if not body.startswith("https://") or any(char.isspace() for char in body):
            raise PasteError("temporary paste host returned an invalid URL")
        return body
