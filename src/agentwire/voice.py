from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from yarl import URL

from agentwire.text import clean_text

_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
_MAX_WAV_BYTES = 32 * 1024 * 1024
_MAX_CONTENT_BYTES = 64 * 1024
# Bound raw output as well as the smaller, cleaned prompt.
_MAX_STDOUT_BYTES = 1024 * 1024
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=30)
_FFMPEG_TIMEOUT = 120
_WHISPER_TIMEOUT = 600

_VOICE_FALLBACK = re.compile(
    r"\[voice(?: (?P<encrypted>encrypted))? "
    r"(?P<duration>[0-9]+(?::[0-9]{2}){1,2}) "
    r"(?P<mime>audio/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126})"
    r"(?: expires=(?P<expiry>[^\]\s]+))?\] (?P<url>[^\s<>]+)",
    re.IGNORECASE,
)
_RFC3339 = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt](?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)
_SILENCE = re.compile(r"\[(?:BLANK_AUDIO|NO_SPEECH)\]", re.IGNORECASE)


class VoiceError(ValueError):
    """A fixed, safe failure category suitable for an owner-visible message."""


@dataclass(frozen=True)
class VoiceMessage:
    duration_seconds: int
    mime_type: str
    fetch_url: str


def _fetch_url(value: str) -> str:
    fetch_url = value.partition("#")[0]
    try:
        parsed = urlsplit(fetch_url)
        valid = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and "\\" not in parsed.netloc
            and not re.search(r"[\x00-\x20\x7f]", value)
        )
        # Accessing port also rejects malformed or out-of-range authorities.
        _ = parsed.port
    except ValueError:
        raise VoiceError("invalid voice URL") from None
    if not valid:
        raise VoiceError("invalid voice URL")
    return fetch_url


def parse_voice_message(text: str) -> VoiceMessage | None:
    match = _VOICE_FALLBACK.fullmatch(text)
    if match is None:
        return None
    if match["encrypted"]:
        raise VoiceError("encrypted voice notes are unsupported")
    mime_type = match["mime"].lower()
    if mime_type not in {"audio/ogg", "audio/mp4"}:
        raise VoiceError("unsupported audio type")

    parts = match["duration"].split(":")
    # Avoid unbounded integer conversion even when called outside IRC's line limit.
    leading = parts[0].lstrip("0") or "0"
    if len(leading) > 3:
        raise VoiceError("voice note exceeds 15 minutes")
    values = [int(leading), *(int(part) for part in parts[1:])]
    if any(value > 59 for value in values[1:]):
        raise VoiceError("invalid voice duration")
    duration_seconds = 0
    for value in values:
        duration_seconds = duration_seconds * 60 + value
    if duration_seconds > 900:
        raise VoiceError("voice note exceeds 15 minutes")

    expiry = match["expiry"]
    if expiry is not None:
        if _RFC3339.fullmatch(expiry) is None:
            raise VoiceError("invalid voice expiry")
        try:
            expires_at = datetime.fromisoformat(expiry.upper().replace("Z", "+00:00"))
        except ValueError:
            raise VoiceError("invalid voice expiry") from None
        if expires_at <= datetime.now(UTC):
            raise VoiceError("voice note expired")

    return VoiceMessage(duration_seconds, mime_type, _fetch_url(match["url"]))


def voice_action_id(owner: str, channel: str, backend: str, sid: str, fetch_url: str) -> str:
    fetch_url = fetch_url.partition("#")[0]
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"agentwire:voice:{owner}:{channel}:{backend}:{sid}:{fetch_url}"
        )
    )


async def _download(fetch_url: str, target: Path) -> None:
    try:
        async with (
            aiohttp.ClientSession(
                timeout=_DOWNLOAD_TIMEOUT,
                headers={"Accept-Encoding": "identity"},
                auto_decompress=False,
                trust_env=False,
            ) as session,
            session.get(URL(fetch_url, encoded=True), allow_redirects=False) as response,
        ):
            if response.status != 200:
                raise VoiceError("audio download rejected")
            encodings = response.headers.getall("Content-Encoding", ["identity"])
            if any(encoding.strip().lower() != "identity" for encoding in encodings):
                raise VoiceError("encoded audio response rejected")
            if (
                response.content_length is not None
                and response.content_length > _MAX_DOWNLOAD_BYTES
            ):
                raise VoiceError("audio download too large")
            with os.fdopen(
                os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
            ) as sink:
                size = 0
                async for chunk in response.content.iter_chunked(64 * 1024):
                    size += len(chunk)
                    if size > _MAX_DOWNLOAD_BYTES:
                        raise VoiceError("audio download too large")
                    sink.write(chunk)
    except VoiceError:
        raise
    except TimeoutError:
        raise VoiceError("audio download timed out") from None
    except (aiohttp.ClientError, OSError, ValueError):
        raise VoiceError("audio download failed") from None


async def _run_media(
    command: list[str], directory: Path, *, capture: bool, timeout: float
) -> bytes:
    failure = "transcription failed" if capture else "audio conversion failed"
    timed_out = "transcription timed out" if capture else "audio conversion timed out"
    # Shield creation so cancellation cannot lose a child between fork and its handle.
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=directory,
            env={
                "HOME": str(directory),
                "TMPDIR": str(directory),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
        )
    )
    try:
        async with asyncio.timeout(timeout):
            process = await asyncio.shield(spawn)
            output = bytearray()
            if capture:
                assert process.stdout is not None
                while chunk := await process.stdout.read(64 * 1024):
                    if len(output) + len(chunk) > _MAX_STDOUT_BYTES:
                        raise VoiceError("transcription too large")
                    output.extend(chunk)
            if await process.wait() != 0:
                raise VoiceError(failure)
            return bytes(output)
    except TimeoutError:
        raise VoiceError(timed_out) from None
    except OSError:
        raise VoiceError(failure) from None
    finally:
        # communicate drains any remaining pipe after SIGKILL, avoiding a full-pipe
        # deadlock in Process.wait when a child exceeded the output bound.
        with contextlib.suppress(OSError):
            process = await spawn
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.communicate()


async def transcribe_voice(message: VoiceMessage, model_path: Path) -> str:
    """Return cleaned text without the bridge's '[voice] ' prefix."""
    fetch_url = _fetch_url(message.fetch_url)
    try:
        ffmpeg = shutil.which("ffmpeg")
        whisper = shutil.which("whisper-cli")
        if ffmpeg is None:
            raise VoiceError("ffmpeg unavailable")
        if whisper is None:
            raise VoiceError("whisper-cli unavailable")
        ffmpeg = str(Path(ffmpeg).resolve())
        whisper = str(Path(whisper).resolve())
        model_path = model_path.absolute()
        with tempfile.TemporaryDirectory(prefix="agentwire-voice-") as name:
            directory = Path(name)
            source = directory / "input"
            wav = directory / "audio.wav"
            await _download(fetch_url, source)
            os.close(os.open(wav, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            await _run_media(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-xerror",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-t",
                    "900",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(wav),
                ],
                directory,
                capture=False,
                timeout=_FFMPEG_TIMEOUT,
            )
            if wav.stat().st_size > _MAX_WAV_BYTES:
                raise VoiceError("decoded audio too large")
            output = await _run_media(
                [whisper, "-m", str(model_path), "-f", str(wav), "-nt", "-np", "-l", "en"],
                directory,
                capture=True,
                timeout=_WHISPER_TIMEOUT,
            )
            try:
                transcript = " ".join(_SILENCE.sub("", clean_text(output.decode("utf-8"))).split())
            except UnicodeDecodeError:
                raise VoiceError("invalid transcription encoding") from None
            if not transcript:
                raise VoiceError("empty transcription")
            if len(("[voice] " + transcript).encode("utf-8")) > _MAX_CONTENT_BYTES:
                raise VoiceError("transcription too large")
            return transcript
    except OSError:
        raise VoiceError("media storage failed") from None
