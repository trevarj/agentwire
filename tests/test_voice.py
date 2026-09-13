from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

from agentwire import voice
from agentwire.voice import (
    VoiceError,
    VoiceMessage,
    parse_voice_message,
    transcribe_voice,
    voice_action_id,
)

_FETCH_URL = "https://media.example/voice.ogg?sig=private%2fsignature&part=a%2Bb&empty="
_FRAGMENT = "#motd-wave=0123abcd&key=private-fragment"
_FALLBACK = f"[voice 0:07 audio/ogg] {_FETCH_URL}{_FRAGMENT}"
_SECRET = "sentinel-voice-parent-credential"
_DOWNLOAD_LIMIT = 25 * 1024 * 1024


@pytest.mark.parametrize("mime,extension", [("audio/ogg", "ogg"), ("audio/mp4", "m4a")])
def test_parses_production_voice_with_expiry_and_waveform(mime: str, extension: str) -> None:
    url = f"https://files.example/voice.{extension}?signature=abc%2fdef+ghi"
    body = f"[voice 0:07 {mime} expires=2099-09-12T13:42:50.123Z] {url}#motd-wave=abcd"
    assert parse_voice_message(body) == VoiceMessage(7, mime, url)


@pytest.mark.parametrize("duration,seconds", [("0:00", 0), ("14:59", 899), ("15:00", 900)])
def test_accepts_duration_through_fifteen_minutes(duration: str, seconds: int) -> None:
    assert parse_voice_message(f"[voice {duration} audio/ogg] {_FETCH_URL}") == VoiceMessage(
        seconds, "audio/ogg", _FETCH_URL
    )


@pytest.mark.parametrize(
    "body",
    [
        "ordinary conversation",
        "Please transcribe " + _FALLBACK,
        _FALLBACK + " then reply",
        _FALLBACK + " https://other.example/audio",
        " " + _FALLBACK,
        _FALLBACK + "\n",
        "[voice 7 audio/ogg] https://files.example/a",
        "[voice 0:7 audio/ogg] https://files.example/a",
        "[voice 0:07 text/plain] https://files.example/a",
        "[voice 0:07 audio/ogg;codecs=opus] https://files.example/a",
        "[voice 0:07 audio/ogg https://files.example/a",
    ],
)
def test_noncanonical_conversation_is_not_voice(body: str) -> None:
    assert parse_voice_message(body) is None


@pytest.mark.parametrize(
    "header,category",
    [
        ("encrypted 0:07 audio/ogg", "encrypted"),
        ("0:07 audio/wav", "unsupported"),
        ("0:60 audio/ogg", "duration"),
        ("1:60:00 audio/ogg", "duration"),
        ("1:00:60 audio/ogg", "duration"),
        ("15:01 audio/ogg", "15 minutes"),
        ("16:00 audio/ogg", "15 minutes"),
        ("1:00:00 audio/ogg", "15 minutes"),
        ("9" * 5000 + ":00 audio/ogg", "15 minutes"),
    ],
)
def test_recognized_unsafe_voice_fails_before_io(header: str, category: str) -> None:
    with pytest.raises(VoiceError, match=category):
        parse_voice_message(f"[voice {header}] {_FETCH_URL}{_FRAGMENT}")


@pytest.mark.parametrize(
    "url",
    [
        "http://files.example/audio",
        "ftp://files.example/audio",
        "https:///audio",
        "https://user@files.example/audio",
        "https://:password@files.example/audio",
        "https://@files.example/audio",
        "https://files.example:bad/audio",
        "https://files.example:65536/audio",
        "https://[invalid]/audio",
        "https://files.example\\other/audio",
        "https://files.example/au\x00dio",
    ],
)
def test_rejects_unsafe_https_authorities(url: str) -> None:
    with pytest.raises(VoiceError, match="URL") as error:
        parse_voice_message(f"[voice 0:07 audio/ogg] {url}")
    assert url not in str(error.value)


@pytest.mark.parametrize(
    "expiry",
    [
        "not-a-date",
        "2099-09-12",
        "2099-09-12T13:42:50",
        "2099-09-12T13:42:50Zjunk",
        "2099-02-30T13:42:50Z",
        "2099-09-12T24:00:00Z",
        "2099-09-12T13:42:50+00:99",
        "2099-09-12T13:42:50+24:00",
    ],
)
def test_rejects_malformed_expiry(expiry: str) -> None:
    with pytest.raises(VoiceError, match="expiry"):
        parse_voice_message(f"[voice 0:07 audio/ogg expires={expiry}] {_FETCH_URL}")


def test_expiry_requires_future_instant_and_accepts_rfc3339_offsets() -> None:
    with pytest.raises(VoiceError, match="expired"):
        parse_voice_message(f"[voice 0:07 audio/ogg expires=2000-01-01T00:00:00Z] {_FETCH_URL}")
    assert parse_voice_message(
        f"[voice 0:07 audio/mp4 expires=2099-01-01t01:02:03.123456789+02:30] {_FETCH_URL}"
    ) == VoiceMessage(7, "audio/mp4", _FETCH_URL)


def test_action_id_ignores_fragments_but_scopes_owner_channel_backend_session_and_query() -> None:
    scope = ("owner", "#pm", "omp", "session-one", _FETCH_URL)
    identifier = voice_action_id(*scope)
    assert identifier == str(uuid.uuid5(uuid.NAMESPACE_URL, "agentwire:voice:" + ":".join(scope)))
    assert voice_action_id(*scope[:-1], _FETCH_URL + _FRAGMENT) == identifier
    assert voice_action_id(*scope[:-1], _FETCH_URL + "#different-fragment") == identifier
    for index, replacement in enumerate(
        ["other-owner", "#project", "codex", "session-two", _FETCH_URL + "changed"]
    ):
        changed = list(scope)
        changed[index] = replacement
        assert voice_action_id(*changed) != identifier


@pytest.fixture
async def media_http(monkeypatch: pytest.MonkeyPatch):
    async def respond(request: web.Request) -> web.StreamResponse:
        return web.Response(body=b"recorded audio")

    state = SimpleNamespace(handler=respond, requests=[], urls=[], sessions=[])

    async def serve(request: web.Request) -> web.StreamResponse:
        state.requests.append((request.raw_path, request.headers.get("Accept-Encoding")))
        return await state.handler(request)

    application = web.Application()
    application.router.add_route("GET", "/{path:.*}", serve)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    client_session = aiohttp.ClientSession

    # Only replace the wire destination: the real aiohttp client still handles
    # streaming, signed-query encoding, headers, deadlines, and redirects.
    class LocalSession:
        def __init__(self, **options):
            self.session = client_session(**options)
            state.sessions.append(self.session)

        async def __aenter__(self):
            await self.session.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self.session.__aexit__(*args)

        def get(self, url, **options):
            url = URL(url, encoded=True)
            state.urls.append(str(url))
            local_url = url.with_scheme("http").with_host("127.0.0.1").with_port(port)
            return self.session.get(local_url, **options)

    monkeypatch.setattr(voice.aiohttp, "ClientSession", LocalSession)
    try:
        yield state
    finally:
        await runner.cleanup()
        assert all(session.closed for session in state.sessions)


@pytest.fixture
async def media_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binaries = tmp_path / "media tools; no shell"
    temporary = tmp_path / "temporary"
    binaries.mkdir()
    temporary.mkdir(mode=0o700)
    model = tmp_path / "model; not a command.bin"
    model.write_bytes(b"test model")
    monkeypatch.setenv("PATH", str(binaries))
    monkeypatch.setenv("AGENTWIRE_SENTINEL_SECRET", _SECRET)
    monkeypatch.setattr(voice.tempfile, "tempdir", str(temporary))
    processes = []
    create_subprocess = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await create_subprocess(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(voice.asyncio, "create_subprocess_exec", spawn)

    def record_path(name: str) -> Path:
        return tmp_path / f"{name}.json"

    def install(name: str, body: str) -> None:
        script = binaries / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, stat, sys, time, wave\n"
            "from pathlib import Path\n"
            "home = Path(os.environ['HOME'])\n"
            f"record = Path({str(record_path(name))!r})\n"
            "data = {'pid': os.getpid(), 'env': dict(os.environ), 'home': str(home),\n"
            "        'directory_mode': stat.S_IMODE(home.stat().st_mode),\n"
            "        'files': {p.name: stat.S_IMODE(p.stat().st_mode) for p in home.iterdir()}}\n"
            "pending = record.with_suffix('.pending')\n"
            "pending.write_text(json.dumps(data))\n"
            "pending.replace(record)\n" + body,
            encoding="utf-8",
        )
        script.chmod(0o700)

    install(
        "ffmpeg",
        "source = Path(sys.argv[sys.argv.index('-i') + 1])\n"
        "assert source.is_file() and '://' not in str(source)\n"
        "with wave.open(sys.argv[-1], 'wb') as wav:\n"
        "    wav.setnchannels(1)\n"
        "    wav.setsampwidth(2)\n"
        "    wav.setframerate(16000)\n"
        "    wav.writeframes(b'\\x00\\x00' * 160)\n",
    )
    install(
        "whisper-cli",
        "assert Path(sys.argv[sys.argv.index('-m') + 1]).read_bytes() == b'test model'\n"
        "with wave.open(sys.argv[sys.argv.index('-f') + 1], 'rb') as wav:\n"
        "    assert wav.getnchannels() == 1 and wav.getframerate() == 16000\n"
        "sys.stdout.buffer.write(b' \\x02Reply\\r\\n  with exactly VOICE_OK.\\t\\n')\n",
    )

    def assert_clean() -> None:
        assert list(temporary.iterdir()) == []
        for process in processes:
            with pytest.raises(ProcessLookupError):
                os.kill(process.pid, 0)
            with pytest.raises(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)

    tools = SimpleNamespace(
        binaries=binaries,
        temporary=temporary,
        model=model,
        install=install,
        record_path=record_path,
        assert_clean=assert_clean,
    )
    try:
        yield tools
    finally:
        # A failed regression must not leave its intentionally sleeping child alive.
        for process in processes:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.communicate()


async def _started(tools, name: str) -> dict:
    async with asyncio.timeout(5):
        path = tools.record_path(name)
        while not path.exists():
            await asyncio.sleep(0.01)
        return json.loads(path.read_text(encoding="utf-8"))


def _message() -> VoiceMessage:
    message = parse_voice_message(_FALLBACK)
    assert message is not None
    return message


def _assert_safe(error: BaseException, tools, caplog, capfd) -> None:
    captured = capfd.readouterr()
    exposed = "".join(traceback.format_exception(error)) + caplog.text + captured.out + captured.err
    for sensitive in (_SECRET, _FETCH_URL, "private-fragment", str(tools.model)):
        assert sensitive not in exposed
    assert len(str(error)) < 100


@pytest.mark.asyncio
async def test_transcribes_with_private_files_no_parent_secret_and_exact_signed_query(
    media_http, media_tools, caplog, capfd
) -> None:
    transcript = await transcribe_voice(_message(), media_tools.model)
    assert transcript == "Reply with exactly VOICE_OK."
    assert media_http.urls == [_FETCH_URL]
    assert media_http.requests == [
        ("/voice.ogg?sig=private%2fsignature&part=a%2Bb&empty=", "identity")
    ]
    for name in ("ffmpeg", "whisper-cli"):
        record = json.loads(media_tools.record_path(name).read_text(encoding="utf-8"))
        assert "AGENTWIRE_SENTINEL_SECRET" not in record["env"]
        assert "PATH" not in record["env"]
        assert record["env"]["TMPDIR"] == record["home"]
        assert record["directory_mode"] == 0o700
        assert record["files"] == {"input": 0o600, "audio.wav": 0o600}
        assert not Path(record["home"]).exists()
    captured = capfd.readouterr()
    assert _FETCH_URL not in caplog.text + captured.out + captured.err
    assert _SECRET not in transcript
    media_tools.assert_clean()


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", ["ffmpeg", "whisper-cli"])
async def test_missing_media_binary_fails_without_downloading(
    binary, media_http, media_tools, caplog, capfd
) -> None:
    (media_tools.binaries / binary).unlink()
    with pytest.raises(VoiceError, match="unavailable") as error:
        await transcribe_voice(_message(), media_tools.model)
    assert media_http.requests == []
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,headers",
    [
        (302, [("Location", "/redirect-target")]),
        (403, []),
        (200, [("Content-Encoding", "gzip")]),
        (200, [("Content-Encoding", "identity"), ("Content-Encoding", "gzip")]),
    ],
)
async def test_rejects_redirects_status_and_nonidentity_encoding_without_children(
    status, headers, media_http, media_tools, caplog, capfd
) -> None:
    async def respond(request):
        return web.Response(status=status, headers=headers, text=_FETCH_URL + _SECRET)

    media_http.handler = respond
    with pytest.raises(VoiceError, match="rejected") as error:
        await transcribe_voice(_message(), media_tools.model)
    assert len(media_http.requests) == 1
    assert not media_tools.record_path("ffmpeg").exists()
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
async def test_declared_oversized_download_is_rejected_before_body(
    media_http, media_tools, caplog, capfd
) -> None:
    async def respond(request):
        return web.Response(headers={"Content-Length": str(_DOWNLOAD_LIMIT + 1)})

    media_http.handler = respond
    with pytest.raises(VoiceError, match="too large") as error:
        await transcribe_voice(_message(), media_tools.model)
    assert not media_tools.record_path("ffmpeg").exists()
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [0, 1])
async def test_streamed_download_limit_is_inclusive_without_declared_length(
    extra, media_http, media_tools, caplog, capfd
) -> None:
    async def respond(request):
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            chunk = b"x" * (64 * 1024)
            for _ in range(_DOWNLOAD_LIMIT // len(chunk)):
                await response.write(chunk)
            if extra:
                await response.write(b"x")
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    media_http.handler = respond
    if extra:
        with pytest.raises(VoiceError, match="too large") as error:
            await transcribe_voice(_message(), media_tools.model)
        assert not media_tools.record_path("ffmpeg").exists()
        _assert_safe(error.value, media_tools, caplog, capfd)
    else:
        assert (
            await transcribe_voice(_message(), media_tools.model) == "Reply with exactly VOICE_OK."
        )
    media_tools.assert_clean()


@pytest.mark.asyncio
async def test_interrupted_download_is_safe_and_removes_partial_file(
    media_http, media_tools, caplog, capfd
) -> None:
    async def respond(request):
        response = web.StreamResponse(headers={"Content-Length": "1000"})
        await response.prepare(request)
        await response.write(b"partial audio")
        request.transport.close()
        return response

    media_http.handler = respond
    with pytest.raises(VoiceError, match="download failed") as error:
        await transcribe_voice(_message(), media_tools.model)
    assert not media_tools.record_path("ffmpeg").exists()
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_download_timeout_or_cancellation_closes_and_cleans_partial_file(
    cancel, media_http, media_tools, monkeypatch, caplog, capfd
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"partial audio")
        started.set()
        await release.wait()
        return response

    media_http.handler = respond
    if not cancel:
        monkeypatch.setattr(
            voice, "_DOWNLOAD_TIMEOUT", aiohttp.ClientTimeout(total=0.2, sock_read=0.2)
        )
    task = asyncio.create_task(transcribe_voice(_message(), media_tools.model))
    try:
        await asyncio.wait_for(started.wait(), 5)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(VoiceError, match="download timed out") as error:
                await asyncio.wait_for(task, 5)
            _assert_safe(error.value, media_tools, caplog, capfd)
    finally:
        release.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, VoiceError):
            await task
    assert not media_tools.record_path("ffmpeg").exists()
    media_tools.assert_clean()


@pytest.mark.asyncio
async def test_oversized_decoded_wav_never_reaches_whisper(
    media_http, media_tools, caplog, capfd
) -> None:
    media_tools.install(
        "ffmpeg",
        "with open(sys.argv[-1], 'wb') as wav:\n    wav.truncate(32 * 1024 * 1024 + 1)\n",
    )
    with pytest.raises(VoiceError, match="decoded audio too large") as error:
        await transcribe_voice(_message(), media_tools.model)
    assert not media_tools.record_path("whisper-cli").exists()
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [b" \n\t", b"[BLANK_AUDIO]\n", b"[NO_SPEECH]", b"\xff"])
async def test_blank_silent_or_invalid_utf8_transcription_is_rejected(
    output, media_http, media_tools, caplog, capfd
) -> None:
    media_tools.install("whisper-cli", f"sys.stdout.buffer.write({output!r})\n")
    with pytest.raises(VoiceError, match="empty|encoding") as error:
        await transcribe_voice(_message(), media_tools.model)
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize("characters", [32764, 32765])
async def test_final_cleaned_utf8_content_limit_includes_voice_prefix(
    characters, media_http, media_tools, caplog, capfd
) -> None:
    media_tools.install(
        "whisper-cli",
        f"sys.stdout.buffer.write((' \\x02' + 'é' * {characters} + '\\r\\n').encode('utf-8'))\n",
    )
    if characters == 32764:
        assert await transcribe_voice(_message(), media_tools.model) == "é" * characters
    else:
        with pytest.raises(VoiceError, match="too large") as error:
            await transcribe_voice(_message(), media_tools.model)
        _assert_safe(error.value, media_tools, caplog, capfd)
    media_tools.assert_clean()


@pytest.mark.asyncio
async def test_unbounded_child_stdout_is_stopped_and_reaped(
    media_http, media_tools, caplog, capfd
) -> None:
    media_tools.install("whisper-cli", "while True:\n    os.write(1, b'x' * 65536)\n")
    with pytest.raises(VoiceError, match="too large") as error:
        await asyncio.wait_for(transcribe_voice(_message(), media_tools.model), 5)
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", ["ffmpeg", "whisper-cli"])
async def test_nonzero_media_child_exit_does_not_disclose_stdout_or_stderr(
    binary, media_http, media_tools, caplog, capfd
) -> None:
    media_tools.install(
        binary,
        f"sys.stderr.write({_SECRET + _FETCH_URL!r})\n"
        f"sys.stdout.write({_SECRET + _FETCH_URL!r})\n"
        "raise SystemExit(23)\n",
    )
    with pytest.raises(VoiceError, match="failed") as error:
        await transcribe_voice(_message(), media_tools.model)
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", ["ffmpeg", "whisper-cli"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_child_timeout_or_cancellation_kills_reaps_and_cleans(
    binary, cancel, media_http, media_tools, monkeypatch, caplog, capfd
) -> None:
    media_tools.install(binary, "time.sleep(60)\n")
    if not cancel:
        setting = "_FFMPEG_TIMEOUT" if binary == "ffmpeg" else "_WHISPER_TIMEOUT"
        monkeypatch.setattr(voice, setting, 1)
    task = asyncio.create_task(transcribe_voice(_message(), media_tools.model))
    try:
        await _started(media_tools, binary)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(VoiceError, match="timed out") as error:
                await asyncio.wait_for(task, 5)
            _assert_safe(error.value, media_tools, caplog, capfd)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, VoiceError):
            await task
    media_tools.assert_clean()


@pytest.mark.asyncio
async def test_spawn_failure_is_safe_and_cleans_downloaded_media(
    media_http, media_tools, caplog, capfd
) -> None:
    (media_tools.binaries / "ffmpeg").write_text(
        "#!/missing-interpreter-private-signature\n", encoding="utf-8"
    )
    with pytest.raises(VoiceError, match="conversion failed") as error:
        await transcribe_voice(_message(), media_tools.model)
    media_tools.assert_clean()
    _assert_safe(error.value, media_tools, caplog, capfd)


@pytest.mark.asyncio
async def test_cancellation_during_spawn_waits_for_child_ownership_and_reaps(
    media_http, media_tools, monkeypatch
) -> None:
    media_tools.install("ffmpeg", "time.sleep(60)\n")
    started = asyncio.Event()
    release = asyncio.Event()
    original_spawn = asyncio.create_subprocess_exec

    async def delayed_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        started.set()
        await release.wait()
        return process

    monkeypatch.setattr(voice.asyncio, "create_subprocess_exec", delayed_spawn)
    task = asyncio.create_task(transcribe_voice(_message(), media_tools.model))
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    finally:
        release.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, VoiceError):
            await task
    media_tools.assert_clean()
