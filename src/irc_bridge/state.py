from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path

from irc_bridge.models import ChannelBinding


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._bindings: dict[str, ChannelBinding] = {}

    async def load(self) -> dict[str, ChannelBinding]:
        async with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._bindings = {}
                return {}
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(f"cannot load bridge state {self.path}: {exc}") from exc
            if raw.get("version") != 1 or not isinstance(raw.get("channels"), dict):
                raise RuntimeError(f"unsupported bridge state format in {self.path}")
            bindings: dict[str, ChannelBinding] = {}
            for channel, item in raw["channels"].items():
                if not isinstance(item, dict):
                    continue
                try:
                    bindings[channel.lower()] = ChannelBinding(
                        backend=str(item["backend"]),
                        session_id=str(item["session_id"]),
                        cwd=str(item["cwd"]),
                    )
                except KeyError:
                    continue
            self._bindings = bindings
            return dict(bindings)

    async def set(self, channel: str, binding: ChannelBinding | None) -> None:
        async with self._lock:
            key = channel.lower()
            if binding is None:
                self._bindings.pop(key, None)
            else:
                self._bindings[key] = binding
            await asyncio.to_thread(self._write)

    def _write(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        payload = {
            "version": 1,
            "channels": {
                channel: {
                    "backend": binding.backend,
                    "session_id": binding.session_id,
                    "cwd": binding.cwd,
                }
                for channel, binding in sorted(self._bindings.items())
            },
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)
            raise
