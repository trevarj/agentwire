# Pi and OMP socket protocols

The `pi` backend reaches sessions over two transports that speak one JSONL
protocol:

- **Live TUI sessions.** A pi extension (`agentwire.ts`, deployed with the
  user's pi configuration) serves one unix socket per running pi process at
  `$XDG_RUNTIME_DIR/agentwire/pi/<pid>.sock` (falling back to the system temp
  directory when no runtime dir exists). Security is same-user filesystem
  permission: the directory is mode 0700.
- **Bridge-owned sessions.** When no live process owns a session, the backend
  spawns `pi --mode rpc` with `AGENTWIRE_SPAWNED=1` in the environment and
  speaks the same protocol over stdio. The environment variable stops the
  extension inside the spawned process from registering a duplicate socket.

Frames are LF-delimited JSON objects, exactly like pi's RPC mode: split
records on `\n` only, tolerate a trailing `\r`, and never treat U+2028/U+2029
as separators. The command surface is a subset of pi's RPC commands, so the
bridge shares one translator for both transports.

## Frames from pi

- `hello` — first frame on a socket connection:
  `{type, pv, sessionId, sessionFile, cwd, sessionName, model, thinkingLevel, busy, waiting, pid}`.
  The durable session identity used by the bridge is the session file stem,
  not `sessionId`.
- `session_changed` — same payload, re-announced after `/new`, `/resume`,
  fork, rename, model, thinking-level, or waiting-for-input changes.
- `agent_start` / `agent_settled` — busy boundary markers. `agent_settled`
  means no retry, compaction retry, or queued continuation remains, and closes
  the bridge's open turn.
- `message_end` — one condensed message:
  `{role: "user" | "assistant", text, toolCalls?, stopReason?, errorMessage?, timestamp?}`.
  Bulk payloads (images, file bodies) never cross the wire. Spawned RPC
  sessions stream pi's native uncondensed `message_end` instead; the backend
  normalizes both.
- `tool_execution_start` — `{toolCallId, toolName, args}` with arguments
  pruned to short scalar fields.
- `tool_execution_end` — `{toolCallId, toolName, isError, output}` with output
  truncated. RPC sessions send pi's native `result` object instead.
- `extension_ui_request` — only observed on spawned RPC sessions. `confirm`
  relays as an Agentwire approval; `select`, `input`, and `editor` relay as
  questions. Fire-and-forget methods are dropped. Live TUI sessions answer
  their dialogs in the TUI, consistent with the privacy model.

## Commands to pi

Every command carries an `id`; the reply is
`{id, type: "response", command, success, data?, error?}`.

- `prompt {message}` — steered automatically when the session is busy.
- `steer {message}` — rejected when idle.
- `follow_up {message}` — queued until the agent finishes.
- `abort`
- `get_state` — the `hello` payload.
- `get_entries {since?, limit?}` — condensed active-branch messages with
  stable entry ids and the active branch `leafId`; `since` is a strict cursor.
  The bridge pages live TUI history in 500-entry windows until it reaches that leaf.
- `get_available_models` / `set_model {provider, modelId}`
- `set_thinking_level {level}`
- `set_session_name {name}`
- `get_session_stats`

## Session identity and discovery

Sessions are keyed by the JSONL file stem
(`~/.pi/agent/sessions/--<cwd-with-dashes>--/<stem>.jsonl`), which is stable
across the TUI, the spawned RPC process, and on-disk history. The backend
scans the socket directory a few times per second for new live sessions,
removes sessions whose stream ends, and refuses to spawn-resume a session
whose file a live process already owns.

## OMP

The distinct `omp` backend has the same two ownership modes. Install
`agentwire-omp.ts` as `~/.omp/agent/extensions/agentwire.ts`; each live OMP TUI then serves a
backend-tagged socket at `$XDG_RUNTIME_DIR/agentwire/omp/<pid>.sock`. The bridge requires
`backend: "omp"` in its `hello` and keeps these sockets separate from Pi.

For a static channel or an OMP session with no live owner, the bridge starts
`omp --mode rpc --approval-mode always-ask --session-dir <cwd-bucket>` with
`AGENTWIRE_SPAWNED=1`. OMP begins stdio RPC with
a `ready` frame and may negotiate its chunked protocol 2; the live extension continues to expose
the canonical `hello`/response protocol above. OMP sessions live under
`~/.omp/agent/sessions`, and its hello advertises `model`, `effort`, and `delivery` settings.
