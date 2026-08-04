# Agentwire

Agentwire exposes live Codex, OpenCode, and Claude sessions as a structured agent harness over IRCv3. A
supporting client renders sessions, turns, plans, tool cards, approvals, questions, queues, usage,
and history while the channel remains a useful readable transcript.

The canonical protocol is [`docs/agentwire-irc-v1.md`](docs/agentwire-irc-v1.md). Its JSON Schema
and interoperability fixtures live in [`protocol/`](protocol/). The old `!command` chat interface
has been removed; all control messages now use the authenticated protocol tag.

## Development

Enter the pinned Nix environment through direnv or explicitly:

```console
nix develop
```

Run the checks with:

```console
nix develop -c sh -c 'ruff format --check src tests && ruff check src tests && pytest -q'
```

The same flake supplies the development environment, checks, package, and deployment commands.

## Configuration and startup

Copy [`config.example.toml`](config.example.toml) to a private location and create its mode-0600
secrets env file. The state path is a private SQLite database containing bindings, action
deduplication, event history, and prompt queues.
On first use of an old JSON state path, Agentwire imports its bindings and preserves the original
as a mode-0600 `.legacy-json` backup.

The foreground stack starts the SSH tunnel, Codex app-server, and bridge. It starts OpenCode only
when an OpenCode-backed channel is configured. Claude needs no server here: when a Claude-backed
channel is configured, the bridge starts one `claude` CLI subprocess per session through the
Claude Agent SDK and stops it with the session. Because there is no shared Claude server, live
session discovery only sees sessions this bridge is running; `agentwire doctor` verifies the
`claude` binary and its credentials (`claude auth status`, or the configured `api_key_env`).

```console
nix run .#stack
```

The bridge requires TLS, SASL, account tags, message tags, server time, batches, echo messages,
labeled responses, standard replies, multiline messages, chat history, and event playback. It
will not connect with a reduced protocol. Configure Ergo 2.19.0 or newer with persistent SQLite
history, a 30-day channel retention period, and stored `TAGMSG` values for
`+trevarj.github.io/agentwire`.

Set a channel topic to activate the harness only after testing:

```text
agentwire:v1;account=your-account;agent=bot-account;backend=codex | Project title
```

The three parameters answer three different questions. `backend` picks the engine (`codex`,
`opencode`, or `claude`) and must match what [`config.example.toml`](config.example.toml) assigns
to that channel. `account` and `agent` are IRC account names, never engine names: `account` is
the controller allowed to issue actions, and `agent` is the bot account whose messages clients
trust as backend state. `agent=claude` would mean an IRC account literally named "claude" — the
engine is chosen only by `backend`. The single-account form is:

```text
agentwire:v1;account=agentwire;agent=agentwire;backend=claude | Project title
```

Both accounts come from the IRCv3 `account` tag, not the nickname, and `agent` must be the
bridge's own account (its configured nickname). Running one SASL account for both — separating
projects by channel — is supported: set `account` and `agent` to the same name and make
`owner_account` match the bridge nickname. Separate accounts remain the safer split, since one
shared credential can both publish state and issue owner commands. Removing the prefix suspends
the harness and pauses queue dispatch without canceling active backend work.

## Reference client

The dependency-free reference library is `agentwire.reference_client`. Its JSONL CLI lets mobile
client development validate topic parsing, fragments, actions, and render state without embedding
Agentwire:

```console
nix develop -c sh -c 'PYTHONPATH=src python -m agentwire.reference_client'
```

Example input:

```json
{"op":"topic","topic":"agentwire:v1;account=trev;agent=agentwire;backend=codex | Test"}
{"op":"ingest","tag":"{\"at\":1785400000000,...}"}
{"op":"action","kind":"sync.request"}
{"op":"state"}
```

Each output is one minified JSON object. An `action` response contains the exact `PRIVMSG` or
`TAGMSG` sequence and tag values the IRC layer should send.

## Privacy model

Workspace paths resolve beneath configured roots. Sensitive question text is never relayed and is
answerable only in an attached TUI, though it may be skipped from IRC. Tool events contain only
safe structured metadata. If a high-confidence secret detector matches an assistant message, the
entire message is omitted from IRC. There is no wire override and no external paste service.
