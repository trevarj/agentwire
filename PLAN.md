# Agentwire: implementation plan

## Outcome

Run one foreground command, `nix run .#stack`, to start an SSH tunnel to Ergo,
the official Codex app-server, an OpenCode HTTP/SSE server, and an owner-only
Agentwire. `#codex` and `#opencode` each bind to one live agent session. The
same sessions remain attachable from the native TUIs.

The stack is deliberately manual and ephemeral: no system service, no offline
message replay, and no ingestion from IRC history. Stopping the foreground
command stops every child process.

## Architecture

```text
IRC client -> Ergo <-TLS/SASL over SSH tunnel-> Agentwire
                                                  |-- Codex app-server (UDS WebSocket)
                                                  `-- OpenCode serve (HTTP + global SSE)
```

- Codex uses the official app-server protocol from the installed release. It
  supports concurrent bridge and `codex --remote unix://...` connections.
- OpenCode uses its stable session HTTP routes, the queued/steered v2 prompt
  route, and the global SSE stream. `opencode attach` can coexist with it.
- Both adapters normalize complete assistant replies, turn boundaries,
  sanitized tool milestones, one-shot approvals, questions, and cancellation.
- The IRC layer uses TLS certificate verification, SASL PLAIN, IRCv3
  `account-tag`, message tags, and multiline batches when negotiated.

## Trust and privacy boundaries

- One dedicated SASL account is used by the bot. Only messages whose exact
  IRCv3 account tag matches the configured owner account are processed; all
  others are silently ignored.
- Ergo channels are registered to the owner, secret/invite-only and
  registered-users-only. The bot is granted channel access without a reusable
  channel key. Channel history is disabled for both bridge channels.
- Workspace paths must be absolute, must exist, and must resolve beneath a
  configured allowlisted root. Symlink escapes are rejected.
- Tool relays contain only category, start/finish, and success/failure. They
  never relay commands, paths, inputs, or tool output.
- Sensitive question prompts are not emitted to IRC and can only be answered
  in an attached TUI. IRC approvals are always one-shot; persistent grants stay
  a TUI-only operation.
- Full replies are uploaded only after `!paste`. A local scanner blocks likely
  credentials; `!paste-force` is the explicit override. Litterbox links are
  public and expire after one hour.
- Live config, certificates, credentials, and persisted state live outside the
  repository with mode 0600. No prompt/reply logs are written.

## IRC behavior

Each channel has one of four useful states: detached, idle with a binding, busy,
or holding a draft that has not been dispatched. A fresh channel can use
`!running` to find active work or `!new <workspace>` to create a session. Only
the channel-to-session/workspace binding persists across restarts; queues,
drafts, request aliases, and watch mode do not.

- Normal text starts a turn while idle. While busy it accumulates in a bounded
  held draft. `!next` commits the draft to the FIFO queue, `!steer` redirects
  the active turn with it, and `!discard` removes it. No draft is dispatched
  implicitly.
- `!running`/`!r` produces a numbered list of active sessions across allowed
  roots. `!sessions [workspace]` lists recent sessions, and `!use N` consumes
  either list. A backend session cannot be attached to two channels.
- Relative workspace names resolve under the configured allowed roots. Missing,
  ambiguous, non-directory, outside-root, and symlink-escape paths are rejected.
- Replies are posted only when complete and previewed at eight lines / 1,200
  UTF-8 bytes. The full reply stays available in the native TUI or via explicit
  `!paste`.
- External TUI turns are mirrored. `!watch quiet|concise|verbose` controls push
  detail; verbose tool activity is capped per turn to avoid IRC flooding. A
  completion/failure line advances the queued prompt.
- Approval and question IDs are short channel-local aliases (`A1`, `Q1`). If a
  TUI resolves one, IRC removes the alias and reports that it was handled in
  another client.
- On a dropped IRC connection the client reconnects with bounded exponential
  backoff, repeats SASL/CAP negotiation, and rejoins. Kicks trigger a rejoin.

Commands:

```text
!help [TOPIC|all]
!running / !r
!new WORKSPACE
!sessions [WORKSPACE]
!use [N] / !attach N
!detach
!status / !s
!last / !l
!watch [quiet|concise|verbose]
!next / !steer [TEXT] / !discard / !cancel
!yes [A1] / !no [A1]
!answer [Q1] ANSWER [ | ANSWER] / !skip [Q1]
!queue / !drop N|all
!paste / !paste-force
```

Compatibility aliases remain available: `!approve`, `!deny`, and `!reject`.

Question answers use `|` between separate questions, comma-separated values for
multi-select questions, and either option numbers or labels.

## Foreground supervisor

`nix run .#stack` validates the private configuration and starts:

1. `ssh -N -T` with `ExitOnForwardFailure` and keepalives, forwarding only a
   loopback port to Ergo's loopback TLS listener.
2. `codex app-server --listen unix://...` in a private runtime directory.
3. `opencode serve` on a loopback-only port with its password supplied through
   the process environment.
4. The bridge, after all three endpoints become ready.

If any component exits, the supervisor terminates the complete process group.
`Ctrl-C` first sends SIGTERM, then SIGKILL after a short grace period.

Companion apps are `nix run .#doctor`, `nix run .#sync-cert`,
`nix run .#codex-tui`, and `nix run .#opencode-tui -- PATH`.

## Deployment and verification

1. Create the dedicated bot account through an authenticated Ergo operator
   connection; register both channels to the owner, apply private modes, grant
   the bot access, and disable their history.
2. Store the generated bot and OpenCode passwords in protected server/local
   credential files. Copy only the public Ergo certificate to the local private
   config directory.
3. Run `nix flake check` for lint, unit, adapter, routing, and state tests.
4. Run `nix run .#doctor`, then start `nix run .#stack`.
5. Prove both adapters with harmless prompts, verify session persistence and
   TUI coexistence, exercise queue/cancel and one-shot request handling, and
   verify untrusted account tags are ignored.
6. Notify the owner nick only after the tunnel, both backends, IRC SASL joins,
   and restored bindings are ready.
