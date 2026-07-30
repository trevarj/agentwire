# Agentwire IRC Protocol v1

Status: implemented protocol for Agentwire and experimental clients. Normative terms such as
MUST and SHOULD are interpreted as in RFC 2119.

## Scope

This extension transports an agent harness through an authenticated IRC channel. IRC remains a
readable transcript and synchronization layer. The JSON payload in the Agentwire message tag is
canonical; message text is only a human-readable preview.

The canonical machine-readable schema is
[`protocol/agentwire-v1.schema.json`](../protocol/agentwire-v1.schema.json). Interoperability
examples are in [`protocol/fixtures/`](../protocol/fixtures/). The Python reference implementation
is `agentwire.reference_client` and its JSONL executable is `agentwire-protocol-client`.

## Channel activation

The first byte of the channel topic MUST begin this exact, case-sensitive prefix:

```text
agentwire:v1;account=agentwire;backend=codex | Human-readable title
```

`account` and `backend` are required. Parameter values use UTF-8 percent encoding. Unknown
parameters MUST be preserved or ignored. ` | ` and everything after it is a human title. Topic
removal or an invalid topic immediately suspends the harness and pauses queued prompts. It does
not cancel a running backend turn.

Clients MUST authenticate the controller by the IRCv3 `account` tag, never by nickname. The
topic account identifies the controller account; the Agentwire bot MUST use a distinct account.
Agentwire additionally requires the topic account to equal its configured owner.

## Required IRC behavior

There is no compatibility mode. Agentwire refuses registration unless the server acknowledges:

- `sasl`, `message-tags`, `account-tag`, `server-time`, `batch`, and `echo-message`
- `labeled-response` and `standard-replies`
- `draft/multiline`, `draft/chathistory`, and `draft/event-playback`

Deployments target Ergo 2.19.0 or newer. They SHOULD use persistent SQLite history, retain the
channel for 30 days, and allow the client-only tag `+trevarj.github.io/agentwire` on `TAGMSG` in
the server history configuration. An IRC network that strips client tags or account tags is not
an Agentwire transport.

## Wire representation

Every protocol message has a client-only tag named:

```text
+trevarj.github.io/agentwire
```

Its value is minified UTF-8 JSON and is canonical. JSON objects sent by this implementation use
lexicographically sorted keys, but receivers MUST NOT rely on key order. Unknown object members
are reserved for future versions and v1 receivers SHOULD reject them when validating against the
schema.

Visible events and prompt actions use `PRIVMSG`; their body is a non-authoritative plain-text
preview capped at 4096 UTF-8 bytes. State, commentary, tool, usage, and control messages use
`TAGMSG`. If a preview needs `draft/multiline`, the Agentwire tag appears only on the opening
`BATCH`; inner `PRIVMSG` lines contain only `batch` and `draft/multiline-concat` tags. No custom
batch type is defined.

The common envelope keys are:

| Key | Meaning |
| --- | --- |
| `v` | Integer protocol version, exactly `1` |
| `k` | Action or event kind |
| `t` | `action` or `event` |
| `id` | Globally unique UUID for this message or action |
| `at` | Unix time in milliseconds |
| `inst` | Sender process instance identifier |
| `epoch` | Live Agentwire connection epoch |
| `device` | Stable client installation identifier, required on actions |
| `sid`, `tid`, `iid`, `rid` | Session, turn, item, and request identifiers |
| `rev` | Non-negative entity revision |
| `reply` | Action UUID to which an event responds |
| `hist` | True only for an Agentwire journal replay |
| `data` | Kind-specific JSON object |

Strings are UTF-8. IDs are opaque except that envelope `id` is a UUID. Timestamps do not
establish liveness or ordering. Clients order revisions by `rev` where present and otherwise use
arrival order, reconciling from snapshots.

## Fragmentation

If the encoded tag name, separator, and escaped value fit IRCv3's 4094-byte tag-section limit,
the envelope is sent directly. Larger values are UTF-8 encoded, base64url encoded without
padding, and split into checked fragments:

```json
{"v":1,"k":"fragment","id":"...","of":"assistant.completed","t":"event","epoch":"...","sid":"...","part":0,"parts":3,"bytes":9000,"sha256":"...","b64":"..."}
```

The first fragment follows the original message visibility rule. Remaining fragments are
`TAGMSG`. A receiver MUST verify uniform metadata, the reconstructed byte count, SHA-256, UTF-8,
and the decoded envelope ID. Limits are 128 KiB reconstructed data, 64 fragments, 16 concurrent
messages, 2 MiB aggregate declared bytes, and 30 seconds from first fragment. Conflicting
duplicates invalidate the message. Exact duplicate fragments are harmless.

## Liveness, acknowledgements, and replay

`agent.hello` supplies an unpredictable epoch after topic activation. Every action other than
`sync.request` MUST carry that exact epoch. Agentwire MUST NOT consume actions from IRC history;
it also rejects `hist:true` actions and playback-tagged messages. Reconnecting clients begin with
`sync.request`, learn the new epoch, then issue new actions.

Agentwire emits `action.accepted` before invoking a backend and exactly one of
`action.succeeded`, `action.failed`, or `action.uncertain` afterward. Each carries the action UUID
in `reply`. Clients MUST NOT automatically retry merely because an acknowledgement is missing.
Actions are deduplicated durably by UUID; a duplicate produces the known status without invoking
the backend again.

`history.request` reads Agentwire's journal, not arbitrary IRC messages. A page is bounded by 200
events, 512 KiB, and 30 days and is enclosed by `history.begin` and `history.end`. Replayed events
carry `hist:true`. IRC message edits affect only readable transcript text; harness state and
Agentwire journal records are immutable.

## Actions

The following kinds are defined. A client MUST enable only those listed in the `actions` array of
`agent.hello`.

- Discovery: `sync.request`, `workspace.list.request`, `session.list.request`, `history.request`.
- Binding: `session.create`, `session.attach`, `session.detach`.
- Optional lifecycle: `session.rename`, `session.fork`, `session.archive`, `session.unarchive`.
- Settings: `settings.update`.
- Turns: `turn.prompt`, `turn.steer`, `turn.cancel`.
- Queue: `queue.edit`, `queue.move`, `queue.delete`, `queue.clear`.
- Requests: `request.respond`, `request.skip`.

This Agentwire advertises discovery, binding, settings, turns, queues, and requests. It does not
advertise optional lifecycle operations until a backend can perform them safely. Session deletion
is intentionally absent.

Prompt and steer text is `data.content` and is capped at 64 KiB. `session.create` uses
`data.cwd`; attach uses `sid` and optional `data.cwd`. Queue edit uses `iid` and `data.content`,
move adds a zero-based `data.position`, and deletion uses `iid`. Approval responses use `rid` and
boolean `data.allow`; question responses use `rid` and `data.answers`, an array aligned with the
questions. Skipping is always explicit.
When an action operating on the current binding supplies `sid`, the bridge MUST reject it if that
session is no longer attached; it must never redirect a delayed action to a newer binding.

`workspace.list.request` without data lists configured allowlisted roots. Supplying an absolute
allowlisted directory as `data.parent` lists its immediate non-hidden child directories.
`workspace.page.data.parent` echoes that directory or is null for the root page; every item has
`path`, `name`, and `hasChildren`. Clients SHOULD lazy-load children when a directory expands and
MAY use any returned path as `session.create.data.cwd` or `session.list.request.data.cwd`.
`session.page.data.cwd` echoes the requested directory or is null for running-session discovery;
`data.cursor` echoes the page cursor or is null for the first page.
If `data.next` is non-null, clients request the next page by returning it as
`session.list.request.data.cursor`; cursors are opaque to clients.

Safe settings are `model`, `effort`, `collaboration`, `delivery`, and `approvalReviewer`.
`collaboration` is `default` or `plan` and requires an explicit model. `delivery` is `queue` or
`steer`; `approvalReviewer` is `manual` or `auto_review`. For Codex,
Auto-review keeps the configured interactive approval policy and sandbox and sets
`approvalsReviewer=auto_review`; it MUST NOT silently use an approval policy equivalent to
“never ask”. Clients SHOULD require one confirmation per session before first enabling it.
The bridge keeps settings per session and MUST reset to safe defaults when binding a session that
has not been configured during the current bridge run; in particular, `auto_review` MUST NOT carry
across a session switch.

`agent.hello.data.settingOptions` MAY advertise backend-sourced picker metadata for those safe
settings. Its `model` member is an array of
`{value, label, efforts, defaultEffort?, default?}` objects.
Clients SHOULD use it for model selection and restrict effort selection to the chosen model's
`efforts`; they MUST tolerate the member or the entire object being absent. The Codex bridge
sources this catalog from app-server `model/list` with hidden models excluded. This object is
discovery metadata only and does not change the strings accepted by `settings.update`.

## Events and client rendering

Bootstrap and state:

- `agent.hello`, `channel.snapshot`, `binding.changed`, `session.snapshot`, `session.status`
- `workspace.page`, `session.page`, `history.begin`, `history.end`
- `action.accepted`, `action.succeeded`, `action.failed`, `action.uncertain`

Harness activity:

- `queue.snapshot`, `queue.item.added`, `queue.item.updated`, `queue.item.moved`,
  `queue.item.removed`
- `turn.started`, `turn.completed`, `turn.failed`
- `assistant.delta`, `assistant.completed`, `plan.updated`
- `tool.started`, `tool.updated`, `tool.completed`, `usage.updated`
- `request.opened`, `request.resolved`, `approval.review.started`,
  `approval.review.completed`

One channel has one global binding. Changing it leaves the old session running and observed by
the backend, but its activity is not rendered in the channel's main timeline. A client presents
the session list as a paged sheet and can reattach when Agentwire reports an inactive-session
request out of band.

Queues are per channel and session, durable, ordered, and limited to 10 items by default. A busy
prompt queues or steers according to the session delivery setting. Canceling a turn does not
clear its queue. Topic suspension pauses queue draining; an IRC outage does not stop already
accepted backend work, which is journaled for later synchronization.

Assistant streaming, when a backend exposes it, is limited to two `assistant.delta` events per
second per item. Final assistant text, prompts, session switches, request cards, failures, and
queue edits/deletions have readable `PRIVMSG` representations. Plans, tools, usage, commentary,
snapshots, and other state are normally `TAGMSG`. Tool cards contain structured safe metadata;
large individual fields are capped at 32 KiB and a card at 64 KiB. Binary artifacts are manifests
only and are never uploaded by this protocol. Token/context metrics are optional; cost is optional
because many backends cannot calculate it reliably.

## Secrets and trust boundary

IRC and Ergo history are not a secret store. Agentwire omits an entire tool field when it is
classified sensitive and omits an entire assistant message when a high-confidence secret detector
matches. It does not send a placeholder value that could disclose length or structure. A redacted
approval may still be allowed or denied, with a client warning. A sensitive question can be
skipped over IRC but answered only in an attached local TUI. There is no per-message wire override.

Agentwire's SQLite database and directory are mode 0600 and 0700 respectively. Deployments MUST
use TLS, SASL, separate controller and bot accounts, and private channel membership.

## Reference client JSONL interface

Run in the Nix development shell:

```console
nix develop -c sh -c 'PYTHONPATH=src python -m agentwire.reference_client'
```

Each input line is an operation: `topic`, `ingest`, `action`, or `state`. `ingest` accepts a raw
Agentwire tag value and outputs the decoded event plus render state. `action` outputs IRC command,
body, and tags for each required fragment. The interface is deterministic except for generated
IDs and timestamps and is intended for mobile-client fixtures and automated interoperability
tests.

## Ergo deployment profile

Use Ergo 2.19.0 or newer. The exact YAML paths can change between Ergo releases, so validate the
shipped example config, then set the equivalent of:

- persistent SQLite history with a 30-day channel expiry and a `CHATHISTORY` page size of at least
  200;
- `TAGMSG` history enabled and `+trevarj.github.io/agentwire` included in its storage whitelist;
- multiline limits of at least 4096 bytes and enough lines for readable previews;
- private, registered channels and account-only access.

Activate a test channel only after the bot and reference client interoperate. Set production
topics last, since the prefix is the client-facing feature flag.
