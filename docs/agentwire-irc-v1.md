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
agentwire:v1;account=trev;agent=agentwire;backend=codex | Human-readable title
```

`account`, `agent`, and `backend` are required, and they answer three different questions:

- `backend=` — which engine runs the session: `codex`, `opencode`, `claude`, `pi`, or `omp`.
- `account=` — the IRC account whose commands the bridge obeys (the owner).
- `agent=` — the IRC account whose messages a client trusts as authoritative backend state
  (the bot).

`account` and `agent` hold IRC account names, never engine names: `agent=claude` would mean "an
IRC account literally named claude", not "the Claude backend". Only `backend` selects the engine,
and it MUST match the backend the deployment has assigned to that channel. A single-account
deployment therefore looks like:

```text
agentwire:v1;account=agentwire;agent=agentwire;backend=claude | Project title
```

Parameter values use UTF-8 percent encoding. Unknown parameters MUST be preserved or ignored.
` | ` and everything after it is a human title. Topic removal or an invalid topic immediately
suspends the harness and pauses queued prompts. It does not cancel a running backend turn.
A suspended channel can publish no events, so Agentwire announces the suspension as a plain
`NOTICE` reading `agentwire suspended: <reason>`: once for a marker that fails validation, and
once for a marker removed from a channel it had activated. A channel that was never activated
stays quiet per topic reply, because an ordinary topic on such a channel is not an event.

Once every configured channel has answered its topic query, Agentwire announces the ones that
did not activate — once per process, and only for channels its own configuration names. A
configured channel is meant to run an agent, so a topic that was never set, or one the server
discarded when an unregistered channel emptied, would otherwise be indistinguishable from a
working channel: the bridge drops every action it receives, and the only visible symptom is a
client that synchronizes forever. Deployments SHOULD register their agent channels so the topic
survives the channel emptying. A marker that was
meant to activate a channel also gets `; set: <topic>`, the corrected topic built from the
deployment's own owner account, authenticated account, and channel backend, preserving any
title, so a topic predating the required `agent` field is repaired by pasting one line.
Agentwire never supplies a missing `agent` itself: clients authenticate events by that account
and ignore a bridge the topic does not name, so activating anyway would publish into a void
that is harder to diagnose than a suspension.

Clients MUST authenticate both identities by the IRCv3 `account` tag, never by nickname: `account`
is the identity whose commands the bridge obeys, and `agent` is the identity whose messages a
client trusts as authoritative backend state. The two MAY be the same account — a single-account
deployment that separates projects by channel is fully supported — at the cost that a compromised
bot credential can then also issue owner commands; separate accounts are RECOMMENDED where that
tradeoff matters. Both fields are required either way. Agentwire additionally requires the topic
account to equal its configured owner and the topic agent to equal its own SASL account, and it
never consumes its own published events as actions.

## Required IRC behavior

There is no compatibility mode. Agentwire refuses registration unless the server acknowledges:

- `sasl`, `message-tags`, `account-tag`, `server-time`, `batch`, and `echo-message`
- `labeled-response` and `standard-replies`
- `draft/multiline`, `draft/chathistory`, and `draft/event-playback`

Agentwire queries `TOPIC` for every configured channel after joining and treats registration as
complete only once each has answered `331` or `332`. A server sends `RPL_TOPIC` unsolicited only
when a topic is set, so a channel with no topic is otherwise indistinguishable from one whose
topic has not arrived: both are silence, and the harness would wait for a line that is never
coming. Topic activation itself publishes no bootstrap events; clients issue `sync.request` once
active and receive one correlated hello and snapshot pair.

Deployments target Ergo 2.19.0 or newer. They SHOULD use persistent SQLite history, retain the
channel for 30 days, and allow the client-only tag `+trevarj.github.io/agentwire` on `TAGMSG` in
the server history configuration. Agentwire pages and fragmented payloads legitimately exceed
Ergo's default five-command burst, so the bot connection MUST be exempt from fakelag or use an
equivalent high-burst profile. A shared server SHOULD define a dedicated oper class containing
only `nofakelag`; granting general operator capabilities to the bot is forbidden. An IRC network
that strips client tags or account tags is not an Agentwire transport. The bridge verifies that
`echo-message` is offered for clients but does not enable it on its own connection.

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
the envelope is sent directly. Larger values are UTF-8 encoded and tested with zlib level-1
compression. Compression is used only when it reduces the number of IRC commands. The selected
bytes are base64url encoded without padding and split into checked fragments:

```json
{"v":1,"k":"fragment","id":"...","of":"assistant.completed","t":"event","epoch":"...","sid":"...","part":0,"parts":2,"bytes":9000,"sha256":"...","encoding":"zlib","b64":"..."}
```

The first fragment follows the original message visibility rule. Remaining fragments are
`TAGMSG`. Compression can produce one fragment even though the original envelope did not fit
directly. A receiver MUST bound decompression by the declared reconstructed size, then verify
uniform metadata, byte count, SHA-256, UTF-8, and decoded envelope ID. Limits are 128 KiB
reconstructed data, 64 fragments, 16 concurrent messages, 2 MiB aggregate declared bytes, and
30 seconds from first fragment. Conflicting duplicates invalidate the message. Exact duplicate
fragments are harmless.

## Liveness, acknowledgements, and replay

`agent.hello` supplies an unpredictable epoch after topic activation. Every action other than
`sync.request` MUST carry that exact epoch. Agentwire MUST NOT consume actions from IRC history;
it also rejects `hist:true` actions and playback-tagged messages. Reconnecting clients begin with
`sync.request`, learn the new epoch, then issue new actions.

Mutating actions emit `action.accepted` before invoking a backend and exactly one of
`action.succeeded`, `action.failed`, or `action.uncertain` afterward. Each carries the action UUID
in `reply`, and mutations are deduplicated durably by UUID. Read actions (`sync.request`, workspace
and session listing, and history) are safe to repeat and return only their reply-correlated data;
`history.end` terminates a history response. Any action can still return `action.failed`. Clients
MUST NOT automatically retry a mutation merely because an acknowledgement is missing.

`history.request` targets the currently attached session using the envelope `sid`; older clients
that omit it target the current binding. A supplied `sid` that differs from the binding is rejected.
Backends with authoritative transcript pagination, including Codex and Claude, provide full
persisted turns.
Other backends fall back to Agentwire's session-indexed journal. History never reads arbitrary IRC
messages. It replays only transcript and request lifecycle events: user prompt, turn, assistant,
plan, tool, usage, request, and approval-review events.
Sync snapshots, discovery pages, action acknowledgements, queue events, and binding/status events are
live state and MUST NOT appear in history pages. A page is bounded by 200 events, 512 KiB, and 30 days
and is enclosed by `history.begin` and `history.end`. To avoid one IRC command per small event,
`history.chunk.data.events` carries arrays of complete event envelopes; chunks are capped below
the common payload limit and use normal checked compression and fragmentation. Agent hello
advertises `compressedFragments` and `historyChunks` capabilities. Replayed nested events carry
`hist:true`, the requested `sid`, and the request UUID in `reply`. `data.cursor` and
`data.next` are opaque backend cursors. IRC message edits affect only readable transcript text; harness state and
Agentwire journal records are immutable.

## Actions

The following kinds are defined. A client MUST enable only those listed in the `actions` array of
`agent.hello`.

- Discovery: `sync.request`, `workspace.list.request`, `session.list.request`, `history.request`.
- Binding: `session.create`, `session.close`, `session.attach`, `session.detach`.
- Optional lifecycle: `session.rename`, `session.fork`, `session.archive`, `session.unarchive`.
- Settings: `settings.update`.
- Turns: `turn.prompt`, `turn.steer`, `turn.cancel`.
- Queue: `queue.edit`, `queue.move`, `queue.delete`, `queue.clear`.
- Requests: `request.respond`, `request.skip`.

This Agentwire advertises discovery, binding, settings, turns, queues, and requests. It does not
advertise optional lifecycle operations until a backend can perform them safely. `session.close`
is advertised only in a managed, dedicated Pi or OMP channel. It requires that channel's bound
`sid`, stops only the bridge-owned local RPC process, preserves its session JSONL for later
resume, then clears the topic and parts the managed channel. Static channels and live TUI
processes cannot be closed this way.

With `[pi].dedicated_channels = true` or `[omp].dedicated_channels = true`, `session.create` on a
configured static channel leaves that channel's binding unchanged. Agentwire joins a
collision-safe `#pi-<session-id-prefix>` or `#omp-<session-id-prefix>` channel, confirms
invite-only and secret modes (`+is`), confirms its canonical activation topic, then invites the
action sender's current nickname and confirms server acceptance. Managed channels survive IRC
reconnects within the bridge process but are not restored after process restart; the backend's
JSONL session remains resumable. They never advertise create, attach, or detach, so they cannot
recursively provision or change their binding. Any create failure stops the new owned process
and removes its runtime channel. Both options default to false.

Prompt and steer text is `data.content` and is capped at 64 KiB. `session.create` uses
`data.cwd`; `session.close` uses `sid`; attach uses `sid` and optional `data.cwd`. Queue edit uses `iid` and `data.content`,
move adds a zero-based `data.position`, and deletion uses `iid`. Approval responses use `rid` and
boolean `data.allow`; question responses use `rid` and `data.answers`, an array aligned with the
questions. Skipping is always explicit.
When an action operating on the current binding supplies `sid`, the bridge MUST reject it if that
session is no longer attached; it must never redirect a delayed action to a newer binding.

`workspace.list.request` without data lists configured allowlisted roots. Supplying an absolute
allowlisted directory as `data.parent` lists its immediate non-hidden child directories.
`workspace.page.data.parent` echoes that directory or is null for the root page; every item has
`path`, `name`, and `hasChildren`. An item MAY also carry `sessionCount`, the number of sessions
directly in that directory as known by the backend; the key is absent when the backend cannot
answer cheaply. Clients SHOULD lazy-load children when a directory expands and
MAY use any returned path as `session.create.data.cwd` or `session.list.request.data.cwd`.
`session.list.request.data.scope` is `workspace` or `live`; older `cwd=null` live discovery remains
valid. `session.page.data.scope` echoes the resolved scope and `data.cwd` echoes the requested
directory or is null for live-session discovery;
each session item includes `busy`, runtime `flags`, and `tuiAttached`, which is true only when
Agentwire can identify that exact top-level thread in a live TUI. Subagent and guardian threads are
not attachable discovery results. `data.cursor` echoes the page cursor or is null for the first page.
If `data.next` is non-null, clients request the next page by returning it as
`session.list.request.data.cursor`; cursors are opaque to clients.
For Claude, live-scope discovery reports only sessions this Agentwire is running or observing: the
Claude Agent SDK owns one CLI subprocess per session and has no shared server to enumerate. An
interactive `claude` session elsewhere on the machine appears in workspace discovery; once attached,
Agentwire follows its transcript and mirrors it live without resuming a competing CLI process, and
the first owner prompt promotes the binding to a driven session.

Safe settings are `model`, `effort`, `collaboration`, `delivery`, and `approvalReviewer`.
`collaboration` is `default` or `plan` and requires an explicit model. `delivery` is `queue` or
`steer`; `approvalReviewer` is `manual` or `auto_review`. For Codex,
Auto-review keeps the configured interactive approval policy and sandbox and sets
`approvalsReviewer=auto_review`; it MUST NOT silently use an approval policy equivalent to
“never ask”. Clients SHOULD require one confirmation per session before first enabling it.
The bridge keeps settings per session and MUST reset to safe defaults when binding a session that
has not been configured during the current bridge run; in particular, `auto_review` MUST NOT carry
across a session switch.

`agent.hello.data.settings` lists only what the bound backend accepts, so clients MUST drive their
settings UI from that list rather than from the full safe-setting vocabulary. Codex advertises all
five; Pi and OMP advertise `model`, `effort`, and `delivery`; OpenCode and Claude advertise
`delivery` alone, and Claude takes its model from deployment configuration rather than from
`settings.update`.

Claude maps its `AskUserQuestion` tool onto question requests: the owner's answers return to the
CLI through the permission callback, and a skip denies that one tool call so the turn continues
unanswered. Claude has no attachable TUI, so a question the bridge redacts as sensitive can only
be skipped.

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
- `workspace.page`, `session.page`, `history.begin`, `history.chunk`, `history.end`
- `action.accepted`, `action.succeeded`, `action.failed`, `action.uncertain`

Harness activity:

- `queue.snapshot`, `queue.item.added`, `queue.item.updated`, `queue.item.moved`,
  `queue.item.removed`
- `turn.started`, `turn.completed`, `turn.failed`
- `assistant.delta`, `assistant.completed`, `plan.updated`
- `tool.started`, `tool.updated`, `tool.completed`, `usage.updated`, `subagent.updated`
- `request.opened`, `request.resolved`, `approval.review.started`,
  `approval.review.completed`

Codex plan notifications, and the Claude backend's translation of its todo list, use
`plan.updated.data` with `plan: true`, `running`, `status`
(`pending`, `inProgress`, or `completed`), `completedSteps`, `totalSteps`, and a display `summary`.
Clients MUST stop an active plan indicator when `running` becomes false or its turn completes, and
SHOULD replace the prior plan card for the same turn instead of appending every update.

One channel has one global binding. Changing it leaves the old session running and observed by
the backend, but its activity is not rendered in the channel's main timeline. A client presents
the session list as a paged sheet and can reattach when Agentwire reports an inactive-session
request out of band.

`session.status` is the one session-owned event that MAY carry the sid of a live or observed
session other than the bound one, so a client can render a session drawer without polling.
Such an event feeds a client-side session status registry only: it MUST NOT alter the bound
session's timeline, busy state, settings, or binding. Its data fields are `busy` and `flags`,
plus `cwd` and `tuiAttached` when the backend knows them; other members are absent rather than
null. `session.status` for the bound sid keeps its existing meaning and MAY also update the
registry entry for that sid. Agentwire coalesces these events per sid to at most two updates per
second, suppressing unchanged payloads and always delivering the newest state once the
window closes, so a client MUST treat the registry as an eventually consistent hint rather than
a turn-accurate signal. Clients that predate this rule
ignore an unknown-sid status event, which is why the extension is additive within v1.

`subagent.updated` is session-owned `TAGMSG` state reporting the autonomous subagents the bound
session is running. Its `data.agents` is the full current list and replaces the client's previous
one rather than merging into it, so an empty list means no agents are tracked. Each entry carries
`id`, `type`, `description` (200 bytes), `status` (`queued`, `running`, `completed`, or `failed`),
and `isBackground`, plus numeric `toolUses`, `durationMs`, and `tokens` when a finished agent
reported them. Agentwire coalesces the event to at most two emissions per second per channel,
suppressing an unchanged list and delivering the newest one once the window closes, and clients
MUST clear the list on `binding.changed` because it describes the bound session only. For the pi
backend this is limited to live TUI sessions, since a bridge-spawned RPC session cannot receive
extension events; other backends and bridge-spawned sessions simply never emit it.

After a successful `session.create` or `session.attach`, Agentwire emits `binding.changed`, then a
`session.snapshot`, then `channel.snapshot`. Clients MUST treat `binding.changed` as a timeline
boundary and clear activity from the previous binding. For Codex, `session.snapshot.data` contains
`status` (`ready`, `running`, or `waiting`) and `recentOutputs`: up to three chronological
`{iid, tid?, phase?, content, omitted}` objects recovered from the resumed thread. Each recovered
output is limited to 4096 UTF-8 bytes and is wholly replaced when high-confidence secret material
is detected. An in-progress Codex turn also includes `recentActivity`: up to six chronological
`{kind, iid, tid?, data}` tool items recovered from the active app-server thread, using the same
safe metadata allowlist as live tool events. Clients SHOULD render those outputs and tool items as
restored session context, or render `status` when both lists are empty. Settings-only
`session.snapshot` events omit both lists and do not replace the timeline.

Queues are per channel and session, durable, ordered, and limited to 10 items by default. A busy
prompt queues or steers according to the session delivery setting. Canceling a turn does not
clear its queue. Topic suspension pauses queue draining; an IRC outage does not stop already
accepted backend work, which is journaled for later synchronization.

Assistant streaming, when a backend exposes it, is limited to two `assistant.delta` events per
second per item. Final assistant text, prompts, session switches, request cards, failures, and
queue edits/deletions have readable `PRIVMSG` representations. Plans, tools, usage, commentary,
snapshots, and other state are normally `TAGMSG`. Tool cards contain structured safe metadata;
the allowlisted fields are `label`, `input`, `output`, `diff`, `status`, `exitCode`, and
`durationMs`. Text fields are capped at 4096 UTF-8 bytes (`label` at 200 bytes and `status` at 80),
and a field is omitted when the high-confidence secret detector matches. Clients SHOULD label tool
cards from `label` (falling back to `kind`) and render available command, output, or diff previews
collapsed by default. Binary artifacts are manifests only and are never uploaded by this protocol.
Token/context metrics are optional; cost is optional because many backends cannot calculate it
reliably.

## Secrets and trust boundary

IRC and Ergo history are not a secret store. Agentwire omits an entire tool field when it is
classified sensitive and omits an entire assistant message when a high-confidence secret detector
matches. It does not send a placeholder value that could disclose length or structure. A redacted
approval may still be allowed or denied, with a client warning. A sensitive question can be
skipped over IRC but answered only in an attached local TUI. There is no per-message wire override.

Agentwire's SQLite database and directory are mode 0600 and 0700 respectively. Deployments MUST
use TLS, SASL, and private channel membership; separate controller and bot accounts are
RECOMMENDED as described under channel activation.

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
- fakelag disabled for the private server, or a bot oper class containing only `nofakelag`;
- private, registered channels and account-only access.

Activate a test channel only after the bot and reference client interoperate. Set production
topics last, since the prefix is the client-facing feature flag.
