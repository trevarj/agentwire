# Agentwire

Agentwire connects two private IRC channels to live Codex and OpenCode
sessions. It is intentionally owner-only: commands are accepted only when Ergo's
IRCv3 `account-tag` exactly matches the configured account.

The normal entry point is one foreground process tree:

```console
nix run .#stack
```

That command starts the SSH tunnel to Ergo, Codex app-server, OpenCode server,
and the bridge. `Ctrl-C` stops the whole tree. Nothing is installed as a system
or user service.

Live server details and all credentials live outside this repository in
`~/.config/agentwire/`. Start from [`config.example.toml`](config.example.toml).
The secrets file is a mode-0600 env file containing only the variables named by
the live config.

Existing installations remain usable during migration: the `irc-bridge`
executable is a compatibility alias, and Agentwire falls back to
`~/.config/irc-bridge/config.toml` when the new config path does not exist.

## IRC workflow

- `!running` (or `!r`) lists active sessions across every allowed workspace;
  `!use <number>` attaches one. `!sessions [workspace]` lists recent sessions.
- `!new <workspace>` creates and binds a session. Workspace names may be
  relative to an allowed root, while absolute paths continue to work.
- A normal message starts a turn while idle. While busy, messages accumulate in
  a held draft: `!next` queues it, `!steer` redirects the active turn with it,
  and `!discard` removes it.
- `!cancel` interrupts the active turn. `!queue` and `!drop <number>|all`
  manage prompts already queued for later turns.
- `!yes [A1]` and `!no [A1]` resolve one-shot approval requests.
  `!answer [Q1] <answer>` and `!skip [Q1]` resolve agent questions.
- `!paste` uploads the last full reply to one-hour Litterbox storage after a
  secret scan; `!paste-force` explicitly overrides a scanner block.
- `!status` (`!s`) shows a compact dashboard and `!last` (`!l`) repeats the
  latest output without invoking the agent.
- `!watch quiet|concise|verbose` controls activity detail per channel. Concise
  is the default after every bridge restart.
- `!help` shows commands relevant to the current state; `!help all` lists the
  complete interface.

The older `!attach`, `!approve`, `!deny`, and `!reject` spellings remain as
compatibility aliases.

Replies are capped in IRC at eight lines or 1,200 UTF-8 bytes. Long replies stay
available in the shared TUI and are uploaded only on the explicit paste command.
Bridge dashboards use multiline Unicode formatting and restrained semantic
emoji; traditional IRC color and bold control codes remain stripped.

Attach local TUIs to the same running sessions with:

```console
nix run .#codex-tui
nix run .#opencode-tui -- "$PWD"
```

Run the local checks with `nix flake check`, and inspect the private deployment
without starting it with `nix run .#doctor`.
