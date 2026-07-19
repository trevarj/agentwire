# IRC agent bridge

`irc-agent-bridge` connects two private IRC channels to live Codex and OpenCode
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
`~/.config/irc-bridge/`. Start from [`config.example.toml`](config.example.toml).
The secrets file is a mode-0600 env file containing only the variables named by
the live config.

## IRC workflow

- `!new <path>` creates and binds a session in an allowlisted workspace.
- `!sessions [path]` lists recent sessions; `!attach <number>` attaches one.
- A normal message starts a turn, or queues when a turn is already active.
- `!steer <text>` redirects the active turn; `!cancel` interrupts it.
- `!approve [A1]` and `!deny [A1]` resolve one-shot approval requests.
- `!answer Q1 <answer>` and `!reject Q1` resolve agent questions.
- `!paste` uploads the last full reply to one-hour Litterbox storage after a
  secret scan; `!paste-force` explicitly overrides a scanner block.
- `!status`, `!queue`, `!drop`, and `!help` cover normal operation.

Replies are capped in IRC at eight lines or 1,200 UTF-8 bytes. Long replies stay
available in the shared TUI and are uploaded only on the explicit paste command.

Attach local TUIs to the same running sessions with:

```console
nix run .#codex-tui
nix run .#opencode-tui -- "$PWD"
```

Run the local checks with `nix flake check`, and inspect the private deployment
without starting it with `nix run .#doctor`.
