# Agentwire protocol rollout

The v1 protocol implementation and canonical specification live in this repository. Rollout is
deliberately staged so the topic prefix remains an explicit feature flag.

1. Validate codec, bounded compressed fragmentation, packed history replay, per-channel action
   isolation, mutation deduplication, queue recovery, and the JSONL reference client locally.
2. Run interoperability tests against Ergo 2.19.0 with persistent SQLite history, 30-day
   retention, Agentwire `TAGMSG` storage enabled, and the bot exempt from fakelag.
3. Implement the protocol in MOTD using the reference fixtures and renderer state transitions.
4. Exercise a non-production channel with distinct owner and bot accounts, including reconnect,
   playback rejection, queue suspension, approvals, sensitive questions, and fragmented cards.
5. Set the production topic prefix last.

No legacy `!command` fallback is retained. Removing the topic suspends protocol processing.
