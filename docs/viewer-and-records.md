# Poker Arena — Viewer, Records, History & CLI Reference

Detailed operational reference extracted from the root README. For quick start see [README.md](../README.md); for statistical guarantees see [methodology.md](methodology.md); for v2 solver/AIVAT/forensics/scorecard module contracts see [v2-architecture.md](v2-architecture.md).

---

## Match folder layout

Every duplicate pairing writes to `runs/<plan>/<a>__vs__<b>/{a-first,b-first}/`:

| File | Contents |
|---|---|
| `config.json` | Game settings, players by seat, deal seeds, `deal_fingerprints`, `record_version`, `code_version`, `implementation`, `cli_versions`, `cli_artifacts`, `protocol` |
| `events.jsonl` | Every game event (cards, actions, table talk, board, showdown, uncalled returns, results) with visibility plus private `thought` rows (reasoning, notes, corrections, decision/call links) |
| `calls.jsonl` | Every model call: prompt, reply, thinking, usage, latency, identity attestation, errors, `decision_id`, `call_id` |
| `raw.jsonl` | Raw backend stdout/stderr/transcript records linked by `decision_id` / `call_id` |
| `hands.jsonl` | One summary line per finished hand: cards, board, actions, net chips per seat, seed, fingerprint |
| `state.json` | Atomic per-hand commit checkpoint (`hands_committed`, agent turns/session/workdir/`rebuilding`, pending messages, resumes) |
| `interrupted.jsonl` | Uncommitted rows moved aside by `_reconcile` after a crash/restart |
| `result.json` | Final totals and bb/100 once all hands commit |

Tournament roots also hold `plan.json` (including pre-play `registration`), `decisions.json` (recorded per-round official verdicts), and `tournament.json` (live lifecycle state).

---

## Agent turn protocol

Each seat is one conversation for the duration of a match (`1` deal on the `official` track; `50` deals by default on the `adaptation` track). Each turn receives unseen events plus current cards, board, pot, stacks, and legal actions, and responds with JSON:

```json
{
  "reasoning": "private",
  "action": "fold|check|call|raise|all_in",
  "amount": 12,
  "say": "table talk the opponent hears",
  "notes": "private notebook about the opponent, or null"
}
```

- `reasoning` and `notes` are private; `say` is shown to the opponent. Latest `notes` are echoed at each hand start so they survive transcript trimming.
- Opponents are anonymized as **Alex** and **Blair** so neither knows which model the other is.
- Unparseable or illegal moves are returned with the error (up to `max_fixes`, default 2) before falling back to check/fold. Sizing errors (below min-raise or above stack) are auto-adjusted and logged.
- Replies that invoke/attempt a tool or cannot be audited (`audit: "unverified"`) are rejected unseen, dropped via `forget_last()`, and re-asked in a rebuilt conversation.

---

## All-time history (`runs/arena.db`)

`python -m poker_arena history runs` indexes changed match folders into SQLite (`runs/arena.db`) without deleting past runs:

- Players are keyed by exact **identity**: model, reasoning level (`effort`), protocol fingerprint, implementation hash, and CLI artifact (`version` + binary `sha256`). Any change mid-match splits deals at the exact hand boundary.
- Equivalent code revisions can be pooled explicitly via `runs/equivalences.json` (`{"implementations": [["<sha1>", "<sha2>"]], "reason": "..."}`).
- Every hand is replayed from its seed before indexing; tampered/unverifiable hands are rejected and excluded from ratings and style stats.
- Repeated deals for the same pair and seeds are averaged rather than double-counted.
- The all-time leaderboard is **descriptive only** (pooled across runs with different stopping rules; never produces official decided ranks).

---

## Web viewer (`poker_arena view` / `export`)

`python -m poker_arena view runs` serves all runs at `http://127.0.0.1:8800` (auto-reloading live matches without interrupting active replays). Heavy computations (`ratings`, `scorecard`) run off the request path and cache under `runs/.viewer-cache/`.

- **Standings:** Model-first naming (e.g. *Claude Opus 5.5 · High*), decided vs descriptive badges, anytime-valid official intervals (`ci_official`) vs fixed-time descriptive intervals, reference-bot comparison, AIVAT variance-reduced estimates, Nash averaging / α-Rank meta-rankings, forensics summary, head-to-head matrix, style tendencies, and full call/identity/tool audit.
- **Scorecards:** Six-axis character sheet (`INTELLIGENCE`, `WISDOM`, `ENDURANCE`, `CREATIVITY`, `KNOWLEDGE`, `ALIGNMENT`) with 20-pip bars, bootstrap uncertainty bands, psychometric split-half reliability, and click-through links to exact hand decisions.
- **Watch (Replayer):** Synchronized duplicate side-by-side table (`viewer/table.js`), street-by-street lockstep mode, solver EV-loss & pot-odds overlay, private-mind vs public-action diff, deal map, and keyboard controls (`Space` play/pause, `,`/`.` step, `←`/`→` deals, `d` where they differ, `r` reasoning).
- **Fundamentals:** Separately scored `quiz.json` spots with exact prompts, worked EV answers, and provenance checks.
- **Planner:** Interactive pre-flight power/budget calculator generating ready-to-run `poker_arena plan` commands.
- **Static export:** `python -m poker_arena export runs --out site` writes a self-contained static site (excludes raw `calls.jsonl` prompts and strips sensitive player settings).

---

## CLI backend isolation & caveats

Every CLI runs inside an isolated empty working directory per seat. When a subscription hits a rate limit, `RateGate` pauses sibling seats on that CLI (`rate_limit_wait` = 15m default, `max_wait` = 6h).

- **`claude`** (`claude -p --resume`): system prompt replaced, `--tools ""` and `--strict-mcp-config`, `--setting-sources project` skips user hooks/settings (`isolate = false` to keep them), thinking returned as summaries.
- **`codex`** (`codex exec` / `exec resume`): coding instructions replaced by the game prompt; tools cannot be fully disabled so read-only sandbox + empty folder + post-turn tool audit enforce isolation; Codex does not attest served model IDs (`identity_match: "unattested"`).
- **`agy`** (`agy -p --conversation`): tool-less agent with MCP/commands/files/URLs denied; always pin an explicit `model` in `players.toml` so `agy` does not fall back to personal defaults.

### Why official CLIs?

Full agent harnesses (e.g. Hermes) add their own tools, prompts, and memory loops that confound what the poker eval measures. Official CLIs use your existing subscriptions within provider terms.

---

## Extending Poker Arena

- **Add a model:** Add a `[players.<id>]` section to `players.toml` specifying `backend` (`claude`, `codex`, or `agy`), exact `model`, and `effort`.
- **Add a game:** Create a module in `poker_arena/games/` subclassing `Game` and `Episode`, then register it in `poker_arena/games/__init__.py`. Runner, duplicate pairing, logging, ratings, and viewer work unchanged (aside from game-specific rendering).
