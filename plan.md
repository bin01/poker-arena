# Plan: blind-review fixes, round 1 → 2

Goal: every area of a blind review (UX, game play, evals, judges, overall) scores above 8.
Round 1 report: `/private/tmp/claude-501/-Users-snehal-Code-poker-arena/7391e5d9-a258-4fd4-b230-fb0d350c6f8f/scratchpad/review/r1-report.md`
(scores ux=5 gameplay=7 evals=3 judges=3 overall=4). The issue numbers below (#1–#22) are that report's.

Product goal (from the owner): identify which **models** (model + reasoning level) win and how good they are at
poker, and let a person watch and understand the games. Later a human will sit at a 3-handed table against two
models, so the table/replay should be a reusable component. Never lead with harness/CLI names.

Work happens in three parallel tracks with **disjoint files**. Do not edit files owned by another track.

| Track | Owns | Issues |
|---|---|---|
| A. Judges & evals | `poker_arena/ratings.py`, `poker_arena/history.py`, `poker_arena/stats.py`, `poker_arena/cli.py` (ratings/table output only), `tests/test_ratings*.py`, `tests/test_history*.py`, `tests/test_calibration.py`, README sections "Ratings"/"How sure", new `docs/methodology.md` | #1 #2 #3 #4 #5(stat side) #6(rule side) #7 #9 #10 #21 |
| B. Viewer & UX | `poker_arena/viewer/*`, `poker_arena/server.py`, `tests/test_viewer.py`, README section "The viewer" | #8 #14 #15 #16 #17 #18 #22, display of #19 corrections, model-first naming, first-visit intuitiveness |
| C. Engine, runner & audit | `poker_arena/runner.py`, `poker_arena/games/*`, `poker_arena/agents.py`, `poker_arena/backends.py`, `tests/test_holdem.py`, `tests/test_agents.py`, `tests/test_backends.py`, `tests/test_rounds.py`, new `tests/test_runner*.py` | #11 #12 #13 #19 #20, #6(enforcement in the round loop), #5(replicates option), deal fingerprints recorded in config |

## Shared contract (all tracks code against this)

### Ranking objective (Track A computes, B displays)
- The **primary** measure is duplicate chip return: **bb/100 over duplicate deals**. A deal's value for player a vs b is
  `x = net_a(both plays) / bb / 2 * 100`.
- Player **strength** = least-squares paired-comparison fit on deals: `x ≈ s_a − s_b`, with `Σ s = 0` within each
  connected component. Units: bb/100 against the average opponent in its component.
- Uncertainty: **block bootstrap** that keeps memory dependence: within each matchup, blocks of `BLOCK = 10`
  consecutive deals are the resampling unit; repeated deals (same fingerprint, same pair) are averaged first into
  one observation. 95% intervals. No interval (null) when a player has fewer than 2 blocks of data.
- **Session Elo** (share of 10-deal sessions won) stays as a secondary, explicitly different statistic, with a
  Bayesian (Laplace-approximation) posterior interval that cannot collapse to a point. Never used for ranking.
- **Components**: players connected by at least one played pairing. Ranks and leader are per component; the
  all-runs view shows each component separately.
- **Rank**: a player's `rank` is shown only when it is separated: the list is ordered by strength, and `rank_tied_with`
  lists players whose difference with it is not significant (pairwise 95% CI with Holm adjustment includes 0).
- **Leader**: a player whose strength beats *every* other player in its component with Holm-adjusted significance.
  Otherwise `leader = null` and the UI says "No established leader yet".

### Sequential design (A defines, C enforces in the runner)
- A plan carries `design = {"alpha": 0.05, "round_deals": 50, "max_rounds": 6, "block": 10}`.
  Family-wise alpha is split over pairings (Bonferroni: `alpha / n_pairings`).
- `ratings.boundary(alpha_pairing, round_deals, max_rounds, block) -> float`: critical |t| for the block-mean
  t statistic checked after each round, calibrated by simulation (cached, deterministic seed). Track A ships a
  calibration test showing the false-winner rate under the null ≤ alpha (+ Monte Carlo error) for normal and
  heavy-tailed (t3) noise.
- `ratings.decide(xs, design, n_pairings, looks_used) -> dict` with keys:
  `deals, blocks, edge, ci95, t, boundary, verdict ('a'|'b'|'even'|'open'|'early'|'exhausted'), looks_used,
  looks_planned, power` where `power = {"deals_for_80pct": {"20": N, "50": N, "100": N}}` (deals needed per pairing
  to detect an edge of 20/50/100 bb/100 with 80% power, from the observed spread), and `even` only when a 90% CI
  lies within ±`EQUIV_BB100 = 15` (TOST).
- Track C's round loop calls `ratings.decide` after each round and stops a pairing on verdict a/b/even or when
  `looks_used == max_rounds` (verdict `exhausted`). Never more looks than planned.

### `ratings.rate(...)` output keys (B renders exactly these)
```
{ objective: "bb/100 over duplicate deals",
  design: {...}, deals, hands,
  components: [[player,...], ...],
  leader: {player, component, margin_bb100, ci95} | null,
  players: [{player, identity, kind ('model'|'bot'), component, strength, strength_ci95 | null, rank | null,
             rank_tied_with: [player], net_bb, hands, deals, session_elo, session_elo_ci95, sessions,
             session_w, session_t, session_l, quarantined (bool), quarantine_reason | null}],
  pairings: [{a, b, matchups:[...], **decide(...)}],
  roster: [{player, status ('playing'|'awaiting'|'done')}],   # from plan.json, so awaiting players show
  notes: [str] }     # honest caveats, e.g. "deals repeated across runs were averaged", "single replicate per pairing"
```
`stats.collect` output keeps its shape, plus `sd_ties` next to `won_sd` (split pots are not wins).

### Identity (A for history, C for config, B for display)
- A model player's identity is `backend:model:effort`. A player without an explicit `model` (or with a bare alias
  like `sonnet`/`opus`) is **quarantined**: shown, but not rated in all-time standings.
- Display name = `label` if set, else `pretty_model(model) · Effort` (already in `server.display_name`).
- Track C writes into each match's `config.json`: `code_version` (git sha + dirty flag), `cli_versions`
  (`claude --version` etc., once per run), `deal_fingerprints` (sha1 of game settings + seed per deal),
  `system_prompts` per seat, and the full effective spec.

### Corrections (C records, B shows)
- `thought` events gain `requested: {action, amount}`, `executed: {action, amount}`, `correction: null | {kind, reason}`
  where kind ∈ `sizing` (auto-adjusted, legal intent) | `retry` (strategically invalid, model was asked again) |
  `fallback` (gave up after retries).

## Track checklists

### A. Judges & evals
- [ ] Strength fit, block bootstrap, components, Holm-separated ranks, leader rule
- [ ] Session Elo with Laplace posterior intervals (no point intervals)
- [ ] `boundary`, `decide`, TOST `even`, power table; remove the old `settle`/Z_STOP and "too close" language
- [ ] Calibration test (null false-winner rate ≤ alpha, normal and t3) — fast enough for the suite (cache/seed)
- [ ] Deal fingerprints: dedupe repeated deals per pair in pooling; history stores seed/fingerprint
- [ ] Quarantine unresolved identities in all-time; components in all-time
- [ ] stats: split pots → `sd_ties`, `won_sd` counts outright wins only
- [ ] CLI `table()`/`verdicts()` show the new objective; README "Ratings" + docs/methodology.md honest about limits

### B. Viewer & UX
- [ ] Model-first everywhere (display names, monograms by family), roster shows awaiting players
- [ ] First-visit clarity: a one-line "what is this" + a short "how to read" legend; obvious primary path
- [ ] Standings: headline = strength bb/100 with CI; no podium/rank unless separated; "No established leader yet";
      components shown separately on all-runs; session Elo secondary
- [ ] Matchup header: net bb headline, deals won secondary; verdict text from `decide`
- [ ] Replay: table-first, compact action log, per-decision reasoning expand, stacks/commitments/actor animated,
      winner shown only at the end; live polling never interrupts playback
- [ ] Corrections shown as requested → executed with reason
- [ ] Live: keep slow polling for unfinished runs; "last action" vs "checked"
- [ ] Mobile 390px: nothing clipped; verify with screenshots
- [ ] Table component structured so a future human seat (3-handed) can reuse it

### C. Engine, runner & audit
- [ ] Resume from persisted manifest; refuse mismatched args before any model call
- [ ] Interrupted hand: rebuild a fresh conversation from committed history instead of reusing a polluted session
- [ ] Audit: system prompts, CLI versions, code version, effective settings, every attempt (incl. failures and
      rate-limit waits) in calls.jsonl/config.json
- [ ] Corrections policy: strategically invalid → retry with explanation; sizing → adjust with notice; record
      requested/executed/reason
- [ ] Finite-number validation (Infinity/NaN) → retry path
- [ ] Round loop enforces the design via `ratings.decide`; `replicates` plan option (fresh-memory repeats)
- [ ] Deal fingerprints and code version in config

## Verification
- `python -m pytest -q` green after each track.
- Screenshots at 1440 and 390 px of standings (live run, all runs, finished small run) and a match replay.
- Fresh blind review (new Codex thread, same rubric). Repeat until every score > 8.

---

# Round 2 → 3

Round 2 report: `/private/tmp/claude-501/-Users-snehal-Code-poker-arena/7391e5d9-a258-4fd4-b230-fb0d350c6f8f/scratchpad/review/r2-report.md`
(ux=6 gameplay=7 evals=5 judges=4 overall=5). Issue numbers below (R2-#n) are that report's. Same three tracks, same file ownership.

## Contract additions

### One design, everywhere (A defines, C uses)
- `ratings.normalize_design(plan: dict) -> dict | None`: the plan's design with defaults filled, `replicates` folded in,
  and a preregistered `target_edge_bb100` (default 50) plus `deals_planned` (power-based: deals per pairing for 80% power
  at the target edge, from a stated prior spread `prior_sd_bb100 = 350`, capped by max_rounds × round_deals × replicates).
  Runner and ratings both call it; nothing else builds a design (R2-#9, #10).

### Inference vs description (A computes, B displays)
- A rank, a leader or a winner is **inferential** only when it comes from the sequential procedure at a completed
  planned look: a pairing verdict `a`/`b`/`even` from `decide`. Everything recomputed on every refresh is
  **descriptive** (R2-#2).
- `rate()` adds `groups: [{id, players, leader | null, front_runner | null, front_runner_margin, ci95, power, ...}]`.
  `leader` (inferential) only when that player has a decided winning verdict against every other player in the group;
  `front_runner` is the descriptive top by point estimate. Top-level `leader`/`leaders` keep meaning "inferential".
  Players get `rank_kind: 'decided' | 'descriptive'`.
- All-time (pooled across runs) output is descriptive only: pairings carry `verdict: 'descriptive'` and the summary
  says so (R2-#3).
- Players carry `in_fit` (bool) and `identity_resolved` (bool) separately from `quarantined` (R2-#13).

### Uncertainty (A)
- Resampling unit = (conversation, block): a conversation is one match folder (one replicate of one round of one
  pairing); blocks of `block` deals within it. Bootstrap jointly over aligned deal blocks shared across pairings in a
  run (same seed range) so common-card correlation is kept; hierarchical: resample conversations, then blocks within
  them (R2-#7, #8). Calibrate coverage with correlated tournament simulations in tests (fast, seeded).

### Identity (A history, C config)
- C writes `protocol` into each match config.json: `{fingerprint, settings}` where settings = the effective backend
  settings that change behaviour (backend, model, effort, isolate, replace_instructions/system prompt sha, tool policy,
  extra_args, max_fixes/retries, thinking display) and fingerprint = sha1 of them.
- A's history identity = `backend:model:effort` + `#` + first 8 chars of the protocol fingerprint when present; players
  with differing protocols never pool; a served-model mismatch (calls' served_model ≠ configured model) quarantines
  the player for that run (R2-#6).

### Fairness and recovery (C)
- Tool use: any model turn that ran a tool (codex `command_execution`/`web_search`/`mcp_tool_call`/file items, agy
  `tool` steps, claude `tool_use` blocks) is rejected: the reply is not used, the seat is asked again with a correction
  of kind `tool_use`, and calls.jsonl records `tool_use: [...]`. Also pass every available codex flag that narrows tools
  (R2-#1).
- Atomic commit per hand: write hand row + checkpoint via temp file + rename (state.json carries `hands_committed`);
  on resume, reconcile hands.jsonl, events.jsonl, calls.jsonl and raw.jsonl to `hands_committed` (R2-#4).
- Bots persist and restore RNG state (R2-#5).
- Every backend attempt keeps stdout, stderr, exit status, session id before/after; if a failed attempt may have
  advanced a stateful CLI conversation, rebuild the conversation from committed turns before retrying (R2-#11).
- Tournament lifecycle file `<out>/tournament.json`: `{status: 'running'|'finished'|'stopped', round, pairings: {"a|b":
  {status: 'playing'|'waiting'|'decided'|'exhausted', verdict, looks_used}}, updated}` written by the runner (R2-#16).

### Viewer (B)
- Per-group hero summaries; no contradictions (R2-#12). Show inferential vs descriptive explicitly: "Decided" badge
  only for inferential results; otherwise "ahead on points".
- Honest wording: "reduces the luck of the cards", no "at least", "difference unresolved" instead of "level" (R2-#14).
- Matchup header: cumulative score + verdict; the selected round's score labelled as such (R2-#15).
- Lifecycle from tournament.json (fallback: plan + decisions.json) (R2-#16).
- Mobile: results first (who's ahead, standings, watch), glossary collapsed, compact rows keep net and sample size,
  scroll cues on wide tables (R2-#17).
- Watch: start at step 0 with "Watch from start"; sticky controls; on mobile a Play 1 / Play 2 switcher (or
  synchronized); "Next deal" continuous playback (R2-#18).
- Table: "to act" (next actor) distinct from "last action" (R2-#19).
- Style table: tendencies with numerator/denominator, no quality thresholds (R2-#20, with A providing counts).
- A target-edge line: "This run was planned to detect a 50 bb/100 edge; that needs ~N deals per pairing; it has M."

### Stats (A)
- `stats.collect` gives each rate as `{n, of}` plus per-opponent breakdown (R2-#20).
