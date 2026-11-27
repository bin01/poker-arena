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

---

# Round 3 → 4

Round 3 report: `/private/tmp/claude-501/-Users-snehal-Code-poker-arena/7391e5d9-a258-4fd4-b230-fb0d350c6f8f/scratchpad/review/r3-report.md`
(ux=7 gameplay=8 evals=5 judges=6 overall=6). Issues R3-#n. Same three tracks, same file ownership.

## Contract additions
- **Inference validity (A):** a verdict needs at least `min_conversations` (default 8) independent conversations per
  pairing; below that the verdict is `early` whatever t is. The boundary is calibrated (and tested) against normal,
  skewed/discrete and heavy-tailed conversation-mean distributions at that minimum; the documented guarantee states its
  assumptions and the worst calibrated case. If a robust alternative (e.g. a studentized bootstrap or a
  sign-flip randomization test on conversation means combined with the minimum) is better calibrated in those cases,
  use it. Zero-variance conversations never produce a decision (R3-#1).
- **Power (A):** plan from conversation-mean variance with explicit dependence assumptions
  (`assumed_icc`, `assumed_ar1`), report `power_scenarios` (optimistic / moderate / pessimistic) and choose
  `max_rounds` by default so the moderate scenario reaches 80% power at the target; a budget cap only via an explicit
  `budget_deals` setting, and then `underpowered: true` (R3-#2, #3).
- **Unregistered runs (A, B):** runs without a recorded design get `design: {registered: false}`; no inferential
  verdicts; any sample-size numbers labelled retrospective (R3-#6).
- **Identity (A, C):** exact canonical model ids with an explicit alias map; served-model mismatch or unresolved identity
  removes the player from inferential results in that run too (R3-#7, #8). Protocol identity includes the match horizon
  and memory-reset schedule (C writes them, A pools on them) (R3-#9).
- **One source of design (C):** the runner executes only from `ratings.normalize_design(plan)`; conflicting
  `plan.replicates` vs `design.replicates` is an error (R3-#10).
- **Session Elo (A):** interval from resampling conversations jointly, or omitted with the assumption stated (R3-#11).
- **Audit (C):** agy turns whose transcript can't be read are `audit: 'unverified'` and rejected/retried like tool use;
  every call carries `audit` status (R3-#4). Each match config records `implementation`: content hashes of the engine,
  agents, backends, ratings modules and prompt templates; resume refuses a different implementation unless
  `--allow-code-change` is given (and then records it); a source snapshot (`git diff` or tarball of the package) is
  saved once per run (R3-#5).
- **Viewer (B):** provisional order labelled as such, numerals only for decided ranks (R3-#12); mobile autoplay plays
  Play 1 then Play 2 before the next deal; desktop offers street-aligned comparison or independent timelines (R3-#13);
  navigation grouped pairing → round → replicate with correct counts (R3-#14); during playback the scoreboard and deal
  map collapse so the table is in the first viewport on mobile, and matchup labels are complete (R3-#15); position split
  and visible "Plays preflop"/"Raises preflop" labels (R3-#16); unregistered runs say so.

---

# Round 4 → 5

Round 4 report: `/private/tmp/claude-501/-Users-snehal-Code-poker-arena/7391e5d9-a258-4fd4-b230-fb0d350c6f8f/scratchpad/review/r4-report.md`
(ux=7 gameplay=8 evals=6 judges=5 overall=6). Issues R4-#n. Same tracks and file ownership.

## Contract additions
- **One adjudicator (A defines, C calls, B shows):** `ratings.adjudicate(pairing_data, design, n_pairings, look, eligibility) -> dict`
  wraps `decide` and applies every eligibility rule (identity attested/consistent, audit complete, protocol consistent,
  registration valid, assumptions supported). Returns decide's keys plus `eligible: bool`, `ineligible_reasons: [..]`,
  `assumptions: {status: 'supported'|'unsupported'|'unknown', checks: {...}}` and a terminal verdict
  `ineligible` when not eligible. The runner stops and records only through `adjudicate`; ratings/CLI/viewer report the
  **recorded** decision as official (`official: true`, with the judge implementation hash) and any recomputation by
  today's code as `retrospective` (R4-#3, #6).
- **Registration (A, C):** `registered` is never recomputed; it is true only when a `registration` artifact written
  before the first hand exists in plan.json: `{design, judge_fingerprint, calibration_id, created}`. An explicit
  `registered: false` stays false. An `--allow-code-change` that changes the engine/backends/judge marks the registration
  `void` with reason (R4-#1, #6).
- **Assumptions (A):** each verdict carries its assumption status (e.g. conversation_share within the calibrated range);
  a decided verdict whose assumptions are `unsupported` is shown as "decided (outside calibrated range)" and never
  counts towards a decided leader unless the design preregistered how to treat it (default: it doesn't). Wording never
  states the error bound without its assumptions (R4-#2).
- **Identity (C records, A uses):** calls carry `requested_model`, `observed_model` (claude: init event `model`;
  agy: transcript model; codex/command: null) and `identity: 'attested'|'unattested'`. A player whose identity is
  unattested is eligible only if the design says `allow_unattested: [backend...]` (default allows codex, since it has no
  attestation, but it's shown as "unattested" everywhere) (R4-#4).
- **Audit eligibility (C, A):** gemini missing tool stats → `unverified`; `opaque` (command backend) never eligible for
  decided results unless preregistered (R4-#5).
- **History identity (A):** includes the implementation fingerprint of behaviour-relevant files and CLI versions;
  an explicit equivalence list may merge them (R4-#7).
- **Pair validation (A):** loaders verify b-first is a-first with seats swapped, same settings, same fingerprints per
  hand; mismatches are rejected and reported (R4-#8).
- **Fit diagnostics (A computes, B shows):** per pairing `fitted_edge`, `residual`, and an `additivity` warning when
  residuals exceed their uncertainty; direct pairwise evidence stays primary (R4-#9).
- **Engine context (C):** action events carry `to_act` (next actor, null at hand end), `to_call`, `pot_before`,
  `pot_odds`; hand_start carries initial `to_act`; each player's `made_hand` description per street is available
  (computed from cards; omniscient fields labelled) (R4-#13, #14).
- **Viewer (B):** interval basis labelled at the number ("within conversations" vs "across conversations"), compact
  outcome row instead of tall hero, one expandable planning panel, audit summary per run (tool rejections, unverified,
  fallbacks, identity coverage) and per-decision audit/identity details, mobile label overflow fixed and mobile stats
  metric selector, made-hand / to-call / pot-odds / duplicate comparison in replay, fit residuals (R4-#10–#15, #2, #9).

---

# Round 5 → 6

Round 5 report: `/private/tmp/claude-501/-Users-snehal-Code-poker-arena/7391e5d9-a258-4fd4-b230-fb0d350c6f8f/scratchpad/review/r5-report.md`
(ux=8 gameplay=8 evals=7 judges=6 overall=7). Issues R5-#n. Same tracks and file ownership.

## Contract additions
- **Record verifier (C writes, A uses):** `games.holdem.verify_hand(game, seed, button, recorded_hand, events) -> {ok, errors}`
  regenerates the deal from the seed, replays the recorded actions legally, and compares cards, board, payouts and
  summary. `runner`/`ratings.pairing_data` run it for every counted hand; the official decision record stores
  `evidence_hash` (hash of the verified hands at that look). A failed verification rejects the hand pair (R5-#4).
- **Decision coverage (C):** eligibility requires every model action event to have exactly one decision (thought) event
  and its call records; missing protocol fingerprints fail; empty evidence never passes (R5-#2).
- **CLI version pinning (C, A):** registration pins CLI versions; resume across a CLI version change is refused unless
  `--allow-code-change`, which voids registration; history identity splits at version changes (R5-#3).
- **Identity (C, A, B):** calls carry `identity_reported: bool` and `identity_match: 'match'|'mismatch'|'unattested'`
  via `ratings.same_model`; every consumer uses `identity_match` (R5-#5).
- **Current validity of historical wins (A):** an official verdict counts toward a current leader/rank only if today's
  evidence check (verifier, coverage, identity) still passes; lost evidence vs rule change are distinguished (R5-#1).
- **Wording (A, B):** "diagnostics passed; distribution assumptions unverified" instead of "within calibrated range" (R5-#6).
- **Power from the full adjudicator (A):** plan/power scenarios simulate the complete `adjudicate` (both arms, diagnostics);
  report pairwise counting power and probability of a decided tournament leader under stated edge scenarios (R5-#7).
- **Equivalence calibration (A):** simulations at ±15 and just outside, skew families, simultaneous pairings; false-"even"
  rates reported in docs and a test (R5-#8).
- **Competence vs peers (C bot, A/B presentation):** a fixed, versioned `reference` bot policy (solid
  tight-aggressive, position-aware, equity/pot-odds based with a fixed bluff frequency) recommended in every plan; and a
  small `quiz` suite of poker decision spots with objectively correct answers (pot-odds calls/folds, all-in EV spots,
  computed by exact/MC equity) that models answer through the same backends, scored separately as "poker
  fundamentals". Present peer-relative winnings and fundamentals separately (R5-#9).
- **Viewer (B):** cache signature includes .json run files (R5-#10); "untestable" fit shown as insufficient evidence
  (R5-#11); when additivity fails, the hero leads with direct matchup results (R5-#12); mobile fixes (R5-#13); playback
  speed control, pause on "why?", jump to divergent decisions and big swings (R5-#14).

---

# Round 6 → 7

Round 6 report: `/private/tmp/claude-501/-Users-snehal-Code-poker-arena/7391e5d9-a258-4fd4-b230-fb0d350c6f8f/scratchpad/review/r6-report.md`
(ux=7 gameplay=8 evals=5 judges=6 overall=6). Issues R6-#n. Same tracks and file ownership.

## Contract additions
- **Assumption-free official verdicts (A):** the official verdict comes from an anytime-valid sequential test for
  bounded outcomes (e.g. a betting / hedged-capital confidence sequence in the style of Waudby-Smith & Ramdas) applied to
  conversation means, which are bounded by the stacks (state the bound exactly). Valid under independence of
  conversations only, with no distribution-shape assumption, at every look. Bonferroni over pairings. The current
  t/sign-flip gate becomes a secondary, clearly labelled "model-based" verdict (or is dropped). Reproduce the reviewer's
  rare-loss counterexample (seed 4901) as a regression: false-verdict rate ≤ alpha. Recompute power, deals needed and
  default budgets for the new test by simulation, and report the cost vs the old gate (R6-#1).
- **Identity (C):** every answering model in a Claude stream (each assistant message's model) is checked; mixed or
  conflicting models → `identity_match: 'mismatch'`; init model is supplementary (R6-#2).
- **Linked audit (C, A):** every decision gets a `decision_id`; every call attempt, raw output row and the executed
  action carry it; accepted audit status is derived from those attempts and verified against the raw backend evidence;
  the judged evidence hash covers hands, events, calls and raw records (R6-#3).
- **Quiz provenance (C, A, B):** quiz answers keep raw output, identity_match, audit, protocol, implementation and CLI
  versions; scores attach only with provenance and eligibility status; fallbacks never count as correct and valid-answer
  accuracy is reported separately (R6-#4, #5). Harder, separately scored categories (closer EV gaps, multi-street lines,
  range inference from described action, sizing choices among 3+ options) with generated variants; no population-style
  intervals unless the sampling population is defined (R6-#9).
- **CLI pinning during a run (C):** re-check the resolved executable path, version and file digest before each launch;
  a change mid-run is recorded and voids registration (R6-#6).
- **Shared committed reader (A, B):** one reader distinguishes verified committed hands from provisional live events;
  stats and replay say which population they cover; rejected hands never enter style totals (R6-#10).
- **Reference characterisation (A, B):** state the reference bot's measured limits (its results against each baseline,
  and that it is a heuristic, not a solver); show model-vs-reference prominently (R6-#8).
- **Viewer (B):** decided order primary in standings, unresolved grouped (R6-#7); serve cached ratings immediately and
  refresh in the background, never block navigation/replay on ratings (R6-#11); all-runs view leads with the current
  model cohort, groups named by participants, cohort selector (R6-#12); mobile replay: compact sticky controls, current
  action and price beside the table, synchronized two-play comparison, larger explanatory text (R6-#13).
