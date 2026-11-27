# Poker Arena

An eval in which language models play **duplicate No-Limit Texas Hold'em** against each other (and against
three baseline bots), with standings in bb/100 that say how sure they are, and a web viewer that replays every
hand with each player's reasoning and table talk.

It borrows its shape from [Clocktower](https://github.com/dwarkeshsp/clocktower): every seat is one ongoing
model conversation for a whole match, the dealer is plain seeded code, side-swapped pairs reduce the luck of the
deal, a preregistered sequential test decides each pairing, and every call is logged so
any result can be traced back to the hand and the reasoning behind it.

Models run through the labs' **official command-line tools signed in with your own subscription** (Claude
Code, Codex, Antigravity), so a tournament costs nothing beyond the plans you already pay for.

## How it works

- **Game.** Heads-up NLHE, blinds 1/2, both stacks reset to 200 (100 big blinds) every hand, the button moving
  every hand. Each match is `deals` hands (50 by default).
- **Duplicate.** Every matchup is two matches on the same deals with the seats swapped. On deal *k* each
  player gets, in turn, both seats' cards against the other player, so being dealt better cards
  matters much less (the two plays of a deal can still go differently). Every matchup in a tournament uses the
  same deals, too.
- **Agents.** Each seat is one conversation for the whole match, so a model remembers earlier hands and what
  its opponent did and said. Each turn it gets everything it saw since its last turn, then its cards, the board,
  the pot, the stacks and its legal options, and answers with JSON:

  ```json
  {"reasoning": "private", "action": "fold|check|call|raise|all_in", "amount": 12,
   "say": "table talk the opponent hears", "notes": "private notebook about the opponent, or null"}
  ```

  `reasoning` and `notes` are never shown to the opponent; `say` is. The latest notes are repeated at the start
  of each hand, so they survive even when a long transcript has to be trimmed. A reply that can't be parsed or
  played is sent back with the error (twice at most), and then the seat checks or folds and the failure is
  logged. Sizing mistakes are corrected and the seat is told (a raise below the minimum becomes the minimum, one
  above the stack becomes an all-in). Opponents are called Alex and Blair; neither knows which model the other is.
- **Bots.** `random` (random legal moves), `station` (always checks or calls) and `rule` (Monte Carlo equity
  against pot odds, value bets, never bluffs). They are free, play instantly, anchor the scale and catch a
  broken prompt or backend at once.
- **Two tracks.** A tournament plays the same players on the same deals twice over, in two separate runs.
  The **official** track starts a fresh conversation for every deal, so its conversations are independent and
  its verdicts come from a test that assumes nothing about the shape of the results; only it decides ranks and
  leaders. The **adaptation** track keeps each conversation for 50 deals, so a model remembers the session and
  can adapt to its opponent; its readings (and its stopping) come from a model-based test with stated
  assumptions, so they are exploratory ("leans ahead", never "wins"), labelled *adaptation · model-based · not
  official* everywhere, and never feed the official standings. The all-time history keeps the tracks apart.
- **Ratings.** Chips decide, as the models are told. Only one thing is a test: each pairing's preregistered
  sequential procedure. On the official track it is a betting confidence sequence on conversation means, which
  the stacks bound (±10,000 bb/100 at 100 big blinds): with no edge, the chance of any false verdict in a
  tournament stays under 5% at every look, for any distribution of results, given independent conversations.
  A **decided** rank or leader is built from those verdicts alone, and not from one reached while the edge
  drifted. Everything else is **descriptive**: a player's strength (its edge in bb/100 against the average
  opponent of its group, from a least-squares fit over duplicate deals) and its fixed-time 95% interval, the
  order by points, session Elo, and every adaptation-track number. [docs/methodology.md](docs/methodology.md) has the method, the assumptions, power, cost and the limits.

## Quick start

```sh
git clone https://github.com/bin01/poker-arena && cd poker-arena
python3 -m venv .venv && . .venv/bin/activate      # Python 3.11+
pip install -e '.[dev]'
python -m pytest -q                                  # engine, evaluator, bots, ratings, a full stand-in model match
# In a sandbox where the only writable place is tests/.tmp (kept in the repository) and TMPDIR is unset or
# read-only, run exactly this from the repository root:
#   python -m pytest -q -p no:cacheprovider
# pytest's capture is sys (pyproject.toml), which needs no temp file, and tests/conftest.py, loaded before any test,
# points Python's and pytest's temp folders (tmp_path included) at tests/.tmp when TMPDIR, /tmp and /var/tmp are
# not writable. -p no:cacheprovider skips .pytest_cache.

# Poker fundamentals, asked of each player -> runs/quiz/quiz.json: the 32 basics plus four harder, separately
# scored categories of 24 generated spots each (close, multistreet, inference, sizing); --category and --limit pick
python -m poker_arena quiz reference claude-haiku-5-5-high --out quiz

# Free and instant: the bots against each other, then the viewer
python -m poker_arena play rule station --deals 50 --out runs/demo
python -m poker_arena view runs/demo                 # http://127.0.0.1:8800

# A tournament in both tracks: plans/cup.json holds one plan per track, played into runs/cup/official and
# runs/cup/adaptation, each registered on its own; `plan` prints the cost of each track before anything plays
python -m poker_arena plan reference rule station --out runs/cup --write plans/cup.json   # --tracks official,adaptation
python -m poker_arena tournament plans/cup.json                       # both tracks at once; --track adaptation for one
python -m poker_arena ratings runs/cup                                # each track judged apart
```

### Models, on your subscriptions

Install and sign in to the CLIs you want to use (`claude`, `codex`, `agy`), then check them:

```sh
python -m poker_arena doctor          # each model player: does it answer, is the JSON valid, does it remember?
```

Players live in [`players.toml`](players.toml). A model player is **a model at a reasoning level**: each entry
names its CLI (`backend`), the exact `model` (always set it: otherwise the CLI uses whatever default your own
settings have) and its reasoning level, `effort`, which defaults to `high`; `--effort` on `tournament` or `play`
sets it for every model. The viewer names the player after the two, e.g. "Claude Opus 5.5 · high"; `label`
overrides that. A bot entry names the `bot`.

| backend | tool | plan | memory |
|---|---|---|---|
| `claude` | Claude Code, `claude -p` | Claude Pro / Max | the CLI keeps the session (`--resume`); tools off; `effort` is `--effort`; thinking summaries shown |
| `codex` | Codex CLI, `codex exec` | ChatGPT Plus / Pro | the CLI keeps the thread (`exec resume`); the game prompt replaces Codex's coding instructions; read-only sandbox |
| `agy` | Google Antigravity CLI, `agy -p` | Google account (Gemini) | the CLI keeps the conversation (`--conversation`); the game prompt is a tool-less agent; MCP, commands, files and URLs denied; thinking read from its transcript |
| `gemini` | Gemini CLI, `gemini -p` (older installs) | Google account | the transcript is re-sent each turn |
| `command` | any CLI reading a prompt on stdin, e.g. `["pi", "-p"]`, `["ollama", "run", "{model}"]` | yours | transcript re-sent |
| `openai_compat` | OpenRouter, Ollama, LM Studio, vLLM (an API, not a subscription) | API key | transcript re-sent |

Every CLI runs in an empty temporary folder so no project files or instructions leak in, and `claude` runs
with `--setting-sources project`, so your personal hooks and settings don't run inside matches (set
`isolate = false` on a player to keep them). When a plan's usage
limit is hit, the call waits (`rate_limit_wait`, 15 minutes by default) and tries again, for up to `max_wait`
(6 hours), so a long tournament pauses instead of failing.

```sh
# A smoke test: one model against two bots, 10 deals each (about 100 model calls)
python -m poker_arena tournament plans/smoke.json

# The pilot: 4 models + 3 bots, round robin, 50 deals per matchup (21 matchups, 2,100 hands)
python -m poker_arena tournament plans/pilot.json --dry-run
python -m poker_arena tournament plans/pilot.json --concurrency 3     # 3 matches at once per CLI

python -m poker_arena ratings runs/pilot
python -m poker_arena view runs/pilot                                 # follows running matches live
```

`tournament` resumes: rerun the same command after a crash or a stop, and each match continues after its last
finished hand. Write your own preregistered plan with `python -m poker_arena plan A B C reference --out runs/mine --write plans/mine.json`:
both tracks by default (`--tracks official` for one), the official one with one deal per conversation
(`--deals`, `--replicates`), the adaptation one with 50 (`--adaptation-deals`, `--adaptation-replicates`,
`--adaptation-budget-deals`); `tournament --track official` plays one track, and `--concurrency` applies to each.

### How sure the results are

Only the sequential test decides anything; the standings beside it describe. Even with duplicate deals one deal
swings a lot (in the first model runs the result per deal had a spread of about 340 bb/100), and a model's
conversations can differ from each other, so the test treats each conversation (a fresh pair of matches) as one
independent unit, and its mean edge as one observation.

**The official test assumes nothing about the shape of the results.** A conversation's mean edge can never be
beyond what the stacks allow: each of a duplicate deal's two plays wins or loses at most one stack, so with
100-big-blind stacks every result lies within ±10,000 bb/100 (`ratings.outcome_bound`, from the game settings).
The test is a **betting confidence sequence** for bounded outcomes (Waudby-Smith and Ramdas): for each player a
bettor stakes part of its capital on each new conversation, using only the earlier ones, never more than it can
lose without going broke; with no real edge its capital is a fair game, and it reaches the bar of 2 / alpha per
pairing with chance at most alpha / 2, by Ville's inequality. So with no real edge the chance of any false
verdict in the whole tournament stays under 5%, **for any distribution of results, at every look and whenever
play stops**, given that conversations are independent (fresh memories, no deal repeated) and that the null is
true conditionally: with no edge, each conversation's expected edge given the ones before it is zero (a common
mean is one case of that). *even* (within ±15 bb/100) has the same guarantee. If the edge drifts (a provider
changes the model mid-run, say), a verdict for a means a's expected edge was positive for at least part of the
run, not that the run's average favours a: +100 bb/100 for 700 conversations and then −100 for 700 averages
zero, yet a is rightly called at conversation 536 (with 3 pairings). So every verdict carries a **drift** diagnostic (the first
half of its conversations against the second, Welch t, flagged at p < 1%), and a decided verdict it flags is
**decided with drift** (`decision_status: decided_with_drift`): shown in plain words ("x ahead early, behind
late"), but counted toward no rank or leader until a confirmation run on a stable model snapshot decides it. Drift
after a pairing stops is never seen, so pin dated model snapshots. Each verdict reports those checks, its bettors'
capital against the bar and its **official interval** (`ci_official`): the anytime-valid interval for the edge,
at level 1 − alpha per pairing, valid whenever play stopped. Every other range (a pairing's t interval, strengths,
the front runner's margin, session Elo) is marked `ci_kind: 'fixed-time descriptive'`: it ignores that play
stopped on the results and that the front runner is picked by them. In seeded round robins that stop as the
runner does, strength intervals still covered about 95% and the front runner's margin about 93%, one family of
scenarios and not a guarantee (`ratings.STRENGTH_COVERAGE`). Round 6's t and sign-flip gate, which a rare-loss distribution inside the
chip bounds fooled 45% of the time, is still shown beside it as a **model-based** result: secondary, never
official, never counted. The adaptation track is judged by that gate, so its results are **exploratory**
(`language: 'exploratory'`): a player *leans ahead on the model-based test*, it never wins.

**What it costs.** Needing nothing has a price set by the bound: a pairing needs about log(2 / alpha per
pairing) × 10,000 / edge independent conversations, however long each is (about 1,450 to detect 50 bb/100 among
3 models). So `plan` now defaults to **one duplicate deal per conversation** (`--deals 1`, 25 conversations per
round): about as many deals as round 6's design (1,475 per pairing for 3 models, 2,150 for 9), but each
conversation is a fresh session, and the run measures play hand by hand rather than what a model learns about
its opponent over a match. Longer conversations measure that, at many more deals (`plan` prints the table: 55,600
deals per pairing at 50 deals per conversation). Shallower stacks (`--stack`) shrink the bound and the cost with
it, but change the game.

A plan carries a **design** (`ratings.normalize_design`): rounds of `round_deals` deals, each round `replicates`
conversations per pairing, a family error `alpha` of 5% split over the pairings (Bonferroni), a **target edge**
(50 bb/100 unless `--target-edge`) and as many rounds as the target needs. The power a plan states is the
official test's, simulated look by look (`ratings.procedure`). It is planned with a conversation effect
(`--assumed-icc`, 0.01) and memory between deals (`--assumed-ar1`, 0.2), shown under no dependence, the assumed
one and a worse one, which barely changes it (and never the error rate); `--budget-deals` caps the size, marking
the plan *underpowered*. For a round robin it also gives the chance that the true best player ends up the
**decided leader**:

```
Planned to detect a 50 bb/100 edge: up to 59 rounds of 25 conversations of 1 deal (1,475 deals per pairing), a
verdict possible from 9 conversations on. The official test is assumption-free: conversation means bounded by
+-10,000 bb/100 (the stacks), a verdict when a bettor's capital reaches 120. Chance that a pairing whose better
player is ahead by the target is decided (simulated; per-deal spread 350): 82% with the assumed dependence (ICC
0.01, AR(1) 0.2), 82% with none, 82% if it is worse; dependence changes only the power, never the error rate.
Chance of naming the true best of the 3 players the decided leader, assumed dependence: 67% if it is 50 bb/100
better than each other player and they are equal, >99% if it is 100 bb/100 better than each other player, 82% if
the players are evenly spaced 50 bb/100 apart.
```

With 9 models (36 pairings) the same defaults give 81% per pairing and 19% for a decided leader when one model is
50 bb/100 better than the rest (every one of its 8 pairings must be decided), more than 99% at twice that.

`plan` also warns when the plan has no **reference** player (`bot = "reference"`): the fixed, versioned bot that
gives every run a common opponent.

**Registered before play.** When a tournament starts, the runner writes a `registration` into the run's
`plan.json` before the first hand: the design, the judge that will decide it (the hash of `ratings.py`), the
calibration id of its test and the CLI versions. Only a run with a valid registration is a test. Nothing
recomputes it: a plan that says `registered: false`, a registration written after play began, a design changed
after it, a code change accepted with `--allow-code-change`, or a CLI version other than the pinned one anywhere
in the run all leave the run descriptive, and the output says why.

**One judge for stopping, recording and reporting.** After each round the runner asks `ratings.adjudicate` about
each pairing, stops on its verdict and records it in `decisions.json`, with the hash of the verified records it
judged (hands, events, call records and raw backend output). The ratings, the command line and the viewer show
that recorded decision as **official**, with the judge that made it (a verdict recorded by an earlier,
model-based judge says so). If today's code would judge the same deals differently, that is shown beside it as a
retrospective recomputation and never replaces it.

**Every counted hand is replayed.** A deal counts only if both of its plays replay from their seeds through the
engine's record verifier: the deal is dealt again, the recorded actions are replayed as legal moves, and the
cards, board, pots, payouts and events must match. A hand edited after the fact (a duplicated ace, a payout no
stack could cover) is rejected, here, in the history and in the style statistics, and the pairing can't be
decided. One reader (`ratings.read_conversation`) says which hands are verified results and which are still
provisional; everything that counts uses it, and it gives each written hand one status (verified, provisional or
rejected, with the reason) and the totals over verified deals, which the viewer shows. A model decision counts
as audited only if its accepted reply, replayed against the engine state before its action, reproduces the
executed action, amount, correction and speech.

**An official verdict counts only while its evidence stands.** Every refresh checks each official decision
against today's folders: the records it judged must all be there, replay, hash as they did, and pass today's
checks (decision coverage, audit, identity). If not, the verdict stays shown as recorded but stops counting
towards a decided rank or leader, with the reason: evidence *lost*, *changed*, *invalid* or *altered* (a record
edited so that the same judge no longer reproduces it), or *fails today* (the same records, judged by checks
that are stricter now).

**Eligibility.** A pairing can only be decided when both players' model identity is resolved and **attested**
(the CLI reported the model that answered and `ratings.same_model` says it is the one requested; reporting some
model is not enough; Codex can't report it, so the default design allows it and shows it as *unattested*; one
policy, `ratings.attestation`, judges games and the quiz alike: attested, unattested where allowed, mismatch
never; a counted result that involves an unattested player carries `identity_qualified: true` into every
headline, and `plan --strict-identity` (stored in the registered design) counts none of them), every
model action has exactly one decision record with its calls, every used reply was audited for tool use (the plain
`command` backend can't be, so it is never decided unless a design allows it; where decisions are linked to
their calls and raw output by a decision id, the judge derives each one's audit from those links), each player's
settings are the same in every match, every duplicate pair checks out and replays, and no deal is beyond the
bound. Otherwise the pairing stops as **ineligible**, with the reasons.

`tests/test_calibration.py` checks the false-winner rate on every test run: the round 6 counterexample (which it
keeps as a regression: it fails on the old gate, 450 of 1,000, and passes on the official test, 0), normal,
heavy-tailed, skewed and poker-like deals, memory, conversation effects of every shape and size, the least
favourable distribution at the bound, pairings that share their cards, false *even* at the ±15 margin, the
coverage of the anytime-valid interval, the advertised power against `decide` itself, and the coverage of the
strength intervals. A pairing ends with one of these verdicts:

- **a is better / b is better** (decided): that player's bettor reached the bar.
- **decided with drift** (not counted): the bar was reached, but the edge changed between early and late
  conversations; it needs a confirmation run on a stable model snapshot.
- **even** (decided): the edge is shown to be within ±15 bb/100; out of reach of the default design at 100 big
  blinds.
- **undecided after every planned look**: the data can't tell the two apart at this size. That is *not*
  evidence that they are equally good.
- **undecided so far** (more rounds to come) or **too early** (fewer conversations than any verdict needs).
- **ineligible** (stopped): an eligibility rule failed; the reasons are listed.
- **descriptive** (no test): pooled across runs, or a run that isn't preregistered (its sample sizes are
  labelled *retrospective*).

Each pairing also reports the deals the official test would need to detect an edge of 20, 50 or 100 bb/100 from
its own observed spread (`power.deals_for_80pct`), a planning number rather than a forecast. Repeated deals (the
same pair on the same cards in another run) are averaged, not counted twice. Players that never met, directly or
through others, are summarized in separate groups, each from its own data. A strength is one number per player
fitted to all its pairings, weighted by how many deals each has; every pairing shows the edge the strengths imply
next to the edge it actually had, and a warning when they disagree by more than their uncertainty. The direct
pairwise results come first.

**Winning here and being good at poker are reported apart, and neither says how good a model is at poker
overall.** Strengths and verdicts are relative to the players met: peer-relative winnings. There is no solver or
expert calibration set and no strong reproducible opponent yet, so the results claim no absolute strength. Each player's direct edge against the reference bot (`vs_reference`) is shown when the group has one; the
reference is a fixed heuristic, not a solver, measured against the other bots (+408 bb/100 against random, +290
against the calling station, +169 against the rule bot, though it wins only half what the rule bot does from
random), and the ratings output carries those limits (`reference`). `python -m poker_arena quiz` scores **poker
fundamentals** (spots with computed answers, in separately scored categories: the clear-cut basics, close calls,
multi-street lines, range inference from a described line, and bet sizing among 3+ options: competence in
stated-range decisions); `ratings` re-derives every answer itself (`ratings.verified_quiz`: each call re-checked
against its raw output, the accepted reply replayed in the spot's engine position, the spot key and the CLI
file checked, whatever the record's labels say), recounts the scores from those answers (a fallback or an answer
it cannot re-derive is never correct, valid-answer accuracy apart), attaches a score to a run's player only
with its provenance (quiz version, protocol, implementation, CLI version and file) and when its answers are
eligible under the same attestation policy as the games (attested, or unattested from a backend the policy
allows, such as Codex; a mismatch never), and prints them as their own section, never mixed into the standings.
The scores are descriptive counts: a category gets a binomial interval only if its spots were sampled
independently from a defined population (`sampling: independent`), never for the fixed basics or spots kept by
answer quotas.

**How long it takes.** A model makes about 2–3 decisions per hand, and a CLI call takes roughly 5–30 seconds,
so 100 hands take 30–90 minutes per model seat. The pilot is about 6,000 model calls: most of a day at
`--concurrency 3`, longer if usage limits pause it. A preregistered 3-model tournament at the defaults is up to
8,850 hands. Start with `plans/smoke.json`.

## What a match leaves behind

`runs/<plan>/<a>__vs__<b>/{a-first,b-first}/`:

| file | what |
|---|---|
| `config.json` | game settings, players by seat, the deal seeds |
| `events.jsonl` | every event (cards, actions, table talk, board, showdown, results) with which seats saw it, and each decision's private reasoning and notes |
| `calls.jsonl` | every model call: the exact prompt, the raw reply, any reasoning the CLI exposes, usage, latency, errors |
| `hands.jsonl` | one line per hand: cards, board, actions, net chips per seat |
| `state.json` | what a stopped match needs to resume |
| `result.json` | totals and bb/100 when the match finishes |

A tournament also writes its `plan.json` into `runs/<plan>/`, so the viewer can show matchups that haven't started.
Each `config.json` records the arena's git commit (`code_version`) and when the match started.

## History across runs

`runs/arena.db` is a SQLite index of every match ever played under `runs/`: each match's settings and code
version, every finished hand, every duplicate deal, and each player's style counts. It is filled from the
folders (only the ones that changed since last time) and never deletes anything, so a run keeps its place in the
history after its folder is removed. A folder that is started over replaces its own old rows.

```sh
python -m poker_arena history runs     # update arena.db and print the all-time leaderboard
```

The all-time leaderboard rates a player by **what it is: its model, reasoning level, protocol and
implementation** (or a bot and its settings), not by its id in one run. The protocol is the fingerprint of every
setting that changes play (instructions, isolation, tool policy, extra arguments, retries, and the conversation
length the model is told); the implementation is the hash of the code that plays (engine, agents, backends,
runner and prompt template, not the judge) and of the model's CLI: its version and the sha256 of its file, so a
rebuilt binary that keeps its version string still splits. So only identical setups pool: the same model at
another level, with other settings, with longer conversations, played by other code or another CLI version or
file, is a different player, and so is one from before protocols were recorded. The split happens at the hand
a change took effect, even in the middle of a match (a CLI upgraded between a stop and a resume, or a code change
accepted with `--allow-code-change`): deals before it, after it, and deals whose two plays straddle it are three
different players. Every hand is replayed from its seed when it is indexed; one that doesn't replay is listed as
rejected and never pooled, and the style counts it stores are of verified results only. When a code change is shown
not to affect play, list the implementations as equivalent in `runs/equivalences.json`
(`{"implementations": [["5e6f7a8b", "9a0b1c2d"]], "reason": "..."}`) and they pool; the leaderboard says so. A player without an exact model version (no `model`, or a bare alias like `sonnet`) is listed but not
rated, since two runs of it may not be the same model; so are a run's deals in which a different model answered
than the one configured (matched by `ratings.same_model`, the one rule everything uses: `claude-sonnet-5` is not
`claude-sonnet-5-5`; only spelling and dated snapshots of the same id are equated). In its own run such a player is shown with a flag, and none of its pairings
is decided. A deal the same pair already played in another run (same seeds, same settings) is
averaged with it rather than counted again. Deals only pool when they were played under the same game settings
(blinds and stacks change what bb/100 means); runs under other settings are listed as left out. Deals between two
entries of the same player don't count. A duplicate pair whose b-first play isn't its a-first play with the seats
swapped is left out and listed.

The all-time leaderboard is **descriptive only**. Its runs had different designs and stopped at different times,
so pooling them is no one's preregistered test: no verdict, rank or leader there is decided. Confirming a
result takes a new run with its own design. The viewer shows it as **All runs** in the run picker.

## The viewer

`python -m poker_arena view runs` serves every run under `runs/` at http://127.0.0.1:8800, with a picker for the
run (a tournament's `out` folder, or a `play --out`). It follows running matches live. A tournament with two tracks
is one entry in the picker: its page shows the official standings first, with a tab (and a compact section below
the matchups) for **Adaptation (models remember the session)**, which has its own standings, matchups and replays,
is marked *exploratory · not official*, never shows a Decided badge, a rank or a leader, words its results as
leanings ("Reference bot leans ahead · model-based", never "wins"), and says in one paragraph how the two tracks differ. The all-runs view pools official-track runs only.

The look is a card room: green felt, ivory cards, chips and brass. Players are always shown as a model and its
reasoning level (e.g. "GPT-6 Luna · High", from `label` in `players.toml` or made from `model` and `effort`), never
by the CLI that runs them. A player whose model wasn't recorded shows as, say, "GPT (model not recorded)".

Results never claim more than the evidence. **Decided** (a gold badge) marks only what the planned sequential test
settled: a matchup's *wins* or *even*, a rank, or a group's leader with a decided win over every other player in its
group. Every decided verdict shows what it rests on as two separate things: the **method** (which test) and the
**evidence** (whether the checks that test needs hold here), and the page never states the test's error bound without
them. When the official verdict comes from the assumption-free test (an anytime-valid test for bounded results, valid
if conversations are independent), the method chip says *assumption-free test* and the evidence chip says *checks
passed*, or names what is wrong: *check failed: independent deals* (its error rate is then not established, and a
leader resting on it is shown as *Leader, but a check failed*, not as decided) or *checks incomplete: conversations*.
A failed or unmade check is never hidden by the method. The older model-based gate's verdict is shown beside it as a
dashed, secondary *model-based: ...* chip that never decides. Verdicts from the model-based gate alone say
*diagnostics passed; distribution assumptions unverified*, *outside calibrated range* or *assumptions unchecked*. A
leader whose wins include one outside the calibrated range is shown as *Leader, outside the calibrated range*, not as a
decided leader. A matchup the judge may not decide (identity, audit, protocol or registration) shows **Ineligible**
with its reasons. A verdict recorded by the run's registered judge is marked *official*; a recomputation by today's
code is *retrospective*, and a verdict the runner recorded that is not a result today is shown as *at the time: ...*.
A verdict the test reached while the edge **drifted** between early and late conversations (the ratings'
`decision_status: decided_with_drift`, or a decided verdict with drift detected) is shown in a dashed pill in its own
words ("Gemini 4 Neon: ahead early, behind late") with a *needs confirmation* chip: never a Decided badge, never a rank,
never a leader; a group whose only clean sweep rests on one says so in its outcome row. A verdict withheld because a
player's model was never confirmed (`decided_identity_unattested`) reads "... ahead · model unconfirmed", and any
headline or verdict involving an unattested model (`identity_qualified`) carries a *model unconfirmed* chip.
Everything recomputed as hands finish is **ahead on points**. The all-runs view pools runs that each stopped on their
own rule, so it is descriptive only. A run without a registration written before play is **unregistered** (a void
registration says so with its reason): nothing in it is decided, and any sample-size figure is labelled
*retrospective*.

- **Standings.** The first screen answers four questions in plain words: who is playing (every model, then the bots,
  by name), who is ahead, whether that is decided, and a **Watch a hand** button (the live table when one is playing).
  Two short sentences above the results say how to read them: *bb/100* is big blinds won per 100 hands (above 0 wins
  chips), and the Decided badge marks only what a test planned before play settled. Operator guidance (such as how to
  add the reference bot to a plan) sits in a folded **Run your own**, never before the results.
  - **Ranges.** An official matchup's primary range is the test's **anytime-valid** interval (`ci_official`, else the
    betting confidence sequence `cs`, with its level, e.g. "anytime-valid 98.3%"): it holds whenever play stops.
    Every other range (strengths, the reference, a matchup's ordinary t interval, shown second) is labelled
    **fixed-time, descriptive** (the ratings' `ci_kind`): play stops on the results and the front runner is picked by
    them, so the page never calls it a 95% range.
  - **Players who met.** When the run (or the all-runs history) holds groups of players who never played each other,
    a compact selector picks one, named by its participants and protocol ("Gemini 4 Neon · Claude Sonnet 5.5 · GPT-6
    Luna · High reasoning · 50-hand conversations", "Baseline bots", "Earlier variants"). The current model cohort
    comes first and is shown by default; `?c=` in the address keeps the choice.
  - **Outcome row**, one per group of players who have met: the decided leader (an ace), or who is ahead on points
    (a face-down card), one sentence on whether it is decided and why ("Not decided yet: 1 of 3 matchups has a
    result, and play continues"), its strength with its fixed-time, descriptive range and the range's basis, and chips for the run's
    registration, how many matchups are decided, the leader's assumption status and any wrong-model calls. When the
    matchups don't fit one number per player (an additivity *warning*), the row leads with the **direct results**
    ("Rule bot beat Random bot · 801 bb/100 · Decided") and shows the fitted strength only as a secondary, qualified
    number.
  - **Against the reference bot**: each player's bb/100 against the fixed, versioned reference policy (a fixed
    heuristic yardstick, not a measure of how good a player is at poker in general) with its range and verdict, models
    first. What the reference is and is not (a heuristic, not a solver) and its own record against the baselines are one
    click away under *What the reference bot is*. A model-only group says plainly that it has no fixed yardstick.
  - **How this is judged**, folded: the test and its assumptions, the target edge ("Planned to detect a 50 bb/100
    edge: ~N deals per matchup; the matchups have M", or *retrospective*), power under the optimistic, moderate and
    pessimistic dependence assumptions, the words used on the page, and the ratings' notes. Caveats live here once.
  - **Standings**, per group: strength in bb/100 against the average opponent with its descriptive range on a shared
    axis. With a cohort picked, the deal count is the cohort's own ("150 duplicate deals in this group"), with the
    whole run's or archive's total beside it, labelled as such.
    A range that rests on blocks of deals inside single conversations is marked *within conversations* at the number
    (hatched on the axis): it leaves out how much fresh matches differ. The **decided order** comes first: decided
    ranks in order, then the players whose order is not decided, grouped by the places they share ("Places 3–4: the
    order of these two is not decided", listed by points within). An **Order: By points** control shows everyone by
    fitted strength instead, labelled descriptive. A numeral is shown only for a decided rank; a group with none is
    titled **Order on points**. The standings and matchups count only verified, committed hands (both plays replayed
    and checked); the style table says which hands it counts. Flags: *identity unresolved*, *model unconfirmed* (the model's tool
    never reported which model answered), *wrong model* or *partly confirmed*. A reference policy is tagged
    *reference*: it is the yardstick, not a contestant.
  - **Matchups**, the direct evidence: each pair's edge with its anytime-valid range (official runs) and its
    fixed-time, descriptive range and basis, its verdict with what it rests
    on, the checks used and conversations it has, and an *off the fit* line only where the fit's additivity check
    warns ("The strengths predict +502; this matchup's own result is +273"). Where too few matchups link the players
    to test the fit, it says *fit: insufficient evidence to assess* instead.
  - **Poker fundamentals**, when the run has a `quiz.json`: one line per model with its score on the scored
    category and on the harder ones ("32/32 basics · 90/96 harder"), linking to its own tab. It is filled in after
    the standings are on screen, so the standings never wait for it.
  - **More detail**, folded sections that remember whether they are open: **Head to head** (one matrix per group,
    shaded only where decided; on a phone the rows show monograms with a legend beneath), **How they play**
    (tendencies, not quality; on a phone a **Show** picker selects one tendency at a time) and the **Audit** (its
    one-line summary on the fold): per model, from `calls.jsonl` and the decisions (computed by the server as
    `run_audit`): decisions and calls, identity, turn audit (verified, opaque, unrecorded), tool-use rejections,
    unaudited turns, retries and resizes, fallbacks, failed tries and rate-limit waits. On a phone each model is an
    expandable card instead of a wide table.
  - **Identity** uses one rule everywhere (audit, standings, replay): a call's `identity_match` from the runner when
    recorded, else the reported model compared with the requested one by `ratings.same_model`. *Model confirmed*
    means the reported model is the one asked for; a backend's bare "attested" (some model was reported) is never
    shown as confirmed.
- **Fundamentals** (only for a run with `quiz.json`): every player answers the same spots, each with an answer worked
  out from the cards. The headline (**Scores · Basics**) is one category, the one the quiz scores (`scored_category`,
  else the basics when there are others), and every figure in it comes from that category's own answers: spots
  answered best out of its spots, the share right among valid answers, fallbacks (no valid answer after every retry:
  never counted right, even where folding was best) and big blinds of expected value lost, with a provenance chip
  (*provenance checked*, *ineligible* with reasons, or *provenance not recorded*). **Harder categories, scored
  separately** gives every other category its own count out of its own spots, with fallbacks and bb lost; none of them
  enters the headline. No error ranges are shown unless the quiz defines its sampling population.
  The reference policy is called out as the yardstick. The **Spots** list is grouped by category (a long category
  shows 12, then *Show all*), with filters for category and for answers (*Players disagree*, *Someone missed*,
  *Everyone right*); a generated spot whose title is its id gets one from its hand and situation ("J9o on the river:
  Villain bets 3 bb into 6 bb"). Each spot opens on the same table component (the deciding seat's cards up, the
  opponent's down), with **the question exactly as the model received it** (the spot's stored `prompt`, else the
  prompt in the answers' call log, else the server's rebuild from `quiz.question`, marked with the quiz code's
  version), and what it states in short (Villain's range, the hand so far, Villain's line, river plan or responses to
  each bet, the allowed choices), beside the worked answer: the pot, price, equity, best answer and every player's
  choice, its EV lost and its reasoning. The price is a **break-even** share only when the call ends the betting
  (river, or someone all in); otherwise it is *immediate pot odds, before future betting*.
  This is separate from the standings: fundamentals, not winnings against these opponents. A quiz file marked
  `"fixture": true` is labelled as made-up layout data (`tests/fixtures/quiz.json` is one).
- **Watch.** The first screen, on a desktop and on a phone, is the table: a one-strip scoreboard (both players by
  model and level, who is up and by how much, bb/100, how many verified deals that counts, and the verdict pill), one
  row of replay controls, then both seats and the board. Everything else comes after the table.
  - **One sequence per pairing.** The replay walks every deal of the pairing in playing order, round by round and
    conversation by conversation (the server's `/api/pairing`). *Deal 226 of 275 · Round 10 · conversation 1*: the
    deal arrows, <kbd>←</kbd> <kbd>→</kbd>, **Auto-next deal** and **Big swings** all cross conversations, so an
    official run of one-deal conversations plays continuously.
  - **Verified totals.** The scoreboard counts verified deals only (both plays committed, checked as a duplicate pair
    and replayed from their seeds, by the same conversation reader the ratings use). A deal still in play or waiting
    for its other play is *provisional*, and one the verifier rejected is *rejected · not counted*; both are marked on
    the deal and counted apart ("2 provisional · not counted"), whether or not the ratings are available.
  - **Matchups**, listed by pairing with the verdict and the edge over verified deals (bb/100, from the ratings). On a
    phone the list folds into an *All matchups* button.
  - **The deal map**, below the tables: every deal of the pairing as a small square, one row per conversation (or per
    round of one-deal conversations), coloured by who won it over both plays, a gold edge for a showdown, hatched for
    provisional and outlined red for rejected; ringed: the biggest swings. Clicking one opens it.
  - **About this result**, folded: the verdict in full, what it rests on, which deals the totals count and the
    verifier's reasons for any rejection.
  - A finished deal opens at its first step with **Watch from start**; a deal still being played follows the latest
    step. Each deal shows **both plays at the table** (same cards, seats swapped). The felt says whose turn it is from
    the engine's recorded `to_act`, the price to call and its pot odds: a break-even share only when the call ends the
    betting (river, or someone all in), else *immediate pot odds, before future betting*; each seat shows its made hand
    ("a pair of queens", plus an open draw on the flop or turn), labelled as computed from both players' cards. On a
    wide screen the tables sit side by side and move either **street by street** (both reach the flop, turn, river and
    result together; the one with fewer actions waits) or **Play 1, then Play 2**; on a narrower one, two tabs above
    the table show what each play just did, tapping one shows its table, and a line says when the same cards drew
    different choices. On a phone the control bar (deal arrows, step, play, scrubber) stays pinned at the top and the
    secondary controls (speed, auto-next, reasoning, where they differ) sit behind **Options**.
  - **Where they differ** jumps to the first decision where the two models, holding the same cards, chose
    differently. **The two plays, street by street** puts each street's choices side by side and fills in as the
    replay goes. **Speed** (slow, normal, fast) sets how long each step stays on screen; opening a decision's **why?**
    pauses playback. Below each table is a one-line-per-action log with the table talk; **why?** opens that decision's
    private reasoning and its record: identity (attested or not, and the model seen), turn audit, tries and time, and
    each rejected try with what it asked and why; **Reasoning** opens all of them. When the game changed a move, the
    log says what the model asked for, what was played and why (*resized*, *asked again*, *tool rejected*,
    *auto-played*).
  Keys: <kbd>space</kbd> play or pause, <kbd>,</kbd> <kbd>.</kbd> step, <kbd>←</kbd> <kbd>→</kbd> deals,
  <kbd>d</kbd> where they differ, <kbd>r</kbd> reasoning. `?deal=12&step=4` in the address opens a deal at a step.
- **Live.** Whether a run is running, between rounds, stopped or finished comes from the runner's `tournament.json`
  (or, for older runs, from `plan.json` and `decisions.json`), never from which folders exist. A run with a table
  playing refreshes every 6 seconds, an unfinished but quiet one every minute. A refresh never interrupts a replay.
  The server recomputes a run's ratings whenever any of its record files changes: hands, events and calls, and the
  `plan.json`, `decisions.json`, `tournament.json`, `config.json` and `quiz.json` files (so a verdict-only change shows
  at once), plus the players file; for all runs these same files are the history's inputs. Ratings are never on a
  page's critical path: they are computed in a separate process, kept in `runs/.viewer-cache/` and served from there
  at once (marked *updating* while a newer version is computed), and the server warms every run when it starts. The
  replay never waits for them. A run computed for the first time shows its players and raw, unchecked matchup scores
  until its standings arrive a few seconds later. Listing runs and matchups never re-reads the archive: each
  conversation is summarized once and kept until one of its record files changes, a run's page reads that run's
  folder only, and the server checks every conversation in the background when it starts.

The table (`viewer/table.js`) is its own component: it builds one snapshot per step from a hand's events and
shows any snapshot, for two or three seats, with a seat's cards optionally face down, so a human seat can reuse it.

```sh
python -m poker_arena status runs                 # the same progress in the terminal
python -m poker_arena status runs --health        # and per model: reply times, retries, fallbacks
python -m poker_arena export runs --out site       # a static copy (no server needed) to host or share
python -m http.server -d site 8000
```

The export holds the viewer and every match's events (cards, actions, talk, reasoning, notes), but not the
raw prompts and replies in `calls.jsonl`, and player settings are reduced to label, model, reasoning level and bot.

## Extending it

- **A model**: add a `[players.name]` entry. Anything with a CLI works through `command`.
- **A game**: write a module in `poker_arena/games/` with a `Game` subclass (rules prompt, response schema, and an
  `Episode` that deals from a seed, says whose turn it is, applies actions and emits events) and register it in
  `games/__init__.py`. The runner, duplicate pairing, logs, ratings and viewer (apart from the hand renderer) don't
  change. The `Holdem` engine already deals 3–9 seats; 6-max still needs a seat rotation for duplicate play and a multi-way rating.

## Why the official CLIs, not Pi or Hermes

[Hermes Agent](https://hermes-agent.nousresearch.com/) is a full agent with its own tools, memory and prompts, so
it would change what the eval measures. [Pi](https://github.com/badlogic/pi-mono)'s `pi-ai` is a good multi-provider
library, but it is TypeScript, and logging into a Claude Pro/Max plan through a third-party tool is against
Anthropic's terms. The official CLIs are the supported way to use each subscription. Pi (or Hermes) can still be
plugged in as a `command` backend if you want to compare harnesses.

## Caveats

- What each CLI lets you control (tested on claude 2.1.332, codex 0.172.0, agy 1.5.14):
  - `claude`: system prompt replaced, tools and MCP off, your settings and hooks skipped, thinking shown as summaries.
  - `codex`: its coding instructions are replaced by the game prompt, but its tools can't be switched off (the
    read-only sandbox and the empty folder are the guard), and its reasoning comes back only as short titles.
  - `agy`: the game prompt runs as an agent with no tools of its own, and MCP servers, commands, files and URLs
    are denied; agy can still add its own system notes to a conversation. Always set a Gemini `model`: agy's
    default comes from your own settings and may not be Gemini (`doctor` shows which model answered). Thinking
    is a condensed version read from agy's transcript.
  Every reply also carries its own `reasoning` field, so the viewer has something for every model.
- Results from different backends of the same model (CLI vs API) shouldn't be mixed on one leaderboard.
- One conversation of 50 deals per pairing decides nothing; a 50 bb/100 edge needs about 2,000 deals per pairing
  in conversations of 50 under moderate dependence (see "How sure the results are"). The first model runs had no
  preregistered design: they are descriptions of those matches.
