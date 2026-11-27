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
- **Ratings.** Chips decide, as the models are told. Only one thing is a test: each pairing's preregistered
  sequential procedure, whose verdicts (*a is better*, *b is better*, *even*) keep the chance of any false
  verdict in a tournament under 5%: exactly when each conversation's result is symmetric about the true edge
  (for instance two interchangeable players), and in simulation for skewed results up to a stated size (beyond it,
  rare large one-sided swings, no test of this size can keep the promise; the methodology shows by how much it
  fails). It treats each conversation (a fresh pair of matches) as one unit and needs at least 8 of them (12 in
  the default design). A run without a preregistered design, and any pairing with a player whose model could not
  be verified, is descriptive only. A **decided** rank or leader is built from those verdicts alone; a leader has
  a decided win over every other player of its group. Everything else is **descriptive**: a player's strength (its
  edge in bb/100 against the average opponent of its group, from a least-squares fit over duplicate deals), its
  95% interval (a bootstrap that resamples whole slates of shared deals, or blocks of 10 deals within one
  conversation when that is all there is), the order by points and the front runner, and session Elo (share of
  10-deal sessions won). [docs/methodology.md](docs/methodology.md) has the method, the assumptions, the
  calibration, power and coverage results and the limits.

## Quick start

```sh
git clone https://github.com/bin01/poker-arena && cd poker-arena
python3 -m venv .venv && . .venv/bin/activate      # Python 3.11+
pip install -e '.[dev]'
python -m pytest -q                                  # engine, evaluator, bots, ratings, a full stand-in model match
# where the system temp folder is read-only: pytest's capture is set to sys (pyproject), which needs no temp file,
# and tests/conftest.py falls back to tests/.tmp; to choose the folder yourself (and skip .pytest_cache):
#   mkdir -p tests/.tmp && TMPDIR=tests/.tmp python -m pytest -q -p no:cacheprovider

# Poker fundamentals: 32 clear-cut decision spots with computed answers, asked of each player -> runs/quiz/quiz.json
python -m poker_arena quiz reference claude-haiku-5-5-high --out quiz

# Free and instant: the bots against each other, then the viewer
python -m poker_arena play rule station --deals 50 --out runs/demo
python -m poker_arena view runs/demo                 # http://127.0.0.1:8800
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
finished hand. Write your own plan with `python -m poker_arena plan A B C --deals 50 --out runs/mine --write plans/mine.json`.

### How sure the results are

Only the sequential test decides anything; the standings beside it describe. Even with duplicate deals one deal
swings a lot (in the first model runs the result per deal had a spread of about 340 bb/100), and a model's
conversations can differ from each other, so the test treats each conversation (a fresh pair of matches) as one
unit: a verdict needs at least 8 conversations, and never comes from conversations that don't differ at all.

A plan carries a **design** (`ratings.normalize_design`): rounds of `round_deals` deals (50), each
round `replicates` conversations per pairing (`plan` defaults to 4), a family error `alpha` of 5% split over the
pairings (Bonferroni), a **target edge** (50 bb/100 unless `--target-edge`) and as many rounds as the target needs.
The power a plan states is the **complete adjudicator's**, simulated (`ratings.procedure`): the chance that the
better player gets a decided verdict that *counts*, after both tests, the minimum number of conversations, every
look and the diagnostics a verdict needs. It is planned with a conversation effect (`--assumed-icc`, 0.01) and
memory between deals (`--assumed-ar1`, 0.2), shown under no dependence, the assumed one and a worse one;
`--budget-deals` caps the size, marking the plan *underpowered*. For a round robin it also gives the chance that
the true best player ends up the **decided leader**:

```
Planned to detect a 50 bb/100 edge: up to 10 rounds of 4 conversations of 50 deals (2,000 deals per pairing), a
verdict from 12 conversations on. Chance that a pairing whose better player is ahead by the target gets a decided
verdict that counts (the complete adjudicator, simulated; per-deal spread 350): 83% with the assumed dependence
(ICC 0.01, AR(1) 0.2), 99% with none, 60% if it is worse (ICC 0.02, AR(1) 0.3). Chance of naming the true best
of the 3 players the decided leader, assumed dependence: 69% if it is 50 bb/100 better than each other player
and they are equal, 71% if it is 100 bb/100 better than each other player, 70% if the players are evenly spaced
50 bb/100 apart.
```

Read the last sentence before planning a big round robin: with 9 models (36 pairings) the same defaults give
83% per pairing but only 22% (one best by 50) to 40% (by 100) for a decided leader, because a large edge is decided
at the first possible look, where the diagnostics are noisy and fail about one time in six
(docs/methodology.md, *Planning*). Through `adjudicate` itself the 3-model numbers come out at 82% (assumed
dependence), 99% and 59%.

`plan` also warns when the plan has no **reference** player (`bot = "reference"`): the fixed, versioned bot that
gives every run a common opponent.

**Registered before play.** When a tournament starts, the runner writes a `registration` into the run's
`plan.json` before the first hand: the design, the judge that will decide it (the hash of `ratings.py`), the
calibration of its bars and the CLI versions. Only a run with a valid registration is a test. Nothing recomputes
it: a plan that says `registered: false`, a registration written after play began, a design changed after it, a
code change accepted with `--allow-code-change`, or a CLI version other than the pinned one anywhere in the run
all leave the run descriptive, and the output says why.

**One judge for stopping, recording and reporting.** After each round the runner asks `ratings.adjudicate` about
each pairing, stops on its verdict and records it in `decisions.json`, with the hash of the verified hands it
judged. The ratings, the command line and the viewer show that recorded decision as **official**, with the judge
that made it. If today's code would judge the same deals differently, that is shown beside it as a retrospective
recomputation and never replaces it.

**Every counted hand is replayed.** A deal counts only if both of its plays replay from their seeds through the
engine's record verifier: the deal is dealt again, the recorded actions are replayed as legal moves, and the
cards, board, pots, payouts and events must match. A hand edited after the fact (a duplicated ace, a payout no
stack could cover) is rejected, here and in the history, and the pairing can't be decided.

**An official verdict counts only while its evidence stands.** Every refresh checks each official decision
against today's folders: the hands it judged must all be there, replay, hash as they did, and their records
(decision coverage, audit, identity) must pass today's checks. If not, the verdict stays shown as recorded but
stops counting towards a decided rank or leader, with the reason: evidence *lost*, *changed*, *invalid* or
*altered* (a record edited so that the same judge no longer reproduces it), or *fails today* (the same records,
judged by checks that are stricter now).

**Eligibility.** A pairing can only be decided when both players' model identity is resolved and **matched**
(the CLI reported the model that answered and `ratings.same_model` says it is the one requested; reporting some
model is not enough; Codex can't report it, so the default design allows it and shows it as *unattested*), every
model action has exactly one decision record with its calls, every used reply was audited for tool use (the plain
`command` backend can't be, so it is never decided unless a design allows it), each player's settings are the
same in every match, and every duplicate pair checks out and replays. Otherwise the pairing stops as
**ineligible**, with the reasons.

After each round a winner needs two tests to agree on the conversations so far: a t-test on conversation means,
against a bar calibrated by simulation for exactly that schedule and for skewed results (a conversation effect of
normal, exponential, lognormal or rare-large-loss shape, holding up to half the spread of a conversation's mean),
and a sign-flip randomization test that is exact for any results symmetric about the true edge. **If either
assumption holds**, then with no real edge the chance of any false verdict in the whole tournament stays under
5%. Beyond both (a pairing whose conversations are mostly small wins and now and then a loss nine times as large,
bigger than the card noise) the error grows to several times the stated rate, and no test of this size can
prevent that; docs/methodology.md has the table. So every verdict reports **diagnostics**: *diagnostics passed;
distribution assumptions unverified* (the data show no sign of leaving the calibrated family, which is not proof
that they are in it), *diagnostics failed* (they are outside it) or *diagnostics incomplete*. A decided verdict
whose diagnostics failed never makes a decided leader. `tests/test_calibration.py` checks the false-winner rate on
every test run for normal, heavy-tailed, skewed and poker-like results, with memory, conversation effects of each
shape, symmetric results of any size, pairings that share their cards, and the counterexamples of earlier
reviews, the false-*even* rate at the ±15 margin, the advertised power against `adjudicate` itself, and the
coverage of the strength intervals. A pairing ends with one of these verdicts:

- **a is better / b is better** (decided): both tests rejected at a planned look.
- **even** (decided): the edge is shown to be within ±15 bb/100 (an equivalence test, each side calibrated so that
  a true edge of ±15 or more is called even at most alpha / n_pairings of the time, skewed results included). At
  the default spread this is out of reach of the default design.
- **undecided after every planned look**: the data can't tell the two apart at this size. That is *not*
  evidence that they are equally good.
- **undecided so far** (more rounds to come) or **too early** (fewer conversations than a verdict needs).
- **ineligible** (stopped): an eligibility rule failed; the reasons are listed.
- **descriptive** (no test): pooled across runs, or a run that isn't preregistered (its sample sizes are
  labelled *retrospective*).

Each pairing also reports the deals it would need to detect an edge of 20, 50 or 100 bb/100 from its own
observed spread of conversation means (`power.deals_for_80pct`), a planning number rather than a forecast.
Repeated deals (the same pair on the same cards in another run) are averaged, not counted twice. Players that
never met, directly or through others, are summarized in separate groups, each from its own data. A strength
is one number per player fitted to all its pairings, weighted by how many deals each has; every pairing shows
the edge the strengths imply next to the edge it actually had, and a warning when they disagree by more than
their uncertainty. The direct pairwise results come first.

**Winning here and being good at poker are reported apart.** Strengths and verdicts are relative to the players
met. Each player's direct edge against the reference bot (`vs_reference`) is shown when the group has one, and
`python -m poker_arena quiz` scores **poker fundamentals** (clear-cut spots with computed answers); `ratings`
prints those scores as their own section and never mixes them into the standings.

**How long it takes.** A model makes about 2–3 decisions per hand, and a CLI call takes roughly 5–30 seconds,
so 100 hands take 30–90 minutes per model seat. The pilot is about 6,000 model calls: most of a day at
`--concurrency 3`, longer if usage limits pause it. Start with `plans/smoke.json`.

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
runner and prompt template, not the judge) and of the model's CLI version. So only identical setups pool: the
same model at another level, with other settings, with longer conversations, played by other code or another CLI
version, is a different player, and so is one from before protocols were recorded. The split happens at the hand
a change took effect, even in the middle of a match (a CLI upgraded between a stop and a resume, or a code change
accepted with `--allow-code-change`): deals before it, after it, and deals whose two plays straddle it are three
different players. Every hand is replayed from its seed when it is indexed; one that doesn't replay is listed as
rejected and never pooled. When a code change is shown
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
run (a tournament's `out` folder, or a `play --out`). It follows running matches live.

The look is a card room: green felt, ivory cards, chips and brass. Players are always shown as a model and its
reasoning level (e.g. "GPT-6 Luna · High", from `label` in `players.toml` or made from `model` and `effort`), never
by the CLI that runs them. A player whose model wasn't recorded shows as, say, "GPT (model not recorded)".

Results never claim more than the evidence. **Decided** (a gold badge) marks only what the planned sequential test
settled: a matchup's *wins* or *even*, a rank, or a group's leader with a decided win over every other player in its
group. Every decided verdict carries its assumption status beside it (*diagnostics passed; distribution assumptions
unverified*, *outside calibrated range* or *assumptions unchecked*), and the page never states the test's error bound
without its assumptions. A
leader whose wins include one outside the calibrated range is shown as *Leader, outside the calibrated range*, not as a
decided leader. A matchup the judge may not decide (identity, audit, protocol or registration) shows **Ineligible**
with its reasons. A verdict recorded by the run's registered judge is marked *official*; a recomputation by today's
code is *retrospective*, and a verdict the runner recorded that is not a result today is shown as *at the time: ...*.
Everything recomputed as hands finish is **ahead on points**. The all-runs view pools runs that each stopped on their
own rule, so it is descriptive only. A run without a registration written before play is **unregistered** (a void
registration says so with its reason): nothing in it is decided, and any sample-size figure is labelled
*retrospective*.

- **Standings.** The first screen answers four questions in plain words: who is playing (every model, then the bots,
  by name), who is ahead, whether that is decided, and a **Watch a hand** button (the live table when one is playing).
  - **Outcome row**, one per group of players who have met: the decided leader (an ace), or who is ahead on points
    (a face-down card), one sentence on whether it is decided and why ("Not decided yet: 1 of 3 matchups has a
    result, and play continues"), its strength with the 95% range and the range's basis, and chips for the run's
    registration, how many matchups are decided, the leader's assumption status and any wrong-model calls. When the
    matchups don't fit one number per player (an additivity *warning*), the row leads with the **direct results**
    ("Rule bot beat Random bot · 801 bb/100 · Decided") and shows the fitted strength only as a secondary, qualified
    number.
  - **How this is judged**, folded: the test and its assumptions, the target edge ("Planned to detect a 50 bb/100
    edge: ~N deals per matchup; the matchups have M", or *retrospective*), power under the optimistic, moderate and
    pessimistic dependence assumptions, the words used on the page, and the ratings' notes. Caveats live here once.
  - **Standings**, per group: strength in bb/100 against the average opponent with its 95% range on a shared axis.
    A range that rests on blocks of deals inside single conversations is marked *within conversations* at the number
    (hatched on the axis): it leaves out how much fresh matches differ. A numeral is shown only for a decided rank; a
    group with none is titled **Order on points**. Flags: *identity unresolved*, *model unconfirmed* (the model's tool
    never reported which model answered), *wrong model* or *partly confirmed*. A reference policy is tagged
    *reference*: it is the yardstick, not a contestant.
  - **Matchups**, the direct evidence: each pair's edge with its 95% range and basis, its verdict with what it rests
    on, the checks used and conversations it has, and an *off the fit* line only where the fit's additivity check
    warns ("The strengths predict +502; this matchup's own result is +273"). Where too few matchups link the players
    to test the fit, it says *fit: insufficient evidence to assess* instead.
  - **Poker fundamentals**, when the run has a `quiz.json`: one line with the best score and the reference policy's,
    linking to its own tab.
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
- **Fundamentals** (only for a run with `quiz.json`): every player answers the same textbook spots (call or fold at a
  price, all-in calls, shove or fold), each with an answer worked out from the cards. Per player: spots answered
  best and big blinds of expected value lost against the best answer; the reference policy is called out as the
  yardstick. Each spot opens on the same table component (the deciding seat's cards up, the opponent's down), with
  the pot, price, break-even share, equity, best answer and every player's choice, its EV lost and its reasoning.
  This is separate from the standings: fundamentals, not winnings against these opponents. A quiz file marked
  `"fixture": true` is labelled as made-up layout data (`tests/fixtures/quiz.json` is one).
- **Watch.** Matchups are listed by pairing (full names, verdict, net), and the selected pairing opens into its
  rounds and replicates (*Round 2 · Rep 1*), each with its own net. On a phone the list folds into an *All matchups*
  button with the round pills beneath it. The header leads with the whole pairing across every round and replicate
  (net big blinds, bb/100 and the cumulative verdict); the selected round and replicate's own score is labelled
  beneath. The **deal map** has a chip per deal, coloured by who won it over both plays (folded on a phone). A
  finished deal opens at its first step with **Watch from start**; a deal still being played follows the latest step.
  Each deal shows **both plays at the table** (same cards, seats swapped). The felt says whose turn it is from the
  engine's recorded `to_act`, the price to call and the break-even share (pot odds); each seat shows its made hand
  ("a pair of queens", plus an open draw on the flop or turn), labelled as computed from both players' cards.
  **The two plays, street by street** puts each street's choices side by side and marks where the model holding the
  same cards chose differently; it fills in as the replay goes. On a wide screen they sit side by side
  and move either **street by street** (both reach the flop, turn, river and result together; the one with fewer
  actions waits) or **Play 1, then Play 2**; on a narrow one a **Play 1 / Play 2** switch shows one, and playback
  plays Play 1 to its end, then Play 2, then the next deal. On a phone, pressing play folds the scoreboard to one
  line and brings the table up. The deal's controls stay on screen while scrolling. Under them, **Where they differ**
  jumps to the first decision where the two models, holding the same cards, chose differently, and **Big swings**
  lists the deals with the largest results (ringed on the deal map). **Speed** (slow, normal, fast) sets how long each
  step stays on screen; opening a decision's **why?** pauses playback. The table marks whose turn it is (**to act**)
  apart from the last action of each seat, and the winner lights up only at the end. With **Auto-next deal** on,
  playback goes on into the next deal. Below each table is a one-line-per-action log with the table talk;
  **why?** opens that decision's private reasoning and its record: identity (attested or not, and the model seen),
  turn audit, tries and time, and each rejected try with what it asked and why; **Reasoning** opens all of them. A
  call shows what it faced and the share it needed. When the game changed a move, the log says what the model asked
  for, what was played and why (*resized*, *asked again*, *tool rejected*, *auto-played*).
  Keys: <kbd>space</kbd> play or pause, <kbd>,</kbd> <kbd>.</kbd> step, <kbd>←</kbd> <kbd>→</kbd> deals,
  <kbd>d</kbd> where they differ, <kbd>r</kbd> reasoning. `?deal=12&step=4` in the address opens a deal at a step.
- **Live.** Whether a run is running, between rounds, stopped or finished comes from the runner's `tournament.json`
  (or, for older runs, from `plan.json` and `decisions.json`), never from which folders exist. A run with a table
  playing refreshes every 6 seconds, an unfinished but quiet one every minute. A refresh never interrupts a replay.
  The server recomputes a run's ratings whenever any of its record files changes: hands, events and calls, and the
  `plan.json`, `decisions.json`, `tournament.json`, `config.json` and `quiz.json` files (so a verdict-only change shows
  at once), plus the players file; for all runs these same files are the history's inputs.

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
