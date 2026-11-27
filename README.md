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

A plan carries a preregistered **design** (`ratings.normalize_design`): rounds of `round_deals` deals (50), each
round `replicates` conversations per pairing (`plan` defaults to 4), a family error `alpha` of 5% split over the
pairings (Bonferroni), a **target edge** (50 bb/100 unless `--target-edge`) and as many rounds as 80% power at that
target needs. Power is planned from the spread of a conversation's mean, which counts the dependence inside a
conversation: a conversation effect (`--assumed-icc`, 0.01) and memory between deals (`--assumed-ar1`, 0.2).
`plan` prints the power under no dependence, the assumed one and a worse one, and `--budget-deals` caps the size,
marking the plan *underpowered*:

```
Planned to detect a 50 bb/100 edge: up to 9 rounds of 4 conversations of 50 deals (1,800 deals per pairing), a
verdict from 12 conversations on. Power at the target, per-deal spread 350: 84% with the assumed dependence
(ICC 0.01, AR(1) 0.2), >99% with none, 65% if it is worse (ICC 0.02, AR(1) 0.3).
```

After each round a winner needs two tests to agree on the conversations so far: a t-test on conversation means,
against a bar calibrated by simulation for exactly that schedule and for skewed results (a conversation effect of
normal, exponential, lognormal or rare-large-loss shape, up to the size of the conversations' card noise), and a
sign-flip randomization test that is exact for any results symmetric about the true edge. So with no real edge
the chance of any false verdict in the whole tournament stays under 5% whenever either assumption holds. Beyond
both (a pairing whose conversations are mostly small wins and now and then a loss nine times as large, bigger than
the card noise) the error grows to several times the stated rate, and no test of this size can prevent that;
docs/methodology.md has the table. `tests/test_calibration.py` checks the false-winner rate on every test run for
normal, heavy-tailed, skewed and poker-like results, with memory, conversation effects of each shape, symmetric
results of any size, pairings that share their cards, and the counterexamples of the last review (two constant
conversations; a zero-mean +50 / −450 distribution), and checks the power and the coverage of the strength
intervals. A pairing ends with one of these verdicts:

- **a is better / b is better** (decided): both tests rejected at a planned look.
- **even** (decided): the edge is shown to be within ±15 bb/100 (an equivalence test). This needs thousands of deals.
- **undecided after every planned look**: the data can't tell the two apart at this size. That is *not*
  evidence that they are equally good.
- **undecided so far** (more rounds to come) or **too early** (fewer conversations than a verdict needs).
- **descriptive** (no test): pooled across runs, a run without a preregistered design (its sample sizes are
  labelled *retrospective*), or a player whose identity is in doubt.

Each pairing also reports the deals it would need to detect an edge of 20, 50 or 100 bb/100 from its own
observed spread of conversation means (`power.deals_for_80pct`), a planning number rather than a forecast.
Repeated deals (the same pair on the same cards in another run) are averaged, not counted twice. Players that
never met, directly or through others, are summarized in separate groups, each from its own data.

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

The all-time leaderboard rates a player by **what it is: its model, reasoning level and protocol** (or a bot and
its settings), not by its id in one run. The protocol is the fingerprint of every setting that changes play
(instructions, isolation, tool policy, extra arguments, retries, and the conversation length the model is told),
so only identical setups pool: the same model at another level, with other settings or with longer conversations,
is a different player, and so is one from before protocols were recorded. A player without an exact model version (no `model`, or a bare alias like `sonnet`) is listed but not
rated, since two runs of it may not be the same model; so are a run's deals in which a different model answered
than the one configured (matched exactly: `claude-sonnet-5` is not `claude-sonnet-5-5`; only spelling and dated
snapshots of the same id are equated). In its own run such a player is shown with a flag, and none of its pairings
is decided. A deal the same pair already played in another run (same seeds, same settings) is
averaged with it rather than counted again. Deals only pool when they were played under the same game settings
(blinds and stacks change what bb/100 means); runs under other settings are listed as left out. Deals between two
entries of the same player don't count. Runs from different code versions do pool, and the leaderboard says so.

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
group. Everything recomputed as hands finish is described as **ahead on points**. The all-runs view pools runs that
each stopped on their own rule, so it is descriptive only: nothing there is decided. A run without a recorded design
(`design.registered: false`, or an older run whose design the ratings fill in) is **unregistered**: a notice says so,
no verdict or rank in it is decided, and any sample-size figure is labelled *retrospective*.

- **Standings.** One line on what the page is and a **Watch the games** button, then the results; *How to read
  this page* is folded below them.
  - **Who is winning?** One card per group of players who have met, from that group's own data: the decided leader
    as an ace, or a face-down card (*No decided leader yet*) naming who is ahead on points, which matchups are
    decided, and the target edge: "Planned to detect a 50 bb/100 edge: ~N deals per pairing; the pairings have M",
    with the plan's power under its optimistic, moderate and pessimistic dependence assumptions, an *underpowered*
    flag when a deal budget caps it, and the minimum number of conversations a verdict needs.
  - **Standings**, per group: strength in bb/100 against the average opponent with its 95% range on a shared axis.
    A numeral is shown only for a decided rank; a group with none is titled **Order on points** (provisional) and
    has no rank column, and "difference with ... unresolved"
    names who it can't be told apart from. Net big blinds, hands and session Elo (secondary) beside it; on a phone
    the net and hands sit under the name. A player in the fit whose exact model is unknown stays in its group with
    an *identity unresolved* flag; players left out of the fit are listed under **Not in the standings** with why.
  - **Matchups**: each pair's edge with its 95% range, its verdict (*wins*, *even* with the Decided badge;
    *undecided*, *no winner*, *too early* otherwise), the planned checks used, how many of the conversations a
    verdict needs it has so far, the deals needed to detect a 20/50/100 bb/100 edge, and a link to watch it.
  - **Head to head** (three or more players): bb/100 for every pair, shaded only where the matchup is decided.
  - **How they play**: tendencies, not quality, under plain names (*Plays preflop*, *Raises preflop*, ...); tapping a
    column name shows what it counts. Each rate shows its count out of its chances (e.g. 144/186), with splits by
    seat (button or big blind) and by opponent when the stats provide them.
- **Watch.** Matchups are listed by pairing (full names, verdict, net), and the selected pairing opens into its
  rounds and replicates (*Round 2 · Rep 1*), each with its own net. On a phone the list folds into an *All matchups*
  button with the round pills beneath it. The header leads with the whole pairing across every round and replicate
  (net big blinds, bb/100 and the cumulative verdict); the selected round and replicate's own score is labelled
  beneath. The **deal map** has a chip per deal, coloured by who won it over both plays (folded on a phone). A
  finished deal opens at its first step with **Watch from start**; a deal still being played follows the latest step.
  Each deal shows **both plays at the table** (same cards, seats swapped). On a wide screen they sit side by side
  and move either **street by street** (both reach the flop, turn, river and result together; the one with fewer
  actions waits) or **Play 1, then Play 2**; on a narrow one a **Play 1 / Play 2** switch shows one, and playback
  plays Play 1 to its end, then Play 2, then the next deal. On a phone, pressing play folds the scoreboard to one
  line and brings the table up. The deal's controls stay on screen while scrolling. The table marks whose turn it is (**to act**)
  apart from the last action of each seat, and the winner lights up only at the end. With **Auto-next deal** on,
  playback goes on into the next deal. Below each table is a one-line-per-action log with the table talk;
  **why?** opens that decision's private reasoning, **Reasoning** opens all of them. When the game changed a move,
  the log says what the model asked for, what was played and why (*resized*, *asked again*, *auto-played*).
  Keys: <kbd>space</kbd> play or pause, <kbd>,</kbd> <kbd>.</kbd> step, <kbd>←</kbd> <kbd>→</kbd> deals,
  <kbd>r</kbd> reasoning. `?deal=12&step=4` in the address opens a deal at a step.
- **Live.** Whether a run is running, between rounds, stopped or finished comes from the runner's `tournament.json`
  (or, for older runs, from `plan.json` and `decisions.json`), never from which folders exist. A run with a table
  playing refreshes every 6 seconds, an unfinished but quiet one every minute. A refresh never interrupts a replay.

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
- One conversation of 50 deals per pairing decides nothing; a 50 bb/100 edge needs about 1,800 deals per pairing
  in conversations of 50 under moderate dependence (see "How sure the results are"). The first model runs had no
  preregistered design: they are descriptions of those matches.
