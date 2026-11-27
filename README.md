# Poker Arena

An eval in which language models play **duplicate No-Limit Texas Hold'em** against each other (and against
three baseline bots), with standings in bb/100 that say how sure they are, and a web viewer that replays every
hand with each player's reasoning and table talk.

It borrows its shape from [Clocktower](https://github.com/dwarkeshsp/clocktower): every seat is one ongoing
model conversation for a whole match, the dealer is plain seeded code, side-swapped pairs cancel the luck of the
deal, one paired-comparison fit turns results into strengths with bootstrap intervals, and every call is logged so
any result can be traced back to the hand and the reasoning behind it.

Models run through the labs' **official command-line tools signed in with your own subscription** (Claude
Code, Codex, Antigravity), so a tournament costs nothing beyond the plans you already pay for.

## How it works

- **Game.** Heads-up NLHE, blinds 1/2, both stacks reset to 200 (100 big blinds) every hand, the button moving
  every hand. Each match is `deals` hands (50 by default).
- **Duplicate.** Every matchup is two matches on the same deals with the seats swapped. On deal *k* each
  player gets, in turn, both seats' cards against the other player, so a player can't win by being dealt
  better cards. Every matchup in a tournament uses the same deals, too.
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
- **Ratings.** Chips decide, as the models are told. A player's **strength** is its edge in bb/100 against the
  average opponent of its group, from a least-squares fit over duplicate deals; the difference of two strengths
  is the predicted edge between them. Intervals are 95%, from a block bootstrap that keeps runs of 10 deals
  together (a model remembers earlier hands). A rank is shown only when a player is separated from every other
  one after Holm's adjustment, and there is a leader only when one player is separated above all the others;
  otherwise the standings say "No established leader yet". Session Elo (share of 10-deal sessions won) is shown
  as a secondary statistic and never decides anything. [docs/methodology.md](docs/methodology.md) has the
  method, the calibration results and the limits.

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

Even with duplicate deals, one deal swings a lot: in the first model runs the result per deal had a spread of
about 340 bb/100. So the number of deals a pairing needs depends on how far apart the two players really are.
For an 80% chance of catching the edge, with the default design and 3 pairings:

| real edge | deals per pairing |
|---|---|
| 100 bb/100 (a mismatch) | ~170 |
| 50 bb/100 (clearly better) | ~680 |
| 20 bb/100 (solidly better) | ~4,200 |

Each pairing's verdict reports this table from its own observed spread (`power.deals_for_80pct`). It is a
planning number, not a forecast for that pairing.

A plan carries a sequential **design**, by default `{"alpha": 0.05, "round_deals": 50, "max_rounds": 6,
"block": 10}`: each pairing plays rounds of 50 deals, and after each round its result is checked against a
stopping bar. Checking after every round gives a false winner more chances, so the bar is calibrated by
simulation for exactly this schedule (a group-sequential design), and the 5% is split over the pairings
(Bonferroni), so that with no real edge the chance of naming any false winner in the whole tournament stays
under 5%. `tests/test_calibration.py` checks that on every test run for normal, heavy-tailed and poker-like
results. A pairing ends with one of these verdicts:

- **a is better / b is better**: the edge crossed the bar.
- **even**: the edge is shown to be within ±15 bb/100 (an equivalence test). This needs thousands of deals.
- **undecided after every planned look**: the data can't tell the two apart at this size. That is *not*
  evidence that they are equally good.
- **undecided so far** (more rounds to come) or **too few deals to judge**.

Repeated deals (the same pair on the same cards in another run) are averaged, not counted twice. Players with
no exact model version recorded (no `model`, or a bare alias like `sonnet`) are not pooled into all-time
standings. Players that never met, directly or through others, are ranked in separate groups.

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

The all-time leaderboard rates a player by **what it is: its model and reasoning level** (or a bot and its
settings), not by its id in one run. Two ids with the same model and level pool their results, and the same model
at another level is a different player. A player without an exact model version (no `model`, or a bare alias
like `sonnet`) is listed but not rated, since two runs of it may not be the same model. A deal the same pair
already played in another run (same seeds, same settings) is averaged with it rather than counted again. Deals only pool when they were played under
the same game settings (blinds and stacks change what bb/100 means); runs under other settings are listed as
left out. Deals between two entries of the same model and level don't count. Runs from different code versions do
pool, and the leaderboard says so. The viewer shows it as **All runs** in the run picker.

## The viewer

`python -m poker_arena view runs` serves every run under `runs/` at http://127.0.0.1:8800, with a picker for the
run (a tournament's `out` folder, or a `play --out`). It follows running matches live.

The look is a card room: green felt, ivory cards, chips and brass. Players are always shown as a model and its
reasoning level (e.g. "GPT-6 Luna · High", from `label` in `players.toml` or made from `model` and `effort`), never
by the CLI that runs them. A player whose model wasn't recorded shows as, say, "GPT (model not recorded)".

- **Standings.** One line on what the page is, a **Watch the games** button, and a short *How to read the
  standings* legend (open on the first visit, remembered once closed). Then:
  - **Who is winning?** The established leader as an ace, only when a model beats every other model by more than
    the noise. Otherwise a face-down card: *No established leader yet*, with who is ahead on points and how many
    deals it would take to tell a 50 bb/100 edge apart.
  - **Standings**: strength in bb/100 against the average opponent, with its 95% range on a shared axis around
    zero. A rank number appears only when a model is clearly separated from the ones below it; otherwise `=`, and
    "level with ..." names who it can't yet be told apart from. Net big blinds and hands, and session Elo (sessions
    won–tied–lost and its interval) as a secondary number. Players still waiting to play are listed. When some
    players have never been connected by games (the all-runs view), each group gets its own table. Players whose
    exact model is unknown are shown under **Not rated**.
  - **Matchups**: each pair's edge with its 95% range and the verdict of the sequential test (*wins*, *even*,
    *undecided*, *no winner*, *too early*), how many of its planned checks it has used, the deals needed to detect
    a 20/50/100 bb/100 edge, and a link to watch it.
  - **Head to head** (three or more players): bb/100 for every pair, shaded when the range excludes zero.
  - **How they play**: VPIP, PFR, aggression after the flop, how often it goes to showdown and wins there
    outright (split pots apart), and how often it talks.
- **Watch.** Every matchup (playing, played, not started). The header leads with the net big blinds over both
  plays and bb/100, deals won as a second line, and the matchup's verdict. A row of chips, one per deal, coloured
  by who won it over both plays. Each deal shows **both plays side by side at the table** (same cards, seats
  swapped): stacks, the chips each seat has put in on this street, the player to act, board and pot all move step
  by step, and the winner lights up only at the end. Below each table is a one-line-per-action log with the table
  talk; **why?** opens that decision's private reasoning (and the CLI's own thinking where it exposes it),
  **Reasoning** opens all of them. When the game changed a move, the log says what the model asked for, what was
  played and why (*resized*, *asked again*, *auto-played*). Each model's notes on its opponent are folded below.
  Keys: <kbd>space</kbd> play or pause, <kbd>,</kbd> <kbd>.</kbd> step, <kbd>←</kbd> <kbd>→</kbd> deals,
  <kbd>r</kbd> reasoning. `?deal=12&step=4` in the address opens a deal at a step.
- **Live.** A run with a table playing refreshes every 6 seconds, an unfinished but quiet one every minute; the
  header shows the last action and when the page last checked. A refresh never interrupts a replay.

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
- 50 deals per pairing separates only large skill gaps; close models need several hundred to thousands (see
  "How sure the results are").
