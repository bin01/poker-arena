# Poker Arena

An eval in which language models play **duplicate No-Limit Texas Hold'em** against each other (and against
three baseline bots), with a leaderboard on the Elo scale and a web viewer that replays every hand with each
player's reasoning and table talk.

It borrows its shape from [Clocktower](https://github.com/dwarkeshsp/clocktower): every seat is one ongoing
model conversation for a whole match, the dealer is plain seeded code, side-swapped pairs cancel the luck of the
deal, one Bradley–Terry fit turns results into ratings with bootstrap intervals, and every call is logged so any
result can be traced back to the hand and the reasoning behind it.

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
- **Ratings.** Deals are grouped into sessions of 10 deals (20 hands). A player wins a session if it is up by
  more than 1 big blind over it (closer is a tie). A player with fewer than 3 sessions is shown as provisional, with
  no interval. One Bradley–Terry fit over all sessions gives an Elo-scale rating (200 points ≈ winning 76% of
  sessions), centred on 1500, with 90% intervals from a bootstrap over sessions within each matchup. bb/100 (big
  blinds won per 100 hands, with a standard error) is reported next to it. Sessions rather than single deals,
  because most deals are tiny pots: counted one by one, a random bot that wins many small pots and loses whole
  stacks rates near the top.

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

### How many deals it takes

Even with duplicate deals, one deal swings a lot: in the first model runs, the result per deal had a spread of
about 350 bb/100. So the number of deals a pairing needs to show who is better depends on how far apart the two are
(95% confidence, 80% chance of catching the edge):

| real gap | deals per pairing |
|---|---|
| 100 bb/100 (a mismatch) | ~100 |
| 50 bb/100 (clearly better) | ~380 |
| 20 bb/100 (solidly better) | ~2,400 |
| 10 bb/100 (slightly better) | ~9,500 |

Rather than guess, play until decided:

```sh
python -m poker_arena tournament plans/pilot.json --until-decided 1000
```

The plan's `deals` becomes a round. After each round, a pairing whose edge is settled stops; the rest play another
round on new deals (the same new deals for every pairing still playing), up to 1,000 deals. Rounds go in their own
folders (`a__vs__b__r2`, ...), each with fresh memories, and pool in the ratings. A pairing is **settled** when its
edge is 2.7 standard errors from even, not 1.96: checking after every round gives a false winner more chances,
and 2.7 keeps the chance under 5% for up to 25 checks (a simulated Pocock boundary). Running it again on a finished
plan with `--until-decided` continues from the rounds already played, so a fixed-length run can be extended.

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
at another level is a different player. Sessions never span runs. Deals only pool when they were played under
the same game settings (blinds and stacks change what bb/100 means); runs under other settings are listed as
left out. Deals between two entries of the same model and level don't count. Runs from different code versions do
pool, and the leaderboard says so. The viewer shows it as **All runs** in the run picker.

## The viewer

`python -m poker_arena view runs` serves every run under `runs/` at http://127.0.0.1:8800, with a picker for the
run (a tournament's `out` folder, or a `play --out`). It follows running matches live.

The look is a card room: green felt, ivory cards, chips and brass. Players are shown by their `label` from
`players.toml` (or a name made from their model), never by the CLI that runs them.

- **Standings.** Progress of the run (hands played, pace, time left), the top three as a hand of cards, then:
  - **Ratings**: Elo with its 90% interval on a shared axis, bb/100 with its 95% margin (coloured only when the
    margin excludes zero), sessions won–tied–lost. Only players with at least 3 sessions are ranked; the rest
    are listed with how far they are from a rating.
  - **Is it settled?**: for each pairing, the leader's edge with its margin, how close it is to settled, and
    roughly how many more deals that takes at the current gap (or "too close to call", meaning over 10,000 more).
    Above it, one line says whether the top player is the winner, and if not, what it still needs.
  - **Head to head**: bb/100 for every pair, shaded when the edge is larger than its margin; a cell opens the matchup.
  - **How they play**: VPIP, PFR, aggression after the flop, how often it goes to showdown and wins there, and
    how often it talks.
- **Matches.** Every matchup (playing, played, not started) with its score. A row of chips, one per deal,
  coloured by who won it over both plays and brighter the bigger the swing. Each deal shows **both plays side by
  side at the table** (same cards, seats swapped):
  hole cards, board, every action in big blinds, the table talk, each decision's private reasoning (and the
  CLI's own thinking where it exposes it), and each player's notes on its opponent. A move the model didn't make
  (no playable answer, so the game checked or folded for it) and an illegal bet size are marked. **Replay** (or <kbd>space</kbd>) deals both plays again step by step. <kbd>←</kbd> <kbd>→</kbd> move between deals,
  <kbd>r</kbd> hides the reasoning.

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
- 50 deals per matchup separates large skill gaps; close models need several hundred.
