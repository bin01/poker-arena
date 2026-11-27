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

Players live in [`players.toml`](players.toml): a `backend` and optional `model` per model, or a `bot`.

| backend | tool | plan | memory |
|---|---|---|---|
| `claude` | Claude Code, `claude -p` | Claude Pro / Max | the CLI keeps the session (`--resume`); tools off |
| `codex` | Codex CLI, `codex exec` | ChatGPT Plus / Pro | the CLI keeps the thread (`exec resume`); read-only sandbox |
| `agy` | Google Antigravity CLI, `agy -p` | Google account (Gemini) | the CLI keeps the conversation (`--conversation`) |
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

## The viewer

`python -m poker_arena view runs/pilot` serves:

- **Leaderboard**: Elo with intervals, bb/100, sessions won, and a head-to-head bb/100 matrix.
- **Matches**: every matchup with a strip of its deals coloured by who won the deal over both plays. Pick a deal
  to see **both plays of it side by side** (same cards, seats swapped): hole cards, board, every action, the
  table talk, and each player's private reasoning and notes.

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

- CLI backends add their own system prompts and limits (Codex's default prompt is about 21K tokens, agy's about 15K), and expose
  reasoning unevenly (`claude -p` shows no thinking, `codex` shows reasoning summaries when asked, `agy` none; agy also keeps its own tools loaded, with anything needing approval refused),
  which is why every reply carries its own `reasoning` field.
- Elo needs whole sessions (10 deals each), so a matchup needs at least 5 deals to count and 10+ to mean much.
- Results from different backends of the same model (CLI vs API) shouldn't be mixed on one leaderboard.
- 50 deals per matchup separates large skill gaps; close models need several hundred.
