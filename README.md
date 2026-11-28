# Poker Arena

Duplicate No-Limit Texas Hold'em evaluation benchmark for LLMs, run through your existing subscription CLIs (`claude`, `codex`, `agy`) with preregistered statistical verdicts, GTO/AIVAT analysis, 4-channel adversarial forensics, 6-axis scorecards, and a live hand-replay viewer.

> **Coding agents & contributors:** start with [**AGENTS.md**](AGENTS.md) for the module map, invariants, test commands, and formal verification workflow.

---

## Quick start

```sh
git clone https://github.com/bin01/poker-arena && cd poker-arena
python3 -m venv .venv && . .venv/bin/activate      # Python 3.11+ (only dep: numpy)
pip install -e '.[dev]'

# Run tests
python -m pytest -q

# Instant free bot demo + web viewer (http://127.0.0.1:8800)
python -m poker_arena play rule station --deals 50 --out runs/demo
python -m poker_arena view runs/demo

# Verify your signed-in model CLIs (configured in players.toml)
python -m poker_arena doctor

# Plan & run a preregistered two-track tournament (resumable on crash/stop)
python -m poker_arena plan reference rule station --out runs/cup --write plans/cup.json
python -m poker_arena tournament plans/cup.json
python -m poker_arena ratings runs/cup
```

---

## Core concepts

- **Duplicate play:** Every matchup plays each seeded deal twice (`a-first` and `b-first`, seats swapped, 100 bb stacks resetting every hand) to cut deal luck.
- **Two tracks:** The **official** track uses 1 deal per fresh conversation (independent observations, assumption-free anytime-valid betting confidence sequence; decides ranks/leaders, assuming each name serves a **stable model** throughout the run and flagged if drift is detected). The **adaptation** track keeps 50-deal conversations (exploratory, model-based, never mixed into official standings).
- **Raw-first integrity:** Every rating, EV-loss metric, forensic alert, and scorecard grade is re-derived directly from immutable match logs (`hands.jsonl`, `events.jsonl`, `calls.jsonl`, `raw.jsonl`, `config.json`, `state.json`). Every counted hand is replayed from its seed and checked for tool use and model identity.
- **Deep evaluation (v2):** Built-in NumPy DCFR solver (plus optional `TexasSolver` / `Slumbot` adapters), unbiased AIVAT variance reduction, Nash averaging & α-Rank meta-game rankings, 4-channel adversarial forensics (sandbox, prompt injection, parser fuzzing, covert intent), suit/seat invariance tests, and a calibrated 6-axis character sheet.
- **Formally verified runner:** Per-hand commits, crash recovery, session rebuilds, and duplicate-pair semaphore concurrency are model-checked with TLA+/TLC (`verification/run_tlc.sh --check-each`).

---

## CLI commands

| Command | Purpose |
|---|---|
| `play A B --deals N --out DIR` | Run one duplicate pair (`a-first` + `b-first`) |
| `plan PLAYERS... --out DIR --write FILE` | Generate a preregistered tournament plan with power/cost estimates |
| `tournament PLAN.json [--concurrency N]` | Run or resume all matchups in a plan |
| `ratings RUN_DIR` | Compute official verdicts, descriptive bb/100 strengths, and eligibility |
| `view RUNS_DIR [--port 8800]` | Launch live web viewer (Standings, Scorecards, Watch, Fundamentals, Planner) |
| `scorecard RUN_DIR` | Build 6-axis character sheets with bootstrap bands and reliability |
| `forensics RUN_DIR` | Run 4-channel adversarial audit + interface-vs-strategy EV split |
| `invariance PLAYER_ID` | Test suit- and seat-relabeling invariance on identical spots |
| `quiz PLAYERS... --out DIR` | Score poker fundamentals across 5 categories of computed spots |
| `history RUNS_DIR` | Update `runs/arena.db` SQLite index and print all-time descriptive standings |
| `status RUNS_DIR [--health]` | Terminal progress, ETA, and per-model latency/retry health |
| `export RUNS_DIR --out SITE_DIR` | Export a self-contained static viewer site |
| `doctor` | Smoke-test configured model CLIs (response, JSON schema, session memory) |
| `solver-check` | Check availability of built-in DCFR, optional TexasSolver, and Slumbot |

---

## Backends (`players.toml`)

Players are defined as `[players.<id>]` entries in [`players.toml`](players.toml) (`backend`, exact `model`, and `effort`, defaulting to `high`, or `bot = "reference"|"rule"|"station"|"random"|"gto"`).

| Backend | CLI / Transport | Conversation state |
|---|---|---|
| `claude` | `claude -p` (Claude Pro/Max) | Stateful (`--resume`), tools off, isolated empty workdir |
| `codex` | `codex exec` (ChatGPT Plus/Pro) | Stateful (`exec resume`), read-only sandbox, isolated empty workdir |
| `agy` | `agy -p` (Google Antigravity) | Stateful (`--conversation`), tool-less agent, isolated empty workdir |
| `gemini` | `gemini -p` | Stateless (re-sends committed transcript) |
| `command` | Any stdin CLI (e.g. `ollama`, `pi`) | Stateless |
| `openai_compat` | OpenAI-compatible HTTP API | Stateless |

---

## Repository map & docs

```text
poker-arena/
├── AGENTS.md                       # Agent entry point: module map, invariants, workflows
├── players.toml                    # Model & baseline bot roster
├── plans/                          # Ready-made tournament plans (smoke, pilot, tracks-demo)
├── poker_arena/                    # Core Python package (engine, runner, ratings, solver, viewer)
├── tests/                          # pytest suite (578+ tests)
├── verification/                   # TLA+/TLC specs + pinned run_tlc.sh runner
├── .agents/skills/                 # Project-local agent skills (verify-runner-protocol)
└── docs/
    ├── methodology.md              # Statistical tests, bounds, calibration, power
    ├── v2-architecture.md          # v2 module contracts (solver, AIVAT, meta-rank, forensics, scorecard)
    └── viewer-and-records.md       # Match log schemas, viewer features, history DB, CLI caveats
```
