# AGENTS.md — Coding Agent Guide for `poker-arena`

Read this first before navigating or editing the codebase. Keep edits minimal, verifiable, and aligned with the invariants below.

---

## 1. Repository & Module Map

```text
poker_arena/
├── games/
│   ├── __init__.py       # Game registry (get_game)
│   └── holdem.py         # Seeded HUNL engine, Episode state machine, Hand replay verifier
├── cards.py              # Fast 5/7-card hand evaluator & Monte Carlo equity
├── backends.py           # Supported CLI adapters (claude, codex, agy), RateGate, session/rebuild state
├── agents.py             # ModelAgent (JSON parse, retries, tool_use/unverified rejection) + baseline bots
├── runner.py             # run_match, _has_uncommitted_logs, _reconcile, verify_match, run_duplicate, run_plan, Lifecycle
├── ratings.py            # Shared reader (read_conversation, _hands, verify_play), betting CS, adjudicate, eligibility, quiz audit
├── history.py            # SQLite archive (runs/arena.db) indexing verified deals across runs
├── stats.py              # Style statistics (VPIP, PFR, AF, showdown, provisional counts)
├── quiz.py               # Poker fundamentals spot bank & runner
├── solver.py             # Built-in NumPy DCFR solver, TexasSolver & Slumbot adapters, calibration/exploiter bots
├── aivat.py              # Unbiased AIVAT chance-node control variates & All-In EV adjustment
├── meta_rank.py          # Max-entropy Nash averaging, α-Rank, empirical dominance cycle detection
├── forensics.py          # 4-channel adversarial audit (sandbox, prompt injection, parser fuzzing, covert intent) + EV split
├── invariance.py         # Suit- and seat-relabeling invariance test harness
├── scorecard.py          # 6-axis character sheet (20-pip bars, bootstrap CIs, split-half reliability, ablation bots)
├── server.py             # HTTP server, off-path background caching (runs/.viewer-cache/), static export()
├── cli.py                # CLI entry point (python -m poker_arena <cmd>)
└── viewer/               # Zero-build vanilla JS/CSS web UI (index.html, app.js, core.js, table.js, style.css)

tests/                    # pytest suite (578+ tests; uses tests/conftest.py for sandbox-safe temp dirs)
verification/             # TLA+/TLC formal models (RunnerCommitRecovery, DuplicateConcurrency) + run_tlc.sh
.agents/skills/           # Project-local skills (verify-runner-protocol)
docs/                     # methodology.md, v2-architecture.md, viewer-and-records.md, archive/
plans/                    # Example tournament plan JSONs (smoke.json, pilot.json, tracks-demo.json)
players.toml              # Configured model and bot players
```

---

## 2. Where to Look First by Task

| Task area | Primary files | Must-check tests / verifiers |
|---|---|---|
| Match commits, crash recovery, CLI session rebuilds, duplicate concurrency | `runner.py`, `backends.py`, `agents.py`, `ratings.py` (`_hands`, `read_conversation`) | **Skill:** `.agents/skills/verify-runner-protocol/SKILL.md`<br>`verification/run_tlc.sh --check-each`<br>`pytest tests/test_runner*.py tests/test_backends.py tests/test_agents.py` |
| Statistical verdicts, eligibility, cache, history | `ratings.py`, `history.py`, `docs/methodology.md` | `pytest tests/test_ratings.py tests/test_evidence.py tests/test_publication.py tests/test_calibration.py tests/test_history.py` |
| Solver, AIVAT, meta-ranking | `solver.py`, `aivat.py`, `meta_rank.py`, `docs/v2-architecture.md` | `pytest tests/test_solver_aivat.py tests/test_verification_suite.py` |
| Forensics, invariance, 6-axis scorecard | `forensics.py`, `invariance.py`, `scorecard.py` | `pytest tests/test_forensics_invariance.py tests/test_scorecard.py tests/test_harness_readiness.py` |
| Web viewer, API endpoints, static export | `server.py`, `viewer/*` | `pytest tests/test_viewer.py tests/test_v2_ui.py tests/test_review_r*.py` |
| Docs (`README.md`, `docs/methodology.md`) | `README.md`, `docs/methodology.md` | `pytest tests/test_review_r9.py` (checks required wording such as `stable model` & `no drift detected`) |

---

## 3. Core Invariants (Do Not Break)

1. **Zero new mandatory Python dependencies:** Stdlib + `numpy` only (`pyproject.toml`). External solvers (`TexasSolver`, `Slumbot`) must remain optional with graceful fallback.
2. **Fail-closed commit boundary:** On modern matches (`record_version >= 2`), `ratings._hands` and `runner.verify_match` must return `0` committed hands if `state.json` is absent, corrupt, or has a non-int/boolean `hands_committed`. Pre-v2 legacy records without `record_version >= 2` retain legacy fallback.
3. **Write-ahead session rebuild:** On resume, `run_match` must detect uncommitted/torn logs (`_has_uncommitted_logs`) and atomically persist `session = None, rebuilding = True` to `state.json` **before** `_reconcile` truncates logs on disk. `Backend.state()` / `restore()` must persist `rebuilding`.
4. **Exception- & cancellation-safe semaphores:** In `runner.run_duplicate`, every `await sem.acquire()` and `on_start()` runs inside `try/finally`, tracking `acquired` and releasing in reverse order. Semaphores are always acquired in `sorted(...)` backend order (`runner.py`).
5. **Raw-first derivation & single reader:** Ratings, style stats, scorecards, forensics, and viewer totals must derive from verified raw logs via `ratings.read_conversation` / `verified_hands` — never trust unverified summary fields.
6. **Model-first UI & honest uncertainty:** Viewer leads with model + reasoning level (never CLI names), distinguishes official anytime-valid intervals from fixed-time descriptive ranges, and never calls exploratory/adaptation results "wins" or "Decided".
7. **Off-path server caching:** Expensive computations (`ratings`, `scorecard`) run off the HTTP request thread and cache in `runs/.viewer-cache/`.

---

## 4. Verification Commands

Use `.venv/bin/python` (or activate `.venv`):

```sh
# Fast targeted tests
.venv/bin/pytest -q tests/test_runner_protocol_fixes.py tests/test_runner_recovery.py

# Full test suite (~4 min due to calibration/solver simulations)
.venv/bin/pytest -q

# In read-only /tmp sandboxes (uses tests/.tmp automatically via tests/conftest.py)
.venv/bin/pytest -q -p no:cacheprovider

# Formal verification (pinned TLA+/TLC v1.7.4, ~3s)
verification/run_tlc.sh --check-each

# Local health checks
.venv/bin/python -m poker_arena solver-check
.venv/bin/python -m poker_arena doctor
```

---

## 5. Project-Local Skills

- **`.agents/skills/verify-runner-protocol/SKILL.md`** — Formal TLA+/TLC verification for `runner.py`, `backends.py`, `agents.py`, and `ratings.py`. Follow it before and after modifying commit ordering, crash recovery, CLI session rebuilds, or duplicate-pairing semaphore concurrency. Only advance its sync commit after confirming the models match the new code and all checks pass.

---

## 6. Repo Gotchas

- **Python `bool` is an `int`:** Always guard JSON integer checks with `isinstance(x, int) and not isinstance(x, bool)` (especially on `hands_committed` and `record_version`).
- **In-memory verification cache:** `ratings._VERIFIED` caches `verify_play` by `(path, _stamp(d), cfg_sig)` and per-hand payload signature; tests mutating files within the same filesystem mtime tick should call `ratings._VERIFIED.clear()`.
- **Gitignored local tools:** `.tools/` holds optional local binaries (Eclipse Temurin JRE, `tla2tools.jar`, `TexasSolver`); never commit `.tools/` or `runs/`.
- **Doc wording assertions:** `tests/test_review_r9.py` asserts exact strings in `README.md` (`"stable model"`) and `docs/methodology.md` (`"no drift detected"`). Run it after doc edits.
