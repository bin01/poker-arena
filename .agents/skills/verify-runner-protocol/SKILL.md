---
name: verify-runner-protocol
description: Formally verify Poker Arena's per-hand commit, crash-recovery, CLI session rebuild, and duplicate-pairing semaphore concurrency protocol with TLA+/TLC. Use before or after editing poker_arena/runner.py (run_match, _has_uncommitted_logs, _reconcile, _write, verify_match, committed_records, run_duplicate, run_plan semaphores), poker_arena/backends.py (Backend.state, restore, rebuild, forget_last, send), poker_arena/agents.py (ModelAgent.state, restore, act fallback/tool rejection), or poker_arena/ratings.py (_is_modern_record, _hands, read_conversation commit boundary).
---

# Verify Runner Protocol (`poker_arena`)

Formal TLA+/TLC verification for Poker Arena's per-hand durability and concurrency protocols:

1. **`verification/RunnerCommitRecovery.tla`** — per-hand commit ordering, crash recovery (`_has_uncommitted_logs` + pre-reconcile rebuild persistence + `_reconcile`), stateful CLI session rebuild/history preservation, and `ratings._hands` / `runner.verify_match` reader visibility.
2. **`verification/DuplicateConcurrency.tla`** — duplicate-pairing semaphore acquisition order, CPython `asyncio.Semaphore` FIFO wait queues, `on_start()` error handling, and sibling/external task cancellation.

**Last verified sync commit:** `36b41c1b773abcf93f176848f15063de22906b41`

> Only advance the sync commit above after you inspect the diff in the covered code, update the `.tla` model (if behavior changed intentionally), re-run the checks, and confirm that every modeled action and atomicity boundary matches the Python code at the new commit.

---

## 1. Check whether a change touches verified code

Run:

```bash
git diff 36b41c1b773abcf93f176848f15063de22906b41 -- \
  poker_arena/runner.py \
  poker_arena/backends.py \
  poker_arena/agents.py \
  poker_arena/ratings.py
```

If any function below changed, this verification applies:

| Spec | File | Functions & line ranges (at `36b41c1`) | What is modeled |
|---|---|---|---|
| `RunnerCommitRecovery` | `poker_arena/runner.py` | `_write` (`215-222`), `_has_uncommitted_logs` (`235-250`), `_reconcile` (`253-281`), `committed_records` (`300-325`), `verify_match` (`562-611`), `run_match` (`678-906`) | `config.json` creation, `state.json` load/checkpoint, pre-reconcile rebuild persistence into `state.json`, per-file log truncation, `hands.jsonl` append before `state.json` replace, `result.json` short-circuit |
| `RunnerCommitRecovery` | `poker_arena/backends.py` | `Backend.state` (`300-302`), `restore` (`304-310`), `rebuild` (`312-317`), `forget_last` (`319-324`), `send` (`352-412`) | Stateful CLI session lifecycle, dirty session tracking on uncommitted turns, persisted `rebuilding` flag across checkpoints and restores |
| `RunnerCommitRecovery` | `poker_arena/agents.py` | `ModelAgent.state`/`restore` (`122-128`), `act` (`135-191`) | Persisted `agents[i]["backend"]` state and `tool_use`/`unverified` rejection triggering `forget_last()` before fallback |
| `RunnerCommitRecovery` | `poker_arena/ratings.py` | `_is_modern_record` (`331-335`), `_hands` (`338-360`), `read_conversation` (`609-665`) | Fail-closed `0` committed hands on modern (`record_version >= 2`) folders before first `state.json` versus checkpoint-bounded reads once `state.json` exists |
| `DuplicateConcurrency` | `poker_arena/runner.py` | `run_duplicate` / `one` (`915-947`), `Lifecycle.pairing` (`1319-1323`), `run_plan` (`1428`, `1454-1458`, `1495-1502`) | Sorted multi-CLI semaphore acquisition (`["agy", "claude"]`), `acquired` tracking inside `try/finally` covering acquires and synchronous `on_start()`, FIFO wait queues, sibling and external cancellation |

---

## 2. Properties verified and user harm

### `verification/RunnerCommitRecovery.tla`

- **`CommitSafety`** — *Property:* For a modern (`record_version >= 2`) match, `ratings._hands` admits only hands strictly below `state.json["hands_committed"]` (and none before the first `state.json`). *Harm if broken:* An external reader or viewer admits uncommitted hand `0` that the runner itself has not committed and will discard and replay on resume.
- **`NoDirtySessionReuse`** — *Property:* After any sequence of crashes (including a crash during recovery), no resumed turn executes on a stateful CLI session that still holds a dropped uncommitted turn. *Harm if broken:* A model plays a replayed hand with leaked memory of cards/actions from an aborted run, silently contaminating the match record.
- **`NoHistoryLossOnRestore`** — *Property:* Whenever a seat with prior committed turns opens a fresh CLI session (`session = None`), it replays its committed transcript (`rebuilding = True`). *Harm if broken:* After a tool-use/unverified fallback hand followed by a restart, the seat silently loses all memory of prior hands.
- **`CommittedLogMatchesCheckpoint`** — *Property:* Whenever the runner is ready to start a hand or finish a match, `hands.jsonl` holds every checkpointed hand ID `0..(hands_committed - 1)`. *Harm if broken:* Resuming a match crashes with a damaged-record error or drops completed hands.
- **`NoDeadlock`** — *Property:* The commit/recovery state machine always has an enabled step until the match finishes. *Harm if broken:* A match gets stuck and cannot resume to completion.

### `verification/DuplicateConcurrency.tla`

- **`BoundedConcurrency`** — *Property:* Available permits for each CLI semaphore stay within `0..CAP` (never negative, never over-released). *Harm if broken:* Semaphore corruption breaks per-CLI concurrency caps or causes crashes.
- **`PermitConservation`** — *Property:* For every CLI semaphore, available permits plus permits held by currently live orders always equal `CAP`. *Harm if broken:* Failed or cancelled duplicate orders permanently leak concurrency slots until the tournament stalls.
- **`NoPermitLeakOnCancel`** — *Property:* Any cancelled duplicate order releases every semaphore permit it acquired. *Harm if broken:* Cancelling a sibling or replicate while it waits on its second CLI semaphore leaks its first CLI semaphore permanently.
- **`NoPermitLeakOnStartError`** — *Property:* Any duplicate order whose `on_start()` callback raises releases every semaphore permit it acquired. *Harm if broken:* A transient write error in `Lifecycle.pairing` leaks both CLI semaphores permanently.
- **`NoDeadlock`** — *Property:* No execution reaches a state where unfinished orders are permanently blocked waiting for leaked or circularly held permits. *Harm if broken:* Unrelated tournament pairings sharing a CLI semaphore hang forever.

---

## 3. How to run the verification

Run the deterministic, SHA-256-pinned TLC (`v1.7.4`) runner from the repo root:

```bash
# Per-property PASS/FAIL summary across every invariant + deadlock (~3s)
verification/run_tlc.sh --check-each

# Standard TLC check across both specs (~1s)
verification/run_tlc.sh
```

You can also target a single spec:

```bash
verification/run_tlc.sh verification/RunnerCommitRecovery.tla
verification/run_tlc.sh verification/DuplicateConcurrency.tla
```

**Always read every TLA+ result with the pstack `/statechart` skill.** Invoke `/statechart` on `verification/RunnerCommitRecovery.tla` and `verification/DuplicateConcurrency.tla` to inspect any broken invariant as its shortest counterexample path in plain words, or to inspect the verified state graph when all invariants pass.

---

## 4. Baseline status at sync commit `36b41c1`

Every invariant and deadlock check passes at sync commit `36b41c1b773abcf93f176848f15063de22906b41`:

| Spec | Reachable states | Properties | Status at `36b41c1` |
|---|---|---|---|
| `RunnerCommitRecovery` | 879 distinct states | `TypeOK`, `CommittedLogMatchesCheckpoint`, `CommitSafety`, `NoDirtySessionReuse`, `NoHistoryLossOnRestore`, `NoDeadlock` | **ALL PASS** |
| `DuplicateConcurrency` | 693 distinct states | `TypeOK`, `BoundedConcurrency`, `PermitConservation`, `NoPermitLeakOnCancel`, `NoPermitLeakOnStartError`, `NoDeadlock` | **ALL PASS** |

Any `FAIL` after a code change is either a **new regression** or **model drift** from an intentional protocol change.

---

## 5. Telling a real bug from a modeling error, and updating the models

When an invariant fails after a code change:

1. **Read the shortest path with `/statechart`.** Translate each TLA+ action name into the exact Python lines cited in the spec header.
2. **Walk the trace against the real Python code:**
   - Check atomicity: does CPython actually yield or allow a crash between step `i` and step `i+1`? (For example, an uncontended `await sem.acquire()` does not yield; synchronous file writes in `run_match` do not yield to sibling asyncio tasks, though an external process or `SIGKILL` can observe the disk between two file operations.)
   - If the trace is **executable on the real Python code**, it is a **real bug** — fix the Python code (never weaken an invariant to make TLC pass).
   - If a step in the trace is **impossible in the current Python code** because the protocol changed intentionally, it is **model drift** — update the `.tla` action to match the new code, re-run `verification/run_tlc.sh --check-each` and `/statechart`, and update the coverage table and sync commit.
3. **Verify the guard can still fail:** After updating a model so a property passes, temporarily test that breaking the guarded behavior makes the property fail, then revert the test.

---

## 6. Bounds, assumptions, and what is NOT covered

- **`RunnerCommitRecovery` bounds:** `MAX_HANDS = 3`, `MAX_CRASHES = 2`, one stateful CLI seat whose `workdir` survives crashes. Assumes `_write()` (`temp + fsync + os.replace`) is atomic per file. Does **not** model torn partial JSON lines, calls/raw-only uncommitted rows, multi-decision turns within a single hand, duplicate rows in `hands.jsonl`, `verify_match` replay checks, duplicate `interrupted.jsonl` rows on a crash mid-`_reconcile`, detached CLI subprocesses surviving `SIGKILL`, or `ratings.read_conversation` pair/replay filtering beyond `_hands`.
- **`DuplicateConcurrency` bounds:** `CAP = 2` (default `--concurrency 2`), two semaphores `{"agy", "claude"}`, one duplicate pairing (`a_first`, `b_first`) plus two single-CLI peers (`peer1`, `peer2`). Does **not** model `RateGate` hold/wait loops (`backends.py:147-164`), `on_start()` errors on peer tasks, or `CAP > 2` (though permit conservation scales identically for any `CAP >= 1`).
- **Not covered by either spec:** `ratings.verify_play` caching (`ratings.py:535-574`, covered by unit tests in `tests/test_runner_protocol_fixes.py`), game rules/hand evaluation (`holdem.py`, `cards.py`), and statistical adjudication (`adjudicate`).
