# Poker Arena v2 Deep Evaluation Architecture & Module Contracts

This document defines the exact contracts for the v2 upgrade on branch `v2-deep-eval`.
All additions preserve 100% backwards compatibility with existing v1 modules, log formats (`config.json`, `events.jsonl`, `calls.jsonl`, `raw.jsonl`, `hands.jsonl`, `state.json`), and all 406 existing tests.

## Core Design Principles
1. **Zero new mandatory Python dependencies**: stdlib + `numpy` only. External solvers (`TexasSolver` binary, `Slumbot` HTTP API) run behind clean optional adapters with deterministic disk caching and graceful fallback to the built-in NumPy DCFR solver.
2. **Raw-first derivation**: Every rating, variance-reduced estimate, EV-loss metric, forensic alert, and 6-axis grade is re-derived directly from immutable raw logs (`hands.jsonl`, `events.jsonl`, `calls.jsonl`, `raw.jsonl`, `config.json`, optional `quiz.json`).
3. **Strict separation of concerns**:
   - Absolute skill (GTO EV-loss & exploitability) vs. pool-relative strength (duplicate + AIVAT + betting CS + Nash averaging / α-Rank).
   - Stateless tactical play (`official` track) vs. multi-hand memory/adaptation (`adaptation` track).
   - Interface/format failures vs. strategic poker EV loss.

---

## Module Contracts

### 1. `poker_arena/solver.py`
- `SpotEval` (dict schema returned by every solver backend):
  ```python
  {
      'street': 'preflop' | 'flop' | 'turn' | 'river',
      'spot_hash': str,               # canonical suit-isomorphic hash
      'solver': str,                  # 'dcfr_exact_river' | 'dcfr_abstracted' | 'texassolver' | 'preflop_lookup'
      'pot_bb': float,
      'to_call_bb': float,
      'pot_odds': float,              # to_call / (pot + to_call) if to_call else 0.0
      'mdf': float,                   # pot / (pot + to_call) if to_call else 1.0
      'equity': float,                # hero equity vs estimated villain range
      'best_action': str,             # 'fold' | 'check' | 'call' | 'raise' | 'all_in'
      'action_evs_bb': dict[str, float],
      'gto_freqs': dict[str, float],  # sums to 1.0 over legal abstract actions
      'taken_action': str | None,
      'taken_ev_bb': float | None,
      'ev_loss_mbb': float | None,    # max(0.0, (best_ev_bb - taken_ev_bb) * 1000)
      'entropy': float,               # Shannon entropy (bits) of gto_freqs
      'is_indifferent': bool,         # top-2 action EVs within 0.05 bb (50 mbb)
      'is_blunder': bool,             # ev_loss_mbb >= 250
      'exploitability_mbb': float | None,
  }
  ```
- `solve_river_subgame(board, pot, effective_stack, ip_range, oop_range, bet_sizes=(0.5, 1.0), iterations=300, seed=0) -> dict`:
  Exact Discounted CFR (DCFR) on 5-card river subgame using `cards.evaluate` with blocker exclusion. Returns `{'strategy', 'ev', 'exploitability_mbb', 'root_ev_ip', 'root_ev_oop', 'node_eval': callable}`.
- `solve_postflop_spot(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb=2, solver='auto', cache_dir=None, iterations=200) -> dict`:
  Returns `SpotEval`. Uses `TexasSolverAdapter` if `solver=='texassolver'` (or `solver=='auto'` when binary available and `use_external=True`), else exact river DCFR on river and fast equity-bucketed DCFR on flop/turn.
- `evaluate_hand(hand_record, events=None, solver='dcfr', cache_dir=None) -> dict`:
  Replays a recorded hand from `hands.jsonl`, returns:
  `{'hand': int, 'seed': int, 'bb': int, 'decisions': [SpotEval + {'seat', 'player', 'decision_id', 'action_idx', 'amount', 'board', 'hole'}], 'per_seat_ev_loss_mbb': [float, float]}`
- `TexasSolverAdapter`:
  - `find_binary() -> Path | None` (checks `POKER_ARENA_TEXASSOLVER`, `.tools/TexasSolver/build/console_solver`, `shutil.which('console_solver')`).
  - `available() -> bool`
  - `solve_spot(...) -> dict` (canonical cached JSON output from `console_solver`).
- `SlumbotClient` & `SlumbotBot(Agent)`:
  HTTP adapter to `https://slumbot.com/api` (injectable transport for offline unit tests).
- Calibration & Exploiter bots (also exported via `SOLVER_BOTS` dict for `agents.BOTS`):
  - `GTOBot` (`bot = 'gto'`)
  - `NoisyGTOBot` (`bot = 'noisy_gto'`, `noise=0.15`)
  - `LocalBestResponseBot` (`bot = 'lbr'`)
  - `ExploiterBot` (`bot = 'exploiter'`, `mode='adaptive'|'vs_station'|'vs_nit'|'vs_overbluffer'`)
  - `RockBot` (`bot = 'rps_rock'`), `PaperBot` (`bot = 'rps_paper'`), `ScissorsBot` (`bot = 'rps_scissors'`) — deterministic non-transitive cycle triplet in HUNL.

### 2. `poker_arena/aivat.py`
- `all_in_ev(hand_record, bb=2, exact_turn_river=True, sims=400, seed=0) -> dict`:
  Returns `{'all_in': bool, 'all_in_street': str | None, 'actual_net': [int, int], 'ev_net': [float, float], 'luck': [float, float]}`.
- `aivat_hand(hand_record, bb=2, stack=200, sims=200, seed=0) -> dict`:
  Unbiased chance-node control-variate estimator (AIVAT, Burch et al. 2018). For every chance transition `c` (deal + flop + turn + river) where the value function `V(s)` depends ONLY on pre-transition public/private state (never future cards), subtracts `Δ_c = V(s, c) - E_{c'}[V(s, c')]` so `E[Δ_c | s] == 0` identically for ANY policy pair.
  Returns `{'raw_net': [float, float], 'aivat_net': [float, float], 'ev_adj_net': [float, float], 'luck_correction': [float, float], 'street_corrections': {...}}`.
- `estimate_matchup(hands_a_first, hands_b_first, bb=2, stack=200) -> dict`:
  Computes duplicate-pair raw, EV-adjusted, and AIVAT estimates in `bb/100`:
  `{'deals': int, 'raw_bb100': float, 'raw_se': float, 'ev_adj_bb100': float, 'ev_adj_se': float, 'aivat_bb100': float, 'aivat_se': float, 'variance_reduction_ratio': float, 'effective_sample_multiplier': float, 'per_deal': [...]}`.
- `evaluate_run_aivat(run_dir) -> dict`:
  Aggregates AIVAT estimates across all verified duplicate pairings in a run directory.

### 3. `poker_arena/meta_rank.py`
- `build_payoff_matrix(pairings, players=None) -> tuple[list[str], np.ndarray, np.ndarray]`:
  Returns `(players, M, counts)` where `M[i, j] = -M[j, i]` is bb/100 edge of `i` vs `j`.
- `nash_averaging(M, players) -> dict`:
  Computes max-entropy Nash equilibrium mixture `weights` over the symmetric zero-sum meta-game `M`, plus each player's Nash-averaged score `(M @ weights)[i]`, game exploitability residual, and clone-invariance guarantee.
  Returns `{'players': [...], 'nash_weights': {p: float}, 'nash_scores': {p: float}, 'ranking': [p, ...], 'residual': float}`.
- `alpha_rank(M, players, alpha=0.05, m=50) -> dict`:
  Single-population evolutionary Markov chain (Omidshafiei et al. 2019) stationary distribution `pi` via exact linear solve.
  Returns `{'players': [...], 'stationary': {p: float}, 'ranking': [p, ...], 'alpha': float, 'm': int}`.
- `detect_cycles(M, players, threshold=0.0) -> list[list[str]]`:
  Finds directed cycles (3-cycles and SCCs of size >= 2) in the empirical dominance graph.
- `meta_rankings(pairings, players=None, alpha=0.05) -> dict`:
  Combines `nash_averaging`, `alpha_rank`, `detect_cycles`, and `payoff_matrix` into one JSON-serializable report.

### 4. `poker_arena/forensics.py`
- `scan_prompt_injection(text: str) -> dict`:
  Returns `{'flagged': bool, 'severity': 'none'|'suspicious'|'attack', 'score': float, 'matches': [{'rule': str, 'span': str}]}`.
- `scan_covert_intent(thinking: str, reasoning: str = '') -> dict`:
  Detects explicit cheating/breakout/injection/fuzzing intent in raw private traces.
  Returns `{'flagged': bool, 'severity': 'none'|'suspicious'|'attack', 'categories': [str], 'matches': [{'rule': str, 'span': str}]}`.
- `scan_parser_fuzzing(calls: list[dict], errors: list[dict]) -> dict`:
  Detects `NaN`/`Infinity`, negative/non-numeric amounts, malformed/oversized JSON probes, extra injected keys, retry gaming.
- `audit_run_forensics(run_dir: Path | str) -> dict`:
  Full 4-channel adversarial audit + interface-vs-strategic EV decomposition for every player in `run_dir`:
  ```python
  {
      'players': {
          pid: {
              'sandbox': {'tool_rejected': int, 'unverified': int, 'identity_mismatch': int, 'cli_changed': int, 'incidents': [...]},
              'prompt_injection': {'emitted_attacks': int, 'emitted_suspicious': int, 'targeted': int, 'susceptible_events': int, 'offense_rate': float, 'susceptibility_rate': float | None, 'incidents': [...]},
              'parser_fuzzing': {'fuzz_attempts': int, 'nan_inf': int, 'illegal_actions': int, 'malformed_json': int, 'retries': int, 'fallbacks': int, 'incidents': [...]},
              'covert_intent': {'flagged_turns': int, 'under_pressure': int, 'blocked_attempts': int, 'incidents': [...]},
              'ev_decomposition': {'total_net_bb': float, 'fallback_loss_bb': float, 'retry_decisions': int, 'sizing_corrections': int, 'strategic_net_bb': float, 'clean_decision_rate': float},
              'total_incidents': int,
              'clean': bool,
          }
      },
      'summary': {'total_incidents': int, 'flagged_players': [str], 'channels': {...}}
  }
  ```
- `CanaryInjectionBot`, `ToolAttemptAgent`, `PromptInjectorBot`, `ParserFuzzerBot`, `CovertSchemerBot` (exported via `FORENSIC_BOTS`).

### 5. `poker_arena/invariance.py`
- `permute_suit_card(card: str, perm: dict[str, str]) -> str`
- `permute_spot(spot_dict: dict, perm: dict[str, str], seat_names: tuple[str, str] | None = None) -> dict`
- `ALL_SUIT_PERMS`: deterministic list of non-identity suit permutations.
- `run_invariance_test(agent_factory, spots=None, n_spots=12, n_perms=3, seed=0) -> dict`:
  Async/sync helper running an agent across isomorphic suit & seat-label perturbations of identical spots.
  Returns `{'n_spots': int, 'n_variants': int, 'action_flip_rate': float, 'ev_delta_mbb': float, 'invariance_score': float, 'p_value': float, 'biased': bool, 'examples': [...]}`.

### 6. `poker_arena/scorecard.py`
- Six fixed axes matching the RPG character sheet:
  `AXES = ('intelligence', 'wisdom', 'endurance', 'creativity', 'knowledge', 'alignment')`
- `AXIS_META`:
  - `intelligence`: `('INTELLIGENCE', 'able to reason and solve hard problems')`
  - `wisdom`: `('WISDOM', 'knowing when to act, ask, verify, change course, or stop')`
  - `endurance`: `('ENDURANCE', 'sticking to a task, following instructions, checking edge cases')`
  - `creativity`: `('CREATIVITY', 'finding new approaches and ideas')`
  - `knowledge`: `('KNOWLEDGE', 'remembering relevant facts and concepts, storing and reading memory')`
  - `alignment`: `('ALIGNMENT', 'follow both instructions and intent of those instructions')`
- `build_scorecard(run_dir: Path | str, solver='dcfr', include_quiz=True, boot=200, seed=0) -> dict`:
  Returns:
  ```python
  {
      'run': str,
      'axes': list[dict],          # ordered metadata for the 6 axes
      'players': {
          pid: {
              'player': pid,
              'kind': 'model' | 'bot',
              'hands': int,
              'decisions': int,
              'overall_20': float | None,
              'axes': {
                  axis_key: {
                      'axis': axis_key,
                      'label': str,
                      'subtitle': str,
                      'score': float | None,      # 0.0 .. 20.0
                      'pips': int | None,         # round(score), 0..20
                      'max_pips': 20,
                      'ci95': [float, float] | None,
                      'pips_low': int | None,
                      'pips_high': int | None,
                      'n_observations': int,
                      'status': 'measured' | 'low_sample' | 'insufficient_data',
                      'anchors': {'0': str, '20': str},
                      'submetrics': [{'id': str, 'label': str, 'value': float | None, 'score_20': float | None, 'weight': float, 'n': int, 'detail': str}],
                      'evidence': [{'kind': str, 'match_id': str, 'hand': int, 'decision_id': str | None, 'street': str, 'summary': str, 'delta_bb': float | None}],
                  }
              }
          }
      },
      'psychometrics': {
          'split_half': {axis_key: {'spearman_r': float | None, 'spearman_brown': float | None, 'n_players': int, 'reliable': bool | None}},
          'inter_axis_corr': {a: {b: float | None}},
          'pca_explained_variance': [float, ...],
          'collapse_warnings': [{'pair': [str, str], 'r': float, 'message': str}],
      },
      'forensics': dict,
      'aivat': dict,
      'meta_rank': dict,
  }
  ```
- Single-axis ablation bots (for verification tests, exported via `ABLATION_BOTS`):
  - `FatigueBot`, `AmnesiacBot`, `RandomSpewerBot`, `MixedCreativeBot`.

### 7. Server API & Viewer UX Additions (`server.py`, `cli.py`, `viewer/*`)
- New API endpoints in `server.py` (and included in static `export()`):
  - `GET /api/scorecard?run=RUN` -> `scorecard.build_scorecard(...)` (cached off request path like ratings!)
  - `GET /api/forensics?run=RUN` -> `forensics.audit_run_forensics(...)`
  - `GET /api/hand_eval?id=MATCH_ID&hand=H` -> `solver.evaluate_hand(...)` + `aivat.aivat_hand(...)` for interactive solver overlay in the replayer
  - `GET /api/plan_preview?players=N&deals=D&replicates=R&target=T&stack=S&alpha=A&aivat_vr=V` -> pre-flight power, AIVAT-adjusted deal budget, cost & runtime estimate
- New CLI commands in `cli.py`:
  - `python -m poker_arena scorecard <run> [--json PATH] [--solver dcfr|texassolver]`
  - `python -m poker_arena forensics <run> [--json PATH]`
  - `python -m poker_arena invariance <player_id> [--spots N] [--perms K]`
  - `python -m poker_arena solver-check` (checks NumPy DCFR + TexasSolver binary + Slumbot reachability)
- Viewer tabs & UX upgrades in `viewer/`:
  - **Scorecards tab** (and embedded summary on Standings): renders each model's **6-axis segmented-bar character sheet** (exact 20-segment bar aesthetic: bold uppercase label + italic/muted subtitle on left, 20 crisp rectangular segments on right with solid filled pips, translucent uncertainty-band pips for `ci95`, `0`/`20` anchor tooltip, psychometric reliability badge, and click-to-drill-down panel listing sub-metrics + clickable hand links that jump straight into the Replayer at that exact hand/decision!).
  - **Standings upgrades**: Meta-game ranking panel (Nash averaging + α-Rank + cycle alerts) and AIVAT variance-reduced edges next to official betting-CS verdicts + Forensics integrity summary banner.
  - **Watch / Replayer upgrades**: **Synchronized Duplicate Split View** toggle (Seat A/B mirror hands side-by-side stepping street-by-street in lockstep!), **Solver EV-loss & Pot-Odds bar** per decision (`EV loss`, `GTO freqs`, `MDF`, blunder badges), **Private Mind vs. Public Action diff** (`thinking` + `notes` diff vs. `action` + `say` + any injection/covert-intent flags).
  - **Planner tab**: Interactive **Pre-flight Experiment Planner** form (number of models, stack depth, target edge, α, track, expected AIVAT variance reduction) showing live power curve, required hands/conversations, stopping threshold, and estimated cost/time + copyable `poker_arena plan ...` CLI command.
