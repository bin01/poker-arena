# How Poker Arena decides who wins

This page explains how results become verdicts and standings, which of them are tests and which are only
descriptions, what the test's error rate rests on (independent conversations and the stacks, nothing about the
shape of the results), what that costs, and what it cannot tell you. The code is `poker_arena/ratings.py` (the design, the test, the standings) and
`poker_arena/history.py` (all-time standings). The calibration claims below are checked by
`tests/test_calibration.py`: small seeded versions on every test run, and the larger runs reported here with
`POKER_ARENA_HEAVY=1 python -m pytest -s tests/test_calibration.py` (about 2 minutes). The evidence rules (record
verification, current validity of official verdicts, identity splits and matching) are checked by
`tests/test_evidence.py` on records the runner plays.

## The question

Which models, each at a given reasoning level, win at heads-up No-Limit Hold'em, and by how much. The models
are told to win chips, so chips decide. The measure is **bb/100 over duplicate deals**: big blinds won per 100
hands, counted only on deals that both players played from both seats.

## Units

**A duplicate deal.** Every conversation of a pairing is two matches on the same seeds with the seats swapped.
Deal *k* is hand *k* of both. On it each player holds each seat's cards once against the other, which reduces
the luck of the cards; it does not remove it, because the two plays of a deal can go differently. Its value for
player *a* against player *b* is

    x = net chips of a over both hands / big blind / 2 × 100          (bb/100)

In the first model runs the spread of `x` was about 340 bb/100, with many exact zeros and heavy tails.

**A conversation** is one such pair of matches with fresh memories: one replicate of one round of one pairing,
one match folder. By default it is a single deal (two one-hand matches); see *Planning* for why. A model remembers everything inside a conversation, so its deals are dependent, and two
conversations of the same pairing can differ for reasons that have nothing to do with the cards (a read the
model formed early, a style it settled into). The conversation is therefore the independent unit, and a
conversation's mean edge is the observation the test uses.

**A slate** is the set of deals that one round and replicate deals to every pairing (the same seeds). The
conversations on one slate share their cards, so results of different pairings on one slate are correlated.

**Repeated deals.** A deal is identified by its fingerprint: a hash of the game settings that shape it, the seed
and the seat holding the button (`ratings.fingerprint`). When a pair has played the same fingerprint more than
once (only possible across runs), those plays are averaged into one observation.

## What is a test and what is a description

Exactly one thing is a test: the **preregistered sequential procedure on each pairing**, applied by one
function, `ratings.adjudicate`. The runner stops a pairing and records its decision only through it, and the
ratings, the command line and the viewer report what it recorded. Its verdicts *a is better*, *b is better* and
*even*, reached at a completed planned look, are the only inferential results, and only when

- the run is **preregistered** (`registration_status`, below): a `registration` written into plan.json before
  the first hand. A run without one (one fixed round, `play`, an older stopping rule, a plan that says
  `registered: false`, a registration written late, a design changed after it, or a code change accepted during
  the run) has `design: {registered: false, ...}`: every pairing there is `verdict: 'descriptive'`, nothing is
  ranked or led, and the only sample-size figure, under `design.retrospective`, is labelled *retrospective*;
- the pairing is **eligible** (see *The adjudicator*): every rule of the design holds for both players and for
  the data, or the verdict is `ineligible`, a terminal verdict that stops the pairing;
- the verdict is **official**: recorded by the runner at that look, by the registered judge with the registered
  calibration. What today's code computes from the same data is a recomputation, labelled `retrospective`;
- its **evidence still stands today** (`evidence_now`, below): the hands it judged are all still here, still
  replay from their seeds, hash as they did when it was recorded, and their records (decision coverage, audit,
  identity) pass today's checks.

A **decided rank** and a **decided leader** are built from those verdicts alone, so they inherit their
guarantee: a false leader needs at least one false verdict. A verdict whose checks failed (its
`assumptions` are `unsupported`, below) does not count unless the design preregistered that it should
(`outside_calibration: 'count'`). A leader is a player with a counting decided win over every other player of
its group; a rank is decided when a player has a counting decided win or loss against every other player of its
group (and the verdicts form no cycle). A group whose only would-be leader relies on a verdict whose checks
failed has `leader: null` and names it in `leader_outside_calibration`.

Everything else is **descriptive**: strengths and their intervals, the order by points, the group's front
runner and its margin, session Elo, style statistics, every all-time number. These are recomputed on every
refresh, would answer differently at a different moment, and are never called a result. Players carry
`rank_kind: 'decided' | 'descriptive'`, `inferential` (whether results involving them can be decided) and a
`position` (order by points); each group carries a `leader` (inferential, or null) next to a `front_runner`
(descriptive); each pairing carries `inferential`.

## The guarantee, and what it needs

With no edge between two players, the official test calls a winner for that pairing with chance at most
`alpha_pairing = alpha / n_pairings`, so that over the tournament the chance of any false verdict is at most
`alpha` (5% by default; Bonferroni, which needs no independence between pairings). The same bound holds for a
false *even*. It holds **at every look and whenever play stops, for any distribution of results**, given two
things, both true by construction:

1. **Conversations are independent**: fresh memories, no shared state, no deal repeated across the pairing's
   conversations (checked: `independent_deals`). They need not be identically distributed, and anything may
   happen inside one (memory, tilt, a read formed early, heavy tails, exact zeros, skew).
2. **A conversation's mean edge is bounded by the stacks**, which is a rule of the game, not an assumption
   about it: each of a duplicate deal's two plays wins or loses at most one stack (stacks reset every hand), so

       |x| = |net chips of a over both plays| / bb / 2 × 100  ≤  100 × stack / bb

   bb/100, and so is the mean of any number of deals. With the default 200-chip stacks and 2-chip big blind
   the bound is **±10,000 bb/100** (`ratings.outcome_bound`, `design.outcome_bound_bb100`; the largest a deal can
   really lose is one chip less than two stacks, −9,975, the round 6 reviewer's value). A deal beyond the bound
   means the record is wrong, and makes the pairing ineligible (`bounded`).

Nothing else is assumed: not normality, not symmetry, not a family of shapes. The reason is the test itself
(below): a bettor's capital is a nonnegative supermartingale under the null, and Ville's inequality bounds the
chance it ever reaches 2 / alpha_pairing. This replaces round 6's guarantee, which was exact only for symmetric
results and otherwise calibrated for a family of shapes; the round 6 reviewer's rare-loss distribution, inside
the chip bounds, broke it (467 false verdicts in 1,000 at a nominal 1.67%). That gate is kept beside the
official verdict as a secondary, **model-based** result (`model_based`), never official (*The model-based
gate*, below).

**The cost of needing nothing.** A mean is not testable without some assumption on the shape (Bahadur and
Savage, 1956), except through a bound. With a bound B the price is set by it: a null distribution that wins a
little nearly always and loses B now and then (probability about edge / B) looks exactly like a real edge until
the loss shows up, so any valid test needs about log(1 / alpha_pairing) × B / edge independent conversations,
however long each is. At B = 10,000 and an edge of 50 bb/100 that is about 900 conversations for 3 models
(the betting test needs about 1,100 to 1,450, depending on their length). Variance-adaptive betting cannot beat
that floor; deals inside a conversation cannot be bet on one by one (the null "a's expected chips are no
higher" says nothing about each deal given the ones before it, so a per-deal martingale is not valid for the
question asked); the levers that remain are the number of deals per conversation (*Planning*: shorter is far
cheaper in deals), the stacks (the bound is proportional to them), the target edge and alpha.

**Every verdict reports its checks** (`assumptions: {status, summary, checks}`, recorded with the decision):
`bounded` (no deal beyond the bound), `independent_deals` (no deal repeated across the pairing's conversations)
and `conversations` (enough for any verdict). `supported` is *checks passed: the guarantee holds for any
distribution of results within the game's bound, given independent conversations*; `unsupported` (a repeated
deal) means the conversations are not shown to be independent, and such a verdict does not count towards a
decided rank or leader unless the design says `outside_calibration: 'count'`; `unknown` is *checks
incomplete*. A decided verdict carries `calibrated` (kept from round 6: true when the checks passed) and a
`standing`.

## The adjudicator: registration, eligibility and official decisions

**Registration.** `ratings.registration(design)` is the artifact the runner writes into plan.json before the
first hand: `{status: 'registered', design, judge_fingerprint, calibration_id, created}`. The judge fingerprint
is the hash of `ratings.py` (the one the runner records for it); the calibration id hashes what sets the
official test (its bound, bettor, bar and margin). `ratings.registration_status(plan, evidence)` never recomputes it from the
design: a run is registered only when the artifact exists, its status is `registered`, its design inputs (alpha,
round deals, block, replicates, max rounds, target, planning assumptions, budget and the eligibility rules) match
the plan's, it was created no later than the first match started, its judge is the one the run started with, and
no code change was accepted during the run. A plan that says `registered: false` is `declined` even beside an
artifact; an `allow_code_change` that changes anything that plays or judges makes it `void`, with the files.

**Eligibility.** `ratings.adjudicate(pairing_data, design, n_pairings, look, eligibility)` runs `decide` and then
every rule; any failure gives `verdict: 'ineligible'` with `ineligible_reasons` (the test's own verdict stays in
`test_verdict`). The facts come from the runner (`runner.eligibility`, from the pairing's match folders): per
player its kind, backend, requested model, protocols, the models its replies named and its audit. The rules:

| rule | ineligible when |
|---|---|
| registration | no valid registration (as above), or a code change during the run |
| identity resolved | no model recorded, or a bare alias (`sonnet`) |
| identity matched | a reply reported another model (`identity_match: mismatch`, by `ratings.same_model`, rechecked by the judge from the models the facts name, whatever summary they carry); no reply at all; or the backend reported no model (`unattested`) and the design's `allow_unattested` (default `codex`, which cannot report it) does not name the backend. An allowed unattested player is shown as unattested everywhere: reporting some model is never a match. |
| audit complete | a used reply whose turn could not be checked for tool use: `opaque` (the `command` backend) unless `allow_opaque` names the backend (default none), or unverified or unrecorded; and, where the records link each decision to its calls (`decision_id`), the audit the judge derives from those links is not complete (below) |
| decision coverage | (from the runner's facts) a model action without exactly one decision record and its calls, decision records without an action, or no recorded actions at all |
| protocol consistent | the player's settings differ between the pairing's matches, or a match records no protocol |
| record verified | a counted hand does not replay from its seed (below), the runner could not verify a match record, or the hands carry no evidence hash |
| data integrity | a duplicate pair failed validation (below), or the pairs were not validated at all; no verified deal at all; a deal beyond the game's bound (more than the stacks allow) |

**Duplicate pairs are validated, and every counted hand is replayed** before it counts, by one reader,
`ratings.read_conversation` (`ratings.check_pair` and `ratings.verify_play`), used by `load_deals`,
`pairing_data`, the ratings, the history and the style statistics, and offered to the viewer
(`ratings.verified_hands`): it says which hands are **verified results** and which are **provisional** (written
but past the play's commit point, or committed while the other play has not finished that hand) or
**rejected**. b-first must seat
a-first's players in the other order, with the same game settings, seeds, deal fingerprints, specs and
protocols; each hand's recorded seed and fingerprint must match the configs and its chips must add up; and each
play of the hand goes through the engine's record verifier (`games.holdem.verify_hand`), with the seed and
button the config assigns it (not the ones the hand records): the deal is regenerated from the seed, the
recorded actions are replayed as legal moves of the seat to act, and the cards, board, every action's pot and
price, the payouts, the summary and the hand's events must equal the replay. Only committed hands count (rows
past `state.json`'s `hands_committed` are an interrupted hand, not a result). A failing conversation or hand is
left out and reported (`rejected`), and the pairing is ineligible; a verifier that is missing or fails to run
rejects everything (fail closed). The round 5 probe (one hand edited to hold a duplicated ace and pay 200,000
chips from 200-chip stacks, in both plays so the pair stays consistent) is rejected by `load_deals`, makes
`adjudicate` ineligible, makes the run's official verdict stop counting, and is kept out of the history
(`tests/test_evidence.py`).

**Evidence hash.** `pairing_data` returns `evidence_hash`, the hash of the verified records it judged (each
deal's conversation, index and the digests of its two plays: the hand record, its events, and since round 7 its
call records and raw backend output rows, `evidence_version: 2`), and `adjudicate` records it in the decision
(`judged_evidence_hash`, `evidence_version`; the runner also records its own hash of the match records as
`evidence_hash`). A decision recorded under version 1 is checked against the version 1 hash (hands and events),
so the new definition never makes an old record look changed. Without verified evidence (`evidence_hash`
missing) a pairing is ineligible.

**Linked audit.** When the runner links every decision to its records by a `decision_id` (on the decision
record, the action it led to, every call attempt and every raw output row), the judge derives each accepted
decision's audit from them (`ratings.linked_audit`): verified only when its action and at least one call carry
the id, the accepted (last) attempt says `verified` and has a raw output row with the same id and attempt, and
the decision record agrees; `opaque` when that attempt comes from a backend with no record. Per player over the
pairing (`linked_audit`, `ratings.linked_status`): `complete`, `opaque`, `incomplete` (any decision unlinked,
unverified or missing its raw row: the pairing is ineligible), `bot`, or `legacy` (records from before decision
ids, judged by the runner's aggregate facts alone, and shown so).

**Official and retrospective.** Each look the runner records is adjudicate's output with the judge fingerprint
and calibration id. The ratings show a pairing's verdict from that record when it was made by the registered
judge with the registered calibration (`official: true`, `judge`); they also adjudicate the same deals with
today's code and show it as `recomputed: {verdict, judge, counts_toward_leader, agrees}`, which never replaces
the record. Without such a record (a missing look, or one written by another judge) the row is
`retrospective: true`, and a decided result there becomes `verdict: 'retrospective'`, which counts for nothing.

**Does an official verdict still count?** Only if its evidence still stands today. Every refresh checks each
official record (`ratings.evidence_now`) and gives it one status; the verdict stays shown as recorded, but
`counts_toward_leader` (and so any decided rank or leader built on it) holds only for `verified`:

| `evidence_now.status` | kind | meaning |
|---|---|---|
| `verified` | | the hands it judged are here, unchanged, replay from their seeds, and today's checks of their records pass |
| `unbound` | evidence | the record carries no evidence hash, so the hands here cannot be shown to be the ones judged |
| `invalid` | evidence | hands it judged no longer pass validation or no longer replay |
| `lost` | evidence | hands or whole conversations it judged are missing (a look whose rounds are no longer all present is still judged at that look, so deleting a folder cannot hide a decision behind an earlier one) |
| `changed` | evidence | the hands hash differently, or the records behind identity, audit or coverage differ from the facts recorded with the decision |
| `altered` | evidence | today's judge is the registered one and has the same evidence and facts, yet does not reproduce the recorded verdict: the record was edited |
| `fails_today` | rules | the evidence and its records are unchanged, but checks that are stricter today reject them |

A change to the test itself (a new judge, other bars) is not an evidence failure: the official verdict stands
and counts, and today's verdict is shown beside it. The round 5 probe (`official`, `eligible` and
`counts_toward_leader` all true while today's recomputation said `ineligible`) now gives
`counts_toward_leader: false` with the reason, and so do hidden decision records, a deleted conversation and an
edited verdict (`tests/test_evidence.py`, on records played by the runner).

## The preregistered design

`ratings.normalize_design(plan)` turns a plan into the one design that the runner plays by, the ratings judge by
and the viewer shows (nothing else builds one):

| key | default | meaning |
|---|---|---|
| `alpha` | 0.05 | chance of any false verdict over the tournament |
| `round_deals` | the plan's `deals` (1 from `plan` on the command line) | deals per conversation |
| `replicates` | 1 in a plan, 25 from `plan` on the command line | conversations per pairing per round (one look per round) |
| `max_rounds` | from the power target | rounds (looks) at most, up to 200; `max_rounds_basis`: `power`, `given` or `budget` |
| `budget_deals` | none | a cap on deals per pairing; a plan it cuts short is `underpowered` |
| `n_pairings`, `alpha_pairing` | | Bonferroni: each pairing is tested at `alpha / n_pairings` |
| `official_test`, `assumption_free` | | the official test's name, `assumption_free: true`, its bound and bar |
| `outcome_bound_bb100` | from the game settings | the most a duplicate deal (and a conversation mean) can move: 100 × stack / bb |
| `threshold` | | the capital a bettor must reach: 2 / alpha_pairing |
| `bet_cap` | 0.9 | the bettor stakes at most this share of what it could lose without going broke |
| `min_conversations` | | the fewest conversations from which a verdict is possible at all (9 for 3 pairings) |
| `target_edge_bb100` | 50 | the edge the plan is powered for, fixed before play |
| `prior_sd_bb100` | 350 | the per-deal spread assumed when planning |
| `assumed_icc` | 0.01 | planning: the share of the per-deal variance that is a conversation effect |
| `assumed_ar1` | 0.2 | planning: AR(1) memory between consecutive deals of a conversation |
| `sd_conversation_bb100` | | the spread of a conversation's mean under those assumptions |
| `power_scenarios` | | per scenario (optimistic, assumed (moderate), pessimistic dependence): `power`, the chance at the target that the better player is decided by the last look (the official test, simulated), and `conversations_used` |
| `power_at_max` | | the moderate scenario's power |
| `power_by_edge` | | power at 20, 50 and 100 bb/100, moderate dependence |
| `power_basis` | | says in words what the power numbers are |
| `leader_chances` | | for a full round robin: the chance that the true best player is named the decided leader, per edge scenario |
| `deals_needed`, `conversations_needed` | | per pairing, for 80% power under the moderate scenario (None: out of reach of 200 rounds) |
| `deals_planned`, `max_deals` | | the deals per pairing the design allows |
| `allow_unattested` | `['codex']` | backends whose players may be eligible without attesting the model that answered |
| `allow_opaque` | `[]` | backends whose players may be eligible although their turns cannot be audited for tool use |
| `outside_calibration` | `'exclude'` | whether a decided verdict whose checks failed counts towards a decided rank or leader (`'count'`) |
| `registered` | | only `false`, for a plan that declines registration or an older stopping rule; whether a run is registered comes from `registration_status`, never from the design |
| `calibration` | | what sets the official test (bound, bet cap, prior, bar, margin); hashed into the calibration id |

`python -m poker_arena plan` writes `max_rounds` (and its basis) into the plan, so a stored plan reads back the
same design even if the planning defaults change later. It also prints the deals the official test would need at
other conversation lengths (`ratings.length_tradeoff`).

## Planning: the official test, simulated

A plan's power is the official test's, from `ratings.procedure`: pairings simulated by their conversation means
(normal, with the spread the scenario gives a conversation of `round_deals` deals, clipped to the bound; 4,000
pairings per number), each bettor run conversation by conversation, looked at after every round as the runner
does, stopping at the first verdict. One simulation per edge and spread answers every number of rounds and of
replicates. The spread of a conversation's mean, with per-deal spread σ, ICC ρ_c and AR(1) memory φ,

    sd(conversation mean)² = ρ_c σ² + (1 − ρ_c) σ² · DE(φ, R) / R,     DE(φ, R) = 1 + 2 Σ_{k<R} (1 − k/R) φ^k

matters little: the bettor's stake is capped at 0.9 / B, so its capital grows by about 0.9 × edge / B per
conversation whatever the spread, and the spread only adds a small penalty (λ² sd² / 2). The dependence
scenarios therefore change the power by a point or two at most, and at one deal per conversation not at all;
they never change the error rate. Through `ratings.decide` on fresh pairings the advertised power is what the
test does (default design: 81.7% advertised; 83.0% with normal deals and 81.7% with poker-like deals, 3,000
pairings each, `tests/test_calibration.py`).

**Conversation length.** Because the bound, not the spread, sets the number of conversations, short
conversations cost far fewer deals (3-model design, 80% at 50 bb/100, moderate dependence, looks after every
conversation):

| deals per conversation | conversations | deals per pairing |
|---|---|---|
| 1 | 1,454 | 1,454 |
| 2 | 1,327 | 2,654 |
| 5 | 1,211 | 6,055 |
| 10 | 1,165 | 11,650 |
| 25 | 1,128 | 28,200 |
| 50 | 1,112 | 55,600 |

So the default is now **one duplicate deal per conversation**: each conversation is the two one-hand matches of
one deal, with fresh memories. The price is what such a run measures: how well a model plays a hand against this
opponent, not what it learns about the opponent over a match. Adaptation is measured on its own track (*Two
tracks*, below), not by lengthening the official conversations. Each conversation is a fresh session for both players, so a run makes many more of them (1,475
conversations of a pairing against round 6's 40), which costs CLI start-ups and system prompts but not the long
contexts of 50-hand conversations.

**The default 3-model design** (3 pairings, 1-deal conversations, 25 per round, target 50 bb/100):

| | value |
|---|---|
| rounds, deals per pairing at most | 59 rounds, 1,475 deals (2,950 hands); 8,850 hands for the tournament |
| bound, bar | ±10,000 bb/100; a bettor's capital must reach 120 (2 / 0.0167) |
| first conversation that can decide | 9 (every conversation at the bound); in practice at a true edge of 50 about 1,200 |
| power at 50 bb/100, any dependence scenario | 81.7% |
| power by edge: 20 / 50 / 100 bb/100 | 0.7% / 81.7% / >99.9% |
| decided leader is the true best: one best by 50 / by 100 / ladder 50 apart | 66.7% / >99.9% / 81.7% |

**The 9-model design** (36 pairings, each at 0.05 / 36 = 0.0014; same defaults):

| | value |
|---|---|
| rounds, deals per pairing at most | 86 rounds, 2,150 deals; 154,800 hands for the tournament |
| power at 50 bb/100, any dependence scenario | 81.5% (about 1,830 conversations used) |
| power by edge: 20 / 50 / 100 bb/100 | 0.1% / 81.5% / >99.9% |
| decided leader is the true best: one best by 50 / by 100 / ladder 50 apart | 19.4% / >99.9% / 81.5% |

**The cost against round 6's gate**, honestly:

| | round 6 gate (model-based) | official test, 1-deal conversations | official test, 50-deal conversations |
|---|---|---|---|
| guarantee | within a family of shapes; broken by the R6 counterexample | any distribution within the bound | any distribution within the bound |
| 3 models: deals per pairing for 80% | 2,000 planned (1,290 used on average at the target) | 1,475 planned (about 1,200 used) | 55,600 |
| 3 models: conversations per pairing | 40 planned (25.8 used) | 1,475 planned (about 1,200 used) | 1,112 |
| 9 models: deals per pairing for 80% | 3,000 planned (2,155 used) | 2,150 planned (about 1,830 used) | 83,850 |
| power of the official test on round 6's budget (10 rounds of 4 × 50) | | 0% (it needs about 1,100 conversations) | |
| decided leader, 3 models, one best by 50 | 69.3% | 66.7% | |
| decided leader, 9 models, one best by 50 | 21.8% | 19.4% | |

In deals, an assumption-free verdict costs about what round 6's model-based one did, *if* conversations are one
deal long; kept at 50 deals it costs 28 times as many. In conversations (fresh sessions) it costs about 35 to
45 times as many either way. The round 6 numbers are its own planner's (`ratings.procedure_model_based`).

## Two tracks: official and adaptation

`python -m poker_arena plan` writes one tournament with two tracks (`runner.make_tournament`; `--tracks` picks
them): the same players and seeds, played into separate run folders (`<out>/official`, `<out>/adaptation`), each
with its own plan.json, `registration` and design (`design.track`, which is part of the registered inputs, so a
run cannot switch tracks after registration).

| | official | adaptation |
|---|---|---|
| conversation | a fresh one every deal (default 25 per round) | 50 deals (default 4 per round): models remember and adapt |
| verdict and stopping | the betting confidence sequence (above): assumption-free given independent conversations | the model-based gate (below), with its stated assumptions |
| power | `procedure` (the official test, simulated) | `procedure_model_based` (the gate, simulated), max_rounds the fewest for 80%, at most the 40 looks and 200 conversations the gate is calibrated for |
| decides ranks and leaders | yes | never: `official: false`, `counts_toward_leader: false` |
| labelled | official | *adaptation · model-based · not official* |
| all-time history | pooled with official runs | pooled only with adaptation runs (`history.all_time(track=...)`), never with official ones |

On the adaptation track `adjudicate` takes its verdict from the gate (`gate`, with the betting test's verdict on
the same data kept as `assumption_free_verdict`), applies every eligibility rule as on the official track, and
returns `track: 'adaptation'`, `official: false`, `model_based: true`, `registered_record` (made by the
registered judge under a valid registration) and `model_based_counts` (eligible, decided, diagnostics passed).
Its assumptions add the gate's diagnostic (`calibrated_family`: the conversation share within the calibrated
family); *supported* there means *diagnostics passed; distribution assumptions unverified*, never a guarantee.
The runner stops on these verdicts and records them like official ones; `rate` reads them as recorded but never
builds a rank or a leader from them: a group reports `model_based_leader` (a counting model-based win over every
other player) beside `leader: null`, and `inferential` is false for every adaptation player.

The default 3-model adaptation design: 10 rounds of 4 conversations of 50 deals (2,000 deals per pairing at
most), a verdict possible from 12 conversations; at 50 bb/100 the gate decides for the better player with
diagnostics passing 83% of the time under the assumed dependence (99% with none, 60% if it is worse), after
about 26 conversations. For 9 models: 15 rounds (3,000 deals), 83%. These powers assume normal conversation
means; the gate's error rate holds only within its calibrated family of shapes (*The model-based gate*), which
is why nothing on this track is official.

## The test

After each round a pairing's verdict comes from `ratings.decide`, inside `ratings.adjudicate`, called by the
runner with every verified deal so far and the conversation of each deal, conversations in play order (round by
round, replicate by replicate: an order fixed before play, never by results).

**The bettors.** Each conversation contributes its mean edge Y, clipped to ±B (a value beyond it makes the
pairing ineligible anyway). For a null mean m, a bettor against "the mean is at most m" holds capital
K_n(m) = Π_{i≤n} (1 + λ_i (Y_i − m)), starting at 1. Its stake λ_i is the aGRAPA fraction (μ − m) / (v + (μ − m)²)
from the mean μ and variance v of the conversations before i (v starts from 350², worth one conversation; μ
starts at m, so the first stake is zero), clipped to [0, 0.9 / (B + m)]. The stake uses only the past, and
1 + λ (Y − m) ≥ 0.1 for every Y ≥ −B, so under any distribution of independent conversations whose mean is at
most m the capital is a nonnegative supermartingale, and by Ville's inequality the chance that it ever reaches
2 / alpha_pairing is at most alpha_pairing / 2. The mirror bettor (on −Y, against −m) tests "the mean is at
least m". Four bettors run: against ≤ 0 (`a`), ≥ 0 (`b`), ≤ −15 and ≥ +15 (`even`).

**Verdicts at a look.** `a` if a's bettor's capital has reached the bar at any conversation so far (its running
maximum: Ville covers every conversation, so looking only at rounds loses nothing), `b` likewise, `even` when both
margin bettors have. Whatever the true edge, a false verdict needs at most two of these bettors, each valid for
that edge, to reach the bar (for an edge of 0: `a` or `b`; at or beyond +15: `b` or `even`), so a pairing gives one
with chance at most alpha_pairing. A verdict reached at one look stands: the runner stops there.

**What it reports.** `capital` (each bettor's best so far: `a`, `b`, and `even`, the smaller of the two margin
bettors) against `official_test.threshold`; `cs`, the anytime-valid interval for the edge (every m whose two
bettors have not reached the bar: it holds the true edge at every look at once with chance at least
1 − alpha_pairing, level `cs_level`); `decided_at_look` and `decided_at_conversation`. `ci95` is an ordinary t
interval, descriptive only. `power.deals_for_80pct` is a planning number from the observed spread of conversation
means (`ratings.conversations_for`, within a few percent of the simulation).

| verdict | meaning |
|---|---|
| `a` / `b` | that player's bettor reached the bar: that player is better. Decided. |
| `even` | both margin bettors reached the bar: the edge is within ±15 bb/100. Decided; out of reach of the default design at 100-big-blind stacks (it needs a few thousand conversations), reachable with shallow stacks. |
| `exhausted` | The last planned look came without a decision. **Not evidence that the two are equally good.** |
| `open` | Undecided and looks remain, or a round is under way. |
| `early` | A completed look with fewer than `min_conversations` conversations, from which no verdict is possible. |
| `ineligible` | An eligibility rule failed (see *The adjudicator*). Terminal: the pairing stops. `test_verdict` keeps what the test said. |
| `retrospective` | A decided result that no official record backs (today's recomputation only). Decides nothing. |
| `descriptive` | No test applies: pooled across runs, or a run that is not preregistered. |

An official verdict recorded by an earlier judge whose test was the round 6 gate stays official as recorded, and
says so (`official_test.assumption_free: false`, a note); today's assumption-free verdict is shown beside it as
the recomputation.

### Calibration

Null false-winner rate per pairing (target `alpha_pairing` = 0.0167, 3-model design), the official test
vectorized exactly as `decide` computes it (checked against `decide` sequence by sequence), over every look of
two long designs: the default (1,475 conversations of 1 deal; 20,000 sequences) and 800 conversations of 50 deals
(2,000 sequences):

| null | 1-deal conversations | 50-deal conversations |
|---|---|---|
| normal, t3, poker-like (40% zeros, t3) or skewed deals | 0 | 0 |
| the round 6 reviewer's rare loss (independent deals: −9,975 with chance 1/799, else 0 or +25) | 0 | 0 |
| least favourable two-point (+50, or −10,000 with chance 50/10,050) | 0.0050 | 0 |
| conversation effects: normal, exponential, lognormal or rare-loss at share 0.5; rare-loss or symmetric two-point at share 0.9; poker-like deals with AR(1) 0.3 | | 0 |
| three pairings on shared slates (any false verdict in the tournament, target 0.05) | 0.0003 | |

The only null that comes near the bound is the least favourable one, a small win nearly always and the largest
loss the stacks allow otherwise: 0.0050, within alpha_pairing / 2 = 0.0083, as Ville's inequality promises.

**The round 6 counterexample**, as the reviewer ran it (seed 4901, 1,000 sequences of 12 conversations of 50
deals, the round 6 default design at look 3): the round 6 gate called a false winner in 450 (425 with its
diagnostics passed; the reviewer's run: 467 and 444); the official test in 0. Over every look of either design
above: official 0; the round 6 t arm over its own design 0.47. `tests/test_calibration.py` keeps it as a
regression: it fails on the round 6 gate and passes on the official test.

**The anytime-valid interval** held the true edge at every look at once in at least 1 − alpha of sequences of the
reviewer's rare-loss deals shifted to +30 (checked at 20, 50, 100 and 200 conversations).

### Equivalence: false "even"

`even` is false whenever the true edge is at or beyond ±15 bb/100; it needs the bettor against that side's
margin to reach the bar, so its chance there is at most alpha_pairing / 2 for any distribution. With 2-big-blind
stacks (bound ±200 bb/100, where `even` is reachable: 0 is called `even` more than half the time with a per-deal
spread of 60), 8,000 sequences each, target 0.0083:

| conversation results | +15 | −15 | +16 | +20 |
|---|---|---|---|---|
| normal, spread 60 | 0.0013 | 0.0018 | 0.0008 | 0 |
| least favourable two-point (0 nearly always, the bound now and then) | 0.0034 | 0.0035 | 0.0015 | 0.0001 |

### The model-based gate (secondary)

`model_based` is the round 6 gate on the same data, computed only for designs it was calibrated for (at least
2 deals per conversation, at most 40 looks and 200 conversations): the conversation-level t (CR1 standard error,
C − 1 degrees of freedom) against a bar calibrated by simulation for a family of null shapes (normal,
exponential, lognormal and rare-large-loss conversation effects up to half a conversation mean's variance),
together with a sign-flip randomization test, and TOST for `even`; with its diagnostic (`conversation_share`
within 0.5). Within that family it keeps its level (t arm alone, 200,000 sequences each: 0.0065 to 0.0157 against
0.0167). Outside it, it does not: the round 6 counterexample above. It is labelled *model-based: secondary,
descriptive, never official; its error rate is not guaranteed*, it never stops an official-track pairing, never
counts towards a rank or a leader, and the official planner never uses it. On the adaptation track it is the
judge (with its own registered design and planner), still never official.

## Strength (descriptive)

Each player gets a strength in bb/100 from a least-squares fit over all observations, `x(a vs b) ≈ s_a − s_b`,
with the strengths summing to zero within each group of connected players: the edge against the average
opponent of the group. The fit assumes edges add up across players, which poker's rock-paper-scissors effects can
break; when every pairing is the only link between its players the assumption cannot be checked, and the notes
say so. The per-pairing edges never depend on it, and they stay the primary evidence.

**Weighting.** Each distinct deal counts once, so a pairing weighs as much as it has deals, and a strength is an
average over the opponents a player actually met, as often as it met them (`fit.method` in the output). Unequal
exposure therefore moves strengths: in the bots' demo run Station sits above Random although Random's direct
edge over Station is positive, because Station met the strong Rule bot in fewer deals.

**Fit diagnostics.** Every pairing carries `observed_edge` (over the deals in the fit), `fitted_edge` (what the
strengths imply, s_a − s_b), `residual` and `residual_se` (the bootstrap spread of the residual, with the same
joint weights as the strengths), and `additivity: {status: 'ok' | 'warning' | 'untestable', z}`: a warning when
the residual is beyond two standard errors, untestable in a group that fits its pairings exactly. A warning goes
into the notes with both numbers: one number per player does not describe those results, so read the pairings.

**Groups.** Players are connected when a chain of played pairings links them; different groups are on separate
scales and every summary is per group, from that group's own data. A group without a reference bot has no fixed
anchor, and the notes say so; plans should include a bot to connect models to the same reference.

**Uncertainty: a joint, stratified bootstrap.** The resampling must keep two dependencies: deals within a
conversation (memory, conversation effects) and pairings on one slate (shared cards). So the units are whole
slates, resampled jointly across every pairing on them:

1. Slates with the same set of pairings form a stratum. A stratum of at least two slates resamples whole slates:
   every conversation on a drawn slate comes with it, so both dependencies are kept (`ci_basis: conversations`).
2. A slate alone in its stratum (for example the only slate of a run with one round) cannot show how slates
   differ. It is split into aligned blocks of 10 deals, which are resampled jointly (`ci_basis: blocks`). Such an
   interval is conditional on the conversations played: it leaves out how much fresh matches differ, and the
   output says so.
3. Each stratum uses the Rao–Wu rescaling bootstrap (k − 1 of its k units drawn with replacement, weighted
   k / (k − 1)), so the bootstrap variance of a linear statistic is unbiased even with two units. Intervals are
   the point ± the Student t quantile on the bootstrap's degrees of freedom (units minus strata) × the bootstrap
   standard error. A player behind no resampled unit gets no interval. Groups never share a stratum.

**Coverage** of the 95% strength intervals in seeded round robins of 4 players (strengths ±60 and ±20 bb/100,
poker-like noise, AR(1) memory, conversation effects and a shared-card effect through style loadings), 1,600
intervals each: 0.953 with 4 slates (conversation effects, shared cards); 0.941 with 1 slate (blocks, no
conversation effect). With a single conversation per pairing the interval cannot see how much conversations
differ (earlier measurement: 0.895 with a conversation effect of the size of the card noise).

## Session Elo (secondary, descriptive)

How often a player wins a session (10 deals of one conversation, or the whole of a shorter one; up by more
than 1 big blind; closer is a tie), as a Bradley–Terry fit
on the Elo scale with a Gaussian prior (1.5 logits), centred on 1500 within each group. Sessions of one
conversation are not independent, so the interval is the wider of two: the Laplace (posterior) interval, which
treats sessions as independent and cannot collapse to a point, and a bootstrap that refits the ratings under the
strength bootstrap's weights, so sessions move with their slate and conversation (`session_elo_basis` says which
one set it). It can disagree with strength (many narrow wins and a few heavy losses), and it never decides
anything.

## Who is a player

A model player is `backend:model:effort#protocol@implementation`, for example
`claude:claude-sonnet-5-5:high#1a2b3c4d@5e6f7a8b` (a bot: `bot:rule:sims=200@5e6f7a8b`). The
protocol is the first 8 characters of the fingerprint the runner records for each seat: a hash of every
setting that changes play (backend, model, effort, isolation, instructions or system prompt, tool policy, extra
arguments, retries and fixes, thinking display) and of the match horizon and memory schedule (hands per
conversation, which the model is told; a fresh conversation at every match). A protocol recorded before the
horizon was part of it gets the horizon in the identity instead (`...#1a2b3c4d~h50`), so conversations of
different lengths never pool. A player from before protocols were recorded is kept apart from later runs.

The implementation is the first 8 characters of a hash of what played the seat (`history.implementation_tag`):
the content hashes of the code that plays (engine, games, agents, backends, runner: everything the runner
records but the judge, `ratings.py`), the prompt template, and for a model the version of its CLI. A change to
parsing, corrections, tool detection or bot behaviour, or a new CLI version, makes another player, from the hand
it took effect (`history.implementation_segments`): a match records the implementation and CLI versions it
started with, each accepted code change (`implementation_changes`) and CLI version change
(`cli_version_changes`) with the hand after which it applied, and each resume's CLI versions (`state.json`); the
history gives every deal its players' identities at that hand of both plays, so the deals before a change pool
under the old identity, the deals after under the new one, and a deal whose two plays straddle the change gets an
identity of its own (pooling with neither). A model whose executable was not recorded is tagged with every
recorded CLI version, so any upgrade splits it. In a preregistered run a CLI version other than the one the
registration pinned, anywhere in the run, voids the registration. When such a change is shown not to matter for
play, an `equivalences.json` beside the database
(`{"implementations": [["5e6f7a8b", "9a0b1c2d"]], "reason": "..."}`) pools each group under its first tag, and
the all-time output lists what it merged.

**Which model answered.** Each call records `requested_model`, `observed_model` (what the CLI reported:
Claude's init event, Antigravity's transcript; Codex and plain commands report nothing), `identity_reported`
and `identity_match`. Reporting a model is not matching it, so there is one rule, `ratings.same_model`, and one
three-way form of it, `ratings.identity_match(requested, observed)`: `match` (a model was reported and it is the
requested one), `mismatch` (another one was reported) or `unattested` (none was). The judge recomputes it from the
models the facts name rather than trusting a summary; `stats` recomputes it for every call (a recorded `match`
never makes a reply a match, a recorded `mismatch` stands) and reports `identity_match` and `identity_reported`
counts per player; the ratings give each player and pairing `identity_match` (the worst over its pairings). The
rule: the same canonical id (lower case, no provider prefix, punctuation and spaces as hyphens, so
"Gemini 4 Neon (High)" is `gemini-4-neon-high`), a dated snapshot of it (`<id>-YYYYMMDD` or `<id>-YYYY-MM-DD`),
or an id listed for it in `ratings.SERVED_ALIASES` (empty by default). Nothing else matches: `claude-sonnet-5`
is not `claude-sonnet-5-5`, `gemini-4-neon-high-0912` is not `gemini-4-neon`, a bare alias (`sonnet`) matches
nothing, and nothing requested matches nothing.

A player is **quarantined** when its identity is unresolved (no model recorded, or a bare alias such as `sonnet`
whose version changes over time), when its settings differ between the matches of one run, or when another model
answered. Within a run a quarantined player stays in the fit with a flag (`in_fit: true`, `quarantined: true`,
`inferential: false`), its pairings are ineligible when the reason holds inside them, and it never takes a
decided rank or the lead. In the all-time standings it is left out of
the fit (`in_fit: false`), listed with its hands and net result; a served-model mismatch quarantines only that
run's deals, under their own identity (`...!served=<model>`).

## All-time standings (descriptive only)

`history.all_time` pools every run of one track (official by default; the `runs` table records each run's
track from its plan.json) in `arena.db` that was played under the same game settings; the tracks never pool. Those runs had
different designs and stopped pairings at different times, so pooling them is no one's preregistered test:
every pairing there has `verdict: 'descriptive'`, there is no leader and no rank, and the notes say so. Repeated
deals are averaged; slates are matched by seed across runs, so runs on the same cards are resampled together.
Every hand is replayed from its seed at sync time, as in a run; duplicate pairs or hands that fail validation
or replay are kept in the `rejected` table and listed, never pooled. A database from before schema 6 is read
again from the folders that still exist (identities gained the hand-level implementation split).
Independent confirmation needs a new run with its own design.

## How good at poker: the reference bot and the quiz

Every number above is **peer-relative**: a strength is bb/100 against the average opponent a player met, and a
verdict compares two players. That says which model wins, not how good any of them is at poker. Two things
answer that, and both are reported apart from the standings:

- **The reference bot** (`bot = "reference"`, version `reference-1`): a fixed heuristic, **not a solver**:
  tight-aggressive, position-aware, equity against the range the opponent's betting implies, pot odds, a fixed
  bluff frequency; it never adapts. `python -m poker_arena plan` warns when a plan has no reference player. When
  a group has one, each player carries `vs_reference` (its direct edge against it, the descriptive interval, the
  anytime-valid interval, the deals, the pairing's verdict and the reference's version), and the output carries
  `reference` (`ratings.REFERENCE`) with its measured limits. Measured against the other bots (2,000 duplicate
  deals per pairing, 40 conversations of 50, 100-big-blind stacks; bb/100, standard error between
  conversations):

  | | random | station | rule |
  |---|---|---|---|
  | reference | +408 (50) | +290 (14) | +169 (24) |
  | rule | +856 (67) | +273 (13) | |
  | station | +44 (78) | | |

  It beats every baseline directly, but wins about half as much from the random bot as the rule bot does, so it
  is not maximally exploitative, and how it does against strong play is unmeasured (no solver or expert
  comparison). A positive edge against it means *beats a solid fixed heuristic*, not *plays well*.
- **Poker fundamentals** (`python -m poker_arena quiz`, written to `quiz.json`): decision spots with computed
  answers, asked of each player through its own backend, each answer kept with its calls, raw output, identity
  and audit. The ratings output carries them as `fundamentals`, **recounted from the answers**: a fallback (no
  playable answer, or no reply) is never correct, whatever it did, even where folding was best; `accuracy` is
  over every spot asked and `valid_accuracy` over the answers the player actually gave, with `fallbacks` apart;
  per category, and the headline is the basics'. Each score carries its `provenance` (quiz version and spots,
  protocol, implementation, CLI version, identity and audit) and `eligibility`: `eligible` only when every
  answer was attested as the requested model and verified against its raw output and the provenance is
  recorded (`unverified` when the record keeps no answers, as quizzes before round 7 did). A score is attached to
  a run's player (same model and reasoning level) only when it is eligible; otherwise it stays under its quiz
  name. The basics are a fixed, hand-picked set, so they carry counts, not intervals; a generated category, whose
  spots are an i.i.d. sample from its generator, carries a 95% interval for that generator's spots, with the
  estimand named. They are never pooled with strengths or verdicts.

## Style statistics (tendencies, not quality)

`stats.collect` counts standard poker tracking statistics per hand dealt, under plain names
(`stats.LABELS`, also in the ratings output as `stat_labels`): *Plays preflop* (VPIP: put money in preflop by
choice), *Raises preflop* (PFR), *Aggression after the flop*, *Goes to showdown*, *Wins at showdown* (a split pot
is not a win; *Splits at showdown* counts those), *Talks*. Every rate comes with its counts
(`counts: {rate: {n, of}}`) and a breakdown by opponent (`by_opponent`) and by seat (`by_position`: button, which
posts the small blind heads-up, or big blind; names in `position_labels`). No value of these is good or bad in
itself. They count **verified results only**, the hands `ratings.read_conversation` accepts, like the verdicts
and the strengths; each player's `population` says so and how many provisional and rejected hands were left
out, so a tampered or interrupted hand never reaches a style total. (Which model answered is read from every
call, result or not: a mismatch anywhere counts.)

## Limits

- **The price of no assumption.** The official test needs about log(2 / alpha_pairing) × bound / edge
  independent conversations, however long they are (about 1,450 for 3 models at 50 bb/100); no valid test can
  need much fewer without assuming something about the shape of the results. Short conversations keep that
  affordable in deals but measure a model's play hand by hand, not its adaptation over a match; long ones cost
  28 times the deals.
- **The model-based result is not a guarantee.** It is shown because it can say something with far fewer
  conversations, under its family of shapes; the round 6 counterexample shows how wrong it can be outside it.
- **Few conversations.** No verdict before `min_conversations` (9 for 3 pairings), and in practice none before
  about 1,000 at the default target; the first model runs had one conversation per pairing, so they are
  descriptions of those matches.
- **The planning assumptions** (per-deal spread, ICC, memory) change only the power, and only a little; each
  pairing's `deals_for_80pct` uses the observed spread. The planner draws normal conversation means;
  poker-like deals gave the same power.
- **A decided leader is hard to get in a large round robin**: it needs all of its pairings decided, so with 9
  players and one best by 50 bb/100 it comes 19% of the time (more than 99.9% at twice that).
- **The record verifier checks what was played, not why.** It shows that the recorded cards, actions and payouts
  are what the engine deals and plays from the seed; it cannot show that a recorded reasoning is the model's.
  Evidence binding is a hash in a file the operator controls, like the registration: it catches edits, missing
  files and records that do not reproduce, not someone who rewrites every record consistently.
- **Independence of conversations** is by design (fresh memories, no shared state, no repeated deal); it is the
  one thing the guarantee needs, and a model provider that carried state across conversations would break it.
- **Additivity** of strengths is an assumption (see Strength).
- **Identity is as good as the CLI's report.** A CLI that doesn't report which model answered (Codex) can't be
  checked; it is eligible only because the default design allows it, and it is shown as unattested.
- **Winnings are relative.** Strength and verdicts say how a model does against the players it met. The fixed
  reference bot gives every run a common, versioned opponent, and the quiz scores fundamentals against computed
  answers; neither is a measure of play against strong humans, and the reference is a heuristic, not a solver.
- **The registration is a record, not a signature.** It proves the design was fixed before the first match
  started only as far as the files are honest: someone editing plan.json and every match config by hand could
  forge it.
- **One game format.** Heads-up, stacks reset every hand, 100 big blinds deep by default (a plan may set other
  stacks with `--stack`; the bound, and the conversations needed, scale with them). Strengths do not carry over
  to other formats.
