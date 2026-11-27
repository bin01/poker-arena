# How Poker Arena decides who wins

This page explains how results become verdicts and standings, which of them are tests and which are only
descriptions, under which assumptions the tests keep their error rate, where they stop keeping it, and what
they cannot tell you. The code is `poker_arena/ratings.py` (the design, the test, the standings) and
`poker_arena/history.py` (all-time standings). The calibration claims below are checked by
`tests/test_calibration.py`: small seeded versions on every test run, and the larger runs reported here with
`POKER_ARENA_HEAVY=1 python -m pytest -s tests/test_calibration.py` (about 8 minutes). The evidence rules (record
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
one match folder. A model remembers everything inside a conversation, so its deals are dependent, and two
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
guarantee: a false leader needs at least one false verdict. A verdict whose diagnostics failed (its
`assumptions` are `unsupported`, below) does not count unless the design preregistered that it should
(`outside_calibration: 'count'`). A leader is a player with a counting decided win over every other player of
its group; a rank is decided when a player has a counting decided win or loss against every other player of its
group (and the verdicts form no cycle). A group whose only would-be leader relies on a verdict whose diagnostics
failed has `leader: null` and names it in `leader_outside_calibration`.

Everything else is **descriptive**: strengths and their intervals, the order by points, the group's front
runner and its margin, session Elo, style statistics, every all-time number. These are recomputed on every
refresh, would answer differently at a different moment, and are never called a result. Players carry
`rank_kind: 'decided' | 'descriptive'`, `inferential` (whether results involving them can be decided) and a
`position` (order by points); each group carries a `leader` (inferential, or null) next to a `front_runner`
(descriptive); each pairing carries `inferential`.

## The guarantee, and its assumptions

With no edge between two players, the procedure calls a winner for that pairing with chance at most
`alpha_pairing = alpha / n_pairings`, so that over the tournament the chance of any false verdict is at most
`alpha` (5% by default; Bonferroni, which needs no independence between pairings). That holds:

1. **Exactly, whenever each conversation's result is symmetric about the pairing's true edge**, whatever its
   shape, tails or discreteness and whatever the dependence between the deals inside it; conversations must be
   independent of each other (fresh memories), not identically distributed. This covers the sharp null of two
   interchangeable players (swapping their names flips the sign of every conversation). It comes from the
   sign-flip arm below, and needs no simulation.
2. **In simulation, for skewed results in a stated family**: card noise plus a conversation effect of normal,
   exponential (skewness 2), lognormal (skewness 6.2) or *rare large loss* shape (+1/3 nine times in ten, −3
   otherwise: the round 3 reviewer's +50 / −450 conversation distribution in standard units), holding up to
   **half** of the variance of a conversation's mean. With 50 deals and a per-deal spread of 350 that is a
   conversation effect with a spread up to about 50 bb/100, an ICC up to about 2%, the size of the pessimistic
   planning scenario. Per-deal results of any shape (heavy tails, exact zeros, skew) are inside it: the mean of
   50 deals is close to normal. The t arm's bar is calibrated on these shapes, so this part carries Monte Carlo
   error (100,000 sequences per shape, target reduced by two standard errors).
3. **Not beyond that.** A pairing whose conversations are mostly small wins and now and then a large loss, with
   the conversation effect larger than the card noise, looks like a steady winner until a loss shows up. No test
   of this size can tell the two apart: a mean is not testable without some assumption on the shape (Bahadur and
   Savage, 1956). The cases measured below show how fast the error grows there.

The minimum number of conversations and the zero-spread rule exist so that degenerate data never decide: two
conversations, or any number of conversations with identical results, give no verdict.

**Every verdict reports diagnostics against these assumptions** (`assumptions: {status, summary, checks}`,
recorded with the decision): `conversation_share` (the estimated share of the conversations' spread that their
cards don't explain) against the calibrated limit of 0.5, `independent_deals` (no deal repeated across the
pairing's conversations) and `conversations` (enough of them to judge). A diagnostic can show that data are
outside the family; it cannot show that they are inside it (a rare large swing not yet seen leaves no trace).
So the words are: `supported` is *diagnostics passed; distribution assumptions unverified*, `unsupported` is
*diagnostics failed: the data are outside the calibrated family* (the bound then holds only if the results are
symmetric, which no data can show), `unknown` is *diagnostics incomplete*. A decided verdict carries `calibrated`
(true, false or null) and a `standing`: `decided`, `decided (diagnostics failed: outside the calibrated family)`
or `decided (diagnostics incomplete)`. Nothing here states the error bound without these conditions.

## The adjudicator: registration, eligibility and official decisions

**Registration.** `ratings.registration(design)` is the artifact the runner writes into plan.json before the
first hand: `{status: 'registered', design, judge_fingerprint, calibration_id, created}`. The judge fingerprint
is the hash of `ratings.py` (the one the runner records for it); the calibration id hashes the calibrated levels
and every constant of the test. `ratings.registration_status(plan, evidence)` never recomputes it from the
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
| audit complete | a used reply whose turn could not be checked for tool use: `opaque` (the `command` backend) unless `allow_opaque` names the backend (default none), or unverified or unrecorded |
| decision coverage | (from the runner's facts) a model action without exactly one decision record and its calls, decision records without an action, or no recorded actions at all |
| protocol consistent | the player's settings differ between the pairing's matches, or a match records no protocol |
| record verified | a counted hand does not replay from its seed (below), the runner could not verify a match record, or the hands carry no evidence hash |
| data integrity | a duplicate pair failed validation (below), or the pairs were not validated at all; no verified deal at all |

**Duplicate pairs are validated, and every counted hand is replayed** before it counts (`ratings.check_pair`
and `ratings.verify_play`, used by `load_deals`, `pairing_data`, the ratings and the history): b-first must seat
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

**Evidence hash.** `pairing_data` returns `evidence_hash`, the hash of the verified hands it judged (each deal's
conversation, index and the digests of its two plays: hand record and events), and `adjudicate` records it in
the decision (`judged_evidence_hash`; the runner also records its own hash of the match records as
`evidence_hash`). Without verified evidence (`evidence_hash` missing) a pairing is ineligible.

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
| `round_deals` | the plan's `deals` (50) | deals per conversation |
| `replicates` | 1 in a plan, 4 from `plan` on the command line | conversations per pairing per round |
| `max_rounds` | from the power target | rounds (looks) at most; `max_rounds_basis`: `power`, `given` or `budget` |
| `budget_deals` | none | a cap on deals per pairing; a plan it cuts short is `underpowered` |
| `n_pairings`, `alpha_pairing` | | Bonferroni: each pairing is tested at `alpha / n_pairings` |
| `nominal_alpha` | | the level each look's t arm is tested at (below) |
| `min_conversations` | at least 8 | the fewest conversations at which a look can decide |
| `target_edge_bb100` | 50 | the edge the plan is powered for, fixed before play |
| `prior_sd_bb100` | 350 | the per-deal spread assumed when planning |
| `assumed_icc` | 0.01 | planning: the share of the per-deal variance that is a conversation effect |
| `assumed_ar1` | 0.2 | planning: AR(1) memory between consecutive deals of a conversation |
| `sd_conversation_bb100` | | the spread of a conversation's mean under those assumptions |
| `power_scenarios` | | per scenario (optimistic, assumed (moderate), pessimistic dependence): `power`, the chance at the target that the better player gets a decided verdict that **counts** (the complete adjudicator, simulated: below), `power_decided` (decided, whatever the diagnostics) and `conversations_used` |
| `power_at_max` | | the moderate scenario's counting power |
| `power_by_edge` | | counting power at 20, 50 and 100 bb/100, moderate dependence |
| `power_basis` | | says in words what the power numbers are |
| `leader_chances` | | for a full round robin: the chance that the true best player is named the decided leader, per edge scenario (below) |
| `equivalence_alpha` | | the two-sided level of the repeated interval behind `even` (each side at half of it), calibrated side by side |
| `deals_needed`, `conversations_needed` | | per pairing, for 80% counting power under the moderate scenario (None: out of reach of MAX_ROUNDS) |
| `deals_planned`, `max_deals` | | the deals per pairing the design allows |
| `allow_unattested` | `['codex']` | backends whose players may be eligible without attesting the model that answered |
| `allow_opaque` | `[]` | backends whose players may be eligible although their turns cannot be audited for tool use |
| `outside_calibration` | `'exclude'` | whether a decided verdict whose diagnostics failed counts towards a decided rank or leader (`'count'`) |
| `registered` | | only `false`, for a plan that declines registration or an older stopping rule; whether a run is registered comes from `registration_status`, never from the design |
| `calibration` | | the null shapes the t arm is calibrated for, and each one's level |

`python -m poker_arena plan` writes `max_rounds` (and its basis) into the plan, so a stored plan reads back the
same design even if the planning defaults change later.

## Planning: the complete adjudicator, simulated

The test's unit is the conversation, so a plan's size comes from the spread of a conversation's mean, not of a
deal. With per-deal spread σ, ICC ρ_c (the share of σ² that is a conversation effect) and AR(1) memory φ between
consecutive deals,

    sd(conversation mean)² = ρ_c σ² + (1 − ρ_c) σ² · DE(φ, R) / R,     DE(φ, R) = 1 + 2 Σ_{k<R} (1 − k/R) φ^k

Three scenarios are reported for every design:

| scenario | ICC | AR(1) | sd of a 50-deal conversation mean (σ = 350) |
|---|---|---|---|
| optimistic | 0 | 0 | 49.5 bb/100 |
| moderate (assumed, `assumed_icc`, `assumed_ar1`) | 0.01 | 0.2 | 69.5 bb/100 |
| pessimistic (ICC doubled, AR(1) + 0.1) | 0.02 | 0.3 | 82.8 bb/100 |

**What the power numbers are.** A plan's power is not the t arm's: it is the chance that the better player gets
a decided verdict that **counts** towards a decided rank or leader, from `ratings.procedure`, which simulates
the complete adjudicator on an eligible pairing (20,000 pairings per number): conversations drawn from the
scenario (normal deals with that memory and conversation effect; block sums drawn from their exact joint law, so
the diagnostic sees what `decide` would compute), looked at round by round as the runner does, each look
deciding only with enough conversations and a spread, the t arm against its calibrated bar, `even` by its own
calibrated interval, and the diagnostics a verdict needs to count (conversation share within the calibrated
0.5). The sign-flip arm is taken to agree whenever the t arm crosses (its spend makes it the looser of the two:
on the same simulated pairings the planner's verdicts and counting decisions agree with `adjudicate`, called
look by look, on 1,000 of 1,000). By default `max_rounds` is the fewest rounds at which the moderate scenario
reaches 80% counting power at the target; `--budget-deals` caps it (`max_rounds_basis: 'budget'`),
`--max-rounds` fixes it (`'given'`), and either way a plan below 80% is marked `underpowered: true`.

**Decided leader.** For a full round robin the design also reports `leader_chances`: the chance that the true
best player is named the decided leader (a counting win over each other player), when it is better than each
other player by the target and they are equal (`one best`), better by twice the target, or when the players are
evenly spaced the target apart (`ladder`). Each of the leader's pairings comes from `procedure`; they are treated
as independent. Pairings that share cards are positively correlated, which keeps the joint chance at or above
the product of the pairings' own chances (checked in a joint simulation).

**The default 3-model design** (3 pairings, 50-deal conversations, 4 per round, target 50 bb/100):

| | value |
|---|---|
| rounds, deals per pairing at most | 10 rounds, 2,000 deals (4,000 hands); 12,000 hands for the tournament |
| first look that can decide | round 3 (12 conversations); bars 4.15, 3.83, 3.67, 3.57, 3.50, 3.45, 3.42, 3.39 at looks 3 to 10 |
| counting power at 50 bb/100: optimistic / moderate / pessimistic | 98.7% / 83.2% / 60.1% (decided at all: 99.8% / 89.2% / 71.4%) |
| through `adjudicate` itself, look by look (4,000 simulated pairings each) | 98.9% / 82.4% / 59.0% (decided 99.9% / 88.6% / 71.1%); moderate with poker-like deals (40% exact zeros, t3 tails): 82.8% |
| counting power by edge, moderate: 20 / 50 / 100 / 200 bb/100 | 11.0% / 83.2% / 84.2% / 81.5% |
| average conversations used, moderate: no edge / 50 / 100 / 200 bb/100 | 39.9 / 25.8 / 13.2 / 12.0 |
| decided leader is the true best: one best by 50 / by 100 / ladder 50 apart | 69.3% / 71.0% / 70.1% |
| the same, one best by 50, simulated jointly with shared cards | 82.1% jointly, equal to the product of its two pairings' own chances (91.1% × 90.1%), with a shared-card component holding 36% of each deal's variance |
| a 600-deal budget (`--budget-deals 600`) | 3 rounds; counting power 62% / 29% / 18%: underpowered |

**The 9-model design** (a round robin of 9 players: 36 pairings, each at alpha / 36 = 0.0014; same defaults):

| | value |
|---|---|
| rounds, deals per pairing at most | 15 rounds, 3,000 deals; 216,000 hands for the tournament |
| first look that can decide | round 4 (16 conversations) |
| counting power at 50 bb/100: optimistic / moderate / pessimistic | 99.8% / 82.7% / 53.3% (decided at all: 99.9% / 85.2% / 59.8%) |
| through `adjudicate`, moderate (1,500 simulated pairings) | 82.0% (decided 84.5%) |
| counting power by edge, moderate: 20 / 50 / 100 / 200 bb/100 | 2.5% / 82.7% / 89.1% / 84.6% |
| decided leader is the true best: one best by 50 / by 100 / ladder 50 apart | 21.8% / 39.8% / 27.0% |

**Why counting power stops near 85% however large the edge.** A large edge is decided at the first look that can
decide (12 conversations in the 3-model design, 16 in the 9-model one), and the pairing stops there. The
conversation-share diagnostic is then estimated from 11 (or 15) degrees of freedom between conversations: with
the moderate scenario's true share of about 0.25 it reads above the calibrated 0.5 about one time in six, and a
verdict whose diagnostics failed does not count. So a clear winner is decided almost surely but counts only
about 84% of the time, and a decided leader in a large round robin, which needs all of its 8 pairings to count,
is unlikely even when one model is far better (40% at twice the target with 9 players). Three ways out, none
taken here because each changes the preregistered procedure: keep a decided pairing playing until its
diagnostics have more conversations (the runner stops on the verdict today; continuing would not change the
false-winner rate, which is fixed at the crossing), raise the minimum number of conversations, or preregister
`outside_calibration: 'count'` and accept the larger error outside the family. The numbers above are what the
default does.

The round 3 default (2 conversations a round, 6 rounds, 600 deals) advertised 53% power; its counting power
under the moderate scenario is far below that.

## The test

After each round a pairing's verdict comes from `ratings.decide`, inside `ratings.adjudicate`, called by the
runner with every validated deal so far and the conversation of each deal. A look can decide only when it has at least `min_conversations`
conversations and they differ at all (a look whose conversations all have the same mean gives no verdict,
`reason` says so).

**t arm.** The conversation-level t: the deal-weighted mean of `x` over the pairing's conversations, divided by
a cluster-robust (CR1) standard error with each conversation as one cluster, on C − 1 degrees of freedom (for
conversations of equal length, the one-sample t-test on conversation means). Every look is tested at the same
nominal level α′ (a Pocock design on the p-value scale, so the bar follows each look's degrees of freedom). α′
is found by simulation: for each shape of the family, the largest level at which at most `alpha_pairing`
(less two Monte Carlo standard errors) of 100,000 null sequences cross at any look; α′ is the smallest of those.
The rare-large-loss shape sets it: for the default design α′ = 0.0016 (normal alone would allow 0.0039), and the
bars at looks 3 to 10 are 4.15, 3.83, 3.67, 3.57, 3.50, 3.45, 3.42 and 3.39.

**Sign-flip arm.** A randomization test of the same statistic. Under the null that each conversation's total
is symmetric about zero, its sign is a fair coin independent of everything else, so the observed signs are
exchangeable with 9,999 random sign vectors (fixed seed, one column per conversation). The arm's error is spent
look by look along the t arm's own crossing curve, rescaled to end at `alpha_pairing`: at each look the
threshold is the lowest that keeps the sign vectors rejected so far, the observed one included, within the
spend. By exchangeability the chance that the observed vector is rejected by the last look is at most
`alpha_pairing`, exactly. A look at which this arm cannot reject at all (with C conversations the most extreme
signs have chance 2^(1−C)) is not used by either arm, which is why the default design's first deciding look has
12 conversations, not 8.

**A winner needs both arms** at the same look (the t arm crossing its bar, the sign-flip arm having rejected by
then). The chance that both happen is at most the smaller of the two arms' chances, so the guarantee holds
whenever either arm's assumptions hold: symmetry, or the calibrated family. The sign-flip arm costs no measurable
power at the target (89.3% decided with both arms, 89.3% with the t arm alone, 1,000 sequences).

**Verdicts.**

| verdict | meaning |
|---|---|
| `a` / `b` | both arms rejected at a planned look: that player is better. Decided. `decided_at_look` says when. |
| `even` | The repeated interval at `equivalence_alpha` lies inside ±15 bb/100 and both one-sided sign-flip tests against ±15 reject: an equivalence test (TOST) at every look, each side calibrated to at most alpha_pairing for every shape of the family (below). Decided. With a spread near 350 bb/100 it is out of reach of the default design. |
| `exhausted` | The last planned look came without a decision. **Not evidence that the two are equally good.** |
| `open` | Undecided and looks remain, or a round is under way. |
| `early` | A completed look with fewer than `min_conversations` conversations (whatever the numbers say), or too few deals for any spread. |
| `ineligible` | An eligibility rule failed (see *The adjudicator*). Terminal: the pairing stops. `test_verdict` keeps what the test said. |
| `retrospective` | A decided result that no official record backs (today's recomputation only). Decides nothing. |
| `descriptive` | No test applies: pooled across runs, or a run that is not preregistered. |

`ci95` is the ordinary 95% interval at the look; `ci_decision` is the edge ± the bar × the standard error, the
interval that matches the t arm; `signflip` says whether the sign-flip arm has rejected. `conversation_share` is
an estimate of how much of the conversations' spread their card noise does not explain (between-conversation
variance beyond what each conversation's own deals show: the variance of a conversation's mean is estimated
from its consecutive blocks of 10 deals, so memory between nearby deals counts as the card noise it is; round 5
ignored memory and so read the moderate scenario's 0.25 as about 0.5); above one half the verdict's diagnostics
fail (`assumptions: unsupported`) and the pairing is outside the calibrated family unless its results are
symmetric.

### Calibration

Null false-winner rate per pairing under the default 3-model design (`alpha_pairing` = 0.0167, 10 looks). The t
arm alone, vectorized, 200,000 sequences each (Monte Carlo standard error about 0.0002 to 0.0003); conversations
of 50 deals with a per-deal spread of 350; "share" is the conversation effect's share of a conversation mean's
variance.

| per-deal results | memory | conversation effect | false winners (t arm) |
|---|---|---|---|
| normal | none | none | 0.0072 |
| t, 3 df | none | none | 0.0067 |
| poker-like (40% exact zeros, t3 otherwise) | none | none | 0.0062 |
| skewed (centred exponential) | none | none | 0.0078 |
| normal | AR(1) 0.2 | normal, share 0.25 | 0.0070 |
| poker-like | AR(1) 0.3 | normal, share 0.5 | 0.0069 |
| normal | none | exponential, share 0.5 | 0.0099 |
| normal | none | lognormal, share 0.5 | 0.0104 |
| poker-like | AR(1) 0.2 | rare large loss, share 0.4 | 0.0116 |
| normal | none | rare large loss, share 0.5 (the calibrating worst case) | **0.0156** |

Through `ratings.decide` itself, both arms, as the runner calls it: poker-like, AR(1) 0.3, exponential effects
at share 0.5, 3,000 sequences: 0.0087. A symmetric two-point conversation effect holding 90% of the variance
(far outside the family, but symmetric): 0.0087 through `decide` (3,000 sequences), as the sign-flip arm
guarantees.

**The round 3 counterexamples.** Two conversations of constant +50 used to give a decided win with zero
uncertainty; now they give `early`, and any number of identical conversations gives no verdict. The zero-mean
distribution in which every deal of a conversation is +50 nine times in ten and −450 otherwise gave a false
first-look winner 82.9% of the time with the round 3 code (reproduced: 1,000 pairings); now 0 of 200 at that
look and 0 of 3,000 over the whole design.

**Beyond the family** (false winners per pairing; the t arm alone with 200,000 sequences, both arms through
`decide` with 2,000):

| conversation effect | t arm | both arms | × alpha_pairing |
|---|---|---|---|
| exponential, share 0.8 | 0.0223 | 0.0160 | 1.0 to 1.3 |
| lognormal, share 0.8 | 0.0323 | 0.0310 | 1.9 |
| rare large loss, share 0.8 | 0.0867 | 0.0875 | 5.2 |
| rare large loss, share 0.9 | 0.1812 | 0.1735 | 10 |
| the reviewer's +50 / −450 conversations plus ordinary card noise (share 0.90) | 0.1849 | 0.1875 | 11 |
| symmetric two-point, share 0.9 | 0.0120 | 0.0070 | within |

This is the worst case, stated plainly: if one conversation in ten of a pairing can be a loss nine times the
usual win, and conversations differ by more than their cards explain, a verdict from a few dozen conversations is
not reliable at the stated level, and nothing at this sample size can make it so.

**Shared cards.** Three pairings among three players on the same slates, with a shared deal effect loaded on each
player's style so that pairings with a common player are correlated, plus memory and conversation effects: the
chance of any false verdict in the tournament is 0.020 (target 0.05, 25,000 tournaments).

### Equivalence: false "even"

`even` is a false verdict whenever the true edge is at or beyond ±15 bb/100. Under the default prior spread
(350 per deal) `even` is out of reach of the default design (the planner gives 0.0% at a true edge of 0), so
these runs use a per-deal spread of 100, at which a true edge of 0 is called `even` 99.4% of the time (t arm and
planner agree). False-even rates per pairing, default 3-model design (target `alpha_pairing` = 0.0167), t arm
of `decide` with its stopping order (a winner first, then the interval), 200,000 sequences each; "mirrored" flips
the skew (rare large wins instead of rare large losses):

| conversation effect (share 0.5) | +15 | −15 | +16 | +20 | mirrored +15 | mirrored −15 | mirrored +16 |
|---|---|---|---|---|---|---|---|
| none (normal deals) | 0.0024 | 0.0024 | 0.0009 | 0.0000 | 0.0025 | 0.0025 | 0.0010 |
| exponential | 0.0082 | 0.0009 | 0.0043 | 0.0003 | 0.0009 | 0.0076 | 0.0003 |
| lognormal | 0.0090 | 0.0006 | 0.0040 | 0.0002 | 0.0007 | 0.0086 | 0.0002 |
| rare large loss | 0.0003 | **0.0134** | 0.0001 | 0.0000 | **0.0135** | 0.0003 | 0.0070 |

Through `decide` itself (both arms; poker-like deals, rare-large-loss effects at share 0.5, edge −15, 3,000
sequences): 0.0163. Three pairings at the margin (+15, −15, +15) on shared slates with exponential effects: any
false `even` in the tournament 0.0155 (target 0.05).

**What changed.** A skewed shape puts most of its crossings on one side. Round 5 calibrated the interval's level
as a two-sided pair (0.0045), so each one-sided test could use nearly the whole pair's budget: false `even` was
0.025 at the heavy side of the rare-large-loss shape, 1.5 times the target. The level is now calibrated side by
side (`ratings.equivalence_level`: for every shape and its mirror, each side's crossing at most alpha_pairing;
0.0017 two-sided for the default design), and `tests/test_calibration.py` checks both the fix and that the old
level fails.

**Power table per pairing.** Each pairing also reports `power.deals_for_80pct`: deals per pairing for an 80%
chance to detect an edge of 20, 50 or 100 bb/100, from its own observed spread of conversation means
(`se × √n`). A planning number for a future run, not a forecast for this pairing.

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

How often a player wins a 10-deal session (up by more than 1 big blind; closer is a tie), as a Bradley–Terry fit
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

`history.all_time` pools every run in `arena.db` that was played under the same game settings. Those runs had
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

- **The reference bot** (`bot = "reference"`): a fixed, versioned, solid tight-aggressive policy (position-aware,
  equity and pot-odds based, with a fixed bluff frequency). `python -m poker_arena plan` warns when a plan has no
  reference player. When a group has one, each player carries `vs_reference` (its direct edge against it, the
  interval, the deals and the pairing's verdict), the same opponent in every run that includes it; the notes say
  when a group has none.
- **Poker fundamentals** (`python -m poker_arena quiz`, written to `quiz.json`): clear-cut heads-up decision spots
  with computed answers (pot-odds calls and folds, all-in spots against a stated range, by exact or seeded Monte
  Carlo equity), asked of each player through its own backend. The ratings output carries them as
  `fundamentals` (per player: answered, correct, accuracy with a 95% Wilson interval, big blinds lost to the best
  answers, fallbacks, the quiz version and spots hash), joined to the run's players by model and reasoning
  level, and the command line prints them as their own section. They are never pooled with strengths or
  verdicts: a model can be weak at fundamentals and still beat weak opponents, or the reverse.

## Style statistics (tendencies, not quality)

`stats.collect` counts standard poker tracking statistics per hand dealt, under plain names
(`stats.LABELS`, also in the ratings output as `stat_labels`): *Plays preflop* (VPIP: put money in preflop by
choice), *Raises preflop* (PFR), *Aggression after the flop*, *Goes to showdown*, *Wins at showdown* (a split pot
is not a win; *Splits at showdown* counts those), *Talks*. Every rate comes with its counts
(`counts: {rate: {n, of}}`) and a breakdown by opponent (`by_opponent`) and by seat (`by_position`: button, which
posts the small blind heads-up, or big blind; names in `position_labels`). No value of these is good or bad in
itself.

## Limits

- **The shape assumption.** The error rate is exact for symmetric conversation results and calibrated for the
  family above; rare large one-sided swings bigger than the card noise break it (see *Beyond the family*). The
  `conversation_share` estimate and its note flag pairings whose conversations differ more than their cards
  explain, but a rare swing not yet seen cannot be estimated.
- **Few conversations.** No verdict before `min_conversations`; the first model runs had one conversation per
  pairing, so they are descriptions of those matches.
- **The planning assumptions** (per-deal spread, ICC, memory) are guesses until a run measures them; the three
  scenarios show how much they matter, and each pairing's `deals_for_80pct` uses the observed spread. The
  planner draws normal deals; poker-like deals gave the same counting power through `adjudicate`.
- **A decided leader is hard to get.** Counting power stops near 85% however large the edge (see *Planning*), so
  the chance of a decided leader falls fast with the number of players: about 70% for 3 players, 22 to 40% for 9.
- **The record verifier checks what was played, not why.** It shows that the recorded cards, actions and payouts
  are what the engine deals and plays from the seed; it cannot show that a recorded reasoning is the model's.
  Evidence binding is a hash in a file the operator controls, like the registration: it catches edits, missing
  files and records that do not reproduce, not someone who rewrites every record consistently.
- **Independence of conversations** is by design (fresh memories, no shared state); a model provider that
  carried state across conversations would break it.
- **Additivity** of strengths is an assumption (see Strength).
- **Identity is as good as the CLI's report.** A CLI that doesn't report which model answered (Codex) can't be
  checked; it is eligible only because the default design allows it, and it is shown as unattested.
- **Winnings are relative.** Strength and verdicts say how a model does against the players it met. The fixed
  reference bot gives every run a common, versioned opponent, and the quiz (*How good at poker*) scores
  fundamentals against computed answers; neither is a measure of play against strong humans.
- **The registration is a record, not a signature.** It proves the design was fixed before the first match
  started only as far as the files are honest: someone editing plan.json and every match config by hand could
  forge it.
- **One game format.** Heads-up, 100 big blinds deep, stacks reset every hand. Strengths do not carry over to
  other formats.
