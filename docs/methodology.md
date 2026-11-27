# How Poker Arena decides who wins

This page explains how results become verdicts and standings, which of them are tests and which are only
descriptions, under which assumptions the tests keep their error rate, where they stop keeping it, and what
they cannot tell you. The code is `poker_arena/ratings.py` (the design, the test, the standings) and
`poker_arena/history.py` (all-time standings). The calibration claims below are checked by
`tests/test_calibration.py`: small seeded versions on every test run, and the larger runs reported here with
`POKER_ARENA_HEAVY=1 python -m pytest -s tests/test_calibration.py` (about 3 minutes).

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

Exactly one thing is a test: the **preregistered sequential procedure on each pairing**. Its verdicts *a is
better*, *b is better* and *even*, reached at a completed planned look, are the only inferential results, and
only when

- the run has a **preregistered design** (`design.registered: true`): a plan whose design was fixed before play.
  A run played as one fixed round (older plans, `play`), or under an older stopping rule, has
  `design: {registered: false, ...}`: every pairing there is `verdict: 'descriptive'`, nothing is ranked or led,
  and the only sample-size figure, under `design.retrospective`, is labelled *retrospective* (what a default
  design would have needed), never "planned";
- both players' **identities are verified** in that run (see *Who is a player*): a pairing with a player whose
  model is unknown, whose settings changed between matches, or for whom another model answered, is
  `verdict: 'descriptive'` with the reason; the test's own verdict is kept as `test_verdict` for the record.

A **decided rank** and a **decided leader** are built from those verdicts alone, so they inherit their
guarantee: a false leader needs at least one false verdict. A leader is a player with a decided win over every
other player of its group; a rank is decided when a player has a decided win or loss against every other player
of its group (and the verdicts form no cycle).

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
| `power_scenarios` | | power at the target under optimistic, assumed (moderate) and pessimistic dependence |
| `power_at_max` | | the moderate scenario's power |
| `deals_needed`, `conversations_needed` | | per pairing, for 80% power under the moderate scenario |
| `deals_planned`, `max_deals` | | the deals per pairing the design allows |
| `registered` | | false for a design not fixed before play |
| `calibration` | | the null shapes the t arm is calibrated for, and each one's level |

`python -m poker_arena plan` writes `max_rounds` (and its basis) into the plan, so a stored plan reads back the
same design even if the planning defaults change later.

## Planning: sample size with dependence

The test's unit is the conversation, so a plan's size comes from the spread of a conversation's mean, not of a
deal. With per-deal spread σ, ICC ρ_c (the share of σ² that is a conversation effect) and AR(1) memory φ between
consecutive deals,

    sd(conversation mean)² = ρ_c σ² + (1 − ρ_c) σ² · DE(φ, R) / R,     DE(φ, R) = 1 + 2 Σ_{k<R} (1 − k/R) φ^k

and the power of the whole procedure (all looks, the calibrated bars, the minimum) is simulated for an edge of
`target / sd(conversation mean)` per conversation. Three scenarios are reported for every design:

| scenario | ICC | AR(1) | sd of a 50-deal conversation mean (σ = 350) |
|---|---|---|---|
| optimistic | 0 | 0 | 49.5 bb/100 |
| moderate (assumed, `assumed_icc`, `assumed_ar1`) | 0.01 | 0.2 | 69.5 bb/100 |
| pessimistic (ICC doubled, AR(1) + 0.1) | 0.02 | 0.3 | 82.8 bb/100 |

By default `max_rounds` is the fewest rounds at which the moderate scenario reaches 80% power at the target.
`--budget-deals` caps it (`max_rounds_basis: 'budget'`), `--max-rounds` fixes it (`'given'`), and either way a
plan below 80% in the moderate scenario is marked `underpowered: true`, with the deals 80% would need.

**The default 3-model design** (3 pairings, 50-deal conversations, 4 per round, target 50 bb/100):

| | value |
|---|---|
| rounds, deals per pairing at most | 9 rounds, 1,800 deals (3,600 hands); 10,800 hands for the tournament |
| first look that can decide | round 3 (12 conversations) |
| power at 50 bb/100: optimistic / moderate / pessimistic | 99.5% / 84.4% / 65.3% |
| simulated from deals (AR(1) noise and conversation effects, 20,000 sequences) | 99.5% / 84.1% / 65.2% |
| power at 100 bb/100, moderate | >99.9% |
| average conversations used, moderate: no edge / 50 / 100 / 200 bb/100 | 35.9 / 24.8 / 13.1 / 12.0 |
| a 600-deal budget (`--budget-deals 600`, the round 3 default's size) | 3 rounds; power 64% / 31% / 20%: underpowered |

The round 3 default (2 conversations a round, 6 rounds, 600 deals) advertised 53% power; under the moderate
scenario it has 26% with the current test, and the reviewer's 17% under the pessimistic one is consistent with
the 18% computed here.

## The test

After each round a pairing's verdict comes from `ratings.decide`, called by the runner with every deal so far
and the conversation of each deal. A look can decide only when it has at least `min_conversations`
conversations and they differ at all (a look whose conversations all have the same mean gives no verdict,
`reason` says so).

**t arm.** The conversation-level t: the deal-weighted mean of `x` over the pairing's conversations, divided by
a cluster-robust (CR1) standard error with each conversation as one cluster, on C − 1 degrees of freedom (for
conversations of equal length, the one-sample t-test on conversation means). Every look is tested at the same
nominal level α′ (a Pocock design on the p-value scale, so the bar follows each look's degrees of freedom). α′
is found by simulation: for each shape of the family, the largest level at which at most `alpha_pairing`
(less two Monte Carlo standard errors) of 100,000 null sequences cross at any look; α′ is the smallest of those.
The rare-large-loss shape sets it: for the default design α′ = 0.0018 (normal alone would allow 0.0043), and the
bars at looks 3 to 9 are 4.09, 3.79, 3.63, 3.53, 3.46, 3.42 and 3.38.

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
power at the target (84.4% with both arms, 84.4% with the t arm alone, 1,000 sequences).

**Verdicts.**

| verdict | meaning |
|---|---|
| `a` / `b` | both arms rejected at a planned look: that player is better. Decided. `decided_at_look` says when. |
| `even` | The repeated (1 − 2·alpha_pairing) interval lies inside ±15 bb/100 and both one-sided sign-flip tests against ±15 reject: an equivalence test (TOST) at every look. Decided. With a spread near 350 bb/100 it needs thousands of deals. |
| `exhausted` | The last planned look came without a decision. **Not evidence that the two are equally good.** |
| `open` | Undecided and looks remain, or a round is under way. |
| `early` | A completed look with fewer than `min_conversations` conversations (whatever the numbers say), or too few deals for any spread. |
| `descriptive` | No test applies: pooled across runs, a run without a design, or a player whose identity is in doubt. |

`ci95` is the ordinary 95% interval at the look; `ci_decision` is the edge ± the bar × the standard error, the
interval that matches the t arm; `signflip` says whether the sign-flip arm has rejected. `conversation_share` is
an estimate of how much of the conversations' spread their card noise does not explain (between-conversation
variance beyond the within-conversation one, ignoring memory, so it errs high); above one half the notes say the
pairing is outside the calibrated range unless its results are symmetric. When a run's `decisions.json` recorded
a different verdict (an earlier version of the test), the pairing shows `recorded_verdict` and the notes say so.

### Calibration

Null false-winner rate per pairing under the default 3-model design (`alpha_pairing` = 0.0167). The t arm alone,
vectorized, 200,000 sequences each (Monte Carlo standard error about 0.0002 to 0.0003); conversations of 50 deals
with a per-deal spread of 350; "share" is the conversation effect's share of a conversation mean's variance.

| per-deal results | memory | conversation effect | false winners (t arm) |
|---|---|---|---|
| normal | none | none | 0.0069 |
| t, 3 df | none | none | 0.0067 |
| poker-like (40% exact zeros, t3 otherwise) | none | none | 0.0063 |
| skewed (centred exponential) | none | none | 0.0079 |
| normal | AR(1) 0.2 | normal, share 0.25 | 0.0072 |
| poker-like | AR(1) 0.3 | normal, share 0.5 | 0.0067 |
| normal | none | exponential, share 0.5 | 0.0097 |
| normal | none | lognormal, share 0.5 | 0.0107 |
| poker-like | AR(1) 0.2 | rare large loss, share 0.4 | 0.0110 |
| normal | none | rare large loss, share 0.5 (the calibrating worst case) | **0.0158** |

Through `ratings.decide` itself, both arms, as the runner calls it: poker-like, AR(1) 0.3, exponential effects
at share 0.5, 3,000 sequences: 0.0093. A symmetric two-point conversation effect holding 90% of the variance
(far outside the family, but symmetric): 0.0150 through `decide` (3,000 sequences), as the sign-flip arm
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
| exponential, share 0.8 | 0.0221 | 0.0230 | 1.4 |
| lognormal, share 0.8 | 0.0318 | 0.0305 | 1.8 |
| rare large loss, share 0.8 | 0.0876 | 0.0810 | 4.9 |
| rare large loss, share 0.9 | 0.1836 | 0.1670 | 10 |
| the reviewer's +50 / −450 conversations plus ordinary card noise (share 0.90) | 0.1850 | 0.1965 | 12 |
| symmetric two-point, share 0.9 | 0.0118 | 0.0130 | within |

This is the worst case, stated plainly: if one conversation in ten of a pairing can be a loss nine times the
usual win, and conversations differ by more than their cards explain, a verdict from a few dozen conversations is
not reliable at the stated level, and nothing at this sample size can make it so.

**Shared cards.** Three pairings among three players on the same slates, with a shared deal effect loaded on each
player's style so that pairings with a common player are correlated, plus memory and conversation effects: the
chance of any false verdict in the tournament is 0.020 (target 0.05, 25,000 tournaments).

**Power table per pairing.** Each pairing also reports `power.deals_for_80pct`: deals per pairing for an 80%
chance to detect an edge of 20, 50 or 100 bb/100, from its own observed spread of conversation means
(`se × √n`). A planning number for a future run, not a forecast for this pairing.

## Strength (descriptive)

Each player gets a strength in bb/100 from a least-squares fit over all observations, `x(a vs b) ≈ s_a − s_b`,
with the strengths summing to zero within each group of connected players: the edge against the average
opponent of the group. The fit assumes edges add up across players, which poker's rock-paper-scissors effects can
break; when every pairing is the only link between its players the assumption cannot be checked, and the notes
say so. The per-pairing edges never depend on it.

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

A model player is `backend:model:effort#protocol`, for example `claude:claude-sonnet-5-5:high#1a2b3c4d`. The
protocol is the first 8 characters of the fingerprint the runner records for each seat: a hash of every
setting that changes play (backend, model, effort, isolation, instructions or system prompt, tool policy, extra
arguments, retries and fixes, thinking display) and of the match horizon and memory schedule (hands per
conversation, which the model is told; a fresh conversation at every match). A protocol recorded before the
horizon was part of it gets the horizon in the identity instead (`...#1a2b3c4d~h50`), so conversations of
different lengths never pool. A player from before protocols were recorded is kept apart from later runs.

**Which model answered.** Where the CLI reports the model that answered (`served_models`), it must be the
configured one: the same canonical id (lower case, no provider prefix, punctuation and spaces as hyphens, so
"Gemini 4 Neon (High)" is `gemini-4-neon-high`), a dated snapshot of it (`<id>-YYYYMMDD` or `<id>-YYYY-MM-DD`),
or an id listed for it in `ratings.SERVED_ALIASES` (empty by default). Nothing else matches: `claude-sonnet-5`
is not `claude-sonnet-5-5`, and `gemini-4-neon-high-0912` is not `gemini-4-neon`.

A player is **quarantined** when its identity is unresolved (no model recorded, or a bare alias such as `sonnet`
whose version changes over time), when its settings differ between the matches of one run, or when another model
answered. Within a run a quarantined player stays in the fit with a flag (`in_fit: true`, `quarantined: true`,
`inferential: false`), and every pairing it plays is descriptive. In the all-time standings it is left out of
the fit (`in_fit: false`), listed with its hands and net result; a served-model mismatch quarantines only that
run's deals, under their own identity (`...!served=<model>`).

## All-time standings (descriptive only)

`history.all_time` pools every run in `arena.db` that was played under the same game settings. Those runs had
different designs and stopped pairings at different times, so pooling them is no one's preregistered test:
every pairing there has `verdict: 'descriptive'`, there is no leader and no rank, and the notes say so. Repeated
deals are averaged; slates are matched by seed across runs, so runs on the same cards are resampled together.
Independent confirmation needs a new run with its own design.

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
  scenarios show how much they matter, and each pairing's `deals_for_80pct` uses the observed spread.
- **Independence of conversations** is by design (fresh memories, no shared state); a model provider that
  carried state across conversations would break it.
- **Additivity** of strengths is an assumption (see Strength).
- **Identity is as good as the CLI's report.** A CLI that doesn't report which model answered can't be checked.
- **One game format.** Heads-up, 100 big blinds deep, stacks reset every hand. Strengths do not carry over to
  other formats.
