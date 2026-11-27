# How Poker Arena decides who wins

This page explains how results become verdicts and standings, which of them are tests and which are only
descriptions, how far they can be trusted, and what they cannot tell you. The code is `poker_arena/ratings.py`
(the design, the test, the standings) and `poker_arena/history.py` (all-time standings). The calibration claims
below are checked by `tests/test_calibration.py`: small seeded versions on every test run, and the larger runs
reported here with `POKER_ARENA_HEAVY=1 python -m pytest -s tests/test_calibration.py`.

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
model formed early, a style it settled into). The conversation is therefore the independent unit.

**A slate** is the set of deals that one round and replicate deals to every pairing (the same seeds). The
conversations on one slate share their cards, so a deal that suits an aggressive style moves every pairing
with an aggressive player the same way: results of different pairings on one slate are correlated.

**Repeated deals.** A deal is identified by its fingerprint: a hash of the game settings that shape it, the seed
and the seat holding the button (`ratings.fingerprint`). When a pair has played the same fingerprint more than
once (only possible across runs), those plays are averaged into one observation.

## What is a test and what is a description

Exactly one thing is a test: the **preregistered sequential procedure on each pairing**. Its verdicts *a is
better*, *b is better* and *even*, reached at a completed planned look, are the only inferential results. The
chance of any false verdict in a whole tournament is at most `alpha` (5% by default).

A **decided rank** and a **decided leader** are built from those verdicts alone, so they inherit the same
guarantee: a false leader needs at least one false verdict. A leader is a player with a decided win over every
other player of its group; a rank is decided when a player has a decided win or loss against every other player
of its group (and the verdicts form no cycle).

Everything else is **descriptive**: strengths and their intervals, the order by points, the group's front
runner and its margin, session Elo, style statistics, and every all-time number. These are recomputed on every
refresh, would answer differently at a different moment, and are never called a result. The output labels them:
players carry `rank_kind: 'decided' | 'descriptive'` and a `position` (order by points); each group carries a
`leader` (inferential, or null) next to a `front_runner` (descriptive).

## The preregistered design

`ratings.normalize_design(plan)` turns a plan into the one design that the runner plays by, the ratings judge by
and the viewer shows (nothing else builds one):

| key | default | meaning |
|---|---|---|
| `alpha` | 0.05 | chance of any false verdict over the tournament |
| `round_deals` | the plan's `deals` (50) | deals per conversation |
| `replicates` | 1 in a plan, 2 from `plan` on the command line | conversations per pairing per round |
| `max_rounds` | 6 | rounds (looks) at most |
| `n_pairings`, `alpha_pairing` | | Bonferroni: each pairing is tested at `alpha / n_pairings` |
| `nominal_alpha` | | the level each look is tested at (below) |
| `target_edge_bb100` | 50 | the edge the plan is powered for, fixed before play |
| `prior_sd_bb100` | 350 | the per-deal spread assumed when planning |
| `deals_needed` | | deals per pairing for 80% power at the target (in conversations of `round_deals`) |
| `deals_planned`, `max_deals` | | the deals needed rounded up to whole rounds and capped at `max_rounds × round_deals × replicates`, and that cap |
| `power_at_max` | | the simulated chance that the whole procedure calls the better player at the target edge |

`python -m poker_arena plan` prints the target in words and warns when the plan cannot reach 80% power, with the
`--max-rounds` that would. A plan with no design (older runs) has no target; its ratings say so.

## The test

After each round a pairing's verdict comes from `ratings.decide`, called by the runner with every deal so far and
the conversation of each deal.

**Statistic.** The conversation-level t: the deal-weighted mean of `x` over the pairing's conversations, divided
by a cluster-robust (CR1) standard error with each conversation as one cluster, on C − 1 degrees of freedom. For
conversations of equal length this is exactly the one-sample t-test on conversation means. It makes no
assumption about dependence inside a conversation (memory of any length is fine) and it counts how much
conversations differ from each other, because that is what it measures. A verdict needs at least **2
conversations**: from one conversation nothing can be said about how a new match would go, so the verdict
stays open and the output says why (`reason`).

**Repeated looks.** Looking after every round gives a false winner more chances. Every look is tested at the
same nominal level α′ (a Pocock design on the p-value scale, so the bar follows each look's degrees of freedom),
and α′ is found by simulating 400,000 null sequences (fixed seed, cached) so that over all planned looks a
pairing with no edge is called with chance `alpha_pairing`. With 3 pairings, 50-deal rounds, 2 replicates and 6
rounds, α′ = 0.0036 and the bars at looks 1 to 6 are 176, 8.3, 5.2, 4.3, 3.9 and 3.7. A look with one
conversation has no bar.

**Verdicts.**

| verdict | meaning |
|---|---|
| `a` / `b` | \|t\| reached the bar at a planned look: that player is better. Decided. |
| `even` | The repeated (1 − 2·alpha_pairing) interval for the edge lies inside ±15 bb/100: an equivalence test (TOST) at every look. Decided. With a spread near 350 bb/100 it needs thousands of deals. |
| `exhausted` | The last planned look came without a decision. **Not evidence that the two are equally good.** |
| `open` | Undecided and looks remain, or a round is under way, or only one conversation. |
| `early` | Too few deals to estimate any spread. |

`ci95` is the ordinary 95% interval at the look; `ci_decision` is the edge ± the bar × the standard error, the
interval that matches the verdict. When a run's `decisions.json` recorded a different verdict (an earlier
version of the test), the pairing shows `recorded_verdict` and the notes say so.

### Calibration

Null false-winner rate per pairing, design as above (`alpha_pairing` = 0.0167), 200,000 sequences each
(Monte Carlo standard error 0.0003), conversations of 50 deals with a per-deal spread of 350:

| per-deal results | memory inside a conversation | conversation effect (sd, in standard errors of a conversation mean) | false winners |
|---|---|---|---|
| normal | none | 0 | 0.0163 |
| t, 3 df | none | 0 | 0.0153 |
| poker-like (40% exact zeros, t3 otherwise) | none | 0 | 0.0143 |
| normal | AR(1), ρ = 0.3 | 0 | 0.0168 |
| normal | none | 1.0 | 0.0166 |
| poker-like | AR(1), ρ = 0.3 | 0.5 | 0.0149 |
| t, 3 df | none | 2.0 | 0.0169 |

Through `ratings.decide` itself, look by look as the runner calls it (poker-like, ρ = 0.3, conversation effect
1.0, 3,000 sequences): 0.0123.

**Shared cards.** Three pairings among three players on the same slates, with a shared deal effect loaded on each
player's style so that pairings with a common player are correlated, plus memory and conversation effects: the
chance of any false verdict in the tournament is 0.046 (target 0.05, 50,000 tournaments). Bonferroni needs no
independence between pairings.

**Power** (3 pairings, spread 350, the procedure simulated):

| design | max deals per pairing | edge 100 bb/100 | edge 50 bb/100 |
|---|---|---|---|
| 1 replicate, 6 rounds of 50 | 300 | 61% | 14% |
| 2 replicates, 6 rounds of 50 | 600 | 99.6% | 53% |
| 2 replicates, 10 rounds of 50 | 1,000 | 100% | 87% |

The wrong player was never called in these simulations. One replicate per round wastes the first look (one
conversation) and leaves few degrees of freedom, which is why `plan` defaults to 2 replicates.

**Power table per pairing.** Each pairing also reports `power.deals_for_80pct`: deals per pairing for an 80%
chance to detect an edge of 20, 50 or 100 bb/100, from its own observed spread (`se × √n`, so it includes how much
its conversations differ). A planning number for a future run, not a forecast for this pairing.

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
   differ. It is split into aligned blocks of 10 deals (deals 0-9 of every pairing on it together, and so on),
   which are resampled jointly (`ci_basis: blocks`). Such an interval is conditional on the conversations
   played: it leaves out how much fresh matches differ, and the output says so.
3. Each stratum uses the Rao–Wu rescaling bootstrap (k − 1 of its k units drawn with replacement, weighted
   k / (k − 1)), so the bootstrap variance of a linear statistic is unbiased even with two units. Intervals are
   the point ± the Student t quantile on the bootstrap's degrees of freedom (units minus strata, over the strata
   that hold the player's pairings) × the bootstrap standard error. A player behind no resampled unit gets no
   interval. Groups never share a stratum.

**Coverage** of the 95% strength intervals in seeded round robins of 4 players (strengths ±60 and ±20 bb/100,
poker-like noise, AR(1) memory, conversation effects and a shared-card effect through style loadings), 1,000 to
1,600 intervals each:

| setting | coverage |
|---|---|
| 4 slates, conversation effects 0.5, shared cards | 0.954 |
| 4 slates, strong shared cards, joint resampling | 0.954 |
| same, but resampling each pairing independently (what the joint bootstrap replaces) | 0.883 |
| 1 slate (blocks), no conversation effect | 0.961 |
| 1 slate (blocks), AR(1) ρ = 0.3 | 0.952 |
| 1 slate (blocks), conversation effect 1.0 | 0.895 |

The last row is the limit stated above: with a single conversation per pairing, the interval cannot see how much
conversations differ.

## Session Elo (secondary, descriptive)

How often a player wins a 10-deal session (up by more than 1 big blind; closer is a tie), as a Bradley–Terry fit
on the Elo scale with a Gaussian prior (1.5 logits), centred on 1500 within each group, with a Laplace
(posterior) interval that cannot collapse to a point. It can disagree with strength (many narrow wins and a few
heavy losses), and it never decides anything.

## Who is a player

A model player is `backend:model:effort#protocol`, for example `claude:claude-sonnet-5-5:high#1a2b3c4d`. The
protocol is the first 8 characters of the fingerprint the runner records for each seat: a hash of every
setting that changes play (backend, model, effort, isolation, instructions or system prompt, tool policy, extra
arguments, retries and fixes, thinking display). Only identical protocols pool in the all-time standings; a
player from before protocols were recorded is kept apart from later runs.

A player is **quarantined** when its identity is unresolved (no model recorded, or a bare alias such as `sonnet`
whose version changes over time), when its settings differ between the matches of one run, or when the model
that answered (as the CLI reports it) is not the configured one. Within a run a quarantined player stays in the
fit and is flagged (`in_fit: true`, `quarantined: true`). In the all-time standings it is left out of the fit
(`in_fit: false`), listed with its hands and net result; a served-model mismatch quarantines only that run's
deals, under their own identity (`...!served=<model>`). `identity_resolved` says separately whether the exact
model is known.

## All-time standings (descriptive only)

`history.all_time` pools every run in `arena.db` that was played under the same game settings. Those runs had
different designs and stopped pairings at different times, so pooling them is no one's preregistered test:
every pairing there has `verdict: 'descriptive'`, there is no leader and no rank, and the notes say so. Repeated
deals are averaged; slates are matched by seed across runs, so runs on the same cards are resampled together.
Independent confirmation needs a new run with its own design.

## Style statistics (tendencies, not quality)

`stats.collect` counts standard poker tracking statistics per hand dealt. Every rate comes with its counts
(`counts: {rate: {n, of}}`) and a breakdown by opponent (`by_opponent`) and by position (`by_position`: button,
which posts the small blind heads-up, or big blind). At showdown, `won_sd` counts pots won outright; a split pot
is counted as `sd_ties`. No value of these is good or bad in itself.

## Limits

- **Few conversations.** Until a pairing has several conversations its verdict cannot be reached, and its
  descriptive interval may rest on blocks within one conversation (see Coverage). The first model runs had one
  conversation per pairing.
- **Small runs.** A 50 bb/100 edge with a spread of 350 needs about 900 deals per pairing; the first model runs had
  50. Their numbers are descriptions of those matches.
- **The prior spread** behind `deals_needed` is an assumption; the per-pairing power table uses the observed one.
- **Additivity** of strengths is an assumption (see Strength).
- **Identity is as good as the CLI's report.** `served_models` shows what answered where the CLI reports it; a
  CLI that doesn't report it can't be checked.
- **One game format.** Heads-up, 100 big blinds deep, stacks reset every hand. Strengths do not carry over to
  other formats.
