# How Poker Arena decides who wins

This page explains how results become standings and verdicts, what the numbers mean, how far they can be
trusted, and what they cannot tell you. The code is `poker_arena/ratings.py` (standings, verdicts, the
stopping rule) and `poker_arena/history.py` (all-time standings). The calibration claims below are checked by
`tests/test_calibration.py` on every test run.

## The question

Which models, each at a given reasoning level, win at heads-up No-Limit Hold'em, and by how much. The models
are told to win chips, so chips decide. The measure is **bb/100 over duplicate deals**: big blinds won per 100
hands, counted only on deals that both players played from both seats.

## The unit: a duplicate deal

Every matchup is two matches on the same seeds with the seats swapped. Deal *k* is hand *k* of both matches. On
it, each player holds each seat's cards once against the other player, so the luck of the cards largely
cancels. Its value for player *a* against player *b* is

    x = net chips of a over both hands / big blind / 2 × 100

which is in bb/100. In the first model runs, the spread of `x` was about 340 bb/100, with many exact zeros (both
plays ended the same way) and heavy tails.

**Repeated deals.** Every run starts from the same seeds, so the same pair can meet on the same cards in more
than one run. A deal is identified by its fingerprint: a hash of the game settings that shape it (everything but
the number of hands), the seed, and the seat holding the button (`ratings.fingerprint`). When a pair has played
the same fingerprint more than once, those plays are averaged into one observation. A repeat is not new
evidence about how the two play those cards; counting it twice would overstate the sample. Runs older than
the fingerprint field are fingerprinted from their recorded game settings and seeds.

## Strength

Each player gets one number, its **strength** in bb/100. It comes from a least-squares fit over all
observations:

    x(a vs b) ≈ s_a − s_b

with the strengths summing to zero within each group of connected players. So a strength is the edge a player
would have against the average opponent of its group, and the difference of two strengths is the predicted edge
between them. Two players who only played each other get `+edge/2` and `−edge/2`.

The fit assumes edges add up across players (if A beats B by 30 and B beats C by 20, A beats C by about 50).
Poker has rock-paper-scissors effects, so that can fail. When every pairing is the only link between its
players, as in a three-player round robin with one pairing per pair, the fit reproduces each pairing exactly and
the assumption cannot be checked; the output says so in its notes. The per-pairing edges are always reported
too, and they make no such assumption.

**Groups.** Players are connected when a chain of played pairings links them. Players in different groups have
never been compared, directly or indirectly, so their strengths are on separate scales. Ranks, leaders and the
all-runs view are per group. A group can only be joined to another by playing a bridge pairing.

## Uncertainty

A model remembers every earlier hand of a match, so deals within a match are not independent. The resampling
unit is therefore a **block of 10 consecutive observations within one matchup**. A block bootstrap (300
resamples, fixed seed) resamples blocks within each pairing, so every pairing keeps its size and the memory
structure inside each block is kept.

Bootstraps with few clusters underestimate the spread, by a factor of about (G − 1)/G in variance, G being the
number of blocks. So the bootstrap standard error is scaled by √(G/(G − 1)), and the 95% interval uses the
Student t quantile with G − 1 degrees of freedom rather than 1.96. A player behind fewer than 2 blocks, or in a
pairing that has only one block (whose spread cannot be resampled), gets no interval at all, and is separated
from no one. The interval is never a point.

## Ranks and the leader

Every pair of players in a group is tested for a difference in strength (bootstrap standard error, t with
G − 1 degrees of freedom), and the p-values are adjusted for the number of pairs with **Holm's** step-down method.

- A player's `rank` is shown only when it is separated from every other player in its group. Otherwise
  `rank_tied_with` lists the players it is not separated from, and `rank_range` gives the best and worst rank
  the data allow (1 + the number of players significantly better; the group size minus the number
  significantly worse).
- The **leader** of a group is the player separated above every other player. If there is none, the output
  says "No established leader yet". The leader's margin over the next player is reported with its interval.

## Verdicts on pairings, and when to stop

Each pairing gets a verdict from `ratings.decide`. The statistic is the block-mean t: the mean of `x` over
the pairing, divided by a standard error that treats blocks as clusters (exact batch means when blocks are
equal). Blocks restart with each matchup, so no block spans two matches.

A tournament plays a pairing in rounds and looks after each one. Looking repeatedly gives a false winner more
chances, so the bar is a **group-sequential boundary**:

- The plan's `design` is `{alpha: 0.05, round_deals: 50, max_rounds: 6, block: 10}` by default. The family
  error is split over the pairings by Bonferroni, so each pairing is tested at `alpha_pairing = alpha / n_pairings`.
- Each look is tested at the same nominal level α′ (a Pocock design on the p-value scale), so the bar follows each
  look's degrees of freedom: high at the first look, with only 5 blocks, and lower later. α′ is found by
  simulating 500,000 null sequences (normal deals, the plan's round size and block size, fixed seed) and solving
  for the level at which the chance of crossing at any look is `alpha_pairing`. The result is deterministic and
  cached. For 3 pairings and the default design, α′ = 0.0040 and the bars are 5.96, 3.84, 3.44, 3.28, 3.19 and
  3.13 at looks 1 to 6.
- A single planned look (a fixed-length run) uses the plain t quantile at `alpha_pairing`.

The verdicts:

| verdict | meaning |
|---|---|
| `a` / `b` | \|t\| reached the bar at a planned look: that player is better. |
| `even` | The repeated (1 − 2·alpha_pairing) interval for the edge lies inside ±15 bb/100: an equivalence test (TOST) valid at every look. The pairing is shown to be close, not just undecided. |
| `exhausted` | The last planned look came and went without a decision. **This is not evidence that the two are equally good**; it only says the data cannot tell at this size. |
| `open` | Undecided, and looks remain (or a round is in progress). |
| `early` | Fewer than 2 blocks of deals: no spread to judge by. |

With a spread near 350 bb/100, `even` needs thousands of deals, so in practice it is rare. That is intended:
"we could not tell them apart" and "they are about equal" are different claims, and only the second earns `even`.

`ci95` is the ordinary 95% interval at the current data. `ci_decision` is the interval that matches the
verdict (the edge ± the bar × the standard error); it is wider, because it pays for the repeated looks.

### Calibration

`tests/test_calibration.py` plays null pairings (no edge) through `ratings.decide` look by look, exactly as the
runner does, for three kinds of per-deal noise, scaled to a spread of 350 bb/100. With 3 pairings
(`alpha_pairing` = 0.0167) and 24,000 sequences each:

| noise | false winners per pairing | any false winner in a 3-pairing tournament |
|---|---|---|
| normal | 0.0174 ± 0.0008 | 0.051 |
| t, 3 degrees of freedom | 0.0142 ± 0.0008 | 0.043 |
| poker-like (40% exact zeros, t3 otherwise) | 0.0127 ± 0.0007 | 0.037 |

On 1,000,000 fresh normal sequences the per-pairing rate is 0.0168 ± 0.0001 against the target 0.0167. The test
suite checks smaller versions of these (2,400 sequences each) on every run, plus a single-look design and power.

Power with the default design, 3 pairings and a spread of 350 bb/100: a real edge of 100 bb/100 is called for the
right player 97% of the time, an edge of 50 bb/100 34% of the time. The wrong player was never called.

Where the guarantee weakens (measured, not part of the test):

- **Dependence longer than a block.** If each deal's result carries over half of the previous one's (AR(1),
  ρ = 0.5, reset at each new match), the per-pairing false-winner rate is 0.025 instead of 0.0167. At ρ = 0.2 it
  is 0.016. Real per-deal results in the first run showed no positive lag-1 correlation, but that run is short.
- **Skewed results.** With strongly skewed noise (a shifted exponential) the rate is 0.026. For two players of
  equal skill the duplicate result is symmetric by construction, so this matters only when two different
  styles happen to have the same mean.

### Power table

Each pairing also reports `power.deals_for_80pct`: the deals per pairing needed for an 80% chance of detecting a
real edge of 20, 50 or 100 bb/100, from the spread observed so far (the cluster standard error times √n, so it
includes the dependence the blocks see):

    n ≈ ((z(1 − α′/2) + z(0.8)) × spread / edge)²

This is a planning number for a future run. It is not a forecast of how many more deals this pairing needs,
which would require knowing the true edge. With a spread of 350 and 3 pairings it is roughly 4,200, 680 and 170
deals.

## Session Elo (secondary)

Session Elo answers a different question: how often a player wins a 10-deal session (20 hands), where winning
means being up by more than 1 big blind over the session (closer is a tie, counted half). It is a Bradley–Terry
fit on the Elo scale with a Gaussian prior (1.5 logits), centred on 1500 within each group, and its 95% interval is
the Laplace approximation of the posterior (the inverse Hessian at the mode). Because of the prior, the interval
cannot collapse to a point, even after three wins in three sessions.

Session Elo can disagree with strength: a player that wins many sessions narrowly and loses a few heavily has a
high session Elo and a low strength. Strength follows the stated objective (chips), so session Elo never decides
ranks or leaders.

## Who is a player

A model player is `backend:model:effort`, for example `claude:claude-sonnet-5-5:high`. Two ids in different
runs with the same identity pool in the all-time standings; the same model at another reasoning level is a
different player.

A player whose model was not recorded, or was recorded only as a bare alias (`sonnet`, `opus`, `haiku`, whose
version changes over time), is **quarantined**: in the all-time standings it is listed with its hands and net
result but not rated, and its deals are left out of the fit, because two runs of "sonnet" may not be the same
model. Within a single run it is rated and flagged. A player whose only opponents were quarantined is listed as
unrated for that reason.

## Style statistics

`stats.collect` counts standard poker tracking statistics per hand dealt. At showdown, `won_sd` counts pots won
outright; a split pot is not a win and is counted as `sd_ties`. `wsd` is outright wins over showdowns.

## Limits

- **Few pairings, one replicate.** Most runs play each pairing once per round as one continuous match. The
  intervals describe the variation of deals within those matches. They do not cover how much a model's play
  would vary across fresh conversations, which only independent replicates (`replicates` in a plan) measure.
- **Memory dependence** beyond 10 deals is not fully absorbed by the blocks (see Calibration).
- **All-time pooling.** All-time verdicts are one test on all pooled data, with alpha split over all pairings.
  Runs that stopped a pairing early because it looked decided bias pooled edges slightly away from zero.
- **Additivity** of strengths is an assumption (see Strength).
- **Identity is as good as the CLI's report.** A model name in a config is what was requested; the backend may
  serve another version. `served_models` in the style statistics shows what answered where the CLI reports it.
- **One game format.** Heads-up, 100 big blinds deep, stacks reset every hand. Strengths do not carry over to
  other formats.
