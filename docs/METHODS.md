# How many games: the sample-size maths

The question "how many games until we know which model is better?" has an exact answer once four things are
fixed. Three are choices; the rest is measured from play. This page gives the formula, the values measured so
far, and what the formula can and cannot promise.

## The unit

A **duplicate deal** is one set of cards (both hole hands and the board) played twice with the seats swapped:
model A holds hand X against B's hand Y, then A holds Y against X. A deal's result is A's chips over both hands,
in big blinds per 100 hands (bb/100). Playing both sides of the same cards removes most of the luck of the deal,
so what is left is mostly the difference in play.

## The formula

To detect a true edge of at least δ bb/100 between two models, with a false-winner rate of α (two-sided) and a
chance 1 − β of catching an edge that really is δ:

    deals = ((z(1 − α/2) + z(1 − β)) × σ / δ)² × D

| symbol | meaning | value used |
|---|---|---|
| δ | smallest edge worth detecting (a choice) | 50 bb/100 |
| α | chance of naming a winner when there is none (a choice) | 5%, so z = 1.96 |
| 1 − β | power: chance of catching a real edge of δ (a choice) | 80%, so z = 0.84 |
| σ | spread of one deal's result (measured) | about 350 bb/100 |
| D | design effect (measured): how much less an extra deal is worth because hands are not independent | about 4.7 |

**Why D exists.** A model plays a whole match (50 deals) as one conversation and remembers earlier hands, so
deals inside one conversation are correlated. With m deals per conversation and correlation ρ between deals of
the same conversation, D = 1 + (m − 1)ρ. D ≈ 4.7 corresponds to ρ ≈ 0.075 at m = 50. This is also why a
pairing must play several separate conversations before any verdict: the effective sample is closer to the
number of conversations than to the number of deals.

## Deals needed per pairing

σ = 350 bb/100, α = 5%, power 80%:

| true edge | if deals were independent (D = 1) | allowing for memory (D ≈ 4.7) |
|---|---|---|
| 100 bb/100 (a mismatch) | ~100 | ~450 |
| 50 bb/100 (clearly better) | ~390 | ~1,800 |
| 30 bb/100 | ~1,070 | ~5,000 |
| 20 bb/100 (solidly better) | ~2,400 | ~11,300 |
| 10 bb/100 (slightly better) | ~9,600 | ~45,000 |

The 1,800-deal cap per pairing is the δ = 50 row. Two strong models that differ by 10 to 20 bb/100 cannot be
told apart on subscription budgets; the honest result for them is "within ±δ of each other".

## Ranking many models

Every game is one model against one other, but all games feed one shared rating, so each model's number uses all
of its games against everyone. A model that has played n deals in total has a rating standard error of roughly
σ√D / √n:

| deals per model (all opponents) | standard error | 95% margin |
|---|---|---|
| 400 | ~38 bb/100 | ±74 |
| 800 | ~27 bb/100 | ±53 |
| 1,600 | ~19 bb/100 | ±37 |

So a screening round in which each model plays about 1,600 deals across its opponents places every model to
within about ±40 bb/100, and further rounds are spent only where neighbouring models' ranges overlap.

## What the maths can and cannot promise

1. **It is a guarantee under assumptions.** σ and D are estimates from the games played so far (39 model-vs-model
   deals when this page was written). The plan is re-sized as data comes in, and the stopping rule is checked by
   simulation, including skewed distributions where a few huge pots dominate.
2. **Equal models can never be separated.** If two models are truly equal, no number of games names a winner.
   That is why a pairing can end "tied within ±δ" instead of with a winner.
3. **Stopping early lowers the average, not the worst case.** Checking after every round and stopping pairings
   whose edge is already clear saves most of the cost on mismatches. Each extra check is another chance to name
   a false winner, so the bar for stopping is stricter than 1.96 standard errors (a group-sequential boundary),
   and the cap is sized for that stricter bar.
4. **The values here are as of 2026-11-27.** If the planner's numbers differ, the planner is right and this page
   should be updated.
