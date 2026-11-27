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
| α | chance of naming a false winner (a choice) | 5% for the whole tournament: with 36 pairings each gets 0.05/36, so z ≈ 3.2 (one pairing alone: z = 1.96) |
| 1 − β | power: chance of catching a real edge of δ (a choice) | 80%, so z = 0.84 |
| σ | spread of one deal's result (measured) | about 350 bb/100 |
| D | design effect (measured): how much less an extra deal is worth because hands are not independent | about 1.5 to 2 |

For a fixed test with these error rates, no method that scores deals the same way can do it with fewer deals:
below this number, a real edge of δ cannot be told apart from luck. That is the sense in which it is a minimum.

**Why D exists.** A model plays a whole match (50 deals) as one conversation and remembers earlier hands, so
deals inside one conversation are correlated. With m deals per conversation and correlation ρ between deals of
the same conversation, D = 1 + (m − 1)ρ; D = 1.5 to 2 at m = 50 means ρ ≈ 0.01 to 0.02. This is also why a
pairing must play several separate conversations before any verdict.

## Deals needed per pairing

σ = 350 bb/100, power 80%:

| true edge | one pairing alone (z = 1.96, D = 1) | each of 36 pairings (z ≈ 3.2, D = 1) | including memory (D = 1.5 to 2) |
|---|---|---|---|
| 100 bb/100 (a mismatch) | ~100 | ~200 | ~300 to 400 |
| 50 bb/100 (clearly better) | ~390 | ~800 | ~1,200 to 1,600 |
| 30 bb/100 | ~1,070 | ~2,200 | ~3,300 to 4,400 |
| 20 bb/100 (solidly better) | ~2,400 | ~5,000 | ~7,500 to 10,000 |
| 10 bb/100 (slightly better) | ~9,600 | ~20,000 | ~30,000 to 40,000 |

The 1,800-deal cap per pairing covers the δ = 50 row with room for the estimates to be off. Two strong models
that differ by 10 to 20 bb/100 cannot be told apart on subscription budgets; the honest result for them is
"within ±δ of each other".

## Ranking many models

Every game is one model against one other, but all games feed one shared rating, so each model's number uses all
of its games against everyone. A model that has played n deals in total has a rating standard error of roughly
σ√D / √n (here with D = 2):

| deals per model (all opponents) | standard error | 95% margin |
|---|---|---|
| 400 | ~25 bb/100 | ±49 |
| 800 | ~18 bb/100 | ±34 |
| 1,600 | ~12 bb/100 | ±24 |

So a screening round in which each model plays about 1,600 deals across its opponents places every model to
within about ±25 bb/100 of the field, and further rounds are spent only where neighbouring models' ranges overlap.

## The biggest lever: less noise per deal

Deals needed grow with σ², so halving σ cuts the games needed by four. Duplicate deals already remove the luck of
who holds which cards. What is left is mostly the luck of the cards still to come: two players all in on the
flop, and the river decides who wins the pot. Variance-reduction estimators from poker-AI research score each
hand by its expected value at the moment the money went in (using the equity calculator the arena already has)
instead of by how the remaining cards fell. How much this lowers σ here is an empirical question: it can be
measured on the games already played before any new games depend on it.

## What the maths can and cannot promise

1. **It is a guarantee under assumptions.** σ and D are estimates from the games played so far (about 100
   model-vs-model deals when this page was written). The plan is re-sized as data comes in, and the stopping rule is checked by
   simulation, including skewed distributions where a few huge pots dominate.
2. **Equal models can never be separated.** If two models are truly equal, no number of games names a winner.
   That is why a pairing can end "tied within ±δ" instead of with a winner.
3. **Stopping early lowers the average, not the worst case.** Checking after every round and stopping pairings
   whose edge is already clear saves most of the cost on mismatches. Each extra check is another chance to name
   a false winner, so the bar for stopping is stricter than 1.96 standard errors (a group-sequential boundary),
   and the cap is sized for that stricter bar.
4. **The values here are as of 2026-11-27.** If the planner's numbers differ, the planner is right and this page
   should be updated.
