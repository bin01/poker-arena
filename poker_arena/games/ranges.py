"""Hold'em hand ranges and equity against them, for the reference bot and the decision quiz.

A range is a list of two-card combos. `top(f)` is the strongest fraction f of all 1326 combos by the Chen
formula (a fixed, published preflop ranking); `parse('TT+, AJs+, KQo')` reads the usual range notation.
`equity(hole, board, combos, ...)` is hole's share of the pot at showdown against an opponent holding a combo of
the range (uniformly, or by `weights`): exact when the board is complete or one card short of it (every runout,
every combo), otherwise Monte Carlo with the given random generator.
"""
from __future__ import annotations
import itertools
import math
import random
from functools import lru_cache

from ..cards import DECK, RANKS, VALUE, evaluate

ALL = [tuple(c) for c in itertools.combinations(DECK, 2)]


def hand_class(combo) -> str:
    """'AKs', 'T9o', 'QQ': the 169 preflop classes."""
    a, b = sorted(combo, key=lambda c: -VALUE[c[0]])
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ('s' if a[1] == b[1] else 'o')


def chen(cls: str) -> float:
    """The Chen formula's score of a preflop class (higher is stronger)."""
    hi, lo = VALUE[cls[0]], VALUE[cls[1]]
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2)
    if hi == lo:
        return max(5, base * 2)
    score = base + (2 if cls.endswith('s') else 0)
    gap = hi - lo - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        score += 1
    return math.ceil(score)


@lru_cache(maxsize=None)
def ranked() -> tuple:
    """Every combo, strongest class first (Chen score, then high card, then low card, then suited)."""
    key = lambda c: (-chen(hand_class(c)), -max(VALUE[x[0]] for x in c), -min(VALUE[x[0]] for x in c),
                     hand_class(c).endswith('o'), c)
    return tuple(sorted(ALL, key=key))


@lru_cache(maxsize=None)
def percentile() -> dict:
    """{class: the fraction of all combos in classes at least as strong}."""
    out, seen, order = {}, 0, ranked()
    classes = {}
    for c in order:
        classes.setdefault(hand_class(c), []).append(c)
    for cls, combos in classes.items():
        seen += len(combos)
        out[cls] = seen / len(ALL)
    return out


def top(f: float) -> list:
    """The strongest fraction f of all combos, whole classes only."""
    pct = percentile()
    keep = {cls for cls, p in pct.items() if p <= f + 1e-9}
    if not keep:
        keep = {min(pct, key=pct.get)}
    return [c for c in ranked() if hand_class(c) in keep]


def _expand(token: str) -> set:
    """One token of range notation: 'QQ', 'TT+', '77-55', 'AKs', 'AKo', 'AK', 'A5s+', 'KTo+', 'random'."""
    t = token.strip()
    if t in ('random', 'any', '100%'):
        return {hand_class(c) for c in ALL}
    plus = t.endswith('+')
    t = t.rstrip('+')
    if '-' in t:
        a, b = t.split('-')
        out = set()
        if a[0] == a[1]:
            lo, hi = sorted((VALUE[a[0]], VALUE[b[0]]))
            return {RANKS[v - 2] * 2 for v in range(lo, hi + 1)}
        lo, hi = sorted((VALUE[a[1]], VALUE[b[1]]))
        for v in range(lo, hi + 1):
            out |= _expand(a[0] + RANKS[v - 2] + a[2:])
        return out
    r1, r2, suf = t[0], t[1], t[2:]
    if r1 == r2:
        return {RANKS[v - 2] * 2 for v in range(VALUE[r1], 15)} if plus else {r1 + r2}
    sufs = [suf] if suf else ['s', 'o']
    kickers = range(VALUE[r2], VALUE[r1]) if plus else [VALUE[r2]]
    return {r1 + RANKS[k - 2] + s for k in kickers for s in sufs}


def parse(text: str) -> list:
    """The combos of a range written in the usual notation, comma separated; 'AhQh' is one exact combo."""
    classes, exact = set(), set()
    for token in text.split(','):
        t = token.strip().replace(' ', '')
        if len(t) == 4 and t[1] in 'shdc' and t[3] in 'shdc':
            exact.add(frozenset((t[:2], t[2:])))
        elif t:
            classes |= _expand(t)
    return [c for c in ALL if hand_class(c) in classes or frozenset(c) in exact]


def _share(mine, theirs) -> float:
    return 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0


def equity(hole, board, combos, rng: random.Random | None = None, sims: int = 2000, weights=None,
           exact: bool | None = None) -> float:
    """hole's expected share of the pot against one opponent holding a combo of `combos` (weighted by `weights`;
    combos that clash with known cards are dropped). Exact when 4 or 5 board cards are known (or `exact`),
    else `sims` Monte Carlo runouts drawn with `rng`."""
    known = set(hole) | set(board)
    pairs = [(c, w) for c, w in zip(combos, weights or [1.0] * len(combos)) if not (set(c) & known) and w > 0]
    if not pairs:
        raise ValueError('no combo of the range is possible with these cards')
    need = 5 - len(board)
    if exact if exact is not None else need <= 1:
        total = weight = 0.0
        for c, w in pairs:
            rest = [x for x in DECK if x not in known and x not in c]
            for run in itertools.combinations(rest, need):
                full = list(board) + list(run)
                total += w * _share(evaluate(list(hole) + full), evaluate(list(c) + full))
                weight += w
        return total / weight
    rng = rng or random.Random(0)
    cum, acc = [], 0.0
    for _, w in pairs:
        acc += w
        cum.append(acc)
    import bisect
    total = 0.0
    for _ in range(sims):
        c = pairs[min(bisect.bisect_right(cum, rng.random() * acc), len(pairs) - 1)][0]
        rest = [x for x in DECK if x not in known and x not in c]
        full = list(board) + rng.sample(rest, need)
        total += _share(evaluate(list(hole) + full), evaluate(list(c) + full))
    return total / sims
