"""Cards, decks and a 7-card hand evaluator (pure Python, no dependencies).

A card is a two-character string: rank (23456789TJQKA) then suit (shdc), e.g. 'Ah', 'Td'.
`evaluate(cards)` scores the best five-card hand among 5-7 cards; a higher tuple is a better hand.
"""
from __future__ import annotations
import random
from collections import Counter

RANKS = '23456789TJQKA'
SUITS = 'shdc'
DECK = [r + s for r in RANKS for s in SUITS]
VALUE = {r: i + 2 for i, r in enumerate(RANKS)}

CATEGORIES = ['high card', 'pair', 'two pair', 'three of a kind', 'straight', 'flush', 'full house',
              'four of a kind', 'straight flush']
NAMES = {14: 'ace', 13: 'king', 12: 'queen', 11: 'jack', 10: 'ten', 9: 'nine', 8: 'eight', 7: 'seven', 6: 'six',
         5: 'five', 4: 'four', 3: 'three', 2: 'deuce'}
PLURAL = {**{v: n + 's' for v, n in NAMES.items()}, 6: 'sixes'}


def shuffled(rng: random.Random) -> list[str]:
    deck = DECK[:]
    rng.shuffle(deck)
    return deck


def _straight_high(values: set[int]) -> int:
    """Highest card of the best straight among these values (0 if none); the wheel A-5 counts, high 5."""
    vs = set(values)
    if 14 in vs:
        vs.add(1)
    for high in range(14, 4, -1):
        if all(v in vs for v in range(high - 4, high + 1)):
            return high
    return 0


def evaluate(cards) -> tuple:
    """(category, tiebreak values...) for the best 5-card hand in 5-7 cards."""
    values = [VALUE[c[0]] for c in cards]
    suits = Counter(c[1] for c in cards)
    flush_suit = next((s for s, n in suits.items() if n >= 5), None)
    if flush_suit:
        fv = sorted((VALUE[c[0]] for c in cards if c[1] == flush_suit), reverse=True)
        sf = _straight_high(set(fv))
        if sf:
            return (8, sf)
    counts = Counter(values)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if groups[0][1] == 4:
        quad = groups[0][0]
        return (7, quad, max(v for v in values if v != quad))
    trips = sorted((v for v, n in counts.items() if n >= 3), reverse=True)
    pairs = sorted((v for v, n in counts.items() if n == 2), reverse=True)
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = max([*trips[1:], *pairs])
        return (6, t, p)
    if flush_suit:
        return (5, *fv[:5])
    st = _straight_high(set(values))
    if st:
        return (4, st)
    if trips:
        t = trips[0]
        kick = sorted((v for v in values if v != t), reverse=True)[:2]
        return (3, t, *kick)
    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        kick = max(v for v in values if v not in (hi, lo))
        return (2, hi, lo, kick)
    if pairs:
        p = pairs[0]
        kick = sorted((v for v in values if v != p), reverse=True)[:3]
        return (1, p, *kick)
    return (0, *sorted(values, reverse=True)[:5])


def describe(score: tuple) -> str:
    """'a pair of queens', 'a flush, ace high', ..."""
    cat = score[0]
    if cat == 8:
        return 'a royal flush' if score[1] == 14 else f'a straight flush, {NAMES[score[1]]} high'
    if cat == 7:
        return f'four {PLURAL[score[1]]}'
    if cat == 6:
        return f'a full house, {PLURAL[score[1]]} full of {PLURAL[score[2]]}'
    if cat == 5:
        return f'a flush, {NAMES[score[1]]} high'
    if cat == 4:
        return f'a straight, {NAMES[score[1]]} high'
    if cat == 3:
        return f'three {PLURAL[score[1]]}'
    if cat == 2:
        return f'two pair, {PLURAL[score[1]]} and {PLURAL[score[2]]}'
    if cat == 1:
        return f'a pair of {PLURAL[score[1]]}'
    return f'{NAMES[score[1]]} high'


def equity(hole, board, rng: random.Random, opponents=1, sims=300) -> float:
    """Monte Carlo chance that `hole` wins (ties count by share) against random hands, given the board."""
    known = set(hole) | set(board)
    rest = [c for c in DECK if c not in known]
    need = 5 - len(board)
    total = 0.0
    for _ in range(sims):
        draw = rng.sample(rest, need + 2 * opponents)
        full = list(board) + draw[:need]
        mine = evaluate(list(hole) + full)
        best, ties, lost = mine, 1, False
        for k in range(opponents):
            opp = evaluate(draw[need + 2 * k: need + 2 * k + 2] + full)
            if opp > mine:
                lost = True
                break
            if opp == mine:
                ties += 1
        if not lost:
            total += 1 / ties
    return total / sims
