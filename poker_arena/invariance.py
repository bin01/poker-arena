"""Isomorphism invariance: does a player decide the same way when nothing that matters has changed?

Poker is symmetric in the suits: relabelling them (spades -> hearts -> diamonds -> clubs -> spades, say) changes no
hand's value, no equity and no best play. Nor do the seat names. A player whose decisions change under such a
relabelling is reacting to something irrelevant (a suit it likes, a name), and that is measured here.

`run_invariance_test(agent_factory, ...)` asks a fresh agent (from `agent_factory()`) each deterministic heads-up
spot (the quiz's hand-picked basics, rebuilt as engine positions) and `n_perms` isomorphic variants of it (a suit
permutation from ALL_SUIT_PERMS, and new seat names), and reports how often the action flips, the mean absolute
EV difference between the variant's action and the base action (both valued on the base spot, whose values every
variant shares), the spread of raise sizes, a binomial test of the flips against a small tolerance, and examples.
"""
from __future__ import annotations
import asyncio
import copy
import itertools
import math
import random
import re
import statistics

from .cards import SUITS

IDENTITY = {s: s for s in SUITS}


def _perm(mapping: str) -> dict:
    return dict(zip(SUITS, mapping))


ALL_SUIT_PERMS = [
    _perm('hdcs'),   # cyclic shift: s->h->d->c->s
    _perm('hscd'),   # red/black swap: s<->h, d<->c
    _perm('cshd'),   # the other cycle: s->c->d->h->s
    _perm('dchs'),   # s->d->h->c->s
    _perm('chds'),   # s<->c
    _perm('shcd'),   # d<->c
    _perm('dhsc'),   # s<->d
    _perm('scdh'),   # h<->c
]
EVERY_SUIT_PERM = [dict(zip(SUITS, p)) for p in itertools.permutations(SUITS)]
SEAT_NAMES = [('Alex', 'Blair'), ('Casey', 'Devon'), ('Emery', 'Finley'), ('Gray', 'Harper'), ('Indy', 'Jordan')]
_CARD = re.compile(r'^[2-9TJQKA][shdc]$')
_COMBO = re.compile(r'^(?:[2-9TJQKA][shdc]){2}$')
TOLERANCE = 0.05      # flip rate an invariant player may show from its own sampling noise
ALPHA = 0.05


def permute_suit_card(card: str, perm: dict) -> str:
    return card[0] + perm.get(card[1], card[1]) if isinstance(card, str) and _CARD.match(card) else card


def permute_cards(cards, perm: dict):
    if isinstance(cards, str):
        if _CARD.match(cards):
            return permute_suit_card(cards, perm)
        return ' '.join(permute_suit_card(c, perm) for c in cards.split()) if all(
            _CARD.match(c) for c in cards.split()) and cards.split() else cards
    return [permute_cards(c, perm) for c in cards]


def _permute_range(text: str, perm: dict) -> str:
    """A range in the usual notation: only exact combos ('AhKh') name suits; 's'/'o' suffixes are suitedness."""
    out = []
    for tok in text.split(','):
        t = tok.strip()
        if _COMBO.match(t.replace(' ', '')):
            t = t.replace(' ', '')
            t = permute_suit_card(t[:2], perm) + permute_suit_card(t[2:], perm)
        out.append(t)
    return ', '.join(out)


CARD_KEYS = {'hole', 'board', 'cards', 'deck_board', 'hand', 'combo'}


def permute_spot(spot_dict: dict, perm: dict, seat_names: tuple | None = None) -> dict:
    """The same spot with every suit relabelled by `perm` (cards, boards, exact combos of a range) and, with
    `seat_names`, new names for the two seats. Nothing else changes."""
    def walk(x, key=None):
        if isinstance(x, dict):
            return {k: walk(v, k) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v, key) for v in x]
        if isinstance(x, tuple):
            return tuple(walk(v, key) for v in x)
        if isinstance(x, str):
            if key == 'range':
                return _permute_range(x, perm)
            if key in CARD_KEYS or _CARD.match(x):
                return permute_cards(x, perm)
        return x
    out = walk(copy.deepcopy(spot_dict))
    if seat_names is not None:
        out['names'] = list(seat_names)
    return out


# ---- spots ---------------------------------------------------------------------------------------------------

def base_spots(n_spots: int = 12, seed: int = 0) -> list[dict]:
    """`n_spots` of the quiz's hand-picked basics (engine positions with a stated villain range), chosen and
    ordered deterministically by `seed`."""
    from .quiz import SPOTS, _basic
    spots = [_basic(t) for t in SPOTS]
    rng = random.Random(seed)
    order = list(range(len(spots)))
    rng.shuffle(order)
    return [spots[i] for i in sorted(order[:n_spots])]


def build_spot(spot: dict):
    """(episode, hero seat, request) for a spot, with its seat names."""
    from .quiz import build
    ep, hero, villain, combos = build(spot)
    ep.names = list(spot.get('names') or ['Hero', 'Villain'])
    return ep, hero, ep.request(hero), combos


def spot_values(spot: dict, sims: int = 1500) -> dict:
    """{abstract action: EV in bb} at the spot, valued the quiz's way (fold = 0; a call ends the betting; a shove
    is called by the stated range): exact on the river and turn, else Monte Carlo from a seed fixed by the spot."""
    from .games import ranges
    ep, hero, req, combos = build_spot(spot)
    villain = 1 - hero
    bb, pot, tc = ep.bb, ep.pot, ep.to_call(hero)
    rng = random.Random(sum(map(ord, spot['id'])))
    board = spot['board']
    eq = ranges.equity(spot['hole'], board, combos, rng, sims=sims, exact=len(board) >= 4)
    if spot['kind'] == 'shove':
        S, h, v = ep.start, ep.committed[hero], ep.committed[villain]
        possible = [c for c in ranges.ALL if not set(c) & set(spot['hole'])]
        p_call = len(combos) / len(possible)
        return {'fold': 0.0, 'all_in': ((1 - p_call) * (h + v) + p_call * (eq * 2 * S - (S - h))) / bb,
                '_equity': eq}
    return {'fold': 0.0, 'call': (eq * (pot + tc) - tc) / bb, '_equity': eq}


def _abstract(action: dict, req) -> tuple[str, int | None]:
    """The engine's reading of a reply as (fold | check | call | raise | all_in, raise-to amount)."""
    kind = str(action.get('action') or '').strip().lower().replace('-', '_').replace(' ', '_')
    kind = {'bet': 'raise', 'allin': 'all_in', 'shove': 'all_in', 'jam': 'all_in'}.get(kind, kind)
    rb = req.data.get('raise_bounds')
    amount = None
    if kind == 'raise':
        try:
            amount = int(round(float(action.get('amount'))))
        except (TypeError, ValueError, OverflowError):
            amount = None
        if rb and amount is not None and amount >= rb[1]:
            kind = 'all_in'
    if kind == 'all_in' and rb:
        amount = rb[1]
    return kind, amount


def _value(values: dict, kind: str):
    """The EV of an action the spot values (fold, call, all_in); None for one it does not (a smaller raise)."""
    return values.get(kind) if not kind.startswith('_') else None


async def _decide(agent_factory, spot: dict) -> dict:
    ep, hero, req, _ = build_spot(spot)
    agent = agent_factory()
    agent.start('', {})
    validate = lambda act, ep=ep: copy.deepcopy(ep).apply(hero, act)
    d = await agent.act(req.prompt, req, validate)
    action = d.action if not d.fallback else ep.fallback(hero)
    kind, amount = _abstract(action, req)
    return {'action': kind, 'amount': amount, 'reasoning': d.reasoning}


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


async def arun_invariance_test(agent_factory, spots=None, n_spots=12, n_perms=3, seed=0, sims=1500) -> dict:
    """See `run_invariance_test`; for callers already inside an event loop."""
    spots = list(spots) if spots is not None else base_spots(n_spots, seed)
    rng = random.Random(seed)
    flips, deltas, examples, sizing_cvs, per_spot = 0, [], [], [], []
    n_variants = 0
    for k, spot in enumerate(spots):
        values = spot_values(spot, sims=sims)
        base = await _decide(agent_factory, spot)
        perms = rng.sample(ALL_SUIT_PERMS, min(n_perms, len(ALL_SUIT_PERMS)))
        variants = []
        for j, perm in enumerate(perms):
            names = SEAT_NAMES[(k + j) % len(SEAT_NAMES)]
            var = permute_spot(spot, perm, names)
            got = await _decide(agent_factory, var)
            n_variants += 1
            flipped = got['action'] != base['action']
            flips += flipped
            vb, vv = _value(values, base['action']), _value(values, got['action'])
            delta = abs(vb - vv) * 1000 if vb is not None and vv is not None else (0.0 if not flipped else None)
            if delta is not None:
                deltas.append(delta)
            variants.append({'perm': ''.join(perm[s] for s in SUITS), 'names': list(names), **got,
                             'flipped': flipped, 'ev_delta_mbb': None if delta is None else round(delta, 1)})
            if flipped and len(examples) < 12:
                examples.append({'spot': spot['id'], 'perm': ''.join(perm[s] for s in SUITS), 'names': list(names),
                                 'base': {'hole': spot['hole'], 'board': spot['board'], 'action': base['action'],
                                          'amount': base['amount']},
                                 'variant': {'hole': var['hole'], 'board': var['board'], 'action': got['action'],
                                             'amount': got['amount']},
                                 'ev_delta_mbb': None if delta is None else round(delta, 1)})
        sizes = [x['amount'] for x in [base] + variants if x['action'] == 'raise' and x['amount'] is not None]
        if len(sizes) >= 2 and statistics.mean(sizes) > 0:
            sizing_cvs.append(statistics.pstdev(sizes) / statistics.mean(sizes))
        per_spot.append({'spot': spot['id'], 'base': base['action'], 'variants': variants,
                         'values_bb': {a: round(v, 3) for a, v in values.items() if not a.startswith('_')}})
    flip_rate = flips / n_variants if n_variants else 0.0
    ev_delta = statistics.mean(deltas) if deltas else 0.0
    sizing_cv = statistics.mean(sizing_cvs) if sizing_cvs else 0.0
    p_value = _binom_sf(flips, n_variants, TOLERANCE) if n_variants else 1.0
    score = max(0.0, min(1.0, (1 - flip_rate) * (1 - min(0.5, sizing_cv))))
    return {'n_spots': len(spots), 'n_variants': n_variants, 'flips': flips, 'action_flip_rate': round(flip_rate, 4),
            'ev_delta_mbb': round(ev_delta, 1), 'sizing_cv': round(sizing_cv, 4), 'invariance_score': round(score, 4),
            'p_value': round(p_value, 6), 'tolerance': TOLERANCE,
            'biased': bool(p_value < ALPHA and flip_rate > TOLERANCE),
            'test': f'one-sided exact binomial: flips among {n_variants} variants against a {TOLERANCE:.0%} noise rate',
            'examples': examples, 'spots': per_spot}


def run_invariance_test(agent_factory, spots=None, n_spots=12, n_perms=3, seed=0, sims=1500) -> dict:
    """{'n_spots', 'n_variants', 'action_flip_rate', 'ev_delta_mbb', 'sizing_cv', 'invariance_score' (0..1, 1 =
    the same decision whatever the labels), 'p_value', 'biased', 'examples', 'spots'} for the agents
    `agent_factory()` makes (each decision gets a fresh one, so memory cannot explain a flip)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun_invariance_test(agent_factory, spots, n_spots, n_perms, seed, sims))
    raise RuntimeError('run_invariance_test called inside an event loop; await arun_invariance_test instead')
