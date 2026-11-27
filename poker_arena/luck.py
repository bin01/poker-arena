"""Expected-value scoring: take the luck of the cards still to come out of a hand's result.

Duplicate deals cancel the luck of who holds which cards. What is left is mostly the luck of the runout: two
players all in on the flop, and the turn and river decide who takes the pot. Once both players are all in (or
the betting is otherwise closed before the river and the hand goes to showdown), nothing they do can change
the result any more, so the hand can be scored by what it was worth at that moment instead of by how the cards
fell:

    expected net = equity x pot - what the player put in

Equity is exact for the turn and the flop (every remaining runout) and a fixed-seed Monte Carlo estimate
before the flop. This is the all-in adjustment used in poker-AI evaluation. It is unbiased (on average it
equals the real result, because the runout is a fair draw from the cards left), and it lowers the spread of
results, so fewer deals are needed for the same certainty. Hands that end with a fold, or with a decision on
the river, are scored as they are.

One catch, special to duplicate play: both plays of a deal share the same runout, so in raw chips the luck of
the river often cancels between the two plays already. Adjusting one play but not the other can undo that and
make the spread worse. So the luck term is used as a control variate: a deal's score is

    chips - c x (chips - expected value)

which is unbiased for any c (the luck term averages zero), equals raw chips at c = 0 and full adjustment at
c = 1. `measure(root)` fits the c that gives the smallest spread on games already played and reports the spread
of per-deal results under all three, and what each means for the number of deals needed.
"""
from __future__ import annotations
import itertools
import json
import math
import random
from pathlib import Path

from .cards import DECK, evaluate

STREET_CARDS = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
PREFLOP_SIMS = 20_000


def heads_up_equity(a, b, board, sims=PREFLOP_SIMS, seed=0) -> float:
    """Chance that hand a beats hand b (ties count half) over the rest of the board."""
    known = set(a) | set(b) | set(board)
    rest = [c for c in DECK if c not in known]
    need = 5 - len(board)
    if need <= 2:
        runs = itertools.combinations(rest, need)
        total = n = 0
        for draw in runs:
            full = list(board) + list(draw)
            total += _score(a, b, full)
            n += 1
        return total / n
    rng = random.Random(seed)
    return sum(_score(a, b, list(board) + rng.sample(rest, need)) for _ in range(sims)) / sims


def _score(a, b, full):
    x, y = evaluate(list(a) + full), evaluate(list(b) + full)
    return 1.0 if x > y else 0.5 if x == y else 0.0


def closed_at(h: dict) -> int | None:
    """How many board cards were out when the betting closed, for a heads-up hand that then went to showdown
    without another decision; None if the hand was decided by a fold or by a decision on the river."""
    if not h.get('showdown') or len(h.get('hole', [])) != 2 or len(h.get('board', [])) != 5:
        return None
    acts = h.get('actions') or []
    if not acts:
        return None
    k = STREET_CARDS.get(acts[-1].get('street'), 5)
    return k if k < 5 else None


def expected_net(h: dict, sims=PREFLOP_SIMS) -> list[float] | None:
    """Each seat's expected net chips at the moment the betting closed, or None if the hand isn't adjusted."""
    k = closed_at(h)
    if k is None:
        return None
    net, won = h['net'], h.get('won') or [0, 0]
    put_in = [w - n for w, n in zip(won, net)]   # chips each seat put in the pot
    pot = sum(put_in)
    eq = heads_up_equity(h['hole'][0], h['hole'][1], h['board'][:k], sims=sims, seed=h.get('hand', 0))
    return [eq * pot - put_in[0], (1 - eq) * pot - put_in[1]]


def _hands(d: Path) -> dict:
    out = {}
    p = d / 'hands.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                h = json.loads(line)
                out[h['hand']] = h
            except (ValueError, KeyError):
                pass
    return out


def deals(root, sims=PREFLOP_SIMS) -> list[dict]:
    """Every finished duplicate deal under root, scored both ways: {'matchup', 'deal', 'bb', 'chips', 'ev',
    'adjusted' (how many of its two hands were adjusted)}. Both values are a's net over the two hands."""
    out = []
    for cfg_path in sorted(Path(root).rglob('a-first/config.json')):
        mdir = cfg_path.parent.parent
        if not (mdir / 'b-first' / 'config.json').exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        bb = cfg['game'].get('big_unit') or cfg['game'].get('bb') or 1
        ha, hb = _hands(mdir / 'a-first'), _hands(mdir / 'b-first')
        for k in sorted(set(ha) & set(hb)):
            ea, eb = expected_net(ha[k], sims), expected_net(hb[k], sims)
            chips = ha[k]['net'][0] + hb[k]['net'][1]   # a is seat 0 of a-first and seat 1 of b-first
            ev = (ea[0] if ea else ha[k]['net'][0]) + (eb[1] if eb else hb[k]['net'][1])
            out.append({'matchup': str(mdir.relative_to(root)), 'deal': k, 'bb': bb, 'chips': chips, 'ev': ev,
                        'adjusted': (ea is not None) + (eb is not None)})
    return out


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def measure(root, sims=PREFLOP_SIMS) -> dict:
    """The spread of per-deal results (bb/100, within each matchup) scored by chips and by expected value."""
    ds = deals(root, sims)
    by = {}
    for d in ds:
        by.setdefault(d['matchup'], []).append(d)
    # deviations from each matchup's own mean, so a real edge between two players doesn't count as noise
    raw, luck = [], []
    for xs in by.values():
        r = [x['chips'] / x['bb'] / 2 * 100 for x in xs]
        lk = [(x['chips'] - x['ev']) / x['bb'] / 2 * 100 for x in xs]
        mr, ml = sum(r) / len(r), sum(lk) / len(lk)
        raw += [v - mr for v in r]
        luck += [v - ml for v in lk]
    var_l = sum(v * v for v in luck)
    c = sum(r * lk for r, lk in zip(raw, luck)) / var_l if var_l else 0.0
    s_raw = _sd(raw)
    s_ev = _sd([r - lk for r, lk in zip(raw, luck)])
    s_cv = _sd([r - c * lk for r, lk in zip(raw, luck)])
    factor = lambda s: (s / s_raw) ** 2 if s_raw and s is not None else None
    return {'deals': len(ds), 'matchups': len(by),
            'hands_adjusted': sum(d['adjusted'] for d in ds), 'hands': 2 * len(ds),
            'sd_chips': s_raw, 'sd_ev': s_ev, 'sd_best': s_cv, 'c': c,
            'ev_factor': factor(s_ev), 'deals_factor': factor(s_cv)}


def report(m: dict) -> str:
    if not m['deals'] or m['sd_chips'] is None:
        return 'Not enough finished duplicate deals to measure.'
    pct = lambda f: f'{(1 - f) * 100:.0f}% fewer deals' if f < 1 else f'{(f - 1) * 100:.0f}% more deals'
    return '\n'.join([
        f'{m["deals"]:,} duplicate deals in {m["matchups"]} matchups; {m["hands_adjusted"]:,} of {m["hands"]:,} hands '
        f'were all in (or closed) before the river.',
        f'Spread of one deal, scored by chips:              {m["sd_chips"]:.0f} bb/100',
        f'Spread of one deal, fully by expected value:      {m["sd_ev"]:.0f} bb/100  ({pct(m["ev_factor"])})',
        f'Spread of one deal, best mix (c = {m["c"]:.2f}):        {m["sd_best"]:.0f} bb/100  ({pct(m["deals_factor"])})',
    ])
