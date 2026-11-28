"""Variance reduction for heads-up hold'em results: all-in EV and AIVAT chance-node control variates.

AIVAT (Burch, Schmid, Moravcik, Morrill and Bowling 2018) subtracts from a hand's result, at every chance node
the hand reached, the luck of that chance event: delta_c = V(state after c) - E_c'[V(state after c')], with V
a fixed function of the cards dealt so far and the pot at the chance node (never of later cards or later
actions). Whatever both players' policies are, E[delta_c | state before c] = 0, so the corrected result has
the same expectation as the raw one: the estimate stays unbiased, and luck that V explains is removed.

Here, for seat 0 (seat 1 is its negative: every correction is zero-sum):
- deal (the hole cards): V = K * (e_pre(h0, h1) - 1/2), K a fixed scale (`deal_scale_bb`), e_pre the preflop
  equity estimated on a fixed set of boards drawn from the unordered pair of hands, so e_pre(h1, h0) =
  1 - e_pre(h0, h1) exactly. The deal is symmetric in the two seats, hence E[V] = 0 exactly.
- flop: V = P_flop * (e_flop - 1/2), e_flop exact (all 990 turn-river runouts); E over flops of e_flop is
  the preflop equity, estimated without bias by e_pre (boards drawn independently of the actual flop).
- turn: V = P_turn * (e_turn - 1/2), E over turns exactly P_turn * (e_flop - 1/2).
- river: V = P_river * (share - 1/2), E over rivers exactly P_river * (e_turn - 1/2).
P_street is the pot when that street's cards are dealt. When the players are all in before the river the
turn and river corrections are exactly the all-in luck (`all_in_ev`) of the runout.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .solver import CARD_ID, fast_rank

STREETS = ('preflop', 'flop', 'turn', 'river')
STREET_CARDS = {'flop': 3, 'turn': 4, 'river': 5}
DEAL_SCALE_BB = 10.0


def _ids(cards):
    return [CARD_ID[c] for c in cards]


@lru_cache(maxsize=None)
def _combos(n: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n), k)), dtype=np.int64).reshape(-1, k)


def _seed(*parts) -> int:
    return int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _shares(h0, h1, boards: np.ndarray) -> np.ndarray:
    n = len(boards)
    a = fast_rank(np.concatenate([np.broadcast_to(np.array(h0), (n, 2)), boards], 1))
    b = fast_rank(np.concatenate([np.broadcast_to(np.array(h1), (n, 2)), boards], 1))
    return 0.5 * (1.0 + np.sign(a - b))


def exact_equity(hole0, hole1, board) -> float:
    """Seat 0's exact showdown share with these hole cards, enumerating every runout of a 3-5 card board."""
    h0, h1, b = _ids(hole0), _ids(hole1), _ids(board)
    need = 5 - len(b)
    if need < 0 or need > 2:
        raise ValueError('exact equity needs a flop, turn or river board')
    known = set(h0) | set(h1) | set(b)
    rest = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    if need == 0:
        boards = np.array([b], dtype=np.int64)
    else:
        runs = rest[_combos(len(rest), need)]
        boards = np.concatenate([np.broadcast_to(np.array(b, dtype=np.int64), (len(runs), len(b))), runs], 1)
    return float(_shares(h0, h1, boards).mean())


def preflop_equity(hole0, hole1, sims: int = 200, seed: int = 0) -> float:
    """Seat 0's preflop equity on `sims` boards drawn from a stream keyed by the unordered pair of hands (and
    `seed`), so swapping the hands gives exactly 1 minus the estimate; unbiased for the exact equity."""
    h0, h1 = _ids(hole0), _ids(hole1)
    key = tuple(sorted([tuple(sorted(h0)), tuple(sorted(h1))]))
    nr = np.random.default_rng(_seed('aivat-pre', key, seed, sims))
    known = set(h0) | set(h1)
    rest = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    boards = rest[np.argsort(nr.random((sims, len(rest))), axis=1)[:, :5]]
    return float(_shares(h0, h1, boards).mean())


def _mc_equity(hole0, hole1, board, sims, seed) -> float:
    h0, h1, b = _ids(hole0), _ids(hole1), _ids(board)
    known = set(h0) | set(h1) | set(b)
    rest = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    nr = np.random.default_rng(_seed('aivat-mc', sorted(h0), sorted(h1), b, seed))
    runs = rest[np.argsort(nr.random((sims, len(rest))), axis=1)[:, :5 - len(b)]]
    boards = np.concatenate([np.broadcast_to(np.array(b, dtype=np.int64), (sims, len(b))), runs], 1)
    return float(_shares(h0, h1, boards).mean())


def _pots(rec: dict, sb=1, bb=2) -> dict:
    """The pot when each street's cards were dealt (the pot after the last action before that street) for the
    streets the hand reached."""
    acts = rec.get('actions') or []
    board = rec.get('board') or []
    out = {}
    for st in ('flop', 'turn', 'river'):
        if len(board) < STREET_CARDS[st]:
            break
        idx = STREETS.index(st)
        before = [a for a in acts if STREETS.index(a['street']) < idx]
        out[st] = float(before[-1]['pot'] if before and 'pot' in before[-1] else
                        _pot_after(before, rec.get('button', 0), sb, bb))
    return out


def _pot_after(actions, button=0, sb=1, bb=2) -> float:
    """The pot after these actions, for records without the 'pot' field."""
    bets, pot, street = {button: sb, 1 - button: bb}, sb + bb, 'preflop'
    for a in actions:
        if a['street'] != street:
            street, bets = a['street'], {0: 0, 1: 0}
        if a['action'] == 'call':
            add = a.get('amount') or (max(bets.values()) - bets[a['seat']])
            pot += add
            bets[a['seat']] += add
        elif a['action'] in ('bet', 'raise'):
            pot += a['amount'] - bets[a['seat']]
            bets[a['seat']] = a['amount']
    return float(pot)


def _committed(rec: dict) -> list[float]:
    won, net = rec.get('won') or [0, 0], rec.get('net') or [0, 0]
    return [won[i] - net[i] for i in range(2)]


def all_in_ev(hand_record, bb=2, exact_turn_river=True, sims=400, seed=0) -> dict:
    """The all-in EV of a hand: when both players were all in with cards to come, each seat's expected net is
    its equity at that moment (exact from the flop on with `exact_turn_river`, else and preflop on `sims`
    boards) times the contested pot, minus what it put in. {'all_in', 'all_in_street', 'actual_net',
    'ev_net', 'luck' (actual minus EV)}."""
    rec = hand_record
    actual = [float(x) for x in rec.get('net') or [0, 0]]
    acts = rec.get('actions') or []
    board = rec.get('board') or []
    last = acts[-1] if acts else None
    out = {'all_in': False, 'all_in_street': None, 'actual_net': [int(x) for x in actual], 'ev_net': list(actual),
           'luck': [0.0, 0.0]}
    if not rec.get('showdown') or last is None or last['action'] == 'fold':
        return out
    street = last['street']
    n_known = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[street]
    if n_known >= 5 or not any(a.get('all_in') for a in acts):
        return out
    hole = rec['hole']
    if n_known == 0:
        eq = preflop_equity(hole[0], hole[1], max(sims, 1), seed)
    elif exact_turn_river:
        eq = exact_equity(hole[0], hole[1], board[:n_known])
    else:
        eq = _mc_equity(hole[0], hole[1], board[:n_known], sims, seed)
    com = _committed(rec)
    pot = sum(com)
    ev = [eq * pot - com[0], (1 - eq) * pot - com[1]]
    out.update(all_in=True, all_in_street=street, ev_net=ev, luck=[actual[0] - ev[0], actual[1] - ev[1]])
    return out


def aivat_hand(hand_record, bb=2, stack=200, sims=200, seed=0, deal_scale_bb=DEAL_SCALE_BB) -> dict:
    """The AIVAT-corrected result of one hand (see the module docstring). {'raw_net', 'aivat_net' (raw minus
    the chance corrections; zero-sum), 'ev_adj_net' (all-in EV), 'luck_correction' (the total subtracted, per
    seat), 'street_corrections' ({'deal', 'flop', 'turn', 'river'}: per seat, for the chance nodes reached),
    'equities' (seat 0's equity after each chance node)}."""
    rec = hand_record
    raw = [float(x) for x in rec.get('net') or [0, 0]]
    hole = rec['hole']
    board = rec.get('board') or []
    e_pre = preflop_equity(hole[0], hole[1], sims, seed)
    pots = _pots(rec, max(1, bb // 2), bb)
    corr = {'deal': deal_scale_bb * bb * (e_pre - 0.5)}
    eqs = {'preflop': e_pre}
    prev = e_pre
    for st in ('flop', 'turn', 'river'):
        if st not in pots:
            break
        e = exact_equity(hole[0], hole[1], board[:STREET_CARDS[st]])
        corr[st] = pots[st] * (e - prev)
        eqs[st] = e
        prev = e
    total = sum(corr.values())
    aivat = [raw[0] - total, -(raw[0] - total)]
    ev = all_in_ev(rec, bb, True, max(sims, 400), seed)
    return {'raw_net': raw, 'aivat_net': aivat, 'ev_adj_net': [float(x) for x in ev['ev_net']],
            'luck_correction': [total, -total], 'street_corrections': {k: [v, -v] for k, v in corr.items()},
            'equities': eqs, 'all_in': ev['all_in'], 'all_in_street': ev['all_in_street']}


def _align(a_hands, b_hands):
    if len(a_hands) == len(b_hands) and all(x.get('hand') == y.get('hand') for x, y in zip(a_hands, b_hands)):
        return list(zip(a_hands, b_hands))
    by = {y.get('hand'): y for y in b_hands}
    return [(x, by[x.get('hand')]) for x in a_hands if x.get('hand') in by]


def _mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return None, None
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None


def estimate_matchup(hands_a_first, hands_b_first, bb=2, stack=200, sims=200, deal_scale_bb=DEAL_SCALE_BB) -> dict:
    """Duplicate estimates of a's edge in bb/100: each deal is hand k of the a-first play (a in seat 0) and of
    the b-first play (a in seat 1), x = a's net over both / bb / 2 * 100, raw, all-in EV adjusted and AIVAT.
    variance_reduction_ratio = raw_se / aivat_se; effective_sample_multiplier = its square (how many raw
    deals one AIVAT deal is worth)."""
    per = []
    for ha, hb in _align(list(hands_a_first), list(hands_b_first)):
        va, vb = aivat_hand(ha, bb, stack, sims, deal_scale_bb=deal_scale_bb), \
            aivat_hand(hb, bb, stack, sims, deal_scale_bb=deal_scale_bb)
        f = 100 / bb / 2
        per.append({'deal': ha.get('hand'), 'raw': (va['raw_net'][0] + vb['raw_net'][1]) * f,
                    'ev_adj': (va['ev_adj_net'][0] + vb['ev_adj_net'][1]) * f,
                    'aivat': (va['aivat_net'][0] + vb['aivat_net'][1]) * f})
    out = {'deals': len(per)}
    for key in ('raw', 'ev_adj', 'aivat'):
        m, se = _mean_se([p[key] for p in per])
        out[f'{key}_bb100'], out[f'{key}_se'] = m, se
    r, a = out['raw_se'], out['aivat_se']
    ratio = r / a if r and a else None
    out.update(variance_reduction_ratio=ratio, effective_sample_multiplier=ratio ** 2 if ratio else None, per_deal=per)
    return out


def evaluate_run_aivat(run_dir) -> dict:
    """AIVAT estimates for every pairing in a run folder, from the duplicate pairs `ratings.read_conversation`
    verifies (only verified deals count). Conversations of one pairing are pooled with the pairing's players
    in sorted order. {'run', 'players', 'pairings': [{'a', 'b', 'conversations', 'deals', 'raw_bb100',
    'raw_se', 'ev_adj_bb100', 'ev_adj_se', 'aivat_bb100', 'aivat_se', 'variance_reduction_ratio',
    'effective_sample_multiplier'}], 'summary': {'pairings', 'deals', 'mean_variance_reduction_ratio'}}."""
    from .ratings import read_conversation
    root = Path(run_dir)
    pools = {}
    for cfg_path in sorted(root.rglob('a-first/config.json')):
        mdir = cfg_path.parent.parent
        if not (mdir / 'b-first' / 'config.json').exists():
            continue
        conv = read_conversation(mdir)
        if not conv or not conv['verified']:
            continue
        ca, _ = conv['configs']
        ha, hb = conv['hands']
        game = ca.get('game') or {}
        p0, p1 = ca['players'][:2]
        a, b = sorted([p0, p1])
        A = [ha[k] for k in conv['verified']]
        B = [hb[k] for k in conv['verified']]
        if p0 != a:
            A, B = B, A
        pool = pools.setdefault((a, b), {'A': [], 'B': [], 'bb': game.get('bb', 2), 'stack': game.get('stack', 200),
                                         'conversations': 0})
        tag = str(mdir.relative_to(root)) if mdir != root else mdir.name
        pool['A'] += [{**h, 'hand': f'{tag}#{h.get("hand")}'} for h in A]
        pool['B'] += [{**h, 'hand': f'{tag}#{h.get("hand")}'} for h in B]
        pool['conversations'] += 1
    pairings = []
    for (a, b), pool in sorted(pools.items()):
        est = estimate_matchup(pool['A'], pool['B'], pool['bb'], pool['stack'])
        est.pop('per_deal', None)
        pairings.append({'a': a, 'b': b, 'conversations': pool['conversations'], **est})
    ratios = [p['variance_reduction_ratio'] for p in pairings if p['variance_reduction_ratio']]
    players = sorted({p['a'] for p in pairings} | {p['b'] for p in pairings})
    return {'run': str(root), 'players': players, 'pairings': pairings,
            'summary': {'pairings': len(pairings), 'deals': sum(p['deals'] for p in pairings),
                        'mean_variance_reduction_ratio': float(np.mean(ratios)) if ratios else None}}


__all__ = ['all_in_ev', 'aivat_hand', 'estimate_matchup', 'evaluate_run_aivat', 'exact_equity', 'preflop_equity']
