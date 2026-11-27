"""Ratings from duplicate play: one Bradley-Terry fit on the Elo scale, plus bb/100.

The unit is a duplicate deal: hand k of a matchup's two matches (a-first and b-first), in which each
player held each seat's cards once, so the luck of the deal cancels. Deals are grouped in order into
sessions of `SESSION` deals (10 deals = 20 hands); a player wins a session if its net chips over it are
more than 1 big blind (0.1bb per deal); closer than that is a tie (half a win). The Elo fit is over sessions:

    P(A wins a 20-hand duplicate session against B) = logistic(r_A - r_B)

fitted by Newton's method with a weak Gaussian prior (as in Clocktower's ratings.py), on the Elo scale
(400 / ln 10 points per logit), centred so the players average 1500 (or with one player anchored).
90% intervals come from a bootstrap that resamples sessions within each matchup.

Why sessions and not single deals: most single deals are tiny pots, and counting them as wins or losses
rewards a player that wins many small pots and loses whole stacks (a random bot rates near the top).
Summing chips over a session weighs pots by size, and the session win is still a clean Elo outcome.

bb/100, big blinds won per 100 hands over every duplicate hand, is the second number, with a standard
error from the per-deal results. It is the poker-native measure and uses every chip.
"""
from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

K = 400 / math.log(10)


def load_deals(root) -> list[dict]:
    """[{'a', 'b', 'matchup', 'deal', 'net_a' (chips, both hands), 'bb'}] from every finished pair under root.
    Pairs whose two matches are both under way count the deals both have finished."""
    deals = []
    for cfg_path in sorted(Path(root).rglob('a-first/config.json')):
        mdir = cfg_path.parent.parent
        a_dir, b_dir = mdir / 'a-first', mdir / 'b-first'
        if not (b_dir / 'config.json').exists():
            continue
        ca = json.loads((a_dir / 'config.json').read_text())
        bb = ca['game'].get('big_unit') or ca['game'].get('bb') or 1
        ha = _hands(a_dir)
        hb = _hands(b_dir)
        a, b = ca['players'][0], ca['players'][1]
        for k in sorted(set(ha) & set(hb)):
            # a-first: a is seat 0; b-first: a is seat 1
            deals.append({'a': a, 'b': b, 'matchup': mdir.name, 'deal': k, 'bb': bb,
                          'net_a': ha[k]['net'][0] + hb[k]['net'][1]})
    return deals


def _hands(d: Path):
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


SESSION = 10       # deals per session (each deal is two hands)
TIE_BB = 0.1       # a session within this many big blinds per deal of even is a tie (1bb over 10 deals)
MIN_SESSIONS = 3   # fewer sessions than this: the rating is provisional and gets no interval


def outcome(g):
    """1, 0.5 or 0 for player a: a session within TIE_BB per deal of even is a tie."""
    band = TIE_BB * g['deals'] * g['bb']
    return 1.0 if g['net_a'] > band else 0.0 if g['net_a'] < -band else 0.5


def sessions(deals, size=SESSION):
    """Deals grouped into sessions per matchup, in deal order: [{'a', 'b', 'matchup', 'net_a', 'deals'}].
    A trailing partial session counts if it has at least half the deals."""
    by = {}
    for d in sorted(deals, key=lambda d: (d['matchup'], d['deal'])):
        key = (d['matchup'], d['deal'] // size)
        s = by.setdefault(key, {'a': d['a'], 'b': d['b'], 'matchup': d['matchup'], 'net_a': 0, 'deals': 0, 'bb': d['bb']})
        s['net_a'] += d['net_a']
        s['deals'] += 1
    return [s for s in by.values() if s['deals'] * 2 >= size]


def fit(games, names, prior=1.5, iters=50):
    """Ratings in logits (mean zero) from sessions; ties count half. The prior (sd 1.5 logits, as in
    Clocktower) keeps a player with one or two results from flying off."""
    N = len(names)
    idx = {n: i for i, n in enumerate(names)}
    if not games:
        return np.zeros(N)
    X = np.zeros((len(games), N))
    y = np.zeros(len(games))
    for k, d in enumerate(games):
        X[k, idx[d['a']]] += 1
        X[k, idx[d['b']]] -= 1
        y[k] = outcome(d)
    w = np.zeros(N)
    pr = np.full(N, 1 / prior ** 2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X.T * (p * (1 - p))) @ X + np.diag(pr)
        step = np.linalg.solve(H, X.T @ (y - p) - pr * w)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w - w.mean()


def bb100(deals, names):
    """Per player: (bb/100 over its duplicate hands, standard error, hands played)."""
    per = defaultdict(list)
    for d in deals:
        x = d['net_a'] / d['bb'] / 2 * 100   # per hand, in bb/100
        per[d['a']].append(x)
        per[d['b']].append(-x)
    out = {}
    for n in names:
        xs = per.get(n, [])
        if not xs:
            out[n] = (None, None, 0)
            continue
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
        out[n] = (m, sd / math.sqrt(len(xs)), 2 * len(xs))
    return out


def ratings(root, boot=300, anchor: tuple[str, float] | None = None, seed=0, session=SESSION) -> dict:
    """The leaderboard of every match under root, with each player's style stats."""
    from .stats import collect
    return {**rate(load_deals(root), boot, anchor, seed, session), 'stats': collect(root)}


def rate(deals, boot=300, anchor: tuple[str, float] | None = None, seed=0, session=SESSION) -> dict:
    """The leaderboard from a list of duplicate deals (as `load_deals` gives them)."""
    names = sorted({d['a'] for d in deals} | {d['b'] for d in deals})
    if not names:
        return {'players': [], 'matchups': [], 'deals': 0, 'sessions': 0, 'session_deals': session,
                'min_sessions': MIN_SESSIONS}
    games = sessions(deals, session)
    w = fit(games, names)

    def scale(v):
        if anchor and anchor[0] in names:
            return anchor[1] + K * (v - v[names.index(anchor[0])])
        return 1500 + K * v

    point = scale(w)
    rng = random.Random(seed)
    games_by = defaultdict(list)
    for g in games:
        games_by[g['matchup']].append(g)
    samples = []
    for _ in range(boot):
        resampled = [rng.choice(gs) for gs in games_by.values() for _ in gs]
        samples.append(scale(fit(resampled, names, iters=20)))
    samples = np.array(samples) if samples else np.zeros((0, len(names)))
    rates = bb100(deals, names)
    players = []
    for i, n in enumerate(names):
        mine = [g for g in games if n in (g['a'], g['b'])]
        wins = sum(1 for g in mine if outcome(g) == (1.0 if g['a'] == n else 0.0))
        ties = sum(1 for g in mine if outcome(g) == 0.5)
        provisional = len(mine) < MIN_SESSIONS
        lo, hi = (np.percentile(samples[:, i], [5, 95]) if len(samples) and not provisional else (None, None))
        m, se, hands = rates[n]
        players.append({'player': n, 'elo': round(float(point[i])), 'provisional': provisional, 'rated': not provisional,
                        'sessions_needed': max(0, MIN_SESSIONS - len(mine)),
                        'net_bb': round(sum((d['net_a'] if d['a'] == n else -d['net_a']) / d['bb']
                                            for d in deals if n in (d['a'], d['b'])), 1),
                        'bb_per_100_ci95': None if se is None else [round(m - 1.96 * se), round(m + 1.96 * se)],
                        'elo90': [round(float(lo)), round(float(hi))] if lo is not None else None,
                        'bb_per_100': None if m is None else round(m, 1),
                        'bb_per_100_se': None if se is None else round(se, 1),
                        'hands': hands, 'sessions': len(mine), 'session_wins': wins, 'session_ties': ties,
                        'session_losses': len(mine) - wins - ties})
    # rated players by rating; the rest after them, by how much they've played (their numbers are too noisy to rank)
    players.sort(key=lambda p: (not p['rated'], -p['elo'] if p['rated'] else -p['hands']))
    return {'players': players, 'matchups': pairs(deals, games), 'deals': len(deals), 'sessions': len(games),
            'min_sessions': MIN_SESSIONS, 'tie_bb_per_deal': TIE_BB, 'z_stop': Z_STOP,
            'session_deals': session, 'anchor': list(anchor) if anchor else None, 'boot': boot}


# When is a pairing settled? Its edge (mean bb/100 over duplicate deals) must be Z_STOP standard errors from
# zero. Not 1.96: a tournament that plays until decided looks after every round, and each look is another
# chance to crown a false winner. 2.7 keeps that chance under 5% for up to 25 looks (a simulated Pocock
# boundary: 2.41 for 5 looks, 2.55 for 10, 2.67 for 20, 2.77 for 40).
Z_STOP = 2.7
MIN_SETTLE = 20      # fewer deals than this: the spread itself is too uncertain to judge
TOO_CLOSE = 10_000   # more deals than this still needed: call it too close


def settle(xs) -> dict:
    """The verdict on one pairing from a's result per duplicate deal, in bb/100."""
    n = len(xs)
    if n < 2:
        return {'deals': n, 'a_bb_per_100': round(xs[0], 1) if xs else None, 'se': None, 'z': None,
                'status': 'early', 'more_deals': None}
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    se = sd / math.sqrt(n)
    z = m / se if se else (math.inf if m else 0.0)
    if abs(z) >= Z_STOP and n >= MIN_SETTLE:
        status, more = 'settled', 0
    elif n < MIN_SETTLE:
        status, more = 'early', MIN_SETTLE - n
    else:
        # at the current edge, the deals it would take for the edge to clear the bar
        need = n * (Z_STOP / abs(z)) ** 2 if z else math.inf
        more = math.ceil(need - n) if need - n < TOO_CLOSE else None
        status = 'open' if more is not None else 'too close'
    return {'deals': n, 'a_bb_per_100': round(m, 1), 'se': round(se, 1), 'z': round(z, 2) if math.isfinite(z) else None,
            'status': status, 'more_deals': more}


def pairs(deals, games=()) -> list:
    """Head to head: one row per pair of players, pooling every matchup between them (rounds, runs)."""
    by = {}
    for d in sorted(deals, key=lambda d: (d['matchup'], d['deal'])):
        x = by.setdefault(frozenset((d['a'], d['b'])), {'a': d['a'], 'b': d['b'], 'xs': [], 'chips': 0,
                                                        'matchups': [], 'a_session_wins': 0, 'b_session_wins': 0})
        sign = 1 if d['a'] == x['a'] else -1
        x['xs'].append(sign * d['net_a'] / d['bb'] / 2 * 100)
        x['chips'] += sign * d['net_a']
        if d['matchup'] not in x['matchups']:
            x['matchups'].append(d['matchup'])
    for g in games:
        x = by[frozenset((g['a'], g['b']))]
        o = outcome(g) if g['a'] == x['a'] else 1 - outcome(g)
        x['a_session_wins'] += o == 1.0
        x['b_session_wins'] += o == 0.0
    out = []
    for x in by.values():
        xs, chips = x.pop('xs'), x.pop('chips')
        out.append({'matchup': x['matchups'][-1], **x, 'net_a_chips': chips, 'hands': 2 * len(xs), **settle(xs)})
    return sorted(out, key=lambda m: (m['a'], m['b']))


def table(r, labels=None) -> str:
    labels = labels or {}
    lines = [f'{"player":28} {"elo":>6} {"90% interval":>14} {"bb/100":>9} {"±se":>6} {"hands":>6} {"sessions W-T-L":>15}']
    for p in r['players']:
        iv = f'{p["elo90"][0]}-{p["elo90"][1]}' if p['elo90'] else 'provisional'
        bb = f'{p["bb_per_100"]:+.1f}' if p['bb_per_100'] is not None else ''
        lines.append(f'{labels.get(p["player"], p["player"]):28} {p["elo"]:>6} {iv:>14} {bb:>9} {p["bb_per_100_se"] or 0:>6.1f} '
                     f'{p["hands"]:>6} {p["session_wins"]:>7}-{p["session_ties"]}-{p["session_losses"]}')
    return '\n'.join(lines)


def verdicts(r, labels=None) -> str:
    """Each pairing: who leads, by how much, and whether that is settled."""
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    lines = []
    for m in sorted(r['matchups'], key=lambda m: (m['status'] != 'settled', m['more_deals'] or 1e9)):
        lead, trail = (m['a'], m['b']) if (m['a_bb_per_100'] or 0) >= 0 else (m['b'], m['a'])
        edge = abs(m['a_bb_per_100'] or 0)
        margin = f' ±{1.96 * m["se"]:.0f}' if m['se'] else ''
        state = ('settled' if m['status'] == 'settled' else f'too early ({m["deals"]} deals)' if m['status'] == 'early'
                 else f'about {m["more_deals"]:,} more deals to settle' if m['status'] == 'open'
                 else f'too close to call after {m["deals"]} deals')
        lines.append(f'{name(lead)} over {name(trail)}: {edge:.0f}{margin} bb/100, {state}')
    return '\n'.join(lines)
