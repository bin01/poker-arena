"""Who wins at duplicate heads-up hold'em, by how much, and how sure that is.

Objective. Models are told to win chips, so chips decide: the measure is bb/100 over duplicate deals. A
duplicate deal is hand k of a matchup's two matches (a-first and b-first): each player holds each seat's cards
once against the other, so the luck of the deal largely cancels. Its value for a against b is

    x = net_a(both plays) / bb / 2 * 100          (bb per 100 hands)

Repeated deals. Every run starts from the same seeds, so the same pair can meet on the same cards in several
runs. A deal is identified by its fingerprint (game settings, seed and button; `fingerprint`), and repeats of one
deal by one pair are averaged into one observation: a repeat is not new evidence about the cards.

Strength. A least-squares paired-comparison fit on observations, x ~ s_a - s_b, with the strengths summing to
zero within each connected component (players linked by played pairings). A strength is in bb/100 against the
average opponent of its component; the difference of two strengths is the predicted edge between them.

Uncertainty. A model remembers earlier hands, so deals within a match are not independent. The resampling unit
is a block of BLOCK consecutive observations within one matchup; a block bootstrap (blocks resampled within
each pairing, so every pairing keeps its size) gives standard errors, inflated by G/(G-1) for the few-cluster
bias and used with Student t quantiles on G-1 degrees of freedom, G being the number of blocks behind the
estimate. A player behind fewer than 2 blocks, or in a pairing of a single block (whose spread can't be
resampled), gets no interval and is separated from no one. Two players are separated when their difference is significant after Holm's
adjustment over every pair in the component; a rank is shown only for a player separated from everyone, and
the leader is the player separated above everyone in its component. Otherwise there is no established leader.

Pairings. Each pairing gets a verdict from `decide`: the block-mean t statistic (cluster-robust, blocks as
clusters) against a group-sequential boundary (`boundary`) calibrated by simulation for the plan's design, with
the family-wise alpha split over pairings (Bonferroni). `even` is an equivalence verdict (TOST at
+-EQUIV_BB100): the repeated confidence interval for the edge lies inside the margin. `exhausted` means every
planned look was used without a decision; that is not evidence that the two are equally good.

Session Elo. A secondary, different statistic: the share of 10-deal sessions won (net over 1 bb), fitted as
Bradley-Terry with a Gaussian prior and reported with a Laplace-approximation posterior interval, which cannot
collapse to a point. It never decides ranks. docs/methodology.md has the details and the limits.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np

OBJECTIVE = 'bb/100 over duplicate deals'
BLOCK = 10                  # consecutive observations per resampling block
EQUIV_BB100 = 15            # margin of the equivalence test behind 'even'
DEFAULT_DESIGN = {'alpha': 0.05, 'round_deals': 50, 'max_rounds': 6, 'block': BLOCK}
POWER_EDGES = (20, 50, 100)
BOOT = 300
SIMS = 500_000              # null sequences behind each simulated boundary
SIM_SEED = 20261126
SESSION = 10                # deals per Elo session
TIE_BB = 0.1                # a session within this many big blinds per deal of even is a tie
ELO_PRIOR = 1.5             # prior sd of a session-Elo rating, in logits
K = 400 / math.log(10)
ALIASES = ('sonnet', 'opus', 'haiku')   # bare CLI aliases: the model version behind them is unknown
_Z = NormalDist()


# Student t without scipy: the regularized incomplete beta by Lentz's continued fraction.

def _betacf(a, b, x):
    tiny = 1e-300
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((a - 1 + m2) * (a + m2)), -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_sf(t, df):
    """P(T > t) for Student t with df degrees of freedom (normal when df is None)."""
    if df is None or df > 1e7:
        return 1 - _Z.cdf(t)
    p = 0.5 * _ibeta(df / 2, 0.5, df / (df + t * t))
    return p if t >= 0 else 1 - p


@lru_cache(maxsize=4096)
def t_ppf(p, df):
    """The p quantile of Student t (p > 0.5)."""
    if df is None or df > 1e7:
        return _Z.inv_cdf(p)
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > 1 - p:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_sf(mid, df) > 1 - p else (lo, mid)
    return (lo + hi) / 2


# What a player is and which deal was played.

def quarantine_reason(identity: str) -> str | None:
    """Why a player's identity is too vague to pool into all-time standings, or None. Identities are
    history.identity's: backend:model:effort, bot:..., or backend:id when no model was recorded."""
    parts = identity.split(':')
    if parts[0] == 'bot':
        return None
    if len(parts) < 3:
        return 'no model recorded: which model played is unknown'
    if parts[1] in ALIASES:
        return f'bare alias "{parts[1]}": the model version behind it is unknown'
    return None


def fingerprint(game: dict, seed, button) -> str | None:
    """A deal's identity: the game settings that shape it (all but the number of hands), its seed, and the seat
    holding the button. The same fingerprint means the same cards in the same positions."""
    if seed is None:
        return None
    from .history import game_key
    return hashlib.sha1(f'{game_key(game)}|{seed}|{button}'.encode()).hexdigest()[:16]


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


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


def load_deals(root) -> list[dict]:
    """[{'a', 'b', 'matchup', 'deal', 'net_a' (chips, both hands), 'bb', 'seed', 'fp', 'planned'}] for every
    duplicate deal under root both of whose plays have finished."""
    root = Path(root)
    deals = []
    for cfg_path in sorted(root.rglob('a-first/config.json')):
        mdir = cfg_path.parent.parent
        a_dir, b_dir = mdir / 'a-first', mdir / 'b-first'
        ca = _json(cfg_path)
        if not ca or not (b_dir / 'config.json').exists():
            continue
        game = ca.get('game', {})
        bb = game.get('big_unit') or game.get('bb') or 1
        seeds = ca.get('seeds', [])
        n = len(ca['players'])
        ha, hb = _hands(a_dir), _hands(b_dir)
        a, b = ca['players'][0], ca['players'][1]
        name = str(mdir.relative_to(root)) if mdir != root else mdir.name
        for k in sorted(set(ha) & set(hb)):
            seed = seeds[k] if k < len(seeds) else None
            # a-first: a is seat 0; b-first: a is seat 1
            deals.append({'a': a, 'b': b, 'matchup': name, 'deal': k, 'bb': bb, 'seed': seed,
                          'fp': fingerprint(game, seed, k % n), 'planned': len(seeds),
                          'net_a': ha[k]['net'][0] + hb[k]['net'][1]})
    return deals


def load_meta(root) -> dict:
    """{player id: {'identity', 'kind', 'quarantine_reason'}} from the match configs under root."""
    from .history import identity
    meta = {}
    for cfg_path in sorted(Path(root).rglob('config.json')):
        cfg = _json(cfg_path) or {}
        for pid, spec in (cfg.get('specs') or {}).items():
            ident = identity(pid, spec)
            meta[pid] = {'identity': ident, 'kind': 'bot' if 'bot' in spec else 'model',
                         'quarantine_reason': quarantine_reason(ident)}
    return meta


def design_of(plan: dict | None) -> dict:
    """The sequential design a run was planned with. Plans from before designs were recorded: a fixed number of
    deals is one look at the end; `until_decided` is rounds of `deals` up to that cap."""
    reps = {'replicates': plan['replicates']} if plan and (plan.get('replicates') or 1) > 1 else {}
    if plan and plan.get('design'):
        return {**DEFAULT_DESIGN, 'round_deals': plan.get('deals'), **plan['design'], **reps}
    if plan and plan.get('until_decided') and plan.get('deals'):
        return {'alpha': 0.05, 'round_deals': plan['deals'], 'max_rounds': max(1, plan['until_decided'] // plan['deals']),
                'block': BLOCK, 'legacy': 'until_decided', **reps}
    return {'alpha': 0.05, 'round_deals': (plan or {}).get('deals'), 'max_rounds': 1, 'block': BLOCK}


# Observations and blocks.

def observations(deals) -> tuple[list[dict], int]:
    """One observation per distinct deal of a pair, in play order: [{'a', 'b', 'x' (a's bb/100), 'plays',
    'matchup', 'deal'}], and how many plays were repeats folded into an earlier observation."""
    groups = {}
    for d in sorted(deals, key=lambda d: (d.get('order') or 0, d['matchup'], d['deal'])):
        if d['a'] == d['b']:
            continue
        key = (frozenset((d['a'], d['b'])), d.get('fp') or (d['matchup'], d['deal']))
        x = d['net_a'] / d['bb'] / 2 * 100
        g = groups.get(key)
        if g is None:
            g = groups[key] = {'a': d['a'], 'b': d['b'], 'xs': [], 'matchup': d['matchup'], 'deal': d['deal']}
        g['xs'].append(x if d['a'] == g['a'] else -x)
    obs = [{'a': g['a'], 'b': g['b'], 'x': sum(g['xs']) / len(g['xs']), 'plays': len(g['xs']),
            'matchup': g['matchup'], 'deal': g['deal']} for g in groups.values()]
    return obs, sum(o['plays'] - 1 for o in obs)


def _chunks(xs, groups, block):
    """(sizes, sums) of the blocks: runs of `block` consecutive values, restarting at each new group."""
    sizes, sums = [], []
    last, fill = object(), block
    for i, x in enumerate(xs):
        g = groups[i] if groups is not None else None
        if g != last or fill == block:
            sizes.append(0)
            sums.append(0.0)
            fill = 0
            last = g
        sizes[-1] += 1
        sums[-1] += x
        fill += 1
    return np.array(sizes, float), np.array(sums, float)


def _cluster_t(sizes, sums):
    """(mean, se) with blocks as clusters: the batch-means estimate, exact for equal blocks."""
    G, N = len(sizes), sizes.sum()
    m = sums.sum() / N
    var = G / (G - 1) * float(((sums - sizes * m) ** 2).sum()) / N ** 2
    return float(m), math.sqrt(var)


# The stopping rule.

@lru_cache(maxsize=None)
def _null_t(R: int, rounds: int, B: int, reps: int = 1, sims: int = SIMS):
    """|t| at every look of `sims` null sequences (normal deals; each round is `reps` matches of R deals, and
    blocks restart with each match), and the number of blocks at each look. Deterministic: one fixed seed."""
    per = ([B] * (R // B) + ([R % B] if R % B else [])) * reps
    sizes = np.array(per * rounds, float)
    Gs = tuple(k * len(per) for k in range(1, rounds + 1))
    rng = np.random.default_rng(SIM_SEED)
    out = np.zeros((sims, rounds))
    cn2 = np.cumsum(sizes ** 2)
    step = 50_000
    for start in range(0, sims, step):
        n = min(step, sims - start)
        s = rng.standard_normal((n, len(sizes))) * np.sqrt(sizes)
        cs, cs2, csn = np.cumsum(s, 1), np.cumsum(s * s, 1), np.cumsum(s * sizes, 1)
        for k, G in enumerate(Gs):
            if G < 2:
                continue
            N = (k + 1) * R * reps
            m = cs[:, G - 1] / N
            q = cs2[:, G - 1] - 2 * m * csn[:, G - 1] + m * m * cn2[G - 1]
            out[start:start + n, k] = np.abs(m) / np.sqrt(np.maximum(G / (G - 1) * q / N ** 2, 1e-300))
    return out, Gs


@lru_cache(maxsize=None)
def nominal_alpha(alpha_pairing: float, round_deals: int, max_rounds: int, block: int = BLOCK, replicates: int = 1) -> float:
    """The two-sided level each look is tested at so that, over all `max_rounds` looks, a pairing with no edge
    is called for either player with chance alpha_pairing (a Pocock design on the p-value scale, so the bar
    follows each look's degrees of freedom), found by bisection on the simulated null sequences."""
    if max_rounds <= 1:
        return alpha_pairing
    T, Gs = _null_t(int(round_deals), int(max_rounds), int(block), int(replicates))
    def false_rate(a):
        c = np.array([t_ppf(1 - a / 2, G - 1) if G >= 2 else np.inf for G in Gs])
        return float((T >= c).any(axis=1).mean())
    lo, hi = math.log(alpha_pairing / max_rounds / 4), math.log(alpha_pairing)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if false_rate(math.exp(mid)) <= alpha_pairing else (lo, mid)
    return math.exp(lo)


def boundary(alpha_pairing: float, round_deals: int, max_rounds: int, block: int = BLOCK, look: int | None = None,
             replicates: int = 1) -> float:
    """Critical |t| for the block-mean t statistic at look `look` (default: the last) of `max_rounds` rounds of
    `round_deals` deals (times `replicates` matches per round), so that under the null the chance any look
    reaches its bar is alpha_pairing. One look is the Student t quantile; more are calibrated by simulation
    (`nominal_alpha`), deterministic and cached. Blocks restart with each match."""
    R, rounds, B, reps = int(round_deals), int(max_rounds), int(block), int(replicates)
    look = rounds if look is None else max(1, min(int(look), rounds))
    G = look * reps * (R // B + (1 if R % B else 0))
    if G < 2:
        return math.inf
    if rounds <= 1:
        return t_ppf(1 - alpha_pairing / 2, G - 1)
    return t_ppf(1 - nominal_alpha(alpha_pairing, R, rounds, B, reps) / 2, G - 1)


def decide(xs, design=None, n_pairings=1, looks_used=None, groups=None) -> dict:
    """The verdict on one pairing from a's result per duplicate deal (bb/100), at look `looks_used` of the
    design. `groups` (one label per deal, e.g. its matchup) keeps blocks from spanning matches.

    verdict: 'a' or 'b' (|t| reached the boundary), 'even' (the repeated (1 - 2 alpha) interval lies inside
    +-EQUIV_BB100), 'exhausted' (the last planned look, undecided), 'open' (undecided, looks remain, or between
    looks), 'early' (fewer than 2 blocks)."""
    d = {**DEFAULT_DESIGN, **(design or {})}
    xs = [float(x) for x in xs]
    n, B, rounds, reps = len(xs), int(d['block']), int(d['max_rounds']), int(d.get('replicates') or 1)
    R = int(d.get('round_deals') or n or 1)
    alpha_p = d['alpha'] / max(1, n_pairings)
    if looks_used is None:
        looks_used = min(rounds, n // (R * reps))
    sizes, sums = _chunks(xs, groups, B)
    G = len(sizes)
    out = {'deals': n, 'blocks': G, 'edge': round(sum(xs) / n, 1) if n else None, 'se': None, 'ci95': None,
           't': None, 'boundary': None, 'verdict': 'early', 'looks_used': looks_used, 'looks_planned': rounds,
           'alpha_pairing': alpha_p, 'equiv_bb100': EQUIV_BB100, 'ci_decision': None, 'equiv_ci': None,
           'sd_per_deal': None, 'power': {'deals_for_80pct': {str(e): None for e in POWER_EDGES}}}
    if G < 2:
        return out
    m, se = _cluster_t(sizes, sums)
    if rounds > 1:
        look = max(1, min(looks_used, rounds))
        bnd = boundary(alpha_p, R, rounds, B, look, reps)
        beq = boundary(min(1.0, 2 * alpha_p), R, rounds, B, look, reps)
        plan_c = _Z.inv_cdf(1 - nominal_alpha(alpha_p, R, rounds, B, reps) / 2)
    else:
        bnd, beq = t_ppf(1 - alpha_p / 2, G - 1), t_ppf(1 - alpha_p, G - 1)
        plan_c = _Z.inv_cdf(1 - alpha_p / 2)
    t = m / se if se > 0 else (math.copysign(math.inf, m) if m else 0.0)
    q = t_ppf(0.975, G - 1)
    iv = lambda c: [round(m - c * se, 1), round(m + c * se, 1)]
    if looks_used < 1:
        verdict = 'open'
    elif abs(t) >= bnd:
        verdict = 'a' if t > 0 else 'b'
    elif m - beq * se > -EQUIV_BB100 and m + beq * se < EQUIV_BB100:
        verdict = 'even'
    elif looks_used >= rounds:
        verdict = 'exhausted'
    else:
        verdict = 'open'
    sigma = se * math.sqrt(n)
    out.update({'edge': round(m, 1), 'se': round(se, 1), 'ci95': iv(q), 't': round(t, 2) if math.isfinite(t) else None,
                'boundary': round(bnd, 3), 'verdict': verdict, 'ci_decision': iv(bnd), 'equiv_ci': iv(beq),
                'sd_per_deal': round(sigma, 1),
                'power': {'deals_for_80pct': {str(e): math.ceil(((plan_c + _Z.inv_cdf(0.8)) * sigma / e) ** 2)
                                              for e in POWER_EDGES}}})
    return out


# Strength.

def _components(names, obs):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    played = defaultdict(int)
    for o in obs:
        parent[find(o['a'])] = find(o['b'])
        played[o['a']] += 1
        played[o['b']] += 1
    comps = defaultdict(list)
    for n in names:
        comps[find(n)].append(n)
    return sorted((sorted(c) for c in comps.values()), key=lambda c: (-sum(played[p] for p in c), c))


def _blocks(obs, block):
    """[{'a', 'b', 'matchup', 'n', 'sx'}]: runs of `block` consecutive observations within each matchup."""
    out, by = [], defaultdict(list)
    for o in obs:
        by[o['matchup']].append(o)
    for mu, os_ in by.items():
        for i in range(0, len(os_), block):
            chunk = os_[i:i + block]
            a = chunk[0]['a']
            out.append({'a': a, 'b': chunk[0]['b'], 'matchup': mu, 'n': len(chunk),
                        'sx': sum(o['x'] if o['a'] == a else -o['x'] for o in chunk)})
    return out


def _strengths(names, comps, blocks, boot, seed):
    """Point strengths and bootstrap draws (boot x players), each summing to zero within every component."""
    P, G = len(names), len(blocks)
    idx = {n: i for i, n in enumerate(names)}
    D = np.zeros((G, P, P))
    dv = np.zeros((G, P))
    for g, b in enumerate(blocks):
        i, j = idx[b['a']], idx[b['b']]
        D[g, i, i] = D[g, j, j] = 1
        D[g, i, j] = D[g, j, i] = -1
        dv[g, i], dv[g, j] = 1, -1
    J = np.zeros((P, P))
    for c in comps:
        ii = [idx[p] for p in c]
        J[np.ix_(ii, ii)] = 1 / len(c)
    n = np.array([b['n'] for b in blocks], float)
    sx = np.array([b['sx'] for b in blocks], float)

    def solve(w):
        L = np.einsum('bg,gij->bij', w * n, D) + J
        return np.linalg.solve(L, ((w * sx) @ dv)[..., None])[..., 0]
    point = solve(np.ones((1, G)))[0]
    rng = np.random.default_rng(seed)
    w = np.zeros((boot, G))
    by = defaultdict(list)
    for g, b in enumerate(blocks):
        by[frozenset((b['a'], b['b']))].append(g)
    rows = np.arange(boot)[:, None]
    for gs in by.values():
        gs = np.array(gs)
        pick = gs[rng.integers(0, len(gs), (boot, len(gs)))]
        np.add.at(w, (np.broadcast_to(rows, pick.shape), pick), 1)
    return point, solve(w) if boot else np.zeros((0, P))


def holm(ps) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(ps)
    adj, run = [1.0] * m, 0.0
    for r, i in enumerate(sorted(range(m), key=lambda i: ps[i])):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


# Session Elo.

def sessions(obs, size=SESSION) -> list[dict]:
    """Observations grouped in order into sessions per matchup: [{'a', 'b', 'matchup', 'net_bb', 'deals'}].
    A trailing partial session counts if it has at least half the deals."""
    by = {}
    for o in obs:
        by.setdefault(o['matchup'], []).append(o)
    out = []
    for mu, os_ in by.items():
        for k in range(0, len(os_), size):
            chunk = os_[k:k + size]
            if 2 * len(chunk) < size:
                continue
            a = chunk[0]['a']
            out.append({'a': a, 'b': chunk[0]['b'], 'matchup': mu, 'deals': len(chunk),
                        'net_bb': sum((o['x'] if o['a'] == a else -o['x']) / 50 for o in chunk)})
    return out


def outcome(s) -> float:
    band = TIE_BB * s['deals']
    return 1.0 if s['net_bb'] > band else 0.0 if s['net_bb'] < -band else 0.5


def _elo(names, comps, games, iters=60):
    """Bradley-Terry MAP ratings (logits) with a N(0, ELO_PRIOR^2) prior, centred within each component, and
    their Laplace-approximation covariance (the inverse Hessian at the mode, centred the same way)."""
    N = len(names)
    idx = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(games), N))
    y = np.array([outcome(g) for g in games], float)
    for k, g in enumerate(games):
        X[k, idx[g['a']]] += 1
        X[k, idx[g['b']]] -= 1
    w = np.zeros(N)
    pr = np.full(N, 1 / ELO_PRIOR ** 2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X.T * (p * (1 - p))) @ X + np.diag(pr)
        step = np.linalg.solve(H, X.T @ (y - p) - pr * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ w))
    cov = np.linalg.inv((X.T * (p * (1 - p))) @ X + np.diag(pr))
    A = np.eye(N)
    for c in comps:
        ii = [idx[q] for q in c]
        A[np.ix_(ii, ii)] -= 1 / len(c)
    return A @ w, A @ cov @ A.T


# The leaderboard.

def rate(deals, boot=BOOT, seed=0, session=SESSION, *, meta=None, design=None, plan=None, pooled=False,
         n_pairings=None, notes=(), playing=()) -> dict:
    """The leaderboard from duplicate deals (as `load_deals` gives them). `meta` gives each player's identity,
    kind and quarantine reason; with `pooled` (all-time standings) quarantined players are shown but not rated.
    `plan` (a tournament's plan.json) gives the roster and the pairings planned; `playing` players with a match
    under way."""
    meta = meta or {}
    design = dict(design or design_of(plan))
    block = int(design.get('block') or BLOCK)
    raw = [d for d in deals if d['a'] != d['b']]
    info = lambda p: meta.get(p) or {'identity': p, 'kind': 'bot' if p.startswith('bot:') else 'model',
                                     'quarantine_reason': None}
    excluded = {p for d in raw for p in (d['a'], d['b'])
                if pooled and info(p)['quarantine_reason']}
    used = [d for d in raw if d['a'] not in excluded and d['b'] not in excluded]
    obs, repeats = observations(used)
    names = sorted({o['a'] for o in obs} | {o['b'] for o in obs})
    comps = _components(names, obs)
    comp_of = {p: i for i, c in enumerate(comps) for p in c}
    blocks = _blocks(obs, block)
    nblocks = defaultdict(int)
    for b in blocks:
        nblocks[b['a']] += 1
        nblocks[b['b']] += 1
    per_pair = defaultdict(int)
    for b in blocks:
        per_pair[frozenset((b['a'], b['b']))] += 1
    thin = {p for k, v in per_pair.items() if v < 2 for p in k}   # a pairing of one block has no spread to resample
    idx = {n: i for i, n in enumerate(names)}

    point, draws = _strengths(names, comps, blocks, boot, seed) if names else (np.zeros(0), np.zeros((0, 0)))
    infl = lambda g: math.sqrt(g / (g - 1)) if g > 1 else math.inf

    def se_of(i, j=None):
        v = draws[:, idx[i]] - (draws[:, idx[j]] if j else 0)
        return float(np.std(v, ddof=1)) if len(v) > 1 else math.inf

    # pairwise separation within each component, Holm over the component's pairs
    sig, diff_ci, leaders = {}, {}, []
    for ci_, c in enumerate(comps):
        pairs = [(i, j) for k, i in enumerate(c) for j in c[k + 1:]]
        ps = []
        for i, j in pairs:
            g = 0 if i in thin or j in thin else min(nblocks[i], nblocks[j])
            se = se_of(i, j) * infl(g)
            dlt = point[idx[i]] - point[idx[j]]
            if g < 2 or not math.isfinite(se):
                ps.append(1.0)
                diff_ci[(i, j)] = None
                continue
            ps.append(2 * t_sf(abs(dlt) / se, g - 1) if se > 0 else (0.0 if dlt else 1.0))
            q = t_ppf(0.975, g - 1)
            diff_ci[(i, j)] = [round(float(dlt - q * se), 1), round(float(dlt + q * se), 1)]
        for (i, j), p in zip(pairs, holm(ps)):
            sig[(i, j)] = sig[(j, i)] = p < 0.05
        order = sorted(c, key=lambda p: -point[idx[p]])
        top = order[0]
        if len(order) > 1 and all(sig[(top, o)] for o in order[1:]):
            second = order[1]
            dci = diff_ci.get((top, second)) or [-x for x in reversed(diff_ci[(second, top)])]
            leaders.append({'player': top, 'component': ci_, 'over': second,
                            'margin_bb100': round(float(point[idx[top]] - point[idx[second]]), 1), 'ci95': dci})
        else:
            leaders.append(None)

    games = sessions(obs, session)
    elo, cov = _elo(names, comps, games) if names else (np.zeros(0), np.zeros((0, 0)))

    tot = defaultdict(lambda: {'net_bb': 0.0, 'deals': 0})
    for d in raw:
        for p, s in ((d['a'], 1), (d['b'], -1)):
            tot[p]['net_bb'] += s * d['net_a'] / d['bb']
            tot[p]['deals'] += 1
    players = []
    for p in names:
        i = idx[p]
        g = nblocks[p]
        se = se_of(p) * infl(g)
        rated = g >= 2 and p not in thin and math.isfinite(se)
        q = t_ppf(0.975, g - 1) if rated else None
        c = comps[comp_of[p]]
        better = sum(1 for o in c if o != p and sig[(p, o)] and point[idx[o]] > point[i])
        worse = sum(1 for o in c if o != p and sig[(p, o)] and point[idx[o]] < point[i])
        tied = [o for o in sorted(c, key=lambda o: -point[idx[o]]) if o != p and not sig[(p, o)]]
        mine = [s for s in games if p in (s['a'], s['b'])]
        won = sum(1 for s in mine if outcome(s) == (1.0 if s['a'] == p else 0.0))
        ties = sum(1 for s in mine if outcome(s) == 0.5)
        sd = math.sqrt(max(cov[i, i], 0))
        players.append({
            'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': comp_of[p],
            'strength': round(float(point[i]), 1), 'strength_se': round(se, 1) if rated else None,
            'strength_ci95': [round(float(point[i] - q * se), 1), round(float(point[i] + q * se), 1)] if rated else None,
            'blocks': g, 'rank': better + 1 if not tied and len(c) > 1 else None,
            'rank_range': [better + 1, len(c) - worse], 'rank_tied_with': tied,
            'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'], 'deals': tot[p]['deals'],
            'session_elo': round(1500 + K * float(elo[i])),
            'session_elo_ci95': [round(1500 + K * (elo[i] - 1.96 * sd)), round(1500 + K * (elo[i] + 1.96 * sd))],
            'sessions': len(mine), 'session_w': won, 'session_t': ties, 'session_l': len(mine) - won - ties,
            'quarantined': bool(info(p)['quarantine_reason']), 'quarantine_reason': info(p)['quarantine_reason'],
            'unrated_reason': None})
    players.sort(key=lambda x: (x['component'], -x['strength']))
    unrated = sorted(excluded | ({p for d in raw for p in (d['a'], d['b'])} - set(names)))
    for p in unrated:
        players.append({
            'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': None,
            'strength': None, 'strength_se': None, 'strength_ci95': None, 'blocks': 0, 'rank': None,
            'rank_range': None, 'rank_tied_with': [], 'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'],
            'deals': tot[p]['deals'], 'session_elo': None, 'session_elo_ci95': None, 'sessions': 0,
            'session_w': 0, 'session_t': 0, 'session_l': 0, 'quarantined': p in excluded,
            'quarantine_reason': info(p)['quarantine_reason'] if p in excluded else None,
            'unrated_reason': info(p)['quarantine_reason'] if p in excluded else 'played only unrated opponents'})

    planned_pairs = []
    for a, b in (plan or {}).get('matchups', []):
        if {a, b} not in [set(x) for x in planned_pairs]:
            planned_pairs.append((a, b))
    observed = len({frozenset((o['a'], o['b'])) for o in obs})
    pairings = _pairings(used, obs, games, design, n_pairings or len(planned_pairs) or max(1, observed), pooled)

    final = {'a', 'b', 'even', 'exhausted'}
    verdict_of = {frozenset((x['a'], x['b'])): x['verdict'] for x in pairings}
    roster_ids = list((plan or {}).get('players') or [])
    for p in [q for m in (plan or {}).get('matchups', []) for q in m] + [x['player'] for x in players]:
        if p not in roster_ids:
            roster_ids.append(p)
    roster = []
    for p in roster_ids:
        theirs = [frozenset(x) for x in planned_pairs if p in x]
        if p in playing:
            status = 'playing'
        elif not tot[p]['deals']:
            status = 'awaiting'
        elif not all(verdict_of.get(k) in final for k in theirs):
            status = 'playing'
        else:
            status = 'done'
        roster.append({'player': p, 'status': status})

    notes = list(notes)
    if repeats:
        notes.append(f'{repeats} plays repeated a deal the same pair had already played (same cards, same seats); '
                     f'each repeated deal was averaged into one observation.')
    if len(comps) > 1:
        notes.append(f'The players form {len(comps)} groups that never played across: strengths, ranks and leaders '
                     f'compare players only within a group.')
    if pooled and excluded:
        notes.append(f'Not rated (identity unresolved): {", ".join(sorted(excluded))}. Their deals are left out.')
    if set(unrated) - excluded:
        notes.append(f'Not rated (every opponent was unrated): {", ".join(sorted(set(unrated) - excluded))}.')
    elif any(x['quarantined'] for x in players):
        notes.append('Some players have no exact model version recorded (marked quarantined): they are rated in this '
                     'run but not pooled into the all-time standings.')
    if any(len(c) > 2 and _is_tree(c, obs) for c in comps):
        notes.append('Every pairing in a group is its only link between those players, so the strengths fit the '
                     'pairings exactly and nothing checks that edges add up across players.')
    reps = int((plan or {}).get('replicates') or 1)
    if not pooled and pairings:
        notes.append((f'Each round of a pairing was {reps} independent matches with fresh memories; within a match'
                      if reps > 1 else 'Each pairing was played as one continuous match per round:') +
                     f' a model remembers earlier hands, so nearby deals are dependent, which blocks of {block} deals '
                     f'absorb only in part.')
    if design.get('legacy') == 'until_decided':
        notes.append('This run stopped pairings with an older rule (|z| >= 2.7); its verdicts are re-judged here with '
                     'the calibrated boundary, so a pairing that stopped may now read as undecided.')
    if pooled:
        notes.append('All-time verdicts are one test on everything pooled, with alpha split over all pairings; runs '
                     'that stopped early when a pairing looked decided bias pooled edges away from zero a little.')
    notes.append('Session Elo is the share of 10-deal sessions won: a different, secondary statistic that never '
                 'decides ranks.')

    return {'objective': OBJECTIVE, 'design': design, 'deals': len(raw), 'hands': 2 * len(raw),
            'observations': len(obs), 'repeats': repeats, 'block': block, 'boot': boot,
            'equiv_bb100': EQUIV_BB100, 'session_deals': session,
            'components': comps, 'leader': leaders[0] if leaders else None, 'leaders': leaders,
            'players': players, 'pairings': pairings, 'matchups': pairings, 'roster': roster, 'notes': notes}


def _is_tree(comp, obs):
    edges = {frozenset((o['a'], o['b'])) for o in obs if o['a'] in comp}
    return len(edges) == len(comp) - 1


def _pairings(raw, obs, games, design, n_pairings, pooled) -> list:
    """One row per pair of players, pooling every matchup between them (rounds and runs), with its verdict."""
    by = {}
    for o in obs:
        x = by.setdefault(frozenset((o['a'], o['b'])), {'a': o['a'], 'b': o['b'], 'xs': [], 'groups': [],
                                                         'matchups': [], 'chips': 0, 'plays': 0, 'planned': {}})
        x['xs'].append(o['x'] if o['a'] == x['a'] else -o['x'])
        x['groups'].append(o['matchup'])
        if o['matchup'] not in x['matchups']:
            x['matchups'].append(o['matchup'])
    for d in raw:
        x = by[frozenset((d['a'], d['b']))]
        x['chips'] += d['net_a'] if d['a'] == x['a'] else -d['net_a']
        x['plays'] += 1
        x['planned'][d['matchup']] = d.get('planned') or 0
    wins = defaultdict(lambda: [0, 0])
    for g in games:
        x = by[frozenset((g['a'], g['b']))]
        o = outcome(g) if g['a'] == x['a'] else 1 - outcome(g)
        wins[frozenset((g['a'], g['b']))][0] += o == 1.0
        wins[frozenset((g['a'], g['b']))][1] += o == 0.0
    rounds = int(design.get('max_rounds') or 1)
    out = []
    for key, x in by.items():
        xs, n = x['xs'], len(x['xs'])
        if pooled:
            dsg, looks, partial = {**design, 'round_deals': n, 'max_rounds': 1}, 1, False
        elif rounds > 1:
            R = int(design['round_deals']) * int(design.get('replicates') or 1)
            dsg, looks, partial = design, min(rounds, n // R), n % R != 0 and n // R < rounds
        else:
            R = sum(x['planned'].values()) or design.get('round_deals') or n
            dsg, looks, partial = {**design, 'round_deals': R}, int(n >= R), n < R
        v = decide(xs, dsg, n_pairings, looks, groups=x['groups'])
        if (partial and v['verdict'] != 'early') or (pooled and v['verdict'] == 'exhausted'):
            v['verdict'] = 'open'
        out.append({'a': x['a'], 'b': x['b'], 'matchups': x['matchups'], 'matchup': x['matchups'][-1],
                    'plays': x['plays'], 'hands': 2 * x['plays'], 'net_a_chips': x['chips'],
                    'a_session_wins': wins[key][0], 'b_session_wins': wins[key][1], 'in_progress': partial, **v})
    return sorted(out, key=lambda m: (m['a'], m['b']))


def ratings(root, boot=BOOT, seed=0, session=SESSION) -> dict:
    """The leaderboard of every match under root, with each player's style stats."""
    from .stats import collect
    root = Path(root)
    plan = _json(root / 'plan.json')
    playing = set()
    for cfg_path in root.rglob('config.json'):
        if cfg_path.parent.name in ('a-first', 'b-first') and not (cfg_path.parent / 'result.json').exists():
            playing |= set((_json(cfg_path) or {}).get('players', []))
    r = rate(load_deals(root), boot, seed, session, meta=load_meta(root), plan=plan, playing=playing)
    return {**r, 'stats': collect(root)}


VERDICT_TEXT = {
    'a': '{a} is better', 'b': '{b} is better',
    'even': 'even: the edge is within ±{e} bb/100',
    'open': 'undecided so far',
    'exhausted': 'undecided after every planned look (not evidence that they are equal)',
    'early': 'too few deals to judge',
}


def _iv(iv):
    return f'[{iv[0]:+.0f}, {iv[1]:+.0f}]' if iv else '—'


def table(r, labels=None) -> str:
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    lines = [f'Strength: {r["objective"]}, against the average opponent in its group (95% block-bootstrap interval).']
    for ci, comp in enumerate(r['components']):
        if len(r['components']) > 1:
            lines.append(f'\nGroup {ci + 1}')
        lines.append(f'{"rank":>5}  {"player":30} {"strength":>8} {"95% interval":>14} {"net bb":>8} {"hands":>6}'
                     f'  {"session Elo (95%)":>20} {"W-T-L":>7}')
        for p in r['players']:
            if p['component'] != ci:
                continue
            rk = str(p['rank']) if p['rank'] else (f'{p["rank_range"][0]}-{p["rank_range"][1]}' if p['rank_range'] else '—')
            elo = f'{p["session_elo"]} ({p["session_elo_ci95"][0]}–{p["session_elo_ci95"][1]})'
            q = ' *' if p['quarantined'] else ''
            lines.append(f'{rk:>5}  {name(p["player"]) + q:30} {p["strength"]:>+8.1f} {_iv(p["strength_ci95"]):>14} '
                         f'{p["net_bb"]:>+8.1f} {p["hands"]:>6}  {elo:>20} '
                         f'{p["session_w"]}-{p["session_t"]}-{p["session_l"]:<3}')
        lead = r['leaders'][ci] if ci < len(r.get('leaders', [])) else None
        if lead:
            lines.append(f'Leader: {name(lead["player"])}, {lead["margin_bb100"]:+.1f} bb/100 over {name(lead["over"])} '
                         f'(95% {_iv(lead["ci95"])}), separated from everyone after Holm adjustment.')
        else:
            lines.append('No established leader yet: no player is separated from every other one.')
    out = [p for p in r['players'] if p['component'] is None]
    if out:
        lines.append('\nNot rated: ' + '; '.join(
            f'{name(p["player"])} ({p.get("unrated_reason") or p["quarantine_reason"]}, {p["hands"]} hands)' for p in out))
    if any(p['quarantined'] and p['component'] is not None for p in r['players']):
        lines.append('* no exact model version recorded: not pooled into all-time standings.')
    lines.append('A rank like 1-3 means the data cannot yet order the player within that range.')
    return '\n'.join(lines)


def verdicts(r, labels=None) -> str:
    """Each pairing: the edge, its interval, the test against the stopping bar, and the verdict."""
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    order = {'a': 0, 'b': 0, 'even': 1, 'exhausted': 2, 'open': 3, 'early': 4}
    lines = []
    for m in sorted(r['pairings'], key=lambda m: (order[m['verdict']], m['a'], m['b'])):
        a, b = name(m['a']), name(m['b'])
        head = f'{a} vs {b}: '
        if m['edge'] is None:
            lines.append(head + 'no deals yet')
            continue
        head += f'{m["edge"]:+.0f} bb/100 for {a} over {m["deals"]} deals'
        if m['ci95']:
            t = m['t'] if m['t'] is not None else 'inf'
            head += f' (95% {_iv(m["ci95"])}), t = {t}'
            head += (f', no planned look yet (bar at the first look {m["boundary"]:.2f})' if m['looks_used'] < 1 else
                     f' vs bar {m["boundary"]:.2f} at look {m["looks_used"]} of {m["looks_planned"]}')
        state = VERDICT_TEXT[m['verdict']].format(a=a, b=b, e=r.get('equiv_bb100', EQUIV_BB100))
        lines.append(f'{head}: {state}.')
        need = m['power']['deals_for_80pct']
        if m['verdict'] in ('open', 'exhausted') and need.get('50'):
            lines.append('    deals per pairing for an 80% chance to detect a real edge of '
                         + ', '.join(f'{e} bb/100: {need[e]:,}' for e in need) + ' (from the observed spread)')
    return '\n'.join(lines)


def summary(r) -> str:
    """The table, the verdicts and the caveats, as the CLI prints them."""
    return table(r, r.get('labels')) + '\n\n' + verdicts(r, r.get('labels')) + '\n\n' + '\n'.join(f'- {n}' for n in r['notes'])
