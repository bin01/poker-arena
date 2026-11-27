"""Who wins at duplicate heads-up hold'em, by how much, and how sure that is.

Objective. Models are told to win chips, so chips decide: the measure is bb/100 over duplicate deals. A
duplicate deal is hand k of a conversation's two matches (a-first and b-first): each player holds each seat's
cards once against the other, which reduces the luck of the cards (it does not remove it: the two plays of a
deal can go differently). Its value for a against b is

    x = net_a(both plays) / bb / 2 * 100          (bb per 100 hands)

Units. A conversation is one duplicate pair of matches with fresh memories (one replicate of one round of one
pairing): a model remembers everything inside it, so its deals are dependent, and two conversations of the
same pairing can differ for reasons that have nothing to do with the cards. A slate is the set of deals one
round and replicate deal to every pairing (the same seeds), so conversations on one slate share their cards.

Inference and description. Exactly one thing here is a test: the preregistered sequential procedure of each
pairing (`normalize_design`, `decide`). Its verdicts 'a', 'b' and 'even', at a completed planned look, are the
only inferential results: the family-wise chance of any false verdict in a tournament is at most `alpha`
(Bonferroni over the pairings, Pocock-type repeated looks calibrated by simulation). A decided rank and a
leader are built from those verdicts alone: a leader has a decided win over every other player of its group.
Everything else (strengths, their intervals, the order by points, the front runner, session Elo, every
all-time number) is descriptive, recomputed on every refresh, and labelled so.

The test. The statistic is the conversation-level t: the deal-weighted mean edge over the pairing's
conversations, with a cluster-robust standard error that treats each conversation as one unit (C - 1 degrees
of freedom). It is valid whatever the dependence inside a conversation and however much conversations differ,
and it needs at least MIN_CONVERSATIONS conversations: from one conversation, no verdict is possible. Each
look is tested at the same nominal level, found by simulation so that over all planned looks a pairing with
no edge is called with chance alpha / n_pairings. `even` is an equivalence verdict (TOST at +-EQUIV_BB100).

Strength (descriptive). A least-squares paired-comparison fit on deals, x ~ s_a - s_b, with the strengths
summing to zero within each connected group. Repeats of one deal by one pair are averaged into one
observation. Its uncertainty comes from a stratified Rao-Wu rescaling bootstrap that resamples jointly across
pairings, so the correlation that shared cards create between pairings is kept: whole slates when a stratum
(slates with the same pairings) has at least two, else aligned blocks of BLOCK deals within the slate (then
the interval is conditional on those conversations, and says so). Intervals use Student t on the bootstrap's
degrees of freedom (units minus strata). tests/test_calibration.py checks their coverage in seeded correlated
tournament simulations. docs/methodology.md has the details and the limits.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np

OBJECTIVE = 'bb/100 over duplicate deals'
BLOCK = 10                  # deals per aligned block in the within-conversation bootstrap and session Elo
EQUIV_BB100 = 15            # margin of the equivalence test behind 'even'
DEFAULT_DESIGN = {'alpha': 0.05, 'round_deals': 50, 'max_rounds': 6, 'block': BLOCK}
TARGET_EDGE = 50            # bb/100: the edge a plan is powered to detect unless it says otherwise
PRIOR_SD = 350              # bb/100: the per-deal spread assumed when planning (first model runs: about 340)
POWER = 0.8
MIN_CONVERSATIONS = 2       # a verdict needs this many independent conversations of the pairing
POWER_EDGES = (20, 50, 100)
BOOT = 300
SIMS = 400_000              # null sequences behind each nominal level
POWER_SIMS = 40_000
SIM_SEED = 20261126
SESSION = 10                # deals per Elo session
TIE_BB = 0.1                # a session within this many big blinds per deal of even is a tie
ELO_PRIOR = 1.5             # prior sd of a session-Elo rating, in logits
K = 400 / math.log(10)
ALIASES = ('sonnet', 'opus', 'haiku')   # bare CLI aliases: the model version behind them is unknown
DECIDED = ('a', 'b', 'even')
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


@lru_cache(maxsize=8192)
def t_ppf(p, df):
    """The p quantile of Student t (p > 0.5)."""
    if df is None or df > 1e7:
        return _Z.inv_cdf(p)
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > 1 - p:
        hi *= 2
    for _ in range(64):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_sf(mid, df) > 1 - p else (lo, mid)
    return (lo + hi) / 2


# What a player is and which deal was played.

def _parts(identity: str):
    served = identity.split('!served=', 1)[1] if '!served=' in identity else None
    return identity.split('!served=', 1)[0].split('#', 1)[0].split(':'), served


def identity_resolved(identity: str) -> bool:
    """Whether the identity names an exact model version (or a bot): not a bare alias, not a missing model."""
    parts, _ = _parts(identity)
    return parts[0] == 'bot' or (len(parts) >= 3 and parts[1] not in ALIASES)


def quarantine_reason(identity: str) -> str | None:
    """Why a player may not be pooled into all-time standings, or None. Identities are history.identity's:
    backend:model:effort[#protocol], bot:..., backend:id when no model was recorded, and `!served=...` when the
    model that answered was not the one configured."""
    parts, served = _parts(identity)
    if parts[0] != 'bot' and len(parts) < 3:
        return 'no model recorded: which model played is unknown'
    if parts[0] != 'bot' and parts[1] in ALIASES:
        return f'bare alias "{parts[1]}": the model version behind it is unknown'
    if served:
        return f'answered by {served}, not the configured {parts[1]}'
    return None


def served_mismatch(model, served) -> list[str]:
    """The models that answered for a player configured as `model` and are not that model."""
    norm = lambda m: re.sub(r'[^a-z0-9]', '', str(m).lower())
    if not model:
        return []
    m = norm(model)
    return sorted({s for s in served or () if s and not (norm(s).startswith(m) or m.startswith(norm(s)))})


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


def _round_of(name: str, cfg: dict) -> tuple[int, int]:
    tail = name.split('__vs__')[-1]
    r = re.search(r'__r(\d+)$', tail)
    x = re.search(r'__x(\d+)', tail)
    return int(cfg.get('round') or (r.group(1) if r else 1)), int(cfg.get('replicate') or (x.group(1) if x else 1))


def load_deals(root) -> list[dict]:
    """[{'a', 'b', 'matchup' (the conversation), 'deal', 'net_a' (chips, both hands), 'bb', 'seed', 'fp',
    'planned', 'slate', 'round', 'replicate'}] for every duplicate deal under root both of whose plays finished."""
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
        rnd, rep = _round_of(mdir.name, ca)
        for k in sorted(set(ha) & set(hb)):
            seed = seeds[k] if k < len(seeds) else None
            deals.append({'a': a, 'b': b, 'matchup': name, 'deal': k, 'bb': bb, 'seed': seed,
                          'fp': fingerprint(game, seed, k % n), 'planned': len(seeds),
                          'slate': seeds[0] if seeds else name, 'round': rnd, 'replicate': rep,
                          'net_a': ha[k]['net'][0] + hb[k]['net'][1]})
    return deals


def protocol_of(cfg: dict, pid: str) -> str | None:
    """The protocol fingerprint the runner recorded for a player in a match config, if any."""
    p = cfg.get('protocol')
    if isinstance(p, dict):
        p = p.get(pid, p if 'fingerprint' in p else None)
    elif isinstance(p, list) and pid in cfg.get('players', []):
        p = p[cfg['players'].index(pid)]
    if isinstance(p, dict):
        p = p.get('fingerprint')
    return p if isinstance(p, str) and p else None


def load_meta(root, served=None) -> dict:
    """{player id: {'identity', 'kind', 'quarantine_reason', 'identity_resolved', 'protocol'}} from the match
    configs under root. `served` ({player id: [models that answered]}) quarantines a served-model mismatch."""
    from .history import identity
    meta, protos, models = {}, defaultdict(set), {}
    for cfg_path in sorted(Path(root).rglob('config.json')):
        cfg = _json(cfg_path) or {}
        for pid, spec in (cfg.get('specs') or {}).items():
            proto = protocol_of(cfg, pid)
            protos[pid].add(proto)
            models[pid] = spec.get('model')
            if pid not in meta:
                ident = identity(pid, spec, proto)
                meta[pid] = {'identity': ident, 'kind': 'bot' if 'bot' in spec else 'model',
                             'identity_resolved': identity_resolved(ident), 'protocol': proto,
                             'quarantine_reason': quarantine_reason(ident)}
    for pid, m in meta.items():
        if len(protos[pid]) > 1:
            m['quarantine_reason'] = m['quarantine_reason'] or 'its settings (protocol) differ between matches'
        wrong = served_mismatch(models.get(pid), (served or {}).get(pid))
        if wrong and m['kind'] == 'model':
            m['quarantine_reason'] = m['quarantine_reason'] or f'answered by {", ".join(wrong)}, not the configured {models[pid]}'
    return meta


# The preregistered design.

_COMPUTED = ('replicates', 'n_pairings', 'alpha_pairing', 'nominal_alpha', 'deals_needed', 'deals_planned',
             'max_deals', 'power_at_max', 'min_conversations', 'conversations_needed')


def _n_pairings(plan: dict) -> int:
    return max(1, len({frozenset(m) for m in plan.get('matchups') or []}))


def normalize_design(plan: dict | None) -> dict | None:
    """The plan's sequential design with every default filled in: alpha, round_deals, max_rounds, block,
    replicates (independent conversations per pairing per round), n_pairings and alpha_pairing (Bonferroni),
    nominal_alpha (the level each look is tested at), and the preregistered target: target_edge_bb100 (default
    50) and prior_sd_bb100 (default 350), the deals per pairing they need for 80% power (deals_needed, in
    conversations of round_deals), deals_planned (that, capped by max_rounds x round_deals x replicates, i.e.
    max_deals) and the simulated power of the whole procedure at the target (power_at_max). The runner, the
    ratings and the viewer all use this one. None for a plan with no sequential design (one fixed round)."""
    if not plan or not plan.get('deals') and not (plan.get('design') or {}).get('round_deals'):
        return None
    given = {k: v for k, v in (plan.get('design') or {}).items() if k not in _COMPUTED}
    if plan.get('design'):
        d = {**DEFAULT_DESIGN, 'round_deals': plan.get('deals') or DEFAULT_DESIGN['round_deals'], **given}
    elif plan.get('until_decided'):
        d = {**DEFAULT_DESIGN, 'round_deals': plan['deals'],
             'max_rounds': max(1, plan['until_decided'] // plan['deals']), 'legacy': 'until_decided'}
    else:
        return None
    reps = int(plan.get('replicates') or (plan.get('design') or {}).get('replicates') or 1)
    return _complete(d, reps, _n_pairings(plan), plan.get('target_edge_bb100') or d.get('target_edge_bb100'),
                     plan.get('prior_sd_bb100') or d.get('prior_sd_bb100'))


def _fixed_design(round_deals, n_pairings) -> dict:
    """A run planned as one fixed round (older plans, `play`): its only look is at the end."""
    return _complete({**DEFAULT_DESIGN, 'round_deals': int(round_deals or 1), 'max_rounds': 1, 'legacy': 'fixed'},
                     1, n_pairings, None, None)


def _complete(d, reps, n_pairings, target, prior) -> dict:
    R, rounds, B = int(d['round_deals']), int(d['max_rounds']), int(d.get('block') or BLOCK)
    target, prior = float(target or TARGET_EDGE), float(prior or PRIOR_SD)
    ap = d['alpha'] / n_pairings
    a1 = nominal_alpha(ap, R, rounds, B, reps)
    conv = conversations_needed(target, prior, R, a1)
    per_round, cap = R * reps, R * reps * rounds
    planned = min(cap, math.ceil(conv * R / per_round) * per_round)
    return {**d, 'round_deals': R, 'max_rounds': rounds, 'block': B, 'replicates': reps, 'n_pairings': n_pairings,
            'alpha_pairing': ap, 'nominal_alpha': a1, 'min_conversations': MIN_CONVERSATIONS,
            'target_edge_bb100': target, 'prior_sd_bb100': prior, 'conversations_needed': conv,
            'deals_needed': conv * R, 'deals_planned': planned, 'max_deals': cap,
            'power_at_max': power_of(target / prior * math.sqrt(R), ap, rounds, reps)}


def conversations_needed(edge, sd, R, a1, power=POWER) -> int:
    """Conversations of R deals one pairing needs for `power` to detect `edge` with the conversation-level t at
    nominal level a1 (normal approximation to the noncentral t)."""
    for C in range(MIN_CONVERSATIONS, 100_000):
        if _Z.cdf(edge * math.sqrt(C * R) / sd - t_ppf(1 - a1 / 2, C - 1)) >= power:
            return C
    return 100_000


# The stopping rule.

def _look_cs(rounds, reps):
    return [k * reps for k in range(1, rounds + 1)]


@lru_cache(maxsize=None)
def _looks(rounds: int, reps: int, shift: float = 0.0, sims: int = SIMS):
    """The conversation-level t at every look of `sims` sequences whose conversation means are N(shift, 1) (a
    round adds `reps` conversations), 0 where a look has too few conversations. Deterministic: one fixed seed."""
    rng = np.random.default_rng(SIM_SEED)
    cs_ = _look_cs(rounds, reps)
    out = np.zeros((sims, rounds))
    step = 100_000
    for start in range(0, sims, step):
        n = min(step, sims - start)
        z = rng.standard_normal((n, rounds * reps)) + shift
        s, s2 = np.cumsum(z, 1), np.cumsum(z * z, 1)
        for k, C in enumerate(cs_):
            if C < MIN_CONVERSATIONS:
                continue
            m = s[:, C - 1] / C
            var = np.maximum(s2[:, C - 1] - C * m * m, 1e-300) / (C - 1)
            out[start:start + n, k] = m / np.sqrt(var / C)
    return out


def _bars(a, rounds, reps):
    return np.array([t_ppf(1 - a / 2, C - 1) if C >= MIN_CONVERSATIONS else np.inf for C in _look_cs(rounds, reps)])


def nominal_alpha(alpha_pairing: float, round_deals: int, max_rounds: int, block: int = BLOCK, replicates: int = 1) -> float:
    """The two-sided level every look is tested at so that, over all `max_rounds` looks, a pairing with no edge
    is called for either player with chance alpha_pairing (a Pocock design on the p-value scale), found by
    bisection on simulated null sequences. With the conversation-level t it depends only on the looks and the
    conversations per round (round_deals and block are kept for the signature)."""
    return _nominal(float(alpha_pairing), int(max_rounds), int(replicates))


@lru_cache(maxsize=None)
def _nominal(alpha_pairing, rounds, reps):
    if sum(C >= MIN_CONVERSATIONS for C in _look_cs(rounds, reps)) <= 1:
        return alpha_pairing
    T = np.abs(_looks(rounds, reps))
    lo, hi = math.log(alpha_pairing / rounds / 4), math.log(alpha_pairing)
    for _ in range(36):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (T >= _bars(math.exp(mid), rounds, reps)).any(axis=1).mean() <= alpha_pairing else (lo, mid)
    return math.exp(lo)


@lru_cache(maxsize=None)
def power_of(shift: float, alpha_pairing: float, max_rounds: int, replicates: int = 1) -> float:
    """The chance the procedure calls the better player when each conversation's mean edge is `shift` standard
    errors of a conversation mean (edge x sqrt(round_deals) / per-deal sd), by simulation."""
    rounds, reps = int(max_rounds), int(replicates)
    a1 = _nominal(float(alpha_pairing), rounds, reps)
    T = _looks(rounds, reps, round(float(shift), 6), POWER_SIMS)
    hit = np.abs(T) >= _bars(a1, rounds, reps)
    first = hit.argmax(axis=1)
    return round(float((hit.any(axis=1) & (T[np.arange(len(T)), first] > 0)).mean()), 3)


def boundary(alpha_pairing: float, round_deals: int, max_rounds: int, block: int = BLOCK, look: int | None = None,
             replicates: int = 1) -> float:
    """Critical |t| at look `look` (default: the last) for the conversation-level t, with look x replicates
    conversations: the Student t quantile at the calibrated nominal level. inf where a look has fewer than
    MIN_CONVERSATIONS conversations (no verdict is possible there)."""
    rounds, reps = int(max_rounds), int(replicates)
    look = rounds if look is None else max(1, min(int(look), rounds))
    C = look * reps
    if C < MIN_CONVERSATIONS:
        return math.inf
    return t_ppf(1 - nominal_alpha(alpha_pairing, round_deals, rounds, block, reps) / 2, C - 1)


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


def _cr1(sizes, sums):
    """(mean, se): the deal-weighted mean and its cluster-robust (CR1) standard error, each cluster one unit."""
    G, N = len(sizes), sizes.sum()
    m = sums.sum() / N
    return float(m), math.sqrt(G / (G - 1) * float(((sums - sizes * m) ** 2).sum()) / N ** 2)


def decide(xs, design=None, n_pairings=1, looks_used=None, groups=None) -> dict:
    """The verdict on one pairing from a's result per duplicate deal (bb/100), at look `looks_used` of the
    design. `groups` labels each deal's conversation (its match folder); without it all deals are one.

    verdict: 'a' or 'b' (|t| reached the bar), 'even' (the repeated (1 - 2 alpha) interval lies inside
    +-EQUIV_BB100), 'exhausted' (the last planned look, undecided), 'open' (undecided: looks remain, or between
    looks), 'early' (too few deals to estimate a spread). A verdict needs MIN_CONVERSATIONS conversations; with
    fewer, `basis` is 'blocks' and the interval is conditional on the one conversation played (`reason`)."""
    d = {**DEFAULT_DESIGN, **(design or {})}
    xs = [float(x) for x in xs]
    n, B, rounds, reps = len(xs), int(d.get('block') or BLOCK), int(d['max_rounds']), int(d.get('replicates') or 1)
    R = int(d.get('round_deals') or n or 1)
    alpha_p = d['alpha'] / max(1, n_pairings)
    if looks_used is None:
        looks_used = min(rounds, n // (R * reps))
    labels = list(groups) if groups is not None else ['.'] * n
    conv = {}
    for g, x in zip(labels, xs):
        c = conv.setdefault(g, [0, 0.0])
        c[0] += 1
        c[1] += x
    bsizes, bsums = _chunks(xs, labels, B)
    C, G = len(conv), len(bsizes)
    out = {'deals': n, 'blocks': G, 'conversations': C, 'edge': round(sum(xs) / n, 1) if n else None, 'se': None,
           'ci95': None, 't': None, 'df': None, 'basis': None, 'boundary': None, 'verdict': 'early',
           'looks_used': looks_used, 'looks_planned': rounds, 'alpha_pairing': alpha_p, 'equiv_bb100': EQUIV_BB100,
           'ci_decision': None, 'equiv_ci': None, 'sd_per_deal': None, 'reason': None,
           'power': {'deals_for_80pct': {str(e): None for e in POWER_EDGES}}}
    if C >= MIN_CONVERSATIONS:
        m, se = _cr1(np.array([c[0] for c in conv.values()], float), np.array([c[1] for c in conv.values()], float))
        df, basis = C - 1, 'conversations'
    elif G >= 2:
        m, se = _cr1(bsizes, bsums)
        df, basis = G - 1, 'blocks'
    else:
        return out
    a1 = nominal_alpha(alpha_p, R, rounds, B, reps) if rounds > 1 else alpha_p
    aeq = nominal_alpha(min(1.0, 2 * alpha_p), R, rounds, B, reps) if rounds > 1 else min(1.0, 2 * alpha_p)
    bnd, beq = t_ppf(1 - a1 / 2, df), t_ppf(1 - aeq / 2, df)
    t = m / se if se > 0 else (math.copysign(math.inf, m) if m else 0.0)
    iv = lambda c: [round(m - c * se, 1), round(m + c * se, 1)]
    can = basis == 'conversations'
    if looks_used < 1:
        verdict = 'open'
    elif can and abs(t) >= bnd:
        verdict = 'a' if t > 0 else 'b'
    elif can and m - beq * se > -EQUIV_BB100 and m + beq * se < EQUIV_BB100:
        verdict = 'even'
    elif looks_used >= rounds:
        verdict = 'exhausted'
    else:
        verdict = 'open'
    sigma = se * math.sqrt(n)
    plan_c = _Z.inv_cdf(1 - a1 / 2)
    out.update({'edge': round(m, 1), 'se': round(se, 1), 'ci95': iv(t_ppf(0.975, df)), 'df': df, 'basis': basis,
                't': round(t, 2) if math.isfinite(t) else None, 'boundary': round(bnd, 3) if can else None,
                'verdict': verdict, 'ci_decision': iv(bnd) if can else None, 'equiv_ci': iv(beq) if can else None,
                'sd_per_deal': round(sigma, 1),
                'reason': None if can else (f'one conversation: a verdict needs at least {MIN_CONVERSATIONS} independent '
                                            f'conversations (fresh matches) of the pairing'),
                'power': {'deals_for_80pct': {str(e): math.ceil(((plan_c + _Z.inv_cdf(POWER)) * sigma / e) ** 2)
                                              for e in POWER_EDGES}}})
    return out


# Strength, and its uncertainty.

def observations(deals, block=BLOCK) -> tuple[list[dict], int]:
    """One observation per distinct deal of a pair, in play order: [{'a', 'b', 'x' (a's bb/100), 'plays',
    'matchup', 'deal', 'slate', 'pos', 'round', 'replicate', 'planned'}], and how many plays were repeats folded
    into an earlier observation. `slate` and `pos` place a deal among the deals every pairing was dealt."""
    groups = {}
    for d in sorted(deals, key=lambda d: (d.get('order') or 0, d['matchup'], d['deal'])):
        if d['a'] == d['b']:
            continue
        key = (frozenset((d['a'], d['b'])), d.get('fp') or (d['matchup'], d['deal']))
        x = d['net_a'] / d['bb'] / 2 * 100
        g = groups.get(key)
        if g is None:
            slate = d.get('slate')
            if slate is None:
                slate = d['seed'] - d['deal'] if d.get('seed') is not None else d['matchup']
            g = groups[key] = {'a': d['a'], 'b': d['b'], 'xs': [], 'matchup': d['matchup'], 'deal': d['deal'],
                               'slate': slate, 'pos': d['deal'] // block, 'round': d.get('round') or 1,
                               'replicate': d.get('replicate') or 1, 'planned': d.get('planned') or 0}
        g['xs'].append(x if d['a'] == g['a'] else -x)
    obs = [{**{k: v for k, v in g.items() if k != 'xs'}, 'x': sum(g['xs']) / len(g['xs']), 'plays': len(g['xs'])}
           for g in groups.values()]
    return obs, sum(o['plays'] - 1 for o in obs)


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


def _cells(obs):
    """[{'a', 'b', 'slate', 'pos', 'n', 'sx'}]: each pairing's observations in one aligned block of a slate."""
    agg = {}
    for o in obs:
        key = (str(o['slate']), o['pos'], frozenset((o['a'], o['b'])))
        c = agg.get(key)
        if c is None:
            c = agg[key] = {'a': o['a'], 'b': o['b'], 'slate': str(o['slate']), 'pos': o['pos'], 'n': 0, 'sx': 0.0}
        c['n'] += 1
        c['sx'] += o['x'] if o['a'] == c['a'] else -o['x']
    return list(agg.values())


def _strata(cells):
    """The bootstrap's strata: [(units, kind, pairings)], a unit being a list of cell indices. Slates with the
    same pairings form a stratum of whole slates when there are at least two ('conversations'); a slate alone
    in its stratum is split into its aligned blocks, grouped by the pairings present ('blocks')."""
    by_slate = defaultdict(list)
    for i, c in enumerate(cells):
        by_slate[c['slate']].append(i)
    pairs = lambda ix: frozenset(frozenset((cells[i]['a'], cells[i]['b'])) for i in ix)
    same = defaultdict(list)
    for s, ix in by_slate.items():
        same[pairs(ix)].append(ix)
    out = []
    for ps, slates in same.items():
        if len(slates) >= 2:
            out.append((slates, 'conversations', ps))
            continue
        by_pos = defaultdict(list)
        for i in slates[0]:
            by_pos[cells[i]['pos']].append(i)
        sub = defaultdict(list)
        for ix in by_pos.values():
            sub[pairs(ix)].append(ix)
        out += [(units, 'blocks', ps2) for ps2, units in sub.items()]
    return out


def _weights(cells, strata, boot, rng):
    """Rao-Wu rescaled bootstrap weights (boot x cells): in each stratum of k units, k - 1 units drawn with
    replacement, each draw weighted k / (k - 1), so the variance of a linear statistic is unbiased; a cell keeps
    the weight of its unit, the same for every pairing in it (shared cards stay together)."""
    W = np.ones((boot, len(cells)))
    rows = np.arange(boot)[:, None]
    for units, _, _ in strata:
        k = len(units)
        if k < 2:
            continue
        counts = np.zeros((boot, k))
        np.add.at(counts, (np.broadcast_to(rows, (boot, k - 1)), rng.integers(0, k, (boot, k - 1))), 1)
        for j, ix in enumerate(units):
            W[:, ix] = (counts[:, j] * k / (k - 1))[:, None]
    return W


def _strengths(names, comps, cells, W):
    """Point strengths and one strength vector per row of weights, each summing to zero within every group."""
    P, G = len(names), len(cells)
    idx = {n: i for i, n in enumerate(names)}
    D = np.zeros((G, P, P))
    dv = np.zeros((G, P))
    for g, c in enumerate(cells):
        i, j = idx[c['a']], idx[c['b']]
        D[g, i, i] = D[g, j, j] = 1
        D[g, i, j] = D[g, j, i] = -1
        dv[g, i], dv[g, j] = 1, -1
    J = np.zeros((P, P))
    for comp in comps:
        ii = [idx[p] for p in comp]
        J[np.ix_(ii, ii)] = 1 / len(comp)
    n = np.array([c['n'] for c in cells], float)
    sx = np.array([c['sx'] for c in cells], float)

    def solve(w):
        L = np.einsum('bg,gij->bij', w * n, D) + J
        return np.linalg.solve(L, ((w * sx) @ dv)[..., None])[..., 0]
    return solve(np.ones((1, G)))[0], (solve(W) if len(W) else np.zeros((0, P)))


def _df_basis(players, strata):
    """Degrees of freedom (units minus strata, over the strata that hold any of the players' pairings) and
    the basis of the interval: 'conversations', 'blocks' or 'mixed'."""
    df, kinds = 0, set()
    for units, kind, ps in strata:
        if len(units) >= 2 and any(set(players) & set(p) for p in ps):
            df += len(units) - 1
            kinds.add(kind)
    return df, (kinds.pop() if len(kinds) == 1 else 'mixed' if kinds else None)


# Session Elo.

def sessions(obs, size=SESSION) -> list[dict]:
    """Observations grouped in order into sessions per conversation: [{'a', 'b', 'matchup', 'net_bb', 'deals'}].
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
         n_pairings=None, notes=(), playing=(), recorded=None) -> dict:
    """The leaderboard from duplicate deals (as `load_deals` gives them). `meta` gives each player's identity,
    kind and quarantine reason; with `pooled` (all-time standings) everything is descriptive and quarantined
    players are listed but left out of the fit. `plan` (a tournament's plan.json) gives the design, the roster
    and the pairings planned; `playing` players with a match under way; `recorded` the run's decisions.json."""
    meta = meta or {}
    raw = [d for d in deals if d['a'] != d['b']]
    planned_pairs = []
    for a, b in (plan or {}).get('matchups', []):
        if {a, b} not in [set(x) for x in planned_pairs]:
            planned_pairs.append((a, b))
    seen = {frozenset((d['a'], d['b'])) for d in raw}
    n_pairings = n_pairings or len(planned_pairs) or max(1, len(seen))
    if pooled:
        design = None
    elif design is None:
        design = normalize_design(plan) if plan else None
        if design is None:
            design = _fixed_design((plan or {}).get('deals') or max((d.get('planned') or 0 for d in raw), default=1),
                                   n_pairings)
    block = int((design or {}).get('block') or BLOCK)
    info = lambda p: meta.get(p) or {'identity': p, 'kind': 'bot' if p.startswith('bot:') else 'model',
                                     'quarantine_reason': None, 'identity_resolved': True}
    excluded = {p for d in raw for p in (d['a'], d['b']) if pooled and info(p)['quarantine_reason']}
    used = [d for d in raw if d['a'] not in excluded and d['b'] not in excluded]
    obs, repeats = observations(used, block)
    names = sorted({o['a'] for o in obs} | {o['b'] for o in obs})
    comps = _components(names, obs)
    comp_of = {p: i for i, c in enumerate(comps) for p in c}
    idx = {n: i for i, n in enumerate(names)}

    cells = _cells(obs)
    for c in cells:                     # groups never share a stratum: one group's pairings don't shape another's
        c['slate'] = f'{comp_of[c["a"]]}|{c["slate"]}'
    strata = _strata(cells)
    rng = np.random.default_rng(seed)
    W = _weights(cells, strata, boot, rng) if names else np.zeros((0, 0))
    point, draws = _strengths(names, comps, cells, W) if names else (np.zeros(0), np.zeros((0, 0)))

    def interval(ps, vec):
        df, basis = _df_basis(ps, strata)
        if df < 1 or len(draws) < 2:
            return None, None, df, basis
        se = float(np.std(draws @ vec, ddof=1))
        est = float(point @ vec)
        q = t_ppf(0.975, df)
        return round(se, 1), [round(est - q * se, 1), round(est + q * se, 1)], df, basis

    def unit(*pairs):
        v = np.zeros(len(names))
        for p, s in pairs:
            v[idx[p]] += s
        return v

    pairings = _pairings(used, obs, design, n_pairings, pooled, recorded or {})
    vmap = {frozenset((x['a'], x['b'])): x for x in pairings}

    def result(p, q):
        """'win', 'loss', 'even' or None: the decided verdict of p against q."""
        x = vmap.get(frozenset((p, q)))
        if pooled or not x or x['verdict'] not in DECIDED:
            return None
        if x['verdict'] == 'even':
            return 'even'
        return 'win' if (x['verdict'] == 'a') == (x['a'] == p) else 'loss'

    games = sessions(obs, session)
    elo, cov = _elo(names, comps, games) if names else (np.zeros(0), np.zeros((0, 0)))
    tot = defaultdict(lambda: {'net_bb': 0.0, 'deals': 0})
    for d in raw:
        for p, s in ((d['a'], 1), (d['b'], -1)):
            tot[p]['net_bb'] += s * d['net_a'] / d['bb']
            tot[p]['deals'] += 1

    players, groups, leaders = [], [], []
    for ci, comp in enumerate(comps):
        order = sorted(comp, key=lambda p: -point[idx[p]])
        decided = {}
        for p in comp:
            res = {q: result(p, q) for q in comp if q != p}
            better = sum(r == 'loss' for r in res.values())
            worse = sum(r == 'win' for r in res.values())
            full = len(comp) > 1 and all(r in ('win', 'loss') for r in res.values())
            decided[p] = (better + 1 if full else None, [better + 1, len(comp) - worse],
                          [q for q in order if q != p and res[q] not in ('win', 'loss')])
        ranks = [v[0] for v in decided.values() if v[0]]
        for p in comp:
            if decided[p][0] and ranks.count(decided[p][0]) > 1:     # a cycle of verdicts: no strict order
                decided[p] = (None,) + decided[p][1:]
        lead = next((p for p in comp if len(comp) > 1 and all(result(p, q) == 'win' for q in comp if q != p)), None)
        front = order[0] if len(order) > 1 else None
        margin = ci95 = None
        if front:
            _, ci95, _, _ = interval([front, order[1]], unit((front, 1), (order[1], -1)))
            margin = round(float(point[idx[front]] - point[idx[order[1]]]), 1)
        mine = [x for x in pairings if x['a'] in comp]
        have = [x['deals'] for x in mine]
        power = {'target_edge_bb100': (design or {}).get('target_edge_bb100'),
                 'deals_needed': (design or {}).get('deals_needed'), 'deals_planned': (design or {}).get('deals_planned'),
                 'max_deals': (design or {}).get('max_deals'), 'power_at_max': (design or {}).get('power_at_max'),
                 'deals_per_pairing': [min(have), max(have)] if have else None}
        leader = None
        if lead:
            second = next(q for q in order if q != lead)
            _, lci, _, _ = interval([lead, second], unit((lead, 1), (second, -1)))
            beats = [(q, vmap[frozenset((lead, q))]) for q in order if q != lead]
            leader = {'player': lead, 'component': ci, 'over': second, 'basis': 'sequential verdicts',
                      'margin_bb100': round(float(point[idx[lead]] - point[idx[second]]), 1), 'ci95': lci,
                      'beats': [{'player': q, 'edge': x['edge'] if x['a'] == lead else -x['edge'],
                                 'looks_used': x['looks_used']} for q, x in beats]}
        leaders.append(leader)
        groups.append({
            'id': ci, 'players': comp, 'leader': lead, 'leader_detail': leader, 'front_runner': front,
            'over': order[1] if front else None, 'front_runner_margin': margin, 'ci95': ci95, 'power': power,
            'pairings': len(mine), 'decided': sum(x['verdict'] in DECIDED for x in mine),
            'has_bot': any(info(p)['kind'] == 'bot' for p in comp),
            'kind': 'descriptive' if pooled else 'decided' if lead else 'descriptive',
            'headline': _headline(lead, front, margin, ci95, mine, pooled)})
        for pos, p in enumerate(order, 1):
            i = idx[p]
            se, sci, df, basis = interval([p], unit((p, 1)))
            mine_s = [s for s in games if p in (s['a'], s['b'])]
            won = sum(1 for s in mine_s if outcome(s) == (1.0 if s['a'] == p else 0.0))
            ties = sum(1 for s in mine_s if outcome(s) == 0.5)
            sd = math.sqrt(max(cov[i, i], 0))
            rank, rng_, tied = decided[p]
            players.append({
                'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': ci,
                'strength': round(float(point[i]), 1), 'strength_se': se, 'strength_ci95': sci, 'df': df,
                'ci_basis': basis, 'position': pos, 'rank': rank, 'rank_kind': 'decided' if rank else 'descriptive',
                'rank_range': rng_, 'rank_tied_with': tied, 'in_fit': True,
                'identity_resolved': bool(info(p).get('identity_resolved', identity_resolved(info(p)['identity']))),
                'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'], 'deals': tot[p]['deals'],
                'session_elo': round(1500 + K * float(elo[i])),
                'session_elo_ci95': [round(1500 + K * (elo[i] - 1.96 * sd)), round(1500 + K * (elo[i] + 1.96 * sd))],
                'sessions': len(mine_s), 'session_w': won, 'session_t': ties, 'session_l': len(mine_s) - won - ties,
                'quarantined': bool(info(p)['quarantine_reason']), 'quarantine_reason': info(p)['quarantine_reason'],
                'unrated_reason': None})
    unrated = sorted(excluded | ({p for d in raw for p in (d['a'], d['b'])} - set(names)))
    for p in unrated:
        players.append({
            'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': None,
            'strength': None, 'strength_se': None, 'strength_ci95': None, 'df': None, 'ci_basis': None,
            'position': None, 'rank': None, 'rank_kind': 'descriptive', 'rank_range': None, 'rank_tied_with': [],
            'in_fit': False, 'identity_resolved': bool(info(p).get('identity_resolved', identity_resolved(info(p)['identity']))),
            'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'], 'deals': tot[p]['deals'],
            'session_elo': None, 'session_elo_ci95': None, 'sessions': 0, 'session_w': 0, 'session_t': 0,
            'session_l': 0, 'quarantined': p in excluded,
            'quarantine_reason': info(p)['quarantine_reason'] if p in excluded else None,
            'unrated_reason': info(p)['quarantine_reason'] if p in excluded else 'played only unrated opponents'})

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

    notes = list(notes) + _notes(pooled, design, pairings, comps, groups, obs, repeats, excluded, unrated, players, block)
    return {'objective': OBJECTIVE, 'design': design, 'pooled': pooled, 'deals': len(raw), 'hands': 2 * len(raw),
            'observations': len(obs), 'repeats': repeats, 'block': block, 'boot': boot,
            'equiv_bb100': EQUIV_BB100, 'session_deals': session, 'min_conversations': MIN_CONVERSATIONS,
            'inference': ('descriptive only: pooled across runs, so no result here is a test' if pooled else
                          'decided results come only from the preregistered sequential test of each pairing; '
                          'strengths, intervals and the order by points are descriptive'),
            'components': comps, 'groups': groups, 'leader': leaders[0] if leaders else None, 'leaders': leaders,
            'players': players, 'pairings': pairings, 'matchups': pairings, 'roster': roster, 'notes': notes}


def _headline(lead, front, margin, ci95, pairings, pooled) -> str:
    """One line per group, from that group's own data only."""
    if lead:
        return f'{lead} is the decided leader: it won its pairing against every other player in this group.'
    if not front:
        return 'Only one player in this group.'
    open_ = sum(x['verdict'] not in DECIDED for x in pairings)
    what = 'pooled across runs, so nothing here is decided' if pooled else (
        f'no decided leader ({open_} of {len(pairings)} pairings undecided)')
    iv = f', 95% {_iv(ci95)}' if ci95 else ''
    return f'{front} is ahead on points by {margin:+.0f} bb/100 over the next player{iv}; {what}.'


def _notes(pooled, design, pairings, comps, groups, obs, repeats, excluded, unrated, players, block) -> list:
    notes = []
    if pooled:
        notes.append('All-time standings pool runs with different designs and stopping times, so they are '
                     'descriptive only: no verdict, rank or leader here is a test result. Only identical '
                     'protocols pool (model, reasoning level and the settings that change play).')
    else:
        notes.append(f'Decided results come only from the preregistered sequential test of each pairing (at most '
                     f'{design["alpha"]:.0%} chance of any false verdict over its {design["n_pairings"]} '
                     f'pairing{"s" if design["n_pairings"] != 1 else ""}); '
                     f'strengths, intervals and the order by points are descriptive.')
        if any(x['basis'] == 'blocks' for x in pairings):
            notes.append(f'Some pairings have one conversation (one fresh match pair): no verdict is possible from '
                         f'one, and their intervals describe that conversation only (blocks of {block} deals), not '
                         f'how a new match would go.')
    if any(p['ci_basis'] in ('blocks', 'mixed') for p in players if p['in_fit']):
        notes.append('Strength intervals marked "blocks" rest on blocks of deals within single conversations, so '
                     'they leave out how much fresh matches differ.')
    if repeats:
        notes.append(f'{repeats} plays repeated a deal the same pair had already played (same cards, same seats); '
                     f'each repeated deal was averaged into one observation.')
    if len(comps) > 1:
        notes.append(f'The players form {len(comps)} groups that never played across: strengths and leaders '
                     f'compare players only within a group.')
    if any(not g['has_bot'] for g in groups if len(g['players']) > 1):
        notes.append('A group without a reference bot has no fixed anchor: its strengths only compare its own players.')
    if pooled and excluded:
        notes.append(f'Not rated (identity unresolved or a different model answered): {", ".join(sorted(excluded))}. '
                     f'Their deals are left out.')
    if set(unrated) - excluded:
        notes.append(f'Not rated (every opponent was unrated): {", ".join(sorted(set(unrated) - excluded))}.')
    elif not pooled and any(x['quarantined'] for x in players):
        notes.append('Some players are quarantined (no exact model version, mixed settings, or another model '
                     'answered): they are rated in this run but not pooled into the all-time standings.')
    if any(len(c) > 2 and _is_tree(c, obs) for c in comps):
        notes.append('Every pairing in a group is its only link between those players, so the strengths fit the '
                     'pairings exactly and nothing checks that edges add up across players.')
    changed = [x for x in pairings if x.get('recorded_verdict') and x['recorded_verdict'] != x['verdict']]
    if changed:
        notes.append('The run recorded different verdicts with an earlier version of the test ('
                     + ', '.join(f'{x["a"]} vs {x["b"]}: {x["recorded_verdict"]}' for x in changed)
                     + '); shown here is the current test on the same looks, and a pairing it stopped is not resumed.')
    if design and design.get('legacy') == 'until_decided':
        notes.append('This run stopped pairings with an older rule (|z| >= 2.7); its verdicts are re-judged here with '
                     'the current test, so a pairing that stopped may now read as undecided.')
    notes.append('Session Elo is the share of 10-deal sessions won: a different, secondary statistic that never '
                 'decides anything.')
    return notes


def _is_tree(comp, obs):
    edges = {frozenset((o['a'], o['b'])) for o in obs if o['a'] in comp}
    return len(edges) == len(comp) - 1


def _pairings(raw, obs, design, n_pairings, pooled, recorded) -> list:
    """One row per pair of players, pooling every conversation between them, with its verdict at the last
    completed planned look (pooled: descriptive only)."""
    by = {}
    for o in obs:
        x = by.setdefault(frozenset((o['a'], o['b'])), {'a': o['a'], 'b': o['b'], 'obs': [], 'matchups': [],
                                                         'chips': 0, 'plays': 0})
        x['obs'].append(o)
        if o['matchup'] not in x['matchups']:
            x['matchups'].append(o['matchup'])
    for d in raw:
        x = by[frozenset((d['a'], d['b']))]
        x['chips'] += d['net_a'] if d['a'] == x['a'] else -d['net_a']
        x['plays'] += 1
    wins = defaultdict(lambda: [0, 0])
    for g in sessions(obs):
        key = frozenset((g['a'], g['b']))
        o = outcome(g) if g['a'] == by[key]['a'] else 1 - outcome(g)
        wins[key][0] += o == 1.0
        wins[key][1] += o == 0.0
    out = []
    for key, x in by.items():
        os_ = x['obs']
        sign = lambda o: o['x'] if o['a'] == x['a'] else -o['x']
        if pooled:
            v = decide([sign(o) for o in os_], {'max_rounds': 1, 'round_deals': len(os_), 'block': BLOCK},
                       n_pairings, 0, groups=[o['matchup'] for o in os_])
            v.update(verdict='descriptive', boundary=None, ci_decision=None, equiv_ci=None, reason=None)
            looks, beyond = 0, 0
        else:
            rounds, reps = int(design['max_rounds']), int(design.get('replicates') or 1)
            conv = defaultdict(lambda: [0, 0])
            for o in os_:
                c = conv[(o['round'], o['replicate'])]
                c[0] += 1
                c[1] = o['planned'] or design['round_deals']
            looks = 0
            for r in range(1, rounds + 1):
                if all(conv[(r, j)][0] and conv[(r, j)][0] >= conv[(r, j)][1] for j in range(1, reps + 1)):
                    looks = r
                else:
                    break
            at = [o for o in os_ if o['round'] <= looks] if looks else os_
            v = decide([sign(o) for o in at], design, n_pairings, looks, groups=[o['matchup'] for o in at])
            beyond = len(os_) - len(at) if looks else 0
        rec = recorded.get(f'{x["a"]}__vs__{x["b"]}') or recorded.get(f'{x["b"]}__vs__{x["a"]}') or []
        last = rec[-1] if rec else {}
        out.append({'a': x['a'], 'b': x['b'], 'matchups': x['matchups'], 'matchup': x['matchups'][-1],
                    'plays': x['plays'], 'hands': 2 * x['plays'], 'net_a_chips': x['chips'],
                    'deals_total': len(os_), 'deals_since_look': beyond, 'in_progress': bool(beyond),
                    'a_session_wins': wins[key][0], 'b_session_wins': wins[key][1], **v,
                    'recorded_verdict': last.get('verdict'), 'recorded_looks': last.get('looks_used')})
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
    stats = collect(root)
    meta = load_meta(root, {p: s.get('served_models') for p, s in stats.items()})
    r = rate(load_deals(root), boot, seed, session, meta=meta, plan=plan, playing=playing,
             recorded=_json(root / 'decisions.json') or {})
    return {**r, 'stats': stats}


VERDICT_TEXT = {
    'a': '{a} is better (decided)', 'b': '{b} is better (decided)',
    'even': 'even (decided): the edge is within ±{e} bb/100',
    'open': 'undecided so far',
    'exhausted': 'undecided after every planned look (not evidence that they are equal)',
    'early': 'too few deals to judge',
    'descriptive': 'descriptive only (pooled across runs, not a test)',
}


def _iv(iv):
    return f'[{iv[0]:+.0f}, {iv[1]:+.0f}]' if iv else '—'


def table(r, labels=None) -> str:
    """The descriptive standings, one block per group, after the group's own one-line summary."""
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    lines = [f'Standings (descriptive): strength = {r["objective"]} against the average opponent in its group, '
             f'with a 95% interval from a bootstrap that keeps shared cards and conversations together.']
    for g in r.get('groups', []):
        if len(r['groups']) > 1:
            lines.append(f'\nGroup {g["id"] + 1}')
        lines.append(f'{"#":>3} {"rank":>5}  {"player":30} {"strength":>8} {"95% interval":>14} {"basis":>13} '
                     f'{"net bb":>8} {"hands":>6}  {"session Elo (95%)":>20} {"W-T-L":>7}')
        for p in r['players']:
            if p['component'] != g['id']:
                continue
            rk = str(p['rank']) if p['rank'] else '—'
            elo = f'{p["session_elo"]} ({p["session_elo_ci95"][0]}–{p["session_elo_ci95"][1]})'
            q = ' *' if p['quarantined'] else ''
            lines.append(f'{p["position"]:>3} {rk:>5}  {name(p["player"]) + q:30} {p["strength"]:>+8.1f} '
                         f'{_iv(p["strength_ci95"]):>14} {p["ci_basis"] or "—":>13} {p["net_bb"]:>+8.1f} {p["hands"]:>6}  '
                         f'{elo:>20} {p["session_w"]}-{p["session_t"]}-{p["session_l"]:<3}')
        lines.append(_named(g['headline'], name, g['players']))
    out = [p for p in r['players'] if p['component'] is None]
    if out:
        lines.append('\nNot rated: ' + '; '.join(
            f'{name(p["player"])} ({p.get("unrated_reason") or p["quarantine_reason"]}, {p["hands"]} hands)' for p in out))
    if any(p['quarantined'] and p['component'] is not None for p in r['players']):
        lines.append('* quarantined: ' + '; '.join(f'{name(p["player"])}: {p["quarantine_reason"]}'
                                                   for p in r['players'] if p['quarantined'] and p['component'] is not None))
    lines.append('# is the order by points (descriptive). rank is shown only when the sequential verdicts decide it.')
    return '\n'.join(lines)


def _named(text, name, ids):
    for p in sorted(ids, key=len, reverse=True):
        text = re.sub(rf'(?<![\w:-]){re.escape(p)}(?![\w:-])', name(p), text)
    return text


def design_line(d) -> str:
    """The preregistered target in words."""
    if not d:
        return ''
    reach = (f'the plan allows at most {d["max_deals"]:,} (power at the target: {d["power_at_max"]:.0%})'
             if d['deals_needed'] > d['max_deals'] else f'the plan allows {d["max_deals"]:,}')
    head = (f'This run had no preregistered target. To detect a {d["target_edge_bb100"]:.0f} bb/100 edge'
            if d.get('legacy') else f'Planned to detect a {d["target_edge_bb100"]:.0f} bb/100 edge')
    return (f'{head} with 80% power, assuming a per-deal spread of {d["prior_sd_bb100"]:.0f} bb/100: that needs about {d["deals_needed"]:,} deals per pairing '
            f'({d["conversations_needed"]} conversations of {d["round_deals"]}); {reach}.')


def verdicts(r, labels=None) -> str:
    """Each pairing: the edge, its interval, the test against the stopping bar, and the verdict."""
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    order = {'a': 0, 'b': 0, 'even': 1, 'exhausted': 2, 'open': 3, 'early': 4, 'descriptive': 5}
    d = r.get('design')
    if r.get('pooled'):
        lines = ['Pairings (descriptive: pooled across runs, nothing here is a test):']
    else:
        lines = [f'Pairings (decided only by the preregistered sequential test: at most {d["alpha"]:.0%} chance of '
                 f'any false verdict over {d["n_pairings"]} pairing{"s" if d["n_pairings"] != 1 else ""}; a verdict needs '
                 f'{MIN_CONVERSATIONS}+ conversations):']
    for m in sorted(r['pairings'], key=lambda m: (order[m['verdict']], m['a'], m['b'])):
        a, b = name(m['a']), name(m['b'])
        head = f'  {a} vs {b}: '
        if m['edge'] is None:
            lines.append(head + 'no deals yet')
            continue
        head += f'{m["edge"]:+.0f} bb/100 for {a} over {m["deals"]} deals in {m["conversations"]} conversation' \
                + ('s' if m['conversations'] != 1 else '')
        if m['ci95']:
            head += f' (95% {_iv(m["ci95"])})'
            if m['boundary'] is not None:
                t = m['t'] if m['t'] is not None else 'inf'
                head += (', no planned look yet' if m['looks_used'] < 1 else
                         f', t = {t} vs bar {m["boundary"]:.2f} at look {m["looks_used"]} of {m["looks_planned"]}')
        state = VERDICT_TEXT[m['verdict']].format(a=a, b=b, e=r.get('equiv_bb100', EQUIV_BB100))
        lines.append(f'{head}: {state}.')
        if m.get('reason') and m['verdict'] != 'descriptive':
            lines.append(f'      {m["reason"]}.')
        if m.get('deals_since_look'):
            lines.append(f'      {m["deals_since_look"]} more deals are in a round under way, not yet looked at.')
        need = m['power']['deals_for_80pct']
        if m['verdict'] in ('open', 'exhausted') and need.get('50'):
            lines.append('      deals per pairing for an 80% chance to detect a real edge of '
                         + ', '.join(f'{e} bb/100: {need[e]:,}' for e in need) + ' (from the observed spread)')
    lead = [x for x in r.get('leaders') or [] if x]
    if not r.get('pooled'):
        lines.append('Decided leader: ' + ('; '.join(f'{name(x["player"])} (won its pairing against every other '
                                                    f'player of its group)' for x in lead) if lead else 'none.'))
        if d:
            lines.append(design_line(d))
    return '\n'.join(lines)


def summary(r) -> str:
    """The verdicts, the standings and the caveats, as the CLI prints them."""
    labels = r.get('labels')
    return verdicts(r, labels) + '\n\n' + table(r, labels) + '\n\n' + '\n'.join(f'- {n}' for n in r['notes'])
