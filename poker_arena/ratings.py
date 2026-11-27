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
pairing (`normalize_design`, `decide`), applied with every eligibility rule by one function, `adjudicate`,
which the runner stops and records on. Its verdicts 'a', 'b' and 'even', at a completed planned look, are the
only inferential results, and only when the run is preregistered (`registration_status`: a registration
written into plan.json before the first hand, never recomputed), the pairing is eligible (identities resolved
and attested, turns audited, protocols consistent, duplicate pairs valid; else the terminal verdict
'ineligible'), and the verdict is official (recorded at that look by the registered judge; what today's code
recomputes is retrospective). Every verdict carries its assumption status (`assumptions`). A decided rank and
a leader are built from official verdicts within the calibrated range alone: a leader has such a win over
every other player of its group. Everything else (strengths, their intervals and fit diagnostics, the order by
points, the front runner, session Elo, every all-time number, every number of a run that is not
preregistered) is descriptive, recomputed on every refresh, and labelled so.

The test. The unit is the conversation; a look can decide only with at least MIN_CONVERSATIONS of them and
some spread between them. A winner needs two arms to agree at a look: the conversation-level t (the
deal-weighted mean edge over its cluster-robust standard error, C - 1 degrees of freedom) beyond a bar
calibrated by simulation, and a sign-flip randomization test of the same statistic with its own error spent
look by look. The sign-flip arm makes the guarantee exact for any conversation results symmetric about the
true edge (any tails, any discreteness, any dependence inside a conversation: e.g. two interchangeable
players). The t arm's bar is calibrated for a stated family of skewed results: conversation effects of
exponential, lognormal or rare-large-loss shape holding up to half a conversation mean's variance. Together:
with no edge, a pairing is called with chance at most alpha / n_pairings in either case (Bonferroni over the
pairings, Pocock-type repeated looks), if those assumptions hold. Beyond the family (large, rare, one-sided
swings) no test of this size can keep that promise; docs/methodology.md gives the measured worst cases. `even` is an equivalence
verdict (TOST at +-EQUIV_BB100, both arms).

Planning. The deals a design needs come from the spread of a conversation's mean, which counts the
dependence inside it: a conversation effect (assumed_icc) and memory between deals (assumed_ar1). A design
reports its power under optimistic, assumed and pessimistic dependence, and by default plays as many rounds
as the assumed (moderate) case needs for 80% power.

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
DEFAULT_DESIGN = {'alpha': 0.05, 'round_deals': 50, 'block': BLOCK}   # max_rounds: from the power target
TARGET_EDGE = 50            # bb/100: the edge a plan is powered to detect unless it says otherwise
PRIOR_SD = 350              # bb/100: the per-deal spread assumed when planning (first model runs: about 340)
ASSUMED_ICC = 0.01          # planning: share of the per-deal variance that is a conversation effect (moderate)
ASSUMED_AR1 = 0.2           # planning: AR(1) memory between consecutive deals of a conversation (moderate)
POWER = 0.8
MAX_ROUNDS = 40
MIN_CONVERSATIONS = 8       # a verdict needs this many independent conversations of the pairing
FAMILY = ('normal', 'exponential', 'lognormal', 'rare_loss')   # null shapes the bars are calibrated for
FAMILY_SHARE = 0.5          # in the skewed shapes, the conversation effect's share of a conversation mean's variance
POWER_EDGES = (20, 50, 100)
BOOT = 300
CAL_SIMS = 100_000          # null sequences per shape behind each nominal level
POWER_SIMS = 40_000
FLIPS = 9_999               # random sign vectors of the sign-flip arm (plus the observed one)
SIM_SEED = 20261126
SESSION = 10                # deals per Elo session
TIE_BB = 0.1                # a session within this many big blinds per deal of even is a tie
ELO_PRIOR = 1.5             # prior sd of a session-Elo rating, in logits
K = 400 / math.log(10)
ALIASES = ('sonnet', 'opus', 'haiku')   # bare CLI aliases: the model version behind them is unknown
DECIDED = ('a', 'b', 'even')
TERMINAL = ('a', 'b', 'even', 'ineligible')   # verdicts that stop a pairing before its last look
ALLOW_UNATTESTED = ('codex',)   # backends that cannot report the model that answered, allowed unless a design says otherwise
ASSUMPTIONS = ('Its error bound is conditional: if the test\'s assumptions hold, a pairing with no real edge is '
               'called for either player with chance at most alpha / n_pairings (so at most alpha over all of '
               'them). It holds exactly when every conversation\'s result is symmetric about the true edge (for '
               'instance two interchangeable players), which no data can confirm, and, by simulation, when results '
               'are skewed but the conversation effect holds at most half the spread of a conversation mean '
               '(docs/methodology.md). Beyond that range the bound is not shown to hold; each verdict says whether '
               'its data look within it.')
GUARANTEE = ASSUMPTIONS
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
    return re.split(r'[#@]', identity.split('!served=', 1)[0], maxsplit=1)[0].split(':'), served


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


SERVED_ALIASES: dict[str, set] = {}   # canonical configured id -> canonical served ids documented as the same model
_SNAPSHOT = re.compile(r'^(?P<base>.+?)-(?:\d{8}|\d{4}-\d{2}-\d{2})$')


def canonical_model(m) -> str:
    """A model id in one spelling: lower case, no provider prefix (models/, google/, ...), and any run of
    spaces, dots, underscores, slashes or brackets as one hyphen, so "Gemini 4 Neon (High)" and
    "gemini-4-neon-high" are the same id. Nothing else is equated."""
    s = str(m).strip().lower()
    s = re.sub(r'^(?:models|google|anthropic|openai|vertex_ai|gemini)/', '', s)
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')


def same_model(configured, served) -> bool:
    """Whether the model that answered is the configured one: the same canonical id, a dated snapshot of it
    (`<id>-YYYYMMDD` or `<id>-YYYY-MM-DD`, the providers' documented way of pinning a version), or an id listed
    for it in SERVED_ALIASES. A shorter or longer id is a different model (claude-sonnet-5 is not
    claude-sonnet-5-5)."""
    a, b = canonical_model(configured), canonical_model(served)
    if a == b or b in SERVED_ALIASES.get(a, ()):
        return True
    snap = _SNAPSHOT.match(b)
    return bool(snap) and snap.group('base') == a


def served_mismatch(model, served) -> list[str]:
    """The models that answered for a player configured as `model` and are not that model."""
    if not model:
        return []
    return sorted({s for s in served or () if s and not same_model(model, s)})


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


def check_pair(ca: dict, cb: dict | None, ha: dict, hb: dict) -> tuple[list[int], list[dict]]:
    """Whether b-first is a-first with the seats swapped, hand by hand: (the hands both plays finished that pass,
    [{'hand' (None: the whole conversation), 'reason'}] for what was rejected). The conversation is rejected
    when b-first is missing its config, seats other players or the same order, plays other game settings, other
    seeds or other deal fingerprints, or gives a player other specs or another protocol. A hand is rejected
    when either play's recorded seed or deal fingerprint differs from the configs', or its chips don't add up."""
    if not cb:
        return [], [{'hand': None, 'reason': 'b-first has no readable config'}]
    pa, pb = list(ca.get('players') or []), list(cb.get('players') or [])
    bad = []
    if len(pa) != 2 or pb != pa[::-1]:
        bad.append(f'b-first seats {pb}, not {pa[::-1]}')
    from .history import game_key
    if game_key(ca.get('game') or {}) != game_key(cb.get('game') or {}):
        bad.append('the two plays have different game settings')
    if (ca.get('seeds') or []) != (cb.get('seeds') or []):
        bad.append('the two plays were dealt different seeds')
    fa, fb = ca.get('deal_fingerprints'), cb.get('deal_fingerprints')
    if fa is not None and fb is not None and fa != fb:
        bad.append('the two plays have different deal fingerprints')
    for p in pa:
        if (ca.get('specs') or {}).get(p) != (cb.get('specs') or {}).get(p):
            bad.append(f'{p} has different specs in the two plays')
        if protocol_of(ca, p) != protocol_of(cb, p):
            bad.append(f'{p} has a different protocol in the two plays')
    if bad:
        return [], [{'hand': None, 'reason': '; '.join(bad)}]
    seeds = ca.get('seeds') or []
    good, out = [], []
    for k in sorted(set(ha) & set(hb)):
        why = []
        for side, h in (('a-first', ha[k]), ('b-first', hb[k])):
            if h.get('seed') is not None and k < len(seeds) and h['seed'] != seeds[k]:
                why.append(f'{side} recorded seed {h["seed"]}, not {seeds[k]}')
            if h.get('fingerprint') is not None and fa is not None and k < len(fa) and h['fingerprint'] != fa[k]:
                why.append(f'{side} recorded another deal fingerprint')
            net = h.get('net')
            if not isinstance(net, list) or len(net) != 2 or sum(net) != 0:
                why.append(f'{side} chips do not add up ({net})')
        if why:
            out.append({'hand': k, 'reason': '; '.join(why)})
        else:
            good.append(k)
    return good, out


def load_deals(root, report: list | None = None) -> list[dict]:
    """[{'a', 'b', 'matchup' (the conversation), 'deal', 'net_a' (chips, both hands), 'bb', 'seed', 'fp',
    'planned', 'slate', 'round', 'replicate'}] for every duplicate deal under root both of whose plays finished
    and that passes `check_pair`; what it rejects is appended to `report` as {'matchup', 'hand', 'reason'}."""
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
        good, rejected = check_pair(ca, _json(b_dir / 'config.json'), ha, hb)
        if report is not None:
            report += [{'matchup': name, 'a': a, 'b': b, **x} for x in rejected]
        for k in good:
            seed = seeds[k] if k < len(seeds) else None
            deals.append({'a': a, 'b': b, 'matchup': name, 'deal': k, 'bb': bb, 'seed': seed,
                          'fp': fingerprint(game, seed, k % n), 'planned': len(seeds),
                          'slate': seeds[0] if seeds else name, 'round': rnd, 'replicate': rep,
                          'net_a': ha[k]['net'][0] + hb[k]['net'][1]})
    return deals


def pairing_data(root, folders, a) -> dict:
    """What `adjudicate` judges for one pairing, from its match folders (relative to root) in play order:
    {'xs' (a's bb/100 per validated duplicate deal), 'groups' (each deal's conversation), 'fps' (deal
    fingerprints), 'conversations', 'rejected' ([{'matchup', 'hand', 'reason'}] from `check_pair`)}."""
    root = Path(root)
    xs, groups, fps, rejected = [], [], [], []
    for f in folders:
        report = []
        for d in load_deals(root / f, report):
            sign = 1 if d['a'] == a else -1
            xs.append(sign * d['net_a'] / d['bb'] / 2 * 100)
            groups.append(f)
            fps.append(d['fp'])
        rejected += [{**x, 'matchup': f} for x in report]
    return {'a': a, 'xs': xs, 'groups': groups, 'fps': fps, 'conversations': list(folders), 'rejected': rejected}


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
    from .history import match_identity
    meta, protos, models = {}, defaultdict(set), {}
    for cfg_path in sorted(Path(root).rglob('config.json')):
        cfg = _json(cfg_path) or {}
        for pid, spec in (cfg.get('specs') or {}).items():
            proto = protocol_of(cfg, pid)
            protos[pid].add(proto)
            models[pid] = spec.get('model')
            if pid not in meta:
                ident = match_identity(cfg, pid)
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


# The judge and its registration.

_SOURCE = Path(__file__).read_bytes()


def judge_fingerprint() -> str:
    """This judge's implementation: the hash of ratings.py as loaded (the hash the runner records for it)."""
    return hashlib.sha256(_SOURCE).hexdigest()[:16]


def calibration_id(design: dict | None) -> str | None:
    """Names the calibration a design's bars come from: its calibrated levels and every constant of the test
    that sets its operating characteristics."""
    if not design or not design.get('calibration'):
        return None
    key = {'calibration': design['calibration'], 'nominal_alpha': design.get('nominal_alpha'), 'flips': FLIPS,
           'seed': SIM_SEED, 'min_conversations': MIN_CONVERSATIONS, 'equiv_bb100': EQUIV_BB100}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


DESIGN_INPUTS = ('alpha', 'round_deals', 'block', 'replicates', 'max_rounds', 'target_edge_bb100', 'prior_sd_bb100',
                 'assumed_icc', 'assumed_ar1', 'budget_deals', 'n_pairings', 'allow_unattested', 'allow_opaque',
                 'outside_calibration')


def _inputs(d: dict) -> dict:
    """What a design was chosen to be (not what this code computes from it), for comparing with a registration."""
    d = d or {}
    defaults = {'block': BLOCK, 'replicates': 1, 'target_edge_bb100': TARGET_EDGE, 'prior_sd_bb100': PRIOR_SD,
                'assumed_icc': ASSUMED_ICC, 'assumed_ar1': ASSUMED_AR1, 'budget_deals': None,
                'allow_unattested': list(ALLOW_UNATTESTED), 'allow_opaque': [], 'outside_calibration': 'exclude'}
    out = {}
    for k in DESIGN_INPUTS:
        v = d.get(k, defaults.get(k))
        out[k] = sorted(v) if isinstance(v, (list, tuple, set)) else float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    return out


def registration(design: dict, created: float | None = None) -> dict:
    """The preregistration of a design, to write into plan.json before the first hand: the design, the judge
    that will decide the run and the calibration of its bars."""
    import time
    return {'status': 'registered', 'design': design, 'judge_fingerprint': judge_fingerprint(),
            'calibration_id': calibration_id(design), 'created': round(created if created is not None else time.time())}


def registration_problems(reg: dict | None, design: dict | None, first_started=None, implementation=None,
                          changes=()) -> list[str]:
    """Why a registration doesn't preregister this design (empty: it does). `first_started` is when the run's
    first match began, `implementation` the one it began with ({'files': {path: hash}}), `changes` the code
    changes accepted during it (allow_code_change)."""
    if not reg:
        return ['no registration was written before play']
    if reg.get('status') == 'void':
        return [f'the registration is void: {reg.get("void_reason") or "the code changed after it"}']
    if reg.get('status') != 'registered':
        return [f'the registration has status {reg.get("status")!r}']
    out = []
    if not reg.get('judge_fingerprint') or reg.get('created') is None:
        out.append('the registration names no judge or no time')
    if design is not None:
        diff = [k for k, v in _inputs(design).items() if _inputs(reg.get('design') or {}).get(k) != v]
        if diff:
            out.append(f'the design differs from the registered one ({", ".join(diff)})')
    if first_started is not None and reg.get('created') is not None and float(reg['created']) > float(first_started):
        out.append('it was written after play began')
    judge = ((implementation or {}).get('files') or {}).get(f'{Path(__file__).parent.name}/ratings.py')
    if judge and reg.get('judge_fingerprint') and judge != reg['judge_fingerprint']:
        out.append('the registered judge is not the one the run started with')
    if changes:
        files = sorted({f for c in changes for f in (c.get('files') or ['(unrecorded files)'])})
        out.append(f'the code changed during the run (allow_code_change: {", ".join(files)})')
    return out


def run_evidence(root) -> dict:
    """What the match folders under root say about when play began and which code changes were accepted."""
    first, changes = None, []
    for cfg_path in Path(root).rglob('config.json'):
        if cfg_path.parent.name not in ('a-first', 'b-first'):
            continue
        cfg = _json(cfg_path) or {}
        if cfg.get('started') is not None:
            first = cfg['started'] if first is None else min(first, cfg['started'])
        changes += cfg.get('implementation_changes') or []
    return {'first_started': first, 'implementation_changes': changes}


def registration_status(plan: dict | None, evidence: dict | None = None) -> dict:
    """Whether a run is preregistered: {'registered', 'status' ('valid' | 'missing' | 'declined' | 'legacy' |
    'void' | 'invalid'), 'reasons', 'judge_fingerprint', 'calibration_id', 'created', 'design'}. Never recomputed
    from the design: only a registration written into plan.json before the first hand counts, an explicit
    `registered: false` stays false, and any code change accepted during the run voids it."""
    plan = plan or {}
    reg = plan.get('registration')
    out = {'registered': False, 'status': 'missing', 'reasons': [], 'judge_fingerprint': (reg or {}).get('judge_fingerprint'),
           'calibration_id': (reg or {}).get('calibration_id'), 'created': (reg or {}).get('created'),
           'design': (reg or {}).get('design')}
    design = normalize_design(plan) if plan else None
    if plan.get('registered') is False or (plan.get('design') or {}).get('registered') is False:
        return {**out, 'status': 'declined', 'reasons': ['the plan says registered: false']}
    if design and design.get('legacy'):
        return {**out, 'status': 'legacy', 'reasons': ['played under an older stopping rule, not a registered design']}
    if not design:
        return {**out, 'reasons': ['the run has no sequential design']}
    ev = evidence or {}
    problems = registration_problems(reg, design, ev.get('first_started'), plan.get('implementation'),
                                     [*(plan.get('implementation_changes') or []), *(ev.get('implementation_changes') or [])])
    if not problems:
        return {**out, 'registered': True, 'status': 'valid'}
    status = 'missing' if not reg else 'void' if reg.get('status') == 'void' or any('code changed' in p for p in problems) else 'invalid'
    return {**out, 'status': status, 'reasons': problems}


# The preregistered design.

_COMPUTED = ('replicates', 'n_pairings', 'alpha_pairing', 'nominal_alpha', 'deals_needed', 'deals_planned',
             'max_deals', 'power_at_max', 'min_conversations', 'conversations_needed', 'power_scenarios',
             'underpowered', 'registered', 'sd_conversation_bb100', 'calibration')
_PLAN_KEYS = ('design', 'deals', 'until_decided', 'replicates', 'matchups', 'target_edge_bb100', 'prior_sd_bb100',
              'registered')


def _n_pairings(plan: dict) -> int:
    return max(1, len({frozenset(m) for m in plan.get('matchups') or []}))


def normalize_design(plan: dict | None) -> dict | None:
    """The plan's sequential design with every default filled in: alpha, round_deals, block, replicates
    (independent conversations per pairing per round), n_pairings and alpha_pairing (Bonferroni), nominal_alpha
    (the level each look is tested at), min_conversations, and the preregistered target: target_edge_bb100
    (default 50), prior_sd_bb100 (the per-deal spread, default 350) and the dependence assumed when planning
    (assumed_icc: the share of the per-deal variance that is a conversation effect; assumed_ar1: memory from one
    deal to the next inside a conversation). max_rounds, unless the plan sets it, is the fewest rounds that give
    80% power at the target under those assumptions (max_rounds_basis 'power'); `budget_deals` caps the deals per
    pairing (basis 'budget'), and then `underpowered` says whether that leaves less than 80%. power_scenarios
    gives the power at the target under optimistic, the assumed (moderate) and pessimistic dependence;
    power_at_max is the moderate one. The eligibility rules are part of the design: allow_unattested (backends
    whose model identity may go unattested, default codex), allow_opaque (backends whose turns may go unaudited,
    default none) and outside_calibration ('exclude', the default: a decided verdict outside the calibrated range
    never counts towards a decided leader; 'count'). `registered` is not computed here: it is False for a plan
    that declines registration or played an older stopping rule (`legacy`), and absent otherwise; only
    `registration_status` says whether a run is preregistered. The runner, the ratings and the viewer all use
    this one. None for a plan with no sequential design."""
    if not plan or not plan.get('deals') and not (plan.get('design') or {}).get('round_deals'):
        return None
    if plan.get('design') is None and not plan.get('until_decided'):
        return None
    key = json.dumps({k: plan.get(k) for k in _PLAN_KEYS if k != 'matchups'} | {'n': _n_pairings(plan)},
                     sort_keys=True, default=str)
    return json.loads(_normalized(key))


@lru_cache(maxsize=256)
def _normalized(key: str) -> str:
    plan = json.loads(key)
    given = {k: v for k, v in (plan.get('design') or {}).items() if k not in _COMPUTED}
    if plan.get('registered') is False or (plan.get('design') or {}).get('registered') is False:
        given['registered'] = False
    if plan.get('design') is not None:
        d = {**DEFAULT_DESIGN, 'round_deals': plan.get('deals') or DEFAULT_DESIGN['round_deals'], **given}
    else:
        d = {**DEFAULT_DESIGN, 'round_deals': plan['deals'],
             'max_rounds': max(1, plan['until_decided'] // plan['deals']), 'legacy': 'until_decided'}
    reps = int(plan.get('replicates') or (plan.get('design') or {}).get('replicates') or 1)
    return json.dumps(_complete(d, reps, plan['n'], plan.get('target_edge_bb100') or d.get('target_edge_bb100'),
                                plan.get('prior_sd_bb100') or d.get('prior_sd_bb100')))


def design_effect(ar1: float, R: int) -> float:
    """Variance of the mean of R deals with AR(1) memory ar1, relative to R independent deals."""
    return 1 + 2 * sum((1 - k / R) * ar1 ** k for k in range(1, R))


def sd_conversation(sd: float, R: int, icc: float, ar1: float) -> float:
    """The spread (bb/100) of one conversation's mean edge: a conversation effect holding a share icc of the
    per-deal variance sd^2, plus the card noise of R deals with AR(1) memory."""
    return math.sqrt(icc * sd * sd + (1 - icc) * sd * sd * design_effect(ar1, R) / R)


def scenarios(icc: float, ar1: float) -> list[tuple[str, float, float]]:
    """The planning scenarios: no dependence, the assumed one, and double the ICC with 0.1 more memory."""
    return [('optimistic', 0.0, 0.0), ('moderate', icc, ar1), ('pessimistic', min(0.5, 2 * icc), round(min(0.9, ar1 + 0.1), 3))]


def rounds_for(shift: float, alpha_pairing: float, replicates: int, power: float = POWER) -> int | None:
    """The fewest rounds (looks) whose simulated power reaches `power` when each conversation's mean edge is
    `shift` of its own spread; None beyond MAX_ROUNDS."""
    reps = int(replicates)
    z = _Z.inv_cdf(1 - alpha_pairing / 8) + _Z.inv_cdf(power)
    r = max(1, math.ceil(max(MIN_CONVERSATIONS, (z / max(shift, 1e-9)) ** 2) / reps))
    if r > MAX_ROUNDS:
        return None if power_of(shift, alpha_pairing, MAX_ROUNDS, reps) < power else MAX_ROUNDS
    if power_of(shift, alpha_pairing, r, reps) >= power:
        while r > 1 and power_of(shift, alpha_pairing, r - 1, reps) >= power:
            r -= 1
        return r
    while r < MAX_ROUNDS:
        r += 1
        if power_of(shift, alpha_pairing, r, reps) >= power:
            return r
    return None


def _complete(d, reps, n_pairings, target, prior) -> dict:
    R, B = int(d['round_deals']), int(d.get('block') or BLOCK)
    target, prior = float(target or TARGET_EDGE), float(prior or PRIOR_SD)
    icc, ar1 = float(d.get('assumed_icc', ASSUMED_ICC)), float(d.get('assumed_ar1', ASSUMED_AR1))
    ap = d['alpha'] / n_pairings
    plan_sd = sd_conversation(prior, R, icc, ar1)
    need = rounds_for(target / plan_sd, ap, reps)
    if d.get('max_rounds'):
        rounds, basis = int(d['max_rounds']), d.get('max_rounds_basis') or 'given'
    else:
        rounds, basis = need or MAX_ROUNDS, 'power'
    if d.get('budget_deals'):
        cap = max(1, int(d['budget_deals']) // (R * reps))
        if rounds > cap:
            rounds, basis = cap, 'budget'
    a1, _, levels, _ = _calibrate(ap, rounds, reps)
    declined = d.get('registered') is False or d.get('legacy')
    power = [{'scenario': name, 'assumed_icc': i, 'assumed_ar1': a,
              'sd_conversation_bb100': round(sd_conversation(prior, R, i, a), 1),
              'power': power_of(target / sd_conversation(prior, R, i, a), ap, rounds, reps)}
             for name, i, a in scenarios(icc, ar1)]
    moderate = power[1]['power']
    cap = R * reps * rounds
    return {**d, 'round_deals': R, 'max_rounds': rounds, 'max_rounds_basis': basis, 'block': B, 'replicates': reps,
            'n_pairings': n_pairings, 'alpha_pairing': ap, 'nominal_alpha': a1,
            'min_conversations': first_verdict(ap, rounds, reps) or MIN_CONVERSATIONS,
            **({'registered': False} if declined else {}), 'target_edge_bb100': target, 'prior_sd_bb100': prior,
            'allow_unattested': sorted(d.get('allow_unattested', ALLOW_UNATTESTED)),
            'allow_opaque': sorted(d.get('allow_opaque', ())),
            'outside_calibration': d.get('outside_calibration', 'exclude'),
            'assumed_icc': icc, 'assumed_ar1': ar1, 'sd_conversation_bb100': round(plan_sd, 1),
            'conversations_needed': need * reps if need else None, 'deals_needed': need * reps * R if need else None,
            'deals_planned': cap, 'max_deals': cap, 'power_at_max': moderate, 'power_scenarios': power,
            'underpowered': moderate < POWER,
            'calibration': {'family': list(FAMILY), 'conversation_effect_share': FAMILY_SHARE,
                            'levels': {k: round(v, 6) for k, v in levels.items()}, 'sims': CAL_SIMS}}


def unregistered(round_deals, n_pairings, replicates=1) -> dict:
    """The design of a run that had none fixed before play: nothing in it is a test. `retrospective` is what a
    default design would have needed for 80% power at the target (moderate dependence), labelled as such."""
    R = int(round_deals or DEFAULT_DESIGN['round_deals'])
    ref = normalize_design({'deals': R, 'design': {}, 'replicates': 4, 'matchups': [[i, -i - 1] for i in range(n_pairings)]})
    return {'registered': False, 'alpha': DEFAULT_DESIGN['alpha'], 'round_deals': R, 'replicates': int(replicates or 1),
            'block': BLOCK, 'n_pairings': n_pairings, 'min_conversations': MIN_CONVERSATIONS,
            'retrospective': {k: ref[k] for k in ('target_edge_bb100', 'prior_sd_bb100', 'assumed_icc', 'assumed_ar1',
                                                  'replicates', 'conversations_needed', 'deals_needed')}}


# The stopping rule.

def _look_cs(rounds, reps):
    return [k * reps for k in range(1, rounds + 1)]


def _p_two_sided(t, df: int):
    """2 P(T > |t|) for Student t with integer df, vectorized (Abramowitz and Stegun 26.7.3-4)."""
    t = np.abs(np.asarray(t, float))
    th = np.arctan(t / math.sqrt(df))
    c2, s = np.cos(th) ** 2, np.sin(th)
    if df == 1:
        return 1 - 2 * th / math.pi
    if df % 2 == 0:
        term = np.ones_like(t)
        acc = term.copy()
        for j in range(1, (df - 2) // 2 + 1):
            term = term * c2 * (2 * j - 1) / (2 * j)
            acc += term
        return np.clip(1 - s * acc, 0.0, 1.0)
    term = np.cos(th)
    acc = term.copy()
    for j in range(1, (df - 3) // 2 + 1):
        term = term * c2 * (2 * j) / (2 * j + 1)
        acc += term
    return np.clip(1 - 2 / math.pi * (th + s * acc), 0.0, 1.0)


def _null(shape, rng, n, C):
    """n x C conversation means with no edge, unit variance: card noise (normal) plus, in the skewed shapes, a
    conversation effect holding FAMILY_SHARE of the variance."""
    z = rng.standard_normal((n, C))
    if shape == 'normal':
        return z
    if shape == 'exponential':
        u = rng.exponential(1.0, (n, C)) - 1
    elif shape == 'lognormal':
        u = (np.exp(rng.standard_normal((n, C))) - math.exp(0.5)) / math.sqrt((math.e - 1) * math.e)
    else:   # rare_loss: a small win nine times in ten, a loss nine times as large otherwise
        u = np.where(rng.random((n, C)) < 0.9, 1 / 3, -3.0)
    return math.sqrt(1 - FAMILY_SHARE) * z + math.sqrt(FAMILY_SHARE) * u


_T_CACHE: dict = {}


def _tcols(shape, reps, shift, sims, stream, looks):
    """The conversation-level t at looks 1..`looks` of `sims` simulated sequences of conversation means
    (`shape` plus `shift`, `reps` conversations a look), one float32 column per look, 0 with fewer than two
    conversations or no spread. Each look's conversations come from their own seed, so the columns of a short
    design are those of a longer one, and are kept for reuse."""
    key = (shape, int(reps), round(float(shift), 6), int(sims), int(stream))
    c = _T_CACHE.get(key)
    if c is None:
        c = _T_CACHE[key] = {'s1': np.zeros(sims), 's2': np.zeros(sims), 'cols': []}
    while len(c['cols']) < looks:
        j = len(c['cols'])
        rng = np.random.default_rng([SIM_SEED, int(stream), int(reps), j, int(round(shift * 1e6))])
        z = _null(shape, rng, sims, reps) + shift
        c['s1'] += z.sum(1)
        c['s2'] += (z * z).sum(1)
        C = (j + 1) * reps
        if C < 2:
            c['cols'].append(np.zeros(sims, np.float32))
            continue
        m = c['s1'] / C
        var = (c['s2'] - C * m * m) / (C - 1)
        ok = var > 1e-12 * (1 + m * m)
        c['cols'].append(np.where(ok, m / np.sqrt(np.where(ok, var, 1.0) / C), 0.0).astype(np.float32))
    return c['cols'][:looks]


def _crossings(cols, reps, eligible, a1):
    """(sequences x looks) whether |t| reached the bar of nominal level a1 at each eligible look."""
    out = np.zeros((len(cols[0]), len(cols)), bool)
    for j, t in enumerate(cols):
        if eligible[j]:
            out[:, j] = np.abs(t) >= t_ppf(1 - a1 / 2, (j + 1) * reps - 1)
    return out


def _levels(ap, rounds, reps, eligible):
    k = max(0, int(math.floor((ap - 2 * math.sqrt(ap * (1 - ap) / CAL_SIMS)) * CAL_SIMS)))
    levels = {}
    for i, shape in enumerate(FAMILY):
        cols = _tcols(shape, reps, 0.0, CAL_SIMS, i, rounds)
        lowest = np.ones(CAL_SIMS)
        for j, t in enumerate(cols):
            if not eligible[j]:
                continue
            df = (j + 1) * reps - 1
            near = np.abs(t) > t_ppf(1 - min(0.25, 10 * ap), df)
            lowest[near] = np.minimum(lowest[near], _p_two_sided(t[near], df))
        levels[shape] = min(ap, float(np.partition(lowest, k)[k]) * (1 - 1e-9))
    a1 = min(levels.values())
    hit = np.logical_or.accumulate(_crossings(_tcols('normal', reps, 0.0, CAL_SIMS, 0, rounds), reps, eligible, a1), 1)
    return a1, tuple(float(x) for x in hit.mean(0)), levels


@lru_cache(maxsize=None)
def _calibrate(alpha_pairing: float, rounds: int, reps: int):
    """(nominal level, the normal-shape cumulative chance of a crossing by each look, {shape: level}, which
    looks can decide). Each shape's level is the largest at which at most alpha_pairing (less two Monte Carlo
    standard errors) of its null sequences cross at any look; the nominal level is the smallest of them (a
    Pocock design on the p-value scale, calibrated for every shape of the family). A look can decide when it
    has MIN_CONVERSATIONS conversations and the sign-flip arm can reject there at all (its error spent by then
    is at least 1.5 x 2^(1 - conversations), the chance of the most extreme signs); a look that cannot is left
    out of the calibration, so its error is not wasted."""
    ap, rounds, reps = float(alpha_pairing), int(rounds), int(reps)
    cs = _look_cs(rounds, reps)
    eligible = tuple(C >= MIN_CONVERSATIONS for C in cs)
    a1, cum, levels = ap, (0.0,) * rounds, {s: ap for s in FAMILY}
    for _ in range(rounds + 1):
        if not any(eligible):
            return ap, (0.0,) * rounds, {s: ap for s in FAMILY}, eligible
        a1, cum, levels = _levels(ap, rounds, reps, eligible)
        spend = [c * ap / cum[-1] if cum[-1] > 0 else 0.0 for c in cum]
        feasible = tuple(e and spend[j] >= 1.5 * 2.0 ** (1 - cs[j]) for j, e in enumerate(eligible))
        if feasible == eligible:
            break
        eligible = feasible
    return a1, cum, levels, eligible


def first_verdict(alpha_pairing: float, rounds: int, reps: int) -> int | None:
    """The fewest conversations at which a look of this design can decide."""
    eligible = _calibrate(float(alpha_pairing), int(rounds), int(reps))[3]
    return next((C for C, e in zip(_look_cs(int(rounds), int(reps)), eligible) if e), None)


def nominal_alpha(alpha_pairing: float, round_deals: int, max_rounds: int, block: int = BLOCK, replicates: int = 1) -> float:
    """The two-sided level every look is tested at so that, over all `max_rounds` looks, a pairing with no edge
    is called for either player with chance at most alpha_pairing for every null shape of the calibration family
    (normal, and skewed conversation effects: exponential, lognormal, rare large losses), found by simulation.
    It depends only on the looks and the conversations per round (round_deals and block are kept for the
    signature)."""
    return _calibrate(float(alpha_pairing), int(max_rounds), int(replicates))[0]


@lru_cache(maxsize=None)
def power_of(shift: float, alpha_pairing: float, max_rounds: int, replicates: int = 1) -> float:
    """The chance the t arm of the procedure calls the better player when each conversation's mean edge is
    `shift` of the spread of a conversation mean, by simulation (normal conversation means). The sign-flip arm
    can only lower it, by under 2 points in the heavy tests."""
    rounds, reps = int(max_rounds), int(replicates)
    a1, _, _, eligible = _calibrate(float(alpha_pairing), rounds, reps)
    cols = _tcols('normal', reps, float(shift), POWER_SIMS, 99, rounds)
    hit = _crossings(cols, reps, eligible, a1)
    first = hit.argmax(1)
    T = np.stack(cols, 1)
    return round(float((hit.any(1) & (T[np.arange(len(T)), first] > 0)).mean()), 3)


def boundary(alpha_pairing: float, round_deals: int, max_rounds: int, block: int = BLOCK, look: int | None = None,
             replicates: int = 1) -> float:
    """Critical |t| at look `look` (default: the last) for the conversation-level t, with look x replicates
    conversations: the Student t quantile at the calibrated nominal level. inf where a look has fewer than
    MIN_CONVERSATIONS conversations (no verdict is possible there)."""
    rounds, reps = int(max_rounds), int(replicates)
    look = rounds if look is None else max(1, min(int(look), rounds))
    C = look * reps
    if not _calibrate(float(alpha_pairing), rounds, reps)[3][look - 1]:
        return math.inf
    return t_ppf(1 - nominal_alpha(alpha_pairing, round_deals, rounds, block, reps) / 2, C - 1)


_FLIP_COLS: list = []


def _flips(G: int) -> np.ndarray:
    """(FLIPS + 1) x G signs: row 0 all +1 (the data as observed), the rest independent fair signs, one
    seeded column per conversation, so the matrix of the first G conversations never depends on later ones."""
    while len(_FLIP_COLS) < G:
        c = len(_FLIP_COLS)
        col = np.random.default_rng([SIM_SEED, 7, c]).integers(0, 2, FLIPS) * 2 - 1
        _FLIP_COLS.append(np.concatenate([[1], col]).astype(float))
    return np.stack(_FLIP_COLS[:G], 1)


def _signflip(sizes, sums, cs, spend, null=0.0, side=0) -> list[bool]:
    """The sign-flip arm, look by look: whether the observed conversations have been rejected at or before each
    look (`cs` conversations at each, `spend` the cumulative chance allowed by then) as having a mean edge other
    than `null` (side 0), above it (side 1) or below it (side -1). Under the null that each conversation's total
    is symmetric about `null` x its deals (independent conversations, any shape, any dependence inside one),
    the observed signs are exchangeable with the random ones, and at every look the threshold is the lowest
    that keeps the rows rejected so far, observed included, within spend x rows: so the chance of a false
    rejection by the last look is at most its spend, exactly."""
    E = _flips(len(sums))
    rows = len(E)
    rejected = np.zeros(rows, bool)
    path = []
    dev = sums - null * sizes
    for c, f in zip(cs, spend):
        if c >= MIN_CONVERSATIONS and f > 0:
            z = E[:, :c] * dev[:c]
            n = sizes[:c]
            N = n.sum()
            m = z.sum(1) / N
            r = z - np.outer(m, n)
            se2 = c / (c - 1) * (r * r).sum(1) / N ** 2
            pos = se2 > 1e-12 * (1 + (m * N) ** 2)
            t = np.where(pos, (np.abs(m) if side == 0 else side * m) / np.sqrt(np.where(pos, se2, 1.0)), 0.0)
            budget = int(math.floor(f * rows + 1e-9)) - int(rejected.sum())
            free = ~rejected & (t > 0)
            vals = t[free]
            if budget >= 1 and len(vals):
                thr = vals.min() if budget >= len(vals) else \
                    np.nextafter(np.partition(vals, len(vals) - budget - 1)[len(vals) - budget - 1], np.inf)
                rejected |= free & (t >= thr)
        path.append(bool(rejected[0]))
    return path


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
    design. `groups` labels each deal's conversation (its match folder), conversations in play order (round by
    round); without it all deals are one.

    A look can decide only with at least `min_conversations` conversations (MIN_CONVERSATIONS or more: the
    first look of the design at which the sign-flip arm can reject at all) and some spread between them. Then a
    winner needs both arms at that look: the conversation-level t beyond the calibrated bar, and the sign-flip
    test (exact when results are symmetric about the true edge) rejecting by then. verdict: 'a' or 'b' (the
    first look where both did, `decided_at_look`), 'even' (the repeated (1 - 2 alpha) interval inside
    +-EQUIV_BB100, and both one-sided sign-flip tests), 'exhausted' (the last planned look, undecided), 'open'
    (undecided, looks remain, or no look yet), 'early' (a completed look with fewer conversations than a verdict
    needs, or too few deals for any spread). The
    numbers describe all the deals given; with one conversation `basis` is 'blocks' and the interval is
    conditional on it."""
    d = {**DEFAULT_DESIGN, 'max_rounds': 1, **(design or {})}
    xs = [float(x) for x in xs]
    n, B, rounds, reps = len(xs), int(d.get('block') or BLOCK), int(d['max_rounds']), int(d.get('replicates') or 1)
    R = int(d.get('round_deals') or n or 1)
    alpha_p = d['alpha'] / max(1, n_pairings)
    if looks_used is None:
        looks_used = min(rounds, n // (R * reps))
    labels = list(groups) if groups is not None else ['.'] * n
    conv = {}
    for g, x in zip(labels, xs):
        c = conv.setdefault(g, [0, 0.0, 0.0])
        c[0] += 1
        c[1] += x
        c[2] += x * x
    sizes = np.array([c[0] for c in conv.values()], float)
    sums = np.array([c[1] for c in conv.values()], float)
    bsizes, bsums = _chunks(xs, labels, B)
    C, G = len(conv), len(bsizes)
    out = {'deals': n, 'blocks': G, 'conversations': C, 'edge': round(sum(xs) / n, 1) if n else None, 'se': None,
           'ci95': None, 't': None, 'df': None, 'basis': None, 'boundary': None, 'verdict': 'early',
           'looks_used': looks_used, 'looks_planned': rounds, 'alpha_pairing': alpha_p, 'equiv_bb100': EQUIV_BB100,
           'min_conversations': MIN_CONVERSATIONS, 'decided_at_look': None, 'signflip': None,
           'conversation_share': None, 'ci_decision': None, 'equiv_ci': None, 'sd_per_deal': None, 'reason': None,
           'power': {'deals_for_80pct': {str(e): None for e in POWER_EDGES}}}
    if C >= 2:
        m, se = _cr1(sizes, sums)
        df, basis = C - 1, 'conversations'
    elif G >= 2:
        m, se = _cr1(bsizes, bsums)
        df, basis = G - 1, 'blocks'
    else:
        out['reason'] = 'too few deals to estimate any spread'
        return out
    iv = lambda c: [round(m - c * se, 1), round(m + c * se, 1)]
    a1, cum, _, eligible = _calibrate(alpha_p, rounds, reps)
    aeq = _calibrate(min(1.0, 2 * alpha_p), rounds, reps)[0]
    need = next((c for c, e in zip(_look_cs(rounds, reps), eligible) if e), None) or MIN_CONVERSATIONS
    out['min_conversations'] = need
    can = C >= need
    bnd = t_ppf(1 - a1 / 2, df) if can else None
    beq = t_ppf(1 - aeq / 2, df) if can else None
    flat = lambda mean, s: s <= 1e-9 * (1 + abs(mean))
    t = 0.0 if flat(m, se) else m / se
    verdict, reason, at, flip = None, None, None, None
    if looks_used < 1:
        verdict = 'open'
    elif not can:
        verdict = 'early'
        reason = (f'{C} conversation{"s" if C != 1 else ""}: a verdict needs at least {need} independent '
                  f'conversations (fresh match pairs) of the pairing')
    else:
        cs = [min(j * reps, C) for j in range(1, looks_used + 1)]
        cs[-1] = C
        spend = [cum[min(j, len(cum) - 1)] * alpha_p / cum[-1] if cum[-1] > 0 else 0.0 for j in range(len(cs))]
        flips = _signflip(sizes, sums, cs, spend)
        flip = flips[-1]
        eq = None
        for j, c in enumerate(cs):
            if c < MIN_CONVERSATIONS or not eligible[min(j, len(eligible) - 1)]:
                continue
            mj, sej = _cr1(sizes[:c], sums[:c])
            if flat(mj, sej):
                continue
            tj = mj / sej
            if abs(tj) >= t_ppf(1 - a1 / 2, c - 1) and flips[j]:
                verdict, at = ('a' if tj > 0 else 'b'), j + 1
                break
            q = t_ppf(1 - aeq / 2, c - 1)
            if mj - q * sej > -EQUIV_BB100 and mj + q * sej < EQUIV_BB100:
                if eq is None:
                    eq = [_signflip(sizes, sums, cs, spend, EQUIV_BB100, -1), _signflip(sizes, sums, cs, spend, -EQUIV_BB100, 1)]
                if eq[0][j] and eq[1][j]:
                    verdict, at = 'even', j + 1
                    break
        if verdict is None:
            verdict = 'exhausted' if looks_used >= rounds else 'open'
            if flat(m, se):
                reason = 'the conversations show no spread at all, so no verdict is possible from them'
            elif abs(t) >= bnd and not flip:
                reason = 'the t arm crossed its bar but the sign-flip arm did not (both are needed)'
    share = None
    if C >= 2:
        within = sum((c[2] - c[1] ** 2 / c[0]) / (c[0] - 1) / c[0] for c in conv.values() if c[0] > 1) / C
        between = float(np.var(sums / sizes, ddof=1))
        share = round(max(0.0, 1 - within / between), 2) if between > 0 else None
    sigma = se * math.sqrt(n)
    plan_c = _Z.inv_cdf(1 - a1 / 2)
    out.update({'edge': round(m, 1), 'se': round(se, 1), 'ci95': iv(t_ppf(0.975, df)), 'df': df, 'basis': basis,
                't': round(t, 2), 'boundary': round(bnd, 3) if can else None, 'verdict': verdict,
                'decided_at_look': at, 'signflip': flip, 'conversation_share': share,
                'ci_decision': iv(bnd) if can else None, 'equiv_ci': iv(beq) if can else None,
                'sd_per_deal': round(sigma, 1), 'reason': reason,
                'power': {'deals_for_80pct': {str(e): math.ceil(((plan_c + _Z.inv_cdf(POWER)) * sigma / e) ** 2)
                                              for e in POWER_EDGES}}})
    return out


# The adjudicator: the only function whose verdicts stop, record or decide anything.

def assumptions(v: dict, fps=None, groups=None) -> dict:
    """What a verdict's error bound rests on, checked against the data: {'status': 'supported' | 'unsupported' |
    'unknown', 'checks': {name: {'status', 'value', 'limit', 'meaning'}}}. The bound holds exactly when
    conversation results are symmetric about the true edge (which no data can confirm) and, by calibration, for
    skewed results whose conversation effect holds at most FAMILY_SHARE of a conversation mean's variance:
    `conversation_share` estimates that share. `independent_deals`: no deal is repeated across the pairing's
    conversations. `conversations`: enough of them to judge at all."""
    checks = {}
    C, need = v.get('conversations') or 0, v.get('min_conversations') or MIN_CONVERSATIONS
    checks['conversations'] = {'status': 'supported' if C >= need else 'unknown', 'value': C, 'limit': need,
                               'meaning': 'independent conversations, at least the number a verdict needs'}
    share = v.get('conversation_share')
    checks['conversation_share'] = {
        'status': 'unknown' if share is None else 'supported' if share <= FAMILY_SHARE else 'unsupported',
        'value': share, 'limit': FAMILY_SHARE,
        'meaning': 'estimated share of the spread of conversation means that is a conversation effect; the test '
                   'is calibrated up to the limit (beyond it the bound holds only if results are symmetric)'}
    if fps is not None and groups is not None:
        seen, repeats = {}, 0
        for fp, g in zip(fps, groups):
            if fp is None:
                continue
            if fp in seen and seen[fp] != g:
                repeats += 1
            seen.setdefault(fp, g)
        checks['independent_deals'] = {'status': 'supported' if not repeats else 'unsupported', 'value': repeats,
                                       'limit': 0, 'meaning': 'deals repeated across conversations of the pairing'}
    states = {c['status'] for c in checks.values()}
    status = 'unsupported' if 'unsupported' in states else 'unknown' if 'unknown' in states else 'supported'
    return {'status': status, 'checks': checks}


def _player_problems(p: str, x: dict | None, design: dict) -> list[str]:
    if not x:
        return [f'{p}: no eligibility facts']
    if x.get('kind') == 'bot':
        return []
    out, backend = [], x.get('backend')
    model = x.get('requested_model')
    if not model:
        out.append(f'{p}: no model recorded, so which model played is unknown')
    elif str(model) in ALIASES:
        out.append(f'{p}: bare alias "{model}", so the model version behind it is unknown')
    ident = x.get('identity')
    if ident == 'mismatch':
        out.append(f'{p}: answered by {", ".join(x.get("observed_models") or ["another model"])}, not the requested {model}')
    elif ident in (None, 'none'):
        out.append(f'{p}: no replies recorded')
    elif ident == 'unattested' and backend not in (design.get('allow_unattested') or ()):
        out.append(f'{p}: its backend ({backend}) did not attest which model answered, and the design does not '
                   f'allow unattested {backend}')
    audit = x.get('audit')
    if audit == 'opaque' and backend not in (design.get('allow_opaque') or ()):
        out.append(f'{p}: its turns could not be audited for tool use (opaque {backend} backend), and the design '
                   f'does not allow that')
    elif audit not in ('complete', 'opaque'):
        out.append(f'{p}: some of its decisions have no complete audit of tool use')
    if x.get('protocol_consistent') is False:
        out.append(f'{p}: its settings (protocol) differ between matches of the pairing')
    return out


def adjudicate(pairing_data: dict, design: dict, n_pairings: int, look: int, eligibility: dict | None) -> dict:
    """The verdict on one pairing at look `look`, with every eligibility rule applied: what the runner stops and
    records on, and what the ratings, the CLI and the viewer report as official when the runner recorded it.

    pairing_data: {'a', 'b', 'xs' (a's bb/100 per duplicate deal), 'groups' (each deal's conversation),
    'fps' (deal fingerprints, optional), 'rejected' (check_pair's rejections)}; or {'root', 'conversations', 'a',
    'b'} to load and validate them with `pairing_data`. eligibility: the runner's facts ({'registration',
    'implementation_changes', 'first_started' (optional), 'players': {pid: {'kind', 'backend', 'requested_model',
    'identity', 'audit', 'protocol_consistent', 'observed_models', ...}}}).

    Returns `decide`'s keys plus: eligible, ineligible_reasons, assumptions ({status, checks}), calibrated
    (True / False / None: assumptions supported / unsupported / unknown), counts_toward_leader, standing,
    test_verdict (decide's verdict), identity_status and audit_status per player, judge_fingerprint (= judge),
    calibration_id, registered_judge and official (made by the registered judge under a valid registration).
    verdict is 'ineligible', a terminal verdict, whenever any rule fails: no valid registration, code changed
    during the run, a player's identity unresolved, mismatched or unattested where the design doesn't allow it,
    its turns not fully audited, its protocol inconsistent, or a duplicate pair that failed validation."""
    data = dict(pairing_data)
    if 'xs' not in data and data.get('root') is not None:
        data = {**pairing_data_of(data), **{k: v for k, v in data.items() if k in ('a', 'b')}}
    xs, groups = list(data.get('xs') or []), data.get('groups')
    v = decide(xs, design, n_pairings, look, groups=groups)
    elig = eligibility or {}
    reasons = []
    if eligibility is None:
        reasons.append('no eligibility facts were given')
    reg = elig.get('registration')
    reasons += registration_problems(reg, design, elig.get('first_started'), None, elig.get('implementation_changes') or ())
    if (design or {}).get('registered') is False:
        reasons.append('the plan declined registration')
    players = elig.get('players') or {}
    named = [p for p in (data.get('a'), data.get('b')) if p] or list(players)
    for p in named:
        reasons += _player_problems(p, players.get(p), design or {})
    rejected = data.get('rejected')
    if rejected is None:
        reasons.append('the duplicate pairs were not validated')
    elif rejected:
        reasons.append(f'{len(rejected)} duplicate pair{"s" if len(rejected) != 1 else ""} failed validation '
                       f'({rejected[0].get("reason")}{"; ..." if len(rejected) > 1 else ""})')
    reasons = list(dict.fromkeys(reasons))
    a = assumptions(v, data.get('fps'), groups)
    calibrated = {'supported': True, 'unsupported': False}.get(a['status'])
    eligible = not reasons
    test = v['verdict']
    count_outside = (design or {}).get('outside_calibration') == 'count'
    counts = eligible and test in DECIDED and (calibrated is True or (calibrated is False and count_outside))
    standing = None
    if eligible and test in DECIDED:
        standing = 'decided' if calibrated else ('decided (outside calibrated range)' if calibrated is False
                                                 else 'decided (assumptions unchecked)')
    judge = judge_fingerprint()
    cal = calibration_id(design)
    official = bool(reg) and not registration_problems(reg, design) and reg.get('judge_fingerprint') == judge \
        and reg.get('calibration_id') == cal
    return {**v, 'verdict': test if eligible else 'ineligible', 'test_verdict': test, 'eligible': eligible,
            'ineligible_reasons': reasons, 'assumptions': a, 'calibrated': calibrated, 'counts_toward_leader': counts,
            'standing': standing,
            'identity_status': {p: (players.get(p) or {}).get('identity') for p in named},
            'audit_status': {p: (players.get(p) or {}).get('audit') for p in named},
            'judge_fingerprint': judge, 'judge': judge, 'calibration_id': cal,
            'registered_judge': (reg or {}).get('judge_fingerprint'), 'official': official}


def pairing_data_of(spec: dict) -> dict:
    return pairing_data(spec['root'], spec.get('conversations') or [], spec['a'])


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
                               'slate': slate, 'pos': d['deal'] // block, 'round': d.get('round') or 1, 'fp': d.get('fp'),
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
                        'net_bb': sum((o['x'] if o['a'] == a else -o['x']) / 50 for o in chunk),
                        'cells': [(str(o['slate']), o['pos'], frozenset((o['a'], o['b']))) for o in chunk]})
    return out


def outcome(s) -> float:
    band = TIE_BB * s['deals']
    return 1.0 if s['net_bb'] > band else 0.0 if s['net_bb'] < -band else 0.5


def _elo_boot(names, comps, games, Ws, iters=40):
    """Session-Elo MAP ratings refitted under each row of bootstrap weights (boot x sessions): a session keeps
    the weight of the slate (or aligned block) it was played on, so sessions of one conversation, and of
    pairings that shared the cards, are resampled together."""
    N = len(names)
    idx = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(games), N))
    for k, g in enumerate(games):
        X[k, idx[g['a']]] += 1
        X[k, idx[g['b']]] -= 1
    y = np.array([outcome(g) for g in games], float)
    w = np.zeros((len(Ws), N))
    pr = 1 / ELO_PRIOR ** 2
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(w @ X.T)))
        grad = (Ws * (y - p)) @ X - pr * w
        H = np.einsum('bs,si,sj->bij', Ws * p * (1 - p), X, X) + pr * np.eye(N)
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        w += step
        if np.abs(step).max() < 1e-9:
            break
    A = np.eye(N)
    for c in comps:
        ii = [idx[q] for q in c]
        A[np.ix_(ii, ii)] -= 1 / len(c)
    return w @ A.T


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
         n_pairings=None, notes=(), playing=(), recorded=None, eligibility=None, rejected=(), registration=None) -> dict:
    """The leaderboard from duplicate deals (as `load_deals` gives them). `meta` gives each player's identity,
    kind and quarantine reason; with `pooled` (all-time standings) everything is descriptive and quarantined
    players are listed but left out of the fit. `plan` (a tournament's plan.json) gives the design, the roster
    and the pairings planned; `playing` players with a match under way; `recorded` the run's decisions.json
    (the official decisions); `eligibility` ({frozenset((a, b)): the runner's facts}) what today's adjudication
    of each pairing is checked with; `rejected` the duplicate pairs that failed validation; `registration`
    the run's `registration_status` (from the plan alone when not given)."""
    meta = meta or {}
    raw = [d for d in deals if d['a'] != d['b']]
    planned_pairs = []
    for a, b in (plan or {}).get('matchups', []):
        if {a, b} not in [set(x) for x in planned_pairs]:
            planned_pairs.append((a, b))
    seen = {frozenset((d['a'], d['b'])) for d in raw}
    n_pairings = n_pairings or len(planned_pairs) or max(1, len(seen))
    reg = None
    if pooled:
        design = None
    else:
        if design is None:
            design = normalize_design(plan) if plan else None
        reg = registration if registration is not None else registration_status(plan)
        if design is None:
            design = unregistered((plan or {}).get('deals') or max((d.get('planned') or 0 for d in raw), default=1),
                                  n_pairings, (plan or {}).get('replicates') or 1)
        elif not reg['registered']:
            design = {**unregistered(design['round_deals'], n_pairings, design['replicates']), 'legacy': design.get('legacy')}
        else:
            design = {**design, 'registered': True}
    block = int((design or {}).get('block') or BLOCK)
    registered = not pooled and bool(design and design.get('registered'))
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
    cell_ix = {(c['slate'], c['pos'], frozenset((c['a'], c['b']))): i for i, c in enumerate(cells)}
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

    pairings = _pairings(used, obs, design, n_pairings, pooled, recorded or {}, eligibility or {}, list(rejected),
                         reg, lambda p: info(p)['quarantine_reason'])
    vmap = {frozenset((x['a'], x['b'])): x for x in pairings}
    _fit_diagnostics(pairings, names, idx, comps, cells, strata, point, draws, W)

    def result(p, q, outside=False):
        """'win', 'loss', 'even' or None: the decided verdict of p against q that counts towards a decided rank
        or leader (official, eligible, within the calibrated range unless the design counts the rest; with
        `outside`, also one outside that range), between players not quarantined in the run."""
        x = vmap.get(frozenset((p, q)))
        if pooled or not x or x['verdict'] not in DECIDED or info(p)['quarantine_reason'] or info(q)['quarantine_reason']:
            return None
        if not (x.get('counts_toward_leader') or (outside and x.get('official') and x.get('eligible')
                                                   and x.get('calibrated') is False)):
            return None
        if x['verdict'] == 'even':
            return 'even'
        return 'win' if (x['verdict'] == 'a') == (x['a'] == p) else 'loss'

    games = sessions(obs, session)
    elo, cov = _elo(names, comps, games) if names else (np.zeros(0), np.zeros((0, 0)))
    elo_draws = np.zeros((0, len(names)))
    if names and games and len(W) >= 2:
        Ws = np.stack([W[:, [cell_ix[k] for k in g['cells'] if k in cell_ix]].mean(1) for g in games], 1)
        elo_draws = _elo_boot(names, comps, games, Ws)
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
        outside = None if lead else next((p for p in comp if len(comp) > 1 and all(
            result(p, q, True) == 'win' for q in comp if q != p)), None)
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
            'leader_outside_calibration': outside,
            'misfit_pairings': sum((x.get('additivity') or {}).get('status') == 'warning' for x in mine),
            'over': order[1] if front else None, 'front_runner_margin': margin, 'ci95': ci95, 'power': power,
            'pairings': len(mine), 'decided': sum(x['verdict'] in DECIDED for x in mine),
            'has_bot': any(info(p)['kind'] == 'bot' for p in comp),
            'kind': 'descriptive' if pooled else 'decided' if lead else 'descriptive',
            'headline': _headline(lead, front, margin, ci95, mine, pooled, outside)})
        for pos, p in enumerate(order, 1):
            i = idx[p]
            se, sci, df, basis = interval([p], unit((p, 1)))
            mine_s = [s for s in games if p in (s['a'], s['b'])]
            won = sum(1 for s in mine_s if outcome(s) == (1.0 if s['a'] == p else 0.0))
            ties = sum(1 for s in mine_s if outcome(s) == 0.5)
            sd = math.sqrt(max(cov[i, i], 0))
            sd_boot = float(np.std(elo_draws[:, i], ddof=1)) if len(elo_draws) >= 2 and df and df >= 1 else None
            q = t_ppf(0.975, df) if sd_boot is not None else 1.96
            elo_basis = 'posterior' if sd_boot is None or sd_boot <= sd else 'bootstrap'
            sd = max(sd, sd_boot or 0.0)
            rank, rng_, tied = decided[p]
            players.append({
                'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': ci,
                'identity_status': _status_of(p, pairings, 'identity_status'),
                'audit_status': _status_of(p, pairings, 'audit_status'),
                'strength': round(float(point[i]), 1), 'strength_se': se, 'strength_ci95': sci, 'df': df,
                'ci_basis': basis, 'position': pos, 'rank': rank, 'rank_kind': 'decided' if rank else 'descriptive',
                'rank_range': rng_, 'rank_tied_with': tied, 'in_fit': True,
                'identity_resolved': bool(info(p).get('identity_resolved', identity_resolved(info(p)['identity']))),
                'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'], 'deals': tot[p]['deals'],
                'session_elo': round(1500 + K * float(elo[i])),
                'session_elo_ci95': [round(1500 + K * (elo[i] - q * sd)), round(1500 + K * (elo[i] + q * sd))],
                'session_elo_basis': elo_basis,
                'inferential': registered and not info(p)['quarantine_reason'],
                'sessions': len(mine_s), 'session_w': won, 'session_t': ties, 'session_l': len(mine_s) - won - ties,
                'quarantined': bool(info(p)['quarantine_reason']), 'quarantine_reason': info(p)['quarantine_reason'],
                'unrated_reason': None})
    unrated = sorted(excluded | ({p for d in raw for p in (d['a'], d['b'])} - set(names)))
    for p in unrated:
        players.append({
            'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': None,
            'identity_status': _status_of(p, pairings, 'identity_status'),
            'audit_status': _status_of(p, pairings, 'audit_status'),
            'strength': None, 'strength_se': None, 'strength_ci95': None, 'df': None, 'ci_basis': None,
            'position': None, 'rank': None, 'rank_kind': 'descriptive', 'rank_range': None, 'rank_tied_with': [],
            'in_fit': False, 'identity_resolved': bool(info(p).get('identity_resolved', identity_resolved(info(p)['identity']))),
            'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'], 'deals': tot[p]['deals'],
            'session_elo': None, 'session_elo_ci95': None, 'session_elo_basis': None, 'inferential': False,
            'sessions': 0, 'session_w': 0, 'session_t': 0,
            'session_l': 0, 'quarantined': p in excluded,
            'quarantine_reason': info(p)['quarantine_reason'] if p in excluded else None,
            'unrated_reason': info(p)['quarantine_reason'] if p in excluded else 'played only unrated opponents'})

    final = {'a', 'b', 'even', 'exhausted', 'ineligible'}
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

    notes = list(notes) + _notes(pooled, design, pairings, comps, groups, obs, repeats, excluded, unrated, players, block,
                                 reg, list(rejected))
    return {'objective': OBJECTIVE, 'design': design, 'registration': reg, 'pooled': pooled, 'deals': len(raw),
            'hands': 2 * len(raw), 'rejected': list(rejected), 'judge_fingerprint': judge_fingerprint(),
            'fit': {'method': 'least squares on distinct duplicate deals: each distinct deal counts once, so a '
                              'pairing weighs as much as it has deals, and a player\'s strength is against the '
                              'opponents it actually met, as often as it met them',
                    'residual': 'per pairing: the observed edge minus the edge the strengths imply, with a '
                                'bootstrap standard error; additivity "warning" when it is beyond 2 of them',
                    'primary': 'the direct pairwise results; strengths summarise them'},
            'observations': len(obs), 'repeats': repeats, 'block': block, 'boot': boot,
            'equiv_bb100': EQUIV_BB100, 'session_deals': session,
            'min_conversations': (design or {}).get('min_conversations') or MIN_CONVERSATIONS,
            'inference': ('descriptive only: pooled across runs, so no result here is a test' if pooled else
                          'descriptive only: this run is not preregistered (no valid registration before play), '
                          'so no result here is a test'
                          if not registered else
                          'decided results come only from the official decisions of the preregistered '
                          'sequential test of each pairing, recorded by the registered judge; strengths, intervals '
                          'and the order by points are descriptive'),
            'guarantee': None if not registered else GUARANTEE,
            'assumptions': None if not registered else ASSUMPTIONS,
            'components': comps, 'groups': groups, 'leader': leaders[0] if leaders else None, 'leaders': leaders,
            'players': players, 'pairings': pairings, 'matchups': pairings, 'roster': roster, 'notes': notes}


def _headline(lead, front, margin, ci95, pairings, pooled, outside=None) -> str:
    """One line per group, from that group's own data only."""
    if lead:
        return (f'{lead} is the decided leader: it won its pairing against every other player in this group, each '
                f'within the range the test is calibrated for.')
    if not front:
        return 'Only one player in this group.'
    open_ = sum(not x.get('counts_toward_leader') for x in pairings)
    what = 'pooled across runs, so nothing here is decided' if pooled else (
        f'no decided leader ({open_} of {len(pairings)} pairings without decided results that count)')
    if outside:
        what += (f'; {outside} won every pairing, but at least one of those verdicts is outside the calibrated range, '
                 f'where its error rate is unknown')
    iv = f', 95% {_iv(ci95)}' if ci95 else ''
    return f'{front} is ahead on points by {margin:+.0f} bb/100 over the next player{iv}; {what}.'


def _notes(pooled, design, pairings, comps, groups, obs, repeats, excluded, unrated, players, block, reg=None,
           rejected=()) -> list:
    notes = []
    if pooled:
        notes.append('All-time standings pool runs with different designs and stopping times, so they are '
                     'descriptive only: no verdict, rank or leader here is a test result. Only identical '
                     'protocols and implementations pool (model, reasoning level, the settings that change play, '
                     'and the code and CLI version that played).')
    elif not design.get('registered'):
        why = '; '.join((reg or {}).get('reasons') or []) or 'no registration was written before play'
        notes.append(f'This run is not preregistered ({why}), so nothing in it is a test: every verdict is '
                     f'descriptive, and any sample size shown for it is retrospective.')
    else:
        notes.append(f'Decided results come only from the preregistered sequential test of each pairing, as the '
                     f'runner recorded it when play stopped (the official decision, by judge '
                     f'{(reg or {}).get("judge_fingerprint")}). {ASSUMPTIONS} Strengths, intervals and the order by '
                     f'points are descriptive.')
        if any(x['verdict'] == 'early' for x in pairings):
            notes.append(f'A verdict needs at least {design.get("min_conversations", MIN_CONVERSATIONS)} conversations (fresh match pairs) of a '
                         f'pairing; pairings with fewer are "too early", whatever their numbers say.')
        if any(x['basis'] == 'blocks' for x in pairings):
            notes.append(f'Some pairings have one conversation: their intervals describe that conversation only '
                         f'(blocks of {block} deals), not how a new match would go.')
        wide = [x for x in pairings if x.get('calibrated') is False]
        if wide:
            notes.append('Outside the calibrated range: ' + ', '.join(f'{x["a"]} vs {x["b"]}' for x in wide) + '. Their '
                         'conversations differ by more than the test was calibrated for (or repeat deals), so their '
                         'error rate is unknown unless their results are symmetric, which no data can show; '
                         + ('the design counts such verdicts anyway.' if design.get('outside_calibration') == 'count'
                            else 'such a verdict never counts towards a decided rank or leader.'))
        bad = [x for x in pairings if x['verdict'] == 'ineligible']
        if bad:
            notes.append('Ineligible for a decided result: ' + '; '.join(
                f'{x["a"]} vs {x["b"]} ({"; ".join(x.get("ineligible_reasons") or [])})' for x in bad) + '.')
        retro = [x for x in pairings if x.get('retrospective') and x.get('test_verdict') in DECIDED + ('ineligible',)]
        if retro:
            notes.append('No decision by the registered judge was recorded at the last look of ' + ', '.join(
                f'{x["a"]} vs {x["b"]}' for x in retro) + '; what is shown there is today\'s recomputation '
                '(retrospective), which decides nothing.')
        changed = [x for x in pairings if x.get('official') and x.get('recomputed') and not x['recomputed']['agrees']]
        if changed:
            notes.append('Today\'s judge would judge some looks differently (' + ', '.join(
                f'{x["a"]} vs {x["b"]}: {x["recomputed"]["verdict"]}' for x in changed) + '); that recomputation is '
                'retrospective: the official verdict is the one recorded by the registered judge.')
        stale = [x for x in pairings if x.get('official') and x.get('integrity_now') is False]
        if stale:
            notes.append('The match folders of ' + ', '.join(f'{x["a"]} vs {x["b"]}' for x in stale) + ' no longer '
                         'pass validation, so their recorded verdicts no longer count towards a decided leader.')
    if rejected:
        notes.append(f'{len(rejected)} duplicate pair{"s" if len(rejected) != 1 else ""} failed validation (the b-first '
                     f'play is not a-first with the seats swapped on the same deals) and were left out: '
                     + '; '.join(f'{x.get("matchup")}{"" if x.get("hand") is None else " hand " + str(x["hand"])}: '
                                 f'{x["reason"]}' for x in list(rejected)[:3]) + ('; ...' if len(rejected) > 3 else '.'))
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
                     'answered): their strengths are shown, but no decided rank or leader involves them and they are '
                     'not pooled into the all-time standings.')
    if any(len(c) > 2 and _is_tree(c, obs) for c in comps):
        notes.append('Every pairing in a group is its only link between those players, so the strengths fit the '
                     'pairings exactly and nothing checks that edges add up across players.')
    misfit = [x for x in pairings if (x.get('additivity') or {}).get('status') == 'warning']
    if misfit:
        notes.append('The strengths do not add up for ' + ', '.join(
            f'{x["a"]} vs {x["b"]} (observed {x["observed_edge"]:+.0f}, the strengths imply {x["fitted_edge"]:+.0f})'
            for x in misfit) + ': one number per player cannot describe these results, so read the pairings '
            'directly; a strength is an average over the opponents a player happened to meet.')
    notes.append('Session Elo is the share of 10-deal sessions won: a different, secondary statistic that never '
                 'decides anything. Its interval is the wider of a posterior one and a bootstrap that resamples '
                 'sessions with their slates and conversations.')
    return notes


def _is_tree(comp, obs):
    edges = {frozenset((o['a'], o['b'])) for o in obs if o['a'] in comp}
    return len(edges) == len(comp) - 1


def _status_of(p, pairings, key):
    """A player's identity or audit status over its pairings: the worst one any of them recorded."""
    order = ['mismatch', 'none', 'incomplete', 'unattested', 'opaque', 'attested', 'complete', 'bot']
    seen = [x[key][p] for x in pairings if isinstance(x.get(key), dict) and x[key].get(p)]
    return min(seen, key=lambda s: order.index(s) if s in order else -1) if seen else None


def _fit_diagnostics(pairings, names, idx, comps, cells, strata, point, draws, W):
    """Per pairing: the edge observed over the deals in the fit, the edge the strengths imply (fitted_edge), the
    residual and its bootstrap standard error, and `additivity` ({'status': 'ok' | 'warning' | 'untestable',
    'z'}): a warning when the residual is beyond 2 standard errors. A group whose pairings form a tree (or two
    players) fits them exactly: untestable."""
    if not names:
        return
    comp_of = {p: i for i, c in enumerate(comps) for p in c}
    tree = {i: len({frozenset((c['a'], c['b'])) for c in cells if comp_of[c['a']] == i}) <= len(comps[i]) - 1
            for i in range(len(comps))}
    n = np.array([c['n'] for c in cells], float)
    for x in pairings:
        if x['a'] not in idx or x['b'] not in idx:
            continue
        ix = [i for i, c in enumerate(cells) if {c['a'], c['b']} == {x['a'], x['b']}]
        sign = np.array([1.0 if cells[i]['a'] == x['a'] else -1.0 for i in ix])
        sx = np.array([cells[i]['sx'] for i in ix]) * sign
        observed = float(sx.sum() / n[ix].sum())
        i, j = idx[x['a']], idx[x['b']]
        fitted = float(point[i] - point[j])
        res = observed - fitted
        se, z, status = None, None, 'untestable' if tree[comp_of[x['a']]] else None
        if status is None and len(W) >= 2 and _df_basis([x['a'], x['b']], strata)[0] >= 1:
            obs_w = (W[:, ix] * sx).sum(1) / (W[:, ix] * n[ix]).sum(1)
            r = obs_w - (draws[:, i] - draws[:, j])
            se = float(np.std(r, ddof=1))
            if se > 1e-9:
                z = res / se
                status = 'warning' if abs(z) > 2 else 'ok'
        x.update({'observed_edge': round(observed, 1), 'fitted_edge': round(fitted, 1), 'residual': round(res, 1),
                  'residual_se': None if se is None else round(se, 1),
                  'additivity': {'status': status or 'untestable', 'z': None if z is None else round(z, 2)}})


_DECISION_KEYS = ('verdict', 'test_verdict', 't', 'boundary', 'signflip', 'decided_at_look', 'ci_decision', 'equiv_ci',
                  'eligible', 'ineligible_reasons', 'assumptions', 'calibrated', 'counts_toward_leader', 'standing',
                  'conversation_share', 'min_conversations', 'identity_status', 'audit_status', 'calibration_id',
                  'looks_used', 'looks_planned', 'edge', 'ci95', 'se', 'df', 'basis', 'deals', 'conversations',
                  'blocks', 'reason', 'sd_per_deal', 'power', 'alpha_pairing')


def _official(rec: dict | None, reg: dict | None) -> bool:
    """Whether a recorded look is an official decision: written by the adjudicator, by the registered judge,
    with the registered calibration."""
    if not rec or not reg or not reg.get('registered') or 'eligible' not in rec or rec.get('official') is False:
        return False
    judge = rec.get('judge_fingerprint') or rec.get('judge')
    return judge == reg.get('judge_fingerprint') and rec.get('calibration_id') == reg.get('calibration_id')


def _pairings(raw, obs, design, n_pairings, pooled, recorded, eligibility=None, rejected=(), reg=None,
              blocked=lambda p: None) -> list:
    """One row per pair of players, pooling every conversation between them, with its verdict at the last
    completed planned look. Descriptive only (verdict 'descriptive', `inferential` False) when pooled or when the
    run is not preregistered. In a preregistered run every row is adjudicated today (`adjudicate`, with the
    pairing's eligibility facts and duplicate-pair validation), and the verdict shown is the official one: the
    decision the runner recorded at that look with the registered judge (`official` True, `judge` its
    fingerprint; `recomputed` says what today's judge makes of it). Without such a record the row is
    `retrospective` (today's recomputation): a decided result there becomes verdict 'retrospective' and counts
    for nothing."""
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
    registered = bool(design and design.get('registered'))
    out = []
    for key, x in by.items():
        os_ = sorted(x['obs'], key=lambda o: (o['round'], o['replicate'], o['matchup'], o['deal']))
        sign = lambda o: o['x'] if o['a'] == x['a'] else -o['x']
        rec = recorded.get(f'{x["a"]}__vs__{x["b"]}') or recorded.get(f'{x["b"]}__vs__{x["a"]}') or []
        if pooled or not registered:
            v = decide([sign(o) for o in os_], {'max_rounds': 1, 'round_deals': len(os_), 'block': BLOCK},
                       n_pairings, 0, groups=[o['matchup'] for o in os_])
            v.update(verdict='descriptive', boundary=None, ci_decision=None, equiv_ci=None, signflip=None,
                     reason=None if pooled else 'this run is not preregistered, so nothing in it is a test',
                     assumptions=assumptions(v, [o['fp'] for o in os_], [o['matchup'] for o in os_]),
                     official=False, retrospective=False, eligible=None, ineligible_reasons=[], calibrated=None,
                     counts_toward_leader=False)
            beyond = 0
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
            mine = [y for y in rejected if {y.get('a'), y.get('b')} == set(key)]
            data = {'a': x['a'], 'b': x['b'], 'xs': [sign(o) for o in at], 'groups': [o['matchup'] for o in at],
                    'fps': [o['fp'] for o in at], 'rejected': mine}
            today = adjudicate(data, design, n_pairings, looks, eligibility.get(key))
            beyond = len(os_) - len(at) if looks else 0
            last = rec[looks - 1] if looks and len(rec) >= looks and rec[looks - 1].get('looks_used') == looks else None
            if _official(last, reg):
                v = {**today, **{k: last[k] for k in _DECISION_KEYS if k in last}, 'official': True,
                     'retrospective': False, 'judge': last.get('judge_fingerprint') or last.get('judge'),
                     'recomputed': {'verdict': today['verdict'], 'judge': today['judge'],
                                    'agrees': today['verdict'] == last.get('verdict')}}
                if mine:
                    v.update(integrity_now=False, counts_toward_leader=False)
            else:
                v = {**today, 'official': False, 'retrospective': looks >= 1, 'counts_toward_leader': False,
                     'recomputed': None}
                if today['verdict'] in DECIDED:
                    v.update(verdict='retrospective',
                             reason='no decision by the registered judge was recorded at this look; this is today\'s '
                                    'recomputation, which decides nothing')
        v['inferential'] = bool(v.get('official')) and v['verdict'] in DECIDED
        last_rec = rec[-1] if rec else {}
        out.append({'a': x['a'], 'b': x['b'], 'matchups': x['matchups'], 'matchup': x['matchups'][-1],
                    'plays': x['plays'], 'hands': 2 * x['plays'], 'net_a_chips': x['chips'],
                    'deals_total': len(os_), 'deals_since_look': beyond, 'in_progress': bool(beyond),
                    'a_session_wins': wins[key][0], 'b_session_wins': wins[key][1], **v,
                    'quarantine': [f'{p}: {blocked(p)}' for p in (x['a'], x['b']) if blocked(p)],
                    'recorded_verdict': last_rec.get('verdict'), 'recorded_looks': last_rec.get('looks_used')})
    return sorted(out, key=lambda m: (m['a'], m['b']))


def run_eligibility(root, folders: list[str], a: str, b: str, reg: dict | None, evidence: dict | None = None) -> dict | None:
    """The eligibility facts of one pairing, gathered by the runner's own `eligibility` from its match folders
    (the same facts its adjudication used), with when play began."""
    try:
        from .runner import eligibility
    except ImportError:
        return None
    facts = eligibility(Path(root), folders, (a, b), reg)
    return {**facts, 'first_started': (evidence or {}).get('first_started')}


def ratings(root, boot=BOOT, seed=0, session=SESSION) -> dict:
    """The leaderboard of every match under root, with each player's style stats and audit summary."""
    from .stats import collect
    root = Path(root)
    plan = _json(root / 'plan.json')
    playing = set()
    for cfg_path in root.rglob('config.json'):
        if cfg_path.parent.name in ('a-first', 'b-first') and not (cfg_path.parent / 'result.json').exists():
            playing |= set((_json(cfg_path) or {}).get('players', []))
    stats = collect(root)
    meta = load_meta(root, {p: s.get('served_models') for p, s in stats.items()})
    rejected = []
    deals = load_deals(root, rejected)
    evidence = run_evidence(root)
    reg = registration_status(plan, evidence) if plan else None
    eligibility = {}
    if reg and reg['registered']:
        folders = defaultdict(list)
        for d in deals:
            k = frozenset((d['a'], d['b']))
            if d['matchup'] not in folders[k]:
                folders[k].append(d['matchup'])
        for k, fs in folders.items():
            a, b = sorted(k)
            eligibility[k] = run_eligibility(root, fs, a, b, plan.get('registration'), evidence)
    r = rate(deals, boot, seed, session, meta=meta, plan=plan, playing=playing,
             recorded=_json(root / 'decisions.json') or {}, eligibility=eligibility, rejected=rejected, registration=reg)
    from .stats import LABELS, POSITIONS
    audit = {p: {**(s.get('audit') or {}), 'identity': s.get('identity'), 'fallbacks': s.get('fallbacks'),
                 'retries': s.get('retries'), 'served_models': s.get('served_models')} for p, s in stats.items()}
    return {**r, 'stats': stats, 'stat_labels': LABELS, 'position_labels': POSITIONS, 'audit': audit}


VERDICT_TEXT = {
    'a': '{a} is better (decided)', 'b': '{b} is better (decided)',
    'even': 'even (decided): the edge is within ±{e} bb/100',
    'open': 'undecided so far',
    'exhausted': 'undecided after every planned look (not evidence that they are equal)',
    'early': 'too early to judge: too few conversations',
    'descriptive': 'descriptive only (not a test)',
    'ineligible': 'ineligible for a decided result',
    'retrospective': 'retrospective only: no decision by the registered judge was recorded here',
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


def _pct(x):
    return '?' if x is None else '>99%' if x >= 0.995 else '<1%' if x < 0.005 else f'{x:.0%}'


def design_line(d) -> str:
    """The preregistered target and its power under each dependence scenario, in words (or, for a run without a
    design, what one would have needed, labelled retrospective)."""
    if not d:
        return ''
    if not d.get('registered', True):
        x = d.get('retrospective') or {}
        if not x.get('deals_needed'):
            return 'No preregistered design: nothing in this run is a test.'
        return (f'No preregistered design: nothing in this run is a test. Retrospective sample size only: detecting a '
                f'{x["target_edge_bb100"]:.0f} bb/100 edge with 80% power would take about {x["deals_needed"]:,} deals '
                f'per pairing ({x["conversations_needed"]} conversations of {d["round_deals"]} deals), assuming a '
                f'per-deal spread of {x["prior_sd_bb100"]:.0f} and moderate dependence (ICC {x["assumed_icc"]}, '
                f'AR(1) {x["assumed_ar1"]}).')
    sc = {x['scenario']: x for x in d.get('power_scenarios') or []}
    per = f'{d["replicates"]} conversation{"s" if d["replicates"] != 1 else ""} of {d["round_deals"]} deals'
    head = (f'Planned to detect a {d["target_edge_bb100"]:.0f} bb/100 edge: up to {d["max_rounds"]} rounds of {per} '
            f'({d["max_deals"]:,} deals per pairing), a verdict from {d["min_conversations"]} conversations on.')
    if sc:
        m = sc['moderate']
        head += (f' Power at the target, per-deal spread {d["prior_sd_bb100"]:.0f}: {_pct(m["power"])} with the assumed '
                 f'dependence (ICC {m["assumed_icc"]}, AR(1) {m["assumed_ar1"]}), {_pct(sc["optimistic"]["power"])} with '
                 f'none, {_pct(sc["pessimistic"]["power"])} if it is worse (ICC {sc["pessimistic"]["assumed_icc"]}, '
                 f'AR(1) {sc["pessimistic"]["assumed_ar1"]}).')
    if d.get('underpowered'):
        why = 'a budget of ' + f'{d["budget_deals"]:,} deals' if d.get('max_rounds_basis') == 'budget' else 'its max_rounds'
        need = f'about {d["deals_needed"]:,}' if d.get('deals_needed') else f'more than {MAX_ROUNDS * d["round_deals"] * d["replicates"]:,}'
        head += f' Underpowered: capped by {why}; 80% would need {need} deals per pairing.'
    return head


def verdicts(r, labels=None) -> str:
    """Each pairing: the edge, its interval, the test against the stopping bar, and the verdict."""
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    order = {'a': 0, 'b': 0, 'even': 1, 'exhausted': 2, 'open': 3, 'early': 4, 'ineligible': 5, 'retrospective': 6,
             'descriptive': 7}
    d = r.get('design')
    reg = r.get('registration') or {}
    if r.get('pooled'):
        lines = ['Pairings (descriptive: pooled across runs, nothing here is a test):']
    elif d and not d.get('registered', True):
        why = '; '.join(reg.get('reasons') or []) or 'no preregistered design'
        lines = [f'Pairings (descriptive: this run is not preregistered ({why}); nothing here is a test):']
    else:
        lines = [f'Pairings (decided only by the preregistered sequential test, as recorded when play stopped by the '
                 f'registered judge {reg.get("judge_fingerprint") or "?"}; if its assumptions hold, at most '
                 f'{d["alpha"]:.0%} chance of any false verdict over {d["n_pairings"]} '
                 f'pairing{"s" if d["n_pairings"] != 1 else ""}; a verdict needs '
                 f'{d.get("min_conversations", MIN_CONVERSATIONS)}+ conversations):']
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
                         f', t = {t} vs bar {m["boundary"]:.2f} at look {m["looks_used"]} of {m["looks_planned"]}'
                         + ('' if m.get('signflip') is None else
                            f', sign-flip test {"rejects" if m["signflip"] else "does not reject"}')
                         + (f', decided at look {m["decided_at_look"]}' if m.get('decided_at_look')
                            and m['decided_at_look'] != m['looks_used'] else ''))
        state = VERDICT_TEXT[m['verdict']].format(a=a, b=b, e=r.get('equiv_bb100', EQUIV_BB100))
        if m['verdict'] in DECIDED and m.get('standing') and m['standing'] != 'decided':
            state += m['standing'][len('decided'):]
        if m.get('official'):
            state += f'; official (recorded by judge {m.get("judge")})'
        elif m.get('retrospective'):
            state += '; retrospective (today\'s recomputation)'
        lines.append(f'{head}: {state}.')
        if m['verdict'] == 'ineligible':
            lines.append('      ' + _named('; '.join(m.get('ineligible_reasons') or []), name, [m['a'], m['b']]) + '.')
        elif m.get('reason') and m['verdict'] != 'descriptive':
            lines.append(f'      {m["reason"]}.')
        st = (m.get('assumptions') or {}).get('status')
        if m['verdict'] not in ('descriptive',) and st:
            share = ((m['assumptions'].get('checks') or {}).get('conversation_share') or {}).get('value')
            lines.append(f'      assumptions: {st}' + (f' (conversation share {share} vs calibrated up to {FAMILY_SHARE})'
                                                      if share is not None else ''))
        if m.get('recomputed') and not m['recomputed']['agrees']:
            lines.append(f'      today\'s judge would say {m["recomputed"]["verdict"]} (retrospective; the recorded '
                         f'verdict stands).')
        if (m.get('additivity') or {}).get('status') == 'warning':
            lines.append(f'      the strengths imply {m["fitted_edge"]:+.0f} bb/100 here, not the observed '
                         f'{m["observed_edge"]:+.0f}: they do not add up across pairings.')
        if m.get('deals_since_look'):
            lines.append(f'      {m["deals_since_look"]} more deals are in a round under way, not yet looked at.')
        need = m['power']['deals_for_80pct']
        if m['verdict'] in ('open', 'exhausted') and need.get('50') and m['conversations'] >= 2:
            lines.append('      deals per pairing for an 80% chance to detect a real edge of '
                         + ', '.join(f'{e} bb/100: {need[e]:,}' for e in need) + ' (from the observed spread of its '
                         'conversations; a planning number, not a forecast)')
    lead = [x for x in r.get('leaders') or [] if x]
    if not r.get('pooled') and (d or {}).get('registered', True):
        lines.append('Decided leader: ' + ('; '.join(f'{name(x["player"])} (won its pairing against every other '
                                                    f'player of its group, within the calibrated range)'
                                                    for x in lead) if lead else 'none.'))
        for g in r.get('groups') or []:
            if g.get('leader_outside_calibration'):
                lines.append(f'  {name(g["leader_outside_calibration"])} won every pairing of its group, but not all '
                             f'within the calibrated range, so it is not a decided leader.')
    if d and not r.get('pooled'):
        lines.append(design_line(d))
    return '\n'.join(lines)


def summary(r) -> str:
    """The verdicts, the standings and the caveats, as the CLI prints them."""
    labels = r.get('labels')
    return verdicts(r, labels) + '\n\n' + table(r, labels) + '\n\n' + '\n'.join(f'- {n}' for n in r['notes'])
