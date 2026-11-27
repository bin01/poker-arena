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
recomputes is retrospective), and its evidence still stands today (`evidence_now`: the hands it judged are
all here, replay from their seeds and hash as they did, and their records pass today's checks). Every
verdict carries the checks of what its guarantee needs (`assumptions`: deals within the game's bound, no deal
repeated across conversations). A decided rank and a leader are built only from official verdicts whose
evidence stands, whose checks passed and whose edge did not drift (`publication`: 'decided_with_drift' counts
for nothing until a confirmation run; under a design's strict_identity neither does an unattested player's
result, which otherwise carries identity_qualified): a leader has such a win over every other player of its
group. Each pairing's primary interval is the anytime-valid one (`ci_official`); every other interval is
'fixed-time descriptive' (`ci_kind`). Everything else (strengths, their intervals and fit diagnostics, the order by points, the front runner, session
Elo, the model-based result, every all-time number, every number of a run that is not preregistered) is
descriptive, recomputed on every refresh, and labelled so.

The test. The unit is the conversation, and its observation is its mean edge, which no distribution can push
beyond the game's bound (`outcome_bound`: each of a duplicate deal's two plays moves at most one stack, so
|x| <= 100 x stack / bb bb/100; 10,000 at 100 big blinds). The official test is a betting confidence sequence
for bounded observations (Waudby-Smith and Ramdas, 2024): for each null mean m a bettor stakes a predictable
fraction of its capital on each new conversation, K_n(m) = prod (1 + lambda_i (Y_i - m)), lambda_i chosen from
the earlier conversations only and capped at BET_CAP / (bound + m), so the capital never goes negative. Under
any distribution of independent conversations whose mean is at most m, K_n(m) is a nonnegative supermartingale,
and Ville's inequality bounds the chance it ever reaches 2 / alpha_pairing by alpha_pairing / 2: at every look,
whenever play stops, with no assumption on the shape of the results. 'a' is a's bettor against m = 0 reaching
that bar, 'b' the mirror one, 'even' both bettors against the margins (+-EQUIV_BB100) reaching it. Whatever the
true edge, a false verdict needs at most two of those bettors, each valid for it, to reach the bar, so a pairing
gives one with chance at most alpha_pairing (Bonferroni over the pairings: alpha over all of them). The null
is conditional: E[Y_t | the conversations before t] <= m at every t (a common mean is one case of it). When the
edge drifts, 'a' means that a's expected edge was positive for at least part of the run, not that the run's
average favours it; `drift` (early against late conversations) is reported beside every verdict. `model_based` is the round 6 gate (a conversation-level
t and a sign-flip test with bars calibrated for a family of shapes): secondary, descriptive, never official;
on the adaptation track it judges, in exploratory words (`language`: a player leans ahead, it never wins).

Planning. A design's power is that of the official test, simulated (`procedure`): the chance that the better
player is decided by the last look, under optimistic, assumed (assumed_icc, assumed_ar1) and pessimistic
dependence; by default it plays as many rounds as the assumed case needs for 80%. Because the bound, not the
spread, sets the cost (a pairing needs about log(2 / alpha_pairing) x bound / edge conversations, whatever their
length), short conversations are far cheaper in deals; `length_tradeoff` gives the deals needed by conversation
length. `leader_chances` gives the chance that a round robin names the true best player its decided leader.

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
DRIFT_ALPHA = 0.01          # the drift diagnostic flags early and late conversations this unlikely to share a mean
DEFAULT_DESIGN = {'alpha': 0.05, 'round_deals': 50, 'block': BLOCK}   # max_rounds: from the power target
TARGET_EDGE = 50            # bb/100: the edge a plan is powered to detect unless it says otherwise
PRIOR_SD = 350              # bb/100: the per-deal spread assumed when planning (first model runs: about 340)
ASSUMED_ICC = 0.01          # planning: share of the per-deal variance that is a conversation effect (moderate)
ASSUMED_AR1 = 0.2           # planning: AR(1) memory between consecutive deals of a conversation (moderate)
POWER = 0.8
MAX_ROUNDS = 200             # looks at most: the official test is valid at every look, so looks cost nothing
MODEL_MAX_ROUNDS = 40       # the model-based gate is computed only for designs this short
MODEL_MAX_CONVERSATIONS = 200
MIN_CONVERSATIONS = 8       # the model-based gate needs this many independent conversations of the pairing
BET_CAP = 0.9               # the official bettor stakes at most this share of what it could lose without going broke
BET_PRIOR_VAR = PRIOR_SD ** 2   # the bettor's prior variance of a conversation mean, worth one conversation
OFFICIAL_TEST = 'betting confidence sequence for bounded outcomes (Waudby-Smith and Ramdas 2024)'
FAMILY = ('normal', 'exponential', 'lognormal', 'rare_loss')   # null shapes the model-based bars are calibrated for
FAMILY_SHARE = 0.5          # in the skewed shapes, the conversation effect's share of a conversation mean's variance
POWER_EDGES = (20, 50, 100)
BOOT = 300
CAL_SIMS = 100_000          # null sequences per shape behind each model-based level
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
ASSUMPTIONS = ('With no edge, a pairing is called for either player with chance at most alpha / n_pairings (so at '
               'most alpha over all of them), at every look and whenever play stops, for any distribution of '
               'results: the official test is a betting confidence sequence for bounded outcomes, which needs that '
               'conversations are independent (fresh memories, no deal repeated across them), that no duplicate '
               'deal moves more than the stacks allow, and that the null holds conditionally (given the '
               'conversations before it, each one\'s expected edge is at most the null value; a common mean is one '
               'case). Reading a verdict as "the better model" needs more: that each name served one stable model, '
               'so that each conversation\'s expected edge stayed the same over the run. If the edge drifts, a '
               'verdict means a player was ahead for part of the run, not on average; the drift diagnostic beside '
               'each verdict can flag that, but "no drift detected" is not evidence that the model stayed the same. '
               '"even" (within the equivalence margin) has the '
               'same guarantee. Each verdict reports those checks. The model-based result shown beside it (a t and '
               'sign-flip test calibrated for a family of shapes) is secondary and never official.')
GUARANTEE = ASSUMPTIONS
STABLE_CONDITION = ('a better model only if each name served one stable model, so that each conversation\'s '
                    'expected edge stayed the same over the run (the conditional-mean null); no drift detected is '
                    'not proof of that, and dated model snapshots should be pinned where the provider offers them')
STRENGTH_KIND = ('centred fitted strength (bb/100): the strengths sum to zero in each group, and the difference '
                 'of two players\' strengths is the edge the fit predicts between them; it is not a player\'s '
                 'winnings against an average opponent')
OBSERVED_KIND = ('observed opponent-weighted winnings (bb/100): the mean of a player\'s direct edges, one per '
                 'opponent it met, each opponent weighted equally')
CI_SEQUENTIAL = 'anytime-valid sequential'
CI_FIXED = 'fixed-time descriptive'
INTERVALS = {
    CI_SEQUENTIAL: 'the official interval of a pairing (`ci_official`): the betting confidence sequence, which covers '
                   'the mean edge at every look at once, so it stays valid whenever play stopped and whatever the '
                   'shape of the results (given independent conversations); its level is 1 - alpha_pairing',
    CI_FIXED: 'an ordinary interval (a t interval on conversation means for a pairing, the bootstrap for strengths, '
              'margins and session Elo): it would cover 95% of the time for a sample of fixed size, but play stopped '
              'when the results said so and the front runner is picked from the same data, so it is not adjusted '
              'for either and its 95% is not a guarantee; descriptive only',
}
STRENGTH_COVERAGE = {'stopped_runs_checked': True, 'guaranteed': False, 'strength': 0.95, 'pairing_t': 0.95,
                     'front_runner_margin': 0.93,
                     'meaning': 'in seeded 4-player round robins whose pairings stop when the official test decides '
                                '(400 tournaments per setting, conversation effects, up to half the pairings '
                                'stopping early; tests/test_publication.py), strength intervals covered 94-95%, '
                                'pairing t intervals 95% and the front runner\'s margin 92-93% (it is picked from the '
                                'same data); one family of scenarios, not a guarantee, so they stay descriptive'}
DRIFT_RULE = ('A decided verdict whose drift diagnostic (early against late conversations of the judged ones) detects '
              'a change of edge is "decided_with_drift": it stays shown, but it counts toward no decided rank or '
              'leader, because it says only that a player was ahead for part of the run; it needs a confirmation run '
              'on a stable model snapshot. Drift after play stopped is never seen, so pin dated model snapshots.')
STRICT_IDENTITY = ('strict identity policy: a pairing with an unattested player (its backend does not report which '
                   'model answered) gets no counted result, whatever the design allows to play')
TRACKS = ('official', 'adaptation')
ADAPTATION_DEALS = 50       # deals per conversation on the adaptation track: models remember the session
ADAPTATION_REPLICATES = 4   # conversations per pairing per round on the adaptation track
ADAPTATION_LABEL = 'adaptation · model-based · not official'
ADAPTATION_TEST = ('model-based gate (a conversation-level t and a sign-flip test, with bars calibrated for a family '
                   'of shapes): its error rate holds only within that family, so its verdicts are never official')
ADAPTATION_ASSUMPTIONS = ('Adaptation track: the same players in long conversations, so models remember and adapt '
                          'to the opponent. Its verdicts come from the model-based gate (a conversation-level t and '
                          'a sign-flip test with bars calibrated for a family of result shapes), which stops play '
                          'by its own registered design; its error rate is not guaranteed outside that family, so '
                          'nothing here is official, ranks no one and names no leader.')
REFERENCE = {   # the reference bot's measured limits (docs/methodology.md, "How good at poker")
    'version': 'reference-1',
    'kind': 'a fixed heuristic, not a solver: tight-aggressive, position-aware, equity against the range the '
            'opponent\'s betting implies, pot odds, a fixed bluff frequency; it never adapts to its opponent',
    'measured_with': '2,000 duplicate deals per pairing (40 conversations of 50), 100-big-blind stacks, bots at '
                     'their default settings; edge in bb/100 with the standard error between conversations',
    'results': [{'a': 'reference', 'b': 'random', 'edge_bb100': 408, 'se': 50},
                {'a': 'reference', 'b': 'station', 'edge_bb100': 290, 'se': 14},
                {'a': 'reference', 'b': 'rule', 'edge_bb100': 169, 'se': 24},
                {'a': 'rule', 'b': 'random', 'edge_bb100': 856, 'se': 67},
                {'a': 'rule', 'b': 'station', 'edge_bb100': 273, 'se': 13},
                {'a': 'station', 'b': 'random', 'edge_bb100': 44, 'se': 78}],
    'limits': 'it beats every baseline bot directly, but wins about half as much from the random bot as the rule '
              'bot does (+408 against +856 bb/100), so it is not a maximally exploitative player, and how it does '
              'against strong play has not been measured (no solver or expert comparison): a positive edge against '
              'it means "beats a solid fixed heuristic", not "plays well"'}
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
_DATE = r'(?:19|20)\d{2}-?(?:0[1-9]|1[0-2])-?(?:0[1-9]|[12]\d|3[01])'
_SNAPSHOT = re.compile(rf'^(?P<base>.+?)-(?:{_DATE})$')


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
    if not configured or not served:
        return False
    a, b = canonical_model(configured), canonical_model(served)
    if not a or not b or a in ALIASES:
        return False
    if a == b or b in SERVED_ALIASES.get(a, ()):
        return True
    snap = _SNAPSHOT.match(b)
    return bool(snap) and snap.group('base') == a


def identity_match(requested, observed) -> str:
    """'match' when a model was reported for a reply and `same_model` says it is the requested one, 'mismatch'
    when one was reported and it is not (or nothing exact was requested), 'unattested' when none was reported.
    `observed` is one model or a list (every one must match). The one rule every consumer uses."""
    seen = [m for m in (observed if isinstance(observed, (list, tuple, set)) else [observed]) if m]
    if not seen:
        return 'unattested'
    return 'match' if all(same_model(requested, m) for m in seen) else 'mismatch'


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
    """{hand: record} of a match folder's committed hands: rows past state.json's hands_committed (a hand
    written before an interruption, which a resume replays) are not part of the record. A folder without a
    state.json is a legacy record (written before commits were recorded): every hand it holds is taken as
    committed, and still counts only if it verifies (`read_conversation` says so in `commit`)."""
    out = {}
    p = d / 'hands.jsonl'
    state = _json(d / 'state.json') if (d / 'state.json').exists() else None
    committed = state.get('hands_committed', state.get('hands')) if isinstance(state, dict) else None
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                h = json.loads(line)
                out[h['hand']] = h
            except (ValueError, KeyError, TypeError):
                pass
    if isinstance(committed, int):
        out = {k: h for k, h in out.items() if isinstance(k, int) and k < committed}
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


def verify_hand(game_cfg: dict, seed, button, hand: dict, events: list | None) -> dict:
    """Whether a recorded hand is what the engine deals and plays from its seed: the game's own record verifier
    (games.<game>.verify_hand: the deal regenerated from the seed, the recorded actions replayed legally, cards,
    board, payouts, summary and events compared). {'ok', 'errors'}; a game with no verifier, or one that fails
    to run, fails closed."""
    try:
        from .games import get_game
        game = get_game((game_cfg or {}).get('name') or 'holdem')(game_cfg or {})
        verify = getattr(game, 'verify', None)
        if verify is None:
            return {'ok': False, 'errors': [f'the game {type(game).__name__} has no record verifier']}
        v = verify(seed, button, hand, events)
        return {'ok': bool(v.get('ok')), 'errors': list(v.get('errors') or [])}
    except Exception as e:      # a record the verifier cannot even read is not verified
        return {'ok': False, 'errors': [f'the verifier failed: {e!r}'[:300]]}


_VERIFIED: dict = {}


def _stamp(d: Path) -> tuple:
    out = []
    for name in ('config.json', 'hands.jsonl', 'events.jsonl', 'state.json', 'calls.jsonl', 'raw.jsonl'):
        try:
            st = (d / name).stat()
            out.append((name, st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((name, None, None))
    return tuple(out)


def _rows_by_hand(p: Path, key: str) -> dict | None:
    if not p.exists():
        return None
    out = defaultdict(list)
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out[e.get(key)].append(e)
    return out


def linked_audit(cfg: dict, hand: int, events: list, calls: list, raw: list, replayed: dict | None = None) -> dict:
    """The audit of one hand's model decisions, derived from the records linked to each by its `decision_id`:
    the decision record (a `thought` event), the action it led to, every call attempt and every raw backend
    output row. Per seat: {'decisions', 'linked', 'verified', 'opaque', 'problems'}. A decision is verified when
    its id is on its action and on at least one call, its accepted (last) call says 'verified' and has a raw
    output row of its own (the same `call_id`, or the same decision id and attempt; a failed try's row does not
    count), and the decision record's own audit says the same; 'opaque' when
    the accepted call comes from a backend that exposes no record. With `replayed` (`runner.replay_decisions`:
    {decision id: problems} from replaying each accepted reply against the engine state before its action), a
    decision also needs a clean replay: its reply must reproduce the executed action, amount, correction and
    speech, else it is inconsistent. Bots make no decisions here."""
    ids = cfg.get('players') or []
    specs = cfg.get('specs') or {}
    out = {}
    acted = {(e.get('data') or {}).get('decision_id') for e in events if e.get('kind') == 'action'}
    by_call, by_raw = defaultdict(list), defaultdict(list)
    for c in calls:
        by_call[c.get('decision_id')].append(c)
    for r in raw:
        if r.get('failed_try') is None:
            by_raw[r.get('call_id') or (r.get('decision_id'), r.get('attempt'))].append(r)
    for e in events:
        data = e.get('data') or {}
        if e.get('kind') != 'thought' or data.get('seat') is None:
            continue
        seat = data['seat']
        if seat >= len(ids) or 'bot' in (specs.get(ids[seat]) or {}):
            continue
        x = out.setdefault(seat, {'decisions': 0, 'linked': 0, 'verified': 0, 'opaque': 0, 'problems': []})
        x['decisions'] += 1
        did = data.get('decision_id')
        if not did:
            continue
        mine = sorted(by_call.get(did) or [], key=lambda c: c.get('attempt') or 0)
        why = []
        if did not in acted:
            why.append('no action carries its decision id')
        if not mine:
            why.append('no call record carries its decision id')
        if why:
            x['problems'].append(f'hand {hand} decision {did}: ' + '; '.join(why))
            continue
        x['linked'] += 1
        got = None if replayed is None else replayed.get(did, replayed.get('*'))
        if replayed is not None and got != []:
            x['problems'].append(f'hand {hand} decision {did}: inconsistent: '
                                 + ('; '.join(got[:2]) if got else 'the replay of its hand did not reach it'))
            continue
        last = mine[-1]
        audit = last.get('audit')
        has_raw = bool(by_raw.get(last.get('call_id') or (did, last.get('attempt'))))
        if audit == 'opaque':
            x['opaque'] += 1
        elif audit == 'verified' and has_raw and data.get('audit') in (None, 'verified'):
            x['verified'] += 1
        else:
            x['problems'].append(f'hand {hand} decision {did}: accepted call audit {audit!r}'
                                 + ('' if has_raw else ', no raw output row'))
    return out


def _replayed(replay, cfg, h, evs, calls) -> dict | None:
    """`runner.replay_decisions` of one hand, or None when there is nothing to replay against (no replayer, no
    events, or a record from before decisions were linked); a replayer that fails says so for every decision."""
    if replay is None or not evs or not any((e.get('data') or {}).get('decision_id') for e in evs):
        return None
    try:
        return replay(cfg, h, evs, calls)
    except Exception as e:
        return {'*': [f'the replay failed: {e!r}'[:200]]}


def verify_play(d: Path, cfg: dict, hands: dict) -> dict:
    """{hand: {'ok', 'errors', 'digest', 'digest_linked', 'audit'}} for every recorded hand of one match folder,
    each checked with `verify_hand` against the seed and button the config assigns it (not the ones the hand
    records), with its rows from events.jsonl when the folder has one. `digest` hashes the hand record and those
    rows (evidence version 1); `digest_linked` also that hand's call records and raw backend output rows
    (version 2: what an official decision is bound to); `audit` is `linked_audit` of the hand. Cached by the
    folder's file sizes and times."""
    d = Path(d)
    key = (str(d.resolve()), _stamp(d))
    hit = _VERIFIED.get(key)
    if hit is not None:
        return hit
    seeds, n = cfg.get('seeds') or [], len(cfg.get('players') or []) or 2
    events = _rows_by_hand(d / 'events.jsonl', 'episode')
    calls = _rows_by_hand(d / 'calls.jsonl', 'hand') or {}
    raw = _rows_by_hand(d / 'raw.jsonl', 'hand') or {}
    try:
        from .runner import replay_decisions
    except ImportError:
        replay_decisions = None
    out = {}
    for k, h in hands.items():
        evs = None if events is None else events.get(k, [])
        if not isinstance(k, int) or k >= len(seeds):
            v = {'ok': False, 'errors': [f'hand {k} has no seed in the config']}
        else:
            v = verify_hand(cfg.get('game') or {}, seeds[k], k % n, h, evs)
        blob = json.dumps({'hand': h, 'events': evs}, sort_keys=True, default=str)
        linked = json.dumps({'hand': h, 'events': evs, 'calls': calls.get(k, []), 'raw': raw.get(k, [])},
                            sort_keys=True, default=str)
        out[k] = {**v, 'digest': hashlib.sha256(blob.encode()).hexdigest()[:24],
                  'digest_linked': hashlib.sha256(linked.encode()).hexdigest()[:24],
                  'audit': linked_audit(cfg, k, evs or [], calls.get(k, []), raw.get(k, []), _replayed(
                      replay_decisions, cfg, h, evs, calls.get(k, [])))}
    if len(_VERIFIED) > 4096:
        _VERIFIED.clear()
    _VERIFIED[key] = out
    return out


EVIDENCE_VERSION = 2        # 1: hands and events; 2: also their call records and raw backend output


def evidence_hash(deals, version: int = EVIDENCE_VERSION) -> str | None:
    """The hash of the verified records behind a set of duplicate deals (each deal's match folder, index and the
    digests of its two plays: version 2 covers hands, events, call records and raw output, version 1 hands and
    events): what an official decision is bound to. None when there are none, or when any deal carries no
    verified evidence."""
    items = []
    for d in deals:
        ev = d.get('evidence_linked' if version >= 2 else 'evidence')
        if not ev:
            return None
        items.append(f'{Path(str(d["matchup"])).name}|{d["deal"]}|{ev}')
    if not items:
        return None
    return hashlib.sha256('\n'.join(sorted(items)).encode()).hexdigest()[:24]


def _rows(p: Path) -> list:
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def read_conversation(mdir) -> dict | None:
    """The one reader of a conversation's records (its a-first and b-first plays), for the ratings, the history,
    the style statistics and the viewer: which hands are results and which are not. {'configs': (a-first,
    b-first), 'hands': (committed hands of each play), 'verified' (hand indices both plays committed, that pass
    `check_pair`, and whose two plays replay from their seeds through `verify_play`: the only results), 'checks'
    (each play's `verify_play`), 'rejected' ([{'hand' (None: the whole conversation), 'reason'}]), 'provisional'
    ({'a-first': [...], 'b-first': [...]}: hands written but not yet results: past the play's commit point, or
    committed while the other play has not finished that hand), 'status' ({side: {hand: 'verified' |
    'provisional' | 'rejected'}} for every hand either play wrote), 'reasons' ({hand (None: the whole
    conversation): why it was rejected}), 'totals' ({'deals', 'net_a'}: a's chips over the verified deals, both
    plays), 'commit' (per play: what its commit point comes from: state.json, or legacy, see `_hands`)}. None when the folder holds no a-first config."""
    mdir = Path(mdir)
    a_dir, b_dir = mdir / 'a-first', mdir / 'b-first'
    ca = _json(a_dir / 'config.json')
    if not ca:
        return None
    cb = _json(b_dir / 'config.json') if (b_dir / 'config.json').exists() else None
    ha, hb = _hands(a_dir), _hands(b_dir) if cb else {}
    unreadable = cb is None and (b_dir / 'config.json').exists()
    written = {side: {h.get('hand') for h in _rows(d / 'hands.jsonl') if isinstance(h, dict)}
               for side, d in (('a-first', a_dir), ('b-first', b_dir))}
    out = {'configs': (ca, cb), 'hands': (ha, hb), 'verified': [], 'checks': ({}, {}), 'rejected': [],
           'provisional': {}}
    if cb:
        good, rejected = check_pair(ca, cb, ha, hb)
        va = verify_play(a_dir, ca, {k: ha[k] for k in good}) if good else {}
        vb = verify_play(b_dir, cb, {k: hb[k] for k in good}) if good else {}
        for k in good:
            bad = [f'{side} hand {k} does not replay from its seed: {"; ".join(v[k]["errors"][:2])}'
                   for side, v in (('a-first', va), ('b-first', vb)) if not v[k]['ok']]
            if bad:
                rejected.append({'hand': k, 'reason': '; '.join(bad)})
            else:
                out['verified'].append(k)
        out.update(checks=(va, vb), rejected=rejected)
    elif unreadable:
        out['rejected'] = [{'hand': None, 'reason': 'b-first has no readable config'}]
    whole = any(x['hand'] is None for x in out['rejected'])
    bad = {x['hand'] for x in out['rejected']}
    for side, mine, other in (('a-first', ha, hb), ('b-first', hb, ha)):
        out['provisional'][side] = sorted(k for k in written[side] if isinstance(k, int) and not whole
                                          and k not in out['verified'] and k not in bad
                                          and (k not in mine or k not in other))
    out['commit'] = {side: 'state.json' if (d / 'state.json').exists() else
                     'legacy: no state.json, so every written hand is taken as committed and counts only if it verifies'
                     for side, d in (('a-first', a_dir), ('b-first', b_dir))}
    good = set(out['verified'])
    out['status'] = {side: {k: 'verified' if k in good else 'rejected' if whole or k in bad else 'provisional'
                            for k in sorted(x for x in written[side] if isinstance(x, int))}
                     for side in ('a-first', 'b-first')}
    out['reasons'] = {x['hand']: x['reason'] for x in out['rejected']}
    out['totals'] = {'deals': len(good), 'net_a': sum(ha[k]['net'][0] + hb[k]['net'][1] for k in out['verified'])}
    return out


def verified_hands(play_dir) -> dict:
    """For one play folder (a-first or b-first): {'verified': set of hand indices that are results,
    'provisional': [...], 'rejected': [...]} from `read_conversation` of its conversation."""
    play_dir = Path(play_dir)
    side = play_dir.name
    conv = read_conversation(play_dir.parent) if side in ('a-first', 'b-first') else None
    if not conv:
        return {'verified': set(), 'provisional': [], 'rejected': []}
    return {'verified': set(conv['verified']), 'provisional': conv['provisional'].get(side, []),
            'rejected': sorted({x['hand'] for x in conv['rejected'] if x['hand'] is not None})
            if not any(x['hand'] is None for x in conv['rejected']) else 'all'}


def load_deals(root, report: list | None = None) -> list[dict]:
    """[{'a', 'b', 'matchup' (the conversation), 'deal', 'net_a' (chips, both hands), 'bb', 'seed', 'fp',
    'planned', 'slate', 'round', 'replicate', 'evidence'}] for every duplicate deal under root that
    `read_conversation` verifies; what it rejects is appended to `report` as {'matchup', 'hand', 'reason'}.
    `evidence` and `evidence_linked` are the digests of the two verified plays (versions 1 and 2), `audit` per
    player the linked audit of its decisions in both plays."""
    root = Path(root)
    deals = []
    for cfg_path in sorted(root.rglob('a-first/config.json')):
        mdir = cfg_path.parent.parent
        if not (mdir / 'b-first' / 'config.json').exists():
            continue
        conv = read_conversation(mdir)
        if not conv:
            continue
        ca, cb = conv['configs']
        ha, hb = conv['hands']
        va, vb = conv['checks']
        game = ca.get('game', {})
        bb = game.get('big_unit') or game.get('bb') or 1
        seeds = ca.get('seeds', [])
        n = len(ca['players'])
        a, b = ca['players'][0], ca['players'][1]
        name = str(mdir.relative_to(root)) if mdir != root else mdir.name
        rnd, rep = _round_of(mdir.name, ca)
        if report is not None:
            report += [{'matchup': name, 'a': a, 'b': b, **x}
                       for x in sorted(conv['rejected'], key=lambda x: (x['hand'] is not None, x['hand'] or 0))]
        for k in conv['verified']:
            seed = seeds[k] if k < len(seeds) else None
            audit = {}
            for v, ids in ((va, ca['players']), (vb, cb['players'])):
                for seat, x in (v[k].get('audit') or {}).items():
                    y = audit.setdefault(ids[seat], {'decisions': 0, 'linked': 0, 'verified': 0, 'opaque': 0, 'problems': []})
                    for f in ('decisions', 'linked', 'verified', 'opaque'):
                        y[f] += x[f]
                    y['problems'] += x['problems']
            deals.append({'a': a, 'b': b, 'matchup': name, 'deal': k, 'bb': bb, 'seed': seed,
                          'fp': fingerprint(game, seed, k % n), 'planned': len(seeds),
                          'slate': seeds[0] if seeds else name, 'round': rnd, 'replicate': rep,
                          'net_a': ha[k]['net'][0] + hb[k]['net'][1],
                          'evidence': hashlib.sha256(f'{va[k]["digest"]}|{vb[k]["digest"]}'.encode()).hexdigest()[:24],
                          'evidence_linked': hashlib.sha256(
                              f'{va[k]["digest_linked"]}|{vb[k]["digest_linked"]}'.encode()).hexdigest()[:24],
                          'audit': audit})
    return deals


def pairing_data(root, folders, a) -> dict:
    """What `adjudicate` judges for one pairing, from its match folders (relative to root) in play order:
    {'xs' (a's bb/100 per validated, verified duplicate deal), 'groups' (each deal's conversation), 'fps' (deal
    fingerprints), 'conversations', 'rejected' ([{'matchup', 'hand', 'reason'}] from `check_pair` and the
    record verifier), 'evidence_hash' (`evidence_hash` of those deals, version 2), 'evidence_hash_v1',
    'evidence_version', 'evidence_deals', 'linked_audit' (`linked_status` per player)}."""
    root = Path(root)
    xs, groups, fps, rejected, kept = [], [], [], [], []
    for f in folders:
        report = []
        if not (root / f).is_dir():
            rejected.append({'matchup': f, 'hand': None, 'reason': 'the match folder is missing'})
            continue
        found = load_deals(root / f, report)
        if not found and not report:
            report.append({'hand': None, 'reason': 'no duplicate deal of this conversation has finished'})
        for d in found:
            sign = 1 if d['a'] == a else -1
            xs.append(sign * d['net_a'] / d['bb'] / 2 * 100)
            groups.append(f)
            fps.append(d['fp'])
            kept.append({**d, 'matchup': f})
        rejected += [{**x, 'matchup': f} for x in report]
    return {'a': a, 'xs': xs, 'groups': groups, 'fps': fps, 'conversations': list(folders), 'rejected': rejected,
            'evidence_hash': evidence_hash(kept), 'evidence_hash_v1': evidence_hash(kept, 1),
            'evidence_version': EVIDENCE_VERSION, 'evidence_deals': len(kept), 'linked_audit': linked_status(kept)}


def linked_status(deals) -> dict:
    """Per player over these deals, the audit derived from decision-linked records: {'status', 'decisions',
    'linked', 'verified', 'opaque', 'problems'}; status 'complete' (every decision linked and verified),
    'opaque' (linked, some from a backend with no record), 'incomplete', 'legacy' (records from before decision
    ids, none linked: the runner's own facts are all there is) or 'bot'."""
    tot = {}
    for d in deals:
        for p, x in (d.get('audit') or {}).items():
            y = tot.setdefault(p, {'decisions': 0, 'linked': 0, 'verified': 0, 'opaque': 0, 'problems': []})
            for f in ('decisions', 'linked', 'verified', 'opaque'):
                y[f] += x[f]
            y['problems'] += x['problems']
    for p, y in tot.items():
        if not y['decisions']:
            y['status'] = 'bot'
        elif not y['linked'] and not y['problems']:
            y['status'] = 'legacy'
        elif y['linked'] == y['decisions'] == y['verified'] + y['opaque'] and not y['problems']:
            y['status'] = 'opaque' if y['opaque'] else 'complete'
        else:
            y['status'] = 'incomplete'
        if not y['problems'] and y['linked'] < y['decisions'] and y['status'] != 'legacy':
            y['problems'].append(f'{y["decisions"] - y["linked"]} decisions carry no decision id')
        y['problems'] = y['problems'][:5]
    return tot


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
    """Names what sets a design's official test: its bound, bettor and bar (`calibration`), and every constant of
    the test that shapes its operating characteristics."""
    if not design or not design.get('calibration'):
        return None
    key = {'calibration': design['calibration'], 'alpha_pairing': design.get('alpha_pairing'),
           'equiv_bb100': EQUIV_BB100, 'bettors': [list(b) for b in BETTORS]}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


DESIGN_INPUTS = ('alpha', 'round_deals', 'block', 'replicates', 'max_rounds', 'target_edge_bb100', 'prior_sd_bb100',
                 'assumed_icc', 'assumed_ar1', 'budget_deals', 'n_pairings', 'allow_unattested', 'allow_opaque',
                 'outside_calibration', 'track', 'strict_identity')


def _inputs(d: dict) -> dict:
    """What a design was chosen to be (not what this code computes from it), for comparing with a registration."""
    d = d or {}
    defaults = {'block': BLOCK, 'replicates': 1, 'target_edge_bb100': TARGET_EDGE, 'prior_sd_bb100': PRIOR_SD,
                'assumed_icc': ASSUMED_ICC, 'assumed_ar1': ASSUMED_AR1, 'budget_deals': None,
                'allow_unattested': list(ALLOW_UNATTESTED), 'allow_opaque': [], 'outside_calibration': 'exclude',
                'track': 'official', 'strict_identity': False}
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
    first, changes, seen = None, [], defaultdict(set)
    for cfg_path in Path(root).rglob('config.json'):
        if cfg_path.parent.name not in ('a-first', 'b-first'):
            continue
        cfg = _json(cfg_path) or {}
        state = _json(cfg_path.parent / 'state.json') or {}
        for vs in [cfg.get('cli_versions') or {}] + [(c.get('to') or {}) for c in cfg.get('cli_version_changes') or []] \
                + [r.get('cli_versions') or {} for r in (state.get('resumes') or []) if isinstance(r, dict)]:
            for exe, ver in vs.items():
                seen[exe].add(str(ver))
        if cfg.get('started') is not None:
            first = cfg['started'] if first is None else min(first, cfg['started'])
        changes += cfg.get('implementation_changes') or []
        changes += [{**c, 'files': [f'CLI {x}' for x in c.get('clis') or ['version']]} for c in cfg.get('cli_version_changes') or []]
    return {'first_started': first, 'implementation_changes': changes,
            'cli_versions_seen': {e: sorted(v) for e, v in seen.items()}}


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
    pinned = (reg or {}).get('cli_versions') or {}
    drift = sorted(f'{e}: {pinned[e]} -> {v}' for e, vs in (ev.get('cli_versions_seen') or {}).items() if e in pinned
                   for v in vs if v != str(pinned[e]))
    if drift and reg and reg.get('status') == 'registered':
        problems.append(f'the code changed during the run (a CLI version differs from the registered one: {"; ".join(drift)})')
    if not problems:
        return {**out, 'registered': True, 'status': 'valid'}
    status = 'missing' if not reg else 'void' if reg.get('status') == 'void' or any('code changed' in p for p in problems) else 'invalid'
    return {**out, 'status': status, 'reasons': problems}


# The preregistered design.

_COMPUTED = ('replicates', 'n_pairings', 'alpha_pairing', 'nominal_alpha', 'deals_needed', 'deals_planned',
             'max_deals', 'power_at_max', 'min_conversations', 'conversations_needed', 'power_scenarios',
             'underpowered', 'registered', 'sd_conversation_bb100', 'calibration', 'equivalence_alpha',
             'power_by_edge', 'power_basis', 'leader_chances', 'outcome_bound_bb100', 'official_test',
             'assumption_free', 'threshold', 'bet_cap', 'equiv_bb100', 'official', 'model_based', 'judge_test',
             'label', 'model_min_conversations', 'nominal_alpha_model')
_PLAN_KEYS = ('design', 'deals', 'until_decided', 'replicates', 'matchups', 'target_edge_bb100', 'prior_sd_bb100',
              'registered', 'game', 'game_config', 'track')


def _n_pairings(plan: dict) -> int:
    return max(1, len({frozenset(m) for m in plan.get('matchups') or []}))


def _n_players(plan: dict) -> int:
    return len({p for m in plan.get('matchups') or [] for p in m})


def normalize_design(plan: dict | None) -> dict | None:
    """The plan's sequential design with every default filled in: alpha, round_deals (deals per conversation),
    block, replicates (independent conversations per pairing per round), n_pairings and alpha_pairing
    (Bonferroni), the official test (`official_test`, `outcome_bound_bb100` from the plan's game settings, the
    bar `threshold` = 2 / alpha_pairing on each bettor's capital, `bet_cap`), min_conversations (the fewest
    conversations from which the official test could decide at all), and the preregistered target:
    target_edge_bb100 (default 50), prior_sd_bb100 (the per-deal spread, default 350) and the dependence
    assumed when planning (assumed_icc, assumed_ar1). max_rounds, unless the plan sets it, is the fewest rounds
    that give 80% power at the target under those assumptions (max_rounds_basis 'power'); `budget_deals` caps
    the deals per pairing (basis 'budget'), and then `underpowered` says whether that leaves less than 80%.
    power_scenarios gives the power at the target under optimistic, the assumed (moderate) and pessimistic
    dependence; power_at_max is the moderate one. The eligibility rules are part of the design: allow_unattested
    (default codex), allow_opaque (default none) and outside_calibration ('exclude', the default: a decided
    verdict whose checks failed never counts towards a decided leader; 'count'). `registered` is not computed
    here: it is False for a plan that declines registration or played an older stopping rule (`legacy`), and
    absent otherwise; only `registration_status` says whether a run is preregistered. The runner, the ratings
    and the viewer all use this one. None for a plan with no sequential design."""
    if not plan or not plan.get('deals') and not (plan.get('design') or {}).get('round_deals'):
        return None
    if plan.get('design') is None and not plan.get('until_decided'):
        return None
    key = json.dumps({k: plan.get(k) for k in _PLAN_KEYS if k != 'matchups'} | {'n': _n_pairings(plan),
                                                                                 'players': _n_players(plan)},
                     sort_keys=True, default=str)
    return json.loads(_normalized(key))


@lru_cache(maxsize=256)
def _normalized(key: str) -> str:
    plan = json.loads(key)
    given = {k: v for k, v in (plan.get('design') or {}).items() if k not in _COMPUTED}
    if plan.get('registered') is False or (plan.get('design') or {}).get('registered') is False:
        given['registered'] = False
    if plan.get('track') and 'track' not in given:
        given['track'] = plan['track']
    if plan.get('design') is not None:
        d = {**DEFAULT_DESIGN, 'round_deals': plan.get('deals') or DEFAULT_DESIGN['round_deals'], **given}
    else:
        d = {**DEFAULT_DESIGN, 'round_deals': plan['deals'],
             'max_rounds': max(1, plan['until_decided'] // plan['deals']), 'legacy': 'until_decided'}
    reps = int(plan.get('replicates') or (plan.get('design') or {}).get('replicates') or 1)
    bound = outcome_bound(plan.get("game") or "holdem", json.dumps(plan.get("game_config") or {}, sort_keys=True))
    return json.dumps(_complete(d, reps, plan['n'], plan.get('target_edge_bb100') or d.get('target_edge_bb100'),
                                plan.get('prior_sd_bb100') or d.get('prior_sd_bb100'), plan.get('players'), bound))


@lru_cache(maxsize=64)
def outcome_bound(game: str = 'holdem', game_config: dict | str | None = None) -> float:
    """The most a duplicate deal can move for one player, in bb/100: each of its two plays wins or loses at most
    one stack (stacks reset every hand), so |x| = |net over both plays| / bb / 2 x 100 <= 100 x stack / bb. Every
    conversation mean lies within it too, whatever its length. 10,000 at the default 200-chip stacks and 2-chip
    big blind."""
    from .games import get_game
    cfg = json.loads(game_config) if isinstance(game_config, str) else dict(game_config or {})
    g = get_game(game or 'holdem')(cfg)
    return 100.0 * float(g.stack) / float(g.bb)


def _bound_of(design: dict | None) -> float:
    return float((design or {}).get('outcome_bound_bb100') or outcome_bound())


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


def _need_rounds(base: dict, target: float) -> int | None:
    """The fewest rounds at which the official test decides for the better player with chance POWER at the
    target edge (moderate dependence); None when not even MAX_ROUNDS do."""
    curve = power_curve({**base, 'max_rounds': MAX_ROUNDS}, target)
    return next((r for r, p in enumerate(curve, 1) if p >= POWER), None)


def track_of(x: dict | None) -> str:
    """'official' or 'adaptation': a plan's, a design's or a verdict's track (official unless it says otherwise)."""
    x = x or {}
    return x.get('track') or (x.get('design') or {}).get('track') or 'official'


def _complete(d, reps, n_pairings, target, prior, players=None, bound=None) -> dict:
    if d.get('track') == 'adaptation':
        return _complete_adaptation(d, reps, n_pairings, target, prior, bound)
    R, B = int(d['round_deals']), int(d.get('block') or BLOCK)
    target, prior = float(target or TARGET_EDGE), float(prior or PRIOR_SD)
    icc, ar1 = float(d.get('assumed_icc', ASSUMED_ICC)), float(d.get('assumed_ar1', ASSUMED_AR1))
    ap = d['alpha'] / n_pairings
    bound = float(bound or outcome_bound())
    plan_sd = sd_conversation(prior, R, icc, ar1)
    base = {'round_deals': R, 'block': B, 'replicates': reps, 'alpha_pairing': ap, 'assumed_icc': icc,
            'assumed_ar1': ar1, 'prior_sd_bb100': prior, 'target_edge_bb100': target, 'outcome_bound_bb100': bound}
    need = _need_rounds(base, target)
    if d.get('max_rounds'):
        rounds, basis = int(d['max_rounds']), d.get('max_rounds_basis') or 'given'
    else:
        rounds, basis = need or MAX_ROUNDS, 'power'
    if d.get('budget_deals'):
        cap = max(1, int(d['budget_deals']) // (R * reps))
        if rounds > cap:
            rounds, basis = cap, 'budget'
    declined = d.get('registered') is False or d.get('legacy')
    full = {**base, 'max_rounds': rounds}
    power = []
    for name, i, a in scenarios(icc, ar1):
        x = procedure(full, target, i, a)
        power.append({'scenario': name, 'assumed_icc': i, 'assumed_ar1': a,
                      'sd_conversation_bb100': round(sd_conversation(prior, R, i, a), 1),
                      'power': x['counting'], 'power_decided': x['right'], 'conversations_used': x['conversations']})
    moderate = power[1]['power']
    by_edge = {str(e): procedure(full, float(e))['counting'] for e in POWER_EDGES}
    cap = R * reps * rounds
    out = {**d, 'track': 'official', 'official': True,
           'round_deals': R, 'max_rounds': rounds, 'max_rounds_basis': basis, 'block': B, 'replicates': reps,
           'n_pairings': n_pairings, 'alpha_pairing': ap,
           'official_test': {'name': OFFICIAL_TEST, 'assumption_free': True, 'bound_bb100': bound,
                             'bet_cap': BET_CAP, 'threshold': round(2 / ap, 3)},
           'assumption_free': True, 'outcome_bound_bb100': bound, 'threshold': round(2 / ap, 3), 'bet_cap': BET_CAP,
           'equiv_bb100': EQUIV_BB100, 'min_conversations': min_conversations(ap),
           **({'registered': False} if declined else {}), 'target_edge_bb100': target, 'prior_sd_bb100': prior,
           'allow_unattested': sorted(d.get('allow_unattested', ALLOW_UNATTESTED)),
           'allow_opaque': sorted(d.get('allow_opaque', ())),
           'outside_calibration': d.get('outside_calibration', 'exclude'),
           'strict_identity': bool(d.get('strict_identity', False)),
           'assumed_icc': icc, 'assumed_ar1': ar1, 'sd_conversation_bb100': round(plan_sd, 1),
           'conversations_needed': need * reps if need else None, 'deals_needed': need * reps * R if need else None,
           'deals_planned': cap, 'max_deals': cap, 'power_at_max': moderate, 'power_scenarios': power,
           'power_by_edge': by_edge,
           'power_basis': ('simulated through the official test (the betting confidence sequence, every look, as '
                           'the runner stops): the chance that the better player gets a decided verdict by the '
                           'last planned look, per pairing; its guarantee needs no assumption on the shape of the '
                           'results, so the dependence scenarios change only the power'),
           'underpowered': moderate < POWER,
           'calibration': {'test': OFFICIAL_TEST, 'bound_bb100': bound, 'bet_cap': BET_CAP,
                           'prior_var': BET_PRIOR_VAR, 'threshold': round(2 / ap, 6), 'equiv_bb100': EQUIV_BB100}}
    if players and players >= 2 and n_pairings == players * (players - 1) // 2:
        out['leader_chances'] = leader_chances(out, players)
    return out


def model_rounds_cap(reps: int) -> int:
    """The most rounds the model-based gate is calibrated for at `reps` conversations a round."""
    return max(1, min(MODEL_MAX_ROUNDS, MODEL_MAX_CONVERSATIONS // max(1, int(reps))))


def _complete_adaptation(d, reps, n_pairings, target, prior, bound=None) -> dict:
    """The adaptation track's design: long conversations (models remember the session), judged and stopped by
    the model-based gate (`_model_based`), powered by simulating that gate (`procedure_model_based`). max_rounds,
    unless given, is the fewest rounds that give 80% power at the target under the assumed dependence, within
    what the gate is calibrated for (`model_rounds_cap`); `underpowered` when that cap or a budget leaves less."""
    R, B = int(d['round_deals']), int(d.get('block') or BLOCK)
    if R < 2:
        raise ValueError('the adaptation track needs conversations of at least 2 deals (models remember the session)')
    target, prior = float(target or TARGET_EDGE), float(prior or PRIOR_SD)
    icc, ar1 = float(d.get('assumed_icc', ASSUMED_ICC)), float(d.get('assumed_ar1', ASSUMED_AR1))
    ap = d['alpha'] / n_pairings
    bound = float(bound or outcome_bound())
    cap = model_rounds_cap(reps)
    base = {'round_deals': R, 'block': B, 'replicates': reps, 'alpha_pairing': ap, 'assumed_icc': icc,
            'assumed_ar1': ar1, 'prior_sd_bb100': prior, 'target_edge_bb100': target,
            'outside_calibration': d.get('outside_calibration', 'exclude')}
    power_at = lambda rounds, edge=target, i=icc, a=ar1: procedure_model_based({**base, 'max_rounds': rounds}, edge, i, a)
    need = next((r for r in range(1, cap + 1) if power_at(r)['counting'] >= POWER), None)
    if d.get('max_rounds'):
        rounds, basis = min(cap, int(d['max_rounds'])), d.get('max_rounds_basis') or 'given'
    else:
        rounds, basis = need or cap, 'power'
    if d.get('budget_deals'):
        most = max(1, int(d['budget_deals']) // (R * reps))
        if rounds > most:
            rounds, basis = most, 'budget'
    power = []
    for name, i, a in scenarios(icc, ar1):
        x = power_at(rounds, target, i, a)
        power.append({'scenario': name, 'assumed_icc': i, 'assumed_ar1': a,
                      'sd_conversation_bb100': round(sd_conversation(prior, R, i, a), 1),
                      'power': x['counting'], 'power_decided': x['right'], 'conversations_used': x['conversations']})
    moderate = power[1]['power']
    a1, _, _, eligible = _calibrate(ap, rounds, reps)
    first = next((c for c, e in zip(_look_cs(rounds, reps), eligible) if e), None)
    deals = R * reps * rounds
    declined = d.get('registered') is False or d.get('legacy')
    return {**d, 'track': 'adaptation', 'official': False, 'model_based': True, 'label': ADAPTATION_LABEL,
            'round_deals': R, 'max_rounds': rounds, 'max_rounds_basis': basis, 'block': B, 'replicates': reps,
            'n_pairings': n_pairings, 'alpha_pairing': ap,
            'official_test': {'name': ADAPTATION_TEST, 'assumption_free': False, 'model_based': True,
                              'bound_bb100': bound},
            'judge_test': 'model_based', 'assumption_free': False, 'outcome_bound_bb100': bound,
            'threshold': round(2 / ap, 3), 'bet_cap': BET_CAP, 'equiv_bb100': EQUIV_BB100,
            'min_conversations': first or MIN_CONVERSATIONS, 'nominal_alpha_model': a1,
            **({'registered': False} if declined else {}), 'target_edge_bb100': target, 'prior_sd_bb100': prior,
            'allow_unattested': sorted(d.get('allow_unattested', ALLOW_UNATTESTED)),
            'allow_opaque': sorted(d.get('allow_opaque', ())),
            'outside_calibration': d.get('outside_calibration', 'exclude'),
            'strict_identity': bool(d.get('strict_identity', False)),
            'assumed_icc': icc, 'assumed_ar1': ar1, 'sd_conversation_bb100': round(sd_conversation(prior, R, icc, ar1), 1),
            'conversations_needed': need * reps if need else None, 'deals_needed': need * reps * R if need else None,
            'deals_planned': deals, 'max_deals': deals, 'power_at_max': moderate, 'power_scenarios': power,
            'power_by_edge': {str(e): power_at(rounds, float(e))['counting'] for e in POWER_EDGES},
            'power_basis': ('simulated through the model-based gate (its t arm against the calibrated bar, every '
                            'look, as the runner stops), under normal conversation means with the assumed '
                            'dependence: the chance that the better player gets a verdict whose diagnostics pass by '
                            'the last planned look; the gate\'s error rate itself holds only within its calibrated '
                            'family of shapes'),
            'underpowered': moderate < POWER,
            'calibration': {'test': 'model_based', 'family': list(FAMILY), 'family_share': FAMILY_SHARE,
                            'cal_sims': CAL_SIMS, 'flips': FLIPS, 'min_conversations': MIN_CONVERSATIONS,
                            'rounds': rounds, 'replicates': reps, 'nominal_alpha': a1,
                            'equiv_bb100': EQUIV_BB100}}


def unregistered(round_deals, n_pairings, replicates=1, bound=None) -> dict:
    """The design of a run that had none fixed before play: nothing in it is a test. `retrospective` is roughly
    what the official test would have needed for 80% power at the default target (moderate dependence),
    labelled as such."""
    R = int(round_deals or DEFAULT_DESIGN['round_deals'])
    ap = DEFAULT_DESIGN['alpha'] / max(1, n_pairings)
    bound = float(bound or outcome_bound())
    n = conversations_for(TARGET_EDGE, sd_conversation(PRIOR_SD, R, ASSUMED_ICC, ASSUMED_AR1), ap, bound)
    return {'registered': False, 'alpha': DEFAULT_DESIGN['alpha'], 'round_deals': R, 'replicates': int(replicates or 1),
            'block': BLOCK, 'n_pairings': n_pairings, 'min_conversations': min_conversations(ap),
            'outcome_bound_bb100': bound,
            'retrospective': {'target_edge_bb100': float(TARGET_EDGE), 'prior_sd_bb100': float(PRIOR_SD),
                              'assumed_icc': ASSUMED_ICC, 'assumed_ar1': ASSUMED_AR1, 'replicates': int(replicates or 1),
                              'conversations_needed': n, 'deals_needed': n * R, 'basis': 'approximate'}}


# The official test: a betting confidence sequence for bounded conversation means.

def _bet(Y, m, bound, state=None, prior_var=BET_PRIOR_VAR, cap=BET_CAP):
    """The log capital of the bettor against H: mean <= m after each observation of Y (rows: independent
    sequences, columns: conversations in play order, every value within +-bound), and its state to carry into
    the next columns. The stake on conversation i is the aGRAPA fraction (mu - m) / (var + (mu - m)^2) from the
    mean mu and variance var of the conversations before it (the variance starts from prior_var, worth one
    conversation), clipped to [0, cap / (bound + m)]: it uses only the past, and 1 + lambda (Y - m) >= 1 - cap
    for every Y >= -bound, so the capital is a nonnegative supermartingale under H for any distribution."""
    Y = np.asarray(Y, float)
    S, T = Y.shape
    m = np.broadcast_to(np.asarray(m, float).reshape(-1, 1) if np.ndim(m) else np.full((S, 1), float(m)), (S, 1))
    n0, s1, s2, lk = state if state is not None else (0, np.zeros(S), np.zeros(S), np.zeros(S))
    c1, c2 = s1[:, None] + np.cumsum(Y, 1), s2[:, None] + np.cumsum(Y * Y, 1)
    p1 = np.concatenate([s1[:, None], c1[:, :-1]], 1)
    p2 = np.concatenate([s2[:, None], c2[:, :-1]], 1)
    k = n0 + np.arange(T)
    mu = np.where(k > 0, p1 / np.maximum(k, 1), m)
    var = (prior_var + np.maximum(p2 - k * mu * mu, 0.0)) / (k + 1)
    gap = mu - m
    lam = np.clip(gap / (var + gap * gap), 0.0, cap / (bound + m))
    path = lk[:, None] + np.cumsum(np.log1p(lam * (Y - m)), 1)
    return path, (n0 + T, c1[:, -1], c2[:, -1], path[:, -1])


def drift(ys) -> dict:
    """Whether a pairing's edge held still: its conversation means in play order, the first half against the
    second (Welch t, two-sided). {'status' ('no drift detected' | 'drift detected' | 'unknown': fewer than 4
    conversations),
    'early', 'late', 'difference', 'p', 'conversations' ([early, late]), 'meaning'}. A diagnostic beside the
    verdict, not a test of it: the official verdict stays valid under drift, but then says only that the edge
    was on that side for part of the run."""
    y = np.asarray(ys, float)
    C = len(y)
    out = {'status': 'unknown', 'early': None, 'late': None, 'difference': None, 'p': None, 'conversations': None,
           'alpha': DRIFT_ALPHA, 'meaning': 'too few conversations to compare early and late play'}
    if C < 4:
        return out
    e, l = y[:C // 2], y[C // 2:]
    me, ml = float(e.mean()), float(l.mean())
    ve, vl = float(e.var(ddof=1)) / len(e), float(l.var(ddof=1)) / len(l)
    diff = ml - me
    if ve + vl <= 1e-12 * (1 + diff * diff):
        p = 0.0 if abs(diff) > 1e-9 else 1.0
    else:
        df = (ve + vl) ** 2 / ((ve * ve / (len(e) - 1) if ve else 0.0) + (vl * vl / (len(l) - 1) if vl else 0.0))
        p = min(1.0, 2 * t_sf(abs(diff) / math.sqrt(ve + vl), max(df, 1.0)))
    found = p < DRIFT_ALPHA
    out.update(status='drift detected' if found else 'no drift detected', early=round(me, 1), late=round(ml, 1),
               difference=round(diff, 1), p=round(p, 6), conversations=[len(e), len(l)],
               meaning=(f'the edge changed during play ({me:+.0f} bb/100 in the first {len(e)} conversations, '
                        f'{ml:+.0f} in the last {len(l)}): a verdict means a player was ahead for part of the run, '
                        f'not that the run\'s average favours it') if found else
               'no drift detected between early and late conversations; this is not evidence that the edge '
               'held still (drift that reverses within each half, or after play stopped, is not seen)')
    return out


def min_conversations(alpha_pairing: float, cap: float = BET_CAP) -> int:
    """The fewest conversations from which the official test could decide at all: the first stake is zero, and
    each later one multiplies the capital by at most 1 + cap (a conversation at the bound)."""
    return 1 + math.ceil(math.log(2 / alpha_pairing) / math.log1p(cap) - 1e-12)


BETTORS = (('a', 1, 0.0), ('b', -1, 0.0), ('lo', 1, -EQUIV_BB100), ('hi', -1, EQUIV_BB100))


def betting(ys, alpha_pairing: float, bound: float, looks=None) -> dict:
    """The official test on conversation means `ys` (play order): the running maximum of each bettor's log
    capital at each look (`looks`: conversations at each look; default every conversation), and the verdict at
    the first look where one reached log(2 / alpha_pairing): 'a' (the bettor against an edge <= 0), 'b' (against
    >= 0), 'even' (both bettors against the margins, <= -EQUIV_BB100 and >= +EQUIV_BB100), else None.
    {'verdict', 'look' (1-based), 'conversation' (where its bar was first reached), 'capital' (per bettor, at the
    last look), 'threshold'}."""
    y = np.clip(np.asarray(ys, float), -bound, bound)
    C = len(y)
    looks = list(range(1, C + 1)) if looks is None else [int(c) for c in looks]
    bar = math.log(2 / alpha_pairing)
    out = {'verdict': None, 'look': None, 'conversation': None, 'threshold': round(2 / alpha_pairing, 3),
           'capital': None}
    if not C:
        return out
    signs = np.array([s for _, s, _ in BETTORS], float)
    nulls = np.array([s * m for _, s, m in BETTORS], float)
    path, _ = _bet(signs[:, None] * y[None, :], nulls, bound)
    best = np.maximum.accumulate(path, 1)
    first = [int(np.argmax(best[i] >= bar)) + 1 if best[i, -1] >= bar else None for i in range(len(BETTORS))]
    for j, c in enumerate(looks):
        hit = {name: best[i, c - 1] >= bar for i, (name, _, _) in enumerate(BETTORS)}
        if hit['a'] or hit['b']:
            pick = 'a' if hit['a'] and (not hit['b'] or first[0] <= first[1]) else 'b'
            out.update(verdict=pick, look=j + 1, conversation=first[0 if pick == 'a' else 1])
            break
        if hit['lo'] and hit['hi']:
            out.update(verdict='even', look=j + 1, conversation=max(first[2], first[3]))
            break
    last = looks[-1] if looks else C
    out['capital'] = {name: round(float(np.exp(min(best[i, last - 1], 700))), 4) for i, (name, _, _) in enumerate(BETTORS)}
    return out


def confidence_sequence(ys, alpha_pairing: float, bound: float) -> list[float] | None:
    """The anytime-valid interval for the mean edge after the conversations `ys`: every m whose two bettors
    (against <= m and against >= m) have not reached 2 / alpha_pairing at any point so far. It covers the true
    mean at every look at once with chance at least 1 - alpha_pairing, whatever the distribution."""
    y = np.clip(np.asarray(ys, float), -bound, bound)
    if not len(y):
        return None
    bar = math.log(2 / alpha_pairing)

    def keep(grid):
        up, _ = _bet(np.broadcast_to(y, (len(grid), len(y))), grid, bound)
        down, _ = _bet(np.broadcast_to(-y, (len(grid), len(y))), -grid, bound)
        return up.max(1) < bar, down.max(1) < bar

    lim = bound * (1 - 1e-6)
    grid = np.linspace(-lim, lim, 401)
    ok_up, ok_down = keep(grid)
    if not ok_up.any() or not ok_down.any():
        return None
    i, j = int(np.argmax(ok_up)), int(len(grid) - 1 - np.argmax(ok_down[::-1]))
    fine = np.linspace(grid[max(0, i - 1)], grid[i], 201)
    lo = fine[int(np.argmax(keep(fine)[0]))]
    fine = np.linspace(grid[j], grid[min(len(grid) - 1, j + 1)], 201)
    d = keep(fine)[1]
    hi = fine[int(len(fine) - 1 - np.argmax(d[::-1]))]
    return [round(float(lo), 1), round(float(hi), 1)]


def conversations_for(edge: float, sd_conv: float, alpha_pairing: float, bound: float, power: float = POWER) -> int:
    """Approximately how many conversations the official test needs to decide an edge with chance `power`:
    with the stake at its cap lambda = BET_CAP / bound, the log capital drifts by g = lambda edge - lambda^2
    (sd^2 + edge^2) / 2 per conversation with spread lambda sd, and must reach log(2 / alpha_pairing); plus the
    first conversation, on which nothing is staked. Within a few percent of the simulation for spreads far
    below the bound."""
    lam, L, z = BET_CAP / bound, math.log(2 / alpha_pairing), _Z.inv_cdf(power)
    g = lam * abs(edge) - lam * lam * (sd_conv ** 2 + edge ** 2) / 2
    if g <= 0:
        return 10 ** 9
    s = lam * sd_conv
    root = (z * s + math.sqrt(z * z * s * s + 4 * g * L)) / (2 * g)
    return 1 + math.ceil(root * root)


PLAN_SIMS = 4_000           # simulated pairings behind each planning number of the official test
_TIMES: dict = {}


def _crossing_times(sd_conv: float, edge: float, alpha_pairing: float, bound: float, T: int, sims: int = PLAN_SIMS,
                    stream: int = 0, even: bool = True) -> dict:
    """For `sims` simulated pairings of conversation means N(edge, sd_conv^2) clipped to the bound, the
    conversation at which each bettor of the official test first reaches its bar (0: not within T), for
    looks after every conversation. Kept for reuse: one simulation answers every number of rounds and of
    replicates."""
    key = (round(float(sd_conv), 6), round(float(edge), 6), float(alpha_pairing), float(bound), int(sims), int(stream),
           bool(even))
    hit = _TIMES.get(key)
    if hit is not None and hit[0] >= T:
        return hit[1]
    names = [b for b in BETTORS if even or b[0] in ('a', 'b')]
    signs = np.repeat(np.array([s for _, s, _ in names], float), sims)[:, None]
    nulls = np.repeat(np.array([s * m for _, s, m in names], float), sims)
    first = np.zeros(len(names) * sims, np.int64)
    rng = np.random.default_rng([SIM_SEED, 47, int(stream), int(round(edge * 1e3)), int(round(sd_conv * 1e3))])
    bar, state, done, step = math.log(2 / alpha_pairing), None, 0, 128
    while done < T:
        n = min(step, T - done)
        y = np.clip(edge + sd_conv * rng.standard_normal((sims, n)), -bound, bound)
        path, state = _bet(signs * np.tile(y, (len(names), 1)), nulls, bound, state)
        hit_ = path >= bar
        new = (first == 0) & hit_.any(1)
        first[new] = done + hit_[new].argmax(1) + 1
        done += n
        if (first.reshape(len(names), sims)[:2] > 0).any(0).all():
            break
    times = {name: first[i * sims:(i + 1) * sims] for i, (name, _, _) in enumerate(names)}
    _TIMES[key] = (T if done >= T else 10 ** 9, times)
    return times


def _plan_verdicts(times: dict, reps: int, rounds: int):
    """(verdict per pairing: 1 'a', -1 'b', 2 'even', 0 undecided; conversations played) from the bettors'
    crossing times, with looks after every `reps` conversations and the runner stopping at the first verdict."""
    big = 10 ** 9
    t = {k: np.where(v > 0, v, big) for k, v in times.items()}
    look = {k: np.where(v < big, -(-v // reps), big) for k, v in t.items()}
    la, lb = look['a'], look['b']
    le = np.maximum(look['lo'], look['hi']) if 'lo' in look else np.full(la.shape, big)
    lw = np.minimum(la, lb)
    win = np.where((la < lb) | ((la == lb) & (t['a'] <= t['b'])), 1, -1)
    sign = np.where(lw <= le, win, 2)
    at = np.minimum(lw, le)
    sign = np.where(at <= rounds, sign, 0)
    return sign, np.where(at <= rounds, at, rounds) * reps


def procedure(design: dict, edge: float, icc: float | None = None, ar1: float | None = None, sd: float | None = None,
              sims: int = PLAN_SIMS, stream: int = 0, even: bool = True) -> dict:
    """What the official test does with one eligible pairing of this design whose true edge is `edge` bb/100
    (planning dependence icc and ar1, per-deal spread sd; the design's own by default), by simulation: the
    verdict at the look the runner stops at. Returns the chance of each outcome: 'counting' and 'right' (the
    better player decided: in the simulation the test's checks always pass, so every decided verdict counts),
    'wrong' (the worse one decided), 'even', 'undecided', and the average conversations played."""
    d = design
    R, reps, rounds = int(d['round_deals']), int(d.get('replicates') or 1), int(d['max_rounds'])
    icc = float(d.get('assumed_icc', ASSUMED_ICC) if icc is None else icc)
    ar1 = float(d.get('assumed_ar1', ASSUMED_AR1) if ar1 is None else ar1)
    sd = float(d.get('prior_sd_bb100') or PRIOR_SD) if sd is None else float(sd)
    sdc = sd_conversation(sd, R, icc, ar1)
    eq = even and abs(edge) < 2 * EQUIV_BB100
    times = _crossing_times(sdc, float(edge), float(d['alpha_pairing']), _bound_of(d), rounds * reps, sims, stream, eq)
    sign, used = _plan_verdicts(times, reps, rounds)
    better = 1 if edge >= 0 else -1
    right = (sign == better) if edge else np.zeros(len(sign), bool)
    wrong = (sign == -better) if edge else (np.abs(sign) == 1)
    n = float(len(sign))
    return {'edge_bb100': float(edge), 'assumed_icc': icc, 'assumed_ar1': ar1, 'prior_sd_bb100': sd,
            'sd_conversation_bb100': round(sdc, 1),
            'counting': round(float(right.sum()) / n, 4), 'right': round(float(right.sum()) / n, 4),
            'wrong': round(float(wrong.sum()) / n, 4), 'even': round(float((sign == 2).sum()) / n, 4),
            'even_counting': round(float((sign == 2).sum()) / n, 4),
            'undecided': round(float((sign == 0).sum()) / n, 4), 'conversations': round(float(used.mean()), 1),
            'sims': int(n)}


def power_curve(design: dict, edge: float, icc=None, ar1=None, sd=None) -> list[float]:
    """The official test's power at `edge` by the end of each round 1..max_rounds."""
    d = design
    R, reps, rounds = int(d['round_deals']), int(d.get('replicates') or 1), int(d['max_rounds'])
    icc = float(d.get('assumed_icc', ASSUMED_ICC) if icc is None else icc)
    ar1 = float(d.get('assumed_ar1', ASSUMED_AR1) if ar1 is None else ar1)
    sd = float(d.get('prior_sd_bb100') or PRIOR_SD) if sd is None else float(sd)
    times = _crossing_times(sd_conversation(sd, R, icc, ar1), float(edge), float(d['alpha_pairing']), _bound_of(d),
                            rounds * reps, even=abs(edge) < 2 * EQUIV_BB100)
    sign, used = _plan_verdicts(times, reps, rounds)
    better = 1 if edge >= 0 else -1
    at = used // reps
    return [round(float(((sign == better) & (at <= r)).mean()), 4) for r in range(1, rounds + 1)]


def length_tradeoff(design: dict, lengths=(1, 2, 5, 10, 25, 50)) -> list[dict]:
    """What the official test needs at the design's target, by conversation length (deals per conversation),
    under the design's assumed dependence: conversations and deals per pairing for 80% power (looks after every
    conversation). The bound, not the spread, sets the number of conversations, so short conversations need
    far fewer deals; long ones measure what a model learns about its opponent within a conversation."""
    target = float(design.get('target_edge_bb100') or TARGET_EDGE)
    out = []
    for R in lengths:
        sdc = sd_conversation(float(design.get('prior_sd_bb100') or PRIOR_SD), int(R),
                              float(design.get('assumed_icc', ASSUMED_ICC)), float(design.get('assumed_ar1', ASSUMED_AR1)))
        T = MAX_ROUNDS * 40
        t = _crossing_times(sdc, target, float(design['alpha_pairing']), _bound_of(design), T, even=False)
        right = np.where(t['a'] > 0, t['a'], 10 ** 9)
        wrong = np.where(t['b'] > 0, t['b'], 10 ** 9)
        hit = np.sort(np.where(right < wrong, right, 10 ** 9))
        n = int(hit[int(math.ceil(POWER * len(hit))) - 1])
        out.append({'round_deals': int(R), 'sd_conversation_bb100': round(sdc, 1),
                    'conversations': n if n < 10 ** 9 else None, 'deals': n * int(R) if n < 10 ** 9 else None})
    return out


# The model-based gate (round 6): secondary, never official.

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


@lru_cache(maxsize=4096)
def _small_p(i: int, reps: int, j: int, cut: float, side: int = 0):
    """(sequences, p-values) of look j's null t of shape FAMILY[i] whose p-value is below `cut`: two-sided
    (side 0) or one-sided towards `side`. Only these can set a calibrated level, so only they are kept."""
    t = _tcols(FAMILY[i], reps, 0.0, CAL_SIMS, i, j + 1)[j]
    df = (j + 1) * reps - 1
    if side:
        idx = np.nonzero(side * t > t_ppf(1 - cut, df))[0]
        return idx, _one_sided(side * t[idx], df)
    idx = np.nonzero(np.abs(t) > t_ppf(1 - cut / 2, df))[0]
    return idx, _p_two_sided(t[idx], df)


def _kth_lowest(i, reps, eligible, cut, k, side=0):
    lowest = np.ones(CAL_SIMS)
    for j, e in enumerate(eligible):
        if e:
            idx, p = _small_p(i, int(reps), j, cut, side)
            np.minimum.at(lowest, idx, p)
    return float(np.partition(lowest, k)[k])


def _levels(ap, rounds, reps, eligible):
    k = max(0, int(math.floor((ap - 2 * math.sqrt(ap * (1 - ap) / CAL_SIMS)) * CAL_SIMS)))
    cut = min(0.5, 20 * ap)
    levels = {shape: min(ap, _kth_lowest(i, reps, eligible, cut, k) * (1 - 1e-9)) for i, shape in enumerate(FAMILY)}
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


def _one_sided(t, df):
    """P(T >= t) for Student t with integer df, vectorized."""
    p = _p_two_sided(t, df) / 2
    return np.where(np.asarray(t) > 0, p, 1 - p)


@lru_cache(maxsize=None)
def equivalence_level(alpha_pairing: float, rounds: int, reps: int) -> float:
    """The two-sided nominal level of the repeated interval behind 'even' (each of its one-sided tests is at half
    of it): for every shape of the calibration family and its mirror image, at most alpha_pairing (less two Monte
    Carlo standard errors) of null sequences cross one chosen side at any look that can decide. A skewed shape
    puts most of its two-sided crossings on one side, so calibrating each side, not the pair, is what keeps a
    false 'even' at an edge of exactly +-EQUIV_BB100 within alpha_pairing on either side."""
    ap, rounds, reps = float(alpha_pairing), int(rounds), int(reps)
    eligible = _calibrate(ap, rounds, reps)[3]
    if not any(eligible):
        return 2 * ap
    k = max(0, int(math.floor((ap - 2 * math.sqrt(ap * (1 - ap) / CAL_SIMS)) * CAL_SIMS)))
    cut = min(0.5, 20 * ap)
    level = min(_kth_lowest(i, reps, eligible, cut, k, side) for i in range(len(FAMILY)) for side in (1, -1))
    return 2 * min(ap, level * (1 - 1e-9))


_P_CACHE: dict = {}


@lru_cache(maxsize=256)
def _block_factor(R: int, B: int, ar1: float):
    """(block sizes, a factor L with L L' the covariance of a conversation's block sums) for R deals of unit
    variance with AR(1) memory ar1, in consecutive blocks of B (blocks of 1 when there would be one block, as
    `decide` then uses every deal)."""
    B = B if -(-R // B) >= 2 else 1
    i = np.arange(R)
    S = np.power(float(ar1), np.abs(i[:, None] - i[None, :])) if ar1 else np.eye(R)
    M = np.zeros((-(-R // B), R))
    M[i // B, i] = 1
    cov = M @ S @ M.T
    return M.sum(1), np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))


def _procedure_cols(R, B, sd, icc, ar1, edge, reps, sims, looks, stream=0):
    """What `decide` and `assumptions` see at looks 1..`looks` of simulated pairings, `reps` conversations of R
    deals a look: per look, the mean edge over the conversations, its standard error and the estimated
    conversation share, as float32 columns. Deals are normal with spread sd, a share icc of their variance a
    normal conversation effect, AR(1) memory ar1 inside a conversation and a true edge; block sums are drawn
    from their exact joint distribution, so the share estimate is the one `decide` computes."""
    key = (int(R), int(B), round(float(sd), 6), round(float(icc), 6), round(float(ar1), 6), round(float(edge), 6),
           int(reps), int(sims), int(stream))
    c = _P_CACHE.get(key)
    if c is None:
        c = _P_CACHE[key] = {'s1': np.zeros(sims), 's2': np.zeros(sims), 'sw': np.zeros(sims), 'cols': []}
    sizes, L = _block_factor(int(R), int(B), float(ar1))
    card, tau = float(sd) * math.sqrt(1 - icc), float(sd) * math.sqrt(icc)
    K = len(sizes)
    while len(c['cols']) < looks:
        j = len(c['cols'])
        rng = np.random.default_rng([SIM_SEED, 31, int(stream), j, *[int(round(v * 1e6)) for v in key[:6]], int(reps)])
        sums = card * rng.standard_normal((sims, reps, K)) @ L.T
        sums += sizes * (edge + tau * rng.standard_normal((sims, reps, 1)))
        m = sums.sum(-1) / R
        w = K / (K - 1) * ((sums - sizes * m[..., None]) ** 2).sum(-1) / R ** 2 if K > 1 else np.full(m.shape, np.nan)
        c['s1'] += m.sum(1)
        c['s2'] += (m * m).sum(1)
        c['sw'] += w.sum(1)
        C = (j + 1) * reps
        if C < 2:
            c['cols'].append((np.zeros(sims, np.float32), np.zeros(sims, np.float32), np.full(sims, np.nan, np.float32)))
            continue
        mean = c['s1'] / C
        var = np.maximum((c['s2'] - C * mean * mean) / (C - 1), 0.0)
        share = np.where(var > 0, np.maximum(0.0, 1 - c['sw'] / C / np.where(var > 0, var, 1.0)), np.nan)
        c['cols'].append((mean.astype(np.float32), np.sqrt(var / C).astype(np.float32),
                          np.round(share, 2).astype(np.float32)))
    return c['cols'][:looks]


def procedure_model_based(design: dict, edge: float, icc: float | None = None, ar1: float | None = None,
                          sd: float | None = None, sims: int = 20_000, stream: int = 0, even: bool = True) -> dict:
    """The round 6 planner, kept to compare costs: what the model-based gate would do with one pairing of this
    design (it needs `max_rounds` and `alpha_pairing`) whose true edge is `edge` bb/100, by simulation: the
    verdict at the look the runner would stop at, from the t arm against its calibrated bar (the sign-flip arm
    taken to agree whenever the t arm crosses), its 'even' test, its minimum number of conversations, and the
    diagnostics it needed to count (conversation share within FAMILY_SHARE). Returns the chance of each
    outcome: 'counting' (the better player decided, and it counts towards a decided rank or leader), 'right'
    (the better player decided, whatever the diagnostics), 'wrong' (the worse one decided), 'even',
    'even_counting', 'undecided', and the average conversations played."""
    d = design
    R, B, reps, rounds = int(d['round_deals']), int(d.get('block') or BLOCK), int(d.get('replicates') or 1), int(d['max_rounds'])
    icc = float(d.get('assumed_icc', ASSUMED_ICC) if icc is None else icc)
    ar1 = float(d.get('assumed_ar1', ASSUMED_AR1) if ar1 is None else ar1)
    sd = float(d.get('prior_sd_bb100') or PRIOR_SD) if sd is None else float(sd)
    cols = _procedure_cols(R, B, sd, icc, ar1, float(edge), reps, sims, rounds, stream)
    sign, counts, used = verdicts_of(cols, d, even)
    better = 1 if edge >= 0 else -1
    right = (sign == better) if edge else np.zeros(sims, bool)
    wrong = (sign == -better) if edge else (np.abs(sign) == 1)
    n = float(sims)
    return {'edge_bb100': float(edge), 'assumed_icc': icc, 'assumed_ar1': ar1, 'prior_sd_bb100': sd,
            'counting': round(float((right & counts).sum()) / n, 4),
            'right': round(float(right.sum()) / n, 4), 'wrong': round(float(wrong.sum()) / n, 4),
            'even': round(float((sign == 2).sum()) / n, 4),
            'even_counting': round(float(((sign == 2) & counts).sum()) / n, 4),
            'undecided': round(float((sign == 0).sum()) / n, 4), 'conversations': round(float(used.mean()), 1),
            'sims': int(sims)}


def verdicts_of(cols, design: dict, even: bool = True):
    """The round 6 planner's model of its gate on an eligible pairing, from per-look statistics (`cols`: per look
    the conversations' mean edge, its standard error and the conversation share, as `_procedure_cols` gives
    them): (verdict per sequence: 1 'a', -1 'b', 2 'even', 0 undecided; whether it counts towards a decided rank
    or leader; conversations played). Looks run in order and the first decision stops the pairing, as the runner
    does; a look decides only with enough conversations and a spread; the t arm against its calibrated bar
    stands for both arms."""
    d = design
    reps, rounds, ap = int(d.get('replicates') or 1), int(d['max_rounds']), float(d['alpha_pairing'])
    a1, _, _, eligible = _calibrate(ap, rounds, reps)
    aeq = equivalence_level(ap, rounds, reps) if even else 0.0
    count_outside = d.get('outside_calibration') == 'count'
    sims = len(cols[0][0])
    sign = np.zeros(sims, int)
    counts = np.zeros(sims, bool)
    used = np.full(sims, rounds * reps, float)
    for j, (m, se, share) in enumerate(cols[:rounds]):
        C = (j + 1) * reps
        if not eligible[j] or C < MIN_CONVERSATIONS:
            continue
        m, se, share = np.asarray(m, float), np.asarray(se, float), np.asarray(share, float)
        live = sign == 0
        ok = live & (se > 1e-9 * (1 + np.abs(m)))
        t = np.where(ok, m / np.where(ok, se, 1.0), 0.0)
        win = ok & (np.abs(t) >= t_ppf(1 - a1 / 2, C - 1))
        eq = np.zeros(sims, bool)
        if aeq > 0:
            q = t_ppf(1 - aeq / 2, C - 1)
            eq = ok & ~win & (m - q * se > -EQUIV_BB100) & (m + q * se < EQUIV_BB100)
        fine = (share <= FAMILY_SHARE) | (count_outside & ~np.isnan(share))
        sign[win] = np.where(t[win] > 0, 1, -1)
        sign[eq] = 2
        counts |= (win | eq) & fine
        used[win | eq] = C
    return sign, counts, used


def leader_chances(design: dict, players: int, icc=None, ar1=None) -> list[dict]:
    """The chance that a round robin of `players` under this design names the true best player its decided
    leader (a counting win over every other player), by edge scenario: 'one best' (it is better than each
    other player by the target edge, the rest equal), 'one best by twice the target', and 'ladder'
    (strengths evenly spaced, the target edge apart). Each of the leader's pairings is simulated through the
    official test (`procedure`) and treated as independent of the others. Pairings that share cards are
    positively correlated, which keeps the joint chance at or above that product (tests/test_calibration.py);
    anything that lowers a single pairing's power (a noisier game than planned) lowers it as a power."""
    T = float(design.get('target_edge_bb100') or TARGET_EDGE)
    if players < 2:
        return []
    p = lambda e: procedure(design, e, icc, ar1)['counting']
    one, two = p(T), p(2 * T)
    ladder = 1.0
    for k in range(1, players):
        ladder *= p(k * T) if k <= 4 else min(1.0, p(4 * T))
    return [{'scenario': 'one best', 'edge_bb100': T, 'p_leader': round(one ** (players - 1), 3)},
            {'scenario': 'one best by twice the target', 'edge_bb100': 2 * T, 'p_leader': round(two ** (players - 1), 3)},
            {'scenario': 'ladder', 'edge_bb100': T, 'p_leader': round(ladder, 3)}]


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


def _within(xs, labels, block) -> list[float]:
    """Per conversation of at least two deals, the variance of its mean that its own deals show: from its
    consecutive blocks of `block` deals (cluster-robust, blocks as clusters, so memory between nearby deals
    counts as the card noise it is), or s^2 / n when it has a single block."""
    by = {}
    for g, x in zip(labels, xs):
        by.setdefault(g, []).append(x)
    out = []
    for v in by.values():
        n = len(v)
        if n < 2:
            continue
        v = np.asarray(v, float)
        k = -(-n // block)
        if k >= 2:
            sz = np.minimum(block, n - block * np.arange(k)).astype(float)
            sm = np.add.reduceat(v, block * np.arange(k))
            out.append(k / (k - 1) * float(((sm - sz * v.mean()) ** 2).sum()) / n ** 2)
        else:
            out.append(float(v.var(ddof=1)) / n)
    return out


def _cr1(sizes, sums):
    """(mean, se): the deal-weighted mean and its cluster-robust (CR1) standard error, each cluster one unit."""
    G, N = len(sizes), sizes.sum()
    m = sums.sum() / N
    return float(m), math.sqrt(G / (G - 1) * float(((sums - sizes * m) ** 2).sum()) / N ** 2)


def decide(xs, design=None, n_pairings=1, looks_used=None, groups=None) -> dict:
    """The verdict on one pairing from a's result per duplicate deal (bb/100), at look `looks_used` of the
    design. `groups` labels each deal's conversation (its match folder), conversations in play order (round by
    round, replicate by replicate); without it all deals are one.

    The official verdict (`verdict`) is the betting confidence sequence's (`betting`) on the conversations'
    mean edges, each clipped to the game's bound (deals beyond it are counted in `out_of_bound`, which the
    adjudicator rejects): 'a' or 'b' at the first look where that player's bettor reached 2 / alpha_pairing
    (`decided_at_look`, `decided_at_conversation`), 'even' where both bettors against the margins did,
    'exhausted' (the last planned look, undecided), 'open' (undecided, looks remain, or no look yet), 'early'
    (a completed look with fewer conversations than any verdict could come from). `capital` is each bettor's
    best capital so far against the bar `official_test.threshold`; `cs` the anytime-valid interval for the edge
    (level `cs_level`), also given as `ci_official` ({interval, level, kind: CI_SEQUENTIAL}), the primary
    interval. `drift` compares early and late conversations (a diagnostic beside the verdict: under drift a
    verdict holds for the conditional null, see `drift`; `publication` withholds it from ranks and leaders).
    The other numbers describe all the deals given (`ci95` is an ordinary t interval, `ci_kind` CI_FIXED: not
    adjusted for the stopping rule, not a test); with one conversation `basis` is 'blocks' and it is
    conditional on it. `model_based` is the round 6
    gate on the same data, secondary and never official (None for designs it was never calibrated for)."""
    d = {**DEFAULT_DESIGN, 'max_rounds': 1, **(design or {})}
    xs = [float(x) for x in xs]
    n, B, rounds, reps = len(xs), int(d.get('block') or BLOCK), int(d['max_rounds']), int(d.get('replicates') or 1)
    R = int(d.get('round_deals') or n or 1)
    alpha_p = d['alpha'] / max(1, n_pairings)
    bound = _bound_of(d)
    if looks_used is None:
        looks_used = min(rounds, n // (R * reps))
    labels = list(groups) if groups is not None else ['.'] * n
    conv = {}
    for g, x in zip(labels, xs):
        c = conv.setdefault(g, [0, 0.0])
        c[0] += 1
        c[1] += x
    sizes = np.array([c[0] for c in conv.values()], float)
    sums = np.array([c[1] for c in conv.values()], float)
    bsizes, bsums = _chunks(xs, labels, B)
    C, G = len(conv), len(bsizes)
    need = min_conversations(alpha_p)
    out = {'deals': n, 'blocks': G, 'conversations': C, 'edge': round(sum(xs) / n, 1) if n else None, 'se': None,
           'ci95': None, 't': None, 'df': None, 'basis': None, 'verdict': 'early',
           'looks_used': looks_used, 'looks_planned': rounds, 'alpha_pairing': alpha_p, 'equiv_bb100': EQUIV_BB100,
           'min_conversations': need, 'decided_at_look': None, 'decided_at_conversation': None,
           'official_test': {'name': OFFICIAL_TEST, 'assumption_free': True, 'bound_bb100': bound, 'bet_cap': BET_CAP,
                             'threshold': round(2 / alpha_p, 3)},
           'assumption_free': True, 'bound_bb100': bound,
           'out_of_bound': sum(abs(x) > bound * (1 + 1e-9) for x in xs), 'capital': None, 'cs': None,
           'cs_level': round(1 - alpha_p, 6), 'ci_official': None, 'ci_kind': CI_FIXED,
           'conversation_share': None, 'sd_per_deal': None, 'reason': None,
           'model_based': None, 'power': {'deals_for_80pct': {str(e): None for e in POWER_EDGES}},
           'drift': drift(np.clip(sums / np.maximum(sizes, 1), -bound, bound))}
    if C >= 2:
        m, se = _cr1(sizes, sums)
        df, basis = C - 1, 'conversations'
    elif G >= 2:
        m, se = _cr1(bsizes, bsums)
        df, basis = G - 1, 'blocks'
    else:
        m = se = df = basis = None
    means = sums / np.maximum(sizes, 1)
    verdict, reason, at, at_c = None, None, None, None
    cs_ = [min(j * reps, C) for j in range(1, looks_used + 1)]
    if cs_:
        cs_[-1] = C
    test = betting(means, alpha_p, bound, cs_ or [C]) if C else None
    if looks_used < 1:
        verdict = 'open'
    elif C < need:
        verdict = 'early'
        reason = (f'{C} conversation{"s" if C != 1 else ""}: the official test cannot decide from fewer than {need} '
                  f'independent conversations (fresh match pairs) of the pairing')
    elif test['verdict']:
        verdict, at, at_c = test['verdict'], test['look'], test['conversation']
    else:
        verdict = 'exhausted' if looks_used >= rounds else 'open'
    if test:
        out['capital'] = {'a': test['capital']['a'], 'b': test['capital']['b'],
                          'even': min(test['capital']['lo'], test['capital']['hi'])}
        out['cs'] = confidence_sequence(means, alpha_p, bound)
        out['ci_official'] = _ci_official(out)
    out.update(verdict=verdict, reason=reason, decided_at_look=at, decided_at_conversation=at_c)
    if m is None:
        out.update(verdict='early', reason=out['reason'] or 'too few deals to estimate any spread')
        return out
    share = None
    if C >= 2:
        w = _within(xs, labels, B)
        between = float(np.var(sums / sizes, ddof=1))
        share = round(max(0.0, 1 - float(np.mean(w)) / between), 2) if between > 0 and w else None
    flat = se <= 1e-9 * (1 + abs(m))
    t = 0.0 if flat else m / se
    sd_conv = se * math.sqrt(C) if C >= 2 else None
    per = n / max(C, 1)
    out.update({'edge': round(m, 1), 'se': round(se, 1), 'ci95': [round(m - t_ppf(0.975, df) * se, 1),
                                                                  round(m + t_ppf(0.975, df) * se, 1)],
                't': round(t, 2), 'df': df, 'basis': basis, 'conversation_share': share,
                'sd_per_deal': round(se * math.sqrt(n), 1),
                'power': {'deals_for_80pct': {str(e): (math.ceil(conversations_for(e, sd_conv, alpha_p, bound) * per)
                                                       if sd_conv is not None else None) for e in POWER_EDGES},
                          'basis': 'official test, approximate, from the observed spread of conversation means'}})
    if d.get('model_based', True) and _model_cheap(rounds, reps, R) and C >= 2:
        out['model_based'] = _model_based(xs, labels, sizes, sums, m, se, df, C, share, d, alpha_p, looks_used,
                                          rounds, reps)
    return out


def _ci_official(v: dict) -> dict | None:
    if not v.get('cs'):
        return None
    return {'interval': v['cs'], 'level': v.get('cs_level'), 'kind': CI_SEQUENTIAL,
            'meaning': 'covers the mean edge at every look at once, whenever play stopped'}


def _model_cheap(rounds: int, reps: int, R: int) -> bool:
    return R >= 2 and rounds <= MODEL_MAX_ROUNDS and rounds * reps <= MODEL_MAX_CONVERSATIONS


MODEL_LABEL = ('model-based (round 6 gate: a conversation-level t and a sign-flip test, with bars calibrated for '
               'a family of shapes): secondary, descriptive, never official; its error rate is not guaranteed')


def _model_based(xs, labels, sizes, sums, m, se, df, C, share, d, alpha_p, looks_used, rounds, reps) -> dict:
    """The round 6 gate on the same deals: both arms (the t arm against its calibrated bar, the sign-flip arm
    with its error spent look by look) for a winner, TOST with both one-sided sign-flip tests for 'even', from
    MIN_CONVERSATIONS conversations, and its diagnostic against the calibrated family (conversation share)."""
    a1, cum, _, eligible = _calibrate(alpha_p, rounds, reps)
    aeq = equivalence_level(alpha_p, rounds, reps)
    need = next((c for c, e in zip(_look_cs(rounds, reps), eligible) if e), None) or MIN_CONVERSATIONS
    flat = lambda mean, s: s <= 1e-9 * (1 + abs(mean))
    can = C >= need
    bnd = t_ppf(1 - a1 / 2, df) if can else None
    beq = t_ppf(1 - aeq / 2, df) if can else None
    t = 0.0 if flat(m, se) else m / se
    verdict, reason, at, flip = None, None, None, None
    if looks_used < 1:
        verdict = 'open'
    elif not can:
        verdict = 'early'
        reason = f'{C} conversations: the model-based gate needs at least {need}'
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
                reason = 'the conversations show no spread at all, so the model-based gate cannot decide'
            elif abs(t) >= bnd and not flip:
                reason = 'the t arm crossed its bar but the sign-flip arm did not (both are needed)'
    status = 'unknown' if share is None or not can else 'supported' if share <= FAMILY_SHARE else 'unsupported'
    iv = lambda c: [round(m - c * se, 1), round(m + c * se, 1)]
    return {'verdict': verdict, 'label': MODEL_LABEL, 'official': False, 't': round(t, 2),
            'boundary': round(bnd, 3) if can else None, 'signflip': flip, 'decided_at_look': at,
            'ci_decision': iv(bnd) if can else None, 'equiv_ci': iv(beq) if can else None, 'nominal_alpha': a1,
            'min_conversations': need, 'conversation_share': share, 'reason': reason,
            'diagnostics': {'status': status, 'summary': MODEL_DIAGNOSTICS[status], 'limit': FAMILY_SHARE}}


MODEL_DIAGNOSTICS = {'supported': 'diagnostics passed; distribution assumptions unverified',
                     'unsupported': 'diagnostics failed: the data are outside the calibrated family',
                     'unknown': 'diagnostics incomplete: some could not be checked'}


# The adjudicator: the only function whose verdicts stop, record or decide anything.

DIAGNOSTICS = {'supported': 'checks passed: the guarantee holds for any distribution of results within the '
                            'game\'s bound, given independent conversations and the conditional null (see drift); reading '
                            'it as the better model also needs each name to have served one stable model',
               'unsupported': 'checks failed: deals are repeated across conversations, so they are not shown to be '
                              'independent',
               'unknown': 'checks incomplete: some could not be made'}


def assumptions(v: dict, fps=None, groups=None) -> dict:
    """What a verdict's error bound rests on, checked against the data: {'status': 'supported' | 'unsupported' |
    'unknown', 'summary', 'checks': {name: {'status', 'value', 'limit', 'meaning'}}}. The official test needs no
    assumption on the shape of the results, only independent conversations (fresh memories, by design) whose
    results the game bounds: `independent_deals` (no deal repeated across the pairing's conversations, which
    would make two of them share cards), `bounded` (no deal beyond the game's bound), `conversations` (enough of
    them for any verdict)."""
    checks = {}
    C, need = v.get('conversations') or 0, v.get('min_conversations') or MIN_CONVERSATIONS
    checks['conversations'] = {'status': 'supported' if C >= need else 'unknown', 'value': C, 'limit': need,
                               'meaning': 'independent conversations, at least the number any verdict needs'}
    bad = v.get('out_of_bound')
    checks['bounded'] = {'status': 'unknown' if bad is None else 'supported' if not bad else 'unsupported',
                         'value': bad, 'limit': 0,
                         'meaning': f'deals beyond the game\'s bound of +-{v.get("bound_bb100") or outcome_bound():,.0f} '
                                    f'bb/100 (more than the stacks allow)'}
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
    return {'status': status, 'summary': DIAGNOSTICS[status], 'checks': checks}


ATTESTATION = {'attested': 'the backend reported which model answered, and it is the requested one',
               'unattested': 'the backend did not report which model answered',
               'mismatch': 'the backend reported another model than the requested one',
               'bot': 'a bot: no model to attest', 'none': 'no reply was recorded'}


def attestation(backend: str | None, identity_match: str | None, allow=ALLOW_UNATTESTED) -> dict:
    """The one attestation policy, for games and the quiz alike: from an `identity_match` ('match' | 'mismatch' |
    'unattested' | 'bot' | None), {'status' ('attested' | 'unattested' | 'mismatch' | 'bot' | 'none'), 'allowed',
    'meaning', 'policy'}. Attested and bots are allowed; a mismatch and no reply never are; unattested is allowed
    only for a backend in `allow` (a design's `allow_unattested`; by default ALLOW_UNATTESTED, the backends that
    cannot report the model) and is shown as unattested wherever it is allowed."""
    status = {'match': 'attested', 'mismatch': 'mismatch', 'unattested': 'unattested', 'bot': 'bot'}.get(
        identity_match or '', 'none')
    allow = sorted(allow or ())
    allowed = status in ('attested', 'bot') or (status == 'unattested' and backend in allow)
    return {'status': status, 'allowed': allowed, 'meaning': ATTESTATION[status],
            'policy': f'unattested allowed for: {", ".join(allow) or "no backend"}'}


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
    wrong = [m for m in x.get('observed_models') or [] if identity_match(model, m) == 'mismatch']
    att = attestation(backend, 'mismatch' if wrong else _identity_of(x), design.get('allow_unattested') or ())
    if att['status'] == 'mismatch':
        out.append(f'{p}: answered by {", ".join(wrong or x.get("observed_models") or ["another model"])}, not the '
                   f'requested {model}')
    elif att['status'] == 'none':
        out.append(f'{p}: no replies recorded')
    elif not att['allowed']:
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
    (True / False / None: the official test's checks supported / unsupported / unknown; the name is kept from
    round 6), counts_toward_leader, standing, linked_audit (per player, the audit derived from the decision-linked
    call and raw records, when `pairing_data` found them),
    test_verdict (decide's verdict), identity_status and audit_status per player, judge_fingerprint (= judge),
    calibration_id, registered_judge and official (made by the registered judge under a valid registration),
    and what the result may claim (`publication`: decision_status, qualifier, identity_qualified, language,
    ci_official). A decided verdict under detected drift, or with an unattested player under the design's
    `strict_identity`, keeps its verdict (play stops on it) but does not count toward a leader.
    verdict is 'ineligible', a terminal verdict, whenever any rule fails: no valid registration, code changed
    during the run, a player's identity unresolved, mismatched or unattested where the design doesn't allow it,
    its turns not fully audited (by the runner's facts, and by the linked records where they exist), its
    protocol inconsistent, a duplicate pair that failed validation, or a deal beyond the game's bound."""
    data = dict(pairing_data)
    if 'xs' not in data and data.get('root') is not None:
        data = {**pairing_data_of(data), **{k: v for k, v in data.items() if k in ('a', 'b')}}
    xs, groups = list(data.get('xs') or []), data.get('groups')
    v = decide(xs, design, n_pairings, look, groups=groups)
    adaptation = track_of(design) == 'adaptation'
    if adaptation:
        v = _adaptation_verdict(v, look)
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
        reasons += [str(q) for q in (players.get(p) or {}).get('problems') or []]
    ver = elig.get('verification')
    if isinstance(ver, dict) and not ver.get('ok'):
        reasons += [str(q) for q in ver.get('problems') or []] or ['the runner could not verify the match records']
    rejected = data.get('rejected')
    if rejected is None:
        reasons.append('the duplicate pairs were not validated')
    elif rejected:
        reasons.append(f'{len(rejected)} duplicate pair{"s" if len(rejected) != 1 else ""} failed validation '
                       f'({rejected[0].get("reason")}{"; ..." if len(rejected) > 1 else ""})')
    if not xs:
        reasons.append('no verified duplicate deal: there is nothing to judge')
    elif not data.get('evidence_hash'):
        reasons.append('the hands were not verified against their seeds (no evidence hash), so they cannot count')
    if v.get('out_of_bound'):
        reasons.append(f'{v["out_of_bound"]} deal{"s" if v["out_of_bound"] != 1 else ""} beyond the game\'s bound of '
                       f'+-{v["bound_bb100"]:,.0f} bb/100, more than the stacks allow: the records cannot be right')
    linked = data.get('linked_audit') or {}
    for p in named:
        x = linked.get(p)
        if not x or (players.get(p) or {}).get('kind') == 'bot' or x.get('status') in ('bot', 'legacy'):
            continue
        allowed = ('complete', 'opaque') if (players.get(p) or {}).get('backend') in ((design or {}).get('allow_opaque') or ()) \
            else ('complete',)
        if x.get('status') not in allowed:
            reasons.append(f'{p}: the audit derived from its linked call records is {x.get("status")}'
                           + (f' ({"; ".join(x.get("problems")[:2])})' if x.get('problems') else ''))
    reasons = list(dict.fromkeys(reasons))
    a = assumptions(v, data.get('fps'), groups)
    if adaptation:
        a = _adaptation_assumptions(a, v.get('gate') or {})
    calibrated = {'supported': True, 'unsupported': False}.get(a['status'])
    eligible = not reasons
    test = v['verdict']
    count_outside = (design or {}).get('outside_calibration') == 'count'
    counts = eligible and test in DECIDED and (calibrated is True or (calibrated is False and count_outside))
    standing = None
    if eligible and test in DECIDED:
        standing = 'decided' if calibrated else ('decided (checks failed: conversations not shown independent)'
                                                 if calibrated is False else 'decided (checks incomplete)')
        if adaptation:
            standing = 'model-based (not official)' if calibrated else (
                'model-based (not official; diagnostics failed)' if calibrated is False else
                'model-based (not official; diagnostics incomplete)')
    judge = judge_fingerprint()
    cal = calibration_id(design)
    registered = bool(reg) and not registration_problems(reg, design) and reg.get('judge_fingerprint') == judge \
        and reg.get('calibration_id') == cal
    track = {'track': 'adaptation', 'model_based': True, 'model_based_counts': counts,
             'registered_record': registered, 'label': ADAPTATION_LABEL} if adaptation else \
        {'track': 'official', 'registered_record': registered}
    out = {**v, 'verdict': test if eligible else 'ineligible', 'test_verdict': test, 'eligible': eligible,
            'ineligible_reasons': reasons, 'assumptions': a, 'calibrated': calibrated,
            'counts_toward_leader': counts and not adaptation, 'standing': standing, **track,
            'identity_status': {p: (players.get(p) or {}).get('identity') for p in named},
            'audit_status': {p: (players.get(p) or {}).get('audit') for p in named},
            'identity_match': {p: _identity_of(players.get(p)) for p in named},
            'attestation': {p: attestation((players.get(p) or {}).get('backend'), _identity_of(players.get(p)),
                                           (design or {}).get('allow_unattested') or ()) for p in named},
            'judge_fingerprint': judge, 'judge': judge, 'calibration_id': cal,
            'registered_judge': (reg or {}).get('judge_fingerprint'), 'official': registered and not adaptation,
            'evidence_hash': data.get('evidence_hash'), 'judged_evidence_hash': data.get('evidence_hash'),
            'evidence_version': data.get('evidence_version', EVIDENCE_VERSION), 'evidence_deals': len(xs),
            'linked_audit': linked or None}
    out.update(publication(out, design, data.get('a'), data.get('b'), not adaptation))
    return out


ADAPTATION_TEXT = {'a': '{a} leans ahead on the model-based test', 'b': '{b} leans ahead on the model-based test',
                   'even': 'the model-based test leans toward even (within ±{e} bb/100)'}


def _side_text(v: dict, side: str, a, b) -> tuple[str, float, float]:
    dr = v.get('drift') or {}
    s = -1.0 if side == 'b' else 1.0
    return (b if side == 'b' else a) or side, s * float(dr.get('early') or 0.0), s * float(dr.get('late') or 0.0)


def _drift_qualifier(v: dict, a, b) -> str:
    word = lambda e: 'ahead' if e > 0 else 'behind' if e < 0 else 'level'
    tail = ('the edge changed during play, so this verdict counts toward no rank or leader: it needs a '
            'confirmation run on a stable model snapshot')
    if v.get('verdict') == 'even':
        _, e, l = _side_text(v, 'a', a, b)
        return f'called even, but {a or "a"} was {e:+.0f} bb/100 early and {l:+.0f} late; {tail}'
    who, e, l = _side_text(v, v.get('verdict'), a, b)
    how = f'{word(e)} early, {word(l)} late' if word(e) != word(l) else f'{word(e)} early and late, by different amounts'
    return f'decided with drift: {who} {how} ({e:+.0f} then {l:+.0f} bb/100); {tail}'


def publication(v: dict, design: dict | None, a=None, b=None, official_run: bool = True) -> dict:
    """What a verdict may claim when it is published, from the verdict itself: {'decision_status', 'standing',
    'counts_toward_leader', 'qualifier' (plain words, or None), 'identity_qualified', 'identity_qualifier',
    'unattested' (players whose backend did not report the model), 'language' ('confirmatory' | 'exploratory' |
    'descriptive'), 'ci_official', 'ci_kind'}. It only ever withholds: a decided verdict under detected drift is
    'decided_with_drift' (DRIFT_RULE), one with an unattested player under `strict_identity` is
    'decided_identity_unattested', and a counted result with an unattested player is `identity_qualified`. The
    adaptation track's verdicts are exploratory: they lean, they never win."""
    d = design or {}
    verdict = v.get('verdict')
    decided = verdict in DECIDED
    unattested = sorted(p for p, x in (v.get('attestation') or {}).items() if (x or {}).get('status') == 'unattested')
    counts = bool(v.get('counts_toward_leader'))
    out = {'unattested': unattested, 'ci_kind': CI_FIXED, 'identity_qualified': False, 'identity_qualifier': None,
           'qualifier': None, 'decision_status': verdict}
    if track_of(d) == 'adaptation' or v.get('track') == 'adaptation':
        e = v.get('equiv_bb100', EQUIV_BB100)
        out.update(language='exploratory', ci_official=None, counts_toward_leader=False,
                   decision_status='model-based lean' if decided else verdict,
                   qualifier=(ADAPTATION_TEXT[verdict].format(a=a or 'a', b=b or 'b', e=e)
                              + ' (exploratory: adaptation track, not official)') if decided else None)
        return out
    if not official_run or verdict == 'descriptive':
        out.update(language='descriptive', ci_official=None)
        return out
    ci = _ci_official(v)
    if ci and (v.get('drift') or {}).get('status') == 'drift detected':
        ci = {**ci, 'drift': True, 'meaning': 'drift detected: the interval is for one mean edge over the run, which '
                                              'the change of edge between early and late play contradicts'}
    out.update(language='confirmatory', ci_official=ci)
    if not decided:
        return out
    standing = v.get('standing')
    if (v.get('drift') or {}).get('status') == 'drift detected':
        status, counts, standing = 'decided_with_drift', False, 'decided_with_drift'
        out['qualifier'] = _drift_qualifier(v, a, b)
    elif unattested and d.get('strict_identity'):
        status, counts = 'decided_identity_unattested', False
        standing = 'decided (not counted: unattested identity under the strict policy)'
        out['qualifier'] = (f'not counted: {", ".join(unattested)} unattested (the backend did not report which model '
                            f'answered), and the design\'s strict identity policy counts only attested players')
    else:
        status = 'decided' if counts else 'decided_not_counted'
        if v.get('verdict') in ('a', 'b'):
            out['interpretation'] = STABLE_CONDITION
    if counts and unattested:
        out.update(identity_qualified=True,
                   identity_qualifier=f'{", ".join(unattested)} unattested: the backend did not report which model '
                                      f'answered, so this result rests on it serving the requested model (allowed by '
                                      f'the design\'s policy)')
    out.update(decision_status=status, standing=standing, counts_toward_leader=counts)
    return out


def _adaptation_verdict(v: dict, look: int) -> dict:
    """An adaptation-track verdict: the model-based gate's (`v['model_based']`, kept as `gate`), with the
    assumption-free test's verdict on the same data kept beside it as `assumption_free_verdict`."""
    gate = v.get('model_based') or {}
    verdict = gate.get('verdict') or ('open' if look < 1 else 'early')
    return {**v, 'verdict': verdict, 'assumption_free_verdict': v.get('verdict'), 'gate': gate or None,
            'model_based': True, 'decided_at_look': gate.get('decided_at_look'),
            'decided_at_conversation': None, 'min_conversations': gate.get('min_conversations') or v['min_conversations'],
            'reason': gate.get('reason') or (v.get('reason') if not gate else None),
            'official_test': {'name': ADAPTATION_TEST, 'assumption_free': False, 'model_based': True,
                              'bound_bb100': v.get('bound_bb100')},
            'assumption_free': False, 't': gate.get('t', v.get('t')), 'boundary': gate.get('boundary'),
            'signflip': gate.get('signflip'), 'ci_decision': gate.get('ci_decision'), 'equiv_ci': gate.get('equiv_ci')}


def _adaptation_assumptions(a: dict, gate: dict) -> dict:
    """What an adaptation verdict rests on: the record checks plus the model-based gate's diagnostic (the
    conversation share within the calibrated family); 'supported' never means the error rate is guaranteed."""
    checks = dict(a.get('checks') or {})
    diag = gate.get('diagnostics') or {}
    checks['calibrated_family'] = {'status': diag.get('status') or 'unknown', 'value': gate.get('conversation_share'),
                                   'limit': FAMILY_SHARE,
                                   'meaning': 'the share of the spread of conversation means that is a conversation '
                                              'effect, within the family of shapes the model-based bars are '
                                              'calibrated for'}
    states = {c['status'] for c in checks.values()}
    status = 'unsupported' if 'unsupported' in states else 'unknown' if 'unknown' in states else 'supported'
    return {'status': status, 'summary': MODEL_DIAGNOSTICS[status], 'checks': checks, 'model_based': True}


def _identity_of(x: dict | None) -> str | None:
    """A player's identity over a pairing, by `identity_match`: 'match', 'mismatch', 'unattested', 'bot', or None
    when there are no facts or replies."""
    if not x:
        return None
    if x.get('kind') == 'bot':
        return 'bot'
    seen = x.get('observed_models') or []
    said = x.get('identity_match') or {'attested': 'match'}.get(x.get('identity'), x.get('identity'))
    if said == 'mismatch' or (seen and identity_match(x.get('requested_model'), seen) == 'mismatch'):
        return 'mismatch'
    if said in (None, 'none'):
        return None
    return 'match' if said == 'match' and seen else 'unattested'


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
                               'replicate': d.get('replicate') or 1, 'planned': d.get('planned') or 0,
                               'evidence': d.get('evidence'), 'evidence_linked': d.get('evidence_linked'),
                               'audit': {}}
        else:
            for k in ('evidence', 'evidence_linked'):
                if g[k] != d.get(k):
                    g[k] = None if not (g[k] and d.get(k)) else \
                        hashlib.sha256(f'{g[k]}|{d[k]}'.encode()).hexdigest()[:24]
        for p, a in (d.get('audit') or {}).items():
            y = g['audit'].setdefault(p, {'decisions': 0, 'linked': 0, 'verified': 0, 'opaque': 0, 'problems': []})
            for f in ('decisions', 'linked', 'verified', 'opaque'):
                y[f] += a[f]
            y['problems'] = y['problems'] + a['problems']
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
    A trailing partial session counts if it has at least half the deals; a conversation shorter than a session
    is one session."""
    by = {}
    for o in obs:
        by.setdefault(o['matchup'], []).append(o)
    out = []
    for mu, os_ in by.items():
        for k in range(0, len(os_), size):
            chunk = os_[k:k + size]
            if 2 * len(chunk) < size and k:
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
    track = 'official' if pooled else track_of(design or plan)
    adaptation = track == 'adaptation'
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
    official = registered and not adaptation
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
        or leader (official, eligible, its checks passed unless the design counts the rest; with `outside`, also
        one whose checks failed), between players not quarantined in the run."""
        x = vmap.get(frozenset((p, q)))
        if pooled or not x or x['verdict'] not in DECIDED or info(p)['quarantine_reason'] or info(q)['quarantine_reason']:
            return None
        if not (x.get('counts_toward_leader') or (outside and x.get('official') and x.get('eligible')
                                                   and x.get('calibrated') is False)):
            return None
        return _side(x, p)

    def model_result(p, q):
        """Like `result`, for the adaptation track's model-based verdicts (never official)."""
        x = vmap.get(frozenset((p, q)))
        if not adaptation or not x or x['verdict'] not in DECIDED or not x.get('model_based_counts') \
                or info(p)['quarantine_reason'] or info(q)['quarantine_reason']:
            return None
        return _side(x, p)

    def _side(x, p):
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
        mb_lead = next((p for p in comp if len(comp) > 1 and all(model_result(p, q) == 'win' for q in comp if q != p)),
                       None)
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
            qualified = [x for _, x in beats if x.get('identity_qualified')]
            leader = {'player': lead, 'component': ci, 'over': second, 'basis': 'sequential verdicts',
                      'interpretation': STABLE_CONDITION,
                      'margin_bb100': round(float(point[idx[lead]] - point[idx[second]]), 1), 'ci95': lci,
                      'ci_kind': CI_FIXED, 'identity_qualified': bool(qualified),
                      'unattested': sorted({p for x in qualified for p in x.get('unattested') or []}),
                      'beats': [{'player': q, 'edge': x['edge'] if x['a'] == lead else -x['edge'],
                                 'looks_used': x['looks_used'], 'ci_official': x.get('ci_official'),
                                 'identity_qualified': bool(x.get('identity_qualified'))} for q, x in beats]}
        leaders.append(leader)
        groups.append({
            'id': ci, 'players': comp, 'leader': lead, 'leader_detail': leader, 'front_runner': front,
            'leader_outside_calibration': outside,
            'misfit_pairings': sum((x.get('additivity') or {}).get('status') == 'warning' for x in mine),
            'over': order[1] if front else None, 'front_runner_margin': margin, 'ci95': ci95, 'ci_kind': CI_FIXED,
            'power': power, 'language': 'descriptive' if pooled or not registered else
            'exploratory' if adaptation else 'confirmatory',
            'identity_qualified': bool(leader and leader['identity_qualified']),
            'drift_pairings': sum(x.get('decision_status') == 'decided_with_drift' for x in mine),
            'pairings': len(mine), 'decided': sum(x['verdict'] in DECIDED for x in mine),
            'has_bot': any(info(p)['kind'] == 'bot' for p in comp),
            'has_reference': any(str(info(p)['identity']).startswith('bot:reference') for p in comp),
            'kind': 'descriptive' if pooled else 'decided' if lead else 'model-based' if mb_lead else 'descriptive',
            'model_based_leader': mb_lead, 'track': track,
            'headline': _headline_adaptation(mb_lead, front, margin, ci95, mine) if adaptation
            else _headline(lead, front, margin, ci95, mine, pooled, outside, leader)})
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
            direct = [x['edge'] if x['a'] == p else -x['edge'] for x in mine
                      if p in (x['a'], x['b']) and x.get('edge') is not None]
            players.append({
                'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': ci,
                'identity_status': _status_of(p, pairings, 'identity_status'),
                'identity_match': _status_of(p, pairings, 'identity_match'),
                'audit_status': _status_of(p, pairings, 'audit_status'),
                'strength': round(float(point[i]), 1), 'strength_kind': STRENGTH_KIND, 'strength_se': se,
                'strength_ci95': sci, 'df': df,
                'observed_edge': round(sum(direct) / len(direct), 1) if direct else None,
                'observed_opponents': len(direct), 'ci_basis': basis, 'ci_kind': CI_FIXED, 'position': pos, 'rank': rank, 'rank_kind': 'decided' if rank else 'descriptive',
                'rank_range': rng_, 'rank_tied_with': tied, 'in_fit': True,
                'identity_resolved': bool(info(p).get('identity_resolved', identity_resolved(info(p)['identity']))),
                'net_bb': round(tot[p]['net_bb'], 1), 'hands': 2 * tot[p]['deals'], 'deals': tot[p]['deals'],
                'session_elo': round(1500 + K * float(elo[i])),
                'session_elo_ci95': [round(1500 + K * (elo[i] - q * sd)), round(1500 + K * (elo[i] + q * sd))],
                'session_elo_basis': elo_basis,
                'inferential': official and not info(p)['quarantine_reason'],
                'sessions': len(mine_s), 'session_w': won, 'session_t': ties, 'session_l': len(mine_s) - won - ties,
                'quarantined': bool(info(p)['quarantine_reason']), 'quarantine_reason': info(p)['quarantine_reason'],
                'unrated_reason': None})
    refs = [p for p in names if str(info(p)['identity']).startswith('bot:reference')]
    for row in players:
        ref = next((q for q in refs if q != row['player'] and frozenset((row['player'], q)) in vmap), None)
        x = vmap.get(frozenset((row['player'], ref))) if ref else None
        row['vs_reference'] = None if not x or x.get('edge') is None else {
            'reference': ref, 'reference_version': REFERENCE['version'],
            'edge': x['edge'] if x['a'] == row['player'] else -x['edge'],
            'cs': None if not x.get('cs') else x['cs'] if x['a'] == row['player'] else [-x['cs'][1], -x['cs'][0]],
            'ci_official': None if not x.get('ci_official') else {
                **x['ci_official'], 'interval': x['ci_official']['interval'] if x['a'] == row['player']
                else [-x['ci_official']['interval'][1], -x['ci_official']['interval'][0]]},
            'ci95': x['ci95'] if not x['ci95'] or x['a'] == row['player'] else [-x['ci95'][1], -x['ci95'][0]],
            'ci_kind': CI_FIXED,
            'deals': x['deals'], 'verdict': x['verdict']}
    unrated = sorted(excluded | ({p for d in raw for p in (d['a'], d['b'])} - set(names)))
    for p in unrated:
        players.append({
            'player': p, 'identity': info(p)['identity'], 'kind': info(p)['kind'], 'component': None,
            'identity_status': _status_of(p, pairings, 'identity_status'),
            'identity_match': _status_of(p, pairings, 'identity_match'),
            'audit_status': _status_of(p, pairings, 'audit_status'),
            'strength': None, 'strength_kind': STRENGTH_KIND, 'strength_se': None, 'strength_ci95': None, 'df': None,
            'observed_edge': None, 'observed_opponents': 0, 'ci_basis': None,
            'ci_kind': CI_FIXED, 'position': None, 'rank': None, 'rank_kind': 'descriptive', 'rank_range': None, 'rank_tied_with': [],
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
                                 reg, list(rejected), adaptation)
    return {'objective': OBJECTIVE, 'design': design, 'registration': reg, 'pooled': pooled, 'deals': len(raw),
            'track': track, 'official': not pooled and not adaptation, 'model_based': adaptation,
            'language': 'descriptive' if pooled or not registered else 'exploratory' if adaptation else 'confirmatory',
            'intervals': INTERVALS, 'strength_coverage': STRENGTH_COVERAGE,
            'label': ADAPTATION_LABEL if adaptation else None,
            'hands': 2 * len(raw), 'rejected': list(rejected), 'judge_fingerprint': judge_fingerprint(),
            'fit': {'method': 'least squares on distinct duplicate deals: each distinct deal counts once, so a '
                              'pairing weighs as much as it has deals; the strengths are centred (they sum to zero '
                              'in each group) and only their differences mean anything: a minus b is the edge the '
                              'fit predicts for a against b',
                    'strength': STRENGTH_KIND, 'observed': OBSERVED_KIND,
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
                          'adaptation track: verdicts come from the model-based gate (stated assumptions, not '
                          'guaranteed), never official, never a rank or a leader; strengths are descriptive'
                          if adaptation else
                          'decided results come only from the official decisions of the preregistered '
                          'sequential test of each pairing, recorded by the registered judge; strengths, intervals '
                          'and the order by points are descriptive'),
            'guarantee': None if not official else GUARANTEE,
            'assumptions': None if not registered else ADAPTATION_ASSUMPTIONS if adaptation else ASSUMPTIONS,
            'components': comps, 'groups': groups, 'leader': leaders[0] if leaders else None, 'leaders': leaders,
            'players': players, 'pairings': pairings, 'matchups': pairings, 'roster': roster, 'notes': notes}


def _fixed(ci95) -> str:
    return f', fixed-time descriptive 95% range {_iv(ci95)}' if ci95 else ''


def _headline(lead, front, margin, ci95, pairings, pooled, outside=None, leader=None) -> str:
    """One line per group, from that group's own data only."""
    if lead:
        text = (f'{lead} is the decided leader: it won its pairing against every other player in this group, each '
                f'an official verdict of the assumption-free test whose evidence stands today and whose checks passed. '
                f'It is {STABLE_CONDITION}.')
        if leader and leader.get('identity_qualified'):
            text += (f' Identity qualified: {", ".join(leader["unattested"])} unattested (the backend did not report '
                     f'which model answered).')
        return text
    if not front:
        return 'Only one player in this group.'
    open_ = sum(not x.get('counts_toward_leader') for x in pairings)
    what = 'pooled across runs, so nothing here is decided' if pooled else (
        f'no decided leader ({open_} of {len(pairings)} pairings without decided results that count)')
    drifted = sum(x.get('decision_status') == 'decided_with_drift' for x in pairings)
    if drifted:
        what += (f'; {drifted} pairing{"s" if drifted != 1 else ""} decided with drift (the edge changed during '
                 f'play), which count for nothing until a confirmation run on a stable model snapshot')
    if outside:
        what += (f'; {outside} won every pairing, but at least one of those verdicts failed its checks (deals '
                 f'repeated across conversations), so its error rate is not established')
    return f'{front} is ahead on points by {margin:+.0f} bb/100 over the next player{_fixed(ci95)}; {what}.'


def _headline_adaptation(lead, front, margin, ci95, pairings) -> str:
    if not front:
        return 'Only one player in this group.'
    if lead:
        return (f'{lead} is ahead of every other player in this group on the model-based test (exploratory: '
                f'adaptation track, not official, so no decided leader); {front} is ahead on points by '
                f'{margin:+.0f} bb/100{_fixed(ci95)}.')
    called = sum(x['verdict'] in DECIDED for x in pairings)
    return (f'{front} is ahead on points by {margin:+.0f} bb/100 over the next player{_fixed(ci95)}; the model-based '
            f'test leans one way in {called} of {len(pairings)} pairings (exploratory: adaptation track, not official).')


def _notes(pooled, design, pairings, comps, groups, obs, repeats, excluded, unrated, players, block, reg=None,
           rejected=(), adaptation=False) -> list:
    notes = []
    if adaptation and design.get('registered'):
        notes.append(ADAPTATION_ASSUMPTIONS)
        bad = [x for x in pairings if x['verdict'] == 'ineligible']
        if bad:
            notes.append('Ineligible for a verdict: ' + '; '.join(
                f'{x["a"]} vs {x["b"]} ({"; ".join(x.get("ineligible_reasons") or [])})' for x in bad) + '.')
        wide = [x for x in pairings if x.get('calibrated') is False]
        if wide:
            notes.append('Diagnostics failed (outside the calibrated family of shapes): '
                         + ', '.join(f'{x["a"]} vs {x["b"]}' for x in wide) + '.')
    elif pooled:
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
        if any((x.get('official_test') or {}).get('assumption_free') is False for x in pairings if x.get('official')):
            notes.append('Some official verdicts were recorded by an earlier judge whose test was model-based (a t and '
                         'sign-flip gate calibrated for a family of shapes), so their error rate is guaranteed only '
                         'within that family; today\'s assumption-free test is shown beside them as a recomputation.')
        if any(x['verdict'] == 'early' for x in pairings):
            notes.append(f'No verdict is possible from fewer than {design.get("min_conversations", MIN_CONVERSATIONS)} '
                         f'conversations (fresh match pairs) of a pairing; pairings with fewer are "too early".')
        if any(x['basis'] == 'blocks' for x in pairings):
            notes.append(f'Some pairings have one conversation: their intervals describe that conversation only '
                         f'(blocks of {block} deals), not how a new match would go.')
        wide = [x for x in pairings if x.get('calibrated') is False]
        if wide:
            notes.append('Checks failed: ' + ', '.join(f'{x["a"]} vs {x["b"]}' for x in wide) + ' repeat deals across '
                         'their conversations, so the conversations are not shown to be independent and the error '
                         'rate is not established; '
                         + ('the design counts such verdicts anyway.' if design.get('outside_calibration') == 'count'
                            else 'such a verdict never counts towards a decided rank or leader.'))
        bad = [x for x in pairings if x['verdict'] == 'ineligible']
        if bad:
            notes.append('Ineligible for a decided result: ' + '; '.join(
                f'{x["a"]} vs {x["b"]} ({"; ".join(x.get("ineligible_reasons") or [])})' for x in bad) + '.')
        drifted = [x for x in pairings if x.get('decision_status') == 'decided_with_drift']
        if drifted:
            notes.append('Decided with drift, so not counted: ' + '; '.join(
                f'{x["a"]} vs {x["b"]} ({x["qualifier"]})' for x in drifted) + f'. {DRIFT_RULE}')
        strict = [x for x in pairings if x.get('decision_status') == 'decided_identity_unattested']
        if strict:
            notes.append('Not counted under the strict identity policy: ' + '; '.join(
                f'{x["a"]} vs {x["b"]} ({", ".join(x["unattested"])} unattested)' for x in strict) + '.')
        qualified = [x for x in pairings if x.get('identity_qualified')]
        if qualified:
            notes.append('Counted with an identity qualification (an unattested player the design allows; '
                         '`plan --strict-identity` counts none): ' + '; '.join(
                             f'{x["a"]} vs {x["b"]} ({", ".join(x["unattested"])} unattested)' for x in qualified) + '.')
        notes.append(f'Each pairing\'s primary interval is the anytime-valid one (`ci_official`, level 1 - alpha / '
                     f'{design.get("n_pairings", 1)}). Every 95% range (pairings, strengths, margins, session Elo) is '
                     f'{CI_FIXED}: {INTERVALS[CI_FIXED]}.')
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
        stale = [x for x in pairings if x.get('official') and x.get('evidence_now', {}).get('status') not in (None, 'verified')]
        if stale:
            notes.append('Official verdicts whose evidence no longer stands today, so they no longer count towards a '
                         'decided rank or leader (they stay shown as recorded): ' + '; '.join(
                             f'{x["a"]} vs {x["b"]}: {x["evidence_now"]["status"]} ({x["evidence_now"]["meaning"]}'
                             + (f': {"; ".join(x["evidence_now"]["reasons"][:2])}' if x['evidence_now']['reasons'] else '')
                             + ')' for x in stale) + '. Lost, changed, invalid or altered evidence is a problem with the '
                         'records; "fails_today" means the same records fail checks that are stricter today.')
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
        notes.append('A group without a bot has no fixed anchor: its strengths only compare its own players.')
    if any(g.get('has_reference') for g in groups):
        notes.append(f'The reference bot ({REFERENCE["version"]}) is {REFERENCE["kind"].split(":")[0]}; measured '
                     f'against the other bots it wins ' + ', '.join(f'{x["edge_bb100"]:+} bb/100 against {x["b"]}'
                                                              for x in REFERENCE['results'] if x['a'] == 'reference')
                     + f'; {REFERENCE["limits"]}.')
    if any(not g.get('has_reference') for g in groups if len(g['players']) > 1):
        notes.append('Strength is relative to the players met. A group without the fixed reference bot (bot = '
                     '"reference") has no common, versioned opponent to say how good its players are at poker; the '
                     'quiz (poker fundamentals) answers that separately.')
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
    notes.append('Session Elo is the share of sessions won (10 deals of one conversation, or the whole of a shorter '
                 'one): a different, secondary statistic that never '
                 'decides anything. Its interval is the wider of a posterior one and a bootstrap that resamples '
                 'sessions with their slates and conversations.')
    return notes


def _is_tree(comp, obs):
    edges = {frozenset((o['a'], o['b'])) for o in obs if o['a'] in comp}
    return len(edges) == len(comp) - 1


def _status_of(p, pairings, key):
    """A player's identity or audit status over its pairings: the worst one any of them recorded."""
    order = ['mismatch', 'none', 'incomplete', 'unattested', 'opaque', 'attested', 'match', 'complete', 'bot']
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
                  'blocks', 'reason', 'sd_per_deal', 'power', 'alpha_pairing', 'official_test', 'assumption_free',
                  'bound_bb100', 'out_of_bound', 'capital', 'cs', 'cs_level', 'decided_at_conversation', 'model_based',
                  'evidence_version', 'linked_audit', 'gate', 'track', 'model_based_counts', 'assumption_free_verdict',
                  'label')
EARLIER_JUDGE = {'name': 'conversation-level t and sign-flip gate (an earlier judge: model-based, its error rate '
                         'not guaranteed beyond a calibrated family of shapes)', 'assumption_free': False}


def _official(rec: dict | None, reg: dict | None) -> bool:
    """Whether a recorded look is an official decision: written by the adjudicator, by the registered judge,
    with the registered calibration."""
    if not rec or not reg or not reg.get('registered') or 'eligible' not in rec:
        return False
    if rec.get('official') is False and not (rec.get('track') == 'adaptation' and rec.get('registered_record')):
        return False
    judge = rec.get('judge_fingerprint') or rec.get('judge')
    return judge == reg.get('judge_fingerprint') and rec.get('calibration_id') == reg.get('calibration_id')


def _summary_facts(facts: dict | None) -> dict:
    """What a pairing's eligibility facts say about each player, in the terms the evidence check compares."""
    out = {}
    for p, x in ((facts or {}).get('players') or {}).items():
        x = x or {}
        out[p] = {'identity': _identity_of(x), 'audit': x.get('audit'), 'protocol_consistent': x.get('protocol_consistent'),
                  'observed_models': sorted(map(str, x.get('observed_models') or [])),
                  'requested_model': x.get('requested_model'), 'problems': sorted(map(str, x.get('problems') or []))}
    return out


EVIDENCE_STATUS = {
    'verified': 'the hands it judged are here, unchanged, replay from their seeds, and today\'s checks of their '
                'records (decision coverage, audit, identity) pass',
    'unbound': 'the decision was recorded without a hash of the hands it judged, so they cannot be shown to be '
               'the hands here',
    'invalid': 'hands it judged no longer pass validation or no longer replay from their seeds',
    'lost': 'hands it judged are missing',
    'changed': 'the hands, or the records of how their decisions were made, differ from those it judged',
    'altered': 'today\'s judge is the registered one, yet it does not reproduce the recorded decision from the same '
               'hands, so the record was altered',
    'fails_today': 'the hands and their records are the ones it judged, but today\'s checks of those records '
                   '(stricter than when it was recorded) reject them',
}


def evidence_now(last: dict, today: dict, data: dict, facts: dict | None, design: dict | None = None) -> dict:
    """Whether an official decision's evidence still stands today: {'status' (a key of EVIDENCE_STATUS), 'kind'
    ('evidence': the evidence is gone or different; 'rules': the same evidence, judged by stricter checks
    today; None when verified), 'reasons'}. Only 'verified' lets the decision count towards a current decided
    rank or leader. A later change to the test itself (another judge, other bars) is not an evidence failure:
    the recorded verdict stands, and today's verdict is shown beside it as a recomputation."""
    out = lambda st, kind, *why: {'status': st, 'kind': kind, 'meaning': EVIDENCE_STATUS[st], 'reasons': list(why)}
    judged, recorded = last.get('judged_evidence_hash'), last.get('evidence_hash')
    if not (judged or recorded):
        return out('unbound', 'evidence')
    if data.get('rejected'):
        r = data['rejected']
        return out('invalid', 'evidence', *[f'{x.get("matchup")}{"" if x.get("hand") is None else " hand " + str(x["hand"])}: '
                                             f'{x.get("reason")}' for x in r[:3]])
    runner = ((facts or {}).get('verification') or {}).get('evidence_hash')
    have = data.get('evidence_hash') if int(last.get('evidence_version') or 1) >= 2 else \
        data.get('evidence_hash_v1', data.get('evidence_hash'))
    same = have == judged if judged else recorded in (have, runner)
    if not have or not same:
        had, have = last.get('evidence_deals') or last.get('deals') or 0, len(data.get('xs') or [])
        if have < had:
            return out('lost', 'evidence', f'{had - have} of the {had} deals it judged are missing')
        return out('changed', 'evidence', 'the verified hands hash differently from the ones it judged')
    now = _summary_facts(facts)
    then = _summary_facts(last.get('eligibility'))
    probs = [q for p in (data.get('a'), data.get('b')) if p
             for q in _player_problems(p, ((facts or {}).get('players') or {}).get(p), design or {})
             + [str(q) for q in (((facts or {}).get('players') or {}).get(p) or {}).get('problems') or []]]
    ver = (facts or {}).get('verification')
    if isinstance(ver, dict) and not ver.get('ok'):
        probs += [str(q) for q in ver.get('problems') or []] or ['the runner could not verify the match records']
    if facts is None:
        probs.append('no eligibility facts could be gathered today')
    if probs:
        if then and any(now.get(p) != then.get(p) for p in (data.get('a'), data.get('b')) if p):
            return out('changed', 'evidence', *probs[:3])
        return out('fails_today', 'rules', *probs[:3])
    if (last.get('judge_fingerprint') or last.get('judge')) == today.get('judge') and (
            today.get('test_verdict') != last.get('test_verdict', last.get('verdict'))
            or today.get('verdict') != last.get('verdict')
            or bool(today.get('counts_toward_leader')) != bool(last.get('counts_toward_leader'))):
        return out('altered', 'evidence', f'recorded {last.get("verdict")}, reproduced {today.get("verdict")}')
    return out('verified', None)


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
            v = decide([sign(o) for o in os_], {'max_rounds': 1, 'round_deals': len(os_), 'block': BLOCK,
                                                'model_based': False, 'outcome_bound_bb100': (design or {}).get('outcome_bound_bb100')},
                       n_pairings, 0, groups=[o['matchup'] for o in os_])
            v.update(verdict='descriptive', model_based=None,
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
            done = looks
            ahead = rec[-1].get('looks_used') if rec and _official(rec[-1], reg) else None
            if isinstance(ahead, int) and done < ahead <= rounds:
                looks = ahead           # an official look whose rounds are no longer all here: judged as lost below
            at = [o for o in os_ if o['round'] <= looks] if looks else os_
            mine = [y for y in rejected if {y.get('a'), y.get('b')} == set(key)]
            data = {'a': x['a'], 'b': x['b'], 'xs': [sign(o) for o in at], 'groups': [o['matchup'] for o in at],
                    'fps': [o['fp'] for o in at], 'rejected': mine, 'evidence_hash': evidence_hash(at),
                    'evidence_hash_v1': evidence_hash(at, 1), 'evidence_version': EVIDENCE_VERSION,
                    'linked_audit': linked_status(at)}
            today = adjudicate(data, design, n_pairings, looks, eligibility.get(key))
            beyond = len(os_) - len(at) if looks else 0
            last = rec[looks - 1] if looks and len(rec) >= looks and rec[looks - 1].get('looks_used') == looks else None
            if _official(last, reg):
                now = evidence_now(last, today, data, eligibility.get(key), design)
                earlier = {} if 'official_test' in last else {'official_test': EARLIER_JUDGE, 'assumption_free': False,
                                                              'cs': None, 'capital': None, 'model_based': None}
                adapt = track_of(design) == 'adaptation'
                v = {**today, **earlier, **{k: last[k] for k in _DECISION_KEYS if k in last}, 'official': not adapt,
                     'recorded': True, 'retrospective': False, 'judge': last.get('judge_fingerprint') or last.get('judge'),
                     'recorded_counts': bool(last.get('counts_toward_leader')),
                     'counts_toward_leader': bool(last.get('counts_toward_leader')) and now['status'] == 'verified'
                     and not adapt,
                     **({'model_based_counts': bool(last.get('model_based_counts')) and now['status'] == 'verified'}
                        if adapt else {}),
                     'evidence_now': now, 'integrity_now': now['status'] == 'verified',
                     'recomputed': {'verdict': today['verdict'], 'judge': today['judge'],
                                    'counts_toward_leader': today['counts_toward_leader'],
                                    'agrees': today['verdict'] == last.get('verdict')}}
            else:
                v = {**today, 'official': False, 'retrospective': looks >= 1, 'counts_toward_leader': False,
                     'recomputed': None, **({'model_based_counts': False} if 'model_based_counts' in today else {})}
                if today['verdict'] in DECIDED:
                    v.update(verdict='retrospective',
                             reason='no decision by the registered judge was recorded at this look; this is today\'s '
                                    'recomputation, which decides nothing')
        v.update(publication(v, design, x['a'], x['b'], registered and not pooled))
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


def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _outcome(a: dict) -> str:
    return a.get('outcome', 'fallback' if a.get('fallback') else 'valid')


def _quiz_score(answers: list[dict]) -> dict:
    """Counts from (verified) answers: a valid answer is one the player gave and the judge re-derived from its raw
    output; only those can be correct. A fallback, a backend failure and an answer the judge could not verify
    are never correct, and lose the worst option's value."""
    valid = [x for x in answers if _outcome(x) == 'valid']
    right = sum(1 for x in valid if x.get('correct'))
    n = len(answers)
    return {'answered': n, 'correct': right, 'accuracy': round(right / n, 4) if n else None,
            'valid_answers': len(valid), 'valid_correct': right,
            'valid_accuracy': round(right / len(valid), 4) if valid else None,
            'fallbacks': sum(_outcome(x) in ('fallback', 'error') for x in answers),
            'unverified_answers': sum(_outcome(x) == 'unverified' for x in answers),
            'ev_loss_bb': round(sum(float(x.get('ev_loss_bb') or 0) for x in answers), 3)}


def _answer_key(doc: dict) -> tuple[dict, list[str]]:
    """({spot id: spot}, problems) of a quiz's answer key: the spots must hash to the recorded `spots_hash` (the
    key the quiz was asked with, unaltered), and each spot's `correct` must be its best option by `ev_bb`."""
    spots = [x for x in doc.get('spots') or [] if isinstance(x, dict)]
    problems = []
    if not spots:
        return {}, ['the quiz keeps no answer key']
    if hashlib.sha256(json.dumps(spots, sort_keys=True).encode()).hexdigest()[:16] != doc.get('spots_hash'):
        problems.append('the answer key is not the one its spots_hash names (altered after the quiz)')
    for x in spots:
        ev = x.get('ev_bb') or {}
        if not ev or x.get('correct') not in ev or ev[x['correct']] < max(ev.values()) or \
                sorted(ev) != sorted(x.get('options') or []):
            problems.append(f'spot {x.get("id")}: its answer is not its best option')
    return {x.get('id'): x for x in spots}, problems


def _check_answer(a: dict, spot: dict | None, spec: dict, effective: dict | None, cli: dict | None) -> dict:
    """One answer re-derived from its record by the quiz's own verifier (`quiz.verify_answer`: the spot rebuilt
    from its id and compared with the copy kept, every call checked against its raw output by its backend's
    parser, every refused reply re-applied and the accepted one played in the engine, so the request, option,
    correctness, outcome and loss must be what the replies give), plus the CLI each call ran compared with the
    player's recorded CLI file. Returns the answer as served: the claims replaced by the verified representation
    ('outcome' 'unverified', never correct, when any check failed), with 'verified', 'problems',
    'identity_match', 'audit' and the calls without their raw payloads."""
    from .backends import BACKENDS, Backend, artifact_changed
    from .quiz import verify_answer
    try:
        v = verify_answer(a, spec, effective, spot)
    except Exception as e:      # a record the verifier cannot even read is not verified
        v = {'status': 'unverifiable', 'identity_match': None, 'answer': None,
             'problems': [f'the quiz verifier failed: {e!r}'[:300]]}
    problems = list(v.get('problems') or [])
    calls = [c for c in a.get('calls') or [] if isinstance(c, dict)]
    needs_cli = bool((BACKENDS.get(spec.get('backend')) or Backend).exe)
    for c in calls:
        if c.get('backend_failed'):
            continue
        if (needs_cli or cli or c.get('cli')) and not (cli and isinstance(c.get('cli'), dict) and c['cli'].get('sha256')
                                                         and not artifact_changed(cli, c['cli'])):
            problems.append(f'attempt {c.get("attempt")}: its CLI is not the player\'s recorded CLI file')
    if v.get('answer') is None and not problems:
        problems.append(f'not verified ({v.get("status")})')
    good = v.get('answer') is not None and v.get('status') in ('verified', 'bot') and not problems
    ev_bb = (spot or {}).get('ev_bb') or {}
    worst = round(max(ev_bb.values()) - min(ev_bb.values()), 3) if ev_bb else None
    keep = {k: val for k, val in a.items() if k not in ('calls', 'correct', 'outcome', 'ev_loss_bb', 'ev_lost_bb',
                                                       'identity_match', 'audit', 'audit_problems', 'executed',
                                                       'action', 'fallback')}
    got = v['answer'] if good else {'outcome': 'unverified', 'correct': False, 'fallback': bool(a.get('fallback')),
                                    'executed': a.get('executed'), 'ev_loss_bb': worst}
    return {**keep, **got, 'action': got.get('executed'), 'ev_lost_bb': got.get('ev_loss_bb'),
            'identity_match': v.get('identity_match') or 'none', 'audit': v.get('status'), 'verified': good,
            'problems': problems, 'calls': [{k: val for k, val in c.items() if k != 'raw'} for c in calls]}


def verify_quiz_player(x: dict, key: dict, key_problems=(), allow=None) -> dict:
    """A quiz player's record re-derived by the judge (`_check_answer` per answer, the answer key's integrity,
    each spot asked once, provenance), whatever the record's own labels say. {'status' ('eligible' | 'bot' |
    'ineligible' | 'unverified': no answers kept), 'reasons', 'attestation' (`attestation`: the same policy as
    the games), 'answers' (as served)}."""
    answers = x.get('answers')
    spec = x.get('spec') or {}
    backend = spec.get('backend')
    if allow is None:
        allow = ((x.get('eligibility') or {}).get('policy') or {}).get('allow_unattested', ALLOW_UNATTESTED)
    if x.get('kind') == 'bot' or 'bot' in spec:
        checked = [_check_answer(a, key.get(a.get('spot')), spec, x.get('effective'), None)
                   for a in answers or [] if isinstance(a, dict)]
        return {'status': 'bot', 'reasons': [], 'attestation': attestation(None, 'bot'), 'answers': checked}
    if not isinstance(answers, list) or not answers:
        return {'status': 'unverified', 'reasons': ['the quiz record keeps no answers, so its score cannot be checked'],
                'attestation': attestation(backend, None, allow), 'answers': []}
    cli = x.get('cli') if isinstance(x.get('cli'), dict) else None
    checked = [_check_answer(a, key.get(a.get('spot')), spec, x.get('effective'), cli) for a in answers
               if isinstance(a, dict)]
    why = list(key_problems)
    bad = [a for a in checked if not a['verified']]
    if bad:
        why.append(f'{len(bad)} answer{"s" if len(bad) != 1 else ""} not verified from the raw output '
                   f'({bad[0]["spot"]}: {bad[0]["problems"][0]})')
    seen = [a.get('spot') for a in checked]
    if len(set(seen)) != len(seen):
        why.append('a spot is answered more than once')
    idents = {a['identity_match'] for a in checked}
    ident = 'mismatch' if 'mismatch' in idents else 'match' if idents == {'match'} else \
        'unattested' if 'unattested' in idents else None
    att = attestation(backend, ident, allow)
    if not att['allowed']:
        why.append(f'{att["status"]}: {att["meaning"]} ({att["policy"]})')
    audits = {a['audit'] for a in checked}
    if audits - {'verified'}:
        why.append(f'some answers\' turns are not verified for tool use ({", ".join(sorted(audits - {"verified"}))})')
    missing = [k for k in ('protocol', 'implementation', 'cli_version') if not x.get(k)]
    if missing:
        why.append('no ' + ', '.join(missing) + ' recorded')
    if cli and x.get('cli_version') and x.get('cli_version') != cli.get('version'):
        why.append(f'the recorded CLI version {x.get("cli_version")!r} is not its CLI file\'s {cli.get("version")!r}')
    return {'status': 'ineligible' if why else 'eligible', 'reasons': list(dict.fromkeys(why)), 'attestation': att,
            'answers': checked}


def _quiz_categories(answers: list[dict], cats: dict) -> dict:
    out = {}
    for c in dict.fromkeys(a.get('category', 'basics') for a in answers):
        sc = _quiz_score([a for a in answers if a.get('category', 'basics') == c])
        meta = cats.get(c) or {}
        design = meta.get('sampling')
        design = design.get('design') if isinstance(design, dict) else design
        iid = design == 'independent'
        sc['interval95'] = _wilson(sc['correct'], sc['answered']) if iid else None
        sc['estimand'] = meta.get('estimand') if iid else (
            'descriptive: ' + ('the spots were kept by answer quotas or another non-independent rule, so they are not '
                               'an independent sample of a population; counts, not a population interval'
                               if meta.get('population') else 'a fixed set of spots; counts, not a population interval'))
        out[c] = sc
    return out


def verified_quiz(doc: dict) -> dict:
    """A quiz.json as it should be served: each player's answers re-derived by the judge (`verify_quiz_player`),
    every score recounted from them (the headline fields from `scored_category`, the basics when asked, each
    category with its own denominators), `eligibility` and `attestation` the judge's, never the record's."""
    key, key_problems = _answer_key(doc)
    cats = doc.get('categories') or {}
    players = {}
    for pid, x in (doc.get('players') or {}).items():
        if not isinstance(x, dict):
            continue
        policy = (doc.get('protocol') or {}).get('allow_unattested') if isinstance(doc.get('protocol'), dict) else None
        v = verify_quiz_player(x, key, key_problems, policy)
        answers = v['answers']
        by_cat = _quiz_categories(answers, cats) if answers else {}
        scored = 'basics' if 'basics' in by_cat else 'all' if by_cat else None
        head = by_cat['basics'] if scored == 'basics' else _quiz_score(answers) if answers else \
            {'answered': x.get('answered'), 'correct': None, 'accuracy': None, 'valid_answers': None,
             'valid_correct': None, 'valid_accuracy': None, 'fallbacks': x.get('fallbacks'), 'unverified_answers': None,
             'ev_loss_bb': x.get('ev_loss_bb')}
        n = head.get('answered') or 0
        head = {**head, 'total': head.get('answered'), 'score': head.get('accuracy'), 'ev_lost_bb': head.get('ev_loss_bb'),
                'fallback_rate': round(head['fallbacks'] / n, 4) if n and head.get('fallbacks') is not None else None,
                'mean_ev_loss_bb': round(head['ev_loss_bb'] / n, 3) if n and head.get('ev_loss_bb') is not None else None}
        players[pid] = {**{k: val for k, val in x.items() if k not in ('answers', 'categories', 'eligibility')},
                        **head, 'scored_category': scored, 'categories': by_cat, 'answers': answers,
                        'eligibility': {'status': v['status'], 'reasons': v['reasons'], 'checked_by': 'judge',
                                        'identity_match': {'attested': 'match'}.get(v['attestation']['status'],
                                                                                    v['attestation']['status'])},
                        'attestation': v['attestation'], 'identity_match': v['attestation']['status'],
                        'audit_status': None if not answers else 'verified' if all(a.get('audit') == 'verified' for a in answers) else 'incomplete'}
    return {**doc, 'players': players, 'verified_by': 'judge',
            'answer_key': {'status': 'altered' if key_problems else 'intact', 'problems': key_problems}}


_QUIZ: dict = {}


def verified_quiz_file(path) -> dict | None:
    """`verified_quiz` of a quiz.json, cached by the file's size and time (the first check of a quiz solves its
    spots, which takes seconds); None when it is not a readable quiz."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
    if key not in _QUIZ:
        doc = _json(path)
        if not isinstance(doc, dict):
            return None
        if len(_QUIZ) > 64:
            _QUIZ.clear()
        _QUIZ[key] = verified_quiz(doc)
    return _QUIZ[key]


MIXED = 'mixed'
MATCH_ONLY = ('system_prompt_sha1', 'horizon')   # protocol settings a match adds to the player's own


def _one(values):
    vals = {json.dumps(v, sort_keys=True, default=str) for v in values}
    return None if not vals else json.loads(vals.pop()) if len(vals) == 1 else MIXED


def player_provenance(root) -> dict:
    """{player id: {'identity', 'settings', 'code', 'cli_version', 'cli_sha256'}} from the match configs under
    root: the player's own effective settings (its protocol without what the match adds), the code that plays it
    (every recorded implementation file but the judge), and its CLI's version and file. A value that differs
    between matches is 'mixed'; one never recorded is None."""
    from .history import match_identity
    seen = defaultdict(lambda: defaultdict(list))
    for cfg_path in sorted(Path(root).rglob('config.json')):
        cfg = _json(cfg_path) or {}
        names = cfg.get('players') or []
        for pid, spec in (cfg.get('specs') or {}).items():
            got = seen[pid]
            got['identity'].append(_base_identity(match_identity(cfg, pid)))
            proto = cfg.get('protocol')
            proto = proto.get(pid) if isinstance(proto, dict) and 'fingerprint' not in proto else proto
            st = proto.get('settings') if isinstance(proto, dict) else None
            got['settings'].append({k: v for k, v in st.items() if k not in MATCH_ONLY} if isinstance(st, dict) else None)
            files = ((cfg.get('implementation') or {}).get('files') or {}) if isinstance(cfg.get('implementation'), dict) else {}
            got['code'].append({k: v for k, v in files.items() if not k.endswith('/ratings.py')} or None)
            if 'bot' in spec:
                continue
            seat = names.index(pid) if pid in names else None
            eff = cfg.get('effective') or []
            exe = (eff[seat] or {}).get('exe') if seat is not None and seat < len(eff) and isinstance(eff[seat], dict) else None
            got['cli_version'].append((cfg.get('cli_versions') or {}).get(exe) if exe else None)
            got['cli_sha256'].append(((cfg.get('cli_artifacts') or {}).get(exe) or {}).get('sha256') if exe else None)
    return {pid: {k: _one(got[k]) for k in ('identity', 'settings', 'code', 'cli_version', 'cli_sha256')}
            for pid, got in seen.items()}


def _base_identity(ident: str) -> str:
    return re.split(r'[#~!@]', str(ident), maxsplit=1)[0]


def quiz_link(x: dict, doc: dict, player: dict | str | None) -> dict:
    """Whether a quiz score may stand for a tournament player of the same model and reasoning level: 'attached'
    only when its answers are eligible (or a bot's) and the player's own settings, the code they share and the CLI
    (version and file) are recorded on both sides and the same; else 'model-level' (a separate assessment of the
    same model and level, shown apart under its own name), with `differences` saying what differs or is
    unrecorded."""
    prov = player if isinstance(player, dict) else {'identity': player}
    if (x.get('eligibility') or {}).get('status') == 'bot':
        return {'status': 'attached', 'differences': []}
    diffs = []
    if (x.get('eligibility') or {}).get('status') != 'eligible':
        diffs.append('its answers are not eligible')
    proto = x.get('protocol') if isinstance(x.get('protocol'), dict) else {}
    mine = (proto.get('settings') or {}).get('player') if isinstance(proto.get('settings'), dict) else proto.get('player')
    theirs = prov.get('settings')
    if mine is None or theirs is None:
        diffs.append('player settings not recorded on both sides')
    elif theirs == MIXED or json.dumps(mine, sort_keys=True, default=str) != json.dumps(theirs, sort_keys=True, default=str):
        diffs.append('different player settings')
    impl = x.get('implementation') if isinstance(x.get('implementation'), dict) else doc.get('implementation')
    mine = {k: v for k, v in ((impl or {}).get('files') or {}).items() if not k.endswith('/ratings.py')} if isinstance(impl, dict) else {}
    theirs = prov.get('code')
    if not mine or not theirs:
        diffs.append('code not recorded on both sides')
    elif theirs == MIXED or any(mine[k] != theirs[k] for k in set(mine) & set(theirs)) or not set(mine) & set(theirs):
        diffs.append('different code')
    cli = x.get('cli') if isinstance(x.get('cli'), dict) else {}
    for key, have, want in (('CLI version', x.get('cli_version') or cli.get('version'), prov.get('cli_version')),
                            ('CLI file', cli.get('sha256'), prov.get('cli_sha256'))):
        if have is None or want is None:
            diffs.append(f'{key} not recorded on both sides')
        elif want == MIXED or have != want:
            diffs.append(f'different {key}')
    return {'status': 'model-level' if diffs else 'attached', 'differences': diffs}


def fundamentals(root, players: dict | None = None) -> dict | None:
    """Poker fundamentals, reported apart from match results: the quiz scores (quiz.py) found in quiz.json at
    root or one folder below it, as the judge re-derives them (`verified_quiz`: every answer re-verified from its
    raw output and bound to the scored option; a fallback or an unverified answer is never correct; accuracy over
    every spot asked, valid-answer accuracy apart), per category, each with its provenance (quiz version and
    spots, protocol, implementation, CLI version), `eligibility` and `attestation` (the games' policy). A score is
    attached to a run's player (`players`: {player id: provenance, as `player_provenance` gives it, or a bare
    identity}) only when `quiz_link` finds it compatible: eligible (or a bot's), the same model and reasoning
    level, and the same player settings, code and CLI. A score of the same model and level that fails that check
    stays under its quiz name as a separately identified model-level assessment (`link`: {'status':
    'model-level', 'player', 'differences'}), never silently the tournament player's. Scores are
    descriptive: an interval only where a category's spots are an independent sample of a defined population
    (`sampling: 'independent'`), never for a fixed set or a sample kept by answer quotas."""
    from .history import identity as ident_of, _base
    root = Path(root)
    paths = [p for p in [root / 'quiz.json', *sorted(root.glob('*/quiz.json'))] if p.exists()]
    if not paths:
        return None
    provs = {p: v if isinstance(v, dict) else {'identity': _base(v)} for p, v in (players or {}).items()}
    by_base = {_base(str(v.get('identity'))): p for p, v in provs.items() if v.get('identity') not in (None, MIXED)}
    out, versions = {}, set()
    for path in paths:
        doc = verified_quiz_file(path)
        if doc is None:
            continue
        versions.add((doc.get('version'), doc.get('spots_hash')))
        for pid, x in doc['players'].items():
            if not x.get('answers') and x.get('answered') is None:
                continue
            base = _base(ident_of(pid, x.get('spec') or {}))
            elig = x['eligibility']
            same = by_base.get(base)
            link = quiz_link(x, doc, provs[same]) if same else {'status': 'none', 'differences': []}
            attach = link['status'] == 'attached'
            name = same if attach else pid
            out[name] = {'quiz_player': pid, 'identity': base, 'attached': attach,
                         'attached_by': 'model, reasoning level, player settings, code and CLI' if attach else None,
                         'link': {**link, 'player': same},
                         **{k: x.get(k) for k in ('answered', 'correct', 'accuracy', 'valid_answers', 'valid_correct',
                                                  'valid_accuracy', 'fallbacks', 'unverified_answers', 'ev_loss_bb')},
                         'scored_category': x['scored_category'], 'categories': x['categories'],
                         'eligibility': {k: elig[k] for k in ('status', 'reasons')}, 'attestation': x['attestation'],
                         'provenance': {'quiz_version': doc.get('version'), 'spots_hash': doc.get('spots_hash'),
                                        'protocol': (x.get('protocol') or {}).get('fingerprint')
                                        if isinstance(x.get('protocol'), dict) else x.get('protocol'),
                                        'implementation': (x.get('implementation') or {}).get('fingerprint')
                                        if isinstance(x.get('implementation'), dict) else x.get('implementation'),
                                        'cli_version': x.get('cli_version'),
                                        'cli_sha256': (x.get('cli') or {}).get('sha256') if isinstance(x.get('cli'), dict) else None,
                                        'identity_match': x['identity_match'], 'audit': x['audit_status']},
                         'version': doc.get('version'), 'spots_hash': doc.get('spots_hash'),
                         'source': str(path.relative_to(root))}
    return {'label': 'poker fundamentals', 'separate_from': 'match results',
            'meaning': 'how often a player finds the computed best answer in decision spots with a fixed answer '
                       'key (a fallback, or an answer the judge cannot re-derive from its raw output, is never '
                       'correct; valid-answer accuracy is shown apart); it measures fundamentals in stated-range '
                       'decisions, not winnings against these opponents and not overall poker strength, and is '
                       'never pooled with strengths or verdicts. A score is attached to a run\'s player only with its '
                       'provenance and when its answers are eligible; the scores are descriptive counts, with an '
                       'interval only for a category sampled independently from a defined population',
            'versions': sorted([v for v in versions if v[0]], key=str), 'players': out}


def track_runs(root) -> dict:
    """{track: folder} of a tournament folder that holds one run per track (<root>/official, <root>/adaptation);
    empty for a run folder itself. The tracks are judged apart, never pooled."""
    root = Path(root)
    if (root / 'plan.json').exists():
        return {}
    return {t: root / t for t in TRACKS if (root / t / 'plan.json').exists()}


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
    prov = player_provenance(root)
    audit = {p: {**(s.get('audit') or {}), 'identity': s.get('identity'), 'fallbacks': s.get('fallbacks'),
                 'retries': s.get('retries'), 'served_models': s.get('served_models')} for p, s in stats.items()}
    return {**r, 'stats': stats, 'stat_labels': LABELS, 'position_labels': POSITIONS, 'audit': audit,
            'reference': REFERENCE, 'tournament': (plan or {}).get('tournament'),
            'populations': {'verdicts and strengths': 'verified results: duplicate deals both plays committed, '
                                                     'validated as pairs and replayed from their seeds',
                            'stats': 'the same verified results (stats.population per player)'},
            'fundamentals': fundamentals(root, {p: {**prov.get(p, {}), 'identity': _base_identity(m['identity'])}
                                                for p, m in meta.items()})}


VERDICT_TEXT = {
    'a': '{a} won the pairing (decided; ' + STABLE_CONDITION + ')',
    'b': '{b} won the pairing (decided; ' + STABLE_CONDITION + ')',
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
    lines = [f'Standings (descriptive): strength = centred fitted strength in {r["objective"]} (zero-sum in each '
             f'group; the difference of two strengths is the edge the fit predicts between them, not anyone\'s '
             f'winnings against an average opponent), with a fixed-time descriptive 95% interval from a bootstrap '
             f'that keeps shared cards and conversations together (not adjusted for when play stopped); observed = '
             f'observed opponent-weighted winnings, the mean of its direct edges, one per opponent met.']
    for g in r.get('groups', []):
        if len(r['groups']) > 1:
            lines.append(f'\nGroup {g["id"] + 1}')
        lines.append(f'{"#":>3} {"rank":>5}  {"player":30} {"strength":>8} {"95% interval":>14} {"basis":>13} '
                     f'{"observed":>8} {"net bb":>8} {"hands":>6}  {"session Elo (95%)":>20} {"W-T-L":>7}')
        for p in r['players']:
            if p['component'] != g['id']:
                continue
            rk = str(p['rank']) if p['rank'] else '—'
            elo = f'{p["session_elo"]} ({p["session_elo_ci95"][0]}–{p["session_elo_ci95"][1]})'
            q = ' *' if p['quarantined'] else ''
            lines.append(f'{p["position"]:>3} {rk:>5}  {name(p["player"]) + q:30} {p["strength"]:>+8.1f} '
                         f'{_iv(p["strength_ci95"]):>14} {p["ci_basis"] or "—":>13} '
                         f'{"—" if p.get("observed_edge") is None else format(p["observed_edge"], "+.1f"):>8} '
                         f'{p["net_bb"]:>+8.1f} {p["hands"]:>6}  '
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
        return (f'No preregistered design: nothing in this run is a test. Retrospective sample size only: the official '
                f'test would need about {x["deals_needed"]:,} deals per pairing ({x["conversations_needed"]:,} '
                f'conversations of {d["round_deals"]} deals) for 80% power at {x["target_edge_bb100"]:.0f} bb/100.')
    if d.get('track') == 'adaptation':
        return _design_line_adaptation(d)
    sc = {x['scenario']: x for x in d.get('power_scenarios') or []}
    per = f'{d["replicates"]} conversation{"s" if d["replicates"] != 1 else ""} of {d["round_deals"]} deal' \
          + ('s' if d['round_deals'] != 1 else '')
    head = (f'Planned to detect a {d["target_edge_bb100"]:.0f} bb/100 edge: up to {d["max_rounds"]} rounds of {per} '
            f'({d["max_deals"]:,} deals per pairing), a verdict possible from {d["min_conversations"]} conversations '
            f'on. The official test is assumption-free: conversation means bounded by +-{_bound_of(d):,.0f} bb/100 '
            f'(the stacks), a verdict when a bettor\'s capital reaches {d.get("threshold") or 2 / d["alpha_pairing"]:,.0f}.')
    if sc:
        m = sc['moderate']
        head += (f' Chance that a pairing whose better player is ahead by the target is decided (simulated; per-deal '
                 f'spread {d["prior_sd_bb100"]:.0f}): {_pct(m["power"])} with the assumed dependence (ICC '
                 f'{m["assumed_icc"]}, AR(1) {m["assumed_ar1"]}), {_pct(sc["optimistic"]["power"])} with none, '
                 f'{_pct(sc["pessimistic"]["power"])} if it is worse; dependence changes only the power, never the '
                 f'error rate.')
    lead = d.get('leader_chances')
    if lead:
        n = (1 + math.isqrt(1 + 8 * d['n_pairings'])) // 2
        head += (f' Chance of naming the true best of the {n} players the decided leader, assumed dependence: '
                 + ', '.join(f'{_pct(x["p_leader"])} if {LEADER_SCENARIO[x["scenario"]].format(e=x["edge_bb100"])}'
                             for x in lead) + '.')
    if d.get('underpowered'):
        why = 'a budget of ' + f'{d["budget_deals"]:,} deals' if d.get('max_rounds_basis') == 'budget' else 'its max_rounds'
        need = f'about {d["deals_needed"]:,}' if d.get('deals_needed') else \
            f'more than {MAX_ROUNDS * d["round_deals"] * d["replicates"]:,}'
        head += f' Underpowered: capped by {why}; 80% would need {need} deals per pairing.'
    if d.get('strict_identity'):
        head += f' {STRICT_IDENTITY[0].upper()}{STRICT_IDENTITY[1:]}.'
    return head


def _design_line_adaptation(d) -> str:
    sc = {x['scenario']: x for x in d.get('power_scenarios') or []}
    head = (f'Adaptation track ({ADAPTATION_LABEL}): up to {d["max_rounds"]} rounds of {d["replicates"]} '
            f'conversation{"s" if d["replicates"] != 1 else ""} of {d["round_deals"]} deals ({d["max_deals"]:,} deals per '
            f'pairing), models remembering the whole conversation; verdicts and stopping by the model-based gate, a '
            f'verdict possible from {d["min_conversations"]} conversations on.')
    if sc:
        m = sc['moderate']
        head += (f' Chance that the model-based test leans toward the stronger player at {d["target_edge_bb100"]:.0f} bb/100 '
                 f'(simulated, normal conversation means): {_pct(m["power"])} with the assumed dependence, '
                 f'{_pct(sc["optimistic"]["power"])} with none, {_pct(sc["pessimistic"]["power"])} if it is worse. Its '
                 f'error rate holds only within its calibrated family of shapes, so its results are exploratory.')
    if d.get('underpowered'):
        why = f'a budget of {d["budget_deals"]:,} deals' if d.get('max_rounds_basis') == 'budget' else \
            'the most rounds the gate is calibrated for' if d.get('max_rounds_basis') == 'power' else 'its max_rounds'
        head += f' Underpowered: capped by {why}.'
    return head


LEADER_SCENARIO = {'one best': 'it is {e:.0f} bb/100 better than each other player and they are equal',
                   'one best by twice the target': 'it is {e:.0f} bb/100 better than each other player',
                   'ladder': 'the players are evenly spaced {e:.0f} bb/100 apart'}


def _capital(m) -> str:
    cap, bar = m.get('capital'), (m.get('official_test') or {}).get('threshold')
    if not cap or not bar:
        return ''
    best = max(('a', 'b', 'even'), key=lambda k: cap.get(k) or 0)
    return f'best capital {cap[best]:,.2f} ({best}) of the {bar:,.0f} a verdict needs'


def verdicts(r, labels=None) -> str:
    """Each pairing: the edge, its intervals, the official test's capital against its bar, and the verdict."""
    labels = labels or {}
    name = lambda p: labels.get(p, p)
    order = {'a': 0, 'b': 0, 'even': 1, 'exhausted': 2, 'open': 3, 'early': 4, 'ineligible': 5, 'retrospective': 6,
             'descriptive': 7}
    d = r.get('design')
    reg = r.get('registration') or {}
    adaptation = r.get('track') == 'adaptation'
    if r.get('pooled'):
        lines = ['Pairings (descriptive: pooled across runs, nothing here is a test):']
    elif adaptation and d and d.get('registered', True):
        lines = [f'Adaptation track ({ADAPTATION_LABEL}): models keep one conversation for {d["round_deals"]} deals and '
                 f'can adapt to the opponent. Verdicts come from the model-based gate, under its own registered '
                 f'design (judge {reg.get("judge_fingerprint") or "?"}); its error rate holds only within the family of '
                 f'shapes its bars are calibrated for, so none of it is official or decides a rank or leader:']
    elif d and not d.get('registered', True):
        why = '; '.join(reg.get('reasons') or []) or 'no preregistered design'
        lines = [f'Pairings (descriptive: this run is not preregistered ({why}); nothing here is a test):']
    else:
        lines = [f'Pairings (decided only by the preregistered sequential test, as recorded when play stopped by the '
                 f'registered judge {reg.get("judge_fingerprint") or "?"}: an assumption-free betting test, at most '
                 f'{d["alpha"]:.0%} chance of any false verdict over {d["n_pairings"]} '
                 f'pairing{"s" if d["n_pairings"] != 1 else ""} for any distribution of results, given independent '
                 f'conversations whose expected edge, given the ones before, is at most the null value; a winner is '
                 f'{STABLE_CONDITION}; drift is flagged per pairing when detected):']
    for m in sorted(r['pairings'], key=lambda m: (order[m['verdict']], m['a'], m['b'])):
        a, b = name(m['a']), name(m['b'])
        head = f'  {a} vs {b}: '
        if m['edge'] is None:
            lines.append(head + 'no deals yet')
            continue
        head += f'{m["edge"]:+.0f} bb/100 for {a} over {m["deals"]} deals in {m["conversations"]} conversation' \
            + ('s' if m['conversations'] != 1 else '')
        fixed = f'fixed-time descriptive 95% range {_iv(m["ci95"])}' if m.get('ci95') else ''
        off = m.get('ci_official') if not adaptation and m['verdict'] != 'descriptive' else None
        if off:
            head += f' (official anytime-valid {off["level"]:.1%} interval {_iv(off["interval"])}' \
                + (f'; {fixed})' if fixed else ')')
        elif fixed:
            head += f' ({fixed})'
        if adaptation and m.get('t') is not None and m['verdict'] != 'descriptive':
            head += f'; model-based t = {m["t"]}' + (f' against its bar {m["boundary"]:.2f}' if m.get('boundary') else '')
        if m['verdict'] != 'descriptive' and m.get('looks_used') is not None:
            head += ', no planned look yet' if m['looks_used'] < 1 else f', look {m["looks_used"]} of {m["looks_planned"]}'
            cap = '' if adaptation else _capital(m)
            head += f', {cap}' if cap else ''
            if m.get('decided_at_look') and m['decided_at_look'] != m['looks_used']:
                head += f', decided at look {m["decided_at_look"]}'
        e = r.get('equiv_bb100', EQUIV_BB100)
        state = VERDICT_TEXT[m['verdict']].format(a=a, b=b, e=e)
        if adaptation and m['verdict'] in DECIDED:
            state = ADAPTATION_TEXT[m['verdict']].format(a=a, b=b, e=e) + ' (exploratory: model-based, not official)'
            if m.get('calibrated') is not True:
                state += ' (diagnostics ' + ('failed' if m.get('calibrated') is False else 'incomplete') + ')'
        elif m.get('decision_status') == 'decided_with_drift':
            state = _named(m['qualifier'], name, [m['a'], m['b']])
        elif m['verdict'] in DECIDED and m.get('standing') and m['standing'].startswith('decided ('):
            state += m['standing'][len('decided'):]
        if adaptation and m.get('recorded'):
            state += f'; recorded by judge {m.get("judge")}'
        elif m.get('official'):
            state += f'; official (recorded by judge {m.get("judge")})'
            if (m.get('official_test') or {}).get('assumption_free') is False:
                state += ', made by an earlier model-based test'
        elif m.get('retrospective'):
            state += '; retrospective (today\'s recomputation)'
        lines.append(f'{head}: {state}.')
        if m['verdict'] == 'ineligible':
            lines.append('      ' + _named('; '.join(m.get('ineligible_reasons') or []), name, [m['a'], m['b']]) + '.')
        elif m.get('reason') and m['verdict'] != 'descriptive':
            lines.append(f'      {m["reason"]}.')
        st = (m.get('assumptions') or {}).get('status')
        if m['verdict'] not in ('descriptive',) and st and st != 'supported':
            lines.append(f'      {DIAGNOSTICS[st]}')
        if (m.get('drift') or {}).get('status') == 'drift detected':
            lines.append(f'      drift detected: {m["drift"]["meaning"]}.')
        if m.get('identity_qualified'):
            lines.append(f'      identity qualified: {_named(m["identity_qualifier"], name, [m["a"], m["b"]])}.')
        mb = m.get('model_based')
        if isinstance(mb, dict) and mb.get('verdict') and m['verdict'] != 'descriptive':
            lines.append(f'      model-based (secondary, exploratory, never official; assumes a family of shapes): '
                         f'{(ADAPTATION_TEXT.get(mb["verdict"]) or VERDICT_TEXT.get(mb["verdict"], mb["verdict"])).format(a=a, b=b, e=e)}'
                         + (f', t = {mb["t"]} vs bar {mb["boundary"]:.2f}' if mb.get('boundary') else ''))
        now = m.get('evidence_now')
        if now and now['status'] != 'verified':
            lines.append(f'      evidence today: {now["status"]} ({now["meaning"]}); it no longer counts towards a '
                         f'decided rank or leader.')
        if m.get('recomputed') and not m['recomputed']['agrees']:
            lines.append(f'      today\'s judge would say {m["recomputed"]["verdict"]} (retrospective; the recorded '
                         f'verdict stands).')
        if (m.get('additivity') or {}).get('status') == 'warning':
            lines.append(f'      the strengths imply {m["fitted_edge"]:+.0f} bb/100 here, not the observed '
                         f'{m["observed_edge"]:+.0f}: they do not add up across pairings.')
        if m.get('deals_since_look'):
            lines.append(f'      {m["deals_since_look"]} more deals are in a round under way, not yet looked at.')
        need = (m.get('power') or {}).get('deals_for_80pct') or {}
        if m['verdict'] in ('open', 'exhausted') and need.get('50') and m['conversations'] >= 2 and not adaptation:
            lines.append('      deals per pairing for an 80% chance that the official test decides a real edge of '
                         + ', '.join(f'{e} bb/100: about {need[e]:,}' for e in need if need[e]) + ' (approximate, from '
                         'the observed spread of its conversations and their length; a planning number, not a forecast)')
    lead = [x for x in r.get('leaders') or [] if x]
    if adaptation:
        mb = [g['model_based_leader'] for g in r.get('groups') or [] if g.get('model_based_leader')]
        lines.append('Ahead of every other player on the model-based test (exploratory: not official, not a decided '
                     'leader): ' + (', '.join(name(p) for p in mb) if mb else 'none') + '.')
    elif not r.get('pooled') and (d or {}).get('registered', True):
        lines.append('Decided leader: ' + ('; '.join(f'{name(x["player"])} (won its pairing against every other '
                                                    f'player of its group: official verdicts, evidence verified today, '
                                                    f'checks passed; {STABLE_CONDITION}'
                                                    + (f'; identity qualified: {", ".join(name(p) for p in x["unattested"])} '
                                                       f'unattested' if x.get('identity_qualified') else '') + ')'
                                                    for x in lead) if lead else 'none.'))
        for g in r.get('groups') or []:
            if g.get('leader_outside_calibration'):
                lines.append(f'  {name(g["leader_outside_calibration"])} won every pairing of its group, but not all '
                             f'with their checks passed, so it is not a decided leader.')
    if d and not r.get('pooled'):
        lines.append(design_line(d))
    return '\n'.join(lines)


def fundamentals_text(f, labels=None) -> str:
    """The quiz scores, as their own section: never mixed with winnings."""
    if not f or not f.get('players'):
        return ''
    name = lambda p: (labels or {}).get(p, p)
    lines = ['Poker fundamentals (the quiz: decision spots with computed answers, every answer re-verified from its raw '
             'output; reported apart from match results, which measure winnings against these opponents; a fallback '
             'is never correct; fundamentals in stated-range decisions, not overall poker strength):']
    for p, x in sorted(f['players'].items(), key=lambda kv: -(kv[1].get('accuracy') or 0)):
        st = (x.get('eligibility') or {}).get('status')
        head = f'  {name(p)}: '
        if x.get('correct') is None:
            lines.append(head + f'{x.get("answered")} answered, not recounted ({"; ".join(x["eligibility"]["reasons"])})')
            continue
        head += (f'{x["correct"]}/{x["answered"]} right in {x.get("scored_category") or "the quiz"}'
                 + (f' ({x["valid_correct"]}/{x["valid_answers"]} of its valid answers)' if x.get('fallbacks') else '')
                 + f', {x["ev_loss_bb"]} bb lost to the best answers'
                 + (f', {x["fallbacks"]} fallbacks' if x.get('fallbacks') else '')
                 + (f', {x["unverified_answers"]} not verified from their raw output (never correct)'
                    if x.get('unverified_answers') else '')
                 + (', unattested (allowed by the policy)' if (x.get('attestation') or {}).get('status') == 'unattested'
                    and st == 'eligible' else ''))
        if st not in ('eligible', 'bot'):
            head += f'; {st}, not attached to a player ({"; ".join(x["eligibility"]["reasons"][:2])})'
        link = x.get('link') or {}
        if link.get('status') == 'model-level':
            head += (f'; a separate model-level assessment of {name(link["player"])}\'s model and reasoning level, '
                     f'not that tournament player ({", ".join(link["differences"])})')
        lines.append(head)
        for c, y in (x.get('categories') or {}).items():
            if c == x.get('scored_category'):
                continue
            iv = y.get('interval95')
            lines.append(f'      {c}: {y["correct"]}/{y["answered"]}' + (f' (95% {iv[0]:.0%}-{iv[1]:.0%} of its '
                                                                          f'generator\'s spots)' if iv else ' (descriptive)'))
    return '\n'.join(lines)


def summary(r) -> str:
    """The verdicts, the standings, poker fundamentals (when a quiz was run) and the caveats, as the CLI
    prints them."""
    labels = r.get('labels')
    fund = fundamentals_text(r.get('fundamentals'), labels)
    return (verdicts(r, labels) + '\n\n' + table(r, labels) + ('\n\n' + fund if fund else '') + '\n\n'
            + '\n'.join(f'- {n}' for n in r['notes']))
