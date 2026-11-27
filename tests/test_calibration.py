"""The stopping rule keeps its promise, and says where it stops keeping it.

Null false-winner rate: with no edge, a pairing played round by round under the default 3-model design is
called for either player at most alpha_pairing of the time (plus Monte Carlo error): for normal, heavy-tailed,
skewed and poker-like per-deal results, memory inside a conversation (AR(1)), conversation effects of normal,
exponential, lognormal and rare-large-loss shape up to the calibrated size, and, through `decide` with its
sign-flip arm, for symmetric results of any size; the family of pairings stays within alpha when pairings
share their cards. The round 3 reviewer's counterexamples (two constant conversations; a zero-mean
conversation distribution of +50 nine times in ten and -450 otherwise) no longer produce verdicts. Power under
the planning scenarios matches the design's numbers. Strength intervals from `rate` cover the true strengths
in seeded correlated tournaments. The fast versions run on every test run; POKER_ARENA_HEAVY=1 runs the larger
ones and prints the numbers docs/methodology.md reports, including the cases beyond the calibrated family."""
import math
import os
import unittest

import numpy as np

from poker_arena import ratings as R
from poker_arena.ratings import boundary, decide, nominal_alpha, normalize_design, t_ppf

HEAVY = bool(os.environ.get('POKER_ARENA_HEAVY'))
THREE = [['a', 'b'], ['a', 'c'], ['b', 'c']]
DESIGN = normalize_design({'deals': 50, 'design': {}, 'replicates': 4, 'matchups': THREE})
OLD = normalize_design({'deals': 50, 'design': {'max_rounds': 6}, 'replicates': 2, 'matchups': THREE})
SD = 350.0
DEALS = 50


def deal_noise(kind, rng, shape):
    if kind == 'normal':
        return rng.standard_normal(shape)
    if kind == 't3':
        return rng.standard_t(3, shape) / math.sqrt(3)
    if kind == 'skewed':
        return rng.exponential(1.0, shape) - 1.0
    # poker-like: about 40% of duplicate deals cancel exactly, the rest heavy-tailed (unit variance overall)
    return np.where(rng.random(shape) < 0.4, 0.0, rng.standard_t(3, shape) / math.sqrt(3 * 0.6))


def effect(shape, rng, size):
    """A unit-variance, zero-mean conversation effect."""
    if shape == 'normal':
        return rng.standard_normal(size)
    if shape == 'exponential':
        return rng.exponential(1.0, size) - 1
    if shape == 'lognormal':
        return (np.exp(rng.standard_normal(size)) - math.exp(0.5)) / math.sqrt((math.e - 1) * math.e)
    if shape == 'rare_loss':                    # the reviewer's +50 nine times in ten, -450 otherwise, in sd units
        return np.where(rng.random(size) < 0.9, 50.0, -450.0) / 150.0
    return np.where(rng.random(size) < 0.5, 1.0, -1.0)     # symmetric two-point


def conversations(rng, n, C, kind='normal', rho=0.0, shape=None, share=0.0, edge=0.0, card=1.0, means=True,
                  common=None, deals=DEALS):
    """n sequences of C conversations of `deals` deals in bb/100: per-deal noise of sd SD x card with AR(1) memory
    rho, a conversation effect of `shape` holding `share` of a conversation mean's variance, an optional shared
    deal effect `common` (n x C x DEALS, added as is), and a true edge. Conversation means (n x C) when
    `means`, else the deals."""
    card_sd = SD * card * math.sqrt(R.design_effect(rho, deals) / deals)
    tau = card_sd * math.sqrt(share / (1 - share)) if share < 1 else SD / math.sqrt(deals)
    out = []
    step = max(1, 2_000_000 // (C * deals))
    for start in range(0, n, step):
        k = min(step, n - start)
        e = deal_noise(kind, rng, (k, C, deals))
        if rho:
            for d in range(1, deals):
                e[..., d] = rho * e[..., d - 1] + math.sqrt(1 - rho * rho) * e[..., d]
        x = SD * card * e + edge
        if shape:
            x = x + tau * effect(shape, rng, (k, C, 1))
        if common is not None:
            x = x + common[start:start + k]
        out.append(x.mean(-1) if means else x)
    return np.concatenate(out)


def t_arm(m, design=DESIGN):
    """(called for anyone, called for a): the t arm of the procedure on conversation means (n x C)."""
    reps, rounds = design['replicates'], design['max_rounds']
    a1, _, _, eligible = R._calibrate(design['alpha_pairing'], rounds, reps)
    hit = np.zeros((len(m), rounds), bool)
    T = np.zeros((len(m), rounds))
    for j in range(rounds):
        C = (j + 1) * reps
        z = m[:, :C]
        mu, s = z.mean(1), z.std(1, ddof=1)
        ok = s > 1e-9 * (1 + np.abs(mu))
        T[:, j] = np.where(ok, mu / np.where(ok, s, 1) * math.sqrt(C), 0.0)
        if eligible[j]:
            hit[:, j] = np.abs(T[:, j]) >= t_ppf(1 - a1 / 2, C - 1)
    first = hit.argmax(1)
    return hit.any(1), hit.any(1) & (T[np.arange(len(T)), first] > 0)


def through_decide(x, design=DESIGN):
    """The verdict of each sequence (n x C x DEALS deals) as the runner gets it: decide on every deal at the
    last look reports the first look at which the procedure decided."""
    reps, rounds = design['replicates'], design['max_rounds']
    groups = np.repeat(np.arange(rounds * reps), DEALS)
    return [decide(list(seq.ravel()), design, len(THREE), rounds, groups=groups)['verdict'] for seq in x]


def bound(p, n, z=2.5):
    return p + z * math.sqrt(p * (1 - p) / n)


AP = DESIGN['alpha_pairing']
C_MAX = DESIGN['max_rounds'] * DESIGN['replicates']


class TestBoundary(unittest.TestCase):
    def test_t_quantiles_and_p_values(self):
        self.assertAlmostEqual(t_ppf(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(t_ppf(0.975, 10), 2.2281, places=3)
        self.assertAlmostEqual(t_ppf(0.995, 29), 2.7564, places=3)
        self.assertAlmostEqual(t_ppf(0.975, None), 1.95996, places=4)
        for df in (1, 2, 3, 8, 11, 35):
            for t in (0.3, 2.0, 4.5, 12.0):
                self.assertAlmostEqual(float(R._p_two_sided(np.array([t]), df)[0]), 2 * R.t_sf(t, df), places=12)

    def test_deterministic_and_sensible(self):
        a = boundary(AP, 50, DESIGN['max_rounds'], 10, replicates=4)
        self.assertEqual(a, boundary(AP, 50, DESIGN['max_rounds'], 10, replicates=4))
        bars = [boundary(AP, 50, DESIGN['max_rounds'], 10, k, 4) for k in range(1, DESIGN['max_rounds'] + 1)]
        first = DESIGN['min_conversations'] // 4
        self.assertTrue(all(b == math.inf for b in bars[:first - 1]))   # no verdict before min_conversations
        live = bars[first - 1:]
        self.assertTrue(all(math.isfinite(b) for b in live))
        self.assertEqual(live, sorted(live, reverse=True))               # more conversations, lower bars
        self.assertTrue(AP / DESIGN['max_rounds'] / 10 < nominal_alpha(AP, 50, DESIGN['max_rounds'], 10, 4) < AP)
        lv = DESIGN['calibration']['levels']
        self.assertAlmostEqual(DESIGN['nominal_alpha'], min(lv.values()), places=6)
        self.assertLess(lv['rare_loss'], lv['normal'])                   # skew costs a lower level

    def test_decide_matches_the_vectorized_statistic(self):
        rng = np.random.default_rng(1)
        x = conversations(rng, 1, C_MAX, 'poker', 0.2, 'normal', 0.3, means=False)[0]
        groups = np.repeat(np.arange(C_MAX), DEALS)
        for look in (3, 5, DESIGN['max_rounds']):
            C = 4 * look
            v = decide(x[:C].ravel(), DESIGN, 3, look, groups=groups[:C * DEALS])
            m = x[:C].mean(1)
            self.assertAlmostEqual(v['t'], m.mean() / m.std(ddof=1) * math.sqrt(C), places=1)
            self.assertEqual((v['conversations'], v['df'], v['basis']), (C, C - 1, 'conversations'))


class TestReviewerCounterexamples(unittest.TestCase):
    """Round 3, issue 1: two constant conversations decided a winner with zero uncertainty, and a zero-mean
    conversation distribution gave a false first-look winner 82% of the time."""

    def test_two_constant_conversations(self):
        for design in (DESIGN, OLD):
            v = decide([50.0] * 100, design, 3, 1, groups=['c1'] * 50 + ['c2'] * 50)
            self.assertEqual(v['verdict'], 'early')

    def test_constant_conversations_never_decide(self):
        n = DESIGN['max_rounds'] * 4
        for value in (50.0, 50.1, -0.3):
            xs = [value] * (n * DEALS)
            v = decide(xs, DESIGN, 3, DESIGN['max_rounds'], groups=np.repeat(np.arange(n), DEALS))
            self.assertEqual(v['verdict'], 'exhausted')
            self.assertIn('no spread', v['reason'])

    def test_zero_mean_skewed_conversations(self):
        """Every deal of a conversation is +50 (nine times in ten) or -450: mean zero, no card noise."""
        n = 3000 if HEAVY else 300
        rng = np.random.default_rng(31)
        x = conversations(rng, n, C_MAX, shape='rare_loss', share=1.0, card=0.0, means=False)
        first_look = [decide(list(s[:2].ravel()), OLD, 3, 1, groups=np.repeat([0, 1], DEALS))['verdict'] for s in x[:200]]
        self.assertEqual(sum(v in ('a', 'b') for v in first_look), 0)    # round 3 code: 82% here
        calls = through_decide(x)
        rate = np.mean([c in ('a', 'b') for c in calls])
        self.assertLessEqual(rate, bound(AP, n, 3))
        anyone, _ = t_arm(x.mean(-1))
        self.assertLessEqual(anyone.mean(), bound(AP, n, 3))
        if HEAVY:
            print(f'\nreviewer distribution, no card noise: {rate:.4f} through decide')


FAMILY_SETTINGS = [                               # (deals, AR(1), conversation effect shape, its share)
    ('normal', 0.0, None, 0.0), ('t3', 0.0, None, 0.0), ('poker', 0.0, None, 0.0), ('skewed', 0.0, None, 0.0),
    ('poker', 0.3, 'normal', 0.5), ('normal', 0.0, 'exponential', 0.5), ('normal', 0.0, 'lognormal', 0.5),
    ('normal', 0.0, 'rare_loss', 0.5), ('poker', 0.2, 'rare_loss', 0.4), ('normal', 0.2, 'normal', 0.25),
]
BEYOND = [('normal', 0.0, 'exponential', 0.8), ('normal', 0.0, 'lognormal', 0.8), ('normal', 0.0, 'rare_loss', 0.8),
          ('normal', 0.0, 'rare_loss', 0.9), ('normal', 0.0, 'symmetric', 0.9)]


class TestNullFalseWinners(unittest.TestCase):
    N = 200_000 if HEAVY else 12_000

    def test_calibrated_family(self):
        """The t arm alone, vectorized: within alpha_pairing for every setting the bars are calibrated for."""
        rates = {}
        for i, (kind, rho, shape, share) in enumerate(FAMILY_SETTINGS):
            m = conversations(np.random.default_rng(100 + i), self.N, C_MAX, kind, rho, shape, share)
            anyone, _ = t_arm(m)
            rates[f'{kind}/ar={rho}/{shape}@{share}'] = round(float(anyone.mean()), 4)
            with self.subTest(kind=kind, rho=rho, shape=shape, share=share):
                self.assertLessEqual(anyone.mean(), bound(AP, self.N))
        if HEAVY:
            print(f'\nt arm, null false-winner rate per pairing (target {AP:.4f}): {rates}')

    def test_symmetric_results_of_any_size_through_decide(self):
        """Symmetric conversation results far beyond the calibrated size (a two-point conversation effect
        holding 90% of the variance): the sign-flip arm keeps the procedure within alpha_pairing."""
        n = 3000 if HEAVY else 300
        x = conversations(np.random.default_rng(41), n, C_MAX, shape='symmetric', share=0.9, means=False)
        rate = np.mean([c in ('a', 'b') for c in through_decide(x)])
        self.assertLessEqual(rate, bound(AP, n, 3))
        if HEAVY:
            t_only, _ = t_arm(conversations(np.random.default_rng(42), self.N, C_MAX, shape='symmetric', share=0.9))
            print(f'\nsymmetric two-point, share 0.9: {rate:.4f} through decide, t arm alone {t_only.mean():.4f}')

    def test_end_to_end_through_decide(self):
        n = 3000 if HEAVY else 200
        x = conversations(np.random.default_rng(11), n, C_MAX, 'poker', 0.3, 'exponential', 0.5, means=False)
        rate = np.mean([c in ('a', 'b') for c in through_decide(x)])
        self.assertLessEqual(rate, bound(AP, n, 3))
        if HEAVY:
            print(f'\nend to end through decide (poker, AR 0.3, exponential effects): {rate:.4f}')

    def test_shared_cards_family(self):
        """Three pairings among three players on the same slates: a deal that suits one style moves every
        pairing with that player. No edges; any false verdict in the tournament at most alpha."""
        n = self.N // 8
        rng = np.random.default_rng(7)
        z = rng.standard_normal((n, C_MAX, DEALS)).astype(np.float32) * SD
        beta = {'a': 0.8, 'b': 0.0, 'c': -0.8}                     # style loadings on the shared deal effect
        anyone = np.zeros(n, bool)
        for p, q in THREE:
            anyone |= t_arm(conversations(rng, n, C_MAX, 'poker', 0.2, 'normal', 0.3,
                                          common=(beta[p] - beta[q]) * 0.6 * z))[0]
        self.assertLessEqual(anyone.mean(), bound(DESIGN['alpha'], n))
        if HEAVY:
            print(f'\nany false verdict, 3 correlated pairings: {anyone.mean():.4f} (target 0.05)')

    @unittest.skipUnless(HEAVY, 'heavy: the cases beyond the calibrated family, for docs/methodology.md')
    def test_report_beyond_the_family(self):
        rows = {}
        for i, (kind, rho, shape, share) in enumerate(BEYOND):
            t_only, _ = t_arm(conversations(np.random.default_rng(200 + i), self.N, C_MAX, kind, rho, shape, share))
            x = conversations(np.random.default_rng(300 + i), 2000, C_MAX, kind, rho, shape, share, means=False)
            both = np.mean([c in ('a', 'b') for c in through_decide(x)])
            rows[f'{shape}@{share}'] = (round(float(t_only.mean()), 4), round(float(both), 4))
        share = 150 ** 2 / (150 ** 2 + (SD / math.sqrt(DEALS)) ** 2)
        t_only, _ = t_arm(conversations(np.random.default_rng(399), self.N, C_MAX, shape='rare_loss', share=share))
        x = conversations(np.random.default_rng(400), 2000, C_MAX, shape='rare_loss', share=share, means=False)
        rows['reviewer +50/-450 with card noise'] = (round(float(t_only.mean()), 4),
                                                     round(float(np.mean([c in ('a', 'b') for c in through_decide(x)])), 4))
        print(f'\nbeyond the family (t arm alone, through decide): {rows}')


REG = R.registration(DESIGN, created=0)
BOTS = {'registration': REG, 'implementation_changes': [], 'conversations': 0,
        'players': {p: {'kind': 'bot', 'identity': 'bot', 'audit': 'complete', 'protocol_consistent': True}
                    for p in ('x', 'y')}}


def planning_deals(rng, n, edge, icc, rho, design=DESIGN, kind='normal'):
    """n pairings of the design's conversations as the planner models them (deals of spread SD, a share icc
    of their variance a normal conversation effect, AR(1) memory rho, a true edge), or with poker-like deals."""
    reps, rounds, R_ = design['replicates'], design['max_rounds'], design['round_deals']
    card = SD * math.sqrt(1 - icc)
    e = deal_noise(kind, rng, (n, rounds * reps, R_))
    for d in range(1, R_):
        e[..., d] = rho * e[..., d - 1] + math.sqrt(1 - rho * rho) * e[..., d]
    return edge + card * e + SD * math.sqrt(icc) * rng.standard_normal((n, rounds * reps, 1))


def look_stats(x, design=DESIGN):
    """What decide and assumptions compute at each look from conversations of equal length (n x C x deals):
    the columns `ratings.verdicts_of` reads."""
    reps, block = design['replicates'], design['block']
    m = x.mean(-1)
    k = -(-x.shape[-1] // block)
    sums = np.add.reduceat(x, block * np.arange(k), axis=-1)
    sizes = np.minimum(block, x.shape[-1] - block * np.arange(k))
    w = k / (k - 1) * ((sums - sizes * m[..., None]) ** 2).sum(-1) / x.shape[-1] ** 2
    cols = []
    for j in range(design['max_rounds']):
        C = (j + 1) * reps
        mean, var = m[:, :C].mean(1), m[:, :C].var(1, ddof=1)
        share = np.where(var > 0, np.round(np.maximum(0, 1 - w[:, :C].mean(1) / np.where(var > 0, var, 1)), 2), np.nan)
        cols.append((mean, np.sqrt(var / C), share))
    return cols


def through_adjudicate(x, design=DESIGN, n_pairings=3):
    """Each pairing judged as the runner judges it: `adjudicate` after every round on the deals so far, stopping
    at the first terminal verdict. [(verdict, counts_toward_leader)]."""
    reps, rounds, R_ = design['replicates'], design['max_rounds'], design['round_deals']
    out = []
    for seq in x:
        for r in range(1, rounds + 1):
            C = r * reps
            data = {'a': 'x', 'b': 'y', 'xs': list(seq[:C].ravel()), 'groups': list(np.repeat(np.arange(C), R_)),
                    'rejected': [], 'evidence_hash': 'simulated'}
            v = R.adjudicate(data, design, n_pairings, r, BOTS)
            if v['verdict'] in R.TERMINAL:
                break
        out.append((v['verdict'], v['counts_toward_leader']))
    return out


class TestPower(unittest.TestCase):
    """R5-#7: the power a plan advertises is the full adjudicator's: both arms, the minimum, the looks, and the
    diagnostics a verdict needs to count. Checked three ways: the planner's decision logic against `adjudicate`
    on the same data, the planner's numbers against `adjudicate` on fresh simulated pairings, and the
    tournament's chance of a decided leader against a joint simulation."""

    def test_planner_logic_is_the_adjudicators(self):
        """On the same simulated pairings, `verdicts_of` (what the planner applies) and `adjudicate` (look by look,
        as the runner calls it) reach the same verdict and the same counting decision."""
        n = 1000 if HEAVY else 100
        x = planning_deals(np.random.default_rng(70), n, DESIGN['target_edge_bb100'], 0.01, 0.2)
        sign, counts, _ = R.verdicts_of(look_stats(x), DESIGN)
        got = through_adjudicate(x)
        words = {1: 'a', -1: 'b', 2: 'even', 0: 'exhausted'}
        agree = np.mean([(words[s], bool(c)) == (v, bool(k)) for s, c, (v, k) in zip(sign, counts, got)])
        self.assertGreaterEqual(agree, 0.95)
        if HEAVY:
            print(f'\nplanner logic vs adjudicate on the same pairings: {agree:.3f} agree ({n})')

    def test_advertised_power_is_the_adjudicators(self):
        """The design's counting power (moderate dependence, at the target) against fresh pairings through
        `adjudicate`: within Monte Carlo error. Heavy: also the other scenarios and poker-like deals."""
        n = 4000 if HEAVY else 300
        cases = [('moderate', 'normal')] + ([('optimistic', 'normal'), ('pessimistic', 'normal'), ('moderate', 'poker')]
                                           if HEAVY else [])
        sc = {x['scenario']: x for x in DESIGN['power_scenarios']}
        for i, (name, kind) in enumerate(cases):
            s = sc[name]
            x = planning_deals(np.random.default_rng(80 + i), n, DESIGN['target_edge_bb100'], s['assumed_icc'],
                               s['assumed_ar1'], kind=kind)
            got = through_adjudicate(x)
            counting = np.mean([v == 'a' and k for v, k in got])
            right = np.mean([v == 'a' for v in (g[0] for g in got)])
            se = math.sqrt(s['power'] * (1 - s['power']) / n)
            with self.subTest(scenario=name, deals=kind):
                self.assertAlmostEqual(counting, s['power'], delta=3.5 * se + 0.01)
                self.assertFalse(any(v == 'b' for v, _ in got))
            if HEAVY:
                print(f'\ncounting power, {name}, {kind} deals: design {s["power"]:.3f} (decided {s["power_decided"]:.3f}), '
                      f'adjudicate {counting:.3f} (decided {right:.3f}), {n} pairings')

    def test_design_is_powered_by_counting_verdicts(self):
        self.assertGreaterEqual(DESIGN['power_at_max'], 0.8)
        fewer = normalize_design({'deals': 50, 'design': {'max_rounds': DESIGN['max_rounds'] - 1}, 'replicates': 4,
                                  'matchups': THREE})
        self.assertLess(fewer['power_at_max'], 0.8)
        for s in DESIGN['power_scenarios']:
            self.assertLessEqual(s['power'], s['power_decided'])
        self.assertIn('complete adjudicator', DESIGN['power_basis'])

    def test_leader_chance(self):
        """The chance of a decided leader under 'one best' is the product of its pairings' counting power; in a
        joint simulation where the leader's pairings share cards (positively correlated), the joint chance is at
        least the product of the pairings' own chances."""
        lead = {x['scenario']: x for x in DESIGN['leader_chances']}
        p = R.procedure(DESIGN, DESIGN['target_edge_bb100'])['counting']
        self.assertAlmostEqual(lead['one best']['p_leader'], p ** 2, places=3)
        nine = normalize_design({'deals': 50, 'design': {}, 'replicates': 4,
                                 'matchups': [[a, b] for i, a in enumerate('abcdefghi') for b in 'abcdefghi'[i + 1:]]})
        self.assertLess({x['scenario']: x for x in nine['leader_chances']}['one best']['p_leader'], lead['one best']['p_leader'])
        if not HEAVY:
            return
        n, T, c = 1500, DESIGN['target_edge_bb100'], 0.6
        rng = np.random.default_rng(90)
        z = rng.standard_normal((n, C_MAX, DEALS)) * SD          # the leader's own luck on the shared cards
        wins = []
        for _ in ('b', 'c'):                                      # the same share of each deal's spread, so each
            x = T + math.sqrt(1 - c * c) * (planning_deals(rng, n, T, 0.01, 0.2) - T) + c * z   # pairing keeps its power
            wins.append(np.array([v == 'a' and k for v, k in through_adjudicate(x)]))
        both, alone = (wins[0] & wins[1]).mean(), (wins[0].mean(), wins[1].mean())
        print(f'\ndecided leader, one best by {T:.0f}, 3 players, pairings sharing {c * c:.0%} of each deal\'s variance: '
              f'{both:.3f}; its pairings alone {alone[0]:.3f} and {alone[1]:.3f} (product {alone[0] * alone[1]:.3f}); '
              f'the design says {lead["one best"]["p_leader"]:.3f}')
        self.assertGreaterEqual(both, alone[0] * alone[1] - 3.5 * math.sqrt(0.25 / n))

    def test_signflip_arm_costs_little_power(self):
        n = 1000 if HEAVY else 120
        x = conversations(np.random.default_rng(60), n, C_MAX, 'normal', 0.2, 'normal', 0.25,
                          edge=DESIGN['target_edge_bb100'], means=False)
        _, right = t_arm(x.mean(-1))
        both = np.mean([c == 'a' for c in through_decide(x)])
        self.assertGreater(both, right.mean() - 3.5 * math.sqrt(0.2 / n))
        if HEAVY:
            print(f'\npower at the target, moderate dependence: t arm {right.mean():.3f}, both arms {both:.3f}')

    def test_power_against_a_real_edge(self):
        anyone, right = t_arm(conversations(np.random.default_rng(3), 4000, C_MAX, edge=100.0))
        self.assertGreater(right.mean(), 0.99)
        self.assertEqual((anyone & ~right).sum(), 0)                 # never the wrong player

    def test_even_needs_a_tight_interval(self):
        rng = np.random.default_rng(5)
        groups = np.repeat(np.arange(24), 50)
        self.assertEqual(decide(20 * rng.standard_normal(1200), DESIGN, 3, 6, groups=groups)['verdict'], 'even')
        self.assertEqual(decide(SD * rng.standard_normal(1200), DESIGN, 3, 6, groups=groups)['verdict'], 'open')

    def test_one_conversation_never_decides(self):
        rng = np.random.default_rng(6)
        v = decide(list(400 + 50 * rng.standard_normal(100)), {**DESIGN, 'max_rounds': 1, 'replicates': 1}, 3, 1)
        self.assertEqual((v['verdict'], v['basis'], v['conversations']), ('early', 'blocks', 1))
        self.assertIsNotNone(v['ci95'])
        self.assertIn('conversation', v['reason'])


EQ_CARD = 100.0 / SD          # per-deal spread 100 bb/100: tight enough for 'even' to be reachable at all
EQ_SHAPES = [(None, 0.0), ('exponential', 0.5), ('lognormal', 0.5), ('rare_loss', 0.5)]


def eq_means(rng, n, edge, shape, share, orient=1, kind='normal', C=C_MAX, means=True, common=None):
    """Conversation means (or deals) with a true edge and the conversation effect in either orientation
    (orient -1 mirrors it: rare large wins instead of rare large losses)."""
    x = conversations(rng, n, C, kind, 0.0, shape, share, edge=orient * edge, card=EQ_CARD, means=means,
                      common=None if common is None else orient * common)
    return orient * x


def tost_arm(m, design=DESIGN, level=None):
    """Whether each sequence of conversation means (n x C) ends 'even' under the t arm of decide: at each
    deciding look a winner first, then the repeated interval inside +-EQUIV_BB100; the first decision stops it."""
    reps, rounds, ap = design['replicates'], design['max_rounds'], design['alpha_pairing']
    a1, _, _, eligible = R._calibrate(ap, rounds, reps)
    aeq = R.equivalence_level(ap, rounds, reps) if level is None else level
    live, even = np.ones(len(m), bool), np.zeros(len(m), bool)
    for j in range(rounds):
        C = (j + 1) * reps
        if not eligible[j]:
            continue
        z = m[:, :C]
        mu, se = z.mean(1), z.std(1, ddof=1) / math.sqrt(C)
        win = live & (np.abs(mu) >= t_ppf(1 - a1 / 2, C - 1) * se)
        q = t_ppf(1 - aeq / 2, C - 1)
        eq = live & ~win & (mu - q * se > -R.EQUIV_BB100) & (mu + q * se < R.EQUIV_BB100)
        even |= eq
        live &= ~(win | eq)
    return even


class TestEquivalence(unittest.TestCase):
    """R5-#8: 'even' (TOST at +-15 bb/100) is a false verdict whenever the true edge is at or beyond the margin.
    Its chance there stays within alpha_pairing on either side, for normal results and for every skewed shape
    of the calibrated family in either orientation (a skewed shape puts most of its crossings on one side, so
    each side is calibrated, not the pair), through the vectorized t arm and through `decide` itself, alone and
    for pairings that share cards."""
    N = 200_000 if HEAVY else 12_000

    def test_false_even_at_and_beyond_the_margin(self):
        rates = {}
        for i, (shape, share) in enumerate(EQ_SHAPES):
            for orient in (1, -1):
                for edge in (15.0, -15.0, 16.0, 20.0):
                    rng = np.random.default_rng([300, i, orient + 1, int(edge) + 100])
                    r = tost_arm(eq_means(rng, self.N, edge, shape, share, orient)).mean()
                    rates[f'{shape or "normal"}{"" if orient > 0 else " mirrored"} @ {edge:+.0f}'] = round(float(r), 4)
                    with self.subTest(shape=shape, orient=orient, edge=edge):
                        self.assertLessEqual(r, bound(AP, self.N))
        if HEAVY:
            print(f'\nfalse even per pairing (t arm, target {AP:.4f}): {rates}')

    def test_calibrating_the_pair_was_not_enough(self):
        """The round 5 level (two-sided, calibrated as a pair) let the heavy side of a skewed shape through."""
        old = R._calibrate(min(1.0, 2 * AP), DESIGN['max_rounds'], 4)[0]
        self.assertGreater(old, R.equivalence_level(AP, DESIGN['max_rounds'], 4))
        worst = max(tost_arm(eq_means(np.random.default_rng([310, o + 1, int(e) + 100]), self.N, e, 'rare_loss', 0.5, o), level=old).mean()
                    for o in (1, -1) for e in (15.0, -15.0))
        self.assertGreater(worst, AP)

    def test_even_is_reachable_where_it_is_true(self):
        r = tost_arm(eq_means(np.random.default_rng(320), self.N // 4, 0.0, None, 0.0)).mean()
        self.assertGreater(r, 0.5)
        p = R.procedure(DESIGN, 0.0, icc=0.0, ar1=0.0, sd=100.0)
        self.assertAlmostEqual(p['even'], r, delta=0.03)
        if HEAVY:
            print(f'\neven at a true edge of 0, per-deal spread 100: t arm {r:.3f}, planner {p["even"]:.3f}; at '
                  f'spread 350 (the default prior): planner {R.procedure(DESIGN, 0.0)["even"]:.3f}')

    def test_false_even_through_decide(self):
        n = 3000 if HEAVY else 300
        x = eq_means(np.random.default_rng(330), n, -15.0, 'rare_loss', 0.5, 1, kind='poker', means=False)
        rate = np.mean([c == 'even' for c in through_decide(x)])
        self.assertLessEqual(rate, bound(AP, n, 3))
        if HEAVY:
            print(f'\nfalse even through decide (poker deals, rare-loss effects, edge -15): {rate:.4f}')

    def test_simultaneous_pairings(self):
        """Three pairings at the margin on shared slates: any false 'even' in the tournament at most alpha."""
        n = self.N // 8
        rng = np.random.default_rng(340)
        z = rng.standard_normal((n, C_MAX, DEALS)).astype(np.float32) * SD * EQ_CARD
        beta = {'a': 0.8, 'b': 0.0, 'c': -0.8}
        anyone = np.zeros(n, bool)
        for (p, q), edge in zip(THREE, (15.0, -15.0, 15.0)):
            x = eq_means(rng, n, edge, 'exponential', 0.5, 1, means=False, common=(beta[p] - beta[q]) * 0.6 * z)
            anyone |= tost_arm(x.mean(-1))
        self.assertLessEqual(anyone.mean(), bound(DESIGN['alpha'], n))
        if HEAVY:
            print(f'\nany false even, 3 pairings at the margin with shared cards: {anyone.mean():.4f} (target 0.05)')


def tournament(rng, players, true, slates, deals=50, su=0.5, rho=0.2, common=0.6, kind='poker'):
    """Duplicate deals of a round robin played on `slates` shared slates (every pairing gets the same seeds in
    a slate). A pair's deal result is the strength difference plus a shared-card effect through style
    loadings, a conversation effect (su x the card noise of a conversation mean) and AR(1) noise."""
    beta = dict(zip(players, np.linspace(-1, 1, len(players))))
    z = rng.standard_normal((1, slates, deals)) * SD
    share = su * su / (1 + su * su)
    out = []
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            x = conversations(rng, 1, slates, kind, rho, 'normal' if su else None, share, true[a] - true[b],
                              means=False, common=(beta[a] - beta[b]) * common * z, deals=deals)[0]
            for s in range(slates):
                for k in range(deals):
                    out.append({'a': a, 'b': b, 'matchup': f'{a}__vs__{b}__r{s + 1}', 'deal': k, 'bb': 2,
                                'net_a': x[s, k] * 4 / 100, 'seed': 1000 + s * deals + k, 'slate': 1000 + s * deals,
                                'round': s + 1, 'replicate': 1, 'planned': deals, 'fp': f'{1000 + s * deals + k}:{k % 2}'})
    return out


class TestCoverage(unittest.TestCase):
    """Strength intervals from rate() in seeded correlated tournaments."""
    TRUE = {'p': 60.0, 'q': 20.0, 'r': -20.0, 's': -60.0}

    def coverage(self, n, slates, **kw):
        rng = np.random.default_rng(2026 + slates)
        hit, total, basis = 0, 0, set()
        for t in range(n):
            r = R.rate(tournament(rng, list(self.TRUE), self.TRUE, slates, **kw), boot=200, seed=t)
            for p in r['players']:
                lo, hi = p['strength_ci95']
                hit += lo <= self.TRUE[p['player']] <= hi
                total += 1
                basis.add(p['ci_basis'])
        return hit / total, basis

    def test_conversations(self):
        cov, basis = self.coverage(400 if HEAVY else 40, 4)
        self.assertEqual(basis, {'conversations'})
        self.assertGreater(cov, 0.88)
        if HEAVY:
            print(f'\ncoverage, 4 slates, conversation effects and shared cards: {cov:.3f}')

    def test_blocks_within_one_conversation(self):
        cov, basis = self.coverage(400 if HEAVY else 40, 1, deals=100, su=0.0, rho=0.0)
        self.assertEqual(basis, {'blocks'})
        self.assertGreater(cov, 0.88)
        if HEAVY:
            print(f'\ncoverage, 1 slate, no conversation effect: {cov:.3f}')


if __name__ == '__main__':
    unittest.main()
