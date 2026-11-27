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


class TestPower(unittest.TestCase):
    def test_planned_power_matches_simulation(self):
        """The design's power under each planning scenario is what simulated tournaments of deals give."""
        n = 20_000 if HEAVY else 3000
        for i, sc in enumerate(DESIGN['power_scenarios']):
            icc, rho = sc['assumed_icc'], sc['assumed_ar1']
            card = SD * math.sqrt(1 - icc) * math.sqrt(R.design_effect(rho, DEALS) / DEALS)
            share = icc * SD * SD / (icc * SD * SD + card * card)
            m = conversations(np.random.default_rng(50 + i), n, C_MAX, 'normal', rho, 'normal', share,
                              edge=DESIGN['target_edge_bb100'], card=math.sqrt(1 - icc))
            _, right = t_arm(m)
            with self.subTest(scenario=sc['scenario']):
                self.assertAlmostEqual(right.mean(), sc['power'], delta=0.035)
            if HEAVY:
                print(f'\npower, {sc["scenario"]}: design {sc["power"]:.3f}, simulated deals {right.mean():.3f}')
        self.assertGreaterEqual(DESIGN['power_at_max'], 0.8)

    def test_signflip_arm_costs_little_power(self):
        n = 1000 if HEAVY else 120
        x = conversations(np.random.default_rng(60), n, C_MAX, 'normal', 0.2, 'normal', 0.25,
                          edge=DESIGN['target_edge_bb100'], means=False)
        _, right = t_arm(x.mean(-1))
        both = np.mean([c == 'a' for c in through_decide(x)])
        self.assertGreater(both, right.mean() - 0.08)
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
