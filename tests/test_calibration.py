"""The stopping rule and the intervals keep their promises under the dependence a real tournament has.

Null false-winner rate: with no edge, a pairing played round by round is called for either player at most
alpha_pairing of the time (plus Monte Carlo error), for normal, heavy-tailed and poker-like per-deal results,
with memory inside a conversation (AR(1)) and conversation effects (fresh matches that differ), and the
family of pairings stays within alpha when pairings share their cards. Strength intervals from `rate` cover
the true strengths in seeded correlated tournaments. The fast versions run on every test run; set
POKER_ARENA_HEAVY=1 for the larger runs whose numbers docs/methodology.md reports."""
import math
import os
import time
import unittest

import numpy as np

from poker_arena import ratings as R
from poker_arena.ratings import boundary, decide, nominal_alpha, normalize_design, t_ppf

HEAVY = bool(os.environ.get('POKER_ARENA_HEAVY'))
N_PAIRINGS = 3
DESIGN = normalize_design({'deals': 50, 'design': {'max_rounds': 6}, 'replicates': 2,
                           'matchups': [['a', 'b'], ['a', 'c'], ['b', 'c']]})
SD = 350.0


def noise(kind, rng, shape):
    if kind == 'normal':
        return rng.standard_normal(shape)
    if kind == 't3':
        return rng.standard_t(3, shape) / math.sqrt(3)
    # poker-like: about 40% of duplicate deals cancel exactly, the rest heavy-tailed (unit variance overall)
    return np.where(rng.random(shape) < 0.4, 0.0, rng.standard_t(3, shape) / math.sqrt(3 * 0.6))


def conversations(rng, n, C, deals, kind='normal', rho=0.0, su=0.0, edge=0.0, common=None):
    """n sequences of C conversations of `deals` deals each, in bb/100: per-deal noise of sd SD with AR(1)
    memory rho inside a conversation, a conversation effect of sd su x (SD / sqrt(deals)), an optional shared
    deal effect `common` (added as is), and a true edge."""
    e = noise(kind, rng, (n, C, deals))
    if rho:
        for k in range(1, deals):
            e[..., k] = rho * e[..., k - 1] + math.sqrt(1 - rho * rho) * e[..., k]
    x = SD * e + edge + su * SD / math.sqrt(deals) * rng.standard_normal((n, C, 1))
    return x if common is None else x + common


def looks_t(x, reps):
    """The conversation-level t at every look (as ratings.decide computes it), x being sequences x conversations
    x deals with conversations in play order, `reps` per round."""
    m_c = x.mean(-1)
    out = []
    for C in range(reps, m_c.shape[1] + 1, reps):
        if C < R.MIN_CONVERSATIONS:
            out.append(np.zeros(len(x)))
            continue
        z = m_c[:, :C]
        out.append(z.mean(1) / np.sqrt(z.var(1, ddof=1) / C))
    return np.stack(out, 1)


def called(x, design=DESIGN):
    """(called for anyone, called for a) per sequence under the design."""
    T = looks_t(x, design['replicates'])
    hit = np.abs(T) >= R._bars(design['nominal_alpha'], design['max_rounds'], design['replicates'])
    first = hit.argmax(1)
    return hit.any(1), hit.any(1) & (T[np.arange(len(T)), first] > 0)


def bound(p, n, z=2.5):
    return p + z * math.sqrt(p * (1 - p) / n)


class TestBoundary(unittest.TestCase):
    def test_t_quantiles(self):
        self.assertAlmostEqual(t_ppf(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(t_ppf(0.975, 10), 2.2281, places=3)
        self.assertAlmostEqual(t_ppf(0.995, 29), 2.7564, places=3)
        self.assertAlmostEqual(t_ppf(0.975, None), 1.95996, places=4)

    def test_deterministic_cached_and_sensible(self):
        t0 = time.time()
        a = boundary(0.05 / 3, 50, 6, 10, replicates=2)
        self.assertEqual(a, boundary(0.05 / 3, 50, 6, 10, replicates=2))
        self.assertLess(time.time() - t0, 10)
        bars = [boundary(0.05 / 3, 50, 6, 10, k) for k in range(1, 7)]
        self.assertEqual(bars[0], math.inf)                          # one conversation: no verdict possible
        self.assertEqual(bars, sorted(bars, reverse=True))           # more conversations, lower bars
        na = nominal_alpha(0.05 / 3, 50, 6, 10)
        self.assertTrue(0.05 / 3 / 6 < na < 0.05 / 3)                # between Bonferroni over looks and no correction
        self.assertEqual(boundary(0.05, 50, 1, 10, replicates=2), t_ppf(0.975, 1))  # one look: the plain quantile
        self.assertEqual(boundary(0.05, 50, 1, 10), math.inf)

    def test_decide_matches_the_simulated_statistic(self):
        rng = np.random.default_rng(1)
        x = conversations(rng, 1, 6, 50, su=0.5)[0]
        T = looks_t(x[None], 2)[0]
        for look in (1, 2, 3):
            v = decide(x[:2 * look].ravel(), DESIGN, N_PAIRINGS, look, groups=np.repeat(np.arange(2 * look), 50))
            self.assertAlmostEqual(v['t'], T[look - 1], places=1)
            self.assertEqual((v['conversations'], v['df'], v['basis']), (2 * look, 2 * look - 1, 'conversations'))


class TestNullFalseWinners(unittest.TestCase):
    """Vectorized: the same statistic and bars as decide, over many sequences and dependence settings."""
    N = 200_000 if HEAVY else 12_000
    SETTINGS = [('normal', 0, 0), ('t3', 0, 0), ('poker', 0, 0), ('normal', 0.3, 0), ('normal', 0, 1.0),
                ('poker', 0.3, 0.5), ('t3', 0, 2.0)]

    def test_dependence_settings(self):
        ap = DESIGN['alpha_pairing']
        C = DESIGN['max_rounds'] * DESIGN['replicates']
        rates = {}
        for i, (kind, rho, su) in enumerate(self.SETTINGS):
            anyone, _ = called(conversations(np.random.default_rng(100 + i), self.N, C, 50, kind, rho, su))
            rates[f'{kind}/rho={rho}/su={su}'] = round(anyone.mean(), 4)
            with self.subTest(kind=kind, rho=rho, su=su):
                self.assertLessEqual(anyone.mean(), bound(ap, self.N))
        if HEAVY:
            print(f'\nnull false-winner rate per pairing (target {ap:.4f}): {rates}')

    def test_shared_cards_family(self):
        """Three pairings among three players on the same slates: a deal that suits one style moves every
        pairing with that player. No edges; any false verdict in the tournament at most alpha."""
        n = self.N // 4
        C = DESIGN['max_rounds'] * DESIGN['replicates']
        rng = np.random.default_rng(7)
        z = rng.standard_normal((n, C, 50)) * SD
        beta = {'a': 0.8, 'b': 0.0, 'c': -0.8}                     # style loadings on the shared deal effect
        anyone = np.zeros(n, bool)
        for p, q in (('a', 'b'), ('a', 'c'), ('b', 'c')):
            anyone |= called(conversations(rng, n, C, 50, 'poker', 0.2, 0.5, common=(beta[p] - beta[q]) * 0.6 * z))[0]
        self.assertLessEqual(anyone.mean(), bound(DESIGN['alpha'], n))
        if HEAVY:
            print(f'\nany false verdict, 3 correlated pairings: {anyone.mean():.4f} (target 0.05)')

    def test_end_to_end_through_decide(self):
        """The runner's loop: decide on all deals so far after each round, stop on a verdict."""
        n = 3000 if HEAVY else 300
        rng = np.random.default_rng(11)
        reps, rounds = DESIGN['replicates'], DESIGN['max_rounds']
        calls = []
        for _ in range(n):
            x = conversations(rng, 1, rounds * reps, 50, 'poker', 0.3, 1.0)[0]
            for r in range(1, rounds + 1):
                v = decide(x[:r * reps].ravel(), DESIGN, N_PAIRINGS, r, groups=np.repeat(np.arange(r * reps), 50))
                if v['verdict'] in ('a', 'b', 'even', 'exhausted'):
                    break
            calls.append(v['verdict'])
        rate = np.mean([c in ('a', 'b') for c in calls])
        self.assertLessEqual(rate, bound(DESIGN['alpha_pairing'], n, 3))
        if HEAVY:
            print(f'\nend to end through decide: {rate:.4f} false winners per pairing')


class TestPower(unittest.TestCase):
    def test_power_against_a_real_edge(self):
        C = DESIGN['max_rounds'] * DESIGN['replicates']
        anyone, right = called(conversations(np.random.default_rng(3), 4000, C, 50, edge=100.0))
        self.assertGreater(right.mean(), 0.95)
        self.assertEqual((anyone & ~right).sum(), 0)                 # never the wrong player

    def test_design_power_matches_simulation(self):
        C = DESIGN['max_rounds'] * DESIGN['replicates']
        _, right = called(conversations(np.random.default_rng(4), 8000, C, 50, edge=DESIGN['target_edge_bb100']))
        self.assertAlmostEqual(right.mean(), DESIGN['power_at_max'], delta=0.03)

    def test_even_needs_a_tight_interval(self):
        rng = np.random.default_rng(5)
        groups = np.repeat(np.arange(4), 50)
        self.assertEqual(decide(20 * rng.standard_normal(200), DESIGN, N_PAIRINGS, 2, groups=groups)['verdict'], 'even')
        self.assertEqual(decide(SD * rng.standard_normal(200), DESIGN, N_PAIRINGS, 2, groups=groups)['verdict'], 'open')

    def test_one_conversation_never_decides(self):
        rng = np.random.default_rng(6)
        v = decide(list(400 + 50 * rng.standard_normal(100)), R._fixed_design(100, 3), 3, 1)
        self.assertEqual((v['verdict'], v['basis'], v['conversations']), ('exhausted', 'blocks', 1))
        self.assertIsNotNone(v['ci95'])
        self.assertIn('conversation', v['reason'])


def tournament(rng, players, true, slates, deals=50, su=0.5, rho=0.2, common=0.6, kind='poker'):
    """Duplicate deals of a round robin played on `slates` shared slates (every pairing gets the same seeds in
    a slate). A pair's deal result is the strength difference plus a shared-card effect through style
    loadings, a conversation effect and AR(1) noise."""
    beta = dict(zip(players, np.linspace(-1, 1, len(players))))
    z = rng.standard_normal((1, slates, deals)) * SD
    out = []
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            x = conversations(rng, 1, slates, deals, kind, rho, su, edge=true[a] - true[b],
                              common=(beta[a] - beta[b]) * common * z)[0]
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
