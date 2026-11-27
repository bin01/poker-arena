"""The stopping rule keeps its promise: with no edge, a pairing played round by round under the default design
is called for either player at most alpha_pairing of the time (plus Monte Carlo error), for normal, heavy-tailed
(t with 3 df) and poker-like (many exact zeros, heavy tails) per-deal results; and the family of pairings stays
within alpha. Every verdict comes from ratings.decide itself, look by look, as the runner calls it."""
import math
import time
import unittest

import numpy as np

from poker_arena.ratings import DEFAULT_DESIGN, boundary, decide, nominal_alpha, t_ppf

N_PAIRINGS = 3
SEQUENCES = 2400
R, LOOKS = DEFAULT_DESIGN['round_deals'], DEFAULT_DESIGN['max_rounds']


def noise(kind, rng, n):
    if kind == 'normal':
        return rng.standard_normal(n)
    if kind == 't3':
        return rng.standard_t(3, n)
    # poker-like: about 40% of duplicate deals cancel exactly, the rest heavy-tailed
    return np.where(rng.random(n) < 0.4, 0.0, rng.standard_t(3, n))


def play(xs, design=DEFAULT_DESIGN, n_pairings=N_PAIRINGS):
    """The verdict a pairing ends with when it is checked after each round and stops when decided."""
    for k in range(1, design['max_rounds'] + 1):
        v = decide(xs[:k * design['round_deals']], design, n_pairings, k)
        if v['verdict'] in ('a', 'b', 'even', 'exhausted'):
            return v
    return v


def false_rate(kind, seed, edge=0.0, sequences=SEQUENCES, design=DEFAULT_DESIGN):
    rng = np.random.default_rng(seed)
    n = design['round_deals'] * design['max_rounds']
    calls = [play(350 * noise(kind, rng, n) + edge, design)['verdict'] for _ in range(sequences)]
    return np.array([c in ('a', 'b') for c in calls]), calls


class TestCalibration(unittest.TestCase):
    def test_t_quantiles(self):
        self.assertAlmostEqual(t_ppf(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(t_ppf(0.975, 10), 2.2281, places=3)
        self.assertAlmostEqual(t_ppf(0.995, 29), 2.7564, places=3)
        self.assertAlmostEqual(t_ppf(0.975, None), 1.95996, places=4)

    def test_boundary_deterministic_cached_and_sensible(self):
        t0 = time.time()
        a = boundary(0.05 / 3, 50, 6, 10)
        self.assertEqual(a, boundary(0.05 / 3, 50, 6, 10))
        self.assertLess(time.time() - t0, 10)
        bars = [boundary(0.05 / 3, 50, 6, 10, k) for k in range(1, 7)]
        self.assertEqual(bars, sorted(bars, reverse=True))           # later looks have more blocks, lower bars
        single = t_ppf(1 - 0.05 / 3 / 2, 29)
        self.assertGreater(bars[-1], single)                         # six looks cost more than one
        na = nominal_alpha(0.05 / 3, 50, 6, 10)
        self.assertTrue(0.05 / 3 / 6 < na < 0.05 / 3)                # between Bonferroni over looks and no correction
        self.assertEqual(boundary(0.05, 50, 1, 10), t_ppf(0.975, 4))  # one look: the plain t quantile

    def test_null_false_winner_rate(self):
        alpha_p = DEFAULT_DESIGN['alpha'] / N_PAIRINGS
        for i, kind in enumerate(('normal', 't3', 'poker')):
            called, _ = false_rate(kind, seed=100 + i)
            se = math.sqrt(alpha_p * (1 - alpha_p) / len(called))
            with self.subTest(kind=kind):
                self.assertLessEqual(called.mean(), alpha_p + 2.5 * se)
                fam = called.reshape(-1, N_PAIRINGS).any(axis=1)        # any false winner in a 3-pairing tournament
                fse = math.sqrt(0.05 * 0.95 / len(fam))
                self.assertLessEqual(fam.mean(), 0.05 + 2.5 * fse)

    def test_fixed_single_look(self):
        design = {'alpha': 0.05, 'round_deals': 100, 'max_rounds': 1, 'block': 10}
        called, calls = false_rate('t3', seed=7, sequences=1500, design=design)
        self.assertLessEqual(called.mean(), 0.05 / 3 + 2.5 * math.sqrt(0.0167 * 0.983 / 1500))
        self.assertNotIn('even', calls)                               # 100 deals can never show equivalence

    def test_power_against_a_real_edge(self):
        called, calls = false_rate('normal', seed=3, edge=100.0, sequences=300)
        self.assertGreater(called.mean(), 0.85)
        self.assertEqual(sum(c == 'b' for c in calls), 0)             # never the wrong player

    def test_even_needs_a_tight_interval(self):
        rng = np.random.default_rng(5)
        xs = 20 * rng.standard_normal(300)                            # tiny spread: equivalence is provable
        self.assertEqual(play(xs)['verdict'], 'even')
        xs = 350 * rng.standard_normal(300)                           # typical spread: it never is
        self.assertIn(play(xs)['verdict'], ('exhausted', 'a', 'b'))


if __name__ == '__main__':
    unittest.main()
