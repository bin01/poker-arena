"""The leaderboard: strengths from duplicate deals, components, separated ranks and the leader rule, repeated
deals averaged, session Elo intervals that never collapse, quarantine, split pots, and the output keys the
viewer reads."""
import json
import unittest

import numpy as np

from poker_arena import ratings as R
from poker_arena.stats import summarize

PLAYER_KEYS = {'player', 'identity', 'kind', 'component', 'strength', 'strength_ci95', 'rank', 'rank_tied_with',
               'net_bb', 'hands', 'deals', 'session_elo', 'session_elo_ci95', 'sessions', 'session_w', 'session_t',
               'session_l', 'quarantined', 'quarantine_reason'}
DECIDE_KEYS = {'deals', 'blocks', 'edge', 'ci95', 't', 'boundary', 'verdict', 'looks_used', 'looks_planned', 'power'}


def deals_between(a, b, edges, n, rng, sd=300.0, matchup=None, seed0=1000):
    """n duplicate deals in which a beats b by `edges` bb/100 on average (bb = 2, so net chips = x / 100 * 2 * 2)."""
    out = []
    for k in range(n):
        x = edges + sd * rng.standard_normal()
        out.append({'a': a, 'b': b, 'matchup': matchup or f'{a}__vs__{b}', 'deal': k, 'bb': 2,
                    'net_a': x * 4 / 100, 'seed': seed0 + k, 'fp': f'{seed0 + k}:{k % 2}', 'planned': n})
    return out


class TestStrength(unittest.TestCase):
    def test_recovers_strengths_and_sums_to_zero(self):
        rng = np.random.default_rng(0)
        true = {'x': 100.0, 'y': 0.0, 'z': -100.0}
        deals = []
        for a, b in (('x', 'y'), ('y', 'z'), ('x', 'z')):
            deals += deals_between(a, b, true[a] - true[b], 400, rng, sd=100)
        r = R.rate(deals, boot=100)
        s = {p['player']: p['strength'] for p in r['players']}
        self.assertAlmostEqual(sum(s.values()), 0, places=0)
        for p, v in true.items():
            self.assertAlmostEqual(s[p], v, delta=15)
        self.assertEqual([p['rank'] for p in r['players']], [1, 2, 3])   # well separated: real ranks
        self.assertEqual(r['leader']['player'], 'x')
        self.assertEqual(r['leader']['over'], 'y')
        for p in r['players']:
            lo, hi = p['strength_ci95']
            self.assertLess(lo, true[p['player']])
            self.assertGreater(hi, true[p['player']])

    def test_no_leader_and_no_ranks_when_close(self):
        rng = np.random.default_rng(1)
        deals = deals_between('x', 'y', 5, 60, rng) + deals_between('y', 'z', 5, 60, rng) + deals_between('x', 'z', 10, 60, rng)
        r = R.rate(deals, boot=100)
        self.assertIsNone(r['leader'])
        self.assertTrue(all(p['rank'] is None for p in r['players']))
        top = r['players'][0]
        self.assertTrue(top['rank_tied_with'])
        self.assertEqual(top['rank_range'][0], 1)
        self.assertIn('No established leader', R.table(r))

    def test_components_are_ranked_apart(self):
        rng = np.random.default_rng(2)
        deals = deals_between('a', 'b', 300, 200, rng, sd=100) + deals_between('c', 'd', 300, 100, rng, sd=100)
        r = R.rate(deals, boot=100)
        self.assertEqual(r['components'], [['a', 'b'], ['c', 'd']])     # the bigger group first
        comp = {p['player']: p['component'] for p in r['players']}
        self.assertEqual((comp['a'], comp['c']), (0, 1))
        self.assertEqual([x['player'] if x else None for x in r['leaders']], ['a', 'c'])
        self.assertEqual(r['leader']['player'], 'a')
        self.assertTrue(any('groups' in n for n in r['notes']))
        for p in r['players']:                                          # each group sums to zero on its own
            self.assertAlmostEqual(p['strength'] + next(q['strength'] for q in r['players']
                                                        if q['component'] == p['component'] and q is not p), 0, 0)

    def test_no_interval_without_two_blocks(self):
        rng = np.random.default_rng(3)
        r = R.rate(deals_between('a', 'b', 50, 8, rng), boot=50)        # 8 deals: one block
        self.assertTrue(all(p['strength_ci95'] is None for p in r['players']))
        self.assertEqual(r['pairings'][0]['verdict'], 'early')
        # a triangle of single-block pairings: each player has two blocks, but no pairing has a spread to resample
        deals = deals_between('a', 'b', 50, 5, rng) + deals_between('b', 'c', 50, 5, rng) + deals_between('a', 'c', 50, 5, rng)
        r = R.rate(deals, boot=50)
        self.assertTrue(all(p['strength_ci95'] is None and p['rank'] is None for p in r['players']))
        self.assertIsNone(r['leader'])


class TestRepeatsAndIdentity(unittest.TestCase):
    def test_repeated_deals_are_averaged(self):
        rng = np.random.default_rng(4)
        one = deals_between('a', 'b', 0, 30, rng, matchup='run1/a__vs__b')
        two = deals_between('b', 'a', 0, 30, rng, matchup='run2/b__vs__a')   # the same seeds, the other way round
        r = R.rate(one + two, boot=50)
        self.assertEqual((r['deals'], r['observations'], r['repeats']), (60, 30, 30))
        [p] = r['pairings']
        self.assertEqual((p['plays'], p['deals']), (60, 30))
        x = {(d['matchup'], d['deal']): d['net_a'] / 4 * 100 for d in one + two}
        mean = np.mean([(x[('run1/a__vs__b', k)] - x[('run2/b__vs__a', k)]) / 2 for k in range(30)])
        self.assertAlmostEqual(p['edge'], mean, places=0)
        self.assertTrue(any('averaged' in n for n in r['notes']))

    def test_fingerprint(self):
        g = {'name': 'holdem', 'sb': 1, 'bb': 2, 'stack': 200, 'hands_per_match': 50}
        self.assertEqual(R.fingerprint(g, 1000, 0), R.fingerprint({**g, 'hands_per_match': 10}, 1000, 0))
        self.assertNotEqual(R.fingerprint(g, 1000, 0), R.fingerprint(g, 1000, 1))      # the button matters
        self.assertNotEqual(R.fingerprint(g, 1000, 0), R.fingerprint({**g, 'stack': 100}, 1000, 0))
        self.assertIsNone(R.fingerprint(g, None, 0))

    def test_quarantine(self):
        self.assertIsNone(R.quarantine_reason('claude:claude-sonnet-5-5:high'))
        self.assertIsNone(R.quarantine_reason('bot:rule:sims=200'))
        self.assertIn('alias', R.quarantine_reason('claude:sonnet:default'))
        self.assertIn('no model', R.quarantine_reason('codex:codex-gpt'))
        rng = np.random.default_rng(5)
        deals = deals_between('good:m:high', 'codex:codex-gpt', 0, 40, rng) + deals_between('good:m:high', 'bot:rule', 0, 40, rng)
        meta = {p: {'identity': p, 'kind': 'model', 'quarantine_reason': R.quarantine_reason(p)}
                for p in ('good:m:high', 'codex:codex-gpt', 'bot:rule')}
        pooled = R.rate(deals, boot=50, meta=meta, pooled=True)
        q = next(p for p in pooled['players'] if p['player'] == 'codex:codex-gpt')
        self.assertTrue(q['quarantined'])
        self.assertIsNone(q['strength'])
        self.assertEqual(q['hands'], 80)
        self.assertEqual(len(pooled['pairings']), 1)                   # its pairing is not judged
        self.assertEqual(pooled['components'], [['bot:rule', 'good:m:high']])
        run = R.rate(deals, boot=50, meta=meta)                         # within one run it is rated, and flagged
        q = next(p for p in run['players'] if p['player'] == 'codex:codex-gpt')
        self.assertTrue(q['quarantined'])
        self.assertIsNotNone(q['strength'])


class TestSessionElo(unittest.TestCase):
    def test_interval_never_collapses(self):
        rng = np.random.default_rng(6)
        r = R.rate(deals_between('w', 'l', 2000, 30, rng, sd=10), boot=20)   # three sessions, all won
        w = next(p for p in r['players'] if p['player'] == 'w')
        self.assertEqual(w['session_w'], 3)
        lo, hi = w['session_elo_ci95']
        self.assertLess(lo, w['session_elo'])
        self.assertGreater(hi - lo, 100)
        self.assertTrue(np.isfinite([lo, hi]).all())

    def test_split_pots_are_not_wins(self):
        s = summarize({'hands': 10, 'showdowns': 4, 'won_sd': 1, 'sd_ties': 2})
        self.assertEqual((s['wsd'], s['sd_ties'], s['showdowns']), (0.25, 2, 4))


class TestContract(unittest.TestCase):
    def test_output_keys(self):
        rng = np.random.default_rng(7)
        deals = deals_between('a', 'b', 40, 50, rng) + deals_between('b', 'c', 40, 50, rng)
        plan = {'players': ['a', 'b', 'c', 'd'], 'matchups': [['a', 'b'], ['b', 'c'], ['c', 'd']], 'deals': 50}
        r = R.rate(deals, boot=30, plan=plan)
        for k in ('objective', 'design', 'deals', 'hands', 'components', 'leader', 'players', 'pairings', 'roster', 'notes'):
            self.assertIn(k, r)
        self.assertEqual(r['objective'], 'bb/100 over duplicate deals')
        self.assertEqual(r['hands'], 200)
        for p in r['players']:
            self.assertLessEqual(PLAYER_KEYS, set(p))
        for m in r['pairings']:
            self.assertLessEqual(DECIDE_KEYS | {'a', 'b', 'matchups'}, set(m))
            self.assertEqual(m['alpha_pairing'], 0.05 / 3)               # Bonferroni over the plan's pairings
            self.assertEqual(set(m['power']['deals_for_80pct']), {'20', '50', '100'})
        self.assertEqual({x['player']: x['status'] for x in r['roster']},
                         {'a': 'done', 'b': 'done', 'c': 'playing', 'd': 'awaiting'})
        json.dumps(r, allow_nan=False)

    def test_decide(self):
        rng = np.random.default_rng(8)
        v = R.decide(list(350 * rng.standard_normal(50)), R.DEFAULT_DESIGN, 3, 1)
        self.assertLessEqual(DECIDE_KEYS, set(v))
        self.assertEqual(v['verdict'], 'open')                         # look 1 of 6, no edge
        self.assertEqual((v['looks_used'], v['looks_planned'], v['blocks']), (1, 6, 5))
        n = v['power']['deals_for_80pct']
        self.assertGreater(n['20'], n['50'])
        self.assertGreater(n['50'], n['100'])
        self.assertEqual(R.decide([1.0] * 5, R.DEFAULT_DESIGN, 3, 1)['verdict'], 'early')
        self.assertEqual(R.decide(list(400 + 50 * rng.standard_normal(50)), R.DEFAULT_DESIGN, 3, 1)['verdict'], 'a')
        self.assertEqual(R.decide(list(350 * rng.standard_normal(300)), R.DEFAULT_DESIGN, 3, 6)['verdict'], 'exhausted')
        self.assertEqual(R.decide(list(350 * rng.standard_normal(80)), R.DEFAULT_DESIGN, 3, 0)['verdict'], 'open')
        # blocks restart with each matchup
        self.assertEqual(R.decide([1.0, 2.0] * 10, groups=['m1'] * 5 + ['m2'] * 15)['blocks'], 3)

    def test_no_misleading_language(self):
        rng = np.random.default_rng(9)
        r = R.rate(deals_between('a', 'b', 0, 300, rng), boot=30,
                   plan={'players': ['a', 'b'], 'matchups': [['a', 'b']], 'deals': 300})
        text = R.summary(r)
        self.assertNotIn('too close', text)
        self.assertIn('not evidence that they are equal', text)


if __name__ == '__main__':
    unittest.main()
