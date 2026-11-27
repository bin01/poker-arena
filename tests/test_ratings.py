"""The leaderboard: decided results only from the sequential test, descriptive strengths with intervals from
the joint bootstrap, groups with their own leader or front runner, repeated deals averaged, session Elo
intervals that never collapse, quarantine apart from fit membership, the one normalized design, and the output
keys the viewer reads."""
import json
import unittest

import numpy as np

from poker_arena import ratings as R
from poker_arena.stats import summarize

PLAYER_KEYS = {'player', 'identity', 'kind', 'component', 'strength', 'strength_ci95', 'rank', 'rank_kind',
               'rank_tied_with', 'position', 'in_fit', 'identity_resolved', 'net_bb', 'hands', 'deals', 'session_elo',
               'session_elo_ci95', 'sessions', 'session_w', 'session_t', 'session_l', 'quarantined', 'quarantine_reason'}
DECIDE_KEYS = {'deals', 'blocks', 'edge', 'ci95', 't', 'boundary', 'verdict', 'looks_used', 'looks_planned', 'power',
               'conversations', 'basis', 'df'}
GROUP_KEYS = {'id', 'players', 'leader', 'front_runner', 'front_runner_margin', 'ci95', 'power', 'headline'}
DESIGN_KEYS = {'alpha', 'round_deals', 'max_rounds', 'block', 'replicates', 'n_pairings', 'alpha_pairing',
               'nominal_alpha', 'target_edge_bb100', 'prior_sd_bb100', 'deals_needed', 'deals_planned', 'max_deals',
               'power_at_max'}


def deals_between(a, b, edge, n, rng, sd=300.0, rounds=1, seed0=1000, run='', planned=None):
    """`rounds` conversations of n duplicate deals in which a beats b by `edge` bb/100 on average (bb = 2), on
    the same slates for every pairing (round r deals seeds seed0 + (r - 1) n ...)."""
    out = []
    for r in range(1, rounds + 1):
        for k in range(n):
            x = edge + sd * rng.standard_normal()
            seed = seed0 + (r - 1) * n + k
            out.append({'a': a, 'b': b, 'matchup': f'{run}{a}__vs__{b}' + (f'__r{r}' if r > 1 else ''), 'deal': k,
                        'bb': 2, 'net_a': x * 4 / 100, 'seed': seed, 'fp': f'{seed}:{k % 2}', 'planned': planned or n,
                        'round': r, 'replicate': 1})
    return out


def plan(players, n, rounds, **kw):
    pairs = [[a, b] for i, a in enumerate(players) for b in players[i + 1:]]
    return {'players': players, 'matchups': kw.pop('matchups', pairs), 'deals': n, 'design': {'max_rounds': rounds},
            'replicates': 1, **kw}


class TestInference(unittest.TestCase):
    def test_decided_leader_and_ranks(self):
        rng = np.random.default_rng(0)
        true = {'x': 100.0, 'y': 0.0, 'z': -100.0}
        deals = []
        for a, b in (('x', 'y'), ('x', 'z'), ('y', 'z')):
            deals += deals_between(a, b, true[a] - true[b], 100, rng, sd=100, rounds=4)
        r = R.rate(deals, boot=100, plan=plan(['x', 'y', 'z'], 100, 4))
        self.assertTrue(all(m['verdict'] == 'a' for m in r['pairings']), [m['verdict'] for m in r['pairings']])
        self.assertEqual([(p['player'], p['rank'], p['rank_kind']) for p in r['players']],
                         [('x', 1, 'decided'), ('y', 2, 'decided'), ('z', 3, 'decided')])
        self.assertEqual(r['leader']['player'], 'x')
        self.assertEqual(r['groups'][0]['leader'], 'x')
        self.assertEqual(r['leader']['basis'], 'sequential verdicts')
        s = {p['player']: p for p in r['players']}
        self.assertAlmostEqual(sum(p['strength'] for p in s.values()), 0, places=0)
        for p, v in true.items():
            lo, hi = s[p]['strength_ci95']
            self.assertTrue(lo < v < hi)
            self.assertEqual(s[p]['ci_basis'], 'conversations')

    def test_points_lead_is_not_a_leader(self):
        """A run whose first look hasn't come: however far ahead, nothing is decided (R2-#2)."""
        rng = np.random.default_rng(1)
        deals = deals_between('x', 'y', 400, 20, rng, sd=100, planned=50) \
            + deals_between('x', 'z', 400, 20, rng, sd=100, planned=50) + deals_between('y', 'z', 0, 20, rng, sd=100, planned=50)
        r = R.rate(deals, boot=100, plan=plan(['x', 'y', 'z'], 50, 4))
        self.assertIsNone(r['leader'])
        self.assertTrue(all(m['verdict'] == 'open' and m['looks_used'] == 0 for m in r['pairings']))
        self.assertTrue(all(p['rank'] is None and p['rank_kind'] == 'descriptive' for p in r['players']))
        g = r['groups'][0]
        self.assertEqual((g['leader'], g['front_runner'], g['kind']), (None, 'x', 'descriptive'))
        self.assertIn('ahead on points', g['headline'])
        self.assertEqual([p['position'] for p in r['players']], [1, 2, 3])
        self.assertIn('No decided leader', R.summary(r).replace('Decided leader: none', 'No decided leader'))

    def test_partial_order(self):
        """x beat y and z (decided); y vs z undecided: x has a decided rank, y and z share 2-3."""
        rng = np.random.default_rng(2)
        deals = deals_between('x', 'y', 300, 100, rng, sd=100, rounds=4) \
            + deals_between('x', 'z', 300, 100, rng, sd=100, rounds=4) + deals_between('y', 'z', 0, 100, rng, rounds=4)
        r = R.rate(deals, boot=50, plan=plan(['x', 'y', 'z'], 100, 4))
        s = {p['player']: p for p in r['players']}
        self.assertEqual((s['x']['rank'], s['y']['rank'], s['z']['rank']), (1, None, None))
        self.assertEqual(s['y']['rank_range'], [2, 3])
        self.assertEqual(s['y']['rank_tied_with'], ['z'])
        self.assertEqual(r['leader']['player'], 'x')

    def test_groups_have_their_own_summaries(self):
        rng = np.random.default_rng(3)
        deals = deals_between('a', 'b', 300, 100, rng, sd=100, rounds=3) + deals_between('c', 'd', 30, 50, rng, rounds=3)
        r = R.rate(deals, boot=50, plan=plan(['a', 'b', 'c', 'd'], 100, 3, matchups=[['a', 'b'], ['c', 'd']]))
        self.assertEqual(r['components'], [['a', 'b'], ['c', 'd']])
        self.assertEqual([g['leader'] for g in r['groups']], ['a', None])
        self.assertEqual(r['groups'][1]['front_runner'] in ('c', 'd'), True)
        self.assertNotIn('a', r['groups'][1]['headline'].split())
        self.assertTrue(any('groups' in n for n in r['notes']))

    def test_one_conversation_gives_no_verdict(self):
        rng = np.random.default_rng(4)
        r = R.rate(deals_between('a', 'b', 500, 100, rng, sd=100), boot=50)
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['basis'], m['conversations']), ('exhausted', 'blocks', 1))
        self.assertIsNone(r['leader'])
        self.assertTrue(all(p['ci_basis'] == 'blocks' for p in r['players']))
        self.assertTrue(any('one conversation' in n for n in r['notes']))

    def test_no_interval_without_spread(self):
        rng = np.random.default_rng(5)
        r = R.rate(deals_between('a', 'b', 50, 8, rng), boot=50)          # 8 deals: one block
        self.assertTrue(all(p['strength_ci95'] is None for p in r['players']))
        self.assertEqual(r['pairings'][0]['verdict'], 'early')

    def test_pooled_is_descriptive(self):
        rng = np.random.default_rng(6)
        deals = deals_between('x', 'y', 400, 100, rng, sd=100, rounds=4)
        r = R.rate(deals, boot=50, pooled=True)
        self.assertEqual(r['pairings'][0]['verdict'], 'descriptive')
        self.assertIsNone(r['leader'])
        self.assertTrue(all(p['rank'] is None for p in r['players']))
        self.assertIn('descriptive', R.summary(r))

    def test_recorded_verdict_kept_apart(self):
        rng = np.random.default_rng(7)
        deals = deals_between('x', 'y', 0, 50, rng, rounds=2)
        r = R.rate(deals, boot=30, plan=plan(['x', 'y'], 50, 4), recorded={'x__vs__y': [{'verdict': 'a', 'looks_used': 2}]})
        self.assertEqual((r['pairings'][0]['verdict'], r['pairings'][0]['recorded_verdict']), ('open', 'a'))
        self.assertTrue(any('earlier version' in n for n in r['notes']))


class TestBootstrap(unittest.TestCase):
    def test_strata_and_joint_weights(self):
        rng = np.random.default_rng(8)
        deals = deals_between('a', 'b', 0, 30, rng, rounds=3) + deals_between('a', 'c', 0, 30, rng, rounds=3) \
            + deals_between('b', 'c', 0, 30, rng)
        obs, _ = R.observations(deals)
        cells = R._cells(obs)
        strata = R._strata(cells)
        kinds = sorted((k, len(u)) for u, k, _ in strata)
        self.assertEqual(kinds, [('blocks', 3), ('conversations', 2)])  # slate 1 alone has b-c: its blocks
        W = R._weights(cells, strata, 50, np.random.default_rng(0))
        same = [i for i, c in enumerate(cells) if c['slate'] == '1000' and c['pos'] == 0]
        self.assertEqual(len(same), 3)                                  # three pairings on one aligned block
        self.assertTrue(all(np.array_equal(W[:, same[0]], W[:, i]) for i in same))
        self.assertTrue((W.sum(1) > 0).all())

    def test_unit_weights_are_unbiased(self):
        """Rao-Wu: the bootstrap variance of a stratum mean matches the textbook s^2 / k."""
        cells = [{'a': 'a', 'b': 'b', 'slate': str(s), 'pos': 0, 'n': 1, 'sx': float(v)}
                 for s, v in enumerate([3.0, -1.0, 4.0, 1.0, -5.0])]
        strata = [([[i] for i in range(5)], 'conversations', frozenset())]
        W = R._weights(cells, strata, 40000, np.random.default_rng(1))
        mean = (W * np.array([c['sx'] for c in cells])).sum(1) / W.sum(1).mean()
        self.assertAlmostEqual(mean.var(), np.var([3, -1, 4, 1, -5], ddof=1) / 5, delta=0.15)


class TestRepeatsAndIdentity(unittest.TestCase):
    def test_repeated_deals_are_averaged(self):
        rng = np.random.default_rng(4)
        one = deals_between('a', 'b', 0, 30, rng, run='run1/')
        two = deals_between('b', 'a', 0, 30, rng, run='run2/')            # the same seeds, the other way round
        r = R.rate(one + two, boot=50, pooled=True)
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
        self.assertIsNone(R.quarantine_reason('claude:claude-sonnet-5-5:high#1a2b3c4d'))
        self.assertIsNone(R.quarantine_reason('bot:rule:sims=200'))
        self.assertIn('alias', R.quarantine_reason('claude:sonnet:default'))
        self.assertIn('no model', R.quarantine_reason('codex:codex-gpt'))
        self.assertIn('answered by gemini-2', R.quarantine_reason('agy:gemini-4:high#ab!served=gemini-2'))
        self.assertTrue(R.identity_resolved('agy:gemini-4:high#ab!served=gemini-2'))
        self.assertFalse(R.identity_resolved('claude:sonnet:default'))
        self.assertEqual(R.served_mismatch('gemini-4-neon-high', ['gemini-4-neon-high']), [])
        self.assertEqual(R.served_mismatch('gemini-4-neon', ['gemini-4-neon-high-0912']), [])
        self.assertEqual(R.served_mismatch('gemini-4-neon', ['gemini-3-flash']), ['gemini-3-flash'])
        rng = np.random.default_rng(5)
        deals = deals_between('good:m:high', 'codex:codex-gpt', 0, 40, rng) + deals_between('good:m:high', 'bot:rule', 0, 40, rng)
        meta = {p: {'identity': p, 'kind': 'model', 'quarantine_reason': R.quarantine_reason(p),
                    'identity_resolved': R.identity_resolved(p)} for p in ('good:m:high', 'codex:codex-gpt', 'bot:rule')}
        pooled = R.rate(deals, boot=50, meta=meta, pooled=True)
        q = next(p for p in pooled['players'] if p['player'] == 'codex:codex-gpt')
        self.assertEqual((q['quarantined'], q['in_fit'], q['identity_resolved'], q['strength']), (True, False, False, None))
        self.assertEqual(q['hands'], 80)
        self.assertEqual(len(pooled['pairings']), 1)
        self.assertEqual(pooled['components'], [['bot:rule', 'good:m:high']])
        run = R.rate(deals, boot=50, meta=meta)                         # within one run it is in the fit, and flagged
        q = next(p for p in run['players'] if p['player'] == 'codex:codex-gpt')
        self.assertEqual((q['quarantined'], q['in_fit'], q['identity_resolved']), (True, True, False))
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

    def test_split_pots_are_not_wins(self):
        s = summarize({'hands': 10, 'showdowns': 4, 'won_sd': 1, 'sd_ties': 2})
        self.assertEqual((s['wsd'], s['sd_ties'], s['showdowns']), (0.25, 2, 4))
        self.assertEqual(s['counts']['wsd'], {'n': 1, 'of': 4})


class TestDesign(unittest.TestCase):
    def test_normalize_design(self):
        p = {'deals': 50, 'design': {'alpha': 0.05, 'max_rounds': 6, 'block': 10, 'round_deals': 50}, 'replicates': 2,
             'matchups': [['a', 'b'], ['a', 'c'], ['b', 'c']]}
        d = R.normalize_design(p)
        self.assertLessEqual(DESIGN_KEYS, set(d))
        self.assertEqual((d['replicates'], d['n_pairings'], d['target_edge_bb100'], d['max_deals']), (2, 3, 50, 600))
        self.assertAlmostEqual(d['alpha_pairing'], 0.05 / 3)
        self.assertLessEqual(d['deals_planned'], d['max_deals'])
        self.assertGreater(d['deals_needed'], d['max_deals'])          # 50 bb/100 at a spread of 350 needs more
        self.assertTrue(0 < d['power_at_max'] < 0.8)
        self.assertEqual(R.normalize_design({**p, 'design': d}), d)     # idempotent: a stored plan reads the same
        self.assertIsNone(R.normalize_design({'deals': 20, 'seed': 0}))
        self.assertEqual(R.normalize_design({'deals': 20, 'until_decided': 60, 'seed': 0})['max_rounds'], 3)
        big = R.normalize_design({**p, 'design': {'max_rounds': 20}, 'target_edge_bb100': 100})
        self.assertEqual(big['deals_planned'] % 100, 0)                 # whole rounds
        self.assertLess(big['deals_planned'], big['max_deals'])
        self.assertGreater(big['power_at_max'], 0.8)

    def test_runner_and_ratings_agree(self):
        """The design a run's ratings judge by is the one the runner played by: replicates included (R2-#9)."""
        rng = np.random.default_rng(9)
        p = {'players': ['a', 'b'], 'matchups': [['a', 'b']], 'deals': 50, 'design': {'max_rounds': 3}, 'replicates': 2}
        deals = []
        for r in (1, 2):
            for j in (1, 2):
                for d in deals_between('a', 'b', 0, 50, rng, seed0=1000 + ((r - 1) * 2 + j - 1) * 50):
                    deals.append({**d, 'round': r, 'replicate': j, 'matchup': f'a__vs__b__x{j}__r{r}'})
        out = R.rate(deals, boot=30, plan=p)
        self.assertEqual(out['design'], R.normalize_design(p))
        [m] = out['pairings']
        self.assertEqual((m['looks_used'], m['conversations'], m['looks_planned']), (2, 4, 3))


class TestContract(unittest.TestCase):
    def test_output_keys(self):
        rng = np.random.default_rng(7)
        deals = deals_between('a', 'b', 40, 50, rng) + deals_between('b', 'c', 40, 50, rng)
        p = {'players': ['a', 'b', 'c', 'd'], 'matchups': [['a', 'b'], ['b', 'c'], ['c', 'd']], 'deals': 50}
        r = R.rate(deals, boot=30, plan=p)
        for k in ('objective', 'design', 'deals', 'hands', 'components', 'groups', 'leader', 'leaders', 'players',
                  'pairings', 'roster', 'notes', 'inference'):
            self.assertIn(k, r)
        self.assertEqual(r['objective'], 'bb/100 over duplicate deals')
        self.assertEqual(r['hands'], 200)
        for p_ in r['players']:
            self.assertLessEqual(PLAYER_KEYS, set(p_))
        for g in r['groups']:
            self.assertLessEqual(GROUP_KEYS, set(g))
        for m in r['pairings']:
            self.assertLessEqual(DECIDE_KEYS | {'a', 'b', 'matchups'}, set(m))
            self.assertEqual(m['alpha_pairing'], 0.05 / 3)               # Bonferroni over the plan's pairings
            self.assertEqual(set(m['power']['deals_for_80pct']), {'20', '50', '100'})
        self.assertEqual({x['player']: x['status'] for x in r['roster']},
                         {'a': 'done', 'b': 'done', 'c': 'playing', 'd': 'awaiting'})
        json.dumps(r, allow_nan=False)

    def test_decide(self):
        rng = np.random.default_rng(8)
        d = R.normalize_design({'deals': 50, 'design': {'max_rounds': 6}, 'replicates': 2, 'matchups': [[1, 2]] * 1})
        g = lambda c: np.repeat(np.arange(c), 50)
        v = R.decide(list(350 * rng.standard_normal(100)), d, 3, 1, groups=g(2))
        self.assertLessEqual(DECIDE_KEYS, set(v))
        self.assertEqual((v['verdict'], v['looks_used'], v['looks_planned'], v['conversations']), ('open', 1, 6, 2))
        n = v['power']['deals_for_80pct']
        self.assertGreater(n['20'], n['50'])
        self.assertGreater(n['50'], n['100'])
        self.assertEqual(R.decide([1.0] * 5, d, 3, 1)['verdict'], 'early')
        self.assertEqual(R.decide(list(400 + 50 * rng.standard_normal(200)), d, 3, 2, groups=g(4))['verdict'], 'a')
        self.assertEqual(R.decide(list(350 * rng.standard_normal(600)), d, 3, 6, groups=g(12))['verdict'], 'exhausted')
        self.assertEqual(R.decide(list(350 * rng.standard_normal(80)), d, 3, 0)['verdict'], 'open')
        self.assertEqual(R.decide([1.0, 2.0] * 10, groups=['m1'] * 5 + ['m2'] * 15)['blocks'], 3)

    def test_no_misleading_language(self):
        rng = np.random.default_rng(9)
        r = R.rate(deals_between('a', 'b', 0, 300, rng), boot=30, plan={'players': ['a', 'b'], 'matchups': [['a', 'b']], 'deals': 300})
        text = R.summary(r)
        for bad in ('too close', 'established', 'cancels', 'at least +', 'level with'):
            self.assertNotIn(bad, text)
        self.assertIn('not evidence that they are equal', text)
        self.assertIn('descriptive', text)


if __name__ == '__main__':
    unittest.main()
