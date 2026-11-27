"""The leaderboard: decided results only from the sequential test of a preregistered design and players of
verified identity, descriptive strengths with intervals from the joint bootstrap, groups with their own
leader or front runner, repeated deals averaged, session Elo intervals that keep conversations together and
never collapse, quarantine apart from fit membership, exact model matching, the one normalized design with
its power scenarios, and the output keys the viewer reads."""
import json
import unittest

import numpy as np

from poker_arena import ratings as R
from poker_arena.stats import summarize

PLAYER_KEYS = {'player', 'identity', 'kind', 'component', 'strength', 'strength_ci95', 'rank', 'rank_kind',
               'rank_tied_with', 'position', 'in_fit', 'identity_resolved', 'net_bb', 'hands', 'deals', 'session_elo',
               'session_elo_ci95', 'session_elo_basis', 'sessions', 'session_w', 'session_t', 'session_l',
               'quarantined', 'quarantine_reason', 'inferential'}
DECIDE_KEYS = {'deals', 'blocks', 'edge', 'ci95', 't', 'boundary', 'verdict', 'looks_used', 'looks_planned', 'power',
               'conversations', 'basis', 'df', 'min_conversations', 'signflip', 'decided_at_look', 'conversation_share'}
GROUP_KEYS = {'id', 'players', 'leader', 'front_runner', 'front_runner_margin', 'ci95', 'power', 'headline'}
DESIGN_KEYS = {'alpha', 'round_deals', 'max_rounds', 'max_rounds_basis', 'block', 'replicates', 'n_pairings',
               'alpha_pairing', 'nominal_alpha', 'target_edge_bb100', 'prior_sd_bb100', 'deals_needed', 'deals_planned',
               'max_deals', 'power_at_max', 'min_conversations', 'conversations_needed', 'power_scenarios',
               'assumed_icc', 'assumed_ar1', 'underpowered', 'registered', 'sd_conversation_bb100'}


def deals_between(a, b, edge, n, rng, sd=300.0, rounds=1, seed0=1000, run='', planned=None, reps=1, effects=None):
    """`rounds` x `reps` conversations of n duplicate deals in which a beats b by `edge` bb/100 on average
    (bb = 2), plus a per-conversation effect from `effects` (one value per conversation, in play order), on the
    same slates for every pairing (round r, replicate j deals seeds seed0 + ((r - 1) reps + j - 1) n ...)."""
    out = []
    c = 0
    for r in range(1, rounds + 1):
        for j in range(1, reps + 1):
            u = effects[c] if effects is not None else 0.0
            c += 1
            for k in range(n):
                x = edge + u + sd * rng.standard_normal()
                seed = seed0 + ((r - 1) * reps + j - 1) * n + k
                name = f'{run}{a}__vs__{b}' + (f'__x{j}' if j > 1 else '') + (f'__r{r}' if r > 1 else '')
                out.append({'a': a, 'b': b, 'matchup': name, 'deal': k, 'bb': 2, 'net_a': x * 4 / 100, 'seed': seed,
                            'fp': f'{seed}:{k % 2}', 'planned': planned or n, 'round': r, 'replicate': j})
    return out


def plan(players, n, rounds, reps=1, **kw):
    pairs = [[a, b] for i, a in enumerate(players) for b in players[i + 1:]]
    return {'players': players, 'matchups': kw.pop('matchups', pairs), 'deals': n, 'design': {'max_rounds': rounds},
            'replicates': reps, **kw}


class TestInference(unittest.TestCase):
    def test_decided_leader_and_ranks(self):
        rng = np.random.default_rng(0)
        true = {'x': 100.0, 'y': 0.0, 'z': -100.0}
        deals = []
        for a, b in (('x', 'y'), ('x', 'z'), ('y', 'z')):
            deals += deals_between(a, b, true[a] - true[b], 50, rng, sd=100, rounds=4, reps=2)
        r = R.rate(deals, boot=100, plan=plan(['x', 'y', 'z'], 50, 4, reps=2))
        self.assertTrue(all(m['verdict'] == 'a' for m in r['pairings']), [m['verdict'] for m in r['pairings']])
        self.assertTrue(all(m['signflip'] and m['inferential'] for m in r['pairings']))
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

    def test_too_few_conversations_is_early(self):
        """Below min_conversations a completed look gives 'early', however large the edge (R3-#1)."""
        rng = np.random.default_rng(10)
        deals = deals_between('x', 'y', 500, 50, rng, sd=50, rounds=3, reps=2)
        r = R.rate(deals, boot=50, plan=plan(['x', 'y'], 50, 6, reps=2))
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['conversations'], m['looks_used'], m['min_conversations']), ('early', 6, 3, 8))
        self.assertIn('at least 8', m['reason'])
        self.assertIsNone(r['leader'])
        self.assertTrue(any('at least 8 conversations' in n for n in r['notes']))

    def test_partial_order(self):
        """x beat y and z (decided); y vs z undecided: x has a decided rank, y and z share 2-3."""
        rng = np.random.default_rng(2)
        deals = deals_between('x', 'y', 300, 50, rng, sd=100, rounds=4, reps=2) \
            + deals_between('x', 'z', 300, 50, rng, sd=100, rounds=4, reps=2) \
            + deals_between('y', 'z', 0, 50, rng, rounds=4, reps=2)
        r = R.rate(deals, boot=50, plan=plan(['x', 'y', 'z'], 50, 4, reps=2))
        s = {p['player']: p for p in r['players']}
        self.assertEqual((s['x']['rank'], s['y']['rank'], s['z']['rank']), (1, None, None))
        self.assertEqual(s['y']['rank_range'], [2, 3])
        self.assertEqual(s['y']['rank_tied_with'], ['z'])
        self.assertEqual(r['leader']['player'], 'x')

    def test_groups_have_their_own_summaries(self):
        rng = np.random.default_rng(3)
        deals = deals_between('a', 'b', 300, 50, rng, sd=100, rounds=3, reps=3) \
            + deals_between('c', 'd', 30, 50, rng, rounds=3, reps=3)
        r = R.rate(deals, boot=50, plan=plan(['a', 'b', 'c', 'd'], 50, 3, reps=3, matchups=[['a', 'b'], ['c', 'd']]))
        self.assertEqual(r['components'], [['a', 'b'], ['c', 'd']])
        self.assertEqual([g['leader'] for g in r['groups']], ['a', None])
        self.assertEqual(r['groups'][1]['front_runner'] in ('c', 'd'), True)
        self.assertNotIn('a', r['groups'][1]['headline'].split())
        self.assertTrue(any('groups' in n for n in r['notes']))

    def test_one_conversation_gives_no_verdict(self):
        rng = np.random.default_rng(4)
        r = R.rate(deals_between('a', 'b', 500, 100, rng, sd=100), boot=50, plan=plan(['a', 'b'], 100, 1))
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['basis'], m['conversations']), ('early', 'blocks', 1))
        self.assertIsNone(r['leader'])
        self.assertTrue(all(p['ci_basis'] == 'blocks' for p in r['players']))
        self.assertTrue(any('one conversation' in n for n in r['notes']))

    def test_no_interval_without_spread(self):
        rng = np.random.default_rng(5)
        r = R.rate(deals_between('a', 'b', 50, 8, rng), boot=50)          # 8 deals: one block
        self.assertTrue(all(p['strength_ci95'] is None for p in r['players']))
        self.assertEqual(r['pairings'][0]['verdict'], 'descriptive')
        self.assertEqual(R.decide([1.0] * 8)['verdict'], 'early')

    def test_unregistered_run_is_descriptive(self):
        """A run without a recorded design (one fixed round, `play`, an older stopping rule) is no test: no
        verdict, rank or leader, and any sample size is labelled retrospective (R3-#6)."""
        rng = np.random.default_rng(11)
        deals = deals_between('x', 'y', 400, 50, rng, sd=50, rounds=8)
        for p in ({'players': ['x', 'y'], 'matchups': [['x', 'y']], 'deals': 50}, None,
                  {'players': ['x', 'y'], 'matchups': [['x', 'y']], 'deals': 50, 'until_decided': 400}):
            r = R.rate(deals, boot=30, plan=p)
            self.assertIs(r['design']['registered'], False)
            self.assertNotIn('target_edge_bb100', r['design'])
            self.assertIn('deals_needed', r['design']['retrospective'])
            self.assertEqual([m['verdict'] for m in r['pairings']], ['descriptive'])
            self.assertFalse(r['pairings'][0]['inferential'])
            self.assertIsNone(r['leader'])
            self.assertTrue(all(q['rank'] is None and not q['inferential'] for q in r['players']))
            text = R.summary(r)
            self.assertIn('No preregistered design', text)
            self.assertIn('Retrospective', text)
            self.assertNotIn('Planned to detect', text)
            self.assertIn('no preregistered design', r['inference'])

    def test_identity_in_doubt_is_descriptive_in_the_run(self):
        """A served-model mismatch or an unresolved model keeps a player out of every decided result of its run,
        whatever the numbers say (R3-#8)."""
        rng = np.random.default_rng(12)
        deals = deals_between('good', 'swapped', 300, 50, rng, sd=100, rounds=4, reps=2) \
            + deals_between('good', 'bot', 300, 50, rng, sd=100, rounds=4, reps=2) \
            + deals_between('swapped', 'bot', 0, 50, rng, sd=100, rounds=4, reps=2)
        meta = {'good': {'identity': 'claude:claude-x-1:high', 'kind': 'model', 'quarantine_reason': None,
                         'identity_resolved': True},
                'swapped': {'identity': 'codex:gpt-y:high', 'kind': 'model', 'identity_resolved': True,
                            'quarantine_reason': 'answered by gpt-old, not the configured gpt-y'},
                'bot': {'identity': 'bot:rule', 'kind': 'bot', 'quarantine_reason': None, 'identity_resolved': True}}
        r = R.rate(deals, boot=30, meta=meta, plan=plan(['good', 'swapped', 'bot'], 50, 4, reps=2))
        v = {frozenset((m['a'], m['b'])): m for m in r['pairings']}
        clean = v[frozenset(('good', 'bot'))]
        self.assertEqual((clean['verdict'], clean['inferential']), ('a', True))
        for other in ('good', 'bot'):
            m = v[frozenset(('swapped', other))]
            self.assertEqual((m['verdict'], m['inferential']), ('descriptive', False))
            self.assertIn('gpt-old', m['reason'])
        self.assertIn(v[frozenset(('swapped', 'good'))]['test_verdict'], ('a', 'b'))
        self.assertIsNone(r['leader'])                                  # good's win over swapped does not count
        s = {p['player']: p for p in r['players']}
        self.assertEqual((s['swapped']['inferential'], s['swapped']['in_fit'], s['swapped']['rank']), (False, True, None))
        self.assertIsNone(s['good']['rank'])
        self.assertTrue(any('identity is in doubt' in n for n in r['notes']))

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
        self.assertEqual((r['pairings'][0]['verdict'], r['pairings'][0]['recorded_verdict']), ('early', 'a'))
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


class TestModelIdentity(unittest.TestCase):
    def test_exact_canonical_ids(self):
        """A shorter or longer id is another model; spelling and documented dated snapshots are not (R3-#7)."""
        self.assertEqual(R.served_mismatch('claude-sonnet-5-5', ['claude-sonnet-5']), ['claude-sonnet-5'])
        self.assertEqual(R.served_mismatch('claude-sonnet-5', ['claude-sonnet-5-5']), ['claude-sonnet-5-5'])
        self.assertEqual(R.served_mismatch('gemini-4-neon', ['gemini-4-neon-high-0912']), ['gemini-4-neon-high-0912'])
        self.assertEqual(R.served_mismatch('claude-sonnet-5-5', ['claude-sonnet-5-5-20261001', 'Claude Sonnet 5.5']), [])
        self.assertEqual(R.served_mismatch('gemini-4-neon-high', ['models/gemini-4-neon-high', 'Gemini 4 Neon (High)']), [])
        self.assertTrue(R.same_model('gpt-6-luna', 'gpt-6-luna-2026-09-30'))
        self.assertFalse(R.same_model('gpt-6-luna', 'gpt-6-luna-mini'))
        R.SERVED_ALIASES['gpt-6-luna'] = {'gpt-6-luna-latest'}
        try:
            self.assertTrue(R.same_model('gpt-6-luna', 'gpt-6-luna-latest'))
        finally:
            R.SERVED_ALIASES.pop('gpt-6-luna')


class TestSessionElo(unittest.TestCase):
    def test_interval_never_collapses(self):
        rng = np.random.default_rng(6)
        r = R.rate(deals_between('w', 'l', 2000, 30, rng, sd=10), boot=20)   # three sessions, all won
        w = next(p for p in r['players'] if p['player'] == 'w')
        self.assertEqual(w['session_w'], 3)
        lo, hi = w['session_elo_ci95']
        self.assertLess(lo, w['session_elo'])
        self.assertGreater(hi - lo, 100)

    def test_interval_counts_conversations(self):
        """Sessions of one conversation move together: when whole conversations are won or lost, the interval
        comes from resampling them, and is wider than one that treats sessions as independent (R3-#11)."""
        rng = np.random.default_rng(13)
        swings = [400.0, -400.0] * 4
        deals = deals_between('p', 'q', 0, 50, rng, sd=20, rounds=8, effects=swings) \
            + deals_between('p', 'bot', 0, 50, rng, sd=20, rounds=8) + deals_between('q', 'bot', 0, 50, rng, sd=20, rounds=8)
        r = R.rate(deals, boot=200, plan=plan(['p', 'q', 'bot'], 50, 8))
        p = next(x for x in r['players'] if x['player'] == 'p')
        self.assertEqual(p['session_elo_basis'], 'bootstrap')
        names = sorted({'p', 'q', 'bot'})
        obs, _ = R.observations([d for d in deals if d['a'] != d['b']])
        _, cov = R._elo(names, [names], R.sessions(obs))
        posterior = 2 * 1.96 * R.K * np.sqrt(cov[names.index('p'), names.index('p')])
        self.assertGreater(p['session_elo_ci95'][1] - p['session_elo_ci95'][0], 1.5 * posterior)

    def test_split_pots_are_not_wins(self):
        s = summarize({'hands': 10, 'showdowns': 4, 'won_sd': 1, 'sd_ties': 2})
        self.assertEqual((s['wsd'], s['sd_ties'], s['showdowns']), (0.25, 2, 4))
        self.assertEqual(s['counts']['wsd'], {'n': 1, 'of': 4})


THREE = [['a', 'b'], ['a', 'c'], ['b', 'c']]


class TestDesign(unittest.TestCase):
    def test_default_design_meets_its_power_target(self):
        """max_rounds comes from the target: the moderate scenario reaches 80%, and every scenario is reported
        (R3-#2, #3)."""
        p = {'deals': 50, 'design': {}, 'replicates': 4, 'matchups': THREE}
        d = R.normalize_design(p)
        self.assertLessEqual(DESIGN_KEYS, set(d))
        self.assertEqual((d['max_rounds_basis'], d['registered'], d['underpowered']), ('power', True, False))
        self.assertEqual(d['min_conversations'], R.first_verdict(d['alpha_pairing'], d['max_rounds'], 4))
        self.assertTrue(d['min_conversations'] >= R.MIN_CONVERSATIONS and d['min_conversations'] % 4 == 0)
        self.assertGreaterEqual(d['power_at_max'], 0.8)
        self.assertEqual(d['deals_needed'], d['max_deals'])
        sc = {x['scenario']: x for x in d['power_scenarios']}
        self.assertEqual(set(sc), {'optimistic', 'moderate', 'pessimistic'})
        self.assertEqual(sc['moderate']['power'], d['power_at_max'])
        self.assertGreater(sc['optimistic']['power'], sc['moderate']['power'])
        self.assertGreater(sc['moderate']['power'], sc['pessimistic']['power'])
        self.assertEqual((sc['moderate']['assumed_icc'], sc['moderate']['assumed_ar1']), (0.01, 0.2))
        fewer = R.normalize_design({**p, 'design': {'max_rounds': d['max_rounds'] - 1}})
        self.assertLess(fewer['power_at_max'], 0.8)                        # the fewest rounds that reach it
        self.assertEqual(R.normalize_design({**p, 'design': d}), d)       # idempotent: a stored plan reads the same
        self.assertIn('Planned to detect a 50 bb/100 edge', R.design_line(d))
        self.assertIn('if it is worse', R.design_line(d))

    def test_dependence_raises_the_deals_needed(self):
        base = {'deals': 50, 'replicates': 4, 'matchups': THREE}
        light = R.normalize_design({**base, 'design': {'assumed_icc': 0.0, 'assumed_ar1': 0.0}})
        heavy = R.normalize_design({**base, 'design': {'assumed_icc': 0.03, 'assumed_ar1': 0.3}})
        self.assertLess(light['deals_needed'], heavy['deals_needed'])
        self.assertAlmostEqual(R.sd_conversation(350, 50, 0, 0), 350 / np.sqrt(50))
        self.assertGreater(R.design_effect(0.3, 50), 1.7)

    def test_budget_cap_is_underpowered(self):
        p = {'deals': 50, 'design': {'budget_deals': 600}, 'replicates': 4, 'matchups': THREE}
        d = R.normalize_design(p)
        self.assertEqual((d['max_rounds'], d['max_rounds_basis'], d['underpowered'], d['max_deals']), (3, 'budget', True, 600))
        self.assertGreater(d['deals_needed'], 600)
        self.assertIn('Underpowered', R.design_line(d))
        given = R.normalize_design({**p, 'design': {'max_rounds': 6}, 'replicates': 2})
        self.assertEqual((given['max_rounds_basis'], given['underpowered']), ('given', True))

    def test_normalize_design(self):
        p = {'deals': 50, 'design': {'alpha': 0.05, 'max_rounds': 6, 'block': 10, 'round_deals': 50}, 'replicates': 2,
             'matchups': THREE}
        d = R.normalize_design(p)
        self.assertEqual((d['replicates'], d['n_pairings'], d['target_edge_bb100'], d['max_deals']), (2, 3, 50, 600))
        self.assertAlmostEqual(d['alpha_pairing'], 0.05 / 3)
        self.assertLess(d['nominal_alpha'], d['alpha_pairing'] / 2)
        self.assertEqual(R.normalize_design({**p, 'design': d}), d)
        self.assertIsNone(R.normalize_design({'deals': 20, 'seed': 0}))
        legacy = R.normalize_design({'deals': 20, 'until_decided': 60, 'seed': 0})
        self.assertEqual((legacy['max_rounds'], legacy['registered']), (3, False))

    def test_runner_and_ratings_agree(self):
        """The design a run's ratings judge by is the one the runner played by: replicates included (R2-#9)."""
        rng = np.random.default_rng(9)
        p = {'players': ['a', 'b'], 'matchups': [['a', 'b']], 'deals': 50, 'design': {'max_rounds': 3}, 'replicates': 2}
        deals = deals_between('a', 'b', 0, 50, rng, rounds=2, reps=2)
        out = R.rate(deals, boot=30, plan=p)
        self.assertEqual(out['design'], R.normalize_design(p))
        [m] = out['pairings']
        self.assertEqual((m['looks_used'], m['conversations'], m['looks_planned']), (2, 4, 3))


class TestContract(unittest.TestCase):
    def test_output_keys(self):
        rng = np.random.default_rng(7)
        deals = deals_between('a', 'b', 40, 50, rng) + deals_between('b', 'c', 40, 50, rng)
        p = {'players': ['a', 'b', 'c', 'd'], 'matchups': [['a', 'b'], ['b', 'c'], ['c', 'd']], 'deals': 50,
             'design': {'max_rounds': 4}, 'replicates': 2}
        r = R.rate(deals, boot=30, plan=p)
        for k in ('objective', 'design', 'deals', 'hands', 'components', 'groups', 'leader', 'leaders', 'players',
                  'pairings', 'roster', 'notes', 'inference', 'min_conversations', 'guarantee'):
            self.assertIn(k, r)
        self.assertEqual(r['objective'], 'bb/100 over duplicate deals')
        self.assertEqual(r['hands'], 200)
        for p_ in r['players']:
            self.assertLessEqual(PLAYER_KEYS, set(p_))
        for g in r['groups']:
            self.assertLessEqual(GROUP_KEYS, set(g))
        for m in r['pairings']:
            self.assertLessEqual(DECIDE_KEYS | {'a', 'b', 'matchups', 'inferential'}, set(m))
            self.assertEqual(m['alpha_pairing'], 0.05 / 3)               # Bonferroni over the plan's pairings
            self.assertEqual(set(m['power']['deals_for_80pct']), {'20', '50', '100'})
        self.assertEqual({x['player']: x['status'] for x in r['roster']},
                         {'a': 'playing', 'b': 'playing', 'c': 'playing', 'd': 'awaiting'})
        json.dumps(r, allow_nan=False)

    def test_decide(self):
        rng = np.random.default_rng(8)
        d = R.normalize_design({'deals': 50, 'design': {'max_rounds': 6}, 'replicates': 2, 'matchups': [[1, 2]] * 1})
        g = lambda c: np.repeat(np.arange(c), 50)
        v = R.decide(list(350 * rng.standard_normal(100)), d, 3, 1, groups=g(2))
        self.assertLessEqual(DECIDE_KEYS, set(v))
        self.assertEqual((v['verdict'], v['looks_used'], v['looks_planned'], v['conversations']), ('early', 1, 6, 2))
        v = R.decide(list(350 * rng.standard_normal(400)), d, 3, 4, groups=g(8))
        self.assertEqual((v['verdict'], v['min_conversations']), ('early', 10))   # 8: no sign-flip verdict possible yet
        v = R.decide(list(350 * rng.standard_normal(500)), d, 3, 5, groups=g(10))
        self.assertEqual(v['verdict'], 'open')
        n = v['power']['deals_for_80pct']
        self.assertGreater(n['20'], n['50'])
        self.assertGreater(n['50'], n['100'])
        self.assertEqual(R.decide([1.0] * 5, d, 3, 1)['verdict'], 'early')
        self.assertEqual(R.decide(list(400 + 50 * rng.standard_normal(500)), d, 3, 5, groups=g(10))['verdict'], 'a')
        self.assertEqual(R.decide(list(-400 + 50 * rng.standard_normal(500)), d, 3, 5, groups=g(10))['verdict'], 'b')
        self.assertEqual(R.decide(list(350 * rng.standard_normal(600)), d, 3, 6, groups=g(12))['verdict'], 'exhausted')
        self.assertEqual(R.decide(list(350 * rng.standard_normal(80)), d, 3, 0)['verdict'], 'open')
        self.assertEqual(R.decide([1.0, 2.0] * 10, groups=['m1'] * 5 + ['m2'] * 15)['blocks'], 3)

    def test_both_arms_are_needed(self):
        """The sign-flip arm alone can veto a t crossing: eight conversations of which one carries the whole
        edge cannot reject symmetry even when the t is large."""
        d = R.normalize_design({'deals': 50, 'design': {'max_rounds': 4}, 'replicates': 2, 'matchups': THREE})
        means = [400.0, 40.0, 38.0, 42.0, 41.0, 39.0, 40.5, 39.5]
        xs = np.repeat(means, 50) + np.tile(np.linspace(-1, 1, 50), 8)
        v = R.decide(list(xs), d, 3, 4, groups=np.repeat(np.arange(8), 50))
        self.assertGreater(v['t'], 1)
        self.assertEqual(v['verdict'], 'exhausted')

    def test_no_misleading_language(self):
        rng = np.random.default_rng(9)
        r = R.rate(deals_between('a', 'b', 0, 50, rng, rounds=4, reps=2), boot=30,
                   plan={'players': ['a', 'b'], 'matchups': [['a', 'b']], 'deals': 50, 'design': {'max_rounds': 4},
                         'replicates': 2})
        text = R.summary(r)
        for bad in ('too close', 'established', 'cancels', 'at least +', 'level with'):
            self.assertNotIn(bad, text)
        self.assertIn('not evidence that they are equal', text)
        self.assertIn('descriptive', text)
        self.assertIn('symmetric', text)


if __name__ == '__main__':
    unittest.main()
