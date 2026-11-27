"""What a published result may claim (round 8 review): a verdict under detected drift is not a counted winner
(R8-#1); every official pairing carries its anytime-valid interval as the primary one and every ordinary
interval is labelled fixed-time descriptive (R8-#2); the adaptation track speaks in exploratory terms (R8-#7);
an optional strict identity policy, and an identity qualification on every counted result that involves an
unattested player (R8-#8)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import test_ratings
from poker_arena import cli
from poker_arena import ratings as R
from test_ratings import MODEL, deals_between, facts, plan, run

HEAVY = bool(os.environ.get('POKER_ARENA_HEAVY'))

THREE = [['x', 'y'], ['x', 'z'], ['y', 'z']]
CODEX = {**MODEL, 'backend': 'codex', 'requested_model': 'gpt-x-1', 'identity': 'unattested', 'observed_models': [],
         'identity_match': 'unattested'}


def strong(rng, a='x', b='y', edge=900, rounds=4, reps=2):
    return deals_between(a, b, edge, 50, rng, sd=20, rounds=rounds, reps=reps)


class TestDriftIsNotACountedWinner(unittest.TestCase):
    """The reviewer's probe: 700 conversations at +100, then 700 at -100 (mean 0). The bettor calls a at
    conversation 536; drift is detected over the judged conversations, so the verdict is decided_with_drift,
    counts toward no rank or leader, and says why in plain words."""

    @classmethod
    def setUpClass(cls):
        cls.p = {'players': ['x', 'y', 'z'], 'matchups': THREE, 'deals': 1, 'replicates': 25,
                 'design': {'max_rounds': 56}}
        cls.d = R.normalize_design(cls.p)
        cls.reg = R.registration(cls.d, 0)
        cls.ds = deals_between('x', 'y', 0, 1, np.random.default_rng(1), sd=0, rounds=56, reps=25,
                               effects=[100.0] * 700 + [-100.0] * 700)
        xs = [d['net_a'] / d['bb'] / 2 * 100 for d in cls.ds]
        cls.data = {'a': 'x', 'b': 'y', 'xs': xs, 'groups': [d['matchup'] for d in cls.ds],
                    'fps': [d['fp'] for d in cls.ds], 'rejected': [], 'evidence_hash': R.evidence_hash(cls.ds)}
        cls.v = R.adjudicate(cls.data, cls.d, 3, 56, facts(cls.reg, 'xy'))

    def test_adjudicate_gives_a_distinct_status(self):
        v = self.v
        self.assertEqual((v['test_verdict'], v['verdict'], v['decided_at_conversation']), ('a', 'a', 536))
        self.assertTrue(v['eligible'])
        self.assertEqual(v['drift']['status'], 'drift detected')
        self.assertEqual(v['standing'], 'decided_with_drift')
        self.assertEqual(v['decision_status'], 'decided_with_drift')
        self.assertFalse(v['counts_toward_leader'])
        self.assertIn('ahead early, behind late', v['qualifier'])
        self.assertIn('confirmation run on a stable model snapshot', v['qualifier'])
        self.assertTrue(v['ci_official']['drift'])

    def test_rate_names_no_winner(self):
        rec = {'x__vs__y': [self.v] * 56}
        r = R.rate(self.ds, boot=20, plan={**self.p, 'registration': self.reg}, recorded=rec,
                   eligibility={frozenset('xy'): facts(self.reg, 'xy')})
        [m] = r['pairings']
        self.assertTrue(m['official'])
        self.assertEqual((m['verdict'], m['decision_status']), ('a', 'decided_with_drift'))
        self.assertFalse(m['counts_toward_leader'])
        self.assertIsNone(r['leader'])
        self.assertTrue(all(p['rank'] is None for p in r['players']))
        self.assertIn('drift', r['groups'][0]['headline'])
        text = R.verdicts(r)
        self.assertIn('decided with drift', text)
        self.assertNotIn('x is better (decided)', text)
        self.assertTrue(any('confirmation run' in n for n in r['notes']))

    def test_an_older_record_that_counted_is_withheld_today(self):
        old = {**self.v, 'counts_toward_leader': True, 'standing': 'decided', 'decision_status': 'decided'}
        r = R.rate(self.ds, boot=20, plan={**self.p, 'registration': self.reg}, recorded={'x__vs__y': [old] * 56},
                   eligibility={frozenset('xy'): facts(self.reg, 'xy')})
        [m] = r['pairings']
        self.assertEqual(m['decision_status'], 'decided_with_drift')
        self.assertFalse(m['counts_toward_leader'])

    def test_a_steady_win_still_counts(self):
        r = run(strong(np.random.default_rng(2)), plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20}), boot=20)
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['decision_status'], m['standing']), ('a', 'decided', 'decided'))
        self.assertTrue(m['counts_toward_leader'])
        self.assertIsNone(m['qualifier'])
        self.assertEqual(r['leader']['player'], 'x')


class TestIntervalKinds(unittest.TestCase):
    def test_official_pairings_lead_with_the_sequential_interval(self):
        r = run(strong(np.random.default_rng(3)), plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20}), boot=20)
        [m] = r['pairings']
        self.assertEqual(m['ci_official']['interval'], m['cs'])
        self.assertEqual(m['ci_official']['level'], m['cs_level'])
        self.assertEqual(m['ci_official']['kind'], 'anytime-valid sequential')
        self.assertEqual(m['ci_kind'], 'fixed-time descriptive')
        for p in r['players']:
            self.assertEqual(p['ci_kind'], 'fixed-time descriptive')
        g = r['groups'][0]
        self.assertEqual(g['ci_kind'], 'fixed-time descriptive')
        self.assertEqual(r['leader']['ci_kind'], 'fixed-time descriptive')
        self.assertIn('fixed-time descriptive', r['intervals'])
        self.assertIn('anytime-valid sequential', r['intervals'])
        self.assertIn('not', r['intervals']['fixed-time descriptive'])
        self.assertIn('fixed-time', R.table(r))

    def test_descriptive_and_adaptation_rows_have_no_official_interval(self):
        rng = np.random.default_rng(4)
        ds = strong(rng)
        pooled = R.rate(ds, boot=20, pooled=True)
        unreg = run(ds, plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20}), boot=20, registered=False)
        for r in (pooled, unreg):
            [m] = r['pairings']
            self.assertIsNone(m['ci_official'])
            self.assertEqual(m['ci_kind'], 'fixed-time descriptive')
        ad = run(deals_between('x', 'y', 300, 10, rng, sd=50, rounds=4, reps=4),
                 plan(['x', 'y'], 10, 4, reps=4, track='adaptation'), boot=20)
        self.assertIsNone(ad['pairings'][0]['ci_official'])

    def test_headlines_never_show_a_bare_95_range(self):
        rng = np.random.default_rng(5)
        r = run(deals_between('x', 'y', 30, 50, rng, rounds=4, reps=2),
                plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20}), boot=20)
        h = r['groups'][0]['headline']
        self.assertIn('fixed-time descriptive', h)


class TestAdaptationIsExploratory(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        true = {'x': 300.0, 'y': 0.0, 'z': -300.0}
        ds = []
        for a, b in (('x', 'y'), ('x', 'z'), ('y', 'z')):
            ds += deals_between(a, b, true[a] - true[b], 10, rng, sd=50, rounds=4, reps=4)
        self.r = run(ds, plan(['x', 'y', 'z'], 10, 4, reps=4, track='adaptation'), boot=20)

    def test_flags_and_words(self):
        r = self.r
        self.assertEqual(r['language'], 'exploratory')
        for m in r['pairings']:
            self.assertEqual(m['verdict'], 'a')
            self.assertEqual(m['language'], 'exploratory')
            self.assertIn('leans', m['qualifier'])
        g = r['groups'][0]
        self.assertEqual(g['language'], 'exploratory')
        text = R.verdicts(r) + g['headline']
        self.assertIn('ahead on the model-based test', text)
        self.assertIn('leans', text)
        for bad in ('is better', ' won ', ' wins', 'better player', 'Won every'):
            self.assertNotIn(bad, text)

    def test_official_rows_are_confirmatory(self):
        r = run(strong(np.random.default_rng(6)), plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20}), boot=20)
        self.assertEqual((r['language'], r['pairings'][0]['language'], r['groups'][0]['language']),
                         ('confirmatory',) * 3)
        self.assertEqual(R.rate(strong(np.random.default_rng(6)), boot=20, pooled=True)['language'], 'descriptive')

    def test_cli_plan_text(self):
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / 'players.toml'
            pf.write_text('[players.rule]\nbot = "rule"\n[players.station]\nbot = "station"\n')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(['plan', 'rule', 'station', '--players', str(pf), '--out', str(Path(d) / 'cup'),
                          '--tracks', 'adaptation', '--write', str(Path(d) / 'p.json')])
        self.assertNotIn('better player', out.getvalue())
        self.assertIn('exploratory', out.getvalue())


class TestIdentityPolicy(unittest.TestCase):
    def setUp(self):
        self.ds = strong(np.random.default_rng(7))

    def adjudicated(self, strict):
        p = plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20},
                 design={'strict_identity': True} if strict else {})
        return run(self.ds, p, boot=20, who={'y': CODEX})

    def test_default_counts_with_a_qualification(self):
        r = self.adjudicated(False)
        [m] = r['pairings']
        self.assertTrue(m['counts_toward_leader'])
        self.assertTrue(m['identity_qualified'])
        self.assertIn('y', m['identity_qualifier'])
        self.assertTrue(r['leader']['identity_qualified'])
        self.assertIn('unattested', r['groups'][0]['headline'])
        self.assertIn('unattested', R.verdicts(r))
        self.assertFalse(R.normalize_design(plan(['x', 'y'], 50, 4))['strict_identity'])

    def test_attested_results_are_not_qualified(self):
        r = run(self.ds, plan(['x', 'y'], 50, 4, reps=2, game_config={'stack': 20}), boot=20)
        self.assertFalse(r['pairings'][0]['identity_qualified'])
        self.assertFalse(r['leader']['identity_qualified'])

    def test_strict_policy_counts_nothing_unattested(self):
        r = self.adjudicated(True)
        self.assertTrue(r['design']['strict_identity'])
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['eligible']), ('a', True))
        self.assertFalse(m['counts_toward_leader'])
        self.assertFalse(m['identity_qualified'])
        self.assertEqual(m['decision_status'], 'decided_identity_unattested')
        self.assertIn('strict', m['qualifier'])
        self.assertIsNone(r['leader'])

    def test_strict_is_part_of_the_registered_design(self):
        p = plan(['x', 'y'], 50, 4, reps=2)
        reg = {**p, 'registration': R.registration(R.normalize_design(p), 0)}
        changed = {**reg, 'design': {**reg['design'], 'strict_identity': True}}
        s = R.registration_status(changed)
        self.assertEqual(s['status'], 'invalid')
        self.assertIn('strict_identity', s['reasons'][0])
        self.assertTrue(R.registration_status(reg)['registered'])

    def test_cli_plan_flag(self):
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / 'players.toml'
            pf.write_text('[players.rule]\nbot = "rule"\n[players.station]\nbot = "station"\n')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(['plan', 'rule', 'station', '--players', str(pf), '--out', str(Path(d) / 'cup'),
                          '--strict-identity', '--write', str(Path(d) / 'p.json')])
            doc = json.loads((Path(d) / 'p.json').read_text())
        for t in doc['tracks'].values():
            self.assertIs(t['design']['strict_identity'], True)
            self.assertIs(R.normalize_design(t)['strict_identity'], True)
        self.assertIn('strict identity policy', out.getvalue().lower())


class TestStoppedRunCoverage(unittest.TestCase):
    """R8-#2: the fixed-time intervals after outcome-dependent stopping, in seeded 4-player round robins whose
    pairings stop when the official test decides (bound 3,000 bb/100, so no deal is clipped; conversation
    effects; about half the pairings stop early). docs/methodology.md reports the 400-tournament numbers
    (POKER_ARENA_HEAVY=1): strengths 0.95, pairing t intervals 0.95, the front runner's margin 0.93."""

    def test_coverage(self):
        k, rounds, n = 450.0, 8, 400 if HEAVY else 30
        true = {'p': 1.5 * k, 'q': 0.5 * k, 'r': -0.5 * k, 's': -1.5 * k}
        ps = list(true)
        hit = tot = phit = ptot = early = pairs = 0
        with mock.patch.object(test_ratings, 'BOUND', 3000.0):
            for t in range(n):
                rng = np.random.default_rng(9000 + t)
                ds = []
                for i, a in enumerate(ps):
                    for b in ps[i + 1:]:
                        ds += deals_between(a, b, true[a] - true[b], 10, rng, sd=250, rounds=rounds, reps=4,
                                            effects=list(60 * rng.standard_normal(rounds * 4)))
                r = run(ds, plan(ps, 10, rounds, reps=4, game_config={'stack': 60}), boot=200)
                for p in r['players']:
                    lo, hi = p['strength_ci95']
                    hit += lo <= true[p['player']] <= hi
                    tot += 1
                for m in r['pairings']:
                    pairs += 1
                    early += m['looks_used'] < rounds
                    if m['ci95']:
                        phit += m['ci95'][0] <= true[m['a']] - true[m['b']] <= m['ci95'][1]
                        ptot += 1
        self.assertGreater(early / pairs, 0.3)
        self.assertGreater(hit / tot, 0.88)
        self.assertGreater(phit / ptot, 0.88)
        if HEAVY:
            print(f'\nstopped runs: strength coverage {hit / tot:.3f}, pairing t {phit / ptot:.3f}, '
                  f'stopped early {early / pairs:.2f}')


if __name__ == '__main__':
    unittest.main()
