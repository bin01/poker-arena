"""Sequential tournaments: the round loop enforces the plan's design through ratings.decide (stops on a
verdict, never more looks than planned), replicates get fresh memories and their own deals, and every
pairing plays the same deals."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from poker_arena import ratings
from poker_arena.runner import make_plan, plan_design, round_seeds, run_plan

PLAYERS = ('[players.rule]\nbot = "rule"\nsims = 30\n[players.random]\nbot = "random"\n'
           '[players.station]\nbot = "station"\n[players.station2]\nbot = "station"\n')


def fake_decide(calls, decided_at=None):
    """Stands in for ratings.decide (the contract's keys). Two calling stations always split (every x is 0);
    any other pairing gets verdict 'a' at look `decided_at`."""
    def decide(xs, design, n_pairings, looks_used, groups=None):
        even_pair = all(x == 0 for x in xs)
        calls.append({'n': len(xs), 'design': design, 'n_pairings': n_pairings, 'looks': looks_used,
                      'stations': even_pair, 'groups': groups})
        verdict = 'exhausted' if looks_used >= design['max_rounds'] else 'open'
        if not even_pair and looks_used == decided_at:
            verdict = 'a'
        return {'deals': len(xs), 'blocks': len(xs) // design['block'], 'edge': 0.0, 'ci95': None, 't': 0.0,
                'boundary': 3.0, 'verdict': verdict, 'looks_used': looks_used, 'looks_planned': design['max_rounds'],
                'power': {'deals_for_80pct': {'20': None, '50': None, '100': None}}}
    return decide


class TestRounds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'players.toml').write_text(PLAYERS)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, **kw):
        return make_plan([], deals=10, out=str(self.root / 'runs' / 't'), players_file=str(self.root / 'players.toml'),
                         matchups=[['station', 'station2'], ['rule', 'random']], **kw)

    def folders(self, plan):
        return sorted(p.name for p in Path(plan['out']).iterdir() if p.is_dir())

    def test_design_defaults_and_legacy(self):
        p = make_plan(['a', 'b'], deals=50)
        self.assertEqual(p['design'], {'alpha': 0.05, 'round_deals': 50, 'max_rounds': 6, 'block': 10})
        self.assertEqual(p['replicates'], 1)
        self.assertEqual(plan_design({'deals': 20, 'until_decided': 60, 'seed': 0})['max_rounds'], 3)
        self.assertIsNone(plan_design({'deals': 20, 'seed': 0}))

    def test_seeds(self):
        p = {**self.plan(replicates=2), 'seed': 100}
        seen = [s for r in (1, 2, 3) for j in (1, 2) for s in round_seeds(p, r, j)]
        self.assertEqual(len(seen), len(set(seen)))                     # no deal repeats inside a plan
        self.assertEqual(round_seeds(p, 1, 1), list(range(100, 110)))
        one = {**self.plan(), 'seed': 100}
        self.assertEqual(round_seeds(one, 2, 1), list(range(110, 120)))  # one replicate: the old layout

    def test_loop_stops_on_verdict_and_never_exceeds_max_rounds(self):
        calls = []
        plan = self.plan(design={'max_rounds': 3})
        out = Path(plan['out'])
        with mock.patch.object(ratings, 'decide', fake_decide(calls, decided_at=2), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertEqual(self.folders(plan), ['rule__vs__random', 'rule__vs__random__r2', 'station__vs__station2',
                                              'station__vs__station2__r2', 'station__vs__station2__r3'])
        self.assertEqual(sorted(c['looks'] for c in calls if c['stations']), [1, 2, 3])
        self.assertEqual(sorted(c['looks'] for c in calls if not c['stations']), [1, 2])
        self.assertTrue(all(c['n_pairings'] == 2 for c in calls))           # alpha is split over every pairing
        self.assertTrue(all(c['n'] == 10 * c['looks'] for c in calls))      # every deal so far, pooled
        self.assertEqual(calls[0]['design'], plan_design(plan))
        self.assertEqual({k: calls[0]['design'][k] for k in ('alpha', 'round_deals', 'max_rounds', 'block')},
                         {'alpha': 0.05, 'round_deals': 10, 'max_rounds': 3, 'block': 10})
        life = json.loads((out / 'tournament.json').read_text())
        self.assertEqual(life['status'], 'finished')
        self.assertEqual(life['round'], 3)
        self.assertEqual(life['pairings'], {
            'station|station2': {'status': 'exhausted', 'verdict': 'exhausted', 'looks_used': 3, 'round': 3},
            'rule|random': {'status': 'decided', 'verdict': 'a', 'looks_used': 2, 'round': 2}})
        dec = json.loads((out / 'decisions.json').read_text())
        self.assertEqual([v['verdict'] for v in dec['rule__vs__random']], ['open', 'a'])
        self.assertEqual([v['verdict'] for v in dec['station__vs__station2']], ['open', 'open', 'exhausted'])
        # every pairing plays the same deals in a round
        c1 = json.loads((out / 'rule__vs__random__r2' / 'a-first' / 'config.json').read_text())
        c2 = json.loads((out / 'station__vs__station2__r2' / 'a-first' / 'config.json').read_text())
        self.assertEqual(c1['seeds'], c2['seeds'])
        self.assertEqual(c1['deal_fingerprints'], c2['deal_fingerprints'])
        self.assertEqual((c1['round'], c1['replicate'], c1['pairing']), (2, 1, ['rule', 'random']))
        manifest = json.loads((out / 'plan.json').read_text())
        self.assertEqual(manifest['design']['max_rounds'], 3)
        self.assertEqual(manifest['specs']['rule'], {'bot': 'rule', 'sims': 30})

        # a rerun resumes from the record: the same looks, no new rounds
        calls.clear()
        with mock.patch.object(ratings, 'decide', fake_decide(calls, decided_at=2), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertEqual(len(self.folders(plan)), 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(json.loads((out / 'decisions.json').read_text())['station__vs__station2'][-1]['looks_used'], 3)

    def test_one_normalized_design(self):
        seen = []
        norm = lambda plan: {'alpha': 0.05, 'round_deals': plan['deals'], 'max_rounds': 2, 'block': 10,
                             'replicates': plan.get('replicates', 1), 'target_edge_bb100': 50, 'deals_planned': 40}
        decide = fake_decide(seen)
        plan = self.plan(design={'max_rounds': 2}, replicates=2)
        plan['matchups'] = [['station', 'station2']]
        with mock.patch.object(ratings, 'normalize_design', norm, create=True), \
                mock.patch.object(ratings, 'decide', decide, create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
            self.assertTrue(all(c['design'] == norm(plan) for c in seen))   # the same design, replicates in it
            stored = json.loads((Path(plan['out']) / 'plan.json').read_text())
            self.assertEqual(stored['design'], norm(plan))
            asyncio.run(run_plan(plan, log=lambda *a: None))                   # and a rerun matches it

    def test_lifecycle_while_playing_and_when_stopped(self):
        out = Path(self.plan()['out'])
        snapshots = []
        inner = fake_decide([])

        def decide(xs, design, n, looks, groups=None):
            snapshots.append(json.loads((out / 'tournament.json').read_text()))
            if not all(x == 0 for x in xs):
                raise RuntimeError('boom')
            return inner(xs, design, n, looks, groups)
        plan = self.plan(design={'max_rounds': 2})
        with mock.patch.object(ratings, 'decide', decide, create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertTrue(all(s['status'] == 'running' and s['pid'] for s in snapshots))
        self.assertIn('playing', {p['status'] for s in snapshots for p in s['pairings'].values()})
        life = json.loads((out / 'tournament.json').read_text())
        self.assertEqual(life['status'], 'stopped')                          # one pairing failed: not finished
        self.assertEqual(life['pairings']['rule|random']['status'], 'waiting')
        self.assertIn('boom', life['pairings']['rule|random']['error'])
        self.assertEqual(life['pairings']['station|station2']['status'], 'exhausted')

    def test_replicates(self):
        calls = []
        plan = self.plan(design={'max_rounds': 2}, replicates=2)
        plan['matchups'] = [['station', 'station2']]
        with mock.patch.object(ratings, 'decide', fake_decide(calls), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        folders = self.folders(plan)
        self.assertEqual(folders, ['station__vs__station2', 'station__vs__station2__r2', 'station__vs__station2__x2',
                                   'station__vs__station2__x2__r2'])
        self.assertEqual([(c['n'], c['looks']) for c in calls], [(20, 1), (40, 2)])
        self.assertEqual(sorted(set(calls[-1]['groups'])), folders)          # one group per match
        first = [json.loads((Path(plan['out']) / f / 'a-first' / 'config.json').read_text())['seeds'][0] for f in folders]
        self.assertEqual(len(set(first)), 4)

    @unittest.skipUnless(hasattr(ratings, 'decide') and hasattr(ratings, 'boundary'), 'ratings.decide not shipped yet')
    def test_with_real_decide(self):
        plan = self.plan(design={'max_rounds': 2})
        asyncio.run(run_plan(plan, log=lambda *a: None))
        for looks in json.loads((Path(plan['out']) / 'decisions.json').read_text()).values():
            self.assertLessEqual(len(looks), 2)
            self.assertTrue(looks[-1]['verdict'] in ('a', 'b', 'even') or len(looks) == 2, looks[-1])


if __name__ == '__main__':
    unittest.main()
