"""Sequential tournaments: the round loop enforces the plan's design through ratings.adjudicate (stops on a
verdict, never more looks than planned), replicates get fresh memories and their own deals, and every
pairing plays the same deals."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from poker_arena import ratings
from poker_arena.runner import PlanConflict, make_plan, plan_design, round_seeds, run_plan

PLAYERS = ('[players.rule]\nbot = "rule"\nsims = 30\n[players.random]\nbot = "random"\n'
           '[players.station]\nbot = "station"\n[players.station2]\nbot = "station"\n')


def fake_adjudicate(calls, decided_at=None, ineligible=None):
    """Stands in for ratings.adjudicate (the contract's signature and keys). Two calling stations always split
    (every x is 0); any other pairing gets verdict 'a' at look `decided_at`. `ineligible(eligibility)` -> reasons
    makes the pairing ineligible."""
    def adjudicate(pairing, design, n_pairings, looks_used, eligibility):
        xs, groups = pairing['xs'], pairing['groups']
        even_pair = all(x == 0 for x in xs)
        calls.append({'n': len(xs), 'design': design, 'n_pairings': n_pairings, 'looks': looks_used,
                      'stations': even_pair, 'groups': groups, 'pairing': pairing, 'eligibility': eligibility})
        reasons = ineligible(eligibility) if ineligible else []
        if reasons:
            return {'deals': len(xs), 'verdict': 'ineligible', 'looks_used': looks_used, 'eligible': False,
                    'ineligible_reasons': reasons, 'assumptions': {'status': 'unknown', 'checks': {}}}
        verdict = 'exhausted' if looks_used >= design['max_rounds'] else 'open'
        if not even_pair and looks_used == decided_at:
            verdict = 'a'
        return {'deals': len(xs), 'blocks': len(xs) // design['block'], 'edge': 0.0, 'ci95': None, 't': 0.0,
                'boundary': 3.0, 'verdict': verdict, 'looks_used': looks_used, 'looks_planned': design['max_rounds'],
                'power': {'deals_for_80pct': {'20': None, '50': None, '100': None}}, 'eligible': True,
                'ineligible_reasons': [], 'assumptions': {'status': 'supported', 'checks': {}}}
    return adjudicate


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
        return sorted(p.name for p in Path(plan['out']).iterdir() if p.is_dir() and p.name != 'source')

    def test_design_defaults_and_legacy(self):
        p = make_plan(['a', 'b'], deals=50)
        self.assertEqual(p['design'], {'alpha': 0.05, 'round_deals': 50, 'block': 10})   # max_rounds from the power target
        self.assertGreater(plan_design(p)['max_rounds'], 1)
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
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate(calls, decided_at=2), create=True):
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
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate(calls, decided_at=2), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertEqual(len(self.folders(plan)), 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(json.loads((out / 'decisions.json').read_text())['station__vs__station2'][-1]['looks_used'], 3)

    def test_one_normalized_design(self):
        seen = []
        norm = lambda plan: {'alpha': 0.05, 'round_deals': plan['deals'], 'max_rounds': 2, 'block': 10,
                             'replicates': plan.get('replicates', 1), 'target_edge_bb100': 50, 'deals_planned': 40}
        decide = fake_adjudicate(seen)
        plan = self.plan(design={'max_rounds': 2}, replicates=2)
        plan['matchups'] = [['station', 'station2']]
        with mock.patch.object(ratings, 'normalize_design', norm, create=True), \
                mock.patch.object(ratings, 'adjudicate', decide, create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
            self.assertTrue(all(c['design'] == norm(plan) for c in seen))   # the same design, replicates in it
            stored = json.loads((Path(plan['out']) / 'plan.json').read_text())
            self.assertEqual(stored['design'], norm(plan))
            asyncio.run(run_plan(plan, log=lambda *a: None))                   # and a rerun matches it

    def test_the_runner_plays_the_normalized_design_only(self):
        norm = lambda plan: {'alpha': 0.05, 'round_deals': 10, 'max_rounds': 1, 'block': 10, 'replicates': 3}
        plan = self.plan(design={'max_rounds': 1})
        plan['matchups'] = [['station', 'station2']]
        calls = []
        with mock.patch.object(ratings, 'normalize_design', norm), \
                mock.patch.object(ratings, 'adjudicate', fake_adjudicate(calls), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertEqual(self.folders(plan), ['station__vs__station2', 'station__vs__station2__x2',
                                              'station__vs__station2__x3'])
        self.assertEqual(round_seeds(plan, 1, 2, norm(plan)), list(range(1010, 1020)))

    def test_conflicting_replicates_are_an_error(self):
        plan = self.plan(design={'max_rounds': 2, 'replicates': 3}, replicates=2)
        with self.assertRaisesRegex(PlanConflict, 'replicates'):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertFalse(Path(plan['out']).exists())                         # before anything is written
        with self.assertRaises(PlanConflict):
            plan_design(plan)
        agreed = self.plan(design={'max_rounds': 2, 'replicates': 2}, replicates=2)
        self.assertEqual(plan_design(agreed)['replicates'], 2)

    def test_lifecycle_while_playing_and_when_stopped(self):
        out = Path(self.plan()['out'])
        snapshots = []
        inner = fake_adjudicate([])

        def decide(pairing, design, n, looks, eligibility):
            snapshots.append(json.loads((out / 'tournament.json').read_text()))
            if not all(x == 0 for x in pairing['xs']):
                raise RuntimeError('boom')
            return inner(pairing, design, n, looks, eligibility)
        plan = self.plan(design={'max_rounds': 2})
        with mock.patch.object(ratings, 'adjudicate', decide, create=True):
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
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate(calls), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        folders = self.folders(plan)
        self.assertEqual(folders, ['station__vs__station2', 'station__vs__station2__r2', 'station__vs__station2__x2',
                                   'station__vs__station2__x2__r2'])
        self.assertEqual([(c['n'], c['looks']) for c in calls], [(20, 1), (40, 2)])
        self.assertEqual(sorted(set(calls[-1]['groups'])), folders)          # one group per match
        first = [json.loads((Path(plan['out']) / f / 'a-first' / 'config.json').read_text())['seeds'][0] for f in folders]
        self.assertEqual(len(set(first)), 4)

    def test_an_ineligible_pairing_stops_with_its_own_status(self):
        calls = []
        plan = self.plan(design={'max_rounds': 3})
        out = Path(plan['out'])
        no_rule = lambda e: ['rule: identity unattested'] if 'rule' in e['players'] else []
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate(calls, ineligible=no_rule), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertEqual([f for f in self.folders(plan) if f.startswith('rule')], ['rule__vs__random'])
        life = json.loads((out / 'tournament.json').read_text())
        self.assertEqual(life['pairings']['rule|random']['status'], 'ineligible')
        self.assertEqual(life['status'], 'finished')
        look = json.loads((out / 'decisions.json').read_text())['rule__vs__random'][-1]
        self.assertEqual((look['verdict'], look['ineligible_reasons']), ('ineligible', ['rule: identity unattested']))
        self.assertIsInstance(look['recorded_at'], int)
        self.assertEqual(json.loads((out / 'plan.json').read_text())['registration']['judge_fingerprint'],
                         ratings.judge_fingerprint())
        self.assertEqual(look['eligibility']['registration']['status'], 'registered')
        self.assertNotIn('design', look['eligibility']['registration'])
        facts = next(c for c in calls if not c['stations'])
        self.assertEqual(facts['pairing']['a'], 'rule')
        self.assertEqual(facts['pairing']['conversations'], ['rule__vs__random'])
        self.assertEqual(facts['eligibility']['registration']['design'], plan_design(plan))
        rule = facts['eligibility']['players']['rule']
        self.assertEqual((rule['kind'], rule['identity'], rule['audit'], rule['protocol_consistent']),
                         ('bot', 'bot', 'complete', True))
        self.assertEqual(rule['decisions'], rule['audit_counts']['bot'])
        # a rerun keeps the record: the pairing is not played again
        calls.clear()
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate(calls, ineligible=no_rule), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertEqual([f for f in self.folders(plan) if f.startswith('rule')], ['rule__vs__random'])

    def test_registration_is_written_before_the_first_hand(self):
        from poker_arena import runner
        plan = self.plan(design={'max_rounds': 1})
        out, seen, real = Path(plan['out']), [], runner.run_match

        async def first(*a, **kw):
            seen.append(json.loads((out / 'plan.json').read_text()).get('registration'))
            return await real(*a, **kw)
        with mock.patch.object(runner, 'run_match', first), \
                mock.patch.object(ratings, 'adjudicate', fake_adjudicate([]), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertTrue(seen and all(r and r['status'] == 'registered' for r in seen))
        self.assertEqual(set(seen[0]), {'status', 'design', 'judge_fingerprint', 'calibration_id', 'created', 'cli_versions',
                                        'cli_artifacts'})
        self.assertEqual(seen[0]['design'], plan_design(plan))

    def test_no_registration_when_declined_or_after_play(self):
        plan = self.plan(design={'max_rounds': 1, 'registered': False})
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate([]), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertNotIn('registration', json.loads((Path(plan['out']) / 'plan.json').read_text()))
        plan = self.plan(design={'max_rounds': 2})
        plan['out'] += '-old'
        path = Path(plan['out']) / 'plan.json'
        with mock.patch.object(ratings, 'adjudicate', fake_adjudicate([]), create=True):
            asyncio.run(run_plan(plan, log=lambda *a: None, only=['station__vs__station2']))
            stored = json.loads(path.read_text())
            del stored['registration']                       # a run started by code that wrote none
            path.write_text(json.dumps(stored))
            asyncio.run(run_plan(plan, log=lambda *a: None))
        self.assertNotIn('registration', json.loads(path.read_text()))   # never granted after hands were played

    @unittest.skipUnless(hasattr(ratings, 'adjudicate'), 'ratings.adjudicate not shipped yet')
    def test_with_real_decide(self):
        plan = self.plan(design={'max_rounds': 2})
        asyncio.run(run_plan(plan, log=lambda *a: None))
        for looks in json.loads((Path(plan['out']) / 'decisions.json').read_text()).values():
            self.assertLessEqual(len(looks), 2)
            self.assertTrue(looks[-1]['verdict'] in ('a', 'b', 'even') or len(looks) == 2, looks[-1])
            self.assertTrue(all(v['official'] and v['eligible'] and 'recorded_at' in v for v in looks), looks[-1])


if __name__ == '__main__':
    unittest.main()
