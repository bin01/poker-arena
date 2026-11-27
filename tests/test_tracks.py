"""Two tracks of one tournament: the official track (fresh conversation every deal, assumption-free verdicts,
the only one that ranks) and the adaptation track (long conversations, judged and stopped by the model-based
gate, never official). Plan generation, adjudication by track, the viewer's data, history kept apart, and an
end-to-end bots run of both tracks."""
import asyncio
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poker_arena import cli, history, server
from poker_arena import ratings as R
from poker_arena.runner import make_tournament, plan_tracks, run_plan
from test_ratings import deals_between, plan, run

VIEWER = Path(server.__file__).with_name('viewer')
PLAYERS = '[players.rule]\nbot = "rule"\nsims = 40\n[players.station]\nbot = "station"\n'


class TestPlan(unittest.TestCase):
    def test_tournament_has_one_plan_per_track(self):
        doc = make_tournament(['a', 'b', 'c'], out='runs/cup')
        self.assertEqual(doc['tournament'], 'cup')
        off, ad = doc['tracks']['official'], doc['tracks']['adaptation']
        self.assertEqual((off['out'], ad['out']), ('runs/cup/official', 'runs/cup/adaptation'))
        self.assertEqual((off['deals'], off['replicates']), (1, 25))
        self.assertEqual((ad['deals'], ad['replicates']), (R.ADAPTATION_DEALS, R.ADAPTATION_REPLICATES))
        self.assertEqual((off['track'], ad['track'], ad['design']['track']), ('official', 'adaptation', 'adaptation'))
        self.assertEqual(off['seed'], ad['seed'])
        self.assertEqual(list(plan_tracks(doc, 'adaptation')), ['adaptation'])
        self.assertEqual(list(plan_tracks(off)), ['official'])
        with self.assertRaises(KeyError):
            plan_tracks(make_tournament(['a', 'b'], tracks=['official']), 'adaptation')
        with self.assertRaises(ValueError):
            make_tournament(['a', 'b'], tracks=['other'])

    def test_adaptation_design_is_model_based_and_powered(self):
        doc = make_tournament(['a', 'b', 'c'], out='runs/cup')
        d = R.normalize_design(doc['tracks']['adaptation'])
        self.assertEqual((d['track'], d['official'], d['model_based'], d['assumption_free']),
                         ('adaptation', False, True, False))
        self.assertEqual(d['round_deals'], 50)
        self.assertEqual(d['max_rounds_basis'], 'power')
        self.assertLessEqual(d['max_rounds'], R.model_rounds_cap(d['replicates']))
        self.assertGreaterEqual(d['power_at_max'], 0.8)
        self.assertEqual(d['calibration']['test'], 'model_based')
        self.assertIn('not official', d['label'])
        o = R.normalize_design(doc['tracks']['official'])
        self.assertEqual((o['track'], o['official'], o['assumption_free']), ('official', True, True))
        self.assertNotEqual(R.calibration_id(d), R.calibration_id(o))
        self.assertNotEqual(R._inputs(d)['track'], R._inputs(o)['track'])
        capped = R.normalize_design({**doc['tracks']['adaptation'], 'design': {'track': 'adaptation', 'budget_deals': 400}})
        self.assertEqual((capped['max_rounds_basis'], capped['max_rounds']), ('budget', 2))
        self.assertTrue(capped['underpowered'])
        with self.assertRaises(ValueError):
            R.normalize_design({**plan(['a', 'b'], 1, 3), 'track': 'adaptation'})

    def test_cli_writes_both_tracks_with_costs(self):
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / 'players.toml'
            pf.write_text(PLAYERS + '[players.ref]\nbot = "reference"\n')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(['plan', 'rule', 'station', 'ref', '--players', str(pf), '--write', str(Path(d) / 'p.json'),
                          '--out', str(Path(d) / 'runs' / 'cup')])
            text = out.getvalue()
            self.assertIn('OFFICIAL track', text)
            self.assertIn('ADAPTATION track', text)
            self.assertIn('hands a round', text)
            self.assertIn('never decide the official leaderboard', text)
            doc = json.loads((Path(d) / 'p.json').read_text())
            self.assertEqual(sorted(doc['tracks']), ['adaptation', 'official'])
            for t, p in doc['tracks'].items():
                self.assertEqual(p['design']['max_rounds'], R.normalize_design(p)['max_rounds'])
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(['plan', 'rule', 'station', '--players', str(pf), '--write', str(Path(d) / 'q.json'),
                          '--out', str(Path(d) / 'runs' / 'q'), '--tracks', 'official'])
            self.assertEqual(list(json.loads((Path(d) / 'q.json').read_text())['tracks']), ['official'])


class TestAdjudicationByTrack(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        true = {'x': 300.0, 'y': 0.0, 'z': -300.0}
        self.deals = []
        for a, b in (('x', 'y'), ('x', 'z'), ('y', 'z')):
            self.deals += deals_between(a, b, true[a] - true[b], 10, rng, sd=50, rounds=4, reps=4)

    def test_adaptation_verdicts_are_model_based_and_never_official(self):
        r = run(self.deals, plan(['x', 'y', 'z'], 10, 4, reps=4, track='adaptation'))
        self.assertEqual((r['track'], r['official'], r['model_based']), ('adaptation', False, True))
        for m in r['pairings']:
            self.assertEqual(m['verdict'], 'a')
            self.assertEqual((m['track'], m['official'], m['model_based'], m['recorded']), ('adaptation', False, True, True))
            self.assertFalse(m['counts_toward_leader'])
            self.assertFalse(m['inferential'])
            self.assertTrue(m['model_based_counts'])
            self.assertEqual(m['gate']['verdict'], 'a')
            self.assertIn('not official', m['standing'])
        self.assertIsNone(r['leader'])
        self.assertEqual(r['leaders'], [None])
        self.assertTrue(all(p['rank'] is None and p['rank_kind'] == 'descriptive' and not p['inferential'] for p in r['players']))
        self.assertEqual(r['groups'][0]['model_based_leader'], 'x')
        self.assertIsNone(r['groups'][0]['leader'])
        self.assertIn('Adaptation track', r['notes'][0])
        text = R.summary(r)
        self.assertIn('model-based, not official', text)
        self.assertNotIn('Decided leader', text)

    def test_official_track_on_the_same_data_decides(self):
        r = run(self.deals, plan(['x', 'y', 'z'], 10, 4, reps=4, game_config={'stack': 20}))
        self.assertEqual((r['track'], r['official']), ('official', True))
        self.assertTrue(all(m['official'] and m['track'] == 'official' for m in r['pairings']))

    def test_adjudicate_stops_on_the_gate(self):
        p = {**plan(['x', 'y'], 10, 4, reps=4), 'track': 'adaptation'}
        d = R.normalize_design(p)
        xs = [x['net_a'] / x['bb'] / 2 * 100 for x in self.deals if (x['a'], x['b']) == ('x', 'y')]
        groups = [x['matchup'] for x in self.deals if (x['a'], x['b']) == ('x', 'y')]
        v = R.adjudicate({'a': 'x', 'b': 'y', 'xs': xs, 'groups': groups, 'rejected': [], 'evidence_hash': 'h'}, d, 1, 4, None)
        self.assertEqual(v['verdict'], 'ineligible')     # no facts: still ineligible, as on the official track
        self.assertEqual(v['test_verdict'], 'a')         # the gate's verdict, not the betting test's
        self.assertNotEqual(v['assumption_free_verdict'], 'a')
        self.assertFalse(v['official'])
        self.assertIn('calibrated_family', v['assumptions']['checks'])


class TestHistoryApart(unittest.TestCase):
    def test_tracks_never_pool(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve() / 'runs'
            _play_both(root, Path(d) / 'players.toml', 'cup')
            history.sync(root)
            off = history.all_time(root / history.DB_NAME, boot=20, players_file='/none')
            ad = history.all_time(root / history.DB_NAME, boot=20, players_file='/none', track='adaptation')
            self.assertEqual([x['run'] for x in off['runs']], ['cup/official'])
            self.assertEqual([x['run'] for x in ad['runs']], ['cup/adaptation'])
            self.assertEqual(off['other_track_runs'], ['cup/adaptation'])
            self.assertTrue(any('never pooled' in n for n in off['notes']))
            self.assertNotEqual(off['deals'], ad['deals'])


def _play_both(root: Path, pf: Path, name: str, adaptation_deals=4, rounds=2):
    pf.write_text(PLAYERS)
    doc = make_tournament(['rule', 'station'], out=str(root / name), players_file=str(pf),
                          official={'deals': 1, 'replicates': 3, 'design': {'max_rounds': rounds}},
                          adaptation={'deals': adaptation_deals, 'replicates': 2, 'design': {'max_rounds': rounds}})

    async def go():
        await asyncio.gather(*(run_plan(p, log=lambda *a: None) for p in doc['tracks'].values()))
    asyncio.run(go())
    return doc


class TestBothTracksEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve() / 'runs'
        cls.doc = _play_both(cls.root, Path(cls.tmp.name) / 'players.toml', 'cup')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_track_is_its_own_registered_run(self):
        for t in ('official', 'adaptation'):
            p = json.loads((self.root / 'cup' / t / 'plan.json').read_text())
            self.assertEqual((p['track'], p['tournament'], p['design']['track']), (t, 'cup', t))
            self.assertEqual(p['registration']['status'], 'registered')
            self.assertEqual(p['registration']['design']['track'], t)
        dec = json.loads((self.root / 'cup' / 'adaptation' / 'decisions.json').read_text())
        look = dec['rule__vs__station'][-1]
        self.assertEqual((look['track'], look['official'], look['model_based']), ('adaptation', False, True))
        self.assertTrue(look['registered_record'])
        hands = sum(1 for _ in (self.root / 'cup' / 'adaptation').glob('*/a-first/hands.jsonl'))
        self.assertEqual(hands, 2 * len(dec['rule__vs__station']))

    def test_ratings_by_track(self):
        off = R.ratings(self.root / 'cup' / 'official', boot=20)
        ad = R.ratings(self.root / 'cup' / 'adaptation', boot=20)
        self.assertEqual((off['track'], off['official'], ad['track'], ad['official']), ('official', True, 'adaptation', False))
        self.assertEqual(off['deals'], 2 * 3)
        self.assertEqual(ad['deals'], 2 * 2 * 4)
        self.assertTrue(all(m['official'] for m in off['pairings']))
        self.assertTrue(all(not m['official'] and m['model_based'] is True and m['recorded'] for m in ad['pairings']))
        self.assertEqual(ad['tournament'], 'cup')
        self.assertEqual(sorted(R.track_runs(self.root / 'cup')), ['adaptation', 'official'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(['ratings', str(self.root / 'cup'), '--boot', '20'])
        self.assertIn('== official track', out.getvalue())
        self.assertIn('== adaptation track', out.getvalue())

    def test_viewer_groups_the_tracks(self):
        rs = {r['run']: r for r in server.runs(self.root)}
        off, ad = rs['cup/official'], rs['cup/adaptation']
        self.assertEqual((off['track'], off['tournament'], off['title']), ('official', 'cup', 'cup'))
        self.assertEqual((ad['track'], ad['tournament'], ad['title']), ('adaptation', 'cup', 'cup'))
        self.assertTrue(off['finished'] and ad['finished'])
        r = server.ratings_for(self.root, 'cup/adaptation', boot=20)
        self.assertEqual((r['track'], r['label']), ('adaptation', R.ADAPTATION_LABEL))
        allr = server.ratings_for(self.root, history.ALL, boot=20)
        self.assertEqual(allr['other_track_runs'], ['cup/adaptation'])

    def test_resume_is_a_no_op(self):
        before = (self.root / 'cup' / 'adaptation' / 'decisions.json').read_text()
        asyncio.run(run_plan(self.doc['tracks']['adaptation'], log=lambda *a: None))
        self.assertEqual(json.loads(before).keys(), json.loads((self.root / 'cup' / 'adaptation' / 'decisions.json').read_text()).keys())

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_viewer_words(self):
        raw = R.ratings(self.root / 'cup' / 'adaptation', boot=20)
        p = {**raw['pairings'][0], 'verdict': 'a', 'track': 'adaptation', 'recorded': True}
        script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))});'
                  f'const rt = Core.normalize({json.dumps({**raw, "pairings": [p], "groups": None})});'
                  'const q = rt.pairings[0]; const v = Core.verdict(q, x => x); const s = Core.sourceOf(q);'
                  'process.stdout.write(JSON.stringify([v.decided, v.tone, v.short, v.long, s.short, rt.groups[0].leader]));')
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        decided, tone, short, long, src, leader = json.loads(res.stdout)
        self.assertFalse(decided)
        self.assertEqual(tone, 'model')
        self.assertIn('model-based', short)
        self.assertIn('not official', long)
        self.assertIn('not official', src)
        self.assertIsNone(leader)


if __name__ == '__main__':
    unittest.main()
