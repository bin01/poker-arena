"""The viewer's data: runs and progress, matchups (planned ones too), matches, ratings with stats, the
static export, and the HTTP server."""
import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from poker_arena import server
from poker_arena.games import get_game
from poker_arena.runner import make_plan, run_plan
from poker_arena.stats import collect

FAKE = [sys.executable, str(Path(__file__).with_name('fake_model.py'))]


class TestViewerData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        players = cls.root / 'players.toml'
        players.write_text(f'[players.fake]\nlabel = "Fake Model 1"\nbackend = "command"\ncommand = {json.dumps(FAKE)}\n'
                           '[players.rule]\nbot = "rule"\nsims = 40\n[players.station]\nbot = "station"\n'
                           '[players.random]\nbot = "random"\n')
        out = cls.root / 'runs' / 'pilot'
        # three of the plan's four matchups are played; the fourth stays "not started"
        plan = make_plan(['fake', 'rule', 'station'], deals=6, out=str(out), players_file=str(players),
                         matchups=[['fake', 'rule'], ['rule', 'station'], ['fake', 'station'], ['station', 'random']])
        asyncio.run(run_plan(plan, only=['fake__vs__rule', 'rule__vs__station', 'fake__vs__station'], log=lambda *a: None))
        cls.runs = cls.root / 'runs'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_and_progress(self):
        rs = server.runs(self.runs)
        self.assertEqual([r['run'] for r in rs], ['pilot'])
        r = rs[0]
        self.assertEqual(r['matchups'], 4)
        self.assertEqual(r['matchups_started'], 3)
        self.assertEqual(r['hands'], 3 * 2 * 6)
        self.assertEqual(r['hands_planned'], 4 * 2 * 6)
        self.assertFalse(r['finished'])

    def test_index_lists_planned_matchups(self):
        ms = server.index(self.runs, 'pilot')
        planned = [m for m in ms if not m['started']]
        self.assertEqual([(m['a'], m['b']) for m in planned], [('station', 'random')])
        played = {m['id']: m for m in ms if m['started']}
        m = played['pilot/fake__vs__rule']
        self.assertEqual(m['pairs_done'], 6)
        self.assertEqual(m['kinds'], {'fake': 'model', 'rule': 'bot'})
        self.assertEqual(m['labels'], {'fake': 'Fake Model 1', 'rule': 'Rule bot'})
        self.assertEqual(planned[0]['labels'].get('fake'), 'Fake Model 1')  # from plan.json
        self.assertTrue(m['finished'])
        self.assertNotIn('times', m)

    def test_match_and_containment(self):
        mt = server.match(self.runs, 'pilot/fake__vs__rule/a-first')
        self.assertEqual(len(mt['hands']), 6)
        kinds = {e['kind'] for evs in mt['events'].values() for e in evs}
        self.assertTrue({'hand_start', 'thought', 'action', 'result'} <= kinds)
        self.assertIsNone(server.match(self.runs, '../../etc'))

    def test_ratings_carry_stats(self):
        r = server.ratings_for(self.runs, 'pilot', boot=20)
        self.assertEqual({p['player'] for p in r['players']}, {'fake', 'rule', 'station'})
        self.assertIn('fake', r['stats'])
        self.assertIsNone(server.ratings_for(self.runs, '../..'))

    def test_style_stats(self):
        s = collect(self.runs / 'pilot')
        self.assertEqual(s['station']['pfr'], 0)          # a calling station never raises
        self.assertEqual(s['station']['af'], 0)
        self.assertEqual(s['rule']['calls'], 0)           # bots make no CLI calls
        self.assertGreater(s['fake']['calls'], s['fake']['decisions'] / 2)
        self.assertGreater(s['fake']['retries'], 0)       # the stand-in garbles every 7th reply
        for p in s.values():
            for k in ('vpip', 'pfr', 'wtsd', 'wsd'):
                self.assertTrue(p[k] is None or 0 <= p[k] <= 1, (k, p[k]))

    def test_static_export(self):
        with tempfile.TemporaryDirectory() as d:
            n = server.export(self.runs, d)
            data = Path(d) / 'data'
            self.assertIn('window.POKER_STATIC', (Path(d) / 'index.html').read_text())
            self.assertTrue((Path(d) / 'app.js').exists())
            self.assertEqual(n, len(list(data.iterdir())))
            self.assertTrue((data / 'index-pilot.json').exists())
            self.assertTrue((data / 'ratings-pilot.json').exists())
            m = json.loads((data / f"match-{server.key('pilot/fake__vs__rule/a-first')}.json").read_text())
            self.assertEqual(m['config']['specs']['fake'], {'label': 'Fake Model 1'})  # no commands or paths

    def test_key_matches_the_app(self):
        self.assertEqual(server.key('pilot/a__vs__b-1/a-first'), 'pilot~2Fa__vs__b-1~2Fa-first')
        self.assertEqual(server.key('.'), '~2E')
        self.assertEqual(server.key('é'), '~C3~A9')

    def test_http(self):
        import socket
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        threading.Thread(target=server.serve, args=(self.runs, port), daemon=True).start()
        base = f'http://127.0.0.1:{port}'
        for _ in range(50):
            try:
                urllib.request.urlopen(base + '/api/runs')
                break
            except OSError:
                time.sleep(0.05)
        get = lambda p: json.loads(urllib.request.urlopen(base + p).read())
        self.assertEqual(get('/api/runs')[0]['run'], 'pilot')
        self.assertEqual(len(get('/api/index?run=pilot')), 4)
        self.assertTrue(get('/api/ratings?run=pilot')['players'])
        self.assertEqual(len(get('/api/match?id=pilot/fake__vs__rule/b-first')['hands']), 6)
        page = urllib.request.urlopen(base + '/').read().decode()
        self.assertIn('app.js', page)
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(base + '/api/match?id=../x')



class TestNames(unittest.TestCase):
    def test_display_names(self):
        self.assertEqual(server.pretty_model('claude-opus-5-5'), 'Claude Opus 5.5')
        self.assertEqual(server.pretty_model('gpt-6-luna'), 'GPT-6 Luna')
        self.assertEqual(server.pretty_model('gemini-4-neon-high'), 'Gemini 4 Neon High')
        self.assertEqual(server.display_name('x', {'backend': 'codex', 'model': 'gpt-6-luna', 'effort': 'high'}), 'GPT-6 Luna · high')
        self.assertEqual(server.display_name('x', {'label': 'Mine', 'model': 'm'}), 'Mine')
        self.assertEqual(server.display_name('codex-gpt', {'backend': 'codex'}), 'codex-gpt')  # no model: the id


if __name__ == '__main__':
    unittest.main()
