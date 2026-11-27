"""The viewer's data: runs and progress, matchups (planned ones too), matches, ratings with stats, the
static export, and the HTTP server."""
import asyncio
import json
import shutil
import subprocess
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
                         matchups=[['fake', 'rule'], ['rule', 'station'], ['fake', 'station'], ['station', 'random']],
                         design={'max_rounds': 1})
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
        self.assertEqual(planned[0]['labels'], {'station': 'Calling station', 'random': 'Random bot'})  # resolved before play
        self.assertEqual(planned[0]['kinds'], {'station': 'bot', 'random': 'bot'})
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
        self.assertEqual({x['player']: x['status'] for x in r['roster']}['random'], 'awaiting')
        self.assertEqual(r['labels']['random'], 'Random bot')
        self.assertEqual(server.roster(self.runs, 'pilot')[-1], {'player': 'random', 'status': 'awaiting'})
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
            for f in ('app.js', 'core.js', 'table.js', 'style.css'):
                self.assertTrue((Path(d) / f).exists(), f)
            self.assertEqual(n, len(list(data.iterdir())))
            self.assertTrue((data / 'index-pilot.json').exists())
            self.assertTrue((data / 'ratings-pilot.json').exists())
            m = json.loads((data / f"match-{server.key('pilot/fake__vs__rule/a-first')}.json").read_text())
            self.assertEqual(m['config']['specs']['fake'], {'label': 'Fake Model 1'})  # no commands or paths

    def test_lifecycle(self):
        run = self.runs / 'pilot'
        t = run / 'tournament.json'
        recorded = t.read_text() if t.exists() else None
        try:
            if recorded is not None:
                t.unlink()
            life = server.lifecycle(self.runs, 'pilot', server.index(self.runs, 'pilot'))
            self.assertEqual(life['status'], 'running')              # from the plan: station-vs-random has not played
            self.assertEqual(life['pairings']['station|random']['status'], 'waiting')
            self.assertIn(life['pairings']['fake|rule']['status'], ('decided', 'exhausted', 'ineligible'))
            t.write_text(json.dumps({'status': 'stopped', 'round': 1, 'pairings': {}, 'updated': 1}))
            r = server.runs(self.runs)[0]
            self.assertEqual((r['status'], r['finished']), ('stopped', False))   # the runner's own file wins
        finally:
            t.unlink(missing_ok=True)
            if recorded is not None:
                t.write_text(recorded)

    def test_lifecycle_waits_for_planned_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / 'r').mkdir()
            (root / 'r' / 'plan.json').write_text(json.dumps({'matchups': [['a', 'b']], 'deals': 10, 'design': {'max_rounds': 3}}))
            ms = [{'a': 'a', 'b': 'b', 'started': True, 'finished': True, 'live': False, 'round': 1}]
            (root / 'r' / 'decisions.json').write_text(json.dumps({'a__vs__b': [{'verdict': 'open', 'looks_used': 1}]}))
            self.assertEqual(server.lifecycle(root, 'r', ms)['status'], 'running')   # between rounds, not finished
            (root / 'r' / 'decisions.json').write_text(json.dumps({'a__vs__b': [{'verdict': 'b', 'looks_used': 1}]}))
            life = server.lifecycle(root, 'r', ms)
            self.assertEqual((life['status'], life['pairings']['a|b']['status']), ('finished', 'decided'))

    def test_run_audit(self):
        a = server.audit(self.runs, 'pilot')
        self.assertEqual(set(a['players']), {'fake'})                  # bots make no calls
        f = a['players']['fake']
        self.assertGreater(f['decisions'], 0)
        self.assertGreaterEqual(f['calls'], f['decisions'])
        self.assertEqual(sum(f['identity'].values()), f['calls'] - f['backend_failures'])
        self.assertEqual(f['identity'], {'unattested': f['calls']})    # a command backend never says which model answered
        self.assertGreater(f['retries'], 0)                            # the stand-in garbles every 7th reply
        self.assertEqual(a['totals']['decisions'], f['decisions'])
        self.assertEqual(server.ratings_for(self.runs, 'pilot', boot=20)['run_audit']['totals'], a['totals'])

    def test_identity_of_a_call(self):
        self.assertEqual(server._identity({'identity_match': 'unattested', 'served_model': 'x'}, 'x'), 'unattested')
        self.assertEqual(server._identity({'served_model': 'claude-opus-5-5'}, 'claude-opus-5-5'), 'match')
        self.assertEqual(server._identity({'observed_model': 'claude-haiku-5'}, 'claude-opus-5-5'), 'mismatch')
        self.assertEqual(server._identity({}, 'claude-opus-5-5'), 'unattested')
        # a reported model is not a matching one: 'attested' from the backend never displays as a match
        probe = {'identity': 'attested', 'requested_model': 'claude-sonnet-5-5', 'observed_model': 'claude-haiku-5-5'}
        self.assertEqual(server._identity(probe, 'claude-sonnet-5-5'), 'mismatch')
        self.assertEqual(server._identity({'identity': 'attested', 'requested_model': 'claude-sonnet-5', 'observed_model': 'claude-sonnet-5-5'}, None), 'mismatch')
        self.assertEqual(server._identity({'identity_match': 'mismatch', 'observed_model': 'claude-opus-5-5'}, 'claude-opus-5-5'), 'mismatch')
        self.assertEqual(server._identity({'observed_models': ['gpt-6-luna', 'gpt-6-sol']}, 'gpt-6-luna'), 'mismatch')

    def test_audit_never_calls_a_wrong_model_confirmed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            m = root / 'r' / 'x__vs__y' / 'a-first'
            m.mkdir(parents=True)
            (m / 'config.json').write_text(json.dumps({'players': ['x', 'y'], 'specs': {'x': {'model': 'claude-sonnet-5-5'}, 'y': {'bot': 'rule'}}}))
            (m / 'calls.jsonl').write_text(json.dumps({'player': 'x', 'identity': 'attested', 'requested_model': 'claude-sonnet-5-5',
                                                       'observed_model': 'claude-haiku-5-5', 'audit': 'verified'}) + '\n')
            a = server.audit(root, 'r')['players']['x']
            self.assertEqual(a['identity'], {'mismatch': 1})
            r = self.run_js_identity(a)
            self.assertEqual(r, ['mismatch', 'wrong model'])

    def run_js_identity(self, a):
        script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))}); const s = Core.identityOf(null, {json.dumps(a)});'
                  ' process.stdout.write(JSON.stringify([s, Core.IDENTITY[s]]));')
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
        return json.loads(res.stdout)

    def test_quiz(self):
        run = self.runs / 'pilot'
        self.assertIsNone(server.quiz(self.runs, 'pilot'))
        self.assertFalse(server.runs(self.runs)[0]['quiz'])
        shutil.copy(Path(__file__).with_name('fixtures') / 'quiz.json', run / 'quiz.json')
        try:
            q = server.quiz(self.runs, 'pilot')
            self.assertTrue(q['fixture'])
            self.assertEqual(q['labels']['reference'], 'Reference bot')
            self.assertEqual(q['labels']['rule'], 'Rule bot')                  # names from the run fill in the rest
            self.assertTrue(server.runs(self.runs)[0]['quiz'])
            self.assertIsNone(server.quiz(self.runs, '../..'))
            with tempfile.TemporaryDirectory() as d:
                server.export(self.runs, d)
                self.assertTrue((Path(d) / 'data' / 'quiz-pilot.json').exists())
        finally:
            (run / 'quiz.json').unlink()

    def test_cache_signature_covers_run_files(self):
        run = self.runs / 'pilot'
        before = server.signature(run)
        names = {Path(p).name for p, *_ in before}
        self.assertTrue({'plan.json', 'config.json', 'hands.jsonl', 'events.jsonl'} <= names)
        plan = run / 'plan.json'
        text = plan.read_text()
        try:
            plan.write_text(text + ' ')                                       # a verdict-only change: no .jsonl touched
            self.assertNotEqual(server.signature(run), before)
            (run / 'decisions.json').write_text('{}')
            self.assertIn('decisions.json', {Path(p).name for p, *_ in server.signature(run)})
        finally:
            plan.write_text(text)
            (run / 'decisions.json').unlink(missing_ok=True)

    def test_lifecycle_ineligible_is_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / 'r').mkdir()
            (root / 'r' / 'plan.json').write_text(json.dumps({'matchups': [['a', 'b']], 'deals': 10, 'design': {'max_rounds': 3}}))
            (root / 'r' / 'decisions.json').write_text(json.dumps({'a__vs__b': [{'verdict': 'ineligible', 'looks_used': 1}]}))
            ms = [{'a': 'a', 'b': 'b', 'started': True, 'finished': True, 'live': False, 'round': 1}]
            life = server.lifecycle(root, 'r', ms)
            self.assertEqual((life['status'], life['pairings']['a|b']['status']), ('finished', 'ineligible'))

    def test_ratings_cache_never_blocks(self):
        import threading as th
        gate, calls = th.Event(), []
        def slow(run):
            calls.append(run)
            gate.wait(5)
            return {'players': [], 'n': len(calls)}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            shutil.copytree(self.runs / 'pilot', root / 'pilot')
            cache = server.RatingsCache(root, compute=slow, wait=0.05)
            self.assertEqual(cache.get('pilot'), ('pending', None))      # never computed: answers at once
            gate.set()
            cache.jobs['pilot'].join(5)
            status, data = cache.get('pilot')
            self.assertEqual((status, data['n'], data['cache']['fresh']), ('ok', 1, True))
            self.assertTrue((root / server.CACHE_DIR).exists())
            self.assertEqual(server.signature(root / 'pilot'), server.signature(root / 'pilot'))
            self.assertFalse(any(server.CACHE_DIR in p for p, *_ in server.signature(root)))
            gate.clear()
            hands = next((root / 'pilot').rglob('hands.jsonl'))
            hands.write_text(hands.read_text() + '\n')
            status, data = cache.get('pilot')                           # records changed: the old answer at once, marked stale
            self.assertEqual((status, data['n'], data['cache']['fresh'], data['cache']['reason']), ('ok', 1, False, 'records'))
            gate.set()
            cache.jobs['pilot'].join(5)
            self.assertEqual(cache.get('pilot')[1]['n'], 2)
            hands.write_text(hands.read_text() + '\n')
            broken = server.RatingsCache(root, compute=lambda run: 1 / 0, wait=2)
            broken.mem['pilot'] = cache.mem['pilot']
            broken.get('pilot')
            broken.jobs['pilot'].join(5)
            status, data = broken.get('pilot')                          # a failed recompute keeps the last answer, says so
            self.assertEqual((status, data['n'], data['cache']['fresh']), ('ok', 2, False))
            self.assertIn('ZeroDivisionError', data['cache']['error'])
            self.assertFalse(broken.jobs['pilot'].is_alive())           # and is not retried until something changes
            cache.get('pilot')
            cache.jobs['pilot'].join(5)
            again = server.RatingsCache(root, compute=lambda run: 1 / 0)
            status, data = again.get('pilot')                           # a restarted server answers from disk
            self.assertEqual((status, data['n'], data['cache']['fresh']), ('ok', 3, True))
            self.assertEqual(again.get('../x')[0], 'missing')
            empty = server.RatingsCache(root / 'pilot', compute=lambda run: time.sleep(0.2) or 1 / 0, wait=0)
            self.assertEqual(empty.get('pilot')[0], 'pending')
            empty.jobs['pilot'].join(5)
            self.assertEqual(empty.get('pilot')[0], 'error')

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
        for _ in range(400):                         # the first answer may be {pending}: computed in the background
            rt = get('/api/ratings?run=pilot')
            if not rt.get('pending'):
                break
            time.sleep(0.05)
        self.assertTrue(rt['players'])
        self.assertTrue(rt['cache']['fresh'])
        self.assertEqual(len(get('/api/match?id=pilot/fake__vs__rule/b-first')['hands']), 6)
        page = urllib.request.urlopen(base + '/').read().decode()
        self.assertIn('app.js', page)
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(base + '/api/match?id=../x')



class TestHistory(unittest.TestCase):
    """The SQLite history: incremental sync, pooling by model and level, deleted and restarted folders,
    other game settings left out, and the viewer's all-runs entry."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.players = self.root / 'players.toml'
        self.players.write_text('[players.rule]\nbot = "rule"\nsims = 40\n[players.station]\nbot = "station"\n'
                                '[players.station2]\nbot = "station"\nlabel = "Calling station"\n[players.random]\nbot = "random"\n')
        self.runs = self.root / 'runs'

    def tearDown(self):
        self.tmp.cleanup()

    def play(self, run, matchups, deals=10, **game):
        plan = make_plan(sorted({p for m in matchups for p in m}), deals=deals, out=str(self.runs / run),
                         players_file=str(self.players), matchups=matchups, game_config=game, design={'max_rounds': 1})
        asyncio.run(run_plan(plan, log=lambda *a: None))

    def test_pooling_and_survival(self):
        from poker_arena import history
        self.play('one', [['rule', 'station'], ['rule', 'random']])
        self.play('two', [['station2', 'rule'], ['station', 'station2']])
        self.assertEqual(history.sync(self.runs)['matches'], 8)
        self.assertEqual(history.sync(self.runs)['matches'], 0)          # nothing changed: nothing read again
        db = self.runs / history.DB_NAME
        r = history.all_time(db, boot=10, players_file=str(self.players))
        full = {p['player'].split('@')[0]: p['player'] for p in r['players']}   # an id may carry an implementation suffix
        self.assertEqual(set(full), {'bot:rule:sims=40', 'bot:station', 'bot:random'})
        rule, station = full['bot:rule:sims=40'], full['bot:station']
        self.assertEqual(sorted(r['ids'][station]), ['station', 'station2'])  # one bot, two ids
        self.assertEqual(r['deals'], 30)                                  # the station-vs-station2 deals don't count
        self.assertEqual({x['run']: x['self_play'] for x in r['runs']}, {'one': 0, 'two': 10})
        pair = next(m for m in r['matchups'] if {m['a'], m['b']} == {rule, station})
        self.assertEqual(pair['plays'], 20)                               # pooled across both runs
        self.assertEqual(pair['deals'], 10)                               # the same 10 deals twice: averaged
        self.assertEqual(r['stats'][station]['hands'], 40)                # as station, then as station2; no self-play
        self.assertEqual(r['labels'][rule], 'Rule bot')

        import shutil
        shutil.rmtree(self.runs / 'two')
        history.sync(self.runs)
        r = history.all_time(db, boot=10, players_file=str(self.players))
        self.assertEqual(r['deals'], 30)                                  # the deleted run is still counted
        rs = server.runs(self.runs)
        self.assertEqual(rs[0]['run'], '*')                               # one run on disk, two in the history
        self.assertEqual(server.ratings_for(self.runs, '*', boot=10)['deals'], 30)

    def test_restarted_folder_and_other_settings(self):
        from poker_arena import history
        self.play('one', [['rule', 'station']])
        self.play('short', [['rule', 'random']], deals=2, stack=40)       # other settings: not pooled
        history.sync(self.runs)
        r = history.all_time(self.runs / history.DB_NAME, boot=10)
        self.assertEqual(r['left_out'], [{'run': 'short', 'reason': 'other game settings'}])
        self.assertNotIn('bot:random', {p['player'].split('@')[0] for p in r['players']})
        import shutil
        shutil.rmtree(self.runs / 'one')
        self.play('one', [['rule', 'station']], deals=4)                  # the same folder, started over
        history.sync(self.runs)
        r = history.all_time(self.runs / history.DB_NAME, boot=10)
        self.assertEqual(r['deals'], 4)

    def test_identity(self):
        from poker_arena.history import identity
        self.assertEqual(identity('x', {'backend': 'claude', 'model': 'm', 'effort': 'high'}), 'claude:m:high')
        self.assertEqual(identity('y', {'backend': 'claude', 'model': 'm'}), 'claude:m:default')
        self.assertEqual(identity('z', {'bot': 'rule', 'sims': 200, 'label': 'R'}), 'bot:rule:sims=200')
        self.assertEqual(identity('c', {'backend': 'command', 'command': ['x']}), 'command:c')

    def test_same_name_told_apart(self):
        from poker_arena.history import _disambiguate
        labels = {'claude:m:high': 'M · high', 'api:m:high': 'M · high', 'bot:rule:sims=40': 'Rule bot',
                  'bot:rule:sims=200': 'Rule bot'}
        _disambiguate(labels)
        self.assertEqual(labels['api:m:high'], 'M · high (api)')
        self.assertEqual(labels['bot:rule:sims=200'], 'Rule bot (sims=200)')


class TestNames(unittest.TestCase):
    def test_display_names(self):
        self.assertEqual(server.pretty_model('claude-opus-5-5'), 'Claude Opus 5.5')
        self.assertEqual(server.pretty_model('gpt-6-luna'), 'GPT-6 Luna')
        self.assertEqual(server.pretty_model('gpt-6.2-sol'), 'GPT-6.2 Sol')
        self.assertEqual(server.pretty_model('gemini-4-neon-high'), 'Gemini 4 Neon High')
        self.assertEqual(server.display_name('x', {'backend': 'codex', 'model': 'gpt-6-luna', 'effort': 'high'}), 'GPT-6 Luna · High')
        self.assertEqual(server.display_name('x', {'backend': 'agy', 'model': 'gemini-4-neon-high', 'effort': 'high'}), 'Gemini 4 Neon · High')
        self.assertEqual(server.display_name('x', {'label': 'Mine', 'model': 'm'}), 'Mine')
        self.assertEqual(server.display_name('codex-gpt', {'backend': 'codex', 'effort': 'high'}),
                         'GPT (model not recorded) · High')   # no model: the family, never the CLI
        self.assertEqual(server.display_name('mine', {'backend': 'command', 'command': ['x']}), 'mine')
        self.assertEqual(server.display_name('s', {'backend': 'claude', 'model': 'claude-sonnet-5-5'}),
                         'Claude Sonnet 5.5 · default effort')   # a run from before levels were set



VIEWER = Path(server.__file__).with_name('viewer')
CONTRACT = {
    'objective': 'bb/100 over duplicate deals', 'design': {'alpha': 0.05, 'round_deals': 50, 'max_rounds': 6, 'block': 10},
    'deals': 120, 'hands': 240, 'components': [['a', 'b'], ['c', 'd']],
    'leader': {'player': 'a', 'component': 0, 'margin_bb100': 31.0, 'ci95': [12.0, 50.0]},
    'players': [
        {'player': 'a', 'identity': 'claude:m:high', 'kind': 'model', 'component': 0, 'strength': 40.0, 'strength_ci95': [20.0, 60.0],
         'rank': 1, 'rank_tied_with': [], 'net_bb': 48.0, 'hands': 120, 'deals': 60, 'session_elo': 1560,
         'session_elo_ci95': [1480, 1640], 'sessions': 6, 'session_w': 4, 'session_t': 0, 'session_l': 2, 'quarantined': False, 'quarantine_reason': None},
        {'player': 'b', 'identity': 'codex:g:high', 'kind': 'model', 'component': 0, 'strength': -40.0, 'strength_ci95': [-60.0, -20.0],
         'rank': 2, 'rank_tied_with': [], 'net_bb': -48.0, 'hands': 120, 'deals': 60, 'session_elo': 1440,
         'session_elo_ci95': [1360, 1520], 'sessions': 6, 'session_w': 2, 'session_t': 0, 'session_l': 4, 'quarantined': False, 'quarantine_reason': None},
        {'player': 'c', 'identity': 'bot:rule', 'kind': 'bot', 'component': 1, 'strength': 5.0, 'strength_ci95': None,
         'rank': None, 'rank_tied_with': ['d'], 'net_bb': 1.0, 'hands': 20, 'deals': 10, 'session_elo': 1500,
         'session_elo_ci95': [1300, 1700], 'sessions': 1, 'session_w': 1, 'session_t': 0, 'session_l': 0, 'quarantined': False, 'quarantine_reason': None},
        {'player': 'd', 'identity': 'claude:sonnet:high', 'kind': 'model', 'component': 1, 'strength': -5.0, 'strength_ci95': None,
         'rank': None, 'rank_tied_with': ['c'], 'net_bb': -1.0, 'hands': 20, 'deals': 10, 'session_elo': 1500,
         'session_elo_ci95': [1300, 1700], 'sessions': 1, 'session_w': 0, 'session_t': 0, 'session_l': 1, 'quarantined': True,
         'quarantine_reason': 'bare alias'}],
    'pairings': [
        {'a': 'a', 'b': 'b', 'matchups': ['a__vs__b'], 'deals': 60, 'blocks': 6, 'edge': 80.0, 'ci95': [40.0, 120.0], 't': 4.2,
         'boundary': 2.9, 'verdict': 'a', 'looks_used': 2, 'looks_planned': 6, 'power': {'deals_for_80pct': {'20': 900, '50': 150, '100': 40}}},
        {'a': 'c', 'b': 'd', 'matchups': ['c__vs__d'], 'deals': 10, 'blocks': 1, 'edge': 10.0, 'ci95': None, 't': None,
         'boundary': 2.9, 'verdict': 'early', 'looks_used': 0, 'looks_planned': 6, 'power': None}],
    'roster': [{'player': 'a', 'status': 'done'}, {'player': 'e', 'status': 'awaiting'}],
    'notes': ['single replicate per pairing'],
}
LEGACY = {'players': [{'player': 'x', 'elo': 1520, 'elo90': [1400, 1640], 'bb_per_100': 12.5, 'bb_per_100_ci95': [-30, 55],
                       'net_bb': 10, 'hands': 80, 'sessions': 4, 'session_wins': 2, 'session_ties': 1, 'session_losses': 1},
                      {'player': 'y', 'elo': 1480, 'elo90': [1360, 1600], 'bb_per_100': -12.5, 'bb_per_100_ci95': [-55, 30],
                       'net_bb': -10, 'hands': 80, 'sessions': 4, 'session_wins': 1, 'session_ties': 1, 'session_losses': 2}],
          'matchups': [{'a': 'x', 'b': 'y', 'matchup': 'x__vs__y', 'deals': 40, 'a_bb_per_100': 12.5, 'se': 20.0, 'z': 0.6,
                        'status': 'open', 'more_deals': 400}],
          'deals': 40, 'z_stop': 2.7}
# a hand: blinds 1/2, button raises to 6, the big blind calls, a flop, a bet and a fold
HAND = [
    {'kind': 'hand_start', 'data': {'hand': 0, 'button': 0, 'stack': 200, 'hole': [['As', 'Kd'], ['7c', '2h']]}},
    {'kind': 'blind', 'data': {'seat': 0, 'amount': 1}}, {'kind': 'blind', 'data': {'seat': 1, 'amount': 2}},
    {'kind': 'thought', 'data': {'seat': 0, 'reasoning': 'strong', 'requested': {'action': 'raise', 'amount': 5.5},
                                 'executed': {'action': 'raise', 'amount': 6}, 'correction': {'kind': 'sizing', 'reason': '5.5 rounded to 6'}}},
    {'kind': 'action', 'data': {'seat': 0, 'action': 'raise', 'amount': 6, 'pot': 8}},
    {'kind': 'thought', 'data': {'seat': 1, 'reasoning': 'odds', 'correction': None}},
    {'kind': 'action', 'data': {'seat': 1, 'action': 'call', 'amount': 4, 'pot': 12}},
    {'kind': 'board', 'data': {'street': 'flop', 'cards': ['Ah', '9c', '3d'], 'board': ['Ah', '9c', '3d']}},
    {'kind': 'action', 'data': {'seat': 1, 'action': 'check', 'pot': 12}},
    {'kind': 'action', 'data': {'seat': 0, 'action': 'bet', 'amount': 8, 'pot': 20}},
    {'kind': 'thought', 'data': {'seat': 1, 'fallback': True}},
    {'kind': 'action', 'data': {'seat': 1, 'action': 'fold', 'pot': 20}},
    {'kind': 'result', 'data': {'won': [20, 0], 'net': [6, -6], 'pot': 20, 'showdown': False, 'board': ['Ah', '9c', '3d']}},
]


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class TestViewerLogic(unittest.TestCase):
    """The app's pure parts (viewer/core.js, viewer/table.js), run in node."""

    def run_js(self, body):
        script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))});'
                  f'const PokerTable = require({json.dumps(str(VIEWER / "table.js"))});'
                  f'const out = (() => {{ {body} }})(); process.stdout.write(JSON.stringify(out));')
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout)

    def test_contract_renders_as_given(self):
        r = self.run_js(f'const r = Core.normalize({json.dumps(CONTRACT)}); return {{contract: r.contract, leaders: r.leaders,'
                        ' components: r.components, ranks: r.players.map(p => p.rank), roster: r.roster, verdicts:'
                        ' r.pairings.map(p => Core.verdict(p).short), power: Core.powerLine(r.pairings[0])};')
        self.assertTrue(r['contract'])
        self.assertEqual([l['player'] for l in r['leaders']], ['a'])
        self.assertEqual(r['components'], [['a', 'b'], ['c', 'd']])
        self.assertEqual(r['ranks'], [1, 2, None, None])
        self.assertEqual(r['verdicts'], ['a wins', 'Too early'])
        self.assertIn('50 bb/100: ~150', r['power'])
        self.assertEqual(r['roster'][1], {'player': 'e', 'status': 'awaiting'})

    def test_no_leader_and_leaders_list(self):
        r = self.run_js(f'return [Core.normalize({{...{json.dumps(CONTRACT)}, leader: null}}).leaders,'
                        f' Core.normalize({{...{json.dumps(CONTRACT)}, leaders: [null, null]}}).leaders];')
        self.assertEqual(r, [[], []])

    def test_legacy_output_is_never_ranked(self):
        r = self.run_js(f'const r = Core.normalize({json.dumps(LEGACY)}); return {{contract: r.contract, leaders: r.leaders,'
                        ' players: r.players.map(p => [p.player, p.strength, p.rank, p.session_elo]), pairing: r.pairings[0]};')
        self.assertFalse(r['contract'])
        self.assertEqual(r['leaders'], [])
        self.assertEqual(r['players'], [['x', 12.5, None, 1520], ['y', -12.5, None, 1480]])
        self.assertEqual(r['pairing']['verdict'], 'open')
        self.assertAlmostEqual(r['pairing']['ci95'][0], 12.5 - 39.2)

    def test_verdict_words(self):
        r = self.run_js('const p = {a: "a", b: "b", edge: -3, looks_used: 3, looks_planned: 6};'
                        ' return ["b", "even", "exhausted", "open"].map(v => Core.verdict({...p, verdict: v}).short);')
        self.assertEqual(r, ['b wins', 'Even', 'No winner', 'Undecided'])

    def test_groups_leader_is_inferential(self):
        r = self.run_js(f'const r = Core.normalize({json.dumps(CONTRACT)});'
                        ' return r.groups.map(g => [g.leader, g.front_runner, g.members]);')
        self.assertEqual(r, [['a', 'a', ['a', 'b']], [None, 'c', ['c', 'd']]])   # d: identity unresolved, still in the fit
        r = self.run_js(f'const c = {json.dumps(CONTRACT)}; c.pairings[0].verdict = "open";'
                        ' return Core.normalize(c).groups[0].leader;')
        self.assertIsNone(r)                                    # a descriptive lead is never a leader

    def test_contract_groups_and_flags(self):
        r = self.run_js(f'const c = {json.dumps(CONTRACT)}; c.groups = [{{id: 0, players: ["a", "b"], leader: null,'
                        ' front_runner: {player: "a"}, front_runner_margin: 80}, {id: 1, players: ["c", "d"], leader: null, front_runner: "c"}];'
                        ' c.players[3] = {...c.players[3], in_fit: false, identity_resolved: false}; c.players[0].rank_kind = "decided";'
                        ' const r = Core.normalize(c); return [r.groups.map(g => [g.leader, g.front_runner, g.members]),'
                        ' r.players.map(p => [p.rank_kind, p.in_fit, p.identity_resolved])];')
        self.assertEqual(r[0], [[None, 'a', ['a', 'b']], [None, 'c', ['c']]])
        self.assertEqual(r[1], [['decided', True, True], ['descriptive', True, True], ['descriptive', True, True], ['descriptive', False, False]])

    def test_pooled_is_descriptive(self):
        r = self.run_js(f'const r = Core.normalize({json.dumps(CONTRACT)}, {{pooled: true}});'
                        ' return [r.leaders, r.groups.map(g => g.leader), r.pairings.map(p => Core.verdict(p).short),'
                        ' r.pairings.map(p => Core.verdict(p).decided)];')
        self.assertEqual(r, [[], [None, None], ['Ahead on points', 'Ahead on points'], [False, False]])

    def test_honest_words(self):
        r = self.run_js('const p = {a: "a", b: "b", edge: 0, looks_used: 3, looks_planned: 6};'
                        ' return [Core.verdict({...p, verdict: "a"}), Core.verdict({...p, verdict: "exhausted"}).long,'
                        ' Core.verdict({...p, verdict: "open", edge: 5}).long];')
        self.assertTrue(r[0]['decided'])
        self.assertIn('Decided after 3 of 6', r[0]['long'])
        self.assertIn('unresolved', r[1])
        self.assertIn('a is ahead on points', r[2])
        for text in (r[1], r[2], r[0]['long']):
            self.assertNotIn('at least', text)

    def test_rates_and_target(self):
        r = self.run_js('return [Core.rateOf({vpip: {n: 30, of: 40}}, "vpip"), Core.rateOf({hands: 10, vpip: 0.5}, "vpip"),'
                        ' Core.rateOf({showdowns: 4, won_sd: 1, wsd: 0.25}, "wsd"), Core.rateOf({counts: {pfr: {n: 2, of: 8}}, pfr: 0.25}, "pfr"),'
                        ' Core.targetEdge({target_edge_bb100: 50, deals_planned: 600}, [{deals: 50}, {deals: 100}]),'
                        ' Core.targetEdge({}, [{deals: 40, power: {deals_for_80pct: {"50": 700}}}]), Core.targetEdge({}, [])];')
        self.assertEqual(r[0], {'x': 0.75, 'n': 30, 'of': 40})
        self.assertEqual(r[1], {'x': 0.5, 'n': 5, 'of': 10})
        self.assertEqual(r[2], {'x': 0.25, 'n': 1, 'of': 4})
        self.assertEqual(r[3], {'x': 0.25, 'n': 2, 'of': 8})
        self.assertEqual(r[4], {'target': 50, 'need': 600, 'allowed': None, 'power': None, 'planned': True, 'have': [50, 100]})
        self.assertEqual(r[5], {'target': 50, 'need': 700, 'planned': False, 'have': [40, 40]})
        self.assertIsNone(r[6])

    def test_unregistered_is_descriptive(self):
        r = self.run_js(f'const c = {json.dumps(CONTRACT)}; c.players[0].rank_kind = "decided";'
                        ' const out = []; for (const design of [{registered: false, target_edge_bb100: 50, deals_needed: 650},'
                        ' {legacy: "fixed", target_edge_bb100: 50, deals_needed: 650}, {registered: true, target_edge_bb100: 50, deals_needed: 650}]) {'
                        '   const r = Core.normalize({...c, design}); out.push([r.unregistered, r.leaders.length, r.groups.map(g => g.leader),'
                        '     r.players.map(p => p.rank), r.pairings.map(p => Core.verdict(p).short), Core.verdict(r.pairings[0]).long,'
                        '     Core.targetEdge(design, r.pairings)]); } return out;')
        unreg, legacy, reg = r
        for x in (unreg, legacy):
            self.assertTrue(x[0])
            self.assertEqual((x[1], x[2], x[3]), (0, [None, None], [None, None, None, None]))
            self.assertEqual(x[4], ['Ahead on points', 'Ahead on points'])
            self.assertIn('no recorded test plan', x[5])
            self.assertTrue(x[6]['retrospective'])
            self.assertFalse(x[6]['planned'])
        self.assertFalse(reg[0])
        self.assertEqual(reg[4], ['a wins', 'Too early'])
        self.assertTrue(reg[6]['planned'])
        self.assertNotIn('retrospective', reg[6])

    def test_design_fields(self):
        r = self.run_js('return [Core.scenarios({power_scenarios: {pessimistic: {power: 0.3, deals_needed: 2000},'
                        ' optimistic: {power_at_max: 0.95}, moderate: 0.8}}), Core.scenarios({power_scenarios: [{name: "moderate", power: 0.5}]}),'
                        ' Core.scenarios({}), Core.normalize({pairings: [{a: "a", b: "b", verdict: "early", conversations: 3, deals: 30}],'
                        ' min_conversations: 8}).pairings.map(p => [Core.conversationsShort(p), Core.verdict(p).long])];')
        self.assertEqual([x['name'] for x in r[0]], ['optimistic', 'moderate', 'pessimistic'])
        self.assertEqual([x['power'] for x in r[0]], [0.95, 0.8, 0.3])
        self.assertEqual(r[0][2]['deals'], 2000)
        self.assertEqual(r[1], [{'name': 'moderate', 'power': 0.5, 'deals': None}])
        self.assertEqual(r[2], [])
        self.assertTrue(r[3][0][0])
        self.assertIn('8 independent conversations', r[3][0][1])

    def test_timeline(self):
        r = self.run_js(f'const a = PokerTable.steps({json.dumps(HAND)}).steps;'
                        ' const b = PokerTable.steps(' + json.dumps(HAND[:6] + [HAND[-1]]) + ').steps;'
                        ' const al = PokerTable.timeline([a, b], "aligned"), sq = PokerTable.timeline([a, b], "sequence");'
                        ' return [a.length, b.length, al.map(f => [f.seg, f.at]), sq.map(f => [f.focus, f.at]),'
                        ' PokerTable.frameOf(sq, 1, 0), PokerTable.frameOf(al, 0, 3), PokerTable.timeline([[], []]).length];')
        na, nb, al, sq, f1, f0, empty = r
        self.assertEqual(len(sq), na + nb)
        self.assertEqual(sq[na - 1], [0, [na - 1, 0]])          # play 1 to its end while play 2 waits
        self.assertEqual(sq[na], [1, [na - 1, 0]])               # then play 2 from its start
        self.assertEqual(f1, na)
        segs = [s for s, _ in al]
        self.assertEqual(segs, sorted(segs, key=['preflop', 'flop', 'turn', 'river', 'end'].index))
        self.assertEqual(al[-1][1], [na - 1, nb - 1])            # both end on their result
        ends = [k for k, (s, _) in enumerate(al) if s == 'end']
        self.assertTrue(all(at[1] < nb - 1 for s, at in al if s != 'end'))   # play 2's result waits for the end segment
        self.assertEqual(al[f0][1][0], 3)
        self.assertTrue(ends)
        self.assertEqual(empty, 1)

    def test_corrections(self):
        r = self.run_js('const f = c => c + " chips"; return ['
                        ' Core.correction({requested: {action: "raise", amount: 5.5}, executed: {action: "raise", amount: 6},'
                        '   correction: {kind: "sizing", reason: "rounded"}}, {action: "raise", amount: 6}, f),'
                        ' Core.correction({requested: {action: "fold"}, executed: {action: "check"},'
                        '   correction: {kind: "retry", reason: "asked again 1x: nothing to call"}}, {action: "check"}, f),'
                        ' Core.correction({fallback: true}, {action: "check"}, f),'
                        ' Core.correction({action: "raise", amount: 1}, {action: "raise", amount: 4, note: "below the minimum"}, f),'
                        ' Core.correction({correction: null}, {action: "call", amount: 2}, f)];')
        self.assertEqual(r[0], {'kind': 'sizing', 'label': 'resized', 'requested': 'raise to 5.5 chips',
                                'executed': 'raise to 6 chips', 'reason': 'rounded'})
        self.assertEqual((r[1]['kind'], r[1]['requested'], r[1]['executed']), ('retry', 'fold', 'check'))
        self.assertEqual(r[2]['kind'], 'fallback')
        self.assertEqual((r[3]['requested'], r[3]['executed'], r[3]['reason']), ('raise to 1 chips', 'raise to 4 chips', 'below the minimum'))
        self.assertIsNone(r[4])

    def test_table_steps(self):
        r = self.run_js(f'return PokerTable.steps({json.dumps(HAND)});')
        steps, log = r['steps'], r['log']
        self.assertEqual(len(steps), 8)   # blinds, raise, call, flop, check, bet, fold, result
        s0 = steps[0]
        self.assertEqual([s['bet'] for s in s0['seats']], [1, 2])
        self.assertEqual([s['stack'] for s in s0['seats']], [199, 198])
        raise_ = steps[1]
        self.assertEqual(raise_['actor'], 0)
        self.assertEqual(raise_['seats'][0]['bet'], 6)
        self.assertEqual(raise_['pot'], 8)
        flop = steps[3]
        self.assertEqual([s['bet'] for s in flop['seats']], [0, 0])   # bets swept into the pot on a new street
        self.assertEqual(flop['collected'], 12)
        self.assertEqual(flop['board'], ['Ah', '9c', '3d'])
        self.assertTrue(all(not s['won'] for st in steps[:-1] for s in st['seats']))   # the winner only at the end
        end = steps[-1]
        self.assertTrue(end['done'])
        self.assertEqual([s['won'] for s in end['seats']], [True, False])
        self.assertEqual([s['stack'] for s in end['seats']], [206, 194])
        self.assertEqual([st['toAct'] for st in steps], [0, 1, None, 1, 0, 1, None, None])   # whose turn, not who just acted
        self.assertEqual([st['actor'] for st in steps[:3]], [None, 0, 1])
        acts = [e for e in log if e['kind'] == 'action']
        self.assertEqual(acts[0]['thought']['correction']['kind'], 'sizing')
        self.assertTrue(acts[-1]['thought']['fallback'])

    def test_table_three_seats(self):
        hand = [{'kind': 'hand_start', 'data': {'button': 2, 'stack': [100, 120, 80], 'hole': [['As', 'Ad'], ['Kc', 'Kd'], ['2c', '7d']]}},
                {'kind': 'blind', 'data': {'seat': 0, 'amount': 1}}, {'kind': 'blind', 'data': {'seat': 1, 'amount': 2}},
                {'kind': 'action', 'data': {'seat': 2, 'action': 'fold'}}]
        r = self.run_js(f'const s = PokerTable.steps({json.dumps(hand)}); return [s.n, s.steps[1].seats.map(x => [x.stack, x.folded])];')
        self.assertEqual(r, [3, [[99, False], [118, False], [80, True]]])


    def test_assumptions_beside_every_verdict(self):
        r = self.run_js('const d = {calibration: {conversation_effect_share: 0.5}}; const p = {a: "a", b: "b", edge: 50, verdict: "a", looks_used: 2, looks_planned: 6};'
                        ' const out = Core.normalize({design: d, pairings: [{...p, conversation_share: 0.62}, {...p, conversation_share: 0.2},'
                        '   {...p}, {...p, conversation_share: 0.9, assumptions: {status: "supported", checks: {conversation_share: {value: 0.9, limit: 1}}}}]}).pairings;'
                        ' return out.map(x => [x.assumption.status, Core.verdict(x).long, Core.verdict(x).qualifier, Core.verdict(x).decided]);')
        self.assertEqual([x[0] for x in r], ['unsupported', 'supported', 'unknown', 'supported'])
        self.assertIn('not established', r[0][1])
        self.assertEqual(r[0][2], 'outside calibrated range')
        self.assertIn('distribution assumptions remain unverified', r[1][1])
        self.assertNotIn('data support', r[1][1])
        self.assertIn('not checked', r[2][1])
        for x in r:
            self.assertTrue(x[3])
            self.assertNotIn('5%', x[1])                            # never an error bound without its assumptions

    def test_unsupported_win_never_makes_a_decided_leader(self):
        r = self.run_js(f'const c = {json.dumps(CONTRACT)}; c.design = {{calibration: {{conversation_effect_share: 0.5}}}};'
                        ' c.pairings[0].conversation_share = 0.7; const g = Core.normalize(c).groups[0];'
                        ' c.pairings[0].conversation_share = 0.1; const h = Core.normalize(c).groups[0];'
                        ' c.groups = [{id: 0, players: ["a", "b"], leader: null, leader_outside_calibration: "a"}]; const k = Core.normalize(c).groups[0];'
                        ' return [g.leader, g.qualified_leader, g.leader_assumption, h.leader, h.leader_assumption, k.leader, k.qualified_leader];')
        self.assertEqual(r, [None, 'a', 'unsupported', 'a', 'supported', None, 'a'])

    def test_ineligible_and_provenance(self):
        r = self.run_js('const p = {a: "a", b: "b", edge: 30};'
                        ' return [Core.verdict({...p, verdict: "ineligible", ineligible_reasons: ["identity unattested"]}),'
                        ' Core.verdict({...p, verdict: "a", eligible: false}).short,'
                        ' Core.sourceOf({...p, verdict: "a", official: true, judge: "abcdef123456"}),'
                        ' Core.sourceOf({...p, verdict: "a", retrospective: true}).kind,'
                        ' Core.sourceOf({...p, verdict: "open", recorded_verdict: "a"}), Core.sourceOf({...p, verdict: "a", recorded_verdict: "a"})];')
        self.assertEqual((r[0]['short'], r[0]['decided'], r[0]['tone']), ('Ineligible', False, 'bad'))
        self.assertIn('identity unattested', r[0]['long'])
        self.assertEqual(r[1], 'Ineligible')
        self.assertEqual(r[2]['kind'], 'official')
        self.assertIn('abcdef12', r[2]['text'])
        self.assertEqual(r[3], 'retrospective')
        self.assertEqual(r[4]['kind'], 'retrospective')
        self.assertIn('a wins', r[4]['text'])
        self.assertIsNone(r[5])

    def test_interval_basis_and_fit(self):
        r = self.run_js('const by = {a: {strength: 300, component: 0}, b: {strength: -200, component: 0}};'
                        ' return [Core.basisOf({ci_basis: "blocks"}).short, Core.basisOf({basis: "conversations"}).key, Core.basisOf({}),'
                        ' Core.fitOf({a: "a", b: "b", edge: 270, ci95: [210, 330]}, by), Core.fitOf({a: "a", b: "b", edge: 480, ci95: [100, 860]}, by),'
                        ' Core.fitOf({a: "a", b: "b", edge: 480, fitted_edge: 500, residual: -20, additivity: {warning: true}}, by),'
                        ' Core.fitOf({a: "a", b: "b", edge: 480, fitted_edge: 480, residual: 0, additivity: {status: "untestable", z: null}}, by),'
                        ' Core.fitOf({a: "a", b: "b", edge: 480, fitted_edge: 100, residual: 380, additivity: {status: "warning", z: 3.1}}, by),'
                        ' Core.fitOf({a: "a", b: "b", edge: 480, fitted_edge: 470, residual: 10, additivity: {status: "ok", z: 0.2}}, by)];')
        self.assertEqual(r[0], 'within conversations')
        self.assertEqual(r[1], 'across')
        self.assertIsNone(r[2])
        self.assertEqual(r[3], {'fitted': 500, 'residual': -230, 'warn': True, 'estimated': True})
        self.assertFalse(r[4]['warn'])
        self.assertEqual(r[5], {'fitted': 500, 'residual': -20, 'warn': True, 'estimated': False})
        self.assertEqual((r[6]['warn'], r[6]['untestable']), (False, True))   # insufficient evidence is not "off the fit"
        self.assertEqual((r[7]['warn'], 'untestable' in r[7]), (True, False))
        self.assertFalse(r[8]['warn'])

    def test_identity_and_names(self):
        r = self.run_js('return [Core.identityOf({attestation: "unattested"}), Core.identityOf(null, {identity: {match: 3}}),'
                        ' Core.identityOf(null, {identity: {match: 3, unattested: 1}}), Core.identityOf(null, {identity: {match: 3, mismatch: 1}}),'
                        ' Core.identityOf(null, {}), Core.identityOf(null, {identity: "unattested"}), Core.nameParts("Claude Sonnet · Default effort (protocol not recorded, 5 hands)"), Core.nameParts("Rule bot")];')
        self.assertEqual(r[:6], ['unattested', 'match', 'partial', 'mismatch', None, 'unattested'])
        self.assertEqual(r[6], ['Claude Sonnet', 'Default effort', 'protocol not recorded, 5 hands'])
        self.assertEqual(r[7], ['Rule bot', '', ''])

    def test_identity_never_trusts_bare_attested(self):
        r = self.run_js('return [Core.identityOf({identity_status: "attested"}, {identity: {attested: 5}}), Core.identityOf({identity_match: "mismatch"}),'
                        ' Core.ASSUME.supported];')
        self.assertEqual(r[:2], [None, 'mismatch'])
        self.assertEqual(r[2], 'diagnostics passed; distribution assumptions unverified')

    def test_quiz_shape(self):
        q = json.loads((Path(__file__).with_name('fixtures') / 'quiz.json').read_text())
        r = self.run_js(f'const q = Core.quizOf({json.dumps(q)}); return [q.suite, q.spots.length, q.players.map(p => [p.player, p.correct, p.total, p.ev_lost_bb, p.reference]),'
                        ' Core.quizOf({players: [{player: "m", score: 0.5, answers: [{spot: "s1", action: "call", ev_loss_bb: 2}]}], spots: [{id: "s1", correct: "fold"}]}),'
                        ' Core.quizOf(null)];')
        self.assertEqual(r[0], 'Poker fundamentals')
        self.assertEqual(r[1], 6)
        self.assertEqual(r[2][0], ['reference', 6, 6, 0, True])            # best first; the reference policy is flagged
        self.assertEqual([p[0] for p in r[2]], ['reference', 'rule', 'station', 'random'])
        self.assertAlmostEqual(r[2][1][3], 0.8)
        alt = r[3]
        self.assertEqual((alt['spots'][0]['best'], alt['players'][0]['correct'], alt['players'][0]['ev_lost_bb'], alt['players'][0]['score']), ('fold', 0, 2, 0.5))
        self.assertIsNone(r[4])

    def test_quiz_fallbacks_categories_provenance(self):
        raw = {'spots': [{'id': 's1', 'category': 'pot_odds', 'correct': 'fold'}, {'id': 's2', 'category': 'sizing', 'correct': 'bet'},
                         {'id': 's3', 'category': 'sizing', 'correct': 'fold'}],
               'players': {'m': {'correct': 3, 'total': 3, 'score': 1.0, 'answers': [
                   {'spot': 's1', 'action': 'fold', 'correct': True},
                   {'spot': 's2', 'action': 'bet', 'correct': True},
                   {'spot': 's3', 'action': 'fold', 'correct': True, 'fallback': True}],
                   'provenance': {'eligibility': {'status': 'ineligible', 'reasons': ['identity mismatch']}, 'cli_version': '2.1'}},
                   'b': {'kind': 'bot', 'answers': [{'spot': 's1', 'action': 'fold', 'correct': True}]}}}
        r = self.run_js(f'const q = Core.quizOf({json.dumps(raw)}); const m = q.players.find(p => p.player === "m");'
                        ' return [m.correct, m.score, m.valid, m.fallbacks, m.valid_accuracy, m.by_category, m.answers.s3.correct, m.provenance,'
                        ' q.categories, q.players.find(p => p.player === "b").provenance.status, Core.quizOf({players: {x: {answers: []}}}).players[0].provenance.status];')
        self.assertEqual(r[:5], [2, 2 / 3, 2, 1, 1.0])                   # a fallback that happened to be best gets no credit
        self.assertEqual(r[5], {'pot_odds': {'correct': 1, 'total': 1, 'valid': 1, 'fallbacks': 0},
                                'sizing': {'correct': 1, 'total': 2, 'valid': 1, 'fallbacks': 1}})
        self.assertFalse(r[6])
        self.assertEqual((r[7]['status'], r[7]['reasons']), ('ineligible', ['identity mismatch']))
        self.assertIn('CLI 2.1', r[7]['facts'])
        self.assertEqual(r[8], ['pot_odds', 'sizing'])
        self.assertEqual((r[9], r[10]), ('bot', 'unrecorded'))

    def test_decided_order_first(self):
        ps = [{'player': 'rule', 'rank': 2, 'rank_kind': 'decided', 'strength': 221},
              {'player': 'ref', 'rank': 1, 'rank_kind': 'decided', 'strength': 217},
              {'player': 'random', 'rank': None, 'rank_kind': 'descriptive', 'rank_range': [3, 4], 'strength': -253},
              {'player': 'station', 'rank': None, 'rank_kind': 'descriptive', 'rank_range': [3, 4], 'strength': -185}]
        r = self.run_js(f'const b = Core.standingBlocks({json.dumps(ps)}); return [b.map(x => [x.kind, x.rank ?? x.range, x.players.map(p => p.player)]),'
                        f' Core.standingBlocks({json.dumps(ps)}.map(p => ({{...p, rank_kind: "descriptive"}}))).map(x => [x.kind, x.players.map(p => p.player)])];')
        self.assertEqual(r[0], [['decided', 1, ['ref']], ['decided', 2, ['rule']], ['unresolved', [3, 4], ['station', 'random']]])
        self.assertEqual(r[1], [['points', ['rule', 'ref', 'station', 'random']]])

    def test_cohorts_lead_with_current_models(self):
        rt = {'players': [{'player': p, 'strength': 0, 'in_fit': p != 'old~h5'} for p in ('bot:rule', 'bot:ref', 'a~h50', 'b~h50', 'old~h5')],
              'pairings': [{'a': 'bot:rule', 'b': 'bot:ref', 'deals': 900, 'edge': 10}, {'a': 'a~h50', 'b': 'b~h50', 'deals': 50, 'edge': 5}],
              'groups': [{'players': ['bot:rule', 'bot:ref']}, {'players': ['a~h50', 'b~h50']}]}
        labels = {'bot:rule': 'Rule bot', 'bot:ref': 'Reference bot', 'a~h50': 'Claude Opus 5.5 · High', 'b~h50': 'GPT-6 Luna · High', 'old~h5': 'Claude Sonnet · default effort (protocol not recorded)'}
        r = self.run_js(f'const rt = Core.normalize({json.dumps(rt)}, {{pooled: true}}); const L = {json.dumps(labels)};'
                        ' const cs = Core.cohorts(rt, {label: p => L[p] || p, kind: p => p.startsWith("bot:") ? "bot" : "model"});'
                        ' return cs.map(c => [c.kind, c.members, Core.cohortName(c, p => L[p] || p)]);')
        self.assertEqual([c[0] for c in r], ['models', 'bots', 'history'])
        self.assertEqual(r[0][2]['title'], 'Claude Opus 5.5 · GPT-6 Luna')
        self.assertEqual(r[0][2]['sub'], 'High reasoning · 50-hand conversations')
        self.assertEqual(r[1][2]['title'], 'Baseline bots')
        self.assertEqual(r[2][1], ['old~h5'])

    def test_assumption_free_official_verdict(self):
        p = {'a': 'x', 'b': 'y', 'edge': 80, 'verdict': 'a', 'official_test': 'betting confidence sequence', 'model_based': {'verdict': 'open'},
             'assumptions': {'status': 'unsupported'}}
        r = self.run_js(f'const rt = Core.normalize({{players: [{{player: "x", strength: 40}}, {{player: "y", strength: -40}}], pairings: [{json.dumps(p)}],'
                        ' design: {registered: true}}); const q = rt.pairings[0]; const v = Core.verdict(q);'
                        ' return [q.assumption.status, v.decided, v.secondary && v.secondary.short, v.long, rt.groups[0].leader, rt.groups[0].leader_assumption,'
                        ' Core.testOf({}, {test: "t-test"}).free];')
        self.assertEqual(r[0], 'free')                 # the official test needs no distribution assumption
        self.assertTrue(r[1])
        self.assertEqual(r[2], 'Undecided')            # the model-based gate is shown, secondary
        self.assertIn('independent', r[3])
        self.assertIn('secondary, never decides', r[3])
        self.assertEqual((r[4], r[5]), ('x', 'free'))  # an assumption-free win makes a decided leader
        self.assertFalse(r[6])

    def test_population(self):
        r = self.run_js('return [Core.populationOf({rejected: [{}, {}], population: {committed: 400, provisional: 6}, stats_population: "verified committed"}),'
                        ' Core.populationOf({})];')
        self.assertEqual(r[0], {'committed': 400, 'provisional': 6, 'rejected': 2, 'verified': True, 'stats': {'kind': 'verified committed'}})
        self.assertEqual(r[1]['committed'], None)

    def test_divergence_and_swings(self):
        hand = HAND
        r = self.run_js(f'const a = PokerTable.steps({json.dumps(hand)}).log; const b = JSON.parse(JSON.stringify(a));'
                        ' const acts = b.filter(e => e.kind === "action"); const last = acts[acts.length - 1]; last.action = last.action === "fold" ? "call" : "fold";'
                        ' return [Core.firstDifference(a, a), Core.firstDifference(a, b), last.street, last.step,'
                        ' Core.bigSwings({0: 5, 1: -40, 2: 12, 3: 0, 4: 30}, 2), Core.bigSwings({0: 5, 1: -40}, 5, 10)];')
        self.assertIsNone(r[0])
        self.assertEqual((r[1]['street'], r[1]['steps'][1]), (r[2], r[3]))
        self.assertEqual(r[4], [{'deal': 1, 'net': -40}, {'deal': 4, 'net': 30}])
        self.assertEqual(r[5], [{'deal': 1, 'net': -40}])

    def test_table_context(self):
        r = self.run_js(f'const s = PokerTable.steps({json.dumps(HAND)}); return [s.steps.map(x => [x.toCall, x.potOdds && Math.round(x.potOdds * 100)]),'
                        ' s.log.filter(e => e.kind === "action").map(e => e.facing && [e.facing.toCall, e.facing.pot]), s.steps[3].seats.map(x => x.made)];')
        calls, facing, made = r
        self.assertEqual(calls[0], [1, 25])          # the button owes 1 into a pot of 3: a call needs 1 / (3 + 1)
        self.assertEqual(calls[1], [4, 33])          # the big blind owes 4 to call the raise to 6, into 8
        self.assertEqual(calls[3], [0, None])        # the flop opens: a check is free
        self.assertEqual(facing, [[1, 3], [4, 8], None, None, [8, 20]])
        self.assertEqual(made, ['a pair of aces', 'ace high'])   # the board's ace

    def test_table_uses_recorded_context(self):
        hand = [dict(e, data=dict(e['data'])) for e in HAND]
        hand[0]['data']['to_act'] = 0
        hand[0]['data']['omniscient'] = {'made_hand': {'0': 'ace high (recorded)', '1': 'seven high'}}
        hand[4]['data'].update(to_act=1, to_call=1, pot_before=3, pot_odds=0.25)
        hand[6]['data'].update(to_act=1)                  # after the call: the flop's first actor, before the flop is dealt
        hand[7]['data']['omniscient'] = {'made_hand': {'0': 'a pair of aces (recorded)', '1': 'seven high'}}
        hand[11]['data'].update(to_act=None)
        r = self.run_js(f'const s = PokerTable.steps({json.dumps(hand)}); return [s.steps.map(x => x.toAct), s.log[1].facing,'
                        ' s.steps[0].seats[0].made, s.steps[3].seats[0].made];')
        self.assertEqual(r[0], [0, 1, 1, 1, 0, 1, None, None])
        self.assertEqual(r[1], {'toCall': 1, 'pot': 3, 'odds': 0.25})
        self.assertEqual(r[2:], ['ace high (recorded)', 'a pair of aces (recorded)'])

    def test_made_hands(self):
        r = self.run_js('const m = PokerTable.madeHand; return [m(["Ah", "Kh"], ["Qh", "Jh", "Th"]), m(["6s", "6d"], ["6c", "2h", "2d"]),'
                        ' m(["As", "2d"], ["3c", "4h", "5d"]), m(["9s", "9d"], []), m(["Ks", "Kd"], ["Kc", "Qh", "Qd", "Qs"]),'
                        ' m(["Ah", "2h"], ["9h", "7h", "Kh", "Qc"]), m(["7c", "7d"], ["2c", "2s", "9h"]),'
                        ' PokerTable.drawOf(["Ah", "2h"], ["Kh", "7h", "9c"]), PokerTable.drawOf(["8c", "9d"], ["Ts", "Jh", "2c"]), PokerTable.drawOf(["8c", "9d"], ["Ts", "Jh", "2c", "3d", "4s"])];')
        self.assertEqual(r, ['a royal flush', 'a full house, sixes full of twos', 'a straight, five high', 'a pair of nines',
                             'a full house, kings full of queens', 'a flush, ace high', 'two pair, sevens and twos',
                             'flush draw', 'straight draw', ''])


class TestEffort(unittest.TestCase):
    def test_default_and_override(self):
        from poker_arena.runner import load_players
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.toml'
            f.write_text('[players.a]\nbackend = "claude"\nmodel = "m"\n[players.b]\nbackend = "codex"\nmodel = "m"\n'
                         'effort = "low"\n[players.c]\nbackend = "command"\ncommand = ["x"]\n[players.d]\nbot = "rule"\n')
            ps = load_players(f)
            self.assertEqual(ps['a']['effort'], 'high')     # the default
            self.assertEqual(ps['b']['effort'], 'low')      # the player's own
            self.assertNotIn('effort', ps['c'])             # a CLI without reasoning levels
            self.assertNotIn('effort', ps['d'])
            self.assertEqual(load_players(f, effort='max')['b']['effort'], 'max')  # tournament-wide override


if __name__ == '__main__':
    unittest.main()
