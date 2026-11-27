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
        self.assertEqual({p['player'] for p in r['players']}, {'bot:rule:sims=40', 'bot:station', 'bot:random'})
        self.assertEqual(sorted(r['ids']['bot:station']), ['station', 'station2'])  # one bot, two ids
        self.assertEqual(r['deals'], 30)                                  # the station-vs-station2 deals don't count
        self.assertEqual({x['run']: x['self_play'] for x in r['runs']}, {'one': 0, 'two': 10})
        pair = next(m for m in r['matchups'] if {m['a'], m['b']} == {'bot:rule:sims=40', 'bot:station'})
        self.assertEqual(pair['plays'], 20)                               # pooled across both runs
        self.assertEqual(pair['deals'], 10)                               # the same 10 deals twice: averaged
        self.assertEqual(r['stats']['bot:station']['hands'], 40)          # as station, then as station2; no self-play
        self.assertEqual(r['labels']['bot:rule:sims=40'], 'Rule bot')

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
        self.assertNotIn('bot:random', {p['player'] for p in r['players']})
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
        acts = [e for e in log if e['kind'] == 'action']
        self.assertEqual(acts[0]['thought']['correction']['kind'], 'sizing')
        self.assertTrue(acts[-1]['thought']['fallback'])

    def test_table_three_seats(self):
        hand = [{'kind': 'hand_start', 'data': {'button': 2, 'stack': [100, 120, 80], 'hole': [['As', 'Ad'], ['Kc', 'Kd'], ['2c', '7d']]}},
                {'kind': 'blind', 'data': {'seat': 0, 'amount': 1}}, {'kind': 'blind', 'data': {'seat': 1, 'amount': 2}},
                {'kind': 'action', 'data': {'seat': 2, 'action': 'fold'}}]
        r = self.run_js(f'const s = PokerTable.steps({json.dumps(hand)}); return [s.n, s.steps[1].seats.map(x => [x.stack, x.folded])];')
        self.assertEqual(r, [3, [[99, False], [118, False], [80, True]]])


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
