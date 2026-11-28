"""The v2 surface: scorecard, forensics, hand_eval and plan_preview endpoints, the static export, the four new
CLI commands, and the viewer's pure helpers (viewer/core.js) in node."""
import asyncio
import contextlib
import io
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from poker_arena import cli, server
from poker_arena.runner import make_plan, run_plan

VIEWER = Path(server.__file__).with_name('viewer')
PLAYERS = '[players.rule]\nbot = "rule"\nsims = 40\n[players.station]\nbot = "station"\n[players.random]\nbot = "random"\n'


class V2Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.players = cls.root / 'players.toml'
        cls.players.write_text(PLAYERS)
        cls.runs = cls.root / 'runs'
        plan = make_plan(['rule', 'station', 'random'], deals=4, out=str(cls.runs / 'pilot'), players_file=str(cls.players),
                         design={'max_rounds': 1})
        asyncio.run(run_plan(plan, log=lambda *a: None))
        cls.match = 'pilot/rule__vs__station/a-first'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestEndpoints(V2Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        threading.Thread(target=server.serve, args=(cls.runs, port, '127.0.0.1', str(cls.players)), daemon=True).start()
        cls.base = f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                urllib.request.urlopen(cls.base + '/api/runs')
                break
            except OSError:
                time.sleep(0.05)

    def get(self, path, wait=True):
        for _ in range(600):
            try:
                with urllib.request.urlopen(self.base + path) as r:
                    data = json.loads(r.read())
                    code = r.status
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
            if code != 202 or not wait:
                return code, data
            time.sleep(0.05)
        self.fail(f'{path} stayed pending')

    def test_scorecard_is_served_from_a_cache_with_labels(self):
        code, sc = self.get('/api/scorecard?run=pilot')
        self.assertEqual(code, 200)
        self.assertEqual([a['axis'] for a in sc['axes']], ['intelligence', 'wisdom', 'endurance', 'creativity', 'knowledge', 'alignment'])
        self.assertEqual(set(sc['players']), {'rule', 'station', 'random'})
        self.assertEqual(sc['labels']['rule'], 'Rule bot')
        self.assertEqual(sc['kinds']['rule'], 'bot')
        ax = sc['players']['rule']['axes']['intelligence']
        self.assertEqual(ax['max_pips'], 20)
        self.assertTrue(sc['cache']['fresh'])
        self.assertTrue((self.runs / server.CACHE_DIR / 'scorecard-pilot.json').exists())
        self.assertEqual(self.get('/api/scorecard?run=nope')[0], 404)
        self.assertEqual(self.get('/api/scorecard?run=*')[0], 404)

    def test_forensics(self):
        code, fx = self.get('/api/forensics?run=pilot')
        self.assertEqual(code, 200)
        self.assertEqual(fx['summary']['total_incidents'], 0)
        self.assertTrue(all(p['clean'] for p in fx['players'].values()))
        self.assertEqual(fx['labels']['station'], 'Calling station')
        self.assertEqual(self.get('/api/forensics?run=../x')[0], 404)

    def test_hand_eval(self):
        code, ev = self.get('/api/hand_eval?id=' + urllib.parse.quote(self.match) + '&hand=1')
        self.assertEqual(code, 200)
        self.assertEqual(ev['hand'], 1)
        self.assertEqual(len(ev['per_seat_ev_loss_mbb']), 2)
        self.assertTrue(ev['decisions'])
        d = ev['decisions'][0]
        for k in ('pot_odds', 'mdf', 'equity', 'best_action', 'gto_freqs', 'ev_loss_mbb', 'is_blunder', 'is_indifferent'):
            self.assertIn(k, d)
        self.assertIn(d['player_id'], ('rule', 'station'))
        self.assertEqual(ev['source'], 'solver')
        self.assertIn('luck_correction', ev['aivat'])
        self.assertEqual(self.get('/api/hand_eval?id=' + urllib.parse.quote(self.match) + '&hand=99')[0], 404)
        self.assertEqual(self.get('/api/hand_eval?id=../x&hand=0')[0], 404)
        self.assertEqual(self.get('/api/hand_eval?id=' + urllib.parse.quote(self.match) + '&hand=x')[0], 400)

    def test_hand_eval_falls_back_to_pot_odds_when_the_solver_breaks(self):
        from poker_arena import solver
        real = solver.evaluate_hand
        solver.evaluate_hand = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
        server.HAND_EVALS.clear()
        try:
            ev = server.hand_eval(self.runs, self.match, 1)
        finally:
            solver.evaluate_hand = real
            server.HAND_EVALS.clear()
        self.assertEqual(ev['source'], 'pot_odds')
        self.assertIn('solver unavailable', ev['note'])
        self.assertTrue(ev['decisions'])
        self.assertTrue(all(d['ev_loss_mbb'] is None and 'pot_odds' in d and 'mdf' in d for d in ev['decisions']))

    def test_plan_preview(self):
        code, p = self.get('/api/plan_preview?players=4&replicates=25&target=50&stack=100&alpha=0.05&aivat_vr=2.5&track=official')
        self.assertEqual(code, 200)
        self.assertEqual(p['pairings'], 6)
        self.assertEqual(p['raw_deals_per_pairing'], p['deals_per_pairing'])
        self.assertAlmostEqual(p['aivat_effective_deals_per_pairing'], p['raw_deals_per_pairing'] / 2.5, delta=0.1)
        self.assertAlmostEqual(p['threshold'], 2 / p['alpha_pairing'])
        self.assertAlmostEqual(p['alpha_pairing'], 0.05 / 6)
        self.assertEqual(len(p['power_scenarios']), 3)
        self.assertGreater(p['estimated_calls'], 0)
        self.assertGreater(p['estimated_runtime_hours'], 0)
        self.assertLess(p['estimated_calls_with_aivat'], p['estimated_calls'])
        self.assertTrue(p['command'].startswith('python -m poker_arena plan model1 model2 model3 model4 --tracks official --deals 1'))
        self.assertIn('--stack 200', p['command'])
        code, a = self.get('/api/plan_preview?track=adaptation&players=3')
        self.assertEqual((code, a['round_deals'], a['track']), (200, 50, 'adaptation'))
        self.assertIn('--adaptation-deals 50', a['command'])
        # out-of-range inputs are clamped, not trusted
        code, c = self.get('/api/plan_preview?players=999&target=0&alpha=9&aivat_vr=0')
        self.assertEqual((c['params']['players'], c['params']['target'], c['params']['aivat_vr']), (12, 15.0, 1.0))
        self.assertEqual(c['params']['alpha'], 0.2)

    def test_ratings_carry_the_v2_extras_without_losing_keys(self):
        code, rt = self.get('/api/ratings?run=pilot')
        self.assertEqual(code, 200)
        for k in ('players', 'pairings', 'labels', 'run_audit', 'meta_rank', 'forensics_summary', 'aivat'):
            self.assertIn(k, rt)
        self.assertEqual(rt['forensics_summary']['total_incidents'], 0)
        self.assertIn('nash', rt['meta_rank'])
        self.assertEqual(rt['aivat']['summary']['pairings'], 3)

    def test_the_viewer_page_has_the_new_tabs(self):
        page = urllib.request.urlopen(self.base + '/').read().decode()
        self.assertIn('data-tab="scorecards"', page)
        self.assertIn('data-tab="planner"', page)
        for tab in ('board', 'matches', 'quiz'):
            self.assertIn(f'data-tab="{tab}"', page)


class TestExport(V2Base):
    def test_export_writes_scorecard_and_forensics(self):
        out = self.root / 'site'
        server.export(self.runs, out, players_file=str(self.players))
        data = out / 'data'
        sc = json.loads((data / 'scorecard-pilot.json').read_text())
        fx = json.loads((data / 'forensics-pilot.json').read_text())
        self.assertEqual(set(sc['players']), {'rule', 'station', 'random'})
        self.assertEqual(sc['labels']['rule'], 'Rule bot')
        self.assertEqual(fx['summary']['total_incidents'], 0)
        self.assertIn('forensics_summary', json.loads((data / 'ratings-pilot.json').read_text()))
        self.assertTrue(all((out / f).exists() for f in ('index.html', 'app.js', 'core.js', 'table.js', 'style.css')))
        self.assertIn('data-tab="scorecards"', (out / 'index.html').read_text())


class TestCli(V2Base):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(list(argv))
        return buf.getvalue()

    def test_scorecard(self):
        out_json = self.root / 'sc.json'
        text = self.run_cli('scorecard', str(self.runs / 'pilot'), '--json', str(out_json), '--players', str(self.players), '--solver', 'heuristic')
        for word in ('INTELLIGENCE', 'WISDOM', 'ENDURANCE', 'CREATIVITY', 'KNOWLEDGE', 'ALIGNMENT', 'Rule bot', '/20', 'split-half'):
            self.assertIn(word, text)
        self.assertRegex(text, r'\[[█▒░]{20}\] \d+\.\d/20 \(\d+\.\d–\d+\.\d\)')
        self.assertEqual(json.loads(out_json.read_text())['method']['solver'], 'heuristic')

    def test_pip_bar_has_twenty_segments(self):
        for ax in ({'pips': 14, 'pips_low': 12, 'pips_high': 16}, {'pips': 0}, {'pips': 20, 'pips_low': 20, 'pips_high': 20}, {'pips': None}):
            self.assertEqual(len(cli.pip_bar(ax)), 22)
        self.assertEqual(cli.pip_bar({'pips': 14, 'pips_low': 12, 'pips_high': 16}), '[' + '█' * 12 + '▒' * 4 + '░' * 4 + ']')

    def test_forensics(self):
        out_json = self.root / 'fx.json'
        text = self.run_cli('forensics', str(self.runs / 'pilot'), '--json', str(out_json), '--players', str(self.players))
        self.assertIn('0 incident(s)', text)
        self.assertIn('EV decomposition', text)
        self.assertIn('covert intent', text)
        self.assertEqual(json.loads(out_json.read_text())['summary']['total_incidents'], 0)

    def test_invariance(self):
        out_json = self.root / 'inv.json'
        text = self.run_cli('invariance', 'station', '--spots', '3', '--perms', '2', '--players', str(self.players), '--json', str(out_json))
        for word in ('action flip rate', 'mean EV delta', 'invariance score', 'verdict'):
            self.assertIn(word, text)
        r = json.loads(out_json.read_text())
        self.assertEqual(r['n_spots'], 3)
        self.assertEqual(r['action_flip_rate'], 0.0)          # a calling station ignores suits
        self.assertFalse(r['biased'])
        with self.assertRaises(SystemExit):
            self.run_cli('invariance', 'ghost', '--players', str(self.players))

    def test_solver_check(self):
        text = self.run_cli('solver-check')
        for word in ('built-in DCFR', 'TexasSolver', 'Slumbot'):
            self.assertIn(word, text)
        self.assertIn('ready', text)

    def test_existing_commands_are_still_there(self):
        for cmd in ('play', 'plan', 'tournament', 'ratings', 'view', 'history', 'export', 'status', 'doctor', 'quiz'):
            with self.assertRaises(SystemExit) as e, contextlib.redirect_stdout(io.StringIO()):
                cli.main([cmd, '--help'])
            self.assertEqual(e.exception.code, 0)


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class TestCoreV2(unittest.TestCase):
    def js(self, body):
        script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))});'
                  f'const out = (() => {{ {body} }})(); process.stdout.write(JSON.stringify(out));')
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout)

    def test_pips_are_twenty_segments_solid_then_band_then_empty(self):
        r = self.js('const f = a => Core.pipSegments(a); return [f({pips: 16, pips_low: 14, pips_high: 18}), f({pips: 7}), f({}), f({pips: 20, pips_low: 19, pips_high: 25}), f({pips: 3, pips_low: 5, pips_high: 2})];')
        a = r[0]
        self.assertEqual(len(a), 20)
        self.assertEqual((a.count('solid'), a.count('band'), a.count('empty')), (14, 4, 2))
        self.assertEqual(a[:14], ['solid'] * 14)
        self.assertEqual((r[1].count('solid'), r[1].count('band')), (7, 0))
        self.assertEqual(r[2], ['empty'] * 20)
        self.assertEqual((r[3].count('solid'), r[3].count('band'), len(r[3])), (19, 1, 20))
        self.assertEqual(r[4].count('solid'), 5)

    def test_readout_and_evidence_links(self):
        r = self.js('return [Core.axisReadout({score: 16.44, ci95: [14.8, 17.93], status: "measured"}), Core.axisReadout({score: 4, status: "low_sample"}),'
                    ' Core.axisReadout({score: null, status: "insufficient_data"}),'
                    ' Core.evidenceHref("pilot", {match_id: "rule__vs__station/a-first", hand: 3}), Core.evidenceHref("pilot", {match_id: "pilot/x__vs__y/b-first", hand: 0}),'
                    ' Core.axisList({}).map(a => a.label), Core.axisList({}).map(a => a.subtitle)[1]];')
        self.assertEqual(r[0], {'score': '16.4 / 20', 'ci': '95% CI 14.8–17.9', 'status': 'measured', 'low': False, 'none': False})
        self.assertTrue(r[1]['low'])
        self.assertTrue(r[2]['none'])
        self.assertEqual(r[3], '#/r/pilot/m/pilot%2Frule__vs__station%2Fa-first?deal=4')
        self.assertEqual(r[4], '#/r/pilot/m/pilot%2Fx__vs__y%2Fb-first?deal=1')
        self.assertEqual(r[5], ['INTELLIGENCE', 'WISDOM', 'ENDURANCE', 'CREATIVITY', 'KNOWLEDGE', 'ALIGNMENT'])
        self.assertEqual(r[6], 'knowing when to act, ask, verify, change course, or stop')

    def test_solver_badges_and_decision_lookup(self):
        r = self.js('const log = [{kind: "start", step: 0}, {kind: "action", step: 1, seat: 0}, {kind: "board", step: 2}, {kind: "action", step: 3, seat: 1}];'
                    ' const decs = [{action_idx: 0, ev_loss_mbb: 10}, {action_idx: 1, ev_loss_mbb: 400}];'
                    ' return [Core.evBadge({ev_loss_mbb: 250}), Core.evBadge({ev_loss_mbb: 249, is_indifferent: true}), Core.evBadge({ev_loss_mbb: 10}), Core.evBadge({is_blunder: true}),'
                    ' Core.decisionAt(log, decs, 2).index, Core.decisionAt(log, decs, 3).decision.ev_loss_mbb, Core.decisionAt(log, decs, 0).upcoming,'
                    ' Core.gtoBars({gto_freqs: {fold: 0.1, call: 0.6, raise: 0.3, all_in: 1e-9}, taken_action: "call"}),'
                    ' Core.aivatPill({luck_correction: [3, -3], all_in: true, all_in_street: "flop"}, 0, 2), Core.aivatPill({luck_correction: [0, 0]}, 0, 2).text];')
        self.assertEqual(r[0]['text'], 'BLUNDER')
        self.assertEqual(r[1]['text'], 'INDIFFERENT')
        self.assertIsNone(r[2])
        self.assertEqual(r[3]['kind'], 'blunder')
        self.assertEqual((r[4], r[5], r[6]), (0, 400, True))
        self.assertEqual([x['action'] for x in r[7]], ['call', 'raise', 'fold'])
        self.assertTrue(r[7][0]['taken'])
        self.assertEqual(r[8]['text'], 'AIVAT: ran lucky 1.5 bb')
        self.assertEqual(r[9], 'AIVAT: no luck to remove')

    def test_note_diff_marks_what_changed(self):
        r = self.js('const d = Core.noteDiff("opp is tight folds to 3bets", "opp is loose calls 3bets wide"); return [d, Core.noteChanged(d), Core.noteChanged(Core.noteDiff("same text", "same text")), Core.noteDiff("", "first note")];')
        text = {t: ' '.join(x['text'] for x in r[0] if x['t'] == t) for t in ('same', 'add', 'del')}
        self.assertIn('opp is', text['same'])
        self.assertIn('loose', text['add'])
        self.assertIn('tight', text['del'])
        self.assertTrue(r[1])
        self.assertFalse(r[2])
        self.assertEqual(r[3], [{'t': 'add', 'text': 'first note'}])

    def test_forensics_pill_and_meta_rows(self):
        r = self.js('const meta = {players: ["a", "b", "c"], nash: {nash_weights: {a: .5, b: .3, c: .2}, nash_scores: {a: 0, b: -10, c: -20}, ranking: ["a", "b", "c"]},'
                    ' alpha_rank: {stationary: {a: .2, b: .5, c: .3}, ranking: ["b", "c", "a"], alpha: .05}, cycles: [["a", "b", "c"]], non_transitive: true};'
                    ' const m = Core.metaRows(meta);'
                    ' return [Core.forensicsPill({total_incidents: 0, flagged_players: []}), Core.forensicsPill({total_incidents: 3, flagged_players: ["x"]}), Core.forensicsPill({total_incidents: 9, flagged_players: ["x", "y"]}),'
                    ' Core.forensicsPill(null), m.rows.map(x => [x.player, x.nash_rank, x.alpha_rank]), m.nonTransitive, Core.cycleText(m.cycles[0]), Core.metaRows({available: false}), Core.metaRows(null),'
                    ' Core.aivatOf({pairings: [{a: "x", b: "y", raw_bb100: 100, raw_se: 9, aivat_bb100: 80, aivat_se: 4, variance_reduction_ratio: 2, effective_sample_multiplier: 4, deals: 10}]}, "y", "x")];')
        self.assertEqual(r[0]['text'], 'Clean · 0 incidents')
        self.assertEqual((r[0]['tone'], r[1]['tone'], r[2]['tone']), ('ok', 'warn', 'bad'))
        self.assertEqual(r[1]['text'], '3 incidents · 1 flagged')
        self.assertIsNone(r[3])
        self.assertEqual(r[4], [['a', 1, 3], ['b', 2, 1], ['c', 3, 2]])
        self.assertTrue(r[5])
        self.assertEqual(r[6], 'a > b > c > a')
        self.assertIsNone(r[7])
        self.assertIsNone(r[8])
        self.assertEqual((r[9]['raw'], r[9]['aivat'], r[9]['multiplier']), (-100, -80, 4))

    def test_planner_fallback_matches_the_servers_defaults_and_clamps(self):
        r = self.js('const p = Core.planFallback({}), a = Core.planFallback({track: "adaptation"}), c = Core.planParams({players: 99, target: 1, alpha: 5, aivat_vr: 0.1, track: "x"});'
                    ' const v = Core.planFallback({aivat_vr: 2.5});'
                    ' return [p.conversations_per_pairing, p.threshold, p.pairings, a.conversations_per_pairing, a.round_deals, c, v.aivat_effective_deals_per_pairing / v.raw_deals_per_pairing, Core.hoursText(0.2), Core.hoursText(60), p.command];')
        self.assertEqual((r[0], r[1], r[2]), (1650, 240, 6))
        self.assertEqual((r[3], r[4]), (44, 50))
        self.assertEqual((r[5]['players'], r[5]['target'], r[5]['aivat_vr'], r[5]['alpha'], r[5]['track']), (12, 15, 1, 0.2, 'official'))
        self.assertAlmostEqual(r[6], 0.4, places=2)
        self.assertEqual(r[7], '12 min')
        self.assertEqual(r[8], '2.5 days')
        self.assertTrue(r[9].startswith('python -m poker_arena plan model1 model2 model3 model4 --tracks official --deals 1 --replicates 25 --stack 200'))

    def test_the_offline_planner_and_the_server_agree_on_the_command(self):
        for q in ({}, {'players': 6, 'target': 100, 'alpha': 0.1, 'sd': 400}, {'track': 'adaptation', 'players': 3}):
            p = server.plan_params({k: [str(v)] for k, v in q.items()})
            js = self.js(f'return Core.planCommand(Core.planParams({json.dumps(q)}));')
            self.assertEqual(js, server.plan_command(p))

    def test_reliability_words_and_correlation_matrix(self):
        r = self.js('return [Core.reliability({spearman_brown: 0.9, reliable: true}), Core.reliability({spearman_brown: 0.2, reliable: false}), Core.reliability({}),'
                    ' Core.corrMatrix({inter_axis_corr: {intelligence: {wisdom: 0.5}}}).map(x => x.length), Core.corrMatrix({inter_axis_corr: {intelligence: {wisdom: 0.5}}})[0][1]];')
        self.assertEqual(r[0]['tone'], 'ok')
        self.assertEqual(r[1]['tone'], 'warn')
        self.assertEqual(r[2]['tone'], 'quiet')
        self.assertEqual(r[3], [6] * 6)
        self.assertEqual(r[4], 0.5)

    def test_pages_and_styles_exist_for_the_new_views(self):
        app = (VIEWER / 'app.js').read_text()
        css = (VIEWER / 'style.css').read_text()
        for needle in ('renderScorecards', 'renderPlanner', 'metaCard', 'splitBarHtml', 'solverHtml', 'mindHtml', "'/scorecards'", "'/planner'", '/api/hand_eval', '/api/plan_preview'):
            self.assertIn(needle, app)
        for needle in ('.pip.solid', '.pip.band', '#2952e3', '#f6f5f0', 'rgba(41, 82, 227, 0.34)', '#deddd7', '.pair.split', '.mp-col.private'):
            self.assertIn(needle, css)
        self.assertIn('width: 257px', css)


if __name__ == '__main__':
    unittest.main()
