"""Round 9 review, material issues: the strength is a centred fit, shown beside observed opponent-weighted
winnings (#1); a win names the better model only under a stable model, and a passed drift check is "no drift
detected" (#2); a quiz score from another protocol or implementation never attaches silently (#3); a street's
closing action shows nobody to act until the next street is dealt (#4)."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poker_arena import ratings as R
from poker_arena.games import get_game
from test_fundamentals import CLI, SHOVE, CALL, answer, player, quiz_doc, tournament
from test_ratings import deals_between, plan, run

VIEWER = Path(__file__).resolve().parent.parent / 'poker_arena' / 'viewer'


def node(body):
    script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))});'
              f'const PokerTable = require({json.dumps(str(VIEWER / "table.js"))});'
              f'const out = (() => {{ {body} }})(); process.stdout.write(JSON.stringify(out));')
    res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


class TestStrengthIsCentred(unittest.TestCase):
    """#1: with edges +103 and +80 for the top player, its centred strength is about +61 while its direct edges
    average +91.5: the output says which is which and gives both."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        ds = (deals_between('g', 'c', 103, 50, rng, sd=0) + deals_between('g', 'l', 80, 50, rng, sd=0, seed0=2000)
              + deals_between('c', 'l', 19, 50, rng, sd=0, seed0=3000))
        cls.r = run(ds, plan(['g', 'c', 'l'], 50, 1), boot=20, registered=False)

    def test_observed_beside_fitted(self):
        g = next(p for p in self.r['players'] if p['player'] == 'g')
        self.assertAlmostEqual(g['observed_edge'], 91.5, delta=0.2)
        self.assertEqual(g['observed_opponents'], 2)
        self.assertLess(g['strength'], 70)
        self.assertIn('centred fitted strength', g['strength_kind'])
        self.assertIn('not a player\'s winnings against an average opponent', g['strength_kind'])
        self.assertIn('observed', self.r['fit'])

    def test_cli_table_names_both(self):
        text = R.table(self.r)
        self.assertIn('centred fitted strength', text)
        self.assertNotIn('against the average opponent in its group', text)
        self.assertIn('observed', text)
        self.assertIn('+91.5', text)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_viewer_fills_observed_for_older_ratings(self):
        r = node('const r = Core.normalize({players: [{player: "g", strength: 61, in_fit: true}, {player: "c", strength: -28, in_fit: true},'
                 ' {player: "l", strength: -33, in_fit: true}], pairings: [{a: "g", b: "c", edge: 103}, {a: "g", b: "l", edge: 80},'
                 ' {a: "c", b: "l", edge: 19}], components: [["g", "c", "l"]]}); return r.players.map(p => [p.player, p.observed_edge, p.observed_opponents]);')
        self.assertEqual(r, [['g', 91.5, 2], ['c', -42, 2], ['l', -49.5, 2]])

    def test_viewer_wording(self):
        app = (VIEWER / 'app.js').read_text()
        self.assertNotIn('against the average opponent', app)
        self.assertIn('Fitted strength', app)
        self.assertIn('observed_edge', app)


class TestWinnerNeedsAStableModel(unittest.TestCase):
    """#2: the visible guarantee and every winner's wording carry the stable-model / conditional-mean condition,
    and a passed drift check is "no drift detected", never "stable"."""

    def test_drift_pass_is_no_drift_detected(self):
        d = R.drift([10.0, -5.0, 8.0, 2.0, 7.0, -1.0, 4.0, 3.0])
        self.assertEqual(d['status'], 'no drift detected')
        self.assertIn('not evidence', d['meaning'])

    def test_guarantee_and_winner_are_qualified(self):
        self.assertIn('stable model', R.GUARANTEE)
        self.assertIn('no drift detected', R.GUARANTEE)
        self.assertIn('stable model', R.VERDICT_TEXT['a'])
        rng = np.random.default_rng(5)
        p = plan(['x', 'y'], 1, 60, reps=25, design={'max_rounds': 60})
        r = run(deals_between('x', 'y', 900, 1, rng, sd=20, rounds=60, reps=25), p, boot=20)
        m = r['pairings'][0]
        self.assertEqual(m['verdict'], 'a', m.get('reason'))
        self.assertIn('stable model', m['interpretation'])
        text = R.summary(r)
        self.assertIn('a better model only if each name served one stable model', text)
        self.assertNotIn('is better (decided)', text)
        if r['leader']:
            self.assertIn('stable model', r['leader']['interpretation'])

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_viewer_winner_wording(self):
        r = node('const p = {a: "x", b: "y", edge: 300, verdict: "a", looks_used: 2, looks_planned: 6,'
                 ' drift: {status: "no drift detected"}, assumption: {method: "free", status: "free"}};'
                 ' return [Core.verdict(p).long, Core.verdict(p).decided, Core.driftQualified(p)];')
        self.assertIn('unchanged model', r[0])
        self.assertTrue(r[1])
        self.assertFalse(r[2])
        app = (VIEWER / 'app.js').read_text()
        self.assertIn('error bound if conversations are independent and each model stays the same', app)
        self.assertIn('no drift detected', app)

    def test_docs_say_no_drift_detected(self):
        root = VIEWER.parent.parent
        doc = (root / 'docs' / 'methodology.md').read_text()
        self.assertNotIn("status: 'stable'", doc)
        self.assertIn('no drift detected', doc)
        self.assertIn('stable model', (root / 'README.md').read_text())


class TestQuizAttachment(unittest.TestCase):
    """#3: the same model and reasoning level is not enough: a quiz run with another implementation, CLI or
    settings stays a separate model-level assessment, with what differs."""

    def fund(self, prov):
        doc = quiz_doc({'opus': player([answer(SHOVE, 'call', 'call'), answer(CALL, 'call', 'call')])})
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'quiz.json').write_text(json.dumps(doc))
            return R.fundamentals(d, {'m1': prov})

    def test_compatible_attaches(self):
        x = self.fund(tournament())['players']['m1']
        self.assertTrue(x['attached'])
        self.assertEqual(x['link']['status'], 'attached')

    def test_other_implementation_does_not_attach(self):
        for over, what in (({'code': {'poker_arena/agents.py': 'zz', 'poker_arena/backends.py': 'b1'}}, 'different code'),
                           ({'cli_version': '9.9 (Claude Code)'}, 'different CLI version'),
                           ({'cli_sha256': 'ef' * 32}, 'different CLI file'),
                           ({'settings': {'backend': 'claude', 'model': 'claude-x-1', 'effort': 'high', 'tools': 'on'}},
                            'different player settings'),
                           ({'settings': None}, 'player settings not recorded on both sides')):
            f = self.fund(tournament(**over))
            self.assertNotIn('m1', f['players'], what)
            x = f['players']['opus']
            self.assertFalse(x['attached'])
            self.assertEqual((x['link']['status'], x['link']['player']), ('model-level', 'm1'))
            self.assertIn(what, x['link']['differences'])
            self.assertIn('separate model-level assessment of m1', R.fundamentals_text(f))

    def test_a_bare_identity_is_not_enough(self):
        f = self.fund('claude:claude-x-1:high#abcd1234@ffff0000')
        self.assertNotIn('m1', f['players'])
        self.assertEqual(f['players']['opus']['link']['status'], 'model-level')

    def test_provenance_from_match_configs(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / 'm1__vs__bot' / 'a-first'
            m.mkdir(parents=True)
            cfg = {'players': ['m1', 'bot'], 'specs': {'m1': {'backend': 'claude', 'model': 'claude-x-1', 'effort': 'high'},
                                                       'bot': {'bot': 'rule'}},
                   'protocol': {'m1': {'fingerprint': 'f', 'settings': {'backend': 'claude', 'model': 'claude-x-1',
                                                                         'effort': 'high', 'horizon': {}, 'system_prompt_sha1': 'x'}}},
                   'implementation': {'files': {'poker_arena/agents.py': 'a1', 'poker_arena/ratings.py': 'r9'}},
                   'effective': [{'exe': 'claude'}, {}], 'cli_versions': {'claude': CLI['version']},
                   'cli_artifacts': {'claude': {'sha256': CLI['sha256']}}}
            (m / 'config.json').write_text(json.dumps(cfg))
            prov = R.player_provenance(d)['m1']
        self.assertEqual(prov['settings'], {'backend': 'claude', 'model': 'claude-x-1', 'effort': 'high'})
        self.assertEqual(prov['code'], {'poker_arena/agents.py': 'a1'})
        self.assertEqual((prov['cli_version'], prov['cli_sha256']), (CLI['version'], CLI['sha256']))
        self.assertTrue(prov['identity'].startswith('claude:claude-x-1:high'))


class TestClosingActionShowsNobodyToAct(unittest.TestCase):
    """#4: small blind calls, big blind checks: the check closes preflop, so its snapshot shows no one to act on
    the preflop board; the big blind is to act only once the flop is dealt."""

    def hand(self):
        ep = get_game('holdem')({'hands_per_match': 2}).episode(0, 1, ['A', 'B'])
        ep.apply(0, {'action': 'call'})
        ep.apply(1, {'action': 'check'})
        return ep

    def test_engine(self):
        ep = self.hand()
        actions = [e for e in ep.events if e.kind == 'action']
        board = next(e for e in ep.events if e.kind == 'board')
        self.assertEqual(actions[0].data['to_act'], 1)
        self.assertIsNone(actions[1].data['to_act'])
        self.assertEqual(board.data['to_act'], 1)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_replay_snapshots(self):
        evs = [e.to_json() for e in self.hand().events]
        legacy = json.loads(json.dumps(evs))
        for e in legacy:
            if e['kind'] == 'board':
                del e['data']['to_act']
        legacy[[i for i, e in enumerate(legacy) if e['kind'] == 'action'][1]]['data']['to_act'] = 1
        r = node(f'return [{json.dumps(evs)}, {json.dumps(legacy)}].map(h => PokerTable.steps(h).steps.map(s => [s.street, s.board.length, s.toAct]));')
        for steps in r:
            self.assertEqual(steps[:4], [['preflop', 0, 0], ['preflop', 0, 1], ['preflop', 0, None], ['flop', 3, 1]])

    def test_older_records_still_verify(self):
        from poker_arena.games.holdem import verify_hand
        G = get_game('holdem')({'hands_per_match': 2})
        ep = G.episode(0, 1, ['A', 'B'])
        ep.apply(0, {'action': 'call'})
        ep.apply(1, {'action': 'check'})
        while not ep.done:
            ep.apply(ep.to_act, {'action': 'check'})
        rec = json.loads(json.dumps(ep.summary()))
        evs = [e.to_json() for e in ep.events]
        self.assertTrue(verify_hand(G, 1, ep.button, rec, evs)['ok'])
        for k, a in enumerate(rec['actions']):
            nxt = rec['actions'][k + 1] if k + 1 < len(rec['actions']) else None
            if a['to_act'] is None and nxt and nxt['street'] != a['street']:
                a['to_act'] = nxt['seat']
        for i, e in enumerate(evs):
            if e['kind'] == 'board':
                e['data'].pop('to_act')
                prev = next(x for x in reversed(evs[:i]) if x['kind'] == 'action')
                prev['data']['to_act'] = next(x for x in evs[i:] if x['kind'] == 'action')['data']['seat']
        self.assertTrue(verify_hand(G, 1, ep.button, rec, evs)['ok'], verify_hand(G, 1, ep.button, rec, evs)['errors'])


if __name__ == '__main__':
    unittest.main()
