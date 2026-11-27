"""The poker fundamentals quiz: its answers are computed (and cross-checked by an independent method), every spot
is clear-cut, and players answer through the same agents and backends, scored with their loss in big blinds."""
import asyncio
import json
import random
import re
import tempfile
import unittest
from pathlib import Path

from poker_arena import backends, quiz
from poker_arena.backends import Backend, Reply
from poker_arena.cards import equity as random_equity
from poker_arena.games import ranges


class Oracle(Backend):
    """Answers every question with the computed answer; `WRONG` names questions it answers wrongly first."""
    WRONG = set()

    async def _call(self, text):
        sid = re.findall(r'Question (\S+)\.', text)[-1]
        spot = next(quiz.solve(x) for x in quiz.SPOTS if x[0] == sid)
        if sid in Oracle.WRONG and 'could not be played' not in text:
            action = 'raise'
        else:
            action = spot['correct']
        return Reply(json.dumps({'reasoning': 'oracle', 'action': action, 'amount': 50 if action == 'raise' else None,
                                 'say': '', 'notes': None}))


class Folder(Backend):
    async def _call(self, text):
        return Reply('{"reasoning": "", "action": "fold", "amount": null, "say": "", "notes": null}')


class TestSpots(unittest.TestCase):
    def test_ranges(self):
        count = lambda t: len(ranges.parse(t))
        self.assertEqual([count(t) for t in ('AA', 'AKs', 'AKo', 'AK', 'TT+', 'A5s+', 'KTo+', '77-55', 'AhQh', 'random')],
                         [6, 4, 12, 16, 30, 36, 36, 18, 1, 1326])
        self.assertEqual(len(ranges.top(1.0)), 1326)
        self.assertLess(ranges.percentile()['AA'], ranges.percentile()['72o'])

    def test_the_quiz_is_clear_cut_and_balanced(self):
        spots = quiz.spots()
        self.assertGreaterEqual(len(spots), 30)
        self.assertEqual(len(spots), len(quiz.SPOTS))
        self.assertEqual(len({s['id'] for s in spots}), len(spots))
        answers = [s['correct'] for s in spots]
        self.assertEqual(set(answers), {'fold', 'call', 'all_in'})
        self.assertLess(answers.count('fold') / len(answers), 0.6)       # folding everything is no strategy
        for s in spots:
            self.assertGreaterEqual(s['gap_bb'], quiz.MIN_GAP_BB, s['id'])
            if s['kind'] == 'call':
                self.assertGreaterEqual(abs(s['equity'] - s['required_equity']), quiz.MIN_EQUITY_GAP, s['id'])
        self.assertEqual(quiz.spots_hash(), quiz.spots_hash())

    def test_values_against_independent_counts(self):
        by = {x[0]: x for x in quiz.SPOTS}
        s = quiz.solve(by['river-call-bluffcatcher'])   # AA 6, AK 8, 88 3, 44 3 beat KJ; 23 missed hands lose
        self.assertEqual(s['equity'], round(23 / 43, 4))
        pot, tc = 24, 12
        self.assertAlmostEqual(s['ev_bb']['call'], (23 / 43 * (pot + tc) - tc) / 2, places=2)
        self.assertEqual(quiz.solve(by['river-fold-pot-bet-vs-value'])['equity'], 0.0)
        for sid in ('pf-call-k4o-10', 'pf-shove-a2o-10'):    # range 'random': the plain equity against any hand
            x = by[sid]
            mc = random_equity(x[3].split(), [], random.Random(5), sims=40000)
            self.assertAlmostEqual(quiz.solve(x)['equity'], mc, delta=0.012)
        s = quiz.solve(by['pf-shove-a7o-10'])   # TT+ and AQ+ without the ace of hearts: 27 pairs, 12 AQ, 12 AK
        self.assertEqual(s['villain_combos'], 27 + 12 + 12)
        self.assertAlmostEqual(s['call_probability'], s['villain_combos'] / 1225, places=4)

    def test_every_spot_is_an_engine_position(self):
        for x in quiz.SPOTS:
            ep, hero, villain, _ = quiz.build(x)
            s = quiz.solve(x)
            self.assertEqual(ep.to_act, hero)
            text = quiz.question(s, ep, hero)
            self.assertIn(f'Question {s["id"]}.', text)
            self.assertIn(s['villain_range'], text)
            self.assertNotIn('Hand ', text.splitlines()[4])
            self.assertTrue(text.rstrip().endswith('.') and 'Your options: fold;' in text)


class TestAnswers(unittest.TestCase):
    def setUp(self):
        backends.BACKENDS.update(oracle=Oracle, folder=Folder)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'q'
        Oracle.WRONG = set()

    def tearDown(self):
        for k in ('oracle', 'folder'):
            backends.BACKENDS.pop(k, None)
        self.tmp.cleanup()

    def test_scores_and_the_record(self):
        Oracle.WRONG = {'river-call-bluffcatcher'}
        players = {'oracle': {'backend': 'oracle', 'model': 'o-1'}, 'folder': {'backend': 'folder'},
                   'station': {'bot': 'station'}}
        doc = asyncio.run(quiz.run_quiz(players, ['oracle', 'folder', 'station'], self.out, log=lambda *a: None))
        self.assertEqual(json.loads((self.out / 'quiz.json').read_text())['players'].keys(), doc['players'].keys())
        n = len(quiz.spots())
        o = doc['players']['oracle']
        self.assertEqual((o['answered'], o['correct'], o['ev_loss_bb'], o['fallbacks']), (n, n, 0.0, 0))
        fixed = next(a for a in o['answers'] if a['spot'] == 'river-call-bluffcatcher')
        self.assertEqual((fixed['correct'], len(fixed['calls']), fixed['errors'][0]['kind']), (True, 2, 'invalid'))
        self.assertIn('not one of this question', fixed['errors'][0]['error'])
        f = doc['players']['folder']
        folds = [s for s in doc['spots'] if s['correct'] == 'fold']
        self.assertEqual(f['correct'], len(folds))
        self.assertAlmostEqual(f['ev_loss_bb'], sum(max(s['ev_bb'].values()) for s in doc['spots']), places=2)
        st = doc['players']['station']
        self.assertTrue(all(a['executed'] in ('call', 'fold') for a in st['answers']))
        self.assertEqual(doc['version'], quiz.VERSION)
        self.assertEqual(doc['label'], 'poker fundamentals')
        a = o['answers'][0]
        self.assertEqual(set(a), {'spot', 'requested', 'executed', 'action', 'correct', 'ev_loss_bb', 'ev_lost_bb',
                                  'say', 'fallback', 'reasoning', 'errors', 'seconds', 'calls'})
        self.assertEqual(a['calls'][0]['requested_model'], 'o-1')
        self.assertIn('identity_match', a['calls'][0])
        s = doc['spots'][0]
        for k in ('id', 'hole', 'board', 'villain_range', 'equity', 'ev_bb', 'correct', 'options', 'gap_bb'):
            self.assertIn(k, s)
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'quiz.json').read_text())
        for k in fixture['spots'][0]:                    # the shape the viewer reads
            self.assertIn(k, s)
        self.assertEqual(doc['suite'], {'name': 'Poker fundamentals', 'version': quiz.VERSION})
        for k in ('action', 'correct', 'ev_lost_bb', 'reasoning'):
            self.assertIn(k, a)
        self.assertEqual((s['best'], s['ev'], s['bb']), (s['correct'], s['ev_bb'], 2))
        # asking one player again replaces only its results
        doc = asyncio.run(quiz.run_quiz(players, ['folder'], self.out, limit=2, log=lambda *a: None))
        self.assertEqual(doc['players']['folder']['answered'], 2)
        self.assertEqual(doc['players']['oracle']['answered'], n)

    def test_the_cli(self):
        from poker_arena.cli import main
        players = Path(self.tmp.name) / 'players.toml'
        players.write_text('[players.station]\nbot = "station"\n[players.reference]\nbot = "reference"\n')
        main(['quiz', 'station', 'reference', '--players', str(players), '--out', str(self.out), '--limit', '3'])
        doc = json.loads((self.out / 'quiz.json').read_text())
        self.assertEqual({p: r['answered'] for p, r in doc['players'].items()}, {'station': 3, 'reference': 3})
        self.assertEqual(doc['reference'], ['reference'])


if __name__ == '__main__':
    unittest.main()
