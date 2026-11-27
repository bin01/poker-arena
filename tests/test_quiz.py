"""The poker fundamentals quiz: its answers are computed (and cross-checked by an independent method), every spot
passes its category's margin, and players answer through the same agents and backends, scored per category with
their loss in big blinds; fallbacks never score, and every answer keeps its raw output and provenance."""
import asyncio
import json
import random
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from poker_arena import backends, quiz
from poker_arena.backends import Backend, Reply
from poker_arena.cards import DECK, evaluate, equity as random_equity
from poker_arena.games import ranges


def spot_of(sid):
    return quiz.solve(next(d for d in quiz.suite() if d['id'] == sid))


def reply_for(option):
    if option.startswith('bet_'):
        return {'action': 'raise', 'amount': int(option.split('_')[1])}
    return {'action': option, 'amount': None}


class Oracle(Backend):
    """Answers every question with the computed answer; `WRONG` names questions it answers wrongly first.
    It names the model that answered: the requested one, or `ANSWER`."""
    WRONG = set()
    ANSWER = None

    async def _call(self, text):
        sid = re.findall(r'Question (\S+)\.', text)[-1]
        spot = spot_of(sid)
        act = {'action': 'raise', 'amount': 50} if sid in Oracle.WRONG and 'could not be played' not in text \
            else reply_for(spot['correct'])
        return Reply(json.dumps({'reasoning': 'oracle', **act, 'say': '', 'notes': None}),
                     meta={'observed_model': Oracle.ANSWER or self.model})


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

    def test_the_basics_are_the_original_clear_cut_spots(self):
        spots = quiz.spots(['basics'])
        self.assertEqual([s['id'] for s in spots], [t[0] for t in quiz.SPOTS])
        answers = [s['correct'] for s in spots]
        self.assertEqual(set(answers), {'fold', 'call', 'all_in'})
        self.assertLess(answers.count('fold') / len(answers), 0.6)       # folding everything is no strategy
        for s in spots:
            self.assertEqual(s['category'], 'basics')
            self.assertGreaterEqual(s['gap_bb'], quiz.MIN_GAP_BB, s['id'])
            if s['kind'] == 'call':
                self.assertGreaterEqual(abs(s['equity'] - s['required_equity']), quiz.MIN_EQUITY_GAP, s['id'])
        self.assertEqual(quiz.spots_hash(), quiz.spots_hash())

    def test_the_harder_categories(self):
        every = quiz.spots()
        self.assertEqual(len({s['id'] for s in every}), len(every))
        for cat in ('close', 'multistreet', 'inference', 'sizing'):
            spots = [s for s in every if s['category'] == cat]
            self.assertEqual(len(spots), quiz.PER_CATEGORY, cat)
            best = Counter(s['correct'] for s in spots)
            self.assertLessEqual(max(best.values()), quiz.MAX_SHARE * quiz.PER_CATEGORY, cat)
            self.assertGreaterEqual(len(best), 2, cat)
            for s in spots:
                lo, hi = quiz.CLOSE_GAP if cat == 'close' else quiz.HARD_GAP_BB
                self.assertTrue(lo <= s['gap_bb'] <= hi, s['id'])
                ordered = sorted(s['ev_bb'].values(), reverse=True)
                self.assertAlmostEqual(s['gap_bb'], ordered[0] - ordered[1], places=2)
                self.assertEqual(s['ev_bb'][s['correct']], ordered[0])
        self.assertTrue(all(s['gap_bb'] < quiz.MIN_GAP_BB for s in every if s['category'] == 'close'))
        sizing = [s for s in every if s['category'] == 'sizing']
        self.assertTrue(all(len(s['options']) >= 5 and s['options'][0] == 'check' for s in sizing))
        self.assertGreaterEqual(len({s['correct'] for s in sizing} - {'check'}), 3)   # different sizes win
        self.assertTrue(all(s['options'] == ['fold', 'call', 'all_in'] for s in every if s['category'] == 'multistreet'))
        self.assertTrue(all(s['street'] == 'river' and 'derived_range' in s for s in every if s['category'] == 'inference'))
        before = [d['id'] for d in quiz.suite()]
        quiz._GENERATED.clear()
        self.assertEqual([d['id'] for d in quiz.suite()], before)        # seeded: the same spots every time

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


def win(hole, combo, board):
    a, b = evaluate(list(hole) + board), evaluate(list(combo) + board)
    return 1.0 if a > b else 0.5 if a == b else 0.0


def hand_type(combo, board):
    """The features written out again, without the quiz's code."""
    cards = list(combo) + list(board)
    cat = evaluate(cards)[0]
    ranks = [c[0] for c in cards]
    order = '23456789TJQKA'
    pairs = [r for r in set(ranks) if ranks.count(r) >= 2]
    top = max(board, key=lambda c: order.index(c[0]))[0]
    suits = Counter(c[1] for c in cards)
    draw = any(n == 4 and any(c[1] == s for c in combo) for s, n in suits.items()) and max(suits.values()) < 5
    return {'no_pair': cat == 0, 'pair_plus': cat >= 1, 'two_pair_plus': cat >= 2, 'trips_plus': cat >= 3,
            'top_pair_plus': cat >= 2 or (cat == 1 and order.index(pairs[0]) >= order.index(top)),
            'flush_draw': draw}


class TestHarderAnswersIndependently(unittest.TestCase):
    """Each generated kind's values recomputed by brute force, straight from its definition."""

    def combos(self, d):
        seen = set(d['hole'] + d['board'])
        return [c for c in ranges.parse(d['range']) if not set(c) & seen]

    def test_inference_ranges(self):
        for d in quiz.generated('inference')[:8]:
            keep = [c for c in self.combos(d)
                    if all(any(hand_type(c, d['board'][:n])[f] for f in d['line'][st])
                           for st, n in (('flop', 3), ('turn', 4), ('river', 5)))]
            s = quiz.solve(d)
            self.assertEqual(s['derived_range'], sorted(''.join(c) for c in keep), d['id'])
            eq = sum(win(d['hole'], c, d['board']) for c in keep) / len(keep)
            self.assertAlmostEqual(s['equity'], round(eq, 4), places=4)
            self.assertAlmostEqual(s['ev_bb']['call'], (eq * (s['pot'] + s['to_call']) - s['to_call']) / 2, places=2)

    def test_sizing_values(self):
        for d in quiz.generated('sizing')[:8]:
            s = quiz.solve(d)
            combos, pot = self.combos(d), s['pot']
            check = sum(win(d['hole'], c, d['board']) * pot for c in combos) / len(combos)
            for opt, rule in d['responses'].items():
                bet = s['all_in_amount'] if opt == 'all_in' else int(opt.split('_')[1])
                total = 0.0
                for c in combos:
                    if any(hand_type(c, d['board'])[f] for f in rule):
                        total += win(d['hole'], c, d['board']) * (pot + 2 * bet) - bet
                    else:
                        total += pot
                self.assertAlmostEqual(s['ev_bb'][opt], (total / len(combos) - check) / 2, places=2, msg=d['id'])

    def test_multistreet_values(self):
        for d in quiz.generated('multistreet')[:3]:
            s = quiz.solve(d)
            combos, board, pot, tc = self.combos(d), d['board'], s['pot'], s['to_call']
            left = s['stack'] - tc
            pairs = [(c, r) for c in combos for r in DECK if r not in set(c) | set(board) | set(d['hole'])]
            policy = {}
            for r in {r for _, r in pairs}:      # hero's best response to the river all-in, card by card
                shove = [c for c, rr in pairs if rr == r and any(hand_type(c, board + [r])[f] for f in d['river_plan'])]
                if shove:
                    eq = sum(win(d['hole'], c, board + [r]) for c in shove) / len(shove)
                    policy[r] = eq * (pot + tc + 2 * left) - tc - left >= -tc
            total = 0.0
            for c, r in pairs:
                full = board + [r]
                if any(hand_type(c, full)[f] for f in d['river_plan']):
                    total += win(d['hole'], c, full) * (pot + tc + 2 * left) - tc - left if policy[r] else -tc
                else:
                    total += win(d['hole'], c, full) * (pot + tc) - tc
            self.assertAlmostEqual(s['ev_bb']['call'], total / len(pairs) / 2, places=2, msg=d['id'])
            shove = 0.0
            for c in combos:
                if any(hand_type(c, board)[f] for f in d['shove_call']):
                    runs = [r for r in DECK if r not in set(c) | set(board) | set(d['hole'])]
                    eq = sum(win(d['hole'], c, board + [r]) for r in runs) / len(runs)
                    shove += eq * (pot + tc + 2 * left) - tc - left
                else:
                    shove += pot
            self.assertAlmostEqual(s['ev_bb']['all_in'], shove / len(combos) / 2, places=2, msg=d['id'])

    def test_every_spot_is_an_engine_position(self):
        for d in quiz.suite():
            ep, hero, villain, _ = quiz.build(d)
            s = quiz.solve(d)
            self.assertEqual(ep.to_act, hero)
            text = quiz.question(s, ep, hero)
            self.assertIn(f'Question {s["id"]}.', text)
            self.assertIn(s['villain_range'], text)
            self.assertTrue(text.rstrip().endswith('.') and 'Your options: ' in text)
            if s['kind'] == 'inference':
                self.assertNotIn('range here is exactly', text)          # the range is never stated
                self.assertIn(quiz.rule_text(d['line']['river']), text)
            for o in s['options']:
                self.assertEqual(quiz._executed(ep, hero, reply_for(o)), (o, None), (s['id'], o))


class TestAnswers(unittest.TestCase):
    def setUp(self):
        backends.BACKENDS.update(oracle=Oracle, folder=Folder)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'q'
        Oracle.WRONG, Oracle.ANSWER = set(), None

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
        n = len(quiz.spots(['basics']))
        o = doc['players']['oracle']
        self.assertEqual((o['answered'], o['correct'], o['ev_loss_bb'], o['fallbacks']), (n, n, 0.0, 0))
        self.assertEqual(o['scored_category'], 'basics')
        for cat, x in o['categories'].items():
            self.assertEqual((x['correct'], x['answered'], x['valid_accuracy']), (x['total'], x['total'], 1.0), cat)
        self.assertEqual(sum(x['answered'] for x in o['categories'].values()), len(doc['spots']))
        fixed = next(a for a in o['answers'] if a['spot'] == 'river-call-bluffcatcher')
        self.assertEqual((fixed['correct'], len(fixed['calls']), fixed['errors'][0]['kind']), (True, 2, 'invalid'))
        self.assertIn('not one of this question', fixed['errors'][0]['error'])
        a = o['answers'][0]
        self.assertEqual((a['outcome'], a['identity_match'], a['audit']), ('valid', 'match', 'verified'))
        self.assertIn('synthetic', a['calls'][0]['raw'])                 # the raw output stays with the answer
        self.assertEqual(o['eligibility'], {'status': 'eligible', 'reasons': [], 'identity_match': 'match',
                                            'audit': 'verified'})
        self.assertEqual(len(o['protocol']['fingerprint']), 16)
        self.assertEqual(o['implementation']['fingerprint'], doc['implementation']['fingerprint'])
        self.assertIn('quiz', doc['implementation'])
        self.assertEqual(set(doc['categories']), set(quiz.CATEGORIES))
        self.assertIsNone(doc['categories']['basics']['population'])
        self.assertEqual(doc['categories']['sizing']['generator']['seed'], quiz.GENERATOR_SEED)

        f = doc['players']['folder']             # folds: never credited for a fallback, even where folding is best
        basics = [s for s in doc['spots'] if s['category'] == 'basics']
        self.assertEqual(f['correct'], sum(s['correct'] == 'fold' for s in basics))
        sz = f['categories']['sizing']
        self.assertEqual((sz['correct'], sz['fallbacks'], sz['valid_answers'], sz['valid_accuracy']),
                         (0, quiz.PER_CATEGORY, 0, None))
        checked = [x for x in f['answers'] if x['category'] == 'sizing' and spot_of(x['spot'])['correct'] == 'check']
        self.assertTrue(checked and all(x['executed'] == 'check' and x['fallback'] and not x['correct']
                                        and x['outcome'] == 'fallback' for x in checked))
        worst = {x['spot']: max(spot_of(x['spot'])['ev_bb'].values()) - min(spot_of(x['spot'])['ev_bb'].values())
                 for x in checked}
        self.assertTrue(all(x['ev_loss_bb'] == round(worst[x['spot']], 3) for x in checked))
        self.assertEqual(f['eligibility']['status'], 'ineligible')     # it names no model
        st = doc['players']['station']
        self.assertTrue(all(x['executed'] in ('call', 'fold', 'check') for x in st['answers']))
        self.assertEqual(st['eligibility']['status'], 'bot')
        self.assertEqual(doc['version'], quiz.VERSION)
        self.assertEqual(doc['label'], 'poker fundamentals')
        self.assertTrue({'spot', 'category', 'requested', 'executed', 'action', 'correct', 'outcome', 'ev_loss_bb',
                         'ev_lost_bb', 'say', 'fallback', 'reasoning', 'errors', 'seconds', 'identity_match', 'audit',
                         'calls'} <= set(a))
        self.assertEqual(a['calls'][0]['requested_model'], 'o-1')
        s = doc['spots'][0]
        for k in ('id', 'category', 'hole', 'board', 'villain_range', 'equity', 'ev_bb', 'correct', 'options', 'gap_bb'):
            self.assertIn(k, s)
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'quiz.json').read_text())
        for k in fixture['spots'][0]:                    # the shape the viewer reads
            self.assertIn(k, s)
        self.assertEqual(doc['suite'], {'name': 'Poker fundamentals', 'version': quiz.VERSION})
        self.assertEqual((s['best'], s['ev'], s['bb']), (s['correct'], s['ev_bb'], 2))
        # asking one player again replaces only its results
        doc = asyncio.run(quiz.run_quiz(players, ['folder'], self.out, limit=2, log=lambda *a: None))
        self.assertEqual(doc['players']['folder']['answered'], 2)
        self.assertEqual(doc['players']['oracle']['answered'], n)
        self.assertEqual(len(doc['spots']), len(quiz.spots()))

    def test_provenance_fails_closed(self):
        players = {'oracle': {'backend': 'oracle', 'model': 'o-1'}}
        Oracle.ANSWER = 'other-9'
        doc = asyncio.run(quiz.run_quiz(players, ['oracle'], self.out, categories=['basics'], limit=2,
                                        log=lambda *a: None))
        e = doc['players']['oracle']['eligibility']
        self.assertEqual((e['status'], e['identity_match']), ('ineligible', 'mismatch'))
        a = doc['players']['oracle']['answers'][0]
        call = dict(a['calls'][0], output='{"action": "call"}')              # a record that disagrees with its raw
        self.assertEqual(quiz._answer_audit(players['oracle'], [call])[1], 'inconsistent')
        self.assertEqual(quiz._answer_audit({'backend': 'claude'}, [dict(a['calls'][0], raw=None)])[1], 'unverifiable')
        moved = dict(a['calls'][0], cli={'path': '/x', 'version': '2', 'sha256': 'b'})
        e = quiz.eligibility('model', [{**a, 'identity_match': 'match', 'calls': [moved]}],
                             {'path': '/x', 'version': '1', 'sha256': 'a'})
        self.assertTrue(any('CLI changed' in r for r in e['reasons']))

    def test_a_backend_failure_is_an_error_not_a_score(self):
        class Broken(Backend):
            async def _call(self, text):
                raise backends.BackendError('down')
        backends.BACKENDS['broken'] = Broken
        self.addCleanup(backends.BACKENDS.pop, 'broken', None)
        doc = asyncio.run(quiz.run_quiz({'b': {'backend': 'broken', 'retries': 0}}, ['b'], self.out,
                                        categories=['basics'], limit=3, log=lambda *a: None))
        r = doc['players']['b']
        self.assertEqual((r['correct'], r['fallbacks'], r['valid_answers']), (0, 3, 0))
        self.assertTrue(all(a['outcome'] == 'error' and not a['correct'] for a in r['answers']))
        self.assertEqual(r['eligibility']['status'], 'ineligible')

    def test_the_cli(self):
        from poker_arena.cli import main
        players = Path(self.tmp.name) / 'players.toml'
        players.write_text('[players.station]\nbot = "station"\n[players.reference]\nbot = "reference"\n')
        main(['quiz', 'station', 'reference', '--players', str(players), '--out', str(self.out), '--limit', '3',
              '--category', 'basics', '--category', 'sizing'])
        doc = json.loads((self.out / 'quiz.json').read_text())
        self.assertEqual({p: r['answered'] for p, r in doc['players'].items()}, {'station': 3, 'reference': 3})
        self.assertEqual({p: sorted(r['categories']) for p, r in doc['players'].items()},
                         {'station': ['basics', 'sizing'], 'reference': ['basics', 'sizing']})
        self.assertEqual(doc['reference'], ['reference'])


if __name__ == '__main__':
    unittest.main()
