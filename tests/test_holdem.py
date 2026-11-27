import asyncio
import random
import unittest

from poker_arena.cards import evaluate, describe
from poker_arena.games import get_game


def H(s):
    return s.split()


class TestEvaluator(unittest.TestCase):
    def test_categories(self):
        cases = [
            ('As Ks Qs Js Ts 2d 3c', 8), ('9h 9d 9s 9c 2d 3c 4h', 7), ('Kh Kd Ks 2c 2d 7h 8s', 6),
            ('2h 7h 9h Jh Kh As Ad', 5), ('5d 6c 7h 8s 9d Kc Kd', 4), ('Ah 2d 3c 4s 5h Kd Qd', 4),
            ('Qh Qd Qs 2c 7d 9h Jh', 3), ('Jh Jd 4s 4c 2d 9h Kh', 2), ('Th Td 2s 5c 7d 9h Ah', 1),
            ('2h 5d 7s 9c Jd Kh Ah', 0)]
        for cards, cat in cases:
            self.assertEqual(evaluate(H(cards))[0], cat, cards)

    def test_ordering_and_kickers(self):
        self.assertGreater(evaluate(H('Ah Ad Ks 7c 5d 3h 2s')), evaluate(H('Ah Ad Qs 7c 5d 3h 2s')))
        self.assertEqual(evaluate(H('Ah 2d 3c 4s 5h Kd Qd')), (4, 5))   # the wheel is five high
        self.assertGreater(evaluate(H('6h 2d 3c 4s 5h Kd Qd')), evaluate(H('Ah 2d 3c 4s 5h Kd Qd')))
        # board plays: split
        self.assertEqual(evaluate(H('2c 3d Ah Kh Qh Jh Th')), evaluate(H('4c 5d Ah Kh Qh Jh Th')))
        self.assertEqual(describe(evaluate(H('Kh Kd Ks 2c 2d 7h 8s'))), 'a full house, kings full of deuces')

    def test_two_trips_make_a_full_house(self):
        self.assertEqual(evaluate(H('Kh Kd Ks 2c 2d 2h 8s')), (6, 13, 2))


def play(ep, policy):
    steps = 0
    while not ep.done:
        s = ep.to_act
        req = ep.request(s)
        err = ep.apply(s, policy(s, req))
        assert err is None, err
        steps += 1
        assert steps < 1000
    return ep


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.G = get_game('holdem')({'hands_per_match': 10})

    def test_random_play_conserves_chips(self):
        for seed in range(1500):
            n = 2 if seed % 4 else 2 + seed % 5
            rng = random.Random(seed)

            def policy(s, req):
                a = rng.choice(req.legal)
                act = {'action': a}
                if a == 'raise':
                    lo, hi = req.data['raise_bounds']
                    act['amount'] = rng.randint(lo, hi)
                return act
            ep = play(self.G.episode(seed % n, seed, [f'P{i}' for i in range(n)]), policy)
            self.assertEqual(sum(ep.payoffs), 0)
            self.assertTrue(all(x >= 0 for x in ep.stacks))

    def test_heads_up_blinds_and_order(self):
        ep = self.G.episode(0, 1, ['A', 'B'])
        self.assertEqual(ep.button, 0)
        self.assertEqual(ep.street_bet, [1, 2])   # button posts the small blind
        self.assertEqual(ep.to_act, 0)            # and acts first preflop
        ep.apply(0, {'action': 'call'})
        self.assertEqual(ep.to_act, 1)            # big blind has the option
        ep.apply(1, {'action': 'check'})
        self.assertEqual(ep.street, 1)
        self.assertEqual(ep.to_act, 1)            # and acts first after the flop

    def test_same_seed_same_cards(self):
        a = self.G.episode(3, 77, ['A', 'B'])
        b = self.G.episode(3, 77, ['B', 'A'])
        self.assertEqual(a.hole, b.hole)
        self.assertEqual(a.deck_board, b.deck_board)

    def test_min_raise_and_corrections(self):
        ep = self.G.episode(0, 5, ['A', 'B'])
        self.assertIsNone(ep.apply(0, {'action': 'raise', 'amount': 3}))   # below the min of 4: raised to 4
        self.assertEqual(ep.history[-1]['amount'], 4)
        self.assertIn('minimum', ep.history[-1]['note'])
        self.assertIsNotNone(ep.apply(1, {'action': 'check'}))             # can't check facing a raise
        self.assertIsNone(ep.apply(1, {'action': 'raise', 'amount': 10}))  # re-raise by 6
        self.assertEqual(ep.raise_bounds(0)[0], 16)                        # next min is 10 + 6
        self.assertIsNone(ep.apply(0, {'action': 'raise', 'amount': 9999}))
        self.assertTrue(ep.history[-1].get('all_in'))
        self.assertIsNone(ep.apply(1, {'action': 'call'}))
        self.assertTrue(ep.done)
        self.assertEqual(len(ep.board), 5)
        self.assertEqual(sum(ep.payoffs), 0)

    def test_correction_policy(self):
        ep = self.G.episode(0, 5, ['A', 'B'])
        for bad in ('Infinity', float('inf'), float('nan'), '-inf', 'lots', True, None):
            err = ep.apply(0, {'action': 'raise', 'amount': bad})              # refused: asked again, no crash
            self.assertIsNotNone(err, bad)
            self.assertEqual(ep.history, [])
        self.assertIn('not a raise', ep.apply(0, {'action': 'raise', 'amount': 2}))   # the bet is already 2
        self.assertIsNone(ep.apply(0, {'action': 'call'}))
        self.assertIn('cannot fold', ep.apply(1, {'action': 'fold'}))      # nothing to call: no silent check
        self.assertIsNone(ep.apply(1, {'action': 'call'}))                  # a call of nothing is a check (sizing)
        last = ep.history[-1]
        self.assertEqual(last['action'], 'check')
        self.assertEqual(last['correction']['kind'], 'sizing')
        self.assertEqual(last['requested'], {'action': 'call', 'amount': None})
        self.assertIsNone(ep.apply(1, {'action': 'bet', 'amount': 7.6}))    # rounded, with a notice
        self.assertEqual((ep.history[-1]['amount'], ep.history[-1]['note']), (8, '7.6 rounded to 8'))
        self.assertIsNone(ep.apply(0, {'action': 'raise', 'amount': '20'}))  # a numeric string is fine
        self.assertNotIn('correction', ep.history[-1])

    def test_raise_against_all_in_is_a_call_but_a_closed_raise_is_refused(self):
        ep = self.G.episode(0, 5, ['A', 'B'])
        ep.stacks[1] = 10 - ep.committed[1]                                  # B has 10 in all
        ep.apply(0, {'action': 'raise', 'amount': 20})
        self.assertIsNone(ep.apply(1, {'action': 'raise', 'amount': 50}))   # can only call all in: same chips
        self.assertEqual(ep.history[-1]['action'], 'call')
        self.assertEqual(ep.history[-1]['correction']['kind'], 'sizing')
        # three-handed: an all-in short of a full raise doesn't reopen the betting for the raiser
        ep = self.G.episode(0, 5, ['A', 'B', 'C'])
        ep.stacks[1] = 14 - ep.committed[1]
        self.assertIsNone(ep.apply(0, {'action': 'raise', 'amount': 10}))
        self.assertIsNone(ep.apply(1, {'action': 'all_in'}))                # to 14: +4, less than the raise of 8
        self.assertIsNone(ep.apply(2, {'action': 'call'}))
        self.assertTrue(ep.raise_closed(0))
        self.assertIn('not reopened', ep.apply(0, {'action': 'raise', 'amount': 40}))
        self.assertIn('not reopened', ep.apply(0, {'action': 'all_in'}))
        self.assertIsNone(ep.apply(0, {'action': 'call'}))

    def test_deal_fingerprints(self):
        G2 = get_game('holdem')({'hands_per_match': 99})
        self.assertEqual(self.G.deal_fingerprint(7), G2.deal_fingerprint(7))   # match length isn't the deal
        self.assertNotEqual(self.G.deal_fingerprint(7), self.G.deal_fingerprint(8))
        self.assertNotEqual(self.G.deal_fingerprint(7), get_game('holdem')({'stack': 100}).deal_fingerprint(7))

    def test_fold_awards_pot(self):
        ep = self.G.episode(0, 9, ['A', 'B'])
        ep.apply(0, {'action': 'fold'})
        self.assertTrue(ep.done)
        self.assertEqual(ep.payoffs, [-1, 1])

    def test_side_pots(self):
        G = get_game('holdem')({'hands_per_match': 10})
        ep = G.episode(0, 11, ['A', 'B', 'C'])
        ep.stacks[1] = 30 - ep.committed[1]   # give the small blind a short stack
        play(ep, lambda s, req: {'action': 'all_in'} if 'all_in' in req.legal else {'action': 'call'})
        self.assertEqual(sum(ep.payoffs), 30 - 200)  # chips are conserved given the short starting stack
        self.assertEqual(sum(ep.result['won']), ep.pot)


class TestReplayContext(unittest.TestCase):
    """What the viewer needs to follow a hand live: who acts next, the price, and every seat's made hand."""

    def test_to_act_to_call_pot_odds_and_made_hands(self):
        G = get_game('holdem')({'hands_per_match': 10})
        for seed in range(300):
            n = 2 if seed % 3 else 3
            rng = random.Random(seed)
            ep = G.episode(seed % n, seed, [f'P{i}' for i in range(n)])
            start = ep.events[0]
            self.assertEqual(start.kind, 'hand_start')
            self.assertEqual(start.data['to_act'], None if ep.done else ep.to_act)
            self.assertEqual(start.data['omniscient']['made_hand'],
                             {i: describe(evaluate(ep.hole[i])) for i in range(n)})
            while not ep.done:
                seat, req = ep.to_act, ep.request(ep.to_act)
                pot, tc = ep.pot, ep.to_call(seat)
                a = rng.choice(req.legal)
                act = {'action': a, **({'amount': rng.randint(*req.data['raise_bounds'])} if a == 'raise' else {})}
                self.assertIsNone(ep.apply(seat, act))
                last = ep.history[-1]
                self.assertEqual(last['to_act'], None if ep.done else ep.to_act)
                self.assertEqual((last['to_call'], last['pot_before']), (tc, pot))
                self.assertEqual(last['pot_odds'], round(tc / (pot + tc), 4) if tc else None)
            actions = [e for e in ep.events if e.kind == 'action']
            self.assertIsNone(actions[-1].data['to_act'] if actions else None)
            for e in ep.events:
                if e.kind == 'board':
                    self.assertEqual(e.data['omniscient']['made_hand'],
                                     {i: describe(evaluate(ep.hole[i] + e.data['board'])) for i in range(n)})
                    self.assertNotIn('omniscient', ep.render(e, 0))
            made = ep.summary()['omniscient']['made_hand']
            self.assertEqual(list(made), ['preflop', 'flop', 'turn', 'river'][:len(made)])

    def test_a_pair_preflop_and_on_the_flop(self):
        G = get_game('holdem')({'hands_per_match': 10})
        ep = G.episode(0, 1, ['A', 'B'])
        ep.hole = [['As', 'Ad'], ['7c', '2h']]
        ep.deck_board = ['Ah', '7d', '9s', 'Kc', '3d']
        ep.made_by_street = {'preflop': ep.made_hands()}
        ep.apply(0, {'action': 'call'})
        ep.apply(1, {'action': 'check'})
        made = ep.summary()['omniscient']['made_hand']
        self.assertEqual(made['preflop'], {0: 'a pair of aces', 1: 'seven high'})
        self.assertEqual(made['flop'], {0: 'three aces', 1: 'a pair of sevens'})


class TestBotsAndRatings(unittest.TestCase):
    def test_bot_match_and_ratings(self):
        import tempfile
        from pathlib import Path
        from poker_arena.runner import run_duplicate
        from poker_arena.ratings import ratings
        G = get_game('holdem')({'hands_per_match': 20})
        with tempfile.TemporaryDirectory() as d:
            res = asyncio.run(run_duplicate(G, ('rule', {'bot': 'rule', 'sims': 60}), ('station', {'bot': 'station'}),
                                            list(range(20)), Path(d), log=lambda *a: None))
            self.assertEqual(len(res), 2)
            r = ratings(d, boot=20)
            self.assertEqual(r['deals'], 20)
            self.assertEqual({p['player'] for p in r['players']}, {'rule', 'station'})
            # duplicate: both matches dealt the same cards
            import json
            ha = [json.loads(l) for l in (Path(d) / 'rule__vs__station/a-first/hands.jsonl').read_text().splitlines()]
            hb = [json.loads(l) for l in (Path(d) / 'rule__vs__station/b-first/hands.jsonl').read_text().splitlines()]
            self.assertEqual([h['hole'] for h in ha], [h['hole'] for h in hb])


class TestVerifier(unittest.TestCase):
    """verify_hand regenerates a recorded hand from its seed and replays its actions; any change to the cards,
    board, actions, payouts or events is caught."""

    @classmethod
    def setUpClass(cls):
        import copy, json, tempfile
        from pathlib import Path
        from poker_arena.runner import run_duplicate
        cls.G = get_game('holdem')({'hands_per_match': 12})
        cls.tmp = tempfile.TemporaryDirectory()
        asyncio.run(run_duplicate(cls.G, ('rule', {'bot': 'rule', 'sims': 30}), ('random', {'bot': 'random'}),
                                  list(range(40, 52)), Path(cls.tmp.name), log=lambda *a: None))
        m = Path(cls.tmp.name) / 'rule__vs__random' / 'a-first'
        rows = lambda f: [json.loads(l) for l in (m / f).read_text().splitlines()]
        cls.hands, cls.events = rows('hands.jsonl'), rows('events.jsonl')
        cls.copy = staticmethod(copy.deepcopy)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, h, events=None):
        from poker_arena.games.holdem import verify_hand
        evs = events if events is not None else [e for e in self.events if e['episode'] == h['hand']]
        return verify_hand(self.G, h['seed'], h['hand'] % 2, h, evs)

    def test_recorded_hands_verify(self):
        for h in self.hands:
            self.assertEqual(self.check(h), {'ok': True, 'errors': []})

    def test_older_records_without_derived_fields_verify(self):
        from poker_arena.games.holdem import DERIVED
        h = max(self.hands, key=lambda h: len(h['actions']))
        old = self.copy(h)
        for k in DERIVED:
            old.pop(k, None)
        for a in old['actions']:
            for k in DERIVED:
                a.pop(k, None)
        self.assertEqual(self.check(old)['errors'], [])
        old['actions'][-1]['pot'] += 1
        self.assertFalse(self.check(old)['ok'])
        wrong = self.copy(h)
        if any('to_call' in a for a in wrong['actions']):
            next(a for a in wrong['actions'] if 'to_call' in a)['to_call'] = 999
            self.assertFalse(self.check(wrong)['ok'])

    def test_tampering_is_caught(self):
        h = max(self.hands, key=lambda h: len(h['actions']))
        cases = {
            'duplicate aces': lambda x: x['hole'].__setitem__(0, ['As', 'As']),
            'payoff': lambda x: x.__setitem__('net', [200000, -200000]),
            'board': lambda x: x['board'].reverse(),
            'amount': lambda x: x['actions'][-1].__setitem__('pot', x['actions'][-1]['pot'] + 1),
            'out of turn': lambda x: x['actions'][0].__setitem__('seat', 1 - x['actions'][0]['seat']),
            'dropped action': lambda x: x['actions'].pop(),
            'won': lambda x: x.__setitem__('won', list(reversed(x['won']))),
        }
        for name, change in cases.items():
            x = self.copy(h)
            change(x)
            r = self.check(x, [])
            self.assertFalse(r['ok'], name)
            self.assertTrue(r['errors'], name)
        from poker_arena.games.holdem import verify_hand
        self.assertFalse(verify_hand(self.G, h['seed'] + 1, h['hand'] % 2, h)['ok'])     # another deal
        self.assertFalse(verify_hand(self.G, h['seed'], 1 - h['hand'] % 2, h)['ok'])     # the other button

    def test_events_are_checked(self):
        h = self.hands[0]
        evs = self.copy([e for e in self.events if e['episode'] == h['hand']])
        hole = next(e for e in evs if e['kind'] == 'hole')
        hole['data']['cards'] = ['Ac', 'Ad']
        self.assertFalse(self.check(h, evs)['ok'])
        evs = [e for e in self.copy(self.events) if e['episode'] == h['hand'] and e['kind'] != 'result']
        self.assertFalse(self.check(h, evs)['ok'])


if __name__ == '__main__':
    unittest.main()
