"""Solver, AIVAT and meta-game ranking: convergence to equilibrium on games with known answers, canonical spot
hashing and caching, the external adapters (offline), hand evaluation, the calibration bots, AIVAT's
unbiasedness and variance reduction, and Nash averaging / alpha-Rank / cycles."""
import asyncio
import itertools
import json
import os
import random
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from poker_arena import aivat as A
from poker_arena import meta_rank as MR
from poker_arena import solver as S
from poker_arena.agents import CallingStation
from poker_arena.cards import DECK, evaluate

BOARD = ['As', 'Ks', 'Qd', '7h', '2c']
NUTS = ['JhTh', 'JdTd', 'JcTc', 'JhTd']                 # straights
AIR = ['5h4h', '5d4d', '5c4c', '5h3h', '5d3d', '5c3c']   # lose to everything
CATCH = ['9h9d', '9h9c', '9d9c', '9s9h', '9s9d', '9s9c']  # beat air, lose to the straight


def key(c):
    return S.COMBO_STR[S.combo_index(S._parse_combo(c))]


_TOY = {}


def toy():
    """Polarized range (4 nut combos, 6 air) against a bluff-catcher, pot 10, stacks 10: a pot-size bet is all in.
    Theory: the bettor bets all its value and bluffs so bluffs are B/(P+2B) = 1/3 of its bets (2 of 6 air
    combos: frequency 1/3); the catcher calls P/(P+B) = 1/2; the bettor's range is worth 3 bb, the catcher's 2."""
    if not _TOY:
        _TOY['r'] = S.solve_river_subgame(BOARD, 10, 10, CATCH, NUTS + AIR, bet_sizes=(1.0,), iterations=600, bb=2,
                                          track=(10, 50, 200, 600))
    return _TOY['r']


class TestEvaluator(unittest.TestCase):
    def test_fast_rank_orders_hands_exactly_like_cards_evaluate(self):
        rng = random.Random(7)
        for k in (5, 6, 7):
            hands = [rng.sample(range(52), k) for _ in range(2500)]
            fast = S.fast_rank(np.array(hands))
            slow = [evaluate([DECK[c] for c in h]) for h in hands]
            for i in range(len(hands) - 1):
                self.assertEqual(int(fast[i] > fast[i + 1]) - int(fast[i] < fast[i + 1]),
                                 (slow[i] > slow[i + 1]) - (slow[i] < slow[i + 1]), (hands[i], hands[i + 1]))


class TestRiverDCFR(unittest.TestCase):
    def test_polarized_vs_bluffcatcher_matches_theory(self):
        r = toy()
        root, facing = r['strategy'][''], r['strategy']['allin']
        self.assertEqual(root['actions'], ['check', 'allin'])
        for c in NUTS:
            self.assertGreater(root['freqs'][key(c)][1], 0.99)          # value always bets
        for c in AIR:
            self.assertAlmostEqual(root['freqs'][key(c)][1], 1 / 3, delta=0.02)  # bluff 1/3 of air
        for c in CATCH:
            self.assertAlmostEqual(facing['freqs'][key(c)][1], 0.5, delta=0.02)  # call 1/2 (MDF)
        self.assertAlmostEqual(r['root_ev_oop'], 3.0, delta=0.02)
        self.assertAlmostEqual(r['root_ev_ip'], 2.0, delta=0.02)
        self.assertAlmostEqual(r['root_ev_oop'] + r['root_ev_ip'], 5.0, places=6)

    def test_indifference_and_clear_actions(self):
        r = toy()
        bluff = r['node_eval']('5h4h')
        self.assertTrue(bluff['is_indifferent'])                     # a mixing bluff: both actions worth 0
        self.assertAlmostEqual(bluff['action_evs_bb']['check'], 0.0, delta=0.05)
        nut = r['node_eval']('JhTh')
        self.assertEqual(nut['best_action'], 'allin')
        self.assertFalse(nut['is_indifferent'])
        self.assertAlmostEqual(nut['action_evs_bb']['allin'], 7.5, delta=0.1)   # called half the time: 0.5*5 + 0.5*10
        catcher = r['node_eval']('9h9d', ('allin',))
        self.assertTrue(catcher['is_indifferent'])
        self.assertAlmostEqual(sum(catcher['gto_freqs'].values()), 1.0, places=9)
        self.assertGreater(catcher['entropy'], 0.99)                 # a 50/50 mix is one bit

    def test_exploitability_falls_to_a_small_residual(self):
        curve = toy()['exploitability_curve']
        mbb = [x for _, x in curve]
        self.assertTrue(all(a > b for a, b in zip(mbb, mbb[1:])), curve)
        self.assertLess(mbb[-1], 5.0)                                # under 0.1% of a 5 bb pot
        self.assertAlmostEqual(toy()['exploitability_mbb'], mbb[-1], places=9)

    def test_card_removal_matches_brute_force(self):
        rng = np.random.default_rng(11)
        board = S._ids(['Ah', '9c', '5d', '3s', 'Td'])
        rk = S._river_ranks(tuple(board))
        live = np.nonzero(rk >= 0)[0]
        hero, opp = rng.choice(live, 120, replace=False), rng.choice(live, 140, replace=False)
        ev = S._ComboEval(S.COMBOS[hero], rk[hero], S.COMBOS[opp], rk[opp])
        W = ((S.COMBO_MASK[hero][:, None] & S.COMBO_MASK[opp][None, :]) == 0).astype(float)
        Sd = W * np.sign(rk[hero][:, None] - rk[opp][None, :])
        Q = rng.random((5, len(opp)))
        np.testing.assert_allclose(ev.mass(Q), Q @ W.T, atol=1e-12)
        np.testing.assert_allclose(ev.sd(Q), Q @ Sd.T, atol=1e-12)

    def test_full_ranges_converge_and_solve_fast(self):
        rng = np.random.default_rng(3)
        board = ['Ah', '9c', '5d', '3s', 'Td']
        w_oop, w_ip = np.zeros(1326), np.zeros(1326)
        w_oop[rng.choice(1326, 250, replace=False)] = rng.random(250) + 0.2
        w_ip[rng.choice(1326, 250, replace=False)] = rng.random(250) + 0.2
        t = time.time()
        r = S.solve_river_subgame(board, 20, 90, w_ip, w_oop, iterations=300)
        self.assertLess(time.time() - t, 0.5)
        r2 = S.solve_river_subgame(board, 20, 90, w_ip, w_oop, iterations=300, track=(10, 50, 300))
        mbb = [x for _, x in r2['exploitability_curve']]
        self.assertTrue(mbb[0] > mbb[1] > mbb[2], mbb)
        self.assertLess(mbb[-1], 50.0)                               # under 0.5% of the 10 bb pot
        # blockers are exact: no combo sharing a board card is in either range
        for p in ('oop', 'ip'):
            self.assertTrue(all(not set(S._parse_combo(c)) & set(board) for c in r['ev'][p]))


def spot_args(hole=('Ah', 'Kd'), board=('Qh', '7h', '2c'), history=None):
    history = history or [{'seat': 0, 'street': 'preflop', 'action': 'raise', 'amount': 5, 'pot': 7},
                          {'seat': 1, 'street': 'preflop', 'action': 'call', 'amount': 3, 'pot': 10},
                          {'seat': 1, 'street': 'flop', 'action': 'check', 'pot': 10}]
    return dict(street='flop', hole=list(hole), board=list(board), pot=10, to_call=0, stacks=[195, 195],
                street_bet=[0, 0], seat=0, button=0, history=history)


def permute(cards, perm):
    return [c[0] + perm[c[1]] for c in cards]


class TestSpotHashAndCache(unittest.TestCase):
    def test_suit_isomorphic_and_seat_label_free(self):
        base = spot_args()
        h = S.spot_hash(**{k: v for k, v in base.items()})
        for perm in ({'s': 'h', 'h': 's', 'd': 'c', 'c': 'd'}, {'s': 'd', 'h': 'c', 'd': 's', 'c': 'h'}):
            iso = dict(base, hole=permute(base['hole'], perm), board=permute(base['board'], perm))
            self.assertEqual(S.spot_hash(**iso), h)
        flop_order = dict(base, board=[base['board'][2], base['board'][0], base['board'][1]])
        self.assertEqual(S.spot_hash(**flop_order), h)
        swapped = dict(base, seat=1, button=1, stacks=base['stacks'][::-1], street_bet=base['street_bet'][::-1],
                       history=[{**x, 'seat': 1 - x['seat']} for x in base['history']])
        self.assertEqual(S.spot_hash(**swapped), h)
        self.assertNotEqual(S.spot_hash(**dict(base, hole=['Ah', 'Kh'])), h)   # a flush draw is another spot
        self.assertNotEqual(S.spot_hash(**dict(base, extra={'sb': 1})), S.spot_hash(**dict(base, extra={'sb': 2})))

    def test_isomorphic_spots_solve_identically_and_hit_the_disk_cache(self):
        base = spot_args()
        perm = {'s': 'c', 'h': 'd', 'd': 'h', 'c': 's'}
        iso = dict(base, hole=permute(base['hole'], perm), board=permute(base['board'], perm))
        S._MEMO.clear()
        a = S.solve_postflop_spot(**base, iterations=60)
        S._MEMO.clear()
        b = S.solve_postflop_spot(**iso, iterations=60)
        self.assertEqual(a['spot_hash'], b['spot_hash'])
        self.assertEqual(a['action_evs_bb'], b['action_evs_bb'])
        self.assertEqual(a['gto_freqs'], b['gto_freqs'])
        self.assertEqual(a['solver'], 'dcfr_abstracted')
        with tempfile.TemporaryDirectory() as d:
            S._MEMO.clear()
            first = S.solve_postflop_spot(**base, iterations=60, cache_dir=d)
            self.assertTrue((Path(d) / f'{first["spot_hash"]}.json').exists())
            S._MEMO.clear()
            with mock.patch.object(S, '_abstract_solution', side_effect=AssertionError('cache missed')):
                again = S.solve_postflop_spot(**iso, iterations=60, cache_dir=d, taken_action='check')
            self.assertEqual(again['action_evs_bb'], first['action_evs_bb'])
            self.assertEqual(again['taken_action'], 'check')
            self.assertGreaterEqual(again['ev_loss_mbb'], 0.0)

    def test_spot_eval_schema(self):
        s = S.solve_postflop_spot(**spot_args(), iterations=40, taken_action='bet', taken_amount=5)
        keys = {'street', 'spot_hash', 'solver', 'pot_bb', 'to_call_bb', 'pot_odds', 'mdf', 'equity', 'best_action',
                'action_evs_bb', 'gto_freqs', 'taken_action', 'taken_ev_bb', 'ev_loss_mbb', 'entropy',
                'is_indifferent', 'is_blunder', 'exploitability_mbb'}
        self.assertEqual(set(s), keys | {'root_equity', 'eval_source', 'ev_reliable', 'node_reach', 'reach_fraction',
                                        'tremble_share', 'best_to', 'taken_to', 'mapped_to', 'sizing_loss_mbb', 'action_loss_mbb'})
        self.assertAlmostEqual(sum(s['gto_freqs'].values()), 1.0, places=9)
        self.assertEqual(set(s['gto_freqs']), {'check', 'raise', 'all_in'})
        self.assertEqual(s['taken_action'], 'raise')
        self.assertIn(s['best_action'], s['action_evs_bb'])
        self.assertGreaterEqual(s['ev_loss_mbb'], 0.0)
        json.dumps(s)


FAKE_SOLVER = '''#!{python}
import json, os, sys
combo = os.environ["FAKE_TX_COMBO"]
tree = {{"node_type": "action_node", "player": 1, "actions": ["CHECK", "BET 5.000000", "BET 195.000000"],
        "strategy": {{"actions": ["CHECK", "BET 5.000000", "BET 195.000000"], "strategy": {{combo: [0.25, 0.70, 0.05]}}}},
        "childrens": {{"CHECK": {{"node_type": "action_node", "player": 0, "actions": ["CHECK", "BET 10.000000"],
                     "strategy": {{"actions": ["CHECK", "BET 10.000000"], "strategy": {{}}}}}}}}}}
json.dump(tree, open("output_result.json", "w"))
'''


class TestTexasSolverAdapter(unittest.TestCase):
    def test_commands(self):
        tx = S.TexasSolverAdapter(binary='/nonexistent/console_solver')
        w = S.range_weights('TT+, AKs')
        cmds = tx.build_commands(['Qs', 'Jh', '2h'], 10, 95, tx.range_text(w), tx.range_text(w))
        for c in ('set_pot 10', 'set_effective_stack 95', 'set_board Qs,Jh,2h', 'set_range_ip ', 'set_range_oop ',
                  'set_bet_sizes oop,flop,bet,50,100', 'set_bet_sizes ip,river,raise,100', 'set_bet_sizes ip,turn,allin',
                  'set_allin_threshold', 'build_tree', 'set_thread_num', 'set_accuracy', 'set_max_iteration',
                  'start_solve', 'set_dump_rounds', 'dump_result output_result.json'):
            self.assertIn(c, cmds)
        self.assertLess(cmds.index('build_tree'), cmds.index('start_solve'))
        self.assertLess(cmds.index('start_solve'), cmds.index('dump_result'))
        self.assertIn('AA:1.000', tx.range_text(w))
        self.assertIn('AKs:1.000', tx.range_text(w))

    def test_parse_dump_tree(self):
        tree = {'node_type': 'action_node', 'actions': ['CHECK', 'BET 5.000000'],
                'strategy': {'actions': ['CHECK', 'BET 5.000000'], 'strategy': {'AhKd': [0.4, 0.6]}},
                'childrens': {'BET 5.000000': {'node_type': 'action_node', 'actions': ['FOLD', 'CALL', 'RAISE 20.0'],
                                               'strategy': {'actions': ['FOLD', 'CALL', 'RAISE 20.0'],
                                                            'strategy': {'QcQd': [0.1, 0.5, 0.4]}}}}}
        tx = S.TexasSolverAdapter(binary='/nonexistent')
        root = tx.parse_result(tree, 'KdAh', [], 195, 2)
        self.assertEqual([o['abstract'] for o in root], ['check', 'raise'])
        self.assertEqual([o['freq'] for o in root], [0.4, 0.6])
        facing = tx.parse_result(tree, 'QdQc', [{'action': 'bet', 'amount': 6}], 195, 2)
        self.assertEqual([o['abstract'] for o in facing], ['fold', 'call', 'raise'])
        self.assertIsNone(tx.parse_result(tree, 'KdAh', [{'action': 'check'}], 195, 2))

    def test_missing_binary_falls_back_to_dcfr(self):
        with mock.patch.dict(os.environ, {'POKER_ARENA_TEXASSOLVER': '/nonexistent/console_solver'}), \
                mock.patch.object(S.shutil, 'which', return_value=None), \
                mock.patch.object(S.TexasSolverAdapter, 'find_binary', staticmethod(lambda: None)):
            self.assertFalse(S.TexasSolverAdapter().available())
            self.assertIsNone(S.TexasSolverAdapter().solve_spot(**spot_args()))
            s = S.solve_postflop_spot(**spot_args(), solver='texassolver', iterations=40)
        self.assertEqual(s['solver'], 'dcfr_abstracted')

    def test_fake_binary_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / 'console_solver'
            exe.write_text(FAKE_SOLVER.format(python=sys.executable))
            exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
            args = spot_args(hole=('Ah', 'Kd'), history=[
                {'seat': 0, 'street': 'preflop', 'action': 'raise', 'amount': 5, 'pot': 7},
                {'seat': 1, 'street': 'preflop', 'action': 'call', 'amount': 3, 'pot': 10}])
            args['seat'], args['button'] = 1, 0          # the big blind acts first on the flop
            with mock.patch.dict(os.environ, {'POKER_ARENA_TEXASSOLVER': str(exe), 'FAKE_TX_COMBO': 'AhKd'}):
                self.assertEqual(S.TexasSolverAdapter.find_binary(), exe)
                S._MEMO.clear()
                s = S.solve_postflop_spot(**args, solver='texassolver', cache_dir=Path(d) / 'cache', iterations=40)
                self.assertEqual(s['solver'], 'texassolver')
                self.assertAlmostEqual(s['gto_freqs']['check'], 0.25)
                self.assertAlmostEqual(s['gto_freqs']['raise'], 0.70)
                self.assertAlmostEqual(s['gto_freqs']['all_in'], 0.05)
                self.assertTrue(all(isinstance(v, float) for v in s['action_evs_bb'].values()))
                self.assertTrue(list((Path(d) / 'cache').glob('tx-*.json')))


class FakeSlumbot:
    """A scripted Slumbot server: the client is the small blind (button) and calls down; Slumbot raises to 300
    preflop, then checks every street; the client wins 300 chips at showdown."""

    def __init__(self):
        self.calls = []
        self.script = [
            {'action': 'cb300', 'board': []},
            {'action': 'cb300c/k', 'board': ['2c', '7d', 'Th']},
            {'action': 'cb300c/kk/k', 'board': ['2c', '7d', 'Th', 'Js']},
            {'action': 'cb300c/kk/kk/k', 'board': ['2c', '7d', 'Th', 'Js', '3h']},
            {'action': 'cb300c/kk/kk/kk', 'board': ['2c', '7d', 'Th', 'Js', '3h'], 'winnings': 300,
             'bot_hole_cards': ['Kc', 'Qc']},
        ]

    def __call__(self, url, payload):
        self.calls.append((url.rsplit('/', 1)[-1], payload))
        if url.endswith('new_hand'):
            return {'token': 'tok', 'action': '', 'client_pos': 1, 'hole_cards': ['Ah', 'Ad'], 'board': []}
        step = self.script.pop(0)
        return {'token': 'tok', 'client_pos': 1, 'hole_cards': ['Ah', 'Ad'], **step}


class TestSlumbot(unittest.TestCase):
    def test_action_strings(self):
        st = S.SlumbotClient.parse_action('b200c/kb300c/kk/b1000f')
        self.assertEqual(st['to_act'], -1)
        self.assertEqual([a['kind'] for a in st['actions']], ['b', 'c', 'k', 'b', 'c', 'k', 'k', 'b', 'f'])
        self.assertEqual(st['totals'], [1500, 500])
        st = S.SlumbotClient.parse_action('cb300')
        self.assertEqual((st['street'], st['to_act'], st['street_bets']), (0, 1, [300, 100]))
        st = S.SlumbotClient.parse_action('b200c/k')
        self.assertEqual((st['street'], st['to_act'], st['street_bets']), (1, 1, [0, 0]))
        arena = S.SlumbotClient.to_arena('b200c/kb300c', client_pos=1)
        self.assertEqual(arena[0], {'seat': 0, 'street': 'preflop', 'action': 'raise', 'amount': 4.0})
        self.assertEqual(arena[1], {'seat': 1, 'street': 'preflop', 'action': 'call', 'amount': 2.0})
        self.assertEqual(arena[3], {'seat': 0, 'street': 'flop', 'action': 'bet', 'amount': 6.0})
        state = S.SlumbotClient.parse_action('b200c/kb300')
        self.assertEqual(S.SlumbotClient.from_arena({'action': 'raise', 'amount': 20}, state, 0), 'b1000')
        self.assertEqual(S.SlumbotClient.from_arena({'action': 'call'}, state, 0), 'c')
        self.assertEqual(S.SlumbotClient.from_arena({'action': 'fold'}, state, 0), 'f')
        self.assertEqual(S.SlumbotClient.from_arena({'action': 'all_in'}, state, 0), 'b19800')
        self.assertEqual(S.SlumbotClient.from_arena({'action': 'check'}, S.SlumbotClient.parse_action('b200c/'), 0), 'k')

    def test_bot_plays_a_hand_offline(self):
        server = FakeSlumbot()
        bot = S.SlumbotBot(agent=CallingStation(), transport=server)
        res = bot.play_hand()
        sent = [p['incr'] for kind, p in server.calls if kind == 'act']
        self.assertEqual(sent, ['c', 'c', 'k', 'k', 'k'])
        self.assertTrue(all(p.get('token') == 'tok' for kind, p in server.calls if kind == 'act'))
        self.assertEqual(res['winnings_bb'], 3.0)
        self.assertEqual(res['bot_hole'], ['Kc', 'Qc'])
        d = S.SlumbotClient.request_data({'client_pos': 1, 'action': 'cb300', 'hole_cards': ['Ah', 'Ad'], 'board': []})
        self.assertEqual((d['to_call'], d['pot'], d['button'], d['street']), (4.0, 8.0, 0, 'preflop'))
        self.assertEqual(d['raise_bounds'], (10.0, 400.0))
        dec = S.run_agent(bot, '', type('R', (), {'data': d, 'legal': ['fold', 'call', 'raise', 'all_in']})(), None)
        self.assertEqual(dec.action, {'action': 'call'})


def royal_fold_record():
    """Seat 0 (button) holds a royal flush on the river and folds it to a small bet."""
    return {'hand': 0, 'button': 0, 'hole': [['Ah', 'Kh'], ['7c', '2d']], 'board': ['Qh', 'Jh', 'Th', '3s', '4d'],
            'actions': [{'seat': 0, 'street': 'preflop', 'action': 'call', 'amount': 1},
                        {'seat': 1, 'street': 'preflop', 'action': 'check'},
                        {'seat': 1, 'street': 'flop', 'action': 'check'}, {'seat': 0, 'street': 'flop', 'action': 'check'},
                        {'seat': 1, 'street': 'turn', 'action': 'check'}, {'seat': 0, 'street': 'turn', 'action': 'check'},
                        {'seat': 1, 'street': 'river', 'action': 'bet', 'amount': 4},
                        {'seat': 0, 'street': 'river', 'action': 'fold'}],
            'net': [-2, 2], 'won': [0, 8], 'seed': None}


class TestAuditSpots(unittest.TestCase):
    def replay(self, button, holes, board, actions):
        record = dict(button=button, hole=[hole.split() for hole in holes], board=board.split(),
                      actions=[dict(action='raise', amount=item) if isinstance(item, int) else dict(action=item)
                               for item in actions])
        decisions = S.evaluate_hand(record)['decisions']
        self.assertEqual(len(decisions), len(actions))
        return decisions

    def test_hand_23(self):
        spots = self.replay(1, ['Ah Ks', '5h Th'], 'Jc 2d Jh 6d 4c',
                            [5, 18, 'call', 12, 'call', 'check', 30, 'call', 'check', 'check'])
        flop, turn, river = spots[3], spots[7], spots[-1]
        self.assertEqual((flop['taken_to'], flop['mapped_to'], flop['best_to']), (12, 18, 36))
        self.assertEqual(flop['sizing_loss_mbb'] + flop['action_loss_mbb'], flop['ev_loss_mbb'])
        self.assertTrue(.22 <= turn['equity'] <= .26)
        self.assertAlmostEqual(turn['root_equity'], .597, delta=.02)
        self.assertLess(turn['ev_loss_mbb'], 100)
        self.assertEqual(river['eval_source'], 'solved')
        self.assertAlmostEqual(river['action_evs_bb']['all_in'], 9.6, delta=.6)
        self.assertAlmostEqual(river['action_evs_bb']['all_in'], river['action_evs_bb']['raise'], delta=.5)

    def test_zero_frequency_river_check_is_non_negative(self):
        # Unreached counterfactual actions must assume optimal continuation (fold to river bet with air), never negative EV
        spots = self.replay(1, ['Jc Qh', 'Kh Js'], '8h 8d 9h Kd 5s',
                            [5, 18, 'call', 16, 'call', 38, 'call', 'check', 45, 'fold'])
        river_check = spots[7]
        self.assertGreaterEqual(river_check['action_evs_bb']['check'], 0.0)
        self.assertAlmostEqual(river_check['action_evs_bb']['check'], 3.53, delta=0.5)

    def test_off_policy_queens_abstain(self):
        spots = self.replay(0, ['Qs Qh', '7h 5s'], '7d Kc 9d 9c 9s',
                            [5, 'call', 'check', 'check', 'check', 5, 'call', 10, 50, 'fold'])
        self.assertFalse(spots[8]['ev_reliable'])
        self.assertFalse(spots[8]['is_blunder'])

    def test_reference_river_policies(self):
        for holes, board, actions, preferred in [
                (['7d 8s', '4d Ts'], '8c 6s As 7h Kh', [5, 'call', 'check', 'check', 10, 'call', 185, 'call'], 'raise'),
                (['4s Ac', 'Kc Js'], '6d 3c 7h 6h As', [5, 'call', 5, 'call', 'check', 'check', 190, 'call'], 'check')]:
            river = self.replay(0, holes, board, actions)[-2]
            self.assertEqual(river['eval_source'], 'solved')
            self.assertGreater(river['gto_freqs'][preferred], river['gto_freqs']['all_in'])
            from poker_arena.games.holdem import Holdem, HoldemHand
            episode = HoldemHand(Holdem({'sb': 1, 'bb': 2, 'stack': 200}), 0, 0, ['OOP', 'IP'])
            episode.hole = [hand.split() for hand in holes]
            episode.deck_board = board.split()
            for action in actions[:-2]:
                move = {'action': 'raise', 'amount': action} if isinstance(action, int) else {'action': action}
                self.assertIsNone(episode.apply(episode.to_act, move))
            options = S.GTOBot().options(episode.request(episode.to_act).data)
            self.assertEqual(max(options, key=lambda item: item['freq'])['abstract'], preferred)

    def test_range_suit_permutation(self):
        args = spot_args()
        perm = dict(s='h', h='c', c='d', d='s')
        indices = [S.combo_index(S._ids(permute([DECK[first], DECK[second]], perm))) for first, second in S.COMBOS]
        original = S.infer_ranges(args['history'], args['board'], args['button'])
        permuted = S.infer_ranges(args['history'], permute(args['board'], perm), args['button'])
        for weights, other in zip(original, permuted):
            np.testing.assert_allclose(weights, other[indices], atol=1e-14, rtol=0)


class TestEvaluateHand(unittest.TestCase):
    def test_blunder_detection(self):
        ev = S.evaluate_hand(royal_fold_record())
        self.assertEqual(len(ev['decisions']), 8)
        last = ev['decisions'][-1]
        self.assertEqual((last['street'], last['taken_action']), ('river', 'fold'))
        self.assertIn(last['best_action'], ('raise', 'all_in'))     # raise the nuts
        evs = last['action_evs_bb']
        self.assertGreater(evs['call'] - evs['fold'], 3.9)          # calling wins the 2 bb pot and the 2 bb bet
        self.assertEqual(last['solver'], 'dcfr_exact_river')
        self.assertTrue(last['is_blunder'])
        self.assertGreater(last['ev_loss_mbb'], 2000)
        self.assertGreater(last['equity'], 0.99)
        self.assertEqual((last['seat'], last['action_idx'], last['hole'], last['board']),
                         (0, 7, ['Ah', 'Kh'], ['Qh', 'Jh', 'Th', '3s', '4d']))
        self.assertGreater(ev['per_seat_ev_loss_mbb'][0], 2000)
        self.assertTrue(all(d['ev_loss_mbb'] >= 0 for d in ev['decisions']))

    def test_real_records_fast_and_consistent(self):
        hands = S.play_match([S.RockBot(), S.ScissorsBot()], list(range(3000, 3060)))
        S._MEMO.clear()
        t = time.time()
        evs = [S.evaluate_hand(h) for h in hands]
        self.assertLess(time.time() - t, 4.0)
        n = 0
        for h, ev in zip(hands, evs):
            self.assertEqual(len(ev['decisions']), len(h['actions']))
            for d, a in zip(ev['decisions'], h['actions']):
                n += 1
                self.assertEqual(d['seat'], a['seat'])
                self.assertGreaterEqual(d['ev_loss_mbb'], 0.0)
                self.assertIn(d['solver'], ('preflop_lookup', 'dcfr_abstracted', 'dcfr_exact_river'))
                self.assertAlmostEqual(sum(d['gto_freqs'].values()), 1.0, places=6)
            for seat in (0, 1):
                self.assertAlmostEqual(ev['per_seat_ev_loss_mbb'][seat],
                                       sum(d['ev_loss_mbb'] for d in ev['decisions'] if d['seat'] == seat))
        self.assertGreater(n, 150)


class TestBots(unittest.TestCase):
    def test_every_bot_plays(self):
        for name, cls in S.SOLVER_BOTS.items():
            h = S.play_hand([cls(seed=1), CallingStation()], 11)
            self.assertEqual(sum(h['net']), 0, name)

    def test_gto_is_fast_and_noisy_gto_is_weaker(self):
        t = time.time()
        hands = S.play_match([S.GTOBot(seed=1), S.NoisyGTOBot(seed=2)], list(range(600, 652)))
        self.assertLess(time.time() - t, 2.0)
        evs = [S.evaluate_hand(h) for h in hands]
        loss = np.array([e['per_seat_ev_loss_mbb'] for e in evs]).sum(0)
        n = np.array([[sum(d['seat'] == s for d in e['decisions']) for s in (0, 1)] for e in evs]).sum(0)
        gto, noisy = loss / n
        self.assertGreater(noisy, 3 * gto)

    def test_local_best_response_takes_the_best_ev_and_exploits(self):
        from poker_arena.games import get_game
        ep = get_game('holdem')({}).episode(0, 5, ['a', 'b'])
        bot = S.LocalBestResponseBot(seed=1)
        d = ep.request(ep.to_act).data
        evs, _ = bot.evs(d)
        dec = S.run_agent(bot, '', ep.request(ep.to_act), None)
        self.assertEqual(dec.action, max(evs, key=lambda x: x[1])[0])
        A_, B_ = S.play_duplicate(lambda s: S.LocalBestResponseBot(seed=s), lambda s: CallingStation(), range(900, 960))
        self.assertGreater(sum(a['net'][0] + b['net'][1] for a, b in zip(A_, B_)), 0)

    def test_exploiter_modes_and_adaptation(self):
        def edge(mode):
            A_, B_ = S.play_duplicate(lambda s: S.ExploiterBot(seed=s, mode=mode), lambda s: CallingStation(),
                                      range(800, 880))
            return sum(a['net'][0] + b['net'][1] for a, b in zip(A_, B_))
        self.assertGreater(edge('vs_station'), edge('vs_nit'))     # the right exploit beats the wrong one
        bot = S.ExploiterBot(seed=1)
        S.play_match([bot, CallingStation()], range(900, 960))
        est = bot.estimates()
        self.assertLess(est['fold_to_bet'], 0.2)                    # it learned the station never folds
        p = bot.params()
        self.assertLess(p['bluff'], S.ExploiterBot.PRESETS['base']['bluff'])
        self.assertGreater(p['size'], S.ExploiterBot.PRESETS['base']['size'])
        with self.assertRaises(ValueError):
            S.ExploiterBot(mode='bogus')


RPS = {'rps_rock': S.RockBot, 'rps_paper': S.PaperBot, 'rps_scissors': S.ScissorsBot}
_RPS = {}


def rps_pairings():
    """Each pair of the triplet over 150 duplicate deals, raw and AIVAT."""
    if not _RPS:
        out = []
        for a, b in itertools.combinations(sorted(RPS), 2):
            X, Y = S.play_duplicate(lambda s: RPS[a](seed=s), lambda s: RPS[b](seed=s), range(150))
            out.append({'a': a, 'b': b, **{k: v for k, v in A.estimate_matchup(X, Y).items() if k != 'per_deal'}})
        _RPS['p'] = out
    return _RPS['p']


class TestAIVAT(unittest.TestCase):
    def test_chance_corrections_have_zero_mean_exactly(self):
        rng = random.Random(4)
        for _ in range(3):
            cards = rng.sample(DECK, 9)
            h0, h1, board = cards[:2], cards[2:4], cards[4:9]
            rest = [c for c in DECK if c not in cards[:7]]
            # E over turn cards of V_turn equals V before the turn; E over rivers of V_river equals V before
            e_flop = A.exact_equity(h0, h1, board[:3])
            self.assertAlmostEqual(np.mean([A.exact_equity(h0, h1, board[:3] + [c]) for c in rest]), e_flop, places=12)
            rest4 = [c for c in rest if c != board[3]]
            e_turn = A.exact_equity(h0, h1, board[:4])
            self.assertAlmostEqual(np.mean([A.exact_equity(h0, h1, board[:4] + [c]) for c in rest4]), e_turn, places=12)
            # the deal: the preflop estimate is exactly antisymmetric, so its mean over the symmetric deal is 0
            self.assertEqual(A.preflop_equity(h0, h1) + A.preflop_equity(h1, h0), 1.0)

    def test_all_in_ev(self):
        rec = {'hand': 0, 'button': 0, 'hole': [['Ah', 'Ad'], ['Kc', 'Ks']], 'board': ['Kd', '7h', '2c', '9s', '3d'],
               'actions': [{'seat': 0, 'street': 'preflop', 'action': 'raise', 'amount': 200, 'all_in': True, 'pot': 202},
                           {'seat': 1, 'street': 'preflop', 'action': 'call', 'amount': 198, 'all_in': True, 'pot': 400}],
               'won': [0, 400], 'net': [-200, 200], 'showdown': True}
        ev = A.all_in_ev(rec, sims=4000)
        self.assertTrue(ev['all_in'])
        self.assertEqual(ev['all_in_street'], 'preflop')
        self.assertAlmostEqual(ev['ev_net'][0], 0.82 * 400 - 200, delta=12)
        self.assertAlmostEqual(sum(ev['ev_net']), 0.0, places=9)
        self.assertAlmostEqual(ev['luck'][0], -200 - ev['ev_net'][0])
        av = A.aivat_hand(rec)
        self.assertAlmostEqual(sum(av['aivat_net']), 0.0, places=9)
        self.assertEqual(set(av['street_corrections']), {'deal', 'flop', 'turn', 'river'})

    def test_unbiased_and_variance_reducing_on_simulated_hands(self):
        hands = S.play_match([S.ScissorsBot(), S.ScissorsBot()], range(5000, 5400))
        vals = [A.aivat_hand(h) for h in hands]
        raw = np.array([v['raw_net'][0] for v in vals])
        adj = np.array([v['aivat_net'][0] for v in vals])
        for v in vals:
            self.assertAlmostEqual(v['aivat_net'][0] + v['aivat_net'][1], 0.0, places=9)
            self.assertAlmostEqual(v['raw_net'][0] - v['aivat_net'][0], v['luck_correction'][0], places=9)
        corr = raw - adj
        se = corr.std(ddof=1) / np.sqrt(len(corr))
        self.assertLess(abs(corr.mean()), 3 * se)                      # no bias: mean correction ~ 0
        self.assertGreater(raw.std(ddof=1) / adj.std(ddof=1), 1.2)     # less variance

    def test_duplicate_estimates(self):
        by = {(p['a'], p['b']): p for p in rps_pairings()}
        rs = by[('rps_rock', 'rps_scissors')]
        self.assertEqual(rs['deals'], 150)
        self.assertLess(rs['aivat_se'], rs['raw_se'])
        self.assertGreater(rs['variance_reduction_ratio'], 1.2)
        self.assertAlmostEqual(rs['effective_sample_multiplier'], rs['variance_reduction_ratio'] ** 2)
        for p in by.values():
            self.assertLess(p['aivat_se'], p['raw_se'])

    def test_evaluate_run_aivat(self):
        from poker_arena.games import get_game
        from poker_arena.runner import run_duplicate
        G = get_game('holdem')({'hands_per_match': 12})
        with tempfile.TemporaryDirectory() as d:
            asyncio.run(run_duplicate(G, ('station', {'bot': 'station'}), ('random', {'bot': 'random'}), list(range(12)),
                                      Path(d), log=lambda *a: None))
            rep = A.evaluate_run_aivat(d)
        self.assertEqual(rep['players'], ['random', 'station'])
        (p,) = rep['pairings']
        self.assertEqual((p['a'], p['b'], p['deals'], p['conversations']), ('random', 'station', 12, 1))
        for k in ('raw_bb100', 'raw_se', 'ev_adj_bb100', 'aivat_bb100', 'aivat_se', 'variance_reduction_ratio'):
            self.assertIsNotNone(p[k])
        json.dumps(rep)


RPS_M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)


def with_weak(k):
    strong = np.array([[0, 20, -10], [-20, 0, 30], [10, -30, 0]], float)
    n = 3 + k
    M = np.zeros((n, n))
    M[:3, :3] = strong
    for w in range(3, n):
        M[:3, w] = [50, 60, 70]
        M[w, :3] = -M[:3, w]
    return M, ['a', 'b', 'c'] + [f'weak{i}' for i in range(k)]


class TestMetaRank(unittest.TestCase):
    def test_rock_paper_scissors_nash(self):
        r = MR.nash_averaging(RPS_M, ['rock', 'paper', 'scissors'])
        for p in ('rock', 'paper', 'scissors'):
            self.assertAlmostEqual(r['nash_weights'][p], 1 / 3, places=9)
            self.assertAlmostEqual(r['nash_scores'][p], 0.0, places=9)
        self.assertLess(r['residual'], 1e-7)

    def test_clone_invariance(self):
        base = MR.nash_averaging(*with_weak(1))
        cloned = MR.nash_averaging(*with_weak(6))                     # five more copies of the weak player
        for p in 'abc':
            self.assertAlmostEqual(cloned['nash_weights'][p], base['nash_weights'][p], places=9)
            self.assertAlmostEqual(cloned['nash_scores'][p], base['nash_scores'][p], places=9)
        self.assertEqual(cloned['ranking'][:3], base['ranking'][:3])
        self.assertTrue(all(cloned['nash_weights'][f'weak{i}'] == 0 for i in range(6)))
        self.assertLess(cloned['residual'], 1e-7)
        # cloning a strong player splits its weight evenly (maximum entropy) and changes no score
        idx = [0, 1, 2, 0]
        M = RPS_M[np.ix_(idx, idx)]
        r = MR.nash_averaging(M, ['rock', 'paper', 'scissors', 'rock2'])
        self.assertAlmostEqual(r['nash_weights']['rock'], 1 / 6, places=7)
        self.assertAlmostEqual(r['nash_weights']['rock2'], 1 / 6, places=7)
        self.assertAlmostEqual(r['nash_weights']['paper'], 1 / 3, places=7)
        self.assertTrue(all(abs(v) < 1e-9 for v in r['nash_scores'].values()))

    def test_alpha_rank_stationary_distribution(self):
        for M, names in ((RPS_M * 50, ['r', 'p', 's']), with_weak(3)):
            r = MR.alpha_rank(M, names, alpha=0.05, m=50)
            pi = np.array([r['stationary'][p] for p in names])
            T = np.array(r['transition'])
            self.assertTrue((pi >= 0).all())
            self.assertAlmostEqual(pi.sum(), 1.0, places=12)
            np.testing.assert_allclose(pi @ T, pi, atol=1e-10)
            np.testing.assert_allclose(T.sum(1), 1.0, atol=1e-12)
        r = MR.alpha_rank(*with_weak(3))
        self.assertTrue(all(r['stationary'][f'weak{i}'] < 1e-6 for i in range(3)))
        r = MR.alpha_rank(RPS_M * 50, ['r', 'p', 's'])
        self.assertAlmostEqual(r['stationary']['r'], 1 / 3, places=9)

    def test_payoff_matrix_and_cycles(self):
        names, M, counts = MR.build_payoff_matrix([{'a': 'x', 'b': 'y', 'aivat_bb100': 30, 'deals': 10},
                                                  {'a': 'y', 'b': 'x', 'raw_bb100': 10, 'deals': 30},
                                                  {'a': 'y', 'b': 'z', 'edge_bb100': 5}])
        self.assertEqual(names, ['x', 'y', 'z'])
        np.testing.assert_allclose(M, -M.T)
        self.assertAlmostEqual(M[0, 1], (10 * 30 - 30 * 10) / 40)
        self.assertEqual(counts[0, 1], 40)
        self.assertEqual(counts[0, 2], 0)
        self.assertEqual(MR.detect_cycles(RPS_M, ['rock', 'paper', 'scissors']), [['paper', 'rock', 'scissors']])
        self.assertEqual(MR.detect_cycles(*with_weak(2)), [['a', 'b', 'c']])
        self.assertEqual(MR.detect_cycles(np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]), ['p', 'q', 'r']), [])

    def test_bot_triplet_is_a_cycle(self):
        pairs = rps_pairings()
        by = {(p['a'], p['b']): p for p in pairs}
        # rock > scissors, scissors > paper, paper > rock, raw and AIVAT (a is the alphabetically first)
        for key_ in ('raw_bb100', 'aivat_bb100'):
            self.assertGreater(by[('rps_paper', 'rps_rock')][key_], 0)
            self.assertGreater(by[('rps_rock', 'rps_scissors')][key_], 0)
            self.assertLess(by[('rps_paper', 'rps_scissors')][key_], 0)
        self.assertGreater(by[('rps_paper', 'rps_rock')]['aivat_bb100'] / by[('rps_paper', 'rps_rock')]['aivat_se'], 2)
        self.assertLess(by[('rps_paper', 'rps_scissors')]['aivat_bb100'] / by[('rps_paper', 'rps_scissors')]['aivat_se'], -2)
        self.assertGreater(by[('rps_rock', 'rps_scissors')]['aivat_bb100'] / by[('rps_rock', 'rps_scissors')]['aivat_se'], 2)
        rep = MR.meta_rankings(pairs)
        self.assertEqual(rep['cycles'], [['rps_paper', 'rps_rock', 'rps_scissors']])
        self.assertTrue(rep['non_transitive'])
        self.assertTrue(rep['complete'])
        self.assertEqual(sorted(rep['nash']['support']), sorted(RPS))
        self.assertLess(rep['nash']['residual'], 1e-7)
        json.dumps(rep)


if __name__ == '__main__':
    unittest.main()
