"""Round 10 review, material issues: the official track is one deal per conversation, enforced by `plan` and by
the registration (#2); unmatched chips go back explicitly, so the hand records the contested pot and the
returned amount, and older records still verify (#3); session Elo says what a session is in this run and sizes
its tie band from it (#4)."""
import asyncio
import contextlib
import copy
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poker_arena import cli
from poker_arena import ratings as R
from poker_arena.games import get_game
from poker_arena.games.holdem import HoldemHand, verify_hand
from poker_arena.runner import make_tournament, run_plan
from test_ratings import deals_between, plan, run
from test_tracks import PLAYERS

VIEWER = Path(__file__).resolve().parent.parent / 'poker_arena' / 'viewer'


def node(body):
    script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))});'
              f'const PokerTable = require({json.dumps(str(VIEWER / "table.js"))});'
              f'const out = (() => {{ {body} }})(); process.stdout.write(JSON.stringify(out));')
    res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def longer(p, deals):
    """An official-track plan edited to play `deals` deals per conversation."""
    return {**p, 'deals': deals, 'design': {**p['design'], 'round_deals': deals}}


class TestOfficialIsOneDeal(unittest.TestCase):
    """#2: the official track promises a fresh conversation every deal."""

    def test_plan_refuses_more_deals_per_conversation(self):
        with self.assertRaises(ValueError) as e:
            make_tournament(['a', 'b'], out='runs/cup', official={'deals': 3})
        self.assertIn('one deal per conversation', str(e.exception))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as x:
            cli.main(['plan', 'a', 'b', '--deals', '3', '--players', '/none'])
        self.assertIn('one deal per conversation', str(x.exception.code))

    def test_adaptation_keeps_long_conversations(self):
        doc = make_tournament(['a', 'b'], out='runs/cup')
        self.assertEqual((doc['tracks']['official']['deals'], doc['tracks']['adaptation']['deals']), (1, R.ADAPTATION_DEALS))
        self.assertEqual(R.contract_problems(R.normalize_design(doc['tracks']['adaptation'])), [])
        self.assertEqual(R.contract_problems(R.normalize_design(doc['tracks']['official'])), [])

    def test_an_edited_official_plan_is_not_official(self):
        doc = make_tournament(['a', 'b'], out='runs/cup', official={'replicates': 2})
        p = longer(doc['tracks']['official'], 4)
        d = R.normalize_design(p)
        self.assertTrue(R.contract_problems(d))
        reg = R.registration_status({**p, 'registration': R.registration(d)})
        self.assertEqual((reg['registered'], reg['status']), (False, 'invalid'))
        self.assertIn('one deal per conversation', reg['reasons'][0])

    def test_the_runner_does_not_register_it(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t).resolve()
            (root / 'players.toml').write_text(PLAYERS)
            doc = make_tournament(['rule', 'station'], out=str(root / 'cup'), players_file=str(root / 'players.toml'),
                                  tracks=['official'], official={'replicates': 1, 'design': {'max_rounds': 1}})
            p = longer(doc['tracks']['official'], 2)
            asyncio.run(run_plan(p, log=lambda *a: None))
            stored = json.loads((root / 'cup' / 'official' / 'plan.json').read_text())
            self.assertNotIn('registration', stored)
            r = R.ratings(root / 'cup' / 'official', boot=20)
            self.assertFalse(r['registration']['registered'])
            self.assertIn('one deal per conversation', '; '.join(r['registration']['reasons']))
            self.assertFalse(any(m.get('official') for m in r['pairings']))


def shove_fold(seed=7):
    ep = HoldemHand(get_game('holdem')({}), 0, seed, ['a', 'b'])
    ep.apply(ep.to_act, {'action': 'raise', 'amount': 200})
    ep.apply(ep.to_act, {'action': 'fold'})
    return ep


def records(ep):
    hand = json.loads(json.dumps({**ep.summary(), 'seed': 7}, default=str))
    events = [json.loads(json.dumps(e.to_json(), default=str)) for e in ep.events]
    return hand, events


def old_format(hand, events):
    """How the same hand was written before uncalled chips were returned explicitly."""
    h, evs = copy.deepcopy(hand), copy.deepcopy(events)
    back = h.pop('uncalled_return')
    h['pot'] += back['amount']
    h['won'][back['seat']] += back['amount']
    evs = [e for e in evs if e['kind'] != 'return']
    res = next(e for e in evs if e['kind'] == 'result')['data']
    res.pop('uncalled_return')
    res['pot'], res['won'] = h['pot'], list(h['won'])
    return h, evs


class TestUncalledReturn(unittest.TestCase):
    """#3: a 200-chip shove then fold contests 4 chips and returns 198."""

    def setUp(self):
        self.G = get_game('holdem')({})
        self.hand, self.events = records(shove_fold())

    def test_the_engine_returns_it(self):
        ret = [e for e in self.events if e['kind'] == 'return']
        self.assertEqual([e['data'] for e in ret], [{'seat': 0, 'amount': 198}])
        self.assertEqual((self.hand['pot'], self.hand['uncalled_return'], self.hand['won'], self.hand['net']),
                         (4, {'seat': 0, 'amount': 198}, [4, 0], [2, -2]))
        self.assertIn('back the uncalled 198', shove_fold().render(shove_fold().events[-2], 1))

    def test_no_return_when_everything_is_matched(self):
        ep = HoldemHand(self.G, 0, 7, ['a', 'b'])
        ep.apply(ep.to_act, {'action': 'raise', 'amount': 200})
        ep.apply(ep.to_act, {'action': 'call'})
        self.assertIsNone(ep.summary()['uncalled_return'])
        self.assertFalse(any(e.kind == 'return' for e in ep.events))
        self.assertEqual(ep.summary()['pot'], 400)

    def test_new_records_verify_and_a_tampered_return_fails(self):
        self.assertEqual(verify_hand(self.G, 7, 0, self.hand, self.events)['errors'], [])
        h = copy.deepcopy(self.hand)
        h['uncalled_return']['amount'] = 98
        self.assertFalse(verify_hand(self.G, 7, 0, h, self.events)['ok'])
        evs = copy.deepcopy(self.events)
        next(e for e in evs if e['kind'] == 'return')['data']['amount'] = 98
        self.assertFalse(verify_hand(self.G, 7, 0, self.hand, evs)['ok'])
        dropped = [e for e in self.events if e['kind'] != 'return']
        self.assertFalse(verify_hand(self.G, 7, 0, self.hand, dropped)['ok'])

    def test_old_records_still_verify(self):
        h, evs = old_format(self.hand, self.events)
        self.assertEqual(verify_hand(self.G, 7, 0, h, evs)['errors'], [])
        h['pot'] += 1
        self.assertFalse(verify_hand(self.G, 7, 0, h, evs)['ok'])

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_the_replay_shows_the_contested_pot_and_the_return(self):
        h, old = old_format(self.hand, self.events)
        for evs in (self.events, old):
            r = node(f'const t = PokerTable.steps({json.dumps(evs)}); const e = t.log[t.log.length - 1];'
                     f'const s = t.steps[e.step]; return {{pot: e.pot, returned: e.returned, final: s.finalPot, '
                     f'winners: e.winners, stacks: s.seats.map(x => x.stack)}};')
            self.assertEqual(r, {'pot': 4, 'returned': {'seat': 0, 'amount': 198}, 'final': 4, 'winners': [0],
                                 'stacks': [202, 198]})


class TestSessionDefinition(unittest.TestCase):
    """#4: session Elo's description, length and tie band are the run's own."""

    def test_one_deal_sessions(self):
        rng = np.random.default_rng(1)
        r = run(deals_between('a', 'b', 40, 1, rng, sd=0, reps=30), plan(['a', 'b'], 1, 1, reps=30), boot=20,
                registered=False)
        s = r['session']
        self.assertEqual((r['session_deals'], s['deals'], s['tie_bb']), (1, 1, 0.1))
        self.assertIn('one duplicate deal', s['text'])
        self.assertTrue(any('one duplicate deal' in n for n in r['notes']))
        self.assertFalse(any('10 deals of one conversation' in n for n in r['notes']))

    def test_ten_deal_sessions(self):
        rng = np.random.default_rng(1)
        r = run(deals_between('a', 'b', 40, 30, rng, sd=0), plan(['a', 'b'], 30, 1), boot=20, registered=False)
        self.assertEqual((r['session_deals'], r['session']['tie_bb']), (10, 1.0))
        self.assertIn('10 deals of one conversation', r['session']['text'])

    def test_the_tie_band_scales_with_the_session(self):
        self.assertEqual(R.outcome({'net_bb': 0.5, 'deals': 1}), 1.0)
        self.assertEqual(R.outcome({'net_bb': 0.5, 'deals': 10}), 0.5)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_the_viewer_uses_the_runs_definition(self):
        src = (VIEWER / 'app.js').read_text()
        self.assertNotIn('10-deal session', src)
        self.assertIn('sessionText(rt)', src)
