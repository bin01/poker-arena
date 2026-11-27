"""Tournaments played until decided: rounds on new deals, settled pairings stop, a cap for the rest; and the
per-pairing verdicts."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from poker_arena import server
from poker_arena.ratings import Z_STOP, load_deals, pairs, ratings, settle
from poker_arena.runner import make_plan, run_plan


class TestSettle(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(settle([100.0] * 3 + [-100.0] * 2)['status'], 'early')     # under 20 deals
        clear = settle([300.0, 100.0] * 15)                                            # +200 ± tiny
        self.assertEqual((clear['status'], clear['more_deals']), ('settled', 0))
        even = settle([100.0, -100.0] * 15)
        self.assertEqual((even['status'], even['more_deals']), ('too close', None))
        lean = settle([400.0, -340.0] * 15)                                            # +30, sd ~375
        self.assertEqual(lean['status'], 'open')
        n = 30 * (Z_STOP / lean['z']) ** 2                                             # where the edge would clear
        self.assertAlmostEqual(lean['more_deals'], n - 30, delta=0.03 * n)              # z is rounded

    def test_pairs_pool_both_orders(self):
        deals = [{'a': 'x', 'b': 'y', 'matchup': 'x__vs__y', 'deal': k, 'net_a': 4, 'bb': 2} for k in range(5)] + \
                [{'a': 'y', 'b': 'x', 'matchup': 'y__vs__x', 'deal': k, 'net_a': 4, 'bb': 2} for k in range(5)]
        [p] = pairs(deals)
        self.assertEqual((p['a'], p['deals'], p['net_a_chips'], p['a_bb_per_100']), ('x', 10, 0, 0.0))
        self.assertEqual(p['matchups'], ['x__vs__y', 'y__vs__x'])


class TestRounds(unittest.TestCase):
    def test_until_decided(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            players = root / 'players.toml'
            players.write_text('[players.rule]\nbot = "rule"\nsims = 40\n[players.random]\nbot = "random"\n'
                               '[players.station]\nbot = "station"\n[players.station2]\nbot = "station"\n')
            out = root / 'runs' / 'rounds'
            plan = make_plan([], deals=20, out=str(out), players_file=str(players),
                             matchups=[['station', 'station2'], ['rule', 'random']])
            plan['until_decided'] = 60
            asyncio.run(run_plan(plan, log=lambda *a: None))
            folders = sorted(p.name for p in out.iterdir() if p.is_dir())
            # two identical players never settle: they play every round up to the cap
            self.assertEqual([f for f in folders if f.startswith('station')],
                             ['station__vs__station2', 'station__vs__station2__r2', 'station__vs__station2__r3'])
            deals = load_deals(out)
            self.assertEqual(len([x for x in deals if x['a'] == 'station']), 60)
            r = ratings(out, boot=10)
            v = {frozenset((m['a'], m['b'])): m for m in r['matchups']}
            self.assertEqual(v[frozenset(('station', 'station2'))]['deals'], 60)       # pooled over the rounds
            self.assertEqual(v[frozenset(('station', 'station2'))]['status'], 'too close')
            rr = v[frozenset(('rule', 'random'))]
            self.assertTrue(rr['status'] == 'settled' or rr['deals'] == 60, rr)
            idx = {m['id']: m for m in server.index(root / 'runs', 'rounds')}
            self.assertEqual(idx['rounds/station__vs__station2__r3']['round'], 3)
            # rounds use new deals: round 2 starts where round 1's seeds end
            import json
            s1 = json.loads((out / 'station__vs__station2' / 'a-first' / 'config.json').read_text())['seeds']
            s2 = json.loads((out / 'station__vs__station2__r2' / 'a-first' / 'config.json').read_text())['seeds']
            self.assertEqual(s2[0], s1[-1] + 1)


if __name__ == '__main__':
    unittest.main()
