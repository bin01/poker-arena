"""Expected-value scoring: exact equities, which hands get adjusted, and that it is unbiased."""
import random
import unittest

from poker_arena.luck import closed_at, expected_net, heads_up_equity


def hand(actions, hole=(('As', 'Ad'), ('Kc', 'Kd')), board=('2h', '7s', '9c', '3d', '4h'), net=(200, -200),
         won=(400, 0), showdown=True):
    return {'hand': 0, 'hole': [list(h) for h in hole], 'board': list(board), 'net': list(net), 'won': list(won),
            'showdown': showdown, 'actions': actions}


ALL_IN_PRE = [{'seat': 0, 'street': 'preflop', 'action': 'raise'}, {'seat': 1, 'street': 'preflop', 'action': 'call'}]


class TestLuck(unittest.TestCase):
    def test_exact_equity(self):
        # a made ace-high flush on the turn against two queens: no river card saves the queens
        self.assertEqual(heads_up_equity(['Ah', 'Kh'], ['Qs', 'Qc'], ['2h', '7h', '9h', '3c']), 1.0)
        # two outs on the river: 2 of 44 cards win for the underdog
        self.assertAlmostEqual(heads_up_equity(['Qs', 'Qc'], ['As', 'Ac'], ['2h', '7d', '9c', '3s']), 2 / 44)

    def test_aces_against_kings_preflop(self):
        self.assertAlmostEqual(heads_up_equity(['As', 'Ad'], ['Kc', 'Kd'], [], sims=20000), 0.82, delta=0.015)

    def test_which_hands_are_adjusted(self):
        self.assertEqual(closed_at(hand(ALL_IN_PRE)), 0)
        self.assertIsNone(closed_at(hand(ALL_IN_PRE, showdown=False)))                     # someone folded
        self.assertIsNone(closed_at(hand(ALL_IN_PRE + [{'seat': 1, 'street': 'river', 'action': 'check'},
                                                       {'seat': 0, 'street': 'river', 'action': 'check'}])))

    def test_expected_net(self):
        ev = expected_net(hand(ALL_IN_PRE))
        self.assertAlmostEqual(ev[0], 0.82 * 400 - 200, delta=6)
        self.assertAlmostEqual(ev[0] + ev[1], 0)                                            # chips are conserved

    def test_unbiased(self):
        """Averaged over runouts, the real result equals the expected value."""
        rng = random.Random(1)
        a, b, flop = ['Js', 'Ts'], ['Ah', 'Kd'], ['9s', '8d', '2s']
        rest = [r + s for r in '23456789TJQKA' for s in 'shdc' if r + s not in a + b + flop]
        from poker_arena.cards import evaluate
        real = []
        for _ in range(4000):
            run = rng.sample(rest, 2)
            x, y = evaluate(a + flop + run), evaluate(b + flop + run)
            real.append(100 if x > y else -100 if x < y else 0)
        eq = heads_up_equity(a, b, flop)
        self.assertAlmostEqual(sum(real) / len(real), eq * 200 - 100, delta=4)

    def test_measure_on_played_games(self):
        """On real duplicate deals, the best mix is never noisier than either pure score."""
        import asyncio, tempfile
        from pathlib import Path
        from poker_arena.luck import measure
        from poker_arena.runner import make_plan, run_plan
        with tempfile.TemporaryDirectory() as d:
            players = Path(d) / 'p.toml'
            players.write_text('[players.rule]\nbot = "rule"\nsims = 40\n[players.random]\nbot = "random"\n'
                               '[players.station]\nbot = "station"\n')
            plan = make_plan([], deals=30, out=str(Path(d) / 'runs'), players_file=str(players),
                             matchups=[['rule', 'random'], ['random', 'station']])
            asyncio.run(run_plan(plan, log=lambda *a: None))
            m = measure(Path(d) / 'runs', sims=2000)
        self.assertEqual(m['deals'], 60)
        self.assertGreater(m['hands_adjusted'], 0)          # a random bot goes all in before the river
        self.assertLessEqual(m['sd_best'], min(m['sd_chips'], m['sd_ev']) + 1e-9)


if __name__ == '__main__':
    unittest.main()
