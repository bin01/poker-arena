import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from poker_arena.agents import parse_json
from poker_arena.games import get_game
from poker_arena.runner import run_duplicate, run_match

FAKE = [sys.executable, str(Path(__file__).with_name('fake_model.py'))]


class TestParse(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(parse_json('{"a": 1}'), {'a': 1})
        self.assertEqual(parse_json('```json\n{"a": {"b": "}"}}\n```'), {'a': {'b': '}'}})
        self.assertEqual(parse_json('Sure! {"action": "call"} hope that helps'), {'action': 'call'})
        self.assertIsNone(parse_json('no json here'))
        self.assertEqual(parse_json('{bad} then {"ok": true}'), {'ok': True})


class TestModelAgent(unittest.TestCase):
    def test_refusals_are_kept(self):
        from poker_arena.agents import ModelAgent
        from poker_arena.backends import BackendError, Reply
        replies = ['nope', '{"action": "raise", "amount": "Infinity"}', '{"action": "call"}']

        class Stub:
            async def send(self, text):
                if not replies:
                    raise BackendError('gone')
                return Reply(replies.pop(0))
        G = get_game('holdem')({'hands_per_match': 1})
        ep = G.episode(0, 1, ['A', 'B'])
        agent = ModelAgent('m', 'command', command=['x'])
        agent.backend = Stub()
        validate = lambda act: ep.__class__.apply(__import__('copy').deepcopy(ep), 0, act)
        d = asyncio.run(agent.act('go', ep.request(0), validate))
        self.assertEqual(d.action['action'], 'call')
        self.assertEqual([e['requested'] for e in d.errors], [None, {'action': 'raise', 'amount': 'Infinity'}])
        self.assertEqual(len(d.calls), 3)
        with self.assertRaises(BackendError) as cm:
            asyncio.run(agent.act('go', ep.request(0), validate))
        self.assertTrue(cm.exception.calls[-1]['backend_failed'])


class TestModelPipeline(unittest.TestCase):
    """A full duplicate pair with a stand-in model CLI (the `command` backend): prompts, JSON parsing,
    retries after malformed replies, table talk, notes, logs and resume."""

    def test_fake_model_pair_and_resume(self):
        G = get_game('holdem')({'hands_per_match': 6})
        fake = {'backend': 'command', 'command': FAKE}
        with tempfile.TemporaryDirectory() as d:
            asyncio.run(run_duplicate(G, ('fake', fake), ('rule', {'bot': 'rule', 'sims': 50}), list(range(6)),
                                      Path(d), log=lambda *a: None))
            m = Path(d) / 'fake__vs__rule' / 'a-first'
            calls = [json.loads(l) for l in (m / 'calls.jsonl').read_text().splitlines()]
            self.assertTrue(calls)
            self.assertTrue(all('prompt' in c and 'output' in c for c in calls))
            # the stand-in remembers earlier turns through the transcript
            self.assertIn('seen', calls[-1]['output'])
            events = [json.loads(l) for l in (m / 'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['kind'] == 'thought' and e['data']['reasoning'] for e in events))
            result = json.loads((m / 'result.json').read_text())
            self.assertEqual(sum(result['net']), 0)

            # resume: drop the result and the last two hands, rerun, get the same number of hands
            (m / 'result.json').unlink()
            st = json.loads((m / 'state.json').read_text())
            st['hands'] = st['hands_committed'] = 4
            (m / 'state.json').write_text(json.dumps(st))
            lines = (m / 'hands.jsonl').read_text().splitlines()[:4]
            (m / 'hands.jsonl').write_text('\n'.join(lines) + '\n')
            cfg = json.loads((m / 'config.json').read_text())
            asyncio.run(run_match(G, [('fake', fake), ('rule', {'bot': 'rule', 'sims': 50})], cfg['seeds'], m,
                                  log=lambda *a: None))
            self.assertEqual(len((m / 'hands.jsonl').read_text().splitlines()), 6)
            events = [json.loads(l) for l in (m / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(1 for e in events if e['kind'] == 'hand_start'), 6)


class TestReferenceBot(unittest.TestCase):
    def test_versioned_deterministic_and_it_beats_the_baselines(self):
        from poker_arena.agents import make_agent
        ref = make_agent('reference', {'bot': 'reference'}, seed=3)
        self.assertEqual(ref.describe()['version'], 'reference-1')
        G = get_game('holdem')({'hands_per_match': 80})
        seeds = list(range(1000, 1080))
        with tempfile.TemporaryDirectory() as d:
            nets = {}
            for opp in ('rule', 'station', 'random'):
                res = asyncio.run(run_duplicate(G, ('reference', {'bot': 'reference'}), (opp, {'bot': opp}),
                                                seeds, Path(d) / opp, log=lambda *a: None))
                nets[opp] = res[0]['net'][0] + res[1]['net'][1]
            again = asyncio.run(run_duplicate(G, ('reference', {'bot': 'reference'}), ('rule', {'bot': 'rule'}),
                                              seeds, Path(d) / 'again', log=lambda *a: None))
            self.assertEqual(again[0]['net'][0] + again[1]['net'][1], nets['rule'])
            first = lambda p: [json.loads(l)['actions'] for l in (p / 'reference__vs__rule' / 'a-first' / 'hands.jsonl')
                               .read_text().splitlines()]
            self.assertEqual(first(Path(d) / 'rule'), first(Path(d) / 'again'))
        for opp, net in nets.items():
            self.assertGreater(net, 0, opp)

    def test_it_reads_position_and_price(self):
        from poker_arena.agents import make_agent
        G = get_game('holdem')({'hands_per_match': 2})
        bot = make_agent('reference', {'bot': 'reference', 'sims': 300})

        def decide(hole, button_seat, script=()):
            ep = G.episode(button_seat, 1, ['A', 'B'])
            ep.hole[0] = hole
            for seat, act in script:
                self.assertIsNone(ep.apply(seat, act))
            self.assertEqual(ep.to_act, 0)
            return asyncio.run(bot.act('', ep.request(0), None)).action['action']
        self.assertEqual(decide(['As', 'Ah'], 0), 'raise')                      # opens aces on the button
        self.assertEqual(decide(['7d', '2c'], 0), 'fold')                       # folds the worst hands there
        self.assertEqual(decide(['7d', '2c'], 1, [(1, {'action': 'call'})]), 'check')   # free look in the big blind
        self.assertEqual(decide(['As', 'Ah'], 1, [(1, {'action': 'all_in'})]), 'call')
        self.assertEqual(decide(['7d', '2c'], 1, [(1, {'action': 'all_in'})]), 'fold')


if __name__ == '__main__':
    unittest.main()
