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
            st['hands'] = 4
            (m / 'state.json').write_text(json.dumps(st))
            lines = (m / 'hands.jsonl').read_text().splitlines()[:4]
            (m / 'hands.jsonl').write_text('\n'.join(lines) + '\n')
            cfg = json.loads((m / 'config.json').read_text())
            asyncio.run(run_match(G, [('fake', fake), ('rule', {'bot': 'rule', 'sims': 50})], cfg['seeds'], m,
                                  log=lambda *a: None))
            self.assertEqual(len((m / 'hands.jsonl').read_text().splitlines()), 6)
            events = [json.loads(l) for l in (m / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(1 for e in events if e['kind'] == 'hand_start'), 6)


if __name__ == '__main__':
    unittest.main()
