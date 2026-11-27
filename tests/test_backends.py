import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from poker_arena.backends import Backend, Reply, make_backend

# A stand-in `agy` that checks the argument layout and answers like `agy --output-format json`.
FAKE_AGY = r'''#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
assert args[-2] == '-p', args            # flags must come before -p
assert '--sandbox' in args and args[args.index('--output-format') + 1] == 'json'
assert args[args.index('--agent') + 1] == 'poker' and args[args.index('--model') + 1] == 'gemini-x'
import os
agent = open('.agents/agents/poker/agent.md').read()   # the system prompt lives in the seat's folder
assert 'tools: []' in agent and 'SYSTEM PROMPT' in agent
assert 'SYSTEM PROMPT' not in args[-1]                  # and not in the message
conv = args[args.index('--conversation') + 1] if '--conversation' in args else 'conv-1'
print(json.dumps({'conversation_id': conv, 'status': 'SUCCESS', 'response': json.dumps({'echo': args[-1][-20:],
                  'resumed': '--conversation' in args}), 'usage': {'input_tokens': 1}}))
'''


class TestAgy(unittest.TestCase):
    def test_args_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / 'agy'
            exe.write_text(FAKE_AGY)
            exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
            old = os.environ['PATH']
            os.environ['PATH'] = d + os.pathsep + old
            try:
                be = make_backend('agy', 'SYSTEM PROMPT', model='gemini-x', label='t', retries=0)
                r1 = asyncio.run(be.send('first turn'))
                r2 = asyncio.run(be.send('second turn'))
            finally:
                os.environ['PATH'] = old
        self.assertFalse(json.loads(r1.text)['resumed'])
        self.assertTrue(json.loads(r2.text)['resumed'])
        self.assertEqual(be.session, 'conv-1')
        self.assertEqual(r1.usage, {'input_tokens': 1})


class Echo(Backend):
    stateful = True

    async def _call(self, text):
        out = await self.run(['echo', self.system, text])
        self.session = 'sess'
        return Reply(out.strip())


class TestAuditAndRebuild(unittest.TestCase):
    def test_argv_raw_and_describe(self):
        be = Echo('SYSTEM ' * 100, timeout=5, retries=0, key='secret')
        r = asyncio.run(be.send('hi'))
        self.assertEqual(r.meta['argv'], ['echo', '<system prompt>', 'hi'])
        self.assertEqual(r.meta['raw']['returncode'], 0)
        d = be.describe()
        self.assertEqual((d['timeout'], d['retries'], d['stateful']), (5, 0, True))
        self.assertFalse({'system', 'turns', 'session', 'key', 'workdir'} & set(d))

    def test_rebuild_replays_committed_turns(self):
        be = Echo('SYS', retries=0)
        asyncio.run(be.send('turn one'))
        st = be.state()
        fresh = Echo('SYS', retries=0)
        fresh.restore(st)
        self.assertEqual((fresh.session, fresh.rebuilding), ('sess', False))   # same folder: resume as is
        gone = dict(st, workdir='/nonexistent/poker-arena')
        lost = Echo('SYS', retries=0)
        lost.restore(gone)
        self.assertEqual((lost.session, lost.rebuilding), (None, True))        # folder gone: rebuild
        r = asyncio.run(lost.send('turn two'))
        self.assertIn('turn one', r.text)
        self.assertIn('interrupted', r.text)
        self.assertEqual(r.meta['rebuilt_from_turns'], 1)
        self.assertFalse(lost.rebuilding)
        self.assertEqual(lost.turns[-1][0], 'turn two')                       # the record keeps the real turn


if __name__ == '__main__':
    unittest.main()
