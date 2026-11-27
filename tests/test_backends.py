import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from poker_arena.backends import make_backend

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


if __name__ == '__main__':
    unittest.main()
