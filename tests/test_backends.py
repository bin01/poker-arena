import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from poker_arena.backends import CODEX_NO_TOOLS, Backend, BackendError, Reply, make_backend

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


# A stand-in `codex exec --json`: logs its argv, runs a "shell command" when the prompt asks for one.
FAKE_CODEX = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['ARGV_LOG'], 'a') as f:
    f.write(json.dumps(args) + '\n')
prompt = sys.stdin.read()
emit = lambda **e: print(json.dumps(e))
emit(type='thread.started', thread_id='th-1')
emit(type='item.completed', item={'id': 'item_0', 'type': 'error', 'message': 'config warning'})
emit(type='turn.started')
if 'TOOL' in prompt:
    emit(type='item.started', item={'id': 'item_1', 'type': 'command_execution', 'command': 'ls -la'})
    emit(type='item.completed', item={'id': 'item_1', 'type': 'command_execution', 'command': 'ls -la'})
    emit(type='item.completed', item={'id': 'item_2', 'type': 'web_search', 'query': 'gto charts'})
emit(type='item.completed', item={'id': 'item_3', 'type': 'reasoning', 'text': 'think'})
emit(type='item.completed', item={'id': 'item_4', 'type': 'agent_message', 'text': '{"action": "check"}'})
emit(type='turn.completed', usage={'input_tokens': 3})
'''

FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, sys
sys.stdin.read()
print(json.dumps({'type': 'system', 'session_id': 'c-1'}))
print(json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'thinking', 'thinking': 'hmm'},
      {'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'ls'}}]}}))
print(json.dumps({'type': 'result', 'result': '{"action": "fold"}', 'session_id': 'c-1', 'usage': {}}))
'''


class OnPath(unittest.TestCase):
    def install(self, name, script):
        d = tempfile.mkdtemp()
        exe = Path(d) / name
        exe.write_text(script)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        old = os.environ['PATH']
        os.environ['PATH'] = d + os.pathsep + old
        self.addCleanup(os.environ.__setitem__, 'PATH', old)
        return Path(d)


class TestToolUse(OnPath):
    def test_codex_flags_and_tool_items(self):
        d = self.install('codex', FAKE_CODEX)
        os.environ['ARGV_LOG'] = str(d / 'argv.jsonl')
        self.addCleanup(os.environ.pop, 'ARGV_LOG', None)
        be = make_backend('codex', 'SYS', model='gpt-x', retries=0)
        r1 = asyncio.run(be.send('a plain turn'))
        r2 = asyncio.run(be.send('TOOL please'))
        argv = [a for a in map(json.loads, (d / 'argv.jsonl').read_text().splitlines()) if a != ['--version']]
        for a in argv:
            joined = ' '.join(a)
            self.assertIn(' '.join(CODEX_NO_TOOLS), joined)
            self.assertIn('--ignore-user-config', a)
            self.assertIn('--ignore-rules', a)
        self.assertEqual(argv[0][:4], ['exec', '--sandbox', 'read-only', '--json'])
        self.assertEqual(argv[1][:5], ['exec', 'resume', 'th-1', '-c', 'sandbox_mode="read-only"'])
        self.assertNotIn('tool_use', r1.meta)
        self.assertEqual(r2.meta['tool_use'], [{'type': 'command_execution', 'name': 'ls -la'},
                                               {'type': 'web_search', 'name': 'gto charts'}])
        self.assertEqual((r2.meta['session_before'], r2.meta['session_after']), ('th-1', 'th-1'))
        p = be.protocol()
        self.assertTrue(p['isolate'])
        self.assertIn('--disable shell_tool', p['tool_policy']['restriction'])
        self.assertNotIn('timeout', p)

    def test_claude_tool_use_blocks(self):
        self.install('claude', FAKE_CLAUDE)
        r = asyncio.run(make_backend('claude', 'SYS', retries=0).send('hi'))
        self.assertEqual(r.meta['tool_use'], [{'type': 'tool_use', 'name': 'Bash', 'input': '{"command": "ls"}'}])
        self.assertEqual(r.thinking, 'hmm')

    def test_agy_tool_calls_from_this_turn_only(self):
        brain = Path(tempfile.mkdtemp())
        logs = brain / 'conv-1' / '.system_generated' / 'logs'
        logs.mkdir(parents=True)
        rows = [{'type': 'USER_INPUT'}, {'type': 'PLANNER_RESPONSE', 'tool_calls': [{'name': 'run_command', 'args': {}}]},
                {'type': 'USER_INPUT'}, {'type': 'PLANNER_RESPONSE', 'thinking': 't', 'model': 'g-1',
                                         'tool_calls': [{'name': 'search_web', 'args': {'query': 'x'}}]},
                {'type': 'PLANNER_RESPONSE', 'content': 'done', 'model': 'g-1'}]
        (logs / 'transcript.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        be = make_backend('agy', 'SYS')
        be.session, be.BRAIN = 'conv-1', str(brain)
        thinking, models, tools, audit = be.read_turn(be._transcript())
        self.assertEqual((thinking, models, audit), ('t', ['g-1'], 'verified'))
        self.assertEqual([t['name'] for t in tools], ['search_web'])

    def test_agy_without_a_readable_transcript_is_unverified(self):
        brain = Path(tempfile.mkdtemp())
        be = make_backend('agy', 'SYS')
        be.session, be.BRAIN = 'conv-1', str(brain)
        self.assertIsNone(be._transcript())                                         # no file
        self.assertEqual(be.read_turn(None), ('', [], [], 'unverified'))
        logs = brain / 'conv-1' / '.system_generated' / 'logs'
        logs.mkdir(parents=True)
        (logs / 'transcript.jsonl').write_text(json.dumps({'type': 'PLANNER_RESPONSE', 'model': 'g'}) + '\n')
        self.assertEqual(be.read_turn(be._transcript())[3], 'unverified')                        # no user turn: not this turn
        (logs / 'transcript.jsonl').write_text(json.dumps({'type': 'USER_INPUT'}) + '\n')
        self.assertEqual(be.read_turn(be._transcript())[3], 'unverified')                        # no response to it yet
        self.assertEqual(be.protocol()['tool_policy']['audit']['default'], 'verified')

    def test_every_reply_carries_an_audit_status(self):
        d = self.install('codex', FAKE_CODEX)
        os.environ['ARGV_LOG'] = str(d / 'argv.jsonl')
        self.addCleanup(os.environ.pop, 'ARGV_LOG', None)
        self.assertEqual(asyncio.run(make_backend('codex', 'SYS', retries=0).send('hi')).meta['audit'], 'verified')
        cmd = make_backend('command', 'SYS', command=['cat'], retries=0)
        self.assertEqual(asyncio.run(cmd.send('hi')).meta['audit'], 'opaque')
        self.assertEqual(cmd.protocol()['tool_policy']['audit']['default'], 'opaque')


FAKE_CLAUDE_INIT = r'''#!/usr/bin/env python3
import json, sys
sys.stdin.read()
print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': 'c-1', 'model': 'claude-x-1-20260101'}))
print(json.dumps({'type': 'assistant', 'message': {'model': 'claude-x-1-20260101', 'content': [{'type': 'text', 'text': 'ok'}]}}))
print(json.dumps({'type': 'result', 'result': '{"action": "fold"}', 'session_id': 'c-1', 'usage': {}}))
'''

FAKE_GEMINI = r'''#!/usr/bin/env python3
import json, os
mode = os.environ.get('GEMINI_MODE')
stats = {'models': {'gemini-x': {}}, 'tools': {'totalCalls': 0, 'byName': {}}}
if mode == 'no-tools':
    stats = {'models': {'gemini-x': {}}}
elif mode == 'two-models':
    stats = {'models': {'gemini-x': {}, 'gemini-y': {}}, 'tools': {'totalCalls': 0, 'byName': {}}}
print(json.dumps({'response': '{"action": "check"}', 'stats': stats}))
'''


class TestIdentity(OnPath):
    """Every reply says which model was asked and which (if any) the backend's own output says answered."""

    def test_claude_attests_from_its_init_event(self):
        self.install('claude', FAKE_CLAUDE_INIT)
        r = asyncio.run(make_backend('claude', 'SYS', model='claude-x-1', effort='high', retries=0).send('hi'))
        self.assertEqual({k: r.meta[k] for k in ('requested_model', 'observed_model', 'identity', 'served_model',
                                                  'requested_effort')},
                         {'requested_model': 'claude-x-1', 'observed_model': 'claude-x-1-20260101',
                          'identity': 'attested', 'served_model': 'claude-x-1-20260101', 'requested_effort': 'high'})
        self.assertEqual((r.meta['identity_reported'], r.meta['identity_match']), (True, 'match'))

    def test_identity_match_is_separate_from_reporting(self):
        from poker_arena.backends import identity_fields
        self.assertEqual(identity_fields('claude-sonnet-5-5', 'claude-sonnet-5-5-20260101'),
                         {'identity_reported': True, 'identity_match': 'match'})
        self.assertEqual(identity_fields('claude-sonnet-5-5', 'claude-haiku-5-5'),
                         {'identity_reported': True, 'identity_match': 'mismatch'})
        self.assertEqual(identity_fields(None, 'claude-haiku-5-5'),
                         {'identity_reported': True, 'identity_match': 'mismatch'})
        self.assertEqual(identity_fields('claude-sonnet-5-5', None),
                         {'identity_reported': False, 'identity_match': 'unattested'})
        be = make_backend('claude', 'SYS', model='claude-sonnet-5-5')
        meta = be.identity('claude-haiku-5-5')
        self.assertEqual((meta['identity_reported'], meta['identity_match']), (True, 'mismatch'))

    def test_a_claude_reply_that_names_no_model_is_unattested(self):
        self.install('claude', FAKE_CLAUDE)
        r = asyncio.run(make_backend('claude', 'SYS', model='claude-x-1', retries=0).send('hi'))
        self.assertEqual((r.meta['observed_model'], r.meta['identity']), (None, 'unattested'))
        self.assertEqual((r.meta['identity_reported'], r.meta['identity_match']), (False, 'unattested'))
        self.assertNotIn('served_model', r.meta)

    def test_codex_and_command_cannot_attest(self):
        d = self.install('codex', FAKE_CODEX)
        os.environ['ARGV_LOG'] = str(d / 'argv.jsonl')
        self.addCleanup(os.environ.pop, 'ARGV_LOG', None)
        r = asyncio.run(make_backend('codex', 'SYS', model='gpt-x', retries=0).send('hi'))
        self.assertEqual((r.meta['requested_model'], r.meta['observed_model'], r.meta['identity']),
                         ('gpt-x', None, 'unattested'))
        r = asyncio.run(make_backend('command', 'SYS', command=['cat'], model='m', retries=0).send('hi'))
        self.assertEqual((r.meta['observed_model'], r.meta['identity'], r.meta['audit']), (None, 'unattested', 'opaque'))

    def test_gemini_tool_stats_and_model(self):
        self.install('gemini', FAKE_GEMINI)
        self.addCleanup(os.environ.pop, 'GEMINI_MODE', None)
        send = lambda: asyncio.run(make_backend('gemini', 'SYS', model='gemini-x', retries=0).send('hi'))
        r = send()
        self.assertEqual((r.meta['audit'], r.meta['observed_model'], r.meta['identity']), ('verified', 'gemini-x', 'attested'))
        os.environ['GEMINI_MODE'] = 'no-tools'
        r = send()
        self.assertEqual(r.meta['audit'], 'unverified')                  # no tool counts: nothing is known
        self.assertNotIn('tool_use', r.meta)
        os.environ['GEMINI_MODE'] = 'two-models'
        r = send()
        self.assertEqual((r.meta['observed_model'], r.meta['observed_models'], r.meta['identity_match']),
                         (None, ['gemini-x', 'gemini-y'], 'mismatch'))      # a second model answered too


def claude_stream(init, *answers):
    lines = [{'type': 'system', 'subtype': 'init', 'session_id': 'c-1', **({'model': init} if init else {})}]
    lines += [{'type': 'assistant', 'message': {**({'model': m} if m else {}), 'content': [{'type': 'text', 'text': 'x'}]}}
              for m in answers]
    lines.append({'type': 'result', 'result': '{"action": "fold"}', 'session_id': 'c-1', 'usage': {}})
    return '\n'.join(json.dumps(x) for x in lines) + '\n'


class TestEveryAnsweringModel(OnPath):
    """R6-#2: identity comes from every assistant message of a Claude turn; the init event is only supplementary."""

    def send(self, stream, model='claude-x-1'):
        self.install('claude', f'#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nprint({stream!r}, end="")\n')
        return asyncio.run(make_backend('claude', 'SYS', model=model, retries=0).send('hi'))

    def test_the_reviewers_case_init_hides_another_answering_model(self):
        r = self.send(claude_stream('claude-x-1', 'claude-other-2'))
        self.assertEqual(r.meta['identity_match'], 'mismatch')
        self.assertEqual((r.meta['init_model'], r.meta['answered_by']), ('claude-x-1', ['claude-other-2']))

    def test_mixed_answering_models_are_a_mismatch(self):
        r = self.send(claude_stream('claude-x-1', 'claude-x-1', 'claude-other-2'))
        self.assertEqual((r.meta['identity_match'], r.meta['observed_model']), ('mismatch', None))
        self.assertEqual(r.meta['observed_models'], ['claude-other-2', 'claude-x-1'])

    def test_one_model_throughout_matches_and_dated_ids_count(self):
        r = self.send(claude_stream('claude-x-1', 'claude-x-1-20260101', 'claude-x-1-20260101'))
        self.assertEqual((r.meta['identity_match'], r.meta['observed_model']), ('match', 'claude-x-1-20260101'))

    def test_init_alone_does_not_attest(self):
        r = self.send(claude_stream('claude-x-1', None))
        self.assertEqual((r.meta['identity_match'], r.meta['init_model']), ('unattested', 'claude-x-1'))

    def test_the_raw_output_says_the_same(self):
        from poker_arena.backends import verify_call
        r = self.send(claude_stream('claude-x-1', 'claude-other-2'))
        call = {'output': r.text, **r.meta}
        self.assertEqual(verify_call('claude', call, r.meta['raw'])['status'], 'verified')
        self.assertEqual(verify_call('claude', {**call, 'identity_match': 'match'}, r.meta['raw'])['status'],
                         'inconsistent')
        self.assertEqual(verify_call('claude', {**call, 'output': 'other'}, r.meta['raw'])['status'], 'inconsistent')
        self.assertEqual(verify_call('claude', call, None)['status'], 'unverifiable')


class TestCliArtifact(OnPath):
    """R6-#6: every launch re-checks the resolved file, its version and its digest."""

    def test_a_changed_file_is_seen_on_the_next_launch(self):
        from poker_arena.backends import cli_artifact
        script = '#!/bin/sh\nif [ "$1" = --version ]; then echo "tool {v}"; exit 0; fi\ncat >/dev/null\necho ok\n'
        d = self.install('tool', script.format(v='1.0'))
        be = make_backend('command', 'SYS', command=['tool'], retries=0)
        r1 = asyncio.run(be.send('hi'))
        self.assertEqual((r1.meta['cli']['version'], r1.meta['cli']['path']), ('tool 1.0', str((d / 'tool').resolve())))
        self.assertEqual(be.cli_transitions, [])
        (d / 'tool').write_text(script.format(v='1.0') + '# patched\n')     # same version string, another file
        r2 = asyncio.run(be.send('hi'))
        self.assertNotEqual(r2.meta['cli']['sha256'], r1.meta['cli']['sha256'])
        (d / 'tool').write_text(script.format(v='2.0'))
        r3 = asyncio.run(be.send('hi'))
        self.assertEqual(r3.meta['cli']['version'], 'tool 2.0')
        self.assertEqual([t['to']['sha256'] for t in be.cli_transitions], [r2.meta['cli']['sha256'], r3.meta['cli']['sha256']])
        self.assertEqual(cli_artifact('tool')['version'], 'tool 2.0')


class TestAttempts(unittest.TestCase):
    def test_a_timeout_keeps_what_was_printed(self):
        class Slow(Backend):
            stateful = True

            async def _call(self, text):
                return Reply(await self.run(['sh', '-c', 'echo partial; echo oops >&2; sleep 30']))
        be = Slow('SYS', timeout=0.5, retries=0)
        be.session = 'old'
        be.turns = [['t1', 'r1']]
        with self.assertRaises(BackendError) as cm:
            asyncio.run(be.send('hi'))
        fa = cm.exception.meta['failed_attempts'][0]
        self.assertEqual(fa['raw']['stdout'], 'partial\n')
        self.assertEqual(fa['raw']['stderr'], 'oops\n')
        self.assertTrue(fa['raw']['timed_out'])
        self.assertTrue(fa['rebuilt'])                                         # it may have stored the turn
        self.assertEqual(fa['session_before'], 'old')
        self.assertEqual((be.session, be.rebuilding), (None, True))

    def test_a_cli_that_never_started_is_not_rebuilt(self):
        class Missing(Backend):
            stateful = True

            async def _call(self, text):
                return Reply(await self.run(['definitely-not-a-cli-xyz']))
        be = Missing('SYS', retries=0)
        be.session = 'kept'
        with self.assertRaises(BackendError) as cm:
            asyncio.run(be.send('hi'))
        self.assertNotIn('rebuilt', cm.exception.meta['failed_attempts'][0])
        self.assertEqual(be.session, 'kept')


if __name__ == '__main__':
    unittest.main()
