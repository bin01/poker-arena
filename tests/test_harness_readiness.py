"""Readiness fixes for a 9-model run: the rate-limit detector and shared gate, agy transcript binding, codex rollout
client echo, the upgraded doctor, staged plans (--vs / --matchups), shared per-CLI semaphores, and sibling
cancellation in a duplicate pair."""
import argparse
import asyncio
import contextlib
import io
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from poker_arena import backends, cli
from poker_arena.backends import (RATE_LIMIT, AgyCLI, Backend, BackendError, Reply, limit_text, make_backend,
                                  rate_gate, reset_rate_gates, retry_wait)
from poker_arena.games import get_game
from poker_arena.runner import make_tournament, run_duplicate, run_plan

JSON_NUMBER = '{"type":"turn.completed","usage":{"input_tokens": 14290, "output_tokens": 12}}'


class TestRateLimitDetection(unittest.TestCase):
    def test_regex_ignores_numbers_and_ids_that_contain_429(self):
        self.assertIsNone(RATE_LIMIT.search('"input_tokens": 14290'))
        self.assertIsNone(RATE_LIMIT.search('session abc429xyz'))
        self.assertIsNone(RATE_LIMIT.search('thread 0429-4429'))
        for text in ('HTTP 429 Too Many Requests', 'HTTP 429: boom', "You've hit your usage limit", 'rate limit exceeded',
                     'RESOURCE_EXHAUSTED', 'please try again in 5 minutes'):
            self.assertTrue(RATE_LIMIT.search(text), text)

    def test_only_stderr_and_error_messages_are_read_never_stdout_json(self):
        raw = {'returncode': 1, 'stderr': 'boom', 'stdout': JSON_NUMBER}
        self.assertFalse(RATE_LIMIT.search(limit_text(raw, f'codex exited 1: {JSON_NUMBER}')))
        failed = json.dumps({'type': 'turn.failed', 'error': {'message': 'boom'},
                             'usage': {'input_tokens': 429, 'cached': 'usage limit in a log'}})
        self.assertFalse(RATE_LIMIT.search(limit_text({'stderr': '', 'stdout': failed})))
        # a number in a line that no longer parses is not a status either
        self.assertFalse(RATE_LIMIT.search(limit_text({'stderr': '', 'stdout': '{"input_tokens": 429, "tru'})))
        self.assertTrue(RATE_LIMIT.search(limit_text({'stderr': 'HTTP 429 Too Many Requests', 'stdout': ''})))
        said = json.dumps({'type': 'turn.failed', 'error': {'message': "You've hit your usage limit. Try again in 7 minutes."}})
        self.assertTrue(RATE_LIMIT.search(limit_text({'stderr': '', 'stdout': JSON_NUMBER + '\n' + said})))
        claude = json.dumps({'type': 'result', 'is_error': True, 'result': 'Claude AI usage limit reached|1800000000'})
        self.assertTrue(RATE_LIMIT.search(limit_text({'stderr': '', 'stdout': claude})))
        agy = json.dumps({'status': 'ERROR', 'error': {'message': 'quota exhausted', 'code': 429}}, indent=2)
        self.assertTrue(RATE_LIMIT.search(limit_text({'stderr': '', 'stdout': agy})))

    def test_a_failure_without_cli_output_is_read_from_its_message(self):
        self.assertTrue(RATE_LIMIT.search(limit_text({}, 'HTTP 429: rate limit')))
        self.assertFalse(RATE_LIMIT.search(limit_text({}, 'HTTP 500: {"input_tokens": 429}')))

    def test_retry_wait_reads_the_time_a_message_names(self):
        now = time.mktime((2026, 11, 27, 10, 0, 0, 0, 0, -1))
        self.assertEqual(retry_wait('Try again in 7 minutes.'), 420)
        self.assertEqual(retry_wait('try again in 1h 30m'), 5400)
        self.assertEqual(retry_wait('retry after 2 hours and 5 minutes'), 7500)
        self.assertEqual(retry_wait('try again in 20s'), 20)
        self.assertEqual(retry_wait('try again at 4:02 PM.', now), 6 * 3600 + 120)
        self.assertEqual(retry_wait('try again at 9:30 AM.', now), 23 * 3600 + 1800)   # already past: tomorrow
        self.assertEqual(retry_wait('try again at Nov 28th, 2026 3:45 PM.', now), 29 * 3600 + 45 * 60)
        self.assertEqual(retry_wait('usage limit reached|1800000000', 1799999000), 1000)
        self.assertIsNone(retry_wait('try again later'))
        self.assertIsNone(retry_wait('HTTP 429'))


class Flaky(Backend):
    exe = 'flaky-cli'
    script: list = []
    calls: list = []

    async def _call(self, text):
        Flaky.calls.append(self.label)
        step = Flaky.script.pop(0) if Flaky.script else None
        if step is not None:
            self.last_call['raw'] = step.get('raw')
            raise BackendError(step['error'])
        return Reply('ok')


class TestSharedRateGate(unittest.TestCase):
    def setUp(self):
        reset_rate_gates()
        self.addCleanup(reset_rate_gates)
        Flaky.script, Flaky.calls = [], []
        self.sleeps = []

        async def sleep(s):
            self.sleeps.append(s)
        patch = mock.patch.object(backends.asyncio, 'sleep', sleep)
        patch.start()
        self.addCleanup(patch.stop)

    def test_one_seat_hitting_a_limit_pauses_its_siblings(self):
        Flaky.script = [{'error': 'flaky-cli exited 1: usage limit reached, try again in 30 seconds'}]
        a, b = Flaky('s', label='a', retries=0), Flaky('s', label='b', retries=0)
        reply = asyncio.run(a.send('x'))
        self.assertEqual(len(self.sleeps), 1)
        self.assertAlmostEqual(self.sleeps[0], 32, delta=1)         # the hinted wait (+2 s), not the 900 s default
        self.assertEqual(reply.meta['failed_attempts'][0]['retry_hint'], 30)
        self.assertTrue(reply.meta['failed_attempts'][0]['rate_limited'])
        self.assertGreater(rate_gate('flaky-cli').until, time.monotonic() + 25)
        self.sleeps.clear()
        asyncio.run(b.send('y'))                                    # b never hit the limit, and still waits at the gate
        self.assertTrue(self.sleeps and 25 < self.sleeps[0] <= 32, self.sleeps)
        self.assertEqual(Flaky.calls, ['a', 'a', 'b'])

    def test_gates_are_per_backend_kind(self):
        Flaky.script = [{'error': 'flaky-cli exited 1: rate limit'}]
        asyncio.run(Flaky('s', label='a', retries=0, rate_limit_wait=40).send('x'))
        self.assertEqual(len(self.sleeps), 1)
        self.assertAlmostEqual(self.sleeps[0], 40, delta=1)
        self.assertEqual(rate_gate('other-cli').until, 0.0)

    def test_numbers_in_stdout_json_are_not_a_rate_limit(self):
        raw = {'returncode': 1, 'stderr': '', 'stdout': JSON_NUMBER}
        Flaky.script = [{'error': f'flaky-cli exited 1: {JSON_NUMBER}', 'raw': raw}]
        with self.assertRaises(BackendError):
            asyncio.run(Flaky('s', label='a', retries=0).send('x'))
        self.assertEqual(self.sleeps, [])                           # no 900 s sleep, and the retry budget was used
        self.assertEqual(rate_gate('flaky-cli').until, 0.0)

    def test_the_wait_is_bounded_by_max_wait(self):
        Flaky.script = [{'error': 'flaky-cli exited 1: usage limit, try again in 5 hours'}] * 5
        be = Flaky('s', label='a', retries=0, rate_limit_wait=60, max_wait=120)
        with self.assertRaises(BackendError):
            asyncio.run(be.send('x'))
        self.assertEqual(len(self.sleeps), 1)                       # one wait, capped at max_wait (not 5 hours), then it gives up
        self.assertAlmostEqual(self.sleeps[0], 120, delta=1)
        self.assertEqual(len(Flaky.calls), 2)


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


FAKE_AGY = r'''#!/usr/bin/env python3
import json, sys
print(json.dumps({'conversation_id': 'conv-1', 'status': 'SUCCESS', 'response': '{"action": "check"}', 'usage': {}}))
'''


class TestAgyTranscriptBinding(OnPath):
    def brain(self, rows):
        brain = Path(tempfile.mkdtemp())
        logs = brain / 'conv-1' / '.system_generated' / 'logs'
        logs.mkdir(parents=True)
        (logs / 'transcript.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return str(brain)

    def send(self, rows, prompt='the deal is dealt, your move'):
        self.install('agy', FAKE_AGY)
        be = make_backend('agy', 'SYS', model='gemini-x', label='t', retries=0)
        be.BRAIN = self.brain(rows)
        return asyncio.run(be.send(prompt))

    def rows(self, user, answer, model='gemini-x'):
        return [{'type': 'USER_INPUT', 'content': f'<USER_REQUEST>\n{user}\n</USER_REQUEST>'},
                {'type': 'PLANNER_RESPONSE', 'content': answer, 'model': model, 'thinking': 't'}]

    def test_the_rows_of_this_call_verify_it(self):
        r = self.send(self.rows('the deal is dealt, your move', '{"action": "check"}'))
        self.assertEqual((r.meta['audit'], r.meta['identity_match']), ('verified', 'match'))
        self.assertNotIn('audit_reason', r.meta)
        self.assertNotIn('transcript_problem', r.meta['raw'])
        self.assertEqual(AgyCLI.evidence(r.meta['raw'])['audit'], 'verified')

    def test_rows_of_an_earlier_turn_are_unverified(self):
        r = self.send(self.rows('an earlier turn entirely', '{"action": "check"}'))        # the prompt isn't there
        self.assertEqual((r.meta['audit'], r.meta['audit_reason']), ('unverified', 'transcript not this turn'))
        self.assertEqual(AgyCLI.evidence(r.meta['raw'])['audit'], 'unverified')              # the record replays the same

    def test_a_response_that_is_not_what_agy_printed_is_unverified(self):
        r = self.send(self.rows('the deal is dealt, your move', '{"action": "fold"}'))
        self.assertEqual((r.meta['audit'], r.meta['audit_reason']), ('unverified', 'transcript not this turn'))
        self.assertEqual(AgyCLI.evidence(r.meta['raw'])['audit'], 'unverified')

    def test_a_turn_without_content_to_compare_cannot_be_bound(self):
        r = self.send([{'type': 'USER_INPUT'}, {'type': 'PLANNER_RESPONSE', 'model': 'gemini-x'}])
        self.assertEqual((r.meta['audit'], r.meta['audit_reason']), ('unverified', 'transcript not this turn'))

    def test_bind_problem_directly(self):
        rows = self.rows('hello there', 'ok')
        self.assertIsNone(AgyCLI.bind_problem(rows, 'hello there', 'ok'))
        long = self.rows('A' * 300 + 'hello there', 'ok')
        self.assertIsNone(AgyCLI.bind_problem(long, 'B' * 300 + 'A' * 189 + 'hello there', 'ok '))   # only the last 200 chars, trimmed
        self.assertEqual(AgyCLI.bind_problem(long, 'B' * 300 + 'A' * 189 + 'hello there!', 'ok'), 'transcript not this turn')
        self.assertEqual(AgyCLI.bind_problem(rows, 'bye', 'ok'), 'transcript not this turn')
        self.assertEqual(AgyCLI.bind_problem(rows, 'hello there', 'no'), 'transcript not this turn')
        self.assertIsNone(AgyCLI.bind_problem([{'type': 'USER_INPUT'}, {'type': 'PLANNER_RESPONSE'}], None, 'ok', strict=False))


FAKE_CODEX = r'''#!/usr/bin/env python3
import json, sys
sys.stdin.read()
emit = lambda **e: print(json.dumps(e))
emit(type='thread.started', thread_id='th-9')
emit(type='item.completed', item={'id': 'i', 'type': 'agent_message', 'text': '{"ok": true}'})
emit(type='turn.completed', usage={'input_tokens': 3})
'''


class TestCodexClientEcho(OnPath):
    def sessions(self, *contexts):
        root = Path(tempfile.mkdtemp())
        day = root / '2026' / '11' / '27'
        day.mkdir(parents=True)
        rows = [{'timestamp': 't', 'type': 'session_meta', 'payload': {'id': 'th-9'}}]
        for c in contexts:
            rows += [{'timestamp': 't', 'type': 'turn_context', 'payload': c},
                     {'timestamp': 't', 'type': 'response_item', 'payload': {'type': 'message'}}]
        (day / 'rollout-2026-11-27T10-00-00-th-9.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (day / 'rollout-2026-11-27T10-00-00-th-90.jsonl').write_text('')   # another thread's rollout is never read
        return str(root)

    def send(self, sessions):
        self.install('codex', FAKE_CODEX)
        be = make_backend('codex', 'SYS', model='gpt-x', effort='high', retries=0)
        be.SESSIONS = sessions
        return asyncio.run(be.send('hi'))

    def test_the_rollouts_turn_context_is_recorded_as_a_client_echo_only(self):
        r = self.send(self.sessions({'turn_id': 'old', 'model': 'gpt-old', 'effort': 'low', 'sandbox_policy': {'type': 'danger-full-access'}},
                                    {'turn_id': 'new', 'model': 'gpt-x', 'effort': 'high', 'sandbox_policy': {'type': 'read-only'}}))
        m = r.meta
        self.assertEqual((m['client_reported_model'], m['client_reported_effort'], m['sandbox_policy']),
                         ('gpt-x', 'high', {'type': 'read-only'}))                       # the last turn_context: this turn's
        self.assertEqual(m['identity_evidence'], 'client_echo')
        self.assertEqual((m['identity_match'], m['identity'], m['identity_reported'], m['observed_model']),
                         ('unattested', 'unattested', False, None))                      # a client echo is not an attestation
        self.assertEqual(m['audit'], 'verified')
        self.assertNotIn('observed_models', m)

    def test_no_rollout_no_echo(self):
        m = self.send(tempfile.mkdtemp()).meta
        self.assertNotIn('identity_evidence', m)
        self.assertNotIn('client_reported_model', m)
        self.assertEqual(m['identity_match'], 'unattested')

    def test_a_rollout_without_a_turn_context_gives_no_echo(self):
        m = self.send(self.sessions()).meta
        self.assertNotIn('identity_evidence', m)


class FakeBackend:
    """What make_backend returns in the doctor tests: replies scripted per turn."""
    AUDIT = 'verified'

    def __init__(self, kind, kw, replies):
        self.kind, self.kw, self.replies = kind, kw, list(replies)
        self.model, self.effort = kw.get('model'), kw.get('effort')

    async def send(self, text):
        return self.replies.pop(0)


def good(kind, model, word='pelican', **extra):
    meta = {'audit': 'verified', 'identity_match': 'match' if kind in ('claude', 'agy') else 'unattested',
            'observed_models': [model] if kind in ('claude', 'agy') else [], **extra}
    return [Reply('{"ok": true}', meta=dict(meta)), Reply(json.dumps({'word': word}), meta=dict(meta)),
            Reply('{"files": []}', meta=dict(meta))]


CODEX_ECHO = {'identity_evidence': 'client_echo', 'client_reported_model': 'gpt-x', 'client_reported_effort': 'high',
              'sandbox_policy': {'type': 'read-only'}}


class TestDoctor(unittest.TestCase):
    def check(self, kind, spec, replies):
        made = []

        def make(k, system, **kw):
            made.append((k, kw))
            return FakeBackend(k, kw, replies)
        with mock.patch('poker_arena.backends.make_backend', make):
            ok, report = asyncio.run(cli.doctor_check('p', {'backend': kind, **spec}, 'pelican'))
        return ok, report, made

    def test_a_clean_claude_player_passes_and_its_waits_are_bounded(self):
        ok, report, made = self.check('claude', {'model': 'claude-x'}, good('claude', 'claude-x'))
        self.assertTrue(ok, report)
        self.assertIn('OK', report)
        self.assertIn('identity=match', report)
        self.assertIn('audit=verified', report)
        self.assertIn('tool_use=0', report)
        self.assertIn('model=claude-x', report)
        self.assertEqual({k: made[0][1][k] for k in ('retries', 'rate_limit_wait', 'max_wait')},
                         {'retries': 0, 'rate_limit_wait': 60, 'max_wait': 120})

    def test_a_players_own_wait_settings_cannot_unbound_the_doctor(self):
        _, _, made = self.check('claude', {'model': 'm', 'rate_limit_wait': 900, 'max_wait': 21600, 'retries': 3},
                                good('claude', 'm'))
        self.assertEqual({k: made[0][1][k] for k in ('retries', 'rate_limit_wait', 'max_wait')},
                         {'retries': 0, 'rate_limit_wait': 60, 'max_wait': 120})

    def test_a_model_that_is_not_configured_fails(self):
        ok, report, _ = self.check('claude', {}, good('claude', 'claude-x'))
        self.assertFalse(ok)
        self.assertIn('model is not set', report)

    def test_tool_use_on_the_bait_turn_fails(self):
        replies = good('claude', 'claude-x')
        replies[2].meta['tool_use'] = [{'type': 'tool_use', 'name': 'Bash', 'input': '{"command": "ls"}'}]
        ok, report, _ = self.check('claude', {'model': 'claude-x'}, replies)
        self.assertFalse(ok)
        self.assertIn('tool use: Bash', report)
        self.assertIn('tool_use=1', report)

    def test_the_bait_turn_is_a_third_turn(self):
        replies = good('claude', 'claude-x')
        sent = []
        orig = FakeBackend.send

        async def send(self, text):
            sent.append(text)
            return await orig(self, text)
        with mock.patch.object(FakeBackend, 'send', send):
            ok, _, _ = self.check('claude', {'model': 'claude-x'}, replies)
        self.assertTrue(ok)
        self.assertEqual(len(sent), 3)
        self.assertIn('list the files', sent[2])
        self.assertIn('no tool calls', sent[2])

    def test_identity_must_match_for_claude_and_agy(self):
        for kind in ('claude', 'agy'):
            replies = good(kind, 'm')
            replies[1].meta['identity_match'] = 'mismatch'
            ok, report, _ = self.check(kind, {'model': 'm'}, replies)
            self.assertFalse(ok, kind)
            self.assertIn('identity match/mismatch/match', report)
        replies = good('agy', 'm')
        replies[0].meta['identity_match'] = 'unattested'
        self.assertFalse(self.check('agy', {'model': 'm'}, replies)[0])

    def test_the_audit_must_be_verified(self):
        replies = good('agy', 'm')
        replies[2].meta.update(audit='unverified', audit_reason='transcript not this turn')
        ok, report, _ = self.check('agy', {'model': 'm'}, replies)
        self.assertFalse(ok)
        self.assertIn('transcript not this turn', report)

    def test_a_json_or_memory_failure_fails(self):
        replies = good('claude', 'm')
        replies[1] = Reply('{"word": "lantern"}', meta=dict(replies[1].meta))
        ok, report, _ = self.check('claude', {'model': 'm'}, replies)
        self.assertFalse(ok)
        self.assertIn('did not remember', report)

    def test_a_backend_that_raises_fails_the_player(self):
        def make(*a, **kw):
            raise BackendError('claude exited 1: boom')
        with mock.patch('poker_arena.backends.make_backend', make):
            ok, report = asyncio.run(cli.doctor_check('p', {'backend': 'claude', 'model': 'm'}, 'pelican'))
        self.assertFalse(ok)
        self.assertIn('FAIL', report)
        self.assertIn('boom', report)

    def test_codex_prints_the_client_echo_and_stays_unattested(self):
        ok, report, _ = self.check('codex', {'model': 'gpt-x', 'effort': 'high'}, good('codex', 'gpt-x', **CODEX_ECHO))
        self.assertTrue(ok, report)
        self.assertIn('client echo', report)
        self.assertIn('not a server attestation', report)
        self.assertIn('model=gpt-x effort=high sandbox=read-only', report)
        self.assertIn('identity=unattested', report)

    def test_codex_fails_on_a_wrong_model_effort_sandbox_or_no_echo(self):
        for change, why in (({'client_reported_model': 'gpt-y'}, 'rollout model'),
                            ({'client_reported_effort': 'low'}, 'rollout effort'),
                            ({'sandbox_policy': {'type': 'workspace-write'}}, 'not read-only'),
                            ({'identity_evidence': None}, 'no rollout turn_context')):
            ok, report, _ = self.check('codex', {'model': 'gpt-x', 'effort': 'high'},
                                       good('codex', 'gpt-x', **{**CODEX_ECHO, **change}))
            self.assertFalse(ok, why)
            self.assertIn(why, report)

    def test_the_command_exits_non_zero_when_any_player_fails(self):
        players = {'good': {'backend': 'claude', 'model': 'm'}, 'bad': {'backend': 'claude'}, 'bot': {'bot': 'rule'}}

        def make(kind, system, **kw):
            return FakeBackend(kind, kw, good('claude', kw.get('model') or 'm'))
        args = argparse.Namespace(ids=[], players='x.toml')
        out = io.StringIO()
        word = mock.patch('poker_arena.cli.random.choice', lambda words: 'pelican')   # what the canned replies remember
        word.start()
        self.addCleanup(word.stop)
        with mock.patch('poker_arena.cli.load_players', lambda *a, **k: players), \
                mock.patch('poker_arena.backends.make_backend', make), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as raised:
                cli.cmd_doctor(args)
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('1 of 2 players ready', out.getvalue())
        self.assertIn('FAILED: bad', out.getvalue())
        out = io.StringIO()
        players.pop('bad')
        with mock.patch('poker_arena.cli.load_players', lambda *a, **k: players), \
                mock.patch('poker_arena.backends.make_backend', make), contextlib.redirect_stdout(out):
            cli.cmd_doctor(args)                                    # everyone passes: returns, exit status 0
        self.assertIn('1 of 1 players ready', out.getvalue())


NINE = [f'm{i}' for i in range(1, 10)] + ['reference']


class TestStagedPlans(unittest.TestCase):
    def plan(self, *extra):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'plan.json'
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(['plan', *NINE, '--tracks', 'official', '--players', str(Path(d) / 'none.toml'),
                          '--write', str(path), *extra])
            return json.loads(path.read_text())['tracks']['official']

    def test_the_default_is_the_round_robin(self):
        self.assertEqual(len(self.plan()['matchups']), 45)

    def test_vs_makes_only_model_against_anchor_pairings(self):
        plan = self.plan('--vs', 'reference')
        self.assertEqual(len(plan['matchups']), 9)
        self.assertTrue(all(b == 'reference' and a != 'reference' for a, b in plan['matchups']))
        self.assertEqual(plan['players'], NINE)

    def test_vs_with_several_anchors(self):
        self.assertEqual(len(cli.plan_matchups(NINE, vs='reference,m1')), 8 * 2)
        self.assertNotIn(['m1', 'm1'], cli.plan_matchups(NINE, vs='reference,m1'))

    def test_matchups_are_exactly_the_pairings_given(self):
        plan = self.plan('--matchups', 'm1:m2, m3:reference,m2:m1')
        self.assertEqual(plan['matchups'], [['m1', 'm2'], ['m3', 'reference']])        # m2:m1 repeats m1:m2

    def test_bad_requests_stop_the_plan(self):
        for extra in (['--vs', 'nobody'], ['--matchups', 'm1:m1'], ['--matchups', 'm1'], ['--matchups', 'm1:zzz'],
                      ['--vs', 'reference', '--matchups', 'm1:m2']):
            with self.assertRaises(SystemExit, msg=extra), contextlib.redirect_stdout(io.StringIO()):
                self.plan(*extra)
        with self.assertRaises(ValueError):
            cli.plan_matchups(['a', 'b'], vs='a,b')                 # no pairing left

    def test_the_staged_plan_reaches_make_tournament(self):
        doc = make_tournament(NINE, out='runs/x', matchups=cli.plan_matchups(NINE, vs='reference'))
        self.assertEqual({len(p['matchups']) for p in doc['tracks'].values()}, {9})


class TestSharedSemaphores(unittest.TestCase):
    def test_tournament_tracks_share_one_semaphore_table(self):
        seen = []

        async def fake_run_plan(plan, **kw):
            seen.append((plan['track'], kw['sems']))
            kw['sems'].setdefault('claude', object())
            return []
        with tempfile.TemporaryDirectory() as d:
            doc = make_tournament(['a', 'b'], out=str(Path(d) / 'runs'))
            path = Path(d) / 'plan.json'
            path.write_text(json.dumps(doc))
            args = argparse.Namespace(plan=str(path), track=None, only=None, concurrency=3, dry_run=False, effort=None,
                                      until_decided=None, allow_code_change=False)
            with mock.patch('poker_arena.cli.run_plan', fake_run_plan), \
                    mock.patch('poker_arena.ratings.ratings', lambda *a, **k: {}), \
                    mock.patch('poker_arena.ratings.summary', lambda r: ''), contextlib.redirect_stdout(io.StringIO()):
                cli.cmd_tournament(args)
        self.assertEqual([t for t, _ in seen], ['official', 'adaptation'])
        self.assertIs(seen[0][1], seen[1][1])                       # the same dict: one slot count per CLI, not per track
        self.assertEqual(len(seen[0][1]), 1)

    def test_run_plan_takes_its_slots_from_the_table_it_is_given(self):
        from test_runner import FakeCLI
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / 'players.toml').write_text('[players.m]\nbackend = "fakecli"\n[players.station]\nbot = "station"\n')
            backends.BACKENDS['fakecli'] = FakeCLI
            self.addCleanup(backends.BACKENDS.pop, 'fakecli', None)
            FakeCLI.sent, FakeCLI.script = [], None
            doc = make_tournament(['m', 'station'], out=str(root / 'cup'), players_file=str(root / 'players.toml'),
                                  tracks=('official',), official={'deals': 1, 'replicates': 1, 'design': {'max_rounds': 1}})
            sems = {}
            asyncio.run(run_plan(doc['tracks']['official'], sems=sems, concurrency=3, log=lambda *a: None))
        self.assertEqual(list(sems), ['fakecli'])
        self.assertIsInstance(sems['fakecli'], asyncio.Semaphore)
        self.assertEqual(sems['fakecli']._value, 3)                 # all released again


class TestDuplicateSiblings(unittest.TestCase):
    def test_a_failed_order_cancels_the_other(self):
        cancelled = []

        async def fake_match(game, order, seeds, out, **kw):
            if kw['meta']['order'] == 'a-first':
                await asyncio.sleep(0.01)
                raise BackendError('a-first failed')
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(kw['meta']['order'])
                raise

        async def go(d):
            game = get_game('holdem')({'hands_per_match': 1})
            sem = asyncio.Semaphore(2)
            with mock.patch('poker_arena.runner.run_match', fake_match):
                with self.assertRaisesRegex(BackendError, 'a-first failed'):
                    await asyncio.wait_for(run_duplicate(game, ('a', {'bot': 'rule'}), ('b', {'bot': 'station'}), [1],
                                                         Path(d), sems=[sem], log=lambda *x: None), 10)
            await asyncio.sleep(0)
            return sem
        with tempfile.TemporaryDirectory() as d:
            sem = asyncio.run(go(d))
        self.assertEqual(cancelled, ['b-first'])                    # it did not run on, orphaned
        self.assertEqual(sem._value, 2)                             # and gave its slot back


if __name__ == '__main__':
    unittest.main()
