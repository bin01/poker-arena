"""Fairness and recovery: tool-using turns are rejected and taken back, failed tries keep their evidence and
never leave a CLI conversation ahead of the record, a hand is committed atomically and every log is cut back to
the commit on resume, bots resume with their random state, and each player's protocol is fingerprinted."""
import asyncio
import json
import tarfile
import unittest
from unittest import mock

from poker_arena import backends, runner
from poker_arena.agents import RandomBot
from poker_arena.backends import BackendError
from poker_arena.runner import ResumeMismatch, make_plan, match_manifest, run_match, run_plan
from test_runner import QUIET, Base, FakeCLI


def strip_t(rows):
    return [{k: v for k, v in r.items() if k != 't'} for r in rows]


class ToolCLI(FakeCLI):
    """Uses a tool on the calls listed in `tool_calls`."""
    tool_calls = set()

    async def _call(self, text):
        reply = await super()._call(text)
        if len(FakeCLI.sent) in ToolCLI.tool_calls:
            reply.meta['tool_use'] = [{'type': 'command_execution', 'name': 'ls -la'}]
        return reply


class TestToolUse(Base):
    def setUp(self):
        super().setUp()
        backends.BACKENDS['toolcli'] = ToolCLI

    def tearDown(self):
        backends.BACKENDS.pop('toolcli', None)
        super().tearDown()

    def test_a_tool_turn_is_rejected_taken_back_and_asked_again(self):
        ToolCLI.tool_calls = {2}
        m = self.d / 'm'
        asyncio.run(run_match(self.G, [('m', {'backend': 'toolcli'}), ('s', {'bot': 'station'})], [1, 2], m, **QUIET))
        retry = FakeCLI.sent[2]
        self.assertIsNone(retry['session'])                                   # a new conversation without that turn
        self.assertIn('Tools are not allowed', retry['text'])
        self.assertIn(FakeCLI.sent[1]['text'], retry['text'])                 # the same question, replayed history
        calls = self.rows(m / 'calls.jsonl')
        bad = [c for c in calls if c.get('rejected') == 'tool_use']
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]['tool_use'], [{'type': 'command_execution', 'name': 'ls -la'}])
        thoughts = [e['data'] for e in self.rows(m / 'events.jsonl') if e['kind'] == 'thought' and e['data']['seat'] == 0]
        tooled = [t for t in thoughts if t['correction'] and t['correction']['kind'] == 'tool_use']
        self.assertEqual(len(tooled), 1)
        self.assertIn('ls -la', tooled[0]['correction']['reason'])
        self.assertEqual(tooled[0]['retries'][0]['kind'], 'tool_use')
        st = json.loads((m / 'state.json').read_text())
        turns = st['agents'][0]['backend']['turns']
        self.assertEqual(len(turns), len(calls) - 1)                          # the record never kept that turn

    def test_tool_use_every_time_falls_back(self):
        ToolCLI.tool_calls = set(range(1, 100))
        m = self.d / 'm'
        asyncio.run(run_match(self.G, [('m', {'backend': 'toolcli', 'max_fixes': 1}), ('s', {'bot': 'station'})], [1],
                              m, **QUIET))
        first = [e['data'] for e in self.rows(m / 'events.jsonl') if e['kind'] == 'thought'][0]
        self.assertEqual(first['correction']['kind'], 'fallback')
        self.assertIn('Tools are not allowed', first['correction']['reason'])


class TestFailedAttempts(Base):
    def test_a_failed_try_keeps_its_output_and_rebuilds_the_conversation(self):
        class Flaky(FakeCLI):
            async def _call(self, text):
                if len(FakeCLI.sent) == 2:   # the third call fails after the CLI ran
                    FakeCLI.sent.append({'session': self.session, 'text': text})
                    self.last_call.update(launched=True, raw={'returncode': 1, 'stdout': '{"partial": 1}',
                                                              'stderr': 'network gone'})
                    raise BackendError('fakecli exited 1: network gone')
                return await super()._call(text)
        backends.BACKENDS['flaky'] = Flaky
        try:
            async def no_sleep(_):
                return None
            with mock.patch.object(backends.asyncio, 'sleep', no_sleep):
                asyncio.run(run_match(self.G, [('m', {'backend': 'flaky'}), ('s', {'bot': 'station'})], [1, 2],
                                      self.d / 'm', **QUIET))
        finally:
            backends.BACKENDS.pop('flaky', None)
        failed, retry = FakeCLI.sent[2], FakeCLI.sent[3]
        self.assertEqual(failed['session'], 's1')
        self.assertIsNone(retry['session'])                                   # not built on top of the failed try
        self.assertIn(FakeCLI.sent[0]['text'], retry['text'])                 # committed turns replayed
        call = next(c for c in self.rows(self.d / 'm' / 'calls.jsonl') if c.get('failed_attempts'))
        fa = call['failed_attempts'][0]
        self.assertEqual((fa['returncode'], fa['session_before'], fa['rebuilt']), (1, 's1', True))
        self.assertNotIn('raw', fa)
        self.assertEqual(call['rebuilt_from_turns'], 2)
        self.assertEqual((call['session_before'], call['session_after']), (None, 's4'))
        raw = [r for r in self.rows(self.d / 'm' / 'raw.jsonl') if 'failed_try' in r]
        self.assertEqual((raw[0]['stderr'], raw[0]['stdout'], raw[0]['returncode']), ('network gone', '{"partial": 1}', 1))


class TestCommit(Base):
    seats = [('m', {'backend': 'fakecli'}), ('station', {'bot': 'station'})]

    def test_resume_reconciles_every_log_to_the_committed_hands(self):
        m = self.d / 'm'
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3, 4], m, **QUIET))
        clean = {f: self.rows(m / f) for f in ('hands.jsonl', 'events.jsonl')}
        # a run killed after hand 3's row reached hands.jsonl but before its checkpoint, mid-write in calls.jsonl
        (m / 'result.json').unlink()
        st = json.loads((m / 'state.json').read_text())
        calls = self.rows(m / 'calls.jsonl')
        st['hands_committed'] = st['hands'] = 2
        st['agents'][0]['backend']['turns'] = [[c['prompt'], c['output']] for c in calls if c['hand'] < 2]
        (m / 'state.json').write_text(json.dumps(st))
        hands = self.rows(m / 'hands.jsonl')[:3]
        (m / 'hands.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in hands))
        with (m / 'calls.jsonl').open('a') as f:
            f.write('{"hand": 3, "prom')
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3, 4], m, **QUIET))
        self.assertEqual([h['hand'] for h in self.rows(m / 'hands.jsonl')], [0, 1, 2, 3])
        self.assertEqual(strip_t(self.rows(m / 'hands.jsonl')), strip_t(clean['hands.jsonl']))
        self.assertEqual(sum(1 for e in self.rows(m / 'events.jsonl') if e['kind'] == 'hand_start'), 4)
        moved = {r['file'] for r in self.rows(m / 'interrupted.jsonl')}
        self.assertTrue({'hands.jsonl', 'events.jsonl', 'calls.jsonl'} <= moved)
        resume = json.loads((m / 'state.json').read_text())['resumes'][0]
        self.assertEqual(resume['torn_lines'], {'calls.jsonl': 1})
        self.assertEqual(resume['dropped_rows']['hands.jsonl'], 1)
        for c in self.rows(m / 'calls.jsonl'):
            self.assertIn('prompt', c)
        self.assertFalse(list(m.glob('*.tmp')))

    def test_checkpoint_is_replaced_whole(self):
        m = self.d / 'm'
        writes = []
        real = __import__('os').replace

        def spy(src, dst):
            writes.append((str(src), str(dst)))
            return real(src, dst)
        with mock.patch('poker_arena.runner.os.replace', spy):
            asyncio.run(run_match(self.G, self.seats, [1, 2], m, **QUIET))
        self.assertIn((str(m / 'state.json.tmp'), str(m / 'state.json')), writes)
        self.assertEqual(json.loads((m / 'state.json').read_text())['hands_committed'], 2)


class Stop(Exception):
    pass


class TestBotsResumeExactly(Base):
    seats = [('random', {'bot': 'random'}), ('rule', {'bot': 'rule', 'sims': 40})]
    seeds = list(range(10, 18))

    def test_interrupted_equals_uninterrupted(self):
        G = self.G.__class__({'hands_per_match': len(self.seeds)})
        asyncio.run(run_match(G, self.seats, self.seeds, self.d / 'clean', **QUIET))
        real, count = RandomBot.act, [0]

        async def act(bot, *a):
            count[0] += 1
            if count[0] == 9:
                raise Stop()
            return await real(bot, *a)
        with mock.patch.object(RandomBot, 'act', act), self.assertRaises(Stop):
            asyncio.run(run_match(G, self.seats, self.seeds, self.d / 'cut', **QUIET))
        self.assertTrue(self.rows(self.d / 'cut' / 'events.jsonl'))
        asyncio.run(run_match(G, self.seats, self.seeds, self.d / 'cut', **QUIET))
        for f in ('hands.jsonl', 'events.jsonl'):
            self.assertEqual(strip_t(self.rows(self.d / 'cut' / f)), strip_t(self.rows(self.d / 'clean' / f)), f)
        self.assertEqual(json.loads((self.d / 'cut' / 'result.json').read_text())['net'],
                         json.loads((self.d / 'clean' / 'result.json').read_text())['net'])
        st = json.loads((self.d / 'cut' / 'state.json').read_text())
        self.assertEqual(len(st['agents'][0]['rng'][1]), 625)


class TestProtocol(Base):
    def test_fingerprint_is_per_player_and_seat_independent(self):
        spec = {'backend': 'fakecli', 'model': 'm-1', 'effort': 'high'}
        one = match_manifest(self.G, [('m', spec), ('s', {'bot': 'station'})], [1, 2])['protocol']
        two = match_manifest(self.G, [('s', {'bot': 'station'}), ('m', spec)], [7, 8])['protocol']
        self.assertEqual(one, two)
        longer = match_manifest(self.G, [('m', spec), ('s', {'bot': 'station'})], [7, 8, 9])['protocol']
        self.assertNotEqual(longer['m']['fingerprint'], one['m']['fingerprint'])   # the horizon is part of it
        self.assertEqual(longer['m']['settings']['horizon']['hands_per_conversation'], 3)
        self.assertIn('every match', longer['m']['settings']['horizon']['memory_reset'])
        self.assertEqual(longer['s'], one['s'])                                     # a bot has no memory
        p = one['m']
        self.assertEqual(len(p['fingerprint']), 16)
        s = p['settings']
        self.assertEqual((s['backend'], s['model'], s['max_fixes'], s['retries']), ('fakecli', 'm-1', 2, 3))
        self.assertIn('system_prompt_sha1', s)
        self.assertIn('tool_policy', s)
        self.assertNotIn('timeout', s)
        self.assertEqual(one['s']['settings'], {'bot': 'station'})
        slow = match_manifest(self.G, [('m', {**spec, 'timeout': 5}), ('s', {'bot': 'station'})], [1, 2])['protocol']
        self.assertEqual(slow['m']['fingerprint'], p['fingerprint'])         # a timeout doesn't change play
        for change in ({'max_fixes': 0}, {'extra_args': ['--x']}, {'model': 'm-2'}):
            other = match_manifest(self.G, [('m', {**spec, **change}), ('s', {'bot': 'station'})], [1, 2])['protocol']
            self.assertNotEqual(other['m']['fingerprint'], p['fingerprint'], change)

    def test_config_records_it_and_resume_checks_it(self):
        m = self.d / 'm'
        seats = [('m', {'backend': 'fakecli'}), ('s', {'bot': 'station'})]
        asyncio.run(run_match(self.G, seats, [1, 2], m, **QUIET))
        cfg = json.loads((m / 'config.json').read_text())
        self.assertEqual(set(cfg['protocol']), {'m', 's'})
        (m / 'result.json').unlink()
        cfg['protocol']['m']['fingerprint'] = 'something-else'
        (m / 'config.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ResumeMismatch, 'protocol'):
            asyncio.run(run_match(self.G, seats, [1, 2], m, **QUIET))
        del cfg['protocol']                                                   # written by older code: adopted
        (m / 'config.json').write_text(json.dumps(cfg))
        asyncio.run(run_match(self.G, seats, [1, 2], m, **QUIET))



class UnverifiedCLI(FakeCLI):
    """Its turn record can't be read on the calls listed in `bad`."""
    bad = set()

    async def _call(self, text):
        reply = await super()._call(text)
        if len(FakeCLI.sent) in UnverifiedCLI.bad:
            reply.meta['audit'] = 'unverified'
        return reply


class TestAudit(Base):
    def setUp(self):
        super().setUp()
        backends.BACKENDS['unverified'] = UnverifiedCLI

    def tearDown(self):
        backends.BACKENDS.pop('unverified', None)
        super().tearDown()

    def test_an_unverified_turn_is_rejected_and_asked_again(self):
        UnverifiedCLI.bad = {2}
        m = self.d / 'm'
        asyncio.run(run_match(self.G, [('m', {'backend': 'unverified'}), ('s', {'bot': 'station'})], [1, 2], m,
                              **QUIET))
        self.assertIsNone(FakeCLI.sent[2]['session'])                         # rebuilt without that turn
        self.assertEqual(FakeCLI.sent[2]['text'].split('=== NOW ===')[-1].strip().split('\n')[1],
                         FakeCLI.sent[1]['text'].split('\n')[0])              # the same question, no notice
        calls = self.rows(m / 'calls.jsonl')
        self.assertTrue(all(c.get('audit') for c in calls))                  # every call carries its status
        self.assertEqual([c['rejected'] for c in calls if c.get('rejected')], ['unverified'])
        thoughts = [e['data'] for e in self.rows(m / 'events.jsonl') if e['kind'] == 'thought' and e['data']['seat'] == 0]
        self.assertEqual([t['correction']['kind'] for t in thoughts if t['correction']], ['unverified'])
        self.assertTrue(all(t['audit'] == 'verified' for t in thoughts))       # every move played was audited
        turns = json.loads((m / 'state.json').read_text())['agents'][0]['backend']['turns']
        self.assertEqual(len(turns), len(calls) - 1)

    def test_no_auditable_turn_stops_the_match_instead_of_playing(self):
        UnverifiedCLI.bad = set(range(1, 100))
        m = self.d / 'm'
        with self.assertRaisesRegex(BackendError, 'audited'):
            asyncio.run(run_match(self.G, [('m', {'backend': 'unverified', 'max_fixes': 1}), ('s', {'bot': 'station'})],
                                  [1, 2], m, **QUIET))
        self.assertEqual([c['rejected'] for c in self.rows(m / 'calls.jsonl')], ['unverified', 'unverified'])
        self.assertFalse(self.rows(m / 'hands.jsonl'))
        UnverifiedCLI.bad = set()
        asyncio.run(run_match(self.G, [('m', {'backend': 'unverified', 'max_fixes': 1}), ('s', {'bot': 'station'})],
                              [1, 2], m, **QUIET))                              # resumable once it can be read
        self.assertEqual(len(self.rows(m / 'hands.jsonl')), 2)


class TestImplementation(Base):
    seats = [('m', {'backend': 'fakecli'}), ('s', {'bot': 'station'})]

    def changed(self, name='poker_arena/backends.py'):
        src = dict(runner.sources())
        src[name] = src[name] + b'\n# changed\n'
        return mock.patch.object(runner, '_SOURCES', src)

    def test_config_records_hashes_and_resume_refuses_other_code(self):
        m = self.d / 'm'
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3], m, **QUIET))
        impl = json.loads((m / 'config.json').read_text())['implementation']
        for f in ('poker_arena/runner.py', 'poker_arena/agents.py', 'poker_arena/backends.py', 'poker_arena/ratings.py',
                  'poker_arena/cards.py', 'poker_arena/games/holdem.py'):
            self.assertIn(f, impl['files'])
        self.assertTrue(impl['prompt_template'] and impl['fingerprint'])
        (m / 'result.json').unlink()
        with self.changed(), self.assertRaisesRegex(ResumeMismatch, 'backends.py.*--allow-code-change'):
            asyncio.run(run_match(self.G, self.seats, [1, 2, 3], m, **QUIET))
        with self.changed():
            asyncio.run(run_match(self.G, self.seats, [1, 2, 3], m, allow_code_change=True, **QUIET))
            cfg = json.loads((m / 'config.json').read_text())
            change = cfg['implementation_changes'][0]
            self.assertEqual((change['from'], change['files']), (impl['fingerprint'], ['poker_arena/backends.py']))
            self.assertEqual(cfg['implementation'], impl)                       # the original stays
            (m / 'result.json').unlink()
            asyncio.run(run_match(self.G, self.seats, [1, 2, 3], m, **QUIET))  # the accepted code resumes freely
        (m / 'result.json').unlink()
        with self.assertRaises(ResumeMismatch):                                  # and the old code is a change again
            asyncio.run(run_match(self.G, self.seats, [1, 2, 3], m, **QUIET))

    def test_plan_refuses_other_code_before_playing_and_keeps_each_source(self):
        players = self.d / 'players.toml'
        players.write_text('[players.a]\nbot = "station"\n[players.b]\nbot = "station"\n')
        plan = make_plan(['a', 'b'], deals=4, out=str(self.d / 'run'), players_file=str(players),
                         design={'max_rounds': 2})
        decide = lambda xs, d, n, k, groups=None: {'verdict': 'exhausted' if k >= 2 else 'open', 'looks_used': k}
        with mock.patch('poker_arena.ratings.decide', side_effect=decide):
            asyncio.run(run_plan(plan, **QUIET))
            out = self.d / 'run'
            stored = json.loads((out / 'plan.json').read_text())
            self.assertEqual(stored['sources'], [f'source/{stored["implementation"]["fingerprint"]}.tar.gz'])
            with tarfile.open(out / stored['sources'][0]) as tar:
                names = tar.getnames()
            self.assertIn('poker_arena/runner.py', names)
            self.assertIn('implementation.json', names)
            with self.changed('poker_arena/ratings.py'), self.assertRaisesRegex(ResumeMismatch, 'ratings.py'):
                asyncio.run(run_plan(plan, **QUIET))
            with self.changed('poker_arena/ratings.py'):
                asyncio.run(run_plan(plan, allow_code_change=True, **QUIET))
                stored = json.loads((out / 'plan.json').read_text())
                self.assertEqual(len(stored['sources']), 2)
                self.assertEqual(stored['implementation_changes'][0]['files'], ['poker_arena/ratings.py'])
                first = f'source/{stored["implementation"]["fingerprint"]}.tar.gz'
                self.assertEqual(stored['implementation_changes'][0]['source'], next(x for x in stored['sources']
                                                                                     if x != first))

    def test_a_viewer_change_is_not_an_implementation_change(self):
        src = dict(runner.sources())
        key = next(k for k in src if k.endswith('server.py'))
        src[key] += b'\n# changed\n'
        before = runner.implementation(self.G)['fingerprint']
        with mock.patch.object(runner, '_SOURCES', src):
            self.assertEqual(runner.implementation(self.G)['fingerprint'], before)


if __name__ == '__main__':
    unittest.main()
