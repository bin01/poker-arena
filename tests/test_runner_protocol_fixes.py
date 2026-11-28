"""Regression tests for the formally verified runner, backend, and ratings protocol fixes."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from poker_arena import backends, ratings, runner
from poker_arena.games.holdem import Holdem


class TestRunnerProtocolFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.G = Holdem()
        ratings._VERIFIED.clear()

    def tearDown(self):
        ratings._VERIFIED.clear()
        self.tmp.cleanup()

    # ---- Finding 1: CommitSafety (first-checkpoint window vs legacy) -----------------------

    def test_modern_match_without_state_json_admits_zero_committed_hands(self):
        root = self.d / 'm__vs__s'
        a_dir, b_dir = root / 'a-first', root / 'b-first'
        asyncio.run(runner.run_match(self.G, [('a', {'bot': 'station'}), ('b', {'bot': 'station'})], [42], a_dir, log=lambda *_: None))
        asyncio.run(runner.run_match(self.G, [('b', {'bot': 'station'}), ('a', {'bot': 'station'})], [42], b_dir, log=lambda *_: None))
        # Simulate crash between put('hands.jsonl') and _write('state.json') on hand 0
        (a_dir / 'state.json').unlink()
        (a_dir / 'result.json').unlink()

        self.assertEqual(ratings._hands(a_dir), {})
        self.assertEqual(runner.verify_match(a_dir)['hands'], 0)
        self.assertFalse(runner.verify_match(a_dir)['ok'])
        conv = ratings.read_conversation(root)
        self.assertEqual(conv['verified'], [])
        self.assertEqual(conv['provisional']['a-first'], [0])
        self.assertIn('0 committed', conv['commit']['a-first'])

    def test_legacy_match_without_state_json_still_admits_verified_hands(self):
        root = self.d / 'leg__vs__s'
        a_dir, b_dir = root / 'a-first', root / 'b-first'
        asyncio.run(runner.run_match(self.G, [('a', {'bot': 'station'}), ('b', {'bot': 'station'})], [42], a_dir, log=lambda *_: None))
        asyncio.run(runner.run_match(self.G, [('b', {'bot': 'station'}), ('a', {'bot': 'station'})], [42], b_dir, log=lambda *_: None))
        # Strip record_version and state.json to represent a genuine pre-v2 legacy record
        cfg = json.loads((a_dir / 'config.json').read_text())
        cfg.pop('record_version', None)
        (a_dir / 'config.json').write_text(json.dumps(cfg))
        (a_dir / 'state.json').unlink()

        self.assertEqual(list(ratings._hands(a_dir).keys()), [0])
        conv = ratings.read_conversation(root)
        self.assertEqual(conv['verified'], [0])
        self.assertTrue(conv['commit']['a-first'].startswith('legacy:'))

    def test_corrupt_and_boolean_state_json_fail_closed(self):
        m = self.d / 'corrupt'
        asyncio.run(runner.run_match(self.G, [('a', {'bot': 'station'}), ('b', {'bot': 'station'})], [42], m, log=lambda *_: None))
        (m / 'state.json').write_text('{corrupt')
        self.assertEqual(ratings._hands(m), {})
        self.assertEqual(runner.verify_match(m)['hands'], 0)
        # JSON true must NOT be accepted as int 1
        (m / 'state.json').write_text(json.dumps({'hands_committed': True}))
        self.assertEqual(ratings._hands(m), {})
        self.assertEqual(runner.verify_match(m)['hands'], 0)

    def test_stateful_backend_without_session_id_keeps_replaying_transcript(self):
        prompts = []

        class NoSessionBackend(backends.Backend):
            stateful = True
            exe = None

            async def _call(self, text: str) -> backends.Reply:
                prompts.append(text)
                return backends.Reply(text='{"action": "call"}')

        b = NoSessionBackend('sys')
        b.rebuilding = True
        asyncio.run(b.send('turn-1'))
        asyncio.run(b.send('turn-2'))
        self.assertIn('THE MATCH SO FAR', prompts[1])
        self.assertIn('turn-1', prompts[1])

    def test_capacity_one_woken_waiter_cancellation_hands_permit_to_next_waiter(self):
        async def scenario():
            sem_agy, sem_claude = asyncio.Semaphore(1), asyncio.Semaphore(1)
            await sem_claude.acquire()
            peer_got_claude = asyncio.Event()
            b_blocked = asyncio.Event()

            async def peer():
                await b_blocked.wait()
                await sem_claude.acquire()
                peer_got_claude.set()
                sem_claude.release()

            with mock.patch.object(runner, 'run_match', lambda *a, **k: asyncio.sleep(10)), \
                    mock.patch.object(runner, 'save_source'):
                dup = asyncio.create_task(runner.run_duplicate(
                    self.G, ('a', {'bot': 'station'}), ('b', {'bot': 'station'}),
                    [1], self.d / 'cap1', sems=[sem_agy, sem_claude],
                ))
                while not sem_claude._waiters:
                    await asyncio.sleep(0)
                b_blocked.set()
                p = asyncio.create_task(peer())
                while len(sem_claude._waiters or ()) < 2:
                    await asyncio.sleep(0)
                # Release claude (wakes dup's waiter) and cancel dup BEFORE it resumes
                sem_claude.release()
                dup.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await dup
                await asyncio.wait_for(p, timeout=2.0)
                self.assertTrue(peer_got_claude.is_set())
                self.assertEqual((sem_agy._value, sem_claude._value), (1, 1))

        asyncio.run(scenario())

    def test_verify_play_cache_respects_modified_hand_or_cfg_with_same_stamp(self):
        m = self.d / 'mod'
        asyncio.run(runner.run_match(self.G, [('a', {'bot': 'station'}), ('b', {'bot': 'station'})], [11], m, log=lambda *_: None))
        cfg = json.loads((m / 'config.json').read_text())
        h0 = ratings._hands(m, cfg)[0]
        v1 = ratings.verify_play(m, cfg, {0: h0})[0]
        self.assertTrue(v1['ok'])
        # Modified in-memory hand with identical folder stamp must NOT return stale cached digest/ok
        tampered = {**h0, 'net': [999, -999]}
        v2 = ratings.verify_play(m, cfg, {0: tampered})[0]
        self.assertNotEqual(v2['digest'], v1['digest'])
        # Modified cfg seeds with identical folder stamp must NOT return stale cached ok=True
        bad_cfg = {**cfg, 'seeds': [99999]}
        v3 = ratings.verify_play(m, bad_cfg, {0: h0})[0]
        self.assertFalse(v3['ok'])

    # ---- Finding 2: NoHistoryLossOnRestore (rebuilding persisted across fallback commit) ---

    def test_tool_use_fallback_preserves_rebuilding_across_crash_and_restore(self):
        prompts_seen = []
        hand_num = [0]
        wd = str(self.d / 'cli-wd-tf')
        Path(wd).mkdir()

        class ToolFallbackBackend(backends.Backend):
            stateful = True
            exe = None

            async def _call(self, text: str) -> backends.Reply:
                prompts_seen.append((hand_num[0], self.session, self.rebuilding, 'THE MATCH SO FAR' in text))
                self.workdir = wd
                if self.session is None:
                    self.session = f's-{len(prompts_seen)}'
                if hand_num[0] == 1:
                    return backends.Reply(
                        text='{"action": "call", "reasoning": "ok"}',
                        meta={'tool_use': [{'name': 'bash'}]},
                    )
                return backends.Reply(text='{"action": "call", "reasoning": "ok"}')

        backends.BACKENDS['test_tool_fb'] = ToolFallbackBackend
        try:
            m = self.d / 'tf'
            players = [('m1', {'backend': 'test_tool_fb', 'max_fixes': 0}), ('b1', {'bot': 'station'})]
            orig_write = runner._write

            def spy_write(path, obj, *a, **k):
                if Path(path).name == 'state.json':
                    hand_num[0] = obj['hands_committed']
                return orig_write(path, obj, *a, **k)

            orig_ep = self.G.episode

            def stop_at_hand2(k, seed, names):
                if k == 2:
                    raise KeyboardInterrupt('crash after hand 1 commit')
                return orig_ep(k, seed, names)

            with mock.patch.object(runner, '_write', spy_write), mock.patch.object(self.G, 'episode', stop_at_hand2):
                with self.assertRaises(KeyboardInterrupt):
                    asyncio.run(runner.run_match(self.G, players, [10, 20, 30], m, log=lambda *_: None))

            st = json.loads((m / 'state.json').read_text())
            b_st = st['agents'][0]['backend']
            self.assertIsNone(b_st['session'])
            self.assertTrue(b_st['rebuilding'])
            self.assertTrue(len(b_st['turns']) > 0)

            before = len(prompts_seen)
            asyncio.run(runner.run_match(self.G, players, [10, 20, 30], m, log=lambda *_: None))
            first_after_resume = prompts_seen[before]
            self.assertIsNone(first_after_resume[1])
            self.assertTrue(first_after_resume[2])   # rebuilding=True
            self.assertTrue(first_after_resume[3])   # transcript replayed!
        finally:
            backends.BACKENDS.pop('test_tool_fb', None)

    # ---- Finding 3: NoDirtySessionReuse (double crash during recovery) ---------------------

    def test_double_crash_during_recovery_never_reuses_dirty_cli_session(self):
        abandoned_sessions = set()
        reused_abandoned = []
        sessions_used_after_resume2 = []
        phase = ['run1']
        call_num = [0]
        hand_idx = [0]
        wd = str(self.d / 'cli-wd-dc')
        Path(wd).mkdir()

        class DirtyCheckBackend(backends.Backend):
            stateful = True
            exe = None

            async def _call(self, text: str) -> backends.Reply:
                call_num[0] += 1
                self.workdir = wd
                if self.session is None:
                    self.session = f'sess-{call_num[0]}'
                if phase[0] == 'resume2':
                    sessions_used_after_resume2.append(self.session)
                if self.session in abandoned_sessions:
                    reused_abandoned.append((call_num[0], self.session))
                if hand_idx[0] == 1:
                    hand_idx[0] = 2
                    abandoned_sessions.add(self.session)
                    raise backends.BackendError('crash #1 mid-hand after CLI stored turn')
                return backends.Reply(text='{"action": "call", "reasoning": "ok"}')

        backends.BACKENDS['test_dirty_chk'] = DirtyCheckBackend
        try:
            m = self.d / 'dc'
            players = [('m1', {'backend': 'test_dirty_chk', 'retries': 0}), ('b1', {'bot': 'station'})]
            orig_write = runner._write

            def spy_write(path, obj, *a, **k):
                if Path(path).name == 'state.json':
                    hand_idx[0] = obj['hands_committed']
                return orig_write(path, obj, *a, **k)

            # Run 1: hand 0 commits (sess-1), hand 1 dirties sess-1 and crashes mid-hand
            with mock.patch.object(runner, '_write', spy_write):
                with self.assertRaises(backends.BackendError):
                    asyncio.run(runner.run_match(self.G, players, [10, 20], m, log=lambda *_: None))
            self.assertEqual(abandoned_sessions, {'sess-1'})

            # Resume #1: crashes during verify_match right after _reconcile truncated late rows
            phase[0] = 'resume1'
            with mock.patch.object(runner, 'verify_match', side_effect=KeyboardInterrupt('crash #2')):
                with self.assertRaises(KeyboardInterrupt):
                    asyncio.run(runner.run_match(self.G, players, [10, 20], m, log=lambda *_: None))

            # state.json must already have cleared sess-1 and persisted rebuilding=True
            st2 = json.loads((m / 'state.json').read_text())
            self.assertIsNone(st2['agents'][0]['backend']['session'])
            self.assertTrue(st2['agents'][0]['backend']['rebuilding'])

            # Resume #2: completes hand 1 on a fresh CLI session without ever touching dirty sess-1
            phase[0] = 'resume2'
            asyncio.run(runner.run_match(self.G, players, [10, 20], m, log=lambda *_: None))
            self.assertEqual(reused_abandoned, [])
            self.assertTrue(sessions_used_after_resume2)
            self.assertNotIn('sess-1', sessions_used_after_resume2)
        finally:
            backends.BACKENDS.pop('test_dirty_chk', None)

    # ---- Finding 4: DuplicateConcurrency semaphore permit conservation ---------------------

    def test_run_duplicate_releases_semaphores_on_sibling_cancel_ext_cancel_and_on_start_error(self):
        async def scenario():
            # Case A: sibling failure with FIFO peer ahead in sem_claude._waiters
            sem_agy, sem_claude = asyncio.Semaphore(2), asyncio.Semaphore(2)
            await sem_claude.acquire()
            b_waiting, a_entered = asyncio.Event(), asyncio.Event()
            peer2_task = None

            async def fake_run_match(game, order, seeds, out, **kw):
                if Path(out).name == 'a-first':
                    a_entered.set()
                    await b_waiting.wait()
                    while len(sem_claude._waiters or ()) < 2:
                        await asyncio.sleep(0)
                    raise RuntimeError('boom in a-first')
                await asyncio.sleep(10)

            orig_acq = sem_agy.acquire
            acq_count = 0

            async def spy_acq():
                nonlocal acq_count, peer2_task
                acq_count += 1
                if acq_count == 2:
                    await a_entered.wait()
                    peer2_task = asyncio.create_task(sem_claude.acquire())
                    while not sem_claude._waiters:
                        await asyncio.sleep(0)
                    r = await orig_acq()
                    b_waiting.set()
                    return r
                return await orig_acq()

            sem_agy.acquire = spy_acq
            with mock.patch.object(runner, 'run_match', fake_run_match), mock.patch.object(runner, 'save_source'):
                with self.assertRaises(RuntimeError):
                    await runner.run_duplicate(
                        self.G, ('a', {'bot': 'station'}), ('b', {'bot': 'station'}),
                        [1], self.d / 'case_a', sems=[sem_agy, sem_claude],
                    )
            if peer2_task:
                await peer2_task
                sem_claude.release()
            sem_claude.release()
            self.assertEqual((sem_agy._value, sem_claude._value), (2, 2))

            # Case B: external cancel while both orders hold sem_agy and wait on sem_claude
            sem_agy, sem_claude = asyncio.Semaphore(2), asyncio.Semaphore(2)
            await sem_claude.acquire()
            await sem_claude.acquire()
            with mock.patch.object(runner, 'run_match', lambda *a, **k: asyncio.sleep(10)), \
                    mock.patch.object(runner, 'save_source'):
                t = asyncio.create_task(runner.run_duplicate(
                    self.G, ('a', {'bot': 'station'}), ('b', {'bot': 'station'}),
                    [1], self.d / 'case_b', sems=[sem_agy, sem_claude],
                ))
                while len(sem_claude._waiters or ()) < 2:
                    await asyncio.sleep(0)
                t.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await t
            sem_claude.release()
            sem_claude.release()
            self.assertEqual((sem_agy._value, sem_claude._value), (2, 2))

            # Case C: on_start() raises after acquiring both semaphores
            sem_agy, sem_claude = asyncio.Semaphore(2), asyncio.Semaphore(2)
            with mock.patch.object(runner, 'save_source'):
                with self.assertRaises(OSError):
                    await runner.run_duplicate(
                        self.G, ('a', {'bot': 'station'}), ('b', {'bot': 'station'}),
                        [1], self.d / 'case_c', sems=[sem_agy, sem_claude],
                        on_start=lambda: (_ for _ in ()).throw(OSError('disk full')),
                    )
            self.assertEqual((sem_agy._value, sem_claude._value), (2, 2))

        asyncio.run(scenario())

    # ---- Runner-up Finding 5: verify_play subset-cache growth ------------------------------

    def test_verify_play_cache_handles_growing_hand_subset_without_keyerror(self):
        root = self.d / 'grow__vs__s'
        a_dir, b_dir = root / 'a-first', root / 'b-first'
        asyncio.run(runner.run_match(self.G, [('a', {'bot': 'station'}), ('b', {'bot': 'station'})], [11, 22], a_dir, log=lambda *_: None))
        asyncio.run(runner.run_match(self.G, [('b', {'bot': 'station'}), ('a', {'bot': 'station'})], [11, 22], b_dir, log=lambda *_: None))

        # First read: b-first is only at hands_committed=1 while a-first already has 2 committed hands
        b_state_path = b_dir / 'state.json'
        b_state = json.loads(b_state_path.read_text())
        b_state['hands_committed'] = b_state['hands'] = 1
        b_state_path.write_text(json.dumps(b_state))

        conv1 = ratings.read_conversation(root)
        self.assertEqual(conv1['verified'], [0])

        # Now b-first catches up to hands_committed=2 WITHOUT touching a-first's files/stamp!
        b_state['hands_committed'] = b_state['hands'] = 2
        b_state_path.write_text(json.dumps(b_state))

        conv2 = ratings.read_conversation(root)
        self.assertEqual(conv2['verified'], [0, 1])


if __name__ == '__main__':
    unittest.main()
