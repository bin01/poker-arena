"""The runner's record and recovery: resuming refuses changed settings, an interrupted hand doesn't survive
in a CLI conversation, the audit trail (prompts, versions, settings, every attempt) and corrections."""
import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from poker_arena import backends
from poker_arena.backends import Backend, BackendError, Reply
from poker_arena.games import get_game
from poker_arena.runner import PlanConflict, ResumeMismatch, make_plan, run_duplicate, run_match, run_plan

FAKE = [sys.executable, str(Path(__file__).with_name('fake_model.py'))]
QUIET = dict(log=lambda *a: None)


def passive(text):
    opts = re.findall(r'Your options: (.*)', text)
    return {'action': 'check' if opts and opts[-1].startswith('check') else 'call', 'amount': None}


class FakeCLI(Backend):
    """A stateful 'CLI' kept in memory: sessions are ids, every payload it got is kept per class."""
    stateful = True
    exe = None
    sent: list = []
    script = None   # optional: f(n_call, text) -> reply dict or BackendError

    async def _call(self, text):
        self.folder()
        FakeCLI.sent.append({'session': self.session, 'text': text})
        n = len(FakeCLI.sent)
        out = FakeCLI.script(n, text) if FakeCLI.script else None
        if isinstance(out, BackendError):
            raise out
        self.session = self.session or f's{n}'
        return Reply(json.dumps({'reasoning': 'fake', 'say': '', 'notes': None, **(out or passive(text))}),
                     meta={'session': self.session})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name).resolve()
        backends.BACKENDS['fakecli'] = FakeCLI
        FakeCLI.sent, FakeCLI.script = [], None
        self.G = get_game('holdem')({'hands_per_match': 4})

    def tearDown(self):
        backends.BACKENDS.pop('fakecli', None)
        self.tmp.cleanup()

    def rows(self, path):
        return [json.loads(l) for l in Path(path).read_text().splitlines()] if Path(path).exists() else []


class TestResume(Base):
    def test_match_refuses_changed_settings(self):
        seats = [('rule', {'bot': 'rule', 'sims': 30}), ('station', {'bot': 'station'})]
        m = self.d / 'm'
        asyncio.run(run_match(self.G, seats, [1, 2, 3, 4], m, **QUIET))
        with self.assertRaisesRegex(ResumeMismatch, 'seeds'):                # even when finished
            asyncio.run(run_match(self.G, seats, [1, 2, 3, 5], m, **QUIET))
        (m / 'result.json').unlink()
        with self.assertRaisesRegex(ResumeMismatch, 'seeds'):
            asyncio.run(run_match(self.G, seats, [1, 2, 3, 5], m, **QUIET))
        with self.assertRaisesRegex(ResumeMismatch, 'specs'):
            asyncio.run(run_match(self.G, [('rule', {'bot': 'rule', 'sims': 31}), seats[1]], [1, 2, 3, 4], m, **QUIET))
        with self.assertRaisesRegex(ResumeMismatch, 'game'):
            asyncio.run(run_match(get_game('holdem')({'hands_per_match': 4, 'stack': 100}), seats, [1, 2, 3, 4], m,
                                  **QUIET))
        asyncio.run(run_match(self.G, seats, [1, 2, 3, 4], m, **QUIET))   # the recorded settings resume
        self.assertTrue((m / 'result.json').exists())

    def test_legacy_config_keys_are_adopted(self):
        seats = [('rule', {'bot': 'rule', 'sims': 30}), ('station', {'bot': 'station'})]
        m = self.d / 'm'
        asyncio.run(run_match(self.G, seats, [1, 2], m, **QUIET))
        (m / 'result.json').unlink()
        cfg = json.loads((m / 'config.json').read_text())
        for k in ('system_prompts', 'deal_fingerprints', 'effective', 'cli_versions'):
            del cfg[k]
        (m / 'config.json').write_text(json.dumps(cfg))
        asyncio.run(run_match(self.G, seats, [1, 2], m, **QUIET))

    def test_plan_refuses_changed_args_before_playing(self):
        players = self.d / 'players.toml'
        players.write_text('[players.rule]\nbot = "rule"\nsims = 30\n[players.station]\nbot = "station"\n')
        plan = make_plan(['rule', 'station'], deals=4, out=str(self.d / 'run'), players_file=str(players),
                         design={'max_rounds': 1})
        with mock.patch('poker_arena.ratings.adjudicate', create=True,
                        side_effect=lambda pd, d, n, k, e: {'verdict': 'exhausted', 'looks_used': k, 'deals': len(pd['xs'])}):
            asyncio.run(run_plan(plan, **QUIET))
            for change in ({'deals': 5, 'design': {**plan['design'], 'round_deals': 5}}, {'seed': 7},
                           {'design': {**plan['design'], 'alpha': 0.1}}, {'replicates': 2}):
                with self.assertRaises(ResumeMismatch, msg=change):
                    asyncio.run(run_plan({**plan, **change}, **QUIET))
            with self.assertRaisesRegex(PlanConflict, 'round_deals'):            # deals that contradict the design
                asyncio.run(run_plan({**plan, 'deals': 5}, **QUIET))
            players.write_text('[players.rule]\nbot = "rule"\nsims = 99\n[players.station]\nbot = "station"\n')
            with self.assertRaisesRegex(ResumeMismatch, 'specs'):
                asyncio.run(run_plan(plan, **QUIET))
            # a changed match folder is caught by the plan before anything plays
            players.write_text('[players.rule]\nbot = "rule"\nsims = 30\n[players.station]\nbot = "station"\n')
            cfg_path = self.d / 'run' / 'rule__vs__station' / 'b-first' / 'config.json'
            cfg = json.loads(cfg_path.read_text())
            cfg['seeds'][0] += 1
            cfg_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ResumeMismatch, 'b-first'):
                asyncio.run(run_plan(plan, **QUIET))


class Attesting(FakeCLI):
    """Names the model that answered: the configured one, or `ANSWER` when set."""
    ANSWER = None

    async def _call(self, text):
        r = await super()._call(text)
        r.meta['observed_model'] = Attesting.ANSWER or self.model + '-20260101'
        return r


class TestEligibilityFacts(Base):
    def facts(self, backend, answer=None):
        from poker_arena.runner import eligibility
        backends.BACKENDS['attesting'] = Attesting
        self.addCleanup(backends.BACKENDS.pop, 'attesting', None)
        Attesting.ANSWER = answer
        a, b = ('m', {'backend': backend, 'model': 'fake-1'}), ('station', {'bot': 'station'})
        out = self.d / backend / str(answer)
        asyncio.run(run_duplicate(self.G, a, b, [1, 2], out, **QUIET))
        return eligibility(out, ['m__vs__station'], ('m', 'station'), {'status': 'registered'})

    def test_identity_audit_and_protocol_per_player(self):
        f = self.facts('fakecli')
        m, st = f['players']['m'], f['players']['station']
        self.assertEqual((m['kind'], m['backend'], m['requested_model'], m['identity']),
                         ('model', 'fakecli', 'fake-1', 'unattested'))
        self.assertEqual((m['audit'], m['audit_counts']), ('complete', {'verified': m['decisions']}))
        self.assertEqual(len(m['protocols']), 1)
        self.assertTrue(m['protocol_consistent'])
        self.assertEqual(st['identity'], 'bot')
        self.assertEqual((f['registration'], f['conversations'], f['implementation_changes']),
                         ({'status': 'registered'}, 1, []))
        att = self.facts('attesting')['players']['m']
        self.assertEqual((att['identity'], att['observed_models']), ('attested', ['fake-1-20260101']))
        self.assertEqual(self.facts('attesting', 'other-9')['players']['m']['identity'], 'mismatch')

    def test_thought_events_carry_identity(self):
        self.facts('attesting')
        out = self.d / 'attesting' / 'None' / 'm__vs__station' / 'a-first'
        t = next(e['data'] for e in self.rows(out / 'events.jsonl') if e['kind'] == 'thought' and e['data']['player'] == 'm')
        self.assertEqual((t['requested_model'], t['observed_model'], t['identity'], t['served_model']),
                         ('fake-1', 'fake-1-20260101', 'attested', 'fake-1-20260101'))
        c = self.rows(out / 'calls.jsonl')[0]
        self.assertEqual((c['requested_model'], c['identity']), ('fake-1', 'attested'))


class TestInterruptedHand(Base):
    seats = [('m', {'backend': 'fakecli'}), ('station', {'bot': 'station'})]

    def stop_after(self, m, hands, mid_hand):
        """Make the folder look like a run that was stopped after `hands` hands (and inside the next one)."""
        (m / 'result.json').unlink()
        st = json.loads((m / 'state.json').read_text())
        hist = [json.loads(l) for l in (m / 'hands.jsonl').read_text().splitlines()]
        (m / 'hands.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in hist[:hands]))
        # the state as it was written at the end of hand `hands`
        turns = [c for c in self.rows(m / 'calls.jsonl') if c['hand'] < hands]
        st['hands'] = st['hands_committed'] = hands
        st['agents'][0]['backend']['turns'] = [[c['prompt'], c['output']] for c in turns]
        st['agents'][0]['backend']['session'] = 's1'
        (m / 'state.json').write_text(json.dumps(st))
        if not mid_hand:
            for f in ('events.jsonl', 'calls.jsonl', 'raw.jsonl'):
                rows = [r for r in self.rows(m / f) if r.get('episode', r.get('hand', 0)) < hands]
                (m / f).write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_mid_hand_stop_rebuilds_the_conversation(self):
        m = self.d / 'm'
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3, 4], m, **QUIET))
        dropped_calls = len([c for c in self.rows(m / 'calls.jsonl') if c['hand'] >= 2])
        self.stop_after(m, 2, mid_hand=True)
        FakeCLI.sent = []
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3, 4], m, **QUIET))
        first = FakeCLI.sent[0]
        self.assertIsNone(first['session'])                                 # a new conversation, not s1
        self.assertIn('interrupted', first['text'])
        self.assertIn('Hand 2 of 4', first['text'])                          # committed turns are replayed
        self.assertNotIn('Hand 3 of 4 · flop', first['text'].split('=== NOW ===')[0])
        self.assertIsNotNone(FakeCLI.sent[1]['session'])                     # and then continued
        calls = self.rows(m / 'calls.jsonl')
        self.assertEqual([c['hand'] for c in calls if c.get('rebuilt_from_turns')], [2])
        self.assertEqual(len([r for r in self.rows(m / 'interrupted.jsonl') if r['file'] == 'calls.jsonl']),
                         dropped_calls)
        st = json.loads((m / 'state.json').read_text())
        self.assertEqual(st['resumes'][0]['after_hand'], 2)
        self.assertEqual(st['resumes'][0]['rebuilt_seats'], [0])
        self.assertEqual(len(self.rows(m / 'hands.jsonl')), 4)
        self.assertEqual(sum(1 for e in self.rows(m / 'events.jsonl') if e['kind'] == 'hand_start'), 4)

    def test_clean_stop_keeps_the_conversation(self):
        m = self.d / 'm'
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3, 4], m, **QUIET))
        self.stop_after(m, 2, mid_hand=False)
        FakeCLI.sent = []
        asyncio.run(run_match(self.G, self.seats, [1, 2, 3, 4], m, **QUIET))
        self.assertEqual(FakeCLI.sent[0]['session'], 's1')
        self.assertNotIn('interrupted', FakeCLI.sent[0]['text'])


class TestAudit(Base):
    def test_config_calls_raw_and_hands(self):
        fake = {'backend': 'command', 'command': FAKE, 'api_key': 'secret'}
        asyncio.run(run_duplicate(self.G, ('fake', fake), ('rule', {'bot': 'rule', 'sims': 30}), [5, 6, 7, 8], self.d,
                                  **QUIET))
        m = self.d / 'fake__vs__rule' / 'a-first'
        cfg = json.loads((m / 'config.json').read_text())
        self.assertEqual(len(cfg['system_prompts']), 2)
        self.assertIn('You are Alex', cfg['system_prompts'][0])
        self.assertEqual(len(cfg['deal_fingerprints']), 4)
        self.assertEqual(cfg['deal_fingerprints'][0], self.G.deal_fingerprint(5))
        self.assertTrue(cfg['cli_versions'][FAKE[0]].startswith('Python'))
        self.assertEqual(cfg['effective'][0]['backend'], 'command')
        self.assertEqual(cfg['effective'][0]['retries'], 3)                 # defaults included
        self.assertEqual(cfg['effective'][1], {'bot': 'rule', 'sims': 30, 'seed': 101})
        self.assertNotIn('api_key', json.dumps(cfg))
        self.assertIn('code_version', cfg)
        self.assertEqual((cfg['round'], cfg['replicate'], cfg['order']), (1, 1, 'a-first'))
        b = json.loads((self.d / 'fake__vs__rule' / 'b-first' / 'config.json').read_text())
        self.assertEqual(cfg['deal_fingerprints'], b['deal_fingerprints'])  # duplicate: the same deals
        calls = self.rows(m / 'calls.jsonl')
        self.assertTrue(all(c['argv'][0] == FAKE[0] and c['system_sha1'] for c in calls))
        self.assertNotIn('raw', calls[0])
        raw = self.rows(m / 'raw.jsonl')
        self.assertEqual(len(raw), len(calls))
        self.assertEqual(raw[0]['stdout'].strip(), calls[0]['output'])
        hands = self.rows(m / 'hands.jsonl')
        self.assertEqual([h['seed'] for h in hands], [5, 6, 7, 8])
        self.assertEqual(hands[0]['fingerprint'], cfg['deal_fingerprints'][0])

    def test_failed_attempts_and_rate_limits_are_logged(self):
        def script(n, text):
            if n == 1:
                return BackendError('HTTP 429: rate limit, try again later')
            if n == 2:
                return BackendError('fakecli exited 1: boom')
        FakeCLI.script = staticmethod(script)
        seats = [('m', {'backend': 'fakecli', 'retries': 1, 'rate_limit_wait': 0}), ('station', {'bot': 'station'})]

        async def no_sleep(_):
            return None
        with mock.patch.object(backends.asyncio, 'sleep', no_sleep):
            asyncio.run(run_match(self.G, seats, [1], self.d / 'm', **QUIET))
        first = self.rows(self.d / 'm' / 'calls.jsonl')[0]
        fa = first['failed_attempts']
        self.assertEqual(len(fa), 2)
        self.assertTrue(fa[0]['rate_limited'])
        self.assertIn('boom', fa[1]['error'])
        self.assertEqual(fa[1]['waited'], 10)

    def test_a_backend_that_gives_up_is_logged_and_stops_the_match(self):
        FakeCLI.script = staticmethod(lambda n, text: BackendError('fakecli exited 1: dead') if n > 2 else None)
        seats = [('m', {'backend': 'fakecli', 'retries': 0}), ('station', {'bot': 'station'})]
        with self.assertRaises(BackendError):
            asyncio.run(run_match(self.G, seats, [1, 2, 3, 4], self.d / 'm', **QUIET))
        calls = self.rows(self.d / 'm' / 'calls.jsonl')
        self.assertTrue(calls[-1]['backend_failed'])
        self.assertIn('dead', calls[-1]['failed_attempts'][0]['error'])


class TestCorrections(Base):
    def thoughts(self, m):
        return [e['data'] for e in self.rows(m / 'events.jsonl') if e['kind'] == 'thought' and e['data']['seat'] == 0]

    def test_retry_sizing_and_fallback(self):
        # call 1: Infinity (refused, asked again); 2: raise to 3 (below the min of 4: sizing)
        # then on the next decision three unplayable replies: the fallback
        replies = {1: {'action': 'raise', 'amount': 'Infinity'}, 2: {'action': 'raise', 'amount': 3},
                   3: {'action': 'fold'}, 4: {'action': 'fold'}, 5: {'action': 'dance'}}

        def script(n, text):
            return replies.get(n)
        FakeCLI.script = staticmethod(script)
        seats = [('m', {'backend': 'fakecli'}), ('rule', {'bot': 'station'})]
        m = self.d / 'm'
        asyncio.run(run_match(self.G, seats, [1, 2], m, **QUIET))
        t = self.thoughts(m)
        self.assertEqual(t[0]['correction']['kind'], 'retry')
        self.assertIn('finite', t[0]['correction']['reason'])
        self.assertIn('minimum', t[0]['correction']['reason'])           # the sizing after the retry, too
        self.assertEqual(t[0]['requested'], {'action': 'raise', 'amount': 'Infinity'})
        self.assertEqual(t[0]['executed'], {'action': 'raise', 'amount': 4})
        self.assertEqual(len(t[0]['retries']), 1)
        # the station calls; seat 0 (big blind... or not) is asked next: folds with nothing to call, three times
        self.assertEqual(t[1]['correction']['kind'], 'fallback')
        self.assertTrue(t[1]['fallback'])
        self.assertEqual(t[1]['executed']['action'], 'check')
        self.assertEqual(t[1]['requested'], {'action': 'dance', 'amount': None})
        self.assertIn('cannot fold', t[1]['correction']['reason'])
        self.assertTrue(all(x['correction'] is None for x in t[2:]))
        self.assertTrue(all(x['requested'] == x['executed'] or x['requested']['action'] in ('check', 'call')
                            for x in t[2:]))
        result = json.loads((m / 'result.json').read_text())
        self.assertEqual(result['fallbacks'][0], 1)
        # the logs stay strict JSON (a browser can't parse a bare Infinity)
        def strict(c):
            raise ValueError(c)
        for f in ('events.jsonl', 'calls.jsonl', 'hands.jsonl'):
            for line in (m / f).read_text().splitlines():
                json.loads(line, parse_constant=strict)

    def test_sizing_only(self):
        FakeCLI.script = staticmethod(lambda n, text: {'action': 'raise', 'amount': 9999} if n == 1 else None)
        m = self.d / 'm'
        asyncio.run(run_match(self.G, [('m', {'backend': 'fakecli'}), ('s', {'bot': 'station'})], [1], m, **QUIET))
        t = self.thoughts(m)[0]
        self.assertEqual(t['correction'], {'kind': 'sizing', 'reason': '9999 is more than you have: all in for 200'})
        self.assertEqual(t['requested'], {'action': 'raise', 'amount': 9999})
        self.assertEqual(t['executed'], {'action': 'raise', 'amount': 200})
        self.assertEqual(t['retries'], [])



class TestTempFolder(unittest.TestCase):
    def test_falls_back_to_a_folder_in_the_repo(self):
        import os
        import conftest
        env = {'TMPDIR': '/nonexistent-read-only/tmp', 'TEMP': '', 'TMP': ''}
        with mock.patch.dict(os.environ, env), mock.patch.object(tempfile, 'gettempdir', side_effect=FileNotFoundError):
            self.assertEqual(Path(conftest._choose()), Path(__file__).resolve().parent / '.tmp')
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'TMPDIR': d}):
            self.assertEqual(conftest._choose(), d)                          # TMPDIR first when writable


if __name__ == '__main__':
    unittest.main()
