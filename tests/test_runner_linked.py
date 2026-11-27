"""R6-#3, #6, #10: every decision links its calls, their raw output and the executed action, and its audit status is
derived from those links re-checked against the raw output; a CLI that changes under a running match is recorded
and voids the registration; the committed record is told apart from provisional rows."""
import asyncio
import json
from unittest import mock

from poker_arena import backends, runner
from poker_arena.runner import committed_records, make_plan, run_match, run_plan
from test_runner import FAKE, QUIET
from test_runner_integrity import Facts


class Linked(Facts):
    def match(self, out, label='a-first'):
        return out / 'm__vs__station' / label

    def thoughts(self, d):
        return [e['data'] for e in self.rows(d / 'events.jsonl') if e['kind'] == 'thought']


class TestLinks(Linked):
    def test_every_decision_links_its_calls_raw_rows_and_action(self):
        out = self.play()
        d = self.match(out)
        cfg = json.loads((d / 'config.json').read_text())
        self.assertEqual(cfg['record_version'], runner.RECORD_VERSION)
        calls, raw = self.rows(d / 'calls.jsonl'), self.rows(d / 'raw.jsonl')
        actions = [e['data'] for e in self.rows(d / 'events.jsonl') if e['kind'] == 'action']
        thoughts = self.thoughts(d)
        self.assertEqual(len({t['decision_id'] for t in thoughts}), len(thoughts))
        self.assertEqual([a['decision_id'] for a in actions], [t['decision_id'] for t in thoughts])
        hands = self.rows(d / 'hands.jsonl')
        self.assertEqual([a['decision_id'] for h in hands for a in h['actions']], [t['decision_id'] for t in thoughts])
        for t in thoughts:
            mine = [c for c in calls if c['decision_id'] == t['decision_id']]
            self.assertEqual([c['call_id'] for c in mine], t['call_ids'])
            if t['player'] == 'm' and not t['fallback']:
                self.assertEqual(t['accepted_call'], t['call_ids'][-1])
                self.assertEqual(t['accepted']['action'], json.loads(mine[-1]['output'])['action'])
        self.assertEqual(sorted(r['call_id'] for r in raw), sorted(c['call_id'] for c in calls))
        f = self.facts(out)
        m = f['players']['m']
        self.assertEqual((m['audit'], m['audit_source'], m['problems']), ('complete', 'linked', []))
        self.assertEqual(m['audit_counts'], {'verified': m['decisions']})
        self.assertEqual(f['players']['station']['audit'], 'complete')
        self.assertEqual(len(f['verification']['record_hash']), 64)
        self.assertEqual(f['verification']['audit_sources'], ['linked'])

    def test_a_doctored_call_is_caught_by_its_raw_output(self):
        out = self.play()
        before = self.facts(out)['verification']
        calls = self.match(out) / 'calls.jsonl'

        def doctor(rows):
            rows[0]['output'] = rows[0]['output'].replace('"reasoning": "fake"', '"reasoning": "edited"')
            return rows
        self.edit(calls, doctor)
        f = self.facts(out)
        m = f['players']['m']
        self.assertEqual(m['audit'], 'incomplete')
        self.assertEqual(m['audit_counts'].get('inconsistent'), 1)
        self.assertTrue(any('not what the raw output says' in p for p in m['problems']))
        self.assertNotEqual(f['verification']['record_hash'], before['record_hash'])
        self.assertEqual(f['verification']['evidence_hash'], before['evidence_hash'])   # the hands did not change

    def test_a_missing_raw_row_or_a_relabelled_decision(self):
        out = self.play()
        self.edit(self.match(out) / 'raw.jsonl', lambda rows: rows[1:])
        m = self.facts(out)['players']['m']
        self.assertEqual((m['audit'], m['audit_counts'].get('unverifiable')), ('incomplete', 1))
        out = self.play('relabel')

        def relabel(rows):
            t = next(r for r in rows if r['kind'] == 'thought' and r['data']['player'] == 'm')
            t['data']['audit'] = 'opaque'
            return rows
        self.edit(self.match(out, 'b-first') / 'events.jsonl', relabel)
        m = self.facts(out)['players']['m']
        self.assertEqual((m['audit'], m['audit_counts'].get('inconsistent')), ('incomplete', 1))
        self.assertTrue(any("labelled 'opaque'" in p for p in m['problems']))

    def test_an_action_that_names_no_decision(self):
        out = self.play()

        def unlink(rows):
            a = next(r for r in rows if r['kind'] == 'action' and r['data']['seat'] == 0)
            del a['data']['decision_id']
            return rows
        self.edit(self.match(out) / 'events.jsonl', unlink)
        f = self.facts(out)
        self.assertTrue(f['verification']['ok'])          # the hands still replay
        self.assertTrue(any('executed action is not the one linked' in p for p in f['problems']))

    def test_a_real_cli_record_is_reparsed(self):
        self.seats = (('m', {'backend': 'command', 'command': FAKE, 'model': 'x'}), ('station', {'bot': 'station'}))
        out = self.play('cmd', deals=(1, 2))
        m = self.facts(out)['players']['m']
        self.assertEqual((m['audit'], m['audit_source']), ('opaque', 'linked'))    # a command CLI shows no tool record
        raw = self.match(out) / 'raw.jsonl'

        def doctor(rows):
            r = next(r for r in rows if 'stdout' in r and 'failed_try' not in r)
            r['stdout'] = '{"action": "fold"}\n'
            return rows
        self.edit(raw, doctor)
        self.assertEqual(self.facts(out)['players']['m']['audit'], 'incomplete')

    def test_older_records_keep_their_labels(self):
        out = self.play()
        for label in ('a-first', 'b-first'):
            cfg = self.match(out, label) / 'config.json'
            doc = json.loads(cfg.read_text())
            del doc['record_version']
            cfg.write_text(json.dumps(doc))
        m = self.facts(out)['players']['m']
        self.assertEqual((m['audit'], m['audit_source']), ('complete', 'labels'))


class TestCommitted(Linked):
    def test_rows_past_the_commit_are_provisional(self):
        m = self.d / 'one'
        asyncio.run(run_match(self.G, list(self.seats), [1, 2, 3], m, **QUIET))
        st = json.loads((m / 'state.json').read_text())
        self.assertEqual(st['log_lines']['hands.jsonl'], 3)
        self.assertEqual(st['log_lines']['calls.jsonl'], len(self.rows(m / 'calls.jsonl')))
        full = committed_records(m)
        self.assertEqual((full['hands_committed'], full['finished'], len(full['hands'])), (3, True, 3))
        self.assertEqual(sum(map(len, full['provisional'].values())), 0)
        st['hands'] = st['hands_committed'] = 2
        (m / 'state.json').write_text(json.dumps(st))
        (m / 'result.json').unlink()
        part = committed_records(m)
        self.assertEqual((part['hands_committed'], part['finished'], len(part['hands'])), (2, False, 2))
        self.assertTrue(part['provisional']['events'] and all(e['episode'] == 2 for e in part['provisional']['events']))
        self.assertTrue(all(c['hand'] < 2 for c in part['calls']))


    def test_the_judge_reads_the_conversation_as_every_reader_does(self):
        out = self.play()
        rows = committed_records(self.match(out))
        self.assertEqual(rows['conversation'], {'verified': [0, 1, 2], 'provisional': [], 'rejected': []})
        self.assertTrue(self.facts(out)['verification']['ok'])
        cfg = self.match(out, 'b-first') / 'config.json'
        doc = json.loads(cfg.read_text())
        doc['specs']['station'] = {**doc['specs']['station'], 'note': 'edited'}   # the hands still replay
        cfg.write_text(json.dumps(doc))
        self.assertEqual(committed_records(self.match(out))['conversation']['rejected'], 'all')
        f = self.facts(out)
        self.assertFalse(f['verification']['ok'])
        self.assertTrue(any('conversation reader' in p for p in f['problems']))


class TestCliMidRun(Linked):
    seats = (('m', {'backend': 'command', 'command': FAKE, 'model': 'x'}), ('station', {'bot': 'station'}))

    def flip_after(self, n):
        real, seen = backends.cli_artifact, []

        def art(exe):
            a = real(exe)
            seen.append(exe)
            return {**a, 'sha256': 'upgraded'} if a and len(seen) > n else a
        return mock.patch.object(backends, 'cli_artifact', art)

    def test_a_cli_that_changes_while_a_match_runs_is_recorded(self):
        m = self.d / 'one'
        with self.flip_after(3):
            asyncio.run(run_match(self.G, list(self.seats), [1, 2, 3], m, **QUIET))
        cfg = json.loads((m / 'config.json').read_text())
        moved = [c for c in cfg['cli_version_changes'] if c['kind'] == 'mid_run']
        self.assertEqual(len(moved), 1)
        self.assertIn('the file changed', moved[0]['clis'][0])
        self.assertEqual(moved[0]['artifact_to']['sha256'], 'upgraded')
        self.assertEqual(cfg['cli_artifacts'][FAKE[0]]['sha256'], 'upgraded')
        calls = self.rows(m / 'calls.jsonl')
        self.assertEqual(calls[-1]['cli']['sha256'], 'upgraded')

    def test_it_voids_the_registration(self):
        players = self.d / 'players.toml'
        players.write_text(f'[players.m]\nbackend = "command"\ncommand = {json.dumps(FAKE)}\nmodel = "x"\n'
                           '[players.station]\nbot = "station"\n')
        plan = make_plan(['m', 'station'], deals=2, out=str(self.d / 'plan'), players_file=str(players),
                         design={'max_rounds': 1})
        decide = lambda pd, d, n, k, e: {'verdict': 'exhausted', 'looks_used': k}
        with self.flip_after(4), mock.patch('poker_arena.ratings.adjudicate', side_effect=decide, create=True):
            asyncio.run(run_plan(plan, **QUIET))
        reg = json.loads((self.d / 'plan' / 'plan.json').read_text())['registration']
        self.assertEqual(reg['status'], 'void')
        self.assertIn('changed while a match was running', reg['void_reason'])
        f = runner.eligibility(self.d / 'plan', ['m__vs__station'], ('m', 'station'), reg)
        self.assertTrue(any(c.get('mid_run') for c in f['implementation_changes']))


class TestReplayedDecisions(Linked):
    """R7-#2: the accepted reply is replayed against the engine state before the action and must produce exactly the
    executed record (action, amount, speech, correction); a refused reply must still be refused."""

    def script(self, first, then=None):
        from test_runner import FakeCLI, passive

        def run(n, text):
            if n == 1:
                return first
            return then(n, text) if then else passive(text)
        FakeCLI.script = staticmethod(run)

    def doctor_reply(self, out, call_id, fn):
        d = self.match(out)

        def calls(rows):
            for r in rows:
                if r['call_id'] == call_id:
                    r['output'] = json.dumps(fn(json.loads(r['output'])))
            return rows

        def raw(rows):
            for r in rows:
                if r.get('call_id') == call_id and 'failed_try' not in r:
                    r['synthetic']['text'] = json.dumps(fn(json.loads(r['synthetic']['text'])))
            return rows
        self.edit(d / 'calls.jsonl', calls)
        self.edit(d / 'raw.jsonl', raw)

    def first_thought(self, out):
        return next(t for t in self.thoughts(self.match(out)) if t['player'] == 'm')

    def test_a_reply_raising_to_4_does_not_verify_an_executed_raise_to_200(self):
        self.script({'action': 'raise', 'amount': 200})
        out = self.play(deals=(1,))
        t = self.first_thought(out)
        self.assertEqual((t['action'], t['amount']), ('raise', 200))
        self.assertEqual(self.facts(out)['players']['m']['audit'], 'complete')
        self.doctor_reply(out, t['accepted_call'], lambda o: {**o, 'amount': 4})
        m = self.facts(out)['players']['m']
        self.assertEqual(m['audit'], 'incomplete')
        self.assertEqual(m['audit_counts'].get('inconsistent'), 1)
        self.assertTrue(any('amount' in p and t['decision_id'] in p for p in m['problems']), m['problems'])

    def test_the_speech_is_bound_too(self):
        self.script({'action': 'call', 'say': 'hello'})
        out = self.play(deals=(1,))
        self.assertEqual(self.facts(out)['players']['m']['audit'], 'complete')
        t = self.first_thought(out)
        self.doctor_reply(out, t['accepted_call'], lambda o: {**o, 'say': 'goodbye'})
        m = self.facts(out)['players']['m']
        self.assertEqual(m['audit'], 'incomplete')
        self.assertTrue(any('say' in p for p in m['problems']), m['problems'])

    def test_a_sizing_correction_is_replayed(self):
        self.script({'action': 'raise', 'amount': 3})
        out = self.play(deals=(1,))
        t = self.first_thought(out)
        self.assertEqual((t['amount'], t['correction']['kind']), (4, 'sizing'))
        self.assertEqual(self.facts(out)['players']['m']['audit'], 'complete')

        def strip(rows):
            for r in rows:
                acts = r['actions'] if 'actions' in r else [r['data']] if r.get('kind') == 'action' else []
                for a in acts:
                    if a.get('decision_id') == t['decision_id']:
                        for k in ('correction', 'requested', 'note'):
                            a.pop(k, None)
            return rows
        d = self.match(out)
        self.edit(d / 'events.jsonl', strip)
        self.edit(d / 'hands.jsonl', strip)
        f = self.facts(out)
        self.assertTrue(f['verification']['ok'])            # the hand itself still replays
        m = f['players']['m']
        self.assertEqual(m['audit'], 'incomplete')
        self.assertTrue(any('correction' in p for p in m['problems']), m['problems'])

    def test_the_decision_record_is_recomputed(self):
        self.script({'action': 'raise', 'amount': 3})
        out = self.play(deals=(1,))
        t = self.first_thought(out)

        def relabel(rows):
            for r in rows:
                if r.get('kind') == 'thought' and r['data']['decision_id'] == t['decision_id']:
                    r['data']['correction'] = None
                    r['data']['requested'] = r['data']['accepted']
            return rows
        self.edit(self.match(out) / 'events.jsonl', relabel)
        m = self.facts(out)['players']['m']
        self.assertEqual(m['audit'], 'incomplete')
        self.assertTrue(any('decision record' in p for p in m['problems']), m['problems'])

    def test_a_refused_reply_must_still_be_refused(self):
        self.script({'action': 'check'})        # facing the big blind: not playable, asked again, then calls
        out = self.play(deals=(1,))
        t = self.first_thought(out)
        self.assertEqual((t['tries'], t['correction']['kind']), (2, 'retry'))
        self.assertEqual(self.facts(out)['players']['m']['audit'], 'complete')
        self.doctor_reply(out, t['call_ids'][0], lambda o: {**o, 'action': 'call'})
        m = self.facts(out)['players']['m']
        self.assertEqual(m['audit'], 'incomplete')
        self.assertTrue(any('refused' in p for p in m['problems']), m['problems'])

    def test_a_fallback_is_the_engine_fallback(self):
        from test_runner import FakeCLI
        FakeCLI.script = staticmethod(lambda n, text: {'action': 'dance'})
        out = self.play(deals=(1,))
        t = self.first_thought(out)
        self.assertTrue(t['fallback'])
        self.assertEqual(self.facts(out)['players']['m']['audit'], 'complete')


class TestResumeCliChange(Linked):
    def test_a_resume_records_the_file_it_replaces(self):
        before = {'codex': {'path': '/bin/codex', 'version': '1', 'sha256': 'a'}}
        after = {'codex': {'path': '/bin/codex', 'version': '1', 'sha256': 'b'}}
        ch = runner.check_versions({'cli_versions': {'codex': '1'}, 'cli_artifacts': before}, {'codex': '1'}, 'm',
                                   True, after)
        self.assertEqual((ch['artifacts_from'], ch['artifacts_to']), (before, after))
