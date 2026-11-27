"""The record is checked, not trusted: every committed hand replays from its seed, every action has exactly one
decision record with its calls, protocols must be present, identity is judged by whether the requested model
answered, and a CLI version change refuses a resume (or, when accepted, voids the registration)."""
import asyncio
import json
from unittest import mock

from poker_arena import backends, runner
from poker_arena.runner import ResumeMismatch, eligibility, make_plan, run_duplicate, run_match, run_plan, verify_match
from test_runner import QUIET, Attesting, Base


class Facts(Base):
    seats = (('m', {'backend': 'fakecli', 'model': 'fake-1'}), ('station', {'bot': 'station'}))

    def play(self, name='run', deals=(1, 2, 3)):
        out = self.d / name
        asyncio.run(run_duplicate(self.G, *self.seats, list(deals), out, **QUIET))
        return out

    def facts(self, out, folders=('m__vs__station',), reg=None):
        return eligibility(out, list(folders), ('m', 'station'), reg or {'status': 'registered'})

    def edit(self, path, fn):
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows = fn(rows)
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class TestVerification(Facts):
    def test_a_clean_record_verifies_and_has_an_evidence_hash(self):
        out = self.play()
        f = self.facts(out)
        v = f['verification']
        self.assertTrue(v['ok'])
        self.assertEqual((v['matches'], v['hands'], v['verified'], v['failed'], v['problems']), (2, 6, 6, [], []))
        self.assertEqual(len(v['evidence_hash']), 64)
        self.assertEqual(f['problems'], [])
        self.assertEqual(self.facts(out)['verification']['evidence_hash'], v['evidence_hash'])

    def test_a_doctored_hand_fails_and_changes_the_hash(self):
        out = self.play()
        before = self.facts(out)['verification']['evidence_hash']
        hands = out / 'm__vs__station' / 'b-first' / 'hands.jsonl'

        def doctor(rows):
            rows[1]['net'] = [200000, -200000]
            rows[1]['hole'][0] = ['As', 'As']
            return rows
        self.edit(hands, doctor)
        f = self.facts(out)
        self.assertFalse(f['verification']['ok'])
        self.assertEqual([(x['match'], x['hand']) for x in f['verification']['failed']], [('m__vs__station/b-first', 1)])
        self.assertNotEqual(f['verification']['evidence_hash'], before)
        self.assertTrue(any('b-first hand 1' in p for p in f['problems']))
        self.assertFalse(verify_match(hands.parent)['ok'])

    def test_empty_evidence_never_passes(self):
        out = self.d / 'nothing'
        (out / 'm__vs__station' / 'a-first').mkdir(parents=True)
        f = self.facts(out)
        self.assertFalse(f['verification']['ok'])
        self.assertTrue(f['problems'])
        self.assertEqual(f['players']['m']['audit'], 'incomplete')
        self.assertFalse(f['players']['m']['protocol_consistent'])

    def test_a_hand_that_does_not_replay_is_not_committed(self):
        m = self.d / 'm'
        with mock.patch.object(type(self.G), 'verify', return_value={'ok': False, 'errors': ['tampered']}), \
                self.assertRaisesRegex(RuntimeError, 'does not replay'):
            asyncio.run(run_match(self.G, list(self.seats), [1, 2], m, **QUIET))
        self.assertEqual(self.rows(m / 'hands.jsonl'), [])
        self.assertFalse((m / 'state.json').exists())

    def test_resume_refuses_a_damaged_record(self):
        m = self.d / 'm'
        asyncio.run(run_match(self.G, list(self.seats), [1, 2, 3], m, **QUIET))
        (m / 'result.json').unlink()
        st = json.loads((m / 'state.json').read_text())
        st['hands'] = st['hands_committed'] = 2
        (m / 'state.json').write_text(json.dumps(st))
        self.edit(m / 'hands.jsonl', lambda rows: [{**rows[0], 'board': rows[0]['board'][::-1]}] + rows[1:])
        with self.assertRaisesRegex(RuntimeError, 'fail verification'):
            asyncio.run(run_match(self.G, list(self.seats), [1, 2, 3], m, **QUIET))


class TestCoverage(Facts):
    def test_a_complete_record(self):
        m = self.facts(self.play())['players']['m']
        c = m['coverage']
        self.assertTrue(c['complete'])
        self.assertEqual(c['actions'], c['decisions'])
        self.assertEqual(c['calls_expected'], c['calls_recorded'])
        self.assertEqual((m['audit'], m['problems']), ('complete', []))

    def test_hidden_decisions_fail_closed(self):
        out = self.play()
        for label in ('a-first', 'b-first'):
            self.edit(out / 'm__vs__station' / label / 'events.jsonl',
                      lambda rows: [r for r in rows if r['kind'] != 'thought'])
        f = self.facts(out)
        m, st = f['players']['m'], f['players']['station']
        self.assertEqual((m['decisions'], m['audit']), (0, 'incomplete'))
        self.assertEqual(st['audit'], 'incomplete')
        self.assertEqual(m['coverage']['missing'], m['coverage']['actions'])
        self.assertTrue(any('without a decision record' in p for p in m['problems']))
        self.assertTrue(f['problems'])

    def test_duplicate_orphan_and_missing_calls(self):
        out = self.play()
        ev = out / 'm__vs__station' / 'a-first' / 'events.jsonl'
        thought = next(r for r in self.rows(ev) if r['kind'] == 'thought' and r['data']['player'] == 'm')
        self.edit(ev, lambda rows: rows + [thought])
        self.edit(out / 'm__vs__station' / 'b-first' / 'calls.jsonl', lambda rows: rows[1:])
        m = self.facts(out)['players']['m']
        self.assertEqual(m['coverage']['orphan'] + m['coverage']['duplicate'], 1)
        self.assertEqual(m['coverage']['calls_recorded'], m['coverage']['calls_expected'] - 1 - thought['data']['tries'])
        self.assertEqual(m['audit'], 'incomplete')

    def test_a_missing_protocol_fails(self):
        out = self.play()
        cfg = out / 'm__vs__station' / 'a-first' / 'config.json'
        doc = json.loads(cfg.read_text())
        del doc['protocol']['m']
        cfg.write_text(json.dumps(doc))
        m = self.facts(out)['players']['m']
        self.assertEqual((m['protocol_consistent'], m['protocols_missing']), (False, 1))
        self.assertTrue(any('no protocol fingerprint' in p for p in m['problems']))


class TestIdentityMatch(Facts):
    def run_with(self, answer):
        backends.BACKENDS['attesting'] = Attesting
        self.addCleanup(backends.BACKENDS.pop, 'attesting', None)
        Attesting.ANSWER = answer
        self.seats = (('m', {'backend': 'attesting', 'model': 'claude-sonnet-5-5'}), ('station', {'bot': 'station'}))
        out = self.play(str(answer))
        return out, self.facts(out)['players']['m']

    def test_the_wrong_model_is_a_mismatch_everywhere(self):
        out, m = self.run_with('claude-haiku-5-5')
        self.assertEqual((m['identity'], m['identity_match']), ('mismatch', 'mismatch'))
        self.assertEqual(m['identity_match_counts'], {'mismatch': m['replies']})
        d = out / 'm__vs__station' / 'a-first'
        call = self.rows(d / 'calls.jsonl')[0]
        self.assertEqual((call['identity_reported'], call['identity_match']), (True, 'mismatch'))
        t = next(e['data'] for e in self.rows(d / 'events.jsonl') if e['kind'] == 'thought' and e['data']['player'] == 'm')
        self.assertEqual((t['identity_reported'], t['identity_match']), (True, 'mismatch'))

    def test_the_requested_model_matches(self):
        _, m = self.run_with(None)
        self.assertEqual((m['identity'], m['identity_match']), ('attested', 'match'))


class TestCliVersions(Facts):
    def versions(self, v):
        return mock.patch.object(runner, 'cli_version', lambda exe: v if exe else None)

    def test_resume_refuses_a_new_cli_version_unless_allowed(self):
        self.seats = (('m', {'backend': 'command', 'command': ['fake-cli'], 'model': 'x'}), ('station', {'bot': 'station'}))
        m = self.d / 'm'
        cfg = {'players': ['m', 'station'], 'cli_versions': {'fake-cli': '1.0'}}
        with self.versions('2.0'):
            self.assertIsNone(runner.check_versions({**cfg, 'cli_versions': {'fake-cli': '2.0'}}, {'fake-cli': '2.0'}, m, False))
            with self.assertRaisesRegex(ResumeMismatch, r'fake-cli: 1.0 -> 2.0.*--allow-code-change'):
                runner.check_versions(cfg, {'fake-cli': '2.0'}, m, False)
            change = runner.check_versions(cfg, {'fake-cli': '2.0'}, m, True)
        self.assertEqual((change['clis'], change['to']), (['fake-cli: 1.0 -> 2.0'], {'fake-cli': '2.0'}))

    def test_a_plan_pins_versions_and_an_accepted_change_voids_the_registration(self):
        players = self.d / 'players.toml'
        players.write_text('[players.m]\nbackend = "fakecli"\nmodel = "fake-1"\n[players.station]\nbot = "station"\n')
        plan = make_plan(['m', 'station'], deals=2, out=str(self.d / 'plan'), players_file=str(players),
                         design={'max_rounds': 2})
        decide = lambda pd, d, n, k, e: {'verdict': 'exhausted' if k >= 2 else 'open', 'looks_used': k}
        from test_runner import FakeCLI
        with mock.patch.object(FakeCLI, 'exe', 'fake-cli'), \
                mock.patch('poker_arena.ratings.adjudicate', side_effect=decide, create=True):
            with self.versions('1.0'):
                asyncio.run(run_plan(plan, only=['nobody'], **QUIET))
            stored = json.loads((self.d / 'plan' / 'plan.json').read_text())
            self.assertEqual((stored['cli_versions'], stored['registration']['cli_versions']),
                             ({'fake-cli': '1.0'}, {'fake-cli': '1.0'}))
            with self.versions('1.1'):
                with self.assertRaisesRegex(ResumeMismatch, 'fake-cli: 1.0 -> 1.1'):
                    asyncio.run(run_plan(plan, **QUIET))
                asyncio.run(run_plan(plan, allow_code_change=True, **QUIET))
                stored = json.loads((self.d / 'plan' / 'plan.json').read_text())
                self.assertEqual(stored['registration']['status'], 'void')
                self.assertIn('fake-cli: 1.0 -> 1.1', stored['registration']['void_reason'])
                self.assertEqual(stored['cli_version_changes'][0]['clis'], ['fake-cli: 1.0 -> 1.1'])
                asyncio.run(run_plan(plan, **QUIET))          # the accepted version resumes freely
            look = json.loads((self.d / 'plan' / 'decisions.json').read_text())['m__vs__station'][-1]
            self.assertEqual(len(look['evidence_hash']), 64)
            self.assertTrue(look['verification']['ok'])

    def test_drift_inside_a_pairing_is_listed_as_a_change(self):
        out = self.play()
        cfg = out / 'm__vs__station' / 'a-first' / 'config.json'
        doc = json.loads(cfg.read_text())
        doc['cli_versions'] = {'fake-cli': '1.0'}
        cfg.write_text(json.dumps(doc))
        f = self.facts(out, reg={'status': 'registered', 'cli_versions': {'fake-cli': '0.9'}})
        self.assertEqual([c['files'] for c in f['implementation_changes'] if c.get('kind') == 'cli_version'],
                         [['CLI fake-cli: 0.9 -> 1.0']])
