"""Round 5 review probes, judge side: an official verdict counts towards a current leader or rank only while
today's evidence check passes, and says why when it does not (lost or changed evidence, an altered record, or
stricter checks today) (R5-#1); every counted hand replays from its seed through the engine's record verifier,
so tampered cards or payoffs never count, here or in the history (R5-#4); the history splits a player at a CLI
version or code change, even mid-match (R5-#3); one canonical model matcher decides identity everywhere
(R5-#5). The real records come from a registered bot tournament played by the runner."""
import asyncio
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from poker_arena import history, ratings as R, stats
from poker_arena.runner import make_plan, run_plan
from test_adjudicate import stand_in
from test_history import SPECS, write_matchup
from test_adjudicate import registered
from test_ratings import MODEL, deals_between, facts, plan, run

PLAYERS = '[players.rule]\nbot = "rule"\nsims = 30\n[players.station]\nbot = "station"\n'


def rows(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def write_rows(p: Path, xs):
    p.write_text(''.join(json.dumps(x) + '\n' for x in xs))


class RealRun(unittest.TestCase):
    """A registered tournament of two bots, played once: rule against a calling station at 10-big-blind stacks
    (so the official test's bound is +-1,000 bb/100 and a verdict takes a few dozen conversations of 10 deals),
    decided before its last look. Each test works on its own copy."""

    _played = None

    @classmethod
    def setUpClass(cls):
        if RealRun._played is None:
            base = tempfile.TemporaryDirectory()
            root = Path(base.name)
            (root / 'players.toml').write_text(PLAYERS)
            p = make_plan([], deals=10, out=str(root / 'run'), players_file=str(root / 'players.toml'),
                          matchups=[['rule', 'station']], design={'max_rounds': 14}, replicates=5,
                          game_config={'stack': 20})
            asyncio.run(run_plan(p, log=lambda *a: None))
            RealRun._played = base
            unittest.addModuleCleanup(base.cleanup)
        cls.pristine = Path(RealRun._played.name)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'copy'
        shutil.copytree(self.pristine, self.root)
        self.run_dir = self.root / 'run'
        R._VERIFIED.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def pairing(self, r=None):
        r = r or R.ratings(self.run_dir, boot=20)
        [m] = r['pairings']
        return r, m

    def folder(self, r=1, x=1):
        name = 'rule__vs__station' + (f'__x{x}' if x > 1 else '') + (f'__r{r}' if r > 1 else '')
        return self.run_dir / name


class TestCurrentEvidence(RealRun):
    """R5-#1."""

    def test_untouched_run_counts(self):
        r, m = self.pairing()
        self.assertEqual((m['verdict'], m['official'], m['evidence_now']['status']), ('a', True, 'verified'))
        self.assertTrue(m['counts_toward_leader'])
        self.assertEqual(r['leader']['player'], 'rule')
        rec = json.loads((self.run_dir / 'decisions.json').read_text())['rule__vs__station'][-1]
        self.assertEqual(rec['judged_evidence_hash'], R.pairing_data(self.run_dir, m['matchups'], 'rule')['evidence_hash'])

    def test_lost_evidence_stops_counting(self):
        shutil.rmtree(self.folder(2, 3))
        r, m = self.pairing()
        self.assertEqual((m['verdict'], m['official'], m['recorded_counts']), ('a', True, True))
        self.assertEqual((m['evidence_now']['status'], m['evidence_now']['kind']), ('lost', 'evidence'))
        self.assertFalse(m['counts_toward_leader'])
        self.assertIsNone(r['leader'])
        self.assertTrue(all(p['rank'] is None for p in r['players']))
        self.assertTrue(any('no longer' in n and 'lost' in n for n in r['notes']), r['notes'])

    def test_the_reviewers_probe_recomputed_ineligible(self):
        """official=True, eligible=True and counts_toward_leader=True with a recomputed verdict of ineligible."""
        from poker_arena import runner
        real = runner.eligibility

        def mismatch(*a, **k):
            f = real(*a, **k)
            f['players']['rule'] = {**f['players']['rule'], 'kind': 'model', 'backend': 'claude',
                                    'requested_model': 'claude-x-1', 'observed_models': ['claude-y-1'],
                                    'identity': 'attested', 'identity_match': 'match', 'problems': []}
            return f
        with mock.patch.object(runner, 'eligibility', mismatch):
            r, m = self.pairing()
        self.assertEqual(m['recomputed']['verdict'], 'ineligible')
        self.assertTrue(m['official'])
        self.assertFalse(m['counts_toward_leader'])
        self.assertEqual(m['evidence_now']['kind'], 'evidence')       # the recorded facts differ: evidence changed
        self.assertIsNone(r['leader'])

    def test_stricter_checks_today_are_a_rule_change(self):
        from poker_arena import runner
        real = runner.eligibility

        def stricter(*a, **k):
            f = real(*a, **k)
            f['players']['station']['problems'] = ['station: a check added after the run']
            return f
        with mock.patch.object(runner, 'eligibility', stricter), \
                mock.patch.object(R, '_summary_facts', lambda facts: {}):
            r, m = self.pairing()
        self.assertEqual((m['evidence_now']['status'], m['evidence_now']['kind']), ('fails_today', 'rules'))
        self.assertFalse(m['counts_toward_leader'])

    def test_hidden_decision_records(self):
        """The round 5 probe that hid thought events: the records behind the decision changed."""
        d = self.folder(1, 1) / 'a-first' / 'events.jsonl'
        write_rows(d, [e for e in rows(d) if e.get('kind') != 'thought'])
        r, m = self.pairing()
        self.assertFalse(m['counts_toward_leader'])
        self.assertNotEqual(m['evidence_now']['status'], 'verified')
        self.assertIsNone(r['leader'])

    def test_an_altered_record_does_not_reproduce(self):
        path = self.run_dir / 'decisions.json'
        doc = json.loads(path.read_text())
        doc['rule__vs__station'][-1].update(verdict='b', test_verdict='b')
        path.write_text(json.dumps(doc))
        r, m = self.pairing()
        self.assertEqual((m['verdict'], m['evidence_now']['status']), ('b', 'altered'))
        self.assertFalse(m['counts_toward_leader'])
        self.assertIsNone(r['leader'])

    def test_a_record_without_an_evidence_hash_is_unbound(self):
        path = self.run_dir / 'decisions.json'
        doc = json.loads(path.read_text())
        for v in doc['rule__vs__station']:
            v.pop('evidence_hash', None)
            v.pop('judged_evidence_hash', None)
        path.write_text(json.dumps(doc))
        r, m = self.pairing()
        self.assertEqual(m['evidence_now']['status'], 'unbound')
        self.assertFalse(m['counts_toward_leader'])

    def test_a_new_judge_keeps_the_verdict(self):
        """Another judge today is a rule change of the test, not of the evidence: the official verdict stands."""
        with mock.patch.object(R, 'judge_fingerprint', return_value='n3wjudge00000000'):
            r, m = self.pairing()
        self.assertEqual((m['official'], m['evidence_now']['status']), (True, 'verified'))
        self.assertTrue(m['counts_toward_leader'])


class TestRecordVerifier(RealRun):
    """R5-#4: the probe that duplicated an ace and paid 200,000 chips from 200-chip stacks."""

    def tamper(self, k=3):
        for side in ('a-first', 'b-first'):
            p = self.folder(1, 2) / side / 'hands.jsonl'
            hs = rows(p)
            h = hs[k]
            h['hole'][1] = list(h['hole'][0])
            h['net'] = [200_000, -200_000] if side == 'a-first' else [-200_000, 200_000]
            write_rows(p, hs)
        return k

    def test_tampered_hands_never_count(self):
        k = self.tamper()
        report = []
        deals = R.load_deals(self.folder(1, 2), report)
        self.assertNotIn(k, [d['deal'] for d in deals])
        self.assertTrue(any(x['hand'] == k and 'replay' in x['reason'] for x in report), report)
        self.assertTrue(all(abs(d['net_a']) <= 2 * 200 for d in R.load_deals(self.run_dir)))
        data = R.pairing_data(self.run_dir, [self.folder(1, 2).name], 'rule')
        self.assertTrue(data['rejected'])
        r, m = self.pairing()
        self.assertEqual(m['evidence_now']['status'], 'invalid')
        self.assertFalse(m['counts_toward_leader'])
        self.assertIsNone(r['leader'])
        self.assertTrue(r['rejected'])

    def test_adjudicate_rejects_them(self):
        self.tamper()
        d = R.normalize_design(json.loads((self.run_dir / 'plan.json').read_text()))
        reg = json.loads((self.run_dir / 'plan.json').read_text())['registration']
        folders = sorted(p.name for p in self.run_dir.iterdir() if p.name.startswith('rule__vs__'))
        v = R.adjudicate({'root': self.run_dir, 'conversations': folders, 'a': 'rule', 'b': 'station'}, d, 1, 3,
                         facts(reg, ('rule', 'station')))
        self.assertEqual(v['verdict'], 'ineligible')
        self.assertTrue(any('replay' in x for x in v['ineligible_reasons']))

    def test_the_history_rejects_them(self):
        k = self.tamper()
        history.sync(self.root)
        con = sqlite3.connect(self.root / history.DB_NAME)
        bad = con.execute('SELECT deal, reason FROM rejected').fetchall()
        big = con.execute('SELECT max(abs(net_a)) FROM deals').fetchone()[0]
        con.close()
        self.assertIn(k, [x[0] for x in bad])
        self.assertLessEqual(big, 400)

    def test_style_totals_count_verified_results_only(self):
        """R6-#10: one reader decides which hands are results; the style statistics use it, so a tampered hand
        never reaches a style total, and each player's stats say which population they cover."""
        before = stats.collect(self.run_dir)['rule']
        self.tamper()
        R._VERIFIED.clear()
        after = stats.collect(self.run_dir)['rule']
        self.assertEqual(after['hands'], before['hands'] - 2)          # the hand, in both plays of its deal
        self.assertEqual(after['population']['rejected_hands'], 2)
        self.assertIn('verified results only', after['population']['counted'])
        v = R.verified_hands(self.folder(1, 2) / 'a-first')
        self.assertNotIn(3, v['verified'])
        self.assertEqual(v['rejected'], [3])

    def test_uncommitted_rows_do_not_count(self):
        d = self.folder(1, 1) / 'a-first'
        state = json.loads((d / 'state.json').read_text())
        state['hands_committed'] = 5
        (d / 'state.json').write_text(json.dumps(state))
        self.assertEqual(sorted(x['deal'] for x in R.load_deals(self.folder(1, 1))), [0, 1, 2, 3, 4])

    def test_uncommitted_rows_are_provisional(self):
        d = self.folder(1, 1) / 'a-first'
        state = json.loads((d / 'state.json').read_text())
        state['hands_committed'] = 5
        (d / 'state.json').write_text(json.dumps(state))
        conv = R.read_conversation(self.folder(1, 1))
        self.assertEqual(conv['verified'], [0, 1, 2, 3, 4])
        self.assertEqual(conv['provisional']['a-first'], [5, 6, 7, 8, 9])
        self.assertEqual(conv['provisional']['b-first'], [5, 6, 7, 8, 9])
        self.assertEqual(stats.collect(self.folder(1, 1))['rule']['population']['provisional_hands'], 10)

    def test_every_written_hand_has_one_status(self):
        """R7-#7: the shared reader says verified, provisional or rejected for every hand either play wrote, and
        totals over the verified ones, for the viewer to use without the ratings."""
        k = self.tamper()
        d = self.folder(1, 2) / 'a-first'
        state = json.loads((d / 'state.json').read_text())
        state['hands_committed'] = 8
        (d / 'state.json').write_text(json.dumps(state))
        R._VERIFIED.clear()
        conv = R.read_conversation(self.folder(1, 2))
        st = conv['status']
        self.assertEqual(st['a-first'][k], 'rejected')
        self.assertEqual(st['b-first'][k], 'rejected')
        self.assertEqual((st['a-first'][8], st['b-first'][9]), ('provisional', 'provisional'))
        self.assertEqual(sorted(h for h, s in st['a-first'].items() if s == 'verified'), conv['verified'])
        self.assertIn('replay', conv['reasons'][k])
        ha, hb = conv['hands']
        self.assertEqual(conv['totals'], {'deals': len(conv['verified']),
                                          'net_a': sum(ha[h]['net'][0] + hb[h]['net'][1] for h in conv['verified'])})

    def test_a_missing_verifier_fails_closed(self):
        with mock.patch('poker_arena.games.holdem.Holdem.verify', side_effect=RuntimeError('broken')):
            report = []
            self.assertEqual(R.load_deals(self.folder(1, 1), report), [])
        self.assertTrue(report and all('verifier failed' in x['reason'] for x in report))

    def test_adjudicate_without_verified_evidence_is_ineligible(self):
        p = {**plan(['x', 'y'], 50, 4, reps=2)}
        p['registration'] = R.registration(R.normalize_design(p), created=0)
        rng = np.random.default_rng(3)
        ds = deals_between('x', 'y', 300, 50, rng, sd=100, rounds=4, reps=2)
        data = {'a': 'x', 'b': 'y', 'xs': [d['net_a'] / 4 * 100 for d in ds], 'groups': [d['matchup'] for d in ds],
                'rejected': []}
        v = R.adjudicate(data, R.normalize_design(p), 1, 4, facts(p['registration'], 'xy'))
        self.assertEqual(v['verdict'], 'ineligible')
        self.assertTrue(any('not verified' in x for x in v['ineligible_reasons']))
        self.assertEqual(R.adjudicate({**data, 'xs': [], 'groups': [], 'evidence_hash': 'h'}, R.normalize_design(p), 1,
                                      4, facts(p['registration'], 'xy'))['verdict'], 'ineligible')


class TestLinkedAudit(unittest.TestCase):
    """R6-#3, judge side: each accepted decision's audit is derived from the records that carry its
    decision_id (the action, every call attempt, the raw output rows), and the evidence an official decision is
    bound to covers the call and raw records too."""
    CFG = {'players': ['m', 'bot'], 'specs': {'m': {'backend': 'claude', 'model': 'claude-x-1'}, 'bot': {'bot': 'rule'}}}

    def records(self):
        events = [{'kind': 'thought', 'episode': 0, 'data': {'seat': 0, 'player': 'm', 'decision_id': 'd1', 'audit': 'verified'}},
                  {'kind': 'action', 'episode': 0, 'data': {'seat': 0, 'decision_id': 'd1'}},
                  {'kind': 'thought', 'episode': 0, 'data': {'seat': 1, 'player': 'bot'}},
                  {'kind': 'action', 'episode': 0, 'data': {'seat': 1}}]
        calls = [{'hand': 0, 'seat': 0, 'player': 'm', 'decision_id': 'd1', 'attempt': 0, 'audit': 'unverified'},
                 {'hand': 0, 'seat': 0, 'player': 'm', 'decision_id': 'd1', 'attempt': 1, 'audit': 'verified'}]
        raw = [{'hand': 0, 'seat': 0, 'decision_id': 'd1', 'attempt': 0, 'stdout': '...'},
               {'hand': 0, 'seat': 0, 'decision_id': 'd1', 'attempt': 1, 'stdout': '...'}]
        return events, calls, raw

    def test_derived_from_linked_attempts(self):
        events, calls, raw = self.records()
        self.assertEqual(R.linked_audit(self.CFG, 0, events, calls, raw)[0],
                         {'decisions': 1, 'linked': 1, 'verified': 1, 'opaque': 0, 'problems': []})
        x = R.linked_audit(self.CFG, 0, events, calls, raw[:1] + [{**raw[1], 'failed_try': 0}])[0]   # only a failed try's
        self.assertEqual((x['verified'], len(x['problems'])), (0, 1))
        x = R.linked_audit(self.CFG, 0, events, [{**c, 'decision_id': 'other'} for c in calls], raw)[0]
        self.assertEqual(x['linked'], 0)
        self.assertNotIn(1, R.linked_audit(self.CFG, 0, events, calls, raw))   # bots make no audited decisions
        bad = R.linked_status([{'audit': {'m': x}}])['m']
        self.assertEqual(bad['status'], 'incomplete')
        legacy = R.linked_status([{'audit': {'m': {'decisions': 3, 'linked': 0, 'verified': 0, 'opaque': 0, 'problems': []}}}])
        self.assertEqual(legacy['m']['status'], 'legacy')

    def test_adjudicate_rejects_an_unlinked_audit(self):
        p = registered(plan(['m', 'y'], 50, 4, reps=2))
        d = R.normalize_design(p)
        rng = np.random.default_rng(3)
        win = {'a': 'm', 'b': 'y', 'xs': list(900 + 20 * rng.standard_normal(400)),
               'groups': list(np.repeat(np.arange(8), 50)), 'rejected': [], 'evidence_hash': 'h',
               'linked_audit': {'m': {'status': 'incomplete', 'problems': ['hand 0 decision d1: no raw output row']}}}
        v = R.adjudicate(win, d, 1, 4, facts(p['registration'], ('m', 'y'), {'m': MODEL}))
        self.assertEqual((v['verdict'], v['test_verdict']), ('ineligible', 'a'))
        self.assertTrue(any('linked call records' in r for r in v['ineligible_reasons']))
        ok = R.adjudicate({**win, 'linked_audit': {'m': {'status': 'complete'}}}, d, 1, 4,
                          facts(p['registration'], ('m', 'y'), {'m': MODEL}))
        self.assertEqual(ok['verdict'], 'a')

    def test_the_evidence_covers_calls_and_raw_output(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            events, calls, raw = self.records()
            write_rows(d / 'calls.jsonl', calls)
            write_rows(d / 'raw.jsonl', raw)
            hand = {'hand': 0, 'net': [1, -1]}
            cfg = {**self.CFG, 'seeds': [7]}
            with mock.patch.object(R, 'verify_hand', lambda *a: {'ok': True, 'errors': []}):
                first = R.verify_play(d, cfg, {0: hand})[0]
                write_rows(d / 'raw.jsonl', raw[:1])
                R._VERIFIED.clear()
                second = R.verify_play(d, cfg, {0: hand})[0]
        self.assertEqual(first['digest'], second['digest'])                   # version 1 cannot see it
        self.assertNotEqual(first['digest_linked'], second['digest_linked'])  # version 2 does


class TestPinnedVersions(RealRun):
    """R5-#3, judge side: a CLI version other than the registered one, anywhere in the run, voids the
    registration, so nothing in it is official."""

    def test_drift_voids(self):
        plan_path = self.run_dir / 'plan.json'
        doc = json.loads(plan_path.read_text())
        doc['registration'] = {**doc['registration'], 'cli_versions': {'claude': '1.0'}}
        plan_path.write_text(json.dumps(doc))
        cfg_path = self.folder(2, 1) / 'b-first' / 'config.json'
        cfg = json.loads(cfg_path.read_text())
        cfg['cli_versions'] = {'claude': '1.1'}
        cfg_path.write_text(json.dumps(cfg))
        s = R.registration_status(doc, R.run_evidence(self.run_dir))
        self.assertEqual((s['registered'], s['status']), (False, 'void'))
        self.assertIn('1.0 -> 1.1', s['reasons'][0])
        r, m = self.pairing()
        self.assertEqual(m['verdict'], 'descriptive')
        self.assertIsNone(r['leader'])


class TestFundamentals(unittest.TestCase):
    """R5-#9, presentation: the quiz is its own section, joined to the run's players by model and level, never
    mixed into strengths or verdicts; plans are told to include the reference bot."""

    def test_separate_section(self):
        """R6-#4, #5, R7-#3, #5: scores are recounted from the answers as the judge re-derives them (a fallback
        never counts, even where folding was best; valid-answer accuracy apart), carry their provenance, attach to
        a run's player only when eligible, and are counts, not population intervals, unless sampled independently."""
        from test_fundamentals import CALL, CALL_B, FOLD_B, SHOVE, answer, player, quiz_doc, reply
        fold_b, call_b = FOLD_B, CALL_B
        miss = answer(fold_b, None, 'fold', text='no JSON here', error='Your reply was not a JSON object.')
        p = player([answer(SHOVE, 'call', 'call'), miss, answer(call_b, 'fold', 'fold'), answer(CALL, 'call', 'call')])
        doc = quiz_doc({'opus': p, 'old': {'spec': {'backend': 'claude', 'model': 'claude-z-1', 'effort': 'high'},
                                           'answered': 30, 'correct': 24, 'accuracy': 0.8, 'ev_loss_bb': 12.5,
                                           'fallbacks': 0}}, spots=(SHOVE, fold_b, call_b, CALL))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'quiz.json').write_text(json.dumps(doc))
            f = R.fundamentals(root, {'m1': 'claude:claude-x-1:high#abcd1234@ffff0000',
                                      'm2': 'claude:claude-z-1:high#abcd1234@ffff0000'})
            bad = json.loads(json.dumps(doc))
            text = reply('fold')
            bad['players']['opus']['answers'][0]['calls'][0].update(output=text)
            (root / 'quiz.json').write_text(json.dumps(bad))
            g = R.fundamentals(root, {'m1': 'claude:claude-x-1:high#abcd1234@ffff0000'})
        self.assertEqual(f['label'], 'poker fundamentals')
        x = f['players']['m1']
        self.assertEqual((x['quiz_player'], x['attached'], x['correct'], x['answered'], x['fallbacks']), ('opus', True, 1, 3, 1))
        self.assertEqual((x['valid_correct'], x['valid_answers'], x['valid_accuracy']), (1, 2, 0.5))
        self.assertIsNone(x['categories']['basics']['interval95'])
        self.assertIsNone(x['categories']['close']['interval95'])
        self.assertEqual(x['provenance']['implementation'], 'i1')
        self.assertEqual(x['eligibility']['status'], 'eligible')
        self.assertNotIn('m2', f['players'])                               # no answers kept: not attached
        self.assertEqual(f['players']['old']['eligibility']['status'], 'unverified')
        self.assertNotIn('m1', g['players'])
        self.assertEqual(g['players']['opus']['eligibility']['status'], 'ineligible')
        text = R.fundamentals_text(f)
        self.assertIn('apart from match results', text)
        self.assertIn('1/3', text)
        self.assertNotIn('95%', text)
        self.assertIsNone(R.fundamentals(Path(tempfile.gettempdir()) / 'no-such-run'))

    def test_reference_edge_and_warning(self):
        rng = np.random.default_rng(12)
        deals = deals_between('m', 'ref', 80, 50, rng, sd=100, rounds=2, reps=2) + \
            deals_between('m', 'n', 0, 50, rng, sd=100, rounds=2, reps=2)
        meta = {'ref': {'identity': 'bot:reference', 'kind': 'bot', 'quarantine_reason': None, 'identity_resolved': True}}
        r = R.rate(deals, boot=20, pooled=True, meta=meta)
        m = next(p for p in r['players'] if p['player'] == 'm')
        self.assertEqual(m['vs_reference']['reference'], 'ref')
        self.assertGreater(m['vs_reference']['edge'], 0)
        self.assertTrue(r['groups'][0]['has_reference'])
        self.assertEqual(m['vs_reference']['reference_version'], R.REFERENCE['version'])
        self.assertTrue(any('not a solver' in n and '+408' in n for n in r['notes']), r['notes'])
        no = R.rate(deals_between('m', 'n', 0, 50, rng, sd=100, rounds=2, reps=2), boot=20, pooled=True)
        self.assertTrue(any('reference' in n and 'quiz' in n for n in no['notes']))

    def test_plan_warns_without_a_reference_player(self):
        import contextlib
        import io
        from poker_arena import cli
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / 'players.toml'
            pf.write_text('[players.rule]\nbot = "rule"\n[players.station]\nbot = "station"\n'
                          '[players.ref]\nbot = "reference"\n')
            for players, warned in ((['rule', 'station'], True), (['rule', 'ref'], False)):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    cli.main(['plan', *players, '--players', str(pf), '--write', str(Path(d) / 'plan.json'),
                              '--out', str(Path(d) / 'run')])
                self.assertEqual('No reference player' in out.getvalue(), warned)
                self.assertIn('official test', out.getvalue())
                self.assertIn('decided leader', out.getvalue())


class TestCliVersionSplit(unittest.TestCase):
    """R5-#3: a CLI version change splits a player's identity in the history from the hand it took effect."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(R, 'verify_hand', stand_in)
        p.start()
        self.addCleanup(p.stop)

    def edit(self, run, side, **kw):
        p = self.root / run / 'm1__vs__m2' / side / 'config.json'
        cfg = json.loads(p.read_text())
        cfg.update(kw)
        p.write_text(json.dumps(cfg))

    def ids(self):
        history.sync(self.root)
        con = sqlite3.connect(self.root / history.DB_NAME)
        out = con.execute('SELECT deal, a_identity FROM deals ORDER BY deal').fetchall()
        con.close()
        return out

    def test_change_mid_match(self):
        impl = {'poker_arena/agents.py': 'a1'}
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, impl=impl, cli='1.0')
        change = [{'after_hand': 10, 'to': {'claude': '2.0'}}]
        self.edit('one', 'a-first', cli_version_changes=change)
        self.edit('one', 'b-first', cli_version_changes=[{'after_hand': 12, 'to': {'claude': '2.0'}}])
        ids = dict(self.ids())
        before, mixed, after = ids[0], ids[11], ids[15]
        self.assertEqual(len({before, mixed, after}), 3)                 # 1.0 / one play of each / 2.0
        self.assertEqual(ids[9], before)
        self.assertEqual(ids[19], after)
        r = history.all_time(self.root / history.DB_NAME, boot=20, players_file='/none')
        self.assertEqual(len([p for p in r['players'] if p['player'].startswith('claude')]), 3)
        self.assertTrue(all(r['labels'].get(p['player']) for p in r['players']))

    def test_change_seen_only_at_a_resume(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, impl={'poker_arena/agents.py': 'a1'}, cli='1.0')
        for side in ('a-first', 'b-first'):
            (self.root / 'one' / 'm1__vs__m2' / side / 'state.json').write_text(json.dumps(
                {'resumes': [{'after_hand': 5, 'cli_versions': {'claude': '1.1'}}]}))
        ids = dict(self.ids())
        self.assertNotEqual(ids[4], ids[5])

    def test_unknown_executable_uses_every_version(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, impl={'poker_arena/agents.py': 'a1'}, cli='1.0')
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, [4, -2] * 10, seed0=3000, impl={'poker_arena/agents.py': 'a1'},
                      cli='2.0')
        for run in ('one', 'two'):
            for side in ('a-first', 'b-first'):
                self.edit(run, side, effective=[{}, {}])
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=20, players_file='/none')
        self.assertEqual(len([p for p in r['players'] if p['player'].startswith('claude')]), 2)

    def test_older_databases_are_read_again(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10)
        history.sync(self.root)
        con = sqlite3.connect(self.root / history.DB_NAME)
        con.execute("UPDATE meta SET v = '5' WHERE k = 'schema'")
        con.commit()
        con.close()
        self.assertEqual(history.sync(self.root)['matches'], 2)


class TestOneMatcher(unittest.TestCase):
    """R5-#5: `same_model` is the one rule, `identity_match` its three-way form, used by eligibility, stats and
    the ratings output."""

    def test_matcher(self):
        m = R.identity_match
        self.assertEqual(m('claude-sonnet-5-5', 'claude-sonnet-5-5-20260101'), 'match')
        self.assertEqual(m('claude-sonnet-5-5', 'claude-haiku-5-5'), 'mismatch')
        self.assertEqual(m('claude-sonnet-5', 'claude-sonnet-5-5'), 'mismatch')
        self.assertEqual(m('claude-sonnet-5-5', None), 'unattested')
        self.assertEqual(m(None, 'none'), 'mismatch')                      # nothing requested matches nothing
        self.assertEqual(m('sonnet', 'sonnet'), 'mismatch')                # a bare alias names no version
        self.assertEqual(m('', ''), 'unattested')
        self.assertFalse(R.same_model('None', 'none') and R.same_model(None, 'None'))
        self.assertEqual(m('gpt-6', ['gpt-6', 'gpt-5']), 'mismatch')
        self.assertEqual(m('gpt-6', 'gpt-6-2026-01-31'), 'match')
        self.assertEqual(m('gpt-6', 'gpt-6-99999999'), 'mismatch')          # not a date: another model
        self.assertEqual(m('gpt-6', 'gpt-6-20261340'), 'mismatch')

    def test_stats_never_trust_a_recorded_match(self):
        calls = [{'requested_model': 'claude-sonnet-5-5', 'observed_model': 'claude-haiku-5-5', 'identity': 'attested',
                  'identity_match': 'match', 'audit': 'verified'},
                 {'requested_model': 'claude-sonnet-5-5', 'observed_model': 'claude-sonnet-5-5', 'identity': 'attested',
                  'audit': 'verified'},
                 {'requested_model': 'claude-sonnet-5-5', 'audit': 'verified', 'identity': 'attested'},
                 {'requested_model': 'claude-sonnet-5-5', 'observed_model': 'claude-sonnet-5-5',
                  'identity_match': 'mismatch', 'audit': 'verified'}]
        total = {}
        for c in calls:
            stats.add(total, stats.audit_counts(c))
        s = stats.summarize(total)
        self.assertEqual(s['identity'], {'match': 1, 'mismatch': 2, 'unattested': 1})
        self.assertEqual(s['identity_match'], s['identity'])
        self.assertEqual(s['identity_reported'], 3)

    def test_eligibility_rechecks_the_models_named(self):
        p = plan(['x', 'y'], 50, 4, reps=2)
        lying = {'x': {**MODEL, 'identity': 'attested', 'identity_match': 'match', 'observed_models': ['claude-y-9']}}
        r = run(deals_between('x', 'y', 300, 50, np.random.default_rng(4), sd=100, rounds=4, reps=2), p, who=lying)
        [m] = r['pairings']
        self.assertEqual(m['verdict'], 'ineligible')
        self.assertEqual(m['identity_match']['x'], 'mismatch')
        self.assertEqual(next(q for q in r['players'] if q['player'] == 'x')['identity_match'], 'mismatch')
        ok = run(deals_between('x', 'y', 300, 50, np.random.default_rng(4), sd=100, rounds=4, reps=2), p,
                 who={'x': MODEL})
        self.assertEqual(next(q for q in ok['players'] if q['player'] == 'x')['identity_match'], 'match')


from test_runner_linked import Linked, TestReplayedDecisions


class TestJudgeReplaysDecisions(Linked):
    """R7-#2, judge side: the ratings' own audit of each decision (verify_play's linked_audit) replays the accepted
    reply against the engine state before its action, so a reply raising to 4 never verifies a raise to 200."""
    script = TestReplayedDecisions.script
    doctor_reply = TestReplayedDecisions.doctor_reply
    first_thought = TestReplayedDecisions.first_thought

    def audit(self, out):
        d = self.match(out)
        cfg = json.loads((d / 'config.json').read_text())
        R._VERIFIED.clear()
        v = R.verify_play(d, cfg, R._hands(d))
        return {k: x['audit'] for k, x in v.items()}

    def test_a_raise_to_4_reply_linked_to_a_raise_to_200(self):
        self.script({'action': 'raise', 'amount': 200})
        out = self.play(deals=(1,))
        t = self.first_thought(out)
        clean = self.audit(out)
        self.assertTrue(all(not s['problems'] for x in clean.values() for s in x.values()), clean)
        self.doctor_reply(out, t['accepted_call'], lambda o: {**o, 'amount': 4})
        bad = [p for x in self.audit(out).values() for s in x.values() for p in s['problems']]
        self.assertTrue(any('inconsistent' in p and t['decision_id'] in p for p in bad), bad)


if __name__ == '__main__':
    unittest.main()
