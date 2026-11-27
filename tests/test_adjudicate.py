"""The judge, finding by finding of the round 4 review: one eligibility-aware adjudicator for stopping,
recording and reporting (R4-#3), registration only from a pre-play artifact (R4-#1), official results pinned to
the registered judge with recomputation labelled retrospective (R4-#6), assumption status on every verdict
(R4-#2), identity attestation and audit completeness as eligibility rules (R4-#4, #5), duplicate pairs
validated before they count (R4-#8) and fit diagnostics beside the strengths (R4-#9)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from poker_arena import ratings as R
from test_ratings import ADJUDICATE_KEYS, MODEL, deals_between, facts, plan, run


def registered(p, created=0):
    return {**p, 'registration': R.registration(R.normalize_design(p), created=created)}


def strong(rng, a='x', b='y', edge=300, rounds=4, reps=2, effects=None):
    return deals_between(a, b, edge, 50, rng, sd=100, rounds=rounds, reps=reps, effects=effects)


def data(xs_by_conv):
    xs = [x for c in xs_by_conv for x in c]
    groups = [f'm{i}' for i, c in enumerate(xs_by_conv) for _ in c]
    return {'a': 'x', 'b': 'y', 'xs': xs, 'groups': groups, 'rejected': []}


class TestRegistration(unittest.TestCase):
    """R4-#1: `registered` is never recomputed from the design."""

    def setUp(self):
        self.p = plan(['x', 'y'], 50, 4, reps=2)

    def test_explicit_false_stays_false(self):
        p = {**self.p, 'design': {'registered': False, 'max_rounds': 9}}
        self.assertIs(R.normalize_design(p)['registered'], False)               # the reviewer's reproduction
        self.assertEqual(R.registration_status(p)['status'], 'declined')
        self.assertEqual(R.registration_status(registered(p))['status'], 'declined')   # even beside an artifact
        r = R.rate(strong(np.random.default_rng(1)), boot=20, plan=registered(p))
        self.assertIs(r['design']['registered'], False)
        self.assertEqual([m['verdict'] for m in r['pairings']], ['descriptive'])

    def test_true_without_an_artifact_is_not_registered(self):
        for p in (self.p, {**self.p, 'design': {**self.p['design'], 'registered': True}}):
            self.assertNotIn('registered', R.normalize_design(p))
            s = R.registration_status(p)
            self.assertEqual((s['registered'], s['status']), (False, 'missing'))
        self.assertTrue(R.registration_status(registered(self.p))['registered'])

    def test_written_after_play_began_is_invalid(self):
        s = R.registration_status(registered(self.p, created=200), {'first_started': 100})
        self.assertEqual((s['registered'], s['status']), (False, 'invalid'))
        self.assertIn('after play began', s['reasons'][0])
        self.assertTrue(R.registration_status(registered(self.p, created=50), {'first_started': 100})['registered'])

    def test_a_changed_design_is_not_the_registered_one(self):
        p = registered(self.p)
        p['design'] = {**p['design'], 'max_rounds': 6}
        s = R.registration_status(p)
        self.assertEqual(s['status'], 'invalid')
        self.assertIn('max_rounds', s['reasons'][0])

    def test_code_changes_void_it(self):
        change = {'t': 5, 'from': 'a', 'to': {'fingerprint': 'b'}, 'files': ['poker_arena/backends.py']}
        for p, ev in (({**registered(self.p), 'implementation_changes': [change]}, None),
                      (registered(self.p), {'implementation_changes': [change]})):
            s = R.registration_status(p, ev)
            self.assertEqual((s['registered'], s['status']), (False, 'void'))
            self.assertIn('backends.py', s['reasons'][0])
        p = registered(self.p)
        p['registration'] = {**p['registration'], 'status': 'void', 'void_reason': 'ratings.py changed'}
        self.assertEqual(R.registration_status(p)['status'], 'void')

    def test_the_judge_must_be_the_one_the_run_started_with(self):
        p = {**registered(self.p), 'implementation': {'files': {'poker_arena/ratings.py': 'somethingelse0000'}}}
        self.assertIn('registered judge', R.registration_status(p)['reasons'][0])
        ok = {**registered(self.p), 'implementation': {'files': {'poker_arena/ratings.py': R.judge_fingerprint()}}}
        self.assertTrue(R.registration_status(ok)['registered'])

    def test_the_judge_fingerprint_is_the_runners_hash(self):
        import hashlib
        src = (Path(R.__file__)).read_bytes()
        self.assertEqual(R.judge_fingerprint(), hashlib.sha256(src).hexdigest()[:16])


class TestAdjudicate(unittest.TestCase):
    """R4-#3, #4, #5: one adjudicator, every eligibility rule, a terminal 'ineligible'."""

    def setUp(self):
        self.p = registered(plan(['x', 'y'], 50, 4, reps=2))
        self.d = R.normalize_design(self.p)
        rng = np.random.default_rng(3)
        self.win = data([list(300 + 100 * rng.standard_normal(50)) for _ in range(8)])

    def judge(self, who=None, pairing=None, design=None, reg='keep'):
        e = facts(self.p['registration'] if reg == 'keep' else reg, 'xy', who)
        return R.adjudicate(pairing or self.win, design or self.d, 1, 4, e)

    def test_keys_and_a_clean_win(self):
        v = self.judge()
        self.assertLessEqual(ADJUDICATE_KEYS, set(v))
        self.assertEqual((v['verdict'], v['eligible'], v['ineligible_reasons'], v['official']), ('a', True, [], True))
        self.assertEqual((v['assumptions']['status'], v['calibrated'], v['counts_toward_leader'], v['standing']),
                         ('supported', True, True, 'decided'))
        self.assertEqual(v['judge'], R.judge_fingerprint())
        self.assertEqual(v['calibration_id'], self.p['registration']['calibration_id'])
        self.assertIn('ineligible', R.TERMINAL)

    def test_nothing_is_eligible_by_default(self):
        self.assertEqual(R.adjudicate(self.win, self.d, 1, 4, None)['verdict'], 'ineligible')
        unvalidated = {k: v for k, v in self.win.items() if k != 'rejected'}
        v = self.judge(pairing=unvalidated)
        self.assertEqual(v['verdict'], 'ineligible')
        self.assertIn('not validated', v['ineligible_reasons'][0])
        self.assertIn('no registration', self.judge(reg=None)['ineligible_reasons'][0])

    def test_identity(self):
        mismatch = {**MODEL, 'identity': 'mismatch', 'observed_models': ['claude-old']}
        cases = [({'x': mismatch}, 'claude-old'),
                 ({'x': {**MODEL, 'requested_model': 'sonnet'}}, 'bare alias'),
                 ({'x': {**MODEL, 'requested_model': None}}, 'no model'),
                 ({'x': {**MODEL, 'identity': 'none'}}, 'no replies'),
                 ({'x': {**MODEL, 'backend': 'command', 'identity': 'unattested'}}, 'did not attest'),
                 ({'x': {**MODEL, 'protocol_consistent': False}}, 'protocol')]
        for who, why in cases:
            v = self.judge(who)
            self.assertEqual((v['verdict'], v['test_verdict'], v['eligible']), ('ineligible', 'a', False), who)
            self.assertTrue(any(why in r for r in v['ineligible_reasons']), (why, v['ineligible_reasons']))
            self.assertFalse(v['counts_toward_leader'])

    def test_unattested_codex_is_allowed_but_shown(self):
        v = self.judge({'x': {**MODEL, 'backend': 'codex', 'identity': 'unattested'}})
        self.assertEqual((v['verdict'], v['identity_status']['x']), ('a', 'unattested'))
        d = R.normalize_design({**self.p, 'design': {**self.p['design'], 'allow_unattested': []}})
        v = R.adjudicate(self.win, d, 1, 4, facts(R.registration(d, 0), 'xy', {'x': {**MODEL, 'backend': 'codex',
                                                                                      'identity': 'unattested'}}))
        self.assertEqual(v['verdict'], 'ineligible')

    def test_audit(self):
        """Opaque (command) turns and incompletely audited ones never give a decided result unless the design
        preregistered it (R4-#5)."""
        for audit, why in (('opaque', 'opaque'), ('incomplete', 'no complete audit')):
            v = self.judge({'x': {**MODEL, 'backend': 'command', 'identity': 'attested', 'audit': audit}})
            self.assertEqual(v['verdict'], 'ineligible')
            self.assertTrue(any(why in r for r in v['ineligible_reasons']))
        d = R.normalize_design({**self.p, 'design': {**self.p['design'], 'allow_opaque': ['command']}})
        v = R.adjudicate(self.win, d, 1, 4, facts(R.registration(d, 0), 'xy',
                                                  {'x': {**MODEL, 'backend': 'command', 'audit': 'opaque'}}))
        self.assertEqual(v['verdict'], 'a')

    def test_rejected_pairs_and_code_changes(self):
        v = self.judge(pairing={**self.win, 'rejected': [{'matchup': 'm1', 'hand': None, 'reason': 'seats'}]})
        self.assertIn('failed validation', v['ineligible_reasons'][0])
        e = facts(self.p['registration'], 'xy')
        e['implementation_changes'] = [{'files': ['poker_arena/agents.py']}]
        self.assertIn('agents.py', R.adjudicate(self.win, self.d, 1, 4, e)['ineligible_reasons'][0])

    def test_rate_uses_it(self):
        """The ratings judge every pairing of a registered run through adjudicate, the runner's function."""
        rng = np.random.default_rng(4)
        with mock.patch.object(R, 'adjudicate', wraps=R.adjudicate) as spy:
            run(strong(rng), plan(['x', 'y'], 50, 4, reps=2), boot=20)
        self.assertGreater(spy.call_count, 1)


class TestOfficial(unittest.TestCase):
    """R4-#6: the official verdict is the recorded one, by the registered judge; recomputation is retrospective."""

    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.p = plan(['x', 'y'], 50, 4, reps=2)

    def test_recorded_by_the_registered_judge(self):
        r = run(strong(self.rng), self.p, boot=20)
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['official'], m['retrospective'], m['judge']), ('a', True, False, R.judge_fingerprint()))
        self.assertTrue(m['recomputed']['agrees'])
        self.assertEqual(r['leader']['player'], 'x')

    def test_todays_judge_differs(self):
        """A run judged by an earlier judge keeps its recorded verdicts; today's judge only adds a retrospective."""
        deals = strong(self.rng)
        with mock.patch.object(R, 'judge_fingerprint', return_value='0ld0judge0000000'):
            p = registered(self.p)
            looks = self._looks(deals, p)
        looks[3] = {**looks[3], 'verdict': 'even'}          # what the earlier judge recorded at the last look
        recorded = {'x__vs__y': looks}
        r = R.rate(deals, boot=20, plan=p, recorded=recorded, eligibility={frozenset('xy'): facts(p['registration'], 'xy')})
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['official'], m['judge']), ('even', True, '0ld0judge0000000'))
        self.assertEqual((m['recomputed']['verdict'], m['recomputed']['agrees'], m['recomputed']['judge']),
                         ('a', False, R.judge_fingerprint()))
        self.assertTrue(any('retrospective' in n and 'registered judge' in n for n in r['notes']))
        self.assertIn("today's judge would say a", R.summary(r))

    def _looks(self, deals, p):
        d = R.normalize_design(p)
        out = []
        for look in range(1, 5):
            at = [x for x in deals if x['round'] <= look]
            out.append(R.adjudicate({'a': 'x', 'b': 'y', 'xs': [x['net_a'] / 4 * 100 for x in at],
                                     'groups': [x['matchup'] for x in at], 'rejected': []},
                                    d, 1, look, facts(p['registration'], 'xy')))
        return out

    def test_a_record_by_another_judge_is_retrospective(self):
        deals = strong(self.rng)
        p = registered(self.p)
        looks = self._looks(deals, p)
        recorded = {'x__vs__y': [{**v, 'judge_fingerprint': 'f0rged0000000000', 'judge': 'f0rged0000000000'}
                                 for v in looks]}
        r = R.rate(deals, boot=20, plan=p, recorded=recorded, eligibility={frozenset('xy'): facts(p['registration'], 'xy')})
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['test_verdict'], m['official'], m['retrospective']), ('retrospective', 'a', False, True))
        self.assertFalse(m['counts_toward_leader'])
        self.assertIsNone(r['leader'])
        self.assertTrue(any('retrospective' in n for n in r['notes']))

    def test_no_record_is_not_official(self):
        p = registered(self.p)
        r = R.rate(strong(self.rng), boot=20, plan=p, eligibility={frozenset('xy'): facts(p['registration'], 'xy')})
        [m] = r['pairings']
        self.assertEqual((m['verdict'], m['official']), ('retrospective', False))
        self.assertIsNone(r['leader'])


class TestAssumptions(unittest.TestCase):
    """R4-#2: every verdict carries its assumption status; outside the calibrated range a decided verdict never
    makes a decided leader unless the design preregistered that it should."""

    def deals(self, rng):
        swings = [60.0, -60.0] * 4
        return (strong(rng, 'x', 'y', 300, effects=swings) + strong(rng, 'x', 'z', 300)
                + deals_between('y', 'z', 0, 50, rng, sd=100, rounds=4, reps=2))

    def test_outside_the_calibrated_range(self):
        rng = np.random.default_rng(6)
        deals = self.deals(rng)
        r = run(deals, plan(['x', 'y', 'z'], 50, 4, reps=2), boot=20)
        v = {frozenset((m['a'], m['b'])): m for m in r['pairings']}
        wide = v[frozenset('xy')]
        self.assertEqual((wide['verdict'], wide['official'], wide['calibrated']), ('a', True, False))
        self.assertGreater(wide['conversation_share'], R.FAMILY_SHARE)
        self.assertEqual(wide['assumptions']['status'], 'unsupported')
        self.assertEqual(wide['assumptions']['checks']['conversation_share']['limit'], R.FAMILY_SHARE)
        self.assertEqual(wide['standing'], 'decided (outside calibrated range)')
        self.assertFalse(wide['counts_toward_leader'])
        self.assertIsNone(r['leader'])
        self.assertEqual(r['groups'][0]['leader_outside_calibration'], 'x')
        self.assertTrue(any('Outside the calibrated range' in n for n in r['notes']))
        self.assertIn('outside calibrated range', R.summary(r))
        for m in r['pairings']:
            self.assertIn(m['assumptions']['status'], ('supported', 'unsupported', 'unknown'))

        counted = run(deals, plan(['x', 'y', 'z'], 50, 4, reps=2, design={'outside_calibration': 'count'}), boot=20)
        self.assertEqual(counted['leader']['player'], 'x')

    def test_wording_states_the_assumptions(self):
        rng = np.random.default_rng(7)
        r = run(strong(rng), plan(['x', 'y'], 50, 4, reps=2), boot=20)
        self.assertIn('if the test\'s assumptions hold', r['guarantee'])
        self.assertIn('symmetric', r['guarantee'])
        text = R.summary(r)
        self.assertIn('assumptions: supported', text)
        self.assertNotIn('at most 5% chance of any false verdict over', text.split('if its assumptions hold, ')[0])

    def test_repeated_deals_are_unsupported(self):
        v = R.assumptions({'conversations': 8, 'min_conversations': 8, 'conversation_share': 0.1},
                          ['f1', 'f2', 'f1'], ['m1', 'm1', 'm2'])
        self.assertEqual((v['status'], v['checks']['independent_deals']['value']), ('unsupported', 1))


def match(root, name, a, b, nets, *, seeds=None, b_players=None, b_seeds=None, b_fp=None, b_net=None):
    m = root / name
    seeds = seeds or [1000 + k for k in range(len(nets))]
    game = {'name': 'holdem', 'big_unit': 2, 'sb': 1, 'bb': 2, 'stack': 200, 'hands_per_match': len(nets)}
    fps = [R.fingerprint(game, s, k % 2) for k, s in enumerate(seeds)]
    for side, players in (('a-first', [a, b]), ('b-first', b_players or [b, a])):
        d = m / side
        d.mkdir(parents=True)
        sd = (b_seeds or seeds) if side == 'b-first' else seeds
        cfg = {'game': game, 'players': players, 'seeds': sd, 'deal_fingerprints': fps,
               'specs': {a: {'bot': 'rule'}, b: {'bot': 'station'}}}
        (d / 'config.json').write_text(json.dumps(cfg))
        rows = []
        for k, net in enumerate(nets):
            n = [net // 2, -(net // 2)] if side == 'a-first' else [-(net // 2), net // 2]
            if side == 'b-first' and b_net and k in b_net:
                n = b_net[k]
            fp = b_fp.get(k, fps[k]) if side == 'b-first' and b_fp else fps[k]
            rows.append(json.dumps({'hand': k, 'net': n, 'seed': sd[k], 'fingerprint': fp}))
        (d / 'hands.jsonl').write_text('\n'.join(rows) + '\n')


class TestPairValidation(unittest.TestCase):
    """R4-#8: a duplicate deal counts only when b-first is a-first with the seats swapped."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mismatches_are_rejected_and_reported(self):
        nets = [4, -2, 6, 0]
        match(self.root, 'ok', 'x', 'y', nets)
        match(self.root, 'seats', 'x', 'y', nets, b_players=['x', 'y'])
        match(self.root, 'seeds', 'x', 'y', nets, b_seeds=[7, 8, 9, 10])
        match(self.root, 'hand', 'x', 'y', nets, b_fp={1: 'other'}, b_net={2: [5, -3]})
        report = []
        deals = R.load_deals(self.root, report)
        got = {(d['matchup'], d['deal']) for d in deals}
        self.assertEqual(got, {('ok', k) for k in range(4)} | {('hand', 0), ('hand', 3)})
        why = {(x['matchup'], x['hand']): x['reason'] for x in report}
        self.assertIn('b-first seats', why[('seats', None)])
        self.assertIn('different seeds', why[('seeds', None)])
        self.assertIn('deal fingerprint', why[('hand', 1)])
        self.assertIn('do not add up', why[('hand', 2)])
        data = R.pairing_data(self.root, ['ok', 'hand'], 'x')
        self.assertEqual((len(data['xs']), len(data['rejected'])), (6, 2))
        p = registered(plan(['x', 'y'], 4, 4, reps=2))
        v = R.adjudicate(data, R.normalize_design(p), 1, 1, facts(p['registration'], 'xy'))
        self.assertEqual(v['verdict'], 'ineligible')
        v = R.adjudicate({'root': self.root, 'conversations': ['ok'], 'a': 'x', 'b': 'y'}, R.normalize_design(p), 1, 1,
                         facts(p['registration'], 'xy'))
        self.assertEqual((v['deals'], v['eligible']), (4, True))


class TestFitDiagnostics(unittest.TestCase):
    """R4-#9: observed against fitted edges, with an additivity warning when they disagree."""

    def test_non_additive_results_are_flagged(self):
        rng = np.random.default_rng(8)
        deals = (deals_between('x', 'y', 300, 50, rng, sd=100, rounds=4, reps=2)
                 + deals_between('y', 'z', 300, 50, rng, sd=100, rounds=4, reps=2)
                 + deals_between('x', 'z', -300, 50, rng, sd=100, rounds=4, reps=2))
        r = R.rate(deals, boot=100, pooled=True)
        for m in r['pairings']:
            self.assertEqual(set(m) >= {'observed_edge', 'fitted_edge', 'residual', 'residual_se', 'additivity'}, True)
            self.assertAlmostEqual(m['residual'], m['observed_edge'] - m['fitted_edge'], delta=0.2)
            self.assertEqual(m['additivity']['status'], 'warning')
        self.assertTrue(any('do not add up' in n for n in r['notes']))
        self.assertIn('weighs', r['fit']['method'])

    def test_additive_results_pass_and_trees_are_untestable(self):
        rng = np.random.default_rng(9)
        true = {'x': 100.0, 'y': 0.0, 'z': -100.0}
        deals = []
        for a, b in (('x', 'y'), ('x', 'z'), ('y', 'z')):
            deals += deals_between(a, b, true[a] - true[b], 50, rng, sd=100, rounds=4, reps=2)
        r = R.rate(deals, boot=100, pooled=True)
        self.assertTrue(all(m['additivity']['status'] == 'ok' for m in r['pairings']), [m['additivity'] for m in r['pairings']])
        tree = R.rate([d for d in deals if {d['a'], d['b']} != {'x', 'z'}], boot=50, pooled=True)
        self.assertTrue(all(m['additivity']['status'] == 'untestable' for m in tree['pairings']))


if __name__ == '__main__':
    unittest.main()


class TestAuditCounts(unittest.TestCase):
    """The audit summary behind the viewer's per-run audit panel (R4-#12, from A's side)."""

    def test_counts(self):
        from poker_arena.stats import audit_counts, summarize
        calls = [{'audit': 'verified', 'identity': 'attested', 'observed_model': 'claude-x-1-20260101'},
                 {'audit': 'opaque'},
                 {'rejected': 'tool_use', 'audit': 'verified', 'identity': 'attested', 'observed_model': 'claude-x-1'},
                 {'audit': 'verified', 'observed_model': 'claude-old'},
                 {'backend_failed': True, 'audit': 'no_reply'}]
        total = {}
        for c in calls:
            for k, v in audit_counts(c, 'claude-x-1').items():
                total[k] = total.get(k, 0) + v
        s = summarize(total)
        self.assertEqual(s['audit'], {'verified': 2, 'opaque': 1, 'unverified': 0, 'unrecorded': 0,
                                      'tool_use_rejected': 1, 'unverified_rejected': 0, 'no_reply': 1})
        self.assertEqual(s['identity'], {'attested': 2, 'unattested': 1, 'mismatch': 1})
