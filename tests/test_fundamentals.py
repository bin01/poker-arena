"""Round 7 review probes on poker fundamentals, judge side: a quiz score is eligible only when the judge re-verifies
every answer from its raw backend output, binds the parsed reply to the scored answer through the engine, checks the
spot key and the CLI that answered (R7-#3); scores are descriptive unless the sampling design supports an interval
(R7-#5); one attestation policy, attested / unattested / mismatch, judges games and quiz alike (R7-#13)."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from poker_arena import quiz, ratings as R

CLI = {'exe': 'claude', 'path': '/opt/claude/2.0', 'version': '2.0 (Claude Code)', 'sha256': 'ab' * 32}
CODEX = {'exe': 'codex', 'path': '/opt/codex.js', 'version': 'codex-cli 1.0', 'sha256': 'cd' * 32}
FILES = {'poker_arena/agents.py': 'a1', 'poker_arena/backends.py': 'b1', 'poker_arena/ratings.py': 'r1'}


def spec_of(backend='claude'):
    return {'backend': backend, 'model': 'claude-x-1' if backend == 'claude' else 'gpt-y', 'effort': 'high'}


def tournament(backend='claude', cli=CLI, **over):
    """A tournament player's provenance (ratings.player_provenance) that matches `player(backend, cli)`."""
    s = spec_of(backend)
    return {'identity': f'{backend}:{s["model"]}:high', 'settings': dict(s),
            'code': {k: v for k, v in FILES.items() if not k.endswith('ratings.py')}, 'cli_version': cli['version'],
            'cli_sha256': cli['sha256'], **over}



def _close():
    for i in range(200):
        d = quiz.spot_def(f'close-{i:03d}')
        s = quiz._accepted(d) if d else None
        if s and s['street'] == 'river':
            return s
    raise AssertionError('no river close spot in the first 200 candidates')


SHOVE = quiz.solve(quiz.spot_def('river-call-bluffcatcher'))     # basics: call is best
FOLD_B = quiz.solve(quiz.spot_def('river-fold-vs-value'))        # basics: fold is best
CALL_B = quiz.solve(quiz.spot_def('river-call-missed-draws'))    # basics: call is best
CALL = _close()


def claude_raw(text, model='claude-x-1'):
    lines = [{'type': 'system', 'subtype': 'init', 'model': model},
             {'type': 'assistant', 'message': {'model': model, 'content': [{'type': 'text', 'text': text}]}},
             {'type': 'result', 'result': text, 'session_id': 's1', 'usage': {}}]
    return {'returncode': 0, 'stdout': '\n'.join(json.dumps(x) for x in lines), 'stderr': ''}


def codex_raw(text):
    lines = [{'type': 'thread.started', 'thread_id': 't1'},
             {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text}},
             {'type': 'turn.completed', 'usage': {}}]
    return {'returncode': 0, 'stdout': '\n'.join(json.dumps(x) for x in lines), 'stderr': ''}


def reply(action, amount=None):
    return json.dumps({'reasoning': 'r', 'action': action, 'amount': amount, 'say': '', 'notes': None})


def answer(spot, action, executed, backend='claude', model='claude-x-1', cli=CLI, amount=None, text=None, error=None):
    text = reply(action, amount) if text is None else text
    raw = claude_raw(text, model) if backend == 'claude' else codex_raw(text)
    match = 'match' if backend == 'claude' and model == 'claude-x-1' else 'mismatch' if backend == 'claude' else 'unattested'
    call = {'attempt': 0, 'output': text, 'raw': raw, 'audit': 'verified', 'cli': dict(cli),
            'requested_model': 'claude-x-1' if backend == 'claude' else 'gpt-y', 'identity_match': match,
            **({'error': error} if error else {})}
    ev = spot['ev_bb']
    fallback = error is not None
    return {'spot': spot['id'], 'category': spot['category'],
            'requested': None if fallback else {'action': action, 'amount': amount},
            'executed': executed, 'action': executed, 'correct': not fallback and executed == spot['correct'],
            'outcome': 'fallback' if fallback else 'valid', 'fallback': fallback,
            'ev_loss_bb': round(max(ev.values()) - (min(ev.values()) if fallback else ev[executed]), 3),
            'identity_match': match, 'audit': 'verified', 'calls': [call]}


def player(answers, backend='claude', cli=CLI):
    spec = spec_of(backend)
    return {'spec': spec, 'kind': 'model', 'answers': answers,
            'protocol': {'fingerprint': 'p1', 'settings': {'quiz': 'quiz-3', 'player': dict(spec)}},
            'implementation': {'fingerprint': 'i1', 'files': dict(FILES)}, 'cli': dict(cli), 'cli_version': cli['version'],
            'eligibility': {'status': 'eligible', 'reasons': []}, 'identity_match': 'match', 'audit_status': 'verified'}


def quiz_doc(players, spots=(SHOVE, CALL), sampling=None):
    spots = [dict(s) for s in spots]
    cats = {'basics': {'population': None}, 'close': {'population': 'generated', 'estimand': 'share of the generator',
                                                      **({'sampling': sampling} if sampling else {})}}
    return {'version': 'quiz-3', 'spots': spots, 'categories': cats, 'players': players,
            'spots_hash': hashlib.sha256(json.dumps(spots, sort_keys=True).encode()).hexdigest()[:16]}


def good():
    return player([answer(SHOVE, 'call', 'call'), answer(CALL, 'call', 'call')])


class TestQuizEvidence(unittest.TestCase):
    """R7-#3: stored labels are claims; the judge re-derives each answer from its raw output."""

    def fund(self, doc, players=None):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'quiz.json').write_text(json.dumps(doc))
            return R.fundamentals(d, players if players is not None else {'m1': tournament()})

    def test_untouched_answers_are_eligible(self):
        f = self.fund(quiz_doc({'opus': good()}))
        x = f['players']['m1']
        self.assertEqual((x['eligibility']['status'], x['attached']), ('eligible', True), x['eligibility'])
        self.assertEqual((x['correct'], x['answered'], x['scored_category']), (1, 1, 'basics'))
        self.assertEqual(x['categories']['close']['correct'], 1)
        self.assertEqual(x['attestation']['status'], 'attested')

    def test_null_raw_output_is_not_eligible(self):
        """The reviewer's probe: raw: null, with stored labels saying match and verified, was eligible."""
        p = good()
        p['answers'][0]['calls'][0]['raw'] = None
        f = self.fund(quiz_doc({'opus': p}))
        self.assertNotIn('m1', f['players'])
        x = f['players']['opus']
        self.assertEqual(x['eligibility']['status'], 'ineligible')
        self.assertTrue(any('raw output' in r for r in x['eligibility']['reasons']), x['eligibility'])

    def test_the_scored_answer_is_bound_to_the_raw_reply(self):
        """The raw reply folds; the record says it called, correctly. The judge scores what was said."""
        p = good()
        a = p['answers'][0]
        text = reply('fold')
        a['calls'][0].update(output=text, raw=claude_raw(text))
        f = self.fund(quiz_doc({'opus': p}))
        x = f['players']['opus']
        self.assertEqual(x['eligibility']['status'], 'ineligible')
        self.assertTrue(any('executed' in r or 'requested' in r for r in x['eligibility']['reasons']), x['eligibility'])
        self.assertEqual(x['correct'], 0)

    def test_an_unknown_or_altered_spot_key_is_not_eligible(self):
        p = good()
        p['answers'][0]['spot'] = 'nowhere'
        self.assertEqual(self.fund(quiz_doc({'opus': p}))['players']['opus']['eligibility']['status'], 'ineligible')
        doc = quiz_doc({'opus': good()})
        doc['spots'][0]['correct'] = 'fold'              # the key edited after the quiz
        x = self.fund(doc)['players']['opus']
        self.assertEqual(x['eligibility']['status'], 'ineligible')
        self.assertTrue(any('answer key' in r for r in x['eligibility']['reasons']), x['eligibility'])

    def test_a_different_cli_is_not_eligible(self):
        p = good()
        p['answers'][1]['calls'][0]['cli'] = {**CLI, 'sha256': 'ef' * 32}   # same version string, another file
        x = self.fund(quiz_doc({'opus': p}))['players']['opus']
        self.assertEqual(x['eligibility']['status'], 'ineligible')
        self.assertTrue(any('CLI' in r for r in x['eligibility']['reasons']), x['eligibility'])
        q = good()
        q['cli'] = None
        self.assertEqual(self.fund(quiz_doc({'opus': q}))['players']['opus']['eligibility']['status'], 'ineligible')

    def test_a_served_mismatch_from_the_raw_output_is_caught(self):
        """The stored label says match; the raw output names another model."""
        p = good()
        a = p['answers'][0]
        a['calls'][0]['raw'] = claude_raw(a['calls'][0]['output'], model='claude-z-9')
        x = self.fund(quiz_doc({'opus': p}))['players']['opus']
        self.assertEqual((x['eligibility']['status'], x['attestation']['status']), ('ineligible', 'mismatch'))

    def test_the_server_serves_the_verified_representation(self):
        p = good()
        a = p['answers'][0]
        text = reply('fold')
        a['calls'][0].update(output=text, raw=claude_raw(text))
        doc = quiz_doc({'opus': p})
        v = R.verified_quiz(copy.deepcopy(doc))
        y = v['players']['opus']
        self.assertEqual(y['eligibility']['status'], 'ineligible')
        self.assertEqual((y['correct'], y['answered'], y['scored_category']), (0, 1, 'basics'))
        b = y['answers'][0]
        self.assertEqual((b['outcome'], b['correct'], b['verified']), ('unverified', False, False))
        self.assertTrue(b['problems'])
        self.assertTrue(all('raw' not in c for c in b['calls']))
        self.assertEqual(y['categories']['close']['answered'], 1)


class TestOneAttestationPolicy(unittest.TestCase):
    """R7-#13: games allowed unattested Codex play while the quiz required attestation."""

    def test_the_same_rule_for_both(self):
        for backend, ident, allowed in (('codex', 'unattested', True), ('command', 'unattested', False),
                                        ('claude', 'match', True), ('claude', 'mismatch', False),
                                        ('claude', 'unattested', False)):
            a = R.attestation(backend, ident)
            self.assertEqual(a['allowed'], allowed, (backend, ident))
            game = R._player_problems('p', {'kind': 'model', 'backend': backend, 'requested_model': 'm-1',
                                            'identity_match': ident, 'observed_models': ['m-1'] if ident == 'match'
                                            else ['m-2'] if ident == 'mismatch' else [], 'audit': 'complete'},
                                      {'allow_unattested': list(R.ALLOW_UNATTESTED)})
            self.assertEqual(not game, allowed, (backend, ident, game))

    def test_unattested_codex_quiz_is_eligible_and_marked(self):
        p = player([answer(SHOVE, 'call', 'call', backend='codex', cli=CODEX)], backend='codex', cli=CODEX)
        p['eligibility'] = {'status': 'ineligible', 'reasons': ['not attested']}   # the quiz's own, older rule
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'quiz.json').write_text(json.dumps(quiz_doc({'luna': p})))
            f = R.fundamentals(d, {'m2': tournament('codex', CODEX)})
        x = f['players']['m2']
        self.assertEqual((x['eligibility']['status'], x['attestation']['status']), ('eligible', 'unattested'))
        self.assertTrue(x['attestation']['allowed'])


class TestDescriptiveScores(unittest.TestCase):
    """R7-#5: answer quotas make a generated category's spots no independent sample, so no binomial interval."""

    def test_no_interval_without_an_independent_sample(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'quiz.json').write_text(json.dumps(quiz_doc({'opus': good()}, sampling={'design': 'quotas'})))
            quota = R.fundamentals(d, {})['players']['opus']['categories']['close']
            (Path(d) / 'quiz.json').write_text(json.dumps(quiz_doc({'opus': good()}, sampling={'design': 'independent'})))
            iid = R.fundamentals(d, {})['players']['opus']['categories']['close']
        self.assertIsNone(quota['interval95'])
        self.assertIn('descriptive', quota['estimand'])
        self.assertIsNotNone(iid['interval95'])


if __name__ == '__main__':
    unittest.main()
