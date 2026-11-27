"""Round 11 review, material issues: cyclic decided wins give no strict ranks (#1); an unreadable transcript line
makes a turn unverified (#2); quiz prompts carry no answer-bearing ids (#3)."""
import asyncio
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poker_arena import quiz as Q
from poker_arena import ratings as R
from poker_arena.backends import AgyCLI, ClaudeCLI, CodexCLI, make_backend, verify_call
from test_ratings import deals_between, plan, run

VIEWER = Path(__file__).resolve().parent.parent / 'poker_arena' / 'viewer'


def node(body):
    script = (f'const Core = require({json.dumps(str(VIEWER / "core.js"))});'
              f'const out = (() => {{ {body} }})(); process.stdout.write(JSON.stringify(out));')
    res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def tournament(wins, players):
    rng = np.random.default_rng(11)
    deals = []
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            edge = 600 if b in wins.get(a, ()) else -600 if a in wins.get(b, ()) else 0
            deals += deals_between(a, b, edge, 10, rng, sd=50, rounds=4, reps=4)
    return run(deals, plan(players, 10, 4, reps=4))


class TestCyclicRanks(unittest.TestCase):
    """#1: decided wins that form a cycle give no strict order inside the cycle."""

    def test_cycle_gets_no_ranks(self):
        wins = {'a': 'c', 'b': 'acd', 'c': 'de', 'd': 'ae', 'e': 'ab'}
        r = tournament(wins, list('abcde'))
        self.assertEqual(sum(x['verdict'] in R.DECIDED for x in r['pairings']), 10)
        for p in r['players']:
            self.assertIsNone(p['rank'], p['player'])
            self.assertEqual(p['rank_kind'], 'descriptive')
            lo, hi = p['rank_range']
            self.assertLess(lo, hi, p['player'])
            self.assertEqual(sorted(p['rank_cycle']), sorted(set('abcde') - {p['player']}))
            self.assertEqual(p['rank_note'], 'decided matchups form a cycle')
        self.assertIsNone(r['leader'])

    def test_chain_keeps_ranks(self):
        r = tournament({'a': 'bc', 'b': 'c'}, list('abc'))
        s = {p['player']: p for p in r['players']}
        self.assertEqual([s[p]['rank'] for p in 'abc'], [1, 2, 3])
        self.assertTrue(all(s[p]['rank_cycle'] == [] and s[p]['rank_note'] is None for p in 'abc'))

    def test_cycle_below_a_clear_winner(self):
        r = tournament({'a': 'bcd', 'b': 'c', 'c': 'd', 'd': 'b'}, list('abcd'))
        s = {p['player']: p for p in r['players']}
        self.assertEqual((s['a']['rank'], s['a']['rank_range']), (1, [1, 1]))
        for p in 'bcd':
            self.assertEqual((s[p]['rank'], s[p]['rank_range']), (None, [2, 4]))
        self.assertEqual(r['leader']['player'], 'a')

    def test_viewer_blocks_a_cycle(self):
        ps = [{'player': 'a', 'rank': 1, 'rank_kind': 'decided', 'rank_range': [1, 1], 'strength': 300},
              *[{'player': p, 'rank': None, 'rank_kind': 'descriptive', 'rank_range': [2, 4], 'strength': s,
                 'rank_cycle': [q for q in 'bcd' if q != p], 'rank_note': 'decided matchups form a cycle'}
                for p, s in (('b', 10), ('c', 20), ('d', -30))]]
        blocks = node(f'return Core.standingBlocks({json.dumps(ps)}).map(b => [b.kind, b.range, b.players.map(p => p.player)]);')
        self.assertEqual(blocks, [['decided', None, ['a']], ['cycle', [2, 4], ['c', 'b', 'd']]])
        only = node(f'return Core.standingBlocks({json.dumps(ps[1:])}).map(b => b.kind);')
        self.assertEqual(only, ['cycle'])
        res = node("return Core.cycleResults(['b', 'c', 'd'], [{a: 'b', b: 'c', verdict: 'a'}, {a: 'c', b: 'd', verdict: 'a'},"
                   " {a: 'b', b: 'd', verdict: 'b'}, {a: 'b', b: 'x', verdict: 'a'}]);")
        self.assertEqual(res, [{'winner': 'b', 'loser': 'c'}, {'winner': 'c', 'loser': 'd'}, {'winner': 'd', 'loser': 'b'}])


class TestUnreadableTranscript(unittest.TestCase):
    """#2: a transcript or stream line that can't be parsed leaves the turn unverified, with its raw bytes kept."""

    def brain(self, text):
        brain = Path(tempfile.mkdtemp())
        logs = brain / 'conv-1' / '.system_generated' / 'logs'
        logs.mkdir(parents=True)
        (logs / 'transcript.jsonl').write_bytes(text)
        be = make_backend('agy', 'SYS')
        be.session, be.BRAIN = 'conv-1', str(brain)
        return be

    def test_unreadable_middle_record_in_memory(self):
        rows = [{'type': 'USER_INPUT'}, {'unreadable_line': '{"type": "PLANNER_RESPONSE", "tool_calls": [{"na'},
                {'type': 'PLANNER_RESPONSE', 'model': 'g-1'}]
        self.assertEqual(AgyCLI.read_turn(rows)[3], 'unverified')
        self.assertEqual(AgyCLI.turn_problem(rows), 'unreadable transcript line')
        self.assertEqual(AgyCLI.evidence({'stdout': '{"status": "SUCCESS", "response": "x"}', 'transcript': rows})['audit'],
                         'unverified')

    def test_transcript_keeps_the_raw_bytes(self):
        bad = b'{"type": "PLANNER_RESPONSE", "tool_calls": [{"name": "run_command"\xff'
        be = self.brain(b'{"type": "USER_INPUT"}\n' + bad + b'\n{"type": "PLANNER_RESPONSE", "model": "g-1"}\n')
        rows = be._transcript()
        self.assertEqual(rows[1]['unreadable_line'].encode('utf-8', 'surrogateescape'), bad)
        self.assertEqual(be.read_turn(rows)[3], 'unverified')
        json.dumps(rows)
        good = self.brain(b'{"type": "USER_INPUT"}\n\n{"type": "PLANNER_RESPONSE", "model": "g-1"}\n')
        self.assertEqual(good.read_turn(good._transcript())[3], 'verified')

    def test_a_corrupted_user_input_is_not_hidden(self):
        be = self.brain(b'{"type": "USER_INPUT"}\n{"type": "PLANNER_RESPONSE"}\n{"type": "USER_IN\n'
                        b'{"type": "PLANNER_RESPONSE", "model": "g-1"}\n')
        self.assertEqual(be.read_turn(be._transcript())[3], 'unverified')
        only = self.brain(b'not json\n')
        self.assertEqual(only._transcript(), [{'unreadable_line': 'not json'}])

    def test_claude_and_codex_streams(self):
        claude = '\n'.join([json.dumps({'type': 'system', 'subtype': 'init', 'session_id': 'c'}), '{"type": "assistant", "mess',
                            json.dumps({'type': 'result', 'result': '{"action": "fold"}', 'session_id': 'c'})]) + '\n'
        self.assertEqual(ClaudeCLI.evidence({'stdout': claude})['audit'], 'unverified')
        self.assertEqual(ClaudeCLI.evidence({'stdout': claude.replace('{"type": "assistant", "mess', '')})['audit'], 'verified')
        codex = '\n'.join([json.dumps({'type': 'thread.started', 'thread_id': 't'}), 'garbage',
                           json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'x'}})])
        self.assertEqual(CodexCLI.evidence({'stdout': codex})['audit'], 'unverified')

    def test_the_seat_rejects_and_asks_again(self):
        from poker_arena.agents import ModelAgent
        from poker_arena.games import get_game
        d = tempfile.mkdtemp()
        exe = Path(d) / 'claude'
        exe.write_text('#!/usr/bin/env python3\nimport json, os, sys\nsys.stdin.read()\n'
                       'p = os.path.join(os.path.dirname(sys.argv[0]), "n")\n'
                       'n = int(open(p).read()) if os.path.exists(p) else 0\n'
                       'if "-p" in sys.argv: open(p, "w").write(str(n + 1))\n'
                       'print(json.dumps({"type": "system", "subtype": "init", "session_id": "c"}))\n'
                       'if n == 0: print("{broken")\n'
                       'print(json.dumps({"type": "result", "result": json.dumps({"action": "call"}), "session_id": "c"}))\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        old = os.environ['PATH']
        os.environ['PATH'] = d + os.pathsep + old
        self.addCleanup(os.environ.__setitem__, 'PATH', old)
        G = get_game('holdem')({'hands_per_match': 1})
        ep = G.episode(0, 1, ['A', 'B'])
        agent = ModelAgent('m', 'claude', retries=0)
        agent.start('SYS', {})
        validate = lambda act: ep.__class__.apply(__import__('copy').deepcopy(ep), 0, act)
        dec = asyncio.run(agent.act('go', ep.request(0), validate))
        self.assertEqual(dec.action['action'], 'call')
        first, second = dec.calls
        self.assertEqual((first['audit'], first['audit_reason'], first['rejected']),
                         ('unverified', 'unreadable output line', 'unverified'))
        self.assertIn('{broken', first['raw']['stdout'])
        self.assertEqual(verify_call('claude', first, first['raw'])['status'], 'unverified')
        self.assertEqual(second['audit'], 'verified')


class TestOpaqueQuizIds(unittest.TestCase):
    """#3: nothing a model sees in a quiz prompt names the spot's answer."""

    def test_no_prompt_carries_an_answer_bearing_id_or_field(self):
        self.assertEqual(Q.VERSION, 'quiz-4')
        words = re.compile(r'fold|call|raise|shove|bet', re.I)
        seen = set()
        for d in Q.suite():
            s = Q.solve(d)
            ep, hero, _, _ = Q.build(d)
            text = Q.question(s, ep, hero)
            [qid] = re.findall(r'^Question (\S+)\.', text)
            self.assertRegex(qid, r'^Q-[0-9A-F]{8}$')
            self.assertIsNone(words.search(qid), qid)
            seen.add(qid)
            self.assertNotIn(s['id'], text)
            for field in ('title', 'explain', 'equity_note'):
                if s.get(field):
                    self.assertNotIn(s[field], text, (s['id'], field))
            if s.get('equity') is not None:
                self.assertNotIn(f'{100 * s["equity"]:.1f}%', text)
            for key in ('correct', 'best', 'answer', 'clear_cut', 'ev_bb', 'gap_bb'):
                self.assertNotRegex(text, rf'["\']?\b{key}\b["\']?\s*[:=]', (s['id'], key))
            self.assertNotIn(s['correct'] + '"', text.split('your only choices are')[0])
        self.assertEqual(len(seen), len(Q.suite()))

    def test_earlier_files_are_labelled(self):
        basics = [Q.solve(d) for d in Q.suite(['basics'])]
        leak = Q.prompt_leak({'version': 'quiz-3', 'spots': basics})
        self.assertEqual(leak['spots'], [t[0] for t in Q.SPOTS])
        self.assertIn('river-fold-vs-value', leak['named_answer'])
        self.assertIn('river-fold-pot-bet-vs-value', leak['named_answer'])
        self.assertNotIn('pf-call-ajo-20', leak['named_answer'])          # names "call"; the answer is fold
        self.assertIn('pf-shove-a7o-10', leak['named_answer'])            # "shove": the answer is all in
        self.assertIsNone(Q.prompt_leak({'version': 'quiz-4', 'spots': basics}))
        self.assertIsNotNone(Q.prompt_leak({'spots': basics}))
        self.assertIsNone(Q.prompt_leak({'version': 'quiz-2', 'spots': [s for s in basics if False]}))
        doc = {'version': 'quiz-3', 'spots': basics, 'players': {
            'm': {'kind': 'model', 'answers': [{'spot': 'river-fold-vs-value', 'category': 'basics', 'correct': True}]}}}
        v = R.verified_quiz(doc)
        self.assertEqual(v['prompt_leak']['version'], 'quiz-3')
        self.assertEqual(v['players']['m']['prompt_leak']['spots'], ['river-fold-vs-value'])
        self.assertIsNone(R.verified_quiz({**doc, 'version': 'quiz-4'})['players']['m']['prompt_leak'])

    def test_viewer_flags_leaked_scores(self):
        note = node("return [Core.leakNote({prompt_leak: {version: 'quiz-3', spots: ['a', 'b'], named_answer: ['a']}}),"
                    " Core.leakNote({}), Core.leakNote({prompt_leak: null})];")
        self.assertIn('quiz-3', note[0]['short'])
        self.assertIn('2 spots', note[0]['long'])
        self.assertEqual(note[1:], [None, None])


if __name__ == '__main__':
    unittest.main()


class TestStandingOrderAroundCycles(unittest.TestCase):
    def test_a_cycle_in_the_middle_sits_between_its_decided_neighbours(self):
        import json, subprocess
        js = """
const C=require('./poker_arena/viewer/core.js');
const ps=[{player:'a',rank_kind:'decided',rank:1,strength:3},{player:'e',rank_kind:'decided',rank:5,strength:-3},
{player:'b',rank_range:[2,4],rank_cycle:['c','d'],strength:1},{player:'c',rank_range:[2,4],rank_cycle:['b','d'],strength:0},
{player:'d',rank_range:[2,4],rank_cycle:['b','c'],strength:-1}];
console.log(JSON.stringify(C.standingBlocks(ps).map(b=>b.players.map(p=>p.player).join(''))));"""
        out = subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout
        self.assertEqual(json.loads(out), ['a', 'bcd', 'e'])


class TestRound12Fixes(unittest.TestCase):
    def test_street_comparison_counts_amounts(self):
        src = open('poker_arena/viewer/app.js').read()
        self.assertIn("${e.seat}:${e.action}:${e.amount ?? ''}", src)

    def test_leader_chances_are_labelled_independence_scenarios(self):
        from poker_arena import ratings
        self.assertIn('not a', ratings.leader_chances.__doc__)
        self.assertNotIn('at or above that product', ratings.leader_chances.__doc__)
