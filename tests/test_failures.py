"""Deals a model's CLI gave up on, end to end: a real plan whose model backend fails, what the viewer's data, the
run's lifecycle and `status` then say, and a rerun that replays those deals and clears every trace of the failure."""
import asyncio
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from poker_arena import backends, cli, ratings, server
from poker_arena.backends import BackendError
from poker_arena.runner import make_plan, run_plan
from test_runner import QUIET, FakeCLI

REFUSAL = ('claude exited 1: ate":"off","is_error":true,"api_error_status":null,"result":"API Error: Fable 5.2\'s '
           'safeguards flagged this message (https://www.anthropic.com/legal/aup). This sometimes happens.\\n\\nTry '
           'rephrasing.\\n\\nDetails: `[reasoning_extraction]`\\n\\nRequest ID: req_1","type":"result","duration_ms":2439}')


async def no_sleep(_):
    return None


class TestFailureText(unittest.TestCase):
    def test_the_cli_message_is_pulled_out_of_its_json(self):
        text = server._error_text(REFUSAL)
        self.assertTrue(text.startswith("API Error: Fable 5.2's safeguards flagged this message"))
        self.assertIn('Details: `[reasoning_extraction]`', text)
        self.assertNotIn('duration_ms', text)

    def test_text_without_json_keeps_its_tail(self):
        self.assertEqual(server._error_text('fakecli exited 1: boom'), 'fakecli exited 1: boom')
        self.assertEqual(server._error_text('x "result":"unterminated'), 'x "result":"unterminated')
        self.assertEqual(server._error_text('claude returned no result: {"result":""}'),
                         'claude returned no result: {"result":""}')   # an empty message says nothing: keep the tail
        self.assertEqual(server._error_text(None), '')
        self.assertEqual(len(server._error_text('y' * 5000)), 600)

    def test_kinds(self):
        kind = lambda e: server.failure_kind(e, server._error_text(e))[0]
        self.assertEqual(kind(REFUSAL), 'refused')
        self.assertEqual(kind('claude timed out after 900s'), 'timeout')
        self.assertEqual(kind("'agy' is not installed or not on PATH"), 'missing_cli')
        self.assertEqual(kind('HTTP 429: rate limit, try again later'), 'rate_limit')
        self.assertEqual(kind('codex: You have hit your usage limit. Try again at 4pm'), 'rate_limit')
        self.assertEqual(kind('claude exited 1: Invalid API key · Please run /login'), 'auth')
        self.assertEqual(kind('m: none of 3 turns could be audited; stopping the match'), 'unaudited')
        self.assertEqual(kind('fakecli exited 1: boom'), 'error')
        # the runner's own prefix outranks words the CLI happened to print before it was killed
        self.assertEqual(kind('claude timed out after 900s'), 'timeout')
        self.assertEqual(server.failure_kind('codex timed out after 60s', 'rate limit')[0], 'timeout')


class TestTerminalLines(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(cli._failure_lines(None, 'runs'), [])
        self.assertEqual(cli._failure_lines({'deals': 0, 'players': []}, 'runs'), [])
        one = {'deals': 1, 'players': [{'player': 'm', 'deals': 1, 'started': 4, 'kinds': [
            {'kind': 'timeout', 'title': 'no reply before the time limit', 'deals': 1,
             'message': 'claude timed out\nafter 900s ' + 'x' * 400, 'example': 'r/m__vs__gto/a-first'}]}]}
        head, row = cli._failure_lines(one, 'runs')
        self.assertIn('1 deal stopped on a model error and is not counted', head)
        self.assertIn('m: 1 of 4 deals, no reply before the time limit: "claude timed out after 900s', row)
        self.assertIn('..." (full error: runs/r/m__vs__gto/a-first/calls.jsonl)', row)   # one line, cut, with the file


class TestFailedDeals(unittest.TestCase):
    """One model ('m', a scripted CLI) against a bot, three one-deal conversations."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.runs = self.root / 'runs'
        backends.BACKENDS['fakecli'] = FakeCLI
        FakeCLI.sent, FakeCLI.script = [], None
        players = self.root / 'players.toml'
        players.write_text('[players.m]\nbackend = "fakecli"\nretries = 1\nlabel = "Fake Model High"\n'
                           '[players.station]\nbot = "station"\n')
        self.plan = make_plan(['m', 'station'], deals=1, out=str(self.runs / 'r'), players_file=str(players),
                              design={'max_rounds': 1, 'min_conversations': 100}, replicates=3)
        server.INVENTORIES.clear()
        server.CONVERSATIONS.clear()
        ratings._VERIFIED.clear()

    def tearDown(self):
        backends.BACKENDS.pop('fakecli', None)
        self.tmp.cleanup()

    def play(self, script=None):
        FakeCLI.script = staticmethod(script) if script else None
        with mock.patch.object(backends.asyncio, 'sleep', no_sleep):
            return asyncio.run(run_plan(self.plan, **QUIET))

    def run_row(self):
        return next(r for r in server.runs(self.runs) if r['run'] == 'r')

    def status_text(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.cmd_status(types.SimpleNamespace(root=str(self.runs), run=None, all=False, health=False))
        return out.getvalue()

    def test_failed_deals_are_named_everywhere_and_a_rerun_clears_them(self):
        results = self.play(lambda n, text: BackendError(REFUSAL))
        self.assertTrue(all(isinstance(x, BackendError) for x in results))

        # the run: every deal failed, by whom, why, and where to look
        row = self.run_row()
        f = row['failures']
        self.assertEqual((f['deals'], f['conversations'], f['interrupted']), (3, 3, 0))
        (who,) = f['players']
        self.assertEqual((who['player'], who['deals'], who['started']), ('m', 3, 3))
        (kind,) = who['kinds']
        self.assertEqual((kind['kind'], kind['deals']), ('refused', 3))
        self.assertTrue(kind['message'].startswith("API Error: Fable 5.2's safeguards flagged"))
        self.assertTrue((self.runs / kind['example'] / 'calls.jsonl').exists())
        self.assertFalse(row['finished'])
        self.assertFalse(row['live'])                       # a failed conversation is never 'playing'

        # the pairing's recorded state is corrected from the call logs, and its error is readable
        p = row['pairings']['m|station']
        self.assertEqual((p['status'], p['failed_deals'], p['error_kind']), ('failed', 3, 'refused'))
        self.assertTrue(p['error'].startswith('API Error:'))
        self.assertIn('BackendError', p['error_raw'])
        self.assertEqual(json.loads((self.runs / 'r' / 'tournament.json').read_text())['status'], 'stopped')
        self.assertEqual({x['player']: x['status'] for x in server.roster(self.runs, 'r')}['m'], 'stopped')
        rated = server.ratings_for(self.runs, 'r', boot=20)   # the standings' roster says so too
        self.assertEqual({x['player']: x['status'] for x in rated['roster']}['m'], 'stopped')

        # each deal: failed, never in progress, never counted
        pr = server.pairing(self.runs, 'r', 'm', 'station')
        self.assertEqual([c['deals'] for c in pr['conversations']], [[[0, None, 'failed', None]]] * 3)
        self.assertEqual((pr['totals']['failed'], pr['totals']['counted'], pr['totals']['provisional'],
                          pr['totals']['playing']), (3, 0, 0, 0))
        for c in pr['conversations']:
            self.assertIn(len(c['failed']), (1, 2))        # the model acts in both plays, unless the bot ended one first
            for bad in c['failed']:
                play = server.match(self.runs, f"{c['id']}/{bad['side']}")
                self.assertEqual((play['failure']['player'], play['failure']['tries'], play['failure']['kind']), ('m', 2, 'refused'))
                self.assertIn('reasoning_extraction', play['failure']['detail'])
            for side in {'a-first', 'b-first'} - {x['side'] for x in c['failed']}:   # a play only stopped with the other
                self.assertIsNone(server.match(self.runs, f"{c['id']}/{side}")['failure'])
        self.assertEqual(len(ratings.load_deals(self.runs / 'r')), 0)                       # nothing leaks into the ratings

        text = self.status_text()
        self.assertIn('3 deals stopped on a model error and are not counted', text)
        self.assertIn("m: 3 of 3 deals, blocked by the provider's safeguards", text)
        self.assertIn('stopped on an error', text)
        self.assertIn('/calls.jsonl', text)

        # the static export carries the same facts
        with tempfile.TemporaryDirectory() as d:
            server.export(self.runs, d)
            data = Path(d) / 'data'
            runs = json.loads((data / 'runs.json').read_text())
            self.assertEqual(next(r for r in runs if r['run'] == 'r')['failures']['deals'], 3)
            m = json.loads((data / f"match-{server.key(kind['example'])}.json").read_text())
            self.assertEqual(m['failure']['kind'], 'refused')
            # ... the CLI's own message, but never its raw output or the runner's raw error
            self.assertTrue(m['failure']['message'].startswith('API Error:'))
            self.assertNotIn('detail', m['failure'])
            everything = ''.join(f.read_text() for f in data.iterdir())
            self.assertNotIn('error_raw', everything)
            self.assertNotIn('duration_ms', everything)
            self.assertNotIn('Request ID', everything)

        # the same plan again replays the deals: nothing is left failed, and they count
        results = self.play()
        self.assertFalse([x for x in results if isinstance(x, Exception)])
        row = self.run_row()
        self.assertEqual(row['failures'], {'deals': 0, 'conversations': 0, 'players': [], 'interrupted': 0})
        self.assertNotIn('failed_deals', row['pairings']['m|station'])
        self.assertNotEqual(row['pairings']['m|station']['status'], 'failed')
        pr = server.pairing(self.runs, 'r', 'm', 'station')
        self.assertEqual([c['deals'][0][2] for c in pr['conversations']], ['verified'] * 3)
        self.assertEqual((pr['totals']['failed'], pr['totals']['verified']), (0, 3))
        self.assertTrue(all(not c['failed'] for c in pr['conversations']))
        self.assertEqual(len(ratings.load_deals(self.runs / 'r')), 3)
        self.assertNotIn('errors:', self.status_text())

    def test_some_deals_fail_and_the_rest_still_count(self):
        """A model that fails in one conversation only: that deal is failed, the others are verified results, and
        the pairing is not called failed while it has results."""
        first = {}

        def script(n, text):   # every call of the first conversation to ask fails; the others answer
            seed = text.split('Your cards:')[1][:8] if 'Your cards:' in text else text[:40]
            first.setdefault('seed', seed)
            return BackendError('fakecli exited 1: boom') if seed == first['seed'] else None
        self.play(script)
        row = self.run_row()
        self.assertEqual(row['failures']['deals'], 1)
        self.assertEqual(row['failures']['players'][0]['kinds'][0]['kind'], 'error')
        pr = server.pairing(self.runs, 'r', 'm', 'station')
        self.assertEqual(sorted(c['deals'][0][2] for c in pr['conversations']), ['failed', 'verified', 'verified'])
        self.assertEqual((pr['totals']['failed'], pr['totals']['verified'], pr['totals']['counted']), (1, 2, 2))
        self.assertEqual(row['pairings']['m|station']['failed_deals'], 1)
        self.assertEqual({x['player']: x['status'] for x in server.roster(self.runs, 'r')}['m'], 'playing')
        # an error that is only a tail of raw output never reaches a static export
        with tempfile.TemporaryDirectory() as d:
            server.export(self.runs, d)
            everything = ''.join(f.read_text() for f in (Path(d) / 'data').iterdir())
            self.assertNotIn('boom', everything)
            self.assertIn(server.NOT_EXPORTED, everything)
            self.assertIn('"kind":"error"', everything)

    def test_an_interrupted_play_is_not_called_failed(self):
        """A play that stopped with no backend error on record (the run was killed) is unfinished, not failed."""
        self.play()
        conv = server.pairing(self.runs, 'r', 'm', 'station')['conversations'][0]
        for f in ('result.json', 'state.json', 'hands.jsonl'):
            (self.runs / conv['id'] / 'a-first' / f).unlink()
        old = time.time() - 10 * server.LIVE_SECONDS      # long enough ago that nothing reads as being played
        for side in server.SIDES:
            os.utime(self.runs / conv['id'] / side / 'events.jsonl', (old, old))
        server.INVENTORIES.clear()
        server.CONVERSATIONS.clear()
        ratings._VERIFIED.clear()
        row = self.run_row()
        self.assertEqual((row['failures']['deals'], row['failures']['interrupted']), (0, 1))
        self.assertIsNone(server.match(self.runs, conv['id'] + '/a-first')['failure'])
        pr = server.pairing(self.runs, 'r', 'm', 'station')
        self.assertEqual(pr['conversations'][0]['deals'][0][2], 'provisional')
        self.assertEqual(pr['totals']['failed'], 0)
        self.assertNotIn('errors:', self.status_text())


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class TestFailedInTheApp(unittest.TestCase):
    """The app's own arithmetic (viewer/core.js, run in node): a failed deal is in the sequence, uncounted."""

    def test_failed_deals_are_listed_and_never_counted(self):
        pr = {'a': 'm', 'b': 'station', 'conversations': [
            {'id': 'r/c1', 'round': 1, 'replicate': 1, 'deals_total': 1, 'deals': [[0, 4.0, 'verified', 'showdown']]},
            {'id': 'r/c2', 'round': 1, 'replicate': 2, 'deals_total': 1, 'deals': [[0, None, 'failed', None]]},
            {'id': 'r/c3', 'round': 1, 'replicate': 3, 'deals_total': 1, 'deals': [[0, -1.5, 'verified', 'flop']]}]}
        core = Path(server.__file__).with_name('viewer') / 'core.js'
        script = (f'const Core = require({json.dumps(str(core))}); const s = Core.dealSequence({json.dumps(pr)}, {{"r/c2": [0]}});'
                  ' process.stdout.write(JSON.stringify([s.map(x => x.status), Core.sequenceTotals(s), Core.sequenceSwings(s).map(x => x.net)]));')
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        statuses, tot, swings = json.loads(res.stdout)
        self.assertEqual(statuses, ['verified', 'failed', 'verified'])     # a deal the page saw played stays failed
        self.assertEqual((tot['failed'], tot['counted'], tot['net'], tot['playing'], tot['provisional']), (1, 2, 2.5, 0, 0))
        self.assertEqual(swings, [4.0, -1.5])


if __name__ == '__main__':
    unittest.main()
