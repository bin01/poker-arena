"""Stakes in dollars: the $10/$20 table people read (stakes.py, viewer/core.js), which must stay display only."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from poker_arena import runner, server, stakes
from poker_arena.games import get_game

VIEWER = Path(server.__file__).with_name('viewer')
PACKAGE = VIEWER.parent
# amounts in big blinds: table amounts, totals, and the small averages of mistake costs
AMOUNTS = [0, 0.5, 1, 3, 62, 100, 1547.5, -3, -100, 0.08, 0.25, 0.35, 0.4999, 0.5, 12.34, -0.0001, 0.0004, 1e6]


class Money(unittest.TestCase):
    def test_format(self):
        self.assertEqual(stakes.money(62), '$1,240')
        self.assertEqual(stakes.money(0.5), '$10')
        self.assertEqual(stakes.money(100), '$2,000')
        self.assertEqual(stakes.money(0), '$0')
        self.assertEqual(stakes.money_signed(62), '+$1,240')
        self.assertEqual(stakes.money_signed(-3), '-$60')
        self.assertEqual(stakes.money_signed(0), '$0')
        self.assertEqual(stakes.money_signed(-0.0001), '$0')      # never "-$0"
        self.assertEqual(stakes.chip_money(6, 2), '$60')
        self.assertEqual(stakes.per_hundred(50), '+$1,000 per 100 hands')

    def test_whole_dollars_and_never_fractional_cents(self):
        self.assertEqual(stakes.money(0.08), '$1.60')              # an average under $10 keeps its cents
        self.assertEqual(stakes.money(12.34), '$247')              # from $10 up: whole dollars
        for bb in AMOUNTS:
            self.assertRegex(stakes.money(bb), r'^\$\d{1,3}(,\d{3})*(\.\d{2})?$', bb)

    def test_table_amounts_are_whole_dollars(self):
        """Every amount the engine can produce is a whole number of chips, so a multiple of $10."""
        for chips in range(0, 401):
            self.assertRegex(stakes.chip_money(chips, 2), r'^\$\d{1,3}(,\d{3})*$')
            self.assertEqual(int(stakes.chip_money(chips, 2)[1:].replace(',', '')), chips * 10)

    def test_the_default_table(self):
        game = get_game('holdem')()
        k = stakes.stakes(game.config)
        self.assertEqual((k['sb'], k['bb'], k['stack'], k['stack_bb'], k['chip']), ('$10', '$20', '$2,000', 100, '$10'))
        self.assertEqual(k['line'], "$10/$20 No-Limit Hold'em · $2,000 stacks")
        self.assertEqual(stakes.stakes({'bb': 2})['line'], "$10/$20 No-Limit Hold'em")
        self.assertEqual(stakes.stakes({'sb': 1, 'bb': 2, 'stack': 400})['stack'], '$4,000')


class DisplayOnly(unittest.TestCase):
    def test_no_dollars_reach_a_model_or_the_fingerprint(self):
        """Dollars are for people. The prompt a model reads and every file the implementation fingerprint covers
        stay in chips, so a pinned run resumes without --allow-code-change."""
        game = get_game('holdem')()
        self.assertNotIn('$', game.system_prompt('a', ['a', 'b'], 50))
        for pattern in runner.IMPLEMENTATION_FILES:
            for f in PACKAGE.glob(pattern):
                text = f.read_text()
                self.assertNotRegex(text, r'(?m)^\s*(?:from|import) .*\bstakes\b', f.name)
                self.assertNotIn('BB_DOLLARS', text, f.name)

    def test_status_prints_dollars(self):
        src = (PACKAGE / 'cli.py').read_text()
        self.assertIn("money_signed(m['pair_net_a'] / m['bb'])", src)
        self.assertNotRegex(src, r"pair_net_a.*\bbb\"")


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class ViewerMoney(unittest.TestCase):
    def node(self, body):
        script = f'const C = require({json.dumps(str(VIEWER / "core.js"))});\n{body}'
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout)

    def test_viewer_and_terminal_agree(self):
        out = self.node(f'const xs = {json.dumps(AMOUNTS)};\n'
                        'console.log(JSON.stringify({bb: C.BB_DOLLARS, money: xs.map(C.money), signed: xs.map(C.moneySigned),'
                        ' line: C.stakesOf({sb: 1, bb: 2, stack: 200}).line, bare: C.stakesOf({bb: 2, stack: null}).line,'
                        ' rate: C.perHundred(50), chips: C.chipMoney(6, 2)}))')
        self.assertEqual(out['bb'], stakes.BB_DOLLARS)
        self.assertEqual(out['money'], [stakes.money(x) for x in AMOUNTS])
        self.assertEqual([s.replace('−', '-') for s in out['signed']], [stakes.money_signed(x) for x in AMOUNTS])
        self.assertEqual(out['line'], stakes.stakes({'sb': 1, 'bb': 2, 'stack': 200})['line'])
        self.assertEqual(out['rate'], stakes.per_hundred(50))
        self.assertEqual(out['chips'], '$60')

    def test_authored_text_and_engine_notes(self):
        out = self.node('const fmt = c => C.chipMoney(c, 2);\n'
                        'console.log(JSON.stringify({prose: C.bbProse("Opponent bets 10 bb into 30 bb; a 20+ bb loss at 50 bb/100"),'
                        ' low: C.correction({}, {action: "raise", amount: 8, note: "7 is below the minimum: raised to 8"}, fmt),'
                        ' pill: C.aivatPill({luck_correction: [5, -5]}, 0, 2).text, note: C.chipNote(2)}))')
        self.assertEqual(out['prose'], 'Opponent bets $200 into $600; a $400+ loss at 50 bb/100')
        self.assertEqual(out['low']['reason'], '$70 is below the minimum: raised to $80')
        self.assertEqual(out['low']['executed'], 'raise to $80')
        self.assertEqual(out['pill'], 'AIVAT: ran lucky $50')
        self.assertIn('1 chip is $10', out['note'])

    def test_pages_format_money_in_one_place(self):
        """No page writes a chip amount as "N bb" itself: amounts go through Core.money. bb/100 is a win rate."""
        for name in ('app.js', 'table.js'):
            src = (VIEWER / name).read_text()
            code = re.sub(r'//.*', '', src)
            self.assertNotRegex(code, r"\} bb\b(?!/100)", name)
            self.assertNotRegex(code, r"['\"`] bb['\"`<]", name)
            self.assertNotIn('toFixed(2)} bb', code)


if __name__ == '__main__':
    unittest.main()
