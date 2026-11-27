"""The history database: deal seeds and fingerprints stored, a schema-1 database upgraded in place, repeated
deals across runs averaged in the all-time standings, and unresolved identities quarantined."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from poker_arena import history
from poker_arena.ratings import fingerprint, ratings

GAME = {'name': 'holdem', 'unit': 'chips', 'big_unit': 2, 'sb': 1, 'bb': 2, 'stack': 200, 'hands_per_match': 20}


def write_matchup(root: Path, run, a, b, specs, nets, seed0=1000):
    """A finished duplicate pair whose deal k gives a `nets[k]` chips over its two hands."""
    m = root / run / f'{a}__vs__{b}'
    for side, players in (('a-first', [a, b]), ('b-first', [b, a])):
        d = m / side
        d.mkdir(parents=True)
        seeds = [seed0 + k for k in range(len(nets))]
        (d / 'config.json').write_text(json.dumps({'game': {**GAME, 'hands_per_match': len(nets)}, 'players': players,
                                                   'seeds': seeds, 'specs': {a: specs[a], b: specs[b]}}))
        lines = []
        for k, net in enumerate(nets):
            half = net // 2 if side == 'a-first' else net - net // 2
            n = [half, -half] if side == 'a-first' else [-half, half]
            lines.append(json.dumps({'hand': k, 'net': n, 'won': [0, 0], 'actions': [], 'board': []}))
        (d / 'hands.jsonl').write_text('\n'.join(lines) + '\n')
        (d / 'result.json').write_text('{}')


SPECS = {'m1': {'backend': 'claude', 'model': 'claude-x-1', 'effort': 'high'},
         'm2': {'backend': 'codex', 'model': 'gpt-y', 'effort': 'high'},
         'alias': {'backend': 'claude', 'model': 'sonnet'},
         'nomodel': {'backend': 'codex', 'effort': 'high'}}


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_repeats_across_runs_are_averaged(self):
        nets = [8, -4, 0, 12, -8, 4, 0, 0, 16, -12] * 2
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, nets)
        write_matchup(self.root, 'two', 'm2', 'm1', SPECS, [-x for x in nets])   # same seeds, sides swapped
        history.sync(self.root)
        con = sqlite3.connect(self.root / history.DB_NAME)
        fps = con.execute("SELECT seed, fp FROM deals WHERE run = 'one' ORDER BY deal").fetchall()
        con.close()
        self.assertEqual(fps[0], (1000, fingerprint(GAME, 1000, 0)))
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        self.assertEqual((r['deals'], r['observations'], r['repeats']), (40, 20, 20))
        [p] = r['pairings']
        self.assertEqual((p['plays'], p['deals']), (40, 20))
        self.assertAlmostEqual(abs(p['edge']), sum(nets) / 2 / 2 * 100 / 20, places=1)

    def test_quarantine_in_all_time(self):
        write_matchup(self.root, 'one', 'm1', 'alias', SPECS, [4, -2] * 10)
        write_matchup(self.root, 'one', 'm1', 'nomodel', SPECS, [4, -2] * 10, seed0=2000)
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [2, -2] * 10, seed0=3000)
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        by = {p['player']: p for p in r['players']}
        self.assertTrue(by['claude:sonnet:default']['quarantined'])
        self.assertTrue(by['codex:nomodel']['quarantined'])
        self.assertIsNone(by['codex:nomodel']['strength'])
        self.assertIsNotNone(by['claude:claude-x-1:high']['strength'])
        self.assertEqual(r['components'], [['claude:claude-x-1:high', 'codex:gpt-y:high']])
        # within its own run the same players are rated, and flagged
        run = ratings(self.root / 'one', boot=30)
        self.assertTrue(next(p for p in run['players'] if p['player'] == 'alias')['quarantined'])
        self.assertEqual(len(run['components'][0]), 4)

    def test_schema_1_database_upgrades(self):
        db = self.root / history.DB_NAME
        con = sqlite3.connect(db)
        con.executescript("""
            CREATE TABLE deals (run TEXT, matchup TEXT, deal INTEGER, a TEXT, b TEXT, a_identity TEXT, b_identity TEXT,
                net_a INTEGER, bb INTEGER, game_key TEXT, PRIMARY KEY (run, matchup, deal));
            CREATE TABLE matches (id TEXT PRIMARY KEY, run TEXT, matchup TEXT, side TEXT, players TEXT, identities TEXT,
                game TEXT, game_key TEXT, code_version TEXT, started REAL, hands_total INTEGER, hands_done INTEGER,
                finished INTEGER, config_sig TEXT, stamp TEXT, synced REAL);
            INSERT INTO deals VALUES ('gone', 'x__vs__y', 0, 'x', 'y', 'bot:x', 'bot:y', 4, 2, '{}');""")
        con.commit()
        con.close()
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10)
        history.sync(self.root)
        con = sqlite3.connect(db)
        self.assertIn('fp', {r[1] for r in con.execute('PRAGMA table_info(deals)')})
        self.assertEqual(con.execute("SELECT fp FROM deals WHERE run = 'gone'").fetchone(), (None,))
        self.assertEqual(con.execute("SELECT count(*) FROM deals WHERE run = 'one' AND fp IS NOT NULL").fetchone(), (20,))
        con.close()


if __name__ == '__main__':
    unittest.main()
