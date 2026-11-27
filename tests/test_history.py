"""The history database: deal seeds and fingerprints stored, older databases upgraded in place, repeated deals
across runs averaged, only identical protocols pooled, unresolved identities and served-model mismatches
quarantined, and the all-time standings descriptive only."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from poker_arena import history
from poker_arena.ratings import fingerprint, ratings

GAME = {'name': 'holdem', 'unit': 'chips', 'big_unit': 2, 'sb': 1, 'bb': 2, 'stack': 200, 'hands_per_match': 20}


def write_matchup(root: Path, run, a, b, specs, nets, seed0=1000, protocols=None, served=None):
    """A finished duplicate pair whose deal k gives a `nets[k]` chips over its two hands. `protocols`
    ({pid: fingerprint}) as the runner records them; `served` ({pid: model}) a model that answered."""
    m = root / run / f'{a}__vs__{b}'
    for side, players in (('a-first', [a, b]), ('b-first', [b, a])):
        d = m / side
        d.mkdir(parents=True)
        seeds = [seed0 + k for k in range(len(nets))]
        cfg = {'game': {**GAME, 'hands_per_match': len(nets)}, 'players': players, 'seeds': seeds,
               'specs': {a: specs[a], b: specs[b]}}
        if protocols:
            cfg['protocol'] = {p: {'fingerprint': f, 'settings': {}} for p, f in protocols.items()}
        (d / 'config.json').write_text(json.dumps(cfg))
        if served:
            (d / 'calls.jsonl').write_text(''.join(json.dumps({'player': p, 'served_model': m, 'attempt': 0}) + '\n'
                                                   for p, m in served.items()))
        lines = []
        for k, net in enumerate(nets):
            half = net // 2 if side == 'a-first' else net - net // 2
            n = [half, -half] if side == 'a-first' else [-half, half]
            lines.append(json.dumps({'hand': k, 'button': k % 2, 'net': n, 'won': [0, 0], 'actions': [], 'board': []}))
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

    def test_protocols_never_pool(self):
        nets = [8, -4, 0, 12, -8, 4, 0, 0, 16, -12] * 2
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, nets, protocols={'m1': 'aaaaaaaa11', 'm2': 'bbbbbbbb22'})
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, nets, seed0=5000, protocols={'m1': 'cccccccc33', 'm2': 'bbbbbbbb22'})
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        ids = {p['player'] for p in r['players']}
        self.assertEqual(ids, {'claude:claude-x-1:high#aaaaaaaa', 'claude:claude-x-1:high#cccccccc', 'codex:gpt-y:high#bbbbbbbb'})
        labels = {r['labels'][i] for i in ids if i.startswith('claude')}
        self.assertEqual(len(labels), 2)                                  # told apart by protocol
        self.assertTrue(all('protocol' in lab for lab in labels))

    def test_served_model_mismatch_is_quarantined(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, served={'m2': 'gpt-old'})
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, [4, -2] * 10, seed0=3000, served={'m2': 'gpt-y'})
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        by = {p['player']: p for p in r['players']}
        bad = 'codex:gpt-y:high!served=gpt-old'
        self.assertEqual((by[bad]['quarantined'], by[bad]['in_fit']), (True, False))
        self.assertTrue(by['codex:gpt-y:high']['in_fit'])                # its other run still counts
        self.assertIn({'run': 'one', 'player': 'codex:gpt-y:high', 'reason': 'answered by gpt-old'}, r['left_out'])
        run = ratings(self.root / 'one', boot=30)
        m2 = next(p for p in run['players'] if p['player'] == 'm2')
        self.assertEqual((m2['quarantined'], m2['in_fit']), (True, True))
        self.assertIn('gpt-old', m2['quarantine_reason'])

    def test_all_time_is_descriptive(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [40, 30] * 30)
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, [40, 30] * 30, seed0=9000)
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        self.assertTrue(r['pooled'])
        self.assertEqual({m['verdict'] for m in r['pairings']}, {'descriptive'})
        self.assertIsNone(r['leader'])
        self.assertIsNone(r['groups'][0]['leader'])
        self.assertTrue(all(p['rank'] is None and p['rank_kind'] == 'descriptive' for p in r['players']))
        self.assertTrue(any('descriptive only' in n for n in r['notes']))

    def test_style_by_opponent_identity(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10)
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        s = r['stats']['claude:claude-x-1:high']
        self.assertEqual(set(s['by_opponent']), {'codex:gpt-y:high'})
        self.assertEqual(s['by_opponent']['codex:gpt-y:high']['hands'], 40)
        self.assertEqual(set(s['by_position']), {'button', 'big_blind'})
        self.assertEqual(s['counts']['vpip']['of'], 40)

    def test_schema_2_database_resyncs(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, protocols={'m1': 'aaaaaaaa11', 'm2': 'bbbbbbbb22'})
        db = self.root / history.DB_NAME
        history.sync(self.root)
        con = sqlite3.connect(db)
        con.execute("UPDATE deals SET a_identity = 'old'")
        con.execute("UPDATE meta SET v = '2' WHERE k = 'schema'")
        con.commit()
        con.close()
        self.assertGreater(history.sync(self.root)['matches'], 0)
        con = sqlite3.connect(db)
        self.assertEqual(con.execute('SELECT DISTINCT a_identity FROM deals').fetchall(), [('claude:claude-x-1:high#aaaaaaaa',)])
        con.close()

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
