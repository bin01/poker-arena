"""The history database: deal seeds and fingerprints stored, older databases upgraded in place, repeated deals
across runs averaged, only identical protocols and implementations pooled (conversation length, code that
plays and CLI version included, unless an equivalence list merges them), duplicate pairs validated,
unresolved identities and served-model mismatches quarantined, and the all-time standings descriptive only."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from poker_arena import history
from poker_arena.ratings import fingerprint, ratings

GAME = {'name': 'holdem', 'unit': 'chips', 'big_unit': 2, 'sb': 1, 'bb': 2, 'stack': 200, 'hands_per_match': 20}


def write_matchup(root: Path, run, a, b, specs, nets, seed0=1000, protocols=None, served=None, legacy=False,
                  impl=None, cli=None, b_players=None):
    """A finished duplicate pair whose deal k gives a `nets[k]` chips over its two hands. `protocols`
    ({pid: fingerprint}) as the runner records them (with the match horizon, or without it when `legacy`);
    `served` ({pid: model}) a model that answered; `impl` ({file: hash}) the implementation recorded and `cli`
    the version of every model's CLI; `b_players` the b-first seats (default: a-first's swapped)."""
    m = root / run / f'{a}__vs__{b}'
    for side, players in (('a-first', [a, b]), ('b-first', b_players or [b, a])):
        d = m / side
        d.mkdir(parents=True)
        seeds = [seed0 + k for k in range(len(nets))]
        cfg = {'game': {**GAME, 'hands_per_match': len(nets)}, 'players': players, 'seeds': seeds,
               'specs': {a: specs[a], b: specs[b]}}
        if impl:
            cfg['implementation'] = {'files': impl, 'prompt_template': 'p0', 'fingerprint': 'f'}
            cfg['effective'] = [{'exe': specs[p].get('backend')} for p in players]
            cfg['cli_versions'] = {specs[p]['backend']: cli for p in players if cli and 'backend' in specs[p]}
        if protocols:
            settings = {} if legacy else {'horizon': {'hands_per_conversation': len(nets)}}
            cfg['protocol'] = {p: {'fingerprint': f, 'settings': settings} for p, f in protocols.items()}
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
        self.assertTrue(by['claude:sonnet:default~h20']['quarantined'])
        self.assertTrue(by['codex:nomodel~h20']['quarantined'])
        self.assertIsNone(by['codex:nomodel~h20']['strength'])
        self.assertIsNotNone(by['claude:claude-x-1:high~h20']['strength'])
        self.assertEqual(r['components'], [['claude:claude-x-1:high~h20', 'codex:gpt-y:high~h20']])
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

    def test_implementations_never_pool_unless_declared_equivalent(self):
        """The code that plays and the CLI version are part of what a player is (R4-#7); the judge is not, and
        an explicit equivalence list can merge implementations shown to play identically."""
        v1 = {'poker_arena/agents.py': 'a1', 'poker_arena/ratings.py': 'j1'}
        v2 = {'poker_arena/agents.py': 'a2', 'poker_arena/ratings.py': 'j1'}
        judge_only = {'poker_arena/agents.py': 'a1', 'poker_arena/ratings.py': 'j2'}
        nets = [4, -2] * 10
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, nets, impl=v1, cli='1.0')
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, nets, seed0=3000, impl=judge_only, cli='1.0')
        write_matchup(self.root, 'three', 'm1', 'm2', SPECS, nets, seed0=5000, impl=v2, cli='1.0')
        write_matchup(self.root, 'four', 'm1', 'm2', SPECS, nets, seed0=7000, impl=v1, cli='2.0')
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        m1 = sorted(p['player'] for p in r['players'] if p['player'].startswith('claude'))
        self.assertEqual(len(m1), 3)                                       # v1 (and judge-only), v2, v1 with CLI 2.0
        tags = sorted({i.split('@')[1] for i in m1})
        self.assertTrue(any('different implementations' in n for n in r['notes']))
        self.assertTrue(all('implementation' in r['labels'][i] for i in m1), [r['labels'][i] for i in m1])
        (self.root / 'equivalences.json').write_text(json.dumps({'implementations': [tags], 'reason': 'test'}))
        merged = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        self.assertEqual(len([p for p in merged['players'] if p['player'].startswith('claude')]), 1)
        self.assertTrue(any('equivalent implementations' in n for n in merged['notes']))

    def test_duplicate_pairs_are_validated(self):
        """A b-first play that is not a-first with the seats swapped never counts (R4-#8)."""
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10)
        write_matchup(self.root, 'one', 'm1', 'alias', SPECS, [4, -2] * 10, seed0=3000, b_players=['m1', 'alias'])
        history.sync(self.root)
        con = sqlite3.connect(self.root / history.DB_NAME)
        self.assertEqual(con.execute("SELECT count(*) FROM deals WHERE matchup = 'm1__vs__alias'").fetchone(), (0,))
        [(deal, why)] = con.execute("SELECT deal, reason FROM rejected").fetchall()
        con.close()
        self.assertEqual(deal, -1)
        self.assertIn('b-first seats', why)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        self.assertEqual(len(r['rejected']), 1)
        self.assertTrue(any('failed validation' in n for n in r['notes']))
        run = ratings(self.root / 'one', boot=30)
        self.assertEqual([x['matchup'] for x in run['rejected']], ['m1__vs__alias'])

    def test_conversation_length_never_pools(self):
        """The horizon is part of what a model player is (R3-#9): the runner's protocol now records it; for an
        older protocol (or none) the identity carries it, so 20-hand and 30-hand conversations stay apart."""
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, protocols={'m1': 'aaaaaaaa11', 'm2': 'bbbbbbbb22'},
                      legacy=True)
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, [4, -2] * 15, seed0=5000,
                      protocols={'m1': 'aaaaaaaa11', 'm2': 'bbbbbbbb22'}, legacy=True)
        write_matchup(self.root, 'three', 'm1', 'm2', SPECS, [4, -2] * 15, seed0=9000,
                      protocols={'m1': 'dddddddd11', 'm2': 'eeeeeeee22'})
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        ids = {p['player'] for p in r['players']}
        self.assertEqual(ids, {'claude:claude-x-1:high#aaaaaaaa~h20', 'claude:claude-x-1:high#aaaaaaaa~h30',
                               'codex:gpt-y:high#bbbbbbbb~h20', 'codex:gpt-y:high#bbbbbbbb~h30',
                               'claude:claude-x-1:high#dddddddd', 'codex:gpt-y:high#eeeeeeee'})
        self.assertEqual(len(r['components']), 3)
        labels = [r['labels'][i] for i in ids if '#aaaaaaaa' in i]
        self.assertTrue(all('hands per conversation' in lab for lab in labels), labels)

    def test_served_model_mismatch_is_quarantined(self):
        write_matchup(self.root, 'one', 'm1', 'm2', SPECS, [4, -2] * 10, served={'m2': 'gpt-old'})
        write_matchup(self.root, 'two', 'm1', 'm2', SPECS, [4, -2] * 10, seed0=3000, served={'m2': 'gpt-y'})
        history.sync(self.root)
        r = history.all_time(self.root / history.DB_NAME, boot=30, players_file='/none')
        by = {p['player']: p for p in r['players']}
        bad = 'codex:gpt-y:high~h20!served=gpt-old'
        self.assertEqual((by[bad]['quarantined'], by[bad]['in_fit']), (True, False))
        self.assertTrue(by['codex:gpt-y:high~h20']['in_fit'])            # its other run still counts
        self.assertIn({'run': 'one', 'player': 'codex:gpt-y:high~h20', 'reason': 'answered by gpt-old'}, r['left_out'])
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
        s = r['stats']['claude:claude-x-1:high~h20']
        self.assertEqual(set(s['by_opponent']), {'codex:gpt-y:high~h20'})
        self.assertEqual(s['by_opponent']['codex:gpt-y:high~h20']['hands'], 40)
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
