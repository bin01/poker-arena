"""Every run ever played, in one SQLite file, and the all-time leaderboard built from it.

The match folders stay the record of what happened; the database is an index of them that outlives them.
`sync` reads every match under a root into `arena.db` there (only folders that changed since the last
sync) and never removes anything, so a run whose folder is deleted keeps its place in the history.

The all-time leaderboard rates a player by what it is, not by its id in one run's players.toml: its model
and reasoning level (`claude:claude-sonnet-5-5:high`), or its bot and settings (`bot:rule:sims=200`).
Two ids with the same model and level pool their results; the same model at another level is a different
player. Only deals played under the same game settings pool (blinds and stacks change what bb/100 means),
so runs under other settings are listed as left out. Runs from different code versions do pool, and the
result says so when they differ.

Tables:
    matches   one row per match folder: run, matchup, seat order, game settings, code version, progress
    players   one row per player id in a run: its identity, display name and spec
    hands     every finished hand's summary (cards, board, actions, result), as hands.jsonl has it
    deals     every duplicate deal both of whose plays have finished: a's net chips over the two hands
    tallies   per match and player, the raw style counts that stats.summarize turns into rates
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from .stats import add, summarize, tally

SCHEMA = 1
DB_NAME = 'arena.db'
ALL = '*'   # the run name the viewer uses for the all-time leaderboard

TABLES = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY, run TEXT, matchup TEXT, side TEXT, players TEXT, identities TEXT,
    game TEXT, game_key TEXT, code_version TEXT, started REAL, hands_total INTEGER, hands_done INTEGER,
    finished INTEGER, config_sig TEXT, stamp TEXT, synced REAL);
CREATE TABLE IF NOT EXISTS players (
    run TEXT, pid TEXT, identity TEXT, label TEXT, kind TEXT, spec TEXT, PRIMARY KEY (run, pid));
CREATE TABLE IF NOT EXISTS hands (match_id TEXT, hand INTEGER, data TEXT, PRIMARY KEY (match_id, hand));
CREATE TABLE IF NOT EXISTS deals (
    run TEXT, matchup TEXT, deal INTEGER, a TEXT, b TEXT, a_identity TEXT, b_identity TEXT,
    net_a INTEGER, bb INTEGER, game_key TEXT, PRIMARY KEY (run, matchup, deal));
CREATE TABLE IF NOT EXISTS tallies (
    match_id TEXT, pid TEXT, identity TEXT, counts TEXT, served TEXT, PRIMARY KEY (match_id, pid));
"""


def identity(pid: str, spec: dict) -> str:
    """What a player is, whatever its id: backend, model and reasoning level, or bot and settings."""
    if 'bot' in spec:
        extra = ''.join(f':{k}={spec[k]}' for k in sorted(spec) if k not in ('bot', 'label'))
        return f'bot:{spec["bot"]}{extra}'
    backend = spec.get('backend', 'unknown')
    if spec.get('model'):
        return f'{backend}:{spec["model"]}:{spec.get("effort") or "default"}'
    return f'{backend}:{pid}'   # a CLI or command without a model setting: only its id says what it is


def game_key(game: dict) -> str:
    """The settings that make deals comparable: everything but the number of hands."""
    return json.dumps({k: v for k, v in sorted(game.items()) if k != 'hands_per_match'}, sort_keys=True)


def connect(db) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), timeout=30)
    con.executescript(TABLES)
    con.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(SCHEMA),))
    return con


def _read(p: Path):
    try:
        return p.read_text()
    except OSError:
        return ''


def _lines(text: str):
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _stamp(d: Path) -> str:
    parts = []
    for name in ('config.json', 'hands.jsonl', 'events.jsonl', 'calls.jsonl', 'result.json'):
        try:
            st = (d / name).stat()
            parts.append(f'{name}:{st.st_size}:{st.st_mtime_ns}')
        except OSError:
            pass
    return '|'.join(parts)


def _run_of(root: Path, mdir: Path) -> str:
    rel = mdir.parent.relative_to(root)
    return str(rel) if str(rel) != '.' else '.'


def sync(root, db=None) -> dict:
    """Bring the database up to date with every match under root. Returns counts of what changed."""
    from .server import display_name
    root = Path(root).resolve()
    con = connect(db or root / DB_NAME)
    changed = {'matches': 0, 'deals': 0}
    with con:
        for cfg_path in sorted(root.rglob('config.json')):
            d = cfg_path.parent
            if d.name not in ('a-first', 'b-first'):
                continue
            mid = str(d.relative_to(root))
            stamp = _stamp(d)
            row = con.execute('SELECT stamp, config_sig FROM matches WHERE id = ?', (mid,)).fetchone()
            if row and row[0] == stamp:
                continue
            text = _read(cfg_path)
            try:
                cfg = json.loads(text)
            except ValueError:
                continue
            sig = hashlib.sha1(text.encode()).hexdigest()
            mdir, run = d.parent, _run_of(root, d.parent)
            if row and row[1] != sig:   # the folder was started over: its old rows describe another match
                con.execute('DELETE FROM hands WHERE match_id = ?', (mid,))
                con.execute('DELETE FROM tallies WHERE match_id = ?', (mid,))
                con.execute('DELETE FROM deals WHERE run = ? AND matchup = ?', (run, mdir.name))
            pids, specs = cfg.get('players', []), cfg.get('specs', {})
            ids = [identity(p, specs.get(p, {})) for p in pids]
            hands = _lines(_read(d / 'hands.jsonl'))
            gk = game_key(cfg.get('game', {}))
            con.execute('INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                mid, run, mdir.name, d.name, json.dumps(pids), json.dumps(ids), json.dumps(cfg.get('game', {})), gk,
                cfg.get('code_version'), cfg.get('started'), len(cfg.get('seeds', [])), len(hands),
                int((d / 'result.json').exists()), sig, stamp, time.time()))
            for p, ident in zip(pids, ids):
                spec = specs.get(p, {})
                con.execute('INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?)', (
                    run, p, ident, display_name(p, spec), 'bot' if 'bot' in spec else 'model',
                    json.dumps({k: v for k, v in spec.items() if k in ('label', 'backend', 'model', 'effort', 'bot', 'sims')})))
            con.executemany('INSERT OR REPLACE INTO hands VALUES (?,?,?)',
                            [(mid, h['hand'], json.dumps(h)) for h in hands if 'hand' in h])
            counts, served = tally(d)
            con.executemany('INSERT OR REPLACE INTO tallies VALUES (?,?,?,?,?)', [
                (mid, p, ids[pids.index(p)] if p in pids else p, json.dumps(c), json.dumps(sorted(served.get(p, ()))))
                for p, c in counts.items()])
            changed['matches'] += 1
            changed['deals'] += _sync_deals(con, mdir, run, gk)
    con.close()
    return changed


def _sync_deals(con, mdir: Path, run: str, gk) -> int:
    """The duplicate deals of one matchup that both plays have finished (a is seat 0 of a-first)."""
    a_dir, b_dir = mdir / 'a-first', mdir / 'b-first'
    if not (a_dir / 'config.json').exists() or not (b_dir / 'config.json').exists():
        return 0
    try:
        cfg = json.loads(_read(a_dir / 'config.json'))
    except ValueError:
        return 0
    a, b = (cfg.get('players') or ['?', '?'])[:2]
    specs = cfg.get('specs', {})
    ids = [identity(a, specs.get(a, {})), identity(b, specs.get(b, {}))]
    bb = cfg.get('game', {}).get('big_unit') or cfg.get('game', {}).get('bb') or 1
    ha = {h['hand']: h for h in _lines(_read(a_dir / 'hands.jsonl')) if 'hand' in h}
    hb = {h['hand']: h for h in _lines(_read(b_dir / 'hands.jsonl')) if 'hand' in h}
    rows = [(run, mdir.name, k, a, b, ids[0], ids[1], ha[k]['net'][0] + hb[k]['net'][1], bb, gk)
            for k in sorted(set(ha) & set(hb))]
    con.executemany('INSERT OR REPLACE INTO deals VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
    return len(rows)


def current_labels(players_file='players.toml') -> dict:
    """{identity: label} for the players in the current players.toml that set a label."""
    try:
        from .runner import load_players
        return {identity(p, s): s['label'] for p, s in load_players(players_file).items() if s.get('label')}
    except Exception:
        return {}


def all_time(db, boot=200, players_file='players.toml', seed=0) -> dict:
    """The leaderboard over every run in the database, one player per model and reasoning level."""
    from .ratings import rate
    con = connect(db)
    rows = con.execute('SELECT run, matchup, deal, a_identity, b_identity, net_a, bb, game_key FROM deals').fetchall()
    matches = con.execute('SELECT run, game_key, code_version, started, hands_done FROM matches').fetchall()
    people = con.execute('SELECT run, pid, identity, label, kind, spec FROM players').fetchall()
    tallies = con.execute('SELECT t.identity, t.counts, t.served, m.game_key, m.identities FROM tallies t '
                          'JOIN matches m ON m.id = t.match_id').fetchall()
    con.close()

    # the settings most deals were played under; deals under other settings are listed, not pooled
    per_key = defaultdict(int)
    for r in rows:
        per_key[r[7]] += 1
    main = max(per_key, key=per_key.get) if per_key else None
    runs = defaultdict(lambda: {'deals': 0, 'self_play': 0, 'code_versions': set(), 'started': None, 'game_keys': set()})
    for run, gk, cv, started, _ in matches:
        x = runs[run]
        x['code_versions'].add(cv)
        x['game_keys'].add(gk)
        if started and (x['started'] is None or started < x['started']):
            x['started'] = started
    deals = []
    for run, mu, k, a, b, net, bb, gk in rows:
        if gk != main:
            continue
        if a == b:   # two ids for the same model and level: no information about how good it is
            runs[run]['self_play'] += 1
            continue
        runs[run]['deals'] += 1
        deals.append({'a': a, 'b': b, 'matchup': mu if run == '.' else f'{run}/{mu}', 'deal': k, 'net_a': net, 'bb': bb})
    r = rate(deals, boot=boot, seed=seed)

    # names: a label from the current players.toml, else the newest one recorded, else one made from the spec
    newest = {}
    kinds, pids = {}, defaultdict(set)
    order = sorted(people, key=lambda p: runs[p[0]]['started'] or 0)
    for run, pid, ident, lab, kind, spec in order:
        newest[ident] = lab
        kinds[ident] = kind
        pids[ident].add(pid)
    labels = {**newest, **{k: v for k, v in current_labels(players_file).items() if k in newest}}
    _disambiguate(labels)

    stats, served = defaultdict(dict), defaultdict(set)
    for ident, counts, sv, gk, seats in tallies:
        if gk == main and len(set(json.loads(seats))) > 1:   # as in the rating: no self-play
            add(stats[ident], json.loads(counts))
            served[ident] |= set(json.loads(sv))
    r['stats'] = {i: summarize(s, served[i]) for i, s in stats.items()}

    included = [run for run, x in runs.items() if x['deals']]
    left_out = [{'run': run, 'reason': 'other game settings'} for run, x in runs.items()
                if x['game_keys'] and main not in x['game_keys']]
    versions = sorted({v for run in included for v in runs[run]['code_versions'] if v})
    r.update({
        'all_time': True, 'labels': labels, 'kinds': kinds,
        'ids': {i: sorted(p) for i, p in pids.items()},
        'game': json.loads(main) if main else None,
        'runs': sorted(({'run': run, 'deals': x['deals'], 'self_play': x['self_play'], 'started': x['started'],
                         'code_versions': sorted(v for v in x['code_versions'] if v)}
                        for run, x in runs.items() if run in included),
                       key=lambda x: x['started'] or 0),
        'left_out': left_out,
        'code_versions': versions,
        'unversioned_runs': sorted(run for run in included if None in runs[run]['code_versions']),
    })
    return r


def _disambiguate(labels: dict):
    """Identities that would show the same name get what tells them apart: the backend of a model (one
    model through two CLIs), the settings of a bot, else the whole identity."""
    def tell(ident):
        parts = ident.split(':')
        return ':'.join(parts[2:]) or 'default' if parts[0] == 'bot' else parts[0]
    by = defaultdict(list)
    for ident, lab in labels.items():
        by[lab].append(ident)
    for lab, idents in by.items():
        if len(idents) > 1:
            unique = len({tell(i) for i in idents}) == len(idents)
            for ident in idents:
                labels[ident] = f'{lab} ({tell(ident) if unique else ident})'


def table(r) -> str:
    from .ratings import table as base, verdicts
    return base(r, labels=r.get('labels')) + '\n\n' + verdicts(r, labels=r.get('labels'))
