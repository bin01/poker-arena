"""Every run ever played, in one SQLite file, and the all-time leaderboard built from it.

The match folders stay the record of what happened; the database is an index of them that outlives them.
`sync` reads every match under a root into `arena.db` there (only folders that changed since the last
sync) and never removes anything, so a run whose folder is deleted keeps its place in the history.

The all-time leaderboard rates a player by what it is, not by its id in one run's players.toml: its model,
reasoning level and protocol (`claude:claude-sonnet-5-5:high#1a2b3c4d`, the protocol being the fingerprint of
every setting that changes play, as the runner records it, the match horizon and memory resets included), or
its bot and settings (`bot:rule:sims=200`), and the implementation that played it (`@5e6f7a8b`: the code that
plays, i.e. the engine, agents, backends, runner and prompt template, not the judge, and for a model the
version and file hash of its CLI). Only identical protocols and implementations pool: the same model with other
instructions, isolation, tool policy, retry policy or conversation length is another player, and so is one
played by other parsing, correction or tool-detection code or another CLI version or CLI file, and a player from before
protocols were recorded. The implementation is taken hand by hand (`implementation_segments`): a code or CLI
version change during a match splits the player at the hand it took effect, and a deal whose two plays
straddle a change belongs to neither side. An explicit equivalence list (`equivalences.json` beside the database:
{"implementations": [["5e6f7a8b", "9a0b1c2d", ...], ...], "reason": "..."}) merges implementations shown to
play identically; each group pools under its first member, and the all-time output lists what was merged. A model whose recorded protocol predates the horizon carries it
in its identity instead (`...#1a2b3c4d~h50`: 50 hands per conversation, memory reset at every match), so
conversations of different lengths never pool. A player whose model was
not recorded, or only as a bare alias (`sonnet`), and a player for whom a different model answered than the
one configured (in that run), is quarantined: listed, but left out of the all-time fit. Only deals played
under the same game settings pool (blinds and stacks change what bb/100 means); runs under other settings
are listed as left out. The same pair often replays the same deal in several runs: each deal row keeps its
seed and fingerprint, and the rating averages repeats into one observation. A duplicate pair is counted only
if its b-first play is its a-first play with the seats swapped (ratings.check_pair) and both of its hands replay
from their seeds through the engine's record verifier (ratings.verify_play), as the one reader of conversations,
ratings.read_conversation, decides for every consumer; the rest are kept in the `rejected` table and listed.

All-time standings are descriptive only. Runs had different designs and stopped pairings at different
times, so pooling them is not any run's preregistered test: no all-time verdict, rank or leader is decided.

Tables:
    matches   one row per match folder: run, matchup, seat order, game settings, code version, progress
    players   one row per player id in a run: its identity, display name and spec
    hands     every finished hand's summary (cards, board, actions, result), as hands.jsonl has it
    deals     every duplicate deal both of whose plays have finished: a's net chips over the two hands, its seed
              and fingerprint (ratings.fingerprint)
    tallies   per match and player, the raw style counts of its verified results that stats.summarize turns into rates
    rejected  duplicate pairs (or single deals of them) that failed validation, with the reason
    runs      each run's track ('official', or 'adaptation': long conversations, model-based) and tournament

Tracks never pool: all-time standings are built from one track's runs at a time (official by default).
"""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from .stats import add, summarize, tally

SCHEMA = 7
DB_NAME = 'arena.db'
ALL = '*'   # the run name the viewer uses for the all-time leaderboard

TABLES = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY, run TEXT, matchup TEXT, side TEXT, players TEXT, identities TEXT,
    game TEXT, game_key TEXT, code_version TEXT, started REAL, hands_total INTEGER, hands_done INTEGER,
    finished INTEGER, config_sig TEXT, stamp TEXT, synced REAL);
CREATE TABLE IF NOT EXISTS players (
    run TEXT, pid TEXT, identity TEXT, label TEXT, kind TEXT, spec TEXT, PRIMARY KEY (run, pid, identity));
CREATE TABLE IF NOT EXISTS hands (match_id TEXT, hand INTEGER, data TEXT, PRIMARY KEY (match_id, hand));
CREATE TABLE IF NOT EXISTS deals (
    run TEXT, matchup TEXT, deal INTEGER, a TEXT, b TEXT, a_identity TEXT, b_identity TEXT,
    net_a INTEGER, bb INTEGER, game_key TEXT, seed INTEGER, fp TEXT, PRIMARY KEY (run, matchup, deal));
CREATE TABLE IF NOT EXISTS tallies (
    match_id TEXT, pid TEXT, identity TEXT, counts TEXT, served TEXT, PRIMARY KEY (match_id, pid));
CREATE TABLE IF NOT EXISTS rejected (
    run TEXT, matchup TEXT, deal INTEGER, reason TEXT, PRIMARY KEY (run, matchup, deal));
CREATE TABLE IF NOT EXISTS runs (run TEXT PRIMARY KEY, track TEXT, tournament TEXT);
"""


def identity(pid: str, spec: dict, protocol: str | None = None, horizon: int | None = None,
             implementation: str | None = None) -> str:
    """What a player is, whatever its id: backend, model, reasoning level and the first 8 characters of its
    protocol fingerprint when one was recorded, plus `~h<hands>` (`horizon`: hands per conversation) when that
    protocol does not include the match horizon; or bot and settings. Then `@<implementation>` when the match
    recorded the implementation that played (see `implementation_tag`)."""
    impl = f'@{implementation}' if implementation else ''
    if 'bot' in spec:
        extra = ''.join(f':{k}={spec[k]}' for k in sorted(spec) if k not in ('bot', 'label'))
        return f'bot:{spec["bot"]}{extra}{impl}'
    backend = spec.get('backend', 'unknown')
    tag = (f'#{protocol[:8]}' if protocol else '') + (f'~h{horizon}' if horizon else '') + impl
    if spec.get('model'):
        return f'{backend}:{spec["model"]}:{spec.get("effort") or "default"}{tag}'
    return f'{backend}:{pid}{tag}'   # a CLI or command without a model setting: only its id says what it is


def _tag(impl: dict | None, cli_versions: dict | None, cfg: dict, pid: str, cli_files: dict | None = None) -> str | None:
    if not isinstance(impl, dict) or not impl.get('files'):
        return None
    files = {k: v for k, v in impl['files'].items() if not k.endswith('/ratings.py')}
    spec = (cfg.get('specs') or {}).get(pid) or {}
    cli = sha = None
    if 'bot' not in spec:
        players, eff = cfg.get('players') or [], cfg.get('effective') or []
        seat = players.index(pid) if pid in players else None
        exe = (eff[seat] or {}).get('exe') if seat is not None and seat < len(eff) else None
        versions, shas = cli_versions or {}, cli_files or {}
        cli = versions.get(exe) if exe and exe in versions else (dict(sorted(versions.items())) or None)
        sha = shas.get(exe) if exe and exe in shas else (dict(sorted(shas.items())) or None)
    key = {'files': files, 'prompt_template': impl.get('prompt_template'), 'cli': cli}
    if sha:
        key['cli_file'] = sha
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]


UNRECORDED = 'unrecorded'   # a CLI file the record lost (replaced by a later one): kept apart from every known file


def _exes_changed(c: dict) -> set:
    """The executables one recorded CLI change touched: the mid-run transition's, or those its drift names."""
    out = {str(a.get('exe')) for a in (c.get('artifact_to'), c.get('artifact_from')) if isinstance(a, dict) and a.get('exe')}
    return out | {str(x).split(':')[0] for x in c.get('clis') or [] if ':' in str(x)}


def _files_at_start(cfg: dict, state: dict | None) -> dict:
    """{exe: sha256} of the CLI files a match started with. The runner keeps the latest file in `cli_artifacts`,
    so for a CLI that changed the start is the change's `artifact_from` / `artifacts_from`, else a resume's record
    from before it, else UNRECORDED."""
    shas = {e: (a or {}).get('sha256') for e, a in (cfg.get('cli_artifacts') or {}).items() if isinstance(a, dict)}
    resumes = sorted((state or {}).get('resumes') or [], key=lambda r: int(r.get('after_hand') or 0))
    done = set()
    for c in sorted(cfg.get('cli_version_changes') or [], key=lambda c: int(c.get('after_hand') or 0)):
        at = int(c.get('after_hand') or 0)
        for e in _exes_changed(c) - done:
            done.add(e)
            frm = c.get('artifact_from') if (c.get('artifact_from') or {}).get('exe') == e else \
                (c.get('artifacts_from') or {}).get(e)
            if not isinstance(frm, dict):
                frm = next(((r.get('cli_artifacts') or {}).get(e) for r in resumes
                            if int(r.get('after_hand') or 0) < at and (r.get('cli_artifacts') or {}).get(e)), None)
            shas[e] = (frm or {}).get('sha256') or UNRECORDED
    return {e: v for e, v in shas.items() if v}


def implementation_segments(cfg: dict, pid: str, state: dict | None = None) -> list[tuple[int, str | None]]:
    """[(first hand, implementation tag)] of what played `pid` in a match, in hand order: the implementation
    and CLI versions the match started with, then each accepted code change (implementation_changes), CLI
    version change (cli_version_changes) and CLI version seen at a resume (state.json `resumes`) from the hand
    it took effect. A tag is the first 8 characters of a hash of the code that plays (every recorded
    implementation file but the judge, ratings.py), the prompt template and, for a model, the version and the
    file (sha256, `cli_artifacts` and each change's artifacts) of its CLI (every recorded CLI when its executable
    is not recorded), so a rebuilt CLI that keeps its version string is another implementation too."""
    impl, clis = cfg.get('implementation'), dict(cfg.get('cli_versions') or {})
    shas = _files_at_start(cfg, state)
    steps = [(int(c.get('after_hand') or 0), 0, 'impl', c.get('to')) for c in cfg.get('implementation_changes') or []]
    steps += [(int(c.get('after_hand') or 0), 1, 'cli', c) for c in cfg.get('cli_version_changes') or []]
    steps += [(int(r.get('after_hand') or 0), 2, 'resume', r) for r in (state or {}).get('resumes') or []
              if r.get('cli_versions') or r.get('cli_artifacts')]
    out = [(0, _tag(impl, clis, cfg, pid, shas))]
    for at, _, kind, value in sorted(steps, key=lambda x: (x[0], x[1])):
        if kind == 'impl' and isinstance(value, dict):
            impl = value
        elif kind == 'cli':
            if isinstance(value.get('to'), dict):
                clis = {**clis, **value['to']}
            to = {**(value.get('artifacts_to') or {})}
            if isinstance(value.get('artifact_to'), dict) and value['artifact_to'].get('exe'):
                to[value['artifact_to']['exe']] = value['artifact_to']
            shas = {**shas, **{e: a.get('sha256') for e, a in to.items() if isinstance(a, dict) and a.get('sha256')}}
        elif kind == 'resume':
            clis = {**clis, **(value.get('cli_versions') or {})}
            shas = {**shas, **{e: a.get('sha256') for e, a in (value.get('cli_artifacts') or {}).items()
                               if isinstance(a, dict) and a.get('sha256')}}
        tag = _tag(impl, clis, cfg, pid, shas)
        if tag != out[-1][1]:
            out = [x for x in out if x[0] < at] + [(at, tag)]
    return out


def tag_at(segments, hand: int) -> str | None:
    tag = segments[0][1] if segments else None
    for at, t in segments:
        if at <= hand:
            tag = t
    return tag


def combined_tag(tags) -> str | None:
    """One tag for play by several implementations (a deal whose two plays, or a match whose hands, were not
    all played by the same one): never the tag of any of them, so such play pools with neither."""
    tags = sorted({t for t in tags if t})
    if len(tags) <= 1:
        return tags[0] if tags else None
    return hashlib.sha1('+'.join(tags).encode()).hexdigest()[:8]


def implementation_tag(cfg: dict, pid: str, state: dict | None = None) -> str | None:
    """The implementation tag of a whole match for `pid` (`implementation_segments`): its one tag, or
    `combined_tag` of them when the code or a CLI changed during the match. None when the match recorded no
    implementation."""
    return combined_tag(t for _, t in implementation_segments(cfg, pid, state))


def match_identity(cfg: dict, pid: str, state: dict | None = None, hand: int | None = None) -> str:
    """A player's identity in one match config (at one hand of it, when `hand` is given)."""
    from .ratings import protocol_of
    tag = implementation_tag(cfg, pid, state) if hand is None else tag_at(implementation_segments(cfg, pid, state), hand)
    return identity(pid, (cfg.get('specs') or {}).get(pid, {}), protocol_of(cfg, pid), legacy_horizon(cfg, pid), tag)


def legacy_horizon(cfg: dict, pid: str) -> int | None:
    """The hands per conversation of a match whose recorded protocol for `pid` leaves the horizon out (older
    runs), so the identity can carry it; None when the protocol has it."""
    p = cfg.get('protocol')
    p = p.get(pid) if isinstance(p, dict) and 'fingerprint' not in p else p
    if isinstance(p, dict) and isinstance(p.get('settings'), dict) and 'horizon' in p['settings']:
        return None
    return len(cfg.get('seeds') or []) or None


def game_key(game: dict) -> str:
    """The settings that make deals comparable: everything but the number of hands."""
    return json.dumps({k: v for k, v in sorted(game.items()) if k != 'hands_per_match'}, sort_keys=True)


def connect(db) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), timeout=30)
    con.executescript(TABLES)
    cols = {r[1] for r in con.execute('PRAGMA table_info(deals)')}
    if 'fp' not in cols:   # schema 1: add the deal fingerprints, and read every folder again to fill them
        with con:
            con.execute('ALTER TABLE deals ADD COLUMN seed INTEGER')
            con.execute('ALTER TABLE deals ADD COLUMN fp TEXT')
            con.execute("UPDATE matches SET stamp = ''")
    row = con.execute("SELECT v FROM meta WHERE k = 'schema'").fetchone()
    if row and int(row[0]) < 7:   # identities gain the protocol, horizon, implementation, its changes, CLI files: read again
        with con:
            con.execute('DROP TABLE players')
            con.executescript(TABLES)
            con.execute("UPDATE matches SET stamp = ''")
    with con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)", (str(SCHEMA),))
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
    for name in ('config.json', 'hands.jsonl', 'events.jsonl', 'calls.jsonl', 'raw.jsonl', 'result.json', 'state.json'):
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
    from .ratings import track_of
    with con:
        for plan_path in sorted(root.rglob('plan.json')):
            plan = _plan(plan_path)
            if plan is None:
                continue
            rel = plan_path.parent.relative_to(root)
            con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?)',
                        (str(rel) if str(rel) != '.' else '.', track_of(plan), plan.get('tournament')))
        for cfg_path in sorted(root.rglob('config.json')):
            d = cfg_path.parent
            if d.name not in ('a-first', 'b-first'):
                continue
            mid = str(d.relative_to(root))
            stamp = _stamp(d) + '#' + _stamp(d.parent / ('b-first' if d.name == 'a-first' else 'a-first'))
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
            state = _state(d)
            ids = [match_identity(cfg, p, state) for p in pids]
            hands = _lines(_read(d / 'hands.jsonl'))
            gk = game_key(cfg.get('game', {}))
            con.execute('INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                mid, run, mdir.name, d.name, json.dumps(pids), json.dumps(ids), json.dumps(cfg.get('game', {})), gk,
                cfg.get('code_version'), cfg.get('started'), len(cfg.get('seeds', [])), len(hands),
                int((d / 'result.json').exists()), sig, stamp, time.time()))
            for p in pids:
                spec = specs.get(p, {})
                every = {match_identity(cfg, p, state)} | {match_identity(cfg, p, state, at)
                                                           for at, _ in implementation_segments(cfg, p, state)}
                for ident in sorted(every):
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


def _plan(p: Path) -> dict | None:
    try:
        x = json.loads(_read(p) or 'null')
    except ValueError:
        return None
    return x if isinstance(x, dict) else None


def _state(d: Path) -> dict:
    try:
        x = json.loads(_read(d / 'state.json') or '{}')
    except ValueError:
        return {}
    return x if isinstance(x, dict) else {}


def _sync_deals(con, mdir: Path, run: str, gk) -> int:
    """The duplicate deals of one matchup that both plays have finished, that pass ratings.check_pair and whose
    hands replay from their seeds (ratings.verify_play), a being seat 0 of a-first; what fails is kept in
    `rejected` (deal -1: the whole conversation). Each deal carries its players' identities at that hand of
    both plays: a deal played on both sides of a code or CLI version change gets an identity of its own."""
    a_dir, b_dir = mdir / 'a-first', mdir / 'b-first'
    if not (a_dir / 'config.json').exists() or not (b_dir / 'config.json').exists():
        return 0
    try:
        cfg = json.loads(_read(a_dir / 'config.json'))
    except ValueError:
        return 0
    try:
        cb = json.loads(_read(b_dir / 'config.json'))
    except ValueError:
        cb = None
    from .ratings import fingerprint, read_conversation
    players = cfg.get('players') or ['?', '?']
    a, b = players[:2]
    game = cfg.get('game', {})
    bb = game.get('big_unit') or game.get('bb') or 1
    seeds = cfg.get('seeds', [])
    conv = read_conversation(mdir) or {'hands': ({}, {}), 'verified': [], 'rejected': []}
    ha, hb = conv['hands']
    good, rejected = list(conv['verified']), list(conv['rejected'])
    con.execute('DELETE FROM deals WHERE run = ? AND matchup = ?', (run, mdir.name))
    con.execute('DELETE FROM rejected WHERE run = ? AND matchup = ?', (run, mdir.name))
    con.executemany('INSERT OR REPLACE INTO rejected VALUES (?,?,?,?)',
                    [(run, mdir.name, -1 if x['hand'] is None else x['hand'], x['reason']) for x in rejected])
    sa, sb = _state(a_dir), _state(b_dir)
    seg = {(side, p): implementation_segments(c, p, st) for side, c, st in (('a', cfg, sa), ('b', cb or {}, sb))
           for p in (a, b)}
    rows = []
    for k in good:
        seed = seeds[k] if k < len(seeds) else None
        ids = []
        for p in (a, b):
            tag = combined_tag([tag_at(seg[('a', p)], k), tag_at(seg[('b', p)], k)])
            ids.append(identity(p, (cfg.get('specs') or {}).get(p, {}), _protocol(cfg, p), legacy_horizon(cfg, p), tag))
        rows.append((run, mdir.name, k, a, b, ids[0], ids[1], ha[k]['net'][0] + hb[k]['net'][1], bb, gk, seed,
                     fingerprint(game, seed, k % len(players))))
    con.executemany('INSERT OR REPLACE INTO deals (run, matchup, deal, a, b, a_identity, b_identity, net_a, bb, '
                    'game_key, seed, fp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    from .server import display_name
    for p, ident in sorted({(a, r[5]) for r in rows} | {(b, r[6]) for r in rows}):
        spec = (cfg.get('specs') or {}).get(p, {})
        con.execute('INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?)', (
            run, p, ident, display_name(p, spec), 'bot' if 'bot' in spec else 'model',
            json.dumps({k: v for k, v in spec.items() if k in ('label', 'backend', 'model', 'effort', 'bot', 'sims')})))
    return len(rows)


def _protocol(cfg, pid):
    from .ratings import protocol_of
    return protocol_of(cfg, pid)


def current_labels(players_file='players.toml') -> dict:
    """{identity: label} for the players in the current players.toml that set a label."""
    try:
        from .runner import load_players
        return {_base(identity(p, s)): s['label'] for p, s in load_players(players_file).items() if s.get('label')}
    except Exception:
        return {}


def _base(ident: str) -> str:
    return re.split(r'[#~!@]', ident, maxsplit=1)[0]


def load_equivalences(path) -> dict:
    """{implementation tag: the tag it pools under} from an equivalences.json ({"implementations": [[tag, ...],
    ...]}); empty when there is none."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for group in doc.get('implementations') or []:
        tags = [str(t)[:8] for t in group if t]
        for t in tags[1:]:
            out[t] = tags[0]
    return out


def _canon(ident: str, eq: dict) -> str:
    return re.sub(r'@([0-9a-f]{8})', lambda m: '@' + eq.get(m.group(1), m.group(1)), ident)


def all_time(db, boot=200, players_file='players.toml', seed=0, equivalences=None, track='official') -> dict:
    """The leaderboard over every run of one track in the database (official by default; the tracks never pool),
    one player per model, reasoning level, protocol and implementation (`equivalences`: {tag: tag it pools
    under}, by default from equivalences.json beside the database). Descriptive only: no verdict, rank or leader
    here is a test result."""
    from .ratings import quarantine_reason, identity_resolved, rate, served_mismatch
    eq = load_equivalences(Path(db).parent / 'equivalences.json') if equivalences is None else dict(equivalences)
    con = connect(db)
    tracks = dict(con.execute('SELECT run, track FROM runs').fetchall())
    mine = lambda run: (tracks.get(run) or 'official') == track
    rows = [r for r in con.execute('SELECT run, matchup, deal, a_identity, b_identity, net_a, bb, game_key, fp, seed '
                                   'FROM deals').fetchall() if mine(r[0])]
    matches = [m for m in con.execute('SELECT run, game_key, code_version, started, hands_done, finished, identities '
                                      'FROM matches').fetchall() if mine(m[0])]
    people = [x for x in con.execute('SELECT run, pid, identity, label, kind, spec FROM players').fetchall() if mine(x[0])]
    tallies = [t for t in con.execute('SELECT m.run, t.identity, t.counts, t.served, m.game_key, m.identities, m.players '
                                      'FROM tallies t JOIN matches m ON m.id = t.match_id').fetchall() if mine(t[0])]
    rejected = [{'run': run, 'matchup': mu, 'hand': None if k < 0 else k, 'reason': why}
                for run, mu, k, why in con.execute('SELECT run, matchup, deal, reason FROM rejected ORDER BY run, matchup, deal')
                if mine(run)]
    others = sorted({r for r, t in tracks.items() if (t or 'official') != track})
    con.close()
    rows = [(run, mu, k, _canon(a, eq), _canon(b, eq), *rest) for run, mu, k, a, b, *rest in rows]
    matches = [(*m[:6], json.dumps([_canon(i, eq) for i in json.loads(m[6] or '[]')])) for m in matches]
    people = [(run, pid, _canon(ident, eq), *rest) for run, pid, ident, *rest in people]
    tallies = [(run, _canon(ident, eq), c, sv, gk, json.dumps([_canon(i, eq) for i in json.loads(idents or '[]')]), pl)
               for run, ident, c, sv, gk, idents, pl in tallies]

    # a model that answered for a player in a run, other than the one configured: that run's deals of it are
    # quarantined under their own identity
    served = defaultdict(set)
    for run, ident, _, sv, *_ in tallies:
        served[(run, ident)] |= set(json.loads(sv or '[]'))
    swap = {}
    for (run, ident), models in served.items():
        parts = _base(ident).split(':')
        wrong = served_mismatch(parts[1], models) if len(parts) >= 3 and parts[0] != 'bot' else []
        if wrong:
            swap[(run, ident)] = f'{ident}!served={",".join(wrong)}'
    rename = lambda run, ident: swap.get((run, ident), ident)

    # the settings most deals were played under; deals under other settings are listed, not pooled
    per_key = defaultdict(int)
    for r in rows:
        per_key[r[7]] += 1
    main = max(per_key, key=per_key.get) if per_key else None
    runs = defaultdict(lambda: {'deals': 0, 'self_play': 0, 'code_versions': set(), 'started': None, 'game_keys': set()})
    playing = set()
    for run, gk, cv, started, _, finished, idents in matches:
        if not finished:
            playing |= {rename(run, i) for i in json.loads(idents or '[]')}
        x = runs[run]
        x['code_versions'].add(cv)
        x['game_keys'].add(gk)
        if started and (x['started'] is None or started < x['started']):
            x['started'] = started
    deals = []
    for run, mu, k, a, b, net, bb, gk, fp, sd in rows:
        if gk != main:
            continue
        a, b = rename(run, a), rename(run, b)
        if a == b:   # two ids for the same model, level and protocol: no information about how good it is
            runs[run]['self_play'] += 1
            continue
        runs[run]['deals'] += 1
        deals.append({'a': a, 'b': b, 'matchup': mu if run == '.' else f'{run}/{mu}', 'deal': k, 'net_a': net, 'bb': bb,
                      'fp': fp, 'seed': sd, 'order': runs[run]['started'] or 0})

    # names: a label from the current players.toml, else the newest one recorded, else one made from the spec
    newest = {}
    kinds, pids = {}, defaultdict(set)
    order = sorted(people, key=lambda p: runs[p[0]]['started'] or 0)
    from .server import display_name
    for run, pid, ident, lab, kind, spec in order:
        ident = rename(run, ident)
        try:
            newest[ident] = display_name(pid, json.loads(spec))   # today's naming, not the one stored at sync time
        except (TypeError, ValueError):
            newest[ident] = lab
        kinds[ident] = kind
        pids[ident].add(pid)
    current = current_labels(players_file)
    labels = {**newest, **{k: current[_base(k)] for k in newest if _base(k) in current}}
    _disambiguate(labels)

    meta = {i: {'identity': i, 'kind': kinds.get(i) or ('bot' if i.startswith('bot:') else 'model'),
                'quarantine_reason': quarantine_reason(i), 'identity_resolved': identity_resolved(i)}
            for i in {p for d in deals for p in (d['a'], d['b'])}}
    left = [run for run, x in runs.items() if x['game_keys'] and main not in x['game_keys']]
    notes = [f'Left out (other game settings): {", ".join(sorted(left))}.'] if left else []
    if sum(1 for d in deals if d['fp'] is None):
        notes.append('Some deals have no recorded seed (their folders were deleted before seeds were indexed), so '
                     'repeats among them cannot be detected.')
    if any('#' not in i for i, m in meta.items() if m['kind'] == 'model'):
        notes.append('Players from runs before protocols were recorded are kept apart from later runs of the same '
                     'model and level (their settings cannot be compared).')
    if len({re.sub(r'@[0-9a-f]{8}', '', i) for i in meta}) < len(meta):
        notes.append('Some models appear more than once because they were played by different implementations '
                     '(code that plays, or CLI versions); they pool only where an equivalence list says those '
                     'implementations play identically.')
    if eq:
        notes.append('Pooled as equivalent implementations (equivalences.json): '
                     + '; '.join(f'{k} as {v}' for k, v in sorted(eq.items())) + '.')
    if rejected:
        notes.append(f'{len(rejected)} duplicate pair{"s" if len(rejected) != 1 else ""} failed validation (b-first is '
                     f'not a-first with the seats swapped on the same deals) and were left out.')
    if others:
        notes.insert(0, f'Only {track}-track runs are pooled here; the tracks are never pooled ('
                        + ('the adaptation track\'s long conversations are judged apart, model-based and not official'
                           if track == 'official' else 'the official track is judged apart') + f'): {len(others)} '
                        f'other run{"s" if len(others) != 1 else ""} left out.')
    r = rate(deals, boot=boot, seed=seed, meta=meta, pooled=True, notes=notes, playing=playing, rejected=rejected)

    stats, servedp = defaultdict(dict), defaultdict(set)
    for run, ident, counts, sv, gk, seats, pl in tallies:
        idents = json.loads(seats or '[]')
        if gk == main and len(set(idents)) > 1:   # as in the rating: no self-play
            ids = dict(zip(json.loads(pl or '[]'), idents))
            add(stats[rename(run, ident)], _by_identity(json.loads(counts), {p: rename(run, i) for p, i in ids.items()}))
            servedp[rename(run, ident)] |= set(json.loads(sv))
    r['stats'] = {i: summarize(s, servedp[i]) for i, s in stats.items()}

    included = [run for run, x in runs.items() if x['deals']]
    left_out = [{'run': run, 'reason': 'other game settings'} for run, x in runs.items()
                if x['game_keys'] and main not in x['game_keys']]
    left_out += [{'run': run, 'player': ident, 'reason': f'answered by {swap[(run, ident)].split("=", 1)[1]}'}
                 for run, ident in sorted(swap)]
    versions = sorted({v for run in included for v in runs[run]['code_versions'] if v})
    if len(versions) > 1:
        r['notes'].insert(0, f'Pooled across {len(versions)} code versions.')
    unversioned = sorted(run for run in included if None in runs[run]['code_versions'])
    if unversioned:
        r['notes'].insert(0, f'From before code versions were recorded: {", ".join(unversioned)}.')
    from .stats import LABELS, POSITIONS
    r.update({
        'all_time': True, 'track': track, 'other_track_runs': others, 'labels': labels, 'kinds': kinds, 'stat_labels': LABELS, 'position_labels': POSITIONS,
        'equivalences': eq,
        'ids': {i: sorted(p) for i, p in pids.items()},
        'game': json.loads(main) if main else None,
        'runs': sorted(({'run': run, 'deals': x['deals'], 'self_play': x['self_play'], 'started': x['started'],
                         'code_versions': sorted(v for v in x['code_versions'] if v)}
                        for run, x in runs.items() if run in included),
                       key=lambda x: x['started'] or 0),
        'left_out': left_out,
        'code_versions': versions,
        'unversioned_runs': unversioned,
    })
    return r


def _by_identity(counts: dict, ids: dict) -> dict:
    """A tally's per-opponent keys (vs:<player id>:...) renamed to the opponent's identity."""
    out = {}
    for k, v in counts.items():
        if k.startswith('vs:'):
            pid, rest = k[3:].rsplit(':', 1)
            k = f'vs:{ids.get(pid, pid)}:{rest}'
        out[k] = v
    return out


def _disambiguate(labels: dict):
    """Identities that would show the same name get what tells them apart: the backend of a model (one
    model through two CLIs), else its protocol (or the model that answered), the settings of a bot, else the
    whole identity."""
    def backend(ident):
        parts = _base(ident).split(':')
        return ':'.join(parts[2:]) or 'default' if parts[0] == 'bot' else parts[0]

    def protocol(ident):
        if '!served=' in ident:
            return 'answered by ' + ident.split('!served=', 1)[1]
        h = re.search(r'~h(\d+)', ident)
        h = f', {h.group(1)} hands per conversation' if h else ''
        return (f'protocol {ident.split("#", 1)[1][:8]}' if '#' in ident else 'protocol not recorded') + h

    def implementation(ident):
        m = re.search(r'@([0-9a-f]{8})', ident)
        return f'implementation {m.group(1)}' if m else 'implementation not recorded'
    by = defaultdict(list)
    for ident, lab in labels.items():
        by[lab].append(ident)
    for lab, idents in by.items():
        if len(idents) > 1:
            for tell in (backend, protocol, implementation, lambda i: i):
                if len({tell(i) for i in idents}) == len(idents):
                    break
            for ident in idents:
                labels[ident] = f'{lab} ({tell(ident)})'


def table(r) -> str:
    from .ratings import summary
    return summary(r)
