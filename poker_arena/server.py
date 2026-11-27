"""The web viewer: `python -m poker_arena view runs` then open http://127.0.0.1:8800

Serves the static app in poker_arena/viewer/ and read-only JSON about the matches under the root. A run is
one folder of matchups (a tournament's `out`, or a `play --out`); the leaderboard is per run, because
mixing a smoke test with a tournament makes neither readable.
    /api/runs                       every run: players, progress, pace and ETA, live or not
    /api/index?run=RUN              the run's matchups (both duplicate matches, and planned ones not started)
    /api/match?id=RUN/MATCHUP/a-first   one match: config, finished hands, and every event by hand
    /api/ratings?run=RUN            the run's leaderboard, per-player stats and its audit summary (recomputed when
                                    any of the run's record files changes)
    /api/quiz?run=RUN               the run's poker-fundamentals quiz (quiz.json), when it has one
The run `*` is every run together: its leaderboard comes from the history database (history.py), one player
per model and reasoning level, and keeps runs whose folders have since been deleted.
Running matches are followed live: the app polls while a match is unfinished.

`export(root, out)` writes the same JSON as files next to a copy of the app, so a run can be published as a
static site (`python -m poker_arena export runs/pilot --out site`).
"""
from __future__ import annotations
import json
import re
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VIEWER = Path(__file__).with_name('viewer')
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
         '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml'}
LIVE_SECONDS = 180    # a match without a result that wrote an event this recently counts as playing
PACE_SECONDS = 1800   # the ETA uses the pace over the last half hour of finished hands


BOT_NAMES = {'random': 'Random bot', 'station': 'Calling station', 'rule': 'Rule bot', 'reference': 'Reference bot'}
FAMILIES = {'claude': 'Claude', 'codex': 'GPT', 'agy': 'Gemini', 'gemini': 'Gemini'}
LABELS: dict = {}   # labels from the current players.toml, which win over the ones recorded in old runs
PLAYERS_FILE = ['players.toml']


def pretty_model(model: str) -> str:
    """'claude-opus-5-5' -> 'Claude Opus 5.5', 'gpt-6-luna' -> 'GPT-6 Luna', 'sonnet' -> 'Claude Sonnet'."""
    if model in ('opus', 'sonnet', 'haiku'):
        return 'Claude ' + model.title()
    out = []
    for tok in model.replace('_', '-').split('-'):
        if tok.isdigit() and out and out[-1][-1].isdigit():
            out[-1] += '.' + tok
        elif out and out[-1] == 'GPT' and tok.replace('.', '').isdigit():
            out[-1] += '-' + tok
        else:
            out.append('GPT' if tok.lower() == 'gpt' else tok if any(c.isdigit() for c in tok) else tok.title())
    return ' '.join(out)


def display_name(pid: str, spec: dict) -> str:
    """The name the viewer shows: the player's `label`, else one made from its model, else its id."""
    if spec.get('label'):
        return spec['label']
    if 'bot' in spec:
        return BOT_NAMES.get(spec['bot'], f'{spec["bot"]} bot')
    if not spec.get('model'):
        family = FAMILIES.get(spec.get('backend'))
        if not family:
            return pid
        name = f'{family} (model not recorded)'
        return name + (f' · {spec["effort"].title()}' if spec.get('effort') else '')
    from .runner import EFFORT_BACKENDS
    effort = spec.get('effort') or ('default' if spec.get('backend') in EFFORT_BACKENDS else '')
    name = pretty_model(str(spec['model']))
    if effort and name.lower().endswith(' ' + effort.lower()):   # gemini-4-neon-high already says its level
        name = name[: -len(effort) - 1]
    return name + (f' · {effort.title() if effort != "default" else "default effort"}' if effort else '')


def load_labels(players_file='players.toml') -> dict:
    """Explicit labels from a players file, if there is one here."""
    try:
        from .runner import load_players
        return {pid: s['label'] for pid, s in load_players(players_file).items() if s.get('label')}
    except Exception:
        return {}


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _lines(p: Path):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _mtime(p: Path):
    try:
        return p.stat().st_mtime
    except OSError:
        return 0


def match_summary(root: Path, d: Path, hands=None):
    cfg = _json(d / 'config.json') or {}
    res = _json(d / 'result.json')
    hands = _lines(d / 'hands.jsonl') if hands is None else hands
    totals = [0] * len(cfg.get('players', []))
    for h in hands:
        totals = [t + x for t, x in zip(totals, h.get('net', []))]
    return {'id': str(d.relative_to(root)), 'label': d.name, 'players': cfg.get('players', []),
            'names': cfg.get('names', []), 'hands_total': len(cfg.get('seeds', [])), 'hands_done': len(hands),
            'net': totals, 'finished': res is not None,
            'live': res is None and time.time() - _mtime(d / 'events.jsonl') < LIVE_SECONDS}


def run_of(root: Path, m: Path) -> str:
    rel = m.parent.relative_to(root)
    return str(rel) if str(rel) != '.' else '.'


def _matchup(root: Path, m: Path):
    """One duplicate pair: both matches, how many deals both have finished, and a's net over those deals."""
    cfg = _json(m / 'a-first' / 'config.json') or {}
    hands = {s: _lines(m / s / 'hands.jsonl') for s in ('a-first', 'b-first')}
    matches = [match_summary(root, m / s, hands[s]) for s in ('a-first', 'b-first') if (m / s / 'config.json').exists()]
    a, b = cfg.get('players', ['?', '?'])[:2]
    ha = {h['hand']: h for h in hands['a-first']}
    hb = {h['hand']: h for h in hands['b-first']}
    both = sorted(set(ha) & set(hb))
    specs = cfg.get('specs', {})
    rep = re.search(r'__x(\d+)', m.name.split('__vs__')[-1])
    rnd = m.name.rsplit('__r', 1)[1] if '__r' in m.name.split('__vs__')[-1] else ''
    return {'id': str(m.relative_to(root)), 'run': run_of(root, m), 'a': a, 'b': b, 'started': True,
            'round': int(rnd) if rnd.isdigit() else 1, 'replicate': int(rep.group(1)) if rep else 1,
            'matches': matches, 'deals_total': len(cfg.get('seeds', [])), 'pairs_done': len(both),
            'pair_net_a': sum(ha[k]['net'][0] + hb[k]['net'][1] for k in both),
            'kinds': {p: 'bot' if 'bot' in s else 'model' for p, s in specs.items()},
            'labels': {p: LABELS.get(p) or display_name(p, s) for p, s in specs.items()},
            'bb': cfg.get('game', {}).get('big_unit') or cfg.get('game', {}).get('bb') or 1,
            'live': any(x['live'] for x in matches), 'finished': len(matches) == 2 and all(x['finished'] for x in matches),
            'updated': max((_mtime(m / s / 'events.jsonl') for s in ('a-first', 'b-first')), default=0),
            'times': [h['t'] for s in hands.values() for h in s if h.get('t')]}


def _plan_specs(plan_path: Path, plan: dict) -> dict:
    """The specs of a plan's players from its players file, if that file can still be found."""
    from .runner import load_players
    name = plan.get('players_file') or PLAYERS_FILE[0]
    for f in (Path(name), plan_path.parent / name, Path(PLAYERS_FILE[0])):
        try:
            return load_players(f, plan.get('effort'))
        except Exception:
            continue
    return {}


def _who(pid, specs: list) -> tuple:
    """(name, kind) for a player that hasn't played yet: from the first spec source that knows it."""
    for src in specs:
        if pid in src:
            s = src[pid]
            return LABELS.get(pid) or display_name(pid, s), 'bot' if 'bot' in s else 'model'
    return LABELS.get(pid) or pid, None


def _runs_and_matchups(root: Path):
    """{run: [matchup]} including matchups a plan.json lists that haven't started."""
    by, seen = {}, {}
    for cfg in sorted(root.rglob('a-first/config.json')):
        m = _matchup(root, cfg.parent.parent)
        by.setdefault(m['run'], []).append(m)
        seen.setdefault(m['run'], {}).update((_json(cfg) or {}).get('specs', {}))
    for plan_path in root.rglob('plan.json'):
        plan = _json(plan_path) or {}
        rel = plan_path.parent.relative_to(root)
        run = str(rel) if str(rel) != '.' else '.'
        have = {(m['a'], m['b']) for m in by.get(run, [])}
        missing = [(a, b) for a, b in plan.get('matchups', []) if (a, b) not in have]
        if not missing:
            continue
        from .runner import EFFORT_BACKENDS
        here = [{p: {'label': lab, **({'bot': True} if m['kinds'].get(p) == 'bot' else {})} for p, lab in m['labels'].items()}
                for m in by.get(run, [])]
        effort = plan.get('effort')
        elsewhere = [{p: {**s, 'effort': effort or s.get('effort')} if effort and s.get('backend') in EFFORT_BACKENDS else s
                      for p, s in specs.items()} for r, specs in seen.items() if r != run]
        sources = here + [_plan_specs(plan_path, plan)] + elsewhere
        for a, b in missing:
            (la, ka), (lb, kb) = _who(a, sources), _who(b, sources)
            by.setdefault(run, []).append({
                'id': str(rel / f'{a}__vs__{b}') if run != '.' else f'{a}__vs__{b}', 'run': run, 'a': a, 'b': b,
                'started': False, 'round': 1, 'matches': [], 'deals_total': plan.get('deals', 0), 'pairs_done': 0,
                'pair_net_a': 0, 'kinds': {p: k for p, k in ((a, ka), (b, kb)) if k}, 'labels': {a: la, b: lb},
                'bb': 1, 'live': False, 'finished': False, 'updated': 0, 'times': []})
    return by


def roster(root: Path, run: str) -> list:
    """Every player of a run, from its plan when there is one, with where it stands: awaiting its first
    matchup, playing, or done."""
    ms = _runs_and_matchups(root).get(run, [])
    plan = _json((root if run == '.' else root / run) / 'plan.json') or {}
    names = list(plan.get('players') or [])
    names += sorted(({m['a'] for m in ms} | {m['b'] for m in ms}) - set(names))
    out = []
    for p in names:
        mine = [m for m in ms if p in (m['a'], m['b'])]
        status = ('done' if mine and all(m['finished'] for m in mine)
                  else 'playing' if any(m['started'] for m in mine) else 'awaiting')
        out.append({'player': p, 'status': status})
    return out


TERMINAL = ('a', 'b', 'even', 'exhausted', 'ineligible')


def _design(plan: dict):
    try:
        from .ratings import normalize_design
    except ImportError:
        from .runner import plan_design as normalize_design
    try:
        return normalize_design(plan)
    except Exception:
        return None


def lifecycle(root: Path, run: str, ms: list) -> dict:
    """Where a run stands: {status: 'running'|'finished'|'stopped', round, pairings: {"a|b": {status, verdict,
    looks_used}}}. From the runner's tournament.json when there is one; else from plan.json and decisions.json
    (a pairing is over on a terminal verdict or once its planned rounds are played); else from the folders."""
    base = root if run == '.' else root / run
    t = _json(base / 'tournament.json')
    if isinstance(t, dict) and t.get('status'):
        return {'source': 'tournament', 'round': t.get('round'), 'pairings': t.get('pairings') or {},
                'status': t['status'], 'updated': t.get('updated')}
    plan = _json(base / 'plan.json')
    if not plan or not plan.get('matchups'):
        done = bool(ms) and all(m['finished'] for m in ms)
        return {'source': 'folders', 'status': 'finished' if done else 'running', 'round': None, 'pairings': {}}
    design = _design(plan) or {}
    rounds = design.get('max_rounds') or 1
    decisions = _json(base / 'decisions.json') or {}
    pairings = {}
    for a, b in plan['matchups']:
        mine = [m for m in ms if {m['a'], m['b']} == {a, b} and m['started']]
        looks = decisions.get(f'{a}__vs__{b}') or []
        last = looks[-1] if looks else {}
        verdict = last.get('verdict')
        played = max((m['round'] for m in mine if m['finished']), default=0)
        if verdict in TERMINAL:
            status = verdict if verdict in ('exhausted', 'ineligible') else 'decided'
        elif played >= rounds and all(m['finished'] for m in mine):
            status = 'exhausted' if design else 'done'
        elif any(m['live'] for m in mine):
            status = 'playing'
        else:
            status = 'waiting'
        pairings[f'{a}|{b}'] = {'status': status, 'verdict': verdict, 'looks_used': last.get('looks_used', 0),
                                'rounds_played': played, 'rounds_planned': rounds}
    over = all(p['status'] in ('decided', 'exhausted', 'ineligible', 'done') for p in pairings.values())
    return {'source': 'plan', 'status': 'finished' if over else 'running',
            'round': max((m['round'] for m in ms if m['started']), default=None), 'pairings': pairings}


def _public(m):
    return {k: v for k, v in m.items() if k != 'times'}


def index(root: Path, run: str | None = None):
    from .history import ALL
    run = None if run == ALL else run
    by = _runs_and_matchups(root)
    out = [m for r, ms in by.items() if run is None or r == run for m in ms]
    # playing first, then started, then planned; alphabetical within each
    return [_public(m) for m in sorted(out, key=lambda m: (not m['live'], not m['started'], m['id']))]


def runs(root: Path):
    """Every run with its progress, most recently updated first."""
    now = time.time()
    out = []
    for run, ms in _runs_and_matchups(root).items():
        planned = sum(2 * m['deals_total'] for m in ms)
        done = sum(x['hands_done'] for m in ms for x in m['matches'])
        times = sorted(t for m in ms for t in m['times'])
        live = any(m['live'] for m in ms)
        recent = [t for t in times if t > now - PACE_SECONDS]
        rate = None  # hands per second
        if live and len(recent) >= 4:
            rate = (len(recent) - 1) / max(1.0, recent[-1] - recent[0])
        life = lifecycle(root, run, ms)
        has_quiz = ((root if run == '.' else root / run) / 'quiz.json').exists()
        out.append({'run': run, 'quiz': has_quiz, 'title': root.name if run == '.' else run.split('/')[-1], 'matchups': len(ms), 'matchups_started': sum(m['started'] for m in ms),
                    'matchups_finished': sum(m['finished'] for m in ms),
                    'players': sorted({m['a'] for m in ms} | {m['b'] for m in ms}),
                    'hands': done, 'hands_planned': planned, 'live': live,
                    'finished': life['status'] == 'finished', 'status': life['status'], 'round': life['round'],
                    'pairings': life['pairings'],
                    'between_rounds': life['status'] == 'running' and not live and bool(ms) and all(m['finished'] for m in ms if m['started']),
                    'updated': max((m['updated'] for m in ms), default=0),
                    'started_at': times[0] if times else None,
                    'hands_per_hour': round(rate * 3600) if rate else None,
                    'eta_seconds': round((planned - done) / rate) if rate and planned > done else None})
    out.sort(key=lambda r: -r['updated'])
    if out and (len(out) > 1 or _history_runs(root) - {r['run'] for r in out}):
        out.insert(0, _all_runs(out))
    return out


def _history_runs(root: Path) -> set:
    """Runs the history database knows, including ones whose folders are gone."""
    from .history import DB_NAME
    import sqlite3
    db = root / DB_NAME
    if not db.exists():
        return set()
    try:
        con = sqlite3.connect(str(db), timeout=5)
        try:
            return {r[0] for r in con.execute('SELECT DISTINCT run FROM matches')}
        finally:
            con.close()
    except sqlite3.Error:
        return set()


def _all_runs(rs):
    """The `*` entry: every run on disk added up (its leaderboard also counts runs only the history has)."""
    from .history import ALL
    return {'run': ALL, 'title': 'All runs', 'all': True, 'runs': len(rs),
            **{k: sum(r[k] for r in rs) for k in ('matchups', 'matchups_started', 'matchups_finished', 'hands', 'hands_planned')},
            'players': sorted({p for r in rs for p in r['players']}),
            'live': any(r['live'] for r in rs), 'finished': all(r['finished'] for r in rs),
            'status': 'finished' if all(r['finished'] for r in rs) else 'running',
            'updated': max(r['updated'] for r in rs),
            'started_at': min((r['started_at'] for r in rs if r['started_at']), default=None),
            'hands_per_hour': None, 'eta_seconds': None}


def match(root: Path, mid: str):
    d = (root / mid).resolve()
    if root not in d.parents or not (d / 'config.json').exists():
        return None
    events = {}
    for e in _lines(d / 'events.jsonl'):
        events.setdefault(e.get('episode', 0), []).append(e)
    return {'summary': match_summary(root, d), 'config': _json(d / 'config.json'),
            'hands': _lines(d / 'hands.jsonl'), 'events': events, 'result': _json(d / 'result.json')}


AUDIT_KEYS = ('decisions', 'calls', 'tool_rejected', 'unverified', 'verified', 'opaque', 'audit_missing', 'fallbacks',
              'retries', 'resized', 'failed_tries', 'rate_limited', 'backend_failures', 'rebuilt')


def _same_model(want, seen) -> bool:
    try:
        from .ratings import same_model
    except ImportError:
        def same_model(a, b):
            norm = lambda x: re.sub(r'[^a-z0-9]+', '-', str(x).lower()).strip('-')
            return norm(a) == norm(b)
    return bool(same_model(want, seen))


def _identity(call: dict, model) -> str:
    """'match' when the backend reported the model it ran and it is the one asked for, 'mismatch' when it
    reported another, else 'unattested'. The runner's own `identity_match` wins; its older `identity` field only
    says whether some model was reported, never that it matched, so it is not trusted for a match."""
    from .ratings import identity_match
    if call.get('identity_match') in ('mismatch', 'unattested'):
        return call['identity_match']   # a recorded doubt stands; a recorded match is re-checked below
    seen = [m for m in (call.get('observed_models') or [call.get('observed_model') or call.get('served_model')]) if m]
    want = call.get('requested_model') or model
    if not want:
        return 'unattested' if not seen else 'mismatch'
    return identity_match(want, seen)   # recomputed from what was reported, never taken from the record


def audit(root: Path, run: str) -> dict:
    """What the records say about how each model's decisions were made, from calls.jsonl and the decisions in
    events.jsonl: tool-use rejections, turns that could not be audited, fallbacks, retries, failed tries and
    rate-limit waits, and how many calls report a model that matches the one asked for. Bots make no calls and are left out."""
    from .history import ALL
    folders = [cfg.parent for cfg in sorted(root.rglob('config.json')) if cfg.parent.name in ('a-first', 'b-first')
               and (run in (None, ALL) or run_of(root, cfg.parent.parent) == run)]
    players: dict = {}
    for d in folders:
        cfg = _json(d / 'config.json') or {}
        specs = cfg.get('specs', {})
        models = {pid: s for pid, s in specs.items() if 'bot' not in s}
        if not models:
            continue
        for pid in models:
            players.setdefault(pid, {**{k: 0 for k in AUDIT_KEYS}, 'identity': {}, 'observed': set(), 'requested': set()})
        for e in _lines(d / 'events.jsonl'):
            x = e.get('data') or {}
            if e.get('kind') != 'thought' or x.get('player') not in models:
                continue
            a = players[x['player']]
            a['decisions'] += 1
            kind = (x.get('correction') or {}).get('kind')
            if x.get('fallback') or kind == 'fallback':
                a['fallbacks'] += 1
            elif kind in ('retry', 'tool_use', 'unverified'):
                a['retries'] += 1
            elif kind == 'sizing':
                a['resized'] += 1
        for c in _lines(d / 'calls.jsonl'):
            pid = c.get('player')
            if pid not in models:
                continue
            a = players[pid]
            a['calls'] += 1
            fails = c.get('failed_attempts') or []
            a['failed_tries'] += len(fails)
            a['rate_limited'] += sum(1 for f in fails if f.get('rate_limited'))
            a['rebuilt'] += 1 if c.get('rebuilt_from_turns') else 0
            if c.get('backend_failed'):
                a['backend_failures'] += 1
                continue
            if c.get('rejected') == 'tool_use':
                a['tool_rejected'] += 1
            elif c.get('rejected') == 'unverified' or c.get('audit') == 'unverified':
                a['unverified'] += 1
            elif c.get('audit') in ('verified', 'opaque'):
                a[c['audit']] += 1
            else:
                a['audit_missing'] += 1
            model = models[pid].get('model')
            ident = _identity(c, model)
            a['identity'][ident] = a['identity'].get(ident, 0) + 1
            for seen in c.get('observed_models') or [c.get('observed_model') or c.get('served_model')]:
                if seen:
                    a['observed'].add(str(seen))
            if c.get('requested_model') or model:
                a['requested'].add(str(c.get('requested_model') or model))
    out = {pid: {**a, 'observed': sorted(a['observed']), 'requested': sorted(a['requested'])} for pid, a in players.items()}
    totals = {k: sum(a[k] for a in out.values()) for k in AUDIT_KEYS}
    totals['identity'] = {}
    for a in out.values():
        for k, n in a['identity'].items():
            totals['identity'][k] = totals['identity'].get(k, 0) + n
    return {'players': out, 'totals': totals, 'recorded': any(a['verified'] or a['opaque'] or a['unverified'] or a['tool_rejected']
                                                              for a in out.values())}


def quiz(root: Path, run):
    """The run's quiz.json (poker-fundamentals spots answered by each player), or None."""
    from .history import ALL
    if run in (None, '', ALL):
        return None
    base = root if run == '.' else (root / run).resolve()
    if base != root and root not in base.parents:
        return None
    data = _json(base / 'quiz.json')
    if not isinstance(data, dict):
        return None
    labels, kinds = {}, {}
    for m in _runs_and_matchups(root).get(run, []):
        labels.update(m['labels'])
        kinds.update(m['kinds'])
    try:   # quiz players named by today's players file when the run's matches don't name them
        from .runner import load_players
        known = load_players(PLAYERS_FILE[0])
        for pid in (data.get('players') or {}):
            if pid not in labels and pid in known:
                labels[pid] = display_name(pid, known[pid])
                kinds[pid] = 'bot' if 'bot' in known[pid] else 'model'
    except Exception:
        pass
    return {**data, 'labels': {**labels, **(data.get('labels') or {})}, 'kinds': {**kinds, **(data.get('kinds') or {})}}


def ratings_for(root: Path, run, boot=200):
    from .ratings import ratings
    from . import history
    if run == history.ALL:
        history.sync(root)
        r = history.all_time(root / history.DB_NAME, boot=boot, players_file=PLAYERS_FILE[0])
        r['run_audit'] = audit(root, history.ALL)
        return r
    base = root if run in (None, '', '.') else (root / run).resolve()
    if root != base and root not in base.parents:
        return None
    r = ratings(base, boot=boot)
    key = run if run not in (None, '') else '.'
    r.setdefault('roster', roster(root, key))
    labels, kinds = {}, {}
    for m in _runs_and_matchups(root).get(key, []):
        labels.update(m['labels'])
        kinds.update(m['kinds'])
    r['labels'] = {**labels, **(r.get('labels') or {})}
    r['kinds'] = {**kinds, **(r.get('kinds') or {})}
    r['run_audit'] = audit(root, key)
    return r


def key(s: str) -> str:
    """A file name for an id or run, the same as the app's `fileKey`: letters, digits, - and _ kept,
    every other byte as ~XX."""
    return ''.join(chr(b) if chr(b).isascii() and (chr(b).isalnum() or chr(b) in '-_') else f'~{b:02X}'
                   for b in s.encode())


def export(root, out, players_file='players.toml') -> int:
    """A static copy of the viewer for the runs under root: the app plus every API answer as a file."""
    root, out = Path(root).resolve(), Path(out)
    LABELS.update(load_labels(players_file))
    PLAYERS_FILE[0] = players_file
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for f in VIEWER.iterdir():
        if f.suffix in TYPES and f.name != 'index.html':
            shutil.copy(f, out / f.name)
    html = (VIEWER / 'index.html').read_text().replace(
        '<!--static-->', f'<script>window.POKER_STATIC = {{exported: {int(time.time())}}};</script>')
    (out / 'index.html').write_text(html)
    files = 0

    def put(name, obj):
        nonlocal files
        (data / name).write_text(json.dumps(obj, separators=(',', ':')))
        files += 1

    rs = runs(root)
    for r in rs:
        r['live'] = False
        r['eta_seconds'] = None
    put('runs.json', rs)
    done = set()
    for r in rs:
        ms = index(root, r['run'])
        for m in ms:
            m['live'] = False
            for x in m['matches']:
                x['live'] = False
                if x['id'] in done:   # the same match again, in the all-runs list
                    continue
                done.add(x['id'])
                data_ = match(root, x['id'])
                specs = data_['config'].get('specs', {})  # only what the viewer shows: no commands, paths or URLs
                data_['config']['specs'] = {p: {k: v for k, v in s.items() if k in ('label', 'model', 'bot', 'effort')}
                                            for p, s in specs.items()}
                put(f'match-{key(x["id"])}.json', data_)
        put(f'index-{key(r["run"])}.json', ms)
        put(f'ratings-{key(r["run"])}.json', ratings_for(root, r['run'], boot=300))
        q = quiz(root, r['run'])
        if q is not None:
            put(f'quiz-{key(r["run"])}.json', q)
    return files


def signature(base: Path, players_file='players.toml') -> tuple:
    """What a run's ratings depend on: every record file under it (hands, events and calls, and the plan,
    decisions, tournament, config and quiz files), and the players file that names them. For all runs the
    history is synced from these same files, so they are its inputs too."""
    files = [p for p in base.rglob('*.json*') if p.suffix in ('.json', '.jsonl')]
    files.append(Path(players_file))
    def stat(p):
        try:
            st = p.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return 0, 0
    return tuple(sorted((str(p), *stat(p)) for p in files))


def serve(root, port=8800, host='127.0.0.1', players_file='players.toml'):
    import threading
    from .history import ALL
    root = Path(root).resolve()  # match folders are resolved before use, so the root must be too
    LABELS.update(load_labels(players_file))
    PLAYERS_FILE[0] = players_file
    cache = {}
    lock = threading.Lock()   # one recompute at a time (and one writer to the history database)

    def ratings_json(run):
        base = root if run in (None, '', '.', ALL) else (root / run).resolve()
        if base != root and root not in base.parents:
            return None
        stamp = signature(base, players_file)
        with lock:
            if cache.get(run, (None,))[0] != stamp:
                cache[run] = (stamp, ratings_for(root, run))
            return cache[run][1]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                run = q.get('run', [None])[0]
                if url.path == '/api/runs':
                    return self.send(200, json.dumps(runs(root)))
                if url.path == '/api/index':
                    return self.send(200, json.dumps(index(root, run)))
                if url.path == '/api/match':
                    data = match(root, q.get('id', [''])[0])
                    return self.send(200, json.dumps(data)) if data else self.send(404, '{"error": "no such match"}')
                if url.path == '/api/quiz':
                    data = quiz(root, run)
                    return self.send(200, json.dumps(data)) if data is not None else self.send(404, '{"error": "no quiz in this run"}')
                if url.path == '/api/ratings':
                    data = ratings_json(run)
                    return self.send(200, json.dumps(data)) if data is not None else self.send(404, '{"error": "no such run"}')
            except Exception as e:  # keep serving; the app shows the error where it happened
                import traceback
                traceback.print_exc()
                return self.send(500, json.dumps({'error': f'{type(e).__name__}: {e}'}))
            name = 'index.html' if url.path in ('/', '') else url.path.lstrip('/')
            f = (VIEWER / name).resolve()
            if f.parent == VIEWER.resolve() and f.suffix in TYPES and f.exists():
                return self.send(200, f.read_bytes(), TYPES[f.suffix])
            self.send(404, 'not found', 'text/plain')

    server = ThreadingHTTPServer((host, port), Handler)
    print(f'Viewing {root} at http://{host}:{port}')
    server.serve_forever()
