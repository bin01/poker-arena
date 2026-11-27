"""The web viewer: `python -m poker_arena view runs` then open http://127.0.0.1:8800

Serves the static app in poker_arena/viewer/ and read-only JSON about the matches under the root. A run is
one folder of matchups (a tournament's `out`, or a `play --out`); the leaderboard is per run, because
mixing a smoke test with a tournament makes neither readable.
    /api/runs                       every run: players, progress, pace and ETA, live or not
    /api/index?run=RUN              the run's matchups (both duplicate matches, and planned ones not started)
    /api/match?id=RUN/MATCHUP/a-first   one match: config, finished hands, and every event by hand
    /api/ratings?run=RUN            the run's leaderboard and per-player stats (recomputed when a hand finishes)
Running matches are followed live: the app polls while a match is unfinished.

`export(root, out)` writes the same JSON as files next to a copy of the app, so a run can be published as a
static site (`python -m poker_arena export runs/pilot --out site`).
"""
from __future__ import annotations
import json
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


BOT_NAMES = {'random': 'Random bot', 'station': 'Calling station', 'rule': 'Rule bot'}
LABELS: dict = {}   # labels from the current players.toml, which win over the ones recorded in old runs


def pretty_model(model: str) -> str:
    """'claude-opus-5-5' -> 'Claude Opus 5.5', 'gpt-6-luna' -> 'GPT-6 Luna', 'sonnet' -> 'Claude Sonnet'."""
    if model in ('opus', 'sonnet', 'haiku'):
        return 'Claude ' + model.title()
    out = []
    for tok in model.replace('_', '-').split('-'):
        if tok.isdigit() and out and out[-1][-1].isdigit():
            out[-1] += '.' + tok
        elif tok.isdigit() and out and out[-1] == 'GPT':
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
        return pid
    return pretty_model(str(spec['model'])) + (f' · {spec["effort"]}' if spec.get('effort') else '')


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
    return {'id': str(m.relative_to(root)), 'run': run_of(root, m), 'a': a, 'b': b, 'started': True,
            'matches': matches, 'deals_total': len(cfg.get('seeds', [])), 'pairs_done': len(both),
            'pair_net_a': sum(ha[k]['net'][0] + hb[k]['net'][1] for k in both),
            'kinds': {p: 'bot' if 'bot' in s else 'model' for p, s in specs.items()},
            'labels': {p: LABELS.get(p) or display_name(p, s) for p, s in specs.items()},
            'bb': cfg.get('game', {}).get('big_unit') or cfg.get('game', {}).get('bb') or 1,
            'live': any(x['live'] for x in matches), 'finished': len(matches) == 2 and all(x['finished'] for x in matches),
            'updated': max((_mtime(m / s / 'events.jsonl') for s in ('a-first', 'b-first')), default=0),
            'times': [h['t'] for s in hands.values() for h in s if h.get('t')]}


def _runs_and_matchups(root: Path):
    """{run: [matchup]} including matchups a plan.json lists that haven't started."""
    by = {}
    for cfg in sorted(root.rglob('a-first/config.json')):
        m = _matchup(root, cfg.parent.parent)
        by.setdefault(m['run'], []).append(m)
    for plan_path in root.rglob('plan.json'):
        plan = _json(plan_path) or {}
        rel = plan_path.parent.relative_to(root)
        run = str(rel) if str(rel) != '.' else '.'
        have = {(m['a'], m['b']) for m in by.get(run, [])}
        for a, b in plan.get('matchups', []):
            if (a, b) not in have:
                by.setdefault(run, []).append({
                    'id': str(rel / f'{a}__vs__{b}') if run != '.' else f'{a}__vs__{b}', 'run': run, 'a': a, 'b': b,
                    'started': False, 'matches': [], 'deals_total': plan.get('deals', 0), 'pairs_done': 0,
                    'pair_net_a': 0, 'kinds': {}, 'labels': {**plan.get('labels', {}), **LABELS}, 'bb': 1, 'live': False, 'finished': False,
                    'updated': 0, 'times': []})
    return by


def _public(m):
    return {k: v for k, v in m.items() if k != 'times'}


def index(root: Path, run: str | None = None):
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
        out.append({'run': run, 'title': root.name if run == '.' else run.split('/')[-1], 'matchups': len(ms), 'matchups_started': sum(m['started'] for m in ms),
                    'matchups_finished': sum(m['finished'] for m in ms),
                    'players': sorted({m['a'] for m in ms} | {m['b'] for m in ms}),
                    'hands': done, 'hands_planned': planned, 'live': live,
                    'finished': all(m['finished'] for m in ms),
                    'updated': max((m['updated'] for m in ms), default=0),
                    'started_at': times[0] if times else None,
                    'hands_per_hour': round(rate * 3600) if rate else None,
                    'eta_seconds': round((planned - done) / rate) if rate and planned > done else None})
    return sorted(out, key=lambda r: -r['updated'])


def match(root: Path, mid: str):
    d = (root / mid).resolve()
    if root not in d.parents or not (d / 'config.json').exists():
        return None
    events = {}
    for e in _lines(d / 'events.jsonl'):
        events.setdefault(e.get('episode', 0), []).append(e)
    return {'summary': match_summary(root, d), 'config': _json(d / 'config.json'),
            'hands': _lines(d / 'hands.jsonl'), 'events': events, 'result': _json(d / 'result.json')}


def ratings_for(root: Path, run, boot=200):
    from .ratings import ratings
    base = root if run in (None, '', '.') else (root / run).resolve()
    if root != base and root not in base.parents:
        return None
    return ratings(base, boot=boot)


def key(s: str) -> str:
    """A file name for an id or run, the same as the app's `fileKey`: letters, digits, - and _ kept,
    every other byte as ~XX."""
    return ''.join(chr(b) if chr(b).isascii() and (chr(b).isalnum() or chr(b) in '-_') else f'~{b:02X}'
                   for b in s.encode())


def export(root, out, players_file='players.toml') -> int:
    """A static copy of the viewer for the runs under root: the app plus every API answer as a file."""
    root, out = Path(root).resolve(), Path(out)
    LABELS.update(load_labels(players_file))
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
    for r in rs:
        ms = index(root, r['run'])
        for m in ms:
            m['live'] = False
            for x in m['matches']:
                x['live'] = False
                data_ = match(root, x['id'])
                specs = data_['config'].get('specs', {})  # only what the viewer shows: no commands, paths or URLs
                data_['config']['specs'] = {p: {k: v for k, v in s.items() if k in ('label', 'model', 'bot', 'effort')}
                                            for p, s in specs.items()}
                put(f'match-{key(x["id"])}.json', data_)
        put(f'index-{key(r["run"])}.json', ms)
        put(f'ratings-{key(r["run"])}.json', ratings_for(root, r['run'], boot=300))
    return files


def serve(root, port=8800, host='127.0.0.1', players_file='players.toml'):
    root = Path(root).resolve()  # match folders are resolved before use, so the root must be too
    LABELS.update(load_labels(players_file))
    cache = {}

    def ratings_json(run):
        base = root if run in (None, '', '.') else (root / run).resolve()
        stamp = tuple(sorted((str(p), _mtime(p)) for p in base.rglob('*.jsonl')))
        if cache.get(str(base), (None,))[0] != stamp:
            cache[str(base)] = (stamp, ratings_for(root, run))
        return cache[str(base)][1]

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
