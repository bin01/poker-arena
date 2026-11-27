"""The web viewer: `python -m poker_arena view runs` then open http://127.0.0.1:8800

Serves the static app in poker_arena/viewer/ and read-only JSON about the matches under the root:
    /api/index                     every matchup (both duplicate matches), with progress and results
    /api/match?id=MATCHUP/a-first  one match: config, finished hands, and every event by hand
    /api/ratings                   the leaderboard (recomputed when a hand finishes anywhere)
Running matches are followed live: the app polls while a match is unfinished.
"""
from __future__ import annotations
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VIEWER = Path(__file__).with_name('viewer')
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
         '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml'}
LIVE_SECONDS = 600


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


def match_summary(root: Path, d: Path):
    cfg = _json(d / 'config.json') or {}
    res = _json(d / 'result.json')
    hands = _lines(d / 'hands.jsonl')
    ev = d / 'events.jsonl'
    totals = [0] * len(cfg.get('players', []))
    for h in hands:
        totals = [t + x for t, x in zip(totals, h.get('net', []))]
    return {'id': str(d.relative_to(root)), 'label': d.name, 'players': cfg.get('players', []),
            'names': cfg.get('names', []), 'hands_total': len(cfg.get('seeds', [])), 'hands_done': len(hands),
            'net': totals, 'finished': res is not None,
            'live': res is None and ev.exists() and time.time() - ev.stat().st_mtime < LIVE_SECONDS}


def index(root: Path):
    out = []
    for cfg in sorted(root.rglob('a-first/config.json')):
        m = cfg.parent.parent
        matches = [match_summary(root, m / s) for s in ('a-first', 'b-first') if (m / s / 'config.json').exists()]
        a, b = matches[0]['players'][:2]
        done = min(x['hands_done'] for x in matches) if len(matches) == 2 else 0
        net_a = 0
        if len(matches) == 2:
            ha = {h['hand']: h for h in _lines(m / 'a-first' / 'hands.jsonl')}
            hb = {h['hand']: h for h in _lines(m / 'b-first' / 'hands.jsonl')}
            net_a = sum(ha[k]['net'][0] + hb[k]['net'][1] for k in set(ha) & set(hb))
        specs = (_json(cfg) or {}).get('specs', {})
        kinds = {p: 'bot' if 'bot' in s else 'model' for p, s in specs.items()}
        out.append({'id': str(m.relative_to(root)), 'a': a, 'b': b, 'matches': matches, 'pairs_done': done, 'kinds': kinds,
                    'pair_net_a': net_a, 'bb': (_json(cfg) or {}).get('game', {}).get('big_unit', 1),
                    'live': any(x['live'] for x in matches), 'finished': all(x['finished'] for x in matches)})
    return out


def match(root: Path, mid: str):
    d = (root / mid).resolve()
    if root.resolve() not in d.parents or not (d / 'config.json').exists():
        return None
    events = {}
    for e in _lines(d / 'events.jsonl'):
        events.setdefault(e.get('episode', 0), []).append(e)
    return {'summary': match_summary(root, d), 'config': _json(d / 'config.json'),
            'hands': _lines(d / 'hands.jsonl'), 'events': events, 'result': _json(d / 'result.json')}


def serve(root, port=8800, host='127.0.0.1'):
    root = Path(root)
    cache = {'key': None, 'value': None}

    def ratings_json():
        from .ratings import ratings
        key = tuple(sorted((str(p), p.stat().st_mtime) for p in root.rglob('hands.jsonl')))
        if key != cache['key']:
            cache['value'], cache['key'] = ratings(root, boot=200), key
        return cache['value']

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
                if url.path == '/api/index':
                    return self.send(200, json.dumps(index(root)))
                if url.path == '/api/match':
                    data = match(root, q.get('id', [''])[0])
                    return self.send(200, json.dumps(data)) if data else self.send(404, '{}')
                if url.path == '/api/ratings':
                    return self.send(200, json.dumps(ratings_json()))
            except Exception as e:  # keep serving; show the error in the app
                return self.send(500, json.dumps({'error': repr(e)}))
            name = 'index.html' if url.path in ('/', '') else url.path.lstrip('/')
            f = (VIEWER / name).resolve()
            if f.parent == VIEWER.resolve() and f.suffix in TYPES and f.exists():
                return self.send(200, f.read_bytes(), TYPES[f.suffix])
            self.send(404, 'not found', 'text/plain')

    server = ThreadingHTTPServer((host, port), Handler)
    print(f'Viewing {root} at http://{host}:{port}')
    server.serve_forever()
