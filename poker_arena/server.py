"""The web viewer: `python -m poker_arena view runs` then open http://127.0.0.1:8800

Serves the static app in poker_arena/viewer/ and read-only JSON about the matches under the root. A run is
one folder of matchups (a tournament's `out`, or a `play --out`); the leaderboard is per run, because
mixing a smoke test with a tournament makes neither readable.
    /api/runs                       every run: players, progress, pace and ETA, live or not
    /api/index?run=RUN              the run's matchups (both duplicate matches, and planned ones not started)
    /api/match?id=RUN/MATCHUP/a-first   one match: config, finished hands, every event by hand, and which of its
                                    deals are verified, provisional or rejected (the shared conversation reader)
    /api/pairing?p=RUN|A|B          one pairing's conversations in playing order, each deal's result and status, and
                                    totals over verified deals only: the replay's deal sequence and scoreboard
    /api/ratings?run=RUN            the run's leaderboard, per-player stats and its audit summary. Served from a
                                    cache at once (its `cache` key says whether it is current); when a record file
                                    changed it is recomputed in the background. 202 {pending} the first time only.
    /api/quiz?run=RUN               the run's poker-fundamentals quiz (quiz.json), when it has one
    /api/scorecard?run=RUN          the six-axis character sheet of every player (scorecard.py), cached like the ratings
    /api/forensics?run=RUN          the four-channel adversarial audit and the interface-vs-strategy EV split
    /api/hand_eval?id=MATCH&hand=H  one hand replayed through the solver, with its AIVAT luck adjustment
    /api/plan_preview?players=N&... the pre-flight planner: deals, hands, calls, time and power of a design
The run `*` is every run together: its leaderboard comes from the history database (history.py), one player
per model and reasoning level, and keeps runs whose folders have since been deleted.
Running matches are followed live: the app polls while a match is unfinished. Conversations are summarized once
and kept in memory until one of their record files changes (`Inventory`), so a page never re-reads the archive.

`export(root, out)` writes the same JSON as files next to a copy of the app, so a run can be published as a
static site (`python -m poker_arena export runs/pilot --out site`).
"""
from __future__ import annotations
import json
import os
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


BOT_NAMES = {
    'random': 'Random bot', 'station': 'Calling station', 'rule': 'Rule bot', 'reference': 'Reference bot',
    'gto': 'GTO DCFR bot', 'noisy_gto': 'Noisy GTO bot', 'lbr': 'Local Best Response', 'exploiter': 'Adaptive Exploiter',
    'slumbot': 'Slumbot', 'rps_rock': 'Rock (cycle)', 'rps_paper': 'Paper (cycle)', 'rps_scissors': 'Scissors (cycle)',
    'canary_injector': 'Canary Injector', 'redteam_injector': 'Red-Team Injector',
    'redteam_fuzzer': 'Red-Team Fuzzer', 'redteam_schemer': 'Red-Team Schemer', 'redteam_tool': 'Red-Team Tooler',
    'fatigue': 'Fatigue bot', 'amnesiac': 'Amnesiac bot', 'notetaker': 'Note-Taker bot',
    'random_spewer': 'Random Spewer', 'mixed_creative': 'Mixed Creative bot',
}
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


def _error_text(err: str) -> str:
    """The readable part of a backend error: the CLI's own final message when its JSON output carries one
    (`"result": "API Error: ..."`), else the tail of whatever it printed."""
    err = str(err or '')
    found = re.findall(r'"result"\s*:\s*("(?:[^"\\]|\\.)*")', err)
    for raw in reversed(found):
        try:
            text = json.loads(raw)
        except ValueError:
            continue
        if isinstance(text, str) and text.strip():
            return text.strip()
    return err.strip()[-600:]


def play_failure(d: Path) -> dict | None:
    """Why a play stopped before its last hand, when a model's backend gave up on a decision: {'hand', 'seat',
    'player', 'decision', 'tries', 'at', 'message', 'detail'}; None for a finished play, one still being played,
    or one that simply has not got further yet. The runner writes a `backend_failed` call as the last line of
    calls.jsonl when every try of a decision failed, then abandons the play; a later resume appends past it."""
    if (d / 'result.json').exists():
        return None
    calls = _lines(d / 'calls.jsonl')
    last = calls[-1] if calls else None
    if not isinstance(last, dict) or not last.get('backend_failed') or not isinstance(last.get('hand'), int):
        return None
    tries = last.get('failed_attempts') if isinstance(last.get('failed_attempts'), list) else []
    text = _error_text(last.get('error'))
    stamps = [x.get('t') for x in tries if isinstance(x, dict) and isinstance(x.get('t'), (int, float))]
    return {'hand': last['hand'], 'seat': last.get('seat'), 'player': last.get('player'),
            'decision': last.get('decision_id'), 'tries': len(tries) or 1, 'at': max(stamps, default=None),
            'message': re.split(r'\n\s*\n', text)[0][:400], 'detail': text[:2000]}


def run_of(root: Path, m: Path) -> str:
    rel = m.parent.relative_to(root)
    return str(rel) if str(rel) != '.' else '.'


SIDES = ('a-first', 'b-first')
RECORDS = ('config.json', 'hands.jsonl', 'result.json', 'state.json', 'events.jsonl', 'calls.jsonl', 'raw.jsonl')


def _stat(path: str):
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _stamp(m: Path) -> tuple:
    """What a conversation's summary depends on: its two plays' config, hands and result files."""
    base = str(m)
    return tuple(_stat(os.path.join(base, s, f)) for s in SIDES for f in RECORDS)


def _walk(base: Path) -> tuple[list[Path], list[Path]]:
    """(conversation folders, folders holding a plan.json) under base, without descending into a conversation."""
    convs, plans, stack = [], [], [str(base)]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        names = {e.name for e in entries}
        if 'a-first' in names:
            if os.path.exists(os.path.join(d, 'a-first', 'config.json')):
                convs.append(Path(d))
            continue
        if 'plan.json' in names:
            plans.append(Path(d))
        stack += [e.path for e in entries if not e.name.startswith('.') and e.is_dir(follow_symlinks=False)]
    return sorted(convs), sorted(plans)


class Inventory:
    """Every conversation under a root, summarized once and kept until one of its record files changes, so
    listing runs and matchups never re-reads the archive: a visit stats the files and parses only what changed.
    Whether a conversation is playing comes from its event log's age, checked on every read."""

    def __init__(self):
        import threading
        self.mem: dict = {}
        self.lock = threading.Lock()

    def conversation(self, root: Path, m: Path) -> dict:
        stamp = _stamp(m)
        hit = self.mem.get(str(m))
        if hit is None or hit[0] != stamp:
            hit = (stamp, self._read(root, m))
            with self.lock:
                self.mem[str(m)] = hit
        x = hit[1]
        now = time.time()
        ages = {s: _mtime(m / s / 'events.jsonl') for s in SIDES}
        matches = [{**y, 'live': not y['finished'] and not x['failed'] and now - ages[y['side']] < LIVE_SECONDS}
                   for y in x['matches']]
        return {**x, 'matches': matches, 'live': any(y['live'] for y in matches),
                'updated': max(ages.values(), default=0)}

    @staticmethod
    def _read(root: Path, m: Path) -> dict:
        cfg = _json(m / 'a-first' / 'config.json') or {}
        hands = {s: _lines(m / s / 'hands.jsonl') for s in SIDES}
        matches = []
        for s in SIDES:
            if (m / s / 'config.json').exists():
                y = match_summary(root, m / s, hands[s])
                matches.append({**y, 'side': s})
        a, b = (cfg.get('players') or ['?', '?'])[:2]
        ha = {h['hand']: h for h in hands['a-first'] if isinstance(h.get('hand'), int)}
        hb = {h['hand']: h for h in hands['b-first'] if isinstance(h.get('hand'), int)}
        both = sorted(set(ha) & set(hb))
        specs = cfg.get('specs', {})
        tail = m.name.split('__vs__')[-1]
        rep = re.search(r'__x(\d+)', tail)
        rnd = tail.rsplit('__r', 1)[1] if '__r' in tail else ''
        game = cfg.get('game', {})
        stage = ['preflop', 'flop', 'flop', 'flop', 'turn', 'river']
        reach = lambda h: 'showdown' if h.get('showdown') else stage[min(len(h.get('board') or []), 5)]
        rank = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        deals = {k: [ha[k]['net'][0] + hb[k]['net'][1], max(reach(ha[k]), reach(hb[k]), key=rank.get)] for k in both}
        return {'id': str(m.relative_to(root)), 'run': run_of(root, m), 'a': a, 'b': b, 'started': True,
                'round': cfg.get('round') or (int(rnd) if rnd.isdigit() else 1),
                'replicate': cfg.get('replicate') or (int(rep.group(1)) if rep else 1),
                'matches': matches, 'deals_total': len(cfg.get('seeds', [])), 'pairs_done': len(both),
                'pair_net_a': sum(x[0] for x in deals.values()),
                'kinds': {p: 'bot' if 'bot' in s else 'model' for p, s in specs.items()},
                'labels': {p: LABELS.get(p) or display_name(p, s) for p, s in specs.items()},
                'bb': game.get('big_unit') or game.get('bb') or 1,
                'finished': len(matches) == 2 and all(x['finished'] for x in matches),
                'failed': sum(play_failure(m / s) is not None for s in SIDES),
                'times': [h['t'] for s in hands.values() for h in s if h.get('t')],
                '_specs': specs, '_deals': deals,
                '_written': {s: sorted({h['hand'] for h in hands[s] if isinstance(h.get('hand'), int)}) for s in SIDES}}

    def scan(self, root: Path, base: Path | None = None) -> tuple[list[dict], list[Path]]:
        convs, plans = _walk(base or root)
        return [self.conversation(root, m) for m in convs], plans


INVENTORIES: dict = {}


def inventory(root: Path) -> Inventory:
    return INVENTORIES.setdefault(str(root), Inventory())


def _matchup(root: Path, m: Path):
    """One duplicate pair: both matches, how many deals both have finished, and a's net over those deals."""
    return inventory(root).conversation(root, m)


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


def _runs_and_matchups(root: Path, only: str | None = None):
    """{run: [matchup]} including matchups a plan.json lists that haven't started. `only` reads one run's folder
    and nothing else (unless a planned player is named only by another run)."""
    base = None if only in (None, '.') else (root / only)
    if base is not None and not base.is_dir():
        return {}
    found, plans = inventory(root).scan(root, base)
    by = {}
    for m in found:
        if only is None or m['run'] == only:
            by.setdefault(m['run'], []).append(m)
    for plan_dir in plans:
        plan_path = plan_dir / 'plan.json'
        rel = plan_dir.relative_to(root)
        run = str(rel) if str(rel) != '.' else '.'
        if only is not None and run != only:
            continue
        plan = _json(plan_path) or {}
        have = {(m['a'], m['b']) for m in by.get(run, [])}
        missing = [(a, b) for a, b in plan.get('matchups', []) if (a, b) not in have]
        if not missing:
            continue
        from .runner import EFFORT_BACKENDS
        here = [{p: {'label': lab, **({'bot': True} if m['kinds'].get(p) == 'bot' else {})} for p, lab in m['labels'].items()}
                for m in by.get(run, [])]
        effort = plan.get('effort')
        sources = here + [_plan_specs(plan_path, plan)]
        if any(not any(p in src for src in sources) for a, b in missing for p in (a, b)):
            seen: dict = {}
            for m in (found if only is None else inventory(root).scan(root)[0]):
                if m['run'] != run:
                    seen.setdefault(m['run'], {}).update(m['_specs'])
            sources += [{p: {**s, 'effort': effort or s.get('effort')} if effort and s.get('backend') in EFFORT_BACKENDS else s
                         for p, s in specs.items()} for specs in seen.values()]
        for a, b in missing:
            (la, ka), (lb, kb) = _who(a, sources), _who(b, sources)
            by.setdefault(run, []).append({
                'id': str(rel / f'{a}__vs__{b}') if run != '.' else f'{a}__vs__{b}', 'run': run, 'a': a, 'b': b,
                'started': False, 'round': 1, 'replicate': 1, 'matches': [], 'deals_total': plan.get('deals', 0), 'pairs_done': 0,
                'pair_net_a': 0, 'kinds': {p: k for p, k in ((a, ka), (b, kb)) if k}, 'labels': {a: la, b: lb},
                'bb': 1, 'live': False, 'finished': False, 'failed': 0, 'updated': 0, 'times': []})
    return by


def roster(root: Path, run: str) -> list:
    """Every player of a run, from its plan when there is one, with where it stands: awaiting its first
    matchup, playing, done, or stopped (every deal it started ended on a backend error and none finished)."""
    ms = _runs_and_matchups(root, run).get(run, [])
    plan = _json((root if run == '.' else root / run) / 'plan.json') or {}
    names = list(plan.get('players') or [])
    names += sorted(({m['a'] for m in ms} | {m['b'] for m in ms}) - set(names))
    out = []
    for p in names:
        mine = [m for m in ms if p in (m['a'], m['b'])]
        stuck = [m for m in mine if m['started'] and not m['finished']]
        status = ('done' if mine and all(m['finished'] for m in mine)
                  else 'stopped' if stuck and all(m.get('failed') for m in stuck) and not any(m['pairs_done'] for m in mine)
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


MATCH_KEYS = ('id', 'hands_total', 'hands_done', 'finished', 'live')


def _public(m):
    return {**{k: v for k, v in m.items() if k != 'times' and not k.startswith('_')},
            'matches': [{k: x[k] for k in MATCH_KEYS} for x in m['matches']]}


def index(root: Path, run: str | None = None):
    """The run's matchups, read from its own folder only (every run's for None or all runs)."""
    from .history import ALL
    run = None if run == ALL else run
    by = _runs_and_matchups(root, run)
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
        base = root if run == '.' else root / run
        has_quiz = (base / 'quiz.json').exists()
        track, tournament = track_info(root, run)
        title = tournament or (root.name if run == '.' else run.split('/')[-1])
        out.append({'run': run, 'quiz': has_quiz, 'title': title, 'track': track, 'tournament': tournament,
                    'matchups': len(ms), 'matchups_started': sum(m['started'] for m in ms),
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


def track_info(root: Path, run: str) -> tuple:
    """(track, tournament) of a run: from its plan.json; a run folder named official or adaptation inside a
    tournament folder belongs to that tournament. (None, None) for a run outside any tournament."""
    plan = _json((root if run == '.' else root / run) / 'plan.json') or {}
    track = plan.get('track') or (plan.get('design') or {}).get('track')
    tail = run.split('/')
    if not track and len(tail) > 1 and tail[-1] in ('official', 'adaptation'):
        track = tail[-1]
    if not track:
        return None, None
    return track, plan.get('tournament') or (tail[-2] if len(tail) > 1 and tail[-1] == track else run.split('/')[-1])


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
    conv = conversation_status(d.parent) if d.name in SIDES else None
    return {'summary': match_summary(root, d), 'config': _json(d / 'config.json'),
            'hands': _lines(d / 'hands.jsonl'), 'events': events, 'result': _json(d / 'result.json'),
            'failure': play_failure(d),
            **({'status': {str(k): v for k, v in conv['status'].items()}, 'rejected': conv['rejected'], 'checked': conv['checked']} if conv else {})}


CONVERSATIONS: dict = {}


def conversation_status(m: Path) -> dict:
    """Which deals of a conversation are results, from the one shared reader (`ratings.read_conversation`):
    {'status': {deal: 'verified' | 'provisional' | 'rejected'}, 'rejected': [{'hand', 'reason'}], 'checked'}.
    'verified': both plays committed it, they pass as a duplicate pair and replay from their seeds; 'provisional':
    written but not a result yet; 'rejected': failed a check, so it counts nowhere. Without the reader every
    finished deal is 'unchecked'. Kept until a record file of the conversation changes."""
    stamp = _stamp(m)
    hit = CONVERSATIONS.get(str(m))
    if hit and hit[0] == stamp:
        return hit[1]
    written = {s: {h.get('hand') for h in _lines(m / s / 'hands.jsonl')} for s in SIDES}
    every = {k for ks in written.values() for k in ks if isinstance(k, int)}
    out = {'status': {}, 'rejected': [], 'checked': False,
           'failed': {s: f for s in SIDES for f in [play_failure(m / s)] if f}}
    try:
        from .ratings import read_conversation
        conv = read_conversation(m)
    except Exception:
        conv = None
    if conv:
        given = conv.get('status') if isinstance(conv.get('status'), dict) else None
        if given and set(given) <= set(SIDES):   # per play: a deal is a result only when both plays say so
            sides = [given.get(x) or {} for x in SIDES]
            pick = lambda k: [d.get(k, d.get(str(k))) for d in sides]
            given = {k: 'rejected' if 'rejected' in pick(k) else 'verified' if all(x == 'verified' for x in pick(k)) else 'provisional'
                     for k in every}
        rejected = [x for x in conv.get('rejected') or [] if isinstance(x, dict)]
        whole = any(x.get('hand') is None for x in rejected)
        bad = {x.get('hand') for x in rejected}
        verified = set(conv.get('verified') or [])
        for k in sorted(every):
            g = given and (given.get(k) or given.get(str(k)))
            out['status'][k] = g if g in ('verified', 'provisional', 'rejected') else \
                'rejected' if whole or k in bad else 'verified' if k in verified else 'provisional'
        out.update(rejected=[{'hand': x.get('hand'), 'reason': str(x.get('reason') or '')} for x in rejected], checked=True)
    else:
        both = written['a-first'] & written['b-first']
        out['status'] = {k: 'unchecked' if k in both else 'provisional' for k in sorted(every)}
    CONVERSATIONS[str(m)] = (stamp, out)
    return out


def pairing(root: Path, run: str, a: str, b: str) -> dict | None:
    """Every conversation of one pairing in playing order (round, then replicate) with each deal's result from
    a's side: {'a', 'b', 'conversations': [{'id', 'round', 'replicate', 'deals_total', 'live', 'finished',
    'deals': [[deal, net_bb or None, status, reach]], 'rejected', 'failed'}], 'totals': {...}}. A deal's status is
    the shared reader's (verified, provisional, rejected, or unchecked without it), plus 'playing' for the deal a
    live conversation is on and 'failed' for the deal a play gave up on after a backend error (`play_failure`).
    Totals count verified deals only (unchecked ones when nothing could be checked)."""
    from .history import ALL
    if run in (None, '', ALL):
        return None
    base = root if run == '.' else (root / run).resolve()
    if base != root and root not in base.parents:
        return None
    ms = [m for m in _runs_and_matchups(root, run).get(run, []) if m['started'] and {m['a'], m['b']} == {a, b}]
    if not ms:
        return None
    ms.sort(key=lambda m: (m['round'], m['replicate'], m['id']))
    convs, tot = [], {'verified': 0, 'net_bb': 0.0, 'won': 0, 'lost': 0, 'even': 0, 'provisional': 0, 'rejected': 0,
                      'unchecked': 0, 'playing': 0, 'failed': 0}
    for m in ms:
        st = conversation_status(root / m['id'])
        sign = 1 if m['a'] == a else -1
        bb = m['bb'] or 1
        deals = []
        # a deal one play gave up on is 'failed', not forever 'in progress': it counts nowhere until it is replayed
        failed = {} if m['finished'] else st.get('failed') or {}
        statuses = dict(st['status'])
        for f in failed.values():
            if statuses.get(f['hand']) not in ('verified', 'rejected'):
                statuses[f['hand']] = 'failed'
        for k, status in sorted(statuses.items()):
            d = m['_deals'].get(k)
            net = round(sign * d[0] / bb, 2) if d else None
            deals.append([k, net if status != 'rejected' else None, status, d[1] if d else None])
            tot[status] = tot.get(status, 0) + 1
            if status in ('verified', 'unchecked') and net is not None:
                tot['net_bb'] += net
                tot['won' if net > 0 else 'lost' if net < 0 else 'even'] += 1
        nxt = max(statuses, default=-1) + 1
        if m['live'] and not failed and nxt < m['deals_total']:
            deals.append([nxt, None, 'playing', None])
            tot['playing'] += 1
        convs.append({'id': m['id'], 'round': m['round'], 'replicate': m['replicate'], 'deals_total': m['deals_total'],
                      'live': m['live'], 'finished': m['finished'], 'deals': deals, 'rejected': st['rejected'],
                      'failed': [{'side': s, 'hand': f['hand'], 'player': f['player'], 'message': f['message']}
                                 for s, f in failed.items()]})
    checked = any(conversation_status(root / m['id'])['checked'] for m in ms)
    tot['net_bb'] = round(tot['net_bb'], 2)
    tot['counted'] = tot['verified'] if checked else tot['unchecked']
    return {'run': run, 'a': a, 'b': b, 'checked': checked, 'conversations': convs, 'totals': tot}


def warm_conversations(root: Path, order: list):
    """Check every conversation in the background, most recent run first, so a replay's first visit is instant.
    It yields between conversations, so pages are served meanwhile."""
    import threading

    def go():
        for run in order:
            for m in _runs_and_matchups(root, run).get(run, []):
                if m['started']:
                    try:
                        conversation_status(root / m['id'])
                    except Exception:
                        pass
                    time.sleep(0.001)
    threading.Thread(target=go, daemon=True).start()


def pairing_key(run: str, a: str, b: str) -> str:
    """The one parameter naming a pairing: 'RUN|A|B' with the two players in sorted order."""
    return '|'.join([run, *sorted([a, b])])


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
    data = _verified_quiz(base / 'quiz.json')
    if not isinstance(data, dict):
        return None
    labels, kinds = {}, {}
    for m in _runs_and_matchups(root, run).get(run, []):
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
    data = {**data, 'players': _quiz_links(base, data)}
    return {**data, 'spots': _with_questions(data), 'labels': {**labels, **(data.get('labels') or {})},
            'kinds': {**kinds, **(data.get('kinds') or {})}}


def _quiz_links(base: Path, data: dict) -> dict:
    """Each quiz player with `link`: whether its score may stand for the run's tournament player of the same model
    and reasoning level (ratings.quiz_link), or is a separate model-level assessment, and what differs."""
    from .history import identity as ident_of
    from .ratings import MIXED, _base_identity, player_provenance, quiz_link
    players = data.get('players') or {}
    if not isinstance(players, dict):
        return players
    prov = player_provenance(base)
    by_base = {v['identity']: p for p, v in prov.items() if v.get('identity') not in (None, MIXED)}
    out = {}
    for pid, x in players.items():
        if not isinstance(x, dict):
            out[pid] = x
            continue
        same = by_base.get(_base_identity(ident_of(pid, x.get('spec') or {})))
        link = quiz_link(x, data, prov[same]) if same else {'status': 'none', 'differences': []}
        out[pid] = {**x, 'link': {**link, 'player': same}}
    return out


def _with_questions(data: dict) -> list:
    """The quiz's spots, each without a stored or recorded prompt given `question`: the text the quiz code sends
    for it (`quiz.question`), rebuilt from its definition when that still matches the stored spot, and
    `question_version`, the quiz code's version, which can differ from the version the quiz ran."""
    spots = data.get('spots') or []
    players = data.get('players') or {}
    asked = set()
    for r in (players.values() if isinstance(players, dict) else players):
        for a in (r or {}).get('answers') or []:
            if isinstance(a, dict) and any(isinstance(c, dict) and c.get('prompt') for c in a.get('calls') or []):
                asked.add(str(a.get('spot')))
    try:
        from . import quiz as Q
    except ImportError:
        return spots
    out = []
    for s in spots:
        if not isinstance(s, dict) or s.get('prompt') or str(s.get('id')) in asked:
            out.append(s)
            continue
        try:
            d = Q.spot_def(s['id'])
            ep, hero, _, _ = Q.build(d)
            same = (list(d['hole']) == list(s.get('hole') or []) and list(d['board']) == list(s.get('board') or [])
                    and d['range'] == s.get('villain_range') and ep.pot == s.get('pot') and ep.to_call(hero) == s.get('to_call'))
            out.append({**s, 'question': Q.question(s, ep, hero), 'question_version': Q.VERSION} if same else s)
        except Exception:
            out.append(s)
    return out


def _verified_quiz(path: Path):
    """The quiz as the judge re-derives it (`ratings.verified_quiz_file`: answers re-verified, scores recounted,
    eligibility and attestation the judge's); the stored file only when the judge cannot read it."""
    if not path.exists():
        return None
    try:
        from .ratings import verified_quiz_file
    except ImportError:
        return _json(path)
    try:
        return verified_quiz_file(path)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def warm_quizzes(root: Path, order: list):
    """Verify each run's quiz in the background (the first check of a quiz takes seconds)."""
    import threading

    def go():
        for run in order:
            base = root if run == '.' else root / run
            try:
                _verified_quiz(base / 'quiz.json')
            except Exception:
                pass
    threading.Thread(target=go, daemon=True).start()


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
    for m in _runs_and_matchups(root, key).get(key, []):
        labels.update(m['labels'])
        kinds.update(m['kinds'])
    r['labels'] = {**labels, **(r.get('labels') or {})}
    r['kinds'] = {**kinds, **(r.get('kinds') or {})}
    r['run_audit'] = audit(root, key)
    r.update(_v2_extras(base))
    return r


def _v2_extras(base: Path) -> dict:
    """What the v2 modules add to a run's ratings, each None until its module (and the run) can give it:
    the meta-game ranking, a lightweight forensic summary and the AIVAT estimates. Never raises."""
    out = {'meta_rank': None, 'forensics_summary': None, 'aivat': None}
    try:
        out['forensics_summary'] = forensics_summary(base)
    except Exception:
        pass
    mr = _optional_module('meta_rank')
    sc = _optional_module('scorecard')
    if mr is not None and sc is not None:
        try:
            pairs = sc._pairings(base)
            out['meta_rank'] = mr.meta_rankings(pairs) if pairs else None
        except Exception:
            pass
    av = _optional_module('aivat')
    if av is not None:
        try:
            out['aivat'] = av.evaluate_run_aivat(base)
        except Exception:
            pass
    return out


# ---- v2: scorecards, forensics, solver hand evaluation and the pre-flight planner --------------------------------

def _optional_module(name: str):
    """A v2 module, or None while it is not there yet (or fails to import): every v2 answer degrades cleanly."""
    import importlib
    try:
        return importlib.import_module(f'.{name}', __package__)
    except Exception:
        return None


def _run_base(root: Path, run) -> Path | None:
    base = root if run in (None, '', '.') else (root / run).resolve()
    return base if base == root or root in base.parents else None


def _run_labels(root: Path, run) -> tuple[dict, dict]:
    labels, kinds = {}, {}
    for m in _runs_and_matchups(root, run if run not in (None, '') else '.').get(run if run not in (None, '') else '.', []):
        labels.update(m['labels'])
        kinds.update(m['kinds'])
    return {**labels, **{p: v for p, v in LABELS.items() if p in labels}}, kinds


def scorecard_for(root: Path, run, solver='dcfr') -> dict | None:
    """The six-axis scorecard of a run with display labels and kinds merged in; None for a run outside the root."""
    from . import history
    if run == history.ALL:
        return None
    base = _run_base(root, run)
    if base is None or not base.exists():
        return None
    mod = _optional_module('scorecard')
    if mod is None:
        return {'available': False, 'reason': 'poker_arena/scorecard.py is not available', 'players': {}, 'axes': []}
    data = mod.build_scorecard(base, solver=solver)
    labels, kinds = _run_labels(root, run)
    data['labels'] = labels
    data['kinds'] = {**kinds, **{p: v.get('kind') for p, v in data.get('players', {}).items() if v.get('kind')}}
    data['run'] = run if run not in (None, '') else '.'
    return data


def _forensics_labelled(root: Path, run) -> dict | None:
    base = _run_base(root, run)
    mod = _optional_module('forensics')
    if base is None or not base.exists() or mod is None:
        return None
    data = mod.audit_run_forensics(base)
    labels, kinds = _run_labels(root, run)
    data['labels'] = labels
    data['kinds'] = kinds
    data['run'] = run if run not in (None, '') else '.'
    return data


FORENSICS_MEMO: dict = {}


def forensics_for(root: Path, run) -> dict | None:
    """The four-channel forensic audit with display labels, kept in memory until a record file of the run changes."""
    from . import history
    if run == history.ALL:
        return None
    base = _run_base(root, run)
    if base is None:
        return None
    stamp = repr(signature(base, PLAYERS_FILE[0]))
    hit = FORENSICS_MEMO.get((str(root), run))
    if hit and hit[0] == stamp:
        return hit[1]
    data = _forensics_labelled(root, run)
    if data is not None:
        FORENSICS_MEMO[(str(root), run)] = (stamp, data)
    return data


def forensics_summary(base: Path) -> dict | None:
    """The small part of the audit the standings show: incident counts per channel and who is flagged."""
    mod = _optional_module('forensics')
    if mod is None:
        return None
    try:
        s = mod.audit_run_forensics(base)
    except Exception:
        return None
    summary = s.get('summary') or {}
    return {'total_incidents': summary.get('total_incidents', 0), 'flagged_players': summary.get('flagged_players', []),
            'channels': {c: (v or {}).get('incidents', 0) for c, v in (summary.get('channels') or {}).items()},
            'players': {p: {'total_incidents': v.get('total_incidents', 0), 'clean': v.get('clean', True)}
                        for p, v in (s.get('players') or {}).items()}}


def _compute_scorecard_in_child(root: str, run, players_file: str, labels: dict):
    LABELS.update(labels)
    PLAYERS_FILE[0] = players_file
    return scorecard_for(Path(root), run)


HAND_EVALS: dict = {}


def _fallback_decisions(rec: dict, config: dict, events: list) -> list:
    """Pot odds, minimum defence frequency and a rough equity for each decision, from the hand record alone,
    for when the solver cannot be used."""
    from . import cards
    import random as _random
    players = config.get('players') or []
    rng = _random.Random(rec.get('seed') or 0)
    bb = (config.get('game') or {}).get('big_unit') or (config.get('game') or {}).get('bb') or 2
    board_all = list(rec.get('board') or [])
    out = []
    for i, a in enumerate(rec.get('actions') or []):
        seat = a.get('seat', 0)
        street = a.get('street') or 'preflop'
        shown = board_all[:{'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(street, len(board_all))]
        hole = (rec.get('hole') or [[], []])[seat]
        to_call, pot = a.get('to_call') or 0, a.get('pot_before') or 0
        try:
            eq = round(cards.equity(hole, shown, rng, sims=150), 3)
        except Exception:
            eq = None
        out.append({'street': street, 'solver': 'pot_odds', 'pot_bb': round(pot / bb, 2), 'to_call_bb': round(to_call / bb, 2),
                    'pot_odds': round(to_call / (pot + to_call), 3) if to_call else 0.0,
                    'mdf': round(pot / (pot + to_call), 3) if to_call else 1.0, 'equity': eq, 'best_action': None,
                    'gto_freqs': {}, 'action_evs_bb': {}, 'taken_action': a.get('action'), 'ev_loss_mbb': None,
                    'is_blunder': False, 'is_indifferent': False, 'seat': seat,
                    'player': players[seat] if seat < len(players) else None, 'decision_id': a.get('decision_id'),
                    'action_idx': i, 'amount': a.get('amount'), 'board': shown, 'hole': hole})
    return out


def hand_eval(root: Path, mid: str, hand: int) -> dict | None:
    """One recorded hand, replayed through the solver (`solver.evaluate_hand`) with the AIVAT luck adjustment
    (`aivat.aivat_hand`): {hand, decisions, per_seat_ev_loss_mbb, aivat, source}. When the solver is missing or
    fails, the decisions carry pot odds, MDF and an equity estimate only, and `source` says so."""
    d = (root / mid).resolve()
    if root not in d.parents or not (d / 'config.json').exists():
        return None
    stamp = (_mtime(d / 'hands.jsonl'), _mtime(d / 'events.jsonl'))
    hit = HAND_EVALS.get((str(d), hand))
    if hit and hit[0] == stamp:
        return hit[1]
    rec = next((h for h in _lines(d / 'hands.jsonl') if h.get('hand') == hand), None)
    if rec is None:
        return None
    config = _json(d / 'config.json') or {}
    events = [e for e in _lines(d / 'events.jsonl') if e.get('episode', 0) == hand]
    game = config.get('game') or {}
    players = config.get('players') or []
    out = {'hand': hand, 'players': players, 'decisions': [], 'per_seat_ev_loss_mbb': [None, None], 'aivat': None,
           'source': 'solver', 'note': None}
    sol = _optional_module('solver')
    try:
        if sol is None:
            raise RuntimeError('poker_arena/solver.py is not available')
        ev = sol.evaluate_hand(rec, events=events, solver='dcfr')
        out['decisions'] = ev.get('decisions', [])
        out['per_seat_ev_loss_mbb'] = ev.get('per_seat_ev_loss_mbb', [None, None])
        for x in out['decisions']:
            if x.get('seat') is not None and x['seat'] < len(players):
                x['player_id'] = players[x['seat']]
    except Exception as e:
        out['source'] = 'pot_odds'
        out['note'] = f'solver unavailable ({type(e).__name__}); showing pot odds and a rough equity only'
        out['decisions'] = _fallback_decisions(rec, config, events)
    av = _optional_module('aivat')
    if av is not None:
        try:
            x = av.aivat_hand(rec, bb=game.get('big_unit') or game.get('bb') or 2, stack=game.get('stack') or 200)
            out['aivat'] = {k: v for k, v in x.items() if k != 'equities'}
        except Exception:
            out['aivat'] = None
    HAND_EVALS[(str(d), hand)] = (stamp, out)
    return out


CALLS_PER_HAND = 5.0      # model decisions per hand, both seats together, counting the occasional retry
SECONDS_PER_CALL = 12.0   # a typical reasoning-model reply
PLAN_PREVIEWS: dict = {}


def plan_command(p: dict) -> str:
    names = ' '.join(f'model{i + 1}' for i in range(p['players']))   # the stack is in big blinds here, chips (2 a blind) on the command line
    parts = ['python -m poker_arena plan', names, f'--tracks {p["track"]}']
    if p['track'] == 'official':
        parts += ['--deals 1', f'--replicates {p["replicates"]}']
    else:
        parts += [f'--adaptation-deals {p["deals"]}', f'--adaptation-replicates {p["replicates"]}']
    parts += [f'--stack {2 * p["stack"]}', f'--target-edge {p["target"]:g}', f'--alpha {p["alpha"]:g}', f'--prior-sd {p["sd"]:g}',
              f'--out runs/{p["track"]}-run', '--write plan.json']
    return ' '.join(parts)


def plan_params(q: dict) -> dict:
    """The planner's inputs from a query string (each value a list, as parse_qs gives them), clamped to the ranges
    the page offers."""
    def num(name, default, lo, hi, cast=float):
        try:
            v = cast(float(q.get(name, [default])[0]))
        except (TypeError, ValueError):
            v = cast(default)
        return max(lo, min(hi, v))
    track = q.get('track', ['official'])[0]
    track = track if track in ('official', 'adaptation') else 'official'
    return {'players': num('players', 4, 2, 12, int), 'track': track,
            'deals': 1 if track == 'official' else num('deals', 50, 2, 200, int),
            'replicates': num('replicates', 25 if track == 'official' else 4, 1, 200, int),
            'target': num('target', 50, 15, 150), 'stack': num('stack', 100, 20, 200, int),
            'alpha': num('alpha', 0.05, 0.001, 0.2), 'aivat_vr': num('aivat_vr', 1.0, 1.0, 5.0),
            'sd': num('sd', 350, 100, 1000)}


def plan_preview(p: dict) -> dict:
    """The planner's answer for plan_params `p`: the design the official (or adaptation) track would get, in raw
    duplicate deals and, with the expected AIVAT variance reduction, in effective deals; its price and its
    power. Read-only: nothing is written."""
    from .ratings import normalize_design
    from .runner import make_tournament
    memo = tuple(sorted(p.items()))
    if memo in PLAN_PREVIEWS:
        return PLAN_PREVIEWS[memo]
    ids = [f'model{i + 1}' for i in range(p['players'])]
    shared = {'alpha': p['alpha'], 'target_edge_bb100': p['target'], 'prior_sd_bb100': p['sd']}
    spec = {'deals': p['deals'], 'replicates': p['replicates'], 'design': shared}
    doc = make_tournament(ids, out='runs/preview', tracks=[p['track']], seed=1000, game_config={'stack': 2 * p['stack']},
                          **{p['track']: spec})
    plan = doc['tracks'][p['track']]
    plan['design'].pop('max_rounds', None)
    d = normalize_design(plan)
    pairings = len(plan['matchups'])
    R, reps, rounds = d['round_deals'], d['replicates'], d['max_rounds']
    deals_pairing = d.get('max_deals') or R * reps * rounds
    conv_pairing = d.get('conversations_needed') or reps * rounds
    vr = p['aivat_vr']
    hands_pairing = 2 * deals_pairing
    hands_total = hands_pairing * pairings
    calls = hands_total * CALLS_PER_HAND
    concurrency = 2
    out = {'params': p, 'track': p['track'], 'pairings': pairings,
           'round_deals': R, 'replicates': reps, 'max_rounds': rounds, 'max_rounds_basis': d.get('max_rounds_basis'),
           'conversations_per_pairing': conv_pairing, 'deals_per_pairing': deals_pairing,
           'hands_per_pairing': hands_pairing, 'hands_total': hands_total,
           'conversations_total': conv_pairing * pairings,
           'raw_deals_per_pairing': deals_pairing,
           'aivat_effective_deals_per_pairing': round(deals_pairing / vr, 1),
           'aivat_deals_saved_per_pairing': round(deals_pairing - deals_pairing / vr, 1),
           'aivat_hands_total': round(hands_total / vr), 'aivat_vr': vr,
           'alpha_pairing': d.get('alpha_pairing'), 'threshold': d.get('threshold'),
           'stopping_boundary': {'rule': 'betting confidence sequence', 'threshold': d.get('threshold'),
                                 'alpha_pairing': d.get('alpha_pairing'), 'outcome_bound_bb100': d.get('outcome_bound_bb100'),
                                 'formula': '2 / alpha_pairing'},
           'power_scenarios': d.get('power_scenarios') or [], 'power_by_edge': d.get('power_by_edge') or {},
           'power_at_max': d.get('power_at_max'), 'underpowered': d.get('underpowered', False),
           'official': bool(d.get('official')), 'label': d.get('label'),
           'estimated_calls': round(calls), 'estimated_calls_with_aivat': round(calls / vr),
           'estimated_runtime_hours': round(calls * SECONDS_PER_CALL / concurrency / 3600, 1),
           'estimated_runtime_hours_with_aivat': round(calls / vr * SECONDS_PER_CALL / concurrency / 3600, 1),
           'assumptions': {'calls_per_hand': CALLS_PER_HAND, 'seconds_per_call': SECONDS_PER_CALL, 'concurrency': concurrency},
           'command': plan_command(p)}
    if len(PLAN_PREVIEWS) > 256:
        PLAN_PREVIEWS.clear()
    PLAN_PREVIEWS[memo] = out
    return out


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
        if not r.get('all'):
            for a, b in sorted({tuple(sorted((m['a'], m['b']))) for m in ms if m['started']}):
                pr = pairing(root, r['run'], a, b)
                if pr:
                    put(f'pairing-{key(pairing_key(r["run"], a, b))}.json', pr)
        put(f'ratings-{key(r["run"])}.json', ratings_for(root, r['run'], boot=300))
        if not r.get('all'):
            for name, fn in (('scorecard', scorecard_for), ('forensics', forensics_for)):
                try:
                    obj = fn(root, r['run'])
                except Exception:
                    obj = None
                if obj is not None:
                    put(f'{name}-{key(r["run"])}.json', obj)
        q = quiz(root, r['run'])
        if q is not None:
            put(f'quiz-{key(r["run"])}.json', q)
    return files


def signature(base: Path, players_file='players.toml') -> tuple:
    """What a run's ratings depend on: every record file under it (hands, events and calls, and the plan,
    decisions, tournament, config and quiz files), and the players file that names them. For all runs the
    history is synced from these same files, so they are its inputs too. The viewer's own cache is not an input."""
    files = [p for p in base.rglob('*.json*') if p.suffix in ('.json', '.jsonl') and CACHE_DIR not in p.parts]
    files.append(Path(players_file))
    def stat(p):
        try:
            st = p.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return 0, 0
    return tuple(sorted((str(p), *stat(p)) for p in files))


CACHE_DIR = '.viewer-cache'


def code_stamp() -> str:
    """The arena's code as it is on disk: a cached leaderboard from other code is never served as current."""
    import hashlib
    pkg = Path(__file__).parent
    h = hashlib.sha256()
    for f in sorted(pkg.rglob('*.py')):
        st = f.stat()
        h.update(f'{f.relative_to(pkg)}:{st.st_mtime_ns}:{st.st_size};'.encode())
    return h.hexdigest()[:16]


def _compute_in_child(root: str, run, players_file: str, labels: dict):
    LABELS.update(labels)
    PLAYERS_FILE[0] = players_file
    return ratings_for(Path(root), run)


class RatingsCache:
    """Leaderboards computed off the request path. A request gets the last result at once, marked fresh when
    no record file (and no code) changed since, else stale while a background thread recomputes it. Results
    are kept on disk under the root, so a restarted server answers immediately too. Only a run never computed
    before waits, briefly; past that the answer is 'pending' and the page asks again. The computing happens in
    a separate process, so pages and replays are served at full speed meanwhile."""

    child_fn = staticmethod(_compute_in_child)
    prefix = 'ratings'

    def __init__(self, root: Path, players_file='players.toml', compute=None, wait=0.5):
        import threading
        self.root, self.players_file, self.wait = root, players_file, wait
        self.compute = compute or self._in_child
        self.pool = None
        self.code = code_stamp()
        self.mem: dict = {}
        self.errors: dict = {}   # run -> (stamp, message) of the last failed recompute; not retried until the stamp changes
        self.jobs: dict = {}
        self.work_lock = threading.Lock()   # one recompute at a time (and one writer to the history database)
        self.jobs_lock = threading.Lock()

    def _in_child(self, run):
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        for attempt in range(2):
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=1)
            try:
                return self.pool.submit(self.child_fn, str(self.root), run, self.players_file, dict(LABELS)).result()
            except BrokenProcessPool:
                self.pool = None
                if attempt:
                    raise

    def base(self, run):
        from .history import ALL
        base = self.root if run in (None, '', '.', ALL) else (self.root / run).resolve()
        return base if base == self.root or self.root in base.parents else None

    def stamp(self, run) -> str:
        import hashlib
        return hashlib.sha256(repr((self.code, signature(self.base(run), self.players_file))).encode()).hexdigest()[:24]

    def _file(self, run) -> Path:
        return self.root / CACHE_DIR / f'{self.prefix}-{key(run or ".")}.json'

    def _load(self, run):
        hit = self.mem.get(run)
        if hit is None:
            hit = _json(self._file(run))
            if isinstance(hit, dict) and 'data' in hit and 'stamp' in hit:
                self.mem[run] = hit
            else:
                hit = None
        return hit

    def _save(self, run, entry):
        self.mem[run] = entry
        try:
            f = self._file(run)
            f.parent.mkdir(exist_ok=True)
            tmp = f.with_suffix('.tmp')
            tmp.write_text(json.dumps(entry, separators=(',', ':')))
            tmp.replace(f)
        except OSError:
            pass   # a read-only root still serves from memory

    def _work(self, run):
        with self.work_lock:
            try:
                stamp = self.stamp(run)
                hit = self.mem.get(run)
                if hit and hit['stamp'] == stamp:
                    return
                t = time.time()
                data = self.compute(run)
                if data is not None:
                    self._save(run, {'stamp': stamp, 'code': self.code, 'computed': time.time(), 'seconds': round(time.time() - t, 2), 'data': data})
                self.errors.pop(run, None)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.errors[run] = (stamp, f'{type(e).__name__}: {e}')

    def refresh(self, run):
        import threading
        with self.jobs_lock:
            job = self.jobs.get(run)
            if job is None or not job.is_alive():
                job = self.jobs[run] = threading.Thread(target=self._work, args=(run,), daemon=True)
                job.start()
            return job

    def warm(self, runs_):
        """Compute every run in the background, most recent first, so the first visit to each is instant."""
        import threading
        def go():
            for run in runs_:
                try:
                    if self._load(run) is None or self._load(run)['stamp'] != self.stamp(run):
                        self.refresh(run).join()
                except Exception:
                    pass
        threading.Thread(target=go, daemon=True).start()

    def get(self, run):
        """('ok', data) with data['cache'] saying how current it is, ('pending', None), ('error', message),
        or ('missing', None) for a run outside the root."""
        if self.base(run) is None:
            return 'missing', None
        stamp = self.stamp(run)
        hit = self._load(run)
        failed = self.errors.get(run, (None, None))
        if (hit is None or hit['stamp'] != stamp) and failed[0] != stamp:
            job = self.refresh(run)
            if hit is None:
                job.join(self.wait)
                hit = self._load(run)
        failed = self.errors.get(run, (None, None))
        if hit is None:
            return ('error', failed[1]) if failed[0] == stamp else ('pending', None)
        fresh = hit['stamp'] == stamp
        meta = {'fresh': fresh, 'computed': hit.get('computed'), 'seconds': hit.get('seconds'),
                'reason': None if fresh else 'code' if hit.get('code') != self.code else 'records',
                'refreshing': not fresh and run in self.jobs and self.jobs[run].is_alive(),
                'error': failed[1] if not fresh and failed[0] == stamp else None}
        if hit['data'] is None:
            return 'missing', None
        return 'ok', {**hit['data'], 'cache': meta}


class ScorecardCache(RatingsCache):
    """The same cache for the six-axis scorecards: computed in a separate process, served at once, stamped by the
    run's record files and the code."""
    child_fn = staticmethod(_compute_scorecard_in_child)
    prefix = 'scorecard'


def _cache_answer(handler, cache, run):
    if run == '*':
        return handler.send(404, '{"error": "the all-runs view has no scorecard; pick a run"}')
    base = _run_base(cache.root, run)
    if base is None or not base.is_dir():
        return handler.send(404, '{"error": "no such run"}')
    status, data = cache.get(run)
    if status == 'ok':
        return handler.send(200, json.dumps(data))
    if status == 'pending':
        return handler.send(202, json.dumps({'pending': True, 'run': run}))
    if status == 'error':
        return handler.send(500, json.dumps({'error': data}))
    return handler.send(404, '{"error": "no such run"}')


def serve(root, port=8800, host='127.0.0.1', players_file='players.toml'):
    root = Path(root).resolve()  # match folders are resolved before use, so the root must be too
    LABELS.update(load_labels(players_file))
    PLAYERS_FILE[0] = players_file
    cache = RatingsCache(root, players_file)
    scores = ScorecardCache(root, players_file)
    order = [r['run'] for r in sorted(runs(root), key=lambda r: (bool(r.get('all')), -r['updated']))]
    cache.warm(order)
    warm_conversations(root, [r for r in order if r != '*'])
    warm_quizzes(root, [r for r in order if r != '*'])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the page moved on before the answer arrived

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
                if url.path == '/api/pairing':
                    parts = q.get('p', [''])[0].split('|')
                    data = pairing(root, '|'.join(parts[:-2]), *parts[-2:]) if len(parts) >= 3 else None
                    return self.send(200, json.dumps(data)) if data else self.send(404, '{"error": "no such pairing"}')
                if url.path == '/api/quiz':
                    data = quiz(root, run)
                    return self.send(200, json.dumps(data)) if data is not None else self.send(404, '{"error": "no quiz in this run"}')
                if url.path == '/api/scorecard':
                    return _cache_answer(self, scores, run)
                if url.path == '/api/forensics':
                    data = forensics_for(root, run)
                    return self.send(200, json.dumps(data)) if data is not None else self.send(404, '{"error": "no such run"}')
                if url.path == '/api/hand_eval':
                    try:
                        hand = int(q.get('hand', ['0'])[0])
                    except ValueError:
                        return self.send(400, '{"error": "hand must be a number"}')
                    data = hand_eval(root, q.get('id', [''])[0], hand)
                    return self.send(200, json.dumps(data)) if data is not None else self.send(404, '{"error": "no such hand"}')
                if url.path == '/api/plan_preview':
                    return self.send(200, json.dumps(plan_preview(plan_params(q))))
                if url.path == '/api/ratings':
                    status, data = cache.get(run)
                    if status == 'ok':
                        return self.send(200, json.dumps(data))
                    if status == 'pending':
                        return self.send(202, json.dumps({'pending': True, 'run': run}))
                    if status == 'error':
                        return self.send(500, json.dumps({'error': data}))
                    return self.send(404, '{"error": "no such run"}')
            except (BrokenPipeError, ConnectionResetError):
                return
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
