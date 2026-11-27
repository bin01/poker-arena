"""Matches, duplicate pairs and tournaments.

A match is one series of hands between two players with fresh memories. A duplicate pair is two
matches on the same deals with the seats swapped: in hand k each player gets, in turn, both seats'
cards against the other player, so the luck of the deal cancels.

A match directory holds:
    config.json       the manifest (game, players by seat, specs, seeds, deal fingerprints, system prompts,
                      each player's protocol) and the audit (code version, CLI versions, effective settings
                      per seat, round/replicate)
    events.jsonl      every event, with which seats could see it (the viewer reads this); `thought` events
                      carry requested / executed / correction for every decision
    calls.jsonl       every model call: prompt, reply, reasoning, usage, latency, command line, CLI session
                      before and after, tools used, errors, and the failed tries and rate-limit waits before it
    raw.jsonl         the CLI's raw output (exit status, stdout, stderr) for every call and every failed try
    hands.jsonl       one line per finished hand: cards, board, actions, net result per seat, seed, fingerprint
    state.json        the checkpoint: hands_committed, notes, CLI sessions and turns, bot random states, resumes
    interrupted.jsonl rows of hands a stopped run had started; they are replayed from scratch
    result.json       written when the match finishes

A hand is committed when its row is in hands.jsonl and state.json (written whole, then renamed into place)
says hands_committed includes it. Rerunning a stopped match first cuts every log back to the committed hands
(rows past them go to interrupted.jsonl), then resumes, only with the recorded settings. If the stopped run
was inside a hand, every CLI conversation is rebuilt from the committed turns, so no seat remembers a hand
that the record dropped. A tournament directory also holds plan.json (the plan and the players' specs, which
a rerun must match), decisions.json (every look of the sequential design) and tournament.json (whether the
runner is running, finished or stopped, and where each pairing stands).
"""
from __future__ import annotations
import asyncio
import copy
import hashlib
import itertools
import json
import os
import time
import tomllib
from pathlib import Path

from .agents import make_agent
from .backends import BackendError, cli_version
from .games import get_game
from .games.holdem import json_safe

NAMES = ['Alex', 'Blair', 'Casey', 'Devon', 'Emery', 'Finley', 'Gray', 'Harper', 'Indy']


DEFAULT_EFFORT = 'high'                      # reasoning level for a model player that doesn't set one
EFFORT_BACKENDS = {'claude', 'codex', 'agy'}  # the CLIs that take a reasoning level


def load_players(path, effort: str | None = None) -> dict:
    """Players from a TOML file. Every model on a CLI that takes a reasoning level gets one: `effort` (a
    tournament-wide override) wins, then the player's own `effort`, then DEFAULT_EFFORT. The level is
    recorded in each match's config, so results always say which level played."""
    data = tomllib.loads(Path(path).read_text())
    players = data.get('players', data)
    for spec in players.values():
        if spec.get('backend') in EFFORT_BACKENDS:
            spec['effort'] = effort or spec.get('effort') or DEFAULT_EFFORT
    return players


def code_version() -> str | None:
    """The git commit the arena is running from (with "+dirty" if it has local changes), or None."""
    import subprocess
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (sha + ('+dirty' if dirty else '')) or None
    except (OSError, subprocess.SubprocessError):
        return None


def _append(path: Path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, default=str) + '\n')


def _write(path: Path, obj, indent=None):
    """Write a JSON file whole: a temp file, flushed to disk, renamed over the old one."""
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(obj, indent=indent, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sync(path: Path):
    if path.exists():
        with path.open('a') as f:
            os.fsync(f.fileno())


LOGS = ('hands.jsonl', 'events.jsonl', 'calls.jsonl', 'raw.jsonl')


def _reconcile(out: Path, done: int, resume: int) -> tuple[dict, dict]:
    """Cut every log back to the first `done` hands. Rows of later hands move to interrupted.jsonl; lines a
    kill left half-written are dropped. Returns ({file: rows moved}, {file: torn lines})."""
    moved, torn = {}, {}
    for fname in LOGS:
        path = out / fname
        if not path.exists():
            continue
        keep, late, bad = [], [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            (keep if r.get('episode', r.get('hand', 0)) < done else late).append(r)
        for r in late:
            _append(out / 'interrupted.jsonl', {'file': fname, 'after_hand': done, 'resume': resume, **r})
        if late or bad:
            tmp = path.with_name(fname + '.tmp')
            tmp.write_text(''.join(json.dumps(r, default=str) + '\n' for r in keep))
            os.replace(tmp, path)
        if late:
            moved[fname] = len(late)
        if bad:
            torn[fname] = bad
    return moved, torn


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


class ResumeMismatch(ValueError):
    """A stopped match or tournament was rerun with settings that would change the experiment."""


def _public(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if 'key' not in k.lower()}


def _same(x):
    return json.loads(json.dumps(x, default=str))


def protocol(game, pid: str, spec: dict, seats: int = 2) -> dict:
    """What a player's results depend on, the same in every seat, match and run: the effective settings that
    change play (backend, model, effort, isolation, instructions, tool policy, extra args, retries, fixes,
    thinking display ... or a bot's settings) and the sha1 of the game's system prompt with the names and the
    number of hands left out. `fingerprint` is the sha1 of the settings."""
    template = game.system_prompt('{player}', ['{player}'] + ['{opponent}'] * (seats - 1), '{hands}')
    agent = make_agent(pid, spec)
    agent.start(template, game.response_schema())
    settings = agent.protocol()
    if agent.kind == 'model':
        settings['system_prompt_sha1'] = hashlib.sha1(template.encode()).hexdigest()[:12]
    settings = _same(settings)
    return {'fingerprint': hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16],
            'settings': settings}


def match_manifest(game, seat_players: list[tuple[str, dict]], seeds: list[int]) -> dict:
    """What defines a match: resuming it with anything different would be another experiment."""
    names, n = NAMES[:len(seat_players)], len(seat_players)
    return _same({'game': game.describe(), 'players': [p for p, _ in seat_players], 'names': names, 'seeds': list(seeds),
                  'deal_fingerprints': [game.deal_fingerprint(s, k % n) for k, s in enumerate(seeds)],
                  'specs': {p: _public(s) for p, s in seat_players},
                  'system_prompts': [game.system_prompt(names[i], names, len(seeds)) for i in range(n)],
                  'protocol': {p: protocol(game, p, s, n) for p, s in seat_players}})


def manifest_diff(stored: dict, fresh: dict) -> list[str]:
    """Keys whose values differ. A key the stored manifest lacks (written by older code) is adopted."""
    return [k for k in fresh if k in stored and _same(stored[k]) != _same(fresh[k])]


def _thought_fields(d, last, err) -> dict:
    """requested / executed / correction for a decision (see HoldemHand.apply for the policy)."""
    executed = {'action': last.get('action'), 'amount': last.get('amount')}
    final = {'action': json_safe(d.action.get('action')), 'amount': json_safe(d.action.get('amount'))}
    if d.fallback or err:
        reasons = [e['error'] for e in d.errors] + ([err] if err else [])
        requested = d.errors[-1]['requested'] if d.errors else (None if d.fallback else final)
        correction = {'kind': 'fallback', 'reason': '; '.join(reasons) or 'no playable reply'}
    elif d.errors:
        requested = d.errors[0]['requested']
        reason = ' Then: '.join([d.errors[0]['error']] + ([last['correction']['reason']] if last.get('correction') else []))
        kind = 'tool_use' if any(e.get('kind') == 'tool_use' for e in d.errors) else 'retry'
        correction = {'kind': kind, 'reason': f'asked again {len(d.errors)}x: {reason}'}
    elif last.get('correction'):
        requested, correction = last['requested'], last['correction']
    else:
        requested, correction = final, None
    return {'requested': requested, 'executed': executed, 'correction': correction, 'retries': d.errors}


async def run_match(game, seat_players: list[tuple[str, dict]], seeds: list[int], out: Path, *,
                    log=print, agent_seed: int = 0, meta: dict | None = None) -> dict:
    """Play one match. `seat_players` is [(player_id, player_spec)] by seat; `seeds` one per hand.
    `meta` (round, replicate, ...) is recorded in the config. Rerunning resumes; a rerun whose game,
    players, specs, seeds or prompts differ from the recorded config raises ResumeMismatch first."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(seat_players)
    fresh = match_manifest(game, seat_players, seeds)
    names, ids = fresh['names'], fresh['players']
    stored = json.loads((out / 'config.json').read_text()) if (out / 'config.json').exists() else None
    if stored is not None:
        diff = manifest_diff(stored, fresh)
        if diff:
            raise ResumeMismatch(f'{out}: the recorded match differs in {", ".join(diff)}; '
                                 f'rerun with the recorded settings or use a new folder')
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    schema = game.response_schema()
    agents = []
    for i, (pid, spec) in enumerate(seat_players):
        a = make_agent(pid, spec, seed=agent_seed * 100 + i)
        a.start(fresh['system_prompts'][i], schema)
        agents.append(a)
    exes = {a.backend.exe for a in agents if a.kind == 'model' and a.backend.exe}
    versions = {e: await asyncio.to_thread(cli_version, e) for e in sorted(exes)}
    if stored is None:
        config = {**fresh, **(meta or {}), 'code_version': code_version(), 'cli_versions': versions,
                  'effective': [_same(a.describe()) for a in agents], 'started': round(time.time())}
        _write(out / 'config.json', config, indent=2)
    system_sha = [hashlib.sha1(p.encode()).hexdigest()[:12] for p in fresh['system_prompts']]

    state = json.loads((out / 'state.json').read_text()) if (out / 'state.json').exists() else {}
    done = state.get('hands_committed', state.get('hands', 0))
    pending = state.get('pending', [''] * n)
    resumes = state.get('resumes', [])
    for a, st in zip(agents, state.get('agents', [])):
        a.restore(st)
    # rows of a hand the stopped run had started: moved to interrupted.jsonl, so the logs match what is
    # replayed, and the CLI conversations (which may still hold those turns) are rebuilt from the committed ones
    dropped, torn = _reconcile(out, done, len(resumes) + 1)
    if len(_read_jsonl(out / 'hands.jsonl')) != done:
        raise RuntimeError(f'{out}: state.json commits {done} hands but hands.jsonl holds '
                           f'{len(_read_jsonl(out / "hands.jsonl"))}; the record is damaged')
    rebuilt = []
    for i, a in enumerate(agents):
        if a.kind == 'model' and (dropped or a.backend.rebuilding):
            a.backend.rebuild()
            rebuilt.append(i)
    if done or dropped or torn:
        resumes.append({'t': round(time.time()), 'after_hand': done, 'code_version': code_version(),
                        'cli_versions': versions, 'dropped_rows': dropped, 'torn_lines': torn, 'rebuilt_seats': rebuilt})
        log(f'{out}: resuming after hand {done}' + (f'; dropped an interrupted hand, rebuilt seats {rebuilt}'
                                                    if dropped else ''))
    totals = [0] * n
    for h in _read_jsonl(out / 'hands.jsonl')[:done]:
        totals = [t + x for t, x in zip(totals, h['net'])]
    fallbacks = state.get('fallbacks', [0] * n)
    started = time.time()

    for k in range(done, len(seeds)):
        ep = game.episode(k, seeds[k], names)
        seen = 0

        def flush():
            nonlocal seen
            for ev in ep.events[seen:]:
                _append(out / 'events.jsonl', ev.to_json())
                for i in range(n):
                    if ev.visible is None or i in ev.visible:
                        line = ep.render(ev, i)
                        if ev.kind == 'hole' and agents[i].kind == 'model':
                            line += f'\nYour notes so far: {agents[i].notes or "(none yet)"}'
                        if line:
                            pending[i] = (pending[i] + '\n' + line).strip()
            seen = len(ep.events)

        def log_calls(seat, calls):
            for c in calls:
                c = dict(c)
                raw = c.pop('raw', None)
                tries = []
                for j, f in enumerate(c.get('failed_attempts') or []):
                    f = dict(f)
                    f_raw = f.pop('raw', None)
                    if f_raw:
                        _append(out / 'raw.jsonl', {'hand': k, 'seat': seat, 'attempt': c.get('attempt'),
                                                    'failed_try': j, **f_raw})
                    tries.append(f)
                if tries:
                    c['failed_attempts'] = tries
                _append(out / 'calls.jsonl', {'hand': k, 'seat': seat, 'player': ids[seat],
                                              'system_sha1': system_sha[seat], **c})
                if raw:
                    _append(out / 'raw.jsonl', {'hand': k, 'seat': seat, 'attempt': c.get('attempt'), **raw}
                            if isinstance(raw, dict) else {'hand': k, 'seat': seat, 'raw': raw})

        flush()
        while not ep.done:
            seat = ep.to_act
            req = ep.request(seat)
            text = (pending[seat] + '\n\n' + req.prompt).strip()
            pending[seat] = ''
            validate = lambda act, ep=ep, seat=seat: copy.deepcopy(ep).apply(seat, act)
            try:
                d = await agents[seat].act(text, req, validate)
            except BackendError as e:
                log_calls(seat, getattr(e, 'calls', []))
                raise
            action = d.action
            if d.fallback:
                fallbacks[seat] += 1
                action = ep.fallback(seat)
            err = ep.apply(seat, action)
            if err:  # a bot or fallback should never be illegal; fall back rather than stall
                fallbacks[seat] += 1
                ep.apply(seat, ep.fallback(seat))
            last = ep.history[-1] if ep.history else {}
            # the private side of the decision, for the viewer: placed just before the public action event
            thought = {'kind': 'thought', 'episode': k, 'visible': [], 'data': {
                'seat': seat, 'player': ids[seat], 'reasoning': d.reasoning, 'thinking': d.thinking,
                'notes': agents[seat].notes if agents[seat].kind == 'model' else '', 'fallback': d.fallback or bool(err),
                'tries': len(d.calls), 'action': last.get('action'), 'amount': last.get('amount'),
                **_thought_fields(d, last, err),
                'seconds': round(sum(c.get('latency') or 0 for c in d.calls), 1),
                'served_model': next((c['served_model'] for c in reversed(d.calls) if c.get('served_model')), None)}}
            _append(out / 'events.jsonl', thought)
            log_calls(seat, d.calls)
            flush()
        summary = {**ep.summary(), 'seed': seeds[k], 'fingerprint': fresh['deal_fingerprints'][k],
                   't': round(time.time(), 1)}  # when it finished: the viewer's progress and ETA
        totals = [t + x for t, x in zip(totals, ep.payoffs)]
        # the commit: the hand's rows reach the disk, then the checkpoint that counts them replaces the old one
        _append(out / 'hands.jsonl', summary)
        for fname in LOGS:
            _sync(out / fname)
        state = {'hands_committed': k + 1, 'hands': k + 1, 'pending': pending, 'agents': [a.state() for a in agents],
                 'fallbacks': fallbacks, 'resumes': resumes}
        _write(out / 'state.json', state)
        log(f'{out.name} hand {k + 1}/{len(seeds)}: ' + ', '.join(f'{ids[i]} {ep.payoffs[i]:+d}' for i in range(n))
            + ' | total ' + ', '.join(f'{ids[i]} {totals[i]:+d}' for i in range(n)))

    result = {'players': ids, 'names': names, 'hands': len(seeds), 'net': totals,
              'bb_per_100': [round(t / game.big_unit / len(seeds) * 100, 1) for t in totals],
              'fallbacks': fallbacks, 'resumes': len(resumes), 'seconds': round(time.time() - started, 1)}
    _write(out / 'result.json', result, indent=2)
    return result


def matchup_name(a, b, round_=1, replicate=1):
    """The folder of a matchup. A sequential tournament adds rounds as `a__vs__b__r2`, ...; replicates (fresh
    memories, their own deals) add `__x2`, ... before the round: `a__vs__b__x2__r3`."""
    return f'{a}__vs__{b}' + (f'__x{replicate}' if replicate > 1 else '') + (f'__r{round_}' if round_ > 1 else '')


async def run_duplicate(game, a: tuple[str, dict], b: tuple[str, dict], seeds, out: Path, *, sems=(), log=print,
                        round_=1, replicate=1, on_start=None):
    """Both matches of a duplicate pair; they share deals and run at the same time (each holding `sems`,
    the concurrency slots of the CLIs it uses, while it plays; `on_start()` is called once one holds them)."""
    out = Path(out) / matchup_name(a[0], b[0], round_, replicate)
    meta = {'pairing': [a[0], b[0]], 'round': round_, 'replicate': replicate}

    async def one(order, label, s):
        for sem in sems:
            await sem.acquire()
        if on_start:
            on_start()
        try:
            return await run_match(game, order, seeds, out / label, log=log, agent_seed=s,
                                   meta={**meta, 'order': label})
        finally:
            for sem in sems:
                sem.release()

    return await asyncio.gather(one([a, b], 'a-first', 1), one([b, a], 'b-first', 2))


DESIGN = {'alpha': 0.05, 'max_rounds': 6, 'block': 10}   # plus round_deals = the plan's deals


def make_plan(player_ids, deals=50, seed=1000, out='runs/pilot', game='holdem', game_config=None,
              players_file='players.toml', concurrency=2, matchups=None, design=None, replicates=1):
    """A tournament. Every pairing plays rounds of `deals` duplicate deals (times `replicates` independent
    pairs, each with fresh memories and its own deals) until `ratings.decide` calls it or `max_rounds` is used."""
    return {'game': game, 'game_config': game_config or {}, 'players_file': players_file, 'players': player_ids,
            'matchups': matchups or [list(p) for p in itertools.combinations(player_ids, 2)],
            'deals': deals, 'seed': seed, 'out': out, 'concurrency': concurrency,
            'design': {**DESIGN, 'round_deals': deals, **(design or {})}, 'replicates': replicates}


def plan_design(plan: dict) -> dict | None:
    """The plan's sequential design, as `ratings.normalize_design` fills it in (replicates included), so the
    runner's stopping rule and the ratings' verdicts use the same one. None: one round and no stopping rule."""
    from . import ratings
    if hasattr(ratings, 'normalize_design'):
        return ratings.normalize_design(plan)
    reps = {'replicates': plan.get('replicates') or 1}
    if plan.get('design'):
        return {**DESIGN, 'round_deals': plan['deals'], **plan['design'], **reps}
    if plan.get('until_decided'):
        return {**DESIGN, 'round_deals': plan['deals'], 'max_rounds': max(1, plan['until_decided'] // plan['deals']),
                **reps}
    return None


def round_seeds(plan: dict, round_: int, replicate: int) -> list[int]:
    """Every pairing plays the same deals in the same round and replicate; no deal repeats within a plan."""
    reps, size = plan.get('replicates', 1), plan['deals']
    start = plan['seed'] + ((round_ - 1) * reps + replicate - 1) * size
    return list(range(start, start + size))


def is_bot(spec):
    return 'bot' in spec


def _pair_xs(out: Path, folders: list[str], a: str, bb: float) -> tuple[list[float], list[str]]:
    """a's result per duplicate deal in bb/100, in play order over the given matchup folders, and the folder
    of each deal (so blocks never span two matches)."""
    xs, groups = [], []
    for f in folders:
        ha = {h['hand']: h for h in _read_jsonl(out / f / 'a-first' / 'hands.jsonl')}
        hb = {h['hand']: h for h in _read_jsonl(out / f / 'b-first' / 'hands.jsonl')}
        first = (json.loads((out / f / 'a-first' / 'config.json').read_text())['players'][0]
                 if (out / f / 'a-first' / 'config.json').exists() else a)
        sign = 1 if first == a else -1
        both = sorted(set(ha) & set(hb))
        xs += [sign * (ha[k]['net'][0] + hb[k]['net'][1]) / bb / 2 * 100 for k in both]
        groups += [f] * len(both)
    return xs, groups


def _record_decision(out: Path, a, b, v: dict):
    path = out / 'decisions.json'
    allv = json.loads(path.read_text()) if path.exists() else {}
    looks = allv.setdefault(matchup_name(a, b), [])
    looks[v['looks_used'] - 1:] = [{'t': round(time.time()), **v}]
    _write(path, allv, indent=2)


TERMINAL = {'a': 'decided', 'b': 'decided', 'even': 'decided'}


class Lifecycle:
    """<out>/tournament.json: whether the runner is running, finished or stopped, the round being played, and
    each pairing's status ('waiting' for its turn or its next round, 'playing', 'decided', 'exhausted'), with
    its latest verdict and looks used. Pairings this invocation doesn't run keep their recorded state."""

    def __init__(self, out: Path, plan: dict, rounds: int):
        self.path, self.rounds = out / 'tournament.json', rounds
        old = json.loads(self.path.read_text()) if self.path.exists() else {}
        recorded = json.loads((out / 'decisions.json').read_text()) if (out / 'decisions.json').exists() else {}
        self.doc = {'status': 'running', 'round': old.get('round', 0), 'pairings': {}, 'pid': os.getpid(),
                    'started': round(time.time()), 'updated': None}
        for a, b in plan['matchups']:
            looks = recorded.get(matchup_name(a, b)) or []
            last = looks[-1] if looks else {}
            status = TERMINAL.get(last.get('verdict')) or (
                'exhausted' if last and last.get('looks_used', 0) >= rounds else 'waiting')
            self.doc['pairings'][f'{a}|{b}'] = {'status': status, 'verdict': last.get('verdict'),
                                                'looks_used': last.get('looks_used', 0),
                                                'round': last.get('looks_used', 0)}
        self.write()

    def write(self):
        self.doc['updated'] = round(time.time(), 1)
        _write(self.path, self.doc, indent=2)

    def pairing(self, a, b, **kw):
        self.doc['pairings'][f'{a}|{b}'].update(kw)
        if kw.get('round'):
            self.doc['round'] = max(self.doc['round'], kw['round'])
        self.write()

    def end(self, status=None):
        done = all(p['status'] in ('decided', 'exhausted') for p in self.doc['pairings'].values())
        self.doc['status'] = status or ('finished' if done else 'stopped')
        self.write()


async def run_plan(plan: dict, *, only=None, concurrency=None, log=print):
    """Every matchup of the plan as duplicate pairs, on the same deals (so every player sees the same
    cards across the whole tournament). `concurrency` caps how many matches use each backend (each CLI)
    at once, so one subscription's rate limits don't stall the others; bot-only matches run freely.

    The plan's `design` is enforced here: after each round of a pairing, `ratings.decide` judges all of its
    deals so far; the pairing stops on a verdict ('a', 'b', 'even') or after `max_rounds` looks, never more.
    Every look is kept in the run's decisions.json. The plan and the players' specs are recorded in
    plan.json; a rerun resumes, and one that would change the experiment raises ResumeMismatch before any
    model is called."""
    players = load_players(plan.get('players_file', 'players.toml'), effort=plan.get('effort'))
    design, reps = plan_design(plan), plan.get('replicates', 1)
    rounds = design['max_rounds'] if design else 1
    cfg = {'hands_per_match': plan['deals'], **plan.get('game_config', {})}
    game = get_game(plan.get('game', 'holdem'))(cfg)
    limit = concurrency or plan.get('concurrency', 2)
    out = Path(plan['out'])
    for a, b in plan['matchups']:
        missing = [p for p in (a, b) if p not in players]
        if missing:
            raise KeyError(f'not in {plan.get("players_file")}: {", ".join(missing)}')
    involved = sorted({p for m in plan['matchups'] for p in m})
    manifest = _same({'game': plan.get('game', 'holdem'), 'game_config': plan.get('game_config', {}),
                      'players': plan['players'], 'matchups': plan['matchups'], 'deals': plan['deals'],
                      'seed': plan['seed'], 'design': design, 'replicates': reps, 'effort': plan.get('effort'),
                      'specs': {p: _public(players[p]) for p in involved}})
    stored = json.loads((out / 'plan.json').read_text()) if (out / 'plan.json').exists() else None
    if stored is not None:
        diff = manifest_diff({**stored, 'design': plan_design(stored), 'replicates': stored.get('replicates', 1)}
                             if 'deals' in stored else stored, manifest)
        if diff:
            raise ResumeMismatch(f'{out}/plan.json: this tournament was planned with different {", ".join(diff)}; '
                                 f'rerun with the recorded plan or use a new `out`')
    bad = []
    for a, b in plan['matchups']:
        for r in range(1, rounds + 1):
            for j in range(1, reps + 1):
                seeds = round_seeds(plan, r, j)
                for label, order in (('a-first', [a, b]), ('b-first', [b, a])):
                    cfg_path = out / matchup_name(a, b, r, j) / label / 'config.json'
                    if cfg_path.exists():
                        diff = manifest_diff(json.loads(cfg_path.read_text()),
                                             match_manifest(game, [(p, players[p]) for p in order], seeds))
                        if diff:
                            bad.append(f'{cfg_path.parent.relative_to(out)} ({", ".join(diff)})')
    if bad:
        raise ResumeMismatch('recorded matches differ from this plan: ' + '; '.join(bad))
    out.mkdir(parents=True, exist_ok=True)
    from .server import display_name
    labels = {pid: display_name(pid, spec) for pid, spec in players.items() if pid in plan['players']}
    _write(out / 'plan.json', {**plan, **manifest, 'labels': labels, 'code_version': code_version(),
                               'created': (stored or {}).get('created') or round(time.time())},
           indent=2)  # for the viewer, and the manifest a rerun must match
    life = Lifecycle(out, plan, rounds)
    sems = {}
    tasks = []

    async def pairing(a, b, mine):
        folders = []
        try:
            for r in range(1, rounds + 1):
                names = [matchup_name(a, b, r, j) for j in range(1, reps + 1)]
                life.pairing(a, b, status='waiting')
                res = await asyncio.gather(*[
                    run_duplicate(game, (a, players[a]), (b, players[b]), round_seeds(plan, r, j), out, sems=mine,
                                  log=log, round_=r, replicate=j,
                                  on_start=lambda r=r: life.pairing(a, b, status='playing', round=r))
                    for j in range(1, reps + 1)])
                folders += names
                if not design:
                    life.pairing(a, b, status='exhausted', looks_used=1)
                    return res
                from . import ratings
                xs, groups = _pair_xs(out, folders, a, game.big_unit)
                v = ratings.decide(xs, design, len(plan['matchups']), r, groups=groups)
                _record_decision(out, a, b, v)
                final = v['verdict'] in TERMINAL or r >= rounds
                life.pairing(a, b, verdict=v['verdict'], looks_used=v['looks_used'],
                             status=TERMINAL.get(v['verdict']) or ('exhausted' if final else 'waiting'))
                if final:
                    who = {'a': f'decided: {a} is better', 'b': f'decided: {b} is better', 'even': 'decided: even (within the equivalence margin)', 'exhausted': 'undecided, every planned look used'}.get(v['verdict'], v['verdict'])
                    log(f'{a} vs {b}: {who} after {r} of {rounds} rounds ({v.get("deals")} deals, '
                        f'{a} {v.get("edge") if v.get("edge") is not None else "?"} bb/100)')
                    return v
        except Exception as e:
            life.pairing(a, b, status='waiting', error=repr(e)[:500])
            raise

    for a, b in plan['matchups']:
        if only and matchup_name(a, b) not in only and a not in only and b not in only:
            continue
        kinds = sorted({players[p]['backend'] for p in (a, b) if not is_bot(players[p])})  # sorted: no deadlock
        mine = [sems.setdefault(k, asyncio.Semaphore(limit)) for k in kinds]
        tasks.append(pairing(a, b, mine))
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        life.end('stopped')
        raise
    for r in results:
        if isinstance(r, Exception):
            log(f'matchup failed (rerun the same command to resume): {r!r}')
    life.end()
    return results
