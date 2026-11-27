"""Matches, duplicate pairs and tournaments.

A match is one series of hands between two players with fresh memories. A duplicate pair is two
matches on the same deals with the seats swapped: in hand k each player gets, in turn, both seats'
cards against the other player, so the luck of the deal cancels.

A match directory holds:
    config.json   the game, the players by seat, the deal seeds
    events.jsonl  every event, with which seats could see it (the viewer reads this)
    calls.jsonl   every model call: prompt, raw reply, reasoning, usage, latency, errors
    hands.jsonl   one line per finished hand: cards, board, actions, net result per seat
    state.json    what a stopped match needs to resume (finished hands, notes, CLI sessions)
    result.json   written when the match finishes

Rerunning a stopped match resumes after its last finished hand.
"""
from __future__ import annotations
import asyncio
import copy
import itertools
import json
import time
import tomllib
from pathlib import Path

from .agents import make_agent
from .games import get_game

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


async def run_match(game, seat_players: list[tuple[str, dict]], seeds: list[int], out: Path, *,
                    log=print, agent_seed: int = 0) -> dict:
    """Play one match. `seat_players` is [(player_id, player_spec)] by seat; `seeds` one per hand."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    n = len(seat_players)
    names = NAMES[:n]
    ids = [p for p, _ in seat_players]
    config = {'game': game.describe(), 'players': ids, 'names': names, 'seeds': seeds,
              'code_version': code_version(), 'started': round(time.time()),
              'specs': {p: {k: v for k, v in s.items() if 'key' not in k.lower()} for p, s in seat_players}}
    if not (out / 'config.json').exists():
        (out / 'config.json').write_text(json.dumps(config, indent=2))
    schema = game.response_schema()
    agents = []
    for i, (pid, spec) in enumerate(seat_players):
        a = make_agent(pid, spec, seed=agent_seed * 100 + i)
        a.start(game.system_prompt(names[i], names, len(seeds)), schema)
        agents.append(a)

    state = json.loads((out / 'state.json').read_text()) if (out / 'state.json').exists() else {}
    done = state.get('hands', 0)
    pending = state.get('pending', [''] * n)
    for a, st in zip(agents, state.get('agents', [])):
        a.restore(st)
    if done:
        log(f'{out}: resuming after hand {done}')
    # drop partial-hand logs from an interrupted run so the logs match what is replayed
    for fname in ('events.jsonl', 'calls.jsonl'):
        rows = [r for r in _read_jsonl(out / fname) if r.get('episode', r.get('hand', 0)) < done]
        (out / fname).write_text(''.join(json.dumps(r, default=str) + '\n' for r in rows))
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

        flush()
        while not ep.done:
            seat = ep.to_act
            req = ep.request(seat)
            text = (pending[seat] + '\n\n' + req.prompt).strip()
            pending[seat] = ''
            validate = lambda act, ep=ep, seat=seat: copy.deepcopy(ep).apply(seat, act)
            d = await agents[seat].act(text, req, validate)
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
                'notes': agents[seat].notes if agents[seat].kind == 'model' else '', 'fallback': d.fallback,
                'tries': len(d.calls), 'action': last.get('action'), 'amount': last.get('amount'),
                'seconds': round(sum(c.get('latency') or 0 for c in d.calls), 1),
                'served_model': next((c['served_model'] for c in reversed(d.calls) if c.get('served_model')), None)}}
            _append(out / 'events.jsonl', thought)
            for c in d.calls:
                _append(out / 'calls.jsonl', {'hand': k, 'seat': seat, 'player': ids[seat], **c})
            flush()
        summary = {**ep.summary(), 't': round(time.time(), 1)}  # when it finished: the viewer's progress and ETA
        totals = [t + x for t, x in zip(totals, ep.payoffs)]
        _append(out / 'hands.jsonl', summary)
        state = {'hands': k + 1, 'pending': pending, 'agents': [a.state() for a in agents], 'fallbacks': fallbacks}
        (out / 'state.json').write_text(json.dumps(state, default=str))
        log(f'{out.name} hand {k + 1}/{len(seeds)}: ' + ', '.join(f'{ids[i]} {ep.payoffs[i]:+d}' for i in range(n))
            + ' | total ' + ', '.join(f'{ids[i]} {totals[i]:+d}' for i in range(n)))

    result = {'players': ids, 'names': names, 'hands': len(seeds), 'net': totals,
              'bb_per_100': [round(t / game.big_unit / len(seeds) * 100, 1) for t in totals],
              'fallbacks': fallbacks, 'seconds': round(time.time() - started, 1)}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def matchup_name(a, b, round_=1):
    """The folder of a matchup; a tournament played until decided adds rounds as `a__vs__b__r2`, ..."""
    return f'{a}__vs__{b}' + (f'__r{round_}' if round_ > 1 else '')


async def run_duplicate(game, a: tuple[str, dict], b: tuple[str, dict], seeds, out: Path, *, sems=(), log=print,
                        round_=1):
    """Both matches of a duplicate pair; they share deals and run at the same time (each holding `sems`,
    the concurrency slots of the CLIs it uses, while it plays)."""
    out = Path(out) / matchup_name(a[0], b[0], round_)

    async def one(order, label, s):
        for sem in sems:
            await sem.acquire()
        try:
            return await run_match(game, order, seeds, out / label, log=log, agent_seed=s)
        finally:
            for sem in sems:
                sem.release()

    return await asyncio.gather(one([a, b], 'a-first', 1), one([b, a], 'b-first', 2))


def make_plan(player_ids, deals=50, seed=1000, out='runs/pilot', game='holdem', game_config=None,
              players_file='players.toml', concurrency=2, matchups=None):
    return {'game': game, 'game_config': game_config or {}, 'players_file': players_file, 'players': player_ids,
            'matchups': matchups or [list(p) for p in itertools.combinations(player_ids, 2)],
            'deals': deals, 'seed': seed, 'out': out, 'concurrency': concurrency}


def is_bot(spec):
    return 'bot' in spec


async def run_plan(plan: dict, *, only=None, concurrency=None, log=print):
    """Every matchup of the plan as a duplicate pair, on the same deals (so every player sees the same
    cards across the whole tournament). `concurrency` caps how many matches use each backend (each CLI)
    at once, so one subscription's rate limits don't stall the others; bot-only matches run freely.

    With `until_decided` (a number of deals) in the plan, `deals` is a round: after each round a pairing
    whose edge is settled (ratings.settle) stops, and the rest play another round on new deals (the same
    new deals for every pairing), until settled or `until_decided` deals. Rerun to resume."""
    players = load_players(plan.get('players_file', 'players.toml'), effort=plan.get('effort'))
    cfg = {'hands_per_match': plan['deals'], **plan.get('game_config', {})}
    game = get_game(plan.get('game', 'holdem'))(cfg)
    seeds = [plan['seed'] + k for k in range(plan['deals'])]
    limit = concurrency or plan.get('concurrency', 2)
    Path(plan['out']).mkdir(parents=True, exist_ok=True)
    from .server import display_name
    labels = {pid: display_name(pid, spec) for pid, spec in players.items() if pid in plan['players']}
    (Path(plan['out']) / 'plan.json').write_text(json.dumps({**plan, 'labels': labels}, indent=2))  # for the viewer
    sems = {}
    tasks = []
    cap = plan.get('until_decided')

    async def pairing(a, b, mine):
        r = 1
        while True:
            round_seeds = seeds if r == 1 else [plan['seed'] + (r - 1) * plan['deals'] + k for k in range(plan['deals'])]
            res = await run_duplicate(game, (a, players[a]), (b, players[b]), round_seeds, Path(plan['out']),
                                      sems=mine, log=log, round_=r)
            if not cap:
                return res
            from .ratings import load_deals, settle
            xs = [(d['net_a'] if d['a'] == a else -d['net_a']) / d['bb'] / 2 * 100
                  for d in load_deals(plan['out']) if {d['a'], d['b']} == {a, b}]
            v = settle(xs)
            if v['status'] == 'settled' or len(xs) + plan['deals'] > cap:
                log(f'{a} vs {b}: {"settled" if v["status"] == "settled" else "stopped at the cap"} after '
                    f'{len(xs)} deals, {a} {v["a_bb_per_100"]:+.0f} bb/100')
                return res
            r += 1

    for a, b in plan['matchups']:
        if only and matchup_name(a, b) not in only and a not in only and b not in only:
            continue
        missing = [p for p in (a, b) if p not in players]
        if missing:
            raise KeyError(f'not in {plan.get("players_file")}: {", ".join(missing)}')
        kinds = sorted({players[p]['backend'] for p in (a, b) if not is_bot(players[p])})  # sorted: no deadlock
        mine = [sems.setdefault(k, asyncio.Semaphore(limit)) for k in kinds]
        tasks.append(pairing(a, b, mine))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            log(f'matchup failed (rerun the same command to resume): {r!r}')
    return results
