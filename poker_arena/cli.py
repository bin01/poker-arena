"""Command line: python -m poker_arena <command> --help"""
from __future__ import annotations
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from .games import get_game
from .runner import load_players, make_plan, run_duplicate, run_plan


def cmd_play(args):
    players = load_players(args.players, effort=args.effort)
    game = get_game(args.game)({'hands_per_match': args.deals})
    seeds = [args.seed + k for k in range(args.deals)]
    res = asyncio.run(run_duplicate(game, (args.a, players[args.a]), (args.b, players[args.b]), seeds, Path(args.out)))
    for label, r in zip(('a-first', 'b-first'), res):
        print(label, dict(zip(r['players'], r['net'])))
    from .ratings import ratings, table, SESSION
    print(table(ratings(args.out, boot=100)))
    if args.deals < SESSION:
        print(f'\nElo needs at least {SESSION // 2} deals per matchup (sessions are {SESSION} deals); '
              f'with {args.deals} only bb/100 is meaningful.')


def cmd_plan(args):
    plan = make_plan(args.players_list, deals=args.deals, seed=args.seed, out=args.out, players_file=args.players,
                     concurrency=args.concurrency)
    text = json.dumps(plan, indent=2)
    if args.write:
        Path(args.write).write_text(text + '\n')
        print(f'wrote {args.write}: {len(plan["matchups"])} matchups, {2 * len(plan["matchups"]) * args.deals} hands')
    else:
        print(text)


def cmd_tournament(args):
    plan = json.loads(Path(args.plan).read_text())
    if args.effort:
        plan['effort'] = args.effort
    if args.dry_run:
        for a, b in plan['matchups']:
            print(f'{a} vs {b}: 2 x {plan["deals"]} hands -> {plan["out"]}/{a}__vs__{b}')
        return
    asyncio.run(run_plan(plan, only=args.only, concurrency=args.concurrency))
    from .ratings import ratings, table
    print(table(ratings(plan['out'], boot=200)))


def cmd_ratings(args):
    from .ratings import ratings, table
    anchor = None
    if args.anchor:
        name, value = args.anchor.split('=')
        anchor = (name, float(value))
    r = ratings(args.root, boot=args.boot, anchor=anchor)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))
    print(table(r))
    print(f'\n{r["deals"]} duplicate deals in {r["sessions"]} sessions of {r["session_deals"]} deals '
          f'({2 * r["session_deals"]} hands). Elo: chance of winning a session; bb/100: big blinds won per 100 hands.')


def cmd_view(args):
    from .server import serve
    serve(args.root, args.port, args.host, players_file=args.players)


def cmd_history(args):
    """Update the history database from the runs under root, and print the all-time leaderboard."""
    from . import history
    root = Path(args.root).resolve()
    db = Path(args.db) if args.db else root / history.DB_NAME
    changed = history.sync(root, db)
    r = history.all_time(db, boot=args.boot, players_file=args.players)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))
    print(f'{db}: {changed["matches"]} match folders updated\n')
    if not r['players']:
        print('No finished duplicate deals yet.')
        return
    print(history.table(r))
    print(f'\n{r["deals"]:,} duplicate deals from {len(r["runs"])} runs, one player per model and reasoning level.')
    for x in r['left_out']:
        print(f'Left out: {x["run"]} (other game settings)')
    if len(r['code_versions']) > 1:
        print(f'Played on {len(r["code_versions"])} code versions: {", ".join(r["code_versions"])}')
    if r['unversioned_runs']:
        print(f'From before code versions were recorded: {", ".join(r["unversioned_runs"])}')


def cmd_export(args):
    from .server import export
    n = export(args.root, args.out)
    print(f'wrote {args.out}: the viewer and {n} data files. Open it with any static file server, e.g.\n'
          f'  python -m http.server -d {args.out} 8000')


def cmd_status(args):
    """Progress of every run under a folder: hands played, matchups playing, pace and time left."""
    from .server import index, runs
    root = Path(args.root).resolve()
    for r in runs(root):
        if r.get('all') or (args.run and r['run'] != args.run):
            continue
        state = 'finished' if r['finished'] else 'playing' if r['live'] else 'paused'
        line = f"{r['run']}: {state}, {r['hands']:,} of {r['hands_planned']:,} hands, " \
               f"{r['matchups_finished']} of {r['matchups']} matchups done"
        if r['hands_per_hour']:
            line += f", {r['hands_per_hour']:,} hands an hour"
        if r['eta_seconds']:
            line += f", about {r['eta_seconds'] / 3600:.1f} h left" if r['eta_seconds'] >= 3600 \
                else f", about {max(1, round(r['eta_seconds'] / 60))} min left"
        print(line)
        for m in index(root, r['run']):
            if m['finished'] and not args.all:
                continue
            mark = 'playing' if m['live'] else 'not started' if not m['started'] else 'done' if m['finished'] else 'paused'
            net = f", {m['a']} {m['pair_net_a'] / m['bb']:+.1f} bb" if m['pairs_done'] else ''
            print(f"  {m['a']} vs {m['b']}: {m['pairs_done']}/{m['deals_total']} deals, {mark}{net}")
        if args.health:
            from .stats import collect
            base = root if r['run'] == '.' else root / r['run']
            for pid, s in sorted(collect(base).items()):
                if not s['calls']:
                    continue
                print(f"  {pid}: {s['decisions']} decisions, {s['latency_mean'] or 0:.1f}s a call (slowest "
                      f"{s['latency_max'] or 0:.0f}s), {s['retries']} retries, {s['fallbacks']} fallbacks, "
                      f"{s['corrected']} sizes corrected" + (f", answered by {', '.join(s['served_models'])}" if s['served_models'] else ''))


def cmd_doctor(args):
    """Checks each model player's backend: it answers, its JSON parses, and it remembers an earlier turn."""
    from .agents import parse_json
    from .backends import make_backend
    players = load_players(args.players)
    ids = args.ids or [p for p, s in players.items() if 'backend' in s]
    word = random.choice(['pelican', 'lantern', 'harbor', 'violet', 'compass', 'meadow'])

    async def check(pid):
        spec = dict(players[pid])
        spec.pop('label', None); spec.pop('description', None)
        kind = spec.pop('backend')
        for k in ('max_fixes', 'max_say_words', 'max_notes_words'):
            spec.pop(k, None)
        be = make_backend(kind, 'You are being tested. Always reply with only a JSON object, no other text.',
                          label=pid, retries=0, **spec)
        try:
            r1 = await be.send(f'Remember this secret word for later: {word}. Reply with {{"ok": true}}.')
            j1 = parse_json(r1.text)
            r2 = await be.send('What was the secret word I gave you? Reply with {"word": "..."}.')
            j2 = parse_json(r2.text) or {}
            remembered = word in str(j2.get('word', r2.text)).lower()
            return (f'{pid:24} OK  json={"yes" if j1 else "NO"}  memory={"yes" if remembered else "NO"}  '
                    f'latency={r1.latency:.0f}s/{r2.latency:.0f}s  reasoning-visible={"yes" if r2.thinking else "no"}'
                    + (f'  served-by={r2.meta["served_model"]}' if r2.meta.get('served_model') else ''))
        except Exception as e:  # report and carry on
            return f'{pid:24} FAIL {str(e)[:300]}'

    async def main():
        for line in await asyncio.gather(*(check(p) for p in ids)):
            print(line)
    asyncio.run(main())


def main(argv=None):
    ap = argparse.ArgumentParser(prog='poker_arena', description='LLM agents playing duplicate poker, rated on the Elo scale.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('play', help='one duplicate pair between two players')
    p.add_argument('a'); p.add_argument('b')
    p.add_argument('--deals', type=int, default=20)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--game', default='holdem')
    p.add_argument('--players', default='players.toml')
    p.add_argument('--out', default='runs/play')
    p.add_argument('--effort', help='reasoning level for every model (default: each player\'s own, else high)')
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser('plan', help='write a round-robin tournament plan')
    p.add_argument('players_list', nargs='+')
    p.add_argument('--deals', type=int, default=50)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--out', default='runs/pilot')
    p.add_argument('--players', default='players.toml')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--write', help='file to write the plan to (default: print it)')
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser('tournament', help='run (or resume) every matchup of a plan')
    p.add_argument('plan')
    p.add_argument('--only', nargs='*', help='only these players or matchups')
    p.add_argument('--concurrency', type=int, help='matches at once per CLI (overrides the plan)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--effort', help='reasoning level for every model, e.g. high (overrides players.toml and the plan)')
    p.set_defaults(fn=cmd_tournament)

    p = sub.add_parser('ratings', help='Elo and bb/100 from every finished duplicate deal under a folder')
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--boot', type=int, default=300)
    p.add_argument('--anchor', help='fix one player, e.g. station=1000 (default: players average 1500)')
    p.add_argument('--json', help='also write the ratings as JSON')
    p.set_defaults(fn=cmd_ratings)

    p = sub.add_parser('view', help='web viewer: leaderboard and hand replays with reasoning')
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--port', type=int, default=8800)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--players', default='players.toml', help='for display names (default: players.toml here)')
    p.set_defaults(fn=cmd_view)

    p = sub.add_parser('history', help='update the history database (runs/arena.db) and print the all-time leaderboard')
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--db', help='database file (default: arena.db in root)')
    p.add_argument('--boot', type=int, default=300)
    p.add_argument('--players', default='players.toml', help='for display names (default: players.toml here)')
    p.add_argument('--json', help='also write the leaderboard as JSON')
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser('export', help='write the viewer and a run\'s data as a static site you can host anywhere')
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--out', default='site')
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser('status', help='progress of each run: hands played, pace, time left')
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--run', help='only this run')
    p.add_argument('--all', action='store_true', help='list finished matchups too')
    p.add_argument('--health', action='store_true', help='per model: reply times, retries, fallbacks, which model answered')
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('doctor', help='check that each model player answers and remembers')
    p.add_argument('ids', nargs='*')
    p.add_argument('--players', default='players.toml')
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
