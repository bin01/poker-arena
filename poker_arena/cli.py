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
    from .ratings import ratings, summary
    print(summary(ratings(args.out, boot=100)))


def cmd_plan(args):
    design = {k: v for k, v in (('alpha', args.alpha), ('max_rounds', args.max_rounds), ('block', args.block)) if v is not None}
    plan = make_plan(args.players_list, deals=args.deals, seed=args.seed, out=args.out, players_file=args.players,
                     concurrency=args.concurrency, design=design, replicates=args.replicates)
    text = json.dumps(plan, indent=2)
    if args.write:
        Path(args.write).write_text(text + '\n')
        d, n = plan['design'], len(plan['matchups'])
        per_round = 2 * n * args.deals * plan['replicates']
        print(f'wrote {args.write}: {n} pairings, {per_round} hands per round, at most {d["max_rounds"]} rounds '
              f'({per_round * d["max_rounds"]} hands); a pairing stops once decided (alpha {d["alpha"]} over all pairings)')
    else:
        print(text)


def cmd_tournament(args):
    plan = json.loads(Path(args.plan).read_text())
    if args.effort:
        plan['effort'] = args.effort
    if args.until_decided:
        plan['until_decided'] = args.until_decided
    if args.dry_run:
        from .runner import plan_design
        d = plan_design(plan)
        reps = plan.get('replicates', 1)
        for a, b in plan['matchups']:
            more = (f', in rounds; stops once decided, at most {d["max_rounds"]} rounds '
                    f'({d["max_rounds"] * plan["deals"]} deals)') if d else ''
            print(f'{a} vs {b}: {reps} x 2 x {plan["deals"]} hands per round -> {plan["out"]}/{a}__vs__{b}{more}')
        return
    asyncio.run(run_plan(plan, only=args.only, concurrency=args.concurrency))
    from .ratings import ratings, summary
    print(summary(ratings(plan['out'])))


def cmd_ratings(args):
    from .ratings import ratings, summary
    r = ratings(args.root, boot=args.boot)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))
    if not r['players']:
        print('No finished duplicate deals yet.')
        return
    print(summary(r))
    print(f'\n{r["deals"]} duplicate deals ({r["hands"]} hands), {r["observations"]} distinct. Strength and verdicts: '
          f'{r["objective"]}; session Elo: share of {r["session_deals"]}-deal sessions won (secondary).')


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
    print(f'\n{r["deals"]:,} duplicate deals from {len(r["runs"])} runs ({r["observations"]:,} distinct), '
          f'one player per model and reasoning level.')
    if len(r['code_versions']) > 1:
        print(f'Code versions: {", ".join(r["code_versions"])}')


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
    ap = argparse.ArgumentParser(prog='poker_arena', description='LLM agents playing duplicate poker, rated by bb/100 over duplicate deals.')
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
    p.add_argument('--max-rounds', type=int, help='rounds of --deals a pairing may play before it stops undecided (default 6)')
    p.add_argument('--alpha', type=float, help='chance of naming any false winner across the whole tournament (default 0.05)')
    p.add_argument('--block', type=int, help='deals per block in the uncertainty estimate (default 10)')
    p.add_argument('--replicates', type=int, default=1, help='independent duplicate pairs per pairing per round, each with fresh memories')
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser('tournament', help='run (or resume) every matchup of a plan')
    p.add_argument('plan')
    p.add_argument('--only', nargs='*', help='only these players or matchups')
    p.add_argument('--concurrency', type=int, help='matches at once per CLI (overrides the plan)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--effort', help='reasoning level for every model, e.g. high (overrides players.toml and the plan)')
    p.add_argument('--until-decided', type=int, metavar='MAX_DEALS',
                   help='play the plan\'s deals as rounds, stopping each pairing once its edge is settled, '
                        'or at MAX_DEALS deals')
    p.set_defaults(fn=cmd_tournament)

    p = sub.add_parser('ratings', help='strength (bb/100), verdicts per pairing and session Elo from every finished '
                                       'duplicate deal under a folder')
    p.add_argument('root', nargs='?', default='runs')
    p.add_argument('--boot', type=int, default=300, help='bootstrap resamples (default 300)')
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
    from .runner import ResumeMismatch
    try:
        args.fn(args)
    except ResumeMismatch as e:
        sys.exit(f'Not resuming: {e}')


if __name__ == '__main__':
    main()
