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
    players = load_players(args.players)
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
    serve(args.root, args.port, args.host)


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
                    f'latency={r1.latency:.0f}s/{r2.latency:.0f}s  reasoning-visible={"yes" if r2.thinking else "no"}')
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
    p.set_defaults(fn=cmd_view)

    p = sub.add_parser('doctor', help='check that each model player answers and remembers')
    p.add_argument('ids', nargs='*')
    p.add_argument('--players', default='players.toml')
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
