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
    res = asyncio.run(run_duplicate(game, (args.a, players[args.a]), (args.b, players[args.b]), seeds, Path(args.out),
                                    allow_code_change=args.allow_code_change))
    for label, r in zip(('a-first', 'b-first'), res):
        print(label, dict(zip(r['players'], r['net'])))
    from .ratings import ratings, summary
    print(summary(ratings(args.out, boot=100)))


def _power_table(d, lengths=False) -> str:
    from .ratings import LEADER_SCENARIO, _pct, length_tradeoff
    rows = [f'  {x["scenario"]:12} ICC {x["assumed_icc"]:<5} AR(1) {x["assumed_ar1"]:<4} spread of a conversation '
            f'{x["sd_conversation_bb100"]:>5.1f} bb/100   decided {_pct(x["power"])}, after about '
            f'{x["conversations_used"]:,.0f} conversations' for x in d.get('power_scenarios') or []]
    out = ['Chance that the better player is decided at the target edge (the official test, simulated look by look as '
           'the runner stops), by how dependent the deals turn out to be; the error rate does not depend on it:'] + rows
    if d.get('power_by_edge'):
        out.append('  assumed dependence, by edge: ' + ', '.join(f'{e} bb/100 {_pct(p)}' for e, p in d['power_by_edge'].items()))
    for x in d.get('leader_chances') or []:
        out.append(f'  decided leader is the true best player {_pct(x["p_leader"])} if '
                   f'{LEADER_SCENARIO[x["scenario"]].format(e=x["edge_bb100"])} (pairings taken as independent)')
    if lengths:
        out.append('Deals per pairing the official test needs for 80% at the target, by conversation length (the bound '
                   'of the stacks, not the spread, sets the number of conversations, so short conversations cost far '
                   'fewer deals; long ones also measure what a model learns about its opponent within a match):')
        out += [f'  {x["round_deals"]:>3} deal{"s" if x["round_deals"] != 1 else " "} per conversation: about '
                + (f'{x["deals"]:,} deals ({x["conversations"]:,} conversations)' if x['deals'] else 'out of reach')
                for x in length_tradeoff(d)]
    return '\n'.join(out)


def _cost(plan: dict, d: dict) -> str:
    n = len(plan['matchups'])
    R, reps, rounds = d['round_deals'], d['replicates'], d['max_rounds']
    per_round = 2 * n * R * reps
    used = next((x['conversations_used'] for x in d.get('power_scenarios') or [] if x['scenario'] == 'moderate'), None)
    line = (f'{n} pairings x {reps} conversation{"s" if reps != 1 else ""} of {R} deal{"s" if R != 1 else ""} a round: '
            f'{per_round:,} hands a round, at most {rounds} rounds = {per_round * rounds:,} hands '
            f'({n * reps * rounds:,} fresh conversations of two matches)')
    if used:
        line += (f'; if one player of each pairing is truly {d["target_edge_bb100"]:.0f} bb/100 ahead, about '
                 f'{round(2 * n * R * min(used, reps * rounds)):,} hands before the pairings stop')
    return line


def _track_text(track: str, plan: dict, d: dict, args) -> list[str]:
    from .ratings import design_line
    out = []
    if track == 'adaptation':
        out.append(f'ADAPTATION track -> {plan["out"]} (models remember the session; model-based, exploratory, not '
                   f'official): ' + _cost(plan, d) + '.')
        out.append(design_line(d))
        out += [f'  {x["scenario"]:12} ICC {x["assumed_icc"]:<5} AR(1) {x["assumed_ar1"]:<4} model-based test leans toward '
                f'the stronger player {_pct_(x["power"])}, after about {x["conversations_used"]:,.0f} conversations'
                for x in d.get('power_scenarios') or []]
        if d.get('power_by_edge'):
            out.append('  assumed dependence, by edge: ' + ', '.join(f'{e} bb/100 {_pct_(p)}' for e, p in d['power_by_edge'].items()))
        out.append('  Its verdicts never decide the official leaderboard, ranks or leaders.')
        return out
    out.append(f'OFFICIAL track -> {plan["out"]} (fresh conversations; assumption-free verdicts, the only ones that '
               f'rank): ' + _cost(plan, d) + f'. At most {d["alpha"]:.0%} chance of any false verdict over all pairings, '
               f'for any distribution of results, given independent conversations (fresh memories) and a conditional '
               f'null (each conversation\'s expected edge, given the ones before, at most the null value): a betting '
               f'confidence sequence on conversation means bounded by the stacks (+-{d["outcome_bound_bb100"]:,.0f} '
               f'bb/100), valid at every look (docs/methodology.md).')
    out.append(design_line(d))
    out.append(_power_table(d, lengths=True))
    if d['underpowered']:
        if d.get('deals_needed'):
            out.append(f'To reach 80% with the assumed dependence, allow about {d["deals_needed"]:,} deals per '
                       f'pairing: drop --max-rounds/--budget-deals, or --max-rounds '
                       f'{-(-d["deals_needed"] // (d["round_deals"] * d["replicates"]))} with --replicates {d["replicates"]}.')
        else:
            out.append('80% at this target is out of reach of this plan\'s shape: shorten --deals (fewer deals per '
                       'conversation), raise --replicates or --target-edge, or play shallower stacks (--stack).')
    return out


def _pct_(x):
    from .ratings import _pct
    return _pct(x)


def cmd_plan(args):
    from .ratings import normalize_design
    from .runner import make_tournament
    shared = {k: v for k, v in (('alpha', args.alpha), ('block', args.block), ('target_edge_bb100', args.target_edge),
                                ('prior_sd_bb100', args.prior_sd), ('assumed_icc', args.assumed_icc),
                                ('assumed_ar1', args.assumed_ar1),
                                ('strict_identity', True if args.strict_identity else None)) if v is not None}
    per = {'official': {'max_rounds': args.max_rounds, 'budget_deals': args.budget_deals},
           'adaptation': {'max_rounds': args.adaptation_max_rounds, 'budget_deals': args.adaptation_budget_deals}}
    tracks = [t.strip() for t in args.tracks.split(',') if t.strip()]
    try:
        doc = make_tournament(args.players_list, out=args.out, tracks=tracks, seed=args.seed, players_file=args.players,
                              concurrency=args.concurrency, game_config={'stack': args.stack} if args.stack else None,
                              official={'deals': args.deals, 'replicates': args.replicates,
                                        'design': {**shared, **{k: v for k, v in per['official'].items() if v is not None}}},
                              adaptation={'deals': args.adaptation_deals, 'replicates': args.adaptation_replicates,
                                          'design': {**shared, **{k: v for k, v in per['adaptation'].items() if v is not None}}})
    except ValueError as e:
        raise SystemExit(f'plan: {e}')
    designs = {}
    for track, plan in doc['tracks'].items():
        if per[track]['max_rounds'] is None:
            plan['design'].pop('max_rounds', None)      # from the power target, not a fixed default
        d = designs[track] = normalize_design(plan)
        plan['design'].update(max_rounds=d['max_rounds'], max_rounds_basis=d['max_rounds_basis'])   # preregistered
    text = json.dumps(doc, indent=2)
    if not args.write:
        print(text)
        return
    Path(args.write).parent.mkdir(parents=True, exist_ok=True)
    Path(args.write).write_text(text + '\n')
    print(f'wrote {args.write}: tournament {doc["tournament"]}, {len(doc["tracks"])} track'
          f'{"s" if len(doc["tracks"]) != 1 else ""} ({", ".join(doc["tracks"])}); every pairing stops once decided or '
          f'ineligible. Each track is registered in its own plan.json when it starts, before its first hand; change '
          f'the plan after that and nothing in that track is a test.')
    for track, plan in doc['tracks'].items():
        print()
        print('\n'.join(_track_text(track, plan, designs[track], args)))
    try:
        specs = load_players(args.players)
    except (OSError, ValueError):
        specs = {}
    if not any(specs.get(p, {}).get('bot') == 'reference' for p in args.players_list):
        print('\nNo reference player in the plan: add the fixed reference bot (bot = "reference" in players.toml) so '
              'every model also plays one common, versioned opponent; without it strengths only compare the '
              'models with each other. `python -m poker_arena quiz` measures poker fundamentals separately.')


def _prepare(plan: dict, args) -> dict:
    plan = dict(plan)
    if args.effort:
        plan['effort'] = args.effort
    if args.until_decided:
        design = dict(plan.get('design') or {'round_deals': plan['deals']})
        reps = int(plan.get('replicates') or design.get('replicates') or 1)
        design.update(max_rounds=max(1, args.until_decided // (plan['deals'] * reps)), max_rounds_basis='given')
        plan['design'] = design
    return plan


def _dry_run(track: str, plan: dict):
    from .ratings import design_line, normalize_design
    d = normalize_design(plan)
    reps = d['replicates'] if d else plan.get('replicates', 1)
    print(f'{track} track:' + (' (model-based, not official)' if track == 'adaptation' else ''))
    for a, b in plan['matchups']:
        more = (f', in rounds; stops once decided, at most {d["max_rounds"]} rounds '
                f'({d["max_deals"]} deals)') if d else ''
        print(f'  {a} vs {b}: {reps} x 2 x {plan["deals"]} hands per round -> {plan["out"]}/{a}__vs__{b}{more}')
    if d:
        print(design_line(d))
        if track != 'adaptation':
            print(_power_table(d))
    else:
        print('No design in this plan: one round, no stopping rule, and no result of it is a test.')


def cmd_tournament(args):
    from .runner import plan_tracks
    doc = json.loads(Path(args.plan).read_text())
    plans = {t: _prepare(p, args) for t, p in plan_tracks(doc, args.track).items()}
    if args.dry_run:
        for track, plan in plans.items():
            _dry_run(track, plan)
        return

    async def main():
        return await asyncio.gather(*(run_plan(p, only=args.only, concurrency=args.concurrency,
                                               allow_code_change=args.allow_code_change) for p in plans.values()))
    asyncio.run(main())
    from .ratings import ratings, summary
    for track, plan in plans.items():
        if len(plans) > 1:
            print(f'\n== {track} track ({plan["out"]}) ==')
        print(summary(ratings(plan['out'])))


def cmd_ratings(args):
    from .ratings import track_runs
    tracks = track_runs(args.root)
    if tracks:
        for i, (track, base) in enumerate(tracks.items()):
            print(f'{"" if not i else chr(10)}== {track} track ({base}) ==')
            _ratings(base, args, f'{args.json[:-5] if args.json and args.json.endswith(".json") else args.json}-{track}.json'
                     if args.json else None)
        return
    _ratings(args.root, args, args.json)


def _ratings(root, args, json_path):
    from .ratings import ratings, summary
    r = ratings(root, boot=args.boot)
    if json_path:
        Path(json_path).write_text(json.dumps(r, indent=2))
    if not r['players']:
        print('No finished duplicate deals yet.')
        return
    print(summary(r))
    reg = r.get('registration') or {}
    if r.get('track') == 'adaptation' and (r.get('design') or {}).get('registered'):
        test = (f'the model-based gate (adaptation track: stated assumptions, never official), recorded when play '
                f'stopped by the registered judge {reg.get("judge_fingerprint")}')
    elif (r.get('design') or {}).get('registered'):
        test = (f'the official decisions of the preregistered, assumption-free sequential test on {r["objective"]}, recorded when '
                f'play stopped by the registered judge {reg.get("judge_fingerprint")} (this judge: '
                f'{r.get("judge_fingerprint")}; anything it recomputes differently is retrospective), each counting '
                f'only while its evidence still verifies today')
    else:
        why = '; '.join(reg.get('reasons') or []) or 'no preregistered design'
        test = f'none (not preregistered: {why}; everything is descriptive)'
    print(f'\n{r["deals"]} duplicate deals ({r["hands"]} hands), {r["observations"]} distinct. Verdicts: {test}; '
          f'strengths: descriptive; session Elo: share of sessions won, where '
          f'{(r.get("session") or {}).get("text") or str(r["session_deals"]) + " deals make a session"} (secondary, '
          f'descriptive).')
    if r.get('rejected'):
        print(f'{len(r["rejected"])} duplicate pairs failed validation and were left out (see the notes).')


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
    print(f'\n{r["deals"]:,} duplicate deals from {len(r["runs"])} runs ({r["observations"]:,} distinct, every hand '
          f'replayed from its seed), one player per model, reasoning level, protocol and implementation (code that '
          f'plays and CLI version, split at the hand a change took effect; equivalences.json beside the database can '
          f'merge implementations). Descriptive only: pooled runs are no one\'s preregistered test.')
    if len(r['code_versions']) > 1:
        print(f'Code versions: {", ".join(r["code_versions"])}')
    if r.get('other_track_runs'):
        x = history.all_time(db, boot=args.boot, players_file=args.players, track='adaptation')
        if x['players']:
            print(f'\n== Adaptation track, all time (models remember the session; model-based, not official; never '
                  f'pooled with the official track) ==\n{history.table(x)}')


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
    p.add_argument('--allow-code-change', action='store_true',
                   help='resume a run started by other code (the change is recorded in the run)')
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser('plan', help='write a round-robin tournament plan')
    p.add_argument('players_list', nargs='+')
    p.add_argument('--deals', type=int, default=1,
                   help='duplicate deals per conversation on the official track: must be 1 (fresh conversations every '
                        'deal); long conversations, where models adapt, are the adaptation track (--adaptation-deals)')
    p.add_argument('--stack', type=int, help='chips per stack (default 200: 100 big blinds); the official test\'s bound, '
                                             'and so the conversations it needs, scale with it')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--out', default='runs/pilot')
    p.add_argument('--players', default='players.toml')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--write', help='file to write the plan to (default: print it)')
    p.add_argument('--max-rounds', type=int, help='rounds of --deals a pairing may play before it stops undecided '
                                                  '(default: the fewest that give 80%% power at the target edge under '
                                                  'the assumed dependence)')
    p.add_argument('--budget-deals', type=int, help='cap on deals per pairing; a plan it cuts short is marked underpowered')
    p.add_argument('--prior-sd', type=float, help='per-deal spread in bb/100 assumed when planning (default 350)')
    p.add_argument('--assumed-icc', type=float, help='share of the per-deal variance that is a conversation effect, '
                                                     'for planning (default 0.01)')
    p.add_argument('--assumed-ar1', type=float, help='memory between consecutive deals of a conversation, AR(1), for '
                                                     'planning (default 0.2)')
    p.add_argument('--alpha', type=float, help='chance of naming any false winner across the whole tournament (default 0.05)')
    p.add_argument('--block', type=int, help='deals per block in the within-conversation intervals (default 10)')
    p.add_argument('--replicates', type=int, default=25,
                   help='official track: independent duplicate pairs (conversations) per pairing per round, each with fresh memories '
                        '(default 25; the official test is valid at every look, so this sets only how often it looks)')
    p.add_argument('--target-edge', type=float, help='the edge in bb/100 the plan is powered to detect (default 50)')
    p.add_argument('--tracks', default='official,adaptation',
                   help='which tracks to plan (default both): "official" (fresh conversations, assumption-free '
                        'verdicts that decide leaders and ranks) and "adaptation" (the same players in long '
                        'conversations, so models remember and adapt; model-based verdicts, never official). Each '
                        'gets its own run folder, <out>/<track>')
    p.add_argument('--adaptation-deals', type=int, default=50, help='deals per conversation on the adaptation track (default 50)')
    p.add_argument('--adaptation-replicates', type=int, default=4,
                   help='conversations per pairing per round on the adaptation track (default 4)')
    p.add_argument('--adaptation-max-rounds', type=int, help='rounds at most on the adaptation track (default: the '
                                                             'fewest that give its model-based gate 80%% power)')
    p.add_argument('--strict-identity', action='store_true',
                   help='strict publication policy, stored in the design: a pairing with an unattested player (a backend '
                        'that does not report which model answered, e.g. codex) gets no counted official result; by '
                        'default it counts, marked identity_qualified')
    p.add_argument('--adaptation-budget-deals', type=int, help='cap on deals per pairing on the adaptation track')
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser('tournament', help='run (or resume) every matchup of a plan, in each of its tracks')
    p.add_argument('plan')
    p.add_argument('--track', action='append', choices=['official', 'adaptation'],
                   help='only this track (repeat for both; default: every track in the plan, played at the same time, '
                        'each with its own --concurrency)')
    p.add_argument('--only', nargs='*', help='only these players or matchups')
    p.add_argument('--concurrency', type=int, help='matches at once per CLI (overrides the plan)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--effort', help='reasoning level for every model, e.g. high (overrides players.toml and the plan)')
    p.add_argument('--until-decided', type=int, metavar='MAX_DEALS',
                   help='set the plan\'s max_rounds so that a pairing plays at most MAX_DEALS deals (it stops '
                        'earlier once decided)')
    p.add_argument('--allow-code-change', action='store_true',
                   help='resume a run started by other code (the change is recorded in the run)')
    p.set_defaults(fn=cmd_tournament)

    p = sub.add_parser('ratings', help='verdicts per pairing (the assumption-free sequential test), descriptive strengths (bb/100) and '
                                       'session Elo from every finished duplicate deal under a folder')
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

    p = sub.add_parser('quiz', help='poker fundamentals: decision spots with computed answers, in separately scored '
                                    'categories (basics, close, multistreet, inference, sizing), asked of each player '
                                    'through its backend; writes runs/<out>/quiz.json')
    p.add_argument('ids', nargs='+')
    p.add_argument('--out', default='quiz')
    p.add_argument('--players', default='players.toml')
    p.add_argument('--effort', help='reasoning level for every model (default: each player\'s own, else high)')
    p.add_argument('--category', action='append', choices=['basics', 'close', 'multistreet', 'inference', 'sizing'],
                   help='only this category (repeat for more; default: all)')
    p.add_argument('--limit', type=int, help='only the first N spots of each category')
    p.add_argument('--concurrency', type=int, default=4, help='spots asked at once per player (default 4)')
    p.add_argument('--allow-unattested', action='append', metavar='BACKEND',
                   help='a backend whose replies name no model may still be eligible (repeat for more; the same '
                        'policy as a match design\'s allow_unattested)')
    p.set_defaults(fn=lambda a: __import__('poker_arena.quiz', fromlist=['main']).main(a))

    args = ap.parse_args(argv)
    from .runner import ResumeMismatch
    try:
        args.fn(args)
    except ResumeMismatch as e:
        sys.exit(f'Not resuming: {e}')


if __name__ == '__main__':
    main()
