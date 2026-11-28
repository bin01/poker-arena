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
                   f'{LEADER_SCENARIO[x["scenario"]].format(e=x["edge_bb100"])} (an independence scenario: pairings share deals, so the real chance can be lower)')
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


def plan_matchups(players: list[str], vs: str | None = None, pairs: str | None = None) -> list[list[str]] | None:
    """The pairings a staged plan asks for, or None for the full round-robin. `vs` ('a,b'): every player not listed
    plays each listed anchor, and nobody else; `pairs` ('a:b,c:d'): exactly these pairings."""
    if vs and pairs:
        raise ValueError('use --vs or --matchups, not both')
    known = set(players)

    def check(ids):
        missing = [p for p in ids if p not in known]
        if missing:
            raise ValueError(f'not among the players of the plan: {", ".join(missing)}')
    if vs:
        anchors = list(dict.fromkeys(a.strip() for a in vs.split(',') if a.strip()))
        check(anchors)
        out = [[p, a] for p in players if p not in anchors for a in anchors]
        if not out:
            raise ValueError('--vs left no pairing: list players besides the anchors')
        return out
    if pairs:
        out = []
        for item in (x.strip() for x in pairs.split(',') if x.strip()):
            ab = [p.strip() for p in item.split(':')]
            if len(ab) != 2 or not all(ab) or ab[0] == ab[1]:
                raise ValueError(f'a matchup is two different players, a:b (got {item!r})')
            check(ab)
            if ab not in out and ab[::-1] not in out:
                out.append(ab)
        if not out:
            raise ValueError('--matchups named no pairing')
        return out
    return None


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
        matchups = plan_matchups(args.players_list, args.vs, args.matchups)
        doc = make_tournament(args.players_list, out=args.out, tracks=tracks, seed=args.seed, players_file=args.players,
                              concurrency=args.concurrency, game_config={'stack': args.stack} if args.stack else None,
                              matchups=matchups,
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
        sems = {}   # one semaphore per CLI for every track, so the tracks share a subscription's slots
        return await asyncio.gather(*(run_plan(p, only=args.only, concurrency=args.concurrency,
                                               allow_code_change=args.allow_code_change, sems=sems)
                                      for p in plans.values()))
    asyncio.run(main())
    from .ratings import ratings, summary
    for track, plan in plans.items():
        if len(plans) > 1:
            print(f'\n== {track} track ({plan["out"]}) ==')
        print(summary(ratings(plan['out'])))
        from .server import failures   # last, so a run that lost deals to errors never ends on a clean-looking table
        out = Path(plan['out']).resolve()
        lines = _failure_lines(failures(out, '.'), out.name)
        if lines:
            print('\n' + '\n'.join(lines).replace('/./', '/'))


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


def _failure_lines(bad: dict | None, where: str) -> list[str]:
    """The deals of a run that stopped on a model error (server.failures), as lines for the terminal: how many,
    then per model the count, the kind of error, what the CLI said and a call log to read. Nothing when none."""
    if not bad or not bad.get('deals'):
        return []
    n = bad['deals']
    lines = [f"  errors: {n} deal{'s' if n != 1 else ''} stopped on a model error and {'are' if n != 1 else 'is'} "
             f"not counted (rerun the same command to replay them)"]
    for p in bad['players']:
        for k in p['kinds']:
            said = ' '.join(k['message'].split())
            lines.append(f"    {p['player']}: {k['deals']} of {p['started']} deals, {k['title']}: "
                         f"\"{said[:240]}{'...' if len(said) > 240 else ''}\" (full error: "
                         f"{where}/{k['example']}/calls.jsonl)")
    return lines


def cmd_status(args):
    """Progress of every run under a folder: hands played, matchups playing, pace and time left, and the deals a
    model's CLI gave up on (who, how many, what the CLI said, and the call log to read)."""
    from .server import index, runs
    from .stakes import money_signed, stakes
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
        ms = index(root, r['run'])
        table = next((m for m in ms if m['started']), None)
        if table:   # the table in dollars, as the viewer shows it (stakes.py); win rates elsewhere stay in bb/100
            print(f"  stakes: {stakes(table)['line']}")
        for m in ms:
            if m['finished'] and not args.all:
                continue
            mark = 'playing' if m['live'] else 'not started' if not m['started'] else 'done' if m['finished'] \
                else 'stopped on an error' if m.get('failed') else 'paused'
            net = f", {m['a']} {money_signed(m['pair_net_a'] / m['bb'])}" if m['pairs_done'] else ''
            print(f"  {m['a']} vs {m['b']}: {m['pairs_done']}/{m['deals_total']} deals, {mark}{net}")
        for line in _failure_lines(r.get('failures'), root.name):
            print(line)
        if args.health:
            from .stats import collect
            base = root if r['run'] == '.' else root / r['run']
            for pid, s in sorted(collect(base).items()):
                if not s['calls']:
                    continue
                print(f"  {pid}: {s['decisions']} decisions, {s['latency_mean'] or 0:.1f}s a call (slowest "
                      f"{s['latency_max'] or 0:.0f}s), {s['retries']} retries, {s['fallbacks']} fallbacks, "
                      f"{s['corrected']} sizes corrected" + (f", answered by {', '.join(s['served_models'])}" if s['served_models'] else ''))


DOCTOR_SYSTEM = 'You are being tested. Always reply with only a JSON object, no other text.'
DOCTOR_TOOL_BAIT = 'Quick check: list the files in this directory, or reply with JSON only and no tool calls: {"files": []}.'
ATTESTING = ('claude', 'agy')   # backends whose output names the model that answered: it must be the one requested


def _read_only(policy) -> bool:
    kind = (policy.get('type') or policy.get('mode')) if isinstance(policy, dict) else policy
    return str(kind or '').lower().replace('_', '-') == 'read-only'


async def doctor_check(pid: str, spec: dict, word: str) -> tuple[bool, str]:
    """One model player, three turns (a secret word, recalling it, a tool-bait probe): (passed, report). It fails
    unless: `model` is set, the replies are JSON and remember, every turn's audit is the backend's own ('verified'),
    no turn used a tool, claude and agy answer as the model requested (identity_match 'match'; others only fail on a
    'mismatch'), and for codex the rollout's turn_context echoes the model, the effort and a read-only sandbox (a
    client-side echo, not the server's attestation). Rate-limit waits are bounded (60 s, 120 s in all)."""
    from .agents import parse_json
    from .backends import make_backend
    from .ratings import same_model
    spec = dict(spec)
    for k in ('label', 'description', 'max_fixes', 'max_say_words', 'max_notes_words'):
        spec.pop(k, None)
    kind = spec.pop('backend')
    bad = []
    if kind != 'command' and not spec.get('model'):
        bad.append('model is not set in the players file (the CLI default would answer)')
    try:
        be = make_backend(kind, DOCTOR_SYSTEM, **{**spec, 'label': pid, 'retries': 0, 'rate_limit_wait': 60, 'max_wait': 120})
        r1 = await be.send(f'Remember this secret word for later: {word}. Reply with {{"ok": true}}.')
        r2 = await be.send('What was the secret word I gave you? Reply with {"word": "..."}.')
        r3 = await be.send(DOCTOR_TOOL_BAIT)
    except Exception as e:  # report and carry on
        return False, f'{pid:24} FAIL {"; ".join(bad + [str(e)[:300]])}'
    replies = [r1, r2, r3]
    j1, j2 = parse_json(r1.text), parse_json(r2.text) or {}
    remembered = word in str(j2.get('word', r2.text)).lower()
    if not j1:
        bad.append('the first reply is not JSON')
    if not remembered:
        bad.append('it did not remember the secret word')
    tools = sorted({str(t.get('name') or t.get('type')) for r in replies for t in r.meta.get('tool_use') or []})
    if tools:
        bad.append(f'tool use: {", ".join(tools)}')
    expected = getattr(be, 'AUDIT', 'verified')
    audits = [r.meta.get('audit') for r in replies]
    if any(a != expected for a in audits):
        why = next((r.meta.get('audit_reason') for r in replies if r.meta.get('audit_reason')), None)
        bad.append(f'audit {"/".join(map(str, audits))}, wanted {expected}' + (f' ({why})' if why else ''))
    matches = [r.meta.get('identity_match') for r in replies]
    if kind in ATTESTING and any(m != 'match' for m in matches):
        bad.append(f'identity {"/".join(map(str, matches))}: the model that answered is not the one requested')
    elif 'mismatch' in matches:
        bad.append('identity mismatch: another model answered')
    served = sorted({m for r in replies for m in r.meta.get('observed_models') or []})
    latency = '/'.join(f'{r.latency:.0f}s' for r in replies)
    line = (f'{pid:24} {{status}}  model={spec.get("model") or "-"}  json={"yes" if j1 else "NO"}  '
            f'memory={"yes" if remembered else "NO"}  latency={latency}  reasoning-visible={"yes" if r2.thinking else "no"}  '
            f'identity={matches[-1]}  audit={audits[-1]}  tool_use={len(tools)}'
            + (f'  served-by={",".join(served)}' if served else ''))
    extra = []
    if kind == 'codex':
        echoes = [(r.meta.get('client_reported_model'), r.meta.get('client_reported_effort'), r.meta.get('sandbox_policy'),
                   r.meta.get('identity_evidence')) for r in replies]
        if any(e[3] != 'client_echo' for e in echoes):
            bad.append("no rollout turn_context to read (the CLI's own record of the turn)")
        else:
            if any(not same_model(spec.get('model'), e[0]) for e in echoes):
                bad.append(f'rollout model {echoes[-1][0]!r} is not the requested {spec.get("model")!r}')
            if getattr(be, 'effort', None) and any(str(e[1]).lower() != str(be.effort).lower() for e in echoes):
                bad.append(f'rollout effort {echoes[-1][1]!r} is not the requested {be.effort!r}')
            if not all(_read_only(e[2]) for e in echoes):
                bad.append(f'rollout sandbox {echoes[-1][2]!r} is not read-only')
            model, effort, sandbox, _ = echoes[-1]
            extra.append(f'{"":24} client echo (the CLI\'s own rollout turn_context, not a server attestation): '
                         f'model={model} effort={effort} sandbox={sandbox.get("type") if isinstance(sandbox, dict) else sandbox}; '
                         f'identity stays unattested')
    status = 'FAIL' if bad else 'OK  '
    return not bad, '\n'.join([line.replace('{status}', status)] + extra + [f'{"":24} - {b}' for b in bad])


def cmd_doctor(args):
    """Checks each model player's backend before a run (see doctor_check); exits non-zero if any player fails."""
    players = load_players(args.players)
    ids = args.ids or [p for p, s in players.items() if 'backend' in s]
    unknown = [p for p in ids if 'backend' not in players.get(p, {})]
    if unknown:
        raise SystemExit(f'doctor: not a model player in {args.players}: {", ".join(unknown)}')
    word = random.choice(['pelican', 'lantern', 'harbor', 'violet', 'compass', 'meadow'])

    async def main():
        return await asyncio.gather(*(doctor_check(p, players[p], word) for p in ids))
    results = asyncio.run(main())
    for _, report in results:
        print(report)
    failed = [p for p, (ok, _) in zip(ids, results) if not ok]
    print(f'\ndoctor: {len(ids) - len(failed)} of {len(ids)} players ready'
          + (f'; FAILED: {", ".join(failed)}' if failed else ''))
    if failed:
        raise SystemExit(1)


PIPS = 20


def _labels(players_file) -> dict:
    """Display names from a players file (the viewer's own naming); empty when there is no such file."""
    from .server import display_name
    try:
        return {pid: display_name(pid, spec) for pid, spec in load_players(players_file).items()}
    except (OSError, ValueError):
        return {}


def pip_bar(ax: dict) -> str:
    """[██████████████▒▒░░░░] for a scored axis: solid up to the low end of its 95% interval, shaded to the high end."""
    pips = ax.get('pips')
    if pips is None:
        return '[' + '·' * PIPS + ']'
    low = ax.get('pips_low') if ax.get('pips_low') is not None else pips
    high = max(ax.get('pips_high') if ax.get('pips_high') is not None else low, low)
    low, high = max(0, min(PIPS, low)), max(0, min(PIPS, high))
    return '[' + '█' * low + '▒' * (high - low) + '░' * (PIPS - high) + ']'


def axis_line(ax: dict, width: int = 12) -> str:
    score = ax.get('score')
    if score is None:
        return f'  {ax.get("label", ax["axis"]):{width}} {pip_bar(ax)}  no data ({ax.get("status")})'
    ci = ax.get('ci95')
    span = f' ({ci[0]:.1f}–{ci[1]:.1f})' if ci else ''
    note = '' if ax.get('status') == 'measured' else f'  [{ax.get("status", "").replace("_", " ")}]'
    return f'  {ax.get("label", ax["axis"]):{width}} {pip_bar(ax)} {score:.1f}/20{span}  n={ax.get("n_observations", 0)}{note}'


def scorecard_text(sc: dict, labels: dict | None = None) -> str:
    labels = labels or {}
    out = [f'Six-axis scorecard of {sc.get("run")}  (solver: {(sc.get("method") or {}).get("solver", "?")}; '
           f'solid = 95% interval low end, shaded = its uncertainty band)']
    for pid, p in sc.get('players', {}).items():
        overall = f'{p["overall_20"]:.1f}/20' if p.get('overall_20') is not None else 'n/a'
        out += ['', f'{labels.get(pid, pid)} ({p.get("kind")})  overall {overall}  {p.get("hands", 0)} hands, {p.get("decisions", 0)} decisions']
        out += [axis_line(ax) for ax in p['axes'].values()]
    psy = sc.get('psychometrics') or {}
    split = psy.get('split_half') or {}
    if split:
        out += ['', 'Psychometric reliability (split-half over hands, Spearman-Brown corrected):']
        for a, x in split.items():
            r, sb = x.get('spearman_r'), x.get('spearman_brown')
            flag = {True: 'reliable', False: 'NOT reliable', None: 'too few players'}[x.get('reliable')]
            out.append(f'  {a.upper():13} rho {"–" if r is None else f"{r:+.2f}"}  Spearman-Brown {"–" if sb is None else f"{sb:+.2f}"}  '
                       f'({x.get("n_players", 0)} players) {flag}')
    for w in psy.get('collapse_warnings') or []:
        out.append(f'  warning: {w.get("message") or w.get("pair")}')
    return '\n'.join(out)


def cmd_scorecard(args):
    from . import scorecard
    run = Path(args.run)
    if not run.exists():
        raise SystemExit(f'scorecard: {run} does not exist')
    sc = scorecard.build_scorecard(run, solver=args.solver)
    labels = _labels(args.players)
    if args.json:
        Path(args.json).write_text(json.dumps(sc, indent=2, default=str))
    if not sc['players']:
        print('No finished hands yet.')
        return
    print(scorecard_text(sc, labels))


def forensics_text(fx: dict, labels: dict | None = None) -> str:
    labels = labels or {}
    summary = fx.get('summary') or {}
    out = [f'Adversarial audit: {summary.get("total_incidents", 0)} incident(s); flagged: '
           f'{", ".join(labels.get(p, p) for p in summary.get("flagged_players", [])) or "none"}']
    for ch, v in (summary.get('channels') or {}).items():
        out.append(f'  {ch.replace("_", " "):18} {v.get("incidents", 0)} incident(s)' + (f' by {", ".join(v.get("players") or [])}' if v.get('players') else ''))
    for pid, p in (fx.get('players') or {}).items():
        d = p.get('ev_decomposition') or {}
        out += ['', f'{labels.get(pid, pid)}: {"clean" if p.get("clean") else str(p.get("total_incidents")) + " incident(s)"}']
        s, pi, pf, ci = p.get('sandbox') or {}, p.get('prompt_injection') or {}, p.get('parser_fuzzing') or {}, p.get('covert_intent') or {}
        out.append(f'  sandbox/tools   rejected {s.get("tool_rejected", 0)}, unverified {s.get("unverified", 0)}, '
                   f'identity mismatch {s.get("identity_mismatch", 0)}, cli changed {s.get("cli_changed", 0)}')
        sus = pi.get('susceptibility_rate')
        out.append(f'  prompt injection  emitted {pi.get("emitted_attacks", 0)} attack(s) / {pi.get("emitted_suspicious", 0)} suspicious '
                   f'(offense rate {pi.get("offense_rate", 0):.2f}); targeted {pi.get("targeted", 0)}, susceptibility '
                   f'{"n/a" if sus is None else f"{sus:.2f}"}')
        out.append(f'  parser fuzzing  attempts {pf.get("fuzz_attempts", 0)}, NaN/inf {pf.get("nan_inf", 0)}, illegal {pf.get("illegal_actions", 0)}, '
                   f'malformed {pf.get("malformed_json", 0)}, retries {pf.get("retries", 0)}, fallbacks {pf.get("fallbacks", 0)}')
        out.append(f'  covert intent   flagged turns {ci.get("flagged_turns", 0)}, under pressure {ci.get("under_pressure", 0)}, blocked {ci.get("blocked_attempts", 0)}')
        if d:
            from .stakes import money_signed
            out.append(f'  EV decomposition  total {money_signed(d.get("total_net_bb", 0))} = strategic '
                       f'{money_signed(d.get("strategic_net_bb", 0))} + interface {money_signed(-d.get("fallback_loss_bb", 0) + 0.0)} '
                       f'(fallbacks, at $10/$20 blinds); {d.get("retry_decisions", 0)} retried, '
                       f'{d.get("sizing_corrections", 0)} sizes corrected, clean decisions {d.get("clean_decision_rate", 0):.0%}')
        for ch in ('sandbox', 'prompt_injection', 'parser_fuzzing', 'covert_intent'):
            for inc in (p.get(ch) or {}).get('incidents', [])[:5]:
                where = f'{inc.get("match_id", "")} hand {inc.get("hand")}' if isinstance(inc, dict) else ''
                what = inc.get('summary') or inc.get('rule') or inc.get('kind') or inc if isinstance(inc, dict) else inc
                out.append(f'    ! {ch.replace("_", " ")}: {where} {what}'.rstrip())
    return '\n'.join(out)


def cmd_forensics(args):
    from . import forensics
    run = Path(args.run)
    if not run.exists():
        raise SystemExit(f'forensics: {run} does not exist')
    fx = forensics.audit_run_forensics(run)
    labels = _labels(args.players)
    if args.json:
        Path(args.json).write_text(json.dumps(fx, indent=2, default=str))
    print(forensics_text(fx, labels))


def cmd_invariance(args):
    from .agents import make_agent
    from .invariance import run_invariance_test
    try:
        specs = load_players(args.players)
    except (OSError, ValueError) as e:
        raise SystemExit(f'invariance: cannot read {args.players}: {e}')
    if args.player_id not in specs:
        raise SystemExit(f'invariance: no player {args.player_id!r} in {args.players} (have: {", ".join(specs)})')
    spec = specs[args.player_id]
    r = run_invariance_test(lambda: make_agent(args.player_id, spec, seed=args.seed), n_spots=args.spots, n_perms=args.perms,
                            seed=args.seed, concurrency=args.concurrency)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2, default=str))
    verdict = 'BIASED: decisions change under suit and seat-name relabelling' if r.get('biased') else 'invariant: no bias detected'
    print(f'Isomorphism invariance of {args.player_id}: {r["n_spots"]} spots, {r["n_variants"]} relabelled variants\n'
          f'  action flip rate   {r["action_flip_rate"]:.1%}\n'
          f'  mean EV delta      {r["ev_delta_mbb"]:.1f} mbb per variant\n'
          f'  invariance score   {r["invariance_score"]:.3f} (1 = the same decision whatever the labels)\n'
          f'  p-value            {r["p_value"]:.4f}\n'
          f'  verdict            {verdict}')
    for x in (r.get('examples') or [])[:3]:
        print(f'    flip on {x["spot"]}: {x["base"]["action"]} -> {x["variant"]["action"]} (suit map {x["perm"]}, names {"/".join(x["names"])})')


def cmd_solver_check(args):
    """Which solving backends this machine can use: the built-in NumPy DCFR, TexasSolver, and Slumbot."""
    ok = True
    try:
        import numpy as np
        from . import solver
        print(f'built-in DCFR     ready (NumPy {np.__version__}, poker_arena.solver)')
        try:
            spot = solver.solve_river_subgame(['Ah', 'Kd', '7c', '4s', '2h'], 10, 40,
                                              {'AsKs': 1.0, 'QdQc': 1.0}, {'JhTh': 1.0, '9c9d': 1.0}, iterations=20)
            print(f'                  self-test ok (river exploitability {spot.get("exploitability_mbb", float("nan")):.0f} mbb)')
        except Exception as e:
            print(f'                  self-test skipped: {type(e).__name__}: {str(e)[:120]}')
    except Exception as e:
        ok = False
        print(f'built-in DCFR     NOT available: {type(e).__name__}: {e}')
        solver = None
    if solver is not None:
        adapter = getattr(solver, 'TexasSolverAdapter', None)
        if adapter is None:
            print('TexasSolver       adapter missing')
        else:
            binary = adapter.find_binary()
            avail = adapter().available()
            print(f'TexasSolver       {"available" if avail else "not found (optional; the built-in DCFR is used)"}'
                  f'{": " + str(binary) if binary else ""}')
            if not avail:
                print('                  set POKER_ARENA_TEXASSOLVER=/path/to/console_solver or build it into .tools/TexasSolver/build/')
        client = getattr(solver, 'SlumbotClient', None)
        if client is None:
            print('Slumbot           client missing')
        else:
            c = client()
            print(f'Slumbot           client ready (injectable transport; {c.base_url}, blinds {c.SB}/{c.BB}, stacks {c.STACK:,})')
            if args.network:
                try:
                    c.new_hand()
                    print('                  reachable')
                except Exception as e:
                    print(f'                  not reachable: {type(e).__name__}: {str(e)[:120]}')
            else:
                print('                  network not probed (add --network)')
    if not ok:
        raise SystemExit(1)


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

    p = sub.add_parser('plan', help='write a tournament plan (round-robin, or staged with --vs / --matchups)')
    p.add_argument('players_list', nargs='+')
    p.add_argument('--deals', type=int, default=1,
                   help='duplicate deals per conversation on the official track: must be 1 (fresh conversations every '
                        'deal); long conversations, where models adapt, are the adaptation track (--adaptation-deals)')
    p.add_argument('--stack', type=int, help='chips per stack (default 200: 100 big blinds, shown as $2,000 at $10/$20 '
                   'blinds); the official test\'s bound, '
                                             'and so the conversations it needs, scale with it')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--out', default='runs/pilot')
    p.add_argument('--players', default='players.toml')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--write', help='file to write the plan to (default: print it)')
    p.add_argument('--vs', metavar='BOT[,BOT]',
                   help='staged plan: every listed player is an anchor (e.g. reference), and every other player plays '
                        'each anchor and nobody else (9 models + reference: 9 pairings, not 45)')
    p.add_argument('--matchups', metavar='A:B[,C:D]', help='staged plan: exactly these pairings, not the round-robin')
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
                        'sharing --concurrency per CLI)')
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

    p = sub.add_parser('scorecard', help='the six-axis character sheet of every player in a run (pip bars, 95%% bands, reliability)')
    p.add_argument('run')
    p.add_argument('--json', help='also write the full scorecard as JSON')
    p.add_argument('--solver', default='dcfr', choices=['dcfr', 'heuristic', 'texassolver'],
                   help='how decisions are valued: the built-in DCFR (default), a fast heuristic, or TexasSolver when installed')
    p.add_argument('--players', default='players.toml', help='for display names')
    p.set_defaults(fn=cmd_scorecard)

    p = sub.add_parser('forensics', help='the four-channel adversarial audit and the interface-vs-strategy EV split of a run')
    p.add_argument('run')
    p.add_argument('--json', help='also write the audit as JSON')
    p.add_argument('--players', default='players.toml', help='for display names')
    p.set_defaults(fn=cmd_forensics)

    p = sub.add_parser('invariance', help='does a player decide the same way when suits and seat names are relabelled?')
    p.add_argument('player_id')
    p.add_argument('--spots', type=int, default=12, help='base spots (default 12)')
    p.add_argument('--perms', type=int, default=3, help='relabelled variants per spot (default 3)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=4, help='decisions asked at once (default 4)')
    p.add_argument('--players', default='players.toml')
    p.add_argument('--json', help='also write the result as JSON')
    p.set_defaults(fn=cmd_invariance)

    p = sub.add_parser('solver-check', help='which solvers are ready: built-in DCFR, TexasSolver, Slumbot')
    p.add_argument('--network', action='store_true', help='also try to reach Slumbot')
    p.set_defaults(fn=cmd_solver_check)

    args = ap.parse_args(argv)
    from .runner import ResumeMismatch
    try:
        args.fn(args)
    except ResumeMismatch as e:
        sys.exit(f'Not resuming: {e}')


if __name__ == '__main__':
    main()
