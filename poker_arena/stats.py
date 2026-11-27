"""How each player plays, and how its backend behaved: poker style stats from hands.jsonl and operational
stats from calls.jsonl and events.jsonl, over every match under a folder.

Style (standard poker tracking definitions, per hand dealt):
    vpip   voluntarily put money in preflop (called or raised; checking the big blind doesn't count)
    pfr    raised preflop
    af     aggression factor after the flop: (bets + raises) / calls
    wtsd   went to showdown, as a share of the hands where it saw the flop
    wsd    won the pot outright at showdown, as a share of its showdowns; a split pot is not a win, and is
           counted separately as a showdown tie (sd_ties)
    talk   share of its decisions that came with table talk

`LABELS` names each rate for readers ("Plays preflop" for vpip, "Raises preflop" for pfr ...) with its
definition, and `POSITIONS` names the seats of `by_position`. These are tendencies, not measures of quality: no value of them is good or bad in itself. Every rate comes
with its counts (`counts`: {rate: {n, of}}) and is broken down by opponent (`by_opponent`) and by position
(`by_position`: button, which posts the small blind heads-up, or big blind), since pooling over opponents and
seats can hide what a player does against whom.

Operations (model players):
    decisions, calls (including retries), retries, fallbacks (a seat that never gave a playable answer and
    was checked or folded for), mean and slowest call time, tokens in and out where the CLI reports them,
    and which models actually answered.

Audit (model players), from calls.jsonl: `audit` counts the replies that were used by how far their turn
could be checked for tool use (verified, opaque: the backend exposes no record, unverified, unrecorded), and
the replies rejected and asked again (tool_use_rejected, unverified_rejected) or that never came (no_reply);
`identity` counts replies by whether the backend reported the model that answered (attested), did not
(unattested), or reported another model than the one requested (mismatch).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path


def _lines(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _ratio(a, b):
    return None if not b else a / b


def tally(d: Path) -> tuple[dict, dict]:
    """Raw counts per player for one match folder: ({player: {count: n}}, {player: {served models}}).
    Counts add up across matches; `summarize` turns them into rates."""
    c = defaultdict(lambda: defaultdict(float))
    served = defaultdict(set)
    d = Path(d)
    cfg = json.loads((d / 'config.json').read_text())
    players = cfg.get('players', [])
    for h in _lines(d / 'hands.jsonl'):
        acts = h.get('actions', [])
        contenders = {int(k) for k in (h.get('best') or {})}
        for seat, p in enumerate(players):
            mine = [a for a in acts if a['seat'] == seat]
            pre = [a for a in mine if a['street'] == 'preflop']
            post = [a for a in mine if a['street'] != 'preflop']
            row = {'hands': 1, 'vpip': any(a['action'] in ('call', 'raise', 'bet') for a in pre),
                   'pfr': any(a['action'] in ('raise', 'bet') for a in pre),
                   'aggr': sum(a['action'] in ('raise', 'bet') for a in post),
                   'calls_post': sum(a['action'] == 'call' for a in post),
                   'saw_flop': len(h.get('board', [])) >= 3 and not any(a['action'] == 'fold' for a in pre),
                   'showdowns': 0, 'won_sd': 0, 'sd_ties': 0,
                   'decisions': len(mine), 'talk': sum(bool(a.get('say')) for a in mine)}
            if h.get('showdown') and seat in contenders:
                won = h.get('won') or [0] * len(players)
                mine_won = seat < len(won) and won[seat] > 0
                others = any(won[o] > 0 for o in contenders if o != seat and o < len(won))
                row.update(showdowns=1, won_sd=mine_won and not others, sd_ties=mine_won and others)
            s = c[p]
            where = 'button' if h.get('button') == seat else 'big_blind' if len(players) == 2 else 'other'
            prefixes = ['', f'pos:{where}:'] + ([f'vs:{players[1 - seat]}:'] if len(players) == 2 else [])
            for k, v in row.items():
                for pre_ in prefixes:
                    s[pre_ + k] += v
            s['corrected'] += sum(bool(a.get('note')) for a in mine)
    for e in _lines(d / 'events.jsonl'):
        if e.get('kind') == 'thought':
            x = e['data']
            c[x['player']]['fallbacks'] += bool(x.get('fallback'))
    for call in _lines(d / 'calls.jsonl'):
        s = c[call['player']]
        s['calls'] += 1
        s['retries'] += call.get('attempt', 0) > 0
        s['latency'] += call.get('latency') or 0
        s['slowest'] = max(s['slowest'], call.get('latency') or 0)
        u = call.get('usage') or {}
        s['tokens_in'] += (u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0) \
            + (u.get('cache_creation_input_tokens') or 0)
        s['tokens_out'] += u.get('output_tokens') or 0
        if call.get('served_model'):
            served[call['player']].add(call['served_model'])
        if call.get('observed_model'):
            served[call['player']].add(call['observed_model'])
        for k, v in audit_counts(call, (cfg.get('specs') or {}).get(call['player'], {}).get('model')).items():
            s[k] += v
    return {p: dict(v) for p, v in c.items()}, {p: set(v) for p, v in served.items()}


AUDIT = ('verified', 'opaque', 'unverified', 'unrecorded', 'tool_use_rejected', 'unverified_rejected', 'no_reply')
IDENTITY = ('attested', 'unattested', 'mismatch')


def audit_counts(call: dict, model=None) -> dict:
    """One call's contribution to the audit and identity counts (keys `audit:<status>`, `identity:<status>`)."""
    if call.get('backend_failed'):
        return {'audit:no_reply': 1}
    if call.get('rejected') in ('tool_use', 'unverified'):
        out = {f'audit:{call["rejected"]}_rejected': 1}
    else:
        a = call.get('audit')
        out = {f'audit:{a if a in ("verified", "opaque", "unverified") else "unrecorded"}': 1}
    observed = call.get('observed_model') or call.get('served_model')
    requested = call.get('requested_model') or model
    if observed and requested:
        from .ratings import same_model
        if not same_model(requested, observed):
            return {**out, 'identity:mismatch': 1}
    ident = call.get('identity') or ('attested' if observed else 'unattested')
    return {**out, f'identity:{ident if ident in IDENTITY else "unattested"}': 1}


def add(into: dict, counts: dict):
    """Add one tally's counts for a player into a running total (slowest is a maximum, the rest sum)."""
    for k, v in counts.items():
        into[k] = max(into.get(k, 0), v) if k == 'slowest' else into.get(k, 0) + v


LABELS = {   # what each rate is called where people read it, and what it counts
    'vpip': {'label': 'Plays preflop', 'short': 'VPIP',
             'definition': 'share of hands in which it put money in preflop by choice: called or raised (posting a '
                           'blind, or checking the big blind, does not count)'},
    'pfr': {'label': 'Raises preflop', 'short': 'PFR', 'definition': 'share of hands in which it raised preflop'},
    'af': {'label': 'Aggression after the flop', 'short': 'AF',
           'definition': 'bets and raises after the flop, per call after the flop'},
    'wtsd': {'label': 'Goes to showdown', 'short': 'WTSD',
             'definition': 'share of the hands where it saw the flop that it took to a showdown'},
    'wsd': {'label': 'Wins at showdown', 'short': 'W$SD',
            'definition': 'share of its showdowns whose pot it won outright (a split pot is not a win)'},
    'sd_ties': {'label': 'Splits at showdown', 'short': 'split',
                'definition': 'share of its showdowns that ended in a split pot'},
    'talk': {'label': 'Talks', 'short': 'talk', 'definition': 'share of its decisions that came with table talk'},
}
POSITIONS = {'button': 'Button (small blind)', 'big_blind': 'Big blind'}

RATES = {'vpip': ('vpip', 'hands'), 'pfr': ('pfr', 'hands'), 'af': ('aggr', 'calls_post'),
         'wtsd': ('showdowns', 'saw_flop'), 'wsd': ('won_sd', 'showdowns'), 'sd_ties': ('sd_ties', 'showdowns'),
         'talk': ('talk', 'decisions')}


def _style(g) -> dict:
    out = {k: _ratio(g(n), g(of)) for k, (n, of) in RATES.items() if k != 'sd_ties'}
    out.update({'hands': int(g('hands')), 'showdowns': int(g('showdowns')), 'won_sd': int(g('won_sd')),
                'sd_ties': int(g('sd_ties')), 'decisions': int(g('decisions')),
                'counts': {k: {'n': int(g(n)), 'of': int(g(of))} for k, (n, of) in RATES.items()}})
    return out


def summarize(s: dict, served=()) -> dict:
    """Rates from summed counts, each with its counts, and broken down by opponent and position."""
    g = lambda k: s.get(k, 0)
    out = {
        **_style(g),
        'corrected': int(g('corrected')), 'fallbacks': int(g('fallbacks')),
        'calls': int(g('calls')), 'retries': int(g('retries')),
        'latency_mean': _ratio(g('latency'), g('calls')), 'latency_max': g('slowest') or None,
        'tokens_in': int(g('tokens_in')) or None, 'tokens_out': int(g('tokens_out')) or None,
        'served_models': sorted(served),
        'audit': {k: int(g(f'audit:{k}')) for k in AUDIT},
        'identity': {k: int(g(f'identity:{k}')) for k in IDENTITY},
    }
    for key, prefix in (('by_opponent', 'vs:'), ('by_position', 'pos:')):
        names = sorted({k[len(prefix):].rsplit(':', 1)[0] for k in s if k.startswith(prefix)})
        out[key] = {n: _style(lambda k, n=n: s.get(f'{prefix}{n}:{k}', 0)) for n in names}
    if not g('calls_post') and g('aggr'):
        out['af_all_aggression'] = True   # never called after the flop: shown as "all aggression"
    return out


def match_dirs(root):
    for cfg_path in sorted(Path(root).rglob('config.json')):
        if (cfg_path.parent / 'hands.jsonl').exists():
            yield cfg_path.parent


def collect(root) -> dict:
    """{player: {...style and operations...}} over every match folder under root."""
    total = defaultdict(dict)
    served = defaultdict(set)
    for d in match_dirs(root):
        counts, sv = tally(d)
        for p, cnt in counts.items():
            add(total[p], cnt)
        for p, ms in sv.items():
            served[p] |= ms
    return {p: summarize(s, served[p]) for p, s in total.items()}
