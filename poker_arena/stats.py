"""How each player plays, and how its backend behaved: poker style stats from hands.jsonl and operational
stats from calls.jsonl and events.jsonl, over every match under a folder.

Style (standard poker tracking definitions, per hand dealt):
    vpip   voluntarily put money in preflop (called or raised; checking the big blind doesn't count)
    pfr    raised preflop
    af     aggression factor after the flop: (bets + raises) / calls
    wtsd   went to showdown, as a share of the hands where it saw the flop
    wsd    won money at showdown, as a share of its showdowns
    talk   share of its decisions that came with table talk

Operations (model players):
    decisions, calls (including retries), retries, fallbacks (a seat that never gave a playable answer and
    was checked or folded for), mean and slowest call time, tokens in and out where the CLI reports them,
    and which models actually answered (agy reports this).
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
            s = c[p]
            s['hands'] += 1
            mine = [a for a in acts if a['seat'] == seat]
            pre = [a for a in mine if a['street'] == 'preflop']
            post = [a for a in mine if a['street'] != 'preflop']
            s['vpip'] += any(a['action'] in ('call', 'raise', 'bet') for a in pre)
            s['pfr'] += any(a['action'] in ('raise', 'bet') for a in pre)
            s['aggr'] += sum(a['action'] in ('raise', 'bet') for a in post)
            s['calls_post'] += sum(a['action'] == 'call' for a in post)
            folded_pre = any(a['action'] == 'fold' for a in pre)
            if len(h.get('board', [])) >= 3 and not folded_pre:
                s['saw_flop'] += 1
            if h.get('showdown') and seat in contenders:
                s['showdowns'] += 1
                s['won_sd'] += (h.get('won') or [0] * (seat + 1))[seat] > 0
            s['decisions'] += len(mine)
            s['talk'] += sum(bool(a.get('say')) for a in mine)
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
    return {p: dict(v) for p, v in c.items()}, {p: set(v) for p, v in served.items()}


def add(into: dict, counts: dict):
    """Add one tally's counts for a player into a running total (slowest is a maximum, the rest sum)."""
    for k, v in counts.items():
        into[k] = max(into.get(k, 0), v) if k == 'slowest' else into.get(k, 0) + v


def summarize(s: dict, served=()) -> dict:
    """Rates from summed counts."""
    g = lambda k: s.get(k, 0)
    out = {
        'hands': int(g('hands')),
        'vpip': _ratio(g('vpip'), g('hands')), 'pfr': _ratio(g('pfr'), g('hands')),
        'af': _ratio(g('aggr'), g('calls_post')) if g('calls_post') else None,
        'wtsd': _ratio(g('showdowns'), g('saw_flop')), 'wsd': _ratio(g('won_sd'), g('showdowns')),
        'talk': _ratio(g('talk'), g('decisions')), 'decisions': int(g('decisions')),
        'corrected': int(g('corrected')), 'fallbacks': int(g('fallbacks')),
        'calls': int(g('calls')), 'retries': int(g('retries')),
        'latency_mean': _ratio(g('latency'), g('calls')), 'latency_max': g('slowest') or None,
        'tokens_in': int(g('tokens_in')) or None, 'tokens_out': int(g('tokens_out')) or None,
        'served_models': sorted(served),
    }
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
