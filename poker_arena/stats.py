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


def collect(root) -> dict:
    """{player: {...style and operations...}} over every match folder under root."""
    c = defaultdict(lambda: defaultdict(float))
    served = defaultdict(set)
    for cfg_path in Path(root).rglob('config.json'):
        d = cfg_path.parent
        if not (d / 'hands.jsonl').exists():
            continue
        cfg = json.loads(cfg_path.read_text())
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
    out = {}
    for p, s in c.items():
        out[p] = {
            'hands': int(s['hands']),
            'vpip': _ratio(s['vpip'], s['hands']), 'pfr': _ratio(s['pfr'], s['hands']),
            'af': _ratio(s['aggr'], s['calls_post']) if s['calls_post'] else (None if not s['aggr'] else float('inf')),
            'wtsd': _ratio(s['showdowns'], s['saw_flop']), 'wsd': _ratio(s['won_sd'], s['showdowns']),
            'talk': _ratio(s['talk'], s['decisions']), 'decisions': int(s['decisions']),
            'corrected': int(s['corrected']), 'fallbacks': int(s['fallbacks']),
            'calls': int(s['calls']), 'retries': int(s['retries']),
            'latency_mean': _ratio(s['latency'], s['calls']), 'latency_max': s['slowest'] or None,
            'tokens_in': int(s['tokens_in']) or None, 'tokens_out': int(s['tokens_out']) or None,
            'served_models': sorted(served[p]),
        }
        if out[p]['af'] == float('inf'):
            out[p]['af'] = None  # never called after the flop: show as "all aggression" in the viewer
            out[p]['af_all_aggression'] = True
    return out
