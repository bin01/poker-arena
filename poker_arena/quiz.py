"""Poker fundamentals: a quiz of heads-up decision spots whose best answer is computed, in separately scored
categories.

Every spot is a real position of the engine (blinds 1/2, a stated stack, the actions so far) and every option's
value, in big blinds, is computed from the hand's equity against what Villain holds: exact (every combo, every
runout) from the flop on, Monte Carlo with SIMS draws from a fixed seed before it.

    basics       the original 32 hand-picked spots: a call that ends the betting (against an all-in, or a river
                 bet with raising ruled out) or an all-in-or-fold first decision, against a stated range. Clear
                 cut: the best option beats the other by at least MIN_GAP_BB, and for calls the equity differs from
                 the price by at least MIN_EQUITY_GAP. A competency check.
    close        the same kind of call, on the turn or river, where the best option beats the other by only
                 CLOSE_GAP bb: the price has to be worked out, not eyeballed.
    multistreet  a turn decision (fold, call, all in) against a stated range and a stated plan for the river:
                 if hero calls, Villain goes all in on the river with a stated set of hands and checks the rest,
                 and hero then calls or folds by the river card; if hero goes all in, Villain calls with a stated
                 set. Calling is valued with hero's best river response on every river card.
    inference    a river call against an all-in where Villain's range is not stated: Villain's preflop raising
                 range and its rule on each street (bet with these hand types, check the rest) are, and the range
                 it holds now follows from the line it took.
    sizing       a river bet with a value or air hand among check and 3+ sizes, against a stated range that calls
                 each size with a stated set of hands and never raises.

The spots of the four harder categories are generated: draws from a seeded generator (GENERATOR_SEED, one stream
per category), each kept only if it is a legal engine position whose best option beats the next one by the
category's margin, until PER_CATEGORY are kept (no answer may be the best in more than MAX_SHARE of them). A
category's spots are therefore an i.i.d. sample from that generator's accepted spots: its accuracy estimates the
share of that population a player answers best (the `estimand` of each category). The basics have no sampling
population.

A player answers each spot in a fresh conversation, through its usual backend, with the game's JSON reply;
the answer is checked by the engine (`apply`) and must be one of the spot's options, else it is sent back
(max_fixes times, then the engine's fallback: fold, or check when there is nothing to call). A fallback, or a
backend that gave no reply, is recorded as such (`outcome` 'fallback' | 'error', `fallback: true`) and never
counts as correct, whatever the fallback did; its loss is the worst option's. Bots answer from the same request
data they play from.

Results go to <out>/quiz.json: the protocol and implementation of the quiz, every spot with its computed
values, and per player its effective settings, protocol fingerprint, implementation, the CLI artifact it ran
(path, version, sha256, re-checked on every call), every answer with its calls and their raw output, the answer's
`identity_match` and `audit` (each call re-checked against its raw output by its backend's parser), and an
`eligibility` status: 'eligible' only when every answer was given by the requested model, every call is verified
against its raw output and the CLI never changed. Scores are kept per category (`categories`), with valid-answer
accuracy apart from fallbacks; the top-level score is the basics'. The quiz is reported apart from match
results, as "poker fundamentals".
"""
from __future__ import annotations
import asyncio
import copy
import hashlib
import itertools
import json
import random
import re
import time
from pathlib import Path

from .games import get_game, ranges
from .games.base import Request
from .cards import DECK, VALUE, evaluate

VERSION = 'quiz-2'
SIMS = 40000
MIN_GAP_BB = 1.0
MIN_EQUITY_GAP = 0.04
CLOSE_GAP = (0.2, 1.0)        # close spots: the best option beats the other by at least 0.2 bb and less than 1 bb
HARD_GAP_BB = (0.5, 10.0)     # multistreet, inference, sizing: the best option beats the next by 0.5 to 10 bb
INFERENCE_EQUITY_GAP = (0.03, 0.3)   # inference: the equity is 3 to 30 points from the price (the line matters)
PER_CATEGORY = 24
MAX_SHARE = 0.5
GENERATOR_SEED = 7001
FLOP = [('button', 'raise', 6), ('bb', 'call', None)]   # a 3 bb open, called: 6 bb in the pot on the flop

# (id, hero's seat: 'button' or 'bb', stack in bb, hero's cards, board, actions before hero's decision
#  [(seat, action, amount)], the opponent's range, kind: 'call' (fold or call) or 'shove' (fold or all in))
SPOTS = [
    ('pf-shove-a7o-10', 'button', 10, 'Ah 7c', '', [], 'TT+, AQ+', 'shove'),
    ('pf-shove-72o-10', 'button', 10, '7d 2c', '', [], 'random', 'shove'),
    ('pf-shove-a2o-10', 'button', 10, 'As 2d', '', [], 'random', 'shove'),
    ('pf-shove-93o-10', 'button', 10, '9h 3c', '', [], 'random', 'shove'),
    ('pf-shove-54s-20', 'button', 20, '5s 4s', '', [], 'JJ+, AK', 'shove'),
    ('pf-shove-k9s-15', 'button', 15, 'Kh 9h', '', [], '99+, AJ+', 'shove'),
    ('pf-shove-j3o-30', 'button', 30, 'Jc 3d', '', [], '22+, A2+, K5+, Q8+, J9+', 'shove'),
    ('pf-shove-kto-8', 'button', 8, 'Kd Tc', '', [], 'random', 'shove'),
    ('pf-shove-62o-25', 'button', 25, '6c 2h', '', [], '22+, A2+, K2+, Q5+, J7+, T8+', 'shove'),
    ('pf-shove-q9s-10', 'button', 10, 'Qs 9s', '', [], '77+, A9+, KQ', 'shove'),
    ('pf-call-k4o-10', 'bb', 10, 'Kc 4d', '', [('button', 'all_in', None)], 'random', 'call'),
    ('pf-call-ajo-20', 'bb', 20, 'Ah Jd', '', [('button', 'all_in', None)], 'QQ+, AK', 'call'),
    ('pf-call-tt-50', 'bb', 50, 'Th Ts', '', [('button', 'all_in', None)], 'JJ+, AKs', 'call'),
    ('pf-call-qq-100', 'bb', 100, 'Qd Qc', '', [('button', 'all_in', None)], 'KK+', 'call'),
    ('pf-call-a8o-10', 'bb', 10, 'Ah 8c', '', [('button', 'all_in', None)], '22+, A2+, K2+, Q2+, J7+, T8+, 98', 'call'),
    ('pf-call-55-15', 'bb', 15, '5h 5c', '', [('button', 'all_in', None)], 'random', 'call'),
    ('pf-call-72o-12', 'bb', 12, '7c 2d', '', [('button', 'all_in', None)], 'random', 'call'),
    ('pf-call-ako-30', 'bb', 30, 'Ad Kh', '', [('button', 'all_in', None)], '55+, A9+, KQ', 'call'),
    ('pf-call-a5s-25', 'bb', 25, 'As 5s', '', [('button', 'all_in', None)], 'TT+, AK', 'call'),
    ('flop-call-nfd-vs-made', 'bb', 100, 'Ah 5h', 'Kh 9h 2c', FLOP + [('bb', 'check', None), ('button', 'all_in', None)],
     'KK, 99, 22, AK, KQ, KJ, K9s', 'call'),
    ('flop-call-set', 'bb', 100, 'Qs Qd', 'Qc 7h 2d', FLOP + [('bb', 'check', None), ('button', 'all_in', None)],
     'AA, KK, AQ, KQ, 77', 'call'),
    ('flop-call-oesd-vs-sets', 'button', 100, 'Js Tc', '9d 8h 2s', FLOP + [('bb', 'all_in', None)],
     'AA, KK, 99, 88, 22', 'call'),
    ('flop-call-combo-draw', 'bb', 100, 'Jh Th', '9h 8h 2c', FLOP + [('bb', 'check', None), ('button', 'all_in', None)],
     'AA, KK, AK', 'call'),
    ('flop-call-tptk-vs-draws', 'button', 100, 'Ac Kd', 'Kh 8h 4c', FLOP + [('bb', 'all_in', None)],
     'AhQh, AhJh, AhTh, QhJh, QhTh, JhTh, Th9h, 9h7h, 7h6h, QJ, JT, T9', 'call'),
    ('turn-call-fd-vs-two-pair', 'bb', 100, '7h 6h', 'Ah Kh 2d 9c',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'all_in', None)],
     'AK, AA, KK', 'call'),
    ('turn-call-tptk', 'bb', 100, 'As Ks', 'Ac 7d 4h 2s',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'all_in', None)],
     'AQ, AJ, AT, KK, QQ, 77, 44, 65s, 53s', 'call'),
    ('turn-call-overpair-vs-nuts', 'button', 100, 'Ks Kd', 'Jh Th 9c 2h',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'all_in', None)],
     'KhQh, AhQh, Qh8h, KQ, Q8s, AhKh', 'call'),
    ('turn-call-gutshot-cheap', 'bb', 15, '8c 7c', 'Jd Ts 3h 2c',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'all_in', None)],
     'AA, KK, QQ, AJ, KJ, QJ', 'call'),
    ('river-call-bluffcatcher', 'bb', 100, 'Kd Jd', 'Ks 8h 4c 2d 3s',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'check', None),
             ('bb', 'check', None), ('button', 'raise', 12)],
     'AA, 88, 44, AK, QTs, T9s, 97s, 76s, J9s, Q9s', 'call'),
    ('river-fold-vs-value', 'bb', 100, 'Kd Jd', 'Ks 8h 4c 2d 3s',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'check', None),
             ('bb', 'check', None), ('button', 'raise', 12)],
     'AA, 88, 44, AK, K8s, 65s, QhTh, JhTh', 'call'),
    ('river-call-missed-draws', 'button', 100, 'Ah Qs', 'Kc 9d 4h 2s 7c',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'check', None),
             ('bb', 'raise', 12)],
     'K9, 99, QJs, JTs, T8s, 65s, 86s', 'call'),
    ('river-fold-pot-bet-vs-value', 'button', 100, 'Ah Qs', 'Kc 9d 4h 2s 7c',
     FLOP + [('bb', 'check', None), ('button', 'check', None), ('bb', 'check', None), ('button', 'check', None),
             ('bb', 'raise', 12)],
     'K9, 99, 77, AK, KQ, KJ', 'call'),
]


CATEGORIES = {
    'basics': {'title': 'Basics', 'about': 'clear-cut pot-odds calls and all-in decisions against a stated range',
               'population': None,
               'estimand': 'none: a fixed, hand-picked set; report counts, not population intervals'},
    'close': {'title': 'Close calls', 'about': f'calls against a stated range where the best option is only '
                                               f'{CLOSE_GAP[0]:g} to {CLOSE_GAP[1]:g} bb better',
              'population': 'generated', 'estimand': 'the share of the close-call generator\'s accepted spots a '
                                                     'player answers best'},
    'multistreet': {'title': 'Multi-street lines', 'about': 'turn decisions whose value depends on the river plan',
                    'population': 'generated', 'estimand': 'the share of the multi-street generator\'s accepted '
                                                           'spots a player answers best'},
    'inference': {'title': 'Range inference', 'about': 'river calls where the range follows from a described line',
                  'population': 'generated', 'estimand': 'the share of the range-inference generator\'s accepted '
                                                         'spots a player answers best'},
    'sizing': {'title': 'Bet sizing', 'about': 'river bets: check or one of 3+ sizes, each with a stated response',
               'population': 'generated', 'estimand': 'the share of the sizing generator\'s accepted spots a player '
                                                      'answers best'},
}


def _cards(text) -> list[str]:
    return text.split() if isinstance(text, str) else list(text)


def _game(stack_bb: int):
    return get_game('holdem')({'sb': 1, 'bb': 2, 'stack': 2 * stack_bb, 'hands_per_match': 1})


# ---- hand features Villain's rules are written in ---------------------------------------------------------

def _top(board) -> int:
    return max(VALUE[c[0]] for c in board)


def _flush_draw(combo, board) -> bool:
    cards = list(combo) + list(board)
    for suit in 'shdc':
        n = sum(c[1] == suit for c in cards)
        if n >= 5:
            return False
        if n == 4 and any(c[1] == suit for c in combo):
            return True
    return False


FEATURES = {
    'no_pair': ('no pair at all (high card only)', lambda c, b: evaluate(list(c) + list(b))[0] == 0),
    'pair_plus': ('a pair or better', lambda c, b: evaluate(list(c) + list(b))[0] >= 1),
    'top_pair_plus': ('top pair or better (a pair at least as high as the highest board card, or two pair or better)',
                      lambda c, b: (lambda s: s[0] >= 2 or (s[0] == 1 and s[1] >= _top(b)))(evaluate(list(c) + list(b)))),
    'two_pair_plus': ('two pair or better', lambda c, b: evaluate(list(c) + list(b))[0] >= 2),
    'trips_plus': ('three of a kind or better', lambda c, b: evaluate(list(c) + list(b))[0] >= 3),
    'flush_draw': ('a flush draw (four cards of one suit, at least one of them its own, and no flush yet)', _flush_draw),
}
FEATURE_NOTE = ('A hand type always means Villain\'s best five cards from its two cards and the board as it is at '
                'that moment.')


def holds(rule, combo, board) -> bool:
    return any(FEATURES[f][1](combo, board) for f in rule)


def rule_text(rule) -> str:
    words = [FEATURES[f][0] for f in rule]
    return words[0] if len(words) == 1 else ', '.join(words[:-1]) + ' or ' + words[-1]


# ---- spots -------------------------------------------------------------------------------------------------

def _basic(t: tuple) -> dict:
    sid, pos, stack, hole, board, script, rng_text, kind = t
    return {'id': sid, 'category': 'basics', 'kind': kind, 'position': pos, 'stack_bb': stack, 'hole': _cards(hole),
            'board': _cards(board), 'script': [list(x) for x in script], 'range': rng_text}


def villain_combos(d: dict) -> list:
    """The combos Villain can hold at the decision: the stated range, or for inference the preflop range kept
    through every street's rule, without combos that use cards hero can see."""
    hole, board = d['hole'], d['board']
    combos = [c for c in ranges.parse(d['range']) if not set(c) & set(hole + board)]
    for street, n in (('flop', 3), ('turn', 4), ('river', 5)):
        rule = (d.get('line') or {}).get(street)
        if rule:
            combos = [c for c in combos if holds(rule, c, board[:n])]
    return combos


def build(d):
    """The engine's hand at the spot's decision: (episode, hero's seat, villain's seat, Villain's combos)."""
    d = _basic(d) if isinstance(d, tuple) else d
    sid = d['id']
    game = _game(d['stack_bb'])
    hero = 0
    button = hero if d['position'] == 'button' else 1
    ep = game.episode(button, 0, ['Hero', 'Villain'])
    hole, board = d['hole'], d['board']
    combos = villain_combos(d)
    if not combos:
        raise ValueError(f'{sid}: the range has no combo possible with these cards')
    dummy = list(combos[0])
    ep.hole = [hole, dummy] if hero == 0 else [dummy, hole]
    rest = [c for c in DECK if c not in hole + board + dummy]
    ep.deck_board = board + rest[:5 - len(board)]
    seat_of = {'button': button, 'bb': 1 - button}
    for who, action, amount in d['script']:
        seat = seat_of[who]
        if ep.to_act != seat:
            raise ValueError(f'{sid}: {who} acts out of turn')
        err = ep.apply(seat, {'action': action, 'amount': amount})
        if err or ep.history[-1].get('correction'):
            raise ValueError(f'{sid}: {who} {action} {amount}: {err or ep.history[-1]["note"]}')
    if ep.done or ep.to_act != hero or ep.board != board:
        raise ValueError(f'{sid}: the script does not reach hero\'s decision on that board')
    return ep, hero, 1 - hero, combos


def _shares(hole, board, combos) -> list[list[float]]:
    """Hero's share of the pot against each combo on every runout to the river: [[share per runout] per combo]."""
    known = set(hole) | set(board)
    out = []
    mine = {}
    for c in combos:
        rest = [x for x in DECK if x not in known and x not in c]
        row = []
        for run in itertools.combinations(rest, 5 - len(board)):
            full = list(board) + list(run)
            key = tuple(run)
            if key not in mine:
                mine[key] = evaluate(list(hole) + full)
            theirs = evaluate(list(c) + full)
            row.append(1.0 if mine[key] > theirs else 0.5 if mine[key] == theirs else 0.0)
        out.append(row)
    return out


_SOLVED: dict = {}


def solve(d) -> dict:
    """The spot's facts and the value of each option in bb (relative to folding, or to checking when there is
    nothing to call), computed once per process."""
    d = _basic(d) if isinstance(d, tuple) else d
    if d['id'] not in _SOLVED:
        _SOLVED[d['id']] = _solve(d)
    return _SOLVED[d['id']]


VERB = {'fold': 'fold', 'call': 'call', 'all_in': 'go all in', 'check': 'check'}


def _verb(o: str) -> str:
    return VERB.get(o) or f'bet {o.split("_")[1]}'


def _solve(d: dict) -> dict:
    sid, kind, hole, board = d['id'], d['kind'], d['hole'], d['board']
    ep, hero, villain, combos = build(d)
    bb, pot, tc = ep.bb, ep.pot, ep.to_call(hero)
    rng = random.Random(int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16))
    exact = len(board) >= 3
    out = {'id': sid, 'category': d['category'], 'kind': kind, 'street': ['preflop', 'flop', 'turn', 'river'][ep.street],
           'position': d['position'], 'stack_bb': d['stack_bb'], 'hole': hole, 'board': board,
           'actions': [{k: h[k] for k in ('seat', 'street', 'action') if k in h} | {'amount': h.get('amount')}
                       for h in ep.history],
           'pot_bb': pot / bb, 'to_call_bb': tc / bb, 'villain_range': d['range'], 'villain_combos': len(combos),
           'equity_method': 'exact' if exact else f'monte carlo, {SIMS} draws'}
    for k in ('line', 'river_plan', 'shove_call', 'responses'):
        if d.get(k):
            out[k] = d[k]
    if kind in ('call', 'inference'):
        if not (ep.stacks[villain] == 0 or tc >= ep.stacks[hero] or ep.street == 3):
            raise ValueError(f'{sid}: a call here does not end the betting')
        eq = ranges.equity(hole, board, combos, rng, sims=SIMS, exact=exact)
        need = tc / (pot + tc)
        ev = {'fold': 0.0, 'call': (eq * (pot + tc) - tc) / bb}
        out.update(equity=round(eq, 4), required_equity=round(need, 4), equity_gap=round(eq - need, 4),
                   options=['fold', 'call'], needed=round(need, 4))
        if kind == 'inference':
            out['derived_range'] = sorted(''.join(c) for c in combos)
        explain = (f'Calling {tc / bb:g} bb to win {(pot + tc) / bb:g} bb needs {need:.1%} equity; against '
                   + ('the stated range' if kind == 'call' else f'the {len(combos)} combinations Villain\'s line leaves')
                   + f' the hand has {eq:.1%}, so the best play is to {{best}} (calling is worth {ev["call"]:+.2f} bb '
                     f'against folding).')
    elif kind == 'shove':
        if ep.history or hero != ep.button:
            raise ValueError(f'{sid}: a shove spot is the first decision, on the button')
        eq = ranges.equity(hole, board, combos, rng, sims=SIMS, exact=exact)
        S, h, v = ep.start, ep.committed[hero], ep.committed[villain]
        possible = [c for c in ranges.ALL if not set(c) & set(hole)]
        p_call = len(combos) / len(possible)
        ev = {'fold': 0.0, 'all_in': ((1 - p_call) * (h + v) + p_call * (eq * 2 * S - (S - h))) / bb}
        out.update(equity=round(eq, 4), call_probability=round(p_call, 4), options=['fold', 'all_in'])
        explain = (f'Villain calls {p_call:.1%} of the time, and then the hand has {eq:.1%} equity; going all in '
                   f'is worth {ev["all_in"]:+.2f} bb against folding, so the best play is to {{best}}.')
    elif kind == 'multistreet':
        if ep.street != 2 or not tc or not ep.raise_bounds(hero):
            raise ValueError(f'{sid}: a multi-street spot is a turn bet hero can call or raise all in against')
        R = ep.stacks[hero] - tc
        if R <= 0 or ep.stacks[villain] < R:
            raise ValueError(f'{sid}: calling must leave both players chips for the river')
        shares = _shares(hole, board, combos)
        rivers = [x for x in DECK if x not in set(hole) | set(board)]
        by_river = {r: {'shove': [], 'check': []} for r in rivers}
        for c, row in zip(combos, shares):
            runs = [x for x in rivers if x not in c]
            for r, share in zip(runs, row):
                by_river[r]['shove' if holds(d['river_plan'], c, board + [r]) else 'check'].append(share)
        weight = 1 / (len(combos) * (len(rivers) - 2))
        total, calls_on = 0.0, []
        for r, g in by_river.items():
            total += sum(x * (pot + tc) - tc for x in g['check'])
            if g['shove']:
                called = sum(x * (pot + tc + 2 * R) - tc - R for x in g['shove'])
                if called >= -tc * len(g['shove']):
                    calls_on.append(r)
                total += max(called, -tc * len(g['shove']))
        ev_call = total * weight
        callers = [row for c, row in zip(combos, shares) if holds(d['shove_call'], c, board)]
        ev_all_in = (sum(sum(row) / len(row) * (pot + tc + 2 * R) - tc - R for row in callers)
                     + pot * (len(combos) - len(callers))) / len(combos)
        eq = sum(map(sum, shares)) / sum(map(len, shares))
        ev = {'fold': 0.0, 'call': ev_call / bb, 'all_in': ev_all_in / bb}
        out.update(equity=round(eq, 4), options=['fold', 'call', 'all_in'], river_calls=len(calls_on),
                   rivers=len(rivers), call_probability=round(len(callers) / len(combos), 4))
        explain = (f'Calling {tc / bb:g} bb is worth {ev["call"]:+.2f} bb: Villain then goes all in on the river with '
                   f'{rule_text(d["river_plan"])}, and the best response calls that all-in on {len(calls_on)} of the '
                   f'{len(rivers)} river cards. Going all in is worth {ev["all_in"]:+.2f} bb: Villain calls '
                   f'{len(callers) / len(combos):.0%} of the time. So the best play is to {{best}}.')
    elif kind == 'sizing':
        if ep.street != 3 or tc or not ep.raise_bounds(hero):
            raise ValueError(f'{sid}: a sizing spot is a river decision with nothing to call')
        shares = [row[0] for row in _shares(hole, board, combos)]
        eq = sum(shares) / len(shares)
        hi = ep.raise_bounds(hero)[1]
        check = eq * pot
        ev = {'check': 0.0}
        for label, rule in d['responses'].items():
            amount = hi if label == 'all_in' else int(label.split('_')[1])
            val = sum(x * (pot + 2 * amount) - amount if holds(rule, c, board) else pot for c, x in zip(combos, shares))
            ev[label] = (val / len(combos) - check) / bb
        out.update(equity=round(eq, 4), options=list(ev), all_in_amount=hi)
        explain = (f'Against the stated range the hand wins {eq:.1%} of the pot at showdown. Relative to checking, '
                   + '; '.join(f'{_verb(o)} is worth {ev[o]:+.2f} bb' for o in ev if o != 'check')
                   + '. So the best play is to {best}.')
    else:
        raise ValueError(f'{sid}: unknown kind {kind!r}')
    out['ev_bb'] = {k: round(x, 3) for k, x in ev.items()}
    best = max(ev, key=ev.get)
    order = sorted(ev.values(), reverse=True)
    out.update(correct=best, best=best, ev=out['ev_bb'], bb=bb, pot=pot, to_call=tc, hero_bet=ep.street_bet[hero],
               villain_bet=ep.street_bet[villain], stack=ep.stacks[hero], villain_stack=ep.stacks[villain],
               button=0 if ep.button == hero else 1, title=d.get('title') or TITLES.get(sid, sid),
               equity_note=('against the range Villain\'s line leaves' if kind == 'inference' else
                            'against the stated range') + f' ({out["equity_method"]})',
               explain=explain.replace('{best}', _verb(best)))
    out['situation'] = (f'Villain is all in for {ep.committed[villain] / bb:g} bb' if ep.stacks[villain] == 0 else
                        f'Villain bets {tc / bb:g} bb into {(pot - tc) / bb:g} bb') if tc and kind != 'shove' else \
        (f'First to act on the button, {d["stack_bb"]} bb deep' if kind == 'shove' else
         f'First to act on the river, {(pot / bb):g} bb in the pot')
    out['gap_bb'] = round(order[0] - order[1], 3)
    if kind in ('call', 'shove') and d['category'] == 'basics':
        out['clear_cut'] = out['gap_bb'] >= MIN_GAP_BB and (kind != 'call' or abs(out['equity_gap']) >= MIN_EQUITY_GAP)
    elif d['category'] == 'close':
        out['clear_cut'] = CLOSE_GAP[0] <= out['gap_bb'] < CLOSE_GAP[1]
    else:
        out['clear_cut'] = HARD_GAP_BB[0] <= out['gap_bb'] <= HARD_GAP_BB[1] and (
            kind != 'inference' or INFERENCE_EQUITY_GAP[0] <= abs(out['equity_gap']) <= INFERENCE_EQUITY_GAP[1])
    return out


TITLES = {
    'pf-shove-a7o-10': 'A7o shove at 10 bb against a tight caller', 'pf-shove-72o-10': '72o shove into any two',
    'pf-shove-a2o-10': 'A2o shove into any two', 'pf-shove-93o-10': '93o shove into any two',
    'pf-shove-54s-20': '54s shove at 20 bb against a premium caller', 'pf-shove-k9s-15': 'K9s shove at 15 bb',
    'pf-shove-j3o-30': 'J3o shove at 30 bb', 'pf-shove-kto-8': 'KTo shove at 8 bb into any two',
    'pf-shove-62o-25': '62o shove at 25 bb against a wide caller', 'pf-shove-q9s-10': 'Q9s shove at 10 bb',
    'pf-call-k4o-10': 'K4o against a 10 bb any-two shove', 'pf-call-ajo-20': 'AJo against a premium shove',
    'pf-call-tt-50': 'Tens against JJ+ and AKs', 'pf-call-qq-100': 'Queens against kings or aces',
    'pf-call-a8o-10': 'A8o against a wide 10 bb shove', 'pf-call-55-15': 'Fives against any-two shove',
    'pf-call-72o-12': '72o against an any-two shove', 'pf-call-ako-30': 'AKo against a 30 bb shove',
    'pf-call-a5s-25': 'A5s against TT+ and AK', 'flop-call-nfd-vs-made': 'Nut flush draw against made hands',
    'flop-call-set': 'Top set facing a flop shove', 'flop-call-oesd-vs-sets': 'Open-ender against sets and overpairs',
    'flop-call-combo-draw': 'Combo draw against overpairs', 'flop-call-tptk-vs-draws': 'Top pair against draws',
    'turn-call-fd-vs-two-pair': 'Flush draw against two pair and sets', 'turn-call-tptk': 'Top pair, top kicker on the turn',
    'turn-call-overpair-vs-nuts': 'Overpair against made straights and flushes',
    'turn-call-gutshot-cheap': 'Gutshot against a short shove', 'river-call-bluffcatcher': 'Bluff-catcher on the river',
    'river-fold-vs-value': 'Top pair against a value-heavy river bet', 'river-call-missed-draws': 'Ace high against missed draws',
    'river-fold-pot-bet-vs-value': 'Ace high against only value',
}




# ---- the generated categories --------------------------------------------------------------------------------

RANGES = ['TT+, AQ+', '77+, AT+, KQ', '22+, A2s+, A9o+, KTs+, KQo, QJs', '55+, A7+, K9+, QT+, JT',
          '22+, A2+, K8+, Q9+, J9+, T9', '99+, AJs+, KQs, AQo+', '22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 98s, 87s, ATo+, KJo+']
RIVER_PLANS = [['two_pair_plus'], ['two_pair_plus', 'no_pair'], ['top_pair_plus'], ['trips_plus', 'no_pair'],
               ['top_pair_plus', 'no_pair']]
SHOVE_CALLS = [['top_pair_plus'], ['two_pair_plus'], ['pair_plus'], ['top_pair_plus', 'flush_draw'],
               ['pair_plus', 'flush_draw']]
LINES = [(['pair_plus', 'flush_draw'], ['top_pair_plus', 'flush_draw'], ['two_pair_plus', 'no_pair']),
         (['pair_plus', 'flush_draw'], ['pair_plus', 'flush_draw'], ['top_pair_plus', 'no_pair']),
         (['top_pair_plus', 'flush_draw'], ['top_pair_plus', 'flush_draw'], ['two_pair_plus', 'no_pair']),
         (['pair_plus'], ['top_pair_plus'], ['two_pair_plus']),
         (['pair_plus', 'flush_draw'], ['two_pair_plus', 'flush_draw'], ['trips_plus', 'no_pair']),
         (['top_pair_plus', 'flush_draw'], ['pair_plus', 'flush_draw'], ['top_pair_plus'])]
LADDERS = [(['pair_plus'], ['top_pair_plus'], ['two_pair_plus'], ['trips_plus']),
           (['pair_plus'], ['two_pair_plus'], ['trips_plus'], ['trips_plus']),
           (['pair_plus'], ['pair_plus'], ['two_pair_plus'], ['trips_plus']),
           (['top_pair_plus'], ['two_pair_plus'], ['two_pair_plus'], ['trips_plus']),
           (['pair_plus'], ['pair_plus'], ['top_pair_plus'], ['two_pair_plus']),
           (['top_pair_plus'], ['top_pair_plus'], ['two_pair_plus'], ['trips_plus']),
           (['pair_plus'], ['top_pair_plus'], ['top_pair_plus'], ['two_pair_plus'])]
CHECKS = [('bb', 'check', None), ('button', 'check', None)]


def _deal(rng, n_board):
    while True:
        cards = rng.sample(DECK, 2 + n_board)
        board = cards[2:]
        if len({c[0] for c in board}) == len(board):     # an unpaired board: "top pair" means one thing
            return cards[:2], board


def _candidate(cat: str, rng: random.Random, i: int) -> dict:
    sid = f'{cat}-{i:03d}'
    if cat == 'close':
        river = rng.random() < 0.7
        hole, board = _deal(rng, 5 if river else 4)
        stack = rng.choice([20, 30, 40, 60, 100])
        if river:
            pot = 12
            bet = rng.choice([pot // 2, pot * 3 // 4, pot])
            script = FLOP + CHECKS * 2 + [('bb', 'check', None), ('button', 'raise', bet)]
        else:
            script = FLOP + CHECKS + [('bb', 'check', None), ('button', 'all_in', None)]
            stack = rng.choice([10, 15, 20, 25])
        return {'id': sid, 'category': cat, 'kind': 'call', 'position': 'bb', 'stack_bb': stack, 'hole': hole,
                'board': board, 'script': script, 'range': rng.choice(RANGES)}
    if cat == 'multistreet':
        hole, board = _deal(rng, 4)
        bet = rng.choice([6, 8, 12])
        return {'id': sid, 'category': cat, 'kind': 'multistreet', 'position': 'bb', 'stack_bb': rng.choice([25, 40, 60]),
                'hole': hole, 'board': board, 'script': FLOP + CHECKS + [('bb', 'check', None), ('button', 'raise', bet)],
                'range': rng.choice(RANGES), 'river_plan': rng.choice(RIVER_PLANS), 'shove_call': rng.choice(SHOVE_CALLS)}
    if cat == 'inference':
        hole, board = _deal(rng, 5)
        flop, turn, river = rng.choice(LINES)
        script = FLOP + [('bb', 'check', None), ('button', 'raise', 8), ('bb', 'call', None),
                         ('bb', 'check', None), ('button', 'raise', 18), ('bb', 'call', None),
                         ('bb', 'check', None), ('button', 'all_in', None)]
        return {'id': sid, 'category': cat, 'kind': 'inference', 'position': 'bb', 'stack_bb': rng.choice([45, 60, 80]),
                'hole': hole, 'board': board, 'script': script, 'range': rng.choice(RANGES),
                'line': {'flop': flop, 'turn': turn, 'river': river}}
    hole, board = _deal(rng, 5)
    ladder = rng.choice(LADDERS)
    script = FLOP + [('bb', 'check', None), ('button', 'raise', 8), ('bb', 'call', None)] + CHECKS
    pot = 28
    sizes = [f'bet_{round(pot / 3)}', f'bet_{round(2 * pot / 3)}', f'bet_{pot}', 'all_in']
    return {'id': sid, 'category': cat, 'kind': 'sizing', 'position': 'bb', 'stack_bb': rng.choice([40, 60, 100]),
            'hole': hole, 'board': board, 'script': script, 'range': rng.choice(RANGES),
            'responses': dict(zip(sizes, ladder))}


_GENERATED: dict = {}


def generated(cat: str) -> list[dict]:
    """The category's spots: candidates from its own seeded stream, kept while they solve and pass the
    category's margin, until PER_CATEGORY (no answer best in more than MAX_SHARE of them)."""
    if cat not in _GENERATED:
        rng = random.Random(f'{GENERATOR_SEED}:{cat}')
        kept, best, i = [], {}, 0
        while len(kept) < PER_CATEGORY and i < 5000:
            d = _candidate(cat, rng, i)
            i += 1
            try:
                s = _solve(d)
            except ValueError:
                continue
            if not s['clear_cut'] or best.get(s['correct'], 0) >= MAX_SHARE * PER_CATEGORY:
                continue
            best[s['correct']] = best.get(s['correct'], 0) + 1
            _SOLVED[d['id']] = s
            kept.append(d)
        _GENERATED[cat] = kept
    return _GENERATED[cat]


def suite(categories=None) -> list[dict]:
    """Every spot definition, basics first, in category order."""
    cats = [c for c in CATEGORIES if not categories or c in categories]
    return [d for c in cats for d in ([_basic(t) for t in SPOTS] if c == 'basics' else generated(c))]


def spots(categories=None) -> list[dict]:
    """The quiz: every accepted spot with its computed answer."""
    return [s for s in (solve(d) for d in suite(categories)) if s['clear_cut']]


def spots_hash() -> str:
    return hashlib.sha256(json.dumps([solve(d) for d in suite()], sort_keys=True).encode()).hexdigest()[:16]


SYSTEM = """You are taking a quiz of single poker decisions: heads-up No-Limit Texas Hold'em, blinds 1/2. Each question is one decision in one hand, separate from every other question.

Each question gives the hand so far, your cards, the board and the stacks, and tells you exactly what Villain holds or how Villain plays: either Villain's range (Villain holds each combination in it equally often, except combinations that use cards you can see) or the exact rules Villain follows, and what Villain will do next where it matters. Take all of it as true. Choose the option with the highest expected value in chips; nothing else counts (no table image, no later hands).

Cards are written as rank then suit: A K Q J T 9 ... 2, and s h d c for spades, hearts, diamonds, clubs.

Reply with only a JSON object (no code fence, no other text):
{"reasoning": "your brief reasoning", "action": "fold" | "check" | "call" | "raise" | "all_in", "amount": null unless you raise or bet (then the total you bet this street), "say": "", "notes": null}"""


def _option_word(o: str) -> str:
    return {'fold': '"fold"', 'call': '"call"', 'all_in': '"all_in"', 'check': '"check"'}.get(o) or \
        f'a bet of {o.split("_")[1]} ("raise" with "amount": {o.split("_")[1]})'


def question(spot: dict, ep, hero) -> str:
    opts = ', '.join(_option_word(o) for o in spot['options'][:-1]) + ' or ' + _option_word(spot['options'][-1])
    lines = [f'Question {spot["id"]}. Stacks were {spot["stack_bb"]} big blinds ({2 * spot["stack_bb"]} chips) each '
             f'at the start of the hand.']
    kind = spot['kind']
    if kind == 'shove':
        lines.append(f'Villain, in the big blind, calls an all-in with exactly this range and folds every other '
                     f'hand: {spot["villain_range"]}. Their hand is otherwise random.')
    elif kind == 'inference':
        line = spot['line']
        lines.append(f'Villain plays this hand by fixed rules, and you know them. Preflop, on the button, it raised to '
                     f'3 bb with exactly these hands and folds every other: {spot["villain_range"]}. On the flop, after '
                     f'your check, it bets with {rule_text(line["flop"])}, and checks everything else. On the turn, '
                     f'after your check, it bets with {rule_text(line["turn"])}, and checks everything else. On the '
                     f'river, after your check, it goes all in with {rule_text(line["river"])}, and checks everything '
                     f'else. {FEATURE_NOTE} Before the flop every hand in its raising range was equally likely '
                     f'(except combinations that use cards you can see).')
    else:
        lines.append(f'Villain\'s range here is exactly: {spot["villain_range"]} '
                     f'({spot["villain_combos"]} combinations possible with the cards you see).')
    if kind == 'multistreet':
        lines.append(f'Villain\'s plan is fixed. If you call, the river is dealt, you check, and Villain goes all in '
                     f'when its hand on the final board is {rule_text(spot["river_plan"])}; otherwise it checks behind '
                     f'and the hand goes to showdown. Facing that all-in you may call or fold. If you go all in now, '
                     f'Villain calls when its hand on this board is {rule_text(spot["shove_call"])}, and folds '
                     f'otherwise. {FEATURE_NOTE}')
    if kind == 'sizing':
        said = [f'a bet of {spot["all_in_amount"] if o == "all_in" else o.split("_")[1]}'
                + (' (all in)' if o == 'all_in' else '') + f' with {rule_text(r)}' for o, r in spot['responses'].items()]
        lines.append('Villain never bets or raises on this river: if you check, it checks behind and the hand goes '
                     'to showdown. If you bet, Villain calls ' + '; '.join(said) + ', and folds everything else. '
                     + FEATURE_NOTE)
    lines.append(f'In this question your only choices are {opts}.')
    table = [l for l in ep.request(hero).prompt.splitlines() if not l.startswith('Your options:')]
    table[0] = re.sub(r'^Hand \d+ of \d+ · ', '', table[0])
    table[0] = table[0][:1].upper() + table[0][1:]
    if ep.history:
        who = lambda i: 'you' if i == hero else 'Villain'
        said = []
        for h in ep.history:
            verb = {'bet': f'bet {h.get("amount")}', 'raise': f'raised to {h.get("amount")}',
                    'call': f'called {h.get("amount")}', 'check': 'checked', 'fold': 'folded'}[h['action']]
            said.append(f'{h["street"]}: {who(h["seat"])} {verb}' + (' (all in)' if h.get('all_in') else ''))
        table.insert(1, 'The hand so far (after the blinds of 1 and 2): ' + '; '.join(said) + '.')
    tc, rb = ep.to_call(hero), ep.raise_bounds(hero)
    choice = {'fold': 'fold', 'check': 'check', 'call': f'call {tc}' + (' (all in)' if tc == ep.stacks[hero] else ''),
              'all_in': f'all_in (to {rb[1] if rb else ep.stacks[hero]})'}
    table.append('Your options: ' + '; '.join(choice.get(o) or f'raise to {o.split("_")[1]}' for o in spot['options'])
                 + '.')
    return '\n'.join(lines) + '\n\n' + '\n'.join(table)


def _executed(ep, hero, action) -> tuple[str | None, str | None]:
    """(option the action amounts to, or None; the engine's error, if any)."""
    trial = copy.deepcopy(ep)
    err = trial.apply(hero, action)
    if err:
        return None, err
    h = trial.history[-1]
    kind = h['action']
    if kind in ('bet', 'raise'):
        return ('all_in' if h.get('all_in') else f'bet_{h.get("amount")}' if not h.get('correction') else 'raise'), None
    return kind, None


def _answer_audit(spec: dict, calls: list) -> tuple[str, str, list]:
    """(identity_match, audit, problems) of one answer: every call re-checked against its raw output."""
    from .backends import verify_call
    if 'bot' in spec:
        return 'bot', 'bot', []
    if not calls:
        return 'none', 'no_reply', ['no call was made']
    idents, audits, problems = set(), set(), []
    for c in calls:
        if c.get('backend_failed'):
            audits.add('no_reply')
            problems.append(f'attempt {c.get("attempt")}: the backend gave no reply')
            continue
        res = verify_call(spec.get('backend'), c, c.get('raw'))
        audits.add(res['status'])
        problems += [f'attempt {c.get("attempt")}: {p}' for p in res['problems']]
        idents.add(c.get('identity_match') if c.get('identity_match') == 'mismatch' else
                   res.get('identity_match') or c.get('identity_match') or 'unattested')
    ident = 'mismatch' if 'mismatch' in idents else 'unattested' if 'unattested' in idents or not idents else 'match'
    audit = ('verified' if audits == {'verified'} else 'opaque' if audits <= {'verified', 'opaque'} else
             next(a for a in ('inconsistent', 'unverifiable', 'unverified', 'no_reply', 'opaque') if a in audits))
    return ident, audit, problems


async def ask(pid: str, spec: dict, spot_def, seed: int = 0) -> dict:
    """One answer: the player's reply to one spot, in a fresh conversation, with every call's raw output."""
    from .agents import make_agent
    from .backends import BackendError
    spot = solve(spot_def)
    ep, hero, _, _ = build(spot_def)
    agent = make_agent(pid, spec, seed=seed)
    agent.start(SYSTEM, get_game('holdem')().response_schema())

    def validate(action):
        option, err = _executed(ep, hero, action)
        if err:
            return err
        if option not in spot['options']:
            return (f'{option!r} is not one of this question\'s choices; choose '
                    f'{" or ".join(spot["options"])}' + (' (a bet is "raise" with that "amount")'
                                                          if spot['kind'] == 'sizing' else '') + '.')
        return None

    req = ep.request(hero)
    req = Request(question(spot, ep, hero), [o for o in req.legal if o in spot['options'] or
                                              (o == 'raise' and any(x == 'all_in' or x.startswith('bet_')
                                                                    for x in spot['options']))], req.data)
    t = time.time()
    error = None
    try:
        d = await agent.act(req.prompt, req, validate)
    except BackendError as e:
        from .agents import Decision
        d, error = Decision({}, calls=list(getattr(e, 'calls', [])), fallback=True), str(e)[:500]
    fallback = d.fallback or bool(validate(d.action))
    action = ep.fallback(hero) if fallback else d.action
    option, _ = _executed(ep, hero, action)
    ev = spot['ev_bb']
    loss = round(max(ev.values()) - (min(ev.values()) if fallback else ev.get(option, min(ev.values()))), 3)
    ident, audit, problems = _answer_audit(spec, d.calls)
    say = str(d.action.get('say') or '') if d.action else ''
    return {'spot': spot['id'], 'category': spot['category'],
            'requested': {'action': d.action.get('action'), 'amount': d.action.get('amount')} if d.action else None,
            'executed': option, 'action': option, 'correct': not fallback and option == spot['correct'],
            'outcome': 'error' if error else 'fallback' if fallback else 'valid',
            'ev_loss_bb': loss, 'ev_lost_bb': loss, 'say': say, 'fallback': fallback, 'reasoning': d.reasoning,
            'errors': d.errors + ([{'error': error, 'kind': 'backend'}] if error else []),
            'seconds': round(time.time() - t, 1), 'identity_match': ident, 'audit': audit,
            'audit_problems': problems, 'calls': d.calls}


def score(answers: list[dict]) -> dict:
    """Accuracy over every spot asked (a fallback is wrong), and apart from it the accuracy of the valid answers."""
    n = len(answers)
    valid = [a for a in answers if a.get('outcome', 'fallback' if a.get('fallback') else 'valid') == 'valid']
    right = sum(bool(a['correct']) for a in valid)
    loss = round(sum(a['ev_loss_bb'] for a in answers), 3)
    vloss = round(sum(a['ev_loss_bb'] for a in valid), 3)
    return {'answered': n, 'total': n, 'correct': right, 'accuracy': round(right / n, 4) if n else None,
            'score': round(right / n, 4) if n else None, 'ev_loss_bb': loss, 'ev_lost_bb': loss,
            'mean_ev_loss_bb': round(loss / n, 3) if n else None, 'fallbacks': n - len(valid),
            'fallback_rate': round((n - len(valid)) / n, 4) if n else None, 'valid_answers': len(valid),
            'valid_correct': right, 'valid_accuracy': round(right / len(valid), 4) if valid else None,
            'valid_ev_loss_bb': vloss, 'valid_mean_ev_loss_bb': round(vloss / len(valid), 3) if valid else None}


def protocol(agent) -> dict:
    """What a quiz score depends on: the quiz version, its system prompt, one fresh conversation per spot, and the
    player's effective settings (runner.protocol's terms). `fingerprint` is the sha1 of it."""
    settings = json.loads(json.dumps({'quiz': VERSION, 'system_prompt_sha1': hashlib.sha1(SYSTEM.encode()).hexdigest()[:12],
                                      'conversation': 'a fresh one per spot', 'player': agent.protocol()}, default=str))
    return {'fingerprint': hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16],
            'settings': settings}


def implementation() -> dict:
    from . import runner
    impl = runner.implementation()
    src = runner.sources().get('poker_arena/quiz.py', b'')
    out = {**impl, 'quiz': hashlib.sha256(src).hexdigest()[:16]}
    out['fingerprint'] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]
    return out


def eligibility(kind: str, answers: list[dict], cli_start: dict | None) -> dict:
    """'eligible' only when every answer is a model's, from the requested model, every call verified against its
    raw output, and the CLI the same file on every call; 'bot' for bots; else 'ineligible' with the reasons."""
    from .backends import artifact_changed
    if kind == 'bot':
        return {'status': 'bot', 'reasons': [], 'identity_match': 'bot', 'audit': 'bot'}
    reasons = []
    idents = {a['identity_match'] for a in answers}
    audits = {a['audit'] for a in answers}
    ident = 'mismatch' if 'mismatch' in idents else 'match' if idents == {'match'} else 'unattested'
    audit = 'verified' if audits == {'verified'} else 'opaque' if audits <= {'verified', 'opaque'} else 'incomplete'
    if ident != 'match':
        n = sum(a['identity_match'] != 'match' for a in answers)
        reasons.append(f'{n} answer(s) not attested as the requested model ({", ".join(sorted(idents - {"match"}))})')
    if audit != 'verified':
        n = sum(a['audit'] != 'verified' for a in answers)
        reasons.append(f'{n} answer(s) not verified against their raw output ({", ".join(sorted(audits - {"verified"}))})')
    moved = sorted({str((c.get('cli') or {}).get('sha256'))[:12] for a in answers for c in a['calls']
                    if c.get('cli') and artifact_changed(cli_start, c['cli'])})
    if moved:
        reasons.append(f'the CLI changed during the quiz (now {", ".join(moved)})')
    return {'status': 'ineligible' if reasons else 'eligible', 'reasons': reasons, 'identity_match': ident,
            'audit': audit}


async def run_quiz(players: dict, ids: list[str], out, *, limit: int | None = None, categories=None,
                   concurrency: int = 4, log=print) -> dict:
    """Ask every player in `ids` every spot of the chosen categories (the first `limit` of each), each in its own
    conversation, and write <out>/quiz.json (existing players' results are kept, a player asked again is
    replaced)."""
    from .agents import make_agent
    from .backends import backend_exe, cli_artifact
    from .runner import _public, _write
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    unknown = [c for c in categories or [] if c not in CATEGORIES]
    if unknown:
        raise ValueError(f'unknown quiz categories: {", ".join(unknown)}; known: {", ".join(CATEGORIES)}')
    every = spots()
    defs = [d for c in CATEGORIES if not categories or c in categories
            for d in [x for x in suite([c]) if solve(x)['clear_cut']][:limit]]
    path = out / 'quiz.json'
    doc = json.loads(path.read_text()) if path.exists() else {}
    if doc.get('spots_hash') not in (None, spots_hash()):
        doc = {'replaced': {'version': doc.get('version'), 'spots_hash': doc.get('spots_hash')}}
    counts = {c: sum(s['category'] == c for s in every) for c in CATEGORIES}
    doc.update(version=VERSION, suite={'name': 'Poker fundamentals', 'version': VERSION}, spots_hash=spots_hash(),
               label='poker fundamentals',
               method=f'equity against what Villain holds: exact from the flop on, else Monte Carlo ({SIMS} draws, '
                      f'fixed seed); basics clear cut (gap >= {MIN_GAP_BB} bb, equity gap >= {MIN_EQUITY_GAP}), close '
                      f'spots {CLOSE_GAP[0]:g} to {CLOSE_GAP[1]:g} bb, the other generated categories '
                      f'{HARD_GAP_BB[0]:g} to {HARD_GAP_BB[1]:g} bb between the best option and the next (inference: '
                      f'equity {INFERENCE_EQUITY_GAP[0]:g} to {INFERENCE_EQUITY_GAP[1]:g} from the price)',
               categories={c: {**meta, 'spots': counts[c],
                               **({'generator': {'seed': GENERATOR_SEED, 'per_category': PER_CATEGORY,
                                                 'max_share_of_one_answer': MAX_SHARE}} if meta['population'] else {})}
                           for c, meta in CATEGORIES.items()},
               protocol={'version': VERSION, 'system_prompt_sha1': hashlib.sha1(SYSTEM.encode()).hexdigest()[:12],
                         'conversation': 'a fresh one per spot', 'scoring': 'per category; the top-level score is '
                         'the basics\'; a fallback (no playable answer after the retries) or a backend failure is '
                         'never correct and its loss is the worst option\'s; valid-answer accuracy is reported apart',
                         'eligibility': 'every answer from the requested model, every call verified against its raw '
                                        'output, the CLI unchanged'},
               implementation=implementation(), spots=every)
    doc.setdefault('players', {})
    sem = asyncio.Semaphore(concurrency)

    async def one(pid, k, x):
        async with sem:
            a = await ask(pid, players[pid], x, seed=k)
            log(f'{pid} {a["spot"]}: {a["executed"]} ({"right" if a["correct"] else a["outcome"] if a["fallback"] else "wrong"}, '
                f'-{a["ev_loss_bb"]} bb)')
            return a

    for pid in ids:
        spec = players[pid]
        agent = make_agent(pid, spec)
        agent.start(SYSTEM, get_game('holdem')().response_schema())
        kind = 'bot' if 'bot' in spec else 'model'
        cli = cli_artifact(backend_exe(spec)) if kind == 'model' else None
        started = round(time.time())
        answers = list(await asyncio.gather(*(one(pid, k, x) for k, x in enumerate(defs))))
        if spec.get('bot') == 'reference':
            doc['reference'] = sorted({*doc.get('reference', []), pid})
        by_cat = {c: score([a for a in answers if a['category'] == c]) for c in CATEGORIES
                  if any(a['category'] == c for a in answers)}
        head = [a for a in answers if a['category'] == 'basics'] or answers
        elig = eligibility(kind, answers, cli)
        doc['players'][pid] = {'spec': _public(spec), 'effective': agent.describe(), 'kind': kind,
                               'started': started, 'finished': round(time.time()), **score(head),
                               'scored_category': 'basics' if 'basics' in by_cat else 'all',
                               'categories': by_cat, 'protocol': protocol(agent), 'implementation': doc['implementation'],
                               'cli': cli, 'cli_version': (cli or {}).get('version'),
                               'identity_match': elig['identity_match'], 'audit_status': elig['audit'],
                               'eligibility': elig, 'answers': answers}
        _write(path, doc, indent=2)
    return doc


def main(args):
    from .runner import load_players
    players = load_players(args.players, effort=args.effort)
    missing = [p for p in args.ids if p not in players]
    if missing:
        raise SystemExit(f'not in {args.players}: {", ".join(missing)}')
    out = Path(args.out) if '/' in str(args.out) else Path('runs') / args.out
    doc = asyncio.run(run_quiz(players, args.ids, out, limit=args.limit, categories=args.category,
                               concurrency=args.concurrency))
    for pid in args.ids:
        r = doc['players'][pid]
        cats = ', '.join(f'{c} {x["correct"]}/{x["answered"]}' for c, x in r['categories'].items())
        print(f'{pid}: {cats}; {r["fallbacks"]} fallbacks in the headline category; '
              f'{r["eligibility"]["status"]} -> {out / "quiz.json"}')
