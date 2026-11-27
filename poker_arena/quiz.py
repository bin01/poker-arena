"""Poker fundamentals: a fixed quiz of heads-up decision spots with objectively correct answers.

Every spot is a real position of the engine (blinds 1/2, a stated stack, the actions so far) in which the
opponent's range is stated exactly, and the decision is one whose value can be computed: a call that ends the
betting (against an all-in, or a river bet with raising ruled out) or an all-in-or-fold first decision against a
stated calling range. The value of each option, in big blinds relative to folding, comes from the hand's equity
against the range: exact (every combo, every runout) from the turn on, Monte Carlo with SIMS draws from a fixed
seed before it. Only clear-cut spots are kept: the best option beats the other by at least MIN_GAP_BB, and for
calls the equity differs from the price by at least MIN_EQUITY_GAP.

A player answers each spot in a fresh conversation, through its usual backend, with the game's JSON reply;
the answer is checked by the engine (`apply`) and must be one of the spot's options, else it is sent back
(max_fixes times, then the engine's fallback, a fold). Bots answer from the same request data they play from.
Results go to <out>/quiz.json: the spots with their computed values, and per player every answer, whether it
was correct, its loss in big blinds against the best option, and its calls. The score is reported apart from
match results, as "poker fundamentals".
"""
from __future__ import annotations
import asyncio
import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path

from .games import get_game, ranges
from .games.base import Request
from .cards import DECK

VERSION = 'quiz-1'
SIMS = 40000
MIN_GAP_BB = 1.0
MIN_EQUITY_GAP = 0.04
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


def _cards(text: str) -> list[str]:
    return text.split()


def _game(stack_bb: int):
    return get_game('holdem')({'sb': 1, 'bb': 2, 'stack': 2 * stack_bb, 'hands_per_match': 1})


def build(spot: tuple):
    """The engine's hand at the spot's decision: (episode, hero's seat, villain's seat, the range's combos)."""
    sid, hero_pos, stack_bb, hole, board, script, rng_text, kind = spot
    game = _game(stack_bb)
    hero = 0
    button = hero if hero_pos == 'button' else 1
    ep = game.episode(button, 0, ['Hero', 'Villain'])
    hole, board = _cards(hole), _cards(board)
    combos = [c for c in ranges.parse(rng_text) if not set(c) & set(hole + board)]
    if not combos:
        raise ValueError(f'{sid}: the range has no combo possible with these cards')
    dummy = list(combos[0])
    ep.hole = [hole, dummy] if hero == 0 else [dummy, hole]
    rest = [c for c in DECK if c not in hole + board + dummy]
    ep.deck_board = board + rest[:5 - len(board)]
    seat_of = {'button': button, 'bb': 1 - button}
    for who, action, amount in script:
        seat = seat_of[who]
        if ep.to_act != seat:
            raise ValueError(f'{sid}: {who} acts out of turn')
        err = ep.apply(seat, {'action': action, 'amount': amount})
        if err or ep.history[-1].get('correction'):
            raise ValueError(f'{sid}: {who} {action} {amount}: {err or ep.history[-1]["note"]}')
    if ep.done or ep.to_act != hero or ep.board != board:
        raise ValueError(f'{sid}: the script does not reach hero\'s decision on that board')
    return ep, hero, 1 - hero, combos


_SOLVED: dict = {}


def solve(spot: tuple) -> dict:
    """The spot's facts and the value of each option in bb relative to folding, from exact or Monte Carlo
    equity against the stated range (computed once per process)."""
    if spot[0] not in _SOLVED:
        _SOLVED[spot[0]] = _solve(spot)
    return _SOLVED[spot[0]]


def _solve(spot: tuple) -> dict:
    sid, hero_pos, stack_bb, hole, board, script, rng_text, kind = spot
    ep, hero, villain, combos = build(spot)
    bb = ep.bb
    hole_c, board_c = _cards(hole), _cards(board)
    rng = random.Random(int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16))
    exact = len(board_c) >= 3
    eq = ranges.equity(hole_c, board_c, combos, rng, sims=SIMS, exact=exact)
    pot, tc = ep.pot, ep.to_call(hero)
    out = {'id': sid, 'kind': kind, 'street': ['preflop', 'flop', 'turn', 'river'][ep.street],
           'position': hero_pos, 'stack_bb': stack_bb, 'hole': hole_c, 'board': board_c,
           'actions': [{k: h[k] for k in ('seat', 'street', 'action') if k in h} | {'amount': h.get('amount')}
                       for h in ep.history],
           'pot_bb': pot / bb, 'to_call_bb': tc / bb, 'villain_range': rng_text, 'villain_combos': len(combos),
           'equity': round(eq, 4), 'equity_method': 'exact' if exact else f'monte carlo, {SIMS} draws'}
    if kind == 'call':
        if not (ep.stacks[villain] == 0 or tc >= ep.stacks[hero] or ep.street == 3):
            raise ValueError(f'{sid}: a call here does not end the betting')
        need = tc / (pot + tc)
        ev = {'fold': 0.0, 'call': (eq * (pot + tc) - tc) / bb}
        out.update(required_equity=round(need, 4), equity_gap=round(eq - need, 4), options=['fold', 'call'])
    else:
        if ep.history or hero != ep.button:
            raise ValueError(f'{sid}: a shove spot is the first decision, on the button')
        S, h, v = ep.start, ep.committed[hero], ep.committed[villain]
        possible = [c for c in ranges.ALL if not set(c) & set(hole_c)]
        p_call = len(combos) / len(possible)
        ev = {'fold': 0.0, 'all_in': ((1 - p_call) * (h + v) + p_call * (eq * 2 * S - (S - h))) / bb}
        out.update(call_probability=round(p_call, 4), options=['fold', 'all_in'])
    out['ev_bb'] = {k: round(x, 3) for k, x in ev.items()}
    best = max(ev, key=ev.get)
    out['correct'] = best
    verb = {'fold': 'fold', 'call': 'call', 'all_in': 'go all in'}
    out.update(title=TITLES.get(sid, sid), best=best, ev=out['ev_bb'], bb=bb, pot=pot, to_call=tc,
               hero_bet=ep.street_bet[hero], villain_bet=ep.street_bet[villain], stack=ep.stacks[hero],
               villain_stack=ep.stacks[villain], button=0 if ep.button == hero else 1,
               situation=(f'Villain is all in for {ep.committed[villain] / bb:g} bb' if ep.stacks[villain] == 0 else
                          f'Villain bets {tc / bb:g} bb into {(pot - tc) / bb:g} bb') if tc and kind == 'call' else
               f'First to act on the button, {stack_bb} bb deep',
               equity_note=f'against the stated range ({out["equity_method"]})')
    if kind == 'call':
        out['needed'] = round(tc / (pot + tc), 4)
        out['explain'] = (f'Calling {tc / bb:g} bb to win {(pot + tc) / bb:g} bb needs {need:.1%} equity; against the '
                          f'stated range the hand has {eq:.1%}, so the best play is to {verb[best]} '
                          f'(calling is worth {ev["call"]:+.2f} bb against folding).')
    else:
        out['explain'] = (f'Villain calls {p_call:.1%} of the time, and then the hand has {eq:.1%} equity; going all in '
                          f'is worth {ev["all_in"]:+.2f} bb against folding, so the best play is to {verb[best]}.')
    out['gap_bb'] = round(abs(ev[best] - min(ev.values())), 3)
    out['clear_cut'] = out['gap_bb'] >= MIN_GAP_BB and (kind != 'call' or abs(out['equity_gap']) >= MIN_EQUITY_GAP)
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


def spots() -> list[dict]:
    """The quiz: every clear-cut spot with its computed answer."""
    return [s for s in (solve(x) for x in SPOTS) if s['clear_cut']]


def spots_hash() -> str:
    return hashlib.sha256(json.dumps([solve(x) for x in SPOTS], sort_keys=True).encode()).hexdigest()[:16]


SYSTEM = """You are taking a quiz of single poker decisions: heads-up No-Limit Texas Hold'em, blinds 1/2. Each question is one decision in one hand, separate from every other question.

Each question gives the hand so far, your cards, the board and the stacks, and states your opponent's (Villain's) range exactly: Villain holds each combination in it equally often (except combinations that use cards you can see). Take the range as true. Choose the option with the highest expected value in chips; nothing else counts (no table image, no later hands).

Cards are written as rank then suit: A K Q J T 9 ... 2, and s h d c for spades, hearts, diamonds, clubs.

Reply with only a JSON object (no code fence, no other text):
{"reasoning": "your brief reasoning", "action": "fold" | "check" | "call" | "raise" | "all_in", "amount": null unless you raise (then the total you bet this street), "say": "", "notes": null}"""


def question(spot: dict, ep, hero) -> str:
    opts = ' or '.join({'fold': '"fold"', 'call': '"call"', 'all_in': '"all_in"'}[o] for o in spot['options'])
    lines = [f'Question {spot["id"]}. Stacks were {spot["stack_bb"]} big blinds ({2 * spot["stack_bb"]} chips) each '
             f'at the start of the hand.']
    if spot['kind'] == 'shove':
        lines.append(f'Villain, in the big blind, calls an all-in with exactly this range and folds every other '
                     f'hand: {spot["villain_range"]}. Their hand is otherwise random.')
    else:
        lines.append(f'Villain\'s range here is exactly: {spot["villain_range"]} '
                     f'({spot["villain_combos"]} combinations possible with the cards you see).')
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
    choice = {'fold': 'fold', 'call': f'call {tc}' + (' (all in)' if tc == ep.stacks[hero] else ''),
              'all_in': f'all_in (to {rb[1] if rb else ep.stacks[hero]})'}
    table.append('Your options: ' + '; '.join(choice[o] for o in spot['options']) + '.')
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
        return ('all_in' if h.get('all_in') else 'raise'), None
    return kind, None


async def ask(pid: str, spec: dict, spot_def: tuple, seed: int = 0) -> dict:
    """One answer: the player's reply to one spot, in a fresh conversation."""
    from .agents import make_agent
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
                    f'{" or ".join(spot["options"])}.')
        return None

    req = ep.request(hero)
    req = Request(question(spot, ep, hero), [o for o in req.legal if o in spot['options'] or
                                              (o == 'raise' and 'all_in' in spot['options'])], req.data)
    t = time.time()
    d = await agent.act(req.prompt, req, validate)
    fallback = d.fallback or bool(validate(d.action))
    action = ep.fallback(hero) if fallback else d.action
    option, _ = _executed(ep, hero, action)
    ev = spot['ev_bb']
    loss = round(max(ev.values()) - ev.get(option, ev['fold']), 3)
    say = str(d.action.get('say') or '') if d.action else ''
    calls = [{k: v for k, v in c.items() if k != 'raw'} for c in d.calls]
    return {'spot': spot['id'], 'requested': {'action': d.action.get('action'), 'amount': d.action.get('amount')}
            if d.action else None, 'executed': option, 'action': option, 'correct': option == spot['correct'],
            'ev_loss_bb': loss, 'ev_lost_bb': loss, 'say': say,
            'fallback': fallback, 'reasoning': d.reasoning, 'errors': d.errors, 'seconds': round(time.time() - t, 1),
            'calls': calls}


def score(answers: list[dict]) -> dict:
    n = len(answers)
    right = sum(a['correct'] for a in answers)
    loss = round(sum(a['ev_loss_bb'] for a in answers), 3)
    return {'answered': n, 'total': n, 'correct': right, 'accuracy': round(right / n, 4) if n else None,
            'score': round(right / n, 4) if n else None, 'ev_loss_bb': loss, 'ev_lost_bb': loss, 'mean_ev_loss_bb': round(loss / n, 3) if n else None,
            'fallbacks': sum(a['fallback'] for a in answers)}


async def run_quiz(players: dict, ids: list[str], out, *, limit: int | None = None, concurrency: int = 4,
                   log=print) -> dict:
    """Ask every player in `ids` every clear-cut spot (the first `limit`), each in its own conversation, and
    write <out>/quiz.json (existing players' results are kept, a player asked again is replaced)."""
    from .agents import make_agent
    from .runner import _public, _write
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    defs = [x for x in SPOTS if solve(x)['clear_cut']][:limit]
    path = out / 'quiz.json'
    doc = json.loads(path.read_text()) if path.exists() else {}
    if doc.get('spots_hash') not in (None, spots_hash()):
        doc = {'replaced': {'version': doc.get('version'), 'spots_hash': doc.get('spots_hash')}}
    doc.update(version=VERSION, suite={'name': 'Poker fundamentals', 'version': VERSION}, spots_hash=spots_hash(),
               label='poker fundamentals',
               method=f'equity against the stated range: exact from the flop on, else Monte Carlo ({SIMS} draws, '
                      f'fixed seed); clear-cut spots only (gap >= {MIN_GAP_BB} bb, equity gap >= {MIN_EQUITY_GAP})',
               spots=[solve(x) for x in defs])
    doc.setdefault('players', {})
    sem = asyncio.Semaphore(concurrency)

    async def one(pid, k, x):
        async with sem:
            a = await ask(pid, players[pid], x, seed=k)
            log(f'{pid} {a["spot"]}: {a["executed"]} ({"right" if a["correct"] else "wrong"}, '
                f'-{a["ev_loss_bb"]} bb)')
            return a

    for pid in ids:
        agent = make_agent(pid, players[pid])
        agent.start(SYSTEM, get_game('holdem')().response_schema())
        started = round(time.time())
        answers = list(await asyncio.gather(*(one(pid, k, x) for k, x in enumerate(defs))))
        if players[pid].get('bot') == 'reference':
            doc['reference'] = sorted({*doc.get('reference', []), pid})
        doc['players'][pid] = {'spec': _public(players[pid]), 'effective': agent.describe(),
                               'kind': 'bot' if 'bot' in players[pid] else 'model',
                               'started': started, 'finished': round(time.time()), **score(answers),
                               'answers': answers}
        _write(path, doc, indent=2)
    return doc


def main(args):
    from .runner import load_players
    players = load_players(args.players, effort=args.effort)
    missing = [p for p in args.ids if p not in players]
    if missing:
        raise SystemExit(f'not in {args.players}: {", ".join(missing)}')
    out = Path(args.out) if '/' in str(args.out) else Path('runs') / args.out
    doc = asyncio.run(run_quiz(players, args.ids, out, limit=args.limit, concurrency=args.concurrency))
    for pid in args.ids:
        r = doc['players'][pid]
        print(f'{pid}: {r["correct"]}/{r["answered"]} right, {r["ev_loss_bb"]} bb lost to the best answers '
              f'({r["fallbacks"]} fallbacks) -> {out / "quiz.json"}')
