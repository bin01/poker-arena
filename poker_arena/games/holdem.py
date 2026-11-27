"""No-Limit Texas Hold'em, 2-9 seats (the eval plays heads-up), stacks reset every hand.

Every hand is dealt from its own seed, so a duplicate match deals the same cards to the same seats
and only the models in those seats change. The button moves one seat each hand.

Actions (the `action` field of a reply): fold, check, call, raise (with `amount`: the total you
bet this street, "raise to"; a first bet on a street is also a raise), all_in. `HoldemHand.apply` has the
correction policy: strategically different or unplayable replies are sent back, fixable sizes are adjusted.
"""
from __future__ import annotations
import math
import random

from ..cards import shuffled, evaluate, describe
from . import register
from .base import Game, Episode, Event, Request

STREETS = ['preflop', 'flop', 'turn', 'river']
ACTIONS = ['fold', 'check', 'call', 'raise', 'all_in']


def whole_chips(x):
    """(chips, None) for a usable amount, else (None, why not). Infinity, NaN, booleans and text are refused."""
    if isinstance(x, bool) or x is None:
        return None, 'The amount is missing.' if x is None else f'{x!r} is not an amount.'
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None, f'{x!r} is not a number.'
    if not math.isfinite(f):
        return None, f'{x!r} is not a finite number.'
    return int(round(f)), None


def json_safe(x):
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    return str(x)


def cards_text(cards):
    return ' '.join(cards) if cards else '(none)'


class HoldemHand(Episode):
    def __init__(self, game: 'Holdem', index: int, seed: int, names: list[str]):
        self.game, self.index, self.names = game, index, names
        n = self.n = len(names)
        self.sb, self.bb, self.start = game.sb, game.bb, game.stack
        self.rng = random.Random(seed)
        deck = shuffled(self.rng)
        self.button = index % n
        self.hole = [[deck[i], deck[i + n]] for i in range(n)]
        self.deck_board = deck[2 * n: 2 * n + 5]
        self.board: list[str] = []
        self.stacks = [self.start] * n
        self.committed = [0] * n      # this hand
        self.street_bet = [0] * n     # this street
        self.folded = [False] * n
        self.acted = [False] * n
        self.can_raise = [True] * n
        self.street = 0
        self.min_raise = self.bb
        self.events: list[Event] = []
        self.history: list[dict] = []  # actions, for the viewer and bots
        self.done = False
        self.payoffs = [0] * n
        self.result = {}
        self.made_by_street = {'preflop': self.made_hands()}

        self._emit('hand_start', {'hand': index, 'button': self.button, 'names': names, 'stack': self.start,
                                  'sb': self.sb, 'bb': self.bb, 'hole': self.hole, 'seed': seed,
                                  'omniscient': {'made_hand': self.made_by_street['preflop']}},
                   visible=[])  # the full deal is logged for the viewer only
        for i in range(n):
            self._emit('hole', {'seat': i, 'cards': self.hole[i]}, visible=[i])
        if n == 2:
            sb_seat, bb_seat = self.button, (self.button + 1) % n
        else:
            sb_seat, bb_seat = (self.button + 1) % n, (self.button + 2) % n
        self.sb_seat, self.bb_seat = sb_seat, bb_seat
        self._post(sb_seat, self.sb, 'small blind')
        self._post(bb_seat, self.bb, 'big blind')
        self.to_act = sb_seat if n == 2 else self._next_active(bb_seat)
        self._check_round()
        self.events[0].data['to_act'] = None if self.done else self.to_act

    # ---- helpers -------------------------------------------------------------------------------
    def _emit(self, kind, data, visible=None):
        self.events.append(Event(kind, data, visible, self.index))

    def _post(self, seat, amount, label):
        amount = min(amount, self.stacks[seat])
        self._put(seat, amount)
        self._emit('blind', {'seat': seat, 'amount': amount, 'label': label})

    def _put(self, seat, amount):
        self.stacks[seat] -= amount
        self.committed[seat] += amount
        self.street_bet[seat] += amount

    def _in_hand(self):
        return [i for i in range(self.n) if not self.folded[i]]

    def _can_act(self, i):
        return not self.folded[i] and self.stacks[i] > 0

    def _next_active(self, seat):
        for k in range(1, self.n + 1):
            j = (seat + k) % self.n
            if self._can_act(j):
                return j
        return None

    def made_hands(self) -> dict:
        """Every seat's best hand from its cards and the board so far, folded seats included. Omniscient: it
        reads every seat's hole cards, so it is for the viewer, never shown to a seat."""
        return {i: describe(evaluate(self.hole[i] + self.board)) for i in range(self.n)}

    @property
    def pot(self):
        return sum(self.committed)

    @property
    def current_bet(self):
        return max(self.street_bet)

    def to_call(self, seat):
        return min(self.current_bet - self.street_bet[seat], self.stacks[seat])

    def raise_bounds(self, seat):
        """(min_to, max_to) for a raise this street, or None when the seat cannot raise."""
        max_to = self.street_bet[seat] + self.stacks[seat]
        if not self.can_raise[seat] or max_to <= self.current_bet:
            return None
        others = [i for i in self._in_hand() if i != seat and self.stacks[i] > 0]
        if not others:
            return None  # everyone else is all in: nothing to raise against
        min_to = min(self.current_bet + self.min_raise, max_to) if self.current_bet else min(self.bb, max_to)
        return min_to, max_to

    def legal(self, seat):
        out = []
        tc = self.to_call(seat)
        if tc > 0:
            out += ['fold', 'call']
        else:
            out += ['check']
        if self.raise_bounds(seat):
            out += ['raise', 'all_in']
        return out

    # ---- the Episode interface ---------------------------------------------------------------
    def request(self, seat) -> Request:
        g, legal = self.game, self.legal(seat)
        tc, rb = self.to_call(seat), self.raise_bounds(seat)
        opp = [i for i in range(self.n) if i != seat]
        lines = [f'Hand {self.index + 1} of {g.hands_per_match} · {STREETS[self.street]}. '
                 f'You are {self.names[seat]}' + (' on the button' if seat == self.button else '') +
                 (' (small blind)' if seat == self.sb_seat else ' (big blind)' if seat == self.bb_seat else '') + '.',
                 f'Your cards: {cards_text(self.hole[seat])}. Board: {cards_text(self.board)}.',
                 f'Pot: {self.pot} (including this street\'s bets). This street you have put in {self.street_bet[seat]}'
                 + ''.join(f', {self.names[i]} {self.street_bet[i]}' for i in opp) + '.',
                 f'Stacks behind: you {self.stacks[seat]}' + ''.join(
                     f', {self.names[i]} {self.stacks[i]}' + (' (folded)' if self.folded[i] else '') for i in opp) + '.']
        opts = []
        if 'check' in legal:
            opts.append('check')
        if 'fold' in legal:
            opts += ['fold', f'call {tc}' + (' (all in)' if tc == self.stacks[seat] else '')]
        if rb:
            word = 'raise' if self.current_bet else 'bet'
            opts.append(f'{word} to between {rb[0]} and {rb[1]} (action "raise", amount = your total for this street)')
            opts.append(f'all_in (to {rb[1]})')
        lines.append('Your options: ' + '; '.join(opts) + '.')
        return Request('\n'.join(lines), legal, {
            'seat': seat, 'hole': self.hole[seat], 'board': list(self.board), 'street': STREETS[self.street],
            'pot': self.pot, 'to_call': tc, 'raise_bounds': rb, 'stack': self.stacks[seat],
            'street_bet': list(self.street_bet), 'stacks': list(self.stacks), 'bb': self.bb,
            'button': self.button, 'history': list(self.history), 'n': self.n})

    def fallback(self, seat):
        return {'action': 'check' if 'check' in self.legal(seat) else 'fold'}

    def raise_closed(self, seat):
        """The seat may not raise only because the betting wasn't reopened (an all-in short of a full raise)."""
        others = [i for i in self._in_hand() if i != seat and self.stacks[i] > 0]
        return not self.can_raise[seat] and bool(others) and self.street_bet[seat] + self.stacks[seat] > self.current_bet

    def apply(self, seat, action: dict) -> str | None:
        """Apply a seat's reply, or return an error (nothing changed) so the seat is asked again.

        The correction policy: an action that isn't allowed, or that would have to become a strategically
        different action (fold when you can check, raise when the betting is closed, a raise that isn't
        above the current bet, a missing or non-finite amount), is an error. A legal intent whose size can't
        be played is adjusted: a call with nothing to call is a check, a raise when you can only call (all in,
        or against an all-in) is a call, a raise below the minimum is the minimum, one above your stack is
        all in. The logged action then carries `requested`, `correction` = {kind: 'sizing', reason} and
        `note` (the reason, which the seat is shown)."""
        assert not self.done and seat == self.to_act
        kind = str(action.get('action') or '').strip().lower().replace('-', '_').replace(' ', '_')
        kind = {'bet': 'raise', 'allin': 'all_in', 'shove': 'all_in', 'jam': 'all_in'}.get(kind, kind)
        legal = self.legal(seat)
        tc, rb = self.to_call(seat), self.raise_bounds(seat)
        note = None
        if kind not in ACTIONS:
            return f'Unknown action {action.get("action")!r}. Use one of: {", ".join(legal)}.'
        if kind == 'check' and 'check' not in legal:
            return f'You cannot check: there is {tc} to call. Choose fold, call' + (' or raise.' if rb else '.')
        if kind == 'fold' and 'fold' not in legal:
            return 'You cannot fold: there is nothing to call. Choose check' + (' or bet.' if rb else '.')
        if kind == 'call' and 'call' not in legal:
            kind, note = 'check', 'nothing to call: treated as a check'
        if kind in ('raise', 'all_in') and not rb:
            if self.raise_closed(seat):
                return ('You cannot raise: the last all-in was less than a full raise, so the betting is not '
                        'reopened for you. Choose ' + ('fold or call.' if 'call' in legal else 'check.'))
            if 'call' in legal:
                kind, note = 'call', 'you can only call here (all in, or against an all-in): treated as a call'
            else:
                kind, note = 'check', 'you cannot bet here: treated as a check'
        to = None
        if kind == 'raise':
            lo, hi = rb
            asked = action.get('amount')
            to, why = whole_chips(asked)
            if why:
                return f'{why} A raise needs "amount": your total for this street, a whole number from {lo} to {hi}.'
            if isinstance(asked, float) and asked != to:
                note = f'{asked} rounded to {to}'
            if to > hi:
                to, note = hi, f'{asked} is more than you have: all in for {hi}'
            elif to <= self.current_bet:
                return (f'A raise to {to} is not a raise: the bet this street is already {self.current_bet}. '
                        f'"amount" is your total for this street (raise to), from {lo} to {hi}.')
            elif to < lo:
                to, note = lo, f'{asked} is below the minimum: raised to {lo}'

        say = str(action.get('say') or '').strip()
        pot_before = self.pot
        entry = {'seat': seat, 'street': STREETS[self.street], 'action': kind, 'say': say, 'to_call': tc,
                 'pot_before': pot_before, 'pot_odds': round(tc / (pot_before + tc), 4) if tc else None}
        if kind == 'fold':
            self.folded[seat] = True
        elif kind == 'check':
            pass
        elif kind == 'call':
            self._put(seat, tc)
            entry['amount'] = tc
            if self.stacks[seat] == 0:
                entry['all_in'] = True
        else:
            if kind == 'all_in':
                to = rb[1]
            increment = to - self.current_bet
            full = increment >= self.min_raise
            opening = self.current_bet == 0
            self._put(seat, to - self.street_bet[seat])
            entry['action'], entry['amount'] = ('bet' if opening else 'raise'), to
            if self.stacks[seat] == 0:
                entry['all_in'] = True
            for j in range(self.n):
                if j != seat:
                    self.acted[j] = False
                    if full:
                        self.can_raise[j] = True
            if full:
                self.min_raise = increment
        if note:
            entry['note'] = note
            entry['requested'] = {'action': json_safe(action.get('action')), 'amount': json_safe(action.get('amount'))}
            entry['correction'] = {'kind': 'sizing', 'reason': note}
        self.acted[seat] = True
        self.can_raise[seat] = False
        entry['pot'] = self.pot
        self.history.append(entry)
        self._emit('action', entry)
        self._advance(seat)
        entry['to_act'] = None if self.done else self.to_act
        return None

    # ---- flow --------------------------------------------------------------------------------
    def _round_over(self):
        live = [i for i in self._in_hand() if self.stacks[i] > 0]
        return all(self.acted[i] and self.street_bet[i] == self.current_bet for i in live)

    def _advance(self, seat):
        if len(self._in_hand()) == 1:
            return self._finish()
        if self._round_over():
            return self._next_street()
        self.to_act = self._next_active(seat)

    def _check_round(self):
        """After the blinds: if nobody can act (everyone all in from the blinds), run it out."""
        live = [i for i in self._in_hand() if self.stacks[i] > 0]
        if len(live) == 0 or (len(live) == 1 and self.street_bet[live[0]] >= self.current_bet):
            self._runout()

    def _next_street(self):
        live = [i for i in self._in_hand() if self.stacks[i] > 0]
        if len(live) <= 1 or self.street == 3:
            if self.street == 3:
                return self._showdown()
            return self._runout()
        self.street += 1
        self._deal_board()
        self.street_bet = [0] * self.n
        self.acted = [False] * self.n
        self.can_raise = [True] * self.n
        self.min_raise = self.bb
        self.to_act = self._next_active(self.button)  # first seat left of the button (heads-up: the big blind)

    def _deal_board(self):
        count = {1: 3, 2: 4, 3: 5}[self.street]
        new = self.deck_board[len(self.board):count]
        self.board = self.deck_board[:count]
        self.made_by_street[STREETS[self.street]] = self.made_hands()
        self._emit('board', {'street': STREETS[self.street], 'cards': new, 'board': list(self.board),
                             'omniscient': {'made_hand': self.made_by_street[STREETS[self.street]]}})

    def _runout(self):
        if len(self._in_hand()) > 1:
            self._emit('runout', {'seats': self._in_hand(),
                                  'hands': {i: self.hole[i] for i in self._in_hand()}})
        while self.street < 3:
            self.street += 1
            self._deal_board()
        self._showdown()

    def _showdown(self):
        contenders = self._in_hand()
        scores = {i: evaluate(self.hole[i] + self.board) for i in contenders}
        self._emit('showdown', {'hands': {i: {'cards': self.hole[i], 'best': describe(scores[i])}
                                          for i in contenders}, 'board': list(self.board)})
        self._award(scores)

    def _finish(self):
        self._award(None)

    def _award(self, scores):
        """Main and side pots by contribution level; ties split, odd chips to the first seat left of the button."""
        contenders = self._in_hand()
        won = [0] * self.n
        levels = sorted({self.committed[i] for i in contenders if self.committed[i] > 0})
        prev = 0
        pots = []
        for lvl in levels:
            size = sum(min(c, lvl) - min(c, prev) for c in self.committed)
            eligible = [i for i in contenders if self.committed[i] >= lvl]
            pots.append((size, eligible))
            prev = lvl
        leftover = sum(self.committed) - sum(p for p, _ in pots)  # folded chips above every contender's level
        if leftover:
            pots[-1] = (pots[-1][0] + leftover, pots[-1][1])
        order = [(self.button + k) % self.n for k in range(1, self.n + 1)]
        for size, eligible in pots:
            if scores is None or len(eligible) == 1:
                winners = eligible
            else:
                best = max(scores[i] for i in eligible)
                winners = [i for i in eligible if scores[i] == best]
            share, odd = divmod(size, len(winners))
            for i in winners:
                won[i] += share
            for i in sorted(winners, key=order.index)[:odd]:
                won[i] += 1
        for i in range(self.n):
            self.stacks[i] += won[i]
        self.payoffs = [self.stacks[i] - self.start for i in range(self.n)]
        self.result = {'won': won, 'net': self.payoffs, 'board': list(self.board), 'pot': self.pot,
                       'showdown': scores is not None and len(contenders) > 1,
                       'best': {i: describe(scores[i]) for i in contenders} if scores else {}}
        self._emit('result', self.result)
        self.done = True
        self.to_act = None

    # ---- text --------------------------------------------------------------------------------
    def render(self, ev: Event, seat: int) -> str:
        d, N = ev.data, self.names
        you = lambda i: 'You' if i == seat else N[i]
        if ev.kind == 'hole':
            return (f'--- Hand {self.index + 1} of {self.game.hands_per_match}. {N[self.button]} has the button. '
                    f'Stacks reset to {self.start} each. Your cards: {cards_text(d["cards"])}.')
        if ev.kind == 'blind':
            return f'{you(d["seat"])} post{"" if d["seat"] == seat else "s"} the {d["label"]} ({d["amount"]}).'
        if ev.kind == 'action':
            i = d['seat']
            verb = {'fold': 'fold', 'check': 'check', 'call': f'call {d.get("amount", 0)}',
                    'bet': f'bet {d.get("amount")}', 'raise': f'raise to {d.get("amount")}'}[d['action']]
            s = '' if i == seat else 's'
            words = verb.split(' ', 1)
            text = f'{you(i)} {words[0]}{s}' + (f' {words[1]}' if len(words) > 1 else '')
            if d.get('all_in'):
                text += ' (all in)'
            if d.get('say'):
                text += f', saying: "{d["say"]}"'
            if d.get('note') and i == seat:
                text += f' [note: {d["note"]}]'
            return text + '.'
        if ev.kind == 'board':
            return f'{d["street"].capitalize()}: {cards_text(d["cards"])} (board {cards_text(d["board"])}).'
        if ev.kind == 'runout':
            return 'All in. Hands: ' + ', '.join(f'{you(int(i))} {cards_text(c)}' for i, c in d['hands'].items()) + '.'
        if ev.kind == 'showdown':
            return 'Showdown: ' + '; '.join(f'{you(int(i))} {cards_text(h["cards"])} ({h["best"]})'
                                            for i, h in d['hands'].items()) + '.'
        if ev.kind == 'result':
            winners = [i for i, w in enumerate(d['won']) if w > 0]
            parts = [f'{you(i)} win{"" if i == seat else "s"} {d["won"][i]}' for i in winners]
            net = d['net'][seat]
            return f'Hand {self.index + 1} over: ' + ', '.join(parts) + f'. Your result this hand: {net:+d}.'
        return ''

    def summary(self):
        return {'hand': self.index, 'button': self.button, 'hole': self.hole, 'board': list(self.board),
                'actions': self.history, **self.result, 'omniscient': {'made_hand': self.made_by_street}}


@register
class Holdem(Game):
    name = 'holdem'
    min_seats, max_seats = 2, 9
    unit = 'chips'

    def __init__(self, config=None):
        super().__init__({'sb': 1, 'bb': 2, 'stack': 200, 'hands_per_match': 50, **(config or {})})
        self.sb, self.bb, self.stack = self.config['sb'], self.config['bb'], self.config['stack']
        self.hands_per_match = self.config['hands_per_match']
        self.big_unit = self.bb

    def episode(self, index, seed, names):
        return HoldemHand(self, index, seed, names)

    def response_schema(self):
        return {
            'type': 'object', 'additionalProperties': False,
            'required': ['reasoning', 'action', 'amount', 'say', 'notes'],
            'properties': {
                'reasoning': {'type': 'string'},
                'action': {'type': 'string', 'enum': ACTIONS},
                'amount': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
                'say': {'type': 'string'},
                'notes': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            }}

    def system_prompt(self, name, names, episodes):
        others = [n for n in names if n != name]
        opp = others[0] if len(others) == 1 else ', '.join(others)
        bbs = self.stack // self.bb
        return f"""You are {name}, playing a match of No-Limit Texas Hold'em against {opp}. The match is {episodes} hands. Play to win as many chips as you can over the whole match.

THE GAME
- Heads-up No-Limit Hold'em with blinds {self.sb}/{self.bb}. Both stacks are reset to {self.stack} chips ({bbs} big blinds) at the start of every hand, so each hand is won or lost on its own, and your score is the total you win across all hands.
- The button moves every hand. The button posts the small blind and acts first before the flop, and last on the flop, turn and river.
- Standard rules: a raise must be at least the size of the previous bet or raise (minimum {self.bb}), you may always go all in, and the best five-card hand from your two cards and the five board cards wins at showdown. Ties split the pot.
- Cards are written as rank then suit: A K Q J T 9 ... 2, and s h d c for spades, hearts, diamonds, clubs. "Th" is the ten of hearts.

THE TABLE
- {opp} is another AI player. You will play the whole match against them, so you can learn their tendencies and they can learn yours.
- Each turn you get everything that happened since your last turn, then a short summary of the hand so far and your legal options.
- Table talk: anything you put in "say" is heard by {opp} along with your action. You can talk, needle, bluff or stay quiet. Keep it short (under 40 words), or "" to say nothing.
- Your "reasoning" and "notes" are private: {opp} never sees them.
- Notes are your notebook for the match: write what you have learned about {opp}'s play that you want to remember (for example how often they bluff, how they size bets, how they react to aggression). What you write replaces your previous notes, and your latest notes are shown to you at the start of every hand. Use null to keep them as they are.

YOUR RESPONSE
Reply with only a JSON object (no code fence, no other text):
{{"reasoning": "your private thinking about this decision, brief", "action": "fold" | "check" | "call" | "raise" | "all_in", "amount": the total you are betting this street when you raise or bet (an integer), otherwise null, "say": "table talk" or "", "notes": "your updated notes" or null}}
To bet when nobody has bet yet, use "raise" with the amount of your bet."""
