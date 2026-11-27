"""Seats: model agents and baseline bots, all with the same `act`.

A model agent keeps one conversation for the whole match (its backend), so it remembers earlier hands,
the table talk and its own replies. Each turn it gets the events it saw since its last turn plus the
request prompt, and answers with the game's JSON (reasoning, action, amount, say, notes). A reply that
can't be parsed or played (the game's `apply` returns an error, which includes strategically different
actions it won't convert) is sent back with the error; after `max_fixes` tries the seat takes the game's
fallback (check or fold). Every attempt is kept in `Decision.calls`, every refusal in `Decision.errors`.
A reply from a turn that ran (or tried) a tool is never used: the turn is taken back, the CLI conversation is
rebuilt without it, and the seat is asked again (an error of kind `tool_use`). So is a reply whose turn could
not be audited (`audit: 'unverified'`, e.g. an agy transcript that can't be read), as an error of kind
`unverified`; if no turn of a decision can be audited the match stops (a BackendError), so it can be resumed
once the record is readable again, rather than playing a move nobody can check. Every call records `audit`.

Bots read the request's structured data and never talk. They cost nothing and anchor the ratings; their
random state is saved with the match, so a resumed match plays exactly as an uninterrupted one.
"""
from __future__ import annotations
import json
import random
import re
from dataclasses import dataclass, field

from .backends import BackendError, make_backend
from .cards import equity, evaluate
from .games.holdem import json_safe


@dataclass
class Decision:
    action: dict
    reasoning: str = ''
    thinking: str = ''
    raw: str = ''
    calls: list = field(default_factory=list)   # one entry per model call made for this decision
    fallback: bool = False
    errors: list = field(default_factory=list)  # [{requested: {action, amount} | None, error, kind}] per refused reply


def parse_json(text: str) -> dict | None:
    """The first JSON object in a reply (code fences and chatter around it are ignored)."""
    if not text:
        return None
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.M)
    start = text.find('{')
    while start != -1:
        depth, in_str, esc = 0, False, False
        for k in range(start, len(text)):
            ch = text[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:k + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = text.find('{', start + 1)
    return None


class Agent:
    kind = 'agent'
    notes: str = ''
    spec: dict = {}

    def start(self, system: str, schema: dict):
        pass

    def describe(self) -> dict:
        """The settings this seat actually plays with, defaults included, for the match's config."""
        return dict(self.spec)

    def protocol(self) -> dict:
        """The settings a player's results depend on, the same in every seat (no seat seeds, no seat names)."""
        return {k: v for k, v in self.describe().items() if k != 'seed'}

    async def act(self, text: str, request, validate) -> Decision:
        raise NotImplementedError

    def state(self):
        return {'notes': self.notes}

    def restore(self, st):
        self.notes = st.get('notes', '')


class ModelAgent(Agent):
    kind = 'model'

    def __init__(self, player_id: str, backend: str, max_fixes: int = 2, max_say_words: int = 40,
                 max_notes_words: int = 250, **backend_kw):
        self.player_id, self.backend_kind, self.backend_kw = player_id, backend, backend_kw
        self.max_fixes, self.max_say_words, self.max_notes_words = max_fixes, max_say_words, max_notes_words
        self.backend = None
        self.notes = ''

    def start(self, system, schema):
        self.backend = make_backend(self.backend_kind, system, label=self.player_id, **self.backend_kw)

    def describe(self):
        return {'backend': self.backend_kind, 'max_fixes': self.max_fixes, 'max_say_words': self.max_say_words,
                'max_notes_words': self.max_notes_words, **(self.backend.describe() if self.backend else self.backend_kw)}

    def protocol(self):
        return {'backend': self.backend_kind, 'max_fixes': self.max_fixes, 'max_say_words': self.max_say_words,
                'max_notes_words': self.max_notes_words, **self.backend.protocol()}

    def state(self):
        return {'notes': self.notes, 'backend': self.backend.state() if self.backend else {}}

    def restore(self, st):
        self.notes = st.get('notes', '')
        if self.backend and st.get('backend'):
            self.backend.restore(st['backend'])

    @staticmethod
    def _cap(text, words):
        w = str(text or '').split()
        return ' '.join(w[:words]) + (' ...' if len(w) > words else '')

    async def act(self, text, request, validate) -> Decision:
        """`validate(action) -> error or None` checks an action without playing it."""
        message = base = text   # base: the message without a tool-use notice
        calls, errors = [], []
        for attempt in range(self.max_fixes + 1):
            try:
                reply = await self.backend.send(message)
            except BackendError as e:  # the CLI gave up: keep what was tried, for the log
                calls.append({'prompt': message, 'output': None, 'attempt': attempt, 'error': str(e)[:2000],
                              'backend_failed': True, 'audit': 'no_reply', **getattr(self.backend, 'identity', dict)(),
                              **getattr(e, 'meta', {})})
                e.calls = calls
                raise
            calls.append({'prompt': message, 'output': reply.text, 'thinking': reply.thinking, 'usage': reply.usage,
                          'latency': reply.latency, 'attempt': attempt, **reply.meta})
            if reply.meta.get('audit') == 'unverified':
                error = 'The turn could not be audited (its record of tool use was unreadable); asked again.'
                calls[-1].update(error=error, rejected='unverified')
                obj = parse_json(reply.text)
                errors.append({'requested': None if obj is None else
                               {'action': json_safe(obj.get('action')), 'amount': json_safe(obj.get('amount'))},
                               'error': error, 'kind': 'unverified'})
                self.backend.forget_last()
                continue
            if reply.meta.get('tool_use'):
                used = ', '.join(sorted({str(t.get('name') or t.get('type')) for t in reply.meta['tool_use']}))
                error = f'Your reply used a tool ({used}). Tools are not allowed in this match.'
                calls[-1].update(error=error, rejected='tool_use')
                obj = parse_json(reply.text)
                errors.append({'requested': None if obj is None else
                               {'action': json_safe(obj.get('action')), 'amount': json_safe(obj.get('amount'))},
                               'error': error, 'kind': 'tool_use'})
                self.backend.forget_last()   # the model never saw that turn happen; nor does the record
                message = f'{error} Decide from the information given here, without any tools.\n\n{base}'
                continue
            obj = parse_json(reply.text)
            if obj is None:
                error = 'Your reply was not a JSON object.'
            else:
                obj['say'] = self._cap(obj.get('say'), self.max_say_words)
                error = validate(obj)
            if not error:
                if obj.get('notes'):
                    self.notes = self._cap(obj['notes'], self.max_notes_words)
                return Decision(obj, str(obj.get('reasoning') or ''), reply.thinking, reply.text, calls, errors=errors)
            calls[-1]['error'] = error
            errors.append({'requested': None if obj is None else
                           {'action': json_safe(obj.get('action')), 'amount': json_safe(obj.get('amount'))},
                           'error': error, 'kind': 'invalid'})
            message = base = (f'That reply could not be played: {error}\nReply again with only the JSON object.\n\n'
                              f'{request.prompt}')
        if calls and all(c.get('rejected') == 'unverified' for c in calls):
            e = BackendError(f'{self.player_id}: none of {len(calls)} turns could be audited; stopping the match')
            e.calls = calls
            raise e
        return Decision({}, '', '', calls[-1]['output'] if calls else '', calls, fallback=True, errors=errors)


# ---- baseline bots ------------------------------------------------------------------------------

class SeededBot(Agent):
    """A bot with its own random generator, saved and restored with the match."""
    kind = 'bot'
    rng: random.Random

    def state(self):
        v, internal, gauss = self.rng.getstate()
        return {'notes': self.notes, 'rng': [v, list(internal), gauss]}

    def restore(self, st):
        super().restore(st)
        if st.get('rng'):
            v, internal, gauss = st['rng']
            self.rng.setstate((v, tuple(internal), gauss))


class RandomBot(SeededBot):
    """Picks a legal action uniformly (never folds when it can check); raises to a random legal size."""
    kind = 'bot'

    def __init__(self, seed=0, **_):
        self.rng = random.Random(seed)

    async def act(self, text, request, validate):
        legal = [a for a in request.legal if not (a == 'fold' and 'check' in request.legal)]
        a = self.rng.choice(legal)
        action = {'action': a}
        if a == 'raise':
            lo, hi = request.data['raise_bounds']
            action['amount'] = self.rng.randint(lo, min(hi, max(lo, lo + request.data['pot'] * 2)))
        return Decision(action, 'random')


class CallingStation(Agent):
    """Checks or calls every bet, never raises or folds."""
    kind = 'bot'

    def __init__(self, **_):
        pass

    async def act(self, text, request, validate):
        return Decision({'action': 'check' if 'check' in request.legal else 'call'}, 'always call')


class RuleBot(SeededBot):
    """A simple equity player: Monte Carlo equity against one random hand, compared with pot odds.
    Value-bets strong hands (about 3/4 pot), calls when the price is right, otherwise checks or folds.
    It does not bluff and does not adapt to its opponent."""
    kind = 'bot'

    def __init__(self, seed=0, sims=300, **_):
        self.rng, self.sims = random.Random(seed), sims

    def describe(self):
        return {**self.spec, 'sims': self.sims}

    async def act(self, text, request, validate):
        d = request.data
        eq = equity(d['hole'], d['board'], self.rng, opponents=d['n'] - 1, sims=self.sims)
        tc, pot, rb, bb = d['to_call'], d['pot'], d['raise_bounds'], d['bb']
        odds = tc / (pot + tc) if tc else 0.0
        why = f'equity {eq:.2f}, pot odds {odds:.2f}'
        if d['street'] == 'preflop':
            strong, ok = 0.60, 0.48
        else:
            strong, ok = 0.72, None
        if rb and eq >= strong:
            current = max(d['street_bet'])
            size = 3 * max(current, bb) if d['street'] == 'preflop' else current + int(0.75 * (pot + tc))
            amount = max(rb[0], min(rb[1], size))
            return Decision({'action': 'raise', 'amount': amount}, why + ': raise')
        if tc == 0:
            return Decision({'action': 'check'}, why + ': check')
        threshold = ok if ok is not None else odds + 0.05
        if d['street'] == 'preflop' and tc <= bb:
            threshold = min(threshold, 0.40)
        if eq >= threshold:
            return Decision({'action': 'call'}, why + ': call')
        return Decision({'action': 'fold'}, why + ': fold')


class ReferenceBot(SeededBot):
    """A fixed, versioned, competent heads-up player: tight-aggressive, position-aware, deciding by equity against
    the range the opponent's actions suggest and the price of the pot. It never adapts to its opponent, so the
    same deal against the same actions always gets the same decision from the same random state.

    Ranges: the opponent's preflop range is the top fraction of hands (ranges.top) its preflop betting implies;
    after the flop every bet or raise it makes keeps the part of that range with the strongest made hands on the
    board (with a share of the rest, for its bluffs). Equity against that range is Monte Carlo with the bot's own
    random generator.

    Preflop: on the button it opens the top 70% to 2.5 big blinds and folds the rest; in the big blind it raises
    the top 30% over a limp. Facing a raise it re-raises its value range (top 12% against an open, 4% against a
    3-bet, 2.5% beyond), re-raises as a bluff with probability BLUFF just outside it against an open, calls when
    its equity, discounted for playing out of position, beats the price, and otherwise folds.
    After the flop: unbet, it bets 2/3 pot with equity >= 0.70 (1/2 pot with >= 0.60 in position), bluffs half pot
    with probability BLUFF with equity < 0.40 (flop and turn only), else checks. Facing a bet it raises to 3x
    with equity >= 0.85, calls with equity >= the pot odds plus a margin (0.05 before the river), else folds.
    A bet or raise of 40% or more of what is left goes all in."""
    kind = 'bot'
    VERSION = 'reference-1'
    BLUFF = 0.2

    def __init__(self, seed=0, sims=400, **_):
        self.rng, self.sims = random.Random(seed), sims

    def describe(self):
        return {**self.spec, 'version': self.VERSION, 'sims': self.sims, 'bluff': self.BLUFF}

    def _opp_range(self, d, opp):
        from .games import ranges
        pre = [h for h in d['history'] if h['street'] == 'preflop']
        raises = [h for h in pre if h['action'] in ('bet', 'raise')]
        theirs = [h for h in raises if h['seat'] == opp]
        acts = [h['action'] for h in pre if h['seat'] == opp]
        if not theirs:
            f = 1.0 if not acts or acts == ['check'] else 0.85 if len(raises) == 0 else 0.6 if len(raises) == 1 else 0.3
        elif len(theirs) == 1:
            f = 0.7 if raises[0]['seat'] == opp else 0.3
        else:
            f = 0.12 if len(theirs) == 2 else 0.05
        if raises and raises[-1]['seat'] == opp and raises[-1].get('all_in'):
            f = min(f, 0.25)
        combos = [c for c in ranges.top(f) if not set(c) & set(d['hole'] + d['board'])]
        weights = [1.0] * len(combos)
        board = d['board']
        if board:
            keep = 1.0
            for h in d['history']:
                if h['seat'] == opp and h['street'] != 'preflop' and h['action'] in ('bet', 'raise'):
                    keep *= 0.6 if h['action'] == 'bet' else 0.4
            if keep < 1.0:
                order = sorted(range(len(combos)), key=lambda i: evaluate(list(combos[i]) + board), reverse=True)
                cut = max(1, int(len(order) * max(keep, 0.1)))
                strong = set(order[:cut])
                weights = [0.85 / cut if i in strong else 0.15 / max(1, len(combos) - cut) for i in range(len(combos))]
        return combos, weights

    def _equity(self, d, opp):
        from .games import ranges
        combos, weights = self._opp_range(d, opp)
        return ranges.equity(d['hole'], d['board'], combos, self.rng, sims=self.sims, weights=weights, exact=False)

    @staticmethod
    def _size(d, to):
        rb = d['raise_bounds']
        to = max(rb[0], min(rb[1], int(to)))
        left = rb[1] - d['street_bet'][d['seat']]
        if to - d['street_bet'][d['seat']] >= 0.4 * left:
            to = rb[1]
        return {'action': 'raise', 'amount': to}

    async def act(self, text, request, validate):
        from .games import ranges
        d = request.data
        seat, n = d['seat'], d['n']
        if n != 2:
            eq = equity(d['hole'], d['board'], self.rng, opponents=n - 1, sims=self.sims)
            return Decision({'action': 'call' if d['to_call'] and eq > 0.5 else 'check' if not d['to_call'] else 'fold'},
                            f'equity {eq:.2f} against random hands')
        opp = 1 - seat
        tc, pot, rb, bb = d['to_call'], d['pot'], d['raise_bounds'], d['bb']
        current = max(d['street_bet'])
        odds = tc / (pot + tc) if tc else 0.0
        in_position = seat == d['button']
        all_in_call = tc >= d['stack'] or d['stacks'][opp] == 0
        if d['street'] == 'preflop':
            pct = ranges.percentile()[ranges.hand_class(d['hole'])]
            raises = sum(h['action'] in ('bet', 'raise') for h in d['history'] if h['street'] == 'preflop')
            why = f'top {pct:.0%} hand, {raises} raise(s) so far'
            if raises == 0 and tc > 0:   # first in on the button
                if pct <= 0.70 and rb:
                    return Decision(self._size(d, 2.5 * bb), why + ': open')
                return Decision({'action': 'fold'}, why + ': fold')
            if raises == 0:              # big blind over a limp
                if pct <= 0.30 and rb:
                    return Decision(self._size(d, 4 * bb), why + ': raise the limp')
                return Decision({'action': 'check'}, why + ': check')
            value = 0.12 if raises == 1 else 0.04 if raises == 2 else 0.025
            if rb and pct <= value:
                return Decision(self._size(d, 3 * current), why + ': re-raise for value')
            if rb and raises == 1 and pct <= 0.35 and self.rng.random() < self.BLUFF:
                return Decision(self._size(d, 3 * current), why + ': re-raise as a bluff')
            eq = self._equity(d, opp)
            realize = 1.0 if all_in_call else 0.9 if in_position else 0.8
            why += f', equity {eq:.2f} vs their range, price {odds:.2f}'
            if eq * realize >= odds:
                return Decision({'action': 'call'}, why + ': call')
            return Decision({'action': 'fold'}, why + ': fold')
        eq = self._equity(d, opp)
        why = f'equity {eq:.2f} vs their range, price {odds:.2f}'
        if tc == 0:
            if rb and eq >= 0.70:
                return Decision(self._size(d, max(bb, 2 * pot / 3)), why + ': value bet')
            if rb and eq >= 0.60 and in_position:
                return Decision(self._size(d, max(bb, pot / 2)), why + ': thin value bet')
            if rb and eq < 0.40 and d['street'] != 'river' and self.rng.random() < self.BLUFF:
                return Decision(self._size(d, max(bb, pot / 2)), why + ': bluff')
            return Decision({'action': 'check'}, why + ': check')
        if rb and eq >= 0.85:
            return Decision(self._size(d, 3 * current), why + ': raise for value')
        margin = 0.0 if d['street'] == 'river' or all_in_call else 0.05
        if eq >= odds + margin:
            return Decision({'action': 'call'}, why + ': call')
        return Decision({'action': 'fold'}, why + ': fold')


BOTS = {'random': RandomBot, 'station': CallingStation, 'rule': RuleBot, 'reference': ReferenceBot}


def make_agent(player_id: str, spec: dict, seed: int = 0) -> Agent:
    """An agent from a player entry in players.toml: {bot = "rule"} or {backend = "claude", model = ...}."""
    spec = dict(spec)
    spec.pop('label', None)
    spec.pop('description', None)
    if 'bot' in spec:
        agent = BOTS[spec['bot']](seed=seed, **{k: v for k, v in spec.items() if k != 'bot'})
        agent.spec = {**spec, 'seed': seed}
        return agent
    if 'backend' not in spec:
        raise ValueError(f'player {player_id!r} needs either `bot` or `backend`')
    return ModelAgent(player_id, spec.pop('backend'), **spec)
