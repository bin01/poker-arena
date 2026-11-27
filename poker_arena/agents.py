"""Seats: model agents and baseline bots, all with the same `act`.

A model agent keeps one conversation for the whole match (its backend), so it remembers earlier hands,
the table talk and its own replies. Each turn it gets the events it saw since its last turn plus the
request prompt, and answers with the game's JSON (reasoning, action, amount, say, notes). A reply that
can't be parsed or played (the game's `apply` returns an error, which includes strategically different
actions it won't convert) is sent back with the error; after `max_fixes` tries the seat takes the game's
fallback (check or fold). Every attempt is kept in `Decision.calls`, every refusal in `Decision.errors`.
A reply from a turn that ran (or tried) a tool is never used: the turn is taken back, the CLI conversation is
rebuilt without it, and the seat is asked again (an error of kind `tool_use`).

Bots read the request's structured data and never talk. They cost nothing and anchor the ratings; their
random state is saved with the match, so a resumed match plays exactly as an uninterrupted one.
"""
from __future__ import annotations
import json
import random
import re
from dataclasses import dataclass, field

from .backends import BackendError, make_backend
from .cards import equity
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
                              'backend_failed': True, **getattr(e, 'meta', {})})
                e.calls = calls
                raise
            calls.append({'prompt': message, 'output': reply.text, 'thinking': reply.thinking, 'usage': reply.usage,
                          'latency': reply.latency, 'attempt': attempt, **reply.meta})
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


BOTS = {'random': RandomBot, 'station': CallingStation, 'rule': RuleBot}


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
