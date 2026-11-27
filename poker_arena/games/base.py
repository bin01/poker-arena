"""The interface every game implements, so the match runner, agents, logs, ratings and viewer
work for any game, not just poker.

A match is a series of episodes (for poker, hands) between the same seats. Each episode is
dealt from its own seed, so a duplicate match can replay the same episodes with the seats swapped.

    game = get_game('holdem')(config)
    ep = game.episode(index, seed, names)
    while not ep.done:
        seat = ep.to_act
        request = ep.request(seat)          # what the seat is asked, plus structured data for bots
        error = ep.apply(seat, action)      # None if accepted, else a message to show the seat
    ep.payoffs                              # chips (or points) won by each seat this episode

Episodes emit Events. Each event says which seats may see it; the runner shows each seat the events
it may see, in order, as text (`Episode.render(event, seat)`), and logs every event for the viewer.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PLAY_ONLY = {'hands_per_match'}   # settings that don't change what a deal is


def deal_fingerprint(game: dict, seed, button=0) -> str:
    """Names one deal across runs: the game settings that shape it, its seed and the seat holding the button.
    Two deals with the same fingerprint are the same cards in the same positions (see ratings.fingerprint)."""
    from ..ratings import fingerprint
    return fingerprint(game, seed, button)


@dataclass
class Event:
    kind: str
    data: dict
    visible: list[int] | None = None      # seats that may see it; None = everyone
    episode: int = 0

    def to_json(self):
        return {'kind': self.kind, 'episode': self.episode, 'visible': self.visible, 'data': self.data}


@dataclass
class Request:
    """What a seat is asked on its turn: a prompt for models and the same facts as data for bots."""
    prompt: str
    legal: list[str]
    data: dict[str, Any] = field(default_factory=dict)


class Episode:
    done: bool
    to_act: int
    payoffs: list[float]
    events: list[Event]

    def request(self, seat: int) -> Request: ...
    def apply(self, seat: int, action: dict) -> str | None: ...   # an error means: ask the seat again
    def fallback(self, seat: int) -> dict: ...          # the action taken when a seat fails to answer
    def render(self, event: Event, seat: int) -> str: ...
    def summary(self) -> dict: ...                       # for result logs and the viewer


class Game:
    name: str = ''
    min_seats: int = 2
    max_seats: int = 2
    unit: str = 'points'          # what payoffs are counted in
    big_unit: float = 1.0         # one "big blind" (or equivalent) in payoff units, for per-100 rates

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def system_prompt(self, name: str, names: list[str], episodes: int) -> str: ...
    def response_schema(self) -> dict: ...
    def episode(self, index: int, seed: int, names: list[str]) -> Episode: ...

    def describe(self) -> dict:
        return {'name': self.name, 'unit': self.unit, 'big_unit': self.big_unit, **self.config}

    def deal_fingerprint(self, seed, button=0) -> str:
        return deal_fingerprint(self.describe(), seed, button)
