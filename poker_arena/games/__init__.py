"""Game registry. Add a game by writing a module with a `Game` subclass and registering it here."""
from .base import Game, Episode, Event, Request, deal_fingerprint

GAMES = {}


def register(cls):
    GAMES[cls.name] = cls
    return cls


def get_game(name: str):
    if not GAMES:
        from . import holdem  # noqa: F401  (registers itself)
    if name not in GAMES:
        raise KeyError(f'unknown game {name!r}; known: {", ".join(GAMES)}')
    return GAMES[name]


__all__ = ['Game', 'Episode', 'Event', 'Request', 'register', 'get_game', 'GAMES', 'deal_fingerprint']
