"""Chips shown as dollars: the table read as a Las Vegas $10/$20 no-limit cash game.

The engine and the models count chips (blinds 1/2, 200-chip stacks, reset every hand). Whatever a person reads
(the viewer, `status`, the docs) shows the same game in dollars: a big blind is always $20, so the default stack
is $2,000 and a 6-chip bet is $60. This is display only. Nothing recorded, rated or sent to a model changes, so
runs stay comparable and a run's implementation fingerprint does not move. Win rates stay in bb/100 (big blinds
won per 100 hands), the measure that does not depend on the stakes; 50 bb/100 is $1,000 per 100 hands here.

The viewer's copy of these rules is in viewer/core.js (`money`, `stakesOf`); tests/test_stakes.py keeps the two
in step.
"""
from __future__ import annotations

BB_DOLLARS = 20


def usd(dollars: float) -> str:
    """1240 -> '$1,240'. Whole dollars; an average under $10 that is not whole keeps its cents ('$1.60')."""
    d = abs(float(dollars or 0))
    whole = int(d + 0.5)
    if d >= 10 or abs(d - whole) < 0.005:
        return f'${whole:,}'
    cents = int(d * 100 + 0.5) / 100
    return '$0' if cents == 0 else f'${cents:.2f}'


def money(bb: float) -> str:
    """An amount in big blinds, as dollars: money(62) -> '$1,240'."""
    return usd(bb * BB_DOLLARS)


def money_signed(bb: float) -> str:
    """money_signed(-3) -> '-$60', money_signed(62) -> '+$1,240'; zero is '$0'."""
    text = money(bb)
    return text if text == '$0' else ('+' if bb > 0 else '-') + text


def chip_money(chips: float, bb: float = 2) -> str:
    """An amount in chips of a game whose big blind is `bb` chips, as dollars."""
    return money(chips / (bb or 2))


def per_hundred(rate: float) -> str:
    """A win rate in bb/100 as dollars: per_hundred(50) -> '+$1,000 per 100 hands'."""
    return f'{money_signed(rate)} per 100 hands'


def stakes(game: dict | None = None) -> dict:
    """The table a game config describes: its blinds and stack in dollars, and one line naming it."""
    game = game or {}
    bb = game.get('big_unit') or game.get('bb') or 2
    sb = game.get('sb') if game.get('sb') is not None else bb / 2
    stack = game.get('stack')
    out = {'sb': chip_money(sb, bb), 'bb': chip_money(bb, bb), 'stack': chip_money(stack, bb) if stack else None,
           'stack_bb': stack / bb if stack else None, 'chip': usd(BB_DOLLARS / bb)}
    out['blinds'] = f'{out["sb"]}/{out["bb"]}'
    out['line'] = f'{out["blinds"]} No-Limit Hold\'em' + (f' · {out["stack"]} stacks' if stack else '')
    return out
