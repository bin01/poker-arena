"""Solver-grade evaluation of heads-up no-limit hold'em decisions (stdlib + NumPy only).

What is here
- `fast_rank`: a vectorized NumPy 5-7 card evaluator whose order is identical to `cards.evaluate` (tested).
- `infer_ranges`: Bayesian range weights for both seats over the 1326 combos, updated action by action with a
  fixed likelihood model of how a sound heads-up player bets with each hand strength.
- `solve_river_subgame`: exact Discounted CFR (Brown and Sandholm 2019; alpha=1.5, beta=0, gamma=2) on a river
  subgame: every combo of both ranges, hand strengths from `cards.evaluate`, card blockers removed exactly.
  The tree: OOP checks or bets (0.5x, 1x pot, all in), IP checks/bets or folds/calls/raises (pot raise, all in),
  at most two raises. Its best-response exploitability (NashConv / 2) is computed exactly.
- `solve_postflop_spot`: one `SpotEval` (docs/v2-architecture.md) for any decision: exact river DCFR on the river,
  an equity-bucketed DCFR (12 equity buckets x 2 draw-potential classes, exact pot and stack geometry, the
  street's betting with the rest of the hand collapsed into equity) on the flop and turn, and a HUNL preflop
  range / position / SPR model before the flop. Spots are hashed suit-isomorphically (`spot_hash`) and solved
  in that canonical frame, so isomorphic spots give identical answers and share a disk cache.
- `TexasSolverAdapter`: an optional external solver (the TexasSolver `console_solver` binary), same schema.
- `SlumbotClient` / `SlumbotBot`: the Slumbot HTTP API with an injectable transport (offline tests).
- `evaluate_hand`: replays a recorded hand and evaluates every decision in it.
- Calibration bots (`SOLVER_BOTS`): GTO, noisy GTO, local best response, exploiter, and a rock-paper-scissors
  triplet of styles that beat each other in a cycle.

EV convention. Inside a street subgame an action's EV is the chips the hand is expected to end up with from the
pot minus what the hero puts in from the start of the street (fold = minus what you already put in this
street); preflop and heuristic spots use chips won from now. Only differences between the actions of one
decision are ever compared (EV loss), so the offset never matters.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .agents import Agent, Decision, SeededBot
from .cards import DECK, VALUE, evaluate
from .games import ranges as _ranges

SOLVER_VERSION = 'solver-2'
ALPHA, BETA, GAMMA = 1.5, 0.0, 2.0
BLUNDER_MBB = 250.0
MIN_GRADE_REACH = 0.001
INDIFFERENT_BB = 0.05
# The fast settings evaluate_hand and GTOBot share (so a GTOBot decision and its evaluation are the same spot).
FAST = {'iterations': 48, 'river_iterations': 120, 'k': 24, 'river_cap': None, 'runouts': 24}

CARD_ID = {c: i for i, c in enumerate(DECK)}
COMBOS = np.array(list(itertools.combinations(range(52), 2)), dtype=np.int64)
COMBO_ID = {(int(a), int(b)): i for i, (a, b) in enumerate(COMBOS)}
COMBO_STR = [DECK[b] + DECK[a] for a, b in COMBOS]
COMBO_MASK = (np.int64(1) << COMBOS[:, 0]) | (np.int64(1) << COMBOS[:, 1])
STREET_CARDS = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}


# ---- card helpers -------------------------------------------------------------------------------

def _ids(cards) -> list[int]:
    return [c if isinstance(c, (int, np.integer)) else CARD_ID[c] for c in cards]


def _mask(ids) -> int:
    m = 0
    for c in ids:
        m |= 1 << int(c)
    return m


def combo_index(cards) -> int:
    a, b = sorted(_ids(cards))
    return COMBO_ID[(a, b)]


def _combo_text(cid: int) -> str:
    return COMBO_STR[cid]


def _parse_combo(x):
    if isinstance(x, str):
        x = x.replace(' ', '')
        return [x[:2], x[2:4]]
    return list(x)


# ---- vectorized evaluator -----------------------------------------------------------------------

def _tables():
    pop = np.array([bin(m).count('1') for m in range(8192)], dtype=np.int64)
    top = np.array([m.bit_length() - 1 for m in range(8192)], dtype=np.int64)
    st = np.full(8192, -1, dtype=np.int64)
    for m in range(8192):
        ext = (m << 1) | ((m >> 12) & 1)      # bit 0: the ace playing low
        for h in range(13, 3, -1):
            pat = 0x1F << (h - 4)
            if ext & pat == pat:
                st[m] = h - 1                  # rank index of the straight's top card (wheel: 3, the five)
                break
    return pop, top, st


_POP, _TOP, _STRAIGHT = _tables()
_BITS = np.int64(1) << np.arange(13, dtype=np.int64)


def _bit(x):
    return np.where(x >= 0, np.int64(1) << np.maximum(x, 0), 0)


def _topn(mask, n):
    out = []
    for _ in range(n):
        t = _TOP[mask]
        out.append(np.maximum(t, 0))
        mask = mask & ~_bit(t)
    return out


def _enc(cat, *fields):
    v = np.int64(cat) << 20
    for k, f in enumerate(fields):
        v = v | (f << (16 - 4 * k))
    return v


def fast_rank(cards) -> np.ndarray:
    """Integer strength of the best five-card hand in each row of 5-7 card ids (index into cards.DECK); a higher
    number is a better hand, ties are equal, and the order is exactly that of `cards.evaluate`."""
    c = np.asarray(cards, dtype=np.int64)
    if c.ndim == 1:
        c = c[None]
    n, k = c.shape
    r, s = c >> 2, c & 3
    rows = np.arange(n)
    counts = np.zeros((n, 13), dtype=np.int64)
    smask = np.zeros((n, 4), dtype=np.int64)
    for j in range(k):
        counts[rows, r[:, j]] += 1
        smask[rows, s[:, j]] |= np.int64(1) << r[:, j]
    anym = (counts > 0).astype(np.int64) @ _BITS
    pm = (counts == 2).astype(np.int64) @ _BITS
    tm = (counts == 3).astype(np.int64) @ _BITS
    qm = (counts == 4).astype(np.int64) @ _BITS
    scnt = _POP[smask]
    fs = scnt.argmax(1)
    flush = scnt[rows, fs] >= 5
    fmask = smask[rows, fs]
    sf = np.where(flush, _STRAIGHT[fmask], -1)
    st = _STRAIGHT[anym]
    q = np.maximum(_TOP[qm], 0)
    t = np.maximum(_TOP[tm], 0)
    p1 = np.maximum(_TOP[pm], 0)
    hi, lo = _topn(pm, 2)
    conds = [sf >= 0, qm > 0, (tm > 0) & ((_POP[tm] >= 2) | (pm > 0)), flush, st >= 0, tm > 0, _POP[pm] >= 2, pm > 0]
    choices = [
        _enc(8, np.maximum(sf, 0)),
        _enc(7, q, np.maximum(_TOP[anym & ~_bit(q)], 0)),
        _enc(6, t, np.maximum(_TOP[(tm & ~_bit(t)) | pm], 0)),
        _enc(5, *_topn(fmask, 5)),
        _enc(4, np.maximum(st, 0)),
        _enc(3, t, *_topn(anym & ~_bit(t), 2)),
        _enc(2, hi, lo, np.maximum(_TOP[anym & ~_bit(hi) & ~_bit(lo)], 0)),
        _enc(1, p1, *_topn(anym & ~_bit(p1), 3)),
    ]
    return np.select(conds, choices, default=_enc(0, *_topn(anym, 5)))


def _share(a, b):
    return 0.5 * (1.0 + np.sign(a - b))


def _random_runouts(known_mask_rows, need: int, nr: np.random.Generator) -> np.ndarray:
    """For each row (a 52-bit mask of cards already out), `need` distinct random cards not in it."""
    m = len(known_mask_rows)
    keys = nr.random((m, 52))
    bits = (np.asarray(known_mask_rows, dtype=np.int64)[:, None] >> np.arange(52)) & 1
    keys[bits.astype(bool)] = 2.0
    return np.argsort(keys, axis=1)[:, :need]


# ---- range model --------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _pre_strength() -> np.ndarray:
    pct = _ranges.percentile()
    return np.array([1.0 - pct[_ranges.hand_class((DECK[a], DECK[b]))] for a, b in COMBOS])


@lru_cache(maxsize=1024)
def _board_strength(board: tuple) -> np.ndarray:
    """Every combo's strength on this board in [0, 1] (0 for combos that clash with it): its made hand's
    percentile among all live combos, blended on the flop and turn with its percentile over sampled runouts
    (so draws count). Deterministic for a board."""
    if not board:
        return _pre_strength()
    bmask = _mask(board)
    idx = np.nonzero((COMBO_MASK & bmask) == 0)[0]
    cc = COMBOS[idx]

    def pct(full, live):
        rk = fast_rank(np.concatenate([cc[live], np.tile(np.array(full, dtype=np.int64), (int(live.sum()), 1))], 1))
        srt = np.sort(rk)
        return (np.searchsorted(srt, rk, 'left') + 0.5 * (np.searchsorted(srt, rk, 'right') -
                                                          np.searchsorted(srt, rk, 'left'))) / max(1, len(rk))

    live = np.ones(len(idx), dtype=bool)
    now = pct(list(board), live)
    out = np.zeros(1326)
    if len(board) >= 5:
        out[idx] = now
        return out
    rng = random.Random(f'strength|{board}')
    rest = [c for c in range(52) if not (bmask >> c) & 1]
    acc, cnt = np.zeros(len(idx)), np.zeros(len(idx))
    for _ in range(6):
        run = rng.sample(rest, 5 - len(board))
        rm = _mask(run)
        lv = (COMBO_MASK[idx] & rm) == 0
        acc[lv] += pct(list(board) + run, lv)
        cnt[lv] += 1
    fut = np.where(cnt > 0, acc / np.maximum(cnt, 1), now)
    out[idx] = 0.5 * now + 0.5 * fut
    return out


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


PRE_CONTINUE = {0: 0.15, 1: 0.35, 2: 0.62, 3: 0.82}
PRE_RAISE = {1: 0.20, 2: 0.85, 3: 0.91, 4: 0.95}


def action_likelihood(s, street: str, action: str, *, n_raises: int = 0, frac: float = 0.0, odds: float = 0.0,
                      all_in: bool = False, button: bool = False, facing: bool = False) -> np.ndarray:
    """P(action | hand strength s) under the fixed range model: aggression from strong hands (plus a few
    bluffs), calls from the middle, folds from the bottom, checks mildly capped. Floored at 2%."""
    s = np.asarray(s, dtype=float)
    a = {'bet': 'raise', 'all_in': 'raise'}.get(action, action)
    if street == 'preflop':
        tc = PRE_CONTINUE.get(n_raises, 0.9)
        tr = PRE_RAISE.get(n_raises + 1, 0.97)
        if n_raises == 0 and not button:
            tr = 0.55                       # the big blind raising a limp
        if all_in:
            tr = max(tr, 0.90)
        if a == 'raise':
            lik = _sig((s - tr) / 0.04) + (0.05 if n_raises >= 1 and not all_in else 0.0)
        elif a == 'call':
            lik = (0.3 * np.ones_like(s) if n_raises == 0 else _sig((s - tc) / 0.06)) * (1 - 0.7 * _sig((s - tr) / 0.04))
        elif a == 'check':
            lik = 1 - 0.7 * _sig((s - 0.6) / 0.05)
        else:
            lik = 1 - _sig((s - tc) / 0.06)
    else:
        tc = 0.30 + 0.5 * odds
        if a == 'raise':
            tr = (0.80 if facing else 0.58 + 0.10 * min(frac, 1.5)) + (0.05 if all_in else 0.0)
            lik = _sig((s - tr) / 0.07) + (0.0 if facing else 0.25) * (1 - s) ** 3
        elif a == 'call':
            lik = _sig((s - tc) / 0.07) * (1 - 0.5 * _sig((s - 0.92) / 0.04))
        elif a == 'check':
            lik = 1 - 0.5 * _sig((s - 0.8) / 0.06)
        else:
            lik = 1 - _sig((s - tc) / 0.07)
    return 0.02 + 0.98 * np.clip(lik, 0.0, 1.0)


def _hist_sig(history) -> tuple:
    return tuple((int(h['seat']), h['street'], h['action'], h.get('amount'), bool(h.get('all_in')))
                 for h in history)


def infer_ranges(history, board, button: int, bb: int = 2, sb: int | None = None):
    """(weights of seat 0, weights of seat 1): 1326-vectors, the prior (uniform) times the likelihood of every
    recorded action under `action_likelihood`. Public information only (no hole cards), so callers remove the
    cards they know. `board` is the board as far as it is known; each action is judged on its street's board."""
    return _public_ranges(_hist_sig(history), tuple(_ids(board)), int(button), int(bb), int(sb if sb is not None else bb // 2))


@lru_cache(maxsize=24)
def _perm_indices(perm):
    return np.array([combo_index([_apply_perm(perm, card) for card in combo]) for combo in COMBOS])


@lru_cache(maxsize=256)
def _public_ranges(sig, board, button, bb, sb):
    if not any(row[1] != 'preflop' for row in sig):
        return _infer(sig, (), button, bb, sb)
    canonical, _ = _canon_cards(board, (), canonical_perm(board))
    weights = _infer(sig, tuple(canonical), button, bb, sb)
    maps = [_perm_indices(perm) for perm in _SUIT_PERMS if _canon_cards(board, (), perm)[0] == canonical]
    return tuple(np.mean([side[indices] for indices in maps], axis=0) for side in weights)


@lru_cache(maxsize=4096)
def _infer(sig: tuple, board: tuple, button: int, bb: int, sb: int):
    if not sig:
        return np.ones(1326), np.ones(1326)
    prev = _infer(sig[:-1], board, button, bb, sb)
    w = [prev[0].copy(), prev[1].copy()]
    st = _replay_state(sig[:-1], button, bb, sb)
    seat, street, action, amount, all_in = sig[-1]
    bets, pot, n_raises = st['street_bet'] if st['street'] == street else [0, 0], st['pot'], \
        st['n_raises'] if st['street'] == street else 0
    to_call = max(bets) - bets[seat]
    odds = to_call / (pot + to_call) if to_call else 0.0
    frac = 0.0
    if action in ('bet', 'raise') and amount is not None:
        frac = max(0.0, (amount - bets[seat] - to_call)) / max(1.0, pot + to_call)
    s = _board_strength(board[:STREET_CARDS[street]]) if street != 'preflop' else _pre_strength()
    lik = action_likelihood(s, street, action, n_raises=n_raises, frac=frac, odds=odds, all_in=all_in,
                            button=seat == button, facing=to_call > 0)
    w[seat] = w[seat] * lik
    w[seat] /= max(w[seat].max(), 1e-300)
    return w[0], w[1]


def _replay_state(sig, button, bb, sb):
    """Pot and this street's bets after these actions (from the blinds)."""
    bets = [0, 0]
    bets[button], bets[1 - button] = sb, bb
    pot, street, nr = sb + bb, 'preflop', 0
    for seat, st, action, amount, _ in sig:
        if st != street:
            street, bets, nr = st, [0, 0], 0
        if action == 'call':
            add = amount if amount is not None else max(bets) - bets[seat]
            bets[seat] += add
            pot += add
        elif action in ('bet', 'raise'):
            pot += amount - bets[seat]
            bets[seat] = amount
            nr += 1
    return {'street_bet': bets, 'pot': pot, 'street': street, 'n_raises': nr}


def range_weights(x) -> np.ndarray:
    """A range as 1326 weights, from weights, range text ('TT+, AKs'), {combo: weight} or a list of combos."""
    if isinstance(x, np.ndarray) and x.shape == (1326,):
        return x.astype(float)
    w = np.zeros(1326)
    if isinstance(x, str):
        for c in _ranges.parse(x):
            w[combo_index(c)] = 1.0
        return w
    if isinstance(x, dict):
        for c, v in x.items():
            w[combo_index(_parse_combo(c))] = float(v)
        return w
    for c in x:
        w[combo_index(_parse_combo(c))] = 1.0
    return w


# ---- the betting tree ---------------------------------------------------------------------------

def _abstract(label: str) -> str:
    if label.startswith('bet') or label.startswith('raise'):
        return 'raise'
    return 'all_in' if label == 'allin' else label


class _Tree:
    """One street's betting tree, node ids in breadth-first order (player 0 = out of position, acts first).
    Contributions are chips put in this street; terminals are folds and showdowns."""

    def __init__(self, pot, stack, bet_sizes=(0.5, 1.0), raise_sizes=(1.0,), max_raises=2, bb=2.0, allin_cap=None):
        self.pot, self.stack, self.bb = float(pot), float(stack), float(bb)
        self.bet_sizes, self.raise_sizes, self.max_raises = tuple(bet_sizes), tuple(raise_sizes), int(max_raises)
        self.allin_cap = None if allin_cap is None else float(allin_cap)
        kind, player, contrib, depth, parent, label, amount, kids, states = [], [], [], [], [], [], [], [], []

        def add(k, pl, cc, d, par, lab, amt, state):
            kind.append(k), player.append(pl), contrib.append(tuple(cc)), depth.append(d), parent.append(par)
            label.append(lab), amount.append(amt), kids.append([]), states.append(state)
            return len(kind) - 1

        if self.stack <= 1e-9:
            add('sd', -1, (0.0, 0.0), 0, -1, None, None, None)
        else:
            add('dec', 0, (0.0, 0.0), 0, -1, None, None, (0, 0, False))
        i = 0
        while i < len(kind):
            if kind[i] == 'dec':
                pl, nb, checked = states[i]
                for lab, spec, amt in self._moves(pl, contrib[i], nb, checked):
                    if spec[0] == 'dec':
                        j = add('dec', spec[1], spec[2], depth[i] + 1, i, lab, amt, (spec[1], spec[3], spec[4]))
                    elif spec[0] == 'fold':
                        j = add('fold', spec[1], spec[2], depth[i] + 1, i, lab, amt, None)
                    else:
                        j = add('sd', -1, spec[1], depth[i] + 1, i, lab, amt, None)
                    kids[i].append(j)
            i += 1
        self.kind, self.player, self.contrib, self.depth = kind, np.array(player), contrib, np.array(depth)
        self.parent, self.label, self.amount, self.kids = np.array(parent), label, amount, kids
        self._flatten()

    def _moves(self, me, c, nb, checked):
        op = 1 - me
        P, S = self.pot, self.stack
        allow_allin = self.allin_cap is None or (S - c[op]) <= self.allin_cap * (P + 2 * c[op])
        out = []
        if c[op] > c[me] + 1e-9:
            out.append(('fold', ('fold', me, c), None))
            cc = list(c)
            cc[me] = c[op]
            out.append(('call', ('sd', tuple(cc)), None))
            if c[op] < S - 1e-9 and nb < 1 + self.max_raises:
                last = c[op] - c[me]
                for f in self.raise_sizes:
                    to = max(c[op] + f * (P + 2 * c[op]), c[op] + max(last, self.bb))
                    if to < 0.98 * S:
                        cc = list(c)
                        cc[me] = to
                        out.append((f'raise{f:g}', ('dec', op, tuple(cc), nb + 1, False), to))
                if allow_allin or not any(o[0].startswith('raise') for o in out):
                    cc = list(c)
                    cc[me] = S
                    out.append(('allin', ('dec', op, tuple(cc), nb + 1, False), S))
        else:
            if me == 0 and not checked:
                out.append(('check', ('dec', 1, c, nb, True), None))
            else:
                out.append(('check', ('sd', c), None))
            for f in self.bet_sizes:
                amt = max(f * P, self.bb)
                if amt < 0.98 * S:
                    cc = list(c)
                    cc[me] = amt
                    out.append((f'bet{f:g}', ('dec', op, tuple(cc), 1, False), amt))
            if allow_allin or not any(o[0].startswith('bet') for o in out):
                cc = list(c)
                cc[me] = S
                out.append(('allin', ('dec', op, tuple(cc), nb + 1, False), S))
        return out

    def _flatten(self):
        N = len(self.kind)
        self.N = N
        dec = [i for i in range(N) if self.kind[i] == 'dec']
        e_par, e_ch = [], []
        in_edge = np.full(N, -1)
        for i in dec:
            for j in self.kids[i]:
                in_edge[j] = len(e_par)
                e_par.append(i)
                e_ch.append(j)
        self.e_par, self.e_ch, self.in_edge = np.array(e_par, dtype=np.int64), np.array(e_ch, dtype=np.int64), in_edge
        owner = self.player[self.e_par] if len(e_par) else np.zeros(0, dtype=np.int64)
        self.e_owner = owner
        loc = np.zeros(len(e_par), dtype=np.int64)
        self.p_edges, self.p_nodes, self.p_starts, self.p_node_of_edge, self.p_nact = [], [], [], [], []
        for p in (0, 1):
            ed = np.nonzero(owner == p)[0]
            loc[ed] = np.arange(len(ed))
            nodes = [i for i in dec if self.player[i] == p]
            starts, noe, nact, pos = [], [], [], 0
            for k, i in enumerate(nodes):
                starts.append(pos)
                n = len(self.kids[i])
                noe += [k] * n
                nact += [n] * n
                pos += n
            self.p_edges.append(ed)
            self.p_nodes.append(np.array(nodes, dtype=np.int64))
            self.p_starts.append(np.array(starts, dtype=np.int64))
            self.p_node_of_edge.append(np.array(noe, dtype=np.int64))
            self.p_nact.append(np.array(nact, dtype=float)[:, None] if nact else np.zeros((0, 1)))
        self.loc = loc
        D = int(self.depth.max()) if N else 0
        self.fwd = [[], []]
        for d in range(1, D + 1):
            ch = np.nonzero(self.depth == d)[0]
            par = self.parent[ch]
            for q in (0, 1):
                m = self.player[par] == q
                self.fwd[q].append((slice(ch[0], ch[-1] + 1), par, loc[in_edge[ch]] if m.all() else None))
        self.bwd = []
        for d in range(D - 1, -1, -1):
            groups = []
            for o in (0, 1):
                nodes = [i for i in dec if self.depth[i] == d and self.player[i] == o]
                if not nodes:
                    continue
                edges, starts = [], []
                for i in nodes:
                    starts.append(len(edges))
                    edges += [int(in_edge[j]) for j in self.kids[i]]
                edges = np.array(edges, dtype=np.int64)
                groups.append((o, np.array(nodes, dtype=np.int64), self.e_ch[edges], np.array(starts, dtype=np.int64),
                               loc[edges]))
            self.bwd.append(groups)
        self.fold_ids = np.array([i for i in range(N) if self.kind[i] == 'fold'], dtype=np.int64)
        self.fold_who = np.array([self.player[i] for i in self.fold_ids], dtype=np.int64)
        self.fold_c = np.array([self.contrib[i][self.player[i]] for i in self.fold_ids], dtype=float)
        self.sd_ids = np.array([i for i in range(N) if self.kind[i] == 'sd'], dtype=np.int64)
        self.sd_c = np.array([self.contrib[i][0] for i in self.sd_ids], dtype=float)

    def child_edges(self, node):
        return [int(self.in_edge[j]) for j in self.kids[node]]

    def match(self, node, act):
        """The child of `node` an action maps to (nearest bet size), or None when the tree can't follow it.
        `act`: a tree label, ('raise', amount), or a history entry {'action', 'amount', 'all_in'}."""
        if self.kind[node] != 'dec':
            return None
        kids = self.kids[node]
        labels = [self.label[j] for j in kids]
        if isinstance(act, str) and act in labels:
            return kids[labels.index(act)]
        if isinstance(act, dict):
            a, amt, allin = act.get('action'), act.get('amount'), bool(act.get('all_in'))
        elif isinstance(act, (tuple, list)):
            a, amt, allin = act[0], act[1] if len(act) > 1 else None, False
        else:
            a, amt, allin = act, None, False
        a = {'bet': 'raise', 'allin': 'all_in', 'shove': 'all_in'}.get(a, a)
        if a in ('check', 'call', 'fold'):
            return kids[labels.index(a)] if a in labels else None
        if a == 'all_in' or allin or (amt is not None and amt >= 0.98 * self.stack):
            if 'allin' in labels:
                return kids[labels.index('allin')]
        sized = [(j, self.amount[j]) for j in kids if self.label[j] != 'allin' and self.amount[j]]
        if not sized:
            return kids[labels.index('allin')] if 'allin' in labels and a in ('raise', 'all_in') else None
        if amt is None:
            return sized[0][0]
        cands = sized + ([(kids[labels.index('allin')], self.stack)] if 'allin' in labels else [])
        return min(cands, key=lambda x: abs(math.log(max(amt, 1e-6) / max(x[1], 1e-6))))[0]

    def path(self, actions):
        node = 0
        for a in actions:
            node = self.match(node, a)
            if node is None:
                return None
        return node

    def history_key(self, node):
        out = []
        while node > 0:
            out.append(self.label[node])
            node = int(self.parent[node])
        return '/'.join(reversed(out))


# ---- payoff evaluators --------------------------------------------------------------------------

class _DenseEval:
    """Class-vs-class payoffs from dense matrices: mass[a, b] (joint weight, 0 for clashing cards) and
    sd[a, b] (mass times 2 x equity - 1)."""

    def __init__(self, W, Sd):
        self.dtype = np.asarray(W).dtype.type
        self.WT, self.SdT = np.ascontiguousarray(W.T), np.ascontiguousarray(Sd.T)

    def mass(self, Q):
        return Q @ self.WT

    def sd(self, Q):
        return Q @ self.SdT


class _ComboEval:
    """Exact combo-vs-combo river payoffs in O(n) per terminal: the opponent's reach sorted by hand strength,
    with every combo sharing a card with the hero's removed through per-card prefix sums."""

    def __init__(self, hero_cards, hero_rank, opp_cards, opp_rank):
        big = int(max(hero_rank.max(initial=0), opp_rank.max(initial=0))) + 2
        self.ord = np.argsort(opp_rank, kind='stable')
        rs = opp_rank[self.ord]
        self.lo = np.searchsorted(rs, hero_rank, 'left')
        self.hi = np.searchsorted(rs, hero_rank, 'right')
        n_o = len(opp_rank)
        cards = opp_cards.ravel()
        combo = np.repeat(np.arange(n_o), 2)
        key = cards * big + opp_rank[combo]
        fo = np.argsort(key, kind='stable')
        keys = key[fo]
        self.fcombo = combo[fo]
        self.pl, self.pr, self.ps, self.pe = [], [], [], []
        for j in (0, 1):
            c = hero_cards[:, j]
            self.pl.append(np.searchsorted(keys, c * big + hero_rank, 'left'))
            self.pr.append(np.searchsorted(keys, c * big + hero_rank, 'right'))
            self.ps.append(np.searchsorted(keys, c * big, 'left'))
            self.pe.append(np.searchsorted(keys, (c + 1) * big, 'left'))
        pos = {(int(a), int(b)): i for i, (a, b) in enumerate(opp_cards)}
        self.same = np.array([pos.get((int(a), int(b)), -1) for a, b in hero_cards])
        self.has_same = self.same >= 0
        self.same_idx = np.maximum(self.same, 0)
        counts = np.bincount(cards, minlength=52)
        live_cards = np.flatnonzero(counts)
        self.card_width = int(counts[live_cards[0]]) if np.all(counts[live_cards] == counts[live_cards[0]]) else 0
        self.card_columns = np.searchsorted(live_cards, hero_cards.T)
        if not np.all(np.isin(hero_cards, live_cards)):
            self.card_width = 0

    @staticmethod
    def _cum(X):
        out = np.zeros((X.shape[0], X.shape[1] + 1))
        np.cumsum(X, axis=1, out=out[:, 1:])
        return out

    def mass(self, Q):
        if self.card_width:
            grouped = Q.take(self.fcombo, axis=1).reshape(len(Q), -1, self.card_width).sum(axis=2)
            first, second = self.card_columns
            return (Q.sum(1, keepdims=True) - grouped.take(first, axis=1) - grouped.take(second, axis=1)
                    + Q.take(self.same_idx, axis=1) * self.has_same)
        F = self._cum(Q[:, self.fcombo])
        tot = Q.sum(1, keepdims=True)
        m = tot - (F[:, self.pe[0]] - F[:, self.ps[0]]) - (F[:, self.pe[1]] - F[:, self.ps[1]])
        return m + Q[:, self.same_idx] * self.has_same

    def sd(self, Q):
        C = self._cum(Q.take(self.ord, axis=1))
        F = self._cum(Q.take(self.fcombo, axis=1))
        value = C.take(self.lo, axis=1) + C.take(self.hi, axis=1) - C[:, -1:]
        for card in (0, 1):
            value -= F.take(self.pl[card], axis=1) - F.take(self.ps[card], axis=1)
            value += F.take(self.pe[card], axis=1) - F.take(self.pr[card], axis=1)
        return value


class _FastRiverEval:
    """Exact blocker-aware river evaluator using float32 BLAS GEMM (52-card incidence for mass,
    exact (n_opp x n_hero) showdown sign matrix for sd)."""
    dtype = np.float32

    def __init__(self, hero_cards, hero_rank, opp_cards, opp_rank, SdT=None):
        n_o = len(opp_rank)
        inc = np.zeros((n_o, 52), dtype=np.float32)
        inc[np.arange(n_o), opp_cards[:, 0]] = 1.0
        inc[np.arange(n_o), opp_cards[:, 1]] = 1.0
        self.inc = np.ascontiguousarray(inc)
        self.hc0 = hero_cards[:, 0]
        self.hc1 = hero_cards[:, 1]
        self.identity = bool(np.array_equal(hero_cards, opp_cards))
        if self.identity:
            self.same_idx = None
            self.has_same = None
        else:
            pos = {(int(a), int(b)): i for i, (a, b) in enumerate(opp_cards)}
            same = np.array([pos.get((int(a), int(b)), -1) for a, b in hero_cards], dtype=np.int64)
            self.has_same = (same >= 0).astype(np.float32)
            self.same_idx = np.maximum(same, 0)
        if SdT is None:
            hm = (np.int64(1) << hero_cards[:, 0]) | (np.int64(1) << hero_cards[:, 1])
            om = (np.int64(1) << opp_cards[:, 0]) | (np.int64(1) << opp_cards[:, 1])
            ok = (om[:, None] & hm[None, :]) == 0
            SdT = (ok * np.sign(hero_rank[None, :] - opp_rank[:, None])).astype(np.float32)
        self.SdT = np.ascontiguousarray(SdT)

    def mass(self, Q):
        q = Q if Q.dtype == np.float32 else np.asarray(Q, dtype=np.float32)
        cs = q @ self.inc
        out = q.sum(1, keepdims=True) - cs[:, self.hc0] - cs[:, self.hc1]
        return out + q if self.identity else out + q[:, self.same_idx] * self.has_same

    def sd(self, Q):
        return (Q if Q.dtype == np.float32 else np.asarray(Q, dtype=np.float32)) @ self.SdT


@lru_cache(maxsize=64)
def _full_river_eval(board: tuple) -> tuple[np.ndarray, _FastRiverEval]:
    rk = _river_ranks(board)
    live = np.nonzero(rk >= 0)[0]
    ev = _FastRiverEval(COMBOS[live], rk[live], COMBOS[live], rk[live])
    return live, ev


# ---- Discounted CFR -----------------------------------------------------------------------------

class _DCFR:
    """Vector-form DCFR over a `_Tree`: per player, regrets and average strategies are (edges x classes)
    arrays; reach is pushed forward and values pulled back one tree depth at a time."""

    def __init__(self, tree: _Tree, priors, evals, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        dt = getattr(evals[0], 'dtype', np.float64) if evals else np.float64
        self.dtype = dt
        self.t, self.prior, self.ev = tree, [np.asarray(p, dtype=dt) for p in priors], evals
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.n = [len(self.prior[0]), len(self.prior[1])]
        self.R = [np.zeros((len(tree.p_edges[p]), self.n[p]), dtype=dt) for p in (0, 1)]
        self.S = [np.zeros((len(tree.p_edges[p]), self.n[p]), dtype=dt) for p in (0, 1)]
        self.inv_nact = [(1.0 / tree.p_nact[p]).astype(dt) for p in (0, 1)]
        self.norm_mat = []
        for p in (0, 1):
            ne = len(tree.p_edges[p])
            nn = len(tree.p_nodes[p])
            if ne:
                M = np.zeros((nn, ne), dtype=dt)
                M[tree.p_node_of_edge[p], np.arange(ne)] = 1.0
                self.norm_mat.append(np.ascontiguousarray(M[tree.p_node_of_edge[p]]))
            else:
                self.norm_mat.append(np.zeros((0, 0), dtype=dt))
        self.bwd_g = []
        for groups in tree.bwd:
            glist = []
            for o, nodes, child, starts, loc in groups:
                G = np.zeros((len(nodes), len(child)), dtype=dt)
                ends = np.append(starts[1:], len(child))
                for k, (a, b) in enumerate(zip(starts, ends)):
                    G[k, a:b] = 1.0
                glist.append((o, nodes, child, starts, np.ascontiguousarray(G), loc))
            self.bwd_g.append(glist)
        self.iterations = 0
        half = tree.pot / 2
        self.coef_f = [np.where(tree.fold_who == p, -(half + tree.fold_c), half + tree.fold_c)[:, None].astype(dt) for p in (0, 1)]
        self.coef_s = (half + tree.sd_c)[:, None].astype(dt)
        self._e_ch = [tree.e_ch[tree.p_edges[p]] for p in (0, 1)]
        self._e_par = [tree.e_par[tree.p_edges[p]] for p in (0, 1)]
        self._V = [np.zeros((tree.N, self.n[p]), dtype=dt) for p in (0, 1)]
        self._reach = [np.empty((tree.N, self.n[p]), dtype=dt) for p in (0, 1)]
        for p in (0, 1):
            self._reach[p][0] = self.prior[p]

    def _norm(self, X, p):
        if not len(X):
            return X
        sums = self.norm_mat[p] @ X
        pos = sums > 0
        out = np.empty_like(X)
        np.divide(X, sums, out=out, where=pos)
        return np.where(pos, out, self.inv_nact[p])

    def current(self, p):
        return self._norm(np.maximum(self.R[p], self.dtype(0.0)), p)

    def average(self, p):
        return self._norm(self.S[p], p)

    def forward(self, q, sig, out=None):
        t = self.t
        if out is None:
            reach = np.empty((t.N, self.n[q]), dtype=self.dtype)
            reach[0] = self.prior[q]
        else:
            reach = out
        for ch, par, loc in t.fwd[q]:
            if loc is None:
                reach[ch] = reach[par]
            else:
                np.multiply(reach[par], sig[loc], out=reach[ch])
        return reach

    def backward(self, p, sig, reach_o, br=False, out=None):
        t = self.t
        V = np.zeros((t.N, self.n[p]), dtype=self.dtype) if out is None else out
        if len(t.fold_ids):
            V[t.fold_ids] = self.coef_f[p] * self.ev[p].mass(reach_o[t.fold_ids])
        if len(t.sd_ids):
            V[t.sd_ids] = self.coef_s * self.ev[p].sd(reach_o[t.sd_ids])
        for groups in self.bwd_g:
            for o, nodes, child, starts, G, loc in groups:
                vals = V[child]
                if o == p:
                    V[nodes] = np.maximum.reduceat(vals, starts, axis=0) if br else G @ (vals * sig[loc])
                else:
                    V[nodes] = G @ vals
        return V

    def run(self, iterations, track=()):
        """Alternating-update DCFR iterations; the exploitability (chips) at the iterations in `track`. The
        strategy and reach of the player not being updated are shared between the two half-iterations."""
        t = self.t
        dt = self.dtype
        curve = []
        track = set(track)
        sig = [self.current(0), self.current(1)]
        reach = [self.forward(p, sig[p], self._reach[p]) for p in (0, 1)]
        has_edges = [bool(len(t.p_edges[0])), bool(len(t.p_edges[1]))]
        for _ in range(iterations):
            self.iterations += 1
            it = self.iterations
            pos_d = dt(it ** self.alpha / (it ** self.alpha + 1))
            neg_d = dt(it ** self.beta / (it ** self.beta + 1))
            avg_d = dt(((it - 1) / it) ** self.gamma)
            for p in (0, 1):
                if not has_edges[p]:
                    continue
                V = self.backward(p, sig[p], reach[1 - p], out=self._V[p])
                R = self.R[p]
                R += V[self._e_ch[p]]
                R -= V[self._e_par[p]]
                pos = R > 0
                np.multiply(R, pos_d, out=R, where=pos)
                np.multiply(R, neg_d, out=R, where=~pos)
                S = self.S[p]
                S *= avg_d
                S += reach[p][self._e_par[p]] * sig[p]
                sig[p] = self.current(p)
                reach[p] = self.forward(p, sig[p], self._reach[p])
            if it in track:
                curve.append((it, self.exploitability()))
        return curve

    def total_mass(self):
        return float(self.prior[0] @ self.ev[0].mass(self.prior[1][None])[0])

    def exploitability(self):
        """NashConv / 2 in chips per hand of the average strategies (exact best responses)."""
        avg = [self.average(0), self.average(1)]
        total = 0.0
        for p in (0, 1):
            V = self.backward(p, avg[p], self.forward(1 - p, avg[1 - p]), br=True)
            total += float(self.prior[p] @ V[0])
        z = self.total_mass()
        return max(0.0, total / z / 2) if z > 0 else 0.0

    def evaluate(self, eps=1e-4):
        """Evaluate counterfactual action EVs against the opponent's average profile (best-responding in hero's
        subtrees so 0%-frequency counterfactual actions aren't corrupted by unvisited uniform regrets), and track
        trembled opponent reach separately for diagnostics."""
        avg = [self.average(0), self.average(1)]
        out = []
        for p in (0, 1):
            o = 1 - p
            sig_o = (1 - eps) * avg[o] + eps / self.t.p_nact[o] if len(avg[o]) else avg[o]
            reach_o = self.forward(o, avg[o])
            out.append((self.backward(p, avg[p], reach_o, br=True), reach_o, self.forward(o, sig_o)))
        return avg, out


class _Solution:
    """A solved street subgame: node_eval(hero class, history) gives a decision's sized action EVs and
    equilibrium frequencies."""

    def __init__(self, tree, cfr, classes, class_of, bb, street, solver, equity=None, curve=None):
        self.tree, self.cfr, self.classes, self.class_of, self.bb = tree, cfr, classes, class_of, float(bb)
        self.street, self.solver, self.equity_of = street, solver, equity
        self.curve = [(it, c / self.bb * 1000) for it, c in (curve or [])]
        self.expl_chips = cfr.exploitability() if len(tree.p_edges[0]) else 0.0
        self.exploitability_mbb = self.expl_chips / self.bb * 1000
        self.avg, self.vals = cfr.evaluate()

    def node_of(self, history):
        return self.tree.path(history)

    def sized(self, player, cls, node):
        t = self.tree
        if node is None or t.kind[node] != 'dec' or t.player[node] != player:
            return None
        V, reach_o, _ = self.vals[player]
        m = self.cfr.ev[player].mass(reach_o[node][None])[0][cls]
        edges = t.child_edges(node)
        half = t.pot / 2
        evs = [(V[t.e_ch[e]][cls] / m + half) / self.bb if m > 0 else 0.0 for e in edges]
        freqs = [float(self.avg[player][t.loc[e]][cls]) for e in edges]
        return [{'label': t.label[t.e_ch[e]], 'abstract': _abstract(t.label[t.e_ch[e]]),
                 'to': t.amount[t.e_ch[e]], 'ev': float(ev), 'freq': f} for e, ev, f in zip(edges, evs, freqs)]

    def reach_stats(self, player, cls, node):
        _, raw, smooth = self.vals[player]
        evaluator = self.cfr.ev[player]
        mass, root, perturbed = evaluator.mass(np.stack([raw[node], raw[0], smooth[node]]))[:, cls]
        share = float(np.clip(1 - mass / perturbed, 0, 1)) if perturbed > 0 else 1.0
        equity = float(np.clip(0.5 + 0.5 * evaluator.sd(raw[node:node+1])[0, cls] / mass, 0, 1)) if mass > 0 else None
        fraction = float(mass / root) if root > 0 else 0.0
        return dict(equity=equity, node_reach=float(mass), reach_fraction=fraction, tremble_share=share,
                    ev_reliable=fraction >= MIN_GRADE_REACH and share <= 0.5)

    def deep_shove_ev(self, player, cls, node):
        """Street-root EV (bb) of an out-of-tree deep all-in shove when allin_cap omitted it from a 1-street tree."""
        t = self.tree
        if node is None or t.kind[node] != 'dec' or t.player[node] != player:
            return None
        c_op = t.contrib[node][1 - player]
        S = t.stack
        dd = S - c_op
        if dd <= 1e-9:
            return None
        _, reach_o, _ = self.vals[player]
        ev = self.cfr.ev[player]
        if not isinstance(ev, _DenseEval):
            return None
        w = reach_o[node] * ev.WT[:, cls]
        tot = float(w.sum())
        if tot <= 0:
            return None
        sd = reach_o[node] * ev.SdT[:, cls]
        eq_h = np.where(w > 0, np.clip(0.5 + 0.5 * sd / np.where(w > 0, w, 1.0), 0.0, 1.0), 0.5)
        total_pot = t.pot + 2 * S
        call = (1.0 - eq_h) >= dd / total_pot
        payoff = np.where(call, eq_h * total_pot - S, t.pot + c_op)
        return float((w @ payoff) / tot / self.bb)

    def probe(self, player, cid, node):
        """Sized options of a combo outside the solved range (river solutions): each action's EV for that hand
        against the opponent's equilibrium strategy, the hand best-responding at its later decisions; the
        frequencies put all weight on the best action."""
        t = self.tree
        if node is None or t.kind[node] != 'dec' or t.player[node] != player or not hasattr(self, 'ranks'):
            return None
        opp = self.classes[1 - player]
        ev1 = _ComboEval(COMBOS[[cid]], self.ranks[[cid]], COMBOS[opp], self.ranks[opp])
        cfr1 = _DCFR(t, [np.ones(1), self.priors[1 - player]] if player == 0 else [self.priors[0], np.ones(1)],
                     [ev1, ev1])
        reach_o = self.vals[player][1]
        V = cfr1.backward(player, None, reach_o, br=True)
        m = ev1.mass(reach_o[node][None])[0][0]
        half = t.pot / 2
        edges = t.child_edges(node)
        evs = [(V[t.e_ch[e]][0] / m + half) / self.bb if m > 0 else 0.0 for e in edges]
        best = max(evs)
        top = [abs(e - best) < 1e-12 for e in evs]
        return [{'label': t.label[t.e_ch[e]], 'abstract': _abstract(t.label[t.e_ch[e]]), 'to': t.amount[t.e_ch[e]],
                 'ev': float(ev), 'freq': (1.0 / sum(top)) if hit else 0.0} for e, ev, hit in zip(edges, evs, top)]

    def node_eval(self, hero_hand, history=()):
        node = self.node_of(history)
        if node is None:
            raise ValueError(f'history {history!r} is not in the tree')
        player = int(self.tree.player[node])
        cid = hero_hand if isinstance(hero_hand, (int, np.integer)) else combo_index(_parse_combo(hero_hand))
        cls = self.class_of[player].get(int(cid))
        if cls is None:
            raise ValueError(f'{_combo_text(int(cid))} is not in the range of the player to act')
        opts = self.sized(player, cls, node)
        evs = {o['label']: o['ev'] for o in opts}
        freqs = {o['label']: o['freq'] for o in opts}
        return {'player': 'oop' if player == 0 else 'ip', 'node': self.tree.history_key(node),
                'actions': [o['label'] for o in opts], **_summary(evs, freqs)}


def _entropy(freqs) -> float:
    return float(-sum(f * math.log2(f) for f in freqs if f > 1e-12))


def _summary(evs: dict, freqs: dict) -> dict:
    best = max(evs, key=evs.get)
    top = sorted(evs.values(), reverse=True)
    return {'action_evs_bb': evs, 'gto_freqs': freqs, 'best_action': best, 'best_ev_bb': evs[best],
            'entropy': _entropy(freqs.values()), 'is_indifferent': len(top) > 1 and top[0] - top[1] <= INDIFFERENT_BB}


# ---- river: exact DCFR --------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _river_ranks(board: tuple) -> np.ndarray:
    """Dense integer strength of every combo on a complete board from `cards.evaluate` (-1: clashes)."""
    bmask = _mask(board)
    out = np.full(1326, -1, dtype=np.int64)
    idx = np.nonzero((COMBO_MASK & bmask) == 0)[0]
    scores = fast_rank(np.column_stack((COMBOS[idx], np.broadcast_to(board, (len(idx), 5)))))
    out[idx] = np.unique(scores, return_inverse=True)[1]
    return out


def _river_solution(board, pot, stack, w_oop, w_ip, iterations=300, bet_sizes=(0.5, 1.0), raise_sizes=(1.0,),
                    max_raises=2, bb=2, track=()):
    board = tuple(int(c) for c in board)
    rk = _river_ranks(board)
    tree = _Tree(pot, stack, bet_sizes, raise_sizes, max_raises, bb)
    sides = []
    for w in (w_oop, w_ip):
        live = np.nonzero((np.asarray(w) > 0) & (rk >= 0))[0]
        if not len(live):
            raise ValueError('a range has no combo that is possible on this board')
        sides.append(live)
    full_live, full_ev = _full_river_eval(board)
    if len(sides[0]) == len(full_live) and len(sides[1]) == len(full_live):
        evals = [full_ev, full_ev]
    elif max(len(sides[0]), len(sides[1])) >= 128:
        evals = [_FastRiverEval(COMBOS[sides[p]], rk[sides[p]], COMBOS[sides[1 - p]], rk[sides[1 - p]]) for p in (0, 1)]
    else:
        evals = [_ComboEval(COMBOS[sides[p]], rk[sides[p]], COMBOS[sides[1 - p]], rk[sides[1 - p]]) for p in (0, 1)]
    priors = [np.asarray(w_oop, dtype=float)[sides[0]], np.asarray(w_ip, dtype=float)[sides[1]]]
    cfr = _DCFR(tree, priors, evals)
    curve = cfr.run(iterations, track)
    class_of = [{int(c): k for k, c in enumerate(sides[p])} for p in (0, 1)]

    def equity(player, cid):
        cls = class_of[player].get(cid)
        if cls is None:
            ev = _ComboEval(COMBOS[[cid]], rk[[cid]], COMBOS[sides[1 - player]], rk[sides[1 - player]])
            cls = 0
        else:
            ev = evals[player]
        q = priors[1 - player][None]
        m = ev.mass(q)[0][cls]
        return float(0.5 + 0.5 * ev.sd(q)[0][cls] / m) if m > 0 else 0.5

    sol = _Solution(tree, cfr, sides, class_of, bb, 'river', 'dcfr_exact_river', equity, curve)
    sol.priors, sol.ranks = priors, rk
    return sol


def solve_river_subgame(board, pot, effective_stack, ip_range, oop_range, bet_sizes=(0.5, 1.0), iterations=300,
                        seed=0, *, bb=2, raise_sizes=(1.0,), max_raises=2, track=()) -> dict:
    """Exact DCFR on a river subgame. Ranges: weights over the 1326 combos, range text, {combo: weight} or a
    list of combos; combos that clash with the board are dropped. Deterministic (`seed` is accepted for the
    common solver signature). Returns {'strategy': {history: {'player', 'actions', 'freqs': {combo: [...]}}},
    'ev': {'oop': {combo: bb}, 'ip': {...}}, 'exploitability_mbb', 'root_ev_ip', 'root_ev_oop' (range-average
    EVs in bb; they sum to the pot), 'node_eval': callable(hero_hand, history=()), 'exploitability_curve'
    ([(iteration, mbb)] at the iterations in `track`), 'iterations'}."""
    b = _ids(board)
    if len(b) != 5:
        raise ValueError('a river subgame needs five board cards')
    sol = _river_solution(b, pot, effective_stack, range_weights(oop_range), range_weights(ip_range), iterations,
                          bet_sizes, raise_sizes, max_raises, bb, track)
    tree, avg = sol.tree, sol.avg
    strategy = {}
    for p in (0, 1):
        for node in tree.p_nodes[p]:
            edges = tree.child_edges(int(node))
            freqs = avg[p][tree.loc[edges]]
            strategy[tree.history_key(int(node))] = {
                'player': 'oop' if p == 0 else 'ip', 'actions': [tree.label[tree.e_ch[e]] for e in edges],
                'freqs': {_combo_text(int(c)): [round(float(x), 6) for x in freqs[:, j]]
                          for j, c in enumerate(sol.classes[p])}}
    ev, root = {}, {}
    half = tree.pot / 2
    for p, name in ((0, 'oop'), (1, 'ip')):
        V, reach_o, _ = sol.vals[p]
        m = sol.cfr.ev[p].mass(sol.priors[1 - p][None])[0]
        V0 = sol.cfr.backward(p, avg[p], sol.cfr.forward(1 - p, avg[1 - p]))[0]
        per = np.where(m > 0, V0 / np.where(m > 0, m, 1) + half, half) / sol.bb
        ev[name] = {_combo_text(int(c)): float(per[j]) for j, c in enumerate(sol.classes[p])}
        root[name] = float((sol.priors[p] @ V0) / sol.cfr.total_mass() + half) / sol.bb
    return {'strategy': strategy, 'ev': ev, 'exploitability_mbb': sol.exploitability_mbb,
            'root_ev_ip': root['ip'], 'root_ev_oop': root['oop'], 'node_eval': sol.node_eval,
            'exploitability_curve': sol.curve, 'iterations': sol.cfr.iterations, 'solution': sol}


# ---- flop and turn: equity-bucketed DCFR --------------------------------------------------------

def _representatives(w, valid, strength, k):
    """Up to k combos standing for a weighted range: strength strata of equal weight, one combo per stratum
    (the one at its middle) carrying the stratum's weight."""
    idx = np.nonzero(valid & (w > 0))[0]
    if len(idx) <= k:
        return idx, w[idx].astype(float)
    order = idx[np.lexsort((idx, strength[idx]))]
    cw = np.cumsum(w[order])
    total = cw[-1]
    edges = np.searchsorted(cw, np.linspace(0, total, k + 1)[1:-1], 'left')
    bounds = np.concatenate([[0], edges, [len(order)]])
    reps, wts = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        seg = order[a:b]
        mid = cw[a] - w[order[a]] + (cw[b - 1] - cw[a] + w[order[a]]) / 2
        j = a + int(np.clip(np.searchsorted(cw[a:b], mid, 'left'), 0, len(seg) - 1))
        reps.append(order[j])
        wts.append(float(w[seg].sum()))
    return np.array(reps, dtype=np.int64), np.array(wts)


def _abstract_solution(board, pot, stack, w_oop, w_ip, hero_p, hero_cid, iterations=200, k=40, bet_sizes=(0.5, 1.0),
                       raise_sizes=(1.0,), max_raises=2, bb=2, n_eq=12, n_pot=2, runouts=30, allin_cap=2.5):
    board = tuple(int(c) for c in board)
    bmask = _mask(board)
    valid = (COMBO_MASK & bmask) == 0
    strength = _board_strength(board)
    hero_cid = int(hero_cid)
    reps = []
    for p, w in ((0, w_oop), (1, w_ip)):
        w = np.asarray(w, dtype=float).copy()
        hw = w[hero_cid] if p == hero_p else 0.0
        if p == hero_p:
            w[hero_cid] = 0.0
        r, rw = _representatives(w, valid, strength, k)
        if p == hero_p:
            r, rw = np.append(r, hero_cid), np.append(rw, max(hw, 1e-3 * max(rw.max(initial=1.0), 1.0)))
        reps.append((r, rw))
    need = 5 - len(board)
    rest = [c for c in range(52) if not (bmask >> c) & 1]
    if need == 1:
        runs = [[c] for c in rest]
    else:
        rng = random.Random(f'runouts|{board}')
        runs = [rng.sample(rest, need) for _ in range(runouts)]
    runs = np.array(runs, dtype=np.int64)
    R = len(runs)
    run_mask = (np.int64(1) << runs).sum(1) if need else np.zeros(R, dtype=np.int64)
    ranks, live = [], []
    for r, _ in reps:
        cc = COMBOS[r]
        full = np.concatenate([np.repeat(cc[None], R, 0), np.broadcast_to(np.array(board), (R, len(r), len(board))),
                               np.repeat(runs[:, None, :], len(r), 1)], 2)
        ranks.append(fast_rank(full.reshape(-1, 7)).reshape(R, len(r)))
        live.append((COMBO_MASK[r][None, :] & run_mask[:, None]) == 0)
    (ro, wo), (ri, wi) = reps
    Wc = (COMBO_MASK[ro][:, None] & COMBO_MASK[ri][None, :]) == 0
    okw = live[0][:, :, None] & live[1][:, None, :] & Wc[None]
    share = _share(ranks[0][:, :, None], ranks[1][:, None, :])
    cnt = okw.sum(0)
    E = np.where(cnt > 0, (share * okw).sum(0) / np.maximum(cnt, 1), 0.5)
    pair_mass = wo[:, None] * wi[None, :] * (Wc & (cnt > 0))
    per_run = []
    for num, den in ((((share * okw) * wi[None, None, :]).sum(2), (okw * wi[None, None, :]).sum(2)),
                     ((((1 - share) * okw) * wo[None, :, None]).sum(1), (okw * wo[None, :, None]).sum(1))):
        per_run.append((np.where(den > 0, num / np.maximum(den, 1e-300), 0.0), den > 0))
    eqs = [(pair_mass * E).sum(1) / np.maximum(pair_mass.sum(1), 1e-300),
           (pair_mass * (1 - E)).sum(0) / np.maximum(pair_mass.sum(0), 1e-300)]
    groups, class_of = [], [{}, {}]
    for p, (r, rw) in enumerate(reps):
        pr, ok_r = per_run[p]
        n_ok = np.maximum(ok_r.sum(0), 1)
        mean = (pr * ok_r).sum(0) / n_ok
        potential = np.sqrt((((pr - mean) ** 2) * ok_r).sum(0) / n_ok)
        hero_row = len(r) - 1 if p == hero_p else -1
        body = np.arange(len(r)) if hero_row < 0 else np.arange(len(r) - 1)
        eb = _weighted_bins(eqs[p][body], rw[body], n_eq)
        pb = (potential[body] > _weighted_quantile(potential[body], rw[body], 0.5)).astype(int) if n_pot > 1 else \
            np.zeros(len(body), dtype=int)
        b = eb * n_pot + pb
        uniq = sorted(set(b.tolist()))
        remap = {x: j for j, x in enumerate(uniq)}
        g = np.array([remap[x] for x in b.tolist()], dtype=np.int64)
        n_cls = len(uniq)
        if hero_row >= 0:
            g = np.append(g, n_cls)
            n_cls += 1
        G = np.zeros((n_cls, len(r)))
        G[g, np.arange(len(r))] = 1.0
        groups.append(G)
        for j, c in enumerate(r):
            class_of[p][int(c)] = int(g[j])
    G0, G1 = groups
    M = (G0 @ pair_mass @ G1.T).astype(np.float32)
    SdM = (G0 @ (pair_mass * (2 * E - 1)) @ G1.T).astype(np.float32)
    tree = _Tree(pot, stack, bet_sizes, raise_sizes, max_raises, bb, allin_cap=allin_cap)
    evals = [_DenseEval(M, SdM), _DenseEval(M.T, -SdM.T)]
    cfr = _DCFR(tree, [np.ones(G0.shape[0]), np.ones(G1.shape[0])], evals)
    cfr.run(iterations)
    hero_eq = float(eqs[hero_p][-1])

    def equity(player, cid):
        return hero_eq if (player == hero_p and int(cid) == hero_cid) else 0.5

    street = 'flop' if len(board) == 3 else 'turn'
    return _Solution(tree, cfr, [groups[0], groups[1]], class_of, bb, street, 'dcfr_abstracted', equity)


def _weighted_quantile(x, w, q):
    if not len(x):
        return 0.0
    o = np.argsort(x, kind='stable')
    cw = np.cumsum(w[o])
    return float(x[o][min(len(x) - 1, int(np.searchsorted(cw, q * cw[-1])))])


def _weighted_bins(x, w, n):
    if not len(x):
        return np.zeros(0, dtype=int)
    o = np.argsort(x, kind='stable')
    cw = np.cumsum(w[o])
    out = np.empty(len(x), dtype=int)
    out[o] = np.minimum(n - 1, (n * (cw - w[o] / 2) / max(cw[-1], 1e-300)).astype(int))
    return out


# ---- canonical spots, hashing, caching ----------------------------------------------------------

_SUIT_PERMS = list(itertools.permutations(range(4)))


def _apply_perm(perm, c):
    return (int(c) & ~3) | perm[int(c) & 3]


def _canon_cards(board_ids, hole_ids, perm):
    b = [_apply_perm(perm, c) for c in board_ids]
    b = sorted(b[:3], reverse=True) + b[3:]
    return b, sorted((_apply_perm(perm, c) for c in hole_ids), reverse=True)


def canonical_perm(board, hole=()) -> tuple:
    """The suit relabelling that puts (board, hole) into canonical form: the lexicographically smallest of the
    24 relabellings, with the flop as a set, then turn, then river, then the hole as a set. Suit-isomorphic
    spots map to identical canonical cards (a canonical choice of 'suits in order of first appearance')."""
    return _canonical_perm(tuple(_ids(board)), tuple(_ids(hole)))


@lru_cache(maxsize=4096)
def _canonical_perm(board, hole):
    return max(_SUIT_PERMS, key=lambda perm: _canon_cards(board, hole, perm))


def spot_hash(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb=2, solver='dcfr',
              extra=None) -> str:
    """A canonical, suit-isomorphic and seat-label-free hash of a decision: cards in canonical suits, seats as
    hero/villain, the action history, the pot and stack geometry, the solver settings."""
    perm = canonical_perm(board, hole)
    b, h = _canon_cards(_ids(board), _ids(hole), perm)
    rel = lambda xs: [xs[seat], xs[1 - seat]]
    obj = {'v': SOLVER_VERSION, 'street': street, 'board': [DECK[c] for c in b], 'hole': [DECK[c] for c in h],
           'pot': round(float(pot), 4), 'to_call': round(float(to_call), 4), 'stacks': rel(list(stacks)),
           'street_bet': rel(list(street_bet)), 'hero_button': seat == button, 'bb': bb, 'solver': solver,
           'history': [['h' if x['seat'] == seat else 'v', x['street'], x['action'], x.get('amount'),
                        bool(x.get('all_in'))] for x in history], 'extra': extra}
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


_MEMO: OrderedDict = OrderedDict()
_MEMO_MAX = 512


def _memo_get(key):
    v = _MEMO.get(key)
    if v is not None:
        _MEMO.move_to_end(key)
    return v


def _memo_put(key, v):
    _MEMO[key] = v
    _MEMO.move_to_end(key)
    while len(_MEMO) > _MEMO_MAX:
        _MEMO.popitem(last=False)


def _cache_read(cache_dir, h):
    if not cache_dir:
        return None
    p = Path(cache_dir) / f'{h}.json'
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _cache_write(cache_dir, h, obj):
    if not cache_dir:
        return
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'.{h}.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(obj, sort_keys=True))
    tmp.replace(d / f'{h}.json')


# ---- SpotEval assembly --------------------------------------------------------------------------

def _legal(to_call, raise_bounds):
    out = ['fold', 'call'] if to_call > 0 else ['check']
    if raise_bounds:
        out += ['raise', 'all_in']
    return out


def _aggregate(opts, legal):
    """Sized options -> abstract action EVs (the best size of each) and frequencies (summed), legal only."""
    evs, freqs = {}, {}
    for o in opts:
        a = o['abstract']
        if legal is not None and a not in legal:
            continue
        if o['ev'] is not None:
            evs[a] = max(evs.get(a, -1e18), o['ev'])
        freqs[a] = freqs.get(a, 0.0) + o['freq']
    total = sum(freqs.values())
    if total <= 0:
        freqs = {a: 1.0 / len(freqs) for a in freqs}
    else:
        freqs = {a: f / total for a, f in freqs.items()}
    return evs, freqs


def _make_spot(street, h, solver, pot, to_call, bb, equity, opts, legal, expl, source='solved'):
    evs, freqs = _aggregate(opts, legal)
    if not freqs:
        evs, freqs = {'check' if to_call == 0 else 'fold': 0.0}, {'check' if to_call == 0 else 'fold': 1.0}
    s = _summary(evs, freqs) if evs else dict(best_action=max(freqs, key=freqs.get), action_evs_bb={},
        gto_freqs=freqs, entropy=_entropy(freqs.values()), is_indifferent=False)
    candidates = [o for o in opts if o['abstract'] == s['best_action']]
    best = max(candidates, key=lambda o: o['ev'] if evs else o['freq'], default={})
    return {'street': street, 'spot_hash': h, 'solver': solver, 'pot_bb': pot / bb, 'to_call_bb': to_call / bb,
            'pot_odds': to_call / (pot + to_call) if to_call else 0.0, 'mdf': pot / (pot + to_call) if to_call else 1.0,
            'equity': equity, 'root_equity': equity, 'eval_source': source,
            'ev_reliable': bool(evs) and (source == 'solved' or street == 'preflop'),
            'reach_fraction': None, 'node_reach': None, 'tremble_share': None,
            'best_to': best.get('to') if best.get('abstract') in ('raise', 'all_in') else None,
            'mapped_to': None, 'taken_to': None, 'sizing_loss_mbb': None, 'action_loss_mbb': None,
            'best_action': s['best_action'], 'action_evs_bb': s['action_evs_bb'],
            'gto_freqs': s['gto_freqs'], 'taken_action': None, 'taken_ev_bb': None, 'ev_loss_mbb': None,
            'entropy': s['entropy'], 'is_indifferent': s['is_indifferent'], 'is_blunder': False,
            'exploitability_mbb': expl}


def _taken(action, amount, all_in=False):
    a = {'bet': 'raise', 'allin': 'all_in', 'shove': 'all_in'}.get(action, action)
    if a == 'raise' and all_in:
        a = 'all_in'
    return a


def _fill_taken(spot, opts, action, amount, all_in=False):
    if action is None:
        return spot
    a = _taken(action, amount, all_in)
    evs = spot['action_evs_bb']
    ev = None
    sized = [o for o in opts if o['abstract'] in ('raise', 'all_in') and o.get('to')]
    if a in ('raise', 'all_in') and amount is not None and sized:
        same = [o for o in sized if o['abstract'] == a] or sized
        mapped = min(same, key=lambda o: abs(math.log(max(amount, 1e-6) / max(o['to'], 1e-6))))
        spot.update(taken_to=amount, mapped_to=mapped['to'])
        ev = mapped['ev']
    if ev is None:
        ev = evs.get(a)
    if ev is None and a == 'all_in':
        ev = evs.get('raise')
    if ev is None and a == 'raise':
        ev = evs.get('all_in')
    if ev is None:
        ev = evs.get({'call': 'check', 'check': 'call'}.get(a, a))
    spot['taken_action'] = a
    if not spot['ev_reliable']:
        spot.update(ev_loss_mbb=0.0, taken_ev_bb=ev)
        return spot
    if ev is None:
        return spot
    loss = max(0.0, (max(evs.values()) - ev) * 1000)
    action_loss = max(0.0, (max(evs.values()) - evs.get(a, ev)) * 1000)
    spot.update(taken_ev_bb=float(ev), ev_loss_mbb=loss, is_blunder=loss >= BLUNDER_MBB,
                action_loss_mbb=action_loss, sizing_loss_mbb=max(0.0, loss - action_loss))
    return spot


# ---- preflop and heuristic spots ----------------------------------------------------------------

def _sample_villain(w, known_mask, n, seed_key):
    valid = ((COMBO_MASK & known_mask) == 0) & (w > 0)
    p = np.where(valid, w, 0.0)
    if p.sum() <= 0:
        p = ((COMBO_MASK & known_mask) == 0).astype(float)
    p = p / p.sum()
    nr = np.random.default_rng(int(hashlib.sha1(seed_key.encode()).hexdigest()[:12], 16))
    return nr.choice(1326, size=n, p=p), nr


def _equity_samples(hole_ids, board_ids, w, n, seed_key):
    """(shares, villain combos): hero's showdown share against n villain combos drawn from the weights, each
    with its own random runout."""
    kmask = _mask(list(hole_ids) + list(board_ids))
    vs, nr = _sample_villain(w, kmask, n, seed_key)
    need = 5 - len(board_ids)
    rows = np.full(n, kmask, dtype=np.int64) | COMBO_MASK[vs]
    run = _random_runouts(rows, need, nr) if need else np.zeros((n, 0), dtype=np.int64)
    board = np.broadcast_to(np.array(list(board_ids), dtype=np.int64), (n, len(board_ids)))
    hero = fast_rank(np.concatenate([np.broadcast_to(np.array(hole_ids), (n, 2)), board, run], 1))
    vil = fast_rank(np.concatenate([COMBOS[vs], board, run], 1))
    return _share(hero, vil), vs


def _raise_options(street, d, eq_fn, real, raise_tos, shove_fn):
    """EVs (chips won from now) of raising to each size. Raises below all in: the villain continues with the
    top `cont` of its range (the larger of 1.5x the minimum defence against the size and the share whose
    equity can stand the price), re-raises (to 3x) with the top 20% of what continues and calls with the
    rest; over a re-raise the hero folds or shoves, whichever is worth more. Every all in (the hero's shove,
    or a shove over a re-raise) meets `shove_fn`: a villain that calls exactly when its equity against the
    hero's actual hand beats the price."""
    seat = d['seat']
    opp = 1 - seat
    pot, sb = d['pot'], d['street_bet']
    total_s, total_o = sb[seat] + d['stacks'][seat], sb[opp] + d['stacks'][opp]
    eff = min(total_s, total_o)
    dead = pot - sb[seat] - sb[opp]
    out = []
    for label, to in raise_tos:
        to = min(to, total_s)
        a = to - sb[seat]
        dd = max(0.0, min(to, total_o) - sb[opp])
        allin = to >= total_s - 1e-9 or to >= total_o - 1e-9
        if allin:
            ev = shove_fn(a, dd)
        else:
            odds_v = dd / (pot + a + dd) if dd > 0 else 0.0
            cont = float(np.clip(max(1.5 * pot / (pot + dd), 0.4 * (1 - odds_v) ** 2), 0.04, 0.95)) if dd > 0 else 1.0
            f, r = 1 - cont, 0.2 * cont
            c = cont - r
            ev = f * pot + c * (eq_fn(cont) * real * (pot + a + dd) - a)
            rr = min(3 * to, eff)
            if rr >= eff - 1e-9:     # the re-raise is all in: call it or fold
                back = eq_fn(r) * (dead + 2 * eff) - (eff - sb[seat])
            else:                    # shove over it: the re-raisers call exactly when their equity allows
                back = shove_fn(eff - sb[seat], eff - rr, frac=r, win=pot + rr - sb[opp], pot_now=dead + eff + rr)
            ev += r * max(-a, back)
        out.append({'label': label, 'abstract': 'all_in' if label == 'allin' else 'raise', 'to': float(to), 'ev': ev})
    return out


def _softmax_freqs(opts, tau):
    evs = np.array([o['ev'] for o in opts])
    z = np.exp((evs - evs.max()) / max(tau, 1e-9))
    z /= z.sum()
    for o, f in zip(opts, z):
        o['freq'] = float(f)
    return opts


def _heuristic_options(d, hole_ids, board_ids, w_v, seed_key, samples=160):
    """Sized options (EVs in bb, chips won from now) for preflop and for spots the tree can't follow: equity
    against the villain's range and against the top fractions of it that continue (each from its own
    deterministic sample), a position/SPR realization factor and fold equity; frequencies a softmax of EVs."""
    seat, street = d['seat'], d['street']
    strength = _board_strength(tuple(board_ids)) if board_ids else _pre_strength()
    kmask = _mask(list(hole_ids) + list(board_ids))
    live = np.where((COMBO_MASK & kmask) == 0, np.asarray(w_v, dtype=float), 0.0)
    shares, _ = _equity_samples(hole_ids, board_ids, live, samples, seed_key)
    eq_all = float(shares.mean())
    memo = {}

    def eq_top(frac, bottom=False):
        """Equity against the strongest (or, with bottom, the weakest) `frac` of the villain's range."""
        key = (round(frac, 3), bottom)
        if key not in memo:
            if bottom:
                cut = _weighted_quantile(strength, live, frac)
                sub = np.where(strength <= cut, live, 0.0)
            else:
                cut = _weighted_quantile(strength, live, max(0.0, 1 - frac)) if frac < 1 else -1.0
                sub = np.where(strength >= cut, live, 0.0)
            memo[key] = float(_equity_samples(hole_ids, board_ids, sub if sub.sum() > 0 else live, samples // 2 + 40,
                                              f'{seed_key}|{key}')[0].mean())
        return memo[key]

    pot, tc, bb = d['pot'], d['to_call'], d['bb']
    ip = seat == d['button']
    stack_total = d['stacks'][seat] + d['street_bet'][seat]
    spr = d['stacks'][seat] / max(pot, 1)
    if street == 'preflop':
        real = (0.97 if ip else 0.86) - (0.0 if ip else 0.02 * max(0.0, math.log2(max(spr, 1) / 10)))
    else:
        real = 1.0 if street == 'river' else (0.98 if ip else 0.92)

    def shove(add, dd, frac=1.0, win=None, pot_now=None):
        """All in for `add` more chips against the top `frac` of the villain's range, which must add `dd` to call
        into `pot_now` (default: the pot after the shove) and calls exactly when its equity against this hand
        beats that price; a fold wins `win` (default: the pot)."""
        key = ('shove', round(frac, 3))
        if key not in memo:
            sub = live
            if frac < 1:
                cut = _weighted_quantile(strength, live, max(0.0, 1 - frac))
                sub = np.where(strength >= cut, live, 0.0)
                sub = sub if sub.sum() > 0 else live
            vs, nr = _sample_villain(sub, kmask, 64, f'{seed_key}|{key}')
            need, reps = 5 - len(board_ids), 8
            rows = np.repeat(np.full(len(vs), kmask, dtype=np.int64) | COMBO_MASK[vs], reps)
            run = _random_runouts(rows, need, nr) if need else np.zeros((len(rows), 0), dtype=np.int64)
            b = np.broadcast_to(np.array(list(board_ids), dtype=np.int64), (len(rows), len(board_ids)))
            hero = fast_rank(np.concatenate([np.broadcast_to(np.array(hole_ids), (len(rows), 2)), b, run], 1))
            vil = fast_rank(np.concatenate([np.repeat(COMBOS[vs], reps, 0), b, run], 1))
            memo[key] = _share(hero, vil).reshape(len(vs), reps).mean(1)
        eq_h = memo[key]
        pot_now = pot + add if pot_now is None else pot_now
        win = pot if win is None else win
        total = pot_now + dd
        call = (1 - eq_h) >= dd / total if dd > 0 else np.ones(len(eq_h), dtype=bool)
        return float(np.where(call, eq_h * total - add, win).mean())

    opts = []
    if tc > 0:
        opts.append({'label': 'fold', 'abstract': 'fold', 'to': None, 'ev': 0.0})
        rl = 1.0 if tc >= d['stacks'][seat] else real
        call_ev = eq_all * rl * (pot + tc) - tc
        if street == 'preflop' and not any(h['street'] == 'preflop' and h['action'] in ('bet', 'raise', 'call')
                                           for h in d['history']) and d.get('raise_bounds'):
            # a limp: the big blind still acts, raising with the top 30% (the limper then folds)
            call_ev = 0.7 * (eq_top(0.7, bottom=True) * rl * (pot + tc) - tc) - 0.3 * tc
        opts.append({'label': 'call', 'abstract': 'call', 'to': None, 'ev': call_ev})
    else:
        opts.append({'label': 'check', 'abstract': 'check', 'to': None, 'ev': eq_all * real * pot})
    rb = d.get('raise_bounds')
    if rb:
        cur = max(d['street_bet'])
        tos = []
        if street == 'preflop':
            n_r = sum(h['action'] in ('bet', 'raise') for h in d['history'] if h['street'] == 'preflop')
            to = 2.5 * bb if n_r == 0 and ip else 3.5 * bb if n_r == 0 else 3 * cur if n_r == 1 else 2.3 * cur
            if to < 0.45 * stack_total:
                tos.append(('raise', float(np.clip(to, rb[0], rb[1]))))
        else:
            for f in (0.5, 1.0):
                to = cur + f * (pot + tc) + tc if tc else f * pot
                if to < 0.9 * stack_total:
                    tos.append((f'raise{f:g}', float(np.clip(to, rb[0], rb[1]))))
        tos.append(('allin', float(rb[1])))
        opts += _raise_options(street, d, eq_top, real, tos, shove)
    for o in opts:
        o['ev'] = float(o['ev']) / bb
    return _softmax_freqs(opts, 0.15 + 0.02 * pot / bb), eq_all


# ---- the spot API -------------------------------------------------------------------------------

def _street_root(pot, stacks, street_bet):
    p0 = pot - sum(street_bet)
    s0 = min(stacks[0] + street_bet[0], stacks[1] + street_bet[1])
    return float(p0), float(s0)


def _spot_core(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb=2, solver='auto',
               cache_dir=None, iterations=200, *, raise_bounds=None, use_external=False, sb=None, k=40,
               bet_sizes=(0.5, 1.0), river_cap=700, runouts=30):
    """(SpotEval without the taken action, sized options [{'label', 'abstract', 'to', 'ev', 'freq'}])."""
    hole_ids, board_ids = _ids(hole), _ids(board)
    if raise_bounds is None and stacks[seat] > 0 and stacks[1 - seat] > 0:
        cur = max(street_bet)
        raise_bounds = (min(cur + bb if cur else bb, street_bet[seat] + stacks[seat]), street_bet[seat] + stacks[seat])
    legal = _legal(to_call, raise_bounds)
    sb = sb if sb is not None else bb // 2
    extra = {'iterations': iterations, 'k': k, 'bet_sizes': list(bet_sizes), 'legal': legal, 'river_cap': river_cap,
             'runouts': runouts, 'sb': sb, 'raise_bounds': list(raise_bounds) if raise_bounds else None}
    h = spot_hash(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb, solver, extra)
    hit = _memo_get(('spot', h)) or _cache_read(cache_dir, h)
    if hit:
        _memo_put(('spot', h), hit)
        return dict(hit['spot']), [dict(o) for o in hit['opts']]
    seed = spot_hash(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb, 'seed')
    perm = canonical_perm(board_ids, hole_ids)
    cb, ch = _canon_cards(board_ids, hole_ids, perm)
    hero_cid = combo_index(ch)
    d = {'seat': seat, 'street': street, 'pot': pot, 'to_call': to_call, 'stacks': list(stacks),
         'street_bet': list(street_bet), 'bb': bb, 'button': button, 'history': list(history),
         'raise_bounds': raise_bounds}
    spot = opts = None
    if solver == 'texassolver' or (solver == 'auto' and use_external):
        ext = TexasSolverAdapter(cache_dir=cache_dir)
        if ext.available():
            got = ext.solve_spot(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb=bb,
                                 raise_bounds=raise_bounds, sb=sb, spot_hash_=h)
            if got:
                spot, opts = got
    if spot is None:
        if street == 'preflop':
            w = infer_ranges(history, (), button, bb, sb)
            opts, eq = _heuristic_options(d, ch, [], w[1 - seat], f'pre|{seed}')
            spot = _make_spot(street, h, 'preflop_lookup', pot, to_call, bb, eq, opts, legal, None, 'heuristic')
        else:
            before = [x for x in history if x['street'] != street]
            here = [x for x in history if x['street'] == street]
            w = infer_ranges(before, cb, button, bb, sb)
            oop_seat = 1 - button
            hero_p = 0 if seat == oop_seat else 1
            p0, s0 = _street_root(pot, stacks, street_bet)
            w_oop, w_ip = w[oop_seat].copy(), w[button].copy()
            sol = None
            try:
                if street == 'river':
                    sol = _river_spot(cb, p0, s0, w_oop, w_ip, iterations, bet_sizes, bb)
                else:
                    deep_in_here = any(x.get('all_in') or (x.get('amount') or 0) > 2.5 * p0 for x in here)
                    acap = None if deep_in_here else 2.5
                    key = ('abs', tuple(cb), p0, s0, _hist_sig(before), button, hero_p, hero_cid, iterations, k,
                           tuple(bet_sizes), runouts, bb, sb, acap)
                    sol = _memo_get(key)
                    if sol is None:
                        sol = _abstract_solution(cb, p0, s0, w_oop, w_ip, hero_p, hero_cid, iterations, k, bet_sizes,
                                                 bb=bb, runouts=runouts, allin_cap=acap)
                        _memo_put(key, sol)
            except ValueError:
                sol = None
            node = sol.node_of(here) if sol else None
            cls = sol.class_of[hero_p].get(hero_cid) if sol else None
            sized = None
            if sol and node is not None and cls is not None:
                sized = sol.sized(hero_p, cls, node)
            if sized:
                if 'all_in' in legal and not any(o['abstract'] == 'all_in' for o in sized):
                    dev = sol.deep_shove_ev(hero_p, cls, node)
                    if dev is not None:
                        cap_ev = max(o['ev'] for o in sized)
                        sized = list(sized) + [{'label': 'allin', 'abstract': 'all_in', 'to': float(s0),
                                                'ev': min(dev, cap_ev), 'freq': 0.0}]
                opts = sized
                eq = sol.equity_of(hero_p, hero_cid)
                spot = _make_spot(street, h, sol.solver, pot, to_call, bb, eq, opts, legal, sol.exploitability_mbb)
                spot.update(sol.reach_stats(hero_p, cls, node))
            else:
                wv = infer_ranges(history, cb, button, bb, sb)[1 - seat]
                opts, eq = _heuristic_options(d, ch, cb, wv, f'heur|{seed}')
                spot = _make_spot(street, h, 'dcfr_exact_river' if street == 'river' else 'dcfr_abstracted', pot,
                                  to_call, bb, eq, opts, legal, None, 'heuristic')
    opts = [{k2: (float(v) if isinstance(v, (np.floating,)) else v) for k2, v in o.items()} for o in opts]
    rec = {'spot': spot, 'opts': opts}
    _memo_put(('spot', h), rec)
    _cache_write(cache_dir, h, rec)
    return dict(spot), [dict(o) for o in opts]


def _river_spot(cb, p0, s0, w_oop, w_ip, iterations, bet_sizes, bb, max_raises=4):
    """Share the full public game, independent of which private hand is queried."""
    return _public_river(tuple(cb), p0, s0, w_oop.tobytes(), w_ip.tobytes(), iterations, tuple(bet_sizes), bb, max_raises)


@lru_cache(maxsize=64)
def _public_river(board, pot, stack, oop, ip, iterations, sizes, bb, max_raises=4):
    return _river_solution(board, pot, stack, np.frombuffer(oop), np.frombuffer(ip), iterations, sizes,
                           max_raises=max_raises, bb=bb)


def solve_postflop_spot(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb=2,
                        solver='auto', cache_dir=None, iterations=200, *, taken_action=None, taken_amount=None,
                        taken_all_in=False, raise_bounds=None, use_external=False, sb=None, k=40, river_cap=700,
                        runouts=30) -> dict:
    """The SpotEval of one decision (any street; the name follows the architecture contract). Ranges come from
    `infer_ranges` of the history; the river is solved exactly, the flop and turn by the bucketed DCFR, preflop
    by the range/position/SPR model; `solver='texassolver'` (or 'auto' with use_external) tries the external
    binary first. With `taken_action` (fold/check/call/raise/bet/all_in, `taken_amount` = raise-to) the taken
    EV and EV loss are filled. Cached in memory and, with `cache_dir`, on disk by `spot_hash`."""
    spot, opts = _spot_core(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb, solver,
                            cache_dir, iterations, raise_bounds=raise_bounds, use_external=use_external, sb=sb, k=k,
                            river_cap=river_cap, runouts=runouts)
    _fill_taken(spot, opts, taken_action, taken_amount, taken_all_in)
    if (iterations in (FAST['iterations'], FAST['river_iterations']) and spot['solver'] != 'texassolver'
            and spot['eval_source'] == 'solved' and (spot['ev_loss_mbb'] or 0) >= BLUNDER_MBB):
        return solve_postflop_spot(street, hole, board, pot, to_call, stacks, street_bet, seat, button,
            history, bb, solver, cache_dir, 300, taken_action=taken_action, taken_amount=taken_amount,
            taken_all_in=taken_all_in, raise_bounds=raise_bounds, sb=sb, k=k, river_cap=river_cap, runouts=runouts)
    return spot


# ---- TexasSolver --------------------------------------------------------------------------------

class TexasSolverAdapter:
    """The optional TexasSolver `console_solver` binary behind the SpotEval schema. Found through
    $POKER_ARENA_TEXASSOLVER, <repo>/.tools/TexasSolver/build/console_solver, <repo>/.tools/TexasSolver/
    console_solver or `console_solver` on PATH. Runs in a fresh temporary folder, results cached by spot_hash;
    any failure returns None so callers fall back to the built-in DCFR."""

    STREET_NAMES = ('flop', 'turn', 'river')

    def __init__(self, binary=None, cache_dir=None, threads=4, accuracy=0.5, max_iteration=200, timeout=120,
                 bet_sizes=(50, 100), raise_sizes=(100,), allin_threshold=1.0):
        self.binary = Path(binary) if binary else self.find_binary()
        self.cache_dir, self.threads, self.accuracy = cache_dir, threads, accuracy
        self.max_iteration, self.timeout = max_iteration, timeout
        self.bet_sizes, self.raise_sizes, self.allin_threshold = tuple(bet_sizes), tuple(raise_sizes), allin_threshold

    @staticmethod
    def find_binary() -> Path | None:
        env = os.environ.get('POKER_ARENA_TEXASSOLVER')
        repo = Path(__file__).resolve().parent.parent
        for cand in ([Path(env)] if env else []) + [repo / '.tools/TexasSolver/build/console_solver',
                                                     repo / '.tools/TexasSolver/console_solver']:
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
        w = shutil.which('console_solver')
        return Path(w) if w else None

    def available(self) -> bool:
        return bool(self.binary and Path(self.binary).is_file() and os.access(self.binary, os.X_OK))

    @staticmethod
    def range_text(w, exclude=()) -> str:
        """A range in TexasSolver notation: hand classes with weights ('AA:1.000,AKs:0.500')."""
        w = np.asarray(w, dtype=float)
        kmask = _mask(_ids(exclude))
        live = np.flatnonzero((w > 0) & ((COMBO_MASK & kmask) == 0))
        if not len(live):
            return ''
        scale = max(1.0, 0.01 / w[live].min())
        classes = {}
        for cid in range(1326):
            if COMBO_MASK[cid] & kmask:
                continue
            cls = _ranges.hand_class((DECK[COMBOS[cid, 0]], DECK[COMBOS[cid, 1]]))
            classes.setdefault(cls, []).append(w[cid])
        if not exclude and all(min(v) == max(v) and float(f'{v[0]*scale:.3f}') == v[0]*scale for v in classes.values()):
            return ','.join(f'{c}:{v[0]*scale:.3f}' for c, v in sorted(classes.items()) if v[0] > 0)
        return ','.join(f'{COMBO_STR[cid]}:{w[cid]*scale:.17g}' for cid in live)

    def build_commands(self, board, pot, effective_stack, range_ip, range_oop, dump='output_result.json') -> str:
        b = ','.join(board)
        lines = [f'set_pot {pot:g}', f'set_effective_stack {effective_stack:g}', f'set_board {b}',
                 f'set_range_ip {range_ip}', f'set_range_oop {range_oop}']
        first = {3: 0, 4: 1, 5: 2}[len(board)]
        for street in self.STREET_NAMES[first:]:
            for side in ('oop', 'ip'):
                lines.append(f'set_bet_sizes {side},{street},bet,' + ','.join(f'{x:g}' for x in self.bet_sizes))
                lines.append(f'set_bet_sizes {side},{street},raise,' + ','.join(f'{x:g}' for x in self.raise_sizes))
                lines.append(f'set_bet_sizes {side},{street},allin')
        lines += [f'set_allin_threshold {self.allin_threshold:g}', 'build_tree', f'set_thread_num {self.threads}',
                  f'set_accuracy {self.accuracy:g}', f'set_max_iteration {self.max_iteration}',
                  'set_print_interval 10', 'set_use_isomorphism 1', 'start_solve', 'set_dump_rounds 1',
                  f'dump_result {dump}']
        return '\n'.join(lines) + '\n'

    def run(self, commands: str) -> dict | None:
        self._last_exploitability_pct = None
        if not self.available():
            return None
        try:
            with tempfile.TemporaryDirectory(prefix='texassolver-') as tmp:
                inp = Path(tmp) / 'input.txt'
                inp.write_text(commands)
                args = [str(self.binary), '-i', str(inp)]
                bdir = Path(self.binary).resolve().parent
                res = next((cand for cand in (bdir / 'resources', bdir.parent / 'resources') if cand.is_dir()), None)
                if res is not None:
                    args += ['-r', str(res)]
                proc = subprocess.run(args, cwd=tmp, timeout=self.timeout, capture_output=True, text=True, check=False)
                # TexasSolver prints "Total exploitability 0.4965554 precent" (sic) on stdout
                for m in re.finditer(r'Total\s+exploitability\s+([0-9.]+)\s+p(?:re|er)cent', proc.stdout or '', re.I):
                    self._last_exploitability_pct = float(m.group(1))
                out = Path(tmp) / 'output_result.json'
                return json.loads(out.read_text()) if out.exists() else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _tx_label(action: str):
        """'CHECK' / 'CALL' / 'FOLD' / 'BET 12.0' / 'RAISE 30.0' -> (kind, amount)."""
        parts = action.strip().split()
        kind = parts[0].lower()
        amt = float(parts[1]) if len(parts) > 1 else None
        return {'bet': 'raise'}.get(kind, kind), amt

    def parse_result(self, tree: dict, hero_combo, history_here, stack, bb, path=None):
        """The sized options of the hero at the node the street's history leads to: [{'label', 'abstract',
        'to', 'ev', 'freq'}]; EVs from the dump's 'evs' when present, else None (the caller fills them)."""
        node = tree
        invested = [0., 0.]
        for h in history_here:
            if not node or node.get('node_type') != 'action_node':
                return None
            acts = node.get('actions') or []
            a = {'bet': 'raise'}.get(h['action'], h['action'])
            want = h.get('amount')
            player = node.get('player', 0)
            if a == 'raise' and want is not None:
                want -= invested[player]
            best, bd = None, None
            for x in acts:
                kind, amt = self._tx_label(x)
                if kind != a:
                    continue
                dist = 0.0 if want is None or amt is None else abs(math.log(max(want, 1e-6) / max(amt, 1e-6)))
                if bd is None or dist < bd:
                    best, bd = x, dist
            if best is None:
                return None
            if path is not None:
                path.append((node, best))
            if a == 'raise':
                invested[player] += self._tx_label(best)[1]
            node = (node.get('childrens') or {}).get(best)
        if not node or node.get('node_type') != 'action_node':
            return None
        strat = node.get('strategy') or {}
        acts = strat.get('actions') or node.get('actions') or []
        table = strat.get('strategy') or {}
        c = _parse_combo(hero_combo)
        probs = table.get(c[0] + c[1]) or table.get(c[1] + c[0])
        if probs is None:
            return None
        evs_tab = (node.get('evs') or {}).get('evs') or {}
        evs = evs_tab.get(c[0] + c[1]) or evs_tab.get(c[1] + c[0])
        out = []
        for j, x in enumerate(acts):
            kind, amt = self._tx_label(x)
            if amt is not None:
                amt += invested[node.get('player', 0)]
            abstract = kind
            if kind in ('raise',) and amt is not None and amt >= 0.98 * stack:
                abstract = 'all_in'
            out.append({'label': x, 'abstract': abstract, 'to': amt, 'freq': float(probs[j]),
                        'ev': None if evs is None else float(evs[j]) / bb})
        return out

    def _river_values(self, root, path, opts, hole, board, weights, pot, bb):
        """Integrate terminal chip payouts against the two policies actually in the native dump."""
        hero = combo_index(_ids(hole))
        live = np.flatnonzero((weights > 0) & ((COMBO_MASK & _mask(_ids([*hole, *board]))) == 0))
        ranks = _river_ranks(tuple(_ids(board)))
        shares = _share(ranks[hero], ranks[live])
        reach = weights[live].copy()
        root_mass = reach.sum()
        root_eq = float(reach @ shares / root_mass)
        smooth = reach.copy()
        target = root
        for parent, label in path:
            target = parent['childrens'][label]
        hero_player = target['player']

        def policy(node):
            table = node['strategy']['strategy']
            ids = [hero] if node['player'] == hero_player else live
            rows = []
            for cid in ids:
                key = COMBO_STR[cid]
                rows.append(table[key] if key in table else table[key[2:] + key[:2]])
            return np.asarray(rows).T

        def contributions(node, label, prior):
            costs = prior.copy()
            actor = node['player']
            kind, amt = self._tx_label(label)
            if amt is not None:
                costs[actor] += amt
            elif kind == 'call':
                costs[actor] = costs[1 - actor]
            return costs

        costs = [0., 0.]
        for node, label in path:
            costs = contributions(node, label, costs)
            if node['player'] != hero_player:
                prob = policy(node)[node['actions'].index(label)]
                reach *= prob
                smooth *= (1 - 1e-4) * prob + 1e-4 / len(node['actions'])
        mass = reach.sum()
        fraction = float(mass / root_mass)
        tremble = float(np.clip(1 - mass / smooth.sum(), 0, 1)) if smooth.sum() else 1.

        def payout(node, label, prior, remaining):
            costs = contributions(node, label, prior)
            if label == 'FOLD':
                payoff = -costs[hero_player] if node['player'] == hero_player else pot + costs[1-hero_player]
                return payoff * remaining.sum()
            child = node.get('childrens', {}).get(label)
            if child is None:
                if label not in ('CHECK', 'CALL'):
                    raise ValueError('missing native continuation')
                return (pot + sum(costs)) * (remaining @ shares) - costs[hero_player] * remaining.sum()
            return sum(payout(child, action, costs, remaining * prob) for action, prob in zip(child['actions'], policy(child)))

        if mass and any(opt['ev'] is None for opt in opts):
            values = [payout(target, opt['label'], costs, reach) / (mass * bb) for opt in opts]
            for opt, value in zip(opts, values):
                opt['ev'] = float(value)
        return dict(equity=float(reach @ shares / mass) if mass else None, root_equity=root_eq,
                    node_reach=float(mass), reach_fraction=fraction, tremble_share=tremble,
                    ev_reliable=fraction >= MIN_GRADE_REACH and tremble <= .5)

    def solve_spot(self, street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb=2,
                   raise_bounds=None, sb=None, spot_hash_=None):
        """Native policy and available EVs; reconstruct missing river EVs from the native policy tree."""
        if street == 'preflop' or not self.available():
            return None
        h = spot_hash_ or spot_hash(street, hole, board, pot, to_call, stacks, street_bet, seat, button, history, bb,
                                    'texassolver')
        key = f'tx-{h}'
        hit = _cache_read(self.cache_dir, key)
        if hit:
            return hit['spot'], hit['opts']
        before = [x for x in history if x['street'] != street]
        here = [x for x in history if x['street'] == street]
        w = infer_ranges(before, board, button, bb, sb)
        p0, s0 = _street_root(pot, stacks, street_bet)
        cmds = self.build_commands(list(board), p0, s0, self.range_text(w[button], board),
                                   self.range_text(w[1 - button], board))
        tree = self.run(cmds)
        if not tree:
            return None
        path = []
        opts = self.parse_result(tree, hole, here, s0, bb, path)
        if not opts:
            return None
        stats = {}
        if street == 'river':
            try:
                stats = self._river_values(tree, path, opts, hole, board, w[1-seat], p0, bb)
            except (KeyError, ValueError):
                pass
        legal = _legal(to_call, raise_bounds if raise_bounds is not None else
                       ((0, 0) if stacks[seat] > 0 and stacks[1 - seat] > 0 else None))
        pct = getattr(self, '_last_exploitability_pct', None)
        expl_mbb = None if pct is None else round(pct / 100.0 * p0 / bb * 1000.0, 2)
        spot = _make_spot(street, h, 'texassolver', pot, to_call, bb, None, opts, legal, expl_mbb)
        spot.update(stats)
        _cache_write(self.cache_dir, key, {'spot': spot, 'opts': opts})
        return spot, opts


# ---- Slumbot ------------------------------------------------------------------------------------

SLUMBOT_URL = 'https://slumbot.com'


def _urllib_transport(url: str, payload: dict, timeout: float = 20.0) -> dict:
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (fixed https endpoint)
        return json.loads(r.read().decode())


class SlumbotClient:
    """The Slumbot API (https://slumbot.com/api/new_hand and /api/act): blinds 50/100, stacks 20,000. Position
    0 is the big blind (acts first after the flop), 1 the small blind (button). Action strings: 'k' check,
    'c' call, 'f' fold, 'b<n>' bet or raise to n this street, streets separated by '/'. `transport(url,
    payload) -> dict` is injectable, so tests run offline."""

    SB, BB, STACK = 50, 100, 20000

    def __init__(self, transport=None, base_url=SLUMBOT_URL, token=None, timeout=20.0):
        self.transport = transport or (lambda url, payload: _urllib_transport(url, payload, timeout))
        self.base_url, self.token = base_url.rstrip('/'), token

    def _post(self, path, payload):
        r = self.transport(f'{self.base_url}/api/{path}', payload)
        if not isinstance(r, dict):
            raise RuntimeError(f'slumbot: unexpected reply {r!r}')
        if r.get('error_msg'):
            raise RuntimeError(f'slumbot: {r["error_msg"]}')
        if r.get('token'):
            self.token = r['token']
        return r

    def login(self, username, password):
        return self._post('login', {'username': username, 'password': password}).get('token')

    def new_hand(self) -> dict:
        return self._post('new_hand', {'token': self.token} if self.token else {})

    def act(self, incr: str) -> dict:
        return self._post('act', {'token': self.token, 'incr': incr})

    @classmethod
    def parse_action(cls, action: str) -> dict:
        """The state an action string leaves (the logic of Slumbot's reference client): {'street' (0-3),
        'to_act' (position, -1 when the hand is over), 'street_bets' [pos0, pos1], 'totals' [pos0, pos1],
        'last_bet_size', 'actions': [{'pos', 'street', 'kind' ('k'|'c'|'f'|'b'), 'to' (street total)}],
        'all_in'}."""
        street, pos, ends, over = 0, 1, False, False
        bets, totals = [cls.BB, cls.SB], [cls.BB, cls.SB]
        last_size, acts = cls.BB - cls.SB, []
        i = 0

        def next_street():
            nonlocal street, pos, bets, ends, last_size, over
            if street == 3:
                over = True
            else:
                street, pos, bets, ends, last_size = street + 1, 0, [0, 0], False, 0

        while i < len(action) and not over:
            ch = action[i]
            i += 1
            if ch == '/':
                continue
            if ch == 'k':
                acts.append({'pos': pos, 'street': street, 'kind': 'k', 'to': None})
                if ends:
                    next_street()
                else:
                    pos, ends = 1 - pos, True
            elif ch == 'c':
                add = max(bets) - bets[pos]
                bets[pos] += add
                totals[pos] += add
                acts.append({'pos': pos, 'street': street, 'kind': 'c', 'to': bets[pos]})
                last_size = 0
                if max(totals) >= cls.STACK:
                    over = True
                elif ends:
                    next_street()
                else:
                    pos, ends = 1 - pos, True
            elif ch == 'f':
                acts.append({'pos': pos, 'street': street, 'kind': 'f', 'to': None})
                over = True
            elif ch == 'b':
                j = i
                while i < len(action) and action[i].isdigit():
                    i += 1
                to = int(action[j:i])
                last_size = to - max(bets)
                totals[pos] += to - bets[pos]
                bets[pos] = to
                acts.append({'pos': pos, 'street': street, 'kind': 'b', 'to': to})
                pos, ends = 1 - pos, True
            else:
                raise ValueError(f'bad slumbot action {action!r}')
        return {'street': street, 'to_act': -1 if over else pos, 'street_bets': bets, 'totals': totals,
                'last_bet_size': last_size, 'actions': acts, 'all_in': max(totals) >= cls.STACK}

    @classmethod
    def to_arena(cls, action: str, client_pos: int, scale: float = 1 / 50) -> list[dict]:
        """The action string as Poker Arena history entries (the client is seat 0, Slumbot seat 1; amounts in
        arena chips, `scale` per Slumbot chip)."""
        st = cls.parse_action(action)
        names = ['preflop', 'flop', 'turn', 'river']
        seat_of = {client_pos: 0, 1 - client_pos: 1}
        out, bets, totals = [], {0: cls.BB, 1: cls.SB}, {0: cls.BB, 1: cls.SB}
        cur = 0
        for a in st['actions']:
            if a['street'] != cur:
                cur, bets = a['street'], {0: 0, 1: 0}
            seat = seat_of[a['pos']]
            if a['kind'] == 'k':
                e = {'seat': seat, 'street': names[cur], 'action': 'check'}
            elif a['kind'] == 'f':
                e = {'seat': seat, 'street': names[cur], 'action': 'fold'}
            elif a['kind'] == 'c':
                add = max(bets.values()) - bets[a['pos']]
                bets[a['pos']] += add
                totals[a['pos']] += add
                e = {'seat': seat, 'street': names[cur], 'action': 'call', 'amount': add * scale}
            else:
                opening = max(bets.values()) == 0
                totals[a['pos']] += a['to'] - bets[a['pos']]
                bets[a['pos']] = a['to']
                e = {'seat': seat, 'street': names[cur], 'action': 'bet' if opening else 'raise',
                     'amount': a['to'] * scale}
            if totals[a['pos']] >= cls.STACK:
                e['all_in'] = True
            out.append(e)
        return out

    @classmethod
    def from_arena(cls, action: dict, state: dict, client_pos: int, scale: float = 1 / 50) -> str:
        """An arena action at this state as the Slumbot increment ('f', 'k', 'c' or 'b<to>')."""
        a = str(action.get('action') or '').lower()
        bets = state['street_bets']
        mine, top = bets[client_pos], max(bets)
        if a == 'fold':
            return 'f' if top > mine else 'k'
        if a in ('check', 'call'):
            return 'c' if top > mine else 'k'
        cap = mine + cls.STACK - state['totals'][client_pos]
        if a == 'all_in':
            to = cap
        else:
            to = int(round(float(action.get('amount') or 0) / scale))
            min_to = top + max(state['last_bet_size'], cls.BB)
            to = max(to, min(min_to, cap))
            to = min(to, cap)
        if to <= top:
            return 'c' if top > mine else 'k'
        return f'b{to}'

    @classmethod
    def request_data(cls, r: dict, scale: float = 1 / 50) -> dict:
        """Arena request data for the client's decision from a Slumbot reply."""
        cp = int(r['client_pos'])
        st = cls.parse_action(r.get('action', ''))
        names = ['preflop', 'flop', 'turn', 'river']
        bets = [st['street_bets'][cp], st['street_bets'][1 - cp]]
        totals = [st['totals'][cp], st['totals'][1 - cp]]
        stacks = [(cls.STACK - t) * scale for t in totals]
        top = max(bets)
        to_call = min(top - bets[0], cls.STACK - totals[0]) * scale
        cap = bets[0] + cls.STACK - totals[0]
        min_to = top + max(st['last_bet_size'], cls.BB)
        rb = None
        if cap > top and stacks[1] > 0:
            rb = (min(min_to, cap) * scale, cap * scale)
        return {'seat': 0, 'hole': list(r.get('hole_cards') or []), 'board': list(r.get('board') or []),
                'street': names[st['street']], 'pot': sum(totals) * scale, 'to_call': to_call, 'raise_bounds': rb,
                'stack': stacks[0], 'street_bet': [b * scale for b in bets], 'stacks': stacks, 'bb': cls.BB * scale,
                'button': 0 if cp == 1 else 1, 'history': cls.to_arena(r.get('action', ''), cp, scale), 'n': 2,
                '_state': st, '_client_pos': cp}


class SlumbotBot(Agent):
    """An arena agent that can be benchmarked against Slumbot: it plays with the wrapped agent (GTOBot by
    default) both as a seat in arena matches (`act`) and, through `play_hands`, as the client in Slumbot's own
    hands (cards are dealt by Slumbot's server, so a Slumbot match is its own record, in Slumbot chips and bb)."""
    kind = 'bot'

    def __init__(self, seed=0, agent=None, client=None, transport=None, **_):
        self.agent = agent or GTOBot(seed=seed)
        self.client = client or SlumbotClient(transport=transport)

    async def act(self, text, request, validate):
        return await self.agent.act(text, request, validate)

    def play_hand(self, max_steps=64) -> dict:
        from .games.base import Request
        r = self.client.new_hand()
        steps = []
        for _ in range(max_steps):
            if 'winnings' in r:
                break
            d = SlumbotClient.request_data(r)
            legal = _legal(d['to_call'], d['raise_bounds'])
            dec = run_agent(self.agent, '', Request('', legal, {k: v for k, v in d.items() if not k.startswith('_')}),
                            lambda a: None)
            incr = SlumbotClient.from_arena(dec.action, d['_state'], d['_client_pos'])
            steps.append({'state': d['history'], 'action': dec.action, 'incr': incr})
            r = self.client.act(incr)
        win = r.get('winnings')
        return {'client_pos': r.get('client_pos'), 'hole': r.get('hole_cards'), 'bot_hole': r.get('bot_hole_cards'),
                'board': r.get('board'), 'action': r.get('action'), 'winnings': win,
                'winnings_bb': None if win is None else win / SlumbotClient.BB, 'decisions': steps}

    def play_hands(self, n: int) -> dict:
        hands = [self.play_hand() for _ in range(n)]
        bbs = [h['winnings_bb'] for h in hands if h['winnings_bb'] is not None]
        return {'hands': len(hands), 'winnings_bb': float(sum(bbs)),
                'bb_per_100': float(np.mean(bbs) * 100) if bbs else None, 'records': hands}


# ---- running bots synchronously -----------------------------------------------------------------

def run_agent(agent, text, request, validate):
    """An agent's Decision, without an event loop for the bots here (their act never awaits)."""
    coro = agent.act(text, request, validate)
    if getattr(agent, 'kind', '') == 'bot':
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value
        coro.close()
        raise RuntimeError(f'{type(agent).__name__}.act awaited; run it with asyncio')
    return asyncio.run(coro)


def play_hand(agents, seed, index=0, config=None) -> dict:
    """One hand between two agents (seat order), as the runner records it in hands.jsonl."""
    from .games import get_game
    game = get_game('holdem')({**(config or {})})
    ep = game.episode(index, seed, ['P0', 'P1'])
    guard = 0
    while not ep.done:
        seat = ep.to_act
        req = ep.request(seat)
        dec = run_agent(agents[seat], '', req, lambda a: None)
        if ep.apply(seat, dec.action or {}):
            ep.apply(seat, ep.fallback(seat))
        guard += 1
        if guard > 200:
            raise RuntimeError('hand did not finish')
    return {**ep.summary(), 'seed': seed}


def play_match(agents, seeds, config=None) -> list[dict]:
    return [play_hand(agents, s, k, config) for k, s in enumerate(seeds)]


def play_duplicate(make_a, make_b, seeds, config=None):
    """(a-first hands, b-first hands): the same seeds with the seats swapped, fresh agents for each play."""
    return (play_match([make_a(1), make_b(1)], seeds, config), play_match([make_b(2), make_a(2)], seeds, config))


# ---- evaluate a recorded hand -------------------------------------------------------------------

def evaluate_hand(hand_record, events=None, solver='dcfr', cache_dir=None, *, game=None, players=None,
                  iterations=None, k=None, river_iterations=None) -> dict:
    """Replay a hand from hands.jsonl through the engine and evaluate every decision: {'hand', 'seed', 'bb',
    'decisions': [SpotEval + {'seat', 'player', 'decision_id', 'action_idx', 'amount', 'board', 'hole'}],
    'per_seat_ev_loss_mbb': [seat 0, seat 1] (sum over the hand)}. Ranges are the Bayesian weights of
    `infer_ranges`, updated after every action of either seat."""
    from .games.holdem import Holdem, HoldemHand
    rec = hand_record
    start = next((e.get('data') or {} for e in events or [] if e.get('kind') == 'hand_start'), {})
    cfg = {'sb': start.get('sb', 1), 'bb': start.get('bb', 2), 'stack': start.get('stack', 200), **(game or {})}
    names = players or start.get('names') or ['P0', 'P1']
    G = Holdem(cfg)
    button = rec.get('button', rec.get('hand', 0) % 2)
    idx = rec.get('hand', 0)
    idx = idx if idx % 2 == button else button
    ep = HoldemHand(G, idx, rec.get('seed') or 0, list(names))
    ep.hole = [list(h) for h in rec['hole']]
    used = {c for h in ep.hole for c in h} | set(rec.get('board') or [])
    ep.deck_board = list(rec.get('board') or []) + [c for c in DECK if c not in used][:5 - len(rec.get('board') or [])]
    it_abs = iterations or FAST['iterations']
    it_riv = river_iterations or iterations or FAST['river_iterations']
    k = k or FAST['k']
    bb = G.bb
    out, loss = [], [0.0, 0.0]
    solver_kind = 'texassolver' if solver == 'texassolver' else 'dcfr'
    for k_i, a in enumerate(rec.get('actions') or []):
        if ep.done:
            break
        seat = ep.to_act
        d = ep.request(seat).data
        it = it_riv if d['street'] == 'river' else it_abs
        spot = solve_postflop_spot(d['street'], d['hole'], d['board'], d['pot'], d['to_call'], d['stacks'],
                                   d['street_bet'], seat, d['button'], d['history'], bb, solver_kind, cache_dir, it,
                                   taken_action=a['action'], taken_amount=a.get('amount'),
                                   taken_all_in=bool(a.get('all_in')), raise_bounds=d['raise_bounds'],
                                   sb=G.sb, k=k, river_cap=FAST['river_cap'], runouts=FAST['runouts'])
        spot.update(seat=seat, player=names[seat] if seat < len(names) else f'seat{seat}',
                    decision_id=a.get('decision_id'), action_idx=k_i, amount=a.get('amount'), board=list(d['board']),
                    hole=list(d['hole']))
        out.append(spot)
        loss[seat] += spot['ev_loss_mbb'] or 0.0
        kind = {'bet': 'raise'}.get(a['action'], a['action'])
        err = ep.apply(seat, {'action': kind, 'amount': a.get('amount') if kind == 'raise' else None})
        if err:
            break
    return {'hand': rec.get('hand'), 'seed': rec.get('seed'), 'bb': bb, 'decisions': out,
            'per_seat_ev_loss_mbb': loss}


# ---- shared bot helpers -------------------------------------------------------------------------

def _option_action(o, d):
    a = o['abstract']
    if a in ('fold', 'check', 'call'):
        return {'action': a}
    if a == 'all_in':
        return {'action': 'all_in'}
    rb = d['raise_bounds']
    if not rb:
        return {'action': 'call' if d['to_call'] else 'check'}
    to = int(round(o['to'])) if o.get('to') else rb[0]
    return {'action': 'raise', 'amount': int(min(rb[1], max(rb[0], to)))}


def _pick(opts, u):
    acc = 0.0
    for o in opts:
        acc += o['freq']
        if u < acc:
            return o
    return opts[-1]


def gto_options(d, settings=None) -> list[dict]:
    """The sized options (with equilibrium frequencies) GTOBot chooses among at this request: exactly the spot
    `evaluate_hand` evaluates, with the same FAST settings unless others are given."""
    st = {**FAST, **(settings or {})}
    it = st['river_iterations'] if d['street'] == 'river' else st['iterations']
    _, opts = _spot_core(d['street'], d['hole'], d['board'], d['pot'], d['to_call'], d['stacks'], d['street_bet'],
                         d['seat'], d['button'], d['history'], d['bb'], 'dcfr', None, it,
                         raise_bounds=d['raise_bounds'], k=st['k'], runouts=st['runouts'], sb=d.get('sb'),
                         river_cap=st['river_cap'], bet_sizes=st.get('bet_sizes', (0.5, 1.0)))
    legal = _legal(d['to_call'], d['raise_bounds'])
    opts = [o for o in opts if o['abstract'] in legal]
    tot = sum(o['freq'] for o in opts)
    for o in opts:
        o['freq'] = o['freq'] / tot if tot > 0 else 1.0 / len(opts)
    return opts


def _hand_level(hole, board) -> int:
    """0 air, 1 weak pair or draw, 2 top pair / overpair, 3 two pair / trips with a hole card, 4 straight+."""
    if not board:
        return 0
    me = evaluate(list(hole) + list(board))
    vals = sorted((VALUE[c[0]] for c in board), reverse=True)
    bcat = _board_cat(board)
    hv = [VALUE[c[0]] for c in hole]
    if me[0] >= 4 and me[0] > bcat:
        return 4
    if me[0] in (2, 3, 6, 7) and me[0] > bcat:
        return 3
    if me[0] >= 1 and me[0] > bcat:
        pair = me[1]
        if (pair >= vals[0] and pair in hv) or (hv[0] == hv[1] and hv[0] > vals[0]):
            return 2
        return 1
    if len(board) < 5:
        suits = [c[1] for c in list(hole) + list(board)]
        if any(suits.count(s) >= 4 and s in (hole[0][1], hole[1][1]) for s in set(suits)):
            return 1
        ranks = set(VALUE[c[0]] for c in list(hole) + list(board))
        if 14 in ranks:
            ranks.add(1)
        if any(all(v in ranks for v in range(lo, lo + 4)) and any(h in range(lo, lo + 4) for h in hv)
               for lo in range(1, 12)):
            return 1
    return 0


def _board_cat(board) -> int:
    vals = [VALUE[c[0]] for c in board]
    cnt = sorted((vals.count(v) for v in set(vals)), reverse=True)
    if len(board) >= 5:
        return evaluate(list(board))[0]
    if cnt[0] >= 3:
        return 3
    if cnt[0] == 2 and len(cnt) > 1 and cnt[1] == 2:
        return 2
    return 1 if cnt[0] == 2 else 0


def _pre_pct(hole) -> float:
    return _ranges.percentile()[_ranges.hand_class(hole)]


def _n_raises(d, street=None):
    st = street or d['street']
    return sum(h['action'] in ('bet', 'raise') for h in d['history'] if h['street'] == st)


def _raise_to(d, to):
    rb = d['raise_bounds']
    if not rb:
        return {'action': 'call' if d['to_call'] else 'check'}
    to = int(round(min(rb[1], max(rb[0], to))))
    if to >= rb[1]:
        return {'action': 'all_in'}
    return {'action': 'raise', 'amount': to}


def _passive(d):
    return {'action': 'check'} if d['to_call'] == 0 else {'action': 'fold'}


def _call(d):
    return {'action': 'check'} if d['to_call'] == 0 else {'action': 'call'}


def _eq_vs_range(d, samples=120, salt=''):
    w = infer_ranges(d['history'], d['board'], d['button'], d['bb'])[1 - d['seat']]
    key = f'eq|{salt}|{d["hole"]}|{d["board"]}|{_hist_sig(d["history"])}'
    shares, vs = _equity_samples(_ids(d['hole']), _ids(d['board']), w, samples, key)
    return float(shares.mean()), shares, vs, w


# ---- calibration bots ---------------------------------------------------------------------------

class GTOBot(SeededBot):
    """Approximate equilibrium play: the HUNL preflop range/position/SPR model before the flop, the solver's
    DCFR frequencies after it (bucketed on flop and turn, exact on the river; small and cached so a hand takes
    milliseconds). It samples its mixed strategy with its own generator, after dropping actions played less
    than PURIFY of the time (noise of a short solve)."""
    kind = 'bot'
    VERSION = 'gto-1'
    PURIFY = 0.05

    def __init__(self, seed=0, iterations=None, k=None, river_iterations=None, runouts=None, river_cap=None,
                 bet_sizes=None, **_):
        self.rng = random.Random(seed)
        self.settings = {**FAST, **({'iterations': int(iterations)} if iterations else {}),
                         **({'k': int(k)} if k else {}), **({'runouts': int(runouts)} if runouts else {}),
                         **({'river_iterations': int(river_iterations)} if river_iterations else {}),
                         **({'bet_sizes': tuple(bet_sizes)} if bet_sizes is not None else {})}

    def describe(self):
        return {**self.spec, 'version': self.VERSION, **self.settings}

    def options(self, d):
        opts = gto_options(d, self.settings)
        keep = [o for o in opts if o['freq'] >= self.PURIFY] or [max(opts, key=lambda o: o['freq'])]
        tot = sum(o['freq'] for o in keep)
        return [{**o, 'freq': o['freq'] / tot} for o in keep]

    async def act(self, text, request, validate):
        d = request.data
        if d.get('n', 2) != 2:
            return Decision(_call(d) if d['to_call'] == 0 else {'action': 'fold'}, 'heads-up only')
        opts = self.options(d)
        o = _pick(opts, self.rng.random())
        return Decision(_option_action(o, d), f'gto: {o["label"]} (freq {o["freq"]:.2f}, ev {o["ev"]:+.2f})')


class NoisyGTOBot(GTOBot):
    """GTOBot that, with probability `noise`, takes a uniformly chosen other legal action instead (a provably
    weaker player for the calibration ladder)."""
    VERSION = 'noisy-gto-1'

    def __init__(self, seed=0, noise=0.20, **kw):
        super().__init__(seed=seed, **kw)
        self.noise = float(noise)

    def describe(self):
        return {**super().describe(), 'noise': self.noise}

    async def act(self, text, request, validate):
        d = request.data
        opts = self.options(d)
        o = _pick(opts, self.rng.random())
        if self.rng.random() < self.noise:
            others = [x for x in gto_options(d, self.settings) if x['abstract'] != o['abstract']]
            if others:
                o = others[self.rng.randrange(len(others))]
                return Decision(_option_action(o, d), f'noise: {o["label"]}')
        return Decision(_option_action(o, d), f'gto: {o["label"]}')


class LocalBestResponseBot(SeededBot):
    """Local best response (Lisy and Bowling 2017): at each decision, every action's EV assuming both players
    check/call to showdown afterwards, against the opponent's inferred range and an opponent action model, and
    the best one is taken. The model's fold probability for a bet is the range model's (how often each hand
    in the range folds at that price) blended with the opponent's observed fold-to-bet rate across hands
    (prior weight `prior` observations)."""
    kind = 'bot'
    VERSION = 'lbr-1'

    def __init__(self, seed=0, samples=120, sizes=(0.5, 1.0), prior=8, **_):
        self.rng, self.samples, self.sizes, self.prior = random.Random(seed), int(samples), tuple(sizes), prior
        self.model = ExploiterBot(seed=seed)

    def describe(self):
        return {**self.spec, 'version': self.VERSION, 'samples': self.samples, 'sizes': list(self.sizes),
                'prior': self.prior}

    def state(self):
        return {**super().state(), 'model': self.model.state()}

    def restore(self, st):
        super().restore(st)
        if st.get('model'):
            self.model.restore(st['model'])

    def evs(self, d):
        eq, shares, vs, w = _eq_vs_range(d, self.samples, 'lbr')
        pot, tc, seat = d['pot'], d['to_call'], d['seat']
        opp = 1 - seat
        st = self.model.stats
        out = []
        if tc > 0:
            out.append(({'action': 'fold'}, 0.0))
            out.append(({'action': 'call'}, eq * (pot + tc) - tc))
        else:
            out.append(({'action': 'check'}, eq * pot))
        rb = d['raise_bounds']
        if rb:
            s = _board_strength(tuple(_ids(d['board']))) if d['board'] else _pre_strength()
            cur = max(d['street_bet'])
            nr = _n_raises(d)
            tos = [min(rb[1], max(rb[0], cur + f * (pot + tc) + tc)) for f in self.sizes] + [rb[1]]
            for to in sorted(set(int(round(t)) for t in tos)):
                a = to - d['street_bet'][seat]
                dd = min(to, d['street_bet'][opp] + d['stacks'][opp]) - d['street_bet'][opp]
                odds = dd / (pot + a + dd) if dd > 0 else 0.0
                kw = dict(n_raises=nr + 1, odds=odds, facing=True)
                lf = action_likelihood(s[vs], d['street'], 'fold', **kw)
                lc = action_likelihood(s[vs], d['street'], 'call', **kw) + \
                    action_likelihood(s[vs], d['street'], 'raise', **kw)
                pf = lf / (lf + lc)
                fp_model = float(pf.mean())
                n_obs = st['faced_bet'] if d['street'] != 'preflop' else 0
                fp = (fp_model * self.prior + st['folded_to_bet']) / (self.prior + n_obs) if n_obs else fp_model
                scale = fp / max(fp_model, 1e-9)
                cont = np.clip(1 - pf * scale, 0.0, 1.0)
                eq_c = float((shares * cont).sum() / max(cont.sum(), 1e-9))
                ev = fp * pot + (1 - fp) * (eq_c * (pot + a + dd) - a)
                act = {'action': 'all_in'} if to >= rb[1] else {'action': 'raise', 'amount': to}
                out.append((act, ev))
        return out, eq

    async def act(self, text, request, validate):
        d = request.data
        self.model._observe(d, text)
        evs, eq = self.evs(d)
        act, ev = max(evs, key=lambda x: x[1])
        self.model._last_bet = (d['street'], act['action'] == 'all_in') if act['action'] in ('raise', 'all_in') else None
        return Decision(act, f'lbr: equity {eq:.2f}, best {act["action"]} ev {ev:+.1f}')


class ExploiterBot(SeededBot):
    """Shifts value-betting, bluffing and calling thresholds to exploit the opponent's leaks. Modes:
    'vs_station' (value-bet thin and big, never bluff, call tighter), 'vs_nit' (steal and bluff a lot, give up
    against aggression), 'vs_overbluffer' (call down light, bluff less), 'adaptive' (estimates the opponent's
    VPIP, fold-to-bet and river betting frequency across hands, with priors, and blends those settings)."""
    kind = 'bot'
    VERSION = 'exploiter-1'
    MODES = ('adaptive', 'vs_station', 'vs_nit', 'vs_overbluffer')
    PRESETS = {
        'base': {'value': 0.62, 'bluff': 0.15, 'call_margin': 0.03, 'size': 0.66, 'open': 0.75, 'defend': 0.6},
        'vs_station': {'value': 0.55, 'bluff': 0.0, 'call_margin': 0.08, 'size': 1.0, 'open': 0.6, 'defend': 0.55},
        'vs_nit': {'value': 0.72, 'bluff': 0.6, 'call_margin': 0.10, 'size': 0.5, 'open': 0.95, 'defend': 0.45},
        'vs_overbluffer': {'value': 0.60, 'bluff': 0.03, 'call_margin': -0.12, 'size': 0.66, 'open': 0.7,
                           'defend': 0.75},
    }

    def __init__(self, seed=0, mode='adaptive', samples=100, **_):
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}')
        self.rng, self.mode, self.samples = random.Random(seed), mode, int(samples)
        self.stats = {'hands': 0, 'vpip': 0, 'faced_bet': 0, 'folded_to_bet': 0, 'river_opps': 0, 'river_bets': 0,
                      'aggr': 0, 'passive': 0}
        self._hand_key, self._seen, self._last_bet, self._vpip = None, 0, None, False

    def describe(self):
        return {**self.spec, 'version': self.VERSION, 'mode': self.mode, 'samples': self.samples}

    def state(self):
        return {**super().state(), 'stats': dict(self.stats)}

    def restore(self, st):
        super().restore(st)
        self.stats.update(st.get('stats') or {})

    def _observe(self, d, text):
        """Update the opponent's counts from the actions seen since this bot's last decision. A hand that ended
        after this bot's bet without another request was a fold (before the river; on the river the table
        text decides)."""
        key = (tuple(d['hole']), d['button'])
        hist = d['history']
        if key != self._hand_key or len(hist) < self._seen:
            if self._last_bet is not None and self._hand_key is not None:
                st, allin = self._last_bet
                if (st != 'river' and not allin) or re.search(r'^(?!You\b).*\bfolds\b', text or '', re.M):
                    self.stats['faced_bet'] += 1
                    self.stats['folded_to_bet'] += 1
            self._hand_key, self._seen, self._last_bet, self._vpip = key, 0, None, False
            self.stats['hands'] += 1
        opp = 1 - d['seat']
        for k, h in enumerate(hist[self._seen:], start=self._seen):
            if h['seat'] != opp:
                continue
            prev = hist[k - 1] if k > 0 else None
            if h['street'] == 'preflop' and h['action'] in ('call', 'bet', 'raise') and not self._vpip:
                self.stats['vpip'] += 1
                self._vpip = True
            if h['street'] != 'preflop' and prev is not None and prev['seat'] != opp and prev['action'] in ('bet', 'raise'):
                self.stats['faced_bet'] += 1
                self.stats['folded_to_bet'] += h['action'] == 'fold'
            if h['street'] == 'river':
                self.stats['river_opps'] += 1
                self.stats['river_bets'] += h['action'] in ('bet', 'raise')
            if h['action'] in ('bet', 'raise'):
                self.stats['aggr'] += 1
            elif h['action'] == 'call':
                self.stats['passive'] += 1
        self._seen = len(hist)

    def estimates(self):
        s = self.stats
        return {'vpip': (s['vpip'] + 0.6 * 6) / (s['hands'] + 6),
                'fold_to_bet': (s['folded_to_bet'] + 0.45 * 6) / (s['faced_bet'] + 6),
                'river_bet': (s['river_bets'] + 0.3 * 6) / (s['river_opps'] + 6),
                'aggression': (s['aggr'] + 1) / (s['passive'] + 1)}

    def params(self):
        if self.mode != 'adaptive':
            return dict(self.PRESETS[self.mode])
        e = self.estimates()
        mix = {'base': 1.0}
        mix['vs_station'] = float(np.clip((0.35 - e['fold_to_bet']) / 0.2, 0, 1)) * 2
        mix['vs_nit'] = float(np.clip((e['fold_to_bet'] - 0.55) / 0.2, 0, 1)) * 2
        mix['vs_overbluffer'] = float(np.clip((e['river_bet'] - 0.45) / 0.2, 0, 1) +
                                      np.clip((e['aggression'] - 2.0) / 2.0, 0, 1))
        tot = sum(mix.values())
        return {k: sum(self.PRESETS[m][k] * v for m, v in mix.items()) / tot for k in self.PRESETS['base']}

    async def act(self, text, request, validate):
        d = request.data
        self._observe(d, text)
        p = self.params()
        act, why = self._decide(d, p)
        if act['action'] in ('raise', 'all_in'):
            self._last_bet = (d['street'], act['action'] == 'all_in')
        else:
            self._last_bet = None
        return Decision(act, f'exploiter[{self.mode}]: {why}')

    def _decide(self, d, p):
        tc, pot = d['to_call'], d['pot']
        if d['street'] == 'preflop':
            pct = _pre_pct(d['hole'])
            nr = _n_raises(d)
            if nr == 0 and tc > 0:
                return (_raise_to(d, 2.5 * d['bb']), 'open') if pct <= p['open'] else ({'action': 'fold'}, 'fold')
            if nr == 0:
                return (_raise_to(d, 4 * d['bb']), 'raise limp') if pct <= 0.3 else ({'action': 'check'}, 'check')
            if pct <= 0.05 or (nr == 1 and pct <= 0.10):
                return _raise_to(d, 3 * max(d['street_bet'])), 'reraise'
            odds = tc / (pot + tc)
            limit = p['defend'] if nr == 1 else p['defend'] * 0.35
            return (_call(d), 'defend') if pct <= limit or odds < 0.2 else ({'action': 'fold'}, 'fold')
        eq, *_ = _eq_vs_range(d, self.samples, 'exp')
        odds = tc / (pot + tc) if tc else 0.0
        if tc == 0:
            if eq >= p['value']:
                return _raise_to(d, max(d['bb'], p['size'] * pot)), f'value {eq:.2f}'
            if eq < 0.35 and self.rng.random() < p['bluff']:
                return _raise_to(d, max(d['bb'], 0.6 * pot)), f'bluff {eq:.2f}'
            return {'action': 'check'}, f'check {eq:.2f}'
        if eq >= max(0.8, p['value'] + 0.15) and d['raise_bounds']:
            return _raise_to(d, 3 * max(d['street_bet'])), f'raise {eq:.2f}'
        if eq >= odds + p['call_margin']:
            return {'action': 'call'}, f'call {eq:.2f} vs {odds:.2f}'
        return {'action': 'fold'}, f'fold {eq:.2f} vs {odds:.2f}'


class _StyleBot(SeededBot):
    kind = 'bot'

    def __init__(self, seed=0, **_):
        self.rng = random.Random(seed)

    def describe(self):
        return {**self.spec, 'version': self.VERSION}

    async def act(self, text, request, validate):
        d = request.data
        act = self.preflop(d) if d['street'] == 'preflop' else self.postflop(d, _hand_level(d['hole'], d['board']))
        return Decision(act, type(self).__name__)


class RockBot(_StyleBot):
    """Rock: a tight trapper and value-caller. Plays only premium hands (top 18%), traps by calling with made
    hands, value-bets the river, folds the rest. It crushes the manic Scissors (who pays it off) and loses to
    Paper (who steals the blinds it gives up)."""
    VERSION = 'rps-rock-1'

    def preflop(self, d):
        pct, nr, tc, bb = _pre_pct(d['hole']), _n_raises(d), d['to_call'], d['bb']
        if nr == 0:
            if pct <= 0.18:
                return _raise_to(d, 3 * bb)
            return {'action': 'fold'} if tc > 0 else {'action': 'check'}
        facing_allin = d['stack'] <= tc
        if facing_allin:
            return {'action': 'call'} if pct <= 0.08 else {'action': 'fold'}
        if pct <= 0.05:
            return _raise_to(d, 3 * max(d['street_bet']))
        if pct <= (0.18 if nr == 1 else 0.10):
            return {'action': 'call'}
        return {'action': 'fold'}

    def postflop(self, d, lvl):
        tc, pot = d['to_call'], d['pot']
        if tc == 0:
            if lvl >= 3:
                return _raise_to(d, (1.0 if d['street'] == 'river' else 0.75) * pot)
            return {'action': 'check'}
        if lvl >= 4:
            return _raise_to(d, 3 * max(d['street_bet'])) if d['street'] == 'river' else {'action': 'call'}
        if lvl >= 3:
            return {'action': 'call'}
        if lvl == 2 and tc <= 0.75 * (pot - tc) and d['stack'] > tc:
            return {'action': 'call'}
        return {'action': 'fold'}


class PaperBot(_StyleBot):
    """Paper: a relentless small-ball stealer and bluffer. Min-raises every hand, bets a third of the pot on
    every street, gives up to resistance without a strong hand and never calls an all-in. It crushes the tight
    Rock (who folds too often) and loses to Scissors (who calls its light bluffs)."""
    VERSION = 'rps-paper-1'

    def preflop(self, d):
        nr, tc, bb = _n_raises(d), d['to_call'], d['bb']
        if d['stack'] <= tc:
            return {'action': 'fold'}
        if nr == 0:
            return _raise_to(d, 2 * bb if tc > 0 else 3 * bb)
        if nr == 1 and max(d['street_bet']) <= 3 * bb and _pre_pct(d['hole']) <= 0.5:
            return {'action': 'call'}
        return {'action': 'fold'}

    def postflop(self, d, lvl):
        tc, pot = d['to_call'], d['pot']
        if tc == 0:
            return _raise_to(d, max(d['bb'], pot / 3))
        if d['stack'] <= tc:
            return {'action': 'fold'}
        if lvl >= 3:
            return {'action': 'call'}
        if lvl >= 2 and tc <= 0.5 * (pot - tc):
            return {'action': 'call'}
        return {'action': 'fold'}


class ScissorsBot(_StyleBot):
    """Scissors: hyper-loose, sticky and shove-happy. Plays almost every hand, shoves any pair over a bet, calls
    small bets with nothing and calls all-ins wide. It crushes the light-bluffing Paper (whose small bets it
    shoves over or calls down) and is crushed by Rock (who only puts money in with the goods)."""
    VERSION = 'rps-scissors-1'

    def preflop(self, d):
        pct, nr, tc = _pre_pct(d['hole']), _n_raises(d), d['to_call']
        if d['stack'] <= tc:
            return {'action': 'call'} if pct <= 0.40 else {'action': 'fold'}
        if nr >= 1:
            return ({'action': 'all_in'} if d['raise_bounds'] else {'action': 'call'}) if pct <= 0.40 else _call(d)
        if pct <= (0.5 if tc > 0 else 0.3):
            return _raise_to(d, 3 * d['bb'])
        return _call(d)

    def postflop(self, d, lvl):
        tc, pot = d['to_call'], d['pot']
        if tc == 0:
            if lvl >= 3:
                return {'action': 'all_in'} if d['raise_bounds'] else {'action': 'check'}
            if lvl >= 1:
                return _raise_to(d, (0.75 if lvl >= 2 else 0.5) * pot)
            return {'action': 'check'}
        if lvl >= 1:
            return {'action': 'all_in'} if d['raise_bounds'] else {'action': 'call'}
        high = max(VALUE[c[0]] for c in d['hole'])
        if tc <= 0.5 * (pot - tc) and (d['street'] == 'flop' or high >= 13):
            return {'action': 'call'}
        return {'action': 'fold'}


SOLVER_BOTS = {'gto': GTOBot, 'noisy_gto': NoisyGTOBot, 'lbr': LocalBestResponseBot, 'exploiter': ExploiterBot,
               'rps_rock': RockBot, 'rps_paper': PaperBot, 'rps_scissors': ScissorsBot}
