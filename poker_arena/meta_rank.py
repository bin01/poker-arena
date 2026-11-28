"""Game-theoretic rankings of a pool of players from their pairwise edges (pure NumPy).

The meta-game: M[i, j] is i's edge over j in bb/100 (antisymmetric). Three views of it:
- Nash averaging (Balduzzi, Tuyls, Perolat and Graepel 2018): the maximum-entropy Nash equilibrium p of the
  symmetric zero-sum game M, and each player's score (M p)_i against it. Players outside every equilibrium's
  support get weight 0, and copies of a player share that player's weight, so adding clones (of weak or of
  strong players) changes no one else's weight and nobody's score: the ranking can't be gamed by flooding the
  pool. The equilibrium is exact: its support and equalities from a simplex LP, the maximum-entropy point of
  the equilibrium polytope from a Newton barrier method in its affine hull.
- alpha-Rank (Omidshafiei et al. 2019): the stationary distribution of the single-population evolutionary
  Markov chain whose mutants fix with Fermi (logistic) probabilities; exact linear solve.
- Cycles: 3-cycles and strongly connected components of size >= 2 in the dominance graph (i -> j when
  M[i, j] > threshold), the signature of non-transitive (rock-paper-scissors) strength.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

EDGE_KEYS = ('edge_bb100', 'aivat_bb100', 'raw_bb100', 'edge', 'mean', 'bb100')
TOL = 1e-10


# ---- payoff matrix ------------------------------------------------------------------------------

def build_payoff_matrix(pairings, players=None, key=None):
    """(players, M, counts) from pairings [{'a', 'b', <edge>, 'deals'?}] where the edge is a's bb/100 over b
    (the first of EDGE_KEYS present, or `key`). Several entries of one pair are averaged weighted by deals;
    M[i, j] = -M[j, i]; counts[i, j] = deals behind the entry (0: never played)."""
    pairings = list(pairings)
    names = list(players) if players else sorted({p['a'] for p in pairings} | {p['b'] for p in pairings})
    ix = {p: i for i, p in enumerate(names)}
    n = len(names)
    tot, cnt = np.zeros((n, n)), np.zeros((n, n))
    for p in pairings:
        if p['a'] not in ix or p['b'] not in ix or p['a'] == p['b']:
            continue
        k = key or next((k for k in EDGE_KEYS if p.get(k) is not None), None)
        if k is None or p.get(k) is None:
            continue
        w = float(p.get('deals') or 1)
        i, j = ix[p['a']], ix[p['b']]
        tot[i, j] += w * float(p[k])
        tot[j, i] -= w * float(p[k])
        cnt[i, j] += w
        cnt[j, i] += w
    M = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), 0.0)
    return names, M, cnt


# ---- a small exact LP solver --------------------------------------------------------------------

def _simplex(c, A_ub, b_ub, A_eq, b_eq, tol=1e-11, max_iter=5000):
    """max c.x s.t. A_ub x <= b_ub (b_ub >= 0), A_eq x = b_eq (b_eq >= 0), x >= 0: (x, value). Dense two-phase
    tableau simplex with Bland's rule (no cycling)."""
    c = np.asarray(c, float)
    n = len(c)
    A_ub = np.asarray(A_ub, float).reshape(-1, n)
    A_eq = np.asarray(A_eq, float).reshape(-1, n)
    b_ub, b_eq = np.asarray(b_ub, float), np.asarray(b_eq, float)
    m1, m2 = len(b_ub), len(b_eq)
    m = m1 + m2
    cols = n + m1 + m2
    T = np.zeros((m + 1, cols + 1))
    T[:m1, :n], T[:m1, n:n + m1], T[:m1, -1] = A_ub, np.eye(m1), b_ub
    T[m1:m, :n], T[m1:m, n + m1:cols], T[m1:m, -1] = A_eq, np.eye(m2), b_eq
    basis = list(range(n, cols))
    art = set(range(n + m1, cols))

    def pivot(r, k):
        T[r] /= T[r, k]
        for i in range(m + 1):
            if i != r and abs(T[i, k]) > 0:
                T[i] -= T[i, k] * T[r]
        basis[r] = k

    def solve(allowed):
        for _ in range(max_iter):
            enter = next((j for j in allowed if T[m, j] < -tol), None)
            if enter is None:
                return True
            ratios = [(T[i, -1] / T[i, enter], basis[i], i) for i in range(m) if T[i, enter] > tol]
            if not ratios:
                raise ValueError('unbounded LP')
            pivot(min(ratios)[2], enter)
        raise RuntimeError('simplex did not converge')

    T[m] = 0.0
    T[m, n + m1:cols] = 1.0
    for i in range(m1, m):
        T[m] -= T[i]
    solve(range(cols))
    if T[m, -1] < -1e-8:
        raise ValueError('infeasible LP')
    for i in range(m):
        if basis[i] in art:
            k = next((j for j in range(n + m1) if abs(T[i, j]) > tol), None)
            if k is not None:
                pivot(i, k)
    T[m] = 0.0
    T[m, :n] = -c
    for i in range(m):
        if basis[i] < n and c[basis[i]] != 0:
            T[m] += c[basis[i]] * T[i]
    solve([j for j in range(n + m1)])
    x = np.zeros(cols)
    for i in range(m):
        x[basis[i]] = T[i, -1]
    return x[:n], float(c @ x[:n])


def _equilibrium_lp(A, objective):
    """max objective.p over the Nash set of the antisymmetric game A: {p >= 0, sum p = 1, A^T p >= 0}."""
    n = len(A)
    return _simplex(objective, -A.T, np.zeros(n), np.ones((1, n)), np.ones(1))


# ---- Nash averaging -----------------------------------------------------------------------------

def _max_entropy(A, S, T_eq, T_in, p_start, iters=60):
    """The maximum-entropy point of {p_S >= 0, sum = 1, (A^T p)_j = 0 for j in S and T_eq, (A^T p)_j >= 0 for j
    in T_in}, from a relative-interior start: Newton on entropy plus a vanishing log barrier, in the
    null space of the equalities."""
    S = np.asarray(S)
    rows = [np.ones(len(S))] + [A[S, j] for j in list(S) + list(T_eq)]
    E = np.array(rows)
    _, sv, vt = np.linalg.svd(E)
    rank = int((sv > 1e-9 * max(1.0, sv.max())).sum())
    N = vt[rank:].T
    p = p_start[S].copy()
    if N.shape[1] == 0:
        return p
    G = A[np.ix_(S, list(T_in))] if len(T_in) else np.zeros((len(S), 0))

    def feasible(q):
        return (q > 0).all() and (G.shape[1] == 0 or ((q @ G) > 0).all())

    for mu in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0):
        for _ in range(iters):
            g = p @ G
            grad = -(np.log(p) + 1)
            hess = -np.diag(1 / p)
            if mu > 0 and G.shape[1]:
                grad = grad + mu * (G / g).sum(1)
                hess = hess - mu * (G / g) @ (G / g).T
            gz = N.T @ grad
            if np.linalg.norm(gz) < 1e-13:
                break
            Hz = N.T @ hess @ N
            dz = -np.linalg.solve(Hz, gz)
            step = 1.0
            while step > 1e-12:
                q = p + step * (N @ dz)
                if feasible(q) or (mu == 0 and (q > 0).all() and (G.shape[1] == 0 or ((q @ G) >= -1e-14).all())):
                    break
                step /= 2
            if step <= 1e-12:
                break
            p = q
    return p


def nash_averaging(M, players) -> dict:
    """The maximum-entropy Nash equilibrium of the symmetric zero-sum meta-game M (antisymmetrized), each
    player's Nash-averaged score (M p)_i and the ranking by it. {'players', 'nash_weights', 'nash_scores',
    'ranking', 'residual' (how much the best player gains against p: 0 at an exact equilibrium), 'support',
    'clone_invariant': True}."""
    M = np.asarray(M, float)
    names = list(players)
    n = len(names)
    if n == 0:
        return {'players': [], 'nash_weights': {}, 'nash_scores': {}, 'ranking': [], 'residual': 0.0, 'support': []}
    A = (M - M.T) / 2
    scale = max(1.0, float(np.abs(A).max()))
    As = A / scale
    sols, S = [], []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        x, v = _equilibrium_lp(As, e)
        sols.append(x)
        if v > 1e-9:
            S.append(i)
    p0 = np.mean([sols[i] for i in S], axis=0) if S else sols[0]
    T = [j for j in range(n) if j not in S]
    T_eq, T_in = [], []
    for j in T:
        _, v = _equilibrium_lp(As, As[:, j])
        (T_in if v > 1e-9 else T_eq).append(j)
    p_S = _max_entropy(As, S, T_eq, T_in, np.clip(p0, 1e-15, None)) if len(S) > 1 else np.ones(1)
    p = np.zeros(n)
    p[S] = np.clip(p_S, 0, None)
    p /= p.sum()
    scores = M @ p
    residual = max(0.0, float((A @ p).max()))
    order = sorted(range(n), key=lambda i: (-round(scores[i], 9), -p[i], names[i]))
    return {'players': names, 'nash_weights': {names[i]: float(p[i]) for i in range(n)},
            'nash_scores': {names[i]: float(scores[i]) for i in range(n)}, 'ranking': [names[i] for i in order],
            'residual': residual, 'support': [names[i] for i in S], 'clone_invariant': True}


# ---- alpha-Rank ---------------------------------------------------------------------------------

def _log_abs_expm1(y):
    return y + math.log1p(-math.exp(-y)) if y > 30 else math.log(abs(math.expm1(y)))


def _fixation(x, m):
    """Fermi fixation probability of a mutant with payoff advantage x (alpha times the payoff difference) in a
    population of m: (1 - e^-x) / (1 - e^-mx), 1/m when x = 0; computed in logs so it never overflows."""
    if abs(x) < 1e-12:
        return 1.0 / m
    return math.exp(_log_abs_expm1(-x) - _log_abs_expm1(-m * x)) if x < 0 else \
        math.exp(math.log(-math.expm1(-x)) - math.log(-math.expm1(-m * x)))


def alpha_rank(M, players, alpha=0.05, m=50) -> dict:
    """alpha-Rank's single-population chain: from a monomorphic population of i, a mutant j appears with
    probability 1/(n-1) and fixes with the Fermi probability of its advantage M[j, i] - M[i, j]; pi is the
    exact stationary distribution (pi T = pi, pi >= 0, sum 1). {'players', 'stationary', 'ranking', 'alpha',
    'm', 'transition', 'residual'}."""
    M = np.asarray(M, float)
    names = list(players)
    n = len(names)
    if n == 1:
        return {'players': names, 'stationary': {names[0]: 1.0}, 'ranking': names, 'alpha': alpha, 'm': m,
                'transition': [[1.0]], 'residual': 0.0}
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                T[i, j] = _fixation(alpha * (M[j, i] - M[i, j]), m) / (n - 1)
        T[i, i] = 1.0 - T[i].sum()
    Q = T.T - np.eye(n)
    Q[-1] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.lstsq(Q, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    order = sorted(range(n), key=lambda i: (-pi[i], names[i]))
    return {'players': names, 'stationary': {names[i]: float(pi[i]) for i in range(n)},
            'ranking': [names[i] for i in order], 'alpha': alpha, 'm': m, 'transition': T.tolist(),
            'residual': float(np.abs(pi @ T - pi).max())}


# ---- cycles -------------------------------------------------------------------------------------

def _sccs(adj):
    n = len(adj)
    index, low, on, stack, out, counter = [None] * n, [0] * n, [False] * n, [], [], [0]

    def visit(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on[v] = True
        for w in adj[v]:
            if index[w] is None:
                visit(w)
                low[v] = min(low[v], low[w])
            elif on[w]:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on[w] = False
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in range(n):
        if index[v] is None:
            visit(v)
    return out


def detect_cycles(M, players, threshold=0.0) -> list[list[str]]:
    """Directed cycles in the dominance graph (i -> j when M[i, j] > threshold): every 3-cycle, as [a, b, c]
    with a beating b beating c beating a (rotated to start at its first name), then every strongly connected
    component of two or more players that is not exactly one of those 3-cycles (members sorted)."""
    M = np.asarray(M, float)
    names = list(players)
    n = len(names)
    beats = M > threshold
    out, seen = [], set()
    for i, j, k in itertools.permutations(range(n), 3):
        if beats[i, j] and beats[j, k] and beats[k, i]:
            cyc = [i, j, k]
            r = cyc.index(min(cyc, key=lambda x: names[x]))
            cyc = cyc[r:] + cyc[:r]
            key = tuple(cyc)
            if key not in seen:
                seen.add(key)
                out.append([names[x] for x in cyc])
    tri = {frozenset(c) for c in out}
    adj = [[j for j in range(n) if beats[i, j]] for i in range(n)]
    for comp in _sccs(adj):
        if len(comp) >= 2:
            members = sorted(names[x] for x in comp)
            if frozenset(members) not in tri:
                out.append(members)
    return out


# ---- the report ---------------------------------------------------------------------------------

def meta_rankings(pairings, players=None, alpha=0.05, m=50, threshold=0.0, key=None) -> dict:
    """Nash averaging, alpha-Rank and cycles of a pool from its pairings, as one JSON-serializable report."""
    names, M, counts = build_payoff_matrix(pairings, players, key)
    nash = nash_averaging(M, names)
    ar = alpha_rank(M, names, alpha, m)
    cycles = detect_cycles(M, names, threshold)
    n = len(names)
    complete = all(counts[i, j] > 0 for i in range(n) for j in range(n) if i != j)
    return {'players': names, 'payoff_matrix': M.tolist(), 'counts': counts.tolist(), 'complete': complete,
            'nash': nash, 'alpha_rank': {k: v for k, v in ar.items() if k != 'transition'}, 'cycles': cycles,
            'non_transitive': bool(cycles)}
