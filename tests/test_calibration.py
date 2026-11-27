"""The official verdict keeps its promise with no assumption on the shape of the results, and the round 6 gate
it replaced does not.

Official test (R6-#1): with no edge, a pairing is called for either player at most alpha_pairing of the time,
at every look, for any distribution of independent conversation means within the game's bound. That is
Ville's inequality, so these tests check the implementation, not the theorem: the reviewer's rare-loss
counterexample (seed 4901), which fooled the round 6 gate 45% of the time, and a table of null scenarios
(per-deal shapes, memory, conversation effects of every shape and size, symmetric and one-sided swings, and the
least favourable two-point distribution at the bound), on the long default design where a bettor has the most
chances to be fooled; false 'even' at and beyond the margin; coverage of the anytime-valid interval at every
look at once; the planner's power against `decide` on fresh pairings; the conversation-length trade-off. The
model-based gate is kept as a secondary result: its own calibration family still holds, and outside it the
counterexample breaks it. Strength intervals from `rate` cover the true strengths in seeded correlated
tournaments. The fast versions run on every test run; POKER_ARENA_HEAVY=1 runs the larger ones and prints the
numbers docs/methodology.md reports."""
import math
import os
import unittest

import numpy as np

from poker_arena import ratings as R
from poker_arena.ratings import decide, normalize_design, t_ppf

HEAVY = bool(os.environ.get('POKER_ARENA_HEAVY'))
THREE = [['a', 'b'], ['a', 'c'], ['b', 'c']]
NINE = [[a, b] for i, a in enumerate('abcdefghi') for b in 'abcdefghi'[i + 1:]]
DESIGN = normalize_design({'deals': 1, 'design': {}, 'replicates': 25, 'matchups': THREE})     # the default
LONG = normalize_design({'deals': 50, 'design': {'max_rounds': 10}, 'replicates': 4, 'matchups': THREE})
MB = LONG                     # the round 6 default design: 10 rounds of 4 conversations of 50 deals
LONGRUN = normalize_design({'deals': 50, 'design': {'max_rounds': 100}, 'replicates': 8, 'matchups': THREE})
SD = 350.0
DEALS = 50


def deal_noise(kind, rng, shape):
    if kind == 'normal':
        return rng.standard_normal(shape)
    if kind == 't3':
        return rng.standard_t(3, shape) / math.sqrt(3)
    if kind == 'skewed':
        return rng.exponential(1.0, shape) - 1.0
    # poker-like: about 40% of duplicate deals cancel exactly, the rest heavy-tailed (unit variance overall)
    return np.where(rng.random(shape) < 0.4, 0.0, rng.standard_t(3, shape) / math.sqrt(3 * 0.6))


def effect(shape, rng, size):
    """A unit-variance, zero-mean conversation effect."""
    if shape == 'normal':
        return rng.standard_normal(size)
    if shape == 'exponential':
        return rng.exponential(1.0, size) - 1
    if shape == 'lognormal':
        return (np.exp(rng.standard_normal(size)) - math.exp(0.5)) / math.sqrt((math.e - 1) * math.e)
    if shape == 'rare_loss':                    # the reviewer's +50 nine times in ten, -450 otherwise, in sd units
        return np.where(rng.random(size) < 0.9, 50.0, -450.0) / 150.0
    return np.where(rng.random(size) < 0.5, 1.0, -1.0)     # symmetric two-point


def conversations(rng, n, C, kind='normal', rho=0.0, shape=None, share=0.0, edge=0.0, card=1.0, means=True,
                  common=None, deals=DEALS):
    """n sequences of C conversations of `deals` deals in bb/100: per-deal noise of sd SD x card with AR(1) memory
    rho, a conversation effect of `shape` holding `share` of a conversation mean's variance, an optional shared
    deal effect `common` (n x C x DEALS, added as is), and a true edge. Conversation means (n x C) when
    `means`, else the deals."""
    card_sd = SD * card * math.sqrt(R.design_effect(rho, deals) / deals)
    tau = card_sd * math.sqrt(share / (1 - share)) if share < 1 else SD / math.sqrt(deals)
    out = []
    step = max(1, 2_000_000 // (C * deals))
    for start in range(0, n, step):
        k = min(step, n - start)
        e = deal_noise(kind, rng, (k, C, deals))
        if rho:
            for d in range(1, deals):
                e[..., d] = rho * e[..., d - 1] + math.sqrt(1 - rho * rho) * e[..., d]
        x = SD * card * e + edge
        if shape:
            x = x + tau * effect(shape, rng, (k, C, 1))
        if common is not None:
            x = x + common[start:start + k]
        out.append(x.mean(-1) if means else x)
    return np.concatenate(out)


def t_arm(m, design=MB):
    """(called for anyone, called for a): the t arm of the procedure on conversation means (n x C)."""
    reps, rounds = design['replicates'], design['max_rounds']
    a1, _, _, eligible = R._calibrate(design['alpha_pairing'], rounds, reps)
    hit = np.zeros((len(m), rounds), bool)
    T = np.zeros((len(m), rounds))
    for j in range(rounds):
        C = (j + 1) * reps
        z = m[:, :C]
        mu, s = z.mean(1), z.std(1, ddof=1)
        ok = s > 1e-9 * (1 + np.abs(mu))
        T[:, j] = np.where(ok, mu / np.where(ok, s, 1) * math.sqrt(C), 0.0)
        if eligible[j]:
            hit[:, j] = np.abs(T[:, j]) >= t_ppf(1 - a1 / 2, C - 1)
    first = hit.argmax(1)
    return hit.any(1), hit.any(1) & (T[np.arange(len(T)), first] > 0)


def through_decide(x, design=DESIGN, key=None):
    """The verdict of each sequence (n x C x deals) as the runner gets it: decide on every deal at the last look
    reports the first look at which the test decided. `key` 'model_based' takes the secondary gate's verdict."""
    reps, rounds, deals = design['replicates'], design['max_rounds'], x.shape[-1]
    groups = np.repeat(np.arange(rounds * reps), deals)
    out = []
    for seq in x:
        v = decide(list(seq[:rounds * reps].ravel()), design, len(THREE), rounds, groups=groups[:seq[:rounds * reps].size])
        out.append(v[key]['verdict'] if key else v['verdict'])
    return out


def bound(p, n, z=2.5):
    return p + z * math.sqrt(p * (1 - p) / n)


AP = DESIGN['alpha_pairing']
C_MAX = DESIGN['max_rounds'] * DESIGN['replicates']


def official(Y, design, chunk=400):
    """(verdict per sequence: 1 a, -1 b, 2 even, 0 undecided; conversations used) of the official test on
    conversation means Y (n x C, play order), looks after every `replicates` conversations: the bettors of
    `ratings.betting`, vectorized over sequences."""
    ap, b = design['alpha_pairing'], design['outcome_bound_bb100']
    reps, rounds = design['replicates'], design['max_rounds']
    Y = np.clip(np.asarray(Y, float)[:, :reps * rounds], -b, b)
    signs, used = [], []
    for i in range(0, len(Y), chunk):
        y = Y[i:i + chunk]
        times = {}
        for name, sgn, m in R.BETTORS:
            path, _ = R._bet(sgn * y, sgn * m, b)
            hit = path >= math.log(2 / ap)
            times[name] = np.where(hit.any(1), hit.argmax(1) + 1, 0)
        s_, u_ = R._plan_verdicts(times, reps, rounds)
        signs.append(s_)
        used.append(u_)
    return np.concatenate(signs), np.concatenate(used)



def rare_loss_deals(rng, shape):
    """The round 6 reviewer's independent deals: -9,975 bb/100 with chance 1/799, else 0 or +25 equally likely.
    Mean exactly zero, spread about 353, within the bound of 100-big-blind stacks."""
    loss = rng.random(shape) < 1 / 799
    return np.where(loss, -9975.0, np.where(rng.random(shape) < 0.5, 0.0, 25.0))


def two_point(rng, shape, win=50.0, bound=10_000.0):
    """The least favourable null for a bettor: a small win nearly always, the largest loss the stacks allow
    otherwise, mean exactly zero."""
    p = win / (win + bound)
    return np.where(rng.random(shape) < p, -bound, win)


class TestReviewerCounterexample(unittest.TestCase):
    """R6-#1: independent deals, -9,975 bb/100 with chance 1/799, else 0 or +25 equally likely (mean zero). The
    round 6 gate, on its default 3-player design at look 3 (12 conversations of 50 deals, seed 4901), called a
    winner in 467 of 1,000 sequences (444 with its diagnostics passed) at a nominal 1.67%."""

    def test_the_round6_gate_fails_and_the_official_test_does_not(self):
        rng = np.random.default_rng(4901)
        x = rare_loss_deals(rng, (1000, 12, DEALS))
        groups = np.repeat(np.arange(12), DEALS)
        official_, model, model_counted = 0, 0, 0
        for seq in x:
            v = decide(list(seq.ravel()), LONG, 3, 3, groups=groups)
            official_ += v['verdict'] in ('a', 'b')
            mb = v['model_based']
            model += mb['verdict'] in ('a', 'b')
            model_counted += mb['verdict'] in ('a', 'b') and mb['diagnostics']['status'] == 'supported'
        self.assertGreater(model, 300)                     # the old gate: false verdicts in about 45% of sequences
        self.assertGreater(model_counted, 250)
        self.assertEqual(official_, 0)                     # the official test: none
        if HEAVY:
            print(f'\nreviewer counterexample, seed 4901, look 3 of the round 6 design: round 6 gate {model}/1000 '
                  f'false ({model_counted} with diagnostics passed); official test {official_}/1000')

    def test_over_whole_designs(self):
        """The same deals over every look of the default design (one deal per conversation: the longest run of
        conversations, so the bettor has the most chances to be fooled) and of a long-conversation design."""
        n = 20_000 if HEAVY else 4_000
        rates = {}
        for name, design, m in (('default', DESIGN, n), ('800 conversations of 50 deals', LONGRUN, n // 10)):
            C, R_ = design['max_rounds'] * design['replicates'], design['round_deals']
            Y = np.concatenate([rare_loss_deals(np.random.default_rng([4901, i]), (min(m, 100), C, R_)).mean(-1)
                                for i in range(-(-m // 100))])[:m]
            sign, _ = official(Y, design)
            rates[name] = float(np.mean(np.abs(sign) == 1))
            with self.subTest(design=name):
                self.assertLessEqual(rates[name], bound(design['alpha_pairing'], m))
        anyone, _ = t_arm(rare_loss_deals(np.random.default_rng(4902), (n, 40, DEALS)).mean(-1))
        self.assertGreater(anyone.mean(), 5 * MB['alpha_pairing'])      # the round 6 t arm, over its whole design
        if HEAVY:
            print(f'\nreviewer counterexample over whole designs, official false-winner rate: {rates}; round 6 t arm '
                  f'over its design {anyone.mean():.4f} (target {AP:.4f})')


NULLS = [   # (name, generator of n x C x R deals in bb/100 with no edge)
    ('normal deals', lambda rng, s: SD * rng.standard_normal(s)),
    ('t3 deals', lambda rng, s: SD * deal_noise('t3', rng, s)),
    ('poker-like deals (40% zeros, t3)', lambda rng, s: SD * deal_noise('poker', rng, s)),
    ('skewed deals (centred exponential)', lambda rng, s: SD * deal_noise('skewed', rng, s)),
    ('reviewer rare loss (R6)', rare_loss_deals),
    ('two-point at the bound', two_point),
]
EFFECTS = [('normal', 0.5), ('exponential', 0.5), ('lognormal', 0.5), ('rare_loss', 0.5), ('rare_loss', 0.9),
           ('symmetric', 0.9)]


def null_means(rng, n, design, kind, effect=None, share=0.0, rho=0.0):
    C, R_ = design['max_rounds'] * design['replicates'], design['round_deals']
    out = []
    for start in range(0, n, 250):
        k = min(250, n - start)
        if effect or rho:
            out.append(conversations(rng, k, C, kind, rho, effect, share, deals=R_))
        else:
            out.append(dict(NULLS)[kind](rng, (k, C, R_)).mean(-1))
    return np.concatenate(out)


class TestOfficialNull(unittest.TestCase):
    """The calibration table: false-winner rate per pairing with no edge, official test, for the default design
    (one deal per conversation) and a design of 50-deal conversations; target alpha_pairing."""
    N = 20_000 if HEAVY else 2_000

    def test_null_scenarios(self):
        rows = {}
        cases = [(k, None, 0.0, 0.0) for k, _ in NULLS] + [('normal', e, sh, 0.0) for e, sh in EFFECTS] + \
            [('poker', 'normal', 0.25, 0.3)]
        for name, design in (('default', DESIGN), ('50-deal', LONGRUN)):
            for i, (kind, effect, share, rho) in enumerate(cases):
                if design is DESIGN and (effect or rho):
                    continue                                     # one deal per conversation: no memory, no effect
                if not HEAVY and design is LONGRUN and i % 3:
                    continue
                rng = np.random.default_rng([500, i, design['round_deals']])
                sign, _ = official(null_means(rng, self.N if design is DESIGN else self.N // 10, design, kind, effect,
                                              share, rho), design)
                r = float(np.mean(np.abs(sign) == 1))
                label = f'{name}: {kind}' + (f', {effect} effect share {share}' if effect else '') + \
                    (f', AR(1) {rho}' if rho else '')
                rows[label] = round(r, 5)
                with self.subTest(case=label):
                    self.assertLessEqual(r, bound(design['alpha_pairing'], len(sign)))
        if HEAVY:
            print(f'\nofficial test, null false-winner rate per pairing (target {AP:.4f}): {rows}')

    def test_through_decide(self):
        """The same, through `decide` itself as the runner calls it, on the least favourable null."""
        n = 400 if HEAVY else 40
        x = two_point(np.random.default_rng(510), (n, C_MAX, 1))
        rate = np.mean([c in ('a', 'b') for c in through_decide(x)])
        self.assertLessEqual(rate, bound(AP, n, 3))
        Y = x.mean(-1)
        sign, _ = official(Y, DESIGN)
        words = {1: 'a', -1: 'b', 2: 'even', 0: 'exhausted'}
        self.assertEqual([words[s] for s in sign], through_decide(x))   # the vectorized test is decide's

    def test_shared_cards(self):
        """Three pairings among three players on the same slates, correlated through a shared deal effect: any
        false verdict in the tournament at most alpha."""
        n = self.N // 2
        rng = np.random.default_rng(520)
        z = SD * rng.standard_normal((n, C_MAX))
        beta = {'a': 0.8, 'b': 0.0, 'c': -0.8}
        anyone = np.zeros(n, bool)
        for p, q in THREE:
            Y = SD * deal_noise('poker', rng, (n, C_MAX)) + (beta[p] - beta[q]) * 0.6 * z
            anyone |= np.abs(official(Y, DESIGN)[0]) == 1
        self.assertLessEqual(anyone.mean(), bound(DESIGN['alpha'], n))
        if HEAVY:
            print(f'\nany false verdict, 3 correlated pairings: {anyone.mean():.4f} (target 0.05)')


EQ = normalize_design({'deals': 1, 'design': {'max_rounds': 40}, 'replicates': 25, 'matchups': THREE,
                       'game_config': {'stack': 4}})      # 2-big-blind stacks: bound 200, where 'even' is reachable


class TestEquivalence(unittest.TestCase):
    """'even' is false whenever the true edge is at or beyond +-15 bb/100: its chance there stays within
    alpha_pairing / 2 on either side for any distribution (both bettors against the margins must reach the
    bar), and 'even' is reached where it is true."""
    N = 8_000 if HEAVY else 1_500

    def test_false_even_at_and_beyond_the_margin(self):
        b, rates = EQ['outcome_bound_bb100'], {}
        C = EQ['max_rounds'] * EQ['replicates']
        for i, edge in enumerate((15.0, -15.0, 16.0, 20.0)):
            least = lambda r, s, e=edge: e - np.sign(e) * two_point(r, s, abs(e), b - abs(e))   # mostly 0, rarely +-b
            for j, (name, gen) in enumerate((('normal', lambda r, s, e=edge: e + 60 * r.standard_normal(s)),
                                             ('least favourable two-point', least))):
                rng = np.random.default_rng([530, i, j])
                Y = np.clip(gen(rng, (self.N, C)), -b, b)
                r = float(np.mean(official(Y, EQ)[0] == 2))
                rates[f'{name} @ {edge:+.0f}'] = round(r, 4)
                with self.subTest(shape=name, edge=edge):
                    self.assertLessEqual(r, bound(EQ['alpha_pairing'] / 2, self.N))
        if HEAVY:
            print(f'\nfalse even per pairing (target {EQ["alpha_pairing"] / 2:.4f}): {rates}')

    def test_even_is_reachable_where_it_is_true(self):
        C = EQ['max_rounds'] * EQ['replicates']
        Y = np.clip(60 * np.random.default_rng(540).standard_normal((400, C)), -200, 200)
        self.assertGreater(np.mean(official(Y, EQ)[0] == 2), 0.5)
        self.assertGreater(R.procedure(EQ, 0.0, sd=60.0)['even'], 0.5)


class TestConfidenceSequence(unittest.TestCase):
    def test_covers_the_true_edge_at_every_look(self):
        """The anytime-valid interval holds the true mean at every look at once with chance >= 1 - alpha_pairing,
        whatever the shape: here the reviewer's rare-loss deals shifted to a true edge of +30."""
        n, C, ap, b = (300 if HEAVY else 60), 200, 0.05, 10_000.0
        rng = np.random.default_rng(550)
        miss = 0
        for _ in range(n):
            y = 30 + rare_loss_deals(rng, C)
            lo = hi = None
            for c in (20, 50, 100, 200):
                lo, hi = R.confidence_sequence(y[:c], ap, b)
                if not lo <= 30 <= hi:
                    miss += 1
                    break
        self.assertLessEqual(miss / n, bound(ap, n, 3))
        lo, hi = R.confidence_sequence(np.full(400, 9000.0), ap, b)
        self.assertGreater(lo, 0)
        self.assertGreaterEqual(9000.0, lo)

    def test_agrees_with_the_verdict(self):
        y = 900 + 20 * np.random.default_rng(560).standard_normal(12)
        t = R.betting(y, 0.05, 1000.0)
        self.assertEqual(t['verdict'], 'a')
        self.assertGreater(R.confidence_sequence(y, 0.05, 1000.0)[0], 0)
        self.assertEqual(R.min_conversations(0.05), 7)
        self.assertIsNone(R.betting(np.full(5, 1000.0), 0.05, 1000.0)['verdict'])   # fewer than min: impossible


def planning_means(rng, n, edge, design, icc=None, rho=None, kind='normal'):
    """Conversation means as the planner draws them (normal deals: spread SD, AR(1), a conversation effect),
    or from poker-like deals."""
    icc = design['assumed_icc'] if icc is None else icc
    rho = design['assumed_ar1'] if rho is None else rho
    C, R_ = design['max_rounds'] * design['replicates'], design['round_deals']
    out = []
    for start in range(0, n, 200):
        k = min(200, n - start)
        e = deal_noise(kind, rng, (k, C, R_))
        for d in range(1, R_):
            e[..., d] = rho * e[..., d - 1] + math.sqrt(1 - rho * rho) * e[..., d]
        x = edge + SD * math.sqrt(1 - icc) * e + SD * math.sqrt(icc) * rng.standard_normal((k, C, 1))
        out.append(x)
    return np.concatenate(out)


class TestPower(unittest.TestCase):
    """R6-#1: the power a plan advertises is the official test's, simulated: checked against the official test
    on fresh pairings (normal and poker-like deals) and through `decide`, and the costs against the round 6
    gate (heavy: printed for docs/methodology.md)."""

    def test_advertised_power_is_the_tests(self):
        n = 3000 if HEAVY else 600
        cases = [('default', DESIGN, 'normal')] + ([('default', DESIGN, 'poker')] if HEAVY else [])
        for name, design, kind in cases:
            x = planning_means(np.random.default_rng(80), n, design['target_edge_bb100'], design, kind=kind)
            sign, used = official(x.mean(-1), design)
            got = float(np.mean(sign == 1))
            se = math.sqrt(design['power_at_max'] * (1 - design['power_at_max']) / n)
            with self.subTest(design=name, deals=kind):
                self.assertAlmostEqual(got, design['power_at_max'], delta=3.5 * se + 0.015)
                self.assertFalse((sign == -1).any())
            if HEAVY:
                print(f'\npower at the target, {name} design, {kind} deals: design {design["power_at_max"]:.3f}, '
                      f'simulated test {got:.3f}, conversations used {used.mean():.0f}')
        m = 20 if not HEAVY else 200
        x = planning_means(np.random.default_rng(81), m, DESIGN['target_edge_bb100'], DESIGN)
        words = {1: 'a', -1: 'b', 2: 'even', 0: 'exhausted'}
        self.assertEqual([words[s] for s in official(x.mean(-1), DESIGN)[0]], through_decide(x))

    def test_design_is_powered_and_minimal(self):
        self.assertGreaterEqual(DESIGN['power_at_max'], 0.8)
        fewer = normalize_design({'deals': 1, 'design': {'max_rounds': DESIGN['max_rounds'] - 1}, 'replicates': 25,
                                  'matchups': THREE})
        self.assertLess(fewer['power_at_max'], 0.8)
        self.assertIn('official test', DESIGN['power_basis'])
        approx = R.conversations_for(50, 350, DESIGN['alpha_pairing'], 10_000)
        self.assertAlmostEqual(approx, DESIGN['conversations_needed'], delta=0.06 * approx + 25)

    def test_length_tradeoff(self):
        rows = R.length_tradeoff(DESIGN, (1, 5, 50))
        deals = [x['deals'] for x in rows]
        self.assertTrue(deals[0] < deals[1] < deals[2])
        self.assertLess(rows[2]['conversations'], rows[0]['conversations'])   # fewer, but much longer
        if HEAVY:
            print(f'\nconversation length trade-off, 3-model default target: {R.length_tradeoff(DESIGN)}')

    def test_leader_chance(self):
        lead = {x['scenario']: x for x in DESIGN['leader_chances']}
        p = R.procedure(DESIGN, DESIGN['target_edge_bb100'])['counting']
        self.assertAlmostEqual(lead['one best']['p_leader'], p ** 2, places=3)

    @unittest.skipUnless(HEAVY, 'heavy: the cost of the official test against the round 6 gate')
    def test_report_costs(self):
        out = {}
        for name, m in (('3 models', THREE), ('9 models', NINE)):
            new = normalize_design({'deals': 1, 'design': {}, 'replicates': 25, 'matchups': m})
            long_ = normalize_design({'deals': 50, 'design': {}, 'replicates': 4, 'matchups': m})
            old = {'round_deals': 50, 'block': 10, 'replicates': 4, 'alpha_pairing': new['alpha_pairing'],
                   'max_rounds': 10 if m is THREE else 15, 'assumed_icc': 0.01, 'assumed_ar1': 0.2,
                   'prior_sd_bb100': SD, 'outside_calibration': 'exclude'}
            o = R.procedure_model_based(old, 50.0)
            on_old = R.procedure({**long_, 'max_rounds': old['max_rounds']}, 50.0)
            out[name] = {'official, 1-deal conversations': (new['deals_needed'], new['power_scenarios'],
                                                            new['power_by_edge'], new['leader_chances']),
                         'official, 50-deal conversations': long_['deals_needed'] or
                         R.length_tradeoff(long_, (50,))[0]['deals'],
                         'round 6 gate (deals planned, counting power, conversations used)':
                             (old['max_rounds'] * 200, o['counting'], o['conversations']),
                         'official test on the round 6 budget': on_old['right']}
        print(f'\ncosts: {out}')


class TestModelBased(unittest.TestCase):
    """The round 6 gate, kept as a secondary result: within its calibrated family it still keeps its level (the t
    arm alone, vectorized, on its own design); outside it, it does not (above), which is why it is never
    official."""
    N = 200_000 if HEAVY else 12_000

    def test_calibrated_family(self):
        rates = {}
        for i, (kind, rho, shape, share) in enumerate(FAMILY_SETTINGS):
            m = conversations(np.random.default_rng(100 + i), self.N, 40, kind, rho, shape, share)
            anyone, _ = t_arm(m)
            rates[f'{kind}/ar={rho}/{shape}@{share}'] = round(float(anyone.mean()), 4)
            with self.subTest(kind=kind, rho=rho, shape=shape, share=share):
                self.assertLessEqual(anyone.mean(), bound(MB['alpha_pairing'], self.N))
        if HEAVY:
            print(f'\nround 6 t arm within its family (target {MB["alpha_pairing"]:.4f}): {rates}')

    def test_two_constant_conversations(self):
        v = decide([50.0] * 100, LONG, 3, 1, groups=['c1'] * 50 + ['c2'] * 50)
        self.assertEqual((v['verdict'], v['model_based']['verdict']), ('early', 'early'))


FAMILY_SETTINGS = [                               # (deals, AR(1), conversation effect shape, its share)
    ('normal', 0.0, None, 0.0), ('poker', 0.0, None, 0.0), ('poker', 0.3, 'normal', 0.5),
    ('normal', 0.0, 'exponential', 0.5), ('normal', 0.0, 'lognormal', 0.5), ('normal', 0.0, 'rare_loss', 0.5),
]


def tournament(rng, players, true, slates, deals=50, su=0.5, rho=0.2, common=0.6, kind='poker'):
    """Duplicate deals of a round robin played on `slates` shared slates (every pairing gets the same seeds in
    a slate). A pair's deal result is the strength difference plus a shared-card effect through style
    loadings, a conversation effect (su x the card noise of a conversation mean) and AR(1) noise."""
    beta = dict(zip(players, np.linspace(-1, 1, len(players))))
    z = rng.standard_normal((1, slates, deals)) * SD
    share = su * su / (1 + su * su)
    out = []
    for i, a in enumerate(players):
        for b in players[i + 1:]:
            x = conversations(rng, 1, slates, kind, rho, 'normal' if su else None, share, true[a] - true[b],
                              means=False, common=(beta[a] - beta[b]) * common * z, deals=deals)[0]
            for s in range(slates):
                for k in range(deals):
                    out.append({'a': a, 'b': b, 'matchup': f'{a}__vs__{b}__r{s + 1}', 'deal': k, 'bb': 2,
                                'net_a': x[s, k] * 4 / 100, 'seed': 1000 + s * deals + k, 'slate': 1000 + s * deals,
                                'round': s + 1, 'replicate': 1, 'planned': deals, 'fp': f'{1000 + s * deals + k}:{k % 2}'})
    return out


class TestCoverage(unittest.TestCase):
    """Strength intervals from rate() in seeded correlated tournaments."""
    TRUE = {'p': 60.0, 'q': 20.0, 'r': -20.0, 's': -60.0}

    def coverage(self, n, slates, **kw):
        rng = np.random.default_rng(2026 + slates)
        hit, total, basis = 0, 0, set()
        for t in range(n):
            r = R.rate(tournament(rng, list(self.TRUE), self.TRUE, slates, **kw), boot=200, seed=t)
            for p in r['players']:
                lo, hi = p['strength_ci95']
                hit += lo <= self.TRUE[p['player']] <= hi
                total += 1
                basis.add(p['ci_basis'])
        return hit / total, basis

    def test_conversations(self):
        cov, basis = self.coverage(400 if HEAVY else 40, 4)
        self.assertEqual(basis, {'conversations'})
        self.assertGreater(cov, 0.88)
        if HEAVY:
            print(f'\ncoverage, 4 slates, conversation effects and shared cards: {cov:.3f}')

    def test_blocks_within_one_conversation(self):
        cov, basis = self.coverage(400 if HEAVY else 40, 1, deals=100, su=0.0, rho=0.0)
        self.assertEqual(basis, {'blocks'})
        self.assertGreater(cov, 0.88)
        if HEAVY:
            print(f'\ncoverage, 1 slate, no conversation effect: {cov:.3f}')


if __name__ == '__main__':
    unittest.main()
