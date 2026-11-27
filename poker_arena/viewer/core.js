// Pure helpers shared by the pages: formatting, the ratings in one shape, verdicts and corrections.
// No DOM here, so tests can load it in node.
(function (root, make) {
  const api = make();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Core = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const MINUS = '−';
  const signed = (x, d = 0) => x == null ? '' : (Math.round(x * 10 ** d) > 0 ? '+' : Math.round(x * 10 ** d) < 0 ? MINUS : '') + Math.abs(Number(x)).toFixed(d);
  const plural = (n, word, many = word + 's') => `${Number(n).toLocaleString('en-US')} ${n === 1 ? word : many}`;
  const trim = v => Number.isInteger(v) ? String(v) : v.toFixed(1).replace(/\.0$/, '');
  const approx = n => n < 100 ? String(n) : Number(n.toPrecision(2)).toLocaleString('en-US');

  // "GPT-6 Luna · High" -> ['GPT-6 Luna', 'High']
  function nameParts(text) {
    const i = String(text).lastIndexOf(' · ');
    return i < 0 ? [String(text), ''] : [text.slice(0, i), text.slice(i + 3)];
  }

  function family(text, kind) {
    const k = String(text).toLowerCase();
    if (kind === 'bot' || /\bbot\b|calling station/.test(k)) return 'bot';
    if (/claude|opus|sonnet|haiku|anthropic/.test(k)) return 'claude';
    if (/gpt|openai|\bo\d/.test(k)) return 'gpt';
    if (/gemini|google/.test(k)) return 'gemini';
    if (/llama|meta/.test(k)) return 'llama';
    if (/grok|xai/.test(k)) return 'grok';
    return 'other';
  }

  // ---- ratings: the contract shape, from either the contract or the older Elo output ----------------
  // Contract: {objective, deals, hands, components, leader, players[{strength, strength_ci95, rank,
  // rank_tied_with, session_elo...}], pairings[{a, b, edge, ci95, verdict, ...}], roster, notes}
  function isContract(r) {
    return !!r && ('objective' in r || 'pairings' in r || (r.players || []).some(p => 'strength' in p));
  }

  const DECIDED = ['a', 'b', 'even'];
  const isDecided = p => !!p && DECIDED.includes(p.verdict);
  const playerId = x => x == null ? null : typeof x === 'object' ? x.player : x;

  // `pooled`: an all-runs view, which is descriptive only: no pairing verdict there is a decision.
  // A run without a recorded design (or with one the ratings invented for an older run) is unregistered.
  const isUnregistered = d => !!d && (d.registered === false || (d.registered == null && !!d.legacy));

  function normalize(r, { pooled = false } = {}) {
    r = r || {};
    const out = isContract(r) ? fromContract(r) : fromLegacy(r);
    const minConv = r.min_conversations ?? (r.design && r.design.min_conversations);
    if (minConv != null) out.pairings = out.pairings.map(p => ({ min_conversations: minConv, ...p }));
    const unregistered = !pooled && isUnregistered(r.design);
    const descriptive = pooled || unregistered;
    if (descriptive) {
      const why = pooled ? 'pooled' : 'unregistered';
      out.pairings = out.pairings.map(p => ({ ...p, verdict_recorded: p.verdict, verdict: 'descriptive', descriptive_reason: why }));
      out.leaders = [];
      out.players = out.players.map(p => ({ ...p, rank_kind: 'descriptive', rank: null }));
    }
    out.pooled = pooled;
    out.unregistered = unregistered;
    out.groups = groupsOf(out, r.groups, descriptive);
    if (descriptive) out.groups.forEach(g => { g.leader = null; });
    return out;
  }

  function fromContract(r) {
    const players = (r.players || []).map(p => ({
      rank_tied_with: [], quarantined: false, ...p,
      identity_resolved: p.identity_resolved ?? !p.quarantined,
      in_fit: p.in_fit ?? p.strength != null,
      rank_kind: p.rank_kind || 'descriptive',
    }));
    const names = players.map(p => p.player);
    const leaders = (Array.isArray(r.leaders) ? r.leaders : r.leader == null ? [] : Array.isArray(r.leader) ? r.leader : [r.leader]).filter(Boolean);
    return {
      ...r, contract: true, players, leaders,
      components: r.components && r.components.length ? r.components : names.length ? [names] : [],
      pairings: r.pairings || [], roster: r.roster || [], notes: r.notes || [],
      hands: r.hands ?? players.reduce((s, p) => s + (p.hands || 0), 0) / 2,
    };
  }

  function fromLegacy(r) {
    const legacy = r.players || [];
    const names = legacy.map(p => p.player);
    const players = legacy.map(p => ({
      player: p.player, kind: (r.kinds || {})[p.player], component: 0,
      strength: p.bb_per_100 ?? null, strength_ci95: p.bb_per_100_ci95 || null,
      rank: null, rank_kind: 'descriptive', rank_tied_with: names.filter(n => n !== p.player),
      net_bb: p.net_bb, hands: p.hands, deals: Math.round((p.hands || 0) / 2),
      session_elo: p.elo ?? null, session_elo_ci95: p.elo90 || null, sessions: p.sessions,
      session_w: p.session_wins, session_t: p.session_ties, session_l: p.session_losses,
      quarantined: false, quarantine_reason: null, in_fit: p.bb_per_100 != null, identity_resolved: true,
    })).sort((x, y) => (y.strength ?? -1e9) - (x.strength ?? -1e9));
    const pairings = (r.matchups || []).map(m => {
      const edge = m.a_bb_per_100 ?? 0;
      const verdict = m.status === 'settled' ? (edge >= 0 ? 'a' : 'b') : m.status === 'early' ? 'early' : 'open';
      return {
        a: m.a, b: m.b, matchups: m.matchups || [m.matchup], deals: m.deals, blocks: null, edge,
        ci95: m.se != null ? [edge - 1.96 * m.se, edge + 1.96 * m.se] : null,
        t: m.z ?? null, boundary: r.z_stop ?? null, verdict, looks_used: null, looks_planned: null, power: null,
      };
    });
    return {
      ...r, contract: false, objective: 'bb/100 over duplicate deals', players, pairings, leaders: [],
      components: names.length ? [players.map(p => p.player)] : [], roster: r.roster || [], notes: r.notes || [],
      deals: r.deals || 0, hands: 2 * (r.deals || 0),
    };
  }

  // A leader is inferential: a decided win over every other player of its group. Without the ratings'
  // own groups, they are rebuilt from the components and the pairings' verdicts.
  function decidedLeader(members, pairings) {
    if (members.length < 2) return null;
    return members.find(p => members.every(q => q === p || pairings.some(x =>
      ((x.a === p && x.b === q && x.verdict === 'a') || (x.a === q && x.b === p && x.verdict === 'b'))))) || null;
  }

  function groupsOf(rt, given, pooled) {
    const fit = new Set(rt.players.filter(p => p.in_fit).map(p => p.player));
    const byId = Object.fromEntries(rt.players.map(p => [p.player, p]));
    const own = Array.isArray(given) && given.length;
    const list = own ? given.map((g, i) => ({ ...g, id: g.id ?? i, players: g.players || rt.components[i] || [] }))
      : rt.components.map((c, i) => ({ id: i, players: c }));
    return list.map((g, i) => {
      const members = g.players.filter(p => fit.has(p));
      const ranked = members.map(p => byId[p]).filter(p => p && p.strength != null).sort((x, y) => y.strength - x.strength);
      const pairings = rt.pairings.filter(p => g.players.includes(p.a) && g.players.includes(p.b));
      const leader = pooled ? null : own ? playerId(g.leader) : decidedLeader(members, pairings);
      const front = playerId(g.front_runner) || (ranked[0] && ranked[0].player) || null;
      const margin = g.front_runner_margin ?? (ranked.length > 1 ? ranked[0].strength - ranked[1].strength : null);
      return { ...g, index: i, members, ranked: ranked.map(p => p.player), pairings,
        leader: leader && members.includes(leader) ? leader : null, front_runner: front, front_runner_margin: margin };
    });
  }

  // Which side a verdict favours, as {winner, loser} or null.
  function verdictSides(p) {
    if (p.verdict === 'a') return { winner: p.a, loser: p.b };
    if (p.verdict === 'b') return { winner: p.b, loser: p.a };
    return null;
  }

  // A verdict in words: {tone, short, long, decided}. `name` turns a player id into display text.
  // Only a/b/even come from the sequential procedure; everything else is described, not decided.
  function verdict(p, name = x => x) {
    const sides = verdictSides(p);
    const looks = p.looks_planned && p.looks_used ? ` after ${p.looks_used} of ${p.looks_planned} planned checks` : '';
    const band = p.equiv_bb100 ?? 15;
    const ahead = (p.edge ?? 0) >= 0 ? p.a : p.b;
    const behind = ahead === p.a ? p.b : p.a;
    const level = !p.edge;
    switch (p.verdict) {
      case 'a': case 'b':
        return { tone: 'won', decided: true, short: `${name(sides.winner)} wins`, long: `Decided${looks}: ${name(sides.winner)} beats ${name(sides.loser)}; the edge cleared the sequential test's bar.` };
      case 'even':
        return { tone: 'even', decided: true, short: 'Even', long: `Decided${looks}: within ±${band} bb/100 of each other.` };
      case 'exhausted':
        return { tone: 'open', decided: false, short: 'No winner', long: `Every planned check is used and the difference is unresolved.${level ? '' : ` ${name(ahead)} is ahead on points only.`}` };
      case 'early': {
        const conv = conversationsShort(p);
        return { tone: 'early', decided: false, short: 'Too early', long: conv
          ? `No verdict before ${p.min_conversations} independent conversations (match pairs); it has ${p.conversations}.`
          : `Too few deals to judge${p.deals ? ` (${p.deals} so far)` : ''}.` };
      }
      case 'descriptive': case 'unregistered': {
        const why = p.descriptive_reason === 'unregistered' || p.verdict === 'unregistered'
          ? 'This run has no recorded test plan, so descriptive only: no test decides it.'
          : 'Pooled across runs, so descriptive only: no test decides it here.';
        return { tone: 'open', decided: false, short: level ? 'Level on points' : 'Ahead on points', long: `${why}${level ? '' : ` ${name(ahead)} is ahead of ${name(behind)} on points.`}` };
      }
      default:
        return { tone: 'open', decided: false, short: 'Undecided', long: `Not decided yet${looks}.${level ? '' : ` ${name(ahead)} is ahead on points.`}` };
    }
  }

  // True when a pairing has fewer independent conversations than a verdict needs.
  const conversationsShort = p => !!p && p.min_conversations != null && p.conversations != null && p.conversations < p.min_conversations;

  // A design's power scenarios as [{name, power, deals}] in a fixed order, from a map or a list.
  const SCENARIOS = ['optimistic', 'moderate', 'pessimistic'];
  function scenarios(design) {
    const raw = design && design.power_scenarios;
    if (!raw) return [];
    const list = Array.isArray(raw) ? raw.map(x => [x.name || x.scenario, x]) : Object.entries(raw);
    return list.filter(([n, x]) => n && x != null).map(([n, x]) => {
      const o = typeof x === 'number' ? { power: x } : x;
      return { name: n, power: o.power_at_max ?? o.power ?? null, deals: o.deals_needed ?? o.deals ?? null };
    }).sort((a, b) => (SCENARIOS.indexOf(a.name) + 99) % 99 - (SCENARIOS.indexOf(b.name) + 99) % 99);
  }

  // "To tell a 50 bb/100 edge apart takes about N deals" from decide()'s power table.
  function powerLine(p) {
    const t = p && p.power && p.power.deals_for_80pct;
    if (!t) return '';
    const bits = ['20', '50', '100'].filter(k => t[k] != null).map(k => `${k} bb/100: ~${approx(t[k])}`);
    return bits.length ? `Deals needed to detect an edge (80% power): ${bits.join(' · ')}` : '';
  }

  // ---- corrections: what the model asked for, what was played, and why ------------------------------
  const ACT = { fold: 'fold', check: 'check', call: 'call', bet: 'bet', raise: 'raise to', all_in: 'all in' };
  function actText(a, bbFmt) {
    if (!a || !a.action) return '';
    const amt = a.amount != null && a.action !== 'fold' && a.action !== 'check' ? ' ' + bbFmt(a.amount) : '';
    return (ACT[a.action] || a.action) + amt;
  }
  const KIND = { sizing: 'resized', retry: 'asked again', fallback: 'auto-played', note: 'changed' };
  // From a thought event (and its action): null, or {kind, label, requested, executed, reason}.
  function correction(thought, action, bbFmt = String) {
    const t = thought || {};
    const c = t.correction || (action && action.correction);
    if (c) {
      return { kind: c.kind, label: KIND[c.kind] || c.kind, requested: actText(t.requested || (action && action.requested), bbFmt),
        executed: actText(t.executed || action, bbFmt), reason: c.reason || '' };
    }
    if ('correction' in t) return null;
    if (t.fallback) return { kind: 'fallback', label: KIND.fallback, requested: '', executed: actText(action, bbFmt),
      reason: 'no playable answer, so the game checked or folded for it' };
    if (action && action.note) return { kind: 'sizing', label: KIND.note, requested: t.action ? actText({ action: t.action, amount: t.amount }, bbFmt) : '',
      executed: actText(action, bbFmt), reason: action.note };
    return null;
  }

  // A rate as {x, n, of}: the stats' own counts when given, else rebuilt from the rate and its base.
  function rateOf(s, k) {
    s = s || {};
    const v = s[k];
    const c = v && typeof v === 'object' ? v : (s.counts || {})[k];
    if (c && typeof c === 'object') return { x: c.of ? c.n / c.of : null, n: c.n ?? null, of: c.of ?? null };
    if (v == null) return { x: null, n: null, of: null };
    const of = { vpip: s.hands, pfr: s.hands, wtsd: v && s.showdowns != null ? Math.round(s.showdowns / v) : null,
      wsd: s.showdowns, talk: s.decisions }[k] ?? null;
    const n = k === 'wsd' && s.won_sd != null ? s.won_sd : k === 'wtsd' && s.showdowns != null ? s.showdowns
      : of != null ? Math.round(v * of) : null;
    return { x: v, n, of };
  }

  // The preregistered target edge and how far some pairings are from it:
  // {target, need, planned (from the design, or from the spread seen so far), have: [min, max]} or null.
  // An unregistered run's numbers are retrospective: `planned` is false and `retrospective` true.
  function targetEdge(design, pairings) {
    const have = pairings.map(p => p.deals || 0);
    const range = have.length ? [Math.min(...have), Math.max(...have)] : [0, 0];
    if (design && design.target_edge_bb100 && (design.deals_needed || design.deals_planned)) {
      if (isUnregistered(design)) {
        return { target: design.target_edge_bb100, need: design.deals_needed || design.deals_planned, planned: false, retrospective: true, have: range };
      }
      return { target: design.target_edge_bb100, need: design.deals_needed || design.deals_planned, allowed: design.max_deals || null,
               power: design.power_at_max ?? null, planned: true, have: range };
    }
    const need = pairings.map(p => p.power?.deals_for_80pct?.['50']).filter(x => x != null);
    if (!need.length) return null;
    return { target: 50, need: Math.max(...need), planned: false, ...(isUnregistered(design) ? { retrospective: true } : {}), have: range };
  }

  return { esc, MINUS, signed, plural, trim, approx, nameParts, family, isContract, normalize, verdict, verdictSides, powerLine, correction, actText,
    isDecided, decidedLeader, playerId, rateOf, targetEdge, isUnregistered, scenarios, conversationsShort };
});
