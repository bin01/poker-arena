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

  function normalize(r) {
    r = r || {};
    if (isContract(r)) {
      const players = (r.players || []).map(p => ({ rank_tied_with: [], quarantined: false, ...p }));
      const names = players.map(p => p.player);
      const leaders = (Array.isArray(r.leaders) ? r.leaders : r.leader == null ? [] : Array.isArray(r.leader) ? r.leader : [r.leader]).filter(Boolean);
      return {
        ...r, contract: true, players, leaders,
        components: r.components && r.components.length ? r.components : names.length ? [names] : [],
        pairings: r.pairings || [], roster: r.roster || [], notes: r.notes || [],
        hands: r.hands ?? players.reduce((s, p) => s + (p.hands || 0), 0) / 2,
      };
    }
    const legacy = r.players || [];
    const names = legacy.map(p => p.player);
    const players = legacy.map(p => ({
      player: p.player, kind: (r.kinds || {})[p.player], component: 0,
      strength: p.bb_per_100 ?? null, strength_ci95: p.bb_per_100_ci95 || null,
      rank: null, rank_tied_with: names.filter(n => n !== p.player),
      net_bb: p.net_bb, hands: p.hands, deals: Math.round((p.hands || 0) / 2),
      session_elo: p.elo ?? null, session_elo_ci95: p.elo90 || null, sessions: p.sessions,
      session_w: p.session_wins, session_t: p.session_ties, session_l: p.session_losses,
      quarantined: false, quarantine_reason: null,
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

  // Which side a verdict favours, as {winner, loser} or null.
  function verdictSides(p) {
    if (p.verdict === 'a') return { winner: p.a, loser: p.b };
    if (p.verdict === 'b') return { winner: p.b, loser: p.a };
    return null;
  }

  // A verdict in words: {tone, short, long}. `name` turns a player id into display text.
  function verdict(p, name = x => x) {
    const sides = verdictSides(p);
    const looks = p.looks_planned && p.looks_used ? ` after ${p.looks_used} of ${p.looks_planned} checks` : '';
    const band = p.equiv_bb100 ?? 15;
    const ahead = (p.edge ?? 0) >= 0 ? p.a : p.b;
    switch (p.verdict) {
      case 'a': case 'b':
        return { tone: 'won', short: `${name(sides.winner)} wins`, long: `${name(sides.winner)} is better than ${name(sides.loser)}${looks}: the edge cleared the bar for calling it.` };
      case 'even':
        return { tone: 'even', short: 'Even', long: `Within ±${band} bb/100 of each other${looks}: as good as each other at this sample.` };
      case 'exhausted':
        return { tone: 'open', short: 'No winner', long: `Every planned check is used and neither is clearly better. ${name(ahead)} is ahead on points only.` };
      case 'early':
        return { tone: 'early', short: 'Too early', long: `Too few deals to judge${p.deals ? ` (${p.deals} so far)` : ''}.` };
      default:
        return { tone: 'open', short: 'Undecided', long: `${name(ahead)} is ahead, but not by enough to call it yet${looks}.` };
    }
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

  return { esc, MINUS, signed, plural, trim, approx, nameParts, family, isContract, normalize, verdict, verdictSides, powerLine, correction, actText };
});
