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

  // "GPT-6 Luna · High" -> ['GPT-6 Luna', 'High', '']; a level's parenthetical is the third part:
  // "Claude Sonnet · Default effort (protocol not recorded)" -> ['Claude Sonnet', 'Default effort', 'protocol not recorded']
  function nameParts(text) {
    text = String(text);
    const i = text.lastIndexOf(' · ');
    if (i < 0) return [text, '', ''];
    const level = text.slice(i + 3), m = level.match(/^(.*?)\s*\((.*)\)\s*$/);
    return m ? [text.slice(0, i), m[1], m[2]] : [text.slice(0, i), level, ''];
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
    const byId = Object.fromEntries(out.players.map(p => [p.player, p]));
    out.pairings = out.pairings.map(p => ({ ...p, assumption: assumptionsOf(p, r.design), fit: fitOf(p, byId) }));
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
      const wins = leader ? pairings.filter(x => { const s = verdictSides(x); return s && s.winner === leader; }) : [];
      const given = g.assumptions && (g.assumptions.status || g.assumptions);
      const statuses = wins.map(x => x.assumption ? x.assumption.status : 'unknown');
      let leaderAssumption = !leader ? null : typeof given === 'string' ? given
        : statuses.includes('unsupported') ? 'unsupported' : statuses.length && statuses.every(x => x === 'supported') ? 'supported' : 'unknown';
      // a leader whose wins include one outside the calibrated range is not a decided leader
      let qualified = pooled ? null : playerId(g.leader_outside_calibration);
      let decided = leader && members.includes(leader) ? leader : null;
      if (decided && leaderAssumption === 'unsupported') { qualified = decided; decided = null; }
      if (qualified) leaderAssumption = 'unsupported';
      return { ...g, index: i, members, ranked: ranked.map(p => p.player), pairings,
        leader: decided, qualified_leader: qualified && members.includes(qualified) ? qualified : null, front_runner: front, front_runner_margin: margin,
        leader_assumption: leaderAssumption, misfits: g.misfit_pairings ?? pairings.filter(x => x.fit && x.fit.warn).length };
    });
  }

  // Which side a verdict favours, as {winner, loser} or null.
  function verdictSides(p) {
    if (p.verdict === 'a') return { winner: p.a, loser: p.b };
    if (p.verdict === 'b') return { winner: p.b, loser: p.a };
    return null;
  }

  // ---- what a verdict rests on: its assumptions, its eligibility, where it was recorded ---------------
  const ASSUME = { supported: 'diagnostics passed; distribution assumptions unverified', unsupported: 'outside calibrated range', unknown: 'assumptions unchecked' };
  const words = k => String(k).replace(/_/g, ' ');
  // The adjudicator's `assumptions` when given; else the conversation share against the share the test was
  // calibrated for. {status: 'supported'|'unsupported'|'unknown', label, detail}
  function assumptionsOf(p, design) {
    const a = p && p.assumptions;
    let status = a ? (typeof a === 'string' ? a : a.status) : null;
    let detail = '';
    if (a && typeof a === 'object' && a.checks) {
      detail = Object.entries(a.checks).map(([k, v]) => v && typeof v === 'object'
        ? `${words(k)}: ${v.status || ''}${v.value != null ? ` ${v.value}` : ''}${v.limit != null ? ` (calibrated up to ${v.limit})` : ''}`.trim()
        : `${words(k)}: ${v}`).join('; ');
    }
    if (!status) {
      const share = p && p.conversation_share;
      const cal = design && design.calibration && design.calibration.conversation_effect_share;
      if (share != null && cal != null) {
        status = share > cal ? 'unsupported' : 'supported';
        detail = `conversations differ by ${Math.round(share * 100)}% of the spread; the test is calibrated up to ${Math.round(cal * 100)}%`;
      } else status = 'unknown';
    }
    if (!ASSUME[status]) status = 'unknown';
    return { status, label: ASSUME[status], detail };
  }

  // {reasons} when a pairing may not be decided (identity, audit, protocol, registration...), else null.
  function ineligible(p) {
    if (!p || (p.verdict !== 'ineligible' && p.eligible !== false)) return null;
    return { reasons: (p.ineligible_reasons || []).map(String) };
  }

  // Where a verdict comes from: {kind: 'official'|'retrospective'|'recorded', text} or null.
  function sourceOf(p, name = x => x) {
    if (!p) return null;
    if (p.retrospective) return { kind: 'retrospective', short: 'retrospective', text: 'retrospective: recomputed by today\'s judge, not the one the run registered' };
    if (p.official) return { kind: 'official', short: 'official', text: `official: recorded when play stopped${p.judge ? ` by judge ${String(p.judge).slice(0, 8)}` : ''}` };
    if (p.recorded_verdict != null && p.recorded_verdict !== p.verdict && DECIDED.concat(['exhausted', 'ineligible']).includes(p.recorded_verdict)) {
      const was = p.recorded_verdict === 'a' ? `${name(p.a)} wins` : p.recorded_verdict === 'b' ? `${name(p.b)} wins`
        : { even: 'even', exhausted: 'no winner', ineligible: 'ineligible' }[p.recorded_verdict];
      return { kind: 'retrospective', short: `at the time: ${was}`, text: `The runner recorded "${was}" during play; that record is not a test result here, and what is shown is today's recomputation.` };
    }
    return null;
  }

  // The interval's basis, said at the number: blocks of deals inside conversations, or whole conversations.
  const BASIS = {
    blocks: { key: 'within', short: 'within conversations', long: 'This range resamples blocks of deals inside single conversations. It leaves out how much fresh conversations (new match pairs) differ, which this data cannot measure, so the real uncertainty is larger.' },
    conversations: { key: 'across', short: 'across conversations', long: 'This range resamples whole conversations (match pairs), so it includes how much fresh conversations differ.' },
    mixed: { key: 'mixed', short: 'partly within conversations', long: 'Part of this range rests on blocks of deals inside single conversations, which leave out how much fresh conversations differ.' },
  };
  const basisOf = x => (x && BASIS[x.ci_basis || x.basis]) || null;

  // How a pairing's direct result compares with what the strengths predict: {fitted, residual, warn, estimated}.
  function fitOf(p, byId) {
    const sa = byId[p.a], sb = byId[p.b];
    let fitted = p.fitted_edge;
    const estimated = fitted == null;
    if (fitted == null && sa && sb && sa.strength != null && sb.strength != null && sa.component === sb.component) fitted = sa.strength - sb.strength;
    if (fitted == null || p.edge == null) return null;
    const residual = p.residual ?? p.edge - fitted;
    const a = p.additivity;
    let warn, untestable = false;
    if (a != null) {
      const status = typeof a === 'string' ? a : typeof a === 'object' ? a.status : null;
      untestable = status === 'untestable';
      warn = typeof a === 'boolean' ? a : status ? status === 'warning' : !!(a.warning ?? a.warn);
    } else warn = !!p.ci95 && Math.abs(residual) > (p.ci95[1] - p.ci95[0]) / 2;
    return untestable ? { fitted, residual, warn, estimated, untestable } : { fitted, residual, warn, estimated };
  }

  // A player's model identity: 'match' (the backend reported the model it ran, and it is the one asked for),
  // 'mismatch' (it reported another), 'unattested' (it reported none) or 'partial' (some calls matched, the rest
  // reported none), from the ratings' own `identity_match` or the run audit's call counts; null when unknown.
  // A bare 'attested' only says some model was reported, so it never stands for a match.
  const IDENTITY = { match: 'model confirmed', mismatch: 'wrong model', unattested: 'model unconfirmed', partial: 'partly confirmed' };
  function identityOf(p, audit) {
    const own = p && (p.identity_match || (p.attestation === 'unattested' || p.identity_status === 'unattested' ? 'unattested' : null)
      || (p.attestation === 'mismatch' || p.identity_status === 'mismatch' ? 'mismatch' : null));
    if (own && IDENTITY[own]) return own;
    const a = audit && audit.identity;
    if (!a) return null;
    if (typeof a === 'string') return IDENTITY[a] ? a : null;
    if (a.mismatch) return 'mismatch';
    if (a.match && a.unattested) return 'partial';
    if (a.match) return 'match';
    return a.unattested ? 'unattested' : null;
  }

  // A verdict in words: {tone, short, long, decided, qualifier}. `name` turns a player id into display text.
  // Only a/b/even come from the sequential procedure; everything else is described, not decided. A decided
  // verdict never states its error bound without the assumptions it rests on.
  function verdict(p, name = x => x) {
    const sides = verdictSides(p);
    const looks = p.looks_planned && p.looks_used ? ` after ${p.looks_used} of ${p.looks_planned} planned checks` : '';
    const band = p.equiv_bb100 ?? 15;
    const ahead = (p.edge ?? 0) >= 0 ? p.a : p.b;
    const behind = ahead === p.a ? p.b : p.a;
    const level = !p.edge;
    const no = ineligible(p);
    if (no) {
      return { tone: 'bad', decided: false, short: 'Ineligible', long: `Not eligible for a decision${no.reasons.length ? `: ${no.reasons.join('; ')}` : ''}.${level ? '' : ` ${name(ahead)} is ahead on points only.`}` };
    }
    const as = p.assumption || { status: 'unknown' };
    const out = as.status === 'unsupported'
      ? ' The test\'s assumptions do not hold here (outside its calibrated range), so its error rate is not established.'
      : as.status === 'supported' ? ' Its error rate holds under the test\'s assumptions: the diagnostics passed, but the distribution assumptions remain unverified.'
      : ' Its error rate holds only under the test\'s assumptions, which are not checked here.';
    const qualifier = as.status === 'unsupported' ? ASSUME.unsupported : '';
    switch (p.verdict) {
      case 'a': case 'b':
        return { tone: 'won', decided: true, qualifier, short: `${name(sides.winner)} wins`, long: `Decided${looks}: ${name(sides.winner)} beats ${name(sides.loser)}; the edge cleared the sequential test's bar.${out}` };
      case 'even':
        return { tone: 'even', decided: true, qualifier, short: 'Even', long: `Decided${looks}: within ±${band} bb/100 of each other.${out}` };
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
  const KIND = { sizing: 'resized', retry: 'asked again', fallback: 'auto-played', note: 'changed', tool_use: 'tool rejected', unverified: 'unaudited, asked again' };
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

  // ---- poker fundamentals: quiz.json in one shape -----------------------------------------------------
  // {suite, version, spots: [{id, title, kind, street, hole, board, pot, to_call, stack, bb, ..., best, ev}],
  //  players: [{player, score, correct, answered, total, ev_lost_bb, reference, answers: {spot: {...}}}]}
  const num = (...xs) => xs.find(x => x != null && isFinite(x)) ?? null;
  function quizOf(raw) {
    if (!raw || typeof raw !== 'object') return null;
    const spots = (raw.spots || raw.items || raw.questions || []).map((x, i) => ({ ...x, id: String(x.id ?? x.spot ?? i),
      best: x.best ?? x.correct ?? x.answer ?? null, bb: x.bb ?? raw.bb ?? 1 }));
    const refs = new Set([].concat(raw.reference || raw.references || []).map(String));
    const list = Array.isArray(raw.players) ? raw.players : Object.entries(raw.players || raw.results || {}).map(([player, r]) => ({ player, ...r }));
    const players = list.map(r => {
      const answers = Array.isArray(r.answers) ? Object.fromEntries(r.answers.map(a => [String(a.spot ?? a.id), a])) : r.answers || {};
      const vals = Object.values(answers);
      const correct = num(r.correct, vals.length ? vals.filter(a => a.correct).length : null);
      const answered = num(r.answered, vals.length || null);
      const total = num(r.total, spots.length || answered);
      const lost = num(r.ev_lost_bb, r.ev_loss_bb, r.regret_bb, vals.length ? vals.reduce((t, a) => t + (num(a.ev_lost_bb, a.ev_loss_bb, a.regret_bb) || 0), 0) : null);
      return { ...r, answers, correct, answered, total, ev_lost_bb: lost,
        score: num(r.score, correct != null && total ? correct / total : null),
        reference: !!(r.reference || refs.has(String(r.player)) || r.kind === 'reference') };
    }).sort((x, y) => (y.score ?? -1) - (x.score ?? -1) || (x.ev_lost_bb ?? 1e9) - (y.ev_lost_bb ?? 1e9));
    return { ...raw, suite: raw.suite_name || (typeof raw.suite === 'string' ? raw.suite : raw.suite && raw.suite.name) || 'Poker fundamentals',
      version: raw.version ?? (raw.suite && raw.suite.version) ?? null, spots, players };
  }

  // ---- replay helpers ----------------------------------------------------------------------------------
  // Where two plays of a duplicate deal first part ways: {street, index, steps: [step in play 1, step in play 2]}
  // (the same seat, holding the same cards, chose differently), or null when every choice is the same.
  function firstDifference(logA, logB) {
    for (const st of ['preflop', 'flop', 'turn', 'river']) {
      const a = (logA || []).filter(e => e.kind === 'action' && e.street === st);
      const b = (logB || []).filter(e => e.kind === 'action' && e.street === st);
      for (let i = 0; i < Math.max(a.length, b.length); i++) {
        const x = a[i], y = b[i];
        if (!x || !y || x.seat !== y.seat || x.action !== y.action || (x.amount ?? null) !== (y.amount ?? null)) {
          return { street: st, index: i, steps: [x ? x.step : null, y ? y.step : null] };
        }
      }
    }
    return null;
  }
  // The deals with the biggest swings over both plays: [{deal, net}] largest first, at most n, and only those at
  // least `min` chips.
  function bigSwings(nets, n = 5, min = 0) {
    return Object.entries(nets || {}).map(([k, v]) => ({ deal: Number(k), net: v })).filter(x => Math.abs(x.net) >= min && x.net)
      .sort((x, y) => Math.abs(y.net) - Math.abs(x.net) || x.deal - y.deal).slice(0, n);
  }

  return { quizOf, firstDifference, bigSwings, IDENTITY, esc, MINUS, signed, plural, trim, approx, nameParts, family, isContract, normalize, verdict, verdictSides, powerLine, correction, actText,
    isDecided, decidedLeader, playerId, rateOf, targetEdge, isUnregistered, scenarios, conversationsShort,
    assumptionsOf, ineligible, sourceOf, basisOf, fitOf, identityOf, ASSUME };
});
