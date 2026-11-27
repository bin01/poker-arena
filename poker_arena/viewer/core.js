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
  // A verdict the test reached while the edge drifted between early and late conversations: it says only that a
  // player was ahead for part of the run, so it is never decided and never makes a leader. From the ratings'
  // `decided_with_drift` status when they give it, else from a decided verdict with drift detected beside it.
  const driftOf = p => { const d = p && p.drift; return !d ? null : typeof d === 'string' ? { status: d } : d; };
  const DRIFT_STATUS = 'decided_with_drift';
  // What a decided win says about the models: only as much as the names stayed the same models throughout.
  const STABLE_NOTE = 'It shows the better model only if each name served one unchanged model for the whole run, so that its expected edge stayed the same; a drift check that finds nothing does not prove that.';
  function driftQualified(p) {
    if (!p || p.track === 'adaptation') return false;
    if ([p.verdict, p.decision_status, p.verdict_status, p.status, p.standing].includes(DRIFT_STATUS)) return true;
    return DECIDED.includes(p.verdict) && /^drift detected/.test((driftOf(p) || {}).status || '');
  }
  const countsForLeader = p => !driftQualified(p) && !idWithheld(p) && p.counts_toward_leader !== false;
  const idWithheld = p => !!p && p.decision_status === 'decided_identity_unattested';
  const isDecided = p => !!p && DECIDED.includes(p.verdict) && !driftQualified(p) && !idWithheld(p);
  const playerId = x => x == null ? null : typeof x === 'object' ? x.player : x;

  // `pooled`: an all-runs view, which is descriptive only: no pairing verdict there is a decision.
  // A run without a recorded design (or with one the ratings invented for an older run) is unregistered.
  const isUnregistered = d => !!d && (d.registered === false || (d.registered == null && !!d.legacy));

  function normalize(r, { pooled = false } = {}) {
    r = r || {};
    const out = isContract(r) ? fromContract(r) : fromLegacy(r);
    const minConv = r.min_conversations ?? (r.design && r.design.min_conversations);
    if (minConv != null) out.pairings = out.pairings.map(p => ({ min_conversations: minConv, ...p }));
    out.players = out.players.map(p => p.observed_edge !== undefined || !p.in_fit ? p : { ...p, ...observedOf(p.player, out.pairings) });
    const byId = Object.fromEntries(out.players.map(p => [p.player, p]));
    out.pairings = out.pairings.map(p => ({ ...p, design_ref: r.design || null, assumption: assumptionsOf(p, r.design), fit: fitOf(p, byId) }));
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

  // Observed opponent-weighted winnings: the mean of a player's direct edges, one per opponent, each weighted equally.
  function observedOf(player, pairings) {
    const edges = (pairings || []).filter(x => (x.a === player || x.b === player) && x.edge != null)
      .map(x => x.a === player ? x.edge : -x.edge);
    return { observed_edge: edges.length ? Math.round(edges.reduce((s, e) => s + e, 0) / edges.length * 10) / 10 : null, observed_opponents: edges.length };
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
    return members.find(p => members.every(q => q === p || pairings.some(x => x.track !== 'adaptation' && countsForLeader(x) &&
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
        : statuses.includes('unsupported') ? 'unsupported' : statuses.length && statuses.every(x => x === 'free') ? 'free'
        : statuses.length && statuses.every(x => x === 'supported' || x === 'free') ? 'supported' : 'unknown';
      // a leader whose wins include one outside the calibrated range is not a decided leader
      let qualified = pooled ? null : playerId(g.leader_outside_calibration);
      let decided = leader && members.includes(leader) ? leader : null;
      // a player that beat everyone only with the help of a drifted verdict: shown as such, never as a leader
      const beat = (x, q, w) => x.track !== 'adaptation' && (DECIDED.includes(x.verdict) || driftQualified(x)) && verdictSides(x)
        && verdictSides(x).winner === w && verdictSides(x).loser === q;
      const loose = !pooled && !decided && members.length > 1 && members.find(w => members.every(q => q === w || pairings.some(x => beat(x, q, w))));
      const driftLeader = decided && wins.some(x => !countsForLeader(x)) ? decided
        : loose && pairings.some(x => verdictSides(x)?.winner === loose && driftQualified(x)) ? loose : null;
      if (driftLeader) decided = null;
      if (decided && leaderAssumption === 'unsupported') { qualified = decided; decided = null; }
      if (qualified) leaderAssumption = 'unsupported';
      return { ...g, index: i, members, ranked: ranked.map(p => p.player), pairings,
        leader: decided, drift_leader: driftLeader, qualified_leader: qualified && members.includes(qualified) ? qualified : null, front_runner: front, front_runner_margin: margin,
        leader_assumption: leaderAssumption, misfits: g.misfit_pairings ?? pairings.filter(x => x.fit && x.fit.warn).length };
    });
  }

  // Which side a verdict favours, as {winner, loser} or null.
  function verdictSides(p) {
    let v = p.verdict;
    if (v === DRIFT_STATUS) {
      const s = [p.side, p.decided_side, p.verdict_side, p.winner].find(x => x != null);
      v = s === p.a ? 'a' : s === p.b ? 'b' : s || ((p.edge ?? 0) >= 0 ? 'a' : 'b');
    }
    if (v === 'a') return { winner: p.a, loser: p.b };
    if (v === 'b') return { winner: p.b, loser: p.a };
    return null;
  }

  // A drift-qualified verdict in words: {phrase: 'ahead early, behind late', early, late} from the winner's side.
  function driftWords(p) {
    const d = driftOf(p) || {}, s = verdictSides(p);
    const flip = s && s.winner === p.b ? -1 : 1;
    const early = d.early != null ? flip * d.early : null, late = d.late != null ? flip * d.late : null;
    const at = x => x > 0 ? 'ahead' : x < 0 ? 'behind' : 'level';
    const phrase = early != null && late != null ? `${at(early)} early, ${at(late)} late` : 'the edge changed over the run';
    return { phrase, early, late };
  }

  // The official pairing's anytime-valid interval (the ratings' `ci_official`, else the betting confidence
  // sequence `cs`) as {lo, hi, level}, in a's orientation; null when the pairing has none or is not official.
  function officialInterval(p) {
    if (!p || p.track === 'adaptation' || ['descriptive', 'unregistered'].includes(p.verdict)) return null;
    const o = p.ci_official ?? p.cs;
    if (!o) return null;
    const iv = Array.isArray(o) ? o : o.interval || o.ci || o.range || (o.lo != null ? [o.lo, o.hi] : null);
    if (!iv || iv[0] == null || iv[1] == null) return null;
    const level = (Array.isArray(o) ? null : o.level ?? o.cs_level) ?? p.cs_level ?? null;
    return { lo: iv[0], hi: iv[1], level };
  }
  // What an ordinary (t or bootstrap) interval is: fixed-time and descriptive, whatever the ratings call it.
  const DESCRIPTIVE_TIP = 'An ordinary interval, computed as if the number of deals had been fixed in advance. Play stops on the results, and the player in front is picked out by them, so this range does not carry a 95% guarantee here: it describes the spread.';
  const ciKind = x => (x && x.ci_kind) || 'fixed-time, descriptive';

  // ---- what a verdict rests on: the test (method) and whether its checks hold here (evidence) ------------
  const ASSUME = { free: 'no distribution assumption; conversations independent', supported: 'diagnostics passed; distribution assumptions unverified',
    unsupported: 'outside calibrated range', unknown: 'assumptions unchecked', incomplete: 'checks incomplete' };

  // Which test made a pairing's official verdict: {free, name, model}. `free`: an anytime-valid test for bounded
  // outcomes (a betting confidence sequence), valid at every check if conversations are independent, with no
  // assumption about the shape of the results. `model`: the older model-based gate's verdict, which is secondary.
  const FREE_RE = /betting|hedg|confidence.?seq|assumption.?free|anytime|bounded|e-?value|martingale/i;
  function testOf(p, design) {
    const pick = o => o && [o.official_test, o.test, o.verdict_test, o.method_official].find(x => x);
    const m = pick(p) || pick(design) || (p && p.verdict_method) || (design && design.verdict_method);
    const name = !m ? '' : typeof m === 'string' ? m : m.name || m.kind || m.method || '';
    const explicit = [m && typeof m === 'object' ? m.assumption_free : undefined, p && p.assumption_free, design && design.assumption_free]
      .find(x => typeof x === 'boolean');
    const free = explicit ?? FREE_RE.test(name);
    const mb = p && (p.model_based ?? p.model_based_verdict ?? p.secondary_verdict);
    const model = mb == null ? null : typeof mb === 'string' ? { verdict: mb } : mb;
    const bound = (m && typeof m === 'object' && (m.bound_bb100 ?? m.bound)) ?? (design && (design.outcome_bound_bb100 ?? design.outcome_bound)) ?? null;
    return { free, name, model, bound };
  }
  const words = k => String(k).replace(/_/g, ' ');
  const FAILED = /^(unsupported|fail|failed|violated|warning|warn)$/i, OPEN = /^(unknown|incomplete|untested|missing)$/i;
  // The method and the evidence, apart: {method: 'free'|'model', status, label, detail, checks, failed, open}.
  // `status` is 'free' only when the assumption-free test's checks all hold (or none were recorded); a failed check
  // makes it 'unsupported' and an unmade one 'incomplete', whatever the method. For the model-based gate it is
  // the adjudicator's own status, else the conversation share against the share the test was calibrated for.
  function assumptionsOf(p, design) {
    const method = testOf(p, design).free ? 'free' : 'model';
    const a = p && p.assumptions;
    const checks = a && typeof a === 'object' && a.checks && typeof a.checks === 'object'
      ? Object.entries(a.checks).map(([k, v]) => v && typeof v === 'object' ? { name: words(k), status: String(v.status || ''), value: v.value ?? null, limit: v.limit ?? null, meaning: v.meaning || '' }
        : { name: words(k), status: String(v), value: null, limit: null, meaning: '' }) : [];
    const failed = checks.filter(c => FAILED.test(c.status)).map(c => c.name);
    const open = checks.filter(c => OPEN.test(c.status)).map(c => c.name);
    let detail = checks.map(c => `${c.name}: ${c.status}${c.value != null ? ` ${c.value}` : ''}${c.limit != null ? ` (${method === 'free' ? 'limit' : 'calibrated up to'} ${c.limit})` : ''}`).join('; ');
    let status = a ? (typeof a === 'string' ? a : a.status) : null;
    if (method === 'free') {
      status = failed.length || (status && FAILED.test(status)) ? 'unsupported' : open.length || (status && OPEN.test(status)) ? 'incomplete' : 'free';
      const label = status === 'unsupported' ? `check failed: ${failed.join(', ') || 'see the checks'}` : status === 'incomplete' ? `checks incomplete: ${open.join(', ') || 'some not made'}` : ASSUME.free;
      return { method, status, label, detail: detail || 'the official test assumes only that conversations (match pairs) are independent', checks, failed, open };
    }
    if (!status) {
      const share = p && p.conversation_share;
      const cal = design && design.calibration && design.calibration.conversation_effect_share;
      if (share != null && cal != null) {
        status = share > cal ? 'unsupported' : 'supported';
        detail = `conversations differ by ${Math.round(share * 100)}% of the spread; the test is calibrated up to ${Math.round(cal * 100)}%`;
      } else status = 'unknown';
    }
    if (!ASSUME[status] || status === 'free') status = 'unknown';
    return { method, status, label: ASSUME[status], detail, checks, failed, open };
  }

  // {reasons} when a pairing may not be decided (identity, audit, protocol, registration...), else null.
  function ineligible(p) {
    if (!p || (p.verdict !== 'ineligible' && p.eligible !== false)) return null;
    return { reasons: (p.ineligible_reasons || []).map(String) };
  }

  // Where a verdict comes from: {kind: 'official'|'retrospective'|'recorded', text} or null.
  function sourceOf(p, name = x => x) {
    if (!p) return null;
    if (p.track === 'adaptation' && p.recorded) return { kind: 'recorded', short: 'model-based · not official', text: `recorded when play stopped${p.judge ? ` by judge ${String(p.judge).slice(0, 8)}` : ''}, by the adaptation track's model-based test: never official` };
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
    const test = testOf(p, p.design_ref);
    const mv = test.model && test.model.verdict;
    const secondary = !test.free || !mv ? null
      : { verdict: mv, short: mv === 'a' ? `${name(p.a)} wins` : mv === 'b' ? `${name(p.b)} wins` : mv === 'even' ? 'Even'
        : mv === 'exhausted' ? 'No winner' : mv === 'early' ? 'Too early' : 'Undecided' };
    const sec = secondary ? ` The model-based test (secondary, never decides) says: ${secondary.short.toLowerCase()}.` : '';
    const out = as.method === 'free' && as.status === 'free'
      ? ` The test makes no assumption about the shape of the results: its error rate holds at every check if conversations are independent and each one's expected edge, given the ones before, is at most the tested value.${sec}`
      : as.method === 'free' && as.status === 'unsupported'
      ? ` The test makes no assumption about the shape of the results, but a check it needs failed here (${as.failed.join(', ') || 'see the checks'}), so its error rate is not established.${sec}`
      : as.method === 'free' && as.status === 'incomplete'
      ? ` The test makes no assumption about the shape of the results, but some checks it needs could not be made (${as.open.join(', ') || 'see the checks'}).${sec}`
      : as.status === 'unsupported'
      ? ' The test\'s assumptions do not hold here (outside its calibrated range), so its error rate is not established.'
      : as.status === 'supported' ? ' Its error rate holds under the test\'s assumptions: the diagnostics passed, but the distribution assumptions remain unverified.'
      : ' Its error rate holds only under the test\'s assumptions, which are not checked here.';
    const qualifier = as.status === 'unsupported' ? (as.method === 'free' ? 'check failed' : ASSUME.unsupported) : '';
    if (p.track === 'adaptation' && DECIDED.includes(p.verdict)) {
      const what = p.verdict === 'even' ? `within ±${band} bb/100 of each other` : `${name(sides.winner)} ahead of ${name(sides.loser)}`;
      const diag = as.status === 'unsupported' ? ' Its diagnostics failed: the results look outside the family of shapes it was calibrated for.'
        : as.status === 'supported' ? ' Its diagnostics passed, but its error rate still rests on those assumptions.' : '';
      return { tone: 'model', decided: false, model: true, qualifier, short: p.verdict === 'even' ? 'Leans even · model-based' : `${name(sides.winner)} leans ahead · model-based`,
        long: `Exploratory${looks}, not official: the model-based test reads this as ${what}. On the adaptation track that test (a t and sign-flip test calibrated for a family of result shapes) stops play, but its error rate is not established for these results, so this is a lead to look into, not a finding; it never decides the official standings, ranks or leaders.${diag}` };
    }
    if (driftQualified(p)) {
      const sides = verdictSides(p), dw = driftWords(p);
      const num = x => x == null ? '' : `${x > 0 ? '+' : x < 0 ? MINUS : ''}${Math.abs(Math.round(x))}`;
      const nums = dw.early != null && dw.late != null ? ` (${num(dw.early)} early, ${num(dw.late)} late, bb/100 for ${name(sides.winner)})` : '';
      return { tone: 'drift', decided: false, drift: true, secondary, short: `${name(sides ? sides.winner : ahead)}: ${dw.phrase}`,
        long: `Not decided: the sequential test's bar was cleared${looks}, but the edge drifted between early and late conversations${nums}. That says only that ${name(sides ? sides.winner : ahead)} was ahead for part of the run, not that it is better: it needs confirmation under a stable model snapshot, and it makes no leader.` };
    }
    if (idWithheld(p) && DECIDED.includes(p.verdict)) {
      const sides = verdictSides(p);
      return { tone: 'open', decided: false, identity: true, secondary, short: sides ? `${name(sides.winner)} ahead · model unconfirmed` : 'Even · model unconfirmed',
        long: `Not counted: the test's bar was cleared${looks}, but a player's tool never reported which model answered, and this run's policy counts only confirmed models.${typeof p.qualifier === 'string' ? ` (${p.qualifier})` : ''}` };
    }
    switch (p.verdict) {
      case 'a': case 'b':
        return { tone: 'won', decided: true, qualifier, secondary, short: `${name(sides.winner)} wins`, long: `Decided${looks}: ${name(sides.winner)} beats ${name(sides.loser)}; the edge cleared the sequential test's bar.${out} ${STABLE_NOTE}` };
      case 'even':
        return { tone: 'even', decided: true, qualifier, secondary, short: 'Even', long: `Decided${looks}: within ±${band} bb/100 of each other.${out}` };
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
        return { tone: 'open', decided: false, secondary, short: 'Undecided', long: `Not decided yet${looks}.${level ? '' : ` ${name(ahead)} is ahead on points.`}${secondary && test.free ? ` The model-based test (secondary, never decides) says: ${secondary.short.toLowerCase()}.` : ''}` };
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

  // ---- standings order: decided ranks first, the unresolved grouped by the places they share ---------
  // [{kind: 'decided', rank, players}, {kind: 'unresolved', range: [lo, hi] | null, players}] or, when nothing is
  // decided, [{kind: 'points', players}]. Inside an unresolved block (and in 'points') players go by strength.
  const byStrength = ps => [...ps].sort((x, y) => (y.strength ?? -1e9) - (x.strength ?? -1e9));
  function standingBlocks(ps) {
    const decided = ps.filter(p => p.rank_kind === 'decided' && p.rank != null).sort((x, y) => x.rank - y.rank);
    if (!decided.length) return [{ kind: 'points', players: byStrength(ps) }];
    const blocks = [];
    for (const p of decided) {
      const last = blocks[blocks.length - 1];
      if (last && last.rank === p.rank) last.players.push(p);
      else blocks.push({ kind: 'decided', rank: p.rank, players: [p] });
    }
    const open = new Map();
    for (const p of ps.filter(x => !decided.includes(x))) {
      const r = Array.isArray(p.rank_range) && p.rank_range.length === 2 && p.rank_range[0] != null ? p.rank_range : null;
      const k = r ? r.join('-') : 'open';
      if (!open.has(k)) open.set(k, { kind: 'unresolved', range: r, players: [] });
      open.get(k).players.push(p);
    }
    const rest = [...open.values()].sort((x, y) => (x.range ? x.range[0] : 1e9) - (y.range ? y.range[0] : 1e9));
    rest.forEach(b => { b.players = byStrength(b.players); });
    return blocks.concat(rest);
  }

  // ---- cohorts: the groups of players who met, named by who is in them and how they played ------------
  // [{key, kind: 'models'|'mixed'|'bots'|'history', group, members, models, deals, protocol}], the current model
  // cohort first: model groups before mixed before bots, more models and more deals first; players left out of
  // the fit (earlier variants with an unknown model or protocol) last.
  function protocolOf(ids, label) {
    const hands = new Set(), notes = new Set();
    for (const id of ids) {
      const m = String(id).match(/~h(\d+)/);
      if (m) hands.add(Number(m[1]));
      const d = nameParts(label(id))[2];
      if (d) d.split(/,\s*/).forEach(x => { if (!/hands per conversation/.test(x)) notes.add(x); else { const k = x.match(/(\d+)/); if (k) hands.add(Number(k[1])); } });
    }
    const bits = [];
    if (hands.size === 1) bits.push(`${[...hands][0]}-hand conversations`);
    else if (hands.size > 1) bits.push(`${[...hands].sort((a, b) => a - b).join(' / ')}-hand conversations`);
    notes.forEach(n => bits.push(n));
    return bits.join(' · ');
  }
  function cohorts(rt, { label = x => x, kind = () => 'model' } = {}) {
    const out = rt.groups.filter(g => g.members.length).map(g => {
      const models = g.members.filter(p => kind(p) !== 'bot');
      const k = !models.length ? 'bots' : models.length === g.members.length ? 'models' : 'mixed';
      return { key: `g${g.index}`, kind: k, group: g, members: g.members, models,
        deals: g.pairings.reduce((t, x) => t + (x.deals || 0), 0), protocol: protocolOf(g.members, label) };
    });
    const order = { models: 0, mixed: 1, bots: 2 };
    out.sort((x, y) => order[x.kind] - order[y.kind] || y.models.length - x.models.length || y.deals - x.deals);
    const left = rt.players.filter(p => !p.in_fit).map(p => p.player);
    if (left.length) out.push({ key: 'earlier', kind: 'history', group: null, members: left, models: left.filter(p => kind(p) !== 'bot'), deals: 0, protocol: protocolOf(left, label) });
    return out;
  }
  // {title, sub}: a cohort in words, e.g. {title: 'Claude Sonnet 5.5 · GPT-6 Luna · Gemini 4 Neon', sub: 'High reasoning · 50-hand conversations'}
  function cohortName(c, label = x => x) {
    const parts = c.members.map(p => nameParts(label(p)));
    const levels = [...new Set(c.models.map(p => nameParts(label(p))[1]).filter(Boolean))];
    const lv = levels.length === 1 ? `${levels[0].replace(/ effort$/i, '').replace(/^./, x => x.toUpperCase())} reasoning` : levels.length ? 'mixed reasoning levels' : '';
    if (c.kind === 'bots') return { title: 'Baseline bots', short: 'Baseline bots', sub: [...new Set(parts.map(x => x[0]))].join(' · ') };
    if (c.kind === 'history') return { title: 'Earlier variants', short: 'Earlier variants', sub: 'model or protocol not recorded, so not rated' };
    const names = [...new Set(c.models.map(p => nameParts(label(p))[0]))];
    const bots = c.members.length - c.models.length;
    const title = names.join(' · ') + (bots ? ` + ${bots === 1 ? '1 bot' : `${bots} bots`}` : '');
    const short = names.length > 2 ? `${names.length} models${bots ? ` + ${bots} bots` : ''}` : title;
    return { title, short, sub: [lv, c.protocol].filter(Boolean).join(' · ') };
  }

  // ---- which hands a figure covers: verified committed results, or provisional live play -------------
  // {committed, provisional, rejected, stats, verified} from the ratings when they say; numbers are null when unknown.
  function populationOf(rt) {
    const P = (rt && (rt.population || rt.populations)) || {};
    const stats = (rt && rt.stats_population) || P.stats || null;
    return { committed: P.committed ?? P.verified ?? null, provisional: P.provisional ?? P.live ?? null,
      rejected: Array.isArray(rt && rt.rejected) ? rt.rejected.length : (P.rejected ?? 0), verified: !!(P.committed != null || P.verified != null),
      stats: typeof stats === 'string' ? { kind: stats } : stats };
  }

  // ---- poker fundamentals: quiz.json in one shape -----------------------------------------------------
  // {suite, version, spots: [{id, title, kind, street, hole, board, pot, to_call, stack, bb, ..., best, ev}],
  //  players: [{player, score, correct, answered, total, ev_lost_bb, reference, answers: {spot: {...}}}]}
  const num = (...xs) => xs.find(x => x != null && isFinite(x)) ?? null;
  const isFallback = a => !!a && (a.fallback === true || a.valid === false || a.status === 'fallback');
  // A quiz player's provenance: {status: 'eligible'|'ineligible'|'unrecorded'|'bot', reasons, identity, facts}
  function provenanceOf(r) {
    if (r.kind === 'bot' || (r.spec && r.spec.bot)) return { status: 'bot', reasons: [], identity: null, facts: [] };
    const e = r.eligibility ?? r.eligible ?? (r.provenance && (r.provenance.eligibility ?? r.provenance.eligible));
    const P = r.provenance || {};
    const identity = P.identity_match || r.identity_match || null;
    const reasons = [].concat((e && typeof e === 'object' && e.reasons) || r.ineligible_reasons || P.reasons || []).map(String);
    let status = e == null ? null : typeof e === 'boolean' ? (e ? 'eligible' : 'ineligible') : typeof e === 'string' ? e : e.status || (e.eligible === false ? 'ineligible' : e.eligible ? 'eligible' : null);
    if (status && !['eligible', 'ineligible'].includes(status)) status = /^(ok|pass|valid|eligible)/i.test(status) ? 'eligible' : 'ineligible';
    if (!status) status = 'unrecorded';
    const facts = [['protocol', P.protocol || r.protocol], ['implementation', P.implementation || r.implementation], ['CLI', P.cli_version || r.cli_version],
      ['audit', P.audit || r.audit_status]].filter(([, v]) => v).map(([k, v]) => `${k} ${typeof v === 'object' ? (v.version || v.status || v.hash || JSON.stringify(v)) : v}`);
    return { status, reasons, identity, facts };
  }
  // The headline is one category's (the one the quiz scores: `scored_category`, else "basics" when there are
  // others), every figure of it from that category's own answers; every other category keeps its own count.
  // The question exactly as a model received it: the spot's stored prompt, else the first prompt recorded in an
  // answer's call log, else the server's reconstruction from today's quiz code. {text, source, ...} or null.
  function spotPrompt(q, s) {
    if (s.prompt) return { text: s.prompt, source: 'stored' };
    const seen = [];
    for (const x of q.players || []) {
      const a = (x.answers || {})[s.id];
      const c = a && (a.calls || [])[0];
      if (c && c.prompt) seen.push(c.prompt);
    }
    if (seen.length) return { text: seen[0], source: 'recorded', answers: seen.length, differs: new Set(seen).size > 1 };
    if (s.question) return { text: s.question, source: 'reconstructed', version: s.question_version || null };
    return null;
  }
  // A hand in the usual short form: ['Qd', '9c'] -> 'Q9o', ['As', 'Ks'] -> 'AKs', ['Td', 'Tc'] -> 'TT'.
  const RANKS = '23456789TJQKA';
  function handCode(hole) {
    if (!hole || hole.length !== 2 || hole.some(c => !c || c.length < 2)) return '';
    const [x, y] = [...hole].sort((a, b) => RANKS.indexOf(b[0]) - RANKS.indexOf(a[0]));
    return x[0] === y[0] ? x[0] + y[0] : x[0] + y[0] + (x[1] === y[1] ? 's' : 'o');
  }
  // A spot's title: its own, or for a generated spot (whose title is its id) one made from the hand and the situation.
  function spotTitle(s) {
    if (s.title && s.title !== s.id) return s.title;
    const hand = handCode(s.hole) || 'A hand', at = `${hand} on the ${s.street || 'river'}`;
    const sit = s.situation || '';
    if (s.kind === 'sizing') return `${at}: check, or how much to bet`;
    if (s.kind === 'multistreet') return `${at}: ${sit || 'facing a bet'}, river plan known`;
    if (s.kind === 'inference') return `${at}: ${sit || 'facing a bet'}, range from its line`;
    return sit ? `${at}: ${sit}` : at;
  }

  function quizOf(raw) {
    if (!raw || typeof raw !== 'object') return null;
    const spots = (raw.spots || raw.items || raw.questions || []).map((x, i) => ({ ...x, id: String(x.id ?? x.spot ?? i),
      best: x.best ?? x.correct ?? x.answer ?? null, bb: x.bb ?? raw.bb ?? 1 }));
    const refs = new Set([].concat(raw.reference || raw.references || []).map(String));
    const list = Array.isArray(raw.players) ? raw.players : Object.entries(raw.players || raw.results || {}).map(([player, r]) => ({ player, ...r }));
    const catOf = s => s.category || s.kind || 'other';
    const spotById = Object.fromEntries(spots.map(s => [s.id, s]));
    const categories = [...new Set(spots.map(catOf))];
    const spotsIn = c => spots.filter(s => catOf(s) === c).length;
    const lostOf = a => num(a.ev_lost_bb, a.ev_loss_bb, a.regret_bb) || 0;
    const players = list.map(r => {
      const raw_ = Array.isArray(r.answers) ? Object.fromEntries(r.answers.map(a => [String(a.spot ?? a.id), a])) : r.answers || {};
      // a fallback (no valid answer after every retry) is never correct, whatever the record says
      const answers = Object.fromEntries(Object.entries(raw_).map(([k, a]) => [k, isFallback(a) ? { ...a, fallback: true, correct: false, credited: !!a.correct } : a]));
      const vals = Object.entries(answers);
      const catKey = (k, a) => spotById[k] ? catOf(spotById[k]) : a.category || 'other';
      const cats = {};
      for (const [k, a] of vals) {
        const c = cats[catKey(k, a)] = cats[catKey(k, a)] || { correct: 0, total: 0, valid: 0, fallbacks: 0, ev_lost_bb: 0 };
        c.total++;
        c.ev_lost_bb += lostOf(a);
        if (isFallback(a)) c.fallbacks++;
        else if (a.outcome === 'unverified' || a.verified === false) c.unverified = (c.unverified || 0) + 1;
        else { c.valid++; if (a.correct) c.correct++; }
      }
      for (const [c, k] of Object.entries(cats)) { k.answered = k.total; k.total = Math.max(k.total, spotsIn(c)); k.ev_lost_bb = Math.round(k.ev_lost_bb * 1000) / 1000; }
      const given = r.by_category || r.categories;
      const by_category = given && typeof given === 'object' && !Array.isArray(given) && !vals.length ? given : cats;
      const scored = r.scored_category ?? raw.scored_category ?? (raw.protocol && raw.protocol.scored_category)
        ?? (categories.length > 1 && categories.includes('basics') ? 'basics' : null);
      const head = scored && by_category[scored] ? by_category[scored] : null;
      let correct, answered, total, valid, fallbacks, lost;
      if (head) {
        correct = num(head.correct, head.valid_correct);
        answered = num(head.answered, head.total);
        total = num(head.total, spotsIn(scored), answered);
        fallbacks = num(head.fallbacks) ?? 0;
        valid = num(head.valid, head.valid_answers, answered != null ? answered - fallbacks : null);
        lost = num(head.ev_lost_bb, head.ev_loss_bb);
      } else {
        fallbacks = vals.length ? vals.filter(([, a]) => isFallback(a)).length : num(r.fallbacks) ?? 0;
        correct = vals.length ? vals.filter(([, a]) => a.correct && !isFallback(a)).length : num(r.correct);
        answered = num(r.answered, vals.length || null);
        total = num(r.total, spots.length || answered);
        valid = vals.length ? vals.length - fallbacks : num(r.valid_answers, r.valid);
        lost = num(r.ev_lost_bb, r.ev_loss_bb, r.regret_bb, vals.length ? vals.reduce((t, [, a]) => t + lostOf(a), 0) : null);
      }
      const rest = Object.entries(by_category).filter(([c]) => head && c !== scored).map(([, k]) => k);
      const others = rest.length ? { correct: rest.reduce((t, k) => t + (num(k.correct) || 0), 0), total: rest.reduce((t, k) => t + (num(k.total) || 0), 0),
        fallbacks: rest.reduce((t, k) => t + (num(k.fallbacks) || 0), 0), categories: rest.length } : null;
      return { ...r, answers, correct, answered, total, valid, fallbacks, ev_lost_bb: lost, by_category, scored_category: head ? scored : null, others,
        valid_accuracy: valid ? (correct != null ? correct / valid : null) : head || vals.length ? null : num(r.valid_accuracy),
        score: correct != null && total ? correct / total : num(r.score),
        provenance: provenanceOf(r),
        reference: !!(r.reference || refs.has(String(r.player)) || r.kind === 'reference') };
    }).sort((x, y) => (y.score ?? -1) - (x.score ?? -1) || (x.ev_lost_bb ?? 1e9) - (y.ev_lost_bb ?? 1e9));
    const scored = players.map(x => x.scored_category).find(Boolean) || null;
    return { ...raw, suite: raw.suite_name || (typeof raw.suite === 'string' ? raw.suite : raw.suite && raw.suite.name) || 'Poker fundamentals',
      version: raw.version ?? (raw.suite && raw.suite.version) ?? null, spots, players, categories, scored_category: scored,
      category_info: raw.categories && typeof raw.categories === 'object' && !Array.isArray(raw.categories) ? raw.categories : {} };
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

  // ---- a pairing's deals in playing order: what the replay's arrows, auto-next and swings walk ---------
  // From /api/pairing: [{id, round, replicate, conv, deal, net, status, reach}], conversation by conversation
  // (round, then replicate), deal by deal. `played` ({id: [deal]}) adds deals a conversation's events show
  // that the pairing's records don't have yet (a deal being played right now).
  function dealSequence(pr, played = {}) {
    const out = [];
    (pr && pr.conversations || []).forEach((c, conv) => {
      const have = new Map((c.deals || []).map(([deal, net, status, reach]) => [deal, { net, status, reach }]));
      for (const k of played[c.id] || []) if (!have.has(k)) have.set(k, { net: null, status: 'playing', reach: null });
      [...have.keys()].sort((x, y) => x - y).forEach(deal => out.push({ id: c.id, round: c.round, replicate: c.replicate, conv, deal,
        deals_total: c.deals_total, ...have.get(deal) }));
    });
    return out;
  }
  const counted = x => x.status === 'verified' || x.status === 'unchecked';
  // The deals with the biggest results across the pairing, counted ones only: [entry], largest first.
  function sequenceSwings(seq, n = 5, min = 0) {
    return seq.filter(x => counted(x) && x.net && Math.abs(x.net) >= min)
      .sort((x, y) => Math.abs(y.net) - Math.abs(x.net) || x.conv - y.conv || x.deal - y.deal).slice(0, n);
  }
  // The scoreboard's numbers from the sequence alone: counted deals only, and what is left out.
  function sequenceTotals(seq) {
    const c = seq.filter(x => counted(x) && x.net != null);
    const net = c.reduce((t, x) => t + x.net, 0);
    return { counted: c.length, net: Math.round(net * 100) / 100, won: c.filter(x => x.net > 0).length, lost: c.filter(x => x.net < 0).length,
      even: c.filter(x => x.net === 0).length, provisional: seq.filter(x => x.status === 'provisional').length,
      playing: seq.filter(x => x.status === 'playing').length, rejected: seq.filter(x => x.status === 'rejected').length,
      checked: !c.length || c.some(x => x.status === 'verified') };
  }

  return { STABLE_NOTE, dealSequence, sequenceSwings, sequenceTotals, standingBlocks, cohorts, cohortName, protocolOf, populationOf, testOf, isFallback, provenanceOf, quizOf, firstDifference, bigSwings, IDENTITY, esc, MINUS, signed, plural, trim, approx, nameParts, family, isContract, normalize, verdict, verdictSides, powerLine, correction, actText,
    isDecided, decidedLeader, driftQualified, countsForLeader, driftWords, officialInterval, ciKind, DESCRIPTIVE_TIP, spotPrompt, spotTitle, handCode, playerId, rateOf, targetEdge, isUnregistered, scenarios, conversationsShort,
    assumptionsOf, ineligible, sourceOf, basisOf, fitOf, identityOf, ASSUME };
});
