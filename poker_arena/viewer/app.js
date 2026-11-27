// Poker Arena viewer: which models win and by how much, and every duplicate deal replayed at the table.
// Works against the live server (/api/...) or a static export (data/*.json).
'use strict';

const { esc, signed, plural, trim, approx, nameParts, family, normalize, verdict, verdictSides, powerLine, correction } = window.Core;
const STATIC = window.POKER_STATIC || null;
const $app = document.getElementById('app');
const $status = document.getElementById('status');
const $run = document.getElementById('run');
const store = {
  get(k, d) { try { const v = localStorage.getItem('poker-arena.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('poker-arena.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};
const state = { runs: [], run: null, index: [], kinds: {}, labels: {}, timer: null, mind: store.get('mind', false),
  matches: new Map(), view: null, replay: null, page: null, checked: null, showPlay: 0, carry: false,
  autoNext: store.get('autonext', true), mode: store.get('timeline', 'aligned'), listOpen: false, mapOpen: null, metric: store.get('metric', 'vpip') };

const LIVE_POLL = 6000, SLOW_POLL = 60000;

// ---- formatting ---------------------------------------------------------------------------------
const cls = x => x > 0 ? 'pos' : x < 0 ? 'neg' : '';
const pct = x => x == null ? '<span class="dim">–</span>' : `${Math.round(x * 100)}%`;
const bbs = (chips, bb) => `${trim(chips / bb)} bb`;
const signedBB = (chips, bb) => chips === 0 ? '0 bb' : `${chips > 0 ? '+' : '−'}${trim(Math.abs(chips) / bb)} bb`;
function duration(s) {
  if (s < 60) return 'under a minute';
  const m = Math.round(s / 60);
  if (m < 60) return plural(m, 'minute');
  const h = Math.floor(m / 60), r = m % 60;
  return r ? `${h} h ${r} min` : plural(h, 'hour');
}
function ago(t) {
  if (!t) return '';
  const s = Date.now() / 1000 - t;
  if (s < 90) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400 * 2) return `${Math.round(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
const clock = d => d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
const runLabel = r => (state.runs.find(x => x.run === r)?.title) || r.split('/').pop();
const runLink = (run, rest = '') => `#/r/${encodeURIComponent(run)}${rest}`;
const matchLink = (id, hand) => runLink(state.run, '/m/' + encodeURIComponent(id) + (hand != null ? `?deal=${hand + 1}` : ''));

// ---- data ---------------------------------------------------------------------------------------
// A static export names each answer by the same key the server's export uses.
function fileKey(s) {
  return Array.from(new TextEncoder().encode(s), b => /[A-Za-z0-9_-]/.test(String.fromCharCode(b))
    ? String.fromCharCode(b) : '~' + b.toString(16).toUpperCase().padStart(2, '0')).join('');
}
async function api(name, param) {
  let url;
  if (STATIC) url = `data/${name}${param ? '-' + fileKey(param[1]) : ''}.json`;
  else url = `/api/${name}${param ? `?${param[0]}=${encodeURIComponent(param[1])}` : ''}`;
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) {
    let detail = '';
    try { detail = (await r.json()).error || ''; } catch (e) { /* not JSON */ }
    throw new Error(detail || `${url} answered ${r.status}`);
  }
  return r.json();
}
async function loadMatch(id) {
  const hit = state.matches.get(id);
  if (hit && hit.result) return hit;  // finished matches don't change
  const m = await api('match', ['id', id]);
  state.matches.set(id, m);
  return m;
}

// ---- names: model first, reasoning level beside it ----------------------------------------------
const isBot = p => state.kinds[p] === 'bot';
const labelText = p => state.labels[p] || p;
const COLORS = { claude: '#c8643f', gpt: '#0f8f6c', gemini: '#3b73d9', llama: '#6c52d9', grok: '#555', bot: '#5d655f' };
function monogram(p, size = '') {
  const [model] = nameParts(labelText(p));
  const words = model.replace(/\(.*?\)/g, '').split(/[\s_-]+/).filter(w => /^[A-Za-z]/.test(w) && !/^bot$/i.test(w));
  const initials = words.length > 1 ? (words[0][0] + words[1][0]).toUpperCase() : (words[0] || model).slice(0, 2).replace(/^./, ch => ch.toUpperCase());
  const fam = family(labelText(p), state.kinds[p]);
  let color = COLORS[fam];
  if (!color) { let h = 0; for (const ch of p) h = (h * 31 + ch.charCodeAt(0)) % 360; color = `hsl(${h} 45% 42%)`; }
  return `<span class="mono ${size}" style="--c:${color}" aria-hidden="true">${esc(initials)}</span>`;
}
const effortText = e => e.replace(/ effort$/, '').replace(/^./, c => c.toUpperCase());
const effortChip = e => e ? `<span class="effort">${esc(effortText(e))}</span>` : '';
function name(p, { effort = true } = {}) {
  const [model, level] = nameParts(labelText(p));
  return `<span class="nm" title="${esc(labelText(p))}">${esc(model)}</span>${effort ? effortChip(level) : ''}${isBot(p) ? '<span class="tag">bot</span>' : ''}`;
}
const plain = p => esc(labelText(p));
// a label's parenthetical (protocol details) as a quiet second line, never inside the level chip
const detail = p => { const d = nameParts(labelText(p))[2]; return d ? `<div class="pdetail" title="${esc(d)}">${esc(d)}</div>` : ''; };
const player = (p, extra = '', { details = true } = {}) => `<div class="player">${monogram(p)}<div class="pname"><div class="name">${name(p)}</div>${details ? detail(p) : ''}${extra}</div></div>`;
const SUITS = ['♠', '♥', '♦', '♣'];
let sectionCount = 0;
const section = (title, aside, body, caption = '', opts = {}) => {
  const suit = SUITS[sectionCount++ % 4];
  return `<section ${opts.id ? `id="${opts.id}"` : ''}><div class="section-head"><h2><span class="suit ${suit === '♥' || suit === '♦' ? 'red' : ''}">${suit}</span>${title}</h2>
  ${aside ? `<span class="aside">${aside}</span>` : ''}</div><div class="card-box ${opts.scroll ? 'scroll' : ''}">${body}</div>
  ${caption ? `<p class="caption">${caption}</p>` : ''}</section>`;
};
const th = (text, tip, extra = '') => `<th class="${extra}" ${tip ? `title="${esc(tip)}"` : ''}>${text}</th>`;

// ---- routing: #/r/RUN (standings), #/r/RUN/m/MATCHUP?deal=K (games) ------------------------------
async function route(opts = {}) {
  clearTimeout(state.timer);
  const soft = !!opts.soft;
  if (!soft) stopPlay();
  const [path, query] = location.hash.replace(/^#\/?/, '').split('?');
  const parts = path.split('/').filter(Boolean);
  const page = parts.slice(0, 4).join('/');
  const fresh = page !== state.page;
  state.page = page;
  if (fresh && !soft) $app.classList.add('busy');
  let cur;
  try {
    state.runs = await api('runs');
    state.checked = new Date();
    if (!state.runs.length) {
      $run.parentElement.hidden = true;
      $app.innerHTML = `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>No games yet</h2>
        <p>Play a match or a tournament, and it shows up here as it plays:</p>
        <p><code>python -m poker_arena play rule station --deals 20 --out runs/demo</code></p></div>`;
      return;
    }
    $run.parentElement.hidden = false;
    const run = parts[0] === 'r' && parts[1] ? decodeURIComponent(parts[1]) : null;
    if (!run || !state.runs.find(r => r.run === run)) {
      location.replace(runLink((state.runs.find(r => !r.all) || state.runs[0]).run));
      return;
    }
    state.run = run;
    const option = r => `<option value="${esc(r.run)}" ${r.run === run ? 'selected' : ''}>
      ${esc(runLabel(r.run))}${r.live && !r.all ? ' · live' : ''}</option>`;
    const all = state.runs.filter(r => r.all), each = state.runs.filter(r => !r.all);
    $run.innerHTML = all.map(option).join('') + (all.length ? `<optgroup label="Runs">${each.map(option).join('')}</optgroup>` : each.map(option).join(''));
    $run.onchange = () => { location.hash = runLink($run.value); };
    const tab = parts[2] === 'm' ? 'matches' : 'board';
    document.querySelectorAll('.top nav a').forEach(a => {
      a.classList.toggle('active', a.dataset.tab === tab);
      a.href = runLink(run, a.dataset.tab === 'matches' ? '/m' : '');
    });
    state.index = await api('index', ['run', run]);
    state.labels = {}; state.kinds = {};   // per run: an id can name another model, or level, in another run
    for (const m of state.index) { Object.assign(state.kinds, m.kinds || {}); Object.assign(state.labels, m.labels || {}); }
    sectionCount = 0;
    if (tab === 'matches') {
      await renderMatches(parts[3] ? decodeURIComponent(parts.slice(3).join('/')) : '', new URLSearchParams(query || ''), soft);
    } else {
      state.view = null;
      const y = window.scrollY;
      await renderBoard();
      if (soft) window.scrollTo(0, y);
    }
    cur = state.runs.find(r => r.run === run);
    if (fresh && !soft) window.scrollTo(0, 0);
  } catch (e) {
    if (!soft) $app.innerHTML = `<div class="card-box empty"><h2>Couldn't load this page</h2><p class="error">${esc(e.message)}</p></div>`;
  }
  $app.classList.remove('busy');
  renderStatus(cur);
  schedule(cur);
}
window.addEventListener('hashchange', () => route());

// Live runs refresh every few seconds; an unfinished run that has gone quiet is still checked once a
// minute, so a paused tournament picks up again on its own. A refresh never interrupts a replay.
function schedule(cur) {
  clearTimeout(state.timer);
  if (STATIC || !cur || cur.finished) return;
  const playing = state.index.some(m => m.live);
  state.timer = setTimeout(poll, playing ? LIVE_POLL : SLOW_POLL);
}
function poll() {
  if (state.replay?.playing) { state.timer = setTimeout(poll, 2000); return; }
  route({ soft: true });
}
function renderStatus(cur) {
  if (STATIC) { $status.innerHTML = `<span class="label">Snapshot · ${new Date(STATIC.exported * 1000).toLocaleDateString()}</span>`; return; }
  if (!cur) { $status.innerHTML = ''; return; }
  const playing = state.index.filter(m => m.live).length;
  const checked = state.checked ? `<span class="label checked" title="When this page last asked for news">checked ${clock(state.checked)}</span>` : '';
  if (playing) $status.innerHTML = `<span class="live-dot"></span><span>Live<span class="label"> · ${plural(playing, 'table')}</span></span>${checked}`;
  else if (cur.finished) $status.innerHTML = `<span class="label">Finished</span>`;
  else if (cur.status === 'stopped') $status.innerHTML = `<span class="idle-dot"></span><span>Stopped<span class="label"> · last action ${ago(cur.updated)}</span></span>${checked}`;
  else if (cur.between_rounds) $status.innerHTML = `<span class="idle-dot"></span><span>Between rounds<span class="label"> · last action ${ago(cur.updated)}</span></span>${checked}`;
  else $status.innerHTML = `<span class="idle-dot"></span><span>Paused<span class="label"> · last action ${ago(cur.updated)}</span></span>${checked}`;
}

// ---- standings ----------------------------------------------------------------------------------
function runHead(r, rt) {
  if (r.all) return historyHead(r, rt);
  const deals = Math.max(0, ...state.index.map(m => m.deals_total || 0));
  const pairings = new Set(state.index.map(m => [m.a, m.b].sort().join('|'))).size;
  const players = new Set([...state.index.flatMap(m => [m.a, m.b]), ...rt.roster.map(x => x.player)]).size;
  const rounds = rt.design?.round_deals ? (rt.design.max_rounds > 1 ? `up to ${rt.design.max_rounds} rounds of ${rt.design.round_deals} deals` : `${rt.design.round_deals} deals per matchup`)
    : state.index.some(m => m.round > 1) ? `rounds of ${deals} deals` : `${deals} deals each`;
  const sub = [plural(players, 'player'), plural(pairings, 'matchup'), rounds];
  const share = r.finished ? 1 : r.hands_planned ? r.hands / r.hands_planned : 0;
  let line;
  if (r.finished) line = `Finished · ${plural(r.hands, 'hand')} played · ${ago(r.updated)}`;
  else if (r.status === 'stopped') line = `Stopped · ${plural(r.hands, 'hand')} played · last action ${ago(r.updated)}`;
  else if (r.between_rounds) line = `${plural(r.hands, 'hand')} played · round ${r.round || 1} done, more rounds to come`;
  else {
    line = `${r.hands.toLocaleString()} of ${plural(r.hands_planned, 'hand')} so far`;
    if (r.eta_seconds != null) line += ` · about ${duration(r.eta_seconds)} left in this round`;
    else if (!r.live) line += ` · paused, last action ${ago(r.updated)}`;
  }
  return `<div class="page-head"><div><div class="eyebrow">${r.live ? '<span class="live-dot"></span> Live tournament' : 'Tournament'}</div>
    <h1>${esc(runLabel(state.run))}</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="meter"><span style="width:${(Math.min(1, share) * 100).toFixed(1)}%"></span></div>
      <div class="line">${line}${r.hands_per_hour ? ` · ${r.hands_per_hour.toLocaleString()} hands an hour` : ''}</div></div></div>`;
}

function historyHead(r, rt) {
  const models = rt.players.filter(p => !isBot(p.player)).length;
  const sub = [plural((rt.runs || []).length, 'run'), plural(models, 'model'), plural(rt.players.length - models, 'bot'),
    plural(2 * (rt.deals || 0), 'duplicate hand')];
  const notes = [];
  if ((rt.left_out || []).length) notes.push(`Left out, played under other game settings: ${rt.left_out.map(x => esc(runLabel(x.run))).join(', ')}.`);
  if ((rt.code_versions || []).length > 1) notes.push(`Played on ${rt.code_versions.length} versions of the arena.`);
  const gone = (rt.runs || []).filter(x => !state.runs.some(y => y.run === x.run)).length;
  if (gone) notes.push(`${plural(gone, 'run')} kept from the history after ${gone === 1 ? 'its folder was' : 'their folders were'} deleted.`);
  const self = (rt.runs || []).reduce((n, x) => n + (x.self_play || 0), 0);
  if (self) notes.push(`${plural(self, 'deal')} between two entries of the same model and level don't count.`);
  return `<div class="page-head"><div><div class="eyebrow">History</div><h1>All runs</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="line">One player per model and reasoning level, pooled across every run
      ${r.started_at ? `since ${new Date(r.started_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}.</div>
      ${notes.length ? `<div class="line soft small">${notes.join(' ')}</div>` : ''}</div></div>`;
}

function intro() {
  return `<div class="intro">
    <p class="lede"><b>AI models play heads-up No-Limit Hold'em against each other.</b> Every deal is played twice with
      the seats swapped, which reduces the luck of the cards.</p>
    <a class="cta" href="${runLink(state.run, '/m')}"><span class="cta-icon">▶</span>Watch the games</a></div>`;
}

const fmtStrength = v => v == null ? '–' : signed(v, 0);
const ciText = ci => ci ? `${signed(ci[0], 0)} to ${signed(ci[1], 0)}` : '';
const groupName = i => 'ABCDEFGH'[i] || String(i + 1);
const pid = x => Core.playerId(x);
// the interval's basis at the number: "within conversations" is flagged, it leaves out between-conversation spread
const basisTag = x => {
  const b = Core.basisOf(x);
  return b ? `<span class="basis ${b.key}" title="${esc(b.long)}">${esc(b.short)}</span>` : '';
};
const chip = (text, tone = '', tip = '') => `<span class="chip ${tone}" ${tip ? `title="${esc(tip)}"` : ''}>${text}</span>`;
const assumeChip = a => !a ? '' : chip(esc(a.label), a.status === 'unsupported' ? 'warn' : a.status === 'supported' ? 'ok' : 'quiet',
  a.status === 'unsupported' ? `Outside the range the test was calibrated for${a.detail ? ` (${a.detail})` : ''}: its error rate is not established.`
    : a.status === 'supported' ? `The test's assumptions hold here${a.detail ? ` (${a.detail})` : ''}.` : 'Whether the test\'s assumptions hold here is not checked.');
const voidReason = (d, reg = state.rt && state.rt.registration) => (reg && reg.status === 'void' && ((reg.reasons || []).join('; ') || 'the code changed after it was registered')) || d && (d.registration_void || (d.registration && (d.registration.status === 'void' || d.registration.void)
  && (d.registration.reason || 'the code changed after it was registered')) || (d.registered === 'void' && (d.void_reason || 'the code changed after it was registered')));

// What kind of run this is, in one chip: a registered test, an unregistered or void one, or pooled history.
function runKind(rt) {
  const d = rt.design || {};
  const v = voidReason(d);
  if (rt.pooled) return chip('pooled history', 'quiet', 'Pooled across runs, each stopped by its own rule: described, never decided.');
  if (v) return chip('registration void', 'warn', `The test plan no longer binds: ${v}. Nothing here is decided.`);
  if (rt.unregistered) return chip('unregistered', 'warn', 'Played before its test plan was recorded: everything here is ahead on points only, and sample sizes are retrospective.');
  if (d.registered || d.registration) return chip('registered plan', 'ok', 'The test plan was recorded before the first hand.');
  return '';
}

// The headline for one group of players: who leads and on what basis, in one compact row.
function outcomeRow(rt, g, multi, first = true) {
  const by = Object.fromEntries(rt.players.map(p => [p.player, p]));
  const n = g.ranked.length;
  const verdicts = g.pairings.map(p => verdict(p, plain));
  const decided = verdicts.filter(v => v.decided).length;
  const ineligible = verdicts.filter(v => v.tone === 'bad').length;
  const settled = !rt.pooled && !rt.unregistered && g.pairings.length && g.pairings.every(p => ['a', 'b', 'even', 'exhausted', 'ineligible'].includes(p.verdict));
  const who = g.leader || g.qualified_leader || (n > 1 ? g.front_runner : null);
  const p = who && by[who];
  let kicker, tone, why;
  if (g.qualified_leader) {
    kicker = 'Leader, outside the calibrated range';
    tone = 'qualified';
    why = 'It won each of its matchups under the planned test, but at least one of those wins is outside the range the test is calibrated for, so it is not counted as a decided leader.';
  } else if (g.leader) {
    kicker = '<span class="decided-badge">Decided</span>leader';
    tone = 'decided';
    why = `Won its matchup against ${g.members.length === 2 ? 'the other player' : `each of the other ${g.members.length - 1} players`} under the planned test${g.leader_assumption === 'supported' ? ', within the range the test is calibrated for' : '; whether the test\'s assumptions hold is not checked'}.`;
  } else if (rt.pooled || rt.unregistered || voidReason(rt.design)) {
    kicker = 'Ahead on points';
    tone = 'points';
    why = n < 2 ? 'Only one player in this group has results so far.' : '';
  } else {
    kicker = settled || rt.finished ? 'No decided leader' : 'No decided leader yet';
    tone = 'open';
    if (n < 2) why = 'Only one player in this group has results so far.';
    else if (g.pairings.every(p => p.verdict === 'early' || !p.deals)) why = rt.finished ? 'Too few deals were played to judge anyone.' : 'Too few deals so far to judge anyone.';
    else why = n > 1 && p ? 'The order below is on points; only the matchups marked Decided are results of the test.' : '';
  }
  const deals = Math.max(0, ...g.pairings.map(x => x.deals || 0));
  const facts = [first ? runKind(rt) : ''];
  if (!rt.pooled && !rt.unregistered && g.pairings.length) facts.push(chip(`${decided} of ${plural(g.pairings.length, 'matchup')} decided`, decided ? '' : 'quiet'));
  if (ineligible) facts.push(chip(`${ineligible} ineligible`, 'bad', 'Matchups the judge may not decide (identity, audit, protocol or registration): see Matchups.'));
  if (g.leader || g.qualified_leader) facts.push(assumeChip({ status: g.leader_assumption || 'unknown', label: Core.ASSUME[g.leader_assumption || 'unknown'], detail: '' }));
  if (deals) facts.push(chip(`${plural(deals, 'deal')}${g.pairings.length > 1 ? ' max' : ''} per matchup`, 'quiet'));
  if (g.misfits) facts.push(chip(`${g.misfits} matchup${g.misfits > 1 ? 's' : ''} off the strength fit`, 'warn', 'Strengths are one number per player; these matchups disagree with what the strengths predict. Read the matchup results directly.'));
  const card = g.leader && tone === 'decided' ? '<span class="mini-card" aria-hidden="true"><b>A</b><i>♠</i></span>'
    : '<span class="mini-card back" aria-hidden="true"></span>';
  const num = p ? `<div class="o-num"><b class="${g.leader ? '' : cls(p.strength)}">${fmtStrength(p.strength)}</b><small>bb/100</small>
      <div class="o-ci">${p.strength_ci95 ? `95% range ${ciText(p.strength_ci95)}` : 'no range yet'} ${basisTag(p)}</div></div>` : '';
  return `<div class="outcome-row ${tone}">
    ${card}
    <div class="o-main"><div class="o-kicker">${multi ? `<span class="o-group">Group ${groupName(g.index)}</span>` : ''}${kicker}</div>
      ${p ? `<div class="o-who">${monogram(who, 'lg')}<div class="o-name">${name(who)}</div></div>` : ''}</div>
    ${num}
    <div class="o-facts">${facts.filter(Boolean).join('')}</div>
    ${why ? `<p class="o-why">${why}</p>` : ''}</div>`;
}

function outcomes(rt) {
  const gs = rt.groups.filter(g => g.ranked.length);
  if (!gs.length) return '';
  const multi = rt.groups.filter(g => g.members.length).length > 1;
  return `<div class="outcomes">${gs.map((g, i) => outcomeRow(rt, g, multi, i === 0)).join('')}</div>`;
}

// The target edge: what a registered run was planned to detect, what that takes, and what it has; for an
// unregistered run the same sum, labelled retrospective.
function targetLine(rt, pairings) {
  const t = Core.targetEdge(rt.design, pairings);
  if (!t) return '';
  const d = rt.design || {};
  const have = t.have[0] === t.have[1] ? plural(t.have[0], 'deal') : `${t.have[0]}–${plural(t.have[1], 'deal')}`;
  const pct = Math.min(100, Math.round(100 * t.have[0] / Math.max(1, t.need)));
  const they = pairings.length > 1 ? 'the matchups have' : 'it has';
  let words;
  if (t.retrospective) words = `<b>Retrospective</b>, not a plan: telling a ${t.target} bb/100 edge apart (80% power) would take ~${approx(t.need)} deals per matchup; ${they} ${have}.`;
  else if (t.planned) words = `Planned to detect a ${t.target} bb/100 edge: ~${approx(t.need)} deals per matchup${t.allowed && t.allowed < t.need ? `, but the plan allows ${approx(t.allowed)}${t.power != null ? ` (power ${Math.round(t.power * 100)}%)` : ''}` : ''}; ${they} ${have}.`;
  else words = `To detect a ${t.target} bb/100 edge (80% power) takes ~${approx(t.need)} deals per matchup at the spread seen so far; ${they} ${have}.`;
  const extra = [];
  if (t.planned && d.underpowered) extra.push(`<span class="flag" title="The plan is capped by a deal budget below what its power target needs">underpowered</span> capped by a budget${d.budget_deals ? ` of ${approx(d.budget_deals)} deals` : ''}`);
  const sc = t.planned ? Core.scenarios(d) : [];
  if (sc.length) extra.push(`Power at the plan's size, by how much conversations differ: ${sc.map(x => `${esc(x.name)} ${x.power != null ? Math.round(x.power * 100) + '%' : '–'}`).join(' · ')}`);
  return `<div class="target"><div class="meter thin"><span style="width:${pct}%"></span></div><span>${words}${extra.length ? `<span class="target-more">${extra.join('<br>')}</span>` : ''}</span></div>`;
}

// Everything about how results are judged, once, folded away: the test, its assumptions, the sample size,
// the words used on the page, and the ratings' notes.
function planPanel(rt) {
  const d = rt.design || {};
  const open = store.get('plan-open', false);
  const v = voidReason(d);
  let test;
  if (rt.pooled) test = 'Pooled across runs, each stopped by its own rule, so nothing here is decided: every order and number is <b>ahead on points</b>.';
  else if (v) test = `This run's test plan is void (${esc(v)}), so nothing in it is decided: every order and verdict is <b>ahead on points</b>.`;
  else if (rt.unregistered) test = 'This run was played before its test plan was recorded, so nothing in it is decided: every order and verdict is <b>ahead on points</b>, and any sample size quoted is retrospective.';
  else if (d.round_deals) test = `Each matchup is checked after every ${d.round_deals} deals, at most ${plural(d.max_rounds, 'time')}${d.min_conversations ? `, never before ${plural(d.min_conversations, 'independent conversation')} (match pairs)` : ''}.
    A matchup is <b>decided</b> when its edge clears a bar set for at most a ${Math.round((d.alpha || 0.05) * 100)}% chance of any wrong verdict across all ${plural(d.n_pairings || rt.pairings.length, 'matchup')},
    <b>under the test's assumptions</b>: a verdict outside the range the test was calibrated for is marked so, and does not make a decided leader.`;
  else test = 'Decided results come only from a planned sequential test; everything else is ahead on points.';
  const target = rt.pooled ? '' : targetLine(rt, rt.pairings);
  const summary = rt.pooled ? 'Pooled across runs: described, never decided'
    : v ? 'Test plan void: described, never decided'
    : rt.unregistered ? 'No recorded test plan: described, never decided'
    : d.round_deals ? `Planned sequential test · ${Math.round((d.alpha || 0.05) * 100)}% error bound under stated assumptions` : 'How results are decided';
  const notes = (rt.notes || []).map(n => `<li>${esc(n)}</li>`).join('');
  return `<details class="plan-panel" ${open ? 'open' : ''}><summary><span class="pp-title">How this is judged</span><span class="pp-sum">${summary}</span></summary>
    <div class="pp-body">
      <p>${test}</p>
      ${target}
      <dl>
        <dt>Decided</dt><dd>A result of the planned test above. Only these carry the badge.</dd>
        <dt>Ahead on points</dt><dd>Everything else: the current estimate, recomputed as hands finish. It shows who leads, not who is better.</dd>
        <dt>Strength</dt><dd>Big blinds won per 100 hands against the average opponent in its group, from one least-squares fit over every deal
          (matchups with more deals weigh more); <b>0</b> is average. Where a matchup disagrees with what the strengths predict, the matchup's own result is the evidence.</dd>
        <dt>95% range</dt><dd><b>across conversations</b>: resamples whole match pairs; <b>within conversations</b>: resamples blocks of deals inside single match pairs,
          so it leaves out how much fresh matches differ, and the real uncertainty is larger.</dd>
        <dt>Group</dt><dd>Players linked by games. Players in different groups never met, so they are not compared.</dd>
        <dt>Identity</dt><dd><b>attested</b>: the model's own tool reported which model answered; <b>unattested</b>: only the configured name is known.</dd>
        <dt>Session Elo</dt><dd>A different number: how often it wins a 10-deal session. For reference, never used to rank.</dd>
      </dl>
      ${notes ? `<div class="pp-notes"><b>Notes from the ratings</b><ul>${notes}</ul></div>` : ''}
    </div></details>`;
}

// The run's audit in one line, with a link to the table.
function auditLine(rt) {
  const a = rt.audit;
  if (!a || !a.totals || !a.totals.calls) return '';
  const t = a.totals, id = t.identity || {};
  const idn = (id.attested || 0) + (id.unattested || 0) + (id.mismatch || 0);
  const bits = [`${t.decisions.toLocaleString()} model decisions`];
  bits.push(a.recorded ? `${t.tool_rejected} tool-use rejections · ${t.unverified} unaudited` : 'turn audit not recorded');
  bits.push(`${plural(t.fallbacks, 'fallback')}`);
  if (idn) bits.push(`identity attested on ${Math.round(100 * (id.attested || 0) / idn)}% of calls${id.mismatch ? ` · <b class="neg">${id.mismatch} mismatched</b>` : ''}`);
  return `<a class="audit-line" href="#audit"><span class="al-k">Audit</span><span>${bits.join(' · ')}</span><span class="al-go">details ›</span></a>`;
}

const fmtN = n => n ? n.toLocaleString() : '<span class="dim">0</span>';
function auditSection(rt) {
  const a = rt.audit;
  if (!a || !a.players || !Object.keys(a.players).length) return '';
  const rows = Object.entries(a.players).filter(([, x]) => x.calls || x.decisions).map(([p, x]) => {
    const id = x.identity || {};
    const idn = (id.attested || 0) + (id.unattested || 0) + (id.mismatch || 0);
    const status = Core.identityOf(null, x);
    const idCell = idn ? `<span class="idst ${status === 'attested' ? 'ok' : status === 'mismatch' ? 'bad' : 'warn'}">${esc(status || '–')}</span>
      <small class="nof">${id.attested || 0}/${idn} calls${x.observed && x.observed.length ? ` · saw ${esc(x.observed.join(', '))}` : ''}</small>` : '<span class="dim">–</span>';
    const audited = x.verified + x.opaque + x.unverified;
    const auditCell = !audited && !x.tool_rejected ? '<span class="dim" title="Calls from before turns were audited">not recorded</span>'
      : `${x.verified}<small class="nof">verified${x.opaque ? ` · ${x.opaque} opaque` : ''}${x.audit_missing ? ` · ${x.audit_missing} unrecorded` : ''}</small>`;
    return `<tr><td>${player(p, '', { details: false })}</td><td class="num">${fmtN(x.decisions)}<small class="nof">${fmtN(x.calls)} calls</small></td>
      <td>${idCell}</td><td class="num">${auditCell}</td><td class="num">${audited || x.tool_rejected ? fmtN(x.tool_rejected) : '<span class="dim">–</span>'}</td><td class="num">${audited || x.tool_rejected ? fmtN(x.unverified) : '<span class="dim">–</span>'}</td>
      <td class="num">${fmtN(x.retries)}<small class="nof">${x.resized} resized</small></td><td class="num">${fmtN(x.fallbacks)}</td>
      <td class="num">${fmtN(x.failed_tries)}<small class="nof">${x.rate_limited} rate-limit waits</small></td></tr>`;
  }).join('');
  if (!rows) return '';
  const head = `<thead><tr><th>Model</th><th class="num">Decisions</th><th title="Did the model's own tool report which model answered?">Identity</th>
    <th class="num" title="Turns whose record of tool use was read in full (verified), or that the backend cannot expose (opaque)">Turn audit</th>
    <th class="num" title="Replies that used a tool: rejected, and the model was asked again">Tool use</th><th class="num" title="Turns whose record could not be read: rejected and asked again">Unaudited</th>
    <th class="num" title="Decisions asked again after an invalid answer; resized: a legal intent adjusted to a legal size">Retries</th>
    <th class="num" title="No playable answer after every retry: the game checked or folded for it">Fallbacks</th><th class="num" title="Calls to the model's tool that failed and were tried again">Failed tries</th></tr></thead>`;
  return section('Audit', 'how each decision was made, from the call records', `<table class="audit-t">${head}<tbody>${rows}</tbody></table>`,
    'From each match\'s call log and decisions. Bots make no calls. Each decision\'s own record (tries, identity, audit) is in the replay under <b>why?</b>.<span class="show-sm"> Swipe sideways for every column.</span>',
    { scroll: true, id: 'audit' });
}

async function renderBoard() {
  const r = state.runs.find(x => x.run === state.run);
  const raw = await api('ratings', ['run', state.run]);
  Object.assign(state.labels, raw.labels || {});
  Object.assign(state.kinds, raw.kinds || {});
  const rt = normalize(raw, { pooled: !!r?.all });
  rt.finished = !!r?.finished;
  state.rt = rt;
  rt.audit = raw.run_audit || null;
  rt.playerAudit = raw.audit || {};
  if (voidReason(rt.design) && !rt.unregistered) {
    rt.pairings = rt.pairings.map(p => ({ ...p, verdict_recorded: p.verdict, verdict: 'descriptive', descriptive_reason: 'unregistered' }));
    rt.groups.forEach(g => { g.leader = null; g.pairings = rt.pairings.filter(p => g.players.includes(p.a) && g.players.includes(p.b)); });
    rt.players = rt.players.map(p => ({ ...p, rank: null, rank_kind: 'descriptive' }));
  }
  // a decided rank never rests on a verdict outside the calibrated range (unless the ratings already say so)
  for (const g of rt.groups) {
    if (!g.qualified_leader) continue;
    rt.players = rt.players.map(p => g.members.includes(p.player) && p.rank_kind === 'decided' && !p.assumptions
      ? { ...p, rank: null, rank_kind: 'descriptive', rank_note: 'outside the calibrated range' } : p);
  }
  state.rt = rt;
  for (const p of rt.players) if (p.kind) state.kinds[p.player] = p.kind;
  let html = runHead(r, rt) + intro();
  const waiting = rt.roster.filter(x => !rt.players.some(p => p.player === x.player));
  if (!rt.players.length) {
    html += `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>The cards are in the air</h2>
      <p>A deal counts once both of its plays are finished. ${state.index.some(m => m.started) ? `Watch the first ones on the <a href="${runLink(state.run, '/m')}"><u>games</u></a> page.` : ''}</p>
      ${waiting.length ? `<div class="waiting-list">${waiting.map(x => player(x.player, `<div class="small soft">${x.status === 'playing' ? 'playing its first deals' : 'waiting to play'}</div>`)).join('')}</div>` : ''}</div>`;
    $app.innerHTML = html + planPanel(rt);
    bindBoard();
    return;
  }
  html += outcomes(rt);
  html += `<div class="under-outcome">${planPanel(rt)}${auditLine(rt)}</div>`;
  html += standings(rt, waiting);
  html += pairingsSection(rt);
  html += headToHead(rt);
  html += styleTable(rt);
  html += auditSection(rt);
  $app.innerHTML = html;
  bindBoard();
}
function bindBoard() {
  $app.querySelector('.plan-panel')?.addEventListener('toggle', e => store.set('plan-open', e.target.open));
  $app.querySelectorAll('.th-def').forEach(b => b.onclick = () => {
    const col = STYLE_COLS.find(c => c[0] === b.dataset.def);
    $app.querySelectorAll('.th-def').forEach(x => x.classList.toggle('on', x.dataset.def === b.dataset.def));
    $app.querySelectorAll('.def-line').forEach(l => { l.innerHTML = `<b>${col[1]}:</b> ${esc(col[2])}`; l.classList.add('on'); });
  });
  $app.querySelectorAll('.metric-pick').forEach(sel => sel.onchange = () => {
    state.metric = sel.value;
    store.set('metric', sel.value);
    $app.querySelectorAll('table.style-t').forEach(t => { t.dataset.metric = sel.value; });
    const col = STYLE_COLS.find(c => c[0] === sel.value);
    $app.querySelectorAll('.def-line').forEach(l => { l.innerHTML = col ? `<b>${col[1]}:</b> ${esc(col[2])}` : ''; l.classList.add('on'); });
  });
}

function axisFor(players) {
  const vals = players.flatMap(p => [p.strength, ...(p.strength_ci95 || [])]).filter(v => v != null && isFinite(v));
  const m = Math.max(10, ...vals.map(Math.abs));
  const step = m > 400 ? 200 : m > 150 ? 100 : m > 60 ? 50 : 20;
  const lim = Math.ceil(m * 1.05 / step) * step;
  return { lim, x: v => (50 + Math.max(-1, Math.min(1, v / lim)) * 50).toFixed(2) + '%' };
}

function standings(rt, waiting) {
  const multi = rt.groups.filter(g => g.members.length).length > 1;
  const by = Object.fromEntries(rt.players.map(p => [p.player, p]));
  let out = '';
  const shown = rt.groups.filter(g => g.members.length);
  shown.forEach((g, k) => {
    const ps = g.members.map(p => by[p]).sort((x, y) => (y.strength ?? -1e9) - (x.strength ?? -1e9));
    const ax = axisFor(ps);
    const ranked = ps.some(p => p.rank_kind === 'decided' && p.rank != null);
    const rows = ps.map(p => standingRow(rt, g, p, ax, ranked)).join('');
    const extra = k === shown.length - 1 ? waitingRows(waiting, ranked) : '';
    const title = multi ? `Group ${groupName(g.index)}` : ranked ? 'Standings' : 'Order on points';
    const aside = ranked ? 'a numeral is a decided rank; the rest are on points' : 'provisional: no rank here is decided';
    const misfit = g.misfits ? ` <span class="flag" title="Some matchups disagree with what these strengths predict; see Matchups">${g.misfits} off the fit</span>` : '';
    out += section(title, `${multi ? `${plural(ps.length, 'player')} who played each other · ` : ''}${aside}${multi ? '' : ` · ${plural(rt.deals, 'duplicate deal')}`}${misfit}`,
      `<div class="board-list ${ranked ? '' : 'unranked'}"><div class="board-head">${ranked ? '<span title="Decided rank"><span class="hide-xs">Rank</span><span class="show-xs">#</span></span>' : ''}<span>${ranked ? 'Player' : 'Player, by points'}</span>
         <span class="r" title="Big blinds won per 100 hands against the average opponent in this group">Strength</span>
         <span class="axis-head"><span>${signed(-ax.lim, 0)}</span><span>0</span><span>${signed(ax.lim, 0)}</span></span>
         <span class="r hide-sm">Net</span><span class="r hide-sm" title="Share of 10-deal sessions won (secondary; not used to rank)">Sessions</span></div>
       ${rows}${extra}</div>`,
      multi && k === 0 ? 'No games connect these groups, so each is ordered on its own.' : '');
  });
  if (!out && waiting.length) out = section('Standings', '', `<div class="board-list unranked">${waitingRows(waiting, false)}</div>`);
  const out_ = rt.players.filter(p => !p.in_fit);
  if (out_.length) {
    out += section('Not in the standings', 'played, but left out of the fit', `<div class="board-list unranked left-out">${out_.map(p => `<div class="brow quarantined">
      ${player(p.player, `<div class="small soft why-out">${esc(p.unrated_reason || p.quarantine_reason || 'none of its opponents is rated')}</div>`)}
      <span class="r strength soft">${p.hands ? `<small>${plural(p.hands, 'hand')}</small>` : ''}</span><span class="axis"></span>
      <span class="r hide-sm soft">${p.net_bb != null ? signed(p.net_bb, 0) + ' bb' : ''}</span><span class="hide-sm"></span></div>`).join('')}</div>`,
      'Left out when its exact model or reasoning level is unknown (its results could mix more than one model), or when it only met players who are left out.');
  }
  return out;
}

function identityFlag(rt, p) {
  if (isBot(p.player)) return '';
  const a = rt.audit && rt.audit.players && rt.audit.players[p.player];
  const st = Core.identityOf(p, (rt.playerAudit || {})[p.player]) || Core.identityOf(p, a);
  if (!st || st === 'attested') return '';
  const tip = st === 'mismatch' ? 'The model that answered is not the one configured.'
    : st === 'unattested' ? 'Its tool never reported which model answered; only the configured name is known.'
    : 'Only some of its calls report which model answered.';
  return `<span class="flag ${st === 'mismatch' ? 'bad' : 'quiet'}" title="${esc(tip)}">identity ${esc(st)}</span>`;
}

function standingRow(rt, g, p, ax, ranked) {
  const ci = p.strength_ci95;
  const decided = p.rank_kind === 'decided' && p.rank != null;
  const tied = (p.rank_tied_with || []).filter(x => x !== p.player && g.members.includes(x));
  const notes = [];
  if (!p.identity_resolved) notes.push(`<span class="flag" title="${esc(p.quarantine_reason || '')}">identity unresolved</span>`);
  const idf = identityFlag(rt, p);
  if (idf) notes.push(idf);
  if (tied.length && tied.length < g.members.length - 1) notes.push(`<span class="small soft">difference with ${tied.map(plain).join(', ')} unresolved</span>`);
  const mob = `<div class="mob-line small soft">${p.net_bb != null ? `<span class="${cls(p.net_bb)}">${signed(p.net_bb, 0)} bb</span> · ` : ''}${plural(p.hands || 0, 'hand')}</div>`;
  const leader = g.leader === p.player;
  const b = Core.basisOf(p);
  const whisker = `<div class="whisker ${b ? b.key : ''}" title="${ci ? `95% range ${ciText(ci)} bb/100${b ? `, ${b.short}` : ''}` : 'no range yet: too few deals'}">
      <span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
      ${p.strength != null ? `<span class="point ${leader ? 'lead' : ''}" style="left:${ax.x(p.strength)}"></span>` : ''}</div>`;
  const elo = p.session_elo != null ? `<span title="Session Elo ${p.session_elo}${p.session_elo_ci95 ? ` (${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]})` : ''}">${p.session_w ?? 0}–${p.session_t ?? 0}–${p.session_l ?? 0}</span>
      <small>Elo ${p.session_elo}${p.session_elo_ci95 ? ` · ${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]}` : ''}</small>` : '';
  const half = ci ? Math.round((ci[1] - ci[0]) / 2) : null;
  return `<div class="brow ${leader ? 'lead' : ''}">
    ${ranked ? `<span class="rank ${decided ? 'sep' : 'prov'}" title="${decided ? 'Rank decided by the matchups' : 'Not decided: placed on points only'}">${decided ? p.rank : '–'}</span>` : ''}
    ${player(p.player, `${notes.length ? `<div class="row-notes">${notes.join(' ')}</div>` : ''}${mob}`)}
    <span class="r strength"><b>${fmtStrength(p.strength)}</b>${ci ? `<small title="${b ? esc(b.long) : '95% range'}">±${half}${b && b.key !== 'across' ? `<i class="basis-mark ${b.key}">${b.key === 'within' ? ' within conv.' : ' partly within'}</i>` : ''}</small>` : '<small>no range yet</small>'}</span>
    <span class="axis">${whisker}</span>
    <span class="r hide-sm"><span class="${cls(p.net_bb)}">${p.net_bb != null ? signed(p.net_bb, 0) + ' bb' : ''}</span><small>${plural(p.hands || 0, 'hand')}</small></span>
    <span class="r hide-sm sess">${elo}</span></div>`;
}

function waitingRows(waiting, ranked) {
  return waiting.map(x => `<div class="brow waiting">${ranked ? '<span class="rank">·</span>' : ''}
    ${player(x.player, `<div class="small soft">${x.status === 'playing' ? 'playing its first deals' : x.status === 'done' ? 'no finished deals' : 'waiting to play'}</div>`)}
    <span class="r strength soft">–</span><span class="axis"></span><span class="hide-sm"></span><span class="hide-sm"></span></div>`).join('');
}

function matchupId(p) {
  const ids = Object.fromEntries(state.index.map(m => [m.id.split('/').pop(), m.id]));
  const mu = (p.matchups || [])[(p.matchups || []).length - 1];
  if (mu && state.index.some(m => m.id === mu)) return mu;
  if (mu && (ids[mu] || ids[String(mu).split('/').pop()])) return ids[mu] || ids[String(mu).split('/').pop()];
  const m = state.index.find(m => m.started && ((m.a === p.a && m.b === p.b) || (m.a === p.b && m.b === p.a)));
  return m ? m.id : null;
}

// A verdict's badge and pill, with what it rests on (assumptions, official or retrospective) beside it.
function verdictCell(p, v, { compact = false } = {}) {
  const src = Core.sourceOf(p, plain);
  const bits = [];
  if (v.decided) bits.push('<span class="decided-badge">Decided</span>');
  bits.push(`<span class="verdict ${v.tone}" title="${esc(v.long)}">${esc(v.short)}</span>`);
  if (v.decided) bits.push(assumeChip(p.assumption));
  if (!compact && src) bits.push(chip(src.short, src.kind === 'official' ? 'ok' : 'quiet', src.text));
  return bits.join('');
}

// Each pairing: the edge with its range, its verdict, and how it compares with the strengths' prediction.
function pairingsSection(rt) {
  if (!rt.pairings.length) return '';
  const order = { a: 0, b: 0, even: 1, exhausted: 2, ineligible: 3, open: 4, descriptive: 4, early: 5 };
  const ax = axisFor(rt.pairings.map(p => ({ strength: Math.abs(p.edge ?? 0), strength_ci95: p.ci95 })));
  const descriptive = rt.pooled || rt.unregistered;
  const rows = [...rt.pairings].sort((x, y) => (order[x.verdict] ?? 6) - (order[y.verdict] ?? 6) || Math.abs(y.edge ?? 0) - Math.abs(x.edge ?? 0)).map(p => {
    const up = (p.edge ?? 0) >= 0, lead = up ? p.a : p.b, trail = up ? p.b : p.a;
    const edge = Math.abs(p.edge ?? 0);
    const ci = p.ci95 ? (up ? p.ci95 : [-p.ci95[1], -p.ci95[0]]) : null;
    const v = verdict(p, plain);
    const full = matchupId(p);
    const checks = p.looks_planned && !descriptive ? (p.looks_used ? ` · checked ${p.looks_used} of ${p.looks_planned}` : ' · not checked yet') : '';
    const conv = !descriptive && Core.conversationsShort(p) ? ` · ${p.conversations} of ${p.min_conversations} conversations a verdict needs` : '';
    const word = v.tone === 'won' ? 'beats' : v.tone === 'even' ? 'even with' : edge ? 'ahead of' : 'level with';
    const fit = p.fit;
    const fitLine = fit && fit.warn ? `<div class="fitline"><span class="flag">off the fit</span> The strengths predict ${signed(up ? fit.fitted : -fit.fitted, 0)} for ${plain(lead)}; this matchup's own result is ${signed(edge, 0)}${fit.estimated ? '' : ` (residual ${signed(up ? fit.residual : -fit.residual, 0)})`}. The direct result is the evidence.</div>` : '';
    const no = Core.ineligible(p);
    const noLine = no ? `<div class="fitline bad"><span class="flag bad">ineligible</span> ${esc(no.reasons.join('; ') || 'the judge may not decide this matchup')}</div>` : '';
    return `<div class="prow ${v.decided ? 'decided' : ''} ${v.tone === 'bad' ? 'ineligible' : ''}">
      <div class="duo-names">${player(lead, '', { details: false })}<span class="over">${word}</span>${player(trail, '', { details: false })}</div>
      <div class="edge"><b>${edge.toFixed(0)}</b><small>bb/100${ci ? ` · ${ciText(ci)}` : ''}</small>${ci ? basisTag(p) : ''}</div>
      <div class="edge-bar"><div class="whisker"><span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
        <span class="point ${v.tone === 'won' ? 'pos' : ''}" style="left:${ax.x(edge)}"></span></div></div>
      <div class="vcell">${verdictCell(p, v)}<small>${plural(p.deals || 0, 'deal')}${checks}${conv}</small></div>
      ${full ? `<a class="watch" href="${matchLink(full)}" title="Watch this matchup">Watch ›</a>` : '<span></span>'}
      ${noLine}${fitLine}</div>`;
  }).join('');
  return section('Matchups', 'who beats whom: the direct evidence, in bb/100 over duplicate deals', `<div class="pair-list">${rows}</div>`,
    descriptive ? 'Ahead on points only: no verdict here is a decision. The range is 95%.' : 'Decided results are the planned test\'s; the rest is ahead on points. Hover a verdict for what it rests on; the range is 95%.');
}

function headToHead(rt) {
  const paired = new Set(rt.pairings.flatMap(p => [p.a, p.b]));
  const cell = {};
  for (const p of rt.pairings) {
    cell[p.a + '|' + p.b] = { v: p.edge, p };
    cell[p.b + '|' + p.a] = { v: -p.edge, p };
  }
  const max = Math.max(1, ...rt.pairings.map(p => Math.abs(p.edge || 0)));
  const short = p => esc(nameParts(labelText(p))[0]);
  const matrix = names => `<table class="matrix"><thead><tr><th></th>${names.map(n => `<th title="${plain(n)}">${monogram(n, 'sm')}</th>`).join('')}</tr></thead><tbody>
    ${names.map(a => `<tr><th><div class="player">${monogram(a, 'sm')}<span class="mx-name">${name(a)}</span></div></th>${names.map(b => {
      if (a === b) return '<td class="cell self"></td>';
      const c = cell[a + '|' + b];
      if (!c) return '<td class="cell dim">·</td>';
      const won = c.p.verdict === 'a' || c.p.verdict === 'b';
      const outside = won && c.p.assumption && c.p.assumption.status === 'unsupported';
      const mine = won && ((c.p.verdict === 'a') === (c.p.a === a));
      const alpha = won ? ((outside ? 0.08 : 0.16) + (outside ? 0.12 : 0.3) * Math.sqrt(Math.abs(c.v) / max)).toFixed(2) : 0;
      const bg = won ? `background:rgba(var(--${mine ? 'pos' : 'neg'}-rgb),${alpha})` : '';
      const full = matchupId(c.p);
      return `<td class="cell ${outside ? 'outside' : ''}" style="${bg}"><a ${full ? `href="${matchLink(full)}"` : ''} title="${short(a)} against ${short(b)}: ${esc(verdict(c.p, plain).short)}${outside ? ' (outside the calibrated range)' : ''}">
        <span class="v ${won ? cls(c.v) : 'soft'}">${signed(c.v, 0)}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  const multi = rt.groups.filter(g => g.ranked.length).length > 1;
  const tables = rt.groups.map(g => g.ranked.filter(p => paired.has(p))).filter(ns => ns.length >= 3)
    .map((ns, i, all) => `${all.length > 1 || multi ? `<div class="mx-group">Group ${groupName(rt.groups.findIndex(g => g.ranked.includes(ns[0])))}</div>` : ''}${matrix(ns)}`);
  if (!tables.length) return '';
  const descriptive = rt.pooled || rt.unregistered;
  const anyOutside = rt.pairings.some(p => Core.isDecided(p) && p.assumption && p.assumption.status === 'unsupported');
  return section('Head to head', 'bb/100, row against column', tables.join(''),
    `${descriptive ? 'Nothing is shaded: no result here is decided.' : `Shaded only where the matchup is decided${anyOutside ? ' (faint and dashed: outside the calibrated range)' : ''}; plain numbers are ahead on points.`} A cell opens that matchup.`, { scroll: true });
}

// Tendencies, not quality: how often each does something, out of how many chances.
const STYLE_COLS = [
  ['vpip', 'Plays preflop', 'Put money in voluntarily before the flop (called or raised; checking the big blind does not count), out of its hands. Also called VPIP.'],
  ['pfr', 'Raises preflop', 'Raised before the flop, out of its hands. Also called PFR.'],
  ['af', 'Bets per call', 'After the flop: its bets and raises for each call. Above 1 leans aggressive.'],
  ['wtsd', 'To showdown', 'Reached the showdown, out of the hands where it saw a flop.'],
  ['wsd', 'Won showdown', 'Won the pot outright, out of its showdowns (split pots not counted).'],
  ['talk', 'Talks', 'Said something to the opponent, out of its decisions.'],
];
const SEATS = { button: 'on the button (small blind)', big_blind: 'in the big blind' };
function styleTable(rt) {
  const st = rt.stats || {};
  const names = rt.players.map(p => p.player).filter(p => st[p] && st[p].hands);
  if (!names.length) return '';
  const cellOf = (s, k) => {
    const r = Core.rateOf(s, k);
    if (r.x == null) return `<td class="num m m-${k}"><span class="dim">–</span></td>`;
    return `<td class="num m m-${k}">${Math.round(r.x * 100)}%${r.of != null ? `<small class="nof">${r.n}/${r.of}</small>` : ''}</td>`;
  };
  const af = s => {
    if (s.af_all_aggression) return '<span title="bet or raised after the flop and never called">all bets</span>';
    if (s.af == null) return '<span class="dim">–</span>';
    const c = s.counts?.af;
    return `${s.af.toFixed(1)}${c ? `<small class="nof">${c.n}/${c.of}</small>` : ''}`;
  };
  const row = (p, s, label) => `<tr class="${s.hands < 100 ? 'thin' : ''}"><td class="who-cell">${label || player(p, '', { details: false })}</td>
      ${cellOf(s, 'vpip')}${cellOf(s, 'pfr')}<td class="num m m-af">${af(s)}</td>${cellOf(s, 'wtsd')}${cellOf(s, 'wsd')}
      ${isBot(p) ? '<td class="num m m-talk"><span class="dim">–</span></td>' : cellOf(s, 'talk')}
      <td class="num soft">${(s.hands || 0).toLocaleString()}</td></tr>`;
  const head = `<thead><tr><th>Player</th>${STYLE_COLS.map(([k, label, tip]) =>
    `<th class="num m m-${k}"><button class="th-def" data-def="${k}" title="${esc(tip)}">${label}</button></th>`).join('')}<th class="num">Hands</th></tr></thead>`;
  const metric = STYLE_COLS.some(c => c[0] === state.metric) ? state.metric : 'vpip';
  const table = body => `<table class="style-t" data-metric="${metric}">${head}<tbody>${body}</tbody></table>`;
  const pick = `<label class="metric-bar"><span>Show</span><select class="metric-pick" aria-label="Which tendency">${STYLE_COLS.map(([k, label]) =>
    `<option value="${k}" ${k === metric ? 'selected' : ''}>${label}</option>`).join('')}</select></label>`;
  const rows = names.map(p => row(p, st[p])).join('');
  const splitOf = (key, label) => names.map(p => {
    const by = st[p][key] || (key === 'by_opponent' ? st[p].opponents : null);
    if (!by || !Object.keys(by).length) return '';
    return `<tr class="split-head"><td colspan="8">${player(p, '', { details: false })}</td></tr>` + Object.entries(by).map(([o, s]) => row(p, s, label(o))).join('');
  }).join('');
  const block = (title, body) => body ? `<details class="split"><summary>${title}</summary><div class="scroll">${table(body)}</div></details>` : '';
  const bySeat = splitOf('by_position', o => `<span class="seat-tag">${esc(SEATS[o] || o.replace(/_/g, ' '))}</span>`);
  const byOpp = splitOf('by_opponent', o => `<span class="soft small">vs</span> ${player(o, '', { details: false })}`);
  return section('How they play', 'tendencies, not quality', pick + table(rows),
    `<span class="def-line" aria-live="polite">Tap a column name for what it counts.</span> Each figure is how often it did something, with the count out of its chances beneath.
    These describe style, pooled over opponents and seats; they do not say who plays better. Rows under 100 hands are dimmed.`, { scroll: true, id: 'style' })
    + block('Split by seat: on the button or in the big blind', bySeat) + block('Split by opponent', byOpp);
}

// ---- games ----------------------------------------------------------------------------------------
// Matchups are grouped pairing → round → replicate: one entry per pairing, its rounds and replicates under it.
const pairKey = m => [m.a, m.b].sort().join('|');
const narrow = () => window.matchMedia('(max-width: 1180px)').matches;
const phone = () => window.matchMedia('(max-width: 860px)').matches;

function pairingsOf(index) {
  const by = new Map();
  for (const m of index) {
    const k = pairKey(m);
    if (!by.has(k)) by.set(k, { key: k, a: m.a, b: m.b, ms: [] });
    by.get(k).ms.push(m);
  }
  return [...by.values()].map(P => {
    P.ms.sort((x, y) => x.round - y.round || x.replicate - y.replicate);
    P.started = P.ms.filter(m => m.started);
    P.live = P.ms.some(m => m.live);
    P.rounds = [...new Set(P.started.map(m => m.round))];
    P.reps = [...new Set(P.started.map(m => m.replicate))];
    P.deals = P.started.reduce((s, m) => s + m.pairs_done, 0);
    P.netA = P.started.reduce((s, m) => s + (m.a === P.a ? 1 : -1) * m.pair_net_a / (m.bb || 1), 0);
    P.latest = P.ms.find(m => m.live) || P.started[P.started.length - 1] || null;
    return P;
  });
}

function scopeText(P) {
  const bits = [];
  if (P.rounds.length > 1) bits.push(plural(P.rounds.length, 'round'));
  if (P.reps.length > 1) bits.push(`${P.reps.length} replicates each`);
  return bits.join(' × ');
}

function matchNav(P, cur, rt) {
  const pr = rt && rt.pairings.find(p => pairKey(p) === P.key);
  const v = pr ? verdict(pr, plain) : null;
  const own = (p) => esc(nameParts(labelText(p))[0]);
  const active = cur && pairKey(cur) === P.key;
  const net = P.deals ? `<b class="${cls(P.netA)}" title="${own(P.a)}'s net over every round">${P.netA > 0 ? '+' : P.netA < 0 ? '−' : ''}${trim(Math.abs(Math.round(P.netA * 10) / 10))} bb</b>` : '';
  const status = !P.started.length ? 'waiting for a free seat'
    : `${v ? `${v.decided ? '<span class="decided-badge">Decided</span>' : ''}<span class="vword ${v.tone}">${esc(v.short)}</span>${v.qualifier ? ` <span class="vq" title="${esc(v.long)}">(${esc(v.qualifier)})</span>` : ''} · ` : ''}${plural(P.deals, 'deal')}`;
  const head = `<a class="phead" ${P.latest ? `href="${matchLink(P.latest.id)}"` : ''}><span class="duo">${monogram(P.a, 'sm')}${monogram(P.b, 'sm')}</span>
      <span class="pair-names">${own(P.a)} <span class="vs">vs</span> ${own(P.b)}</span>${P.live ? '<span class="live-dot" title="Playing now"></span>' : ''}${net}</a>
    <div class="pmeta">${status}</div>`;
  let rounds = '';
  if (active && P.started.length > 1) {
    const pill = (m, text) => `<a class="rep ${m === cur ? 'on' : ''} ${m.live ? 'live' : ''}" href="${matchLink(m.id)}" title="${m.pairs_done} of ${m.deals_total} deals">
      <span>${text}</span>${m.pairs_done ? `<b class="${cls((m.a === P.a ? 1 : -1) * m.pair_net_a)}">${signedBB((m.a === P.a ? 1 : -1) * m.pair_net_a, m.bb)}</b>` : ''}</a>`;
    rounds = P.reps.length > 1
      ? P.rounds.map(r => `<div class="rrow"><span class="rlab">Round ${r}</span>${P.started.filter(m => m.round === r).map(m => pill(m, `Rep ${m.replicate}`)).join('')}</div>`).join('')
      : `<div class="rrow">${P.started.map(m => pill(m, `Round ${m.round}`)).join('')}</div>`;
    rounds = `<div class="rounds" aria-label="Rounds and replicates">${rounds}</div>`;
  }
  return `<div class="pgroup ${active ? 'active' : ''} ${P.started.length ? '' : 'planned'}">${head}${rounds}</div>`;
}

async function ratingsForRun() {
  if (state.run === '*') return null;
  try {
    const raw = await api('ratings', ['run', state.run]);
    const rt = normalize(raw);
    rt.audit = raw.run_audit || null;
    return rt;
  } catch (e) { return null; }
}

async function renderMatches(id, params, soft) {
  if (!state.index.length) {
    $app.innerHTML = '<div class="card-box empty"><h2>No matchups in this run yet</h2></div>';
    return;
  }
  const started = state.index.filter(m => m.started);
  let cur = state.index.find(m => m.id === id);
  if (!cur && id) {
    const P = pairingsOf(state.index).find(P => P.ms.some(m => m.id.split('/').pop() === id.split('/').pop()));
    cur = P && P.latest;
  }
  cur = cur || started.find(m => m.live) || started[0] || state.index[0];
  Object.assign(state.labels, cur.labels || {});
  const rt = await ratingsForRun();
  const pairs = pairingsOf(state.index);
  const groups = [['Playing now', pairs.filter(P => P.live)], ['Played', pairs.filter(P => !P.live && P.started.length)],
    ['Not started', pairs.filter(P => !P.started.length)]];
  const curP = pairs.find(P => P.key === pairKey(cur));
  const where = curP && curP.started.length > 1 ? `Round ${cur.round}${curP.reps.length > 1 ? ` · rep ${cur.replicate}` : ''}` : '';
  const toggle = `<button class="mtoggle" aria-expanded="${!!state.listOpen}"><span>${state.listOpen ? 'Hide matchups' : `All ${plural(pairs.length, 'matchup')}`}</span><i>▾</i></button>
    ${where ? `<span class="mwhere">${where}</span>` : ''}`;
  const list = `<div class="mbar">${toggle}</div>` + groups.filter(g => g[1].length).map(([title, ps]) =>
    `<div class="group">${title}</div>` + ps.map(P => matchNav(P, cur, rt)).join('')).join('');

  let body, after = null;
  try {
    if (cur.started) [body, after] = await matchBody(cur, params, rt, curP);
    else body = `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>${plain(cur.a)} vs ${plain(cur.b)}</h2><p>Not started yet: it plays when a seat frees up.</p></div>`;
  } catch (e) {
    body = `<div class="card-box empty"><h2>Couldn't load this matchup</h2><p class="error">${esc(e.message)}</p></div>`;
  }
  const keep = soft ? {
    why: [...document.querySelectorAll('.play .ev.open')].map(e => e.closest('.play').dataset.play + ':' + e.dataset.step),
    notes: [...document.querySelectorAll('.play details.notes[open]')].map(e => e.closest('.play').dataset.play),
  } : null;
  const scroll = window.scrollY;
  const listScroll = document.querySelector('.mlist')?.scrollTop || 0;
  $app.innerHTML = `<div class="layout"><div class="card-box mlist ${state.listOpen ? 'open' : ''}" role="navigation" aria-label="Matchups">${list}</div><div class="mbody">${body}</div></div>`;
  if (soft) window.scrollTo(0, scroll);
  const ml = $app.querySelector('.mlist');
  ml.scrollTop = listScroll;
  ml.querySelector('.mtoggle').onclick = () => {
    state.listOpen = !state.listOpen;
    ml.classList.toggle('open', state.listOpen);
    const b = ml.querySelector('.mtoggle');
    b.setAttribute('aria-expanded', String(state.listOpen));
    b.querySelector('span').textContent = state.listOpen ? 'Hide matchups' : `All ${plural(pairs.length, 'matchup')}`;
  };
  ml.querySelectorAll('a[href]').forEach(a => a.addEventListener('click', () => { state.listOpen = false; }));
  if (!soft) ml.querySelector('.rep.on')?.scrollIntoView({ block: 'nearest', inline: 'center' });
  if (after) after();
  if (keep) {
    for (const k of keep.why) {
      const [j, step] = k.split(':');
      $app.querySelector(`.play[data-play="${j}"] .ev[data-step="${step}"]`)?.classList.add('open');
    }
    for (const j of keep.notes) $app.querySelector(`.play[data-play="${j}"] details.notes`)?.setAttribute('open', '');
  }
}

function go(cur, k) {
  if (!state.view || k < 0 || k >= state.view.total || !state.view.played.has(k)) return;
  history.replaceState(null, '', matchLink(cur.id, k));
  route();
}

function chipColor(v, max) {
  if (v === 0) return '#55605a';
  const t = Math.sqrt(Math.abs(v) / max);
  const base = v > 0 ? 'var(--pos-chip)' : 'var(--neg-chip)';
  return `color-mix(in srgb, ${base} ${Math.round(35 + 65 * t)}%, #24302a)`;
}

const timelineMode = () => narrow() ? 'sequence' : state.mode;

async function matchBody(cur, params, rt, P) {
  const [ma, mb] = await Promise.all(cur.matches.map(x => loadMatch(x.id)));
  const pairing = rt ? rt.pairings.find(p => pairKey(p) === pairKey(cur)) || null : null;
  const bb = ma.config.game.big_unit || ma.config.game.bb || 1;
  const ha = Object.fromEntries(ma.hands.map(h => [h.hand, h]));
  const hb = mb ? Object.fromEntries(mb.hands.map(h => [h.hand, h])) : {};
  const total = ma.config.seeds.length;
  const played = new Set([...Object.keys(ma.events), ...Object.keys(mb ? mb.events : {})].map(Number));
  const done = [...Array(total).keys()].filter(k => ha[k] && hb[k]);
  const nets = Object.fromEntries(done.map(k => [k, ha[k].net[0] + hb[k].net[1]]));
  const wanted = params.has('deal') ? Number(params.get('deal')) - 1 : null;
  const pick = wanted != null && played.has(wanted) ? wanted : !played.size ? 0 : cur.live ? Math.max(...played) : Math.min(...played);
  state.view = { cur, pick, total, played };

  // the headline is the whole pairing (every round and replicate); the selected one's own score sits beneath it
  const net = done.reduce((s, k) => s + nets[k], 0);
  const wins = done.filter(k => nets[k] > 0).length, losses = done.filter(k => nets[k] < 0).length;
  const multi = P && P.started.length > 1;
  const sign = P && P.a !== cur.a ? -1 : 1;
  const cumBB = multi ? sign * P.netA : net / bb;
  const cumDeals = multi ? P.deals : done.length;
  const leader = cumBB > 0 ? cur.a : cumBB < 0 ? cur.b : null;
  const rate = cumDeals ? cumBB / (2 * cumDeals) * 100 : 0;
  const v = pairing ? verdict(pairing, plain) : null;
  const short = p => esc(nameParts(labelText(p))[0]);
  const face = (p, side) => `<div class="face ${side} ${leader === p ? 'ahead' : ''}">${monogram(p, 'xl')}<div class="face-name">${name(p)}</div></div>`;
  const roundNet = net / bb, roundLead = roundNet > 0 ? cur.a : roundNet < 0 ? cur.b : null;
  const which = multi ? `Round ${cur.round}${P.reps.length > 1 ? ` · replicate ${cur.replicate} of ${P.reps.length}` : ''}` : '';
  const roundLine = multi ? `<div class="score-round"><b>${which}</b>: ${done.length
      ? (roundLead ? `${short(roundLead)} up ${trim(Math.abs(Math.round(roundNet * 10) / 10))} bb` : 'even') + ` · deals won ${wins}–${losses}` : 'no deals finished'} · ${done.length} of ${total} played</div>`
    : `<div class="score-cap">deals won ${wins}–${losses}${done.length - wins - losses ? ` · ${done.length - wins - losses} even` : ''} · ${done.length} of ${total} played</div>`;
  const big = cumDeals ? (leader ? `<div class="score-who">${plain(leader)} is up</div><div class="score-big">${trim(Math.abs(Math.round(cumBB * 10) / 10))}<small>bb</small></div>
      <div class="score-rate">${trim(Math.abs(Math.round(rate * 10) / 10))} bb/100 hands</div>` : '<div class="score-big even">Even</div>') : '<div class="score-who">No deals finished yet</div>';
  const scoreboard = `<div class="scoreboard card-box">
      ${face(cur.a, 'left')}
      <div class="score-mid">
        ${multi ? `<div class="score-scope">Whole matchup · ${plural(cumDeals, 'deal')}<span class="hide-xs"> · ${scopeText(P)}</span></div>` : ''}
        ${big}
        ${v ? `<div class="score-verdict"><span class="vline">${verdictCell(pairing, v)}</span><span class="small soft vlong">${esc(v.long)}</span></div>` : ''}
        ${roundLine}
      </div>
      ${face(cur.b, 'right')}
      <div class="score-mini">${leader ? `<b>${short(leader)}</b> up ${trim(Math.abs(Math.round(cumBB * 10) / 10))} bb` : 'Even'}${v ? ` · ${v.decided ? 'decided: ' : ''}${esc(v.short.toLowerCase())}${v.qualifier ? ` (${esc(v.qualifier)})` : ''}` : ''}${which ? ` · ${which}` : ''}</div></div>`;

  const max = Math.max(1, ...done.map(k => Math.abs(nets[k])));
  const strip = Array.from({ length: total }, (_, k) => {
    const sel = k === pick ? 'sel' : '';
    if (!(k in nets)) return `<button class="chipbtn ${played.has(k) ? 'half' : 'empty'} ${sel}" ${played.has(k) ? `data-deal="${k}" title="Deal ${k + 1}: being played"` : 'disabled'}><span>${k + 1}</span></button>`;
    const x = nets[k];
    return `<button data-deal="${k}" class="chipbtn ${sel}" style="--c:${chipColor(x, max)}" title="Deal ${k + 1}: ${x === 0 ? 'even' : `${plain(x > 0 ? cur.a : cur.b)} won ${trim(Math.abs(x) / bb)} bb`} over both plays"><span>${k + 1}</span></button>`;
  }).join('');
  const legend = `<div class="strip-legend">
    <span><span class="swatch" style="background:var(--pos-chip)"></span>${short(cur.a)} won the deal</span>
    <span><span class="swatch" style="background:var(--neg-chip)"></span>${short(cur.b)} won it</span>
    <span>brighter, bigger swing</span>
    <span class="keys"><span class="kbd">←</span><span class="kbd">→</span> deals · <span class="kbd">space</span> play · <span class="kbd">,</span><span class="kbd">.</span> step · <span class="kbd">r</span> reasoning</span></div>`;
  const mapOpen = state.mapOpen ?? !phone();
  const dealMap = `<details class="dealmap" ${mapOpen ? 'open' : ''}><summary><span>Deal map</span><span class="soft small">${done.length} of ${plural(total, 'deal')} finished · tap a chip to jump</span></summary>
    <div class="strip" style="--cols:${total <= 30 ? Math.max(total, 16) : Math.ceil(total / Math.ceil(total / 25))}">${strip}</div>${legend}</details>`;

  const order = [...played].sort((x, y) => x - y);
  const i = order.indexOf(pick);
  const dealNet = pick in nets ? nets[pick] : null;
  const outcome = dealNet == null ? 'in progress' : dealNet === 0 ? 'even over both plays'
    : `<b>${short(dealNet > 0 ? cur.a : cur.b)}</b> wins it by <b class="gold">${trim(Math.abs(dealNet) / bb)} bb</b> over both plays`;

  const plays = [ma, mb].filter(Boolean).map((m, j) => ({ m, j, names: m.config.players.map(x => esc(nameParts(labelText(x))[0])),
    ...PokerTable.steps(m.events[pick] || []) }));
  const mode = timelineMode();
  const frames = PokerTable.timeline(plays.map(p => p.steps), mode);
  const last = frames.length - 1;
  const key = cur.id + '#' + pick;
  // a finished deal opens at its first step; a deal still being played follows the latest step
  const following = !(pick in nets);
  const carry = state.carry;
  state.carry = false;
  const rp = state.replay && state.replay.key === key ? state.replay
    : { key, s: following && !carry ? last : 0, follow: following && !carry, playing: !!carry, fresh: true };
  if (rp.fresh && params.has('step')) { rp.s = Math.max(0, Math.min(last, Number(params.get('step')) - 1 || 0)); rp.follow = rp.s >= last; }
  if (rp.follow && !rp.playing) rp.s = last;
  rp.frames = frames;
  rp.mode = mode;
  rp.last = last;
  rp.s = Math.min(rp.s, last);
  state.replay = rp;
  if (rp.fresh && mode === 'sequence') state.showPlay = frames[rp.s].focus ?? state.showPlay;
  state.view.plays = plays;
  state.view.bb = bb;

  const controls = `<div class="controls card-box" role="group" aria-label="Replay">
    <div class="deal-title"><h2>${multi ? `<span class="round">${which}</span>` : ''}Deal ${pick + 1}<small> of ${total}</small></h2>
      <span class="nav"><button data-deal="${order[i - 1]}" ${i > 0 ? '' : 'disabled'} aria-label="Previous deal" title="Previous deal (←)">←</button>
      <button data-deal="${order[i + 1]}" ${i < order.length - 1 ? '' : 'disabled'} aria-label="Next deal" title="Next deal (→)">→</button></span></div>
    <div class="ctls">
      <button class="ctl" data-ctl="start" title="Back to the deal" aria-label="Back to the start">⏮</button>
      <button class="ctl" data-ctl="prev" title="Step back (,)" aria-label="Step back">◀</button>
      <button class="ctl play" data-ctl="play" title="Play step by step (space)">▶ Play</button>
      <button class="ctl" data-ctl="next" title="Step forward (.)" aria-label="Step forward">▶</button>
      <button class="ctl" data-ctl="end" title="To the end" aria-label="To the end">⏭</button></div>
    <input type="range" class="scrub" min="0" max="${last}" value="${rp.s}" aria-label="Step">
    <span class="step-text"></span>
    <span class="deal-outcome soft">${outcome}</span></div>`;
  const viewBar = `<div class="view-bar">
    <span class="play-switch" role="group" aria-label="Which play">${plays.map(p => `<button class="toggle" data-show="${p.j}" aria-pressed="${state.showPlay === p.j}">${p.j ? 'Play 2<span class="hide-xs"> · swapped</span>' : 'Play 1'}</button>`).join('')}</span>
    <span class="mode-switch" role="group" aria-label="How the two plays move">
      <button class="toggle" data-mode="aligned" aria-pressed="${state.mode === 'aligned'}" title="Both plays reach the flop, turn, river and result together">Streets side by side</button>
      <button class="toggle" data-mode="sequence" aria-pressed="${state.mode === 'sequence'}" title="Play 1 from start to end, then Play 2">Play 1, then Play 2</button></span>
    <span class="seq-note small soft">Play 1 plays out, then Play 2, then the next deal</span>
    <span class="spacer"></span>
    <button class="toggle" id="auto-toggle" aria-pressed="${state.autoNext}" title="When a deal ends, go on to the next one">Auto-next deal</button>
    <button class="toggle" id="mind-toggle" aria-pressed="${state.mind}" title="Show every decision's reasoning (r)">Reasoning</button></div>`;

  const body = `${scoreboard}${dealMap}${controls}${viewBar}
    <div class="pair ${state.mind ? 'show-mind' : ''}" data-show="${state.showPlay}">${plays.map(p => playPanel(p, cur, bb)).join('')}</div>
    <p class="omni small soft">Hand names under each seat use both players' cards, which the players themselves never see.</p>
    ${comparePanel(plays, cur, bb)}`;
  const after = () => {
    const mbody = $app.querySelector('.mbody');
    mbody.classList.toggle('playing', !!rp.playing);
    $app.querySelectorAll('[data-deal]').forEach(b => b.onclick = () => go(cur, Number(b.dataset.deal)));
    $app.querySelectorAll('[data-ctl]').forEach(b => b.onclick = () => control(b.dataset.ctl));
    $app.querySelector('.dealmap')?.addEventListener('toggle', e => { state.mapOpen = e.target.open; });
    const sc = $app.querySelector('.scrub');
    if (sc) sc.oninput = () => { stopPlay(); setStep(Number(sc.value)); };
    $app.querySelectorAll('.ev[data-step]').forEach(el => el.addEventListener('click', e => {
      if (e.target.closest('.why')) return;
      stopPlay();
      setStep(PokerTable.frameOf(state.replay.frames, Number(el.closest('.play').dataset.play), Number(el.dataset.step)));
    }));
    $app.querySelectorAll('.why').forEach(b => b.onclick = () => {
      const ev = b.closest('.ev');
      ev.classList.toggle('open');
      b.setAttribute('aria-expanded', String(ev.classList.contains('open')));
    });
    document.getElementById('mind-toggle')?.addEventListener('click', toggleMind);
    document.getElementById('auto-toggle')?.addEventListener('click', e => {
      state.autoNext = !state.autoNext;
      store.set('autonext', state.autoNext);
      e.currentTarget.setAttribute('aria-pressed', String(state.autoNext));
    });
    $app.querySelectorAll('.play-switch [data-show]').forEach(b => b.onclick = () => {
      const j = Number(b.dataset.show);
      showPlay(j);
      const r = state.replay;
      if (r && r.mode === 'sequence') setStep(PokerTable.frameOf(r.frames, j, r.frames[r.s].at[j]));
    });
    $app.querySelectorAll('.mode-switch [data-mode]').forEach(b => b.onclick = () => {
      state.mode = b.dataset.mode;
      store.set('timeline', state.mode);
      $app.querySelectorAll('.mode-switch [data-mode]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.mode === state.mode)));
      rebuildTimeline();
    });
    const fresh = rp.fresh;
    rp.fresh = false;
    applyStep(fresh ? null : 'keep');
    if (fresh) $app.querySelectorAll('.ptable').forEach(t => t.classList.add('deal-in'));
    if (rp.playing) schedulePlay();
  };
  return [body, after];
}

function showPlay(j) {
  state.showPlay = j;
  $app.querySelector('.pair')?.setAttribute('data-show', j);
  $app.querySelectorAll('.play-switch [data-show]').forEach(x => x.setAttribute('aria-pressed', String(Number(x.dataset.show) === j)));
}

// The same moment of the deal on a new timeline, after the mode or the screen width changes.
function rebuildTimeline() {
  const v = state.view, rp = state.replay;
  if (!v || !v.plays || !rp) return;
  const mode = timelineMode();
  if (mode === rp.mode) return;
  const f = rp.frames[rp.s], atEnd = rp.s >= rp.last;
  const j = f.focus ?? (mode === 'sequence' && narrow() ? state.showPlay : 0);
  const frames = PokerTable.timeline(v.plays.map(p => p.steps), mode);
  rp.frames = frames;
  rp.mode = mode;
  rp.last = frames.length - 1;
  rp.s = atEnd ? rp.last : Math.max(0, PokerTable.frameOf(frames, j, f.at[j]));
  applyStep('keep');
}
window.matchMedia('(max-width: 1180px)').addEventListener('change', rebuildTimeline);

// The duplicate comparison: the same cards played twice with the seats swapped, street by street, each
// model's choices beside its choices in the other play. It fills in as the replay reaches each step.
function comparePanel(plays, cur, bb) {
  if (plays.length < 2 || plays.some(p => !p.steps.length)) return '';
  const fmt = c => bbs(c, bb);
  const VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to', all_in: 'all in' };
  const who = (p, seat) => p.m.config.players[seat];
  const rows = ['preflop', 'flop', 'turn', 'river'].map(st => {
    const acts = plays.map(p => p.log.filter(e => e.kind === 'action' && e.street === st));
    if (acts.every(a => !a.length)) return '';
    const sig = (p, a) => a.map(e => `${e.seat}:${e.action}`).join(' ');
    const same = sig(plays[0], acts[0]) === sig(plays[1], acts[1]);
    const cells = plays.map((p, j) => `<div class="cmp-cell" data-play="${j}">${acts[j].map(e => `<span class="ca" data-step="${e.step}">${monogram(who(p, e.seat), 'xs')}<span class="verb ${e.action}">${VERB[e.action] || esc(e.action)}${['call', 'bet', 'raise'].includes(e.action) ? ' ' + fmt(e.amount) : ''}</span></span>`).join('') || '<span class="soft small">no action</span>'}</div>`).join('');
    return `<div class="cmp-row" data-seg="${st}"><div class="cmp-st">${st[0].toUpperCase() + st.slice(1)}<span class="cmp-tag ${same ? 'same' : 'diff'}">${same ? 'same choices' : 'differs'}</span></div>${cells}</div>`;
  }).join('');
  const results = plays.map((p, j) => {
    const r = p.log.find(e => e.kind === 'result');
    if (!r) return `<div class="cmp-cell" data-play="${j}"><span class="soft small">not finished</span></div>`;
    const net = r.net[p.m.config.players.indexOf(cur.a)] ?? 0;
    return `<div class="cmp-cell" data-play="${j}"><span class="ca" data-step="${r.step}">${monogram(cur.a, 'xs')}<b class="${cls(net)}">${signedBB(net, bb)}</b>${r.showdown ? '<span class="soft small">showdown</span>' : ''}</span></div>`;
  }).join('');
  return `<details class="cmp card-box" ${phone() ? '' : 'open'}><summary><span>The two plays, street by street</span><span class="soft small">the same cards go to the same seat, held by the other model: did it choose the same?</span></summary>
    <div class="cmp-grid"><div class="cmp-row cmp-head"><div></div><div>Play 1</div><div>Play 2 · swapped</div></div>${rows}
    <div class="cmp-row" data-seg="end"><div class="cmp-st">${short3(cur.a)} net</div>${results}</div></div></details>`;
}
const short3 = p => esc(nameParts(labelText(p))[0]);

function playPanel(p, cur, bb) {
  const { m, j } = p;
  const players = m.config.players;
  const title = j === 0 ? 'Play 1' : 'Play 2 · seats swapped';
  if (!p.steps.length) return `<div class="card-box play"><div class="play-head"><h3>${title}</h3></div><div class="pending">Not dealt yet.</div></div>`;
  const btn = p.button;
  const seats = players.map((pid, i) => {
    const [model, level] = nameParts(labelText(pid));
    return { name: `<span class="nm" title="${esc(labelText(pid))}">${esc(model)}</span>`, effort: esc(level ? effortText(level) : ''), badge: monogram(pid, 'lg') };
  });
  const fmt = c => bbs(c, bb);
  const shortName = i => esc(nameParts(labelText(players[i]))[0]);
  const thoughtsBy = {};
  let log = '';
  for (const e of p.log) {
    const s = e.step;
    if (e.kind === 'start') {
      log += `<div class="street" data-step="${s}"><span>Preflop</span></div>
        <div class="ev quiet" data-step="${s}"><span class="what">Blinds: ${e.blinds.map(b => `${shortName(b.seat)} ${fmt(b.amount)}`).join(' · ')}</span></div>`;
    } else if (e.kind === 'board') {
      log += `<div class="street" data-step="${s}"><span>${esc(e.street)}</span>${PokerTable.cards(e.cards, 0, 'sm')}</div>`;
    } else if (e.kind === 'runout') {
      log += `<div class="ev quiet" data-step="${s}"><span class="what">All in: the rest of the board is dealt.</span></div>`;
    } else if (e.kind === 'showdown') {
      log += `<div class="street" data-step="${s}"><span>Showdown</span></div>` + Object.entries(e.hands).map(([i, h]) =>
        `<div class="showdown" data-step="${s}"><span class="actor">${shortName(Number(i))}</span>${PokerTable.cards(h.cards, 0, 'sm')}<span class="soft">${esc(h.best)}</span></div>`).join('');
    } else if (e.kind === 'result') {
      const how = e.showdown ? 'at showdown' : 'without a showdown';
      log += `<div class="outcome" data-step="${s}">${e.winners.length > 1 ? `Split pot ${how}` : `<b>${shortName(e.winners[0])}</b> takes <b class="gold">${fmt(e.pot)}</b> ${how}`}</div>`;
    } else if (e.kind === 'action') {
      const t = e.thought;
      if (t?.notes) thoughtsBy[e.seat] = t.notes;
      const amt = ['call', 'bet', 'raise'].includes(e.action) ? ` ${fmt(e.amount)}` : '';
      const VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to', all_in: 'all in' };
      const corr = correction(t, e, fmt);
      const badges = [];
      if (e.all_in) badges.push('<span class="badge strong">all in</span>');
      if (corr) badges.push(`<span class="badge ${corr.kind === 'fallback' ? 'bad' : 'warn'}" title="${esc(corr.reason)}">${esc(corr.label)}</span>`);
      const corrLine = corr ? `<div class="corr ${corr.kind}">${corr.requested ? `Asked <b>${esc(corr.requested)}</b> → played <b>${esc(corr.executed)}</b>` : `Played <b>${esc(corr.executed)}</b>`}${corr.reason ? `<span class="soft">: ${esc(corr.reason)}</span>` : ''}</div>` : '';
      const hasMind = t && (t.reasoning || t.thinking);
      const record = recordOf(t, players[e.seat], fmt);
      const mind = hasMind || record ? `<div class="mind">${t && t.reasoning ? esc(t.reasoning) : ''}${t && t.thinking
        ? `<span class="lbl">The model's own thinking</span><span class="thinking">${esc(t.thinking)}</span>` : ''}${record}</div>` : '';
      const f = e.facing;
      const facing = f && e.action !== 'raise' && e.action !== 'all_in' ? `<span class="facing" title="What it had to put in to call, and the share of the final pot a call needs to win to break even">facing ${fmt(f.toCall)} · needs ${Math.round(f.odds * 100)}%</span>`
        : f ? `<span class="facing" title="What it had to put in to call">facing ${fmt(f.toCall)}</span>` : '';
      log += `<div class="ev move" data-step="${s}">${monogram(players[e.seat], 'xs')}<span class="actor">${shortName(e.seat)}</span>
        <span class="what"><span class="verb ${e.action}">${VERB[e.action] || esc(e.action)}${amt}</span>${badges.join('')}${facing}</span>
        ${hasMind || record ? `<button class="why" aria-expanded="false" title="This decision's private reasoning and its record">${hasMind ? 'why?' : 'record'}</button>` : ''}
        ${e.say ? `<div class="say-row"><span class="say">${esc(e.say)}</span></div>` : ''}${corrLine}${mind}</div>`;
    }
  }
  const notes = Object.entries(thoughtsBy);
  const noteBlock = notes.length ? `<details class="notes"><summary>Notes each model keeps on its opponent</summary>${notes.map(([i, n]) =>
    `<span class="lbl">${shortName(Number(i))}</span><div>${esc(n)}</div>`).join('')}</details>` : '';
  const button = btn != null ? `${shortName(btn)} on the button` : '';
  return `<div class="card-box play" data-play="${j}"><div class="play-head"><h3>${title}</h3><span class="small soft">${button}</span></div>
    <div class="table-slot">${PokerTable.html(seats)}</div><div class="log">${log}</div>${noteBlock}</div>`;
}

// One decision's record: which model answered (attested or not), whether its turn was audited, and every
// try before the one that was played.
const TRY_WORD = { tool_use: 'used a tool, rejected', unverified: 'could not be audited, rejected', invalid: 'invalid', illegal: 'illegal' };
function recordOf(t, pid, fmt) {
  if (!t || isBot(pid)) return '';
  const seen = t.observed_model || t.served_model || null;
  const want = t.requested_model || null;
  const id = t.identity || (seen ? (want && want !== seen ? 'mismatch' : 'attested') : 'unattested');
  const bits = [`identity <b class="idst ${id === 'attested' ? 'ok' : id === 'mismatch' ? 'bad' : 'warn'}">${esc(id)}</b>${seen ? ` (${esc(seen)}${want && want !== seen ? `, asked for ${esc(want)}` : ''})` : ''}`];
  bits.push(t.audit ? `turn audit <b>${esc(t.audit)}</b>` : 'turn audit not recorded');
  if (t.tries) bits.push(plural(t.tries, 'try', 'tries'));
  if (t.seconds != null) bits.push(`${t.seconds} s`);
  const tries = (t.retries || []).map((r, k) => `<li>Try ${k + 1}${r.requested && r.requested.action ? `: asked <b>${esc(Core.actText(r.requested, fmt))}</b>` : ''}
    → ${esc(TRY_WORD[r.kind] || 'rejected')}${r.error ? `: ${esc(r.error)}` : ''}</li>`).join('');
  return `<span class="lbl">Record</span><span class="record">${bits.join(' · ')}</span>${tries ? `<ol class="tries">${tries}</ol>` : ''}`;
}

// ---- replay: one timeline of frames drives both plays (see PokerTable.timeline) ---------------------
const SEG_NAME = { preflop: 'Preflop', flop: 'Flop', turn: 'Turn', river: 'River', end: 'Result' };
function applyStep(mode) {
  const v = state.view, rp = state.replay;
  if (!v || !v.plays || !rp || !rp.frames) return;
  const s = rp.s, f = rp.frames[s];
  if (rp.mode === 'sequence' && narrow() && f.focus != null && f.focus !== state.showPlay) showPlay(f.focus);
  v.plays.forEach(p => {
    const panel = $app.querySelector(`.play[data-play="${p.j}"]`);
    if (!panel || !p.steps.length) return;
    const mine = Math.min(f.at[p.j] ?? 0, p.steps.length - 1);
    const moving = f.focus == null || f.focus === p.j;
    const prev = mode !== 'keep' && p.shown != null && p.shown < mine ? p.steps[p.shown] : null;
    PokerTable.show(panel, p.steps[mine], { fmt: c => bbs(c, v.bb), prev, names: p.names });
    p.shown = mine;
    panel.classList.toggle('held', !moving && s < rp.last);
    panel.querySelectorAll('[data-step]').forEach(el => {
      const k = Number(el.dataset.step);
      el.classList.toggle('future', k > mine);
      el.classList.toggle('now', k === mine && moving && s < rp.last);
    });
  });
  $app.querySelectorAll('.cmp-cell[data-play]').forEach(cell => {
    const j = Number(cell.dataset.play), p = v.plays[j];
    const mine = p && p.steps.length ? Math.min(f.at[j] ?? 0, p.steps.length - 1) : -1;
    cell.querySelectorAll('[data-step]').forEach(el => el.classList.toggle('future', Number(el.dataset.step) > mine));
  });
  $app.querySelectorAll('.cmp-row[data-seg]').forEach(row => {
    const items = row.querySelectorAll('[data-step]');
    const shown = [...items].filter(x => !x.classList.contains('future')).length;
    row.classList.toggle('pending', !shown);
    row.classList.toggle('complete', shown === items.length);
  });
  $app.querySelectorAll('.cmp-grid').forEach(g => g.classList.toggle('empty', !g.querySelector('.cmp-row[data-seg]:not(.pending)')));
  const sc = $app.querySelector('.scrub');
  if (sc) { sc.max = rp.last; sc.value = s; }
  const txt = $app.querySelector('.step-text');
  if (txt) {
    if (s >= rp.last) txt.textContent = 'End of the deal';
    else if (rp.mode === 'sequence' && f.focus != null) {
      const p = v.plays[f.focus];
      txt.textContent = `Play ${f.focus + 1} · step ${(f.at[f.focus] ?? 0) + 1} of ${p ? p.steps.length : 1}`;
    } else txt.textContent = `${SEG_NAME[f.seg] || ''} · step ${s + 1} of ${rp.last + 1}`;
  }
  $app.querySelector('.deal-outcome')?.classList.toggle('spoiler', s < rp.last);
  const pb = $app.querySelector('[data-ctl="play"]');
  if (pb) pb.innerHTML = rp.playing ? '❚❚ Pause' : s >= rp.last ? '↺ Replay' : s === 0 ? '▶ Watch<span class="hide-xs"> from start</span>' : '▶ Play';
}
function setStep(s) {
  const rp = state.replay;
  if (!rp) return;
  rp.s = Math.max(0, Math.min(rp.last, s));
  rp.follow = rp.s >= rp.last;
  applyStep();
}
function schedulePlay() {
  clearTimeout(state.replay?.timer);
  const rp = state.replay;
  if (!rp || !rp.playing) return;
  const f = rp.frames[rp.s], g = rp.frames[rp.s + 1];
  const handover = g && f.focus != null && g.focus !== f.focus;
  rp.timer = setTimeout(() => {
    if (!rp.playing) return;
    if (rp.s >= rp.last) { nextDeal(rp); return; }
    setStep(rp.s + 1);
    schedulePlay();
  }, handover ? 1800 : rp.s >= rp.last - 1 ? 1500 : 1100);
}
// At the end of a deal, playback goes on to the next played deal, unless that is switched off.
function nextDeal(rp) {
  const v = state.view;
  const order = v ? [...v.played].sort((a, b) => a - b) : [];
  const next = order[order.indexOf(v?.pick) + 1];
  if (!state.autoNext || next == null) { stopPlay(); return; }
  rp.timer = setTimeout(() => {
    if (!rp.playing) return;
    state.carry = true;
    go(v.cur, next);
  }, 1400);
}
// On a phone, watching folds the scoreboard and the deal map away and brings the table up.
function bringTableUp() {
  const mbody = $app.querySelector('.mbody');
  if (!mbody || mbody.classList.contains('playing')) return;
  mbody.classList.add('playing');
  if (!phone()) return;
  const map = $app.querySelector('.dealmap');
  if (map && map.open) { map.open = false; state.mapOpen = false; }
  const ctl = $app.querySelector('.controls');
  if (ctl) window.scrollTo({ top: Math.max(0, ctl.getBoundingClientRect().top + window.scrollY - 64), behavior: 'smooth' });
}
function startPlay() {
  const rp = state.replay;
  if (!rp) return;
  if (rp.s >= rp.last) setStep(0);
  rp.playing = true;
  rp.follow = false;
  bringTableUp();
  applyStep('keep');
  schedulePlay();
}
function stopPlay() {
  const rp = state.replay;
  if (!rp) return;
  clearTimeout(rp.timer);
  if (rp.playing) { rp.playing = false; applyStep('keep'); }
}
function control(c) {
  const rp = state.replay;
  if (!rp) return;
  if (c === 'play') return rp.playing ? stopPlay() : startPlay();
  stopPlay();
  if (c === 'start') setStep(0);
  else if (c === 'end') setStep(rp.last);
  else if (c === 'prev') setStep(rp.s - 1);
  else if (c === 'next') setStep(rp.s + 1);
}

function toggleMind() {
  state.mind = !state.mind;
  store.set('mind', state.mind);
  document.querySelector('.pair')?.classList.toggle('show-mind', state.mind);
  document.getElementById('mind-toggle')?.setAttribute('aria-pressed', String(state.mind));
}

document.addEventListener('keydown', e => {
  if (!state.view || e.metaKey || e.ctrlKey || e.altKey || /input|select|textarea|button/i.test(e.target.tagName)) return;
  const v = state.view;
  const order = [...v.played].sort((a, b) => a - b);
  const i = order.indexOf(v.pick);
  if (e.key === 'ArrowRight' || e.key === 'j') { if (i < order.length - 1) go(v.cur, order[i + 1]); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'k') { if (i > 0) go(v.cur, order[i - 1]); e.preventDefault(); }
  else if (e.key === 'r') toggleMind();
  else if (e.key === ' ') { control('play'); e.preventDefault(); }
  else if (e.key === ',') control('prev');
  else if (e.key === '.') control('next');
});

route();
