// Poker Arena viewer: a leaderboard per run, and every duplicate deal replayed side by side with each
// player's reasoning. Works against the live server (/api/...) or a static export (data/*.json).
'use strict';

const STATIC = window.POKER_STATIC || null;
const $app = document.getElementById('app');
const $status = document.getElementById('status');
const $run = document.getElementById('run');
const store = {
  get(k, d) { try { const v = localStorage.getItem('poker-arena.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('poker-arena.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};
const state = { runs: [], run: null, index: [], kinds: {}, via: {}, timer: null, mind: store.get('mind', true),
  matches: new Map(), view: null };

// ---- formatting ---------------------------------------------------------------------------------
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const MINUS = '−';
const signed = (x, d = 0) => x == null ? '' : (x > 0 ? '+' : x < 0 ? MINUS : '') + Math.abs(Number(x)).toFixed(d);
const cls = x => x > 0 ? 'pos' : x < 0 ? 'neg' : '';
const plural = (n, word, many = word + 's') => `${Number(n).toLocaleString()} ${n === 1 ? word : many}`;
const pct = x => x == null ? '<span class="dim">–</span>' : `${Math.round(x * 100)}%`;
const trim = v => Number.isInteger(v) ? String(v) : v.toFixed(1).replace(/\.0$/, '');
const bbs = (chips, bb) => `${trim(chips / bb)} bb`;
const signedBB = (chips, bb) => chips === 0 ? '0 bb' : `${chips > 0 ? '+' : MINUS}${trim(Math.abs(chips) / bb)} bb`;
const secs = s => s == null ? '<span class="dim">–</span>' : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
const compact = n => n == null ? '–' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? Math.round(n / 1e3) + 'K' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(Math.round(n));
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
  if (s < 3600) return `${Math.round(s / 60)} minutes ago`;
  if (s < 86400 * 2) return `${Math.round(s / 3600)} hours ago`;
  return new Date(t * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
const runLabel = r => r === '.' ? 'This folder' : r.split('/').pop();
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

// ---- small pieces -------------------------------------------------------------------------------
const SUIT = { s: '♠', h: '♥', d: '♦', c: '♣' };
function card(c) {
  if (!c) return '<span class="pc slot"></span>';
  const rank = c[0] === 'T' ? '10' : c[0];
  const red = c[1] === 'h' || c[1] === 'd';
  return `<span class="pc ${red ? 'red' : ''}" title="${c}">${rank}<small>${SUIT[c[1]] || c[1]}</small></span>`;
}
const cards = (cs, n = 0, size = '') => {
  const list = [...(cs || [])];
  while (list.length < n) list.push(null);
  return `<span class="cards ${size}">${list.map(card).join('')}</span>`;
};
const isBot = p => state.kinds[p] === 'bot';
const name = p => `${esc(p)}${isBot(p) ? '<span class="tag">bot</span>' : ''}`;
const player = p => `<div class="player"><span class="name">${name(p)}</span>${state.via[p] && !isBot(p)
  ? `<span class="via">${esc(state.via[p])}</span>` : ''}</div>`;
const section = (title, aside, body, caption = '') => `<section><div class="section-head"><h2>${title}</h2>
  ${aside ? `<span class="aside">${aside}</span>` : ''}</div><div class="card-box scroll">${body}</div>
  ${caption ? `<p class="caption">${caption}</p>` : ''}</section>`;
const th = (label, tip, extra = '') => `<th class="${extra}" ${tip ? `title="${esc(tip)}"` : ''}>${label}</th>`;

// ---- routing: #/r/RUN (leaderboard), #/r/RUN/m/MATCHUP?deal=K (matches) -----------------------
async function route() {
  clearTimeout(state.timer);
  const [path, query] = location.hash.replace(/^#\/?/, '').split('?');
  const parts = path.split('/').filter(Boolean);
  try {
    state.runs = await api('runs');
    if (!state.runs.length) {
      $run.parentElement.hidden = true;
      $app.innerHTML = `<div class="card-box empty"><h2>No games yet</h2>
        <p>Play a match or a tournament, and it shows up here as it plays:</p>
        <p><code>python -m poker_arena play rule station --deals 20 --out runs/demo</code></p></div>`;
      return;
    }
    $run.parentElement.hidden = false;
    const run = parts[0] === 'r' && parts[1] ? decodeURIComponent(parts[1]) : null;
    if (!run || !state.runs.find(r => r.run === run)) {
      location.replace(runLink(state.runs[0].run));
      return;
    }
    state.run = run;
    $run.innerHTML = state.runs.map(r => `<option value="${esc(r.run)}" ${r.run === run ? 'selected' : ''}>
      ${esc(runLabel(r.run))}${r.live ? ' · live' : ''}</option>`).join('');
    $run.onchange = () => { location.hash = runLink($run.value); };
    const tab = parts[2] === 'm' ? 'matches' : 'board';
    document.querySelectorAll('nav a').forEach(a => {
      a.classList.toggle('active', a.dataset.tab === tab);
      a.href = runLink(run, a.dataset.tab === 'matches' ? '/m' : '');
    });
    state.index = await api('index', ['run', run]);
    for (const m of state.index) { Object.assign(state.kinds, m.kinds || {}); Object.assign(state.via, m.via || {}); }
    if (tab === 'matches') {
      await renderMatches(parts[3] ? decodeURIComponent(parts.slice(3).join('/')) : '', new URLSearchParams(query || ''));
    } else {
      state.view = null;
      await renderBoard();
    }
    const cur = state.runs.find(r => r.run === run);
    const playing = state.index.filter(m => m.live).length;
    $status.innerHTML = STATIC ? `<span class="label">Snapshot from ${new Date(STATIC.exported * 1000).toLocaleString()}</span>`
      : playing ? `<span class="live-dot"></span><span>Live<span class="label"> · ${plural(playing, 'matchup')} playing</span></span>`
      : cur.finished ? '' : `<span class="label">Paused · last hand ${ago(cur.updated)}</span>`;
    if (playing && !STATIC) state.timer = setTimeout(route, 6000);
  } catch (e) {
    $app.innerHTML = `<div class="card-box empty"><h2>Couldn't load this page</h2><p class="error">${esc(e.message)}</p></div>`;
  }
}
window.addEventListener('hashchange', route);

// ---- leaderboard --------------------------------------------------------------------------------
function runHead() {
  const r = state.runs.find(x => x.run === state.run);
  const deals = Math.max(0, ...state.index.map(m => m.deals_total || 0));
  const sub = [plural(r.players.length, 'player'), plural(r.matchups, 'matchup'), `${deals} deals each, played from both seats`];
  const share = r.hands_planned ? r.hands / r.hands_planned : 0;
  let line;
  if (r.finished) line = `Finished · ${plural(r.hands, 'hand')} · last hand ${ago(r.updated)}`;
  else {
    line = `${r.hands.toLocaleString()} of ${plural(r.hands_planned, 'hand')}`;
    if (r.eta_seconds != null) line += ` · about ${duration(r.eta_seconds)} left`;
    else if (!r.live) line += ` · paused, last hand ${ago(r.updated)}`;
  }
  return `<div class="page-head"><div><h1>${esc(runLabel(state.run))}</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="meter"><span style="width:${(share * 100).toFixed(1)}%"></span></div>
      <div class="line">${line}${r.hands_per_hour ? ` · ${r.hands_per_hour.toLocaleString()} hands an hour` : ''}</div></div></div>`;
}

async function renderBoard() {
  const r = await api('ratings', ['run', state.run]);
  let html = runHead();
  if (!r.players.length) {
    $app.innerHTML = html + `<div class="card-box empty"><h2>Nothing to score yet</h2>
      <p>A deal counts once both of its plays are finished. The first ones are on the <a href="${runLink(state.run, '/m')}"><u>Matches</u></a> page as they play.</p></div>`;
    return;
  }
  const rated = r.players.filter(p => p.rated);
  const rest = r.players.filter(p => !p.rated);
  const per = r.session_deals, need = r.min_sessions;
  if (!rated.length) {
    const most = Math.max(...r.players.map(p => p.sessions));
    html += `<div class="notice"><div><b>Too early to rank.</b> A rating needs ${plural(need, 'session')} per player,
      and a session is ${per} duplicate deals (${2 * per} hands). The most-played player has ${most}. Below is what has
      happened so far, which is not yet a ranking.</div></div>`;
  }
  if (rated.length) html += ratingsTable(r, rated);
  if (rest.length) html += unratedTable(r, rest, rated.length > 0);
  html += headToHead(r);
  html += styleTable(r);
  html += reliabilityTable(r);
  $app.innerHTML = html;
}

function bbCell(p) {
  if (p.bb_per_100 == null) return '<span class="dim">–</span>';
  const ci = p.bb_per_100_ci95;
  const clear = ci && (ci[0] > 0 || ci[1] < 0);
  const margin = ci ? Math.round((ci[1] - ci[0]) / 2) : null;
  return `<span class="${clear ? cls(p.bb_per_100) : 'soft'}" style="font-weight:600">${signed(p.bb_per_100, 0)}</span>
    ${margin != null ? `<span class="ci">±${margin}</span>` : ''}`;
}

function ratingsTable(r, rated) {
  const lo0 = Math.min(...rated.map(p => p.elo90[0])), hi0 = Math.max(...rated.map(p => p.elo90[1]));
  const lo = Math.floor((lo0 - 20) / 50) * 50, hi = Math.ceil((hi0 + 20) / 50) * 50;
  const x = v => ((v - lo) / (hi - lo) * 100).toFixed(2) + '%';
  const centre = 1500 > lo && 1500 < hi ? `<span class="mid" style="left:${x(1500)}"></span>` : '';
  const rows = rated.map((p, i) => `<tr>
    <td class="rank">${i + 1}</td><td>${player(p.player)}</td>
    <td class="num"><span class="elo">${p.elo}</span></td>
    <td class="axis-cell"><div class="whisker" title="90% interval ${p.elo90[0]}–${p.elo90[1]}"><span class="track"></span>${centre}
      <span class="range" style="left:${x(p.elo90[0])};width:calc(${x(p.elo90[1])} - ${x(p.elo90[0])})"></span>
      <span class="point" style="left:${x(p.elo)}"></span></div></td>
    <td class="num ci">${p.elo90[0]}–${p.elo90[1]}</td>
    <td class="num">${bbCell(p)}</td>
    <td class="num">${p.session_wins}–${p.session_ties}–${p.session_losses}</td>
    <td class="num soft">${p.hands.toLocaleString()}</td></tr>`).join('');
  const table = `<table><thead><tr><th></th><th>Player</th>${th('Elo', '', 'num')}
    <th class="axis-cell"><div class="axis-labels"><span>${lo}</span>${centre ? '<span>1500</span>' : ''}<span>${hi}</span></div></th>
    ${th('90% range', 'Where the rating lands in 90% of bootstrap resamples of the sessions', 'num')}
    ${th('bb/100', 'Big blinds won per 100 hands, ± its 95% margin', 'num')}
    ${th('Sessions', `Sessions won, tied and lost. A session is ${r.session_deals} duplicate deals`, 'num')}
    ${th('Hands', '', 'num')}</tr></thead><tbody>${rows}</tbody></table>`;
  return section('Ratings', `${plural(r.sessions, 'session')} · ${plural(r.deals, 'duplicate deal')}`, table,
    `One Bradley–Terry fit over every session of ${r.session_deals} duplicate deals (${2 * r.session_deals} hands). A gap of
    200 points means the higher player wins about 76% of sessions; a session within 1 big blind is a tie. bb/100 is coloured
    only when its 95% range excludes zero.`);
}

function unratedTable(r, rest, others) {
  const need = r.min_sessions;
  const rows = rest.map(p => `<tr><td>${player(p.player)}</td>
    <td class="num ${cls(p.net_bb)}" style="font-weight:600">${signed(p.net_bb, 0)} bb</td>
    <td class="num">${bbCell(p)}</td><td class="num soft">${p.hands.toLocaleString()}</td>
    <td><span class="meter thin"><span style="width:${Math.min(100, p.sessions / need * 100)}%"></span></span>
      <span class="small soft" style="margin-left:8px">${p.sessions} of ${need} sessions</span></td></tr>`).join('');
  return section(others ? 'Not yet rated' : 'Results so far', '', `<table><thead><tr><th>Player</th>${th('Net', 'Big blinds won over every duplicate deal', 'num')}
    ${th('bb/100', 'Big blinds won per 100 hands, ± its 95% margin', 'num')}${th('Hands', '', 'num')}<th>Toward a rating</th></tr></thead>
    <tbody>${rows}</tbody></table>`,
    'A player is rated once it has played enough sessions for the number to mean something. Until then its net result is mostly the luck of a few big pots.');
}

function headToHead(r) {
  if (!r.matchups.length) return '';
  const names = r.players.map(p => p.player);
  const cell = {};
  for (const m of r.matchups) {
    cell[m.a + '|' + m.b] = { v: m.a_bb_per_100, se: m.se, hands: m.hands, id: m.matchup };
    cell[m.b + '|' + m.a] = { v: -m.a_bb_per_100, se: m.se, hands: m.hands, id: m.matchup };
  }
  const ids = Object.fromEntries(state.index.map(m => [m.id.split('/').pop(), m.id]));
  const max = Math.max(1, ...r.matchups.map(m => Math.abs(m.a_bb_per_100)));
  const table = `<table class="matrix"><thead><tr><th></th>${names.map(n => `<th>${esc(n)}</th>`).join('')}</tr></thead><tbody>
    ${names.map(a => `<tr><th>${name(a)}</th>${names.map(b => {
      if (a === b) return '<td class="cell self"></td>';
      const c = cell[a + '|' + b];
      if (!c) return '<td class="cell dim">·</td>';
      const clear = Math.abs(c.v) > 1.96 * c.se;
      const alpha = clear ? (0.08 + 0.22 * Math.sqrt(Math.abs(c.v) / max)).toFixed(2) : 0;
      const bg = clear ? `background:rgba(var(--${c.v > 0 ? 'pos' : 'neg'}-rgb),${alpha})` : '';
      const link = ids[c.id] ? matchLink(ids[c.id]) : '';
      return `<td class="cell" style="${bg}"><a href="${link}" title="${plural(c.hands, 'hand')}, ±${Math.round(1.96 * c.se)} bb/100">
        <span class="v ${clear ? cls(c.v) : 'soft'}">${signed(c.v, 0)}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  return section('Head to head', 'bb/100, row player against column player', table,
    'Shaded when the edge is larger than its 95% margin. Click a cell to replay that matchup.');
}

function styleTable(r) {
  const st = r.stats || {};
  const names = r.players.map(p => p.player).filter(p => st[p] && st[p].hands);
  if (!names.length) return '';
  const bar = v => v == null ? '' : `<span class="bar"><span style="width:${Math.round(v * 100)}%"></span></span>`;
  const rows = names.map(p => {
    const s = st[p];
    const af = s.af_all_aggression ? '<span title="bet or raised after the flop and never called">∞</span>'
      : s.af == null ? '<span class="dim">–</span>' : s.af.toFixed(1);
    return `<tr><td>${player(p)}</td>
      <td class="num">${pct(s.vpip)}${bar(s.vpip)}</td><td class="num">${pct(s.pfr)}${bar(s.pfr)}</td>
      <td class="num">${af}</td><td class="num">${pct(s.wtsd)}</td><td class="num">${pct(s.wsd)}</td>
      <td class="num">${isBot(p) ? '<span class="dim">–</span>' : pct(s.talk)}</td>
      <td class="num soft">${s.hands.toLocaleString()}</td></tr>`;
  }).join('');
  return section('How they play', '', `<table><thead><tr><th>Player</th>
    ${th('VPIP', 'Voluntarily put money in the pot: called or raised before the flop', 'num')}
    ${th('PFR', 'Raised before the flop', 'num')}
    ${th('Aggression', 'After the flop: bets and raises per call', 'num')}
    ${th('To showdown', 'Of the hands where it saw a flop, how many reached showdown', 'num')}
    ${th('Won at showdown', 'Of its showdowns, how many it won money in', 'num')}
    ${th('Table talk', 'Share of its decisions that came with something said to the opponent', 'num')}
    ${th('Hands', '', 'num')}</tr></thead><tbody>${rows}</tbody></table>`,
    `<b>VPIP</b> and <b>PFR</b>: how often it plays a hand, and how often it raises with it. Heads-up, strong players are
    usually well above 50%. <b>Aggression</b> above 2 means it bets far more than it calls after the flop.`);
}

function reliabilityTable(r) {
  const st = r.stats || {};
  const names = r.players.map(p => p.player).filter(p => !isBot(p) && st[p] && st[p].calls);
  if (!names.length) return '';
  const count = (n, cls_, tip) => n ? `<span class="badge ${cls_}" title="${tip}">${n.toLocaleString()}</span>` : '<span class="dim">0</span>';
  const rows = names.map(p => {
    const s = st[p];
    const per = (n) => n && s.decisions ? compact(n / s.decisions) : '–';
    return `<tr><td>${player(p)}</td><td class="num">${s.decisions.toLocaleString()}</td>
      <td class="num">${secs(s.latency_mean)}</td><td class="num soft">${secs(s.latency_max)}</td>
      <td class="num">${count(s.retries, 'warn', 'calls that were a second or third try after an unusable reply')}</td>
      <td class="num">${count(s.fallbacks, 'bad', 'decisions where no usable reply came and the seat checked or folded')}</td>
      <td class="num">${count(s.corrected, 'warn', 'bet sizes that were out of range and got corrected')}</td>
      <td class="num soft">${s.tokens_in ? `${per(s.tokens_in)} in · ${per(s.tokens_out)} out` : '–'}</td>
      <td class="soft small">${s.served_models.length ? s.served_models.map(esc).join(', ') : '<span class="dim">not reported</span>'}</td></tr>`;
  }).join('');
  return section('Reliability', 'model players', `<table><thead><tr><th>Player</th>${th('Decisions', '', 'num')}
    ${th('Mean reply', 'Average seconds per CLI call', 'num')}${th('Slowest', '', 'num')}
    ${th('Retries', 'Replies that could not be parsed or played and were sent back', 'num')}
    ${th('Fell back', 'Decisions with no usable reply after the retries: the seat checked or folded', 'num')}
    ${th('Sizes fixed', 'Bet sizes below the minimum or above the stack, corrected and explained to the seat', 'num')}
    ${th('Tokens per decision', 'As reported by the CLI, including its own built-in prompt', 'num')}
    ${th('Answered by', 'The model the CLI says replied (agy reports this)')}</tr></thead><tbody>${rows}</tbody></table>`,
    'A fallback costs the seat its decision, so a player with many of them is being scored partly on its CLI, not its poker.');
}

// ---- matches ------------------------------------------------------------------------------------
async function renderMatches(id, params) {
  if (!state.index.length) {
    $app.innerHTML = '<div class="card-box empty"><h2>No matchups in this run yet</h2></div>';
    return;
  }
  const started = state.index.filter(m => m.started);
  const cur = state.index.find(m => m.id === id) || started.find(m => m.live) || started[0] || state.index[0];
  const groups = [['Playing', state.index.filter(m => m.live)], ['Played', started.filter(m => !m.live)],
    ['Not started', state.index.filter(m => !m.started)]];
  const list = groups.filter(g => g[1].length).map(([title, ms]) => `<div class="group">${title}</div>` + ms.map(m => {
    const meta = !m.started ? 'waiting for a free seat' : `${m.pairs_done} of ${m.deals_total} deals`
      + (m.pairs_done ? ` · <span class="${cls(m.pair_net_a)}">${esc(m.a)} ${signedBB(m.pair_net_a, m.bb)}</span>` : '');
    return `<a href="${m.started ? matchLink(m.id) : ''}" class="${m === cur ? 'active' : ''} ${m.started ? '' : 'planned'}">
      <div class="row"><span class="pair-names">${esc(m.a)}<span class="vs">vs</span>${esc(m.b)}</span>${m.live ? '<span class="live-dot"></span>' : ''}</div>
      <div class="meta">${meta}</div></a>`;
  }).join('')).join('');

  let body;
  try {
    body = cur.started ? await matchBody(cur, params) : `<div class="card-box empty"><h2>${esc(cur.a)} vs ${esc(cur.b)}</h2><p>Not started yet.</p></div>`;
  } catch (e) {
    body = `<div class="card-box empty"><h2>Couldn't load this matchup</h2><p class="error">${esc(e.message)}</p></div>`;
  }
  const scroll = window.scrollY;
  const listScroll = document.querySelector('.mlist')?.scrollTop || 0;
  $app.innerHTML = `<div class="layout"><div class="card-box mlist">${list}</div><div>${body}</div></div>`;
  window.scrollTo(0, scroll);
  const ml = document.querySelector('.mlist');
  if (ml) ml.scrollTop = listScroll;
  $app.querySelectorAll('[data-deal]').forEach(b => b.onclick = () => go(cur, Number(b.dataset.deal)));
  const t = document.getElementById('mind-toggle');
  if (t) t.onclick = toggleMind;
}

function go(cur, k) {
  if (!state.view || k < 0 || k >= state.view.total || !state.view.played.has(k)) return;
  history.replaceState(null, '', matchLink(cur.id, k));
  route();
}

function toggleMind() {
  state.mind = !state.mind;
  store.set('mind', state.mind);
  document.querySelector('.pair')?.classList.toggle('hide-mind', !state.mind);
  const t = document.getElementById('mind-toggle');
  if (t) t.setAttribute('aria-pressed', String(state.mind));
}

document.addEventListener('keydown', e => {
  if (!state.view || e.metaKey || e.ctrlKey || e.altKey || /input|select|textarea/i.test(e.target.tagName)) return;
  const v = state.view;
  const order = [...v.played].sort((a, b) => a - b);
  const i = order.indexOf(v.pick);
  if (e.key === 'ArrowRight' || e.key === 'j') { if (i < order.length - 1) go(v.cur, order[i + 1]); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'k') { if (i > 0) go(v.cur, order[i - 1]); e.preventDefault(); }
  else if (e.key === 'r') toggleMind();
});

async function matchBody(cur, params) {
  const [ma, mb] = await Promise.all(cur.matches.map(x => loadMatch(x.id)));
  const bb = ma.config.game.big_unit || ma.config.game.bb || 1;
  const ha = Object.fromEntries(ma.hands.map(h => [h.hand, h]));
  const hb = mb ? Object.fromEntries(mb.hands.map(h => [h.hand, h])) : {};
  const total = ma.config.seeds.length;
  const played = new Set([...Object.keys(ma.events), ...Object.keys(mb ? mb.events : {})].map(Number));
  const done = [...Array(total).keys()].filter(k => ha[k] && hb[k]);
  const nets = Object.fromEntries(done.map(k => [k, ha[k].net[0] + hb[k].net[1]]));
  const wanted = params.has('deal') ? Number(params.get('deal')) - 1 : null;
  const pick = wanted != null && played.has(wanted) ? wanted : played.size ? Math.max(...played) : 0;
  state.view = { cur, pick, total, played };

  // summary over the deals both plays have finished
  const net = done.reduce((s, k) => s + nets[k], 0);
  const wins = done.filter(k => nets[k] > 0).length, losses = done.filter(k => nets[k] < 0).length;
  const stats = `<div class="score">
    <div class="stat"><div class="k">${esc(cur.a)} net</div><div class="v ${cls(net)}">${done.length ? signedBB(net, bb) : '–'}</div></div>
    <div class="stat"><div class="k">bb/100</div><div class="v ${cls(net)}">${done.length ? signed(net / bb / (2 * done.length) * 100, 0) : '–'}</div></div>
    <div class="stat"><div class="k">Deals won</div><div class="v">${wins}<span class="dim"> – </span>${losses}${done.length - wins - losses
      ? `<span class="small soft" style="font-weight:500"> · ${done.length - wins - losses} even</span>` : ''}</div></div>
    <div class="stat"><div class="k">Played</div><div class="v">${done.length}<span class="small soft" style="font-weight:500"> of ${total} deals</span></div></div></div>`;

  const max = Math.max(1, ...done.map(k => Math.abs(nets[k])));
  const strip = Array.from({ length: total }, (_, k) => {
    const sel = k === pick ? 'sel' : '';
    if (!(k in nets)) return `<button ${played.has(k) ? `data-deal="${k}" class="half ${sel}" title="Deal ${k + 1}: one play finished"` : 'disabled'}>${k + 1}</button>`;
    const v = nets[k];
    const a = v === 0 ? 0 : 0.14 + 0.76 * Math.sqrt(Math.abs(v) / max);
    const style = v === 0 ? '' : `background:rgba(var(--${v > 0 ? 'pos' : 'neg'}-rgb),${a.toFixed(2)});${a > 0.5 ? 'color:#fff' : ''}`;
    return `<button data-deal="${k}" class="${sel}" style="${style}" title="Deal ${k + 1}: ${esc(cur.a)} ${signedBB(v, bb)} over both plays">${k + 1}</button>`;
  }).join('');
  const legend = `<div class="strip-legend">
    <span><span class="swatch" style="background:rgba(var(--pos-rgb),.7)"></span>${esc(cur.a)} won the deal</span>
    <span><span class="swatch" style="background:rgba(var(--neg-rgb),.7)"></span>${esc(cur.b)} won it</span>
    <span>darker is a bigger pot</span><span class="keys" style="margin-left:auto"><span class="kbd">←</span> <span class="kbd">→</span> move between deals ·
    <span class="kbd">r</span> reasoning</span></div>`;

  const order = [...played].sort((x, y) => x - y);
  const i = order.indexOf(pick);
  const dealNet = pick in nets ? nets[pick] : null;
  const bar = `<div class="deal-bar"><h2>Deal ${pick + 1}</h2>
    <div class="nav"><button data-deal="${order[i - 1]}" ${i > 0 ? '' : 'disabled'} aria-label="Previous deal">←</button>
      <button data-deal="${order[i + 1]}" ${i < order.length - 1 ? '' : 'disabled'} aria-label="Next deal">→</button></div>
    <span class="soft">${dealNet == null ? 'in progress' : dealNet === 0 ? 'even over both plays'
      : `<b class="${cls(dealNet)}">${esc(cur.a)} ${signedBB(dealNet, bb)}</b> over both plays`}</span>
    <span class="spacer"></span>
    <button class="toggle" id="mind-toggle" aria-pressed="${state.mind}" title="Show each decision's private reasoning (r)">Reasoning</button></div>`;

  return `<div class="page-head"><div><h1>${name(cur.a)} <span class="dim" style="font-weight:400">vs</span> ${name(cur.b)}</h1>
      <div class="sub">Every deal is played twice with the same cards, once from each seat, so the luck of the cards cancels out.</div></div></div>
    ${stats}<div class="strip-wrap"><div class="strip" style="--cols:${total <= 30 ? total : Math.ceil(total / Math.ceil(total / 25))}">${strip}</div>${legend}</div>${bar}
    <div class="pair ${state.mind ? '' : 'hide-mind'}">${playPanel(ma, pick, 'Play 1', bb)}${mb ? playPanel(mb, pick, 'Play 2 · seats swapped', bb) : ''}</div>`;
}

const VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to' };

function playPanel(m, k, title, bb) {
  const evs = m.events[k] || [];
  const start = evs.find(e => e.kind === 'hand_start');
  if (!start) return `<div class="card-box play"><div class="play-head"><h3>${title}</h3></div><div class="pending">Not dealt yet.</div></div>`;
  const d = start.data, players = m.config.players, aliases = m.config.names;
  const result = evs.find(e => e.kind === 'result')?.data;
  const role = i => i === d.button ? 'button, small blind' : 'big blind';
  const seats = [0, 1].map(i => `<div class="seat ${result && result.net[i] > 0 ? 'winner' : ''}">
      <div class="who"><div class="name">${name(players[i])}${i === d.button ? '<span class="chip" title="button">D</span>' : ''}</div>
        <div class="alias">as ${esc(aliases[i])} · ${role(i)}</div></div>
      ${cards(d.hole[i], 2)}
      <div class="net ${result ? cls(result.net[i]) : ''}">${result ? signedBB(result.net[i], bb) : ''}</div></div>`).join('');
  const board = result?.board || [...evs].reverse().find(e => e.kind === 'board')?.data.board || [];

  let log = '', thought = null, pot = 0;
  const notes = {};
  const short = i => esc(players[i]);
  const blinds = evs.filter(e => e.kind === 'blind').map(e => `${short(e.data.seat)} ${bbs(e.data.amount, bb)}`);
  log += `<div class="street">Preflop</div><div class="ev quiet"><span class="actor">Blinds</span><span class="what"><span class="verb">${blinds.join(' · ')}</span></span></div>`;
  pot = evs.filter(e => e.kind === 'blind').reduce((s, e) => s + e.data.amount, 0);
  for (const e of evs) {
    const x = e.data;
    if (e.kind === 'thought') { thought = x; if (x.notes) notes[x.seat] = x.notes; }
    else if (e.kind === 'action') {
      pot = x.pot ?? pot;
      const t = thought && thought.seat === x.seat ? thought : null;
      thought = null;
      const amt = x.action === 'call' || x.action === 'bet' || x.action === 'raise' ? ` ${bbs(x.amount, bb)}` : '';
      const badges = [];
      if (x.all_in) badges.push('<span class="badge strong">all in</span>');
      if (t?.fallback) badges.push('<span class="badge bad" title="No usable reply after the retries: the seat checked or folded">fell back</span>');
      else if (t && t.tries > 1) badges.push(`<span class="badge warn" title="The first reply couldn't be used and was sent back">${t.tries} tries</span>`);
      if (x.note) badges.push(`<span class="badge warn" title="${esc(x.note)}">size fixed</span>`);
      if (t?.served_model) badges.push(`<span class="badge" title="The model the CLI says answered">${esc(t.served_model)}</span>`);
      if (t?.seconds) badges.push(`<span class="badge" title="Time the CLI took">${t.seconds < 10 ? t.seconds.toFixed(1) : Math.round(t.seconds)} s</span>`);
      const mind = t && (t.reasoning || t.thinking) ? `<div class="mind">${t.reasoning ? esc(t.reasoning) : ''}${t.thinking
        ? `<span class="lbl">Thinking from the CLI</span><span class="thinking">${esc(t.thinking)}</span>` : ''}</div>` : '';
      log += `<div class="ev"><span class="actor" title="${esc(players[x.seat])} as ${esc(aliases[x.seat])}">${short(x.seat)}</span>
        <span class="what"><span class="verb ${x.action}">${VERB[x.action] || esc(x.action)}${amt}</span>${badges.join('')}</span>
        ${x.say ? `<span class="say">${esc(x.say)}</span>` : ''}${mind}</div>`;
    } else if (e.kind === 'board') {
      log += `<div class="street">${esc(x.street)} ${cards(x.cards, 0, 'sm')}<span class="pot">pot ${bbs(pot, bb)}</span></div>`;
    } else if (e.kind === 'runout') {
      log += `<div class="ev quiet"><span class="actor"></span><span class="what"><span class="verb">All in, so the rest of the board is dealt.</span></span></div>`;
    } else if (e.kind === 'showdown') {
      log += `<div class="street">Showdown</div>` + Object.entries(x.hands).map(([i, h]) =>
        `<div class="showdown"><span class="actor">${short(Number(i))}</span>${cards(h.cards, 0, 'sm')}<span class="soft">${esc(h.best)}</span></div>`).join('');
    }
  }
  if (result) {
    const winners = result.won.map((w, i) => w > 0 ? i : null).filter(i => i != null);
    const how = result.showdown ? 'at showdown' : 'without a showdown';
    log += `<div class="outcome">${winners.length === 2 ? `Split pot ${how}` : `<b>${short(winners[0])}</b> wins ${bbs(result.pot, bb)} ${how}`}</div>`;
  }
  const noteBlock = Object.keys(notes).length ? `<div class="notes mind">${Object.entries(notes).map(([i, n]) =>
    `<span class="lbl">${short(Number(i))}'s notes on its opponent</span>${esc(n)}`).join('')}</div>` : '';
  return `<div class="card-box play"><div class="play-head"><h3>${title}</h3></div>
    <div class="seats">${seats}</div>
    <div class="boardline">${cards(board, 5)}<div class="pot">${result ? `Pot <b>${bbs(result.pot, bb)}</b>` : `<span class="live-dot"></span> playing`}</div></div>
    <div class="log">${log}</div>${noteBlock}</div>`;
}

route();
