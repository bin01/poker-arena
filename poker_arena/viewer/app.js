// Poker Arena viewer: the standings of a run, and every duplicate deal replayed at the table with each
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
const state = { runs: [], run: null, index: [], kinds: {}, labels: {}, timer: null, mind: store.get('mind', true),
  matches: new Map(), view: null, shown: null, replay: null };

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

// ---- cards, chips and names ---------------------------------------------------------------------
const SUIT = { s: '♠', h: '♥', d: '♦', c: '♣' };
function card(c, i = 0) {
  if (!c) return '<span class="pc slot"></span>';
  const rank = c[0] === 'T' ? '10' : c[0];
  const red = c[1] === 'h' || c[1] === 'd';
  return `<span class="pc ${red ? 'red' : ''}" style="--i:${i}" title="${c}"><b>${rank}</b><i>${SUIT[c[1]] || c[1]}</i></span>`;
}
const cards = (cs, n = 0, size = '', step = null) => {
  const list = [...(cs || [])];
  while (list.length < n) list.push(null);
  return `<span class="cards ${size}">${list.map((c, i) => step && c ? card(c, i).replace('<span class="pc', `<span data-step="${step[i] ?? ''}" class="pc`) : card(c, i)).join('')}</span>`;
};
const isBot = p => state.kinds[p] === 'bot';
const labelText = p => state.labels[p] || p;
const label = p => esc(labelText(p));   // the display name from players.toml, else the player id
// A round badge per player: its initials, coloured by model family.
function monogram(p, size = '') {
  const text = labelText(p);
  const words = text.replace(/\(.*?\)/g, '').split(/[\s_-]+/).filter(w => /^[A-Za-z]/.test(w) && !/^bot$/i.test(w));
  const initials = words.length > 1 ? (words[0][0] + words[1][0]).toUpperCase() : (words[0] || text).slice(0, 2).replace(/^./, ch => ch.toUpperCase());
  const key = (text + ' ' + p).toLowerCase();
  let color;
  if (isBot(p)) color = '#5d655f';
  else if (/claude|opus|sonnet|haiku|anthropic/.test(key)) color = '#c8643f';
  else if (/gpt|codex|openai|o\d/.test(key)) color = '#0f8f6c';
  else if (/gemini|google|agy/.test(key)) color = '#3b73d9';
  else if (/llama|meta/.test(key)) color = '#6c52d9';
  else if (/grok|xai/.test(key)) color = '#444';
  else { let h = 0; for (const ch of key) h = (h * 31 + ch.charCodeAt(0)) % 360; color = `hsl(${h} 45% 42%)`; }
  return `<span class="mono ${size}" style="--c:${color}" aria-hidden="true">${esc(initials)}</span>`;
}
const name = p => `<span class="nm" title="${esc(p)}">${label(p)}</span>${isBot(p) ? '<span class="tag">bot</span>' : ''}`;
const player = p => `<div class="player">${monogram(p)}<span class="name">${name(p)}</span></div>`;
const SUITS = ['♠', '♥', '♦', '♣'];
let sectionCount = 0;
const section = (title, aside, body, caption = '') => {
  const suit = SUITS[sectionCount++ % 4];
  return `<section><div class="section-head"><h2><span class="suit ${suit === '♥' || suit === '♦' ? 'red' : ''}">${suit}</span>${title}</h2>
  ${aside ? `<span class="aside">${aside}</span>` : ''}</div><div class="card-box scroll">${body}</div>
  ${caption ? `<p class="caption">${caption}</p>` : ''}</section>`;
};
const th = (text, tip, extra = '') => `<th class="${extra}" ${tip ? `title="${esc(tip)}"` : ''}>${text}</th>`;

// ---- routing: #/r/RUN (standings), #/r/RUN/m/MATCHUP?deal=K (matches) ---------------------------
async function route() {
  clearTimeout(state.timer);
  stopReplay();
  const [path, query] = location.hash.replace(/^#\/?/, '').split('?');
  const parts = path.split('/').filter(Boolean);
  // a new page (not a live refresh or another deal of the same matchup): show that it's coming, then start at the top
  const page = parts.slice(0, 4).join('/');
  const fresh = page !== state.page;
  state.page = page;
  if (fresh) $app.classList.add('busy');
  try {
    state.runs = await api('runs');
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
      // the newest run, not the all-runs history, is where a visit starts
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
    document.querySelectorAll('nav a').forEach(a => {
      a.classList.toggle('active', a.dataset.tab === tab);
      a.href = runLink(run, a.dataset.tab === 'matches' ? '/m' : '');
    });
    state.index = await api('index', ['run', run]);
    state.labels = {}; state.kinds = {};   // per run: an id can name another model, or level, in another run
    for (const m of state.index) { Object.assign(state.kinds, m.kinds || {}); Object.assign(state.labels, m.labels || {}); }
    sectionCount = 0;
    if (tab === 'matches') {
      await renderMatches(parts[3] ? decodeURIComponent(parts.slice(3).join('/')) : '', new URLSearchParams(query || ''));
    } else {
      state.view = null;
      await renderBoard();
    }
    const cur = state.runs.find(r => r.run === run);
    const playing = state.index.filter(m => m.live).length;
    $status.innerHTML = STATIC ? `<span class="label">Snapshot · ${new Date(STATIC.exported * 1000).toLocaleDateString()}</span>`
      : playing ? `<span class="live-dot"></span><span>Live<span class="label"> · ${plural(playing, 'table')} playing</span></span>`
      : cur.finished ? '' : `<span class="label">Paused · last hand ${ago(cur.updated)}</span>`;
    if (playing && !STATIC) state.timer = setTimeout(route, 6000);
    if (fresh) window.scrollTo(0, 0);
  } catch (e) {
    $app.innerHTML = `<div class="card-box empty"><h2>Couldn't load this page</h2><p class="error">${esc(e.message)}</p></div>`;
  }
  $app.classList.remove('busy');
}
window.addEventListener('hashchange', route);

// ---- standings ----------------------------------------------------------------------------------
function runHead(rt) {
  const r = state.runs.find(x => x.run === state.run);
  if (r.all) return historyHead(r, rt);
  const deals = Math.max(0, ...state.index.map(m => m.deals_total || 0));
  const rounds = state.index.some(m => m.round > 1);
  const pairings = new Set(state.index.map(m => [m.a, m.b].sort().join('|'))).size;   // rounds of a pairing count once
  const sub = [plural(r.players.length, 'player'), plural(pairings, 'matchup'),
    rounds ? `rounds of ${deals} duplicate deals, until settled` : `${deals} duplicate deals each`];
  const share = r.hands_planned ? r.hands / r.hands_planned : 0;
  let line;
  if (r.finished) line = `Finished · ${plural(r.hands, 'hand')} · last hand ${ago(r.updated)}`;
  else {
    line = `${r.hands.toLocaleString()} of ${plural(r.hands_planned, 'hand')}`;
    if (r.eta_seconds != null) line += ` · about ${duration(r.eta_seconds)} left`;
    else if (!r.live) line += ` · paused, last hand ${ago(r.updated)}`;
  }
  return `<div class="page-head"><div><div class="eyebrow">Tournament</div><h1>${esc(runLabel(state.run))}</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="meter"><span style="width:${(share * 100).toFixed(1)}%"></span></div>
      <div class="line">${line}${r.hands_per_hour ? ` · ${r.hands_per_hour.toLocaleString()} hands an hour` : ''}</div></div></div>`;
}

// The all-runs page: every run pooled, one player per model and reasoning level.
function historyHead(r, rt) {
  const models = rt.players.filter(p => !isBot(p.player)).length;
  const sub = [plural(rt.runs.length, 'run'), plural(models, 'model'), plural(rt.players.length - models, 'bot'),
    plural(2 * rt.deals, 'duplicate hand')];
  const notes = [];
  if (rt.left_out.length) notes.push(`Left out, because they were played under other game settings: ${rt.left_out.map(x => esc(runLabel(x.run))).join(', ')}.`);
  if (rt.code_versions.length > 1) notes.push(`Played on ${rt.code_versions.length} versions of the arena (${rt.code_versions.map(esc).join(', ')}).`);
  if (rt.unversioned_runs.length) notes.push(`${plural(rt.unversioned_runs.length, 'run')} from before versions were recorded: ${rt.unversioned_runs.map(x => esc(runLabel(x))).join(', ')}.`);
  const gone = rt.runs.filter(x => !state.runs.some(y => y.run === x.run)).length;
  if (gone) notes.push(`${plural(gone, 'run')} kept from the history after ${gone === 1 ? 'its folder was' : 'their folders were'} deleted.`);
  const self = rt.runs.reduce((n, x) => n + x.self_play, 0);
  if (self) notes.push(`${plural(self, 'deal')} between two entries of the same model and level don't count.`);
  return `<div class="page-head"><div><div class="eyebrow">History</div><h1>All runs</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="line">One player per model and reasoning level, pooled across every run
      ${r.started_at ? `since ${new Date(r.started_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}</div>
      ${notes.length ? `<div class="line soft small">${notes.join(' ')}</div>` : ''}</div></div>`;
}

async function renderBoard() {
  const r = await api('ratings', ['run', state.run]);
  Object.assign(state.labels, r.labels || {});
  Object.assign(state.kinds, r.kinds || {});
  let html = runHead(r);
  if (!r.players.length) {
    $app.innerHTML = html + `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>The cards are in the air</h2>
      <p>A deal counts once both of its plays are finished. Watch the first ones on the <a href="${runLink(state.run, '/m')}"><u>Matches</u></a> page.</p></div>`;
    return;
  }
  const rated = r.players.filter(p => p.rated);
  const rest = r.players.filter(p => !p.rated);
  const per = r.session_deals, need = r.min_sessions;
  if (!rated.length) {
    const most = Math.max(...r.players.map(p => p.sessions));
    html += `<div class="notice"><span class="suit">♣</span><div><b>Too early to rank.</b> A rating needs ${plural(need, 'session')} per player,
      and a session is ${per} duplicate deals (${2 * per} hands). The most-played player has ${most}. Below is what has
      happened so far, which is not yet a ranking.</div></div>`;
  }
  if (rated.length) html += podium(rated) + ratingsTable(r, rated);
  if (rest.length) html += unratedTable(r, rest, rated.length > 0);
  html += settledSection(r);
  html += headToHead(r);
  html += styleTable(r);
  $app.innerHTML = html;
}

// The top three as a fanned hand of cards: ace, king, queen.
function podium(rated) {
  const top = rated.slice(0, 3);
  const ranks = [['A', '♠', ''], ['K', '♥', 'red'], ['Q', '♦', 'red']];
  const order = top.length === 3 ? [1, 0, 2] : top.length === 2 ? [0, 1] : [0];
  return `<div class="podium n${top.length}">${order.map(i => {
    const p = top[i], [rk, st, red] = ranks[i];
    return `<div class="hero place${i + 1}">
      <span class="idx ${red}">${rk}<i>${st}</i></span><span class="idx br ${red}">${rk}<i>${st}</i></span>
      ${monogram(p.player, 'xl')}
      <div class="hero-name">${label(p.player)}</div>
      <div class="hero-elo">${p.elo}</div>
      <div class="hero-sub">Elo ${p.elo90[0]}–${p.elo90[1]}</div>
      <div class="hero-row"><span><b class="${p.bb_per_100 > 0 ? 'pos' : p.bb_per_100 < 0 ? 'neg' : ''}">${signed(p.bb_per_100, 0)}</b> bb/100</span>
        <span><b>${p.session_wins}–${p.session_ties}–${p.session_losses}</b> sessions</span></div></div>`;
  }).join('')}</div>`;
}

function bbCell(p) {
  if (p.bb_per_100 == null) return '<span class="dim">–</span>';
  const ci = p.bb_per_100_ci95;
  const clear = ci && (ci[0] > 0 || ci[1] < 0);
  const margin = ci ? Math.round((ci[1] - ci[0]) / 2) : null;
  return `<span class="${clear ? cls(p.bb_per_100) : 'soft'}" style="font-weight:650">${signed(p.bb_per_100, 0)}</span>
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
  return section('Standings', `${plural(r.sessions, 'session')} · ${plural(r.deals, 'duplicate deal')}`, table,
    `One Bradley–Terry fit over every session of ${r.session_deals} duplicate deals (${2 * r.session_deals} hands). A gap of
    200 points means the higher player wins about 76% of sessions; a session within 1 big blind is a tie. bb/100 is coloured
    only when its 95% range excludes zero.`);
}

function unratedTable(r, rest, others) {
  const need = r.min_sessions;
  const rows = rest.map(p => `<tr><td>${player(p.player)}</td>
    <td class="num ${cls(p.net_bb)}" style="font-weight:650">${signed(p.net_bb, 0)} bb</td>
    <td class="num">${bbCell(p)}</td><td class="num soft">${p.hands.toLocaleString()}</td>
    <td><span class="meter thin"><span style="width:${Math.min(100, p.sessions / need * 100)}%"></span></span>
      <span class="small soft" style="margin-left:8px">${p.sessions} of ${need} sessions</span></td></tr>`).join('');
  return section(others ? 'Not yet rated' : 'Results so far', '', `<table><thead><tr><th>Player</th>${th('Net', 'Big blinds won over every duplicate deal', 'num')}
    ${th('bb/100', 'Big blinds won per 100 hands, ± its 95% margin', 'num')}${th('Hands', '', 'num')}<th>Toward a rating</th></tr></thead>
    <tbody>${rows}</tbody></table>`,
    'A player is rated once it has played enough sessions for the number to mean something. Until then its net result is mostly the luck of a few big pots.');
}

// How sure are we: each pairing's edge against the bar for calling it, and roughly how many more deals that takes.
const approx = n => n < 100 ? String(n) : Number(n.toPrecision(2)).toLocaleString();
function verdictText(m) {
  if (m.status === 'settled') return '<span class="verdict settled">Settled</span>';
  if (m.status === 'early') return `<span class="verdict">Too early</span><span class="soft small"> · ${m.deals} of ${m.deals + m.more_deals} deals</span>`;
  if (m.status === 'too close') return '<span class="verdict close">Too close to call</span>';
  return `<span class="verdict open">About ${approx(m.more_deals)} more deals</span>`;
}
function winnerLine(r, rated, byPair) {
  const top = (rated.length ? rated : r.players)[0];
  if (!top || r.players.length < 2) return '';
  const others = r.players.filter(p => p.player !== top.player);
  const vs = others.map(o => ({ o, m: byPair[[top.player, o.player].sort().join('|')] }));
  const sign = ({ o, m }) => m && (m.a === top.player ? 1 : -1) * m.a_bb_per_100;
  const missing = vs.filter(v => !v.m), lost = vs.filter(v => v.m?.status === 'settled' && sign(v) < 0);
  const open = vs.filter(v => v.m && !(v.m.status === 'settled' && sign(v) > 0));
  const who = esc(labelText(top.player));
  if (!missing.length && !open.length)
    return `<b>${who} is the winner.</b> It beats every other player, and each of those edges is settled.`;
  if (lost.length) return `<b>${who} leads the ratings, but loses to ${lost.map(v => esc(labelText(v.o.player))).join(' and ')}</b> by a settled margin. The ratings weigh every pairing; this one goes the other way.`;
  const hardest = open.filter(v => v.m.status !== 'too close').sort((x, y) => (y.m.more_deals ?? 0) - (x.m.more_deals ?? 0))[0];
  const close = open.filter(v => v.m.status === 'too close');
  let line = `<b>${who} leads, but it isn't settled yet.</b> `;
  if (close.length) line += `Against ${close.map(v => esc(labelText(v.o.player))).join(' and ')} it's too close to call: they may be about as good as each other. `;
  else if (hardest) line += `The closest is ${esc(labelText(hardest.o.player))}: about ${approx(hardest.m.more_deals)} more deals at the current gap. `;
  if (missing.length) line += `It hasn't played ${missing.map(v => esc(labelText(v.o.player))).join(' or ')}.`;
  return line;
}
function settledSection(r) {
  if (!r.matchups.length) return '';
  const rated = r.players.filter(p => p.rated);
  const byPair = Object.fromEntries(r.matchups.map(m => [[m.a, m.b].sort().join('|'), m]));
  const order = { settled: 0, open: 1, early: 2, 'too close': 3 };
  const rows = [...r.matchups].sort((x, y) => order[x.status] - order[y.status] || (x.more_deals ?? 1e9) - (y.more_deals ?? 1e9))
    .map(m => {
      const up = (m.a_bb_per_100 ?? 0) >= 0, lead = up ? m.a : m.b, trail = up ? m.b : m.a;
      const edge = Math.abs(m.a_bb_per_100 ?? 0);
      const sure = m.z == null ? (m.status === 'settled' ? 1 : 0) : Math.min(1, Math.abs(m.z) / r.z_stop);
      return `<tr><td><div class="player">${monogram(lead)}<span class="name">${name(lead)}</span></div></td>
        <td class="soft small">over</td><td><div class="player">${monogram(trail)}<span class="name">${name(trail)}</span></div></td>
        <td class="num"><b class="${m.status === 'settled' ? 'pos' : ''}">${edge.toFixed(0)}</b>${m.se ? `<span class="ci"> ±${Math.round(1.96 * m.se)}</span>` : ''}</td>
        <td><span class="meter thin sure ${m.status === 'settled' ? 'done' : ''}" title="${m.z == null ? '' : `${Math.abs(m.z).toFixed(1)} of the ${r.z_stop} standard errors needed`}"><span style="width:${(sure * 100).toFixed(0)}%"></span></span></td>
        <td>${verdictText(m)}</td><td class="num soft">${m.deals.toLocaleString()}</td></tr>`;
    }).join('');
  const table = `<table><thead><tr><th>Ahead</th><th></th><th>Behind</th>${th('Edge', 'bb/100 the leader wins, ± its 95% margin', 'num')}
    ${th('How sure', `How far the edge is toward the bar for calling it: ${r.z_stop} standard errors from even`)}<th>Verdict</th>${th('Deals', '', 'num')}</tr></thead>
    <tbody>${rows}</tbody></table>`;
  const line = winnerLine(r, rated, byPair);
  return section('Is it settled?', 'each pairing, over every deal it has played', (line ? `<div class="verdict-line"><span class="suit">♠</span><div>${line}</div></div>` : '') + table,
    `A pairing is settled when the leader's edge is ${r.z_stop} standard errors from even. That's stricter than the usual 1.96,
    because a tournament played until decided checks after every round, and every check is another chance to name the
    wrong winner. "More deals" assumes the current edge holds. Close models need thousands of deals, so "too close to
    call" (over 10,000 more) usually means the two play about as well as each other.`);
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
  const table = `<table class="matrix"><thead><tr><th></th>${names.map(n => `<th title="${label(n)}">${monogram(n, 'sm')}</th>`).join('')}</tr></thead><tbody>
    ${names.map(a => `<tr><th>${player(a)}</th>${names.map(b => {
      if (a === b) return '<td class="cell self"></td>';
      const c = cell[a + '|' + b];
      if (!c) return '<td class="cell dim">·</td>';
      const clear = Math.abs(c.v) > 1.96 * c.se;
      const alpha = clear ? (0.12 + 0.3 * Math.sqrt(Math.abs(c.v) / max)).toFixed(2) : 0;
      const bg = clear ? `background:rgba(var(--${c.v > 0 ? 'pos' : 'neg'}-rgb),${alpha})` : '';
      const full = state.index.some(m => m.id === c.id) ? c.id : ids[c.id];
      const link = full ? matchLink(full) : '';
      return `<td class="cell" style="${bg}"><a href="${link}" title="${label(a)} against ${label(b)}: ${plural(c.hands, 'hand')}, ±${Math.round(1.96 * c.se)} bb/100">
        <span class="v ${clear ? cls(c.v) : 'soft'}">${signed(c.v, 0)}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  return section('Head to head', 'bb/100, row player against column player', table,
    'Shaded when the edge is larger than its 95% margin. Click a cell to watch that matchup.');
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
    return `<tr class="${s.hands < 100 ? 'thin' : ''}"><td>${player(p)}</td>
      <td class="num">${pct(s.vpip)}${bar(s.vpip)}</td><td class="num">${pct(s.pfr)}${bar(s.pfr)}</td>
      <td class="num">${af}</td><td class="num">${pct(s.wtsd)}</td><td class="num">${pct(s.wsd)}</td>
      <td class="num">${isBot(p) ? '<span class="dim">–</span>' : pct(s.talk)}</td>
      <td class="num soft">${s.hands.toLocaleString()}${s.hands < 100 ? '<span class="tag" title="Under 100 hands these numbers swing a lot">few hands</span>' : ''}</td></tr>`;
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
    usually well above 50%. <b>Aggression</b> above 2 means it bets far more than it calls after the flop. Rows with under 100 hands are dimmed: at that size one or two hands move these numbers a lot.`);
}

// ---- matches ------------------------------------------------------------------------------------
async function renderMatches(id, params) {
  if (!state.index.length) {
    $app.innerHTML = '<div class="card-box empty"><h2>No matchups in this run yet</h2></div>';
    return;
  }
  const started = state.index.filter(m => m.started);
  const cur = state.index.find(m => m.id === id) || started.find(m => m.live) || started[0] || state.index[0];
  const own = (m, p) => esc((m.labels || {})[p] || labelText(p));   // across runs, one id can be two players
  Object.assign(state.labels, cur.labels || {});
  const groups = [['Playing now', state.index.filter(m => m.live)], ['Played', started.filter(m => !m.live)],
    ['Not started', state.index.filter(m => !m.started)]];
  const list = groups.filter(g => g[1].length).map(([title, ms]) => `<div class="group">${title}</div>` + ms.map(m => {
    const where = state.run === '*' ? `${esc(runLabel(m.run))} · ` : '';
    const count = (state.index.some(x => x.round > 1 && x.a === m.a && x.b === m.b) ? `round ${m.round} · ` : '') + (where ? `${m.pairs_done}/${m.deals_total} deals` : `${m.pairs_done} of ${m.deals_total} deals`);
    const meta = `<span>${where}${!m.started ? 'waiting for a free seat' : count}</span>`
      + (m.pairs_done ? `<b class="${cls(m.pair_net_a)}" title="${own(m, m.a)}'s net over both plays">${signedBB(m.pair_net_a, m.bb)}</b>` : '');
    return `<a href="${m.started ? matchLink(m.id) : ''}" class="${m === cur ? 'active' : ''} ${m.started ? '' : 'planned'}">
      <div class="row"><span class="duo">${monogram(m.a, 'sm')}${monogram(m.b, 'sm')}</span>
        <span class="pair-names">${own(m, m.a)}<span class="vs">vs</span>${own(m, m.b)}</span>${m.live ? '<span class="live-dot"></span>' : ''}</div>
      <div class="meta">${meta}</div></a>`;
  }).join('')).join('');

  let body;
  try {
    body = cur.started ? await matchBody(cur, params) : `<div class="card-box empty"><h2>${label(cur.a)} vs ${label(cur.b)}</h2><p>Not started yet.</p></div>`;
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
  document.getElementById('mind-toggle')?.addEventListener('click', toggleMind);
  document.getElementById('replay')?.addEventListener('click', startReplay);
  // deal the cards in when a new deal is opened (not on every live refresh)
  const key = cur.id + '#' + state.view?.pick;
  if (state.shown !== key) {
    state.shown = key;
    document.querySelector('.pair')?.classList.add('deal-in');
  }
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
  document.getElementById('mind-toggle')?.setAttribute('aria-pressed', String(state.mind));
}

// Replay: every element of a play carries the step it belongs to; reveal them one step at a time,
// both plays together, the pot following along.
function startReplay() {
  stopReplay();
  const pair = document.querySelector('.pair');
  if (!pair) return;
  pair.classList.remove('deal-in');
  const plays = [...pair.querySelectorAll('.play')];
  const last = Math.max(0, ...[...pair.querySelectorAll('[data-step]')].map(e => Number(e.dataset.step) || 0));
  pair.querySelectorAll('[data-step]').forEach(e => e.classList.add('hidden-step'));
  let step = 0;
  const btn = document.getElementById('replay');
  if (btn) { btn.textContent = '■ Stop'; btn.onclick = () => { stopReplay(true); }; }
  const tick = () => {
    pair.querySelectorAll(`[data-step="${step}"]`).forEach(e => { e.classList.remove('hidden-step'); e.classList.add('revealed'); });
    for (const pl of plays) {
      const pots = JSON.parse(pl.dataset.pots || '[]');
      const now = pots.filter(([s]) => s <= step).pop();
      const potEl = pl.querySelector('.pot-amt');
      if (now && potEl) potEl.textContent = now[1];
    }
    step += 1;
    state.replay = step <= last ? setTimeout(tick, 900) : null;
    if (!state.replay) stopReplay(false);
  };
  tick();
}
function stopReplay(showAll = true) {
  clearTimeout(state.replay);
  state.replay = null;
  const pair = document.querySelector('.pair');
  if (showAll && pair) {
    pair.querySelectorAll('.hidden-step').forEach(e => e.classList.remove('hidden-step'));
    pair.querySelectorAll('.play').forEach(pl => {
      const pots = JSON.parse(pl.dataset.pots || '[]');
      const potEl = pl.querySelector('.pot-amt');
      if (pots.length && potEl) potEl.textContent = pots[pots.length - 1][1];
    });
  }
  const btn = document.getElementById('replay');
  if (btn) { btn.textContent = '▶ Replay'; btn.onclick = startReplay; }
}

document.addEventListener('keydown', e => {
  if (!state.view || e.metaKey || e.ctrlKey || e.altKey || /input|select|textarea|button/i.test(e.target.tagName)) return;
  const v = state.view;
  const order = [...v.played].sort((a, b) => a - b);
  const i = order.indexOf(v.pick);
  if (e.key === 'ArrowRight' || e.key === 'j') { if (i < order.length - 1) go(v.cur, order[i + 1]); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'k') { if (i > 0) go(v.cur, order[i - 1]); e.preventDefault(); }
  else if (e.key === 'r') toggleMind();
  else if (e.key === ' ') { state.replay ? stopReplay(true) : startReplay(); e.preventDefault(); }
});

// A poker chip for each deal: green when the first-named player won it over both plays, red when the
// other did; the stronger the colour, the bigger the swing.
function chipColor(v, max) {
  if (v === 0) return '#55605a';
  const t = Math.sqrt(Math.abs(v) / max);
  const base = v > 0 ? 'var(--pos-chip)' : 'var(--neg-chip)';
  return `color-mix(in srgb, ${base} ${Math.round(35 + 65 * t)}%, #24302a)`;
}

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

  const net = done.reduce((s, k) => s + nets[k], 0);
  const wins = done.filter(k => nets[k] > 0).length, losses = done.filter(k => nets[k] < 0).length;
  const face = (p, side) => `<div class="face ${side}">${monogram(p, 'xl')}<div class="face-name">${label(p)}</div>
      ${isBot(p) ? '<span class="tag">bot</span>' : ''}</div>`;
  const leader = net > 0 ? cur.a : net < 0 ? cur.b : null;
  const scoreboard = `<div class="scoreboard card-box">
      ${face(cur.a, 'left')}
      <div class="score-mid">
        <div class="score-big"><span class="${wins >= losses ? 'lead' : ''}">${wins}</span><span class="dash">:</span><span class="${losses >= wins ? 'lead' : ''}">${losses}</span></div>
        <div class="score-cap">deals won${done.length - wins - losses ? ` · ${done.length - wins - losses} even` : ''}</div>
        <div class="score-net">${done.length ? (leader ? `<b>${label(leader)}</b> up <b class="gold">${trim(Math.abs(net) / bb)} bb</b>
          · ${trim(Math.abs(net / bb / (2 * done.length) * 100))} bb/100` : 'Dead even') : 'No deals finished yet'}</div>
        <div class="score-cap">${done.length} of ${total} deals played</div>
      </div>
      ${face(cur.b, 'right')}</div>`;

  const max = Math.max(1, ...done.map(k => Math.abs(nets[k])));
  const strip = Array.from({ length: total }, (_, k) => {
    const sel = k === pick ? 'sel' : '';
    if (!(k in nets)) return `<button class="chipbtn ${played.has(k) ? 'half' : 'empty'} ${sel}" ${played.has(k) ? `data-deal="${k}" title="Deal ${k + 1}: being played"` : 'disabled'}><span>${k + 1}</span></button>`;
    const v = nets[k];
    return `<button data-deal="${k}" class="chipbtn ${sel}" style="--c:${chipColor(v, max)}" title="Deal ${k + 1}: ${v === 0 ? 'even' : `${label(v > 0 ? cur.a : cur.b)} won ${trim(Math.abs(v) / bb)} bb`} over both plays"><span>${k + 1}</span></button>`;
  }).join('');
  const legend = `<div class="strip-legend">
    <span><span class="swatch" style="background:var(--pos-chip)"></span>${label(cur.a)} won the deal</span>
    <span><span class="swatch" style="background:var(--neg-chip)"></span>${label(cur.b)} won it</span>
    <span>brighter chip, bigger swing</span><span class="keys"><span class="kbd">←</span><span class="kbd">→</span> deals
    · <span class="kbd">space</span> replay · <span class="kbd">r</span> reasoning</span></div>`;

  const order = [...played].sort((x, y) => x - y);
  const i = order.indexOf(pick);
  const dealNet = pick in nets ? nets[pick] : null;
  const rounds = state.index.some(m => m.round > 1 && m.a === cur.a && m.b === cur.b);
  const bar = `<div class="deal-bar"><h2>${rounds ? `<span class="round">Round ${cur.round}</span>` : ''}Deal ${pick + 1}</h2>
    <div class="nav"><button data-deal="${order[i - 1]}" ${i > 0 ? '' : 'disabled'} aria-label="Previous deal">←</button>
      <button data-deal="${order[i + 1]}" ${i < order.length - 1 ? '' : 'disabled'} aria-label="Next deal">→</button></div>
    <span class="soft">${dealNet == null ? 'in progress' : dealNet === 0 ? 'even over both plays'
      : `<b>${label(dealNet > 0 ? cur.a : cur.b)}</b> wins it by <b class="gold">${trim(Math.abs(dealNet) / bb)} bb</b> over both plays`}</span>
    <span class="spacer"></span>
    <button class="toggle" id="replay" title="Replay both plays step by step (space)">▶ Replay</button>
    <button class="toggle" id="mind-toggle" aria-pressed="${state.mind}" title="Show each decision's private reasoning (r)">Reasoning</button></div>`;

  return `${scoreboard}<div class="strip-wrap"><div class="strip" style="--cols:${total <= 30 ? total : Math.ceil(total / Math.ceil(total / 25))}">${strip}</div>${legend}</div>${bar}
    <div class="pair ${state.mind ? '' : 'hide-mind'}">${playPanel(ma, pick, 'Play 1', bb)}${mb ? playPanel(mb, pick, 'Play 2 · seats swapped', bb) : ''}</div>`;
}

const VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to' };

function chipStack(n) {
  return `<span class="stack" aria-hidden="true">${Array.from({ length: n }, (_, i) => `<i style="--k:${i}"></i>`).join('')}</span>`;
}

function playPanel(m, k, title, bb) {
  const evs = m.events[k] || [];
  const start = evs.find(e => e.kind === 'hand_start');
  if (!start) return `<div class="card-box play"><div class="play-head"><h3>${title}</h3></div><div class="pending">Not dealt yet.</div></div>`;
  const d = start.data, players = m.config.players, aliases = m.config.names;
  const result = evs.find(e => e.kind === 'result')?.data;

  // walk the events once: the log, the step each thing appears at, and the pot after each step
  let log = '', thought = null, step = 0;
  const notes = {}, pots = [], boardSteps = [];
  const short = i => label(players[i]);
  let pot = evs.filter(e => e.kind === 'blind').reduce((s, e) => s + e.data.amount, 0);
  const blinds = evs.filter(e => e.kind === 'blind').map(e => `${short(e.data.seat)} ${bbs(e.data.amount, bb)}`);
  log += `<div class="street" data-step="0">Preflop</div><div class="ev quiet" data-step="0"><span class="actor">Blinds</span><span class="what"><span class="verb">${blinds.join(' · ')}</span></span></div>`;
  pots.push([0, bbs(pot, bb)]);
  for (const e of evs) {
    const x = e.data;
    if (e.kind === 'thought') { thought = x; if (x.notes) notes[x.seat] = x.notes; }
    else if (e.kind === 'action') {
      step += 1;
      pot = x.pot ?? pot;
      pots.push([step, bbs(pot, bb)]);
      const t = thought && thought.seat === x.seat ? thought : null;
      thought = null;
      const amt = x.action === 'call' || x.action === 'bet' || x.action === 'raise' ? ` ${bbs(x.amount, bb)}` : '';
      const badges = [];
      if (x.all_in) badges.push('<span class="badge strong">all in</span>');
      if (t?.fallback) badges.push('<span class="badge bad" title="The model gave no playable answer, so the game checked or folded for it">not its move</span>');
      if (x.note) badges.push(`<span class="badge warn" title="${esc(x.note)}">illegal size, corrected</span>`);
      const mind = t && (t.reasoning || t.thinking) ? `<div class="mind">${t.reasoning ? esc(t.reasoning) : ''}${t.thinking
        ? `<span class="lbl">Thinking</span><span class="thinking">${esc(t.thinking)}</span>` : ''}</div>` : '';
      log += `<div class="ev" data-step="${step}">${monogram(players[x.seat], 'xs')}<span class="actor">${short(x.seat)}</span>
        <span class="what"><span class="verb ${x.action}">${VERB[x.action] || esc(x.action)}${amt}</span>${badges.join('')}</span>
        ${x.say ? `<div class="say-row"><span class="say">${esc(x.say)}</span></div>` : ''}${mind}</div>`;
    } else if (e.kind === 'board') {
      step += 1;
      x.cards.forEach(() => boardSteps.push(step));
      log += `<div class="street" data-step="${step}">${esc(x.street)} ${cards(x.cards, 0, 'sm')}<span class="pot">pot ${bbs(pot, bb)}</span></div>`;
    } else if (e.kind === 'runout') {
      log += `<div class="ev quiet" data-step="${step}"><span class="actor"></span><span class="what"><span class="verb">All in. The rest of the board is dealt.</span></span></div>`;
    } else if (e.kind === 'showdown') {
      step += 1;
      log += `<div class="street" data-step="${step}">Showdown</div>` + Object.entries(x.hands).map(([i, h]) =>
        `<div class="showdown" data-step="${step}"><span class="actor">${short(Number(i))}</span>${cards(h.cards, 0, 'sm')}<span class="soft">${esc(h.best)}</span></div>`).join('');
    }
  }
  const endStep = step + 1;
  if (result) {
    const winners = result.won.map((w, i) => w > 0 ? i : null).filter(i => i != null);
    const how = result.showdown ? 'at showdown' : 'without a showdown';
    log += `<div class="outcome" data-step="${endStep}">${winners.length === 2 ? `Split pot ${how}` : `<b>${short(winners[0])}</b> takes <b class="gold">${bbs(result.pot, bb)}</b> ${how}`}</div>`;
  }
  const board = result?.board || [...evs].reverse().find(e => e.kind === 'board')?.data.board || [];

  // the table: seat 1 across the felt, seat 0 nearest the viewer
  const seat = (i, where) => {
    const won = result && result.net[i] > 0;
    const netEl = result ? `<span class="seat-net ${cls(result.net[i])}" data-step="${endStep}">${signedBB(result.net[i], bb)}</span>` : '';
    return `<div class="seat ${where} ${won ? 'winner' : ''}">
      ${monogram(players[i], 'lg')}
      <div class="who"><div class="name">${name(players[i])}</div><div class="alias">${esc(aliases[i])} · ${i === d.button ? 'button' : 'big blind'}</div></div>
      ${cards(d.hole[i], 2, 'hole')}${netEl}
      ${i === d.button ? '<span class="dealer" title="Dealer button">D</span>' : ''}</div>`;
  };
  const potChips = Math.max(1, Math.min(7, Math.round(Math.log2(1 + (result ? result.pot : pot) / bb))));
  const table = `<div class="table-wrap">
      ${seat(1, 'top')}
      <div class="felt"><div class="felt-inner">
        <div class="board">${cards(board, 5, '', boardSteps)}</div>
        <div class="pot-line">${chipStack(potChips)}<span>Pot <b class="pot-amt">${bbs(result ? result.pot : pot, bb)}</b></span>
          ${result ? '' : '<span class="live-dot"></span>'}</div>
      </div></div>
      ${seat(0, 'bottom')}</div>`;

  const noteBlock = Object.keys(notes).length ? `<div class="notes mind">${Object.entries(notes).map(([i, n]) =>
    `<span class="lbl">${short(Number(i))}'s notes on its opponent</span>${esc(n)}`).join('')}</div>` : '';
  return `<div class="card-box play" data-pots='${esc(JSON.stringify(pots))}'><div class="play-head"><h3>${title}</h3></div>
    ${table}<div class="log">${log}</div>${noteBlock}</div>`;
}

route();
