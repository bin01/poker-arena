// Poker Arena viewer: leaderboard, matchups, and duplicate hand replays with each player's reasoning.
'use strict';

const $app = document.getElementById('app');
const $status = document.getElementById('status');
const state = { index: [], ratings: null, kinds: {}, timer: null, showMind: false };

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const signed = (x, d = 0) => x == null ? '' : (x > 0 ? '+' : '') + Number(x).toFixed(d);
const cls = x => x > 0 ? 'pos' : x < 0 ? 'neg' : '';
const SUIT = { s: '♠', h: '♥', d: '♦', c: '♣' };

async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function card(c) {
  if (!c) return '<span class="card empty"></span>';
  const rank = c[0] === 'T' ? '10' : c[0];
  const red = c[1] === 'h' || c[1] === 'd';
  return `<span class="card ${red ? 'red' : ''}">${rank}<small>${SUIT[c[1]] || c[1]}</small></span>`;
}
const cards = (cs, n = 0) => {
  const list = [...(cs || [])];
  while (list.length < n) list.push(null);
  return `<span class="cards">${list.map(card).join('')}</span>`;
};
const who = p => `${esc(p)}${state.kinds[p] === 'bot' ? '<span class="bot-tag">bot</span>' : ''}`;

// ---- routing -------------------------------------------------------------------------------
async function route() {
  clearTimeout(state.timer);
  const hash = location.hash.replace(/^#\/?/, '');
  const [tab, ...rest] = hash.split('/');
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('active',
    (a.dataset.tab === 'board' && !tab) || a.dataset.tab === tab));
  try {
    state.index = await get('/api/index');
    for (const m of state.index) Object.assign(state.kinds, m.kinds || {});
    if (tab === 'matches') {
      const params = new URLSearchParams(location.hash.split('?')[1] || '');
      await renderMatches(decodeURIComponent((rest.join('/') || '').split('?')[0]), params);
    } else {
      await renderBoard();
    }
    const live = state.index.filter(m => m.live).length;
    $status.textContent = live ? `${live} matchup${live > 1 ? 's' : ''} live · refreshing` : `${state.index.length} matchups`;
    if (live) state.timer = setTimeout(route, 8000);
  } catch (e) {
    $app.innerHTML = `<div class="panel empty-state">Could not load data: ${esc(e.message)}</div>`;
  }
}
window.addEventListener('hashchange', route);

// ---- leaderboard ---------------------------------------------------------------------------
async function renderBoard() {
  const r = state.ratings = await get('/api/ratings');
  if (!r.players || !r.players.length) {
    $app.innerHTML = `<div class="panel empty-state">No finished duplicate hands yet.<br>
      Run <code>python -m poker_arena play rule station --deals 20</code> or a tournament, then refresh.</div>`;
    return;
  }
  const lo = Math.min(...r.players.map(p => p.elo90 ? p.elo90[0] : p.elo)) - 20;
  const hi = Math.max(...r.players.map(p => p.elo90 ? p.elo90[1] : p.elo)) + 20;
  const x = v => ((v - lo) / (hi - lo) * 100).toFixed(1) + '%';
  const rows = r.players.map((p, i) => `<tr>
      <td class="num muted">${i + 1}</td>
      <td>${who(p.player)}</td>
      <td class="num"><b>${p.elo}</b></td>
      <td><span class="interval">${p.elo90 ? `<span style="left:${x(p.elo90[0])};width:calc(${x(p.elo90[1])} - ${x(p.elo90[0])})"></span>` : ''}<i style="left:${x(p.elo)}"></i></span>
          <span class="muted">${p.elo90 ? p.elo90.join('–') : ''}</span></td>
      <td class="num ${cls(p.bb_per_100)}">${signed(p.bb_per_100, 1)}</td>
      <td class="num muted">±${p.bb_per_100_se ?? ''}</td>
      <td class="num">${p.hands}</td>
      <td class="num">${p.session_wins}–${p.session_ties}–${p.session_losses}</td></tr>`).join('');
  const names = r.players.map(p => p.player);
  const cell = {};
  for (const m of r.matchups) {
    cell[m.a + '|' + m.b] = { v: m.a_bb_per_100, se: m.se, n: m.deals };
    cell[m.b + '|' + m.a] = { v: -m.a_bb_per_100, se: m.se, n: m.deals };
  }
  const matrix = `<table class="matrix"><tr><th>row vs column (bb/100)</th>${names.map(n => `<th>${esc(n)}</th>`).join('')}</tr>
    ${names.map(a => `<tr><th>${esc(a)}</th>${names.map(b => {
      if (a === b) return '<td class="muted">·</td>';
      const c = cell[a + '|' + b];
      return c ? `<td class="${cls(c.v)}" title="±${c.se} over ${c.n} duplicate deals">${signed(c.v, 0)}</td>` : '<td class="muted">–</td>';
    }).join('')}</tr>`).join('')}</table>`;
  $app.innerHTML = `
    <h2>Leaderboard</h2>
    <div class="panel scroll"><table>
      <tr><th></th><th>Player</th><th class="num">Elo</th><th>90% interval</th><th class="num">bb/100</th><th class="num">se</th>
          <th class="num">Hands</th><th class="num">Sessions W–T–L</th></tr>${rows}</table>
      <p class="muted">Elo: a Bradley–Terry fit over ${r.sessions} duplicate sessions of ${r.session_deals} deals
      (${2 * r.session_deals} hands, each deal played from both seats), centred on 1500. A 200-point gap means the higher
      player wins about 76% of sessions. bb/100: big blinds won per 100 hands over every duplicate hand. ${r.deals} duplicate deals in all.</p>
    </div>
    <h3>Head to head</h3>
    <div class="panel scroll">${matrix}</div>`;
}

// ---- matches -------------------------------------------------------------------------------
async function renderMatches(id, params) {
  if (!state.index.length) {
    $app.innerHTML = '<div class="panel empty-state">No matches yet.</div>';
    return;
  }
  const cur = state.index.find(m => m.id === id) || state.index[0];
  const list = state.index.map(m => `<a href="#/matches/${encodeURIComponent(m.id)}" class="${m === cur ? 'active' : ''}">
      <div class="row"><span>${who(m.a)} <span class="muted">vs</span> ${who(m.b)}</span>
        ${m.live ? '<span class="live">live</span>' : ''}</div>
      <div class="row muted"><span>${m.pairs_done} / ${m.matches[0]?.hands_total ?? '?'} deals</span>
        <span class="${cls(m.pair_net_a)}">${esc(m.a)} ${signed(m.pair_net_a / (m.bb || 1), 0)} bb</span></div></a>`).join('');

  const [ma, mb] = await Promise.all(cur.matches.map(x => get('/api/match?id=' + encodeURIComponent(x.id))));
  const ha = Object.fromEntries(ma.hands.map(h => [h.hand, h]));
  const hb = mb ? Object.fromEntries(mb.hands.map(h => [h.hand, h])) : {};
  const total = ma.config.seeds.length;
  const done = Object.keys(ha).map(Number);
  const pick = params.has('hand') ? Number(params.get('hand')) : (done.length ? Math.max(...done) : 0);
  const strip = Array.from({ length: total }, (_, k) => {
    const a = ha[k], b = hb[k];
    if (!a && !b) return `<button disabled class="muted">${k + 1}</button>`;
    const net = (a ? a.net[0] : 0) + (b ? b.net[1] : 0);
    return `<button class="${cls(net)} ${a && b ? '' : 'half'} ${k === pick ? 'sel' : ''}" data-hand="${k}"
      title="${esc(cur.a)} ${signed(net)} chips over both seats">${k + 1}</button>`;
  }).join('');
  const pairNet = (ha[pick] ? ha[pick].net[0] : 0) + (hb[pick] ? hb[pick].net[1] : 0);

  $app.innerHTML = `<div class="layout">
    <div class="panel list">${list}</div>
    <div>
      <h2>${who(cur.a)} <span class="muted">vs</span> ${who(cur.b)}</h2>
      <div class="muted">Duplicate deals: each deal is played twice with the same cards, once from each seat.
        Colours show ${esc(cur.a)}'s net over both plays.</div>
      <div class="strip">${strip}</div>
      <div class="controls">
        <b>Deal ${pick + 1}</b>
        <span class="${cls(pairNet)}">${esc(cur.a)} ${signed(pairNet)} chips over both seats</span>
        <label><input type="checkbox" id="mind" ${state.showMind ? 'checked' : ''}> open all reasoning</label>
      </div>
      <div class="pair">${handPanel(ma, pick, 'Play 1')}${mb ? handPanel(mb, pick, 'Play 2 (seats swapped)') : ''}</div>
    </div></div>`;
  $app.querySelectorAll('.strip button[data-hand]').forEach(b => b.onclick = () => {
    location.hash = `#/matches/${encodeURIComponent(cur.id)}?hand=${b.dataset.hand}`;
  });
  document.getElementById('mind').onchange = e => {
    state.showMind = e.target.checked;
    $app.querySelectorAll('details.mind').forEach(d => d.open = state.showMind);
  };
}

function handPanel(m, k, title) {
  const evs = m.events[k] || [];
  const start = evs.find(e => e.kind === 'hand_start');
  if (!start) return `<div class="panel"><h3>${title}</h3><div class="muted">Not played yet.</div></div>`;
  const d = start.data, players = m.config.players, names = m.config.names;
  const result = evs.find(e => e.kind === 'result');
  const showdown = evs.find(e => e.kind === 'showdown');
  const board = (result || {}).data?.board || [...evs].reverse().find(e => e.kind === 'board')?.data.board || [];
  const seat = i => `<div class="seat" ${i === 1 ? 'style="align-items:flex-end;text-align:right"' : ''}>
      <span class="who">${esc(names[i])}${d.button === i ? '<span class="btn-chip" title="button">D</span>' : ''}</span>
      <span class="model">${who(players[i])}</span>
      ${cards(d.hole[i], 2)}
      ${result ? `<span class="${cls(result.data.net[i])}"><b>${signed(result.data.net[i])}</b></span>` : ''}
    </div>`;

  let html = '', street = 'preflop', thought = null;
  const label = s => `<div class="street"><span>${s}</span></div>`;
  html += label('preflop');
  for (const e of evs) {
    const x = e.data;
    if (e.kind === 'blind') html += `<div class="note">${esc(names[x.seat])} posts the ${x.label} (${x.amount})</div>`;
    else if (e.kind === 'thought') thought = x;
    else if (e.kind === 'action') {
      if (x.street !== street) { street = x.street; }
      const verb = { fold: 'folds', check: 'checks', call: `calls ${x.amount}`, bet: `bets ${x.amount}`, raise: `raises to ${x.amount}` }[x.action] || x.action;
      const t = thought && thought.seat === x.seat ? thought : null;
      thought = null;
      const mind = t && (t.reasoning || t.thinking || t.notes) ? `<details class="mind" ${state.showMind ? 'open' : ''}>
          <summary>${esc(names[x.seat])}'s reasoning${t.tries > 1 ? ` · ${t.tries} tries` : ''}${t.fallback ? ' · fell back' : ''}</summary>
          ${t.reasoning ? `<pre>${esc(t.reasoning)}</pre>` : ''}
          ${t.thinking ? `<pre class="muted">${esc(t.thinking)}</pre>` : ''}
          ${t.notes ? `<pre><b>Notes:</b> ${esc(t.notes)}</pre>` : ''}</details>` : '';
      html += `<div class="step"><div class="head"><span class="actor">${esc(names[x.seat])}</span>
          <span class="act ${x.action}">${verb}${x.all_in ? ' (all in)' : ''}</span>
          <span class="muted">pot ${x.pot}</span>${x.note ? `<span class="note">· ${esc(x.note)}</span>` : ''}</div>
          ${x.say ? `<div class="say">${esc(x.say)}</div>` : ''}${mind}</div>`;
    } else if (e.kind === 'board') {
      street = x.street;
      html += `<div class="street"><span>${x.street}</span>${cards(x.cards)}</div>`;
    } else if (e.kind === 'runout') html += `<div class="note">All in: the board is run out.</div>`;
    else if (e.kind === 'showdown') {
      html += `<div class="street"><span>showdown</span></div>` + Object.entries(x.hands).map(([i, h]) =>
        `<div class="showdown-row"><b>${esc(names[i])}</b> ${cards(h.cards)} <span>${esc(h.best)}</span></div>`).join('');
    }
  }
  if (result) {
    const r = result.data;
    const winners = r.won.map((w, i) => w > 0 ? `${esc(names[i])} wins ${w}` : null).filter(Boolean).join(', ');
    html += `<div class="result">${winners}${r.showdown ? '' : ' (no showdown)'}</div>`;
  }
  return `<div class="panel"><h3>${title}</h3>
    <div class="felt"><div class="seats">${seat(0)}${seat(1)}</div>
      <div class="board">${cards(board, 5)}</div>
      <div class="pot">${result ? `pot ${result.data.pot}` : 'in progress'}${showdown ? ' · showdown' : ''}</div></div>
    <div class="timeline">${html}</div></div>`;
}

route();
