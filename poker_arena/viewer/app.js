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
  matches: new Map(), view: null, replay: null, page: null, checked: null };

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
const player = (p, extra = '') => `<div class="player">${monogram(p)}<div class="pname"><div class="name">${name(p)}</div>${extra}</div></div>`;
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
    document.querySelectorAll('nav a').forEach(a => {
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
  const share = r.hands_planned ? r.hands / r.hands_planned : 0;
  let line;
  if (r.finished) line = `Finished · ${plural(r.hands, 'hand')} played · ${ago(r.updated)}`;
  else {
    line = `${r.hands.toLocaleString()} of ${plural(r.hands_planned, 'hand')}`;
    if (r.eta_seconds != null) line += ` · about ${duration(r.eta_seconds)} left`;
    else if (!r.live) line += ` · paused, last action ${ago(r.updated)}`;
  }
  return `<div class="page-head"><div><div class="eyebrow">${r.live ? '<span class="live-dot"></span> Live tournament' : 'Tournament'}</div>
    <h1>${esc(runLabel(state.run))}</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="meter"><span style="width:${(share * 100).toFixed(1)}%"></span></div>
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
      ${r.started_at ? `since ${new Date(r.started_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}</div>
      ${notes.length ? `<div class="line soft small">${notes.join(' ')}</div>` : ''}</div></div>`;
}

function intro(rt) {
  const open = store.get('howto', true);
  return `<div class="intro">
    <p class="lede"><b>AI models play heads-up No-Limit Hold'em against each other.</b> Every deal is played twice with
      the seats swapped, so the luck of the cards cancels out and what is left is how well each model plays.</p>
    <div class="intro-row">
      <a class="cta" href="${runLink(state.run, '/m')}"><span class="cta-icon">▶</span>Watch the games</a>
      <details class="howto" ${open ? 'open' : ''}><summary>How to read the standings</summary>
        <dl>
          <dt>Strength</dt><dd>Big blinds won per 100 hands against the average opponent${rt.contract ? '' : ' (here: against the opponents it met)'}.
            <b class="pos">+20</b> is a clear edge; <b>0</b> is average. The bar is its 95% range: the true value is very likely inside.</dd>
          <dt>Rank</dt><dd>A number only when a model is clearly better than the ones below it. Otherwise <b>=</b>: it can't
            yet be told apart from its neighbours.</dd>
          <dt>Leader</dt><dd>A model that beats every other model by more than the noise. Until one does, the page says
            <i>No established leader yet</i>.</dd>
          <dt>Matchups</dt><dd>Each pair's verdict: <i>wins</i>, <i>even</i> (within ±15 bb/100), <i>undecided</i>, or
            <i>too early</i>.</dd>
          <dt>Session Elo</dt><dd>A second, different number: how often it wins a 10-deal session. Shown for reference, not used to rank.</dd>
        </dl></details>
    </div></div>`;
}

const fmtStrength = v => v == null ? '–' : signed(v, 0);

function hero(rt) {
  const ranked = rt.players.filter(p => !p.quarantined && p.strength != null);
  if (!ranked.length) return '';
  const multi = rt.components.length > 1;
  if (rt.leaders.length) {
    return `<div class="hero-row">${rt.leaders.map(l => {
      const p = rt.players.find(x => x.player === l.player) || {};
      const ci = l.ci95 ? `${signed(l.ci95[0], 0)} to ${signed(l.ci95[1], 0)}` : '';
      return `<div class="hero-card">
        <span class="idx">A<i>♠</i></span><span class="idx br">A<i>♠</i></span>
        <div class="hero-kicker">${multi ? `Leader of group ${groupName(l.component)}` : 'Established leader'}</div>
        ${monogram(l.player, 'xl')}
        <div class="hero-name">${name(l.player)}</div>
        <div class="hero-num">${fmtStrength(p.strength)}<small>bb/100</small></div>
        <div class="hero-sub">Beats every other model by more than the noise${l.margin_bb100 != null ? `: at least ${signed(l.margin_bb100, 0)} bb/100 over the next best` : ''}${ci ? ` (95% range ${ci})` : ''}.</div>
      </div>`;
    }).join('')}${multi && rt.leaders.length < rt.components.length ? noLeaderCard(rt, true) : ''}</div>`;
  }
  return `<div class="hero-row">${noLeaderCard(rt, false)}</div>`;
}
function noLeaderCard(rt, partial) {
  const ranked = rt.players.filter(p => !p.quarantined && p.strength != null);
  const top = ranked[0];
  const deals = rt.deals || 0;
  const powers = rt.pairings.map(p => p.power?.deals_for_80pct?.['50']).filter(x => x != null);
  const need = powers.length ? Math.max(...powers) : null;
  let why;
  if (ranked.length < 2) why = ranked.length === rt.players.length ? 'Only one model has results so far.' : 'Only one model has rated results so far.';
  else if (deals < 20) why = `Only ${plural(deals, 'deal')} so far: far too few to separate anyone.`;
  else why = `The gaps are still inside the noise after ${plural(deals, 'deal')}.`;
  return `<div class="hero-card back">
    <div class="back-face" aria-hidden="true"><span>?</span></div>
    <div class="hero-body">
      <div class="hero-kicker">${partial ? 'Other groups' : 'Who is winning?'}</div>
      <div class="hero-title">No established leader yet</div>
      <p>${why} ${top && ranked.length > 1 ? `<b>${plain(top.player)}</b> is ahead on points at <b class="${cls(top.strength)}">${fmtStrength(top.strength)} bb/100</b>${top.strength_ci95 ? `, but its 95% range runs from ${signed(top.strength_ci95[0], 0)} to ${signed(top.strength_ci95[1], 0)}` : ''}.` : ''}</p>
      ${need ? `<p class="soft small">To detect a 50 bb/100 edge with 80% power takes about ${approx(need)} deals per pairing.</p>` : ''}
    </div></div>`;
}
const groupName = i => 'ABCDEFGH'[i] || String(i + 1);

async function renderBoard() {
  const raw = await api('ratings', ['run', state.run]);
  Object.assign(state.labels, raw.labels || {});
  Object.assign(state.kinds, raw.kinds || {});
  const rt = normalize(raw);
  for (const p of rt.players) if (p.kind) state.kinds[p.player] = p.kind;
  const r = state.runs.find(x => x.run === state.run);
  let html = runHead(r, rt) + intro(rt);
  const waiting = rt.roster.filter(x => !rt.players.some(p => p.player === x.player));
  if (!rt.players.length) {
    html += `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>The cards are in the air</h2>
      <p>A deal counts once both of its plays are finished. ${state.index.some(m => m.started) ? `Watch the first ones on the <a href="${runLink(state.run, '/m')}"><u>games</u></a> page.` : ''}</p>
      ${waiting.length ? `<div class="waiting-list">${waiting.map(x => player(x.player, `<div class="small soft">${x.status === 'playing' ? 'playing its first deals' : 'waiting to play'}</div>`)).join('')}</div>` : ''}</div>`;
    $app.innerHTML = html;
    bindBoard();
    return;
  }
  html += hero(rt);
  html += standings(rt, waiting);
  html += pairingsSection(rt);
  html += headToHead(rt);
  html += styleTable(rt);
  if (rt.notes.length) html += `<div class="notes-box"><span class="suit">♣</span><div><b>Worth knowing.</b> ${rt.notes.map(esc).join(' ')}</div></div>`;
  $app.innerHTML = html;
  bindBoard();
}
function bindBoard() {
  $app.querySelector('.howto')?.addEventListener('toggle', e => store.set('howto', e.target.open));
}

function axisFor(players) {
  const vals = players.flatMap(p => [p.strength, ...(p.strength_ci95 || [])]).filter(v => v != null && isFinite(v));
  const m = Math.max(10, ...vals.map(Math.abs));
  const step = m > 400 ? 200 : m > 150 ? 100 : m > 60 ? 50 : 20;
  const lim = Math.ceil(m * 1.05 / step) * step;
  return { lim, x: v => (50 + Math.max(-1, Math.min(1, v / lim)) * 50).toFixed(2) + '%' };
}

function standings(rt, waiting) {
  const multi = rt.components.length > 1;
  let out = '';
  rt.components.forEach((comp, ci) => {
    const ps = rt.players.filter(p => comp.includes(p.player) && !p.quarantined);
    if (!ps.length) return;
    const ax = axisFor(ps);
    const rows = ps.map(p => standingRow(rt, p, ax)).join('');
    const extra = ci === rt.components.length - 1 ? waitingRows(waiting) : '';
    const title = multi ? `Group ${groupName(ci)}` : 'Standings';
    out += section(title, multi ? `${plural(ps.length, 'player')} who have played each other` : `${plural(rt.deals, 'duplicate deal')} · ${plural(rt.hands || 2 * rt.deals, 'hand')}`,
      `<div class="board-list"><div class="board-head"><span></span><span>Player</span>
         <span class="r" title="Big blinds won per 100 hands against the average opponent">Strength</span>
         <span class="axis-head"><span>${signed(-ax.lim, 0)}</span><span>0</span><span>${signed(ax.lim, 0)}</span></span>
         <span class="r hide-sm">Net</span><span class="r hide-sm" title="Share of 10-deal sessions won (secondary; not used to rank)">Sessions</span></div>
       ${rows}${extra}</div>`,
      multi && ci === 0 ? 'No games connect these groups yet, so each is ranked on its own. A match between them would join them.' : '');
  });
  if (!out && waiting.length) out = section('Standings', '', `<div class="board-list">${waitingRows(waiting)}</div>`);
  const inComp = new Set(rt.components.flat());
  const q = rt.players.filter(p => p.quarantined || !inComp.has(p.player))
    .map(p => p.quarantined ? p : { ...p, quarantine_reason: p.quarantine_reason || 'has only played players who are not rated' });
  if (q.length) {
    out += section('Not rated', 'results shown, kept out of the standings', `<div class="board-list">${q.map(p => `<div class="brow quarantined">
      <span class="rank">–</span>${player(p.player, `<div class="small soft">${esc(p.quarantine_reason || 'the model or its version was not recorded')}</div>`)}
      <span class="r strength soft">${fmtStrength(p.strength)}</span><span class="axis"></span>
      <span class="r hide-sm soft">${p.net_bb != null ? signed(p.net_bb, 0) + ' bb' : ''}</span><span class="hide-sm"></span></div>`).join('')}</div>`,
      'A player is left out when its exact model or reasoning level is unknown, so its results could mix more than one model, or when none of its opponents is rated.');
  }
  return out;
}

function standingRow(rt, p, ax) {
  const ci = p.strength_ci95;
  const clear = ci && (ci[0] > 0 || ci[1] < 0);
  const tied = (p.rank_tied_with || []).filter(x => x !== p.player);
  const comp = rt.components.find(c => c.includes(p.player)) || [];
  const tiedText = p.rank == null && tied.length && tied.length < comp.length - 1
    ? `<div class="small soft">level with ${tied.map(plain).join(', ')}</div>` : '';
  const leader = rt.leaders.some(l => l.player === p.player);
  const whisker = `<div class="whisker" title="${ci ? `95% range ${signed(ci[0], 0)} to ${signed(ci[1], 0)} bb/100` : 'no range yet: too few deals'}">
      <span class="zero"></span>${ci ? `<span class="range ${clear ? cls(p.strength) : ''}" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
      ${p.strength != null ? `<span class="point ${clear ? cls(p.strength) : ''}" style="left:${ax.x(p.strength)}"></span>` : ''}</div>`;
  const elo = p.session_elo != null ? `<span title="Session Elo ${p.session_elo}${p.session_elo_ci95 ? ` (${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]})` : ''}">${p.session_w ?? 0}–${p.session_t ?? 0}–${p.session_l ?? 0}</span>
      <small>Elo ${p.session_elo}${p.session_elo_ci95 ? ` · ${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]}` : ''}</small>` : '';
  return `<div class="brow ${leader ? 'lead' : ''}">
    <span class="rank ${p.rank != null ? 'sep' : ''}" title="${p.rank != null ? 'Clearly better than everyone below' : 'Not yet separable from its neighbours'}">${p.rank ?? '='}</span>
    ${player(p.player, tiedText)}
    <span class="r strength"><b class="${clear ? cls(p.strength) : ''}">${fmtStrength(p.strength)}</b>${ci ? `<small>±${Math.round((ci[1] - ci[0]) / 2)}</small>` : '<small>no range yet</small>'}</span>
    <span class="axis">${whisker}</span>
    <span class="r hide-sm"><span class="${cls(p.net_bb)}">${p.net_bb != null ? signed(p.net_bb, 0) + ' bb' : ''}</span><small>${plural(p.hands || 0, 'hand')}</small></span>
    <span class="r hide-sm sess">${elo}</span></div>`;
}

function waitingRows(waiting) {
  return waiting.map(x => `<div class="brow waiting"><span class="rank">·</span>
    ${player(x.player, `<div class="small soft">${x.status === 'playing' ? 'playing its first deals' : x.status === 'done' ? 'no finished deals' : 'waiting to play'}</div>`)}
    <span class="r strength soft">–</span><span class="axis"></span><span class="hide-sm"></span><span class="hide-sm"></span></div>`).join('');
}

// Each pairing: the edge with its range, and decide()'s verdict.
function pairingsSection(rt) {
  if (!rt.pairings.length) return '';
  const ids = Object.fromEntries(state.index.map(m => [m.id.split('/').pop(), m.id]));
  const order = { a: 0, b: 0, even: 1, exhausted: 2, open: 3, early: 4 };
  const ax = axisFor(rt.pairings.map(p => ({ strength: Math.abs(p.edge ?? 0), strength_ci95: p.ci95 })));
  const rows = [...rt.pairings].sort((x, y) => (order[x.verdict] ?? 5) - (order[y.verdict] ?? 5) || Math.abs(y.edge ?? 0) - Math.abs(x.edge ?? 0)).map(p => {
    const up = (p.edge ?? 0) >= 0, lead = up ? p.a : p.b, trail = up ? p.b : p.a;
    const edge = Math.abs(p.edge ?? 0);
    const ci = p.ci95 ? (up ? p.ci95 : [-p.ci95[1], -p.ci95[0]]) : null;
    const v = verdict(p, plain);
    const mu = (p.matchups || [])[p.matchups?.length - 1];
    const full = mu && (state.index.some(m => m.id === mu) ? mu : ids[mu] || ids[String(mu).split('/').pop()]);
    const checks = p.looks_planned ? (p.looks_used ? ` · checked ${p.looks_used} of ${plural(p.looks_planned, 'time')}` : ' · not checked yet') : '';
    const power = powerLine(p);
    return `<div class="prow">
      <div class="duo-names">${player(lead)}<span class="over">${v.tone === 'won' ? 'beats' : 'ahead of'}</span>${player(trail)}</div>
      <div class="edge"><b>${edge.toFixed(0)}</b><small>bb/100${ci ? ` · ${signed(ci[0], 0)} to ${signed(ci[1], 0)}` : ''}</small></div>
      <div class="edge-bar"><div class="whisker"><span class="zero"></span>${ci ? `<span class="range ${v.tone === 'won' ? 'pos' : ''}" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
        <span class="point ${v.tone === 'won' ? 'pos' : ''}" style="left:${ax.x(edge)}"></span></div></div>
      <div class="vcell"><span class="verdict ${v.tone}" title="${esc(v.long)}">${esc(v.short)}</span>
        <small>${plural(p.deals || 0, 'deal')}${checks}</small></div>
      ${full ? `<a class="watch" href="${matchLink(full)}" title="Watch this matchup">Watch ›</a>` : '<span></span>'}
      ${power ? `<div class="power">${esc(power)}</div>` : ''}</div>`;
  }).join('');
  const design = rt.design && rt.design.round_deals ? `Each pairing is checked after every ${rt.design.round_deals} deals, at most ${plural(rt.design.max_rounds, 'time')}, against a bar set so the chance of naming a wrong winner stays under ${Math.round(rt.design.alpha * 100)}% across all pairings.` : '';
  return section('Matchups', 'who beats whom, in bb/100 over duplicate deals', `<div class="pair-list">${rows}</div>`,
    `${design} <b>Wins</b>: the edge cleared that bar. <b>Even</b>: the two are within ±15 bb/100 of each other. <b>Undecided</b>: neither yet. The range is 95%.`);
}

function headToHead(rt) {
  const paired = new Set(rt.pairings.flatMap(p => [p.a, p.b]));
  const names = rt.players.filter(p => !p.quarantined && paired.has(p.player)).map(p => p.player);
  if (names.length < 3 || !rt.pairings.length) return '';
  const cell = {};
  for (const p of rt.pairings) {
    const lo = p.ci95 ? p.ci95[0] : null, hi = p.ci95 ? p.ci95[1] : null;
    const clear = lo != null && (lo > 0 || hi < 0);
    cell[p.a + '|' + p.b] = { v: p.edge, clear, p };
    cell[p.b + '|' + p.a] = { v: -p.edge, clear, p };
  }
  const max = Math.max(1, ...rt.pairings.map(p => Math.abs(p.edge || 0)));
  const ids = Object.fromEntries(state.index.map(m => [m.id.split('/').pop(), m.id]));
  const table = `<table class="matrix"><thead><tr><th></th>${names.map(n => `<th title="${plain(n)}">${monogram(n, 'sm')}</th>`).join('')}</tr></thead><tbody>
    ${names.map(a => `<tr><th>${player(a)}</th>${names.map(b => {
      if (a === b) return '<td class="cell self"></td>';
      const c = cell[a + '|' + b];
      if (!c) return '<td class="cell dim">·</td>';
      const alpha = c.clear ? (0.12 + 0.3 * Math.sqrt(Math.abs(c.v) / max)).toFixed(2) : 0;
      const bg = c.clear ? `background:rgba(var(--${c.v > 0 ? 'pos' : 'neg'}-rgb),${alpha})` : '';
      const mu = (c.p.matchups || []).slice(-1)[0];
      const full = mu && (state.index.some(m => m.id === mu) ? mu : ids[String(mu).split('/').pop()]);
      return `<td class="cell" style="${bg}"><a ${full ? `href="${matchLink(full)}"` : ''} title="${plain(a)} against ${plain(b)}">
        <span class="v ${c.clear ? cls(c.v) : 'soft'}">${signed(c.v, 0)}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  return section('Head to head', 'bb/100, row against column', table,
    'Shaded when the 95% range excludes zero. A cell opens that matchup.', { scroll: true });
}

function styleTable(rt) {
  const st = rt.stats || {};
  const names = rt.players.map(p => p.player).filter(p => st[p] && st[p].hands);
  if (!names.length) return '';
  const bar = v => v == null ? '' : `<span class="bar"><span style="width:${Math.round(v * 100)}%"></span></span>`;
  const rows = names.map(p => {
    const s = st[p];
    const af = s.af_all_aggression ? '<span title="bet or raised after the flop and never called">∞</span>'
      : s.af == null ? '<span class="dim">–</span>' : s.af.toFixed(1);
    const ties = s.sd_ties ? ` <small class="soft" title="split pots at showdown, not counted as wins">+${s.sd_ties} split</small>` : '';
    return `<tr class="${s.hands < 100 ? 'thin' : ''}"><td>${player(p)}</td>
      <td class="num">${pct(s.vpip)}${bar(s.vpip)}</td><td class="num">${pct(s.pfr)}${bar(s.pfr)}</td>
      <td class="num">${af}</td><td class="num">${pct(s.wtsd)}</td><td class="num">${pct(s.wsd)}${ties}</td>
      <td class="num">${isBot(p) ? '<span class="dim">–</span>' : pct(s.talk)}</td>
      <td class="num soft">${s.hands.toLocaleString()}</td></tr>`;
  }).join('');
  return section('How they play', '', `<table><thead><tr><th>Player</th>
    ${th('VPIP', 'Voluntarily put money in the pot: called or raised before the flop', 'num')}
    ${th('PFR', 'Raised before the flop', 'num')}
    ${th('Aggression', 'After the flop: bets and raises per call', 'num')}
    ${th('To showdown', 'Of the hands where it saw a flop, how many reached showdown', 'num')}
    ${th('Won showdown', 'Of its showdowns, how many it won outright (split pots apart)', 'num')}
    ${th('Talk', 'Share of its decisions that came with something said to the opponent', 'num')}
    ${th('Hands', '', 'num')}</tr></thead><tbody>${rows}</tbody></table>`,
    `<b>VPIP</b> and <b>PFR</b>: how often it plays a hand, and how often it raises with it; heads-up, strong players are
    usually above 50%. <b>Aggression</b> above 2: it bets far more than it calls after the flop. Rows under 100 hands are dimmed.`, { scroll: true });
}

// ---- games ----------------------------------------------------------------------------------------
async function renderMatches(id, params, soft) {
  if (!state.index.length) {
    $app.innerHTML = '<div class="card-box empty"><h2>No matchups in this run yet</h2></div>';
    return;
  }
  const started = state.index.filter(m => m.started);
  const cur = state.index.find(m => m.id === id) || started.find(m => m.live) || started[0] || state.index[0];
  const own = (m, p) => esc(nameParts((m.labels || {})[p] || labelText(p))[0]);
  Object.assign(state.labels, cur.labels || {});
  const groups = [['Playing now', state.index.filter(m => m.live)], ['Played', started.filter(m => !m.live)],
    ['Not started', state.index.filter(m => !m.started)]];
  const list = groups.filter(g => g[1].length).map(([title, ms]) => `<div class="group">${title}</div>` + ms.map(m => {
    const where = state.run === '*' ? `${esc(runLabel(m.run))} · ` : '';
    const round = state.index.some(x => x.round > 1 && x.a === m.a && x.b === m.b) ? `round ${m.round} · ` : '';
    const meta = `<span>${where}${!m.started ? 'waiting for a free seat' : `${round}${m.pairs_done}/${m.deals_total} deals`}</span>`
      + (m.pairs_done ? `<b class="${cls(m.pair_net_a)}" title="${own(m, m.a)}'s net over both plays">${signedBB(m.pair_net_a, m.bb)}</b>` : '');
    return `<a ${m.started ? `href="${matchLink(m.id)}"` : ''} class="${m === cur ? 'active' : ''} ${m.started ? '' : 'planned'}">
      <div class="row"><span class="duo">${monogram(m.a, 'sm')}${monogram(m.b, 'sm')}</span>
        <span class="pair-names">${own(m, m.a)}<span class="vs">vs</span>${own(m, m.b)}</span>${m.live ? '<span class="live-dot"></span>' : ''}</div>
      <div class="meta">${meta}</div></a>`;
  }).join('')).join('');

  let body, after = null;
  try {
    if (cur.started) [body, after] = await matchBody(cur, params);
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
  const listScrollX = document.querySelector('.mlist')?.scrollLeft || 0;
  $app.innerHTML = `<div class="layout"><div class="card-box mlist" role="navigation" aria-label="Matchups">${list}</div><div class="mbody">${body}</div></div>`;
  if (soft) window.scrollTo(0, scroll);
  const ml = document.querySelector('.mlist');
  if (ml) {
    ml.scrollTop = listScroll;
    ml.scrollLeft = listScrollX;
    if (!soft) ml.querySelector('a.active')?.scrollIntoView({ block: 'nearest', inline: 'center' });
  }
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

async function pairingFor(cur) {
  if (state.run === '*') return null;
  try {
    const rt = normalize(await api('ratings', ['run', state.run]));
    return rt.pairings.find(p => (p.a === cur.a && p.b === cur.b) || (p.a === cur.b && p.b === cur.a)) || null;
  } catch (e) { return null; }
}

async function matchBody(cur, params) {
  const [[ma, mb], pairing] = await Promise.all([Promise.all(cur.matches.map(x => loadMatch(x.id))), pairingFor(cur)]);
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

  // the headline is chips: net big blinds over both plays of every finished deal
  const net = done.reduce((s, k) => s + nets[k], 0);
  const wins = done.filter(k => nets[k] > 0).length, losses = done.filter(k => nets[k] < 0).length;
  const leader = net > 0 ? cur.a : net < 0 ? cur.b : null;
  const rate = done.length ? net / bb / (2 * done.length) * 100 : 0;
  const v = pairing ? verdict(pairing, plain) : null;
  const face = (p, side) => `<div class="face ${side} ${leader === p ? 'ahead' : ''}">${monogram(p, 'xl')}<div class="face-name">${name(p)}</div></div>`;
  const scoreboard = `<div class="scoreboard card-box">
      ${face(cur.a, 'left')}
      <div class="score-mid">
        ${done.length ? (leader ? `<div class="score-who">${plain(leader)} is up</div><div class="score-big">${trim(Math.abs(net) / bb)}<small>bb</small></div>
          <div class="score-rate">${trim(Math.abs(Math.round(rate * 10) / 10))} bb/100 hands</div>`
          : '<div class="score-big even">Even</div>') : '<div class="score-who">No deals finished yet</div>'}
        <div class="score-cap">deals won ${wins}–${losses}${done.length - wins - losses ? ` · ${done.length - wins - losses} even` : ''} · ${done.length} of ${total} played</div>
        ${v ? `<div class="score-verdict"><span class="verdict ${v.tone}">${esc(v.short)}</span><span class="small soft">${esc(v.long)}</span></div>` : ''}
      </div>
      ${face(cur.b, 'right')}</div>`;

  const max = Math.max(1, ...done.map(k => Math.abs(nets[k])));
  const strip = Array.from({ length: total }, (_, k) => {
    const sel = k === pick ? 'sel' : '';
    if (!(k in nets)) return `<button class="chipbtn ${played.has(k) ? 'half' : 'empty'} ${sel}" ${played.has(k) ? `data-deal="${k}" title="Deal ${k + 1}: being played"` : 'disabled'}><span>${k + 1}</span></button>`;
    const x = nets[k];
    return `<button data-deal="${k}" class="chipbtn ${sel}" style="--c:${chipColor(x, max)}" title="Deal ${k + 1}: ${x === 0 ? 'even' : `${plain(x > 0 ? cur.a : cur.b)} won ${trim(Math.abs(x) / bb)} bb`} over both plays"><span>${k + 1}</span></button>`;
  }).join('');
  const short = p => esc(nameParts(labelText(p))[0]);
  const legend = `<div class="strip-legend">
    <span><span class="swatch" style="background:var(--pos-chip)"></span>${short(cur.a)} won the deal</span>
    <span><span class="swatch" style="background:var(--neg-chip)"></span>${short(cur.b)} won it</span>
    <span>brighter, bigger swing</span></div>`;

  const order = [...played].sort((x, y) => x - y);
  const i = order.indexOf(pick);
  const dealNet = pick in nets ? nets[pick] : null;
  const rounds = state.index.some(m => m.round > 1 && m.a === cur.a && m.b === cur.b);
  const outcome = dealNet == null ? 'in progress' : dealNet === 0 ? 'even over both plays'
    : `<b>${short(dealNet > 0 ? cur.a : cur.b)}</b> wins it by <b class="gold">${trim(Math.abs(dealNet) / bb)} bb</b> over both plays`;
  const bar = `<div class="deal-bar"><h2>${rounds ? `<span class="round">Round ${cur.round}</span>` : ''}Deal ${pick + 1}<small> of ${total}</small></h2>
    <div class="nav"><button data-deal="${order[i - 1]}" ${i > 0 ? '' : 'disabled'} aria-label="Previous deal" title="Previous deal (←)">←</button>
      <button data-deal="${order[i + 1]}" ${i < order.length - 1 ? '' : 'disabled'} aria-label="Next deal" title="Next deal (→)">→</button></div>
    <span class="deal-outcome soft">${outcome}</span></div>`;

  const plays = [ma, mb].filter(Boolean).map((m, j) => ({ m, j, ...PokerTable.steps(m.events[pick] || []) }));
  const last = Math.max(0, ...plays.map(p => p.steps.length - 1));
  const key = cur.id + '#' + pick;
  const rp = state.replay && state.replay.key === key ? state.replay : { key, s: last, follow: true, playing: false, fresh: true };
  if (rp.fresh && params.has('step')) { rp.s = Math.max(0, Math.min(last, Number(params.get('step')) - 1 || 0)); rp.follow = rp.s >= last; }
  if (rp.follow && !rp.playing) rp.s = last;
  rp.last = last;
  state.replay = rp;
  state.view.plays = plays;
  state.view.bb = bb;

  const controls = `<div class="controls card-box" role="group" aria-label="Replay">
    <button class="ctl" data-ctl="start" title="Back to the deal">⏮</button>
    <button class="ctl" data-ctl="prev" title="Step back (,)">◀</button>
    <button class="ctl play" data-ctl="play" title="Play both hands step by step (space)">▶ Play</button>
    <button class="ctl" data-ctl="next" title="Step forward (.)">▶</button>
    <button class="ctl" data-ctl="end" title="To the end">⏭</button>
    <input type="range" class="scrub" min="0" max="${last}" value="${rp.s}" aria-label="Step">
    <span class="step-text"></span>
    <button class="toggle" id="mind-toggle" aria-pressed="${state.mind}" title="Show every decision's reasoning (r)">Reasoning</button>
    <span class="keys"><span class="kbd">←</span><span class="kbd">→</span> deals · <span class="kbd">space</span> play · <span class="kbd">,</span><span class="kbd">.</span> step · <span class="kbd">r</span> reasoning</span></div>`;

  const body = `${scoreboard}<div class="strip-wrap"><div class="strip" style="--cols:${total <= 30 ? Math.max(total, 16) : Math.ceil(total / Math.ceil(total / 25))}">${strip}</div>${legend}</div>
    ${bar}${controls}
    <div class="pair ${state.mind ? 'show-mind' : ''}">${plays.map(p => playPanel(p, cur, bb)).join('')}</div>`;
  const after = () => {
    $app.querySelectorAll('[data-deal]').forEach(b => b.onclick = () => go(cur, Number(b.dataset.deal)));
    $app.querySelectorAll('[data-ctl]').forEach(b => b.onclick = () => control(b.dataset.ctl));
    const sc = $app.querySelector('.scrub');
    if (sc) sc.oninput = () => { stopPlay(); setStep(Number(sc.value)); };
    $app.querySelectorAll('.ev[data-step]').forEach(el => el.addEventListener('click', e => {
      if (e.target.closest('.why')) return;
      stopPlay(); setStep(Number(el.dataset.step));
    }));
    $app.querySelectorAll('.why').forEach(b => b.onclick = () => {
      const ev = b.closest('.ev');
      ev.classList.toggle('open');
      b.setAttribute('aria-expanded', String(ev.classList.contains('open')));
    });
    document.getElementById('mind-toggle')?.addEventListener('click', toggleMind);
    const fresh = rp.fresh;
    rp.fresh = false;
    applyStep(fresh ? null : 'keep');
    if (fresh) $app.querySelectorAll('.ptable').forEach(t => t.classList.add('deal-in'));
    if (rp.playing) schedulePlay();
  };
  return [body, after];
}

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
      const mind = hasMind ? `<div class="mind">${t.reasoning ? esc(t.reasoning) : ''}${t.thinking
        ? `<span class="lbl">The model's own thinking</span><span class="thinking">${esc(t.thinking)}</span>` : ''}</div>` : '';
      log += `<div class="ev move" data-step="${s}">${monogram(players[e.seat], 'xs')}<span class="actor">${shortName(e.seat)}</span>
        <span class="what"><span class="verb ${e.action}">${VERB[e.action] || esc(e.action)}${amt}</span>${badges.join('')}</span>
        ${hasMind ? '<button class="why" aria-expanded="false" title="This decision\'s private reasoning">why?</button>' : ''}
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

// ---- replay: one step index drives both plays; each play stops at its own last step ---------------
function applyStep(mode) {
  const v = state.view, rp = state.replay;
  if (!v || !v.plays || !rp) return;
  const s = rp.s;
  v.plays.forEach(p => {
    const panel = $app.querySelector(`.play[data-play="${p.j}"]`);
    if (!panel || !p.steps.length) return;
    const mine = Math.min(s, p.steps.length - 1);
    const prev = mode !== 'keep' && p.shown != null && p.shown < mine ? p.steps[p.shown] : null;
    PokerTable.show(panel, p.steps[mine], { fmt: c => bbs(c, v.bb), prev });
    p.shown = mine;
    panel.querySelectorAll('[data-step]').forEach(el => {
      const k = Number(el.dataset.step);
      el.classList.toggle('future', k > mine);
      el.classList.toggle('now', k === mine && s < rp.last);
    });
  });
  const sc = $app.querySelector('.scrub');
  if (sc) { sc.max = rp.last; sc.value = s; }
  const txt = $app.querySelector('.step-text');
  if (txt) txt.textContent = s >= rp.last ? 'End of the hand' : `Step ${s + 1} of ${rp.last + 1}`;
  $app.querySelector('.deal-outcome')?.classList.toggle('spoiler', s < rp.last);
  const pb = $app.querySelector('[data-ctl="play"]');
  if (pb) pb.textContent = rp.playing ? '❚❚ Pause' : s >= rp.last ? '↺ Replay' : '▶ Play';
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
  rp.timer = setTimeout(() => {
    if (!rp.playing) return;
    if (rp.s >= rp.last) { stopPlay(); return; }
    setStep(rp.s + 1);
    if (rp.s >= rp.last) stopPlay(); else schedulePlay();
  }, 1100);
}
function startPlay() {
  const rp = state.replay;
  if (!rp) return;
  if (rp.s >= rp.last) setStep(0);
  rp.playing = true;
  rp.follow = false;
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
