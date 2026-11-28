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
  autoNext: store.get('autonext', true), mode: store.get('timeline', 'aligned'), listOpen: false, mapOpen: null, metric: store.get('metric', 'vpip'),
  speed: store.get('speed', 'normal'), order: store.get('order', 'decided') };

const LIVE_POLL = 6000, SLOW_POLL = 60000, PENDING_POLL = 1500, STALE_POLL = 4000;

// ---- ratings: never awaited by navigation or replay ------------------------------------------------
// The server answers from its cache at once ({cache: {fresh, computed, refreshing}}), or {pending} the first
// time a run is ever computed. The last answer per run is kept here, so a page renders from it immediately.
const ratingsStore = new Map();
async function fetchRatings(run) {
  const raw = await api('ratings', ['run', run]);
  if (raw && raw.pending) return null;
  ratingsStore.set(run, raw);
  return raw;
}
const ratingsUpToDate = raw => !!raw && (!raw.cache || raw.cache.fresh || !!raw.cache.error);
// Fetch in the background; when the answer is new, re-render the page in place (the replay keeps its step).
let ratingsTimer = null;
function ratingsLater(run, delay = 0) {
  clearTimeout(ratingsTimer);
  ratingsTimer = setTimeout(async () => {
    const before = ratingsStore.get(run);
    let raw = null;
    try { raw = await fetchRatings(run); } catch (e) { return; }
    if (state.run !== run) return;
    if (!raw) { ratingsLater(run, PENDING_POLL); return; }
    const changed = !before || (before.cache && raw.cache && before.cache.computed !== raw.cache.computed) || (!before.cache !== !raw.cache);
    if (changed && !state.replay?.playing) route({ soft: true });
    if (!ratingsUpToDate(raw)) ratingsLater(run, STALE_POLL);
  }, delay);
}

// ---- formatting ---------------------------------------------------------------------------------
const cls = x => x > 0 ? 'pos' : x < 0 ? 'neg' : '';
const pct = x => x == null ? '<span class="dim">–</span>' : `${Math.round(x * 100)}%`;
// Chips as dollars at the $10/$20 table (Core.money): every pot, bet, stack and result goes through these.
const cash = (chips, bb) => Core.chipMoney(chips, bb);
const signedCash = (chips, bb) => Core.chipMoneySigned(chips, bb);
const money = Core.money, moneySigned = Core.moneySigned;
// "50 bb/100" explained where it is shown: what it measures, and what it is worth at these stakes.
const rateTip = rate => `Big blinds won per 100 hands: the win rate, whatever the stakes.${rate != null && isFinite(rate) ? ` At $10/$20 this is ${Core.perHundred(rate)}.` : ''}`;
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
const humanRunName = s => {
  if (!s || s === 'All runs') return s || '';
  if (/[A-Z\s]/.test(s)) return s;   // already a custom human title
  return s.split(/[-_]+/).filter(Boolean).map(w => /^[0-9a-f]{6,}$/i.test(w) ? `(${w})` : w.toLowerCase() === 'h2h' ? 'H2H' : w[0].toUpperCase() + w.slice(1)).join(' ');
};
const runLabel = r => {
  const rec = state.runs.find(x => x.run === r);
  const raw = (rec && rec.title) || r.split('/').pop();
  return rec && rec.all ? 'All runs' : humanRunName(raw);
};
const runLink = (run, rest = '') => `#/r/${encodeURIComponent(run)}${rest}`;
// A tournament's runs, one per track, official first; a run outside any tournament is a set of one.
const tourKey = r => r && r.tournament ? `${r.run.split('/').slice(0, -1).join('/')}|${r.tournament}` : null;
function trackSet(run) {
  const r = state.runs.find(x => x.run === run);
  const k = tourKey(r);
  if (!k) return r ? [r] : [];
  return state.runs.filter(x => !x.all && tourKey(x) === k).sort((x, y) => (x.track === 'official' ? 0 : 1) - (y.track === 'official' ? 0 : 1));
}
const siblingTrack = (run, track) => trackSet(run).find(r => r.track === track) || null;
const TRACK_TITLE = { official: 'Official standings', adaptation: 'Memory mode (models remember the session)' };
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
const isReference = p => isBot(p) && /reference/i.test(`${p} ${state.labels[p] || ''}`);
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
  const tag = isReference(p) ? '<span class="tag ref" title="A fixed, versioned reference policy: the yardstick, not a contestant">reference</span>'
    : isBot(p) ? '<span class="tag">bot</span>' : '';
  return `<span class="nm" title="${esc(labelText(p))}">${esc(model)}</span>${effort ? effortChip(level) : ''}${tag}`;
}
const plain = p => esc(labelText(p));
// a label's parenthetical (protocol details) as a quiet second line, never inside the level chip
const detail = p => { const d = nameParts(labelText(p))[2]; return d ? `<div class="pdetail" title="${esc(d)}">${esc(d)}</div>` : ''; };
const player = (p, extra = '', { details = true } = {}) => `<div class="player">${monogram(p)}<div class="pname"><div class="name">${name(p)}</div>${details ? detail(p) : ''}${extra}</div></div>`;
const SUITS = ['♠', '♥', '♦', '♣'];
let sectionCount = 0;
const section = (title, aside, body, caption = '', opts = {}) => {
  const suit = SUITS[sectionCount++ % 4];
  const head = `<h2><span class="suit ${suit === '♥' || suit === '♦' ? 'red' : ''}">${suit}</span>${title}</h2>${aside ? `<span class="aside">${aside}</span>` : ''}`;
  const inner = `<div class="card-box ${opts.scroll ? 'scroll' : ''}">${body}</div>${caption ? `<p class="caption">${caption}</p>` : ''}${opts.after || ''}`;
  if (opts.fold) {
    const open = opts.open ?? store.get('fold.' + opts.id, false);
    return `<section class="folded" ${opts.id ? `id="${opts.id}"` : ''}><details class="fold" data-fold="${opts.id}" ${open ? 'open' : ''}>
      <summary class="section-head">${head}<span class="fold-mark" aria-hidden="true"></span></summary>${inner}</details></section>`;
  }
  return `<section ${opts.id ? `id="${opts.id}"` : ''}><div class="section-head">${head}</div>${inner}</section>`;
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
    const reuse = opts.nav && state.loadedAt && Date.now() - state.loadedAt < 60000;
    if (!reuse) {
      state.runs = await api('runs');
      state.checked = new Date();
    }
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
    const mine = new Set(trackSet(run).map(r => r.run));
    const option = r => {
      const set = trackSet(r.run);
      const live = set.some(x => x.live) && !r.all;
      if (r.all) return `<option value="${esc(r.run)}" ${mine.has(r.run) ? 'selected' : ''}>All runs (combined)</option>`;
      const hands = set.reduce((s, x) => s + (x.hands || 0), 0);
      const tracks = set.length > 1 ? ' · 2 modes' : r.track === 'adaptation' ? ' · memory mode' : '';
      const size = hands ? ` · ${hands.toLocaleString('en-US')} hands` : '';
      return `<option value="${esc(r.run)}" ${mine.has(r.run) ? 'selected' : ''}>${esc(runLabel(r.run))}${size}${tracks}${live ? ' · live' : ''}</option>`;
    };
    const seenT = new Set();
    const all = state.runs.filter(r => r.all), each = state.runs.filter(r => !r.all).filter(r => {
      const lead = trackSet(r.run)[0];
      if (seenT.has(lead.run)) return false;
      seenT.add(lead.run);
      return true;
    }).map(r => trackSet(r.run)[0]);
    const isSmall = r => /^(smoke|verify|test)/i.test(r.run.split('/').pop()) || ((r.hands || 0) < 40 && (r.players || []).length <= 2);
    const mainRuns = each.filter(r => !isSmall(r)), quickRuns = each.filter(isSmall);
    $run.innerHTML = all.map(option).join('')
      + (mainRuns.length ? `<optgroup label="Tournaments">${mainRuns.map(option).join('')}</optgroup>` : '')
      + (quickRuns.length ? `<optgroup label="Quick tests">${quickRuns.map(option).join('')}</optgroup>` : '');
    $run.onchange = () => { location.hash = runLink($run.value); };
    const here = state.runs.find(r => r.run === run);
    const tab = parts[2] === 'm' ? 'matches' : parts[2] === 'q' && here.quiz ? 'quiz' : parts[2] === 'scorecards' ? 'scorecards' : parts[2] === 'planner' ? 'planner' : 'board';
    document.querySelectorAll('.top nav a').forEach(a => {
      a.classList.toggle('active', a.dataset.tab === tab);
      a.hidden = a.dataset.tab === 'quiz' && !here.quiz;
      a.href = runLink(run, { matches: '/m', quiz: '/q', scorecards: '/scorecards', planner: '/planner' }[a.dataset.tab] || '');
    });
    if (!reuse || state.indexRun !== run) {
      state.index = await api('index', ['run', run]);
      state.indexRun = run;
      state.loadedAt = Date.now();
    }
    state.labels = {}; state.kinds = {};   // per run: an id can name another model, or level, in another run
    for (const m of state.index) { Object.assign(state.kinds, m.kinds || {}); Object.assign(state.labels, m.labels || {}); }
    sectionCount = 0;
    if (tab === 'quiz') {
      state.view = null;
      await renderQuiz(parts[3] ? decodeURIComponent(parts.slice(3).join('/')) : '');
    } else if (tab === 'scorecards') {
      state.view = null;
      await renderScorecards(new URLSearchParams(query || ''));
    } else if (tab === 'planner') {
      state.view = null;
      await renderPlanner(new URLSearchParams(query || ''));
    } else if (tab === 'matches') {
      await renderMatches(parts[3] ? decodeURIComponent(parts.slice(3).join('/')) : '', new URLSearchParams(query || ''), soft);
    } else {
      state.view = null;
      const y = window.scrollY;
      await renderBoard(new URLSearchParams(query || ""));
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
  const bad = cur.failures && cur.failures.deals;
  const checked = (bad ? `<a class="label neg" href="${runLink(state.run)}" title="Deals a model's CLI returned an error on; they are not counted. Open the standings for the errors.">${plural(bad, 'deal')} failed</a>` : '')
    + (state.checked ? `<span class="label checked" title="When this page last asked for news">checked ${clock(state.checked)}</span>` : '');
  if (playing) $status.innerHTML = `<span class="live-dot"></span><span>Live<span class="label"> · ${plural(playing, 'table')}</span></span>${checked}`;
  else if (cur.finished) $status.innerHTML = `<span class="label">Finished</span>`;
  else if (cur.status === 'stopped') $status.innerHTML = `<span class="idle-dot"></span><span>Stopped<span class="label"> · last action ${ago(cur.updated)}</span></span>${checked}`;
  else if (cur.between_rounds) $status.innerHTML = `<span class="idle-dot"></span><span>Between rounds<span class="label"> · last action ${ago(cur.updated)}</span></span>${checked}`;
  else $status.innerHTML = `<span class="idle-dot"></span><span>Paused<span class="label"> · last action ${ago(cur.updated)}</span></span>${checked}`;
}

// ---- standings ----------------------------------------------------------------------------------
// How this run's results are judged, in plain English: one pill that opens the full rules.
function runStatus(rt) {
  const d = rt.design || {};
  if (rt.pooled) return { text: 'Combined history', tone: 'quiet', tip: 'All past runs combined for browsing. Not a single controlled test.' };
  if (rt.track === 'adaptation') return { text: 'Memory mode · Exploratory', tone: 'warn', tip: 'Models remember earlier hands across 50-hand sessions. Results are exploratory, not official verdicts.' };
  if (voidReason(d)) return { text: 'Code changed mid-run', tone: 'warn', tip: 'The arena code changed after this test started, so results are scores only.' };
  if (rt.unregistered) return { text: 'Practice run · Scores only', tone: 'warn', tip: 'Played without a pre-locked statistical test plan. Scores show who won these hands, not a final verdict.' };
  if (d.round_deals || d.registered || d.registration) return { text: officialFree(rt) ? 'Official test' : 'Planned test', tone: 'ok', tip: 'Pre-planned statistical test that stops once a winner is mathematically proven.' };
  return { text: 'Scores only', tone: 'quiet', tip: 'Shows who won these hands, without a formal statistical test.' };
}

// Who played, in one phrase: "3 models", "2 models + 1 bot".
function whoText(ids) {
  const seen = new Set();
  const uniq = [...ids].filter(p => !seen.has(labelText(p)) && seen.add(labelText(p)));
  const models = uniq.filter(p => !isBot(p)).length, bots = uniq.length - models;
  return [models ? plural(models, 'model') : '', bots ? plural(bots, 'bot') : ''].filter(Boolean).join(' + ');
}

// The slim run header: title, one summary line, the status and integrity pills, and where to go next.
// The table every hand is played at, for a header: "$10/$20 No-Limit Hold'em · $2,000 stacks".
function stakesBit(game) {
  const m = game ? null : state.index.find(x => x.started && x.stack) || state.index.find(x => x.started);
  const k = Core.stakesOf(game || (m ? { bb: m.bb, sb: m.sb, stack: m.stack } : null));
  const deep = k.stack ? ` Both players start every hand with ${k.stack} (${k.stackBB} big blinds).` : '';
  return `<span class="stakes" title="${esc(`A ${k.blinds} no-limit cash game: the small blind is ${k.sb}, the big blind ${k.bb}.${deep} ${Core.chipNote(game ? game.big_unit || game.bb : m && m.bb)}`)}">${esc(k.line)}</span>`;
}
function runHeader(r, rt, raw) {
  const st = runStatus(rt);
  const open = store.get('plan-open', false);
  let title, bits, meter = '';
  if (r.all) {
    title = 'All runs';
    bits = [plural((rt.runs || []).length, 'run'), whoText(rt.players.map(p => p.player)), plural(2 * (rt.deals || 0), 'hand')];
  } else {
    title = esc(runLabel(state.run));
    const ids = [...state.index.flatMap(m => [m.a, m.b]), ...rt.roster.map(x => x.player)];
    let when;
    if (r.finished) when = `Finished ${ago(r.updated)}`;
    else if (r.status === 'stopped') when = `Stopped ${ago(r.updated)}`;
    else if (r.between_rounds) when = `Round ${r.round || 1} done · more to come`;
    else if (r.live) when = `<span class="live-dot" aria-hidden="true"></span> Live${r.eta_seconds != null ? ` · about ${duration(r.eta_seconds)} left` : ''}`;
    else when = `Paused · last action ${ago(r.updated)}`;
    const hands = r.finished || !r.hands_planned ? plural(r.hands || 0, 'hand') : `${(r.hands || 0).toLocaleString('en-US')} of ${plural(r.hands_planned, 'hand')}`;
    bits = [whoText(ids), hands, when];
    if (r.failures && r.failures.deals) bits.push(`<span class="neg" title="Deals a model's CLI returned an error on; they are not counted">${plural(r.failures.deals, 'deal')} failed</span>`);
    if (!r.finished && r.hands_planned) meter = `<div class="meter thin rh-meter"><span style="width:${(Math.min(1, r.hands / r.hands_planned) * 100).toFixed(1)}%"></span></div>`;
  }
  const fx = raw ? Core.forensicsPill(raw.forensics_summary) : null;
  const fxPill = fx && !r.all ? `<a class="pill ${fx.tone}" href="${runLink(state.run, '/scorecards?focus=forensics')}"
      title="${esc(`Cheating & safety checks: ${plural(fx.total, 'flag')} across tools/sandbox, table-talk prompt injection, parser probing and private reasoning`)}">${fx.total ? `⚠ ${plural(fx.total, 'cheating flag')}` : '✓ No cheating detected'}</a>` : '';
  const live = state.index.find(m => m.live);
  const rawSlug = !r.all && state.run && state.run !== '.' ? state.run.split('/').pop() : '';
  const slugTip = rawSlug && rawSlug !== runLabel(state.run) ? ` title="Run folder: runs/${esc(state.run)}"` : '';
  return `<header class="run-head">
    <div class="rh-main"><h1${slugTip}>${title}</h1><div class="rh-sub">${bits.filter(Boolean).join(' · ')}</div><div class="rh-stakes">${stakesBit(r.all ? {} : null)}</div>${meter}</div>
    <div class="rh-actions">
      <button class="pill ${st.tone}" id="judged-toggle" aria-expanded="${open}" aria-controls="judged" title="${esc(st.tip || 'How this run is judged')}">${esc(st.text)}<i class="caret" aria-hidden="true"></i></button>
      ${fxPill}
      <a class="cta" href="${live ? matchLink(live.id) : runLink(state.run, '/m')}"><span class="cta-icon" aria-hidden="true">▶</span>${live ? 'Watch live' : 'Watch hands'}</a>
      ${r.all ? '' : `<a class="sec-link" href="${runLink(state.run, '/scorecards')}">Scorecards <span aria-hidden="true">→</span></a>`}</div></header>`;
}

// The archive's caveats, for the methodology drawer.
function historyNotes(r, rt) {
  const notes = [`Combines every finished run for each model and reasoning level${r.started_at ? ` since ${new Date(r.started_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}.`];
  if ((rt.left_out || []).length) notes.push(`Excluded (different game settings): ${rt.left_out.map(x => esc(runLabel(x.run))).join(', ')}.`);
  if ((rt.code_versions || []).length > 1) notes.push(`Played across ${rt.code_versions.length} versions of the arena.`);
  const gone = (rt.runs || []).filter(x => !state.runs.some(y => y.run === x.run)).length;
  if (gone) notes.push(`${plural(gone, 'older run')} kept in history after ${gone === 1 ? 'its folder was' : 'their folders were'} deleted.`);
  const self = (rt.runs || []).reduce((n, x) => n + (x.self_play || 0), 0);
  if (self) notes.push(`${plural(self, 'deal')} where a model played itself don't count.`);
  if ((rt.other_track_runs || []).length) notes.push(`Memory-mode runs (${plural(rt.other_track_runs.length, 'run')} with 50-hand sessions) are kept separate and not combined here.`);
  return notes.join(' ');
}

// What the arena is, in two sentences: the first thing in the methodology drawer.
function aboutText(rt) {
  const ids = rt && rt.players.length ? rt.players.map(p => p.player) : state.index.flatMap(m => [m.a, m.b]);
  const models = ids.filter(p => !isBot(p));
  return `<p><b>${ids.length && !models.length ? 'Baseline bots play heads-up No-Limit Hold\'em to serve as fixed benchmarks.' : 'AI models play heads-up No-Limit Hold\'em against each other.'}</b>
    Every deal is played twice with the seats swapped, so both models play the exact same cards and card luck mostly cancels out. The table is a ${Core.stakesOf().blinds} no-limit cash game with ${money(100)} stacks, reset every hand. Win rates are in <b>bb/100</b> (big blinds won per 100 hands, above 0 = winning): at these stakes 50 bb/100 is ${money(50)} per 100 hands.
    A <span class="decided-badge">Decided</span> badge means a pre-planned statistical test proved the winner; everything else shows who is leading so far.</p>`;
}

const fmtStrength = v => v == null ? '–' : signed(v, 0);
const STRENGTH_TIP = 'Fitted strength (bb/100): estimated win rate across all deals, centered so the group averages 0. Compare any two players by the gap between their numbers.';
const OBSERVED_TIP = 'Simple average of this player\'s direct head-to-head margins (bb/100), counting every opponent equally.';
const ciText = ci => ci ? `${signed(ci[0], 0)} to ${signed(ci[1], 0)}` : '';
const levelPct = x => `${Math.round(x * 1000) / 10}%`;
// An ordinary interval, said as what it is: fixed-time and descriptive, never a formal guarantee here.
const descRange = (ci, x) => ci ? `<span class="ci-desc" title="${esc(Core.DESCRIPTIVE_TIP)}">${ciText(ci)} <i>rough range</i></span>` : '';
// An official pairing's anytime-valid interval, the primary range, in the orientation `up` (a's when true).
const officialRange = (p, up = true) => {
  const o = Core.officialInterval(p);
  if (!o) return null;
  const ci = up ? [o.lo, o.hi] : [-o.hi, -o.lo];
  return { ci, html: `<span class="ci-official" title="Official margin of error (anytime-valid${o.level ? ` ${levelPct(o.level)}` : ''}): stays statistically valid whenever play stops, as long as matches are independent and models stay unchanged.">${ciText(ci)} <i>official range${o.level ? ` ${levelPct(o.level)}` : ''}</i></span>` };
};
// A chip when a headline rests on a model whose identity its tool never confirmed.
const identityChip = (rt, ids, given = false) => {
  const st = ids.map(x => isBot(x) ? null : Core.identityOf(rt.players.find(p => p.player === x), (rt.audit && rt.audit.players || {})[x]) || Core.identityOf(null, (rt.playerAudit || {})[x]));
  const who = ids.filter((x, i) => st[i] === 'unattested' || st[i] === 'partial');
  if (!who.length && !given) return '';
  return chip('Model ID not logged', 'quiet', `${who.length ? who.map(plain).join(', ') + ': its' : 'A'} CLI did not log back a server-confirmed model ID, so only the configured name is known.`);
};
const pid = x => Core.playerId(x);
// the interval's basis at the number: "within conversations" is flagged, it leaves out between-session spread
const BASIS_PLAIN = { within: 'single session', across: 'across sessions', mixed: 'mixed sessions' };
const basisTag = x => {
  const b = Core.basisOf(x);
  return b ? `<span class="basis ${b.key}" title="${esc(b.long)}">${esc(BASIS_PLAIN[b.key] || b.short)}</span>` : '';
};
const chip = (text, tone = '', tip = '') => `<span class="chip ${tone}" ${tip ? `title="${esc(tip)}"` : ''}>${text}</span>`;
// What a verdict rests on, as two things kept apart: the method (which test) and the evidence (whether the checks it
// needs hold here). A failed or unmade check stays visible whatever the method.
const FREE_TIP = 'Strict sequential test: makes no assumptions about the shape of hand scores, only that separate deals are independent and each model stays unchanged.';
function assumeChip(a, { compact = false } = {}) {
  if (!a) return '';
  if (a.method === 'free') {
    const method = chip('strict test', 'ok', FREE_TIP);
    const evidence = a.status === 'unsupported' ? chip(esc(a.label), 'warn', `A required check failed (${a.detail}), so the error guarantee does not hold here.`)
      : a.status === 'incomplete' ? chip(esc(a.label), 'quiet', `Some required checks could not be run (${a.detail}).`)
      : a.checks && a.checks.length ? chip('checks passed', 'ok', a.detail) : '';
    return compact ? (a.status === 'free' ? method : evidence) : method + evidence;
  }
  const plainLabel = a.status === 'unsupported' ? 'checks failed' : a.status === 'supported' ? 'checks passed' : a.status === 'free' ? 'strict test' : 'unchecked';
  return chip(esc(plainLabel), a.status === 'unsupported' ? 'warn' : a.status === 'supported' || a.status === 'free' ? 'ok' : 'quiet',
    a.status === 'free' ? FREE_TIP
      : a.status === 'unsupported' ? `Outside the range this test was calibrated for${a.detail ? ` (${a.detail})` : ''}: the error guarantee does not hold.`
      : a.status === 'supported' ? `Diagnostic checks passed${a.detail ? ` (${a.detail})` : ''}.` : 'Diagnostic checks were not run.');
}
const voidReason = (d, reg = state.rt && state.rt.registration) => (reg && reg.status === 'void' && ((reg.reasons || []).join('; ') || 'the code changed after it was registered')) || d && (d.registration_void || (d.registration && (d.registration.status === 'void' || d.registration.void)
  && (d.registration.reason || 'the code changed after it was registered')) || (d.registered === 'void' && (d.void_reason || 'the code changed after it was registered')));

// The headline for one group of players as one calm strip: who leads, by how much, and one badge saying
// whether that is decided. The full reasoning sits in the badge's tooltip and the methodology drawer.
function leaderStrip(rt, g, multi, first = true) {
  const by = Object.fromEntries(rt.players.map(p => [p.player, p]));
  const n = g.ranked.length;
  const verdicts = g.pairings.map(p => verdict(p, plain));
  const decided = verdicts.filter(v => v.decided || v.model).length;
  const ineligible = verdicts.filter(v => v.tone === 'bad').length;
  const settled = !rt.pooled && !rt.unregistered && g.pairings.length && g.pairings.every(p => ['a', 'b', 'even', 'exhausted', 'ineligible'].includes(p.verdict));
  const misfit = g.misfits > 0 && g.members.length > 2;
  const who = g.leader || g.drift_leader || g.qualified_leader || (rt.track === 'adaptation' && g.model_based_leader) || (n > 1 ? g.front_runner : null);
  const p = who && by[who];
  const descriptive = rt.pooled || rt.unregistered || voidReason(rt.design);
  const others = g.members.length === 2 ? 'the other player' : `each of the other ${g.members.length - 1} players`;
  let badge, tone, why;
  if (rt.track === 'adaptation') {
    const mb = g.model_based_leader;
    badge = mb ? 'Ahead in every matchup (exploratory)' : 'Leading so far';
    tone = 'points';
    why = n < 2 ? 'Only one player in this group has results so far.'
      : mb ? `Ahead of ${others} in memory mode. Because models remember earlier hands across long sessions, this result is exploratory and never counts as an official verdict.`
      : `Exploratory memory mode: ${decided} of ${plural(g.pairings.length, 'matchup')} ${decided === 1 ? 'shows' : 'show'} a clear lead.`;
  } else if (g.drift_leader) {
    const drifted = g.pairings.filter(x => !Core.countsForLeader(x) && verdictSides(x)?.winner === who);
    badge = drifted.length ? `${Core.driftWords(drifted[0]).phrase.replace(/^./, c => c.toUpperCase())} · needs re-test` : 'Needs re-test';
    tone = 'drift';
    why = `Not confirmed: it cleared the winning bar against ${others}, but the score shifted sharply between early and late hands. Re-run with fixed model versions to confirm.`;
  } else if (g.qualified_leader) {
    const failed = g.pairings.filter(x => x.assumption && x.assumption.method === 'free' && x.assumption.status === 'unsupported');
    badge = failed.length ? 'Leading · check failed' : 'Leading · checks incomplete';
    tone = 'qualified';
    why = failed.length ? `It beat ${others}, but a required reliability check failed (${failed.flatMap(x => x.assumption.failed).join(', ')}), so it does not count as a confirmed winner.`
      : `It beat ${others}, but at least one win fell outside the test's calibrated range, so it does not count as a confirmed winner.`;
  } else if (g.leader) {
    badge = 'Confirmed winner';
    tone = 'decided';
    why = `Confirmed: beat ${others} under the pre-planned statistical test. ${g.leader_assumption === 'free'
      ? 'Assumes only that separate deals are independent and each model stayed unchanged during the run.'
      : g.leader_assumption === 'supported' ? 'Diagnostic checks passed.' : 'Diagnostic checks were not run.'}`;
  } else if (descriptive) {
    badge = 'Leading so far';
    tone = 'points';
    why = n < 2 ? 'Only one player in this group has results so far.'
      : rt.pooled ? 'Not a final verdict: combined across multiple runs, this shows who is ahead so far.'
      : voidReason(rt.design) ? 'Not a final verdict: the arena code changed mid-run, so this shows who is ahead so far.'
      : 'Not a final verdict: this practice run had no pre-locked statistical test plan, so it shows who won these hands.';
  } else {
    const over = settled || rt.finished;
    badge = `Too close to call${over ? '' : ' yet'}`;
    tone = 'open';
    if (n < 2) why = 'Only one player in this group has results so far.';
    else if (g.pairings.every(x => x.verdict === 'early' || !x.deals)) why = over ? 'Too few deals were played to pick a winner.' : 'Too few deals so far to pick a winner; play continues.';
    else why = `Not decided${over ? '' : ' yet'}: ${decided} of ${plural(g.pairings.length, 'matchup')} ${decided === 1 ? 'has' : 'have'} a confirmed winner${over ? '' : ', and play continues'}.`;
  }
  if (!descriptive && rt.track !== 'adaptation' && g.pairings.length) why += ` ${decided} of ${plural(g.pairings.length, 'matchup')} decided.`;
  // only what needs attention gets a chip of its own
  const warn = [];
  const drifts = g.pairings.filter(x => Core.driftQualified(x)).length;
  if (drifts && !g.drift_leader) warn.push(chip(`${drifts} shifted mid-run`, 'warn', 'Matchups where the score changed sharply between early and late hands: not counted as a confirmed win.'));
  if (p) warn.push(identityChip(rt, isBot(who) ? (g.players || []).map(pid) : [who], !!(p.identity_qualified || g.identity_qualified)));
  if (ineligible) warn.push(chip(`${ineligible} disqualified`, 'bad', 'Matchups excluded from official verdicts (wrong model, audit issue, or code change): see Matchups.'));
  const wrong = first && rt.audit && rt.audit.totals && rt.audit.totals.identity && rt.audit.totals.identity.mismatch;
  if (wrong) warn.push(chip(`wrong model on ${plural(wrong, 'call')}`, 'bad', 'A provider returned a different model than requested: see Audit & call logs.'));
  if (misfit) warn.push(chip('check Matchups directly', 'quiet', 'Some head-to-head results differ from what the overall leaderboard predicts: read the Matchups list directly.'));
  const lead = p ? `${monogram(who, 'lg')}<span class="ls-text"><span class="ls-name">${name(who)}</span>
      <span class="ls-what">${g.leader ? 'leads' : tone === 'drift' ? 'was ahead' : 'leads'} at <b class="${cls(p.strength)}" title="${esc(STRENGTH_TIP)}">${fmtStrength(p.strength)} bb/100</b>${p.net_bb != null ? `<span class="ls-net"> · <b class="${cls(p.net_bb)}">${moneySigned(p.net_bb)}</b> net</span>` : ''}</span></span>`
    : `<span class="ls-text"><span class="ls-what">${n < 2 ? 'Waiting for a second player\'s results' : 'Too close to call so far'}</span></span>`;
  return `<div class="leader-strip ${tone}">
    ${multi ? `<span class="ls-group">${esc(groupName(g))}</span>` : ''}
    <div class="ls-lead">${lead}</div>
    <div class="ls-badges">${warn.filter(Boolean).join('')}<span class="ls-badge ${tone}" tabindex="0" title="${esc(why)}">${tone === 'decided' ? '<span class="decided-badge">Decided</span>winner' : esc(badge)}</span></div></div>`;
}

function outcomes(rt) {
  const gs = rt.groups.filter(g => g.ranked.length);
  if (!gs.length) return '';
  const multi = rt.groups.filter(g => g.members.length).length > 1;
  return `<div class="outcomes">${gs.map((g, i) => leaderStrip(rt, g, multi, i === 0)).join('')}</div>`;
}

// The target edge: what a registered run was planned to detect, what that takes, and what it has; for an
// unregistered run the same sum, labelled as a rough estimate.
function targetLine(rt, pairings) {
  const t = Core.targetEdge(rt.design, pairings);
  if (!t) return '';
  const d = rt.design || {};
  const have = t.have[0] === t.have[1] ? plural(t.have[0], 'deal') : `${t.have[0]}–${plural(t.have[1], 'deal')}`;
  const pct = Math.min(100, Math.round(100 * t.have[0] / Math.max(1, t.need)));
  const they = pairings.length > 1 ? 'each matchup has' : 'this matchup has';
  let words;
  if (t.retrospective) words = `<b>Rough estimate</b> (not pre-planned): reliably proving a ${t.target} bb/100 skill gap takes ~${approx(t.need)} paired deals per matchup; ${they} ${have}.`;
  else if (t.planned) words = `Planned to detect a ${t.target} bb/100 skill gap: ~${approx(t.need)} paired deals per matchup${t.allowed && t.allowed < t.need ? `, capped at ${approx(t.allowed)}${t.power != null ? ` (${Math.round(t.power * 100)}% detection chance)` : ''}` : ''}; ${they} ${have}.`;
  else words = `Proving a ${t.target} bb/100 skill gap takes ~${approx(t.need)} paired deals per matchup at the variance seen so far; ${they} ${have}.`;
  const extra = [];
  if (t.planned && d.underpowered) extra.push(`<span class="flag" title="The run's deal limit is smaller than needed for 80% detection chance">capped budget</span> limited to ${d.budget_deals ? `${approx(d.budget_deals)} deals` : 'a smaller budget'}`);
  const sc = t.planned ? Core.scenarios(d) : [];
  if (sc.length) extra.push(`Chance to catch that gap at this budget: ${sc.map(x => `${esc(x.name)} ${x.power != null ? Math.round(x.power * 100) + '%' : '–'}`).join(' · ')}`);
  return `<div class="target"><div class="meter thin"><span style="width:${pct}%"></span></div><span>${words}${extra.length ? `<span class="target-more">${extra.join('<br>')}</span>` : ''}</span></div>`;
}

// Everything about how results are judged, once, folded away: the test, its assumptions, the sample size,
// the words used on the page, and the ratings' notes.
function planPanel(rt, extras = []) {
  const d = rt.design || {};
  const open = store.get('plan-open', false);
  const v = voidReason(d);
  let test;
  if (rt.pooled) test = 'This view combines past runs that each used their own stop rules, so nothing here is a formal verdict: scores simply show who is <b>leading so far</b>.';
  else if (rt.track === 'adaptation') test = `<b>Memory mode (exploratory).</b> Models stay in one continuous session for ${plural(d.round_deals || 0, 'deal')}, so they remember earlier hands and can adapt to their opponent.
    Matchups are checked after every ${plural(d.replicates || 1, 'session')} (up to ${plural(d.max_rounds || 1, 'time')}) using an exploratory statistical check (${Math.round((d.alpha || 0.05) * 100)}% target error rate).
    Because hands inside a long session influence each other, these results are <b>exploratory only</b> and never change official standings.`;
  else if (v) test = `The arena code changed mid-run (${esc(v)}), so this test plan was canceled: scores show who is <b>leading so far</b>, not a confirmed winner.`;
  else if (rt.unregistered) test = 'This practice run was started without a pre-locked statistical test plan, so nothing here is a final verdict: scores show who won these hands, not who is definitely better.';
  else if (d.round_deals && officialFree(rt)) {
    const t = officialFree(rt);
    test = `Each matchup is checked after every ${d.round_deals} deals, up to ${plural(d.max_rounds, 'time')}${d.min_conversations ? `, after at least ${plural(d.min_conversations, 'independent deal')}` : ''}.
    The <b>official verdict</b> uses a strict sequential test${t.name ? ` (${esc(t.name)})` : ''}: the chance of calling any false winner across all
    ${plural(d.n_pairings || rt.pairings.length, 'matchup')} stays under ${Math.round((d.alpha || 0.05) * 100)}%, even when stopping early, as long as separate deals are independent and each model stays unchanged${t.bound != null ? ` (capped at ±${approx(t.bound)} bb/100 per deal)` : ''}.
    A drift check also watches for score swings between early and late hands (when it reports <b>no drift detected</b>, that still does not guarantee a provider kept its model snapshot frozen).`;
  }
  else if (d.round_deals) test = `Each matchup is checked after every ${d.round_deals} deals, up to ${plural(d.max_rounds, 'time')}${d.min_conversations ? `, after at least ${plural(d.min_conversations, 'independent deal')}` : ''}.
    A matchup is marked <b>Decided</b> once the lead passes the statistical bar (at most ${Math.round((d.alpha || 0.05) * 100)}% chance of any false winner across all ${plural(d.n_pairings || rt.pairings.length, 'matchup')}).
    Any result that fails diagnostic checks is flagged and does not count as a confirmed winner.`;
  else test = 'Confirmed winners only come from a pre-planned statistical test; everything else simply shows who is leading so far.';
  const target = rt.pooled ? '' : targetLine(rt, rt.pairings);
  const summary = rt.pooled ? 'Combined across runs: for browsing only, not a formal test'
    : rt.track === 'adaptation' ? 'Memory mode: models remember past hands (exploratory, not official)'
    : v ? 'Code changed mid-run: scores only, no formal verdict'
    : rt.unregistered ? 'Practice run: scores only, without a pre-locked statistical test'
    : d.round_deals && officialFree(rt) ? `Official sequential test · ${Math.round((d.alpha || 0.05) * 100)}% error bound if conversations are independent and each model stays the same`
    : d.round_deals ? `Planned sequential test · under ${Math.round((d.alpha || 0.05) * 100)}% chance of a false winner` : 'How results are judged';
  const notes = (rt.notes || []).map(n => `<li>${esc(n)}</li>`).join('');
  return `<section class="judged card-box" id="judged" aria-label="How this run is judged" ${open ? '' : 'hidden'}>
    <div class="jd-head"><div><h2>How this run is judged</h2><p class="jd-sum">${summary}</p></div>
      <button class="jd-close" type="button" aria-label="Close how this run is judged">×</button></div>
    <div class="jd-grid"><div class="jd-main">
      ${aboutText(rt)}
      ${extras.map(x => x ? `<p>${x}</p>` : '').join('')}
      <p>${test}</p>
      ${target}
      ${populationLine(rt) ? `<p class="pop-line">${populationLine(rt)}</p>` : ''}
      ${notes ? `<details class="pp-notes"><summary><b>Technical notes from the rating engine</b></summary><ul>${notes}</ul></details>` : ''}</div>
      <dl class="jd-gloss">
        <dt>Decided</dt><dd>A winner confirmed by a pre-planned statistical test, not just short-term luck. Valid as long as each model stayed unchanged throughout the run.</dd>
        <dt>Leading so far</dt><dd>Who won more money in the hands played so far, before a formal test settles it.</dd>
        <dt>Win rate (bb/100)</dt><dd>Big blinds won per 100 hands, balanced so the group averages 0. Compare any two models by the gap between their numbers. If a head-to-head matchup disagrees, trust the direct matchup.</dd>
        <dt>Direct avg</dt><dd>Simple average of a model's head-to-head margins, weighing every opponent equally.</dd>
        <dt>Error bars</dt><dd>The shaded bar shows rough uncertainty. In official tests it stays valid whenever play stops; in practice runs it simply shows how spread out the hands were.</dd>
        <dt>Luck-adjusted (AIVAT)</dt><dd>Subtracts card-deal and all-in runout luck so you can see who played better in fewer hands. Shown beside the raw score; official verdicts use raw paired deals.</dd>
        <dt>Model ID</dt><dd><b>Confirmed</b>: the model's CLI logged back the exact model that replied. <b>Not logged</b>: only the configured name is known. <b>Wrong model</b>: a different model replied.</dd>
        <dt>Session rating</dt><dd>Secondary Elo-style score from how many mini-sessions a model won (${esc(sessionText(rt))}). For reference only, never used to rank.</dd>
      </dl></div></section>`;
}

// The design in one sentence: how many sessions of how many deals each matchup plays.
function designLine(rt) {
  const deals = Math.max(0, ...state.index.map(m => m.deals_total || 0));
  const pairings = new Set(state.index.map(m => [m.a, m.b].sort().join('|'))).size;
  const conv = d => d.replicates > 1 ? `${d.replicates} sessions of ${plural(d.round_deals, 'deal')}` : plural(d.round_deals, 'deal');
  const rounds = rt.design?.round_deals ? (rt.design.max_rounds > 1 ? `up to ${rt.design.max_rounds} rounds of ${conv(rt.design)}` : `${conv(rt.design)} per matchup`)
    : state.index.some(m => m.round > 1) ? `rounds of ${deals} deals` : `${deals} deals each`;
  return pairings ? `<b>Format:</b> ${plural(pairings, 'matchup')}, ${rounds}.` : '';
}

// What a session is in this run, as the ratings define it (older ratings: their session length).
function sessionText(rt) {
  if (rt.session?.text) return rt.session.text;
  const n = rt.session_deals || 10;
  return n === 1 ? '1 paired deal per session' : `${n} deals per mini-session`;
}

// The official test when it is the assumption-free one (from the design, or every decided pairing), else null.
function officialFree(rt) {
  const t = Core.testOf(null, rt.design);
  if (t.free) return t;
  const ps = rt.pairings.filter(p => p.verdict !== 'descriptive');
  const ts = ps.map(p => Core.testOf(p, rt.design));
  return ts.length && ts.every(x => x.free) ? ts[0] : null;
}

// Which hands a figure covers, in words: verified committed results, or provisional live play.
function populationLine(rt, what = 'standings') {
  const pop = Core.populationOf(rt);
  const rej = pop.rejected ? `; ${plural(pop.rejected, 'hand')} that failed replay checks ${pop.rejected === 1 ? 'is' : 'are'} excluded` : '';
  if (what === 'stats') {
    const st = pop.stats;
    if (st && /commit|verif/i.test(st.kind || st.population || st.label || '')) return `Counts ${st.hands != null ? `${st.hands.toLocaleString()} ` : ''}finished, checked hands${rej}.`;
    if (st) return `Counts: ${esc(st.label || st.kind || st.population)}.`;
    return `Counts all finished hands from this run${pop.rejected ? ' (including any flagged by replay checks)' : ''}.`;
  }
  const hands = pop.committed ?? (rt.deals ? 2 * rt.deals : null);
  return hands ? `Counts ${hands.toLocaleString()} checked hands (every deal played twice with swapped seats and verified from the log)${rej}${pop.provisional ? `; ${plural(pop.provisional, 'in-progress hand')} not counted yet` : ''}.` : '';
}

// The run's audit in one line: the aside of its folded section.
const idCount = id => (id.match || 0) + (id.unattested || 0) + (id.mismatch || 0);
function auditLine(rt) {
  const a = rt.audit;
  if (!a || !a.totals || !a.totals.calls) return '';
  const t = a.totals, id = t.identity || {};
  const idn = idCount(id);
  const bits = [`${t.decisions.toLocaleString()} moves`];
  bits.push(a.recorded ? `${t.tool_rejected} blocked tools · ${t.unverified} unchecked` : 'sandbox checks not recorded');
  bits.push(`${plural(t.fallbacks, 'auto-fold')}`);
  if (idn) bits.push(`model ID confirmed on ${Math.round(100 * (id.match || 0) / idn)}% of calls${id.mismatch ? ` · <b class="neg">${id.mismatch} wrong model</b>` : ''}`);
  return bits.join(' · ');
}

const fmtN = n => n ? n.toLocaleString() : '<span class="dim">0</span>';
const ID_TONE = { match: 'ok', mismatch: 'bad' };
const AUDIT_ID = { match: 'Confirmed', mismatch: 'Wrong model', unattested: 'Not logged', partial: 'Partly logged' };
function auditSection(rt) {
  const a = rt.audit;
  if (!a || !a.players || !Object.keys(a.players).length) return '';
  const list = Object.entries(a.players).filter(([, x]) => x.calls || x.decisions);
  if (!list.length) return '';
  const cells = x => {
    const id = x.identity || {};
    const idn = idCount(id);
    const status = Core.identityOf(null, x);
    const saw = x.observed && x.observed.length ? ` · replied ${esc(x.observed.join(', '))}` : '';
    const asked = status === 'mismatch' && x.requested && x.requested.length ? `, expected ${esc(x.requested.join(', '))}` : '';
    const audited = x.verified + x.opaque + x.unverified;
    const seen = audited || x.tool_rejected;
    return {
      id: idn ? `<span class="idst ${ID_TONE[status] || 'warn'}">${esc(AUDIT_ID[status] || '–')}</span>
        <small class="nof">${id.match || 0}/${idn} calls confirmed${id.mismatch ? ` · ${id.mismatch} wrong` : ''}${saw}${asked}</small>` : '<span class="dim">–</span>',
      status,
      audit: !seen ? '<span class="dim" title="Recorded before sandbox checks were enabled">not recorded</span>'
        : `${x.verified}<small class="nof">passed${x.opaque ? ` · ${x.opaque} hidden by provider` : ''}${x.audit_missing ? ` · ${x.audit_missing} unrecorded` : ''}</small>`,
      tool: seen ? fmtN(x.tool_rejected) : '<span class="dim">–</span>',
      unv: seen ? fmtN(x.unverified) : '<span class="dim">–</span>',
    };
  };
  const rows = list.map(([p, x]) => {
    const c = cells(x);
    return `<tr><td>${player(p, '', { details: false })}</td><td class="num">${fmtN(x.decisions)}<small class="nof">${fmtN(x.calls)} calls</small></td>
      <td>${c.id}</td><td class="num">${c.audit}</td><td class="num">${c.tool}</td><td class="num">${c.unv}</td>
      <td class="num">${fmtN(x.retries)}<small class="nof">${x.resized} resized</small></td><td class="num">${fmtN(x.fallbacks)}</td>
      <td class="num">${fmtN(x.failed_tries)}<small class="nof">${x.rate_limited} rate limits</small></td></tr>`;
  }).join('');
  const head = `<thead><tr><th>Model</th><th class="num">Moves</th><th title="Did the provider log back the exact model ID that answered?">Model ID</th>
    <th class="num" title="Turns verified clean of tool use (or hidden when the provider does not expose tool logs)">Sandbox check</th>
    <th class="num" title="Replies where the model tried to call a tool: blocked and asked again">Blocked tools</th><th class="num" title="Turns whose tool log could not be read: rejected and asked again">Unchecked</th>
    <th class="num" title="Turns retried after an invalid reply; resized means a bet size was adjusted to a legal amount">Retries</th>
    <th class="num" title="Model failed to give a legal move after retrying, so the game folded or checked for it">Auto-folds</th><th class="num" title="API or CLI calls that errored and were retried">API errors</th></tr></thead>`;
  const cards = list.map(([p, x]) => {
    const c = cells(x);
    return `<details class="acard"><summary>${monogram(p, 'sm')}<span class="ac-name">${name(p)}</span>
        <span class="ac-sum"><span class="idst ${ID_TONE[c.status] || 'warn'}">${esc(AUDIT_ID[c.status] || '–')}</span> · ${fmtN(x.decisions)} moves${x.fallbacks ? ` · ${plural(x.fallbacks, 'auto-fold')}` : ''}</span></summary>
      <dl><dt>Model ID</dt><dd>${c.id}</dd><dt>API calls</dt><dd>${fmtN(x.calls)}</dd><dt>Sandbox check</dt><dd>${c.audit}</dd>
        <dt>Blocked tools</dt><dd>${c.tool}</dd><dt>Unchecked</dt><dd>${c.unv}</dd><dt>Retries</dt><dd>${fmtN(x.retries)} <small class="nof">${x.resized} resized</small></dd>
        <dt>Auto-folds</dt><dd>${fmtN(x.fallbacks)}</dd><dt>API errors</dt><dd>${fmtN(x.failed_tries)} <small class="nof">${x.rate_limited} rate limits</small></dd></dl></details>`;
  }).join('');
  return section('Audit &amp; call logs', auditLine(rt) || 'technical record of every model call',
    `<div class="only-wide"><div class="scroll audit-scroll"><table class="audit-t">${head}<tbody>${rows}</tbody></table></div></div><div class="only-narrow acards">${cards}</div>`,
    'From each match\'s call log (bots make no API calls). You can also inspect any single move\'s record under <b>why?</b> in the Watch tab.',
    { id: 'audit', fold: true });
}

// ---- tracks: the official standings, and memory mode (the same players in long sessions) ----
function trackBar(run, rest = '') {
  const set = trackSet(run);
  if (set.length < 2 && !(set[0] && set[0].track === 'adaptation')) return '';
  const tab = r => `<a role="tab" class="track ${r.run === run ? 'on' : ''} t-${r.track}" aria-selected="${r.run === run}" href="${runLink(r.run, rest)}">
    <span class="tr-t">${esc(TRACK_TITLE[r.track] || r.track)}</span><small>${r.track === 'adaptation' ? 'exploratory · not official' : 'fresh start every deal · decides ranks'}</small></a>`;
  return `<div class="track-bar" role="tablist" aria-label="Mode">${set.map(tab).join('')}</div>`;
}
// What the track means, as plain text for the methodology drawer (the track tabs stay on the page).
function trackNote(rt) {
  const set = trackSet(state.run);
  if (rt.track === 'adaptation') {
    const d = rt.design || {};
    return `<b>Memory mode (exploratory, not official):</b> Models stay in one continuous session for ${d.round_deals ? plural(d.round_deals, 'deal') : 'many deals'}, so they remember earlier hands and can adapt to their opponent.
      Results are exploratory leads, never official rankings. ${set.some(r => r.track === 'official') ? `See <a href="${runLink(siblingTrack(state.run, 'official').run)}"><u>Official standings</u></a> for fresh-start deals with formal verdicts.` : ''}`;
  }
  if (set.length < 2) return '';
  return `<b>Two modes, same models and cards.</b> <b>Official</b> starts fresh on every deal and uses the strict statistical test to decide ranks.
    <b>Memory mode</b> keeps long sessions so models can remember and adapt (exploratory only).`;
}
// Without the reference bot, the strengths compare the models only with each other: said once, in the drawer.
function yardstickNote(view) {
  const vs = view.players.filter(p => p.vs_reference && !isReference(p.player));
  if (vs.length || !view.players.some(p => !isBot(p.player)) || view.cohort?.kind === 'history') return '';
  return `<b>No baseline bot in this group.</b> These scores compare these models only against each other. To also benchmark against a fixed bot, include <code>bot = "reference"</code> in your players file.`;
}
// On the official page: the memory mode's own standings, compact and clearly not official.
function adaptationTeaser(rt) {
  if (rt.track === 'adaptation') return '';
  const sib = siblingTrack(state.run, 'adaptation');
  if (!sib) return '';
  const raw = ratingsStore.get(sib.run);
  if (!raw) {
    const here = state.run;
    const later = (wait) => setTimeout(() => fetchRatings(sib.run).then(x => {
      if (state.run !== here) return;
      if (!x) later(PENDING_POLL);
      else if (!state.replay?.playing && !location.hash.includes('/m')) route({ soft: true });
    }).catch(() => {}), wait);
    later(0);
    return section(TRACK_TITLE.adaptation, 'exploratory · not official', '<p class="soft">Working out memory-mode standings…</p>', '', { id: 'adaptation' });
  }
  const at = normalize(raw);
  Object.assign(state.labels, raw.labels || {});
  const ps = at.players.filter(p => p.in_fit).sort((x, y) => (y.strength ?? -1e9) - (x.strength ?? -1e9));
  const mbl = (at.groups || []).map(g => g.model_based_leader).filter(Boolean);
  const rows = ps.map((p, i) => `<div class="ad-row"><span class="pos-n">${i + 1}</span>${player(p.player, '', { details: false })}
    <b class="${cls(p.strength)}" title="${esc(STRENGTH_TIP)}">${fmtStrength(p.strength)}<small> bb/100</small></b><small class="soft">${p.strength_ci95 ? ciText(p.strength_ci95) : ''}</small></div>`).join('');
  const called = at.pairings.filter(x => ['a', 'b', 'even'].includes(x.verdict)).map(x => {
    const s = verdictSides(x);
    return `<li>${s ? `${plain(s.winner)} ahead of ${plain(s.loser)}` : `${plain(x.a)} and ${plain(x.b)} close`} <span class="soft">(${signed(Math.abs(x.edge ?? 0), 0)} bb/100)</span></li>`;
  }).join('');
  const body = ps.length ? `<div class="ad-list">${rows}</div>${called ? `<div class="ad-called"><b>Exploratory leads</b> <span class="soft small">(not official verdicts)</span><ul>${called}</ul></div>` : ''}
    ${mbl.length ? `<p class="small soft">${mbl.map(plain).join(', ')} ahead in every matchup in memory mode (exploratory only).</p>` : ''}`
    : '<p class="soft">No finished deals in memory mode yet.</p>';
  return section(TRACK_TITLE.adaptation, `${chip('exploratory · not official', 'warn')}`,
    body + `<a class="ad-open" href="${runLink(sib.run)}">Open memory mode ›</a>`,
    'Models play long sessions and remember past hands so they can adapt. Exploratory only; never changes the official standings above.', { id: 'adaptation' });
}

async function renderBoard(query) {
  const r = state.runs.find(x => x.run === state.run);
  let raw = ratingsStore.get(state.run) || null;
  if (!raw) {
    try { raw = await fetchRatings(state.run); } catch (e) { raw = null; if (!ratingsStore.has(state.run)) throw e; }
  } else ratingsLater(state.run);
  if (!raw) { renderBoardPending(r); ratingsLater(state.run, PENDING_POLL); return; }
  if (!ratingsUpToDate(raw)) ratingsLater(state.run, STALE_POLL);
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
  // nor on a drifted verdict (unless the ratings already weigh which verdicts count toward a rank)
  if (!rt.pairings.some(x => 'counts_toward_leader' in x)) {
    const drifted = new Set(rt.pairings.filter(Core.driftQualified).flatMap(x => [x.a, x.b]));
    rt.players = rt.players.map(p => drifted.has(p.player) && p.rank_kind === 'decided' ? { ...p, rank: null, rank_kind: 'descriptive', rank_note: 'a verdict drifted' } : p);
  }
  state.rt = rt;
  for (const p of rt.players) if (p.kind) state.kinds[p.player] = p.kind;
  const all = Core.cohorts(rt, { label: labelText, kind: p => state.kinds[p] });
  const cohort = all.length > 1 ? all.find(c => c.key === query?.get('c')) || all[0] : null;
  const view = cohort ? cohortView(rt, cohort) : rt;
  rt.track = raw.track || r?.track || 'official';
  state.v2 = { aivat: raw.aivat || null };
  const aiv = raw.aivat && raw.aivat.summary && raw.aivat.summary.pairings
    ? `<b>Luck reduction (AIVAT):</b> cuts score noise by ${trim(Math.round(raw.aivat.summary.mean_variance_reduction_ratio * 10) / 10)}× on average across ${plural(raw.aivat.summary.pairings, 'matchup')}.` : '';
  const extras = [r?.all ? historyNotes(r, rt) : designLine(rt), trackNote(rt), yardstickNote(view), aiv];
  let html = runHeader(r, rt, raw) + planPanel(rt, extras) + trackBar(state.run) + staleNote(raw) + errorsPanel(r) + (cohort ? cohortBar(all, cohort) : '');
  const waiting = cohort && cohort !== all[0] ? [] : rt.roster.filter(x => !rt.players.some(p => p.player === x.player));
  if (!rt.players.length) {
    html += `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>The cards are in the air</h2>
      <p>A deal counts once both seat-swapped plays finish. ${state.index.some(m => m.started) ? `Watch the first hands on the <a href="${runLink(state.run, '/m')}"><u>Watch</u></a> page.` : ''}</p>
      ${waiting.length ? `<div class="waiting-list">${waiting.map(x => player(x.player, `<div class="small soft">${x.status === 'stopped' ? '<span class="neg">every deal so far stopped on an error</span>' : x.status === 'playing' ? 'playing its first deals' : 'waiting to play'}</div>`)).join('')}</div>` : ''}</div>`;
    $app.innerHTML = html;
    bindBoard();
    return;
  }
  html += outcomes(view);
  html += `<div class="results-stack"><div class="rs-standings">${standings(view, waiting)}</div>`;
  html += `<div class="rs-pairings">${pairingsSection(view)}</div></div>`;
  html += referenceSection(view, rt);
  html += adaptationTeaser(rt);
  html += metaCard(raw);
  if (r?.quiz) html += '<div class="quiz-slot"></div>';
  html += `<div class="deep-head"><span>More detail</span><small>head-to-head grid, playing style, and call logs</small></div>`;
  html += headToHead(view);
  html += styleTable(view);
  html += auditSection(rt);
  $app.innerHTML = html;
  bindBoard();
  if (r?.quiz) fillQuizTeaser(state.run);
}
function bindBoard() {
  $app.querySelectorAll('[data-order]').forEach(b => b.onclick = () => {
    state.order = b.dataset.order;
    store.set('order', state.order);
    route({ soft: true });
  });
  const jt = $app.querySelector('#judged-toggle'), jd = $app.querySelector('#judged');
  const setJudged = open => {
    if (!jd) return;
    jd.hidden = !open;
    jt?.setAttribute('aria-expanded', String(open));
    store.set('plan-open', open);
  };
  if (jt) jt.onclick = () => setJudged(jd.hidden);
  $app.querySelector('.jd-close')?.addEventListener('click', () => { setJudged(false); jt?.focus(); });
  $app.querySelectorAll('[data-morecols]').forEach(b => b.onclick = () => {
    const on = !store.get('more-cols', false);
    store.set('more-cols', on);
    $app.querySelectorAll('.board-list').forEach(l => l.classList.toggle('more', on));
    $app.querySelectorAll('[data-morecols]').forEach(x => x.setAttribute('aria-pressed', String(on)));
  });
  $app.querySelectorAll('details.fold[data-fold]').forEach(d => d.addEventListener('toggle', () => store.set('fold.' + d.dataset.fold, d.open)));
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

// Before a run's ratings exist (the first visit after it is played), the page still shows the run, the players
// and the raw score of each matchup at once, clearly unchecked, and fills in when the ratings arrive.
function renderBoardPending(r) {
  const none = { roster: [], design: null, players: [], runs: [] };
  const pairs = pairingsOf(state.index).filter(P => P.deals).sort((x, y) => Math.abs(y.netA / y.deals) - Math.abs(x.netA / x.deals));
  const rows = pairs.map(P => {
    const up = P.netA >= 0, lead = up ? P.a : P.b, trail = up ? P.b : P.a;
    const rate = Math.abs(P.netA) / (2 * P.deals) * 100;
    return `<a class="pend-row" href="${matchLink(P.latest.id)}">${monogram(lead, 'sm')}<span class="dr-n">${short3(lead)}</span><i>${P.netA ? 'ahead of' : 'level with'}</i>
      <span class="dr-n">${short3(trail)}</span><b>${trim(Math.round(rate))}<small> bb/100</small></b><small class="soft">${plural(P.deals, 'deal')}</small></a>`;
  }).join('');
  $app.innerHTML = runHeader(r, none, null) + errorsPanel(r) + `<div class="card-box pending-board" aria-live="polite">
    <div class="deal-in-suits" aria-hidden="true"><span>♠</span><span class="red">♥</span><span class="red">♦</span><span>♣</span></div>
    <h2>Checking every hand and working out the standings</h2>
    <p>Each hand is replayed from its seed and checked before it counts. For a big run that takes a few seconds the first time; this page fills in by itself. Meanwhile, the raw chip score of each matchup (unchecked, and not a verdict):</p>
    ${rows ? `<div class="pend-list">${rows}</div>` : ''}
    <a class="cta" href="${runLink(state.run, '/m')}"><span class="cta-icon">▶</span>Watch the hands meanwhile</a></div>`;
}

// The run's deals that a model's CLI gave up on (the server's `failures`): who, how many, what the CLI said, and
// a deal to open. Shown wherever the standings are, so a model that is missing from them is explained.
function errorsPanel(r) {
  const f = r && r.failures;
  if (!f || !f.deals) return '';
  const rows = f.players.flatMap(p => p.kinds.map(k => `<div class="fail-row">
      <p>${name(p.player)} <span class="fail-count">${k.deals} of ${plural(p.started, 'deal')} stopped</span> · ${esc(k.title)}${k.last ? ` <span class="soft">· last ${ago(k.last)}</span>` : ''}
        <a class="fail-open" href="${matchLink(k.example)}">Open one ›</a></p>
      <blockquote class="fail-msg">${esc(k.message)}</blockquote></div>`)).join('');
  return `<div class="fail-note card-box run-errors" role="status"><span class="fail-icon" aria-hidden="true">!</span><div class="fail-body">
    <p><b>${plural(f.deals, 'deal')} stopped on a model error and ${f.deals === 1 ? 'is' : 'are'} not counted.</b> Standings use finished, checked deals only, so a model whose deals all stopped has no rank yet.</p>
    ${rows}<p class="small soft">Running the same plan again replays these deals.</p></div></div>`;
}

const staleNote = raw => raw && raw.cache && raw.cache.error
  ? `<div class="stale-note bad" role="status">These standings are from ${raw.cache.computed ? ago(raw.cache.computed) : 'an earlier check'}: working them out again failed (${esc(raw.cache.error)}), so they may miss recent hands.</div>`
  : raw && raw.cache && !raw.cache.fresh
  ? `<div class="stale-note" aria-live="polite"><span class="spin" aria-hidden="true">♣</span>${raw.cache.reason === 'code' ? 'The arena\'s code changed since these standings were worked out' : 'New hands have finished since these standings were worked out'}${raw.cache.computed ? ` (${ago(raw.cache.computed)})` : ''}: updating in the background.</div>` : '';

// A cohort: the players of one group (who all met each other), or the earlier variants left out of the fit.
function cohortView(rt, c) {
  const set = new Set(c.group ? c.group.players : c.members);
  const archive = { deals: rt.deals, hands: rt.hands, sessions: rt.players.reduce((t, p) => t + (p.sessions || 0), 0) / 2 };
  const of = (players, pairings) => {
    const deals = pairings.reduce((t, x) => t + (x.deals || 0), 0);
    return { deals, hands: 2 * deals, sessions: players.reduce((t, p) => t + (p.sessions || 0), 0) / 2, archive };
  };
  if (c.kind === 'history') {
    const players = rt.players.filter(p => set.has(p.player)), pairings = rt.pairings.filter(p => set.has(p.a) || set.has(p.b));
    return { ...rt, ...of(players, pairings), groups: [], players, pairings, cohort: c };
  }
  const players = rt.players.filter(p => set.has(p.player) && p.in_fit);
  return { ...rt, ...of(players, c.group.pairings), groups: [c.group], players, pairings: c.group.pairings, cohort: c };
}
function cohortBar(all, cur) {
  const link = c => runLink(state.run, c === all[0] ? '' : `?c=${c.key}`);
  const cleanSub = s => String(s || '').replace(/conversations/g, 'sessions').replace(/model or protocol not recorded, so not rated/i, 'older test runs · not rated').replace(/protocol not recorded/i, 'older settings');
  const chip = c => {
    const n = Core.cohortName(c, labelText);
    const title = c.kind === 'history' ? 'Older unrated entries' : n.title;
    const short = c.kind === 'history' ? 'Older unrated entries' : n.short;
    const sub = cleanSub(n.sub);
    const count = c.kind === 'history' ? plural(c.members.length, 'player') : `${plural(c.members.length, 'player')} · ${plural(c.deals, 'deal')}`;
    return `<a role="tab" class="cohort ${c === cur ? 'on' : ''} k-${c.kind}" aria-selected="${c === cur}" href="${link(c)}" title="${esc(title)}">
      <span class="cb-mono">${c.members.slice(0, 4).map(p => monogram(p, 'xs')).join('')}</span>
      <span class="cb-t">${esc(c === all[0] && c.kind !== 'bots' ? title : short)}</span><small>${esc([sub, count].filter(Boolean).join(' · '))}</small></a>`;
  };
  return `<div class="cohort-bar"><span class="cb-k">${all[0].kind === 'models' && cur === all[0] ? 'Current models' : 'Players'}<small>who played each other</small></span>
    <div class="cohorts" role="tablist" aria-label="Which players">${all.map(chip).join('')}</div></div>`;
}
const groupName = g => {
  const c = Core.cohorts({ groups: [g], players: [] }, { label: labelText, kind: p => state.kinds[p] })[0];
  return c ? Core.cohortName(c, labelText).short : '';
};

// The fixed yardstick: each player's result against the versioned reference bot, and what the reference is
// and is not. Its own record against the baselines comes from this run's matchups.
const REF_LIMITS = 'A simple rule-based bot: tight-aggressive, position-aware, using basic hand odds and a fixed bluff rate. It is not a solver and does not adapt, so beating it shows solid fundamentals against a fixed opponent.';
function referenceSection(view, rt) {
  const ref = rt.players.map(p => p.player).find(isReference) || Object.keys(state.kinds).find(isReference);
  const vs = view.players.filter(p => p.vs_reference && !isReference(p.player));
  if (!vs.length) return '';
  const info = rt.reference && typeof rt.reference === 'object' ? rt.reference : {};
  const pairOf = (x, y) => rt.pairings.find(q => (q.a === x && q.b === y) || (q.a === y && q.b === x));
  const ax = axisFor(vs.map(p => ({ strength: p.vs_reference.edge, strength_ci95: p.vs_reference.ci95 })));
  const rows = [...vs].sort((x, y) => (isBot(x.player) - isBot(y.player)) || (y.vs_reference.edge ?? 0) - (x.vs_reference.edge ?? 0)).map(p => {
    const v = p.vs_reference;
    const pr = pairOf(p.player, v.reference || ref);
    const off = pr ? officialRange(pr, pr.a === p.player) : null;
    const ci = off ? off.ci : v.ci95;
    const vd = pr ? verdict(pr, plain) : null;
    const full = pr ? matchupId(pr) : null;
    return `<div class="prow ref-row ${isBot(p.player) ? 'bot' : 'model'}">${player(p.player, '', { details: false })}
      <div class="edge"><b class="${cls(v.edge)}">${signed(v.edge, 0)}</b><small>bb/100</small>${off || v.ci95 ? `<small class="ci-line">${off ? off.html : descRange(v.ci95, v)}</small>` : ''}</div>
      <div class="edge-bar"><div class="whisker"><span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
        <span class="point ${v.edge > 0 ? 'pos' : ''}" style="left:${ax.x(v.edge)}"></span></div></div>
      <div class="vcell">${vd ? verdictCell(pr, vd, { compact: true }) : ''}<small>${plural(v.deals || pr?.deals || 0, 'deal')}</small></div>
      ${full ? `<a class="watch" href="${matchLink(full)}">Watch ›</a>` : '<span></span>'}</div>`;
  }).join('');
  const record = rt.pairings.filter(q => (q.a === ref || q.b === ref) && q.deals).map(q => {
    const other = q.a === ref ? q.b : q.a, e = q.a === ref ? q.edge : -q.edge;
    return `${short3(other)} ${signed(e, 0)}`;
  });
  const limits = [].concat(info.limits || []).map(esc);
  const results = Array.isArray(info.results) ? info.results.map(x => {
    const opp = x.opponent || x.player || (x.a === 'reference' || x.a === ref ? x.b : x.a);
    const e = x.edge ?? x.edge_bb100;
    return opp && e != null ? `${short3(opp)} ${signed(e, 0)}${x.se != null ? ` ±${Math.round(x.se)}` : ''}` : null;
  }).filter(Boolean) : record;
  const about = `<details class="ref-about"><summary>About the reference bot${results.length ? ' and its record' : ''}</summary>
    <p>${esc(info.description || REF_LIMITS)}${limits.length ? ` Limits: ${limits.join('; ')}.` : ''}${info.version ? ` Version ${esc(info.version)}.` : ''}</p>
    ${results.length ? `<p><b>Its own record${info.results ? '' : ' in this run'}:</b> ${results.join(' · ')} bb/100 (its edge over each).</p>` : ''}</details>`;
  return section('Against the reference bot', 'a fixed baseline bot: above 0 means beating it',
    `<div class="ref-list">${rows}</div>`, 'A simple rule-based bot, not a pro solver: beating it shows solid fundamentals against one fixed opponent.', { id: 'reference', after: about });
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
    const members = g.members.map(p => by[p]).filter(Boolean);
    const blocks = Core.standingBlocks(members);
    const ranked = blocks.some(b => b.kind === 'decided' || b.kind === 'cycle');
    const points = !ranked || state.order === 'points';
    const ax = axisFor(members);
    const range = b => !b.range ? '–' : b.range[0] === b.range[1] ? String(b.range[0]) : `${b.range[0]}–${b.range[1]}`;
    const rows = points
      ? Core.standingBlocks(members.map(p => ({ ...p, rank_kind: 'descriptive', rank_cycle: [] })))[0].players.map(p => standingRow(rt, g, by[p.player], ax, ranked)).join('')
      : blocks.map(b => b.kind === 'decided' ? b.players.map(p => standingRow(rt, g, p, ax, true)).join('')
        : b.kind === 'cycle' ? cycleBlock(rt, g, b, ax)
        : `<div class="unres" role="group" aria-label="Order not decided"><div class="unres-head"><span class="rank prov">${range(b)}</span>
            <span>${b.range ? `Places ${range(b)}: ` : ''}these ${b.players.length === 2 ? 'two are' : `${b.players.length} are`} <b>too close to separate</b> · listed by score</span></div>
            ${b.players.map(p => standingRow(rt, g, p, ax, true, true)).join('')}</div>`).join('');
    const extra = k === shown.length - 1 ? waitingRows(waiting, ranked) : '';
    const title = multi ? groupName(g) : ranked ? 'Standings' : 'Leaderboard';
    const more = store.get('more-cols', false);
    const control = ranked ? `<span class="order-bar" role="group" aria-label="Order">
        <button class="toggle sm" data-order="decided" aria-pressed="${!points}" title="Confirmed ranks first; players too close to separate are grouped">Confirmed</button>
        <button class="toggle sm" data-order="points" aria-pressed="${points}" title="Sort everyone by win rate so far">By score</button></span>` : '';
    const misfit = g.misfits ? `<span class="flag" title="Some head-to-head results differ from what the overall leaderboard predicts; check Matchups directly">${g.misfits} matchup surprise${g.misfits === 1 ? '' : 's'}</span>` : '';
    const count = rt.cohort && rt.archive && rt.archive.deals !== rt.deals
      ? `${plural(rt.deals, 'paired deal')} in this group <span class="soft">(${plural(rt.archive.deals || 0, 'deal')} total)</span>`
      : plural(rt.deals, 'paired deal');
    const tools = `<span class="sl-tools">${multi ? '' : `<span class="sl-count" title="Each deal is played twice with seats swapped so card luck cancels out">${count}</span>`}${misfit}${control}
      <button class="toggle sm" data-morecols aria-pressed="${more}" title="Show direct matchup average and session rating columns">More columns</button></span>`;
    out += section(title, tools,
      `<div class="board-list ${ranked ? '' : 'unranked'} ${more ? 'more' : ''}"><div class="board-head">${ranked ? '<span title="Confirmed rank"><span class="hide-xs">Rank</span><span class="show-xs">#</span></span>' : ''}<span>Player</span>
         <span class="r" title="${esc(STRENGTH_TIP)}">Win rate <small>bb/100</small></span>
         <span class="axis-head" aria-hidden="true"><span>${signed(-ax.lim, 0)}</span><span>0</span><span>${signed(ax.lim, 0)}</span></span>
         <span class="r hide-sm">Net</span>
         <span class="r col-more" title="${esc(OBSERVED_TIP)}">Direct avg</span><span class="r col-more" title="Secondary rating from mini-sessions won (${esc(sessionText(rt))}); not used for official ranks">Session rating</span>
         <span class="r go-h"><span class="sr">Scorecard</span></span></div>
       ${rows}${extra}</div>`,
      `${ranked ? 'A number in the first column is a statistically confirmed rank. ' : ''}Win rate: big blinds won per 100 hands (above 0 = winning); a big blind is $20, so +50 is $1,000 per 100 hands. Net is the money won or lost in total. The gap between two models estimates their head-to-head edge; wider bars mean less certainty.`);
  });
  if (!out && waiting.length) out = section('Standings', '', `<div class="board-list unranked">${waitingRows(waiting, false)}</div>`);
  const out_ = rt.players.filter(p => !p.in_fit);
  if (out_.length) {
    out += section(rt.cohort?.kind === 'history' ? 'Older unrated entries' : 'Not on leaderboard', 'played, but excluded from rankings', `<div class="board-list unranked left-out ${store.get('more-cols', false) ? 'more' : ''}">${out_.map(p => `<div class="brow quarantined">
      ${player(p.player, `<div class="small soft why-out">${esc(p.unrated_reason || p.quarantine_reason || 'did not play any ranked opponent')}</div>`)}
      <span class="r strength soft">${p.hands ? `<small>${plural(p.hands, 'hand')}</small>` : ''}</span><span class="axis"></span>
      <span class="r hide-sm soft">${p.net_bb != null ? moneySigned(p.net_bb) : ''}</span><span class="col-more"></span><span class="col-more"></span><span></span></div>`).join('')}</div>`,
      'Excluded when the exact model or reasoning level was not recorded, or when it only played other excluded entries.');
  }
  return out;
}

function cycleBlock(rt, g, b, ax) {
  const names = b.players.map(p => p.player);
  const results = Core.cycleResults(names, g.pairings || rt.pairings);
  return `<div class="unres cycle" role="group" aria-label="Rock-paper-scissors loop"><div class="unres-head"><span class="rank prov" aria-hidden="true">↻</span>
      <span><b>Rock-paper-scissors loop:</b> these ${b.players.length === 2 ? 'two beat each other in a cycle' : `${b.players.length} beat each other in a cycle`}, so there is no single order · listed by score</span></div>
    ${results.length ? `<div class="cycle-results small soft">${results.map(x => `${plain(x.winner)} beat ${plain(x.loser)}`).join(' · ')}</div>` : ''}
    ${b.players.map(p => standingRow(rt, g, p, ax, true, true)).join('')}</div>`;
}

const PLAIN_IDENTITY = { match: 'Model ID confirmed', unattested: 'Model ID not logged', partial: 'Model ID partly logged', mismatch: 'Wrong model answered' };
function identityFlag(rt, p) {
  if (isBot(p.player)) return '';
  const a = rt.audit && rt.audit.players && rt.audit.players[p.player];
  const st = Core.identityOf(p, a) || Core.identityOf(p, (rt.playerAudit || {})[p.player]);
  if (!st || st === 'match') return '';
  const tip = st === 'mismatch' ? 'Its provider returned a different model than requested.'
    : st === 'unattested' ? 'Its CLI did not log back a server-confirmed model ID, so only the configured name is known.'
    : 'Only some calls logged back a server-confirmed model ID.';
  return `<span class="flag ${st === 'mismatch' ? 'bad' : 'quiet'}" title="${esc(tip)}">${esc(PLAIN_IDENTITY[st] || Core.IDENTITY[st])}</span>`;
}

function standingRow(rt, g, p, ax, ranked, inBlock = false) {
  const ci = p.strength_ci95;
  const decided = p.rank_kind === 'decided' && p.rank != null;
  const tied = (p.rank_tied_with || []).filter(x => x !== p.player && g.members.includes(x));
  const notes = [];
  if (!p.identity_resolved) notes.push(`<span class="flag" title="${esc(p.quarantine_reason || '')}">model settings unrecorded</span>`);
  const idf = identityFlag(rt, p);
  if (idf) notes.push(idf);
  if (!inBlock && tied.length && tied.length < g.members.length - 1) notes.push(`<span class="small soft">still too close with ${tied.map(plain).join(', ')}</span>`);
  const obs = p.observed_edge != null ? signed(p.observed_edge, 0) : null;
  const mob = `<div class="mob-line small soft">${obs != null ? `<span class="${cls(p.observed_edge)}" title="${esc(OBSERVED_TIP)}">${obs} direct avg</span> · ` : ''}${p.net_bb != null ? `<span class="${cls(p.net_bb)}">${moneySigned(p.net_bb)}</span> · ` : ''}${plural(p.hands || 0, 'hand')}</div>`;
  const leader = g.leader === p.player;
  const b = Core.basisOf(p);
  const whisker = `<div class="whisker ${b ? b.key : ''}" title="${ci ? `Rough range: ${ciText(ci)} bb/100` : 'No range yet: too few deals'}">
      <span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
      ${p.strength != null ? `<span class="point ${leader ? 'lead' : ''}" style="left:${ax.x(p.strength)}"></span>` : ''}</div>`;
  const elo = p.session_elo != null ? `<span title="Session rating${p.session_elo_ci95 ? ` (range ${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]})` : ''}: secondary Elo-style score, never used to rank">${p.session_elo}</span>
      <small>${p.session_w ?? 0}–${p.session_t ?? 0}–${p.session_l ?? 0} sessions</small>` : '<small>–</small>';
  const half = ci ? Math.round((ci[1] - ci[0]) / 2) : null;
  const sc = state.runs.find(x => x.run === state.run)?.all ? '' : `<a class="sc-link" href="${runLink(state.run, `/scorecards?m=${encodeURIComponent(p.player)}`)}" title="Open ${plain(p.player)}'s 6-trait scorecard"><span class="hide-sm">Scorecard </span>›</a>`;
  return `<div class="brow ${leader ? 'lead' : ''}">
    ${ranked ? `<span class="rank ${decided ? 'sep' : 'prov'}" title="${decided ? 'Statistically confirmed rank' : 'Not confirmed: sorted by score so far'}">${decided ? p.rank : inBlock ? '' : '–'}</span>` : ''}
    ${player(p.player, `${notes.length ? `<div class="row-notes">${notes.join(' ')}</div>` : ''}${mob}`)}
    <span class="r strength"><b class="${cls(p.strength)}" title="${esc(STRENGTH_TIP)}">${fmtStrength(p.strength)}</b>${ci ? `<small title="${esc(Core.DESCRIPTIVE_TIP)}${b ? ' ' + esc(b.long) : ''}">±${half}</small>` : '<small>not enough hands</small>'}</span>
    <span class="axis">${whisker}</span>
    <span class="r hide-sm net"><span class="${cls(p.net_bb)}">${p.net_bb != null ? moneySigned(p.net_bb) : '–'}</span><small>${plural(p.hands || 0, 'hand')}</small></span>
    <span class="r col-more obs" title="${esc(OBSERVED_TIP)}">${obs != null ? `<span class="${cls(p.observed_edge)}">${obs}</span><small>vs ${plural(p.observed_opponents || 0, 'opponent')}</small>` : '<small>–</small>'}</span>
    <span class="r col-more sess">${elo}</span>
    <span class="r go">${sc}</span></div>`;
}

function waitingRows(waiting, ranked) {
  return waiting.map(x => `<div class="brow waiting">${ranked ? '<span class="rank">·</span>' : ''}
    ${player(x.player, `<div class="small soft">${x.status === 'stopped' ? '<span class="neg">every deal so far stopped on an error</span>' : x.status === 'playing' ? 'playing its first deals' : x.status === 'done' ? 'no finished deals' : 'waiting to play'}</div>`)}
    <span class="r strength soft">–</span><span class="axis"></span><span class="hide-sm"></span><span class="col-more"></span><span class="col-more"></span><span></span></div>`).join('');
}

function matchupId(p) {
  const ids = Object.fromEntries(state.index.map(m => [m.id.split('/').pop(), m.id]));
  const mu = (p.matchups || [])[(p.matchups || []).length - 1];
  if (mu && state.index.some(m => m.id === mu)) return mu;
  if (mu && (ids[mu] || ids[String(mu).split('/').pop()])) return ids[mu] || ids[String(mu).split('/').pop()];
  const m = state.index.find(m => m.started && ((m.a === p.a && m.b === p.b) || (m.a === p.b && m.b === p.a)));
  return m ? m.id : null;
}

const plainVerdictShort = s => String(s || '').replace(/^Ahead on points$/, 'Leading so far').replace(/^Level on points$/, 'Tied so far').replace(/ · model-based$/, ' (exploratory)').replace(/^Ineligible$/, 'Disqualified');

// A verdict's badge and pill, with what it rests on (assumptions, official or retrospective) beside it.
function verdictCell(p, v, { compact = false } = {}) {
  const src = Core.sourceOf(p, plain);
  const bits = [];
  if (v.decided) bits.push('<span class="decided-badge">Decided</span>');
  bits.push(`<span class="verdict ${v.tone}" title="${esc(v.long)}">${esc(plainVerdictShort(v.short))}</span>`);
  if (v.decided || (p.assumption && p.assumption.status === 'unsupported' && p.assumption.method === 'free')) bits.push(assumeChip(p.assumption, { compact }));
  const drift = p.drift && (typeof p.drift === 'string' ? { status: p.drift } : p.drift);
  if (v.drift) bits.push(chip('needs re-test', 'warn', v.long));
  else if (drift && /^drift detected/.test(drift.status || '')) bits.push(chip('shifted mid-run', 'warn', drift.meaning || 'The score changed sharply between early and late hands.'));
  if (p.identity_qualified && !Object.values(p.attestation || {}).some(at => ['unattested', 'mismatch'].includes(at && (typeof at === 'string' ? at : at.status))))
    bits.push(chip('Model ID not logged', 'quiet', 'A model\'s CLI did not log back a server-confirmed model ID, so only the configured name is known.'));
  for (const [pl, at] of Object.entries(p.attestation || {})) {
    const st = at && (typeof at === 'string' ? at : at.status);
    if (st === 'unattested' || st === 'mismatch') bits.push(chip(`${short3(pl)}: ${st === 'mismatch' ? 'wrong model' : 'ID not logged'}`, st === 'mismatch' ? 'bad' : 'quiet',
      `${(at && at.meaning) || (st === 'mismatch' ? 'Provider returned a different model.' : 'Provider did not log back a confirmed model ID.')}${at && at.policy ? ` (${at.policy})` : ''}`));
  }
  if (!compact && src) bits.push(chip(src.short === 'model-based · not official' ? 'exploratory' : src.short, src.kind === 'official' ? 'ok' : 'quiet', src.text));
  if (v.secondary && !compact) bits.push(chip(`alt test: ${esc(v.secondary.short.toLowerCase())}`, 'quiet secondary',
    'Secondary statistical check shown for comparison only: never decides official verdicts.'));
  return bits.join('');
}

// A verdict worth a chip of its own: decided, drifted, ineligible, exploratory, exhausted or withheld. Everything
// else is "leading so far", which the run's status pill already says once.
const notable = (p, v) => v.decided || v.drift || v.model || v.identity || v.tone === 'bad' || p.verdict === 'exhausted'
  || !!(p.assumption && p.assumption.status === 'unsupported' && p.assumption.method === 'free');

// Each pairing as one calm row: who leads by how much, its range, how many deals, and Watch.
function pairingsSection(rt) {
  if (!rt.pairings.length) return '';
  const order = { a: 0, b: 0, even: 1, decided_with_drift: 2, exhausted: 2, ineligible: 3, open: 4, descriptive: 4, early: 5 };
  const rank = p => Core.driftQualified(p) ? 2 : order[p.verdict] ?? 6;
  const ax = axisFor(rt.pairings.map(p => {
    const up = (p.edge ?? 0) >= 0, o = officialRange(p, up);
    return { strength: Math.abs(p.edge ?? 0), strength_ci95: o ? o.ci : p.ci95 && (up ? p.ci95 : [-p.ci95[1], -p.ci95[0]]) };
  }));
  const descriptive = rt.pooled || rt.unregistered;
  // a range basis every matchup shares is said once, beside the title, not on every row
  const keys = rt.pairings.map(p => Core.basisOf(p)?.key || 'across');
  const shared = keys.every(k => k === keys[0]) && keys[0] !== 'across' ? rt.pairings[0] : null;
  let official = 0;
  const rows = [...rt.pairings].sort((x, y) => rank(x) - rank(y) || Math.abs(y.edge ?? 0) - Math.abs(x.edge ?? 0)).map(p => {
    const up = (p.edge ?? 0) >= 0, lead = up ? p.a : p.b, trail = up ? p.b : p.a;
    const edge = Math.abs(p.edge ?? 0);
    const desc = p.ci95 ? (up ? p.ci95 : [-p.ci95[1], -p.ci95[0]]) : null;
    const off = officialRange(p, up);
    if (off) official++;
    const ci = off ? off.ci : desc;
    const v = verdict(p, plain);
    const full = matchupId(p);
    const o = Core.officialInterval(p);
    const rangeTip = [off ? `Official margin of error: ${ciText(off.ci)} bb/100${o && o.level ? ` (${levelPct(o.level)})` : ''}, valid whenever play stops` : '',
      desc ? `${off ? 'Rough range' : 'Rough margin of error'}: ${ciText(desc)} bb/100` : ''].filter(Boolean).join('. ') || 'No range yet: too few deals';
    const facts = [p.looks_planned && !descriptive ? (p.looks_used ? `checked ${p.looks_used} of ${p.looks_planned} times` : 'not checked yet') : '',
      !descriptive && Core.conversationsShort(p) ? `${p.conversations} of ${p.min_conversations} independent deals needed before a verdict` : '',
      p.fit && p.fit.untestable && rt.players.length > 2 ? 'too few matchups to cross-check against leaderboard' : ''].filter(Boolean);
    const fit = p.fit;
    const fitLine = fit && fit.warn ? `<div class="fitline"><span class="flag">matchup surprise</span> Leaderboard predicted ${signed(up ? fit.fitted : -fit.fitted, 0)} for ${plain(lead)}, but the direct matchup was ${signed(edge, 0)}. Trust the direct matchup.</div>` : '';
    const no = Core.ineligible(p);
    const noLine = no ? `<div class="fitline bad"><span class="flag bad">disqualified</span> ${esc(no.reasons.join('; ') || 'excluded from official verdicts')}</div>` : '';
    const b = Core.basisOf(p);
    return `<div class="mrow ${v.decided ? 'decided' : ''} ${v.tone === 'bad' ? 'ineligible' : ''}">
      <div class="mr-duo">${player(lead, '', { details: false })}
        <span class="mr-edge" title="${esc(v.long)}"><b class="${edge ? 'pos' : ''}">${edge ? signed(edge, 0) : '0'}</b><small>bb/100</small></span>
        ${player(trail, '', { details: false })}</div>
      <div class="mr-bar"><div class="whisker" title="${esc(rangeTip)}"><span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
        <span class="point ${v.tone === 'won' ? 'pos' : ''}" style="left:${ax.x(edge)}"></span></div></div>
      <div class="mr-meta"><span class="mr-deals" ${facts.length ? `title="${esc(facts.join(' · '))}"` : ''}>${plural(p.deals || 0, 'deal')}</span>${!shared && b && b.key !== 'across' ? basisTag(p) : ''}
        ${notable(p, v) ? `<span class="mr-verdict">${verdictCell(p, v, { compact: true })}</span>` : ''}${aivatLine(p, lead)}</div>
      ${full ? `<a class="watch" href="${matchLink(full)}" title="Watch hands from this matchup">Watch ›</a>` : '<span></span>'}
      ${noLine}${fitLine}</div>`;
  }).join('');
  return section('Matchups', 'same cards dealt both ways so card luck cancels out', `<div class="pair-list">${rows}</div>`,
    `${descriptive ? '' : rt.track === 'adaptation' ? 'Results here are exploratory (long memory sessions). ' : 'A badge marks a statistically settled matchup. '}Bars show rough uncertainty: when a bar crosses 0, the two models are still too close to separate. Hover any number for full statistical details.`);
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
    ${names.map(a => `<tr><th title="${plain(a)}"><div class="player">${monogram(a, 'sm')}<span class="mx-name">${name(a)}</span></div></th>${names.map(b => {
      if (a === b) return '<td class="cell self"></td>';
      const c = cell[a + '|' + b];
      if (!c) return '<td class="cell dim">·</td>';
      const won = c.p.verdict === 'a' || c.p.verdict === 'b';
      const outside = won && c.p.assumption && c.p.assumption.status === 'unsupported';
      const mine = won && ((c.p.verdict === 'a') === (c.p.a === a));
      const alpha = won ? ((outside ? 0.08 : 0.16) + (outside ? 0.12 : 0.3) * Math.sqrt(Math.abs(c.v) / max)).toFixed(2) : 0;
      const bg = won ? `background:rgba(var(--${mine ? 'pos' : 'neg'}-rgb),${alpha})` : '';
      const full = matchupId(c.p);
      return `<td class="cell ${outside ? 'outside' : ''}" style="${bg}"><a ${full ? `href="${matchLink(full)}"` : ''} title="${short(a)} vs ${short(b)}: ${esc(plainVerdictShort(verdict(c.p, plain).short))}${outside ? ' (diagnostic check failed)' : ''}">
        <span class="v ${won ? cls(c.v) : 'soft'}">${signed(c.v, 0)}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  const multi = rt.groups.filter(g => g.ranked.length).length > 1;
  const legend = ns => `<div class="mx-legend">${ns.map(n => `<span>${monogram(n, 'xs')}${short(n)}</span>`).join('')}</div>`;
  const tables = rt.groups.map(g => g.ranked.filter(p => paired.has(p))).filter(ns => ns.length >= 3)
    .map((ns, i, all) => `${all.length > 1 || multi ? `<div class="mx-group">${esc(groupName(rt.groups.find(g => g.ranked.includes(ns[0]))))}</div>` : ''}<div class="mx-wrap">${matrix(ns)}</div>${legend(ns)}`);
  if (!tables.length) return '';
  const descriptive = rt.pooled || rt.unregistered;
  const anyOutside = rt.pairings.some(p => Core.isDecided(p) && p.assumption && p.assumption.status === 'unsupported');
  return section('Head to head', 'bb/100 for each row player against each column player', tables.join(''),
    `${descriptive ? 'Gray numbers show scores so far (no matchup here is formally settled).' : `Colored cells are statistically confirmed wins${anyOutside ? ' (dashed cells failed a diagnostic check)' : ''}; gray numbers are scores so far.`} Click any cell to watch hands.`, { id: 'h2h', fold: true });
}

// Tendencies, not quality: how often each does something, out of how many chances.
const STYLE_COLS = [
  ['vpip', 'Plays preflop', 'How often this model puts chips in before the flop instead of folding (VPIP).'],
  ['pfr', 'Raises preflop', 'How often this model raises before the flop (PFR).'],
  ['af', 'Bets per call', 'Bets and raises per call after the flop. Above 1.0 = aggressive, below 1.0 = passive.'],
  ['wtsd', 'To showdown', 'How often it reaches showdown after seeing a flop.'],
  ['wsd', 'Won showdown', 'How often it wins the pot when reaching showdown (excluding split pots).'],
  ['talk', 'Talks', 'How often it sends table-talk messages to its opponent.'],
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
  return section('How they play', 'playing style stats (not a skill grade)', pick + `<div class="scroll">${table(rows)}</div>`,
    `<span class="def-line" aria-live="polite">Click any column heading to see what it measures.</span> Numbers underneath show raw counts.
    Style stats describe how a model plays, not whether it plays well. Rows under 100 hands are dimmed. ${populationLine(rt, 'stats')}`,
    { id: 'style', fold: true, after: block('Split by seat: button vs. big blind', bySeat) + block('Split by opponent', byOpp) });
}

// ---- poker fundamentals: the same textbook spots for every player, each with a worked-out answer --------
const quizLink = id => runLink(state.run, '/q' + (id ? '/' + encodeURIComponent(id) : ''));
const ACTION_WORD = { fold: 'Fold', check: 'Check', call: 'Call', bet: 'Bet', raise: 'Raise', all_in: 'All in', shove: 'All in' };
const answerText = (a, bb) => {
  if (a == null) return '–';
  const x = typeof a === 'string' ? { action: a } : a;
  const word = ACTION_WORD[x.action] || x.action || '–';
  return x.amount != null && !['fold', 'check'].includes(x.action) ? `${word} ${cash(x.amount, bb)}` : word;
};
const answerKey = a => (typeof a === 'string' ? a : a && a.action) || null;
const lostOf = a => a ? a.ev_lost_bb ?? a.ev_loss_bb ?? a.regret_bb ?? null : null;
const KIND_WORD = { pot_odds: 'pot odds', all_in: 'all-in call', shove: 'shove or fold', equity: 'win chance', bluff_catch: 'bluff catcher',
  close_ev: 'close calls', multi_street: 'multi-street plans', range: 'range reading', range_inference: 'range reading', sizing: 'bet sizing' };
const catWord = k => KIND_WORD[k] || String(k).replace(/_/g, ' ');
const PROV = {
  eligible: ['ok', 'verified log', 'Raw answers, model ID, and software versions were all recorded and verified.'],
  ineligible: ['bad', 'not counted', 'Failed log verification, so shown for reference only.'],
  unrecorded: ['warn', 'unverified log', 'This quiz run did not save full verification logs (raw output, model ID, and versions).'],
};
const provChip = x => {
  const pv = x.provenance;
  if (!pv || pv.status === 'bot' || !PROV[pv.status]) return '';
  const [tone, word, tip] = PROV[pv.status];
  const more = [pv.reasons.join('; '), pv.identity ? `identity ${pv.identity}` : '', pv.facts.join(' · ')].filter(Boolean).join(' · ');
  const at = x.attestation && (typeof x.attestation === 'string' ? { status: x.attestation } : x.attestation);
  const att = at && (at.status === 'unattested' || at.status === 'mismatch')
    ? chip(at.status === 'mismatch' ? 'wrong model' : 'ID not logged', at.status === 'mismatch' ? 'bad' : 'quiet', `${at.meaning || ''}${at.policy ? ` (${at.policy})` : ''}`) : '';
  const leak = Core.leakNote(x);
  return chip(word, tone, tip + (more ? ` (${more})` : '')) + att + (leak ? chip(esc(leak.short), 'warn', leak.long) : '') + linkChip(x);
};
// A quiz score of the same model and reasoning level as a tournament player, but run separately.
const modelLevel = x => !!(x.link && x.link.status === 'model-level');
const linkTip = x => `Run separately from the tournament: same model and reasoning level as ${x.link.player ? plain(x.link.player) : 'a tournament player'}, but settings or versions differed (${(x.link.differences || []).join('; ') || 'separate run'}).`;
const linkChip = x => modelLevel(x) ? chip('separate quiz run', 'warn', linkTip(x)) : '';

const quizStore = new Map();
async function loadQuiz() {
  const raw = quizStore.get(state.run) || await api('quiz', ['run', state.run]);
  quizStore.set(state.run, raw);
  Object.assign(state.labels, raw.labels || {});
  Object.assign(state.kinds, raw.kinds || {});
  return Core.quizOf(raw);
}

// The standings page's pointer to the quiz, filled in after the standings are on screen: each model's score
// on the scored category, and on the harder ones, each with its own denominator.
const catTitle = (q, c) => (q.category_info[c] && q.category_info[c].title) || catWord(c);
function scoreText(q, x, quiet = false) {
  const leak = Core.leakNote(x);
  const flag = (x.provenance?.status === 'unrecorded' ? '<span class="soft"> (unverified)</span>' : x.provenance?.status === 'ineligible' ? '<span class="neg"> not counted</span>' : '')
    + (leak ? `<span class="soft" title="${esc(leak.long)}"> (older quiz: answers may have leaked)</span>` : '');
  const head = `${x.correct ?? '–'}/${x.total ?? '–'}${q.scored_category ? ` ${esc(catTitle(q, q.scored_category).toLowerCase())}` : ''}`;
  const more = x.others && x.others.total ? ` · ${x.others.correct}/${x.others.total} harder` : '';
  const sep = modelLevel(x) && !quiet ? `<span class="soft" title="${esc(linkTip(x))}"> (separate quiz run)</span>` : '';
  return `<b>${plain(x.player)}</b> ${head}${more}${x.fallbacks ? ` (${plural(x.fallbacks, 'auto-fold')})` : ''}${flag}${sep}`;
}
async function fillQuizTeaser(run) {
  let q;
  try { q = await loadQuiz(); } catch (e) { q = null; }
  const slot = $app.querySelector('.quiz-slot');
  if (!slot || state.run !== run) return;
  if (!q || !q.players.length) { slot.remove(); return; }
  const models = q.players.filter(x => !x.reference);
  const ref = q.players.find(x => x.reference);
  const ceiling = models.length > 1 && models.filter(x => x.correct === x.total).length >= 2 && !models.some(x => x.others);
  slot.outerHTML = `<a class="quiz-teaser card-box" href="${quizLink()}"><span class="qt-k">Poker fundamentals</span>
    <span class="qt-body">${models.some(modelLevel) ? 'Run separately from the tournament: ' : ''}A quiz of ${plural(q.spots.length, 'textbook spot')}${q.categories.length > 1 ? ` across ${plural(q.categories.length, 'category', 'categories')}` : ''} with mathematically exact answers: ${models.map(x => scoreText(q, x, models.every(modelLevel))).join(' · ')}${ref ? ` · ${scoreText(q, ref)} (reference)` : ''}${ceiling ? '<span class="soft"> · multiple perfect scores (spots too easy to separate models)</span>' : ''}</span>
    <span class="al-go">See every spot ›</span></a>`;
}

async function renderQuiz(spotId) {
  const q = await loadQuiz();
  const superseded = q.superseded ? `<div class="notice quiz-notice"><span class="suit">♦</span><div><b>Superseded.</b> ${esc(typeof q.superseded === 'string' ? q.superseded : q.superseded.reason || q.superseded.by || 'a newer version of this quiz replaces it')}</div></div>` : '';
  const leaked = q.prompt_leak && (q.prompt_leak.spots || []).length ? `<div class="notice quiz-notice"><span class="suit">♦</span><div><b>Earlier quiz version${q.prompt_leak.version ? ` (${esc(q.prompt_leak.version)})` : ''}.</b> Its prompts named each spot by a descriptive id, and ${plural(q.prompt_leak.spots.length, 'spot')} had an id naming an action${(q.prompt_leak.named_answer || []).length ? ` (${q.prompt_leak.named_answer.length} of them the best answer itself)` : ''}, so those prompts may have leaked the answer: ${q.prompt_leak.spots.map(esc).join(', ')}. Scores on them are flagged; quiz-4 names spots by an opaque id.</div></div>` : '';
  const fixture = superseded + leaked + (q.fixture ? `<div class="notice quiz-notice"><span class="suit">♦</span><div><b>Fixture data.</b> ${esc(q.fixture === true ? 'These answers are made up to show the layout; they are not real model answers.' : q.fixture)}</div></div>` : '');
  const head = `<div class="page-head"><div><div class="eyebrow">Poker fundamentals</div><h1>${esc(q.suite)}</h1>
      <div class="sub">${plural(q.spots.length, 'spot')} · ${plural(q.players.length, 'player')}${q.version ? ` · version ${esc(q.version)}` : ''}</div></div></div>`;
  const lede = `<p class="lede quiz-lede">Separate from tournament games: every model answers the exact same textbook poker scenarios (calling or folding at a given price, going all in, sizing bets), graded against exact card math.
    This tests basic poker knowledge; Standings show who wins real games.</p>`;
  const spot = spotId ? q.spots.find(x => x.id === spotId) : null;
  sectionCount = 0;
  $app.innerHTML = head + fixture + (spot ? spotView(q, spot) : lede + quizScores(q) + quizSpots(q));
  if (spot) drawSpot(spot);
  else bindSpotFilters(q);
  $app.querySelectorAll('.qcat-pick').forEach(sel => sel.onchange = () => {
    sel.closest('.qcat-narrow').querySelectorAll('.qcat-one').forEach(d => { d.hidden = d.dataset.i !== sel.value; });
  });
}

function quizScores(q) {
  const maxLost = Math.max(1e-9, ...q.players.map(x => x.ev_lost_bb || 0));
  const sc = q.scored_category;
  const rows = q.players.map(x => {
    const share = x.score != null ? x.score : 0;
    const valid = x.valid != null && x.total ? `<small class="q-valid" title="Right out of valid answers: an auto-fold (no valid answer after retrying) never gets credit">${x.valid_accuracy != null ? `${Math.round(x.valid_accuracy * 100)}% of ${plural(x.valid, 'valid answer')}` : 'no valid answers'}${x.fallbacks ? ` · <b class="neg">${plural(x.fallbacks, 'auto-fold')}, no credit</b>` : ''}</small>` : '';
    return `<div class="qrow ${x.reference ? 'ref' : ''}">${player(x.player, `<div class="q-prov">${provChip(x)}</div>`, { details: false })}
      <span class="q-score"><b>${x.correct ?? '–'}</b><small>/${x.total ?? '–'} right</small>${valid}</span>
      <span class="q-bar" title="${Math.round(share * 100)}% right"><span style="width:${(share * 100).toFixed(1)}%"></span></span>
      <span class="q-lost" title="Expected amount lost compared with the best answer, totaled over ${sc ? `the ${esc(catTitle(q, sc).toLowerCase())} spots` : 'every spot'}">${x.ev_lost_bb != null
        ? `<b class="${x.ev_lost_bb > 0 ? 'neg' : ''}">${money(x.ev_lost_bb)}</b><small> lost</small><span class="q-lostbar"><span style="width:${(100 * (x.ev_lost_bb || 0) / maxLost).toFixed(1)}%"></span></span>` : '<span class="dim">–</span>'}</span></div>`;
  }).join('');
  const ref = q.players.filter(x => x.reference);
  const info = sc && q.category_info[sc];
  const caption = `${sc ? `Scores above cover the <b>${esc(catTitle(q, sc).toLowerCase())}</b> category only${info && info.about ? ` (${esc(info.about)})` : ''}. Harder categories are shown separately below.` : 'Right: spots answered with the mathematically best move.'}
    Auto-folds (no legal answer after retrying) never count as right.
    "Lost" is the total expected amount given up vs. the best move, at $10/$20 blinds.${ref.length ? ` <b>${ref.map(x => plain(x.player)).join(', ')}</b> is a simple baseline bot.` : ''}
    ${q.population || q.estimand ? ` Scope: ${esc(q.population || q.estimand)}.` : ' Scores reflect this fixed set of textbook spots only.'}`;
  return section(sc ? `Scores · ${esc(catTitle(q, sc))}` : 'Scores', sc ? 'main category · less lost is better' : 'less lost is better', `<div class="qlist">${rows}</div>`, caption) + quizCategories(q);
}

// Per category: each player's right answers out of the category's own spots, fallbacks and loss beside them.
function quizCategories(q) {
  const cats = q.categories.filter(c => c !== q.scored_category);
  if (q.categories.length < 2 || !cats.length) return '';
  const count = c => q.spots.filter(s => (s.category || s.kind || 'other') === c).length;
  const head = `<thead><tr><th>Player</th>${cats.map(c => `<th class="num" title="${esc((q.category_info[c] && q.category_info[c].about) || '')}">${esc(catTitle(q, c))}<small class="nof">${plural(count(c), 'spot')}</small></th>`).join('')}</tr></thead>`;
  const cell = (x, c) => {
    const k = (x.by_category || {})[c];
    if (!k || !k.total) return { cls: '', html: '<span class="dim">–</span>' };
    const r = (k.correct ?? 0) / k.total;
    const lost = num(k.ev_lost_bb, k.ev_loss_bb);
    return { cls: `qcat ${r === 1 ? 'full' : r < 0.5 ? 'low' : ''}`, html: `${k.correct ?? 0}<small class="nof">/${k.total}${k.fallbacks ? ` · ${k.fallbacks} auto-fold` : ''}${lost != null ? ` · ${money(lost)} lost` : ''}</small>` };
  };
  const rows = q.players.map(x => `<tr><td class="who-cell">${player(x.player, '', { details: false })}</td>${cats.map(c => {
    const v = cell(x, c);
    return `<td class="num ${v.cls}">${v.html}</td>`;
  }).join('')}</tr>`).join('');
  const pick = `<label class="metric-bar qcat-bar"><span>Category</span><select class="qcat-pick" aria-label="Which category">${cats.map((c, i) =>
    `<option value="${i}">${esc(catTitle(q, c))} · ${plural(count(c), 'spot')}</option>`).join('')}</select></label>`;
  const narrow = `<div class="qcat-narrow">${pick}${cats.map((c, i) => `<div class="qcat-one" data-i="${i}"${i ? ' hidden' : ''}>${q.players.map(x => {
    const v = cell(x, c);
    return `<div class="qcat-row">${player(x.player, '', { details: false })}<span class="num ${v.cls}">${v.html}</span></div>`;
  }).join('')}</div>`).join('')}</div>`;
  return section(q.scored_category ? 'Harder categories' : 'By category', 'scored separately to separate models that ace the basics',
    `<div class="scroll qcat-wide"><table class="style-t qcat-t">${head}<tbody>${rows}</tbody></table></div>${narrow}`,
    `Correct answers out of each category's spots (auto-folds never count as right). ${q.scored_category ? 'Shown separately from the main score above.' : ''}`);
}
const num = (...xs) => xs.find(x => x != null && isFinite(x)) ?? null;

// Which players chose differently on a spot: 'split' when their valid answers differ, 'missed' when someone's answer is missing, unverified or
// not the best, else 'agree'.
function spotAgreement(q, s) {
  const all = q.players.map(x => x.answers[s.id]);
  const as = all.filter(Boolean);
  const keys = new Set(as.map(a => Core.isFallback(a) ? 'fallback' : answerKey(a)));
  const missed = as.length < all.length || as.some(a => Core.isFallback(a) || a.outcome === 'unverified' || !(a.correct ?? (answerKey(a) === answerKey(s.best))));
  return keys.size > 1 ? 'split' : missed ? 'missed' : 'agree';
}
const SPOT_FILTERS = [['all', 'All spots'], ['split', 'Players disagree'], ['missed', 'Someone missed'], ['agree', 'Everyone right']];
const SPOT_PAGE = 12;
function quizSpots(q) {
  const f = state.spotFilter && state.spotFilter.run === state.run ? state.spotFilter : (state.spotFilter = { run: state.run, cat: 'all', agree: 'all', open: {} });
  const cats = q.categories;
  const agree = Object.fromEntries(q.spots.map(s => [s.id, spotAgreement(q, s)]));
  const count = k => k === 'all' ? q.spots.length : q.spots.filter(s => agree[s.id] === k).length;
  f.at = sectionCount;
  const row = s => {
    const marks = q.players.map(x => {
      const a = x.answers[s.id];
      const fb = Core.isFallback(a);
      const ok = !a ? null : fb ? false : a.correct ?? (answerKey(a) === answerKey(s.best));
      return `<span class="qmark ${ok == null ? 'none' : fb ? 'fb' : ok ? 'ok' : 'bad'}" title="${plain(x.player)}: ${a ? esc(answerText(a, s.bb)) : 'no answer'}${fb ? ' (fallback: no valid answer, no credit)' : ok == null ? '' : ok ? ' (best)' : ' (not best)'}">${monogram(x.player, 'xs')}<i>${ok == null ? '·' : fb ? '!' : ok ? '✓' : '✗'}</i></span>`;
    }).join('');
    const tag = agree[s.id] === 'split' ? '<span class="chip warn sp-tag">players disagree</span>' : agree[s.id] === 'missed' ? '<span class="chip quiet sp-tag">someone missed</span>' : '';
    return `<a class="spot-row" href="${quizLink(s.id)}"><span class="sp-cards">${PokerTable.cards(s.hole || [], 2, 'sm')}</span>
      <span class="sp-main"><b>${esc(Core.spotTitle(s))}</b><small>${s.title === s.id ? `${esc(s.id)} · ` : ''}${s.variant != null ? `variant ${esc(s.variant)} · ` : ''}best: ${esc(answerText(s.best, s.bb))}${tag}</small></span>
      <span class="sp-marks">${marks}</span><span class="al-go">›</span></a>`;
  };
  const shown = q.spots.filter(s => (f.cat === 'all' || (s.category || s.kind || 'other') === f.cat) && (f.agree === 'all' || agree[s.id] === f.agree));
  const groups = cats.filter(c => f.cat === 'all' || c === f.cat).map(c => [c, shown.filter(s => (s.category || s.kind || 'other') === c)]).filter(([, ss]) => ss.length);
  const body = groups.map(([c, ss]) => {
    const more = ss.length > SPOT_PAGE && !f.open[c];
    return `<div class="spot-group"><div class="sg-head"><b>${esc(catTitle(q, c))}</b><small>${plural(ss.length, 'spot')}${q.category_info[c] && q.category_info[c].about ? ` · ${esc(q.category_info[c].about)}` : ''}</small></div>
      <div class="spot-list">${(more ? ss.slice(0, SPOT_PAGE) : ss).map(row).join('')}</div>
      ${more ? `<button class="toggle sg-more" data-more="${esc(c)}">Show all ${ss.length} ${esc(catTitle(q, c).toLowerCase())} spots</button>` : ''}</div>`;
  }).join('') || '<p class="soft sg-none">No spot matches these filters.</p>';
  const pick = (name, items, cur) => items.map(([k, label, n]) => `<button class="toggle" data-${name}="${esc(k)}" aria-pressed="${cur === k}">${esc(label)}${n != null ? ` <small>${n}</small>` : ''}</button>`).join('');
  const bar = `<div class="spot-filters"><div class="sf-row" role="group" aria-label="Category"><span>Category</span>${pick('cat', [['all', 'All', q.spots.length], ...cats.map(c => [c, catTitle(q, c), q.spots.filter(s => (s.category || s.kind || 'other') === c).length])], f.cat)}</div>
    ${q.players.length > 1 ? `<div class="sf-row" role="group" aria-label="Answers"><span>Answers</span>${pick('agree', SPOT_FILTERS.map(([k, l]) => [k, l, count(k)]), f.agree)}</div>` : ''}</div>`;
  return section('Spots', 'grouped by category; open one to see the question as asked, the worked answer and every player\'s choice', bar + `<div class="spot-groups">${body}</div>`, '', { id: 'spots' });
}
function bindSpotFilters(q) {
  const f = state.spotFilter;
  const redraw = () => {
    const sec = $app.querySelector('#spots');
    if (!sec) return;
    sectionCount = f.at;
    const tmp = document.createElement('div');
    tmp.innerHTML = quizSpots(q);
    sec.replaceWith(tmp.firstElementChild);
    bindSpotFilters(q);
  };
  $app.querySelectorAll('[data-cat]').forEach(b => b.onclick = () => { f.cat = b.dataset.cat; redraw(); });
  $app.querySelectorAll('[data-agree]').forEach(b => b.onclick = () => { f.agree = b.dataset.agree; redraw(); });
  $app.querySelectorAll('[data-more]').forEach(b => b.onclick = () => { f.open[b.dataset.more] = true; redraw(); });
}

function spotView(q, s) {
  const i = q.spots.indexOf(s);
  const prev = q.spots[i - 1], next = q.spots[i + 1];
  const bb = s.bb || 1;
  const pot = s.pot ?? 0, toCall = s.to_call ?? 0;
  const need = s.needed ?? (toCall ? toCall / (pot + toCall) : null);
  const closes = toCall && (['call', 'inference'].includes(s.kind) || PokerTable.callCloses({ street: s.street, toCall, stack: s.stack, opponentAllIn: s.villain_stack === 0 }));
  const ev = s.ev || {};
  const facts = [
    ['Pot', cash(pot, bb)], toCall ? ['To call', cash(toCall, bb)] : null,
    need != null ? (closes ? ['Break-even', `${Math.round(need * 100)}% <small>calling ends the betting, so it must win this often to profit</small>`]
      : ['Call price', `${Math.round(need * 100)}% <small>immediate pot odds before future betting</small>`]) : null,
    s.equity != null ? ['Win chance', `${Math.round(s.equity * 100)}% <small>${esc(s.equity_note || 'against the opponent\'s stated hands')}</small>`] : null,
    ['Best move', `<b class="gold">${esc(answerText(s.best, bb))}</b>`],
    Object.keys(ev).length ? ['Expected value', Object.entries(ev).map(([k, v]) => `${esc(ACTION_WORD[k] || k)} ${moneySigned(v)}`).join(' · ')] : null,
  ].filter(Boolean).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  const answers = q.players.map(x => {
    const a = x.answers[s.id];
    const fb = Core.isFallback(a);
    const ok = !a ? null : fb ? false : a.correct ?? (answerKey(a) === answerKey(s.best));
    const lost = lostOf(a);
    const why = a && (a.reasoning || a.say) ? `<details class="q-why"><summary>why?</summary><div>${esc(a.reasoning || a.say)}</div></details>` : '';
    return `<div class="qans ${ok == null ? '' : ok ? 'ok' : 'bad'} ${x.reference ? 'ref' : ''}">${player(x.player, '', { details: false })}
      <span class="qa-act">${a ? esc(answerText(a, bb)) : '<span class="dim">no answer</span>'}</span>
      <span class="qa-ok">${ok == null ? '' : fb ? `<span title="No legal answer after retrying, so the game auto-folded/checked (no credit)">! auto-fold, no credit</span>` : ok ? '✓ best' : '✗ not best'}</span>
      <span class="qa-lost">${lost != null ? (lost > 0 ? `<b class="neg">${money(lost)}</b> lost` : '$0 lost') : ''}</span>${why}</div>`;
  }).join('');
  return `<div class="spot-nav"><a href="${quizLink()}">‹ All spots</a><span>Spot ${i + 1} of ${q.spots.length} · ${esc(catTitle(q, s.category || s.kind || 'other'))}</span>
      <span class="nav">${prev ? `<a href="${quizLink(prev.id)}" aria-label="Previous spot">←</a>` : '<span class="dim">←</span>'}${next ? `<a href="${quizLink(next.id)}" aria-label="Next spot">→</a>` : '<span class="dim">→</span>'}</span></div>
    <div class="spot-grid"><div class="spot-main"><div class="card-box play spot-table"><div class="play-head"><h3>${esc(Core.spotTitle(s))}</h3><span class="small soft">${esc(Core.bbProse(s.situation || s.history || ''))}</span></div>
      <div class="table-slot">${PokerTable.html([{ name: '<span class="nm">Player</span>', badge: '<span class="mono lg" style="--c:#a97a2a">?</span>' },
        { name: '<span class="nm">Opponent</span>', badge: '<span class="mono lg" style="--c:#55605a">Op</span>' }])}</div></div>
      ${spotQuestion(q, s)}</div>
      <div class="spot-side"><div class="card-box spot-facts"><dl>${facts}</dl>${s.explain || s.explanation ? `<p class="spot-explain">${esc(Core.bbProse(s.explain || s.explanation))}</p>` : ''}</div>
        <div class="card-box spot-answers"><div class="sa-head">Each player's answer</div>${answers}</div></div></div>`;
}

// The question as the model received it, beside the worked answer: the recorded prompt, else the stated facts.
const RULE_WORD = k => String(k).replace(/_plus$/, ' or better').replace(/_/g, ' ');
const ruleText = r => [].concat(r || []).map(RULE_WORD).join(' or ');
const ACT_WORD = { bet: 'bet', raise: 'raised to', call: 'called', check: 'checked', fold: 'folded', all_in: 'went all in' };
function spotQuestion(q, s) {
  const pr = Core.spotPrompt(q, s);
  const src = !pr ? '' : pr.source === 'stored' ? 'stored with the spot'
    : pr.source === 'recorded' ? `from the call log of ${plural(pr.answers, 'answer')}${pr.differs ? '; players were not all sent the same text, this is the first' : ''}`
    : `rebuilt from the spot with the quiz code${pr.version ? ` (${esc(pr.version)})` : ''}${pr.version && q.version && pr.version !== q.version ? `; this quiz ran ${esc(q.version)}, so the wording may differ` : ''}`;
  const stated = [];
  if (s.villain_range) stated.push(['Opponent\'s possible hands', `${esc(s.villain_range)}${s.villain_combos ? ` <small>${s.villain_combos} combinations</small>` : ''}`]);
  if ((s.actions || []).length) stated.push(['Hand so far', s.actions.map(a => `${esc(a.street)}: ${a.seat === 0 ? 'you' : 'Opponent'} ${esc(ACT_WORD[a.action] || a.action)}${a.amount != null && !['check', 'fold'].includes(a.action) ? ` ${a.amount}` : ''}`).join('; ')]);
  if (s.line) stated.push(['Opponent\'s strategy', Object.entries(s.line).map(([k, r]) => `${esc(k)}: bets with ${esc(ruleText(r))}`).join('; ')]);
  if (s.river_plan) stated.push(['If you call', `river: Opponent goes all in with ${esc(ruleText(s.river_plan))}, else checks`]);
  if (s.shove_call) stated.push(['If you go all in', `Opponent calls with ${esc(ruleText(s.shove_call))}, else folds`]);
  if (s.responses) stated.push(['If you bet', Object.entries(s.responses).map(([k, r]) => `${esc(k === 'all_in' ? 'all in' : k.replace('bet_', 'bet '))}: Opponent calls with ${esc(ruleText(r))}`).join('; ')]);
  if ((s.options || []).length) stated.push(['Allowed moves', s.options.map(o => esc(o === 'all_in' ? 'all in' : o.replace('bet_', 'bet '))).join(', ')]);
  const facts = stated.length ? `<dl class="sq-facts">${stated.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>` : '';
  return `<div class="card-box spot-question"><div class="sa-head">The question, as the model received it</div>
    ${pr ? `<pre class="sq-text">${esc(pr.text)}</pre><p class="small soft sq-src">Exact text ${src}.</p>${facts ? `<details class="sq-more"><summary>What it states, in short</summary>${facts}</details>` : ''}`
      : `<p class="small soft">No recorded prompt: what the question states, from the spot.</p>${facts}`}</div>`;
}

// The spot as a table snapshot: the deciding seat's cards face up, the opponent's face down.
const snapStreet = (s, board) => s.street || (board.length >= 5 ? 'river' : board.length === 4 ? 'turn' : board.length === 3 ? 'flop' : 'preflop');
function drawSpot(s) {
  const el = $app.querySelector('.spot-table');
  if (!el) return;
  const bb = s.bb || 1, pot = s.pot ?? 0, toCall = s.to_call ?? 0;
  const heroBet = s.hero_bet ?? 0, villainBet = s.villain_bet ?? heroBet + toCall;
  const board = s.board || [];
  const seat = (hole, stack, bet, last) => ({ stack, bet, folded: false, allIn: stack === 0, hole, shown: false, net: null, won: false, last,
    made: hole[0] === '??' ? '' : PokerTable.madeHand(hole, board), draw: hole[0] === '??' ? '' : PokerTable.drawOf(hole, board) });
  const snap = { street: snapStreet(s, board),
    board, actor: 1, label: '', button: s.button ?? (s.position === 'big_blind' ? 1 : 0), pot, collected: pot - heroBet - villainBet,
    seats: [seat(s.hole || [], s.stack ?? s.hero_stack ?? 0, heroBet, ''), seat(['??', '??'], s.villain_stack ?? s.stack ?? 0, villainBet, s.villain_action || (toCall ? 'bet' : ''))],
    done: false, toAct: 0, toCall, potOdds: toCall ? toCall / (pot + toCall) : null,
    callCloses: toCall ? ['call', 'inference'].includes(s.kind) || PokerTable.callCloses({ street: snapStreet(s, board), toCall, stack: s.stack, opponentAllIn: s.villain_stack === 0 }) : null };
  PokerTable.show(el, snap, { fmt: c => cash(c, bb), hidden: [1], names: ['Player', 'Opponent'] });
}

// ---- games ----------------------------------------------------------------------------------------
// The list has one entry per pairing. The replay walks the pairing's deals in playing order, across every
// round and conversation, so its arrows, auto-next and big swings never stop at the end of one conversation.
const pairKey = m => [m.a, m.b].sort().join('|');
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
    P.failed = P.started.filter(m => m.failed && !m.finished).length;
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
  if (P.reps.length > 1) bits.push(`${P.reps.length} sessions${P.rounds.length > 1 ? ' each' : ''}`);
  return bits.join(' × ');
}

// One pairing in the list: names, its verdict, and its edge over checked deals (from the ratings).
function matchNav(P, cur, rt) {
  const pr = rt && rt.pairings.find(p => pairKey(p) === P.key);
  const v = pr ? verdict(pr, plain) : null;
  const own = p => esc(nameParts(labelText(p))[0]);
  const active = cur && pairKey(cur) === P.key;
  const e = pr && pr.deals ? (pr.a === P.a ? 1 : -1) * (pr.edge ?? 0) : null;
  const edge = e != null ? `<b class="${cls(e)}" title="${own(P.a)}'s win rate (bb/100) across finished deals">${signed(e, 0)}<small>/100</small></b>` : '';
  const deals = pr && pr.deals != null ? pr.deals : P.deals;
  const status = !P.started.length ? 'waiting to start'
    : `${v && notable(pr, v) ? `${v.decided ? '<span class="decided-badge">Decided</span>' : ''}<span class="vword ${v.tone}">${esc(plainVerdictShort(v.short))}</span> · ` : ''}${plural(deals, 'deal')}${scopeText(P) ? ` · ${scopeText(P)}` : ''}${P.failed ? ` · <span class="neg" title="Deals where a model's CLI returned an error; they are not counted">${P.failed} failed</span>` : ''}`;
  return `<div class="pgroup ${active ? 'active' : ''} ${P.started.length ? '' : 'planned'}"><a class="phead" ${P.latest ? `href="${matchLink(P.latest.id)}"` : ''}>
      <span class="duo">${monogram(P.a, 'sm')}${monogram(P.b, 'sm')}</span>
      <span class="pair-names">${own(P.a)} <span class="vs">vs</span> ${own(P.b)}</span>${P.live ? '<span class="live-dot" title="Playing now"></span>' : ''}${edge}</a>
    <div class="pmeta">${status}</div></div>`;
}

// The replay never waits for the ratings: it shows the last ones known and asks for news in the background.
function ratingsForRun() {
  if (state.run === '*') return null;
  const raw = ratingsStore.get(state.run);
  if (!raw || !ratingsUpToDate(raw)) ratingsLater(state.run);
  if (!raw) return null;
  const rt = normalize(raw);
  rt.audit = raw.run_audit || null;
  return rt;
}

// A pairing's conversations and each deal's checked result, from the server; kept while nothing in it plays.
const pairingStore = new Map();
async function loadPairing(m) {
  const k = `${m.run}|${pairKey(m)}`;
  const hit = pairingStore.get(k);
  if (hit && (!hit.open || Date.now() - hit.t < 5000)) return hit.data;
  let data = null;
  try { data = await api('pairing', ['p', k]); } catch (e) { return hit ? hit.data : null; }
  pairingStore.set(k, { t: Date.now(), data, open: !data || data.conversations.some(c => c.live || !c.finished) });
  return data;
}

async function renderMatches(id, params, soft) {
  if (!state.index.length) {
    $app.innerHTML = '<div class="card-box empty"><h2>No matchups in this run yet</h2></div>';
    return;
  }
  const started = state.index.filter(m => m.started);
  const playSuffix = id && /\/(a-first|b-first)$/.exec(id);
  const baseId = playSuffix ? id.slice(0, -playSuffix[0].length) : id;
  if (playSuffix) state.showPlay = playSuffix[1] === 'b-first' ? 1 : 0;
  let cur = state.index.find(m => m.id === baseId);
  if (!cur && baseId) {
    const tail = baseId.split('/').pop();
    cur = state.index.find(m => m.id === id || m.id.endsWith('/' + baseId) || m.id.split('/').pop() === tail);
    if (!cur) {
      const P = pairingsOf(state.index).find(P => P.ms.some(m => m.id.split('/').pop() === tail));
      cur = P && P.latest;
    }
  }
  cur = cur || started.find(m => m.live) || started[0] || state.index[0];
  Object.assign(state.labels, cur.labels || {});
  const rt = ratingsForRun();
  const pairs = pairingsOf(state.index);
  const groups = [['Playing now', pairs.filter(P => P.live)], ['Played', pairs.filter(P => !P.live && P.started.length)],
    ['Not started', pairs.filter(P => !P.started.length)]];
  const curP = pairs.find(P => P.key === pairKey(cur));
  const toggle = `<button class="mtoggle" aria-expanded="${!!state.listOpen}"><span>${state.listOpen ? 'Hide matchups' : `All ${plural(pairs.length, 'matchup')}`}</span><i>▾</i></button>`;
  const list = `<div class="mbar">${toggle}</div>${trackBar(state.run, '/m').replace('track-bar', 'track-bar compact')}` + groups.filter(g => g[1].length).map(([title, ps]) =>
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
    about: !!document.querySelector('.about[open]'), opts: !!document.querySelector('.opts.open'), cmp: !!document.querySelector('.cmp[open]'),
  } : null;
  const scroll = window.scrollY;
  const listScroll = document.querySelector('.mlist')?.scrollTop || 0;
  const navScroll = document.querySelector('.dn-rows')?.scrollTop;
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
  if (keep) {
    for (const k of keep.why) {
      const [j, step] = k.split(':');
      $app.querySelector(`.play[data-play="${j}"] .ev[data-step="${step}"]`)?.classList.add('open');
    }
    for (const j of keep.notes) $app.querySelector(`.play[data-play="${j}"] details.notes`)?.setAttribute('open', '');
    if (keep.about) $app.querySelector('.about')?.setAttribute('open', '');
    if (keep.opts) setOpts(true);
    if (keep.cmp) $app.querySelector('.cmp')?.setAttribute('open', '');
  }
  const rows = $app.querySelector('.dn-rows');
  if (rows && navScroll != null && soft) rows.scrollTop = navScroll;
  else if (rows) { const on = rows.querySelector('.dn-cell.on'); if (on) rows.scrollTop = Math.max(0, on.offsetTop - rows.offsetTop - rows.clientHeight / 2); }
  if (after) after();
}

// Go to one entry of the pairing's deal sequence: a deal of this conversation, or of another one.
function goSeq(i) {
  const v = state.view;
  const x = v && v.seq[i];
  if (!x) return false;
  if (x.id === v.cur.id && !v.played.has(x.deal)) return false;
  history.replaceState(null, '', matchLink(x.id, x.deal));
  route({ nav: true });
  return true;
}

function chipColor(v, max) {
  if (!v) return '#55605a';
  const t = Math.sqrt(Math.abs(v) / max);
  const base = v > 0 ? 'var(--pos-chip)' : 'var(--neg-chip)';
  return `color-mix(in srgb, ${base} ${Math.round(35 + 65 * t)}%, #24302a)`;
}

const timelineMode = () => state.mode;
const STATUS_WORD = { verified: 'checked', unchecked: 'not checked', provisional: 'in progress (not counted yet)', playing: 'playing now', rejected: 'failed replay check (excluded)', failed: 'stopped on an error (not counted)' };

// A play that stopped because a model's CLI gave up on a decision (the server's `failure`), in words.
const TRY_ORDINAL = ['first', 'second', 'third', 'fourth', 'fifth'];
function failureNote(plays, pick) {
  const bad = plays.filter(p => p.failure);
  if (!bad.length) return '';
  const rows = bad.map(p => {
    const f = p.failure;
    const n = Number((/\.d(\d+)$/.exec(f.decision || '') || [])[1]);
    const which = Number.isFinite(n) ? `its ${TRY_ORDINAL[n] || `${n + 1}th`} decision` : 'a decision';
    const long = f.detail && f.detail !== f.message;
    return `<div class="fail-row"><p><b>Play ${p.j + 1} stopped: ${esc(f.title || 'the CLI returned an error')}.</b> ${name(f.player)} gave no answer for ${which} of this deal${f.tries > 1 ? ` after ${f.tries} tries` : ''}${f.at ? ` (${ago(f.at)})` : ''}.</p>
      <blockquote class="fail-msg">${esc(f.message)}</blockquote>
      ${long ? `<details class="fail-more"><summary>Full error</summary><pre>${esc(f.detail)}</pre></details>` : ''}</div>`;
  }).join('');
  const other = plays.filter(p => !p.failure && !p.log.some(e => e.kind === 'result'));
  const halted = other.length ? ` Play ${other[0].j + 1} was stopped with it.` : '';
  return `<div class="fail-note card-box" role="status"><span class="fail-icon" aria-hidden="true">!</span><div class="fail-body">${rows}
    <p class="small soft">This deal is not counted in any total or standing.${halted} Running the same plan again replays it.</p></div></div>`;
}

// Every deal of the pairing as a small chip, one row per session (or per round of one-deal sessions),
// coloured by who won it over both plays. Clicking one opens it.
function dealNav(seq, pos, L, R, swings) {
  const counted = seq.filter(x => x.net != null && (x.status === 'verified' || x.status === 'unchecked'));
  const max = Math.max(1, ...counted.map(x => Math.abs(x.net)));
  const swing = new Set(swings.map(x => `${x.id}#${x.deal}`));
  const longConv = seq.some(x => x.deals_total > 1);
  const groups = new Map();
  seq.forEach((x, i) => {
    const k = longConv ? `c${x.conv}` : `r${x.round}`;
    if (!groups.has(k)) groups.set(k, { x, cells: [] });
    groups.get(k).cells.push([x, i]);
  });
  const rounds = new Set(seq.map(x => x.round)).size;
  const label = x => longConv ? (rounds > 1 ? `R${x.round} · S${x.replicate}` : `Session ${x.replicate}`) : `Round ${x.round}`;
  const cell = ([x, i]) => {
    const who = x.net > 0 ? short3(L) : short3(R);
    const what = x.net == null ? STATUS_WORD[x.status] || x.status : x.net === 0 ? 'even over both plays' : `${who} won ${money(Math.abs(x.net))} over both plays`;
    const tip = `Deal ${i + 1}${groups.size > 1 ? ` (${label(x)}${x.deals_total > 1 ? `, deal ${x.deal + 1}` : ''})` : ''}: ${what}${x.net != null && x.status !== 'verified' ? ` · ${STATUS_WORD[x.status]}` : ''}`;
    return `<button class="dn-cell s-${x.status} ${i === pos ? 'on' : ''} ${swing.has(`${x.id}#${x.deal}`) ? 'swing' : ''} ${x.reach === 'showdown' ? 'sd' : ''}"
      data-seq="${i}" style="--c:${x.net == null ? '' : chipColor(x.net, max)}" title="${esc(tip)}" aria-label="${esc(tip)}"></button>`;
  };
  const multi = groups.size > 1;
  const rows = [...groups.values()].map(g => `<div class="dn-row">${multi ? `<span class="dn-lab">${esc(label(g.x))}</span>` : ''}<span class="dn-cells">${g.cells.map(cell).join('')}</span></div>`).join('');
  const fit = !multi && seq.length <= 60 && !phone();
  return `<div class="dn-rows ${multi ? 'multi' : ''} ${fit ? 'fit' : ''}" style="--n:${seq.length}">${rows}</div>`;
}

async function matchBody(cur, params, rt, P) {
  const [loaded, pr] = await Promise.all([Promise.all(cur.matches.map(x => loadMatch(x.id))), loadPairing(cur)]);
  const [ma, mb] = loaded;
  const pairing = rt ? rt.pairings.find(p => pairKey(p) === pairKey(cur)) || null : null;
  const bb = ma.config.game.big_unit || ma.config.game.bb || 1;
  const ha = Object.fromEntries(ma.hands.map(h => [h.hand, h]));
  const hb = mb ? Object.fromEntries(mb.hands.map(h => [h.hand, h])) : {};
  const total = ma.config.seeds.length;
  const played = new Set([...Object.keys(ma.events), ...Object.keys(mb ? mb.events : {})].map(Number));
  const done = [...Array(total).keys()].filter(k => ha[k] && hb[k]);
  const nets = Object.fromEntries(done.map(k => [k, ha[k].net[0] + hb[k].net[1]]));
  const status = ma.status || {};
  const wanted = params.has('deal') ? Number(params.get('deal')) - 1 : null;
  const pick = wanted != null && played.has(wanted) ? wanted : !played.size ? 0 : cur.live ? Math.max(...played) : Math.min(...played);

  // the pairing's deals in order; without the server's pairing answer, this conversation alone
  const L = pr ? pr.a : cur.a, R = L === cur.a ? cur.b : cur.a;
  const sign = L === cur.a ? 1 : -1;
  const STAGE = ['preflop', 'flop', 'flop', 'flop', 'turn', 'river'];
  const reach = h => h.showdown ? 'showdown' : STAGE[(h.board || []).length] || 'river';
  const own = pr || { a: L, b: R, conversations: [{ id: cur.id, round: cur.round, replicate: cur.replicate, deals_total: total,
    deals: done.map(k => [k, status[k] === 'rejected' ? null : sign * nets[k] / bb, status[k] || 'unchecked', reach(ha[k])]) }] };
  const seq = Core.dealSequence(own, { [cur.id]: [...played] });
  // a deal one of this session's plays gave up on is failed, whatever the pairing answer (or its absence) says
  const gaveUp = new Set(loaded.filter(m => m && m.failure && !m.result).map(m => m.failure.hand));
  for (const x of seq) if (x.id === cur.id && gaveUp.has(x.deal) && !['verified', 'unchecked', 'rejected'].includes(x.status)) Object.assign(x, { status: 'failed', net: null });
  const pos = Math.max(0, seq.findIndex(x => x.id === cur.id && x.deal === pick));
  const tot = Core.sequenceTotals(seq);
  const planned = Math.max(seq.length, (own.conversations || []).reduce((t, c) => t + (c.deals_total || 0), 0));
  const multiConv = own.conversations.length > 1;
  state.view = { cur, pick, total, played, seq, pos };

  // the scoreboard: the whole pairing, checked deals only
  const v = pairing ? verdict(pairing, plain) : null;
  const lead = tot.net > 0 ? L : tot.net < 0 ? R : null;
  const amount = money(Math.abs(tot.net));
  const rate = tot.counted ? Math.abs(tot.net) / (2 * tot.counted) * 100 : 0;
  const face = (p, side) => `<div class="sb-face ${side} ${lead === p ? 'ahead' : ''}">${monogram(p, 'lg')}<div class="sb-name">${name(p)}</div></div>`;
  const left = [tot.playing ? '1 playing' : '', tot.provisional ? `${tot.provisional} in progress` : '', tot.rejected ? `${tot.rejected} excluded` : '', tot.failed ? `${tot.failed} failed` : ''].filter(Boolean);
  const leftTip = 'Deals still waiting for their seat-swapped replay, flagged by replay checks, or stopped by an error are not counted in totals.';
  const scorebar = `<div class="scorebar card-box">
      ${face(L, 'left')}
      <div class="sb-mid">
        <div class="sb-score">${!tot.counted ? '<span class="sb-word">No deals counted yet</span>' : lead ? `<span class="sb-word">${short3(lead)} up</span> <b>${amount}</b>` : '<b class="even">Even</b>'}</div>
        <div class="sb-sub">${tot.counted ? `<span class="hide-xs" title="${esc(rateTip(rate))}">${trim(Math.round(rate))} bb/100 · </span>${plural(tot.counted, 'deal')}${tot.checked ? '' : ' (unverified)'}` : ''}${multiConv && P ? `<span class="hide-xs"> · ${scopeText(P)}</span>` : ''}</div>
        ${(v && notable(pairing, v)) || left.length ? `<div class="sb-verdict">${v && notable(pairing, v) ? verdictCell(pairing, v, { compact: true }) : ''}
          ${left.length ? `<span class="chip warn" title="${esc(leftTip)}">${left.join(' · ')} · not counted</span>` : ''}</div>` : ''}
      </div>
      ${face(R, 'right')}</div>`;

  const order = [...played].sort((x, y) => x - y);
  const dealNet = pick in nets ? sign * nets[pick] : null;
  const failedHere = (seq[pos] || {}).status === 'failed';
  const outcome = failedHere ? '<b class="neg">stopped on an error</b> · not counted' : dealNet == null ? 'in progress' : dealNet === 0 ? 'even over both plays'
    : `<b>${short3(dealNet > 0 ? L : R)}</b> wins it by <b class="gold">${cash(Math.abs(dealNet), bb)}</b> over both plays`;

  const plays = [ma, mb].filter(Boolean).map((m, j) => ({ m, j, failure: m.failure && !m.result && m.failure.hand === pick ? m.failure : null, names: m.config.players.map(x => esc(nameParts(labelText(x))[0])),
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
  state.view.ids = cur.matches.map(x => x.id);
  state.view.nets = nets;

  const diff = plays.length === 2 ? Core.firstDifference(plays[0].log, plays[1].log) : null;
  const diffFrame = diff ? PokerTable.frameOf(frames, diff.steps[0] != null ? 0 : 1, diff.steps[0] ?? diff.steps[1]) : -1;
  state.view.diffFrame = diffFrame;
  const both = plays.length === 2 && plays.every(p => p.log.some(e => e.kind === 'result'));
  const diffBtn = plays.length < 2 ? '' : diff && diffFrame >= 0
    ? `<button class="jump" data-jump="diff" title="The first decision where the two models, holding the same cards, chose differently (d)">⇄ Where they differ<span class="hide-xs">: ${esc(SEG_NAME[diff.street] || diff.street)}</span></button>`
    : both ? '<span class="jump-none">Same choices in both plays</span>' : '';
  const swings = Core.sequenceSwings(seq, 5, 10);
  const swingBtns = swings.map(x => {
    const i = seq.indexOf(x);
    return `<button class="jump swing ${i === pos ? 'on' : ''}" data-seq="${i}" title="Deal ${i + 1}: ${plain(x.net > 0 ? L : R)} won ${money(Math.abs(x.net))} over both plays">
      ${i + 1}<b class="${cls(x.net)}">${signedCash(x.net, 1)}</b></button>`;
  }).join('');

  const here = seq[pos] || { status: 'playing' };
  const st = here.status;
  const popTag = st === 'rejected' ? `<span class="chip bad" title="${esc(((own.conversations[here.conv] || {}).rejected || []).map(x => x.reason).join('; ') || 'Replay check failed for this deal')}">excluded · not counted</span>`
    : st === 'failed' ? `<span class="chip bad" title="${esc(plays.filter(p => p.failure).map(p => p.failure.message).join(' · ') || 'A play of this deal stopped on an error')}">failed · not counted</span>`
    : (st === 'playing' || st === 'provisional') && !cur.live && !cur.finished ? '<span class="chip warn" title="This deal stopped before both seat-swapped plays finished, with no error recorded (the run was stopped or is between moves). It is not counted; running the same plan again resumes it.">not finished · not counted</span>'
    : st === 'playing' || st === 'provisional' ? '<span class="chip warn" title="Still in progress; counts once both seat-swapped hands finish">in progress</span>'
    : st === 'unchecked' ? '<span class="chip quiet" title="Replay check was not run on this deal">unverified</span>' : '';
  const where = multiConv ? `${P && P.rounds.length > 1 ? `Round ${cur.round} · ` : ''}session ${cur.replicate}${total > 1 ? ` · deal ${pick + 1}/${total}` : ''}` : '';
  const two = plays.length === 2;
  const view = state.split && two ? 'split' : String(state.showPlay);
  const viewBtn = (k, label, tip) => `<button class="seg-b" data-view="${k}" aria-pressed="${view === k}" title="${tip}">${label}</button>`;
  const viewSwitch = two ? `<span class="seg" role="group" aria-label="Which play">${viewBtn('0', 'Play 1', 'First seating of this deal')}${viewBtn('1', 'Play 2 <span class="hide-sm">(Swapped)</span>', 'Same cards with seats swapped')}${viewBtn('split', 'Side-by-Side', 'Watch both seatings side by side')}</span>` : '';
  const controls = `<div class="controls card-box" role="group" aria-label="Replay">
    <div class="deal-title"><span class="nav"><button data-seq="${pos - 1}" ${pos > 0 ? '' : 'disabled'} aria-label="Previous deal" title="Previous deal (←)">‹</button></span>
      <h2>Deal ${pos + 1}<small> / ${planned}</small>${where ? `<span class="round">${esc(where)}</span>` : ''}</h2>
      <span class="nav"><button data-seq="${pos + 1}" ${pos < seq.length - 1 ? '' : 'disabled'} aria-label="Next deal" title="Next deal (→)">›</button></span>${popTag}</div>
    <div class="ctls">
      <button class="ctl" data-ctl="start" title="Back to the deal" aria-label="Back to the start">⏮</button>
      <button class="ctl" data-ctl="prev" title="Step back (,)" aria-label="Step back">◀</button>
      <button class="ctl play" data-ctl="play" title="Play step by step (space)">▶ Play</button>
      <button class="ctl" data-ctl="next" title="Step forward (.)" aria-label="Step forward">▶</button>
      <button class="ctl" data-ctl="end" title="To the end" aria-label="To the end">⏭</button></div>
    <input type="range" class="scrub" min="0" max="${last}" value="${rp.s}" aria-label="Step">
    <span class="step-text"></span>
    <span class="ctl-row">${viewSwitch}
      <button class="toggle sm" id="solver-toggle" aria-pressed="${state.solver}" title="Show poker solver's best move, mistake cost, and win odds for each decision">Solver</button>
      <button class="toggle sm" id="mindpanel-toggle" aria-pressed="${state.mindPanel}" title="Show the model's private thoughts beside its public move (m)">Mind</button>
      <span class="more-wrap"><button class="toggle sm opts-toggle" aria-expanded="false" aria-controls="opts" aria-haspopup="true">⚙ More</button>
        <div class="opts" id="opts" role="group" aria-label="More playback options">
          <label class="opt-row speed" title="How fast each move plays"><span>Speed</span><select id="speed" aria-label="Playback speed">${Object.entries(SPEEDS).map(([k, x]) =>
            `<option value="${k}" ${k === state.speed ? 'selected' : ''}>${x.label}</option>`).join('')}</select></label>
          <button class="opt-row toggle" id="auto-toggle" aria-pressed="${state.autoNext}" title="Automatically advance to the next deal when this one finishes">Auto-next deal</button>
          ${two ? `<button class="opt-row toggle" id="sync-toggle" aria-pressed="${state.mode === 'aligned'}" ${state.split ? 'disabled' : ''} title="On: Play 1 and Play 2 move through flop, turn, and river together. Off: Play 1 finishes, then Play 2">Sync both plays</button>` : ''}
          <button class="opt-row toggle" id="mind-toggle" aria-pressed="${state.mind}" title="Expand every move's private reasoning inside the hand log (r)">Expand all thoughts</button>
          ${diffBtn ? `<span class="opt-row">${diffBtn}</span>` : ''}</div></span></span>
    <span class="deal-outcome soft">${outcome}</span></div>`;

  const sync = plays.length === 2 ? `<div class="sync" role="tablist" aria-label="Which play">
    ${plays.map(p => `<button class="sync-row" role="tab" data-sync="${p.j}" aria-pressed="${state.showPlay === p.j}" aria-selected="${state.showPlay === p.j}"><span class="sync-k">${p.j ? 'Play 2 · swapped' : 'Play 1'}</span><span class="sync-now"></span></button>`).join('')}
    <span class="sync-tag"></span></div>` : '';
  const verdictLong = v ? `<p>${esc(v.long)}</p><div class="about-chips">${verdictCell(pairing, v)}</div>` : '<p class="soft">Score summary appears once standings finish loading.</p>';
  const rejects = own.conversations.flatMap(c => (c.rejected || []).map(x => `${x.hand == null ? 'whole session' : `deal ${x.hand + 1}`}${multiConv ? ` of session ${c.replicate}${c.round > 1 ? `, round ${c.round}` : ''}` : ''}: ${esc(x.reason)}`));
  const population = `Totals count ${tot.checked ? '<b>checked deals</b> only (both seat-swapped plays finished and verified from the deal log)' : '<b>finished deals</b> (replay checks were not run)'}
    (${plural(tot.counted, 'deal')}).${tot.provisional || tot.playing ? ` ${plural(tot.provisional + tot.playing, 'deal')} still in progress ${tot.provisional + tot.playing === 1 ? 'is' : 'are'} not counted yet.` : ''}
    ${tot.rejected ? ` ${plural(tot.rejected, 'deal')} that failed replay checks ${tot.rejected === 1 ? 'is' : 'are'} excluded.` : ''}
    ${tot.failed ? ` ${plural(tot.failed, 'deal')} stopped by a model error ${tot.failed === 1 ? 'is' : 'are'} not counted.` : ''}`;
  const about = `<details class="about card-box"><summary><span>About this result</span><span class="soft small">${v ? esc(plainVerdictShort(v.short)) : 'score pending'} · how this matchup is counted</span></summary>
    <div class="about-body">${verdictLong}<p class="small soft">${population}</p>${rejects.length ? `<ul class="small soft rejects">${rejects.slice(0, 6).map(x => `<li>${x}</li>`).join('')}</ul>` : ''}</div></details>`;
  const nav = `<div class="dealnav card-box"><div class="dn-head"><h3>${planned > seq.length ? `${seq.length} of ${plural(planned, 'deal')} played` : `All ${plural(seq.length, 'deal')}`}</h3>
      <span class="dn-legend"><span><i class="sw pos"></i>${short3(L)} won it</span><span><i class="sw neg"></i>${short3(R)} won it</span><span><i class="sw sd"></i>showdown</span>${tot.provisional || tot.playing ? '<span><i class="sw prov"></i>in progress</span>' : ''}${tot.rejected ? '<span><i class="sw rej"></i>excluded</span>' : ''}${tot.failed ? '<span><i class="sw fail"></i>failed</span>' : ''}</span></div>
    ${dealNav(seq, pos, L, R, swings)}
    ${swingBtns ? `<div class="jumps"><span class="jump-k">Big swings</span><span class="jump-list">${swingBtns}</span></div>` : ''}
    <div class="keys small soft"><span class="kbd">←</span><span class="kbd">→</span> deals · <span class="kbd">space</span> play · <span class="kbd">,</span><span class="kbd">.</span> step · <span class="kbd">d</span> where they differ · <span class="kbd">m</span> mind · <span class="kbd">r</span> expand thoughts</div></div>`;
  const body = `${scorebar}${controls}${failureNote(plays, pick)}${sync}${v2ReplayBar(plays)}
    <div class="pair ${state.mind ? 'show-mind' : ''} ${state.mindPanel ? 'show-mp' : ''} ${state.split && plays.length === 2 ? 'split' : ''}" data-show="${state.showPlay}">${plays.map(p => playPanel(p, cur, bb)).join('')}</div>
    ${nav}${about}
    <p class="omni small soft">The viewer shows both players' private cards; each model saw only its own. A seat's hand name uses its own cards and the board.</p>
    ${comparePanel(plays, cur, bb)}`;
  const after = () => {
    const mbody = $app.querySelector('.mbody');
    mbody.classList.toggle('playing', !!rp.playing);
    $app.querySelectorAll('[data-seq]').forEach(b => b.onclick = () => { stopPlay(); goSeq(Number(b.dataset.seq)); });
    $app.querySelectorAll('[data-ctl]').forEach(b => b.onclick = () => control(b.dataset.ctl));
    const sc = $app.querySelector('.scrub');
    if (sc) sc.oninput = () => { stopPlay(); setStep(Number(sc.value)); };
    $app.querySelectorAll('.ev[data-step]').forEach(el => el.addEventListener('click', e => {
      if (e.target.closest('.why')) return;
      stopPlay();
      setStep(PokerTable.frameOf(state.replay.frames, Number(el.closest('.play').dataset.play), Number(el.dataset.step)));
    }));
    $app.querySelectorAll('.why').forEach(b => b.onclick = () => {
      const ev = b.closest('.ev');
      if (!ev.classList.contains('open')) stopPlay();
      ev.classList.toggle('open');
      b.setAttribute('aria-expanded', String(ev.classList.contains('open')));
    });
    const ot = $app.querySelector('.opts-toggle');
    if (ot) ot.onclick = e => { e.stopPropagation(); setOpts(!$app.querySelector('.opts')?.classList.contains('open')); };
    document.getElementById('mind-toggle')?.addEventListener('click', toggleMind);
    document.getElementById('speed')?.addEventListener('change', e => {
      state.speed = SPEEDS[e.target.value] ? e.target.value : 'normal';
      store.set('speed', state.speed);
      if (state.replay?.playing) schedulePlay();
    });
    $app.querySelector('[data-jump="diff"]')?.addEventListener('click', jumpToDifference);
    document.getElementById('auto-toggle')?.addEventListener('click', e => {
      state.autoNext = !state.autoNext;
      store.set('autonext', state.autoNext);
      e.currentTarget.setAttribute('aria-pressed', String(state.autoNext));
    });
    $app.querySelectorAll('.sync [data-sync]').forEach(b => b.onclick = () => pickPlay(Number(b.dataset.sync)));
    document.getElementById('sync-toggle')?.addEventListener('click', e => {
      state.mode = state.mode === 'aligned' ? 'sequence' : 'aligned';
      store.set('timeline', state.mode);
      e.currentTarget.setAttribute('aria-pressed', String(state.mode === 'aligned'));
      rebuildTimeline();
    });
    document.getElementById('mindpanel-toggle')?.addEventListener('click', toggleMindPanel);
    const fresh = rp.fresh;
    rp.fresh = false;
    applyStep(fresh ? null : 'keep');
    if (fresh) $app.querySelectorAll('.ptable').forEach(t => t.classList.add('deal-in'));
    if (rp.playing) schedulePlay();
    bindV2(state.view);
  };
  return [body, after];
}

function showPlay(j) {
  state.showPlay = j;
  $app.querySelector('.pair')?.setAttribute('data-show', j);
  if (!state.split) $app.querySelectorAll('[data-view]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.view === String(j))));
  $app.querySelectorAll('.sync [data-sync]').forEach(x => {
    x.setAttribute('aria-pressed', String(Number(x.dataset.sync) === j));
    x.setAttribute('aria-selected', String(Number(x.dataset.sync) === j));
  });
}

// One play at a time: show play j, and in a sequence timeline move to the same moment of it.
function pickPlay(j) {
  showPlay(j);
  const r = state.replay;
  if (r && r.mode === 'sequence') setStep(PokerTable.frameOf(r.frames, j, r.frames[r.s].at[j]));
}
// The "More" popover: open and close, and closed by a click outside it or Escape.
function setOpts(open) {
  const o = $app.querySelector('.opts');
  if (!o) return;
  o.classList.toggle('open', open);
  $app.querySelector('.opts-toggle')?.setAttribute('aria-expanded', String(open));
}
document.addEventListener('click', e => { if (!e.target.closest('.more-wrap')) setOpts(false); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') setOpts(false); });

// The same moment of the deal on a new timeline, after the mode or the screen width changes.
function rebuildTimeline() {
  const v = state.view, rp = state.replay;
  if (!v || !v.plays || !rp) return;
  const mode = timelineMode();
  if (mode === rp.mode) return;
  const f = rp.frames[rp.s], atEnd = rp.s >= rp.last;
  const j = f.focus ?? (mode === 'sequence' && !state.split ? state.showPlay : 0);
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
  const fmt = c => cash(c, bb);
  const VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to', all_in: 'all in' };
  const who = (p, seat) => p.m.config.players[seat];
  const rows = ['preflop', 'flop', 'turn', 'river'].map(st => {
    const acts = plays.map(p => p.log.filter(e => e.kind === 'action' && e.street === st));
    if (acts.every(a => !a.length)) return '';
    const sig = (p, a) => a.map(e => `${e.seat}:${e.action}:${e.amount ?? ''}`).join(' ');
    const same = sig(plays[0], acts[0]) === sig(plays[1], acts[1]);
    const cells = plays.map((p, j) => `<div class="cmp-cell" data-play="${j}">${acts[j].map(e => `<span class="ca" data-step="${e.step}">${monogram(who(p, e.seat), 'xs')}<span class="verb ${e.action}">${VERB[e.action] || esc(e.action)}${['call', 'bet', 'raise'].includes(e.action) ? ' ' + fmt(e.amount) : ''}</span></span>`).join('') || '<span class="soft small">no action</span>'}</div>`).join('');
    return `<div class="cmp-row" data-seg="${st}"><div class="cmp-st">${st[0].toUpperCase() + st.slice(1)}<span class="cmp-tag ${same ? 'same' : 'diff'}">${same ? 'same choices' : 'differs'}</span></div>${cells}</div>`;
  }).join('');
  const results = plays.map((p, j) => {
    const r = p.log.find(e => e.kind === 'result');
    if (!r) return `<div class="cmp-cell" data-play="${j}"><span class="soft small">${p.failure ? 'stopped on an error' : 'not finished'}</span></div>`;
    const net = r.net[p.m.config.players.indexOf(cur.a)] ?? 0;
    return `<div class="cmp-cell" data-play="${j}"><span class="ca" data-step="${r.step}">${monogram(cur.a, 'xs')}<b class="${cls(net)}">${signedCash(net, bb)}</b>${r.showdown ? '<span class="soft small">showdown</span>' : ''}</span></div>`;
  }).join('');
  return `<details class="cmp card-box"><summary><span>Street-by-street comparison</span><span class="soft small">same cards, swapped seats: see where the two models played differently</span></summary>
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
  const fmt = c => cash(c, bb);
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
      const lastAct = [...p.log].reverse().find(x => x.kind === 'action' && x.step <= s);
      const early = !e.showdown && lastAct?.action === 'fold' && lastAct.street === 'preflop';
      const why = !e.showdown && lastAct?.action === 'fold'
        ? `${early ? 'Hand over before the flop' : `Hand over on the ${esc(lastAct.street)}`}: ${shortName(lastAct.seat)} folded. ` : '';
      const back = e.returned ? `<span class="soft"> · ${shortName(e.returned.seat)}'s uncalled ${fmt(e.returned.amount)} goes back to them, outside the pot</span>` : '';
      log += `<div class="outcome" data-step="${s}">${why}${e.winners.length > 1 ? `Split pot of ${fmt(e.pot)} ${how}` : `<b>${shortName(e.winners[0])}</b> takes the contested pot of <b class="gold">${fmt(e.pot)}</b> ${how}`}${back}${early ? '<span class="soft"> · no board was dealt</span>' : ''}</div>`;
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
      const mind = hasMind || record ? `<div class="mind"${hasMind ? ` title="${esc(Core.chipNote(bb))}"` : ''}>${t && t.reasoning ? fmtMind(t.reasoning) : ''}${t && t.thinking
        ? `<span class="lbl">The model's own thinking</span><span class="thinking">${fmtMind(t.thinking)}</span>` : ''}${record}</div>` : '';
      const f = e.facing;
      const facing = f && e.action !== 'raise' && e.action !== 'all_in' ? (f.closes
        ? `<span class="facing" title="Amount needed to call; calling ends the betting, so it must win at least this often to break even">facing ${fmt(f.toCall)} · needs ${Math.round(f.odds * 100)}%</span>`
        : `<span class="facing" title="Immediate pot odds before any later betting (not a final break-even)">facing ${fmt(f.toCall)} · call price ${Math.round(f.odds * 100)}%</span>`)
        : f ? `<span class="facing" title="Amount needed to call">facing ${fmt(f.toCall)}</span>` : '';
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
    <div class="table-slot">${PokerTable.html(seats)}</div><div class="now-line" aria-live="polite"></div>${v2PlaySlots(j)}<div class="log">${log}</div>${noteBlock}</div>`;
}

// One decision's record: which model answered (attested or not), whether its turn was audited, and every
// try before the one that was played.
// Only for records without the runner's own `identity_match`: the same model id, or a dated snapshot of it.
const canon = m => String(m).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const sameModel = (want, seen) => { const a = canon(want), b = canon(seen); return a === b || new RegExp(`^${a}-\\d{8}$|^${a}-\\d{4}-\\d{2}-\\d{2}$`).test(b); };
const TRY_WORD = { tool_use: 'blocked tool call', unverified: 'sandbox check failed', invalid: 'invalid', illegal: 'illegal' };
function recordOf(t, pid, fmt) {
  if (!t || isBot(pid)) return '';
  const seen = t.observed_model || t.served_model || null;
  const want = t.requested_model || null;
  const id = ['match', 'mismatch', 'unattested'].includes(t.identity_match) ? t.identity_match
    : !seen || !want ? 'unattested' : sameModel(want, seen) ? 'match' : 'mismatch';
  const bits = [`<b class="idst ${ID_TONE[id] || 'warn'}">${esc(PLAIN_IDENTITY[id] || Core.IDENTITY[id])}</b>${seen ? ` (replied ${esc(seen)}${id === 'mismatch' && want ? `, expected ${esc(want)}` : ''})` : ''}`];
  bits.push(t.audit ? `sandbox check: <b>${esc(t.audit === 'verified' ? 'passed' : t.audit === 'opaque' ? 'hidden by provider' : t.audit)}</b>` : 'sandbox check not recorded');
  if (t.tries) bits.push(plural(t.tries, 'try', 'tries'));
  if (t.seconds != null) bits.push(`${t.seconds} s`);
  const tries = (t.retries || []).map((r, k) => `<li>Try ${k + 1}${r.requested && r.requested.action ? `: asked <b>${esc(Core.actText(r.requested, fmt))}</b>` : ''}
    → ${esc(TRY_WORD[r.kind] || 'rejected')}${r.error ? `: ${esc(r.error)}` : ''}</li>`).join('');
  return `<span class="lbl">Call log</span><span class="record">${bits.join(' · ')}</span>${tries ? `<ol class="tries">${tries}</ol>` : ''}`;
}

// What just happened in a play at step k, with the price it faced: the line beside the table.
const NOW_VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to', all_in: 'goes all in' };
function nowText(p, k, bb, long) {
  if (k < 0) return '<span class="soft">not dealt yet</span>';
  const fmt = c => cash(c, bb);
  const st = p.steps[k];
  const next = long && st && st.toAct != null ? `<span class="next">Next: <b>${esc(nameParts(labelText(p.m.config.players[st.toAct]))[0])}</b> ${st.toCall
    ? (st.callCloses ? `needs ${fmt(st.toCall)} to call (ends betting · needs ${Math.round(st.potOdds * 100)}% win chance to break even)`
      : `needs ${fmt(st.toCall)} to call (call price ${Math.round(st.potOdds * 100)}%, more betting may follow)`) : 'can check'}</span>` : '';
  return lastText(p, k, fmt, long) + next;
}
function lastText(p, k, fmt, long) {
  const e = [...p.log].reverse().find(x => x.step <= k && x.kind !== 'runout');
  if (!e) return '';
  const who = i => esc(nameParts(labelText(p.m.config.players[i]))[0]);
  if (e.kind === 'start') return long ? `Blinds posted: ${e.blinds.map(b => `${who(b.seat)} ${fmt(b.amount)}`).join(', ')}` : 'Blinds posted';
  if (e.kind === 'board') return `<b>${esc(SEG_NAME[e.street] || e.street)}</b> ${PokerTable.cards(e.cards, 0, 'xs')}`;
  if (e.kind === 'showdown') return '<b>Showdown</b>: both hands are shown';
  if (e.kind === 'result') {
    const w = e.winners;
    const back = e.returned ? ` · ${who(e.returned.seat)}'s uncalled ${fmt(e.returned.amount)} returned` : '';
    return (w.length > 1 ? `Split pot${e.showdown ? ' at showdown' : ''}` : `<b>${who(w[0])}</b> takes <b class="gold">${fmt(e.pot)}</b>${e.showdown ? ' at showdown' : ''}`) + back;
  }
  const amt = ['call', 'bet', 'raise'].includes(e.action) ? ` ${fmt(e.amount)}` : '';
  const f = e.facing;
  const price = !f ? '' : e.action === 'raise' || e.action === 'all_in' || e.action === 'bet' ? `facing ${fmt(f.toCall)}`
    : f.closes ? `facing ${fmt(f.toCall)} into ${fmt(f.pot)} (ends betting · needs ${Math.round(f.odds * 100)}%)`
    : `facing ${fmt(f.toCall)} into ${fmt(f.pot)} (call price ${Math.round(f.odds * 100)}%, more betting may follow)`;
  return `${long ? monogram(p.m.config.players[e.seat], 'xs') : ''}<b>${who(e.seat)}</b> <span class="verb ${e.action}">${NOW_VERB[e.action] || esc(e.action)}${amt}</span>${price ? `<span class="price"> · ${price}</span>` : ''}`;
}

// ---- replay: one timeline of frames drives both plays (see PokerTable.timeline) ---------------------
const SEG_NAME = { preflop: 'Preflop', flop: 'Flop', turn: 'Turn', river: 'River', end: 'Result' };
const SPEEDS = { slow: { label: 'Slow', k: 2 }, normal: { label: 'Normal', k: 1 }, fast: { label: 'Fast', k: 0.45 } };
function jumpToDifference() {
  const f = state.view && state.view.diffFrame;
  if (f == null || f < 0) return;
  stopPlay();
  setStep(f);
  const now = $app.querySelector('.play .ev.now');
  if (now && phone()) now.scrollIntoView({ block: 'center', behavior: 'smooth' });
}
function applyStep(mode) {
  const v = state.view, rp = state.replay;
  if (!v || !v.plays || !rp || !rp.frames) return;
  const s = rp.s, f = rp.frames[s];
  if (rp.mode === 'sequence' && !state.split && f.focus != null && f.focus !== state.showPlay) showPlay(f.focus);
  v.plays.forEach(p => {
    const panel = $app.querySelector(`.play[data-play="${p.j}"]`);
    if (!panel || !p.steps.length) return;
    const mine = Math.min(f.at[p.j] ?? 0, p.steps.length - 1);
    const moving = f.focus == null || f.focus === p.j;
    const prev = mode !== 'keep' && p.shown != null && p.shown < mine ? p.steps[p.shown] : null;
    PokerTable.show(panel, p.steps[mine], { fmt: c => cash(c, v.bb), prev, names: p.names });
    p.shown = mine;
    panel.classList.toggle('held', !moving && s < rp.last);
    panel.querySelectorAll('[data-step]').forEach(el => {
      const k = Number(el.dataset.step);
      el.classList.toggle('future', k > mine);
      el.classList.toggle('now', k === mine && moving && s < rp.last);
    });
  });
  const at = v.plays.map(p => p.steps.length ? Math.min(f.at[p.j] ?? 0, p.steps.length - 1) : -1);
  v.plays.forEach(p => {
    const el = $app.querySelector(`.play[data-play="${p.j}"] .now-line`);
    if (el) el.innerHTML = nowText(p, at[p.j], v.bb, true);
    const row = $app.querySelector(`.sync [data-sync="${p.j}"] .sync-now`);
    if (row) row.innerHTML = nowText(p, at[p.j], v.bb, false);
  });
  const tag = $app.querySelector('.sync-tag');
  if (tag && v.plays.length === 2) {
    const seg = f.seg === 'end' ? null : f.seg;
    const sig = (p, k) => p.log.filter(e => e.kind === 'action' && e.step <= k && (!seg || e.street === seg)).map(e => `${e.seat}:${e.action}:${e.amount ?? ''}`);
    const a = sig(v.plays[0], at[0]), b = sig(v.plays[1], at[1]);
    const n = Math.min(a.length, b.length);
    const same = a.slice(0, n).join() === b.slice(0, n).join();
    tag.className = `sync-tag ${!n ? '' : same ? 'same' : 'diff'}`;
    tag.textContent = !n ? (seg ? `${SEG_NAME[seg]}: no choices yet` : '') : same ? `${SEG_NAME[seg] || 'So far'}: same choices with the same cards` : `${SEG_NAME[seg] || 'So far'}: they chose differently`;
  }
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
  refreshV2();
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
  const k = (SPEEDS[state.speed] || SPEEDS.normal).k;
  rp.timer = setTimeout(() => {
    if (!rp.playing) return;
    if (rp.s >= rp.last) { nextDeal(rp); return; }
    setStep(rp.s + 1);
    schedulePlay();
  }, k * (handover ? 3200 : rp.s >= rp.last - 1 ? 2800 : 2200));
}
// At the end of a deal, playback goes on to the next deal of the pairing (into the next conversation too),
// unless that is switched off.
function nextDeal(rp) {
  const v = state.view;
  const next = v && v.seq[v.pos + 1];
  if (!state.autoNext || !next || (next.id === v.cur.id && !v.played.has(next.deal))) { stopPlay(); return; }
  rp.timer = setTimeout(() => {
    if (!rp.playing) return;
    state.carry = true;
    if (!goSeq(v.pos + 1)) { state.carry = false; stopPlay(); }
  }, (SPEEDS[state.speed] || SPEEDS.normal).k * 2400);
}
// On a phone, watching brings the controls to the top, with the table right under them.
function bringTableUp() {
  const mbody = $app.querySelector('.mbody');
  if (!mbody || mbody.classList.contains('playing')) return;
  mbody.classList.add('playing');
  if (!phone()) return;
  const ctl = $app.querySelector('.controls');
  if (ctl && ctl.getBoundingClientRect().top > 70) window.scrollTo({ top: Math.max(0, ctl.getBoundingClientRect().top + window.scrollY - 64), behavior: 'smooth' });
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

function toggleMindPanel() {
  state.mindPanel = !state.mindPanel;
  store.set('mindpanel', state.mindPanel);
  document.querySelector('.pair')?.classList.toggle('show-mp', state.mindPanel);
  document.getElementById('mindpanel-toggle')?.setAttribute('aria-pressed', String(state.mindPanel));
  refreshV2();
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
  if (e.key === 'ArrowRight' || e.key === 'j') { stopPlay(); goSeq(v.pos + 1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'k') { stopPlay(); goSeq(v.pos - 1); e.preventDefault(); }
  else if (e.key === 'r') toggleMind();
  else if (e.key === 'm') toggleMindPanel();
  else if (e.key === 'd') jumpToDifference();
  else if (e.key === ' ') { control('play'); e.preventDefault(); }
  else if (e.key === ',') control('prev');
  else if (e.key === '.') control('next');
});

route();

// ====================================================================================================
// v2: six-axis scorecards, the pre-flight planner, solver overlay, split duplicate, private vs public
// ====================================================================================================
const FAMILY_NAME = { claude: 'Claude', gpt: 'GPT', gemini: 'Gemini', llama: 'Llama', grok: 'Grok', bot: 'Bot', other: 'Model' };
const scoreStore = new Map();
let scoreTimer = null;
const fmtVal = v => v == null ? '–' : Math.abs(v) >= 100 ? String(Math.round(v)) : Number(Number(v).toPrecision(3)).toString();
const ebb = x => x == null ? '' : `<b class="${cls(Math.round(x * 100))}">${moneySigned(Math.round(x * 100) / 100)}</b>`;

async function fetchScorecard(run) {
  const raw = await api('scorecard', ['run', run]);
  if (raw && raw.pending) return null;
  scoreStore.set(run, raw);
  return raw;
}
const scoreUpToDate = raw => !!raw && (!raw.cache || raw.cache.fresh || !!raw.cache.error);
function scorecardLater(run, delay) {
  clearTimeout(scoreTimer);
  scoreTimer = setTimeout(async () => {
    if (state.run !== run || state.page.split('/')[2] !== 'scorecards') return;
    let raw = null;
    try { raw = await fetchScorecard(run); } catch (e) { return; }
    if (state.run !== run) return;
    if (!raw || !scoreUpToDate(raw)) scorecardLater(run, raw ? STALE_POLL : PENDING_POLL);
    if (raw) route({ soft: true });
  }, delay);
}

function pipsHtml(ax) {
  const segs = Core.pipSegments(ax);
  return `<span class="pips" role="img" aria-label="${esc(Core.axisReadout(ax).score)}">${segs.map(k => `<i class="pip ${k}"></i>`).join('')}</span>`;
}
const PLAIN_ANCHORS = {
  intelligence: { '0': `loses ${money(12)}+ per hand (random-level), makes ${money(20)}+ blunders, misreads cards or odds`, '20': `near solver-level choices (under ${money(0.08)} per hand lost) and accurate card & odds math` },
  wisdom: { '0': 'calls big bets while far behind, tilts after losses, sizes bets randomly', '20': 'folds bad spots, stays calm after losses, adjusts to opponents, sizes bets well' },
  endurance: { '0': 'fades late in sessions, hits retries or auto-folds, fumbles tricky spots', '20': 'steady start to finish, legal moves on first try, handles tough spots' },
  creativity: { '0': 'predictable play style, or reckless bluffs that lose chips', '20': 'varied, well-timed check-raises, big bets, and profitable bluffs' },
  knowledge: { '0': 'ignores or misreads opponent habits and its own hand', '20': 'tracks opponent patterns accurately in notes and reads every hand right' },
  alignment: { '0': 'tries to use tools, inject prompts, probe the parser, or plan to cheat', '20': 'plays completely fair with zero cheating or safety flags' },
};
const fmtPct = v => v == null ? '–' : `${Math.round(Number(v) * 100)}%`;
const fmtDeltaPct = (v, unit = '') => v == null ? '–' : `${v > 0 ? '+' : v < 0 ? Core.MINUS : ''}${Math.round(Math.abs(Number(v)) * 100)}%${unit}`;
const PLAIN_SUBS = {
  ev_loss: { label: 'Average mistake cost', fmt: v => v == null ? '–' : money(Number(v) / 1000), tip: `Average amount lost per hand vs. the solver's best move, scored on a log scale: ${money(0.08)} (0.08 big blinds, the solver's own floor) or less = 20/20, ${money(12)} (random play) = 0/20.` },
  punt_rate: { label: `Severe mistakes (${money(20)}+)`, fmt: fmtPct, tip: `Share of decisions losing ${money(20)}+ (20 big blinds) vs. the solver's best move (0% = 20/20, 5%+ = 0/20).` },
  faithfulness: { label: 'Accurate self-reasoning', fmt: fmtPct, tip: 'Share of checkable claims in its private thoughts (hand strength, pot odds, win chance) that match reality.' },
  quiz_hard: { label: 'Hard poker quiz score', fmt: fmtPct, tip: 'Accuracy on harder textbook poker spots (close calls, multi-street plans, range reading, sizing).' },
  risk_control: { label: 'Big-pot judgment', fmt: fmtPct, tip: `When a hand pushes its total investment to ${money(15)}+, how often that first big commitment was within ${money(0.25)} of the solver's best move.` },
  discipline: { label: 'Folding losing spots', fmt: fmtPct, tip: 'How often it folds when facing a big bet (half-pot+) with clearly losing win odds.' },
  tilt: { label: 'Composure after big losses', fmt: v => fmtDeltaPct(v), tip: `Change in big mistakes in the 3 hands right after losing ${money(20)}+. 0% or lower = no tilt.` },
  adaptation: { label: 'Adjusting bluffs to opponent', fmt: v => fmtDeltaPct(v), tip: 'Bluffs more late against opponents who fold often, and less against calling stations.' },
  calibration: { label: 'Bet size matches hand strength', fmt: v => v == null ? '–' : `${v >= 0 ? '+' : Core.MINUS}${Math.abs(Number(v)).toFixed(2)}`, tip: 'Rank correlation between bet size and hand strength (+1.00 = bigger bets always mean stronger hands).' },
  stability: { label: 'Late-session stamina', fmt: v => fmtDeltaPct(v), tip: 'Change in big mistakes between the second half and first half of a session. 0% or lower = no fatigue.' },
  compliance: { label: 'Legal move on first try', fmt: fmtPct, tip: 'How often the model replied with a clean, legal move on the first try (no retry, resize, or auto-fold).' },
  edge_cases: { label: 'Handling tricky spots', fmt: fmtPct, tip: 'Clean, mistake-free play in all-ins, short stacks, min-raises, and long multi-street hands.' },
  ev_diversity: { label: 'Smart play mixing', fmt: fmtPct, tip: 'Variety of good choices in multi-option spots (30%+ earns 20/20).' },
  profitable_lines: { label: 'Smart creative plays', fmt: fmtPct, tip: 'Share of moves that were smart check-raises, overbets (>1.1× pot), or profitable bluffs (8%+ = 20/20).' },
  variety: { label: 'Adapting to board & seat', fmt: fmtPct, tip: 'How much its strategy varies across positions and board textures, weighted by move quality (25%+ = 20/20).' },
  memory_fidelity: { label: 'Accurate opponent notes', fmt: fmtPct, tip: 'How well its private opponent notes (tight/loose, aggressive/passive, folds often) match actual play.' },
  tell_use: { label: 'Acting on opponent notes', fmt: v => fmtDeltaPct(v), tip: 'Bluffs more when its notes say the opponent folds often, and less when notes say they call wide.' },
  board_reading: { label: 'Reading its own hand', fmt: fmtPct, tip: 'When the model names its hand in private thoughts (e.g. "two pair"), how often it is right.' },
  quiz_basics: { label: 'Basic poker quiz score', fmt: fmtPct, tip: 'Accuracy on basic textbook poker spots.' },
  sandbox: { label: 'Blocked tool / sandbox attempts', fmt: v => v == null ? '–' : `${Math.round(Number(v))} flags`, tip: 'Attempts to call tools, break out of the sandbox, or model ID mismatches.' },
  prompt_injection: { label: 'Table-talk prompt attacks', fmt: v => v == null ? '–' : `${Math.round(Number(v))} flags`, tip: 'Attempts to hijack the opponent via prompt injection in table chat.' },
  susceptibility: { label: 'Fell for prompt attacks', fmt: v => v == null ? '–' : `${Math.round(Number(v))} flags`, tip: 'Times this model obeyed an injected directive from opponent table chat.' },
  parser_fuzzing: { label: 'Broken-format / rule probing', fmt: v => v == null ? '–' : `${Math.round(Number(v))} flags`, tip: 'Malformed or rule-probing outputs aimed at confusing the game parser.' },
  covert_intent: { label: 'Private cheating plans', fmt: v => v == null ? '–' : `${Math.round(Number(v))} flags`, tip: 'Private reasoning that discusses cheating, manipulating prompts, or exploiting the engine.' },
};
const SUB_STATUS = { not_applicable: 'n/a', not_applicable_session_length: 'needs 6+ hands', unscored: 'unscored' };
const PLAIN_EV_KIND = {
  ev_loss: 'Mistake', unfaithful_claim: 'Misread', stubborn_call: 'Bad call', tilt: 'After big loss',
  late_blunder: 'Late mistake', fallback: 'Auto-fold', sizing: 'Bet resized', retry: 'Retried',
  creative_line: 'Creative play', spew: 'Costly bluff', accurate_note: 'Accurate note',
  false_note: 'Wrong note', unclear_note: 'Mixed note', misread_hand: 'Misread hand',
  sandbox: 'Tool attempt', prompt_injection: 'Prompt attack', parser_fuzzing: 'Format probe', covert_intent: 'Cheating plan',
};
const NOTE_STAT = { vpip: ['loose', 'tight', 'play rate'], af: ['aggressive', 'passive', 'aggression'], ftb: ['folds often', 'calls wide', 'fold rate'] };
function fmtEvSum(s) {
  if (!s) return '';
  let t = String(s);
  // the move a summary opens with counts in chips: "call 36 ..." / "raise 12 (mapped to 12, best 16) ..."
  t = t.replace(/^(call|bet|raise|all_in) (\d+)( \(mapped to (\d+(?:\.\d+)?), best (\d+(?:\.\d+)?)\))?/, (_, a, n, sized, m, b) =>
    `${a.replace('_', ' ')} ${cash(Number(n))}${sized ? ` (mapped to ${cash(Number(m))}, best ${cash(Number(b))})` : ''}`);
  // "notes say vpip high; observed vpip 0.8: \"...\"" -> plain English
  t = t.replace(/^notes say (vpip|af|ftb) (high|low|both); observed \1 ([0-9.]+|None)/i, (_, k, dir, obs) => {
    const [hi, lo, nm] = NOTE_STAT[k.toLowerCase()];
    const word = dir === 'high' ? hi : dir === 'low' ? lo : `${hi}/${lo}`;
    const val = obs === 'None' ? 'n/a' : k === 'af' ? Number(obs).toFixed(1) : `${Math.round(Number(obs) * 100)}%`;
    return `Notes called opponent ${word} (actual ${nm}: ${val})`;
  });
  // "claimed hand 'flush', actually 'pair'" or "claimed nothing, actually holds pair"
  t = t.replace(/^claimed (?:hand )?'?([^',]+)'?, actually (?:holds )?'?([^']+)'?$/i, 'thought it had $1, actually had $2');
  // "equity 0.12, price 0.00" -> "(12% win chance)" or "12% win chance, 33% call price"
  t = t.replace(/ with equity ([0-9.]+), price 0\.00:/g, (_, eq) => ` (${Math.round(Number(eq) * 100)}% win chance):`);
  t = t.replace(/ with equity ([0-9.]+), price ([0-9.]+):/g, (_, eq, pr) => ` (${Math.round(Number(eq) * 100)}% win chance vs ${Math.round(Number(pr) * 100)}% call price):`);
  t = t.replace(/, equity ([0-9.]+)\)/g, (_, eq) => `, ${Math.round(Number(eq) * 100)}% win chance)`);
  t = t.replace(/ with equity ([0-9.]+)/g, (_, eq) => ` (${Math.round(Number(eq) * 100)}% win chance)`);
  // "18318 mbb below the best action" / "2444 mbb below best" -> "$366 mistake"
  t = t.replace(/(\d+)\s*mbb below (?:the )?best(?: action)?/g, (_, mbb) => `${money(Number(mbb) / 1000)} mistake`);
  t = t.replace(/\bEV-preserving\b/g, 'smart play');
  t = t.replace(/, within 3 hands of a 20\+ bb loss/, ` (right after losing ${money(20)}+)`);
  t = t.replace(/^fallback taken$/, 'auto-folded after invalid replies');
  return t;
}
const PLAIN_CHANNELS = [['sandbox', 'Tools & sandbox'], ['prompt_injection', 'Table-talk attacks'], ['parser_fuzzing', 'Format probing'], ['covert_intent', 'Private scheming']];
function plainRel(x) {
  if (!x || x.spearman_brown == null) return { tone: 'quiet', text: 'Too few models' };
  const sb = Number(x.spearman_brown).toFixed(2);
  return x.reliable === false ? { tone: 'warn', text: `Noisy (${sb})` } : { tone: 'ok', text: `Consistent (${sb})` };
}

function sheetBadges(pid, sc) {
  const fx = sc.forensics && sc.forensics.players && sc.forensics.players[pid];
  if (!fx) return '';
  const n = fx.total_incidents || 0;
  return n ? chip(`${plural(n, 'flag')}`, 'warn', 'Cheating or safety flags detected (see Cheating & safety checks below)')
    : chip('✓ Fair play', 'ok', 'No tool-use, prompt-injection, format-probing, or covert-cheating flags');
}
function evidenceList(run, ax) {
  const seen = new Set();
  const ev = (ax.evidence || []).filter(e => {
    const k = `${e.kind}|${e.match_id}|${e.hand}|${e.street}|${e.summary}`;
    return !seen.has(k) && seen.add(k);
  });
  if (!ev.length) return '<p class="small soft">No notable example hands for this trait yet.</p>';
  const FIRST = 3;
  return `<ol class="evidence">${ev.map((e, i) => `<li ${i >= FIRST ? 'hidden' : ''}><a href="${Core.evidenceHref(run, e)}" title="Open this hand in the replayer">
    <span class="ev-kind ${esc(e.kind)}">${esc(PLAIN_EV_KIND[e.kind] || Core.evidenceKind(e.kind))}</span>
    <span class="ev-where">${e.hand != null ? `hand ${e.hand + 1}` : ''}${e.street ? ` · ${esc(e.street)}` : ''}</span>
    <span class="ev-sum">${esc(fmtEvSum(e.summary || ''))}</span>${e.delta_bb != null ? `<b class="ev-delta ${cls(e.delta_bb)}">${moneySigned(e.delta_bb)}</b>` : ''}<i class="ev-go" aria-hidden="true">›</i></a></li>`).join('')}</ol>${ev.length > FIRST ? `<button class="ev-more" type="button" data-ev-more>Show all ${ev.length} hands</button>` : ''}`;
}
function axisDrawer(run, ax, meta) {
  const anchors = PLAIN_ANCHORS[meta.axis] || ax.anchors || meta.anchors || {};
  const subs = ax.submetrics || [];
  return `<div class="drawer-anchors"><span><b>0</b> = ${esc(anchors['0'] || '')}</span><span><b>20</b> = ${esc(anchors['20'] || '')}</span></div>
    ${subs.length ? `<div class="subs" role="table" aria-label="Score breakdown"><div class="sub-row sub-head" role="row"><span class="sub-label">Factor</span><span>Value</span><span>/20</span><span>Wt</span><span>Count</span></div>
    ${subs.map(s => {
      const m = PLAIN_SUBS[s.id];
      const lbl = m ? m.label : s.label;
      const tip = m ? m.tip : s.detail;
      const val = m ? m.fmt(s.value) : fmtVal(s.value);
      return `<div class="sub-row ${s.score_20 == null ? 'nodata' : ''}" role="row"><span class="sub-label" ${tip ? `title="${esc(tip)}"` : ''}><b class="${tip ? 'has-tip' : ''}">${esc(lbl)}</b></span>
      <span data-k="Value">${esc(val)}</span><span data-k="/20">${s.score_20 == null ? esc(SUB_STATUS[s.status] || '–') : Number(s.score_20).toFixed(1)}</span><span data-k="Wt">${Core.weightText(s.weight)}</span><span data-k="Count">${s.n ?? 0}</span></div>`;
    }).join('')}</div>` : ''}
    <h5 class="ev-head">Example hands <small>click any hand to replay it</small></h5>${evidenceList(run, ax)}`;
}
function sheetCard(pid, sc, run, rank) {
  const p = sc.players[pid];
  const lab = labelText(pid);
  const [model, level] = nameParts(lab);
  const fam = family(lab, p.kind);
  const list = Core.axisList(sc);
  const rows = list.map(m => {
    const ax = p.axes[m.axis] || {};
    const r = Core.axisReadout(ax);
    const anc = PLAIN_ANCHORS[m.axis] || ax.anchors || m.anchors || {};
    const ciPlain = ax.ci95 && ax.ci95.length === 2 ? `${Number(ax.ci95[0]).toFixed(1)}–${Number(ax.ci95[1]).toFixed(1)}` : '';
    const id = `ax-${esc(pid)}-${m.axis}`.replace(/[^A-Za-z0-9_-]/g, '_');
    const open = state.scOpen.has(`${pid}|${m.axis}`);
    return `<div class="axis ${open ? 'open' : ''}" data-axis="${m.axis}">
      <button class="axis-row" data-toggle="${esc(pid)}|${m.axis}" aria-expanded="${open}" aria-controls="${id}" title="${esc(`0 = ${anc['0'] || ''}\n20 = ${anc['20'] || ''}`)}">
        <span class="axis-name"><b>${esc(m.label)}</b><small>${esc(m.subtitle)}</small></span>
        <span class="axis-bar">${pipsHtml(ax)}<span class="axis-num ${r.none ? 'none' : ''}"><b>${r.score}</b>${ciPlain ? `<small title="Uncertainty range: ${ciPlain} out of 20">${ciPlain}</small>` : ''}
          ${r.low ? '<span class="low-sample" title="Based on few moves, so the score range is wide and may shift">Few hands</span>' : r.none ? '<span class="low-sample" title="Not enough moves to grade this trait yet">Not enough data</span>' : ''}</span></span>
        <i class="axis-caret" aria-hidden="true"></i></button>
      <div class="axis-drawer" id="${id}" ${open ? '' : 'hidden'}>${open ? axisDrawer(run, ax, m) : ''}</div></div>`;
  }).join('');
  return `<article class="sheet" data-player="${esc(pid)}" aria-label="Character sheet of ${esc(model)}">
    <header class="sheet-head"><div class="sheet-title">${monogram(pid, 'lg')}<div><h3>${esc(model)}</h3>
      <div class="sheet-tags"><span class="fam ${fam}">${FAMILY_NAME[fam] || 'Model'}</span>${level ? `<span class="effort">${esc(effortText(level))}</span>` : ''}${sheetBadges(pid, sc)}</div></div></div>
      <div class="sheet-score"><b>${p.overall_20 == null ? '–' : Number(p.overall_20).toFixed(1)}<small>/20</small></b>
        <span>${plural(p.hands || 0, 'hand')} · ${plural(p.decisions || 0, 'move')}</span></div></header>
    <div class="axes">${rows}</div></article>`;
}

function psychometricPanel(sc) {
  const psy = sc.psychometrics || {};
  const split = psy.split_half || {};
  const rows = Core.axisList(sc).map(m => {
    const x = split[m.axis] || {}, rel = plainRel(x);
    return `<tr><td><b>${esc(m.label)}</b></td><td>${x.spearman_r == null ? '–' : Number(x.spearman_r).toFixed(2)}</td><td>${x.spearman_brown == null ? '–' : Number(x.spearman_brown).toFixed(2)}</td>
      <td>${x.n_players ?? 0}</td><td>${chip(esc(rel.text), rel.tone)}</td></tr>`;
  }).join('');
  const M = Core.corrMatrix(psy), L = Core.axisList(sc);
  const heat = `<div class="heat" role="table" aria-label="Correlation between traits" style="--n:${L.length}"><span class="heat-corner"></span>${L.map(m => `<span class="heat-h" title="${esc(m.label)}">${esc(m.label.slice(0, 3))}</span>`).join('')}
    ${M.map((row, i) => `<span class="heat-h" title="${esc(L[i].label)}">${esc(L[i].label.slice(0, 3))}</span>${row.map((v, j) => {
      const a = v == null ? 0 : Math.min(1, Math.abs(v));
      const bg = v == null ? 'transparent' : v >= 0 ? `rgba(41, 82, 227, ${(a * 0.85).toFixed(2)})` : `rgba(217, 73, 62, ${(a * 0.85).toFixed(2)})`;
      return `<span class="heat-c ${i === j ? 'diag' : ''}" style="background:${bg}" title="${esc(L[i].label)} × ${esc(L[j].label)}: ${v == null ? 'n/a' : v.toFixed(2)}">${v == null ? '·' : i === j ? '' : v.toFixed(1).replace('0.', '.').replace('-.', '−.')}</span>`;
    }).join('')}`).join('')}</div>`;
  const warns = (psy.collapse_warnings || []).map(w => `<li class="warn-line">${chip('overlap', 'warn')} ${esc(w.message || (w.pair || []).join(' ↔ '))}</li>`).join('');
  const pca = (psy.pca_explained_variance || []).slice(0, 6);
  const n = Object.keys(sc.players || {}).length;
  return section('Grading reliability check', `advanced · ${plural(n, 'model')}${n < 8 ? ' (small sample)' : ''} · checks whether the 6 traits stay consistent and distinct`,
    `<div class="psy"><div class="psy-split"><div class="scroll"><table class="psy-t"><thead><tr>${th('Trait')}${th('Half-vs-half', 'Rank correlation (Spearman ρ) between two random halves of the hands')}${th('Consistency', 'Spearman-Brown split-half reliability (0.70+ = consistent)')}${th('Models')}${th('Status')}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="small soft">Hands are split into two halves to test consistency. With only a few models, treat these numbers as a rough check.</p></div>
      <div class="psy-corr"><h4>Trait overlap</h4>${heat}<p class="small soft">Shows whether two traits move together. Strong blue or red off the diagonal means two traits are measuring the same thing.</p>
      ${pca.length ? `<div class="pca" title="Principal components (PCA): share of variance each distinct skill factor explains">${pca.map(x => `<span style="height:${Math.round(x * 100)}%"><i>${Math.round(x * 100)}%</i></span>`).join('')}</div><p class="small soft">Distinct skill dimensions captured across traits</p>` : ''}
      ${warns ? `<ul class="warns">${warns}</ul>` : ''}</div></div>`, '', { id: 'psychometrics', fold: true, open: false });
}

function forensicsPanel(sc, run) {
  const fx = sc.forensics;
  if (!fx || !fx.players) return '';
  const ids = Object.keys(fx.players);
  const cell = (p, ch) => {
    const x = p[ch] || {};
    const n = (x.incidents || []).length;
    let detail = '';
    if (ch === 'sandbox') detail = `${x.tool_rejected || 0} blocked · ${x.unverified || 0} unchecked`;
    else if (ch === 'prompt_injection') detail = `${Math.round((x.offense_rate || 0) * 100)}% sent · ${x.susceptibility_rate == null ? 'n/a' : Math.round(x.susceptibility_rate * 100) + '% fell for'}`;
    else if (ch === 'parser_fuzzing') detail = `${x.fuzz_attempts || 0} probes · ${x.fallbacks || 0} auto-folds`;
    else detail = `${x.flagged_turns || 0} flagged · ${x.blocked_attempts || 0} blocked`;
    return `<td data-k="${esc(ch)}"><span class="fx-n ${n ? 'hit' : ''}">${n}</span><small>${detail}</small></td>`;
  };
  const rows = ids.map(pid => {
    const p = fx.players[pid], d = p.ev_decomposition || {};
    const total = d.total_net_bb ?? 0, strat = d.strategic_net_bb ?? 0, iface = total - strat;
    const span = Math.max(Math.abs(total), Math.abs(strat), Math.abs(iface), 1);
    const bar = (v, k) => `<span class="evd-bar ${k} ${v < 0 ? 'neg' : 'pos'}" style="width:${Math.round(Math.abs(v) / span * 100)}%"></span>`;
    return `<tr><th scope="row">${player(pid, '', { details: false })}</th>${PLAIN_CHANNELS.map(([ch]) => cell(p, ch)).join('')}
      <td class="evd" data-k="Poker vs format errors"><div class="evd-row"><small>Poker play</small>${bar(strat, 'strat')}${ebb(strat)}</div>
        <div class="evd-row"><small>Format errors</small>${bar(iface, 'iface')}${ebb(iface)}</div>
        <small class="soft">${d.retry_decisions || 0} retried · ${d.sizing_corrections || 0} resized · ${Math.round((d.clean_decision_rate ?? 1) * 100)}% clean</small></td></tr>`;
  }).join('');
  const incidents = ids.flatMap(pid => PLAIN_CHANNELS.flatMap(([ch, name]) => ((fx.players[pid][ch] || {}).incidents || []).map(i => ({ ...i, pid, name }))));
  const list = incidents.length ? `<details class="fx-list" open><summary>${plural(incidents.length, 'flagged incident')}</summary><ol class="evidence">${incidents.slice(0, 40).map(i => `<li><a href="${Core.evidenceHref(run, i)}">
    <span class="ev-kind incident">${esc(i.name)}</span><span class="ev-where">${plain(i.pid)} · hand ${(i.hand ?? 0) + 1}${i.street ? ' · ' + esc(i.street) : ''}</span>
    <span class="ev-sum">${esc(i.summary || '')}${i.excerpt ? ` <q>${esc(i.excerpt)}</q>` : ''}</span><span class="sev ${esc(i.severity)}">${esc(i.severity || '')}</span><i class="ev-go" aria-hidden="true">›</i></a></li>`).join('')}</ol></details>` : '<p class="small soft fx-clean">All clear: no cheating or safety issues detected.</p>';
  const total = fx.summary && fx.summary.total_incidents != null ? fx.summary.total_incidents : incidents.length;
  const status = total ? `⚠ ${plural(total, 'flag')} across ${PLAIN_CHANNELS.length} safety checks` : `✓ All clear (0 flags across ${PLAIN_CHANNELS.length} checks)`;
  return section('Cheating &amp; safety checks', `<span class="${total ? 'warn-t' : 'ok-t'}">${status}</span>`,
    `<div class="fx-wrap"><table class="fx-t"><thead><tr><th>Model</th>${PLAIN_CHANNELS.map(([, n]) => th(esc(n))).join('')}${th('Poker vs format errors', 'Net result split into actual poker decisions vs. money lost to formatting retries or auto-folds')}</tr></thead><tbody>${rows}</tbody></table></div>${list}`,
    'Invalid replies replaced by an auto-fold or check are tracked separately from poker strategy.', { id: 'forensics', fold: true, open: total > 0 });
}

async function renderScorecards(query) {
  const run = state.run;
  if (state.runs.find(r => r.run === run)?.all) {
    $app.innerHTML = `<div class="card-box empty"><h2>Scorecards are per run</h2><p>Pick a specific run from the dropdown at the top to inspect its character sheets.</p></div>`;
    return;
  }
  let raw = scoreStore.get(run) || null;
  if (!raw) {
    try { raw = await fetchScorecard(run); } catch (e) { throw new Error(STATIC ? 'This export has no scorecard for this run.' : e.message); }
  } else scorecardLater(run, STALE_POLL);
  if (!raw) {
    $app.innerHTML = `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>Grading this run</h2><p>Replaying every hand through the poker solver to grade six traits. This page fills in automatically when ready.</p></div>`;
    scorecardLater(run, PENDING_POLL);
    return;
  }
  if (!scoreUpToDate(raw)) scorecardLater(run, STALE_POLL);
  Object.assign(state.labels, raw.labels || {});
  Object.assign(state.kinds, raw.kinds || {});
  const ids = Object.keys(raw.players || {}).sort((a, b) => (raw.players[b].overall_20 ?? -1) - (raw.players[a].overall_20 ?? -1));
  if (!ids.length) {
    $app.innerHTML = `<div class="card-box empty"><h2>No hands to grade yet</h2><p>Play a few hands and each model's character sheet will appear here.</p></div>`;
    return;
  }
  const saved = (state.scSel && state.scSelRun === run ? state.scSel : null) || store.get('sheets.' + run, null);
  let sel = (saved || []).filter(p => ids.includes(p));
  if (!sel.length) sel = ids.slice(0, Math.min(4, ids.length));
  if (query && query.get('m')) { const m = query.get('m').split(',').filter(p => ids.includes(p)); if (m.length) sel = m; }
  state.scSel = sel; state.scSelRun = run;
  const picker = `<div class="sheet-pick" role="group" aria-label="Models to compare">${ids.map(p => `<button class="toggle pick" data-pick="${esc(p)}" aria-pressed="${sel.includes(p)}">${monogram(p, 'xs')}${esc(nameParts(labelText(p))[0])}</button>`).join('')}
    <span class="spacer"></span><button class="toggle" data-pick="*" title="Show every model">All</button></div>`;
  const stale = raw.cache && !raw.cache.fresh ? `<p class="stale-note small">Showing previous grades while this run re-grades in the background${raw.cache.error ? ` (last attempt failed: ${esc(raw.cache.error)})` : ''}.</p>` : '';
  const m = raw.method || {};
  const grid = `<div class="sheets n${Math.min(sel.length, 4)}">${sel.map(p => sheetCard(p, raw, run)).join('')}</div>`;
  $app.innerHTML = `<div class="scorecards"><div class="section-head sc-head"><h2><span class="suit">♠</span>Character sheets</h2>
      <span class="aside">six traits graded 0–20 from game logs and poker solver analysis</span></div>
    ${stale}${picker}<p class="sc-note">How well each model plays (0–20), independent of card luck. Short-run chip results can differ from play quality.</p>${grid}
    <p class="caption">Dark blue = confident minimum score · Light blue = uncertainty range · Click any row to inspect the breakdown and example hands.</p>
    ${psychometricPanel(raw)}${forensicsPanel(raw, run)}</div>`;
  bindScorecards(raw);
  const focus = query && query.get('focus');
  if (focus) setTimeout(() => {
    const el = document.getElementById(focus);
    el?.querySelector('details.fold')?.setAttribute('open', '');
    el?.scrollIntoView({ block: 'start' });
  }, 0);
}
function bindScorecards(raw) {
  $app.querySelector('.sheets')?.addEventListener('click', e => {
    const b = e.target.closest('[data-ev-more]');
    if (!b) return;
    b.previousElementSibling?.querySelectorAll('li[hidden]').forEach(li => { li.hidden = false; });
    b.remove();
  });
  $app.querySelectorAll('[data-pick]').forEach(b => b.onclick = () => {
    const ids = Object.keys(raw.players);
    const p = b.dataset.pick;
    state.scSel = p === '*' ? ids : state.scSel.includes(p) ? (state.scSel.length > 1 ? state.scSel.filter(x => x !== p) : state.scSel) : [...state.scSel, p];
    store.set('sheets.' + state.run, state.scSel);
    route({ soft: true });
  });
  $app.querySelectorAll('.axis-row').forEach(b => b.onclick = () => {
    const k = b.dataset.toggle;
    if (state.scOpen.has(k)) state.scOpen.delete(k); else state.scOpen.add(k);
    const wrap = b.closest('.axis'), open = state.scOpen.has(k);
    const [pid, axis] = k.split('|');
    wrap.classList.toggle('open', open);
    b.setAttribute('aria-expanded', String(open));
    const dr = wrap.querySelector('.axis-drawer');
    dr.hidden = !open;
    if (open) {
      const meta = Core.axisList(raw).find(a => a.axis === axis);
      dr.innerHTML = axisDrawer(state.run, raw.players[pid].axes[axis] || {}, meta);
    }
  });
}

// ---- the pre-flight planner ---------------------------------------------------------------------------
let planTimer = null, planSeq = 0;
const PLAN_ALPHAS = [0.01, 0.05, 0.1];
const PLAN_PRESETS = [
  { key: 'quick', label: 'Quick Check', sub: '2 models · 100 bb/100 gap', q: { players: 2, target: 100, stack: 100, alpha: 0.05, aivat_vr: 1.5, track: 'official' } },
  { key: 'standard', label: 'Standard Eval', sub: '4 models · 50 bb/100 gap', q: { players: 4, target: 50, stack: 100, alpha: 0.05, aivat_vr: 1.5, track: 'official' } },
  { key: 'publication', label: 'Publication Grade', sub: '4 models · 35 bb/100 gap · 1% error', q: { players: 4, target: 35, stack: 100, alpha: 0.01, aivat_vr: 1.5, track: 'official' } },
];
const presetOf = p => PLAN_PRESETS.find(x => Object.entries(x.q).every(([k, v]) => typeof v === 'number' ? Math.abs(p[k] - v) < 1e-9 : p[k] === v)) || null;
const chipsText = bb => `${money(bb)} (${trim(bb)} big blinds)`;
function planForm(p) {
  const num = (id, label, min, max, step, val, unit = '', tip = '') => `<label class="pf" title="${esc(tip)}"><span class="pf-k">${label}</span>
    <input type="range" id="${id}-r" min="${min}" max="${max}" step="${step}" value="${val}" aria-label="${esc(label)}"><span class="pf-v"><input type="number" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}" inputmode="decimal"><em ${id === 'stack' ? 'id="stack-unit"' : ''}>${unit}</em></span></label>`;
  const cur = presetOf(p);
  return `<form class="plan-form card-box" onsubmit="return false" aria-label="Experiment design">
    <div class="presets" role="group" aria-label="Presets">${PLAN_PRESETS.map(x => `<button type="button" class="preset" data-preset="${x.key}" aria-pressed="${cur === x}"><b>${x.label}</b><small>${x.sub}</small></button>`).join('')}</div>
    <div class="pf"><span class="pf-k">Test mode</span><span class="seg" role="group" aria-label="Test mode">
      <button type="button" class="seg-b" data-track="official" aria-pressed="${p.track === 'official'}" title="Fresh start every deal: strictest statistical test">Independent hands (strictest)</button>
      <button type="button" class="seg-b" data-track="adaptation" aria-pressed="${p.track === 'adaptation'}" title="Long 50-hand sessions where models remember and adapt to opponents">50-hand sessions (tests memory)</button></span></div>
    ${num('players', 'Models', 2, 12, 1, p.players, 'playing each other', 'Models in the round-robin tournament')}
    ${num('target', 'Skill gap to detect', 15, 150, 5, p.target, 'bb/100', 'Smaller gaps take more hands to prove (50 = moderate gap, 100 = large gap)')}
    ${num('stack', 'Stack depth', 20, 200, 5, p.stack, `bb · ${money(p.stack)}`, 'Starting stack every hand, in big blinds ($20 each)')}
    <details class="plan-adv" ${store.get('plan-adv', false) ? 'open' : ''}><summary>Advanced statistical settings</summary><div class="plan-adv-body">
      <div class="pf"><span class="pf-k" title="Maximum allowed chance of naming any false winner across the whole tournament (family-wise α)">False-winner risk (<span class="no-upper">α</span>)</span><span class="seg" role="group" aria-label="False-winner risk">${PLAN_ALPHAS.map(a =>
        `<button type="button" class="seg-b" data-alpha="${a}" aria-pressed="${Math.abs(p.alpha - a) < 1e-9}">${Math.round(a * 100)}%</button>`).join('')}</span></div>
      ${num('sd', 'Hand-to-hand swing (SD)', 100, 1000, 10, p.sd, 'bb/100', 'Typical score spread of one paired deal (past runs average around 340 bb/100)')}
      ${num('aivat_vr', 'Luck reduction (AIVAT)', 1, 5, 0.1, p.aivat_vr, '×', 'Expected speedup from subtracting card luck (1.0× = standard duplicate, 1.5×–2.5× = typical AIVAT)')}
      ${num('replicates', p.track === 'adaptation' ? 'Sessions per check' : 'Deals per check', 1, 200, 1, p.replicates, '', 'How many independent deals or sessions finish before each statistical checkpoint')}
      ${p.track === 'adaptation' ? num('deals', 'Deals per session', 2, 200, 1, p.deals, '', 'Number of deals models remember in one continuous session') : ''}</div></details></form>`;
}
function planResult(r, live) {
  const p = r.params, N = x => Number(x).toLocaleString('en-US');
  const raw = r.raw_deals_per_pairing, eff = r.aivat_effective_deals_per_pairing;
  const save = p.aivat_vr > 1 ? `<b>${trim(Math.round(p.aivat_vr * 10) / 10)}×</b> fewer` : 'no reduction assumed';
  const ps = r.power_scenarios || [];
  // on the official track every deal is a fresh session, so the dependence scenarios coincide: one sentence
  const same = ps.length > 1 && ps.every(x => Math.round(x.power * 100) === Math.round(ps[0].power * 100) && Math.round(x.conversations_used) === Math.round(ps[0].conversations_used));
  const power = !ps.length ? '' : same
    ? `<p class="power-one"><b>${Math.round(ps[0].power * 100)}% chance of catching a real ${p.target} bb/100 skill gap</b> (~${N(Math.round(ps[0].conversations_used))} independent deals per matchup).</p>`
    : `<div class="scroll"><table class="psy-t"><thead><tr>${th('Scenario')}${th('Session drift', 'Assumed intraclass correlation (ICC) across sessions')}${th('Hand carryover', 'Assumed hand-to-hand correlation (AR1) within a session')}${th('Catch rate', 'Statistical power: chance of detecting the target skill gap')}${th('Sessions needed')}</tr></thead><tbody>${ps.map(x => `<tr><td>${esc(x.scenario)}</td><td>${x.assumed_icc}</td><td>${x.assumed_ar1}</td><td>${Math.round(x.power * 100)}%</td><td>${N(Math.round(x.conversations_used))}</td></tr>`).join('')}</tbody></table></div>`;
  const track = p.track === 'official' ? 'Strict sequential test' : 'Memory-mode test (exploratory)';
  const stopTip = r.threshold == null ? '' : `Stopping bar: ${N(Math.round(r.threshold))} (2 / α_pairing, α_pairing = ${r.alpha_pairing == null ? '–' : Number(r.alpha_pairing).toPrecision(2)})`;
  return `<div class="plan-out">
    <div class="plan-src">${live ? chip('live estimate', 'ok', 'Simulated checkpoint-by-checkpoint through the real statistical test') : chip('offline estimate', 'quiet', 'Formula estimate; start the viewer server for simulated accuracy')}
      ${r.underpowered ? chip('needs more hands for 80% accuracy', 'warn') : ''}<span class="small soft" ${stopTip ? `title="${esc(stopTip)}"` : ''}>${track}${r.max_rounds ? ` · up to ${r.max_rounds} checkpoints` : ''}</span></div>
    <div class="plan-tiles">
      <div class="tile"><small>Total hands</small><b>${N(r.hands_total)}</b><em>${N(r.hands_per_pairing)} per matchup · ${plural(r.pairings, 'matchup')}</em></div>
      <div class="tile"><small>Independent deals</small><b>${N(r.conversations_total)}</b><em>${N(r.conversations_per_pairing)} per matchup</em></div>
      <div class="tile"><small>Est. model calls</small><b>${N(r.estimated_calls)}</b><em>${p.aivat_vr > 1 ? `${N(r.estimated_calls_with_aivat)} luck-adjusted` : `${r.assumptions.calls_per_hand} a hand`}</em></div>
      <div class="tile"><small>Est. runtime</small><b>${Core.hoursText(r.estimated_runtime_hours)}</b><em>${p.aivat_vr > 1 ? `${Core.hoursText(r.estimated_runtime_hours_with_aivat)} luck-adjusted` : `${r.assumptions.seconds_per_call}s a call, ${r.assumptions.concurrency} at once`}</em></div></div>
    <div class="plan-bars card-box"><h4>Deals per matchup for the same accuracy</h4>
      <div class="pb"><span>Standard paired deals</span><i style="width:100%"></i><b>${N(raw)} deals</b></div>
      <div class="pb aivat"><span>Luck-adjusted (AIVAT)</span><i style="width:${Math.max(3, Math.round(eff / raw * 100))}%"></i><b>${N(Math.round(eff))} deals</b></div>
      <p class="small soft">${save}: luck-adjustment (AIVAT) strips out card-runout luck so fewer hands reach the same precision. The official test still uses standard paired deals.</p></div>
    ${power ? `<div class="card-box"><h4>Chance of catching a ${p.target} bb/100 skill gap</h4>${power}</div>` : ''}
    <div class="plan-cmd card-box"><h4>Run it</h4><div class="cmd"><code id="plan-cmd">${esc(r.command)}</code><button class="toggle" id="copy-cmd">Copy</button></div>
      <p class="small soft">Replace model1… with names from your players.toml. <code>--stack ${p.stack * 2}</code> is ${chipsText(p.stack)}. ${r.approximate || !live ? 'Add the reference bot so every model also plays one common baseline.' : ''}</p></div></div>`;
}
async function renderPlanner(query) {
  const saved = store.get('plan', {});
  const p = Core.planParams({ ...Core.PLAN_DEFAULTS, ...saved, ...Object.fromEntries(query ? query.entries() : []) });
  $app.innerHTML = `<div class="planner"><div class="section-head"><h2><span class="suit red">♦</span>Pre-flight planner</h2>
      <span class="aside">estimate hands, API calls, and runtime before starting a tournament</span></div>
    <div class="plan-grid">${planForm(p)}<div class="plan-result" aria-live="polite"></div></div></div>`;
  const read = () => {
    const g = id => { const el = $app.querySelector('#' + id); return el ? Number(el.value) : undefined; };
    const q = { track: $app.querySelector('[data-track][aria-pressed="true"]')?.dataset.track || 'official', alpha: Number($app.querySelector('[data-alpha][aria-pressed="true"]')?.dataset.alpha || 0.05) };
    for (const k of ['players', 'stack', 'target', 'sd', 'aivat_vr', 'deals', 'replicates']) { const v = g(k); if (v !== undefined) q[k] = v; }
    return q;
  };
  const refresh = async () => {
    const q = Core.planParams(read());
    store.set('plan', q);
    const mine = ++planSeq;
    const out = $app.querySelector('.plan-result');
    let r, live = false;
    if (!STATIC) {
      try {
        const res = await fetch('/api/plan_preview?' + new URLSearchParams(Object.entries(q).map(([k, v]) => [k, String(v)])), { cache: 'no-store' });
        if (res.ok) { r = await res.json(); live = true; }
      } catch (e) { /* offline: the formulas below */ }
    }
    if (mine !== planSeq || !out.isConnected) return;
    out.innerHTML = planResult(r || Core.planFallback(q), live);
    $app.querySelector('#copy-cmd')?.addEventListener('click', async e => {
      const text = $app.querySelector('#plan-cmd').textContent;
      try { await navigator.clipboard.writeText(text); } catch (err) {
        const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch (x) { /* blocked */ } ta.remove();
      }
      e.target.textContent = 'Copied ✓';
      setTimeout(() => { if (e.target.isConnected) e.target.textContent = 'Copy'; }, 1600);
    });
  };
  const later = (ms = 220) => { clearTimeout(planTimer); planTimer = setTimeout(refresh, ms); };
  const unit = () => {
    const el = $app.querySelector('#stack-unit'), v = Number($app.querySelector('#stack')?.value);
    if (el && v) el.textContent = `bb · ${money(v)}`;
  };
  const presets = () => {
    const cur = presetOf(Core.planParams(read()));
    $app.querySelectorAll('[data-preset]').forEach(b => b.setAttribute('aria-pressed', String(!!cur && cur.key === b.dataset.preset)));
  };
  $app.querySelectorAll('.plan-form input[type=range]').forEach(r => {
    const n = $app.querySelector('#' + r.id.replace(/-r$/, ''));
    r.oninput = () => { n.value = r.value; unit(); presets(); later(); };
    n.oninput = () => { if (n.value !== '') { r.value = n.value; unit(); presets(); later(400); } };
  });
  $app.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => {
    const x = PLAN_PRESETS.find(y => y.key === b.dataset.preset);
    store.set('plan', Core.planParams({ ...Core.PLAN_DEFAULTS, ...x.q }));
    renderPlanner(new URLSearchParams());
  });
  $app.querySelector('.plan-adv')?.addEventListener('toggle', e => store.set('plan-adv', e.target.open));
  $app.querySelectorAll('[data-track]').forEach(b => b.onclick = () => {
    const q = Core.planParams({ ...read(), track: b.dataset.track, deals: undefined, replicates: undefined });
    store.set('plan', q);
    renderPlanner(new URLSearchParams());
  });
  $app.querySelectorAll('[data-alpha]').forEach(b => b.onclick = () => {
    $app.querySelectorAll('[data-alpha]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    presets();
    later(0);
  });
  refresh();
}

// ---- standings: forensics status, meta-game, AIVAT ----------------------------------------------------
function metaCard(raw) {
  const m = Core.metaRows(raw.meta_rank);
  if (!m || m.rows.length < 2) return '';
  const maxW = Math.max(0.0001, ...m.rows.map(r => r.weight || 0)), maxM = Math.max(0.0001, ...m.rows.map(r => r.mass || 0));
  const rows = m.rows.map(r => `<div class="meta-row"><span class="meta-name">${player(r.player, '', { details: false })}</span>
    <span class="meta-c" title="Nash equilibrium weight: optimal share to pick this model against the whole field"><small>Balanced share</small><span class="bar"><i style="width:${Math.round((r.weight || 0) / maxW * 100)}%"></i></span><b>${r.weight == null ? '–' : Math.round(r.weight * 100) + '%'}</b></span>
    <span class="meta-c" title="Nash-averaged edge (bb/100) against the optimal mix of opponents"><small>Balanced edge</small><b class="${cls(r.score)}">${r.score == null ? '–' : signed(r.score, 0)}</b></span>
    <span class="meta-c" title="α-Rank stationary mass: long-run share an evolving population spends on this model"><small>Evolution share</small><span class="bar alpha"><i style="width:${Math.round((r.mass || 0) / maxM * 100)}%"></i></span><b>${r.mass == null ? '–' : Math.round(r.mass * 100) + '%'}</b></span></div>`).join('');
  const cycles = m.cycles.map(c => `<li class="cycle-alert"><b>⟳ Cycle</b> ${esc(Core.cycleText(c, x => nameParts(labelText(x))[0]))}</li>`).join('');
  // a real cycle is news and opens on the page; a transitive field only agrees with the standings, so it folds away
  return section(m.nonTransitive ? 'Matchup balance · Rock-paper-scissors loop found' : 'Matchup balance <span class="h-sub">(no rock-paper-scissors loops · agrees with standings)</span>',
    m.nonTransitive ? 'some models beat each other in a loop, so one ladder cannot tell the whole story' : '',
    `<div class="meta"><div class="meta-rows">${rows}</div>${m.nonTransitive ? `<div class="cycle-box"><p><b>Rock-paper-scissors loop:</b> some players beat each other in a cycle, so no single linear ranking fits. Nash averaging and α-Rank balance the whole field.</p>${cycles ? `<ul>${cycles}</ul>` : ''}</div>` : '<p class="small soft meta-note">No rock-paper-scissors loops: the ranking is clean and matches Nash averaging and α-Rank.</p>'}</div>`,
    'Game-theory rankings (Nash averaging & α-Rank) check whether weaker opponents or rock-paper-scissors loops distort the leaderboard.', m.nonTransitive ? { id: 'metagame' } : { id: 'metagame', fold: true });
}
function aivatLine(p, lead) {
  const a = Core.aivatOf(state.v2 && state.v2.aivat, p.a, p.b);
  if (!a || a.aivat == null) return '';
  const up = lead === p.a ? 1 : -1;
  return `<small class="aivat-line" title="Luck-adjusted edge (AIVAT) subtracts deal and runout luck. Raw duplicate edge ${signed(up * a.raw, 0)} ± ${Math.round(a.raw_se || 0)}; luck-adjusted ${signed(up * a.aivat, 0)} ± ${Math.round(a.se || 0)} bb/100 (precision of ${trim(Math.round((a.multiplier || a.vr || 1) * 10) / 10)}× as many raw hands).">Luck-adjusted ${signed(up * a.aivat, 0)} ± ${Math.round(a.se || 0)}</small>`;
}

// ---- the replayer: split duplicate, solver overlay, private mind vs public action ---------------------
state.split = store.get('split', false);
state.solver = store.get('solver', true);
state.solverOpen = store.get('solver-open', false);
state.mindPanel = store.get('mindpanel', false);
state.scOpen = new Set();
const evalStore = new Map(), evalFailed = new Map();
let fxStore = null;
async function fetchHandEval(id, hand) {
  if (STATIC) return null;
  try {
    const r = await fetch(`/api/hand_eval?id=${encodeURIComponent(id)}&hand=${hand}`, { cache: 'no-store' });
    return r.ok ? await r.json() : null;
  } catch (e) { return null; }
}
async function loadForensicsIndex() {
  if (fxStore && fxStore.run === state.run) return fxStore;
  const run = state.run;
  fxStore = { run, by: new Map() };
  try {
    const fx = await api('forensics', ['run', run]);
    for (const [pid, p] of Object.entries(fx.players || {})) for (const [ch] of Core.CHANNELS) for (const i of (p[ch] || {}).incidents || []) {
      const mid = String(i.match_id || '').replace(new RegExp('^' + run.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/'), '');
      const k = `${mid}|${i.hand}`;
      if (!fxStore.by.has(k)) fxStore.by.set(k, []);
      fxStore.by.get(k).push({ ...i, channel: ch, pid });
    }
  } catch (e) { /* no forensics in this export */ }
  return fxStore;
}
const relId = id => String(id).replace(new RegExp('^' + state.run.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/'), '');

// The side-by-side summary bar; the toggles that used to sit here live in the playback bar now.
function v2ReplayBar(plays) {
  if (plays.length < 2) return '';
  return `<div class="split-bar card-box" ${state.split ? '' : 'hidden'} aria-live="polite"></div>`;
}
function v2PlaySlots(j) {
  return `<div class="solver-box" data-play="${j}"></div><div class="mind-box" data-play="${j}"></div>`;
}
function atStep(p, f) { return p.steps.length ? Math.min(f.at[p.j] ?? 0, p.steps.length - 1) : -1; }

const ACT_NAME = { fold: 'Fold', check: 'Check', call: 'Call', bet: 'Bet', raise: 'Raise', all_in: 'All in' };
const actName = a => ACT_NAME[a] || String(a || '–').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
// The solver's reading of the decision on screen: one line by default, the full breakdown on request.
function solverHtml(p, mine, ev, bb, finished) {
  if (!state.solver) return '';
  if (ev === undefined) return '<div class="sv muted"><b>Solver</b> · reading the hand…</div>';
  if (!ev) return STATIC ? '<div class="sv muted"><b>Solver</b> · needs the live viewer server.</div>' : finished ? '<div class="sv muted"><b>Solver</b> · no reading for this hand.</div>' : '';
  const at = Core.decisionAt(p.log, ev.decisions, mine);
  if (!at || !at.decision) return '';
  const d = at.decision, badge = Core.evBadge(d), players = p.m.config.players;
  const who = nameParts(labelText(players[d.seat ?? at.action.seat] || ''))[0];
  const bars = Core.gtoBars(d);
  const loss = d.ev_loss_mbb;
  const tile = (k, v, tip = '') => `<div class="sv-t" ${tip ? `title="${esc(tip)}"` : ''}><small>${k}</small><b>${v}</b></div>`;
  const av = Core.aivatPill(ev.aivat, d.seat ?? 0, bb);
  const lossHtml = loss == null ? '–' : money(loss / 1000);
  const evBB = loss == null ? '' : ` · cost ${loss > 0 ? Core.MINUS : ''}${money(Math.abs(loss) / 1000)}`;
  const luck = av && Math.abs(av.luck_bb || 0) > 0.05;
  const luckLabel = !av ? '' : Math.abs(av.luck_bb || 0) <= 0.05 ? 'Card luck: neutral'
    : `Card luck: ${moneySigned(av.luck_bb)} ${av.luck_bb > 0 ? 'lucky' : 'unlucky'}`;
  const avPill = av ? `<span class="aivat-pill ${av.luck_bb > 0.05 ? 'pos' : av.luck_bb < -0.05 ? 'neg' : ''}" title="${esc(av.text)} (AIVAT subtracts card-runout luck from the hand result)">${esc(luckLabel)}${av.allin ? ` · all-in ${esc(av.street || '')}` : ''}</span>` : '';
  const open = state.solverOpen;
  const narrowSeg = bars.some(x => x.freq < 0.14);
  const badgeWord = badge ? (badge.kind === 'blunder' ? 'BIG MISTAKE' : badge.kind === 'indifferent' ? 'CLOSE CALL' : badge.text) : '';
  const badgeTip = badge ? (badge.kind === 'blunder' ? `Lost at least ${money(0.25)} vs. the solver's best move` : badge.kind === 'indifferent' ? 'Two or more moves have nearly equal expected value here' : '') : '';
  return `<div class="sv ${badge ? badge.kind : ''} ${open ? 'open' : ''}">
    <div class="sv-line"><span class="sv-sum"><b>Solver</b><span class="sv-who"> · ${esc(who)} ${esc(actName(d.taken_action || at.action.action).toLowerCase())}${at.upcoming ? ' (first decision)' : ''}</span>
      · Best: <b>${esc(actName(d.best_action))}</b><span title="Expected amount lost compared with the solver's best move">${evBB}</span></span>
      ${badge ? `<span class="badge ev-${badge.kind}" title="${esc(badgeTip)}">${badgeWord}</span>` : ''}${luck ? avPill : ''}
      <button class="sv-more" type="button" aria-expanded="${open}">${open ? 'Hide' : 'Details'} <i aria-hidden="true">${open ? '▴' : '▾'}</i></button></div>
    ${open ? `<div class="sv-detail"><div class="sv-tiles">${tile('Call price', Core.pctText(d.pot_odds), 'Pot odds: share of the final pot you must win to break even on a call')}${tile('Defend target', Core.pctText(d.mdf), 'Minimum defense frequency (MDF): how often you must continue to avoid being exploited by bluffs')}
      ${tile('Win chance', Core.pctText(d.equity), 'Estimated equity against the opponent\'s likely range')}${tile('Best move', esc(actName(d.best_action)))}
      <div class="sv-t loss ${badge ? badge.kind : ''}" title="Expected value lost against the best move (${loss == null ? '–' : trim(Math.round(loss / 10) / 100) + ' big blinds'})"><small>Mistake cost</small><b>${lossHtml}</b></div></div>
    ${bars.length ? `<div class="gto" role="img" aria-label="GTO frequencies: ${esc(bars.map(x => `${actName(x.action)} ${Math.round(x.freq * 100)}%`).join(', '))}">${bars.map(x => `<span class="gto-seg ${esc(x.action)} ${x.taken ? 'taken' : ''}" style="flex:${Math.max(x.freq, 0.04)}" title="${esc(actName(x.action))} ${Math.round(x.freq * 100)}%${x.taken ? ' (played)' : ''}">${x.freq >= 0.14 ? `${esc(actName(x.action))} ${Math.round(x.freq * 100)}%` : ''}</span>`).join('')}</div>
      ${narrowSeg ? `<div class="gto-legend">${bars.map(x => `<span><i class="gto-sw ${esc(x.action)}"></i>${esc(actName(x.action))} ${Math.round(x.freq * 100)}%${x.taken ? ' · played' : ''}</span>`).join('')}</div>` : ''}` : ''}
    ${ev.note ? `<div class="sv-note">${esc(ev.note)}</div>` : ''}
    <div class="sv-foot">${!luck ? avPill : ''}
      ${(ev.per_seat_ev_loss_mbb || []).some(x => x != null) ? `<span class="small soft">Total mistake cost this hand: ${ev.per_seat_ev_loss_mbb.map((x, i) => `${esc(nameParts(labelText(players[i] || ''))[0])} ${x == null ? '–' : money(x / 1000)}`).join(' · ')}</span>` : ''}</div></div>` : ''}</div>`;
}

// Model prose, safely: escape everything first, then only **bold**, *italic* and `code` become markup.
function fmtMind(text) {
  return esc(text)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)\*(?!\w)/g, '$1<em>$2</em>');
}

function notesBaseline(m, hand, seat) {
  const eps = Object.keys(m.events || {}).map(Number).filter(e => e < hand).sort((a, b) => b - a);
  for (const ep of eps) {
    const ts = (m.events[ep] || []).filter(e => e.kind === 'thought' && (e.data || {}).seat === seat);
    if (ts.length) return { notes: ts[ts.length - 1].data.notes || '', hand: ep };
  }
  return null;
}
// The table names the models saw this hand ("Alex", "Blair"), each with the model behind it.
function aliasKey(m, hand) {
  const start = (m.events[hand] || []).find(e => e.kind === 'hand_start');
  const names = (start && start.data && start.data.names) || m.config.names || [];
  const players = m.config.players || [];
  const pairs = names.map((n, i) => n && players[i] ? `${esc(n)} = ${esc(nameParts(labelText(players[i]))[0])}` : '').filter(Boolean);
  return pairs.length ? `<span class="alias-key" title="The seat names the models saw at the table">${pairs.join(' · ')}</span>` : '';
}
function mindHtml(p, mine, pick, fmt, flags) {
  const act = [...p.log].reverse().find(e => e.kind === 'action' && e.step <= mine);
  if (!act) return '';
  const t = act.thought, players = p.m.config.players, pid = players[act.seat];
  const hasMind = t && (t.thinking || t.reasoning || t.notes);
  const VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to', all_in: 'all in' };
  const amt = ['call', 'bet', 'raise'].includes(act.action) ? ` ${fmt(act.amount)}` : '';
  const corr = correction(t, act, fmt);
  const base = t ? notesBaseline(p.m, pick, act.seat) : null;
  let notes = '';
  if (t && t.notes) {
    const segs = Core.noteDiff(base ? base.notes : '', t.notes);
    notes = Core.noteChanged(segs)
      ? `<div class="notes-diff">${segs.map(s => s.t === 'same' ? esc(s.text) + ' ' : `<${s.t === 'add' ? 'ins' : 'del'}>${esc(s.text)}</${s.t === 'add' ? 'ins' : 'del'}> `).join('')}</div>`
      : '<div class="notes-diff same">Notes unchanged since the previous hand.</div>';
  } else if (t) notes = '<div class="notes-diff same">No notes kept.</div>';
  const fl = [];
  if (corr) fl.push(`<span class="badge ${corr.kind === 'fallback' ? 'bad' : 'warn'}" title="${esc(corr.reason || '')}">${esc(corr.label)}</span>`);
  for (const i of flags.filter(x => !x.decision_id || !t || x.decision_id === t.decision_id)) {
    if (i.channel === 'prompt_injection') fl.push(`<span class="badge ${i.severity === 'attack' ? 'bad' : 'warn'}" title="${esc(i.summary || '')}">prompt attack</span>`);
    else if (i.channel === 'covert_intent') fl.push(`<span class="badge bad" title="${esc(i.summary || '')}">cheating plan</span>`);
    else if (i.channel === 'parser_fuzzing') fl.push(`<span class="badge warn" title="${esc(i.summary || '')}">format probe</span>`);
    else fl.push(`<span class="badge warn" title="${esc(i.summary || '')}">${esc(String(i.channel).replace(/_/g, ' '))}</span>`);
  }
  const think = t && (t.thinking || t.reasoning) ? fmtMind(t.thinking || t.reasoning) : '';
  return `<div class="mp"><div class="mp-col private"><h4><span aria-hidden="true">🜏</span> Private thoughts <small>hidden from opponent</small></h4>
      ${think ? `<div class="mp-think">${think}</div><p class="chip-note">${esc(Core.chipNote(state.view && state.view.bb))}</p>` : `<div class="mp-think none">${hasMind ? '' : 'No private thoughts recorded.'}</div>`}
      ${t ? `<h5>Opponent notes (changes)<small>${base ? `since hand ${base.hand + 1}` : 'first notes'}</small>${aliasKey(p.m, pick)}</h5>${notes}` : ''}</div>
    <div class="mp-col public"><h4><span aria-hidden="true">♠</span> Public move <small>visible to opponent</small></h4>
      <div class="mp-act"><b>${esc(nameParts(labelText(pid || ''))[0])}</b> <span class="verb ${esc(act.action)}">${VERB[act.action] || esc(act.action)}${amt}</span></div>
      ${act.say ? `<div class="mp-say"><small>Table talk</small>“${esc(act.say)}”</div>` : '<div class="mp-say none">Said nothing.</div>'}
      ${fl.length ? `<div class="mp-flags">${fl.join('')}</div>` : ''}</div></div>`;
}

function splitBarHtml(v, f) {
  const [p0, p1] = v.plays;
  const a = v.cur.a;
  const col = p => {
    const k = atStep(p, f);
    const st = k >= 0 ? p.steps[k] : null;
    if (!st) return `<div class="sb-col"><small>${p.j ? 'Play 2 (swapped)' : 'Play 1'}</small><span class="soft">not dealt</span></div>`;
    const seat = p.m.config.players.indexOf(a);
    const net = st.done ? st.seats[seat].net : null;
    return `<div class="sb-col"><small>${p.j ? 'Play 2 (swapped)' : 'Play 1'} · ${esc(SEG_NAME[st.street] || st.street)}</small>
      <div class="sb-board">${PokerTable.cards(st.board, 0, 'sm') || '<span class="soft small">no board</span>'}</div>
      <div class="sb-line">Pot <b class="gold">${cash(st.pot, v.bb)}</b><span class="sb-net">${short3(a)} ${net == null ? '<span class="soft">in play</span>' : ebb(net / v.bb)}</span></div></div>`;
  };
  const nets = v.plays.map(p => {
    const st = p.steps[p.steps.length - 1];
    return st && st.done && p.steps[atStep(p, f)]?.done ? st.seats[p.m.config.players.indexOf(a)].net / v.bb : null;
  });
  const part = x => x == null ? '<span class="soft">in play</span>' : ebb(x);
  // the combined equation only once a play has a result; until then both plays are simply in play
  const sum = nets.some(x => x != null)
    ? `<div class="sb-sum">${short3(a)}: Play 1 ${part(nets[0])} <span class="op">+</span> Play 2 ${part(nets[1])}${nets.every(x => x != null) ? ` <span class="op">=</span> combined ${part(nets[0] + nets[1])}` : ''}</div>`
    : '<div class="sb-sum soft">Both hands in progress: combined net appears once a hand finishes.</div>';
  return `${col(p0)}${col(p1)}${sum}`;
}

function refreshV2() {
  const v = state.view, rp = state.replay;
  if (!v || !v.plays || !rp || !rp.frames) return;
  const f = rp.frames[rp.s];
  const flagsFor = p => (fxStore && fxStore.by.get(`${relId(v.ids[p.j])}|${v.pick}`)) || [];
  v.plays.forEach(p => {
    const k = atStep(p, f);
    if (k < 0) return;
    const sv = $app.querySelector(`.solver-box[data-play="${p.j}"]`);
    if (sv) {
      const ev = evalStore.get(v.ids[p.j] + '#' + v.pick);
      sv.innerHTML = solverHtml(p, k, ev, v.bb, v.pick in (v.nets || {}));
    }
    const mb = $app.querySelector(`.mind-box[data-play="${p.j}"]`);
    if (mb) mb.innerHTML = state.mindPanel ? mindHtml(p, k, v.pick, c => cash(c, v.bb), flagsFor(p)) : '';
  });
  const sbar = $app.querySelector('.split-bar');
  if (sbar && state.split && v.plays.length === 2) sbar.innerHTML = splitBarHtml(v, f);
}
function loadV2(v) {
  loadForensicsIndex().then(() => { if (state.view === v) refreshV2(); });
  if (!state.solver) return;
  v.plays.forEach(p => {
    const key = v.ids[p.j] + '#' + v.pick;
    if (evalStore.has(key)) {
      const hit = evalStore.get(key);
      if (hit !== null || Date.now() - (evalFailed.get(key) || 0) < 20000) return;
    }
    evalStore.set(key, undefined);
    fetchHandEval(v.ids[p.j], v.pick).then(ev => {
      evalStore.set(key, ev);
      if (!ev) evalFailed.set(key, Date.now());
      if (state.view === v) refreshV2();
    });
  });
}
function bindV2(v) {
  const apply = () => {
    const split = state.split && v.plays.length === 2;
    document.querySelector('.mbody')?.classList.toggle('split', split);
    $app.querySelector('.pair')?.classList.toggle('split', split);
    const sb = $app.querySelector('.split-bar');
    if (sb) sb.hidden = !split;
    const view = split ? 'split' : String(state.showPlay);
    $app.querySelectorAll('[data-view]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.view === view)));
    const sync = document.getElementById('sync-toggle');
    if (sync) { sync.disabled = split; sync.setAttribute('aria-pressed', String(state.mode === 'aligned')); }
  };
  // Play 1, Play 2, or both side by side: side by side always steps street by street
  $app.querySelectorAll('[data-view]').forEach(b => b.onclick = () => {
    const k = b.dataset.view;
    state.split = k === 'split';
    store.set('split', state.split);
    if (state.split && state.mode !== 'aligned') { state.mode = 'aligned'; store.set('timeline', 'aligned'); rebuildTimeline(); }
    apply();
    if (!state.split) pickPlay(Number(k));
    refreshV2();
  });
  const st = document.getElementById('solver-toggle');
  if (st) st.onclick = () => {
    state.solver = !state.solver;
    store.set('solver', state.solver);
    st.setAttribute('aria-pressed', String(state.solver));
    if (state.solver) loadV2(v);
    refreshV2();
  };
  // the solver strip is redrawn every step, so its open state lives in `state`
  $app.querySelector('.pair')?.addEventListener('click', e => {
    if (!e.target.closest('.sv-more')) return;
    state.solverOpen = !state.solverOpen;
    store.set('solver-open', state.solverOpen);
    refreshV2();
  });
  apply();
  loadV2(v);
  refreshV2();
}
