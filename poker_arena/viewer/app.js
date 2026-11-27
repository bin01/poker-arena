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
// A tournament's runs, one per track, official first; a run outside any tournament is a set of one.
const tourKey = r => r && r.tournament ? `${r.run.split('/').slice(0, -1).join('/')}|${r.tournament}` : null;
function trackSet(run) {
  const r = state.runs.find(x => x.run === run);
  const k = tourKey(r);
  if (!k) return r ? [r] : [];
  return state.runs.filter(x => !x.all && tourKey(x) === k).sort((x, y) => (x.track === 'official' ? 0 : 1) - (y.track === 'official' ? 0 : 1));
}
const siblingTrack = (run, track) => trackSet(run).find(r => r.track === track) || null;
const TRACK_TITLE = { official: 'Official standings', adaptation: 'Adaptation (models remember the session)' };
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
    const open = store.get('fold.' + opts.id, false);
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
      const tracks = set.length > 1 ? ' · 2 tracks' : r.track === 'adaptation' ? ' · adaptation' : '';
      return `<option value="${esc(r.run)}" ${mine.has(r.run) ? 'selected' : ''}>${esc(runLabel(r.run))}${tracks}${live ? ' · live' : ''}</option>`;
    };
    const seenT = new Set();
    const all = state.runs.filter(r => r.all), each = state.runs.filter(r => !r.all).filter(r => {
      const lead = trackSet(r.run)[0];
      if (seenT.has(lead.run)) return false;
      seenT.add(lead.run);
      return true;
    }).map(r => trackSet(r.run)[0]);
    $run.innerHTML = all.map(option).join('') + (all.length ? `<optgroup label="Runs">${each.map(option).join('')}</optgroup>` : each.map(option).join(''));
    $run.onchange = () => { location.hash = runLink($run.value); };
    const here = state.runs.find(r => r.run === run);
    const tab = parts[2] === 'm' ? 'matches' : parts[2] === 'q' && here.quiz ? 'quiz' : 'board';
    document.querySelectorAll('.top nav a').forEach(a => {
      a.classList.toggle('active', a.dataset.tab === tab);
      a.hidden = a.dataset.tab === 'quiz' && !here.quiz;
      a.href = runLink(run, { matches: '/m', quiz: '/q' }[a.dataset.tab] || '');
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
  const conv = d => d.replicates > 1 ? `${d.replicates} conversations of ${plural(d.round_deals, 'deal')}` : plural(d.round_deals, 'deal');
  const rounds = rt.design?.round_deals ? (rt.design.max_rounds > 1 ? `up to ${rt.design.max_rounds} rounds of ${conv(rt.design)}` : `${conv(rt.design)} per matchup`)
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
    `${plural(2 * (rt.deals || 0), 'duplicate hand')} in the whole archive`];
  const notes = [];
  if ((rt.left_out || []).length) notes.push(`Left out, played under other game settings: ${rt.left_out.map(x => esc(runLabel(x.run))).join(', ')}.`);
  if ((rt.code_versions || []).length > 1) notes.push(`Played on ${rt.code_versions.length} versions of the arena.`);
  const gone = (rt.runs || []).filter(x => !state.runs.some(y => y.run === x.run)).length;
  if (gone) notes.push(`${plural(gone, 'run')} kept from the history after ${gone === 1 ? 'its folder was' : 'their folders were'} deleted.`);
  const self = (rt.runs || []).reduce((n, x) => n + (x.self_play || 0), 0);
  if (self) notes.push(`${plural(self, 'deal')} between two entries of the same model and level don't count.`);
  if ((rt.other_track_runs || []).length) notes.push(`Official tracks only: ${plural(rt.other_track_runs.length, 'adaptation-track run')} (long conversations, model-based, never official) ${rt.other_track_runs.length === 1 ? 'is' : 'are'} kept apart, never pooled here.`);
  return `<div class="page-head"><div><div class="eyebrow">History</div><h1>All runs</h1><div class="sub">${sub.join(' · ')}</div></div>
    <div class="progress-block"><div class="line">One player per model and reasoning level, pooled across every run
      ${r.started_at ? `since ${new Date(r.started_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}.</div>
      ${notes.length ? `<div class="line soft small">${notes.join(' ')}</div>` : ''}</div></div>`;
}

function intro(rt) {
  const all = rt && rt.cohort ? rt.players.map(p => p.player) : rt && rt.players.length ? [...rt.players.filter(p => p.in_fit).map(p => p.player), ...rt.roster.map(x => x.player)]
    : state.index.flatMap(m => [m.a, m.b]);
  const seen = new Set();
  const ids = all.filter(p => !seen.has(labelText(p)) && seen.add(labelText(p)));
  const models = ids.filter(p => !isBot(p)), bots = ids.filter(p => isBot(p));
  const who = ids.length ? `<div class="who-plays"><span class="wp-k">Playing</span>${[...models, ...bots].map(p =>
    `<span class="wp">${monogram(p, 'sm')}${name(p, { effort: true })}</span>`).join('')}</div>` : '';
  const live = state.index.find(m => m.live);
  return `<div class="intro">
    <p class="lede"><b>${ids.length && !models.length ? 'Baseline bots play heads-up No-Limit Hold\'em: the fixed opponents models are measured against.' : 'AI models play heads-up No-Limit Hold\'em against each other.'}</b> Every deal is played twice with
      the seats swapped, which reduces the luck of the cards.</p>
    <p class="lede-2">Results are in <b>bb/100</b>: big blinds won per 100 hands, so above 0 means winning chips. A <span class="decided-badge">Decided</span>
      badge means a test planned before play settled that result; everything else shows who is ahead so far, not who is better.</p>
    <a class="cta" href="${live ? matchLink(live.id) : runLink(state.run, '/m')}"><span class="cta-icon">▶</span>${live ? 'Watch the live table' : 'Watch a hand'}</a>${who}</div>`;
}

const fmtStrength = v => v == null ? '–' : signed(v, 0);
const STRENGTH_TIP = 'Centred fitted strength (bb/100): one least-squares fit over every deal, summing to zero in the group. The difference between two players\' strengths is the edge the fit predicts when they meet; it is not a player\'s winnings against an average opponent.';
const OBSERVED_TIP = 'Observed opponent-weighted winnings (bb/100): the average of this player\'s direct matchup edges, one per opponent it met, each opponent weighted equally.';
const ciText = ci => ci ? `${signed(ci[0], 0)} to ${signed(ci[1], 0)}` : '';
const levelPct = x => `${Math.round(x * 1000) / 10}%`;
// An ordinary interval, said as what it is: fixed-time and descriptive, never a 95% guarantee here.
const descRange = (ci, x) => ci ? `<span class="ci-desc" title="${esc(Core.DESCRIPTIVE_TIP)}">${ciText(ci)} <i>${esc(Core.ciKind(x))}</i></span>` : '';
// An official pairing's anytime-valid interval, the primary range, in the orientation `up` (a's when true).
const officialRange = (p, up = true) => {
  const o = Core.officialInterval(p);
  if (!o) return null;
  const ci = up ? [o.lo, o.hi] : [-o.hi, -o.lo];
  return { ci, html: `<span class="ci-official" title="Anytime-valid: it holds at every check whenever play stops, at the level that controls the chance of any wrong verdict across the matchups${o.level ? ` (${levelPct(o.level)})` : ''}, if conversations are independent and the edge holds still.">${ciText(ci)} <i>anytime-valid${o.level ? ` ${levelPct(o.level)}` : ''}</i></span>` };
};
// A chip when a headline rests on a model whose identity its tool never confirmed.
const identityChip = (rt, ids, given = false) => {
  const st = ids.map(x => isBot(x) ? null : Core.identityOf(rt.players.find(p => p.player === x), (rt.audit && rt.audit.players || {})[x]) || Core.identityOf(null, (rt.playerAudit || {})[x]));
  const who = ids.filter((x, i) => st[i] === 'unattested' || st[i] === 'partial');
  if (!who.length && !given) return '';
  return chip('model unconfirmed', 'quiet', `${who.length ? who.map(plain).join(', ') + ': its' : 'A'} tool never reported which model answered, so this result is qualified: only the configured name is known.`);
};
const pid = x => Core.playerId(x);
// the interval's basis at the number: "within conversations" is flagged, it leaves out between-conversation spread
const basisTag = x => {
  const b = Core.basisOf(x);
  return b ? `<span class="basis ${b.key}" title="${esc(b.long)}">${esc(b.short)}</span>` : '';
};
const chip = (text, tone = '', tip = '') => `<span class="chip ${tone}" ${tip ? `title="${esc(tip)}"` : ''}>${text}</span>`;
// What a verdict rests on, as two things kept apart: the method (which test) and the evidence (whether the checks it
// needs hold here). A failed or unmade check stays visible whatever the method.
const FREE_TIP = 'An anytime-valid test for bounded results: no assumption about the shape of the results, only that conversations (match pairs) are independent. A win names the better model only if each name served one unchanged model for the whole run.';
function assumeChip(a, { compact = false } = {}) {
  if (!a) return '';
  if (a.method === 'free') {
    const method = chip('assumption-free test', 'ok', FREE_TIP);
    const evidence = a.status === 'unsupported' ? chip(esc(a.label), 'warn', `A check the test needs failed (${a.detail}), so its error rate is not established here.`)
      : a.status === 'incomplete' ? chip(esc(a.label), 'quiet', `Some checks it needs could not be made (${a.detail}).`)
      : a.checks && a.checks.length ? chip('checks passed', 'ok', a.detail) : '';
    return compact ? (a.status === 'free' ? method : evidence) : method + evidence;
  }
  return chip(esc(a.label), a.status === 'unsupported' ? 'warn' : a.status === 'supported' || a.status === 'free' ? 'ok' : 'quiet',
    a.status === 'free' ? FREE_TIP
      : a.status === 'unsupported' ? `Outside the range the test was calibrated for${a.detail ? ` (${a.detail})` : ''}: its error rate is not established.`
      : a.status === 'supported' ? `The test's assumptions hold here${a.detail ? ` (${a.detail})` : ''}.` : 'Whether the test\'s assumptions hold here is not checked.');
}
const voidReason = (d, reg = state.rt && state.rt.registration) => (reg && reg.status === 'void' && ((reg.reasons || []).join('; ') || 'the code changed after it was registered')) || d && (d.registration_void || (d.registration && (d.registration.status === 'void' || d.registration.void)
  && (d.registration.reason || 'the code changed after it was registered')) || (d.registered === 'void' && (d.void_reason || 'the code changed after it was registered')));

// What kind of run this is, in one chip: a registered test, an unregistered or void one, or pooled history.
function runKind(rt) {
  const d = rt.design || {};
  const v = voidReason(d);
  if (rt.pooled) return chip('pooled history', 'quiet', 'Pooled across runs, each stopped by its own rule: described, never decided.');
  if (rt.track === 'adaptation') return chip('adaptation · exploratory · not official', 'warn', 'Long conversations, read by a model-based test whose error rate is not established: exploratory, never official, never a rank or a leader.');
  if (v) return chip('registration void', 'warn', `The test plan no longer binds: ${v}. Nothing here is decided.`);
  if (rt.unregistered) return chip('unregistered', 'warn', 'Played before its test plan was recorded: everything here is ahead on points only, and sample sizes are retrospective.');
  if (d.registered || d.registration) return chip('registered plan', 'ok', 'The test plan was recorded before the first hand.');
  return '';
}

// The headline for one group of players, in plain words: who is ahead, whether that is decided, and on what.
// When the matchups do not fit one number per player, the direct results lead and the fitted number is secondary.
function outcomeRow(rt, g, multi, first = true) {
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
  let kicker, tone, why;
  if (rt.track === 'adaptation') {
    const mb = g.model_based_leader;
    kicker = mb ? 'Ahead in every matchup · exploratory' : 'Ahead on points · exploratory track';
    tone = 'points';
    why = n < 2 ? 'Only one player in this group has results so far.'
      : mb ? `The model-based test reads it as ahead of ${others}. That test's error rate is not established for these results, so this is exploratory: not a leader, and it never changes the official standings.`
      : `Exploratory: ${decided} of ${plural(g.pairings.length, 'matchup')} ${decided === 1 ? 'has' : 'have'} a model-based reading. ${p ? `${plain(who)} leads on points.` : ''}`;
  } else if (g.drift_leader) {
    const drifted = g.pairings.filter(x => !Core.countsForLeader(x) && verdictSides(x)?.winner === who);
    kicker = `${drifted.length ? esc(Core.driftWords(drifted[0]).phrase.replace(/^./, c => c.toUpperCase())) : 'Edge drifted'} · needs confirmation`;
    tone = 'drift';
    why = `Not a leader: the test's bar was cleared against ${others}, but the edge drifted between early and late conversations, so this says only that it was ahead for part of the run. It needs a fresh run under a stable model snapshot to count.`;
  } else if (g.qualified_leader) {
    const failed = g.pairings.filter(x => x.assumption && x.assumption.method === 'free' && x.assumption.status === 'unsupported');
    kicker = failed.length ? 'Leader, but a check failed' : 'Leader, outside the calibrated range';
    tone = 'qualified';
    why = failed.length ? `It won its matchup against ${others}, but a check the test needs failed for at least one of those wins (${esc(failed.flatMap(x => x.assumption.failed).join(', '))}), so it does not count as a decided leader.`
      : `It won its matchup against ${others}, but at least one of those wins is outside the range the test is calibrated for, so it does not count as a decided leader.`;
  } else if (g.leader) {
    kicker = '<span class="decided-badge">Decided</span>leader';
    tone = 'decided';
    why = `Decided: it beat ${others} under the planned test. ${g.leader_assumption === 'free'
      ? `The test assumes nothing about the shape of the results, only that conversations are independent. ${Core.STABLE_NOTE}`
      : g.leader_assumption === 'supported' ? 'The diagnostics passed; the distribution assumptions remain unverified.' : 'Whether the test\'s assumptions hold is not checked.'}`;
  } else if (descriptive) {
    kicker = misfit ? 'Ahead on points, head to head' : 'Ahead on points';
    tone = 'points';
    why = n < 2 ? 'Only one player in this group has results so far.'
      : rt.pooled ? 'Not decided: pooled across runs, this shows who is ahead, never who is better.'
      : voidReason(rt.design) ? 'Not decided: this run\'s test plan is void, so it shows who is ahead, not who is better.'
      : 'Not decided: this run had no recorded test plan, so it shows who is ahead, not who is better.';
  } else {
    const over = settled || rt.finished;
    kicker = misfit ? `No decided leader${over ? '' : ' yet'} · head to head` : `No decided leader${over ? '' : ' yet'}`;
    tone = 'open';
    if (n < 2) why = 'Only one player in this group has results so far.';
    else if (g.pairings.every(x => x.verdict === 'early' || !x.deals)) why = over ? 'Too few deals were played to judge anyone.' : 'Too few deals so far to judge anyone; play continues.';
    else why = `Not decided${over ? '' : ' yet'}: ${decided} of ${plural(g.pairings.length, 'matchup')} ${decided === 1 ? 'has' : 'have'} a result${over ? '' : ', and play continues'}. ${p ? `${plain(who)} leads on points.` : ''}`;
  }
  const facts = [first ? runKind(rt) : ''];
  if (!descriptive && g.pairings.length) facts.push(chip(`${decided} of ${plural(g.pairings.length, 'matchup')} ${rt.track === 'adaptation' ? 'with an exploratory reading' : 'decided'}`, decided && rt.track !== 'adaptation' ? '' : 'quiet'));
  const drifts = g.pairings.filter(x => Core.driftQualified(x)).length;
  if (drifts) facts.push(chip(`${drifts} drifted · needs confirmation`, 'warn', 'Matchups where the test\'s bar was cleared while the edge changed between early and late conversations: not decided, and never a leader.'));
  if (p) facts.push(identityChip(rt, isBot(who) ? (g.players || []).map(pid) : [who], !!(p.identity_qualified || g.identity_qualified)));
  if (ineligible) facts.push(chip(`${ineligible} ineligible`, 'bad', 'Matchups the judge may not decide (identity, audit, protocol or registration): see Matchups.'));
  if (g.leader || g.qualified_leader) {
    const free = g.pairings.some(x => x.assumption && x.assumption.method === 'free');
    facts.push(free && g.leader_assumption === 'free' ? chip('assumption-free test', 'ok', FREE_TIP) + chip('checks passed', 'ok', 'Every check the test needs holds for the leader\'s wins.')
      : assumeChip({ status: g.leader_assumption || 'unknown', label: free && g.leader_assumption === 'unsupported' ? 'a check failed' : Core.ASSUME[g.leader_assumption || 'unknown'], detail: '' }));
  }
  const wrong = first && rt.audit && rt.audit.totals && rt.audit.totals.identity && rt.audit.totals.identity.mismatch;
  if (wrong) facts.push(chip(`wrong model on ${plural(wrong, 'call')}`, 'bad', 'A backend reported a different model from the one asked for: see the audit.'));
  const card = g.leader && tone === 'decided' ? '<span class="mini-card" aria-hidden="true"><b>A</b><i>♠</i></span>'
    : '<span class="mini-card back" aria-hidden="true"></span>';
  const strength = p ? `<b class="${g.leader ? '' : cls(p.strength)}" title="${esc(STRENGTH_TIP)}">${fmtStrength(p.strength)}</b><small>bb/100</small>` : '';
  const observed = p && p.observed_edge != null ? `<div class="o-obs" title="${esc(OBSERVED_TIP)}">observed <b class="${cls(p.observed_edge)}">${signed(p.observed_edge, 0)}</b> vs ${p.observed_opponents === 1 ? 'its opponent' : `${p.observed_opponents} opponents, averaged`}</div>` : '';
  const num = !p ? '' : misfit
    ? `<div class="o-num secondary" title="One number per player does not describe these results: read the direct results"><span class="o-fit-k">Fitted strength, poor fit</span>${strength}${observed}</div>`
    : `<div class="o-num"><span class="o-fit-k" title="${esc(STRENGTH_TIP)}">centred fitted strength</span>${strength}<div class="o-ci">${p.strength_ci95 ? `range ${descRange(p.strength_ci95, p)}` : 'no range yet'} ${basisTag(p)}</div>${observed}</div>`;
  const direct = misfit ? directResults(g) : '';
  return `<div class="outcome-row ${tone} ${misfit ? 'has-direct' : ''}">
    ${card}
    <div class="o-main"><div class="o-kicker">${multi ? `<span class="o-group">${esc(groupName(g))}</span>` : ''}${kicker}</div>
      ${p && !(misfit && !g.leader && !g.qualified_leader) ? `<div class="o-who">${monogram(who, 'lg')}<div class="o-name">${name(who)}</div></div>` : ''}
      ${misfit && !g.leader && !g.qualified_leader ? '<div class="o-lead">One strength per player cannot describe every matchup\'s margin here, so here is each matchup directly.</div>' : ''}</div>
    ${num}
    <div class="o-facts">${facts.filter(Boolean).join('')}</div>
    ${direct}
    ${why ? `<p class="o-why">${why}</p>` : ''}</div>`;
}

// A group's matchups as plain sentences, biggest edge first: the evidence when strengths don't fit.
function directResults(g) {
  const rows = [...g.pairings].filter(x => x.deals).sort((x, y) => Math.abs(y.edge ?? 0) - Math.abs(x.edge ?? 0)).map(x => {
    const up = (x.edge ?? 0) >= 0, lead = up ? x.a : x.b, trail = up ? x.b : x.a;
    const v = verdict(x, plain);
    const word = v.tone === 'won' ? 'beat' : v.tone === 'even' ? 'even with' : x.edge ? 'ahead of' : 'level with';
    const full = matchupId(x);
    return `<li><a ${full ? `href="${matchLink(full)}"` : ''}>${monogram(lead, 'sm')}<span class="dr-n">${short3(lead)}</span><i>${word}</i><span class="dr-n">${short3(trail)}</span>
      <b class="dr-e">${Math.abs(x.edge ?? 0).toFixed(0)}<small> bb/100</small></b>${v.decided ? '<span class="decided-badge">Decided</span>' : `<span class="chip quiet">${esc(v.short.toLowerCase())}</span>`}</a></li>`;
  }).join('');
  return rows ? `<ol class="o-direct">${rows}</ol>` : '';
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
  else if (rt.track === 'adaptation') test = `<b>Adaptation track.</b> The same players keep one conversation for ${plural(d.round_deals || 0, 'deal')}, so models remember and can adapt to the opponent.
    Each matchup is checked after every round of ${plural(d.replicates || 1, 'conversation')}, at most ${plural(d.max_rounds || 1, 'time')}, and stops on a reading of the <b>model-based</b> test
    (a conversation-level t and a sign-flip test with bars calibrated for a family of result shapes). Its ${Math.round((d.alpha || 0.05) * 100)}% error bound holds only under those assumptions,
    so its readings are <b>exploratory</b> and <b>never official</b>: they rank no one, name no leader and never change the official standings.`;
  else if (v) test = `This run's test plan is void (${esc(v)}), so nothing in it is decided: every order and verdict is <b>ahead on points</b>.`;
  else if (rt.unregistered) test = 'This run was played before its test plan was recorded, so nothing in it is decided: every order and verdict is <b>ahead on points</b>, and any sample size quoted is retrospective.';
  else if (d.round_deals && officialFree(rt)) {
    const t = officialFree(rt);
    test = `Each matchup is checked after every ${d.round_deals} deals, at most ${plural(d.max_rounds, 'time')}${d.min_conversations ? `, never before ${plural(d.min_conversations, 'independent conversation')} (match pairs)` : ''}.
    The <b>official verdict</b> comes from an anytime-valid sequential test for bounded results${t.name ? ` (${esc(t.name)})` : ''}: at every check, the chance of any wrong verdict across all
    ${plural(d.n_pairings || rt.pairings.length, 'matchup')} is at most ${Math.round((d.alpha || 0.05) * 100)}%, assuming that conversations are independent and that with no edge each one's expected edge, given the ones before, is zero or less, and nothing about the shape of the results${t.bound != null ? `; each conversation's result is bounded by ±${approx(t.bound)} bb/100` : ''}.
    A win says which model is better only if each name served one unchanged model for the whole run (pin dated model snapshots where the provider offers them). The drift check compares early and late conversations: when it finds nothing, that reads <b>no drift detected</b>, not proof that the models held still.
    The earlier <b>model-based</b> test (whose error rate rests on distribution assumptions) is still shown beside a verdict, labelled secondary; it never decides.`;
  }
  else if (d.round_deals) test = `Each matchup is checked after every ${d.round_deals} deals, at most ${plural(d.max_rounds, 'time')}${d.min_conversations ? `, never before ${plural(d.min_conversations, 'independent conversation')} (match pairs)` : ''}.
    A matchup is <b>decided</b> when its edge clears a bar set for at most a ${Math.round((d.alpha || 0.05) * 100)}% chance of any wrong verdict across all ${plural(d.n_pairings || rt.pairings.length, 'matchup')},
    <b>under the test's assumptions</b>. Where its diagnostics pass, the verdict says so, but the distribution assumptions behind the error bound remain unverified;
    a verdict outside the range the test was calibrated for is marked so, and does not make a decided leader.`;
  else test = 'Decided results come only from a planned sequential test; everything else is ahead on points.';
  const target = rt.pooled ? '' : targetLine(rt, rt.pairings);
  const summary = rt.pooled ? 'Pooled across runs: described, never decided'
    : rt.track === 'adaptation' ? 'Adaptation track · model-based test · exploratory, not official'
    : v ? 'Test plan void: described, never decided'
    : rt.unregistered ? 'No recorded test plan: described, never decided'
    : d.round_deals && officialFree(rt) ? `Assumption-free sequential test · ${Math.round((d.alpha || 0.05) * 100)}% error bound if conversations are independent and each model stays the same`
    : d.round_deals ? `Planned sequential test · ${Math.round((d.alpha || 0.05) * 100)}% error bound under stated assumptions` : 'How results are decided';
  const notes = (rt.notes || []).map(n => `<li>${esc(n)}</li>`).join('');
  return `<details class="plan-panel" ${open ? 'open' : ''}><summary><span class="pp-title">How this is judged</span><span class="pp-sum">${summary}</span></summary>
    <div class="pp-body">
      <p>${test}</p>
      ${target}
      ${populationLine(rt) ? `<p class="pop-line">${populationLine(rt)}</p>` : ''}
      <dl>
        <dt>Decided</dt><dd>A result of the planned test above. Only these carry the badge. A decided win names the better model only if each name served one unchanged model for the whole run (its expected edge stayed the same); the drift check below can catch a change, but finding none does not prove there was none.</dd>
        <dt>Ahead on points</dt><dd>Everything else: the current estimate, recomputed as hands finish. It shows who leads, not who is better.</dd>
        <dt>Fitted strength</dt><dd>Centred fitted strength, in bb/100, from one least-squares fit over every deal (matchups with more deals weigh more).
          The strengths sum to zero in each group, so only differences mean anything: the gap between two players is the edge the fit predicts when they meet.
          It is not a player's winnings against an average opponent. Where a matchup disagrees with what the strengths predict, the matchup's own result is the evidence.</dd>
        <dt>Observed</dt><dd>Observed opponent-weighted winnings: the average of a player's direct matchup edges in bb/100, one per opponent it met, each opponent weighted equally.</dd>
        <dt>Ranges</dt><dd>An official matchup's first range is the test's <b>anytime-valid</b> interval: it holds whenever play stops, so it is the one to read.
          Every other range (strengths, the reference, ordinary matchup ranges) is <b>fixed-time, descriptive</b>: computed as if the number of deals had been fixed,
          while play stops on the results and the front runner is picked by them, so it carries no 95% guarantee.
          <b>across conversations</b>: resamples whole match pairs; <b>within conversations</b>: resamples blocks of deals inside single match pairs,
          so it leaves out how much fresh matches differ, and the real uncertainty is larger.</dd>
        <dt>Needs confirmation</dt><dd>The test's bar was cleared, but the edge drifted between early and late conversations: a player was ahead for part of the run. Never decided, never a leader.</dd>
        <dt>Players who met</dt><dd>Strengths compare only players linked by games; players who never met (for example the baseline bots and a model-only tournament) are shown separately, picked above the results.</dd>
        <dt>Identity</dt><dd><b>model confirmed</b>: the model's own tool reported which model answered, and it is the one asked for; <b>wrong model</b>: it reported
          a different one; <b>model unconfirmed</b>: it reported none, so only the configured name is known.</dd>
        <dt>Fit</dt><dd><b>off the fit</b>: a matchup's result disagrees with what the strengths predict; <b>insufficient evidence to assess fit</b>: too few
          matchups link these players to test it (always so with two players).</dd>
        <dt>Session Elo</dt><dd>A different number: how often it wins a session, where ${esc(sessionText(rt))}. For reference, never used to rank.</dd>
      </dl>
      ${notes ? `<div class="pp-notes"><b>Notes from the ratings</b><ul>${notes}</ul></div>` : ''}
    </div></details>`;
}

// What a session is in this run, as the ratings define it (older ratings: their session length).
function sessionText(rt) {
  if (rt.session?.text) return rt.session.text;
  const n = rt.session_deals || 10;
  return n === 1 ? 'a session is one duplicate deal' : `a session is ${n} deals of one conversation`;
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
  const rej = pop.rejected ? `; ${plural(pop.rejected, 'hand')} rejected by the verifier ${pop.rejected === 1 ? 'is' : 'are'} left out` : '';
  if (what === 'stats') {
    const st = pop.stats;
    if (st && /commit|verif/i.test(st.kind || st.population || st.label || '')) return `Counts only verified, committed hands${st.hands != null ? ` (${st.hands.toLocaleString()})` : ''}${rej}; hands still being played are left out.`;
    if (st) return `Counts: ${esc(st.label || st.kind || st.population)}.`;
    return `Counts every finished hand on record, including hands still waiting for their other play${pop.rejected ? ', and any the verifier rejected' : ''}: these are descriptive, not checked results.`;
  }
  const hands = pop.committed ?? (rt.deals ? 2 * rt.deals : null);
  return hands ? `Counts ${hands.toLocaleString()} verified, committed hands (both plays of each deal replayed and checked)${rej}${pop.provisional ? `; ${plural(pop.provisional, 'live hand')} not counted yet` : ''}.` : '';
}

// The run's audit in one line: the aside of its folded section.
const idCount = id => (id.match || 0) + (id.unattested || 0) + (id.mismatch || 0);
function auditLine(rt) {
  const a = rt.audit;
  if (!a || !a.totals || !a.totals.calls) return '';
  const t = a.totals, id = t.identity || {};
  const idn = idCount(id);
  const bits = [`${t.decisions.toLocaleString()} model decisions`];
  bits.push(a.recorded ? `${t.tool_rejected} tool-use rejections · ${t.unverified} unaudited` : 'turn audit not recorded');
  bits.push(`${plural(t.fallbacks, 'fallback')}`);
  if (idn) bits.push(`model confirmed on ${Math.round(100 * (id.match || 0) / idn)}% of calls${id.mismatch ? ` · <b class="neg">${id.mismatch} wrong model</b>` : ''}`);
  return bits.join(' · ');
}

const fmtN = n => n ? n.toLocaleString() : '<span class="dim">0</span>';
const ID_TONE = { match: 'ok', mismatch: 'bad' };
function auditSection(rt) {
  const a = rt.audit;
  if (!a || !a.players || !Object.keys(a.players).length) return '';
  const list = Object.entries(a.players).filter(([, x]) => x.calls || x.decisions);
  if (!list.length) return '';
  const cells = x => {
    const id = x.identity || {};
    const idn = idCount(id);
    const status = Core.identityOf(null, x);
    const saw = x.observed && x.observed.length ? ` · reported ${esc(x.observed.join(', '))}` : '';
    const asked = status === 'mismatch' && x.requested && x.requested.length ? `, asked for ${esc(x.requested.join(', '))}` : '';
    const audited = x.verified + x.opaque + x.unverified;
    const seen = audited || x.tool_rejected;
    return {
      id: idn ? `<span class="idst ${ID_TONE[status] || 'warn'}">${esc(Core.IDENTITY[status] || '–')}</span>
        <small class="nof">${id.match || 0}/${idn} calls confirmed${id.mismatch ? ` · ${id.mismatch} wrong` : ''}${saw}${asked}</small>` : '<span class="dim">–</span>',
      status,
      audit: !seen ? '<span class="dim" title="Calls from before turns were audited">not recorded</span>'
        : `${x.verified}<small class="nof">verified${x.opaque ? ` · ${x.opaque} opaque` : ''}${x.audit_missing ? ` · ${x.audit_missing} unrecorded` : ''}</small>`,
      tool: seen ? fmtN(x.tool_rejected) : '<span class="dim">–</span>',
      unv: seen ? fmtN(x.unverified) : '<span class="dim">–</span>',
    };
  };
  const rows = list.map(([p, x]) => {
    const c = cells(x);
    return `<tr><td>${player(p, '', { details: false })}</td><td class="num">${fmtN(x.decisions)}<small class="nof">${fmtN(x.calls)} calls</small></td>
      <td>${c.id}</td><td class="num">${c.audit}</td><td class="num">${c.tool}</td><td class="num">${c.unv}</td>
      <td class="num">${fmtN(x.retries)}<small class="nof">${x.resized} resized</small></td><td class="num">${fmtN(x.fallbacks)}</td>
      <td class="num">${fmtN(x.failed_tries)}<small class="nof">${x.rate_limited} rate-limit waits</small></td></tr>`;
  }).join('');
  const head = `<thead><tr><th>Model</th><th class="num">Decisions</th><th title="Did the model's own tool report which model answered, and is it the one asked for?">Identity</th>
    <th class="num" title="Turns whose record of tool use was read in full (verified), or that the backend cannot expose (opaque)">Turn audit</th>
    <th class="num" title="Replies that used a tool: rejected, and the model was asked again">Tool use</th><th class="num" title="Turns whose record could not be read: rejected and asked again">Unaudited</th>
    <th class="num" title="Decisions asked again after an invalid answer; resized: a legal intent adjusted to a legal size">Retries</th>
    <th class="num" title="No playable answer after every retry: the game checked or folded for it">Fallbacks</th><th class="num" title="Calls to the model's tool that failed and were tried again">Failed tries</th></tr></thead>`;
  const cards = list.map(([p, x]) => {
    const c = cells(x);
    return `<details class="acard"><summary>${monogram(p, 'sm')}<span class="ac-name">${name(p)}</span>
        <span class="ac-sum"><span class="idst ${ID_TONE[c.status] || 'warn'}">${esc(Core.IDENTITY[c.status] || '–')}</span> · ${fmtN(x.decisions)} decisions${x.fallbacks ? ` · ${plural(x.fallbacks, 'fallback')}` : ''}</span></summary>
      <dl><dt>Identity</dt><dd>${c.id}</dd><dt>Calls</dt><dd>${fmtN(x.calls)}</dd><dt>Turn audit</dt><dd>${c.audit}</dd>
        <dt>Tool use rejected</dt><dd>${c.tool}</dd><dt>Unaudited</dt><dd>${c.unv}</dd><dt>Retries</dt><dd>${fmtN(x.retries)} <small class="nof">${x.resized} resized</small></dd>
        <dt>Fallbacks</dt><dd>${fmtN(x.fallbacks)}</dd><dt>Failed tries</dt><dd>${fmtN(x.failed_tries)} <small class="nof">${x.rate_limited} rate-limit waits</small></dd></dl></details>`;
  }).join('');
  return section('Audit', auditLine(rt) || 'how each decision was made, from the call records',
    `<div class="only-wide scroll"><table class="audit-t">${head}<tbody>${rows}</tbody></table></div><div class="only-narrow acards">${cards}</div>`,
    'From each match\'s call log and decisions. Bots make no calls. Each decision\'s own record (tries, identity, audit) is in the replay under <b>why?</b>.',
    { id: 'audit', fold: true });
}

// ---- tracks: the official standings, and the adaptation track (the same players in long conversations) ----
function trackBar(run, rest = '') {
  const set = trackSet(run);
  if (set.length < 2 && !(set[0] && set[0].track === 'adaptation')) return '';
  const tab = r => `<a role="tab" class="track ${r.run === run ? 'on' : ''} t-${r.track}" aria-selected="${r.run === run}" href="${runLink(r.run, rest)}">
    <span class="tr-t">${esc(TRACK_TITLE[r.track] || r.track)}</span><small>${r.track === 'adaptation' ? 'exploratory · not official' : 'fresh conversation every deal · decides ranks'}</small></a>`;
  return `<div class="track-bar" role="tablist" aria-label="Track">${set.map(tab).join('')}</div>`;
}
function trackNote(rt) {
  const set = trackSet(state.run);
  if (rt.track === 'adaptation') {
    const d = rt.design || {};
    return `<details class="track-note adaptation" role="note" ${phone() ? '' : 'open'}><summary>Exploratory track: models remember the session; nothing here is official</summary><span class="tn-k">${chip('adaptation · exploratory · not official', 'warn')}</span>
      <p>Here the same players keep one conversation for ${d.round_deals ? plural(d.round_deals, 'deal') : 'many deals'}, so a model remembers earlier hands and can adapt to its opponent.
      Its readings come from a <b>model-based</b> test, which assumes the results follow a family of shapes it was calibrated for; its error rate is not established here, so every result is <b>exploratory</b>: a lead to look into, not a finding.
      Nothing here is official: it never ranks anyone or names a leader. ${set.some(r => r.track === 'official') ? `The <a href="${runLink(siblingTrack(state.run, 'official').run)}"><u>official standings</u></a> use a fresh conversation for every deal and a test that assumes nothing about the shape of the results (it needs independent conversations and models that stay the same).` : ''}</p></details>`;
  }
  if (set.length < 2) return '';
  return `<details class="track-note" role="note" ${phone() ? '' : 'open'}><summary>Official track: decides ranks · the adaptation track is exploratory</summary><p>Two tracks, same players and cards. <b>Official</b>: a fresh conversation for every deal, judged by a test that assumes nothing about the shape of the results (it needs independent conversations and models that stay the same); only it decides ranks and leaders.
    <b>Adaptation</b>: long conversations, so models remember and adapt to the opponent; read by a model-based test, exploratory and never official.</p></details>`;
}
// On the official page: the adaptation track's own standings, compact and clearly not official.
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
    return section(TRACK_TITLE.adaptation, 'model-based · not official', '<p class="soft">Working out the adaptation track\'s standings…</p>', '', { id: 'adaptation' });
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
  const body = ps.length ? `<div class="ad-list">${rows}</div>${called ? `<div class="ad-called"><b>Exploratory readings</b> <span class="soft small">(model-based test, error rate not established)</span><ul>${called}</ul></div>` : ''}
    ${mbl.length ? `<p class="small soft">${mbl.map(plain).join(', ')} ahead in every matchup by the model-based test: exploratory, not a leader.</p>` : ''}`
    : '<p class="soft">No finished deals on the adaptation track yet.</p>';
  return section(TRACK_TITLE.adaptation, `${chip('exploratory · not official', 'warn')}`,
    body + `<a class="ad-open" href="${runLink(sib.run)}">Open the adaptation track ›</a>`,
    'Ordered by points (descriptive). Long conversations, so models can adapt to the opponent; its readings come from a model-based test, are exploratory and never change the official standings above.', { id: 'adaptation' });
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
  let html = runHead(r, rt) + trackBar(state.run) + staleNote(raw) + '<div class="top-stack">' + trackNote(rt) + intro(view) + (cohort ? cohortBar(all, cohort) : '');
  const waiting = cohort && cohort !== all[0] ? [] : rt.roster.filter(x => !rt.players.some(p => p.player === x.player));
  if (!rt.players.length) {
    html += `<div class="card-box empty"><div class="big-suits">♠ ♥ ♦ ♣</div><h2>The cards are in the air</h2>
      <p>A deal counts once both of its plays are finished. ${state.index.some(m => m.started) ? `Watch the first ones on the <a href="${runLink(state.run, '/m')}"><u>games</u></a> page.` : ''}</p>
      ${waiting.length ? `<div class="waiting-list">${waiting.map(x => player(x.player, `<div class="small soft">${x.status === 'playing' ? 'playing its first deals' : 'waiting to play'}</div>`)).join('')}</div>` : ''}</div>`;
    $app.innerHTML = html + '</div>' + planPanel(rt);
    bindBoard();
    return;
  }
  html += outcomes(view) + '</div>';
  html += referenceSection(view, rt);
  html += `<div class="results-stack"><div class="under-outcome rs-plan">${planPanel(rt)}</div>`;
  html += `<div class="rs-standings">${standings(view, waiting)}</div>`;
  html += `<div class="rs-pairings">${pairingsSection(view)}</div></div>`;
  html += adaptationTeaser(rt);
  if (r?.quiz) html += '<div class="quiz-slot"></div>';
  html += `<div class="deep-head"><span>More detail</span><small>the full grid, play style and audit, folded away</small></div>`;
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
  $app.querySelector('.plan-panel')?.addEventListener('toggle', e => store.set('plan-open', e.target.open));
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
  $app.innerHTML = runHead(r, none) + intro(null) + `<div class="card-box pending-board" aria-live="polite">
    <div class="deal-in-suits" aria-hidden="true"><span>♠</span><span class="red">♥</span><span class="red">♦</span><span>♣</span></div>
    <h2>Checking every hand and working out the standings</h2>
    <p>Each hand is replayed from its seed and checked before it counts. For a big run that takes a few seconds the first time; this page fills in by itself. Meanwhile, the raw chip score of each matchup (unchecked, and not a verdict):</p>
    ${rows ? `<div class="pend-list">${rows}</div>` : ''}
    <a class="cta" href="${runLink(state.run, '/m')}"><span class="cta-icon">▶</span>Watch the hands meanwhile</a></div>`;
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
  const chip = c => {
    const n = Core.cohortName(c, labelText);
    const count = c.kind === 'history' ? plural(c.members.length, 'player') : `${plural(c.members.length, 'player')} · ${plural(c.deals, 'deal')}`;
    return `<a role="tab" class="cohort ${c === cur ? 'on' : ''} k-${c.kind}" aria-selected="${c === cur}" href="${link(c)}" title="${esc(n.title)}">
      <span class="cb-mono">${c.members.slice(0, 4).map(p => monogram(p, 'xs')).join('')}</span>
      <span class="cb-t">${esc(c === all[0] && c.kind !== 'bots' ? n.title : n.short)}</span><small>${esc([n.sub, count].filter(Boolean).join(' · '))}</small></a>`;
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
const REF_LIMITS = 'A fixed, versioned heuristic: tight-aggressive, position-aware, equity and pot-odds based, with a fixed bluff frequency. It is not a solver and does not adapt, so beating it shows sound play against a simple fixed policy, not optimal play.';
function referenceSection(view, rt) {
  const ref = rt.players.map(p => p.player).find(isReference) || Object.keys(state.kinds).find(isReference);
  const vs = view.players.filter(p => p.vs_reference && !isReference(p.player));
  const models = view.players.filter(p => !isBot(p.player));
  if (!vs.length) {
    if (!models.length || view.cohort?.kind === 'history') return '';
    return `<div class="ref-none card-box"><span class="suit">♣</span><div><b>No fixed yardstick in this group.</b> These strengths compare these models only with each other, not with a fixed heuristic yardstick.
      <details class="run-own"><summary>Run your own</summary><p>To measure models against a fixed heuristic yardstick, add the reference bot to the plan's players (<code>bot = "reference"</code> in the players file). The fundamentals quiz is a separate check of single decisions.</p></details></div></div>`;
  }
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
  const about = `<details class="ref-about"><summary>What the reference bot is${results.length ? ', and its own record' : ''}</summary>
    <p>${esc(info.description || REF_LIMITS)}${limits.length ? ` Limits: ${limits.join('; ')}.` : ''}${info.version ? ` Version ${esc(info.version)}.` : ''}</p>
    ${results.length ? `<p><b>Its own record${info.results ? '' : ' in this run'}:</b> ${results.join(' · ')} bb/100 (its edge over each).</p>` : ''}</details>`;
  return section('Against the reference bot', 'a fixed heuristic yardstick: bb/100 won against it, so above 0 beats it',
    `<div class="ref-list">${rows}</div>`, 'A fixed heuristic, not a solver or an expert: beating it shows sound play against one simple policy, not how good a player is at poker in general.', { id: 'reference', after: about });
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
            <span>${b.range ? `Places ${range(b)}: ` : ''}the order of these ${b.players.length === 2 ? 'two' : b.players.length} is <b>not decided</b> · listed by points</span></div>
            ${b.players.map(p => standingRow(rt, g, p, ax, true, true)).join('')}</div>`).join('');
    const extra = k === shown.length - 1 ? waitingRows(waiting, ranked) : '';
    const title = multi ? groupName(g) : ranked ? 'Standings' : 'Order on points';
    const control = ranked ? `<div class="order-bar" role="group" aria-label="Order"><span>Order</span>
        <button class="toggle" data-order="decided" aria-pressed="${!points}" title="Decided ranks first; players whose order is not decided are grouped">Decided</button>
        <button class="toggle" data-order="points" aria-pressed="${points}" title="Everyone by fitted strength: descriptive, not a ranking">By points</button></div>` : '';
    const aside = !ranked ? 'provisional: no rank here is decided'
      : points ? 'sorted by points (descriptive); a numeral is a decided rank'
      : 'decided ranks first; the rest grouped where their order is not decided';
    const misfit = g.misfits ? ` <span class="flag" title="Some matchups disagree with what these strengths predict; see Matchups">${g.misfits} off the fit</span>` : '';
    const count = rt.cohort && rt.archive && rt.archive.deals !== rt.deals
      ? ` · ${plural(rt.deals, 'duplicate deal')} in this group <span class="soft">(${plural(rt.archive.deals || 0, 'deal')} across the whole ${state.runs.find(x => x.run === state.run)?.all ? 'archive' : 'run'})</span>`
      : ` · ${plural(rt.deals, 'duplicate deal')}`;
    out += section(title, `${aside}${multi ? '' : count}${misfit}`,
      `${control}<div class="board-list ${ranked ? '' : 'unranked'}"><div class="board-head">${ranked ? '<span title="Decided rank"><span class="hide-xs">Rank</span><span class="show-xs">#</span></span>' : ''}<span>${ranked ? 'Player' : 'Player, by points'}</span>
         <span class="r" title="${esc(STRENGTH_TIP)}">Fitted strength</span>
         <span class="axis-head"><span>${signed(-ax.lim, 0)}</span><span>0</span><span>${signed(ax.lim, 0)}</span></span>
         <span class="r hide-sm" title="${esc(OBSERVED_TIP)}">Observed</span><span class="r hide-sm">Net</span><span class="r hide-sm" title="Share of sessions won, where ${esc(sessionText(rt))} (secondary; not used to rank)">Sessions</span></div>
       ${rows}${extra}</div>`,
      `${points && ranked ? 'Sorted by fitted strength, which describes results and decides nothing: a numeral is a rank the matchups decided. ' : ''}Fitted strengths are centred: the difference of two predicts their matchup edge. Observed is what each won: the average of its matchup edges, one per opponent.`);
  });
  if (!out && waiting.length) out = section('Standings', '', `<div class="board-list unranked">${waitingRows(waiting, false)}</div>`);
  const out_ = rt.players.filter(p => !p.in_fit);
  if (out_.length) {
    out += section(rt.cohort?.kind === 'history' ? 'Earlier variants' : 'Not in the standings', 'played, but left out of the fit', `<div class="board-list unranked left-out">${out_.map(p => `<div class="brow quarantined">
      ${player(p.player, `<div class="small soft why-out">${esc(p.unrated_reason || p.quarantine_reason || 'none of its opponents is rated')}</div>`)}
      <span class="r strength soft">${p.hands ? `<small>${plural(p.hands, 'hand')}</small>` : ''}</span><span class="axis"></span>
      <span class="hide-sm"></span><span class="r hide-sm soft">${p.net_bb != null ? signed(p.net_bb, 0) + ' bb' : ''}</span><span class="hide-sm"></span></div>`).join('')}</div>`,
      'Left out when its exact model or reasoning level is unknown (its results could mix more than one model), or when it only met players who are left out.');
  }
  return out;
}

function cycleBlock(rt, g, b, ax) {
  const names = b.players.map(p => p.player);
  const results = Core.cycleResults(names, g.pairings || rt.pairings);
  return `<div class="unres cycle" role="group" aria-label="Decided matchups form a cycle"><div class="unres-head"><span class="rank prov" aria-hidden="true">↻</span>
      <span><b>Decided matchups form a cycle</b>, so these ${b.players.length === 2 ? 'two' : b.players.length} have no order · listed by points</span></div>
    ${results.length ? `<div class="cycle-results small soft">${results.map(x => `${plain(x.winner)} beat ${plain(x.loser)}`).join(' · ')}</div>` : ''}
    ${b.players.map(p => standingRow(rt, g, p, ax, true, true)).join('')}</div>`;
}

function identityFlag(rt, p) {
  if (isBot(p.player)) return '';
  const a = rt.audit && rt.audit.players && rt.audit.players[p.player];
  const st = Core.identityOf(p, a) || Core.identityOf(p, (rt.playerAudit || {})[p.player]);
  if (!st || st === 'match') return '';
  const tip = st === 'mismatch' ? 'Its tool reported a different model from the one asked for.'
    : st === 'unattested' ? 'Its tool never reported which model answered; only the configured name is known.'
    : 'Only some of its calls report the model asked for; the rest report none.';
  return `<span class="flag ${st === 'mismatch' ? 'bad' : 'quiet'}" title="${esc(tip)}">${esc(Core.IDENTITY[st])}</span>`;
}

function standingRow(rt, g, p, ax, ranked, inBlock = false) {
  const ci = p.strength_ci95;
  const decided = p.rank_kind === 'decided' && p.rank != null;
  const tied = (p.rank_tied_with || []).filter(x => x !== p.player && g.members.includes(x));
  const notes = [];
  if (!p.identity_resolved) notes.push(`<span class="flag" title="${esc(p.quarantine_reason || '')}">identity unresolved</span>`);
  const idf = identityFlag(rt, p);
  if (idf) notes.push(idf);
  if (!inBlock && tied.length && tied.length < g.members.length - 1) notes.push(`<span class="small soft">difference with ${tied.map(plain).join(', ')} unresolved</span>`);
  const obs = p.observed_edge != null ? signed(p.observed_edge, 0) : null;
  const mob = `<div class="mob-line small soft">${obs != null ? `<span class="${cls(p.observed_edge)}" title="${esc(OBSERVED_TIP)}">${obs} observed</span> · ` : ''}${p.net_bb != null ? `<span class="${cls(p.net_bb)}">${signed(p.net_bb, 0)} bb</span> · ` : ''}${plural(p.hands || 0, 'hand')}</div>`;
  const leader = g.leader === p.player;
  const b = Core.basisOf(p);
  const whisker = `<div class="whisker ${b ? b.key : ''}" title="${ci ? `Range ${ciText(ci)} bb/100, ${esc(Core.ciKind(p))}${b ? `, ${b.short}` : ''}` : 'no range yet: too few deals'}">
      <span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
      ${p.strength != null ? `<span class="point ${leader ? 'lead' : ''}" style="left:${ax.x(p.strength)}"></span>` : ''}</div>`;
  const elo = p.session_elo != null ? `<span title="Session Elo ${p.session_elo}${p.session_elo_ci95 ? ` (${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]})` : ''}">${p.session_w ?? 0}–${p.session_t ?? 0}–${p.session_l ?? 0}</span>
      <small>Elo ${p.session_elo}${p.session_elo_ci95 ? ` · ${p.session_elo_ci95[0]}–${p.session_elo_ci95[1]}` : ''}</small>` : '';
  const half = ci ? Math.round((ci[1] - ci[0]) / 2) : null;
  return `<div class="brow ${leader ? 'lead' : ''}">
    ${ranked ? `<span class="rank ${decided ? 'sep' : 'prov'}" title="${decided ? 'Rank decided by the matchups' : 'Not decided: placed on points only'}">${decided ? p.rank : inBlock ? '' : '–'}</span>` : ''}
    ${player(p.player, `${notes.length ? `<div class="row-notes">${notes.join(' ')}</div>` : ''}${mob}`)}
    <span class="r strength"><b>${fmtStrength(p.strength)}</b>${ci ? `<small title="${esc(Core.DESCRIPTIVE_TIP)}${b ? ' ' + esc(b.long) : ''}">±${half}${b && b.key !== 'across' ? `<i class="basis-mark ${b.key}">${b.key === 'within' ? ' within conv.' : ' partly within'}</i>` : ''}</small>` : '<small>no range yet</small>'}</span>
    <span class="axis">${whisker}</span>
    <span class="r hide-sm obs" title="${esc(OBSERVED_TIP)}">${obs != null ? `<span class="${cls(p.observed_edge)}">${obs}</span><small>vs ${plural(p.observed_opponents || 0, 'opponent')}</small>` : '<small>–</small>'}</span>
    <span class="r hide-sm"><span class="${cls(p.net_bb)}">${p.net_bb != null ? signed(p.net_bb, 0) + ' bb' : ''}</span><small>${plural(p.hands || 0, 'hand')}</small></span>
    <span class="r hide-sm sess">${elo}</span></div>`;
}

function waitingRows(waiting, ranked) {
  return waiting.map(x => `<div class="brow waiting">${ranked ? '<span class="rank">·</span>' : ''}
    ${player(x.player, `<div class="small soft">${x.status === 'playing' ? 'playing its first deals' : x.status === 'done' ? 'no finished deals' : 'waiting to play'}</div>`)}
    <span class="r strength soft">–</span><span class="axis"></span><span class="hide-sm"></span><span class="hide-sm"></span><span class="hide-sm"></span></div>`).join('');
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
  if (v.decided || (p.assumption && p.assumption.status === 'unsupported' && p.assumption.method === 'free')) bits.push(assumeChip(p.assumption, { compact }));
  const drift = p.drift && (typeof p.drift === 'string' ? { status: p.drift } : p.drift);
  if (v.drift) bits.push(chip('needs confirmation', 'warn', v.long));
  else if (drift && /^drift detected/.test(drift.status || '')) bits.push(chip('drift detected', 'warn', drift.meaning || 'The edge changed between early and late conversations: the verdict means a player was ahead for part of the run.'));
  if (p.identity_qualified && !Object.values(p.attestation || {}).some(at => ['unattested', 'mismatch'].includes(at && (typeof at === 'string' ? at : at.status))))
    bits.push(chip('model unconfirmed', 'quiet', 'A player\'s tool never reported which model answered, so this result is qualified: only the configured name is known.'));
  for (const [pl, at] of Object.entries(p.attestation || {})) {
    const st = at && (typeof at === 'string' ? at : at.status);
    if (st === 'unattested' || st === 'mismatch') bits.push(chip(`${short3(pl)}: ${st === 'mismatch' ? 'wrong model' : 'unattested'}`, st === 'mismatch' ? 'bad' : 'quiet',
      `${(at && at.meaning) || (st === 'mismatch' ? 'Its backend reported another model.' : 'Its backend cannot report which model answered.')}${at && at.policy ? ` (${at.policy})` : ''}`));
  }
  if (!compact && src) bits.push(chip(src.short, src.kind === 'official' ? 'ok' : 'quiet', src.text));
  if (v.secondary && !compact) bits.push(chip(`model-based: ${esc(v.secondary.short.toLowerCase())}`, 'quiet secondary',
    'The older model-based test (a t / sign-flip gate whose error rate rests on distribution assumptions). Shown for comparison only: it never decides.'));
  return bits.join('');
}

// Each pairing: the edge with its range, its verdict, and how it compares with the strengths' prediction.
function pairingsSection(rt) {
  if (!rt.pairings.length) return '';
  const order = { a: 0, b: 0, even: 1, decided_with_drift: 2, exhausted: 2, ineligible: 3, open: 4, descriptive: 4, early: 5 };
  const rank = p => Core.driftQualified(p) ? 2 : order[p.verdict] ?? 6;
  const ax = axisFor(rt.pairings.map(p => {
    const up = (p.edge ?? 0) >= 0, o = officialRange(p, up);
    return { strength: Math.abs(p.edge ?? 0), strength_ci95: o ? o.ci : p.ci95 && (up ? p.ci95 : [-p.ci95[1], -p.ci95[0]]) };
  }));
  const descriptive = rt.pooled || rt.unregistered;
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
    const checks = p.looks_planned && !descriptive ? (p.looks_used ? ` · checked ${p.looks_used} of ${p.looks_planned}` : ' · not checked yet') : '';
    const conv = !descriptive && Core.conversationsShort(p) ? ` · ${p.conversations} of ${p.min_conversations} conversations a verdict needs` : '';
    const word = v.tone === 'won' ? 'beats' : v.tone === 'even' ? 'even with' : edge ? 'ahead of' : 'level with';
    const fit = p.fit;
    const fitLine = fit && fit.warn ? `<div class="fitline"><span class="flag">off the fit</span> The strengths predict ${signed(up ? fit.fitted : -fit.fitted, 0)} for ${plain(lead)}; this matchup's own result is ${signed(edge, 0)}${fit.estimated ? '' : ` (residual ${signed(up ? fit.residual : -fit.residual, 0)})`}. The direct result is the evidence.</div>` : '';
    const untestable = fit && fit.untestable && rt.players.length > 2
      ? `<span class="chip quiet" title="Too few matchups link these players to test whether this one fits the strengths">fit: insufficient evidence to assess</span>` : '';
    const no = Core.ineligible(p);
    const noLine = no ? `<div class="fitline bad"><span class="flag bad">ineligible</span> ${esc(no.reasons.join('; ') || 'the judge may not decide this matchup')}</div>` : '';
    return `<div class="prow ${v.decided ? 'decided' : ''} ${v.tone === 'bad' ? 'ineligible' : ''}">
      <div class="duo-names">${player(lead, '', { details: false })}<span class="over">${word}</span>${player(trail, '', { details: false })}</div>
      <div class="edge"><b>${edge.toFixed(0)}</b><small>bb/100</small>${off ? `<small class="ci-line">${off.html}</small>` : ''}${desc ? `<small class="ci-line ci-second">${off ? 'ordinary ' : ''}${descRange(desc, p)}</small>` : ''}${desc ? basisTag(p) : ''}</div>
      <div class="edge-bar"><div class="whisker"><span class="zero"></span>${ci ? `<span class="range" style="left:${ax.x(ci[0])};width:calc(${ax.x(ci[1])} - ${ax.x(ci[0])})"></span>` : ''}
        <span class="point ${v.tone === 'won' ? 'pos' : ''}" style="left:${ax.x(edge)}"></span></div></div>
      <div class="vcell">${verdictCell(p, v)}<small>${plural(p.deals || 0, 'deal')}${checks}${conv}</small>${untestable}</div>
      ${full ? `<a class="watch" href="${matchLink(full)}" title="Watch this matchup">Watch ›</a>` : '<span></span>'}
      ${noLine}${fitLine}</div>`;
  }).join('');
  return section('Matchups', 'who beats whom: the direct evidence, in bb/100 over duplicate deals', `<div class="pair-list">${rows}</div>`,
    `${descriptive ? 'Ahead on points only: no verdict here is a decision.' : rt.track === 'adaptation' ? 'Readings here are the model-based test\'s: they stop play on this track, but they are exploratory and none is official. Hover one for what it rests on.' : 'Decided results are the planned test\'s, and name the better model only if each name served one unchanged model for the whole run; the rest is ahead on points. Hover a verdict for what it rests on.'}
     ${official ? 'The first range is the official test\'s anytime-valid interval, which holds whenever play stops; the ordinary range under it is fixed-time and descriptive.' : 'Ranges are fixed-time and descriptive: play stops on the results, so they carry no 95% guarantee.'}`);
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
      return `<td class="cell ${outside ? 'outside' : ''}" style="${bg}"><a ${full ? `href="${matchLink(full)}"` : ''} title="${short(a)} against ${short(b)}: ${esc(verdict(c.p, plain).short)}${outside ? ' (outside the calibrated range)' : ''}">
        <span class="v ${won ? cls(c.v) : 'soft'}">${signed(c.v, 0)}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
  const multi = rt.groups.filter(g => g.ranked.length).length > 1;
  const legend = ns => `<div class="mx-legend">${ns.map(n => `<span>${monogram(n, 'xs')}${short(n)}</span>`).join('')}</div>`;
  const tables = rt.groups.map(g => g.ranked.filter(p => paired.has(p))).filter(ns => ns.length >= 3)
    .map((ns, i, all) => `${all.length > 1 || multi ? `<div class="mx-group">${esc(groupName(rt.groups.find(g => g.ranked.includes(ns[0]))))}</div>` : ''}<div class="mx-wrap">${matrix(ns)}</div>${legend(ns)}`);
  if (!tables.length) return '';
  const descriptive = rt.pooled || rt.unregistered;
  const anyOutside = rt.pairings.some(p => Core.isDecided(p) && p.assumption && p.assumption.status === 'unsupported');
  return section('Head to head', 'every matchup in one grid: bb/100, row against column', tables.join(''),
    `${descriptive ? 'Nothing is shaded: no result here is decided.' : `Shaded only where the matchup is decided${anyOutside ? ' (faint and dashed: outside the calibrated range)' : ''}; plain numbers are ahead on points.`} A cell opens that matchup.`, { id: 'h2h', fold: true });
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
  return section('How they play', 'tendencies, not quality', pick + `<div class="scroll">${table(rows)}</div>`,
    `<span class="def-line" aria-live="polite">Tap a column name for what it counts.</span> Each figure is how often it did something, with the count out of its chances beneath.
    These describe style, pooled over opponents and seats; they do not say who plays better. Rows under 100 hands are dimmed. <b>${populationLine(rt, 'stats')}</b>`,
    { id: 'style', fold: true, after: block('Split by seat: on the button or in the big blind', bySeat) + block('Split by opponent', byOpp) });
}

// ---- poker fundamentals: the same textbook spots for every player, each with a worked-out answer --------
const quizLink = id => runLink(state.run, '/q' + (id ? '/' + encodeURIComponent(id) : ''));
const ACTION_WORD = { fold: 'Fold', check: 'Check', call: 'Call', bet: 'Bet', raise: 'Raise', all_in: 'All in', shove: 'All in' };
const answerText = (a, bb) => {
  if (a == null) return '–';
  const x = typeof a === 'string' ? { action: a } : a;
  const word = ACTION_WORD[x.action] || x.action || '–';
  return x.amount != null && !['fold', 'check'].includes(x.action) ? `${word} ${bbs(x.amount, bb)}` : word;
};
const answerKey = a => (typeof a === 'string' ? a : a && a.action) || null;
const lostOf = a => a ? a.ev_lost_bb ?? a.ev_loss_bb ?? a.regret_bb ?? null : null;
const KIND_WORD = { pot_odds: 'pot odds', all_in: 'all-in call', shove: 'shove or fold', equity: 'equity', bluff_catch: 'bluff catcher',
  close_ev: 'close EV', multi_street: 'multi-street line', range: 'range reading', range_inference: 'range reading', sizing: 'bet sizing' };
const catWord = k => KIND_WORD[k] || String(k).replace(/_/g, ' ');
const PROV = {
  eligible: ['ok', 'provenance checked', 'The answers kept their raw output, the model that answered matches the one asked for, and the protocol, code and CLI versions are recorded.'],
  ineligible: ['bad', 'ineligible', 'This score fails the provenance check, so it is shown but not counted.'],
  unrecorded: ['warn', 'provenance not recorded', 'This quiz run did not keep the evidence needed to check the score (raw output, which model answered, protocol, code and CLI versions): read it as unverified.'],
};
const provChip = x => {
  const pv = x.provenance;
  if (!pv || pv.status === 'bot' || !PROV[pv.status]) return '';
  const [tone, word, tip] = PROV[pv.status];
  const more = [pv.reasons.join('; '), pv.identity ? `identity ${pv.identity}` : '', pv.facts.join(' · ')].filter(Boolean).join(' · ');
  const at = x.attestation && (typeof x.attestation === 'string' ? { status: x.attestation } : x.attestation);
  const att = at && (at.status === 'unattested' || at.status === 'mismatch')
    ? chip(at.status === 'mismatch' ? 'wrong model' : 'unattested', at.status === 'mismatch' ? 'bad' : 'quiet', `${at.meaning || ''}${at.policy ? ` (${at.policy})` : ''}`) : '';
  const leak = Core.leakNote(x);
  return chip(word, tone, tip + (more ? ` (${more})` : '')) + att + (leak ? chip(esc(leak.short), 'warn', leak.long) : '') + linkChip(x);
};
// A quiz score of the same model and reasoning level as a tournament player, but not shown to be the same player
// (settings, code or CLI differ or were not recorded): a separate, model-level assessment, never the player's own.
const modelLevel = x => !!(x.link && x.link.status === 'model-level');
const linkTip = x => `A separate model-level assessment: the same model and reasoning level as ${x.link.player ? plain(x.link.player) : 'a tournament player'}, but not shown to be the same player (${(x.link.differences || []).join('; ') || 'provenance differs'}), so it is not that player's score.`;
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
  const flag = (x.provenance?.status === 'unrecorded' ? '<span class="soft"> (unverified)</span>' : x.provenance?.status === 'ineligible' ? '<span class="neg"> ineligible</span>' : '')
    + (leak ? `<span class="soft" title="${esc(leak.long)}"> (earlier version: answers may have leaked)</span>` : '');
  const head = `${x.correct ?? '–'}/${x.total ?? '–'}${q.scored_category ? ` ${esc(catTitle(q, q.scored_category).toLowerCase())}` : ''}`;
  const more = x.others && x.others.total ? ` · ${x.others.correct}/${x.others.total} harder` : '';
  const sep = modelLevel(x) && !quiet ? `<span class="soft" title="${esc(linkTip(x))}"> (model-level, separate run)</span>` : '';
  return `<b>${plain(x.player)}</b> ${head}${more}${x.fallbacks ? ` (${plural(x.fallbacks, 'fallback')})` : ''}${flag}${sep}`;
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
    <span class="qt-body">${models.some(modelLevel) ? 'Model-level, run apart from these tournament players: ' : ''}A separate quiz of ${plural(q.spots.length, 'spot')}${q.categories.length > 1 ? ` in ${plural(q.categories.length, 'category', 'categories')}` : ''} with worked-out answers: ${models.map(x => scoreText(q, x, models.every(modelLevel))).join(' · ')}${ref ? ` · ${scoreText(q, ref)} (reference)` : ''}${ceiling ? '<span class="soft"> · several perfect scores: these spots are too easy to separate these models</span>' : ''}</span>
    <span class="al-go">See every spot ›</span></a>`;
}

async function renderQuiz(spotId) {
  const q = await loadQuiz();
  const superseded = q.superseded ? `<div class="notice quiz-notice"><span class="suit">♦</span><div><b>Superseded.</b> ${esc(typeof q.superseded === 'string' ? q.superseded : q.superseded.reason || q.superseded.by || 'a newer version of this quiz replaces it')}</div></div>` : '';
  const leaked = q.prompt_leak && (q.prompt_leak.spots || []).length ? `<div class="notice quiz-notice"><span class="suit">♦</span><div><b>Earlier quiz version${q.prompt_leak.version ? ` (${esc(q.prompt_leak.version)})` : ''}.</b> Its prompts named each spot by a descriptive id, and ${plural(q.prompt_leak.spots.length, 'spot')} had an id naming an action${(q.prompt_leak.named_answer || []).length ? ` (${q.prompt_leak.named_answer.length} of them the best answer itself)` : ''}, so those prompts may have leaked the answer: ${q.prompt_leak.spots.map(esc).join(', ')}. Scores on them are flagged; quiz-4 names spots by an opaque id.</div></div>` : '';
  const fixture = superseded + leaked + (q.fixture ? `<div class="notice quiz-notice"><span class="suit">♦</span><div><b>Fixture data.</b> ${esc(q.fixture === true ? 'These answers are made up to show the layout; they are not real model answers.' : q.fixture)}</div></div>` : '');
  const head = `<div class="page-head"><div><div class="eyebrow">Poker fundamentals</div><h1>${esc(q.suite)}</h1>
      <div class="sub">${plural(q.spots.length, 'spot')} · ${plural(q.players.length, 'player')}${q.version ? ` · version ${esc(q.version)}` : ''}</div></div></div>`;
  const lede = `<p class="lede quiz-lede">Separate from the games: every player answers the same spots, such as calling or folding at a price or
    going all in, and each spot has an answer worked out from the cards (equity against the stated hands, and the pot odds).
    This measures fundamentals; the standings measure who wins against these particular opponents.</p>`;
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
    const valid = x.valid != null && x.total ? `<small class="q-valid" title="Right out of the answers that were valid: a fallback (no valid answer after every retry) is never counted right">${x.valid_accuracy != null ? `${Math.round(x.valid_accuracy * 100)}% of ${plural(x.valid, 'valid answer')}` : 'no valid answers'}${x.fallbacks ? ` · <b class="neg">${plural(x.fallbacks, 'fallback')}, no credit</b>` : ''}</small>` : '';
    return `<div class="qrow ${x.reference ? 'ref' : ''}">${player(x.player, `<div class="q-prov">${provChip(x)}</div>`, { details: false })}
      <span class="q-score"><b>${x.correct ?? '–'}</b><small>/${x.total ?? '–'} right</small>${valid}</span>
      <span class="q-bar" title="${Math.round(share * 100)}% right"><span style="width:${(share * 100).toFixed(1)}%"></span></span>
      <span class="q-lost" title="Expected big blinds given up against the best answer, added over ${sc ? `the ${esc(catTitle(q, sc).toLowerCase())} spots` : 'every spot'}">${x.ev_lost_bb != null
        ? `<b class="${x.ev_lost_bb > 0 ? 'neg' : ''}">${trim(Math.round(x.ev_lost_bb * 10) / 10)}</b><small> bb lost</small><span class="q-lostbar"><span style="width:${(100 * (x.ev_lost_bb || 0) / maxLost).toFixed(1)}%"></span></span>` : '<span class="dim">–</span>'}</span></div>`;
  }).join('');
  const ref = q.players.filter(x => x.reference);
  const info = sc && q.category_info[sc];
  const caption = `${sc ? `Every figure in this list is from the <b>${esc(catTitle(q, sc).toLowerCase())}</b> spots only${info && info.about ? ` (${esc(info.about)})` : ''}: right answers, the share right among valid answers, fallbacks and big blinds lost all count the same ${plural(q.spots.filter(s => (s.category || s.kind) === sc).length, 'spot')}. The other categories are scored separately below, each out of its own spots.` : 'Right: spots answered with the best action, out of all spots.'}
    A fallback (no valid answer after every retry, so the game folded or checked for it) never counts as right, even where folding happened to be best.
    Bb lost: the expected big blinds given up against the best answer, added over the spots.${ref.length ? ` <b>${ref.map(x => plain(x.player)).join(', ')}</b> is a fixed, versioned
    reference policy (solid, position-aware, pot-odds based): the yardstick. A model below it misses spots a simple rule gets right.` : ''}
    ${q.population || q.estimand ? ` Population: ${esc(q.population || q.estimand)}.` : ' These are counts on this fixed set of spots, not estimates for poker in general, so no error ranges are given.'}`;
  return section(sc ? `Scores · ${esc(catTitle(q, sc))}` : 'Scores', sc ? 'the scored category; fewer bb lost is better' : 'fewer bb lost is better', `<div class="qlist">${rows}</div>`, caption) + quizCategories(q);
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
    return { cls: `qcat ${r === 1 ? 'full' : r < 0.5 ? 'low' : ''}`, html: `${k.correct ?? 0}<small class="nof">/${k.total}${k.fallbacks ? ` · ${k.fallbacks} fb` : ''}${lost != null ? ` · ${trim(Math.round(lost * 10) / 10)} bb lost` : ''}</small>` };
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
  return section(q.scored_category ? 'Harder categories, scored separately' : 'By category', 'each out of its own spots: they separate players the basics cannot',
    `<div class="scroll qcat-wide"><table class="style-t qcat-t">${head}<tbody>${rows}</tbody></table></div>${narrow}`,
    `Right answers out of the category's own spots; fb: fallbacks, which never count as right. ${q.scored_category ? 'None of these enter the headline score above.' : ''}
     ${cats.some(c => { const i = q.category_info[c] || {}; const sm = i.sampling && (i.sampling.design || i.sampling); return sm !== 'independent'; }) ? 'Counts on these spots, descriptive: no error ranges, since the spots are not an independent sample of a defined population.' : ''}`);
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
    ['Pot', bbs(pot, bb)], toCall ? ['To call', bbs(toCall, bb)] : null,
    need != null ? (closes ? ['Break-even', `${Math.round(need * 100)}% <small>the call ends the betting, so it must win this often to pay</small>`]
      : ['Pot odds', `${Math.round(need * 100)}% <small>immediate pot odds, before future betting: not a full break-even</small>`]) : null,
    s.equity != null ? ['Equity', `${Math.round(s.equity * 100)}% <small>${esc(s.equity_note || 'against the stated hands')}</small>`] : null,
    ['Best answer', `<b class="gold">${esc(answerText(s.best, bb))}</b>`],
    Object.keys(ev).length ? ['Expected value', Object.entries(ev).map(([k, v]) => `${esc(ACTION_WORD[k] || k)} ${signed(v, 1)} bb`).join(' · ')] : null,
  ].filter(Boolean).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  const answers = q.players.map(x => {
    const a = x.answers[s.id];
    const fb = Core.isFallback(a);
    const ok = !a ? null : fb ? false : a.correct ?? (answerKey(a) === answerKey(s.best));
    const lost = lostOf(a);
    const why = a && (a.reasoning || a.say) ? `<details class="q-why"><summary>why?</summary><div>${esc(a.reasoning || a.say)}</div></details>` : '';
    return `<div class="qans ${ok == null ? '' : ok ? 'ok' : 'bad'} ${x.reference ? 'ref' : ''}">${player(x.player, '', { details: false })}
      <span class="qa-act">${a ? esc(answerText(a, bb)) : '<span class="dim">no answer</span>'}</span>
      <span class="qa-ok">${ok == null ? '' : fb ? `<span title="No valid answer after every retry, so the game played a fallback for it: never credited${a.credited ? ', though the fallback happened to be the best action' : ''}">! fallback, no credit</span>` : ok ? '✓ best' : '✗ not best'}</span>
      <span class="qa-lost">${lost != null ? (lost > 0 ? `<b class="neg">${trim(Math.round(lost * 10) / 10)}</b> bb lost` : '0 bb lost') : ''}</span>${why}</div>`;
  }).join('');
  return `<div class="spot-nav"><a href="${quizLink()}">‹ All spots</a><span>Spot ${i + 1} of ${q.spots.length} · ${esc(catTitle(q, s.category || s.kind || 'other'))}</span>
      <span class="nav">${prev ? `<a href="${quizLink(prev.id)}" aria-label="Previous spot">←</a>` : '<span class="dim">←</span>'}${next ? `<a href="${quizLink(next.id)}" aria-label="Next spot">→</a>` : '<span class="dim">→</span>'}</span></div>
    <div class="spot-grid"><div class="spot-main"><div class="card-box play spot-table"><div class="play-head"><h3>${esc(Core.spotTitle(s))}</h3><span class="small soft">${esc(s.situation || s.history || '')}</span></div>
      <div class="table-slot">${PokerTable.html([{ name: '<span class="nm">Player</span>', badge: '<span class="mono lg" style="--c:#a97a2a">?</span>' },
        { name: '<span class="nm">Opponent</span>', badge: '<span class="mono lg" style="--c:#55605a">Op</span>' }])}</div></div>
      ${spotQuestion(q, s)}</div>
      <div class="spot-side"><div class="card-box spot-facts"><dl>${facts}</dl>${s.explain || s.explanation ? `<p class="spot-explain">${esc(s.explain || s.explanation)}</p>` : ''}</div>
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
  if (s.villain_range) stated.push(['Villain\'s range', `${esc(s.villain_range)}${s.villain_combos ? ` <small>${s.villain_combos} combinations possible</small>` : ''}`]);
  if ((s.actions || []).length) stated.push(['The hand so far', s.actions.map(a => `${esc(a.street)}: ${a.seat === 0 ? 'you' : 'Villain'} ${esc(ACT_WORD[a.action] || a.action)}${a.amount != null && !['check', 'fold'].includes(a.action) ? ` ${a.amount}` : ''}`).join('; ')]);
  if (s.line) stated.push(['Villain\'s line', Object.entries(s.line).map(([k, r]) => `${esc(k)}: bets with ${esc(ruleText(r))}`).join('; ')]);
  if (s.river_plan) stated.push(['If you call', `river: Villain goes all in with ${esc(ruleText(s.river_plan))}, else checks`]);
  if (s.shove_call) stated.push(['If you go all in', `Villain calls with ${esc(ruleText(s.shove_call))}, else folds`]);
  if (s.responses) stated.push(['If you bet', Object.entries(s.responses).map(([k, r]) => `${esc(k === 'all_in' ? 'all in' : k.replace('bet_', 'bet '))}: Villain calls with ${esc(ruleText(r))}`).join('; ')]);
  if ((s.options || []).length) stated.push(['Allowed choices', s.options.map(o => esc(o === 'all_in' ? 'all in' : o.replace('bet_', 'bet '))).join(', ')]);
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
  PokerTable.show(el, snap, { fmt: c => bbs(c, bb), hidden: [1], names: ['Player', 'Opponent'] });
}

// ---- games ----------------------------------------------------------------------------------------
// The list has one entry per pairing. The replay walks the pairing's deals in playing order, across every
// round and conversation, so its arrows, auto-next and big swings never stop at the end of one conversation.
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
  if (P.reps.length > 1) bits.push(`${P.reps.length} conversations${P.rounds.length > 1 ? ' each' : ''}`);
  return bits.join(' × ');
}

// One pairing in the list: names, its verdict, and its edge over verified deals (from the ratings).
function matchNav(P, cur, rt) {
  const pr = rt && rt.pairings.find(p => pairKey(p) === P.key);
  const v = pr ? verdict(pr, plain) : null;
  const own = p => esc(nameParts(labelText(p))[0]);
  const active = cur && pairKey(cur) === P.key;
  const e = pr && pr.deals ? (pr.a === P.a ? 1 : -1) * (pr.edge ?? 0) : null;
  const edge = e != null ? `<b class="${cls(e)}" title="${own(P.a)}'s edge in bb/100 over verified deals">${signed(e, 0)}<small>/100</small></b>` : '';
  const deals = pr && pr.deals != null ? pr.deals : P.deals;
  const status = !P.started.length ? 'waiting for a free seat'
    : `${v ? `${v.decided ? '<span class="decided-badge">Decided</span>' : ''}<span class="vword ${v.tone}">${esc(v.short)}</span> · ` : ''}${plural(deals, 'deal')}${scopeText(P) ? ` · ${scopeText(P)}` : ''}`;
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
  let cur = state.index.find(m => m.id === id);
  if (!cur && id) {
    const P = pairingsOf(state.index).find(P => P.ms.some(m => m.id.split('/').pop() === id.split('/').pop()));
    cur = P && P.latest;
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
    about: !!document.querySelector('.about[open]'), opts: !!document.querySelector('.opts.open'),
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
    if (keep.opts) $app.querySelector('.opts')?.classList.add('open');
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
const STATUS_WORD = { verified: 'verified', unchecked: 'not checked', provisional: 'provisional: not counted yet', playing: 'being played', rejected: 'rejected by the verifier: counts nowhere' };

// Every deal of the pairing as a small chip, one row per conversation (or per round of one-deal conversations),
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
  const label = x => longConv ? (rounds > 1 ? `R${x.round} · C${x.replicate}` : `Conv. ${x.replicate}`) : `Round ${x.round}`;
  const cell = ([x, i]) => {
    const who = x.net > 0 ? short3(L) : short3(R);
    const what = x.net == null ? STATUS_WORD[x.status] || x.status : x.net === 0 ? 'even over both plays' : `${who} won ${trim(Math.abs(x.net))} bb over both plays`;
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
  const pos = Math.max(0, seq.findIndex(x => x.id === cur.id && x.deal === pick));
  const tot = Core.sequenceTotals(seq);
  const planned = Math.max(seq.length, (own.conversations || []).reduce((t, c) => t + (c.deals_total || 0), 0));
  const multiConv = own.conversations.length > 1;
  state.view = { cur, pick, total, played, seq, pos };

  // the scoreboard: the whole pairing, verified deals only
  const v = pairing ? verdict(pairing, plain) : null;
  const lead = tot.net > 0 ? L : tot.net < 0 ? R : null;
  const amount = trim(Math.abs(Math.round(tot.net * 10) / 10));
  const rate = tot.counted ? Math.abs(tot.net) / (2 * tot.counted) * 100 : 0;
  const face = (p, side) => `<div class="sb-face ${side} ${lead === p ? 'ahead' : ''}">${monogram(p, 'lg')}<div class="sb-name">${name(p)}</div></div>`;
  const left = [tot.playing ? 'one in play' : '', tot.provisional ? `${tot.provisional} provisional` : '', tot.rejected ? `${tot.rejected} rejected` : ''].filter(Boolean);
  const leftTip = 'Provisional: one play is still to finish or commit it. Rejected: the verifier could not reproduce it from its seed and actions. Neither counts in any total.';
  const scorebar = `<div class="scorebar card-box">
      ${face(L, 'left')}
      <div class="sb-mid">
        <div class="sb-score">${!tot.counted ? '<span class="sb-word">No deals counted yet</span>' : lead ? `<span class="sb-word">${short3(lead)} up</span> <b>${amount}<small> bb</small></b>` : '<b class="even">Even</b>'}</div>
        <div class="sb-sub">${tot.counted ? `<span class="hide-xs">${trim(Math.round(rate))} bb/100 · </span>${plural(tot.counted, tot.checked ? 'verified deal' : 'unchecked deal')}` : ''}${multiConv && P ? `<span class="hide-xs"> · ${scopeText(P)}</span>` : ''}</div>
        <div class="sb-verdict">${v ? verdictCell(pairing, v, { compact: true }) : `<span class="chip quiet" title="The standings are being worked out; the verdict appears when they arrive">verdict pending</span>`}
          ${left.length ? `<span class="chip warn" title="${esc(leftTip)}">${left.join(' · ')} · not counted</span>` : ''}</div>
      </div>
      ${face(R, 'right')}</div>`;

  const order = [...played].sort((x, y) => x - y);
  const dealNet = pick in nets ? sign * nets[pick] : null;
  const outcome = dealNet == null ? 'in progress' : dealNet === 0 ? 'even over both plays'
    : `<b>${short3(dealNet > 0 ? L : R)}</b> wins it by <b class="gold">${trim(Math.abs(dealNet) / bb)} bb</b> over both plays`;

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
    return `<button class="jump swing ${i === pos ? 'on' : ''}" data-seq="${i}" title="Deal ${i + 1}: ${plain(x.net > 0 ? L : R)} won ${trim(Math.abs(x.net))} bb over both plays">
      ${i + 1}<b class="${cls(x.net)}">${signedBB(x.net, 1)}</b></button>`;
  }).join('');

  const here = seq[pos] || { status: 'playing' };
  const st = here.status;
  const popTag = st === 'rejected' ? `<span class="chip bad" title="${esc(((own.conversations[here.conv] || {}).rejected || []).map(x => x.reason).join('; ') || 'The verifier could not reproduce this deal from its seed and actions')}">rejected · not counted</span>`
    : st === 'playing' || st === 'provisional' ? '<span class="chip warn" title="Shown as it happens; counted only once both plays finish and are checked">provisional</span>'
    : st === 'unchecked' ? '<span class="chip quiet" title="This viewer could not run the verifier on this deal">not checked</span>' : '';
  const where = multiConv ? `${P && P.rounds.length > 1 ? `Round ${cur.round} · ` : ''}${total > 1 ? 'conv.' : 'conversation'} ${cur.replicate}${total > 1 ? ` · deal ${pick + 1}/${total}` : ''}` : '';
  const controls = `<div class="controls card-box" role="group" aria-label="Replay">
    <div class="deal-title"><span class="nav"><button data-seq="${pos - 1}" ${pos > 0 ? '' : 'disabled'} aria-label="Previous deal" title="Previous deal (←)">‹</button></span>
      <h2>Deal ${pos + 1}<small> of ${planned}</small>${where ? `<span class="round">${esc(where)}</span>` : ''}</h2>
      <span class="nav"><button data-seq="${pos + 1}" ${pos < seq.length - 1 ? '' : 'disabled'} aria-label="Next deal" title="Next deal (→)">›</button></span>${popTag}</div>
    <div class="ctls">
      <button class="ctl" data-ctl="start" title="Back to the deal" aria-label="Back to the start">⏮</button>
      <button class="ctl" data-ctl="prev" title="Step back (,)" aria-label="Step back">◀</button>
      <button class="ctl play" data-ctl="play" title="Play step by step (space)">▶ Play</button>
      <button class="ctl" data-ctl="next" title="Step forward (.)" aria-label="Step forward">▶</button>
      <button class="ctl" data-ctl="end" title="To the end" aria-label="To the end">⏭</button></div>
    <input type="range" class="scrub" min="0" max="${last}" value="${rp.s}" aria-label="Step">
    <span class="step-text"></span>
    <button class="opts-toggle toggle" aria-expanded="false" aria-controls="opts">Options</button>
    <div class="opts" id="opts">
      <span class="mode-switch" role="group" aria-label="How the two plays move">
        <button class="toggle" data-mode="aligned" aria-pressed="${state.mode === 'aligned'}" title="Both plays reach the flop, turn, river and result together">Streets side by side</button>
        <button class="toggle" data-mode="sequence" aria-pressed="${state.mode === 'sequence'}" title="Play 1 from start to end, then Play 2">Play 1, then Play 2</button></span>
      ${diffBtn}<span class="spacer"></span>
      <label class="speed" title="How long each step stays on screen while playing"><span>Speed</span><select id="speed" aria-label="Playback speed">${Object.entries(SPEEDS).map(([k, x]) =>
        `<option value="${k}" ${k === state.speed ? 'selected' : ''}>${x.label}</option>`).join('')}</select></label>
      <button class="toggle" id="auto-toggle" aria-pressed="${state.autoNext}" title="When a deal ends, go on to the next deal of this matchup, across conversations">Auto-next deal</button>
      <button class="toggle" id="mind-toggle" aria-pressed="${state.mind}" title="Show every decision's reasoning (r)">Reasoning</button></div>
    <span class="deal-outcome soft">${outcome}</span></div>`;

  const sync = plays.length === 2 ? `<div class="sync" role="tablist" aria-label="Which play">
    ${plays.map(p => `<button class="sync-row" role="tab" data-sync="${p.j}" aria-pressed="${state.showPlay === p.j}" aria-selected="${state.showPlay === p.j}"><span class="sync-k">${p.j ? 'Play 2 · swapped' : 'Play 1'}</span><span class="sync-now"></span></button>`).join('')}
    <span class="sync-tag"></span></div>` : '';
  const verdictLong = v ? `<p>${esc(v.long)}</p><div class="about-chips">${verdictCell(pairing, v)}</div>` : '<p class="soft">The verdict appears when this run\'s standings are worked out.</p>';
  const rejects = own.conversations.flatMap(c => (c.rejected || []).map(x => `${x.hand == null ? 'the whole conversation' : `deal ${x.hand + 1}`}${multiConv ? ` of conversation ${c.replicate}${c.round > 1 ? `, round ${c.round}` : ''}` : ''}: ${esc(x.reason)}`));
  const population = `Every total here counts ${tot.checked ? '<b>verified deals</b> only: both plays committed, checked as a duplicate pair and replayed from their seeds' : '<b>finished deals</b>: this viewer could not run the verifier, so they are not checked'}
    (${plural(tot.counted, 'deal')}).${tot.provisional || tot.playing ? ` ${plural(tot.provisional + tot.playing, 'deal')} still in play or waiting for the other play ${tot.provisional + tot.playing === 1 ? 'is' : 'are'} provisional and left out.` : ''}
    ${tot.rejected ? ` ${plural(tot.rejected, 'deal')} the verifier rejected ${tot.rejected === 1 ? 'counts' : 'count'} nowhere.` : ''}`;
  const about = `<details class="about card-box"><summary><span>About this result</span><span class="soft small">${v ? esc(v.short) : 'verdict pending'} · what it counts and rests on</span></summary>
    <div class="about-body">${verdictLong}<p class="small soft">${population}</p>${rejects.length ? `<ul class="small soft rejects">${rejects.slice(0, 6).map(x => `<li>${x}</li>`).join('')}</ul>` : ''}</div></details>`;
  const nav = `<div class="dealnav card-box"><div class="dn-head"><h3>${planned > seq.length ? `${seq.length} of ${plural(planned, 'deal')} played` : `All ${plural(seq.length, 'deal')}`}</h3>
      <span class="dn-legend"><span><i class="sw pos"></i>${short3(L)} won it</span><span><i class="sw neg"></i>${short3(R)} won it</span><span><i class="sw sd"></i>showdown</span>${tot.provisional || tot.playing ? '<span><i class="sw prov"></i>provisional</span>' : ''}${tot.rejected ? '<span><i class="sw rej"></i>rejected</span>' : ''}</span></div>
    ${dealNav(seq, pos, L, R, swings)}
    ${swingBtns ? `<div class="jumps"><span class="jump-k">Big swings</span><span class="jump-list">${swingBtns}</span></div>` : ''}
    <div class="keys small soft"><span class="kbd">←</span><span class="kbd">→</span> deals · <span class="kbd">space</span> play · <span class="kbd">,</span><span class="kbd">.</span> step · <span class="kbd">d</span> where they differ · <span class="kbd">r</span> reasoning</div></div>`;
  const body = `${scorebar}${controls}${sync}
    <div class="pair ${state.mind ? 'show-mind' : ''}" data-show="${state.showPlay}">${plays.map(p => playPanel(p, cur, bb)).join('')}</div>
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
    if (ot) ot.onclick = () => {
      const o = $app.querySelector('.opts');
      o.classList.toggle('open');
      ot.setAttribute('aria-expanded', String(o.classList.contains('open')));
    };
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
    $app.querySelectorAll('.sync [data-sync]').forEach(b => b.onclick = () => {
      const j = Number(b.dataset.sync);
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
  $app.querySelectorAll('.sync [data-sync]').forEach(x => {
    x.setAttribute('aria-pressed', String(Number(x.dataset.sync) === j));
    x.setAttribute('aria-selected', String(Number(x.dataset.sync) === j));
  });
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
    const sig = (p, a) => a.map(e => `${e.seat}:${e.action}:${e.amount ?? ''}`).join(' ');
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
      const mind = hasMind || record ? `<div class="mind">${t && t.reasoning ? esc(t.reasoning) : ''}${t && t.thinking
        ? `<span class="lbl">The model's own thinking</span><span class="thinking">${esc(t.thinking)}</span>` : ''}${record}</div>` : '';
      const f = e.facing;
      const facing = f && e.action !== 'raise' && e.action !== 'all_in' ? (f.closes
        ? `<span class="facing" title="What it had to put in to call; the call ends the betting, so this is the share of the final pot a call needs to win to break even">facing ${fmt(f.toCall)} · needs ${Math.round(f.odds * 100)}%</span>`
        : `<span class="facing" title="What it had to put in to call, and the immediate pot odds: the price now, before any future betting, so not a full break-even">facing ${fmt(f.toCall)} · pot odds ${Math.round(f.odds * 100)}%, before future betting</span>`)
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
    <div class="table-slot">${PokerTable.html(seats)}</div><div class="now-line" aria-live="polite"></div><div class="log">${log}</div>${noteBlock}</div>`;
}

// One decision's record: which model answered (attested or not), whether its turn was audited, and every
// try before the one that was played.
// Only for records without the runner's own `identity_match`: the same model id, or a dated snapshot of it.
const canon = m => String(m).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const sameModel = (want, seen) => { const a = canon(want), b = canon(seen); return a === b || new RegExp(`^${a}-\\d{8}$|^${a}-\\d{4}-\\d{2}-\\d{2}$`).test(b); };
const TRY_WORD = { tool_use: 'used a tool, rejected', unverified: 'could not be audited, rejected', invalid: 'invalid', illegal: 'illegal' };
function recordOf(t, pid, fmt) {
  if (!t || isBot(pid)) return '';
  const seen = t.observed_model || t.served_model || null;
  const want = t.requested_model || null;
  const id = ['match', 'mismatch', 'unattested'].includes(t.identity_match) ? t.identity_match
    : !seen || !want ? 'unattested' : sameModel(want, seen) ? 'match' : 'mismatch';
  const bits = [`<b class="idst ${ID_TONE[id] || 'warn'}">${esc(Core.IDENTITY[id])}</b>${seen ? ` (reported ${esc(seen)}${id === 'mismatch' && want ? `, asked for ${esc(want)}` : ''})` : ''}`];
  bits.push(t.audit ? `turn audit <b>${esc(t.audit)}</b>` : 'turn audit not recorded');
  if (t.tries) bits.push(plural(t.tries, 'try', 'tries'));
  if (t.seconds != null) bits.push(`${t.seconds} s`);
  const tries = (t.retries || []).map((r, k) => `<li>Try ${k + 1}${r.requested && r.requested.action ? `: asked <b>${esc(Core.actText(r.requested, fmt))}</b>` : ''}
    → ${esc(TRY_WORD[r.kind] || 'rejected')}${r.error ? `: ${esc(r.error)}` : ''}</li>`).join('');
  return `<span class="lbl">Record</span><span class="record">${bits.join(' · ')}</span>${tries ? `<ol class="tries">${tries}</ol>` : ''}`;
}

// What just happened in a play at step k, with the price it faced: the line beside the table.
const NOW_VERB = { fold: 'folds', check: 'checks', call: 'calls', bet: 'bets', raise: 'raises to', all_in: 'goes all in' };
function nowText(p, k, bb, long) {
  if (k < 0) return '<span class="soft">not dealt yet</span>';
  const fmt = c => bbs(c, bb);
  const st = p.steps[k];
  const next = long && st && st.toAct != null ? `<span class="next">Next: <b>${esc(nameParts(labelText(p.m.config.players[st.toAct]))[0])}</b> ${st.toCall
    ? (st.callCloses ? `must put in ${fmt(st.toCall)} to call, which ends the betting: it pays if it wins ${Math.round(st.potOdds * 100)}% of the time`
      : `must put in ${fmt(st.toCall)} to call: immediate pot odds ${Math.round(st.potOdds * 100)}%, before future betting`) : 'can check'}</span>` : '';
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
    : f.closes ? `facing ${fmt(f.toCall)} into ${fmt(f.pot)}: the call ends the betting, so it needs ${Math.round(f.odds * 100)}%`
    : `facing ${fmt(f.toCall)} into ${fmt(f.pot)}: immediate pot odds ${Math.round(f.odds * 100)}%, before future betting`;
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
  else if (e.key === 'd') jumpToDifference();
  else if (e.key === ' ') { control('play'); e.preventDefault(); }
  else if (e.key === ',') control('prev');
  else if (e.key === '.') control('next');
});

route();
