// The poker table: a hand's events turned into one snapshot per step, and a table that shows a snapshot.
// Any number of seats from 2 to 3 (and a seat can hide its cards), so a human seat can reuse it later.
// `PokerTable.steps` is pure (node can load it); `html` builds the table once, `show` updates it in place.
(function (root, make) {
  const api = make();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PokerTable = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SUIT = { s: '♠', h: '♥', d: '♦', c: '♣' };
  function card(c, i = 0, extra = '') {
    if (!c) return `<span class="pc slot ${extra}"></span>`;
    if (c === '??') return `<span class="pc back ${extra}" style="--i:${i}"></span>`;
    const rank = c[0] === 'T' ? '10' : c[0];
    const red = c[1] === 'h' || c[1] === 'd';
    return `<span class="pc ${red ? 'red' : ''} ${extra}" style="--i:${i}" title="${c}"><b>${rank}</b><i>${SUIT[c[1]] || c[1]}</i></span>`;
  }
  function cards(cs, n = 0, size = '') {
    const list = [...(cs || [])];
    while (list.length < n) list.push(null);
    return `<span class="cards ${size}">${list.map((c, i) => card(c, i)).join('')}</span>`;
  }

  const STREET_NAMES = { preflop: 'Preflop', flop: 'Flop', turn: 'Turn', river: 'River' };

  // The best hand a seat holds with the board so far, in the engine's words ('a pair of queens'), plus an
  // open draw on the flop or turn. The engine records these itself (`omniscient.made_hand`); this is the
  // fallback for hands logged before it did.
  const RANKS = '23456789TJQKA';
  const NAMES = { 14: 'ace', 13: 'king', 12: 'queen', 11: 'jack', 10: 'ten', 9: 'nine', 8: 'eight', 7: 'seven', 6: 'six', 5: 'five', 4: 'four', 3: 'three', 2: 'two' };
  const PLURAL = r => r === 6 ? 'sixes' : NAMES[r] + 's';
  function straightHigh(ranks) {
    const u = new Set(ranks);
    if (u.has(14)) u.add(1);
    for (let h = 14; h >= 5; h--) if ([0, 1, 2, 3, 4].every(k => u.has(h - k))) return h;
    return 0;
  }
  function madeHand(hole, board) {
    const cs = [...(hole || []), ...(board || [])].filter(c => c && c !== '??').map(c => ({ r: RANKS.indexOf(c[0]) + 2, s: c[1] }));
    if (cs.length < 2 || cs.some(c => c.r < 2)) return '';
    const count = {}, suits = {};
    cs.forEach(c => { count[c.r] = (count[c.r] || 0) + 1; (suits[c.s] = suits[c.s] || []).push(c.r); });
    const g = Object.entries(count).map(([r, n]) => [n, +r]).sort((a, b) => b[0] - a[0] || b[1] - a[1]);
    const fs = Object.keys(suits).find(x => suits[x].length >= 5);
    let made;
    const sf = fs && straightHigh(suits[fs]);
    if (sf) made = sf === 14 ? 'a royal flush' : `a straight flush, ${NAMES[sf]} high`;
    else if (g[0][0] === 4) made = `four ${PLURAL(g[0][1])}`;
    else if (g[0][0] === 3 && g[1] && g[1][0] >= 2) made = `a full house, ${PLURAL(g[0][1])} full of ${PLURAL(g[1][1])}`;
    else if (fs) made = `a flush, ${NAMES[Math.max(...suits[fs])]} high`;
    else if (straightHigh(cs.map(c => c.r))) made = `a straight, ${NAMES[straightHigh(cs.map(c => c.r))]} high`;
    else if (g[0][0] === 3) made = `three ${PLURAL(g[0][1])}`;
    else if (g[0][0] === 2 && g[1] && g[1][0] === 2) made = `two pair, ${PLURAL(g[0][1])} and ${PLURAL(g[1][1])}`;
    else if (g[0][0] === 2) made = `a pair of ${PLURAL(g[0][1])}`;
    else made = `${NAMES[g[0][1]]} high`;
    return made;
  }
  // 'flush draw' / 'straight draw' when the flop or turn leaves one open and it is not already made
  function drawOf(hole, board) {
    const n = (board || []).length;
    if (n < 3 || n > 4) return '';
    const cs = [...hole, ...board].map(c => ({ r: RANKS.indexOf(c[0]) + 2, s: c[1] }));
    const suits = {};
    cs.forEach(c => { suits[c.s] = (suits[c.s] || 0) + 1; });
    const mine = new Set(hole.map(c => c[1]));
    if (Object.entries(suits).some(([x, k]) => k === 4 && mine.has(x))) return 'flush draw';
    const ranks = new Set(cs.map(c => c.r));
    if (straightHigh([...ranks])) return '';
    for (let h = 13; h >= 5; h--) if ([0, 1, 2, 3].every(k => ranks.has(h - k)) && h - 3 > 2) return 'straight draw';
    return '';
  }

  // A hand's events -> {steps: [snapshot], log: [entry]}. Step 0 is the blinds posted; every action,
  // new street, showdown and the result is a step. An entry's `step` is the snapshot it produces.
  // A snapshot's `actor` is who just acted (its last action); `toAct` is whose turn it is now, or null.
  function steps(events) {
    const evs = events || [];
    const start = evs.find(e => e.kind === 'hand_start');
    if (!start) return { steps: [], log: [], n: 0 };
    const d = start.data;
    const n = (d.hole || []).length || 2;
    const startStack = Array.isArray(d.stack) ? d.stack : Array(n).fill(d.stack ?? 0);
    const seats = Array.from({ length: n }, (_, i) => ({ stack: startStack[i], bet: 0, folded: false, allIn: false,
      hole: (d.hole || [])[i] || [], shown: false, net: null, won: false, last: '', made: '', draw: '' }));
    let collected = 0, board = [], street = 'preflop', actor = null, returned = null;
    const committed = Array(n).fill(0);
    // whose turn it is, as the engine recorded it (undefined in logs from before it did)
    const out = [], log = [];
    let rec = 'to_act' in d ? d.to_act : undefined;
    const made = recorded => seats.forEach((s, i) => {
      const r = recorded && (recorded[i] ?? recorded[String(i)]);
      s.made = r || madeHand(s.hole, board);
      s.draw = drawOf(s.hole, board);
    });
    made(d.omniscient && d.omniscient.made_hand);
    const owed = i => Math.max(0, Math.min(seats[i].stack, Math.max(...seats.map(x => x.bet)) - seats[i].bet));
    const snap = (label, extra = {}) => {
      const pot = collected + seats.reduce((s, x) => s + x.bet, 0);
      out.push({ step: out.length, street, board: [...board], actor, label, button: d.button, pot, collected,
        seats: seats.map(s => ({ ...s, hole: [...s.hole] })), done: false, recordedToAct: rec, ...extra });
      return out.length - 1;
    };
    const sweep = () => { for (const s of seats) { collected += s.bet; s.bet = 0; if (!s.folded) s.last = ''; } };
    const put = (i, amt) => { const a = Math.min(amt, seats[i].stack); seats[i].stack -= a; seats[i].bet += a; committed[i] += a; if (seats[i].stack === 0) seats[i].allIn = true; };
    // unmatched chips go back to their seat: the engine says so with a 'return' event; for hands recorded
    // before it did, the amount is the largest contribution's excess over the next largest
    const giveBack = (i, amt) => {
      const fromBet = Math.min(seats[i].bet, amt);
      seats[i].bet -= fromBet; collected -= amt - fromBet; seats[i].stack += amt; committed[i] -= amt;
      returned = { seat: i, amount: amt };
    };

    for (const e of evs) if (e.kind === 'blind') put(e.data.seat, e.data.amount);
    log.push({ kind: 'start', step: snap('Blinds posted'),
      blinds: evs.filter(e => e.kind === 'blind').map(e => ({ seat: e.data.seat, amount: e.data.amount })) });

    const thoughts = {};
    for (const e of evs) {
      const x = e.data || {};
      if (e.kind === 'thought') thoughts[x.seat] = x;
      else if (e.kind === 'action') {
        const s = seats[x.seat];
        const potBefore = x.pot_before ?? collected + seats.reduce((t, y) => t + y.bet, 0);
        const toCall = x.to_call ?? owed(x.seat);
        const facing = toCall > 0 ? { toCall, pot: potBefore, odds: x.pot_odds ?? toCall / (potBefore + toCall),
          closes: callCloses({ street, toCall, stack: s.stack, opponentAllIn: seats.some((y, j) => j !== x.seat && !y.folded && y.allIn) }) } : null;
        if (x.action === 'fold') s.folded = true;
        else if (x.action === 'call') put(x.seat, x.amount || 0);
        else if (x.action === 'bet' || x.action === 'raise' || x.action === 'all_in') put(x.seat, Math.max(0, (x.amount || 0) - s.bet));
        if (x.all_in) s.allIn = true;
        actor = x.seat;
        s.last = x.action;
        const t = thoughts[x.seat] || null;
        delete thoughts[x.seat];
        if ('to_act' in x) rec = x.to_act;
        else rec = undefined;
        log.push({ kind: 'action', step: snap(x.action, { lastAmount: x.amount }), seat: x.seat, street, action: x.action,
          amount: x.amount, all_in: !!x.all_in, say: x.say || '', note: x.note || null, thought: t, pot: x.pot, facing });
      } else if (e.kind === 'board') {
        // the action before a new street closed the betting: nobody acts until the street is dealt
        const prev = log[log.length - 1];
        if (prev && prev.kind === 'action') out[prev.step].recordedToAct = null;
        sweep();
        street = x.street;
        board = x.board || [...board, ...(x.cards || [])];
        actor = null;
        made(x.omniscient && x.omniscient.made_hand);
        if ('to_act' in x) rec = x.to_act;
        log.push({ kind: 'board', step: snap(STREET_NAMES[street] || street), street, cards: x.cards || [] });
      } else if (e.kind === 'return') {
        giveBack(x.seat, x.amount);
      } else if (e.kind === 'runout') {
        log.push({ kind: 'runout', step: out.length - 1 });
      } else if (e.kind === 'showdown') {
        sweep();
        actor = null;
        rec = null;
        for (const [i, h] of Object.entries(x.hands || {})) { seats[i].shown = true; seats[i].best = h.best; }
        log.push({ kind: 'showdown', step: snap('Showdown'), hands: x.hands || {} });
      } else if (e.kind === 'result') {
        let pot = x.pot;
        if (!returned && !('uncalled_return' in x)) {
          const top = committed.indexOf(Math.max(...committed));
          const excess = committed[top] - Math.max(0, ...committed.filter((_, i) => i !== top));
          if (excess > 0) { giveBack(top, excess); pot = (x.pot ?? 0) - excess; }
        }
        sweep();
        actor = null;
        rec = null;
        if (x.board) board = x.board;
        const won = x.won || [];
        const contested = i => (won[i] || 0) - (returned && returned.seat === i && !('uncalled_return' in x) ? returned.amount : 0);
        seats.forEach((s, i) => { s.net = (x.net || [])[i] ?? 0; s.won = contested(i) > 0; s.stack = startStack[i] + s.net; });
        collected = 0;
        const winners = seats.map((s, i) => s.won ? i : null).filter(i => i != null);
        log.push({ kind: 'result', step: snap('Result', { done: true, finalPot: pot, returned, winners, showdown: !!x.showdown }),
          winners, pot, returned, showdown: !!x.showdown, net: x.net || [] });
      }
    }
    // who acts next: the recorded `to_act` when the log has it, else the seat of the next action; and what
    // that seat must put in to call, with the share of the pot it would need to win to break even
    out.forEach((st, k) => {
      const at = log.findIndex(e => e.step > k);
      const next = at < 0 ? null : log[at];
      st.toAct = st.done ? null : st.recordedToAct !== undefined ? st.recordedToAct : (next && next.kind === 'action' ? next.seat : null);
      delete st.recordedToAct;
      const me = st.toAct != null ? st.seats[st.toAct] : null;
      const owe = me ? Math.max(0, Math.min(me.stack, Math.max(...st.seats.map(x => x.bet)) - me.bet)) : 0;
      st.toCall = me ? owe : null;
      st.potOdds = owe ? owe / (st.pot + owe) : null;
      st.callCloses = owe ? callCloses({ street: st.street, toCall: owe, stack: me.stack, opponentAllIn: st.seats.some((y, j) => j !== st.toAct && !y.folded && y.allIn) }) : null;
    });
    return { steps: out, log, n, button: d.button };
  }

  // Whether calling ends the betting for the hand (river, or someone all in): only then are the pot odds a full
  // break-even; otherwise they are the immediate price, before any later betting.
  function callCloses({ street, toCall, stack, opponentAllIn }) {
    return street === 'river' || !!opponentAllIn || (toCall > 0 && stack != null && toCall >= stack);
  }
  const priceText = (odds, closes) => closes ? `break-even ${Math.round(odds * 100)}%` : `immediate pot odds ${Math.round(odds * 100)}%, before future betting`;

  // One timeline over the two plays of a deal: frames of {at: [step per play], focus, seg}.
  // 'sequence': the first play from start to end, then the next (focus is the play that moves).
  // 'aligned': street by street, so both plays reach the flop, turn, river and result together; a play
  // with fewer actions on a street holds while the other catches up.
  const SEGMENTS = ['preflop', 'flop', 'turn', 'river', 'end'];
  const segOf = st => st.done || st.label === 'Showdown' ? 'end' : st.street;
  function timeline(lists, mode = 'aligned') {
    const plays = lists.map(l => l || []);
    const frames = [];
    if (mode === 'sequence') {
      plays.forEach((l, j) => l.forEach((st, k) => frames.push({
        focus: j, seg: segOf(st), at: plays.map((m, i) => i === j ? k : i < j ? Math.max(0, m.length - 1) : 0) })));
    } else {
      const segs = plays.map(l => {
        const by = {};
        l.forEach((st, k) => (by[segOf(st)] = by[segOf(st)] || []).push(k));
        return by;
      });
      const cur = plays.map(() => 0);
      for (const key of SEGMENTS) {
        const len = Math.max(0, ...segs.map(s => (s[key] || []).length));
        for (let o = 0; o < len; o++) {
          segs.forEach((s, j) => { const ks = s[key]; if (ks && ks.length) cur[j] = ks[Math.min(o, ks.length - 1)]; });
          frames.push({ focus: null, seg: key, at: [...cur] });
        }
      }
    }
    return frames.length ? frames : [{ focus: 0, seg: 'preflop', at: plays.map(() => 0) }];
  }
  // The first frame that shows step `k` of play `j` (as the moving play, in a sequence).
  function frameOf(frames, j, k) {
    const i = frames.findIndex(f => f.at[j] === k && (f.focus == null || f.focus === j));
    return i < 0 ? frames.findIndex(f => f.at[j] >= k) : i;
  }

  const wager = (i, where) => `<div class="wager ${where[i]}" data-bet="${i}"><span class="chip-dot"></span><b></b></div>`;

  // The table's markup for `seats` = [{name (html), effort, badge (html), role}]. Seat 0 sits nearest the
  // viewer; with three, the other two sit across the felt, left and right.
  function html(seats, opts = {}) {
    const n = seats.length;
    const where = n === 3 ? ['bottom', 'top-left', 'top-right'] : ['bottom', 'top'];
    const seat = (s, i) => `<div class="seat ${where[i]}" data-seat="${i}">
        <span class="seat-badge">${s.badge || ''}<span class="dealer" title="Dealer button">D</span></span>
        <div class="who"><div class="name">${s.name}${s.effort ? `<span class="effort">${s.effort}</span>` : ''}</div>
          <div class="sub"><span class="stack-amt"></span><span class="seat-act"></span><span class="seat-net"></span></div>
          <div class="made" title="Best hand so far from this seat's own cards and the board; viewers see both hands, each player saw only its own"></div></div>
        <span class="hole"></span><span class="turn">to act</span></div>`;
    const top = seats.map((s, i) => i ? seat(s, i) : '').join('');
    return `<div class="ptable n${n} ${opts.compact ? 'compact' : ''}">
      <div class="seats-top">${top}</div>
      <div class="felt"><div class="felt-inner">
        <div class="wagers">${seats.map((_, i) => i ? wager(i, where) : '').join('')}</div>
        <div class="board">${cards([], 5)}</div>
        <div class="pot-line"><span class="pot-chips"></span><span><span class="step-label"></span>Pot <b class="pot-amt">0</b></span></div>
        <div class="to-act-line" aria-live="polite" title="Break-even: the share of the time a call must win to pay for itself at this price"></div>
        <div class="wagers">${wager(0, where)}</div>
      </div></div>
      <div class="seats-bottom">${seat(seats[0], 0)}</div></div>`;
  }

  const VERB = { fold: 'Folded', check: 'Check', call: 'Call', bet: 'Bet', raise: 'Raise', all_in: 'All in' };

  // Update a table made by `html` to a snapshot. `fmt` formats chips (e.g. as big blinds);
  // `hidden` lists seats whose cards stay face down until shown at showdown.
  function show(el, snap, { fmt = String, hidden = [], prev = null, names = [], made = true } = {}) {
    if (!el || !snap) return;
    const t = el.querySelector('.ptable') || el;
    snap.seats.forEach((s, i) => {
      const seat = t.querySelector(`.seat[data-seat="${i}"]`);
      if (!seat) return;
      seat.classList.toggle('to-act', snap.toAct === i);
      seat.classList.toggle('last-actor', snap.actor === i && snap.toAct !== i);
      seat.classList.toggle('folded', s.folded);
      seat.classList.toggle('winner', snap.done && s.won);
      seat.classList.toggle('button', snap.button === i);
      seat.querySelector('.stack-amt').textContent = fmt(s.stack);
      const act = seat.querySelector('.seat-act');
      act.textContent = s.allIn && !s.folded ? 'All in' : s.last ? VERB[s.last] || s.last : '';
      act.className = 'seat-act ' + (s.allIn && !s.folded ? 'all_in' : s.last || '');
      const face = hidden.includes(i) && !s.shown ? ['??', '??'] : s.hole;
      const holeKey = face.join(',') + (s.folded ? 'f' : '');
      const hole = seat.querySelector('.hole');
      if (hole.dataset.key !== holeKey) { hole.dataset.key = holeKey; hole.innerHTML = cards(face, 2, 'hole'); }
      const mh = seat.querySelector('.made');
      if (mh) mh.textContent = made && !(hidden.includes(i) && !s.shown) && s.made ? s.made + (s.draw && snap.board.length < 5 ? ` · ${s.draw}` : '') : '';
      const net = seat.querySelector('.seat-net');
      net.textContent = snap.done && s.net != null ? (s.net > 0 ? '+' : s.net < 0 ? '−' : '') + fmt(Math.abs(s.net)) : '';
      net.className = 'seat-net ' + (snap.done ? (s.net > 0 ? 'pos' : s.net < 0 ? 'neg' : '') : '');
      const bet = t.querySelector(`.wager[data-bet="${i}"]`);
      if (bet) {
        bet.classList.toggle('on', s.bet > 0);
        bet.querySelector('b').textContent = s.bet > 0 ? fmt(s.bet) : '';
        const grew = prev && prev.seats[i] && s.bet > prev.seats[i].bet;
        if (grew) { bet.classList.remove('pop'); void bet.offsetWidth; bet.classList.add('pop'); }
      }
    });
    const board = t.querySelector('.board');
    const key = snap.board.join(',');
    if (board.dataset.key !== key) {
      const before = (board.dataset.key || '').split(',').filter(Boolean).length;
      board.dataset.key = key;
      board.innerHTML = `<span class="cards">${[0, 1, 2, 3, 4].map(i => card(snap.board[i], i - before,
        i >= before && snap.board[i] && prev ? 'flip' : '')).join('')}</span>`;
    }
    t.querySelector('.pot-amt').textContent = fmt(snap.done ? snap.finalPot ?? 0 : snap.pot);
    const chips = Math.max(1, Math.min(6, Math.round(Math.log2(1 + (snap.done ? snap.finalPot || 0 : snap.pot) / 2))));
    const pc = t.querySelector('.pot-chips');
    if (pc.dataset.n !== String(chips)) { pc.dataset.n = chips; pc.innerHTML = `<span class="stack">${Array.from({ length: chips }, (_, k) => `<i style="--k:${k}"></i>`).join('')}</span>`; }
    const street = snap.done ? '' : STREET_NAMES[snap.street] || '';
    t.querySelector('.step-label').textContent = street ? street + ' · ' : '';
    const turn = t.querySelector('.to-act-line');
    if (turn) {
      const who = snap.toAct != null && names[snap.toAct];
      turn.innerHTML = !who ? '' : `<b>${who}</b> to act<span class="price">${snap.toCall
        ? `call ${fmt(snap.toCall)} · ${priceText(snap.potOdds, snap.callCloses)}` : 'can check'}</span>`;
    }
  }

  return { steps, timeline, frameOf, html, show, card, cards, SUIT, madeHand, drawOf, callCloses, priceText };
});
