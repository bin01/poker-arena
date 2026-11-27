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
      hole: (d.hole || [])[i] || [], shown: false, net: null, won: false, last: '' }));
    let collected = 0, board = [], street = 'preflop', actor = null;
    const out = [], log = [];
    const snap = (label, extra = {}) => {
      out.push({ step: out.length, street, board: [...board], actor, label, button: d.button,
        pot: collected + seats.reduce((s, x) => s + x.bet, 0), collected,
        seats: seats.map(s => ({ ...s, hole: [...s.hole] })), done: false, ...extra });
      return out.length - 1;
    };
    const sweep = () => { for (const s of seats) { collected += s.bet; s.bet = 0; if (!s.folded) s.last = ''; } };
    const put = (i, amt) => { const a = Math.min(amt, seats[i].stack); seats[i].stack -= a; seats[i].bet += a; if (seats[i].stack === 0) seats[i].allIn = true; };

    for (const e of evs) if (e.kind === 'blind') put(e.data.seat, e.data.amount);
    log.push({ kind: 'start', step: snap('Blinds posted'),
      blinds: evs.filter(e => e.kind === 'blind').map(e => ({ seat: e.data.seat, amount: e.data.amount })) });

    const thoughts = {};
    for (const e of evs) {
      const x = e.data || {};
      if (e.kind === 'thought') thoughts[x.seat] = x;
      else if (e.kind === 'action') {
        const s = seats[x.seat];
        if (x.action === 'fold') s.folded = true;
        else if (x.action === 'call') put(x.seat, x.amount || 0);
        else if (x.action === 'bet' || x.action === 'raise' || x.action === 'all_in') put(x.seat, Math.max(0, (x.amount || 0) - s.bet));
        if (x.all_in) s.allIn = true;
        actor = x.seat;
        s.last = x.action;
        const t = thoughts[x.seat] || null;
        delete thoughts[x.seat];
        log.push({ kind: 'action', step: snap(x.action, { lastAmount: x.amount, recordedToAct: x.to_act }), seat: x.seat, street, action: x.action,
          amount: x.amount, all_in: !!x.all_in, say: x.say || '', note: x.note || null, thought: t, pot: x.pot });
      } else if (e.kind === 'board') {
        sweep();
        street = x.street;
        board = x.board || [...board, ...(x.cards || [])];
        actor = null;
        log.push({ kind: 'board', step: snap(STREET_NAMES[street] || street), street, cards: x.cards || [] });
      } else if (e.kind === 'runout') {
        log.push({ kind: 'runout', step: out.length - 1 });
      } else if (e.kind === 'showdown') {
        sweep();
        actor = null;
        for (const [i, h] of Object.entries(x.hands || {})) { seats[i].shown = true; seats[i].best = h.best; }
        log.push({ kind: 'showdown', step: snap('Showdown'), hands: x.hands || {} });
      } else if (e.kind === 'result') {
        sweep();
        actor = null;
        if (x.board) board = x.board;
        const won = x.won || [];
        seats.forEach((s, i) => { s.net = (x.net || [])[i] ?? 0; s.won = (won[i] || 0) > 0; s.stack = startStack[i] + s.net; });
        collected = 0;
        const winners = seats.map((s, i) => s.won ? i : null).filter(i => i != null);
        log.push({ kind: 'result', step: snap('Result', { done: true, finalPot: x.pot, winners, showdown: !!x.showdown }),
          winners, pot: x.pot, showdown: !!x.showdown, net: x.net || [] });
      }
    }
    // who acts next: the recorded `to_act` when the log has it, else the seat of the next action
    out.forEach((st, k) => {
      const at = log.findIndex(e => e.step > k);
      const next = at < 0 ? null : log[at];
      st.toAct = st.done ? null : st.recordedToAct ?? (next && next.kind === 'action' ? next.seat : null);
      delete st.recordedToAct;
    });
    return { steps: out, log, n, button: d.button };
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
          <div class="sub"><span class="stack-amt"></span><span class="seat-act"></span><span class="seat-net"></span></div></div>
        <span class="hole"></span><span class="turn">to act</span></div>`;
    const top = seats.map((s, i) => i ? seat(s, i) : '').join('');
    return `<div class="ptable n${n} ${opts.compact ? 'compact' : ''}">
      <div class="seats-top">${top}</div>
      <div class="felt"><div class="felt-inner">
        <div class="wagers">${seats.map((_, i) => i ? wager(i, where) : '').join('')}</div>
        <div class="board">${cards([], 5)}</div>
        <div class="pot-line"><span class="pot-chips"></span><span><span class="step-label"></span>Pot <b class="pot-amt">0</b></span></div>
        <div class="to-act-line" aria-live="polite"></div>
        <div class="wagers">${wager(0, where)}</div>
      </div></div>
      <div class="seats-bottom">${seat(seats[0], 0)}</div></div>`;
  }

  const VERB = { fold: 'Folded', check: 'Check', call: 'Call', bet: 'Bet', raise: 'Raise', all_in: 'All in' };

  // Update a table made by `html` to a snapshot. `fmt` formats chips (e.g. as big blinds);
  // `hidden` lists seats whose cards stay face down until shown at showdown.
  function show(el, snap, { fmt = String, hidden = [], prev = null, names = [] } = {}) {
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
    if (turn) turn.textContent = snap.toAct != null && names[snap.toAct] ? `${names[snap.toAct]} to act` : '';
  }

  return { steps, html, show, card, cards, SUIT };
});
