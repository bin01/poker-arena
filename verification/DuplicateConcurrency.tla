------------------------- MODULE DuplicateConcurrency -------------------------
(*
 * Formal model of Poker Arena's concurrent duplicate-pairing semaphore
 * acquisition, FIFO wait queues, and sibling/external cancellation.
 *
 * Sync commit: 36b41c1b773abcf93f176848f15063de22906b41
 * Covered code:
 *   - poker_arena/runner.py:  run_duplicate / one (915-947),
 *                             Lifecycle.pairing called by on_start (1319-1323, 1457),
 *                             run_plan semaphore setup & gathers (1428, 1454-1458, 1495-1502)
 *   - poker_arena/cli.py:     --concurrency default 2 (659, 703); runner.py defaults (952, 1353)
 *
 * Scope & bounds:
 *   - Two CLI semaphores ("agy", "claude") with capacity CAP = 2.
 *   - One duplicate pairing ("a_first", "b_first") acquiring both semaphores in
 *     sorted order ["agy", "claude"], plus two concurrent single-CLI peers
 *     ("peer1", "peer2") needing only "claude".
 *   - Faithfully models CPython asyncio.Semaphore non-yielding fast path, FIFO
 *     wait queues, and one()'s `acquired = []` tracked inside `try/finally`:
 *       * Both `await sem.acquire()` and `on_start()` execute INSIDE `try/finally`,
 *         releasing every acquired semaphore in reverse order on return, error,
 *         or cancellation at any `await` point.
 *   - Out of scope: RateGate timing/backoff loops (backends.py:147-164) and
 *     on_start() exceptions on peer tasks.
 *)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    CAP          \* Per-CLI semaphore capacity (2 = default concurrency)

ASSUME CAP \in Nat /\ CAP >= 1

Sems    == {"agy", "claude"}
DupPair == {"a_first", "b_first"}
Peers   == {"peer1", "peer2"}
Orders  == DupPair \cup Peers

Sibling(o) == IF o = "a_first" THEN "b_first" ELSE "a_first"

PCs == {"init", "wait_agy", "wait_claude", "running", "done", "failed", "cancelled"}
TerminalPCs == {"done", "failed", "cancelled"}

RECURSIVE Remove(_, _)
Remove(seq, elem) ==
    IF seq = <<>> THEN <<>>
    ELSE IF Head(seq) = elem THEN Remove(Tail(seq), elem)
    ELSE <<Head(seq)>> \o Remove(Tail(seq), elem)

VARIABLES
    avail,       \* [Sems -> 0..CAP]: currently available semaphore permits
    q,           \* [Sems -> Seq(Orders)]: FIFO wait queues matching Semaphore._waiters
    pc,          \* [Orders -> PCs]: control point of each task
    held         \* [Orders -> SUBSET Sems]: permits currently held

vars == <<avail, q, pc, held>>

TypeOK ==
    /\ avail \in [Sems -> 0..CAP]
    /\ \A s \in Sems : Len(q[s]) <= Cardinality(Orders)
    /\ \A s \in Sems : \A i \in 1..Len(q[s]) : q[s][i] \in Orders
    /\ pc \in [Orders -> PCs]
    /\ held \in [Orders -> SUBSET Sems]

Init ==
    /\ avail = [s \in Sems |-> CAP]
    /\ q = [s \in Sems |-> <<>>]
    /\ pc = [o \in Orders |-> "init"]
    /\ held = [o \in Orders |-> {}]

\* Note: CanTakeNow(s) requires q[s] = <<>>, so any released permit while q[s] is
\* non-empty is exclusively reserved for Head(q[s]) — matching CPython's
\* asyncio.Semaphore._wake_up_next() reservation and cancel-on-wakeup re-release.
CanTakeNow(s) == avail[s] > 0 /\ q[s] = <<>>

ReleaseHeld(o) ==
    [s \in Sems |-> IF s \in held[o] THEN avail[s] + 1 ELSE avail[s]]

\* --- Duplicate pair entry: non-yielding fast path INSIDE try/finally (runner.py:913-926) ---

DupStartFast(o) ==
    /\ o \in DupPair
    /\ pc[o] = "init"
    /\ CanTakeNow("agy")
    /\ CanTakeNow("claude")
    /\ avail' = [avail EXCEPT !["agy"] = @ - 1, !["claude"] = @ - 1]
    /\ held' = [held EXCEPT ![o] = {"agy", "claude"}]
    /\ pc' = [pc EXCEPT ![o] = "running"]
    /\ UNCHANGED q

\* on_start() runs inside try/finally -> acquires both, raises, finally releases both in reverse order!
DupStartFastOnStartRaises(o) ==
    /\ o \in DupPair
    /\ pc[o] = "init"
    /\ CanTakeNow("agy")
    /\ CanTakeNow("claude")
    /\ pc' = [pc EXCEPT ![o] = "failed"]
    /\ UNCHANGED <<avail, held, q>>

DupBlockOnSecond(o) ==
    /\ o \in DupPair
    /\ pc[o] = "init"
    /\ CanTakeNow("agy")
    /\ ~CanTakeNow("claude")
    /\ avail' = [avail EXCEPT !["agy"] = @ - 1]
    /\ held' = [held EXCEPT ![o] = {"agy"}]
    /\ q' = [q EXCEPT !["claude"] = Append(@, o)]
    /\ pc' = [pc EXCEPT ![o] = "wait_claude"]

DupBlockOnFirst(o) ==
    /\ o \in DupPair
    /\ pc[o] = "init"
    /\ ~CanTakeNow("agy")
    /\ q' = [q EXCEPT !["agy"] = Append(@, o)]
    /\ pc' = [pc EXCEPT ![o] = "wait_agy"]
    /\ UNCHANGED <<avail, held>>

\* --- FIFO wakeups for duplicate pair orders ---

DupWakeFirstFast(o) ==
    /\ o \in DupPair
    /\ pc[o] = "wait_agy"
    /\ q["agy"] # <<>> /\ Head(q["agy"]) = o
    /\ avail["agy"] > 0
    /\ CanTakeNow("claude")
    /\ avail' = [avail EXCEPT !["agy"] = @ - 1, !["claude"] = @ - 1]
    /\ held' = [held EXCEPT ![o] = {"agy", "claude"}]
    /\ q' = [q EXCEPT !["agy"] = Tail(@)]
    /\ pc' = [pc EXCEPT ![o] = "running"]

\* Wakes on "agy", takes "claude", on_start() raises inside try/finally -> releases both!
DupWakeFirstFastOnStartRaises(o) ==
    /\ o \in DupPair
    /\ pc[o] = "wait_agy"
    /\ q["agy"] # <<>> /\ Head(q["agy"]) = o
    /\ avail["agy"] > 0
    /\ CanTakeNow("claude")
    /\ q' = [q EXCEPT !["agy"] = Tail(@)]
    /\ pc' = [pc EXCEPT ![o] = "failed"]
    /\ UNCHANGED <<avail, held>>

DupWakeFirstBlockSecond(o) ==
    /\ o \in DupPair
    /\ pc[o] = "wait_agy"
    /\ q["agy"] # <<>> /\ Head(q["agy"]) = o
    /\ avail["agy"] > 0
    /\ ~CanTakeNow("claude")
    /\ avail' = [avail EXCEPT !["agy"] = @ - 1]
    /\ held' = [held EXCEPT ![o] = {"agy"}]
    /\ q' = [q EXCEPT !["agy"] = Tail(@), !["claude"] = Append(@, o)]
    /\ pc' = [pc EXCEPT ![o] = "wait_claude"]

DupWakeSecond(o) ==
    /\ o \in DupPair
    /\ pc[o] = "wait_claude"
    /\ q["claude"] # <<>> /\ Head(q["claude"]) = o
    /\ avail["claude"] > 0
    /\ avail' = [avail EXCEPT !["claude"] = @ - 1]
    /\ held' = [held EXCEPT ![o] = @ \cup {"claude"}]
    /\ q' = [q EXCEPT !["claude"] = Tail(@)]
    /\ pc' = [pc EXCEPT ![o] = "running"]

\* Wakes on "claude" (already holds "agy"), on_start() raises inside try/finally -> releases both!
DupWakeSecondOnStartRaises(o) ==
    /\ o \in DupPair
    /\ pc[o] = "wait_claude"
    /\ q["claude"] # <<>> /\ Head(q["claude"]) = o
    /\ avail["claude"] > 0
    /\ avail' = ReleaseHeld(o)
    /\ held' = [held EXCEPT ![o] = {}]
    /\ q' = [q EXCEPT !["claude"] = Tail(@)]
    /\ pc' = [pc EXCEPT ![o] = "failed"]

\* --- Concurrent single-CLI peers (e.g., claude vs bot): acquire only "claude" ---

PeerStartFast(p) ==
    /\ p \in Peers
    /\ pc[p] = "init"
    /\ CanTakeNow("claude")
    /\ avail' = [avail EXCEPT !["claude"] = @ - 1]
    /\ held' = [held EXCEPT ![p] = {"claude"}]
    /\ pc' = [pc EXCEPT ![p] = "running"]
    /\ UNCHANGED q

PeerBlock(p) ==
    /\ p \in Peers
    /\ pc[p] = "init"
    /\ ~CanTakeNow("claude")
    /\ q' = [q EXCEPT !["claude"] = Append(@, p)]
    /\ pc' = [pc EXCEPT ![p] = "wait_claude"]
    /\ UNCHANGED <<avail, held>>

PeerWake(p) ==
    /\ p \in Peers
    /\ pc[p] = "wait_claude"
    /\ q["claude"] # <<>> /\ Head(q["claude"]) = p
    /\ avail["claude"] > 0
    /\ avail' = [avail EXCEPT !["claude"] = @ - 1]
    /\ held' = [held EXCEPT ![p] = {"claude"}]
    /\ q' = [q EXCEPT !["claude"] = Tail(@)]
    /\ pc' = [pc EXCEPT ![p] = "running"]

\* --- Normal completion or failure INSIDE try/finally ---

OrderSucceeds(o) ==
    /\ o \in Orders
    /\ pc[o] = "running"
    /\ avail' = ReleaseHeld(o)
    /\ held' = [held EXCEPT ![o] = {}]
    /\ pc' = [pc EXCEPT ![o] = "done"]
    /\ UNCHANGED q

OrderFails(o) ==
    /\ o \in Orders
    /\ pc[o] = "running"
    /\ avail' = ReleaseHeld(o)
    /\ held' = [held EXCEPT ![o] = {}]
    /\ pc' = [pc EXCEPT ![o] = "failed"]
    /\ UNCHANGED q

\* --- Cancellation inside try/finally (runner.py:913-934) ---
\* Because every acquire is tracked in `acquired` inside `try/finally`, cancellation at
\* ANY await point (wait_agy, wait_claude, running) releases every permit in held[o].

CancelOrder(o) ==
    /\ pc[o] \in {"wait_agy", "wait_claude", "running"}
    /\ q' = [s \in Sems |-> Remove(q[s], o)]
    /\ avail' = ReleaseHeld(o)
    /\ held' = [held EXCEPT ![o] = {}]
    /\ pc' = [pc EXCEPT ![o] = "cancelled"]

CancelSibling(o) ==
    /\ o \in DupPair
    /\ pc[Sibling(o)] = "failed"
    /\ CancelOrder(o)

\* External cancellation of run_duplicate cancels EVERY live order in DupPair together
\* (matching runner.py:928-934 `except BaseException: for t in tasks: t.cancel()`).
ExternalCancelPair ==
    LET live == {o \in DupPair : pc[o] \in {"wait_agy", "wait_claude", "running"}}
    IN /\ live # {}
       /\ \A o \in DupPair : pc[o] \in {"wait_agy", "wait_claude", "running"} \cup TerminalPCs
       /\ q' = [s \in Sems |->
               LET q1 == IF "a_first" \in live THEN Remove(q[s], "a_first") ELSE q[s]
               IN  IF "b_first" \in live THEN Remove(q1, "b_first") ELSE q1]
       /\ avail' = [s \in Sems |-> avail[s] + Cardinality({o \in live : s \in held[o]})]
       /\ held' = [o \in Orders |-> IF o \in live THEN {} ELSE held[o]]
       /\ pc' = [o \in Orders |-> IF o \in live THEN "cancelled" ELSE pc[o]]

Done ==
    /\ \A o \in Orders : pc[o] \in TerminalPCs
    /\ UNCHANGED vars

Next ==
    \/ \E o \in DupPair :
           \/ DupStartFast(o)
           \/ DupStartFastOnStartRaises(o)
           \/ DupBlockOnSecond(o)
           \/ DupBlockOnFirst(o)
           \/ DupWakeFirstFast(o)
           \/ DupWakeFirstFastOnStartRaises(o)
           \/ DupWakeFirstBlockSecond(o)
           \/ DupWakeSecond(o)
           \/ DupWakeSecondOnStartRaises(o)
           \/ CancelSibling(o)
    \/ ExternalCancelPair
    \/ \E p \in Peers :
           \/ PeerStartFast(p)
           \/ PeerBlock(p)
           \/ PeerWake(p)
    \/ \E o \in Orders :
           \/ OrderSucceeds(o)
           \/ OrderFails(o)
    \/ Done

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------

LiveOrders == {o \in Orders : pc[o] \notin TerminalPCs}

BoundedConcurrency ==
    \A s \in Sems : 0 <= avail[s] /\ avail[s] <= CAP

PermitConservation ==
    \A s \in Sems :
        avail[s] + Cardinality({o \in LiveOrders : s \in held[o]}) = CAP

NoPermitLeakOnCancel ==
    \A o \in DupPair : pc[o] = "cancelled" => held[o] = {}

NoPermitLeakOnStartError ==
    \A o \in DupPair : pc[o] = "failed" => held[o] = {}

=============================================================================
