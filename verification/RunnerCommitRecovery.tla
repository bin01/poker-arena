------------------------- MODULE RunnerCommitRecovery -------------------------
(*
 * Formal model of Poker Arena's per-hand commit, crash-recovery, session-rebuild,
 * and external reader visibility protocol.
 *
 * Sync commit: 36b41c1b773abcf93f176848f15063de22906b41
 * Covered code:
 *   - poker_arena/runner.py:  _write (215-222), _has_uncommitted_logs (235-250),
 *                             _reconcile (253-281), committed_records (300-325),
 *                             verify_match (562-611), run_match (678-906)
 *   - poker_arena/agents.py:  ModelAgent.state/restore (122-128),
 *                             ModelAgent.act tool_use/unverified fallback (135-191)
 *   - poker_arena/backends.py: Backend.state (300-302), restore (304-310),
 *                              rebuild (312-317), forget_last (319-324),
 *                              send (352-412)
 *   - poker_arena/ratings.py:  _is_modern_record (331-335), _hands (338-360),
 *                              read_conversation (609-665)
 *
 * Scope & bounds:
 *   - One modern (record_version = 2) match folder with MAX_HANDS = 3 hands
 *     (indices 0, 1, 2 — covers pre-first checkpoint, mid-match crash recovery,
 *     and post-tool-fallback resume), one stateful CLI seat whose workdir
 *     survives crashes, and up to MAX_CRASHES = 2 process crashes at any
 *     non-terminal step.
 *   - Atomic file replacement (_write via temp+fsync+os.replace) is one step;
 *     appending to hands.jsonl and writing state.json are separate steps.
 *   - Recovery persists rebuild intent into state.json BEFORE _reconcile cuts
 *     hands.jsonl and events.jsonl one file at a time (runner.py:725-739).
 *   - Reader visibility models ratings._hands(d, cfg) for modern matches
 *     (record_version >= 2), failing closed to {} when state.json is absent.
 *   - Out of scope: torn partial JSON lines, duplicate rows in hands.jsonl,
 *     detached CLI subprocesses surviving SIGKILL, replay hash mismatches.
 *)

EXTENDS Naturals, FiniteSets

CONSTANTS
    MAX_HANDS,     \* Number of hands in the match (3 covers pre-checkpoint, recovery, post-fallback)
    MAX_CRASHES    \* Maximum process crashes explored (2 covers crash during recovery)

ASSUME MAX_HANDS \in Nat /\ MAX_HANDS >= 1
ASSUME MAX_CRASHES \in Nat /\ MAX_CRASHES >= 1

Hands == 0..(MAX_HANDS - 1)
MaxSid == MAX_HANDS + MAX_CRASHES + 2
Sids == 1..MaxSid
OptSids == 0..MaxSid   \* 0 represents Python None

VARIABLES
    \* --- Durable on-disk state (survives Crash) ---
    cfg_exists,        \* config.json written with record_version = 2
    events_hands,      \* subset of Hands with rows in events.jsonl
    hands_log,         \* subset of Hands with rows in hands.jsonl
    state_exists,      \* state.json exists on disk
    state_committed,   \* state.json["hands_committed"] in 0..MAX_HANDS
    state_session,     \* state.json["agents"][0]["backend"]["session"]
    state_has_turns,   \* bool(state.json["agents"][0]["backend"]["turns"])
    state_rebuilding,  \* state.json["agents"][0]["backend"]["rebuilding"] (backends.py:301)
    result_exists,     \* result.json exists on disk
    cli_dirty,         \* [Sids -> BOOLEAN]: CLI session on disk holds an uncommitted turn
    next_sid,          \* next fresh CLI session id to allocate

    \* --- Volatile in-memory process state (wiped on Crash) ---
    pc,                \* program counter in run_match
    mem_done,          \* local `done` from state.json
    mem_k,             \* current hand index k
    mem_session,       \* agent.backend.session in memory
    mem_has_turns,     \* bool(agent.backend.turns) in memory
    mem_rebuilding,    \* agent.backend.rebuilding in memory

    \* --- Verification ghost state ---
    crashes,           \* number of process crashes so far
    used_dirty_session,\* TRUE iff SendCliTurn resumed a CLI session with cli_dirty[sid] = TRUE
    lost_history       \* TRUE iff SendCliTurn opened a fresh CLI session with turns>0 and rebuilding=FALSE

diskVars == <<cfg_exists, events_hands, hands_log, state_exists, state_committed,
              state_session, state_has_turns, state_rebuilding, result_exists,
              cli_dirty, next_sid>>
memVars  == <<pc, mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding>>
vars     == <<cfg_exists, events_hands, hands_log, state_exists, state_committed,
              state_session, state_has_turns, state_rebuilding, result_exists,
              cli_dirty, next_sid,
              pc, mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
              crashes, used_dirty_session, lost_history>>

PCs == {"init", "load_state", "persist_rebuild", "reconcile_hands", "reconcile_events",
        "start_loop", "hand_start", "cli_turn", "append_hand", "write_state",
        "write_result", "finished", "crashed"}

TypeOK ==
    /\ cfg_exists \in BOOLEAN
    /\ events_hands \subseteq Hands
    /\ hands_log \subseteq Hands
    /\ state_exists \in BOOLEAN
    /\ state_committed \in 0..MAX_HANDS
    /\ state_session \in OptSids
    /\ state_has_turns \in BOOLEAN
    /\ state_rebuilding \in BOOLEAN
    /\ result_exists \in BOOLEAN
    /\ cli_dirty \in [Sids -> BOOLEAN]
    /\ next_sid \in 1..(MaxSid + 1)
    /\ pc \in PCs
    /\ mem_done \in 0..MAX_HANDS
    /\ mem_k \in 0..MAX_HANDS
    /\ mem_session \in OptSids
    /\ mem_has_turns \in BOOLEAN
    /\ mem_rebuilding \in BOOLEAN
    /\ crashes \in 0..MAX_CRASHES
    /\ used_dirty_session \in BOOLEAN
    /\ lost_history \in BOOLEAN

Init ==
    /\ cfg_exists = FALSE
    /\ events_hands = {}
    /\ hands_log = {}
    /\ state_exists = FALSE
    /\ state_committed = 0
    /\ state_session = 0
    /\ state_has_turns = FALSE
    /\ state_rebuilding = FALSE
    /\ result_exists = FALSE
    /\ cli_dirty = [s \in Sids |-> FALSE]
    /\ next_sid = 1
    /\ pc = "init"
    /\ mem_done = 0
    /\ mem_k = 0
    /\ mem_session = 0
    /\ mem_has_turns = FALSE
    /\ mem_rebuilding = FALSE
    /\ crashes = 0
    /\ used_dirty_session = FALSE
    /\ lost_history = FALSE

\* First run writes config.json with record_version = 2
WriteConfig ==
    /\ pc = "init"
    /\ cfg_exists' = TRUE
    /\ pc' = "load_state"
    /\ UNCHANGED <<events_hands, hands_log, state_exists, state_committed, state_session,
                   state_has_turns, state_rebuilding, result_exists, cli_dirty, next_sid,
                   mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* Short-circuit on result.json, else restore state.json including rebuilding flag (backends.py:304-310)
LoadState ==
    /\ pc = "load_state"
    /\ IF result_exists THEN
           /\ pc' = "finished"
           /\ UNCHANGED <<mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding>>
       ELSE
           /\ mem_done' = IF state_exists THEN state_committed ELSE 0
           /\ mem_session' = IF state_exists THEN state_session ELSE 0
           /\ mem_has_turns' = IF state_exists THEN state_has_turns ELSE FALSE
           /\ mem_rebuilding' =
                  IF state_exists THEN (state_rebuilding \/ (state_session = 0 /\ state_has_turns))
                  ELSE FALSE
           /\ mem_k' = 0
           /\ pc' = "persist_rebuild"
    /\ UNCHANGED <<diskVars, crashes, used_dirty_session, lost_history>>

\* runner.py:725-736: peek uncommitted logs; if rebuild needed, rebuild AND persist state.json BEFORE _reconcile!
PersistRebuildBeforeReconcile ==
    /\ pc = "persist_rebuild"
    /\ LET needsRebuild == (\E h \in (hands_log \cup events_hands) : h >= mem_done) \/ mem_rebuilding
       IN IF needsRebuild THEN
              /\ mem_session' = 0
              /\ mem_rebuilding' = mem_has_turns
              /\ IF state_exists THEN
                     /\ state_session' = 0
                     /\ state_rebuilding' = mem_has_turns
                 ELSE
                     /\ UNCHANGED <<state_session, state_rebuilding>>
          ELSE
              /\ UNCHANGED <<mem_session, mem_rebuilding, state_session, state_rebuilding>>
    /\ pc' = "reconcile_hands"
    /\ UNCHANGED <<cfg_exists, events_hands, hands_log, state_exists, state_committed,
                   state_has_turns, result_exists, cli_dirty, next_sid, mem_done, mem_k,
                   mem_has_turns, crashes, used_dirty_session, lost_history>>

\* _reconcile cuts hands.jsonl first, then events.jsonl (non-atomic across files)
ReconcileHandsLog ==
    /\ pc = "reconcile_hands"
    /\ hands_log' = {h \in hands_log : h < mem_done}
    /\ pc' = "reconcile_events"
    /\ UNCHANGED <<cfg_exists, events_hands, state_exists, state_committed, state_session,
                   state_has_turns, state_rebuilding, result_exists, cli_dirty, next_sid,
                   mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

ReconcileEventsLog ==
    /\ pc = "reconcile_events"
    /\ events_hands' = {h \in events_hands : h < mem_done}
    /\ pc' = "start_loop"
    /\ UNCHANGED <<cfg_exists, hands_log, state_exists, state_committed, state_session,
                   state_has_turns, state_rebuilding, result_exists, cli_dirty, next_sid,
                   mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

StartHandsLoop ==
    /\ pc = "start_loop"
    /\ mem_k' = mem_done
    /\ pc' = IF mem_done < MAX_HANDS THEN "hand_start" ELSE "write_result"
    /\ UNCHANGED <<diskVars, mem_done, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* flush() appends hand_start/hole/blind for hand mem_k to events.jsonl before asking any seat
FlushHandStart ==
    /\ pc = "hand_start"
    /\ mem_k < MAX_HANDS
    /\ events_hands' = events_hands \cup {mem_k}
    /\ pc' = "cli_turn"
    /\ UNCHANGED <<cfg_exists, hands_log, state_exists, state_committed, state_session,
                   state_has_turns, state_rebuilding, result_exists, cli_dirty, next_sid,
                   mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* Normal CLI turn (creates fresh session if mem_session=0, else resumes)
SendCliTurn ==
    /\ pc = "cli_turn"
    /\ IF mem_session = 0 THEN
           /\ next_sid <= MaxSid
           /\ lost_history' = (lost_history \/ (mem_has_turns /\ ~mem_rebuilding))
           /\ mem_session' = next_sid
           /\ mem_has_turns' = TRUE
           /\ mem_rebuilding' = FALSE
           /\ cli_dirty' = [cli_dirty EXCEPT ![next_sid] = TRUE]
           /\ next_sid' = next_sid + 1
           /\ UNCHANGED used_dirty_session
       ELSE
           /\ used_dirty_session' = (used_dirty_session \/ cli_dirty[mem_session])
           /\ mem_has_turns' = TRUE
           /\ cli_dirty' = [cli_dirty EXCEPT ![mem_session] = TRUE]
           /\ UNCHANGED <<mem_session, mem_rebuilding, next_sid, lost_history>>
    /\ pc' = "append_hand"
    /\ UNCHANGED <<cfg_exists, events_hands, hands_log, state_exists, state_committed,
                   state_session, state_has_turns, state_rebuilding, result_exists,
                   mem_done, mem_k, crashes>>

\* agents.py:160-170,191 + backends.py:319-324: reply rejected for tool_use exhausts max_fixes ->
\* forget_last() drops session (mem_session'=0, mem_rebuilding'=mem_has_turns) and hand plays fallback
SendCliTurnToolFallback ==
    /\ pc = "cli_turn"
    /\ mem_has_turns
    /\ mem_session # 0
    /\ used_dirty_session' = (used_dirty_session \/ cli_dirty[mem_session])
    /\ cli_dirty' = [cli_dirty EXCEPT ![mem_session] = TRUE]
    /\ mem_session' = 0
    /\ mem_rebuilding' = TRUE
    /\ pc' = "append_hand"
    /\ UNCHANGED <<cfg_exists, events_hands, hands_log, state_exists, state_committed,
                   state_session, state_has_turns, state_rebuilding, result_exists,
                   next_sid, mem_done, mem_k, mem_has_turns, crashes, lost_history>>

\* Append finished hand summary to hands.jsonl and fsync logs
AppendHandsLog ==
    /\ pc = "append_hand"
    /\ hands_log' = hands_log \cup {mem_k}
    /\ pc' = "write_state"
    /\ UNCHANGED <<cfg_exists, events_hands, state_exists, state_committed, state_session,
                   state_has_turns, state_rebuilding, result_exists, cli_dirty, next_sid,
                   mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* Atomic _write of state.json committing hand mem_k (Backend.state() persists rebuilding)
WriteStateCheckpoint ==
    /\ pc = "write_state"
    /\ state_exists' = TRUE
    /\ state_committed' = mem_k + 1
    /\ state_session' = mem_session
    /\ state_has_turns' = mem_has_turns
    /\ state_rebuilding' = mem_rebuilding
    /\ cli_dirty' = IF mem_session \in Sids THEN [cli_dirty EXCEPT ![mem_session] = FALSE] ELSE cli_dirty
    /\ mem_k' = mem_k + 1
    /\ pc' = IF mem_k + 1 < MAX_HANDS THEN "hand_start" ELSE "write_result"
    /\ UNCHANGED <<cfg_exists, events_hands, hands_log, result_exists, next_sid,
                   mem_done, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* Write result.json once all hands are committed
WriteResult ==
    /\ pc = "write_result"
    /\ result_exists' = TRUE
    /\ pc' = "finished"
    /\ UNCHANGED <<cfg_exists, events_hands, hands_log, state_exists, state_committed,
                   state_session, state_has_turns, state_rebuilding, cli_dirty, next_sid,
                   mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* Process crash (SIGKILL / BackendError / Ctrl-C) at any non-terminal step, wiping volatile memory
Crash ==
    /\ pc \notin {"init", "finished", "crashed"}
    /\ crashes < MAX_CRASHES
    /\ pc' = "crashed"
    /\ mem_done' = 0
    /\ mem_k' = 0
    /\ mem_session' = 0
    /\ mem_has_turns' = FALSE
    /\ mem_rebuilding' = FALSE
    /\ crashes' = crashes + 1
    /\ UNCHANGED <<diskVars, used_dirty_session, lost_history>>

\* Operator reruns the command to resume a stopped match (or idempotently reruns a finished match)
Restart ==
    /\ pc = "crashed"
    /\ pc' = IF cfg_exists THEN "load_state" ELSE "init"
    /\ UNCHANGED <<diskVars, mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

RerunFinished ==
    /\ pc = "finished"
    /\ pc' = "load_state"
    /\ UNCHANGED <<diskVars, mem_done, mem_k, mem_session, mem_has_turns, mem_rebuilding,
                   crashes, used_dirty_session, lost_history>>

\* Valid terminal state stuttering
Done ==
    /\ pc = "finished"
    /\ UNCHANGED vars

Next ==
    \/ WriteConfig
    \/ LoadState
    \/ PersistRebuildBeforeReconcile
    \/ ReconcileHandsLog
    \/ ReconcileEventsLog
    \/ StartHandsLoop
    \/ FlushHandStart
    \/ SendCliTurn
    \/ SendCliTurnToolFallback
    \/ AppendHandsLog
    \/ WriteStateCheckpoint
    \/ WriteResult
    \/ Crash
    \/ Restart
    \/ RerunFinished
    \/ Done

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* External ratings reader model: ratings._hands(d, cfg) for modern matches
   (record_version >= 2) at ratings.py:331-358 and 621. *)

ReaderCommittedHands ==
    IF ~cfg_exists THEN {}
    ELSE IF state_exists THEN {h \in hands_log : h < state_committed}
    ELSE {}    \* ratings.py:345-347: modern match without state.json fails closed to 0 committed hands

TrueCommittedHands ==
    IF state_exists THEN 0..(state_committed - 1) ELSE {}

CommitSafety == ReaderCommittedHands = TrueCommittedHands

NoDirtySessionReuse == ~used_dirty_session

NoHistoryLossOnRestore == ~lost_history

CommittedLogMatchesCheckpoint ==
    pc \in {"hand_start", "write_result", "finished"} =>
        ({h \in hands_log : h < state_committed} = TrueCommittedHands)

=============================================================================
