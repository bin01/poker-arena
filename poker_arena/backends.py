"""Model backends: how a seat's conversation reaches a model.

Subscription backends drive the labs' official command-line tools, signed in with your own plan, so a
match costs nothing beyond the subscription:

    claude   Claude Code (`claude -p`), Claude Pro/Max. The CLI keeps the conversation (--resume).
    codex    Codex CLI (`codex exec`), ChatGPT Plus/Pro. The CLI keeps the conversation (exec resume).
    agy      Google Antigravity CLI (`agy -p`), Google account: Gemini models. The CLI keeps the conversation.
    gemini   Gemini CLI (`gemini -p`), for installs that still have it. Stateless: the transcript is re-sent.
    command  Any other CLI that reads a prompt on stdin and prints a reply (pi -p, ollama run, ...).
             Stateless.

One API backend is included for completeness: `openai_compat` (OpenRouter, Ollama, LM Studio, vLLM, or
any OpenAI-compatible chat endpoint), also stateless.

Every backend has the same shape: `await backend.send(text)` returns a Reply with the model's text, any
reasoning the tool exposes, usage and timing, and in `meta` the command line it ran (`argv`, prompts elided),
the tool's raw output (`raw`), the CLI session before and after, and every failed try before it
(`failed_attempts`, rate-limit waits included, each with its exit status, stdout and stderr). A failed try
that may have advanced a CLI's own conversation is followed by a rebuild, so the retry never builds on a
turn the record lacks. `meta['tool_use']` lists any tool the model ran or tried in its turn (the seat rejects
such a reply). `meta['audit']` says how far the turn could be checked: 'verified' (the backend's own record of
the turn was read in full), 'unverified' (a record should exist but couldn't be read, or a line of it couldn't be
parsed, `audit_reason` says which; the raw lines are kept: the seat rejects the reply
and asks again, as for tool use; gemini output without tool counts is one) or 'opaque' (this backend never exposes
one; its protocol says so, and the judge never counts such turns towards a decided result unless the design
preregistered it). Every reply also says who answered: `observed_models`, every model the backend's own output
names as answering the turn (claude: each assistant message's model, with the init event's model as supplementary
evidence that must agree; agy: each transcript step's; gemini: every model in its stats; openai_compat: the
response's; codex and command: none), `observed_model` (the one model when they agree), `identity_reported` and
`identity_match` ('match' only when every named model is the requested one, by ratings.same_model; 'mismatch' when
any is another; 'unattested' when none is named; the field every consumer judges identity by) and the older
`identity` (only whether a model was named). `evidence(raw)` re-derives a call's text, tool use, models and audit
from its retained raw output alone, and `verify_call` checks a recorded call against it. Every launch re-checks
the CLI it runs (`cli_artifact`: resolved path, version, sha256 of the file) and records it as `meta['cli']`; a
change from the pinned artifact (`cli_pin`) is kept in `cli_transitions`. `state()` / `restore()` let a stopped match resume; `rebuild()` starts a CLI conversation over
from the turns the match committed, for when the CLI's own conversation may hold turns the record dropped.
`protocol()` is what a player's results depend on, for its identity across runs.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field

RATE_LIMIT = re.compile(r'usage limit|rate.?limit|quota|too many requests|429|resource.?exhausted|limit reached'
                        r'|try again (?:at|in|later)', re.I)


class BackendError(RuntimeError):
    pass


_ARTIFACTS: dict = {}


def cli_artifact(exe: str | None) -> dict | None:
    """The CLI `exe` resolves to now: {exe, path (symlinks resolved), version (`--version`, first line), sha256}.
    Re-checked on every call: the file is re-hashed and asked its version only when its stat changed."""
    if not exe:
        return None
    path = shutil.which(exe)
    if not path:
        return {'exe': exe, 'path': None, 'version': None, 'sha256': None}
    real = os.path.realpath(path)
    try:
        st = os.stat(real)
    except OSError:
        return {'exe': exe, 'path': real, 'version': None, 'sha256': None}
    key = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)
    cached = _ARTIFACTS.get(real)
    if not cached or cached[0] != key:
        digest = hashlib.sha256()
        try:
            with open(real, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    digest.update(chunk)
            sha = digest.hexdigest()
        except OSError:
            sha = None
        version = None
        try:
            out = subprocess.run([real, '--version'], capture_output=True, text=True, timeout=30,
                                 stdin=subprocess.DEVNULL)
            lines = (out.stdout.strip() or out.stderr.strip()).splitlines()
            version = lines[0][:200] if lines else None
        except (OSError, subprocess.SubprocessError):
            pass
        cached = _ARTIFACTS[real] = (key, {'path': real, 'version': version, 'sha256': sha})
    return {'exe': exe, **cached[1]}


def artifact_changed(old: dict | None, new: dict | None) -> bool:
    return bool(old and new) and any(old.get(k) != new.get(k) for k in ('path', 'version', 'sha256'))


def cli_version(exe: str | None) -> str | None:
    """`<exe> --version`, first line, re-checked whenever the resolved file changes."""
    a = cli_artifact(exe)
    return a['version'] if a else None


def identity_fields(requested, observed) -> dict:
    """{identity_reported, identity_match} for one reply (see Backend.identity). `observed` is one model or every
    model the output named: all of them must be the requested one."""
    from .ratings import same_model
    models = [m for m in (observed if isinstance(observed, (list, tuple, set)) else [observed]) if m]
    if not models:
        return {'identity_reported': False, 'identity_match': 'unattested'}
    return {'identity_reported': True,
            'identity_match': 'match' if requested and all(same_model(requested, m) for m in models) else 'mismatch'}


def backend_exe(spec: dict) -> str | None:
    """The CLI a player's backend runs (None for bots and API backends), without starting it."""
    kind = spec.get('backend')
    if kind == 'command':
        return (spec.get('command') or [None])[0]
    cls = BACKENDS.get(kind)
    return getattr(cls, 'exe', None) if cls else None


@dataclass
class Reply:
    text: str
    thinking: str = ''
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    meta: dict = field(default_factory=dict)


class Backend:
    stateful = False
    exe: str | None = None   # the CLI this backend runs, for the version in the match's config
    PRIVATE = {'system', 'turns', 'session', 'workdir', 'label', 'key', 'rebuilding', 'last_call', 'cli_pin',
               'cli_transitions'}
    NOT_PROTOCOL = {'timeout', 'rate_limit_wait', 'max_wait', 'print_timeout'}   # settings that don't change play
    TOOLS = 'none offered'   # how this CLI keeps tools away from the model (tool_policy)
    AUDIT = 'verified'       # what a turn's record shows by default (see the module docstring)
    AUDIT_SOURCE = "the backend's own output"

    def __init__(self, system: str, *, model: str | None = None, timeout: float = 900, retries: int = 3,
                 rate_limit_wait: float = 900, max_wait: float = 6 * 3600, max_history_chars: int = 400_000,
                 extra_args: list[str] | None = None, label: str = '', **_ignored):
        self.system, self.model, self.timeout, self.retries = system, model, timeout, retries
        self.rate_limit_wait, self.max_wait = rate_limit_wait, max_wait
        self.max_history_chars = max_history_chars
        self.extra_args = list(extra_args or [])
        self.label = label
        self.turns: list[list[str]] = []   # [user, assistant] pairs (kept by stateless backends)
        self.session: str | None = None    # the CLI's own conversation id (stateful backends)
        self.workdir: str | None = None    # one empty folder per seat, kept for the match (CLIs key sessions by folder)
        self.rebuilding = False            # the next turn opens a new conversation that replays self.turns
        self.last_call: dict = {}
        self.cli_pin: dict | None = None   # the CLI artifact this seat must keep running (set by the runner)
        self.cli_transitions: list = []    # [{t, exe, from, to}] every launch that found another artifact

    def describe(self) -> dict:
        """Effective settings, defaults included (never the system prompt, conversation or keys)."""
        out = {k: v for k, v in vars(self).items() if k not in self.PRIVATE and not k.startswith('_')}
        return {**out, 'stateful': self.stateful}

    def version(self) -> str | None:
        return cli_version(self.exe)

    def tool_policy(self) -> dict:
        return {'restriction': self.TOOLS, 'tool_turns': 'rejected and asked again in a rebuilt conversation',
                'audit': {'default': self.AUDIT, 'source': self.AUDIT_SOURCE,
                          'unverified_turns': 'rejected and asked again in a rebuilt conversation'}}

    def protocol(self) -> dict:
        """The settings that change how this seat plays (not timeouts or waits), with the tool policy."""
        out = {k: v for k, v in self.describe().items() if k not in self.NOT_PROTOCOL}
        return {**out, 'tool_policy': self.tool_policy()}

    # ---- state for resuming a stopped match -------------------------------------------------
    def state(self):
        return {'turns': self.turns, 'session': self.session, 'workdir': self.workdir}

    def restore(self, st):
        self.turns, self.session = st.get('turns', []), st.get('session')
        workdir = st.get('workdir')
        self.workdir = workdir if workdir and os.path.isdir(workdir) else None
        if self.session and not self.workdir:   # the CLI's session lived in a folder that is gone
            self.rebuild()

    def rebuild(self):
        """Drop the CLI's conversation; the next turn starts a new one that replays the committed turns.
        Stateless backends re-send their (committed) turns anyway."""
        if self.stateful:
            self.session = None
            self.rebuilding = bool(self.turns)

    def forget_last(self):
        """Take back the last turn (a reply that was rejected unseen, e.g. one that used a tool): it leaves the
        record, and a CLI's conversation, which holds it, is rebuilt without it."""
        if self.turns:
            self.turns.pop()
        self.rebuild()

    def _session_from(self, stdout: str) -> str | None:
        """The CLI session a (possibly failed) run's output names, if any."""
        return None

    # ---- the transcript a stateless backend re-sends ----------------------------------------
    def transcript(self, text: str, system: bool = True) -> str:
        """System prompt, earlier turns (oldest dropped first when too long) and the new turn, as one prompt."""
        turns = list(self.turns)
        def build(ts):
            parts = ([self.system, ''] if system else [
                'This match continues in a new conversation (the last one was interrupted or reset); your earlier '
                'turns are replayed.',
                '']) + ['=== THE MATCH SO FAR (your earlier turns and replies) ===']
            if len(ts) < len(self.turns):
                parts.append(f'[{len(self.turns) - len(ts)} earlier turns omitted for length; your notes carry what '
                             f'you chose to remember]')
            for u, a in ts:
                parts += ['', '>>> GAME:', u, '', '>>> YOU:', a]
            parts += ['', '=== NOW ===', '>>> GAME:', text, '', '>>> YOU (reply with the JSON object only):']
            return '\n'.join(parts)
        prompt = build(turns)
        while len(prompt) > self.max_history_chars and turns:
            turns = turns[max(1, len(turns) // 10):]
            prompt = build(turns)
        return prompt

    async def send(self, text: str) -> Reply:
        waited, attempt, failed = 0.0, 0, []
        while True:
            payload = self.transcript(text, system=False) if self.rebuilding else text
            start, t, before = time.monotonic(), round(time.time(), 1), self.session
            self.last_call = {}
            try:
                reply = await self._call(payload)
            except BackendError as e:
                msg = str(e)
                raw = self.last_call.get('raw') or {}
                after = self._session_from(raw.get('stdout') or '') or self.session
                rec = {'t': t, 'latency': round(time.monotonic() - start, 2), 'error': msg[:2000],
                       'argv': self.last_call.get('argv'), 'returncode': raw.get('returncode'),
                       'session_before': before, 'session_after': after, 'raw': raw or None}
                # a CLI that ran may have stored the turn in its conversation: drop that conversation, so the
                # retry replays the committed turns instead of building on one the record lacks
                if self.stateful and self.last_call.get('launched') and (before or after):
                    self.rebuild()
                    rec['rebuilt'] = True
                failed.append(rec)
                if RATE_LIMIT.search(msg) and waited < self.max_wait:
                    rec.update(rate_limited=True, waited=self.rate_limit_wait)
                    print(f'[{self.label}] rate limited; waiting {self.rate_limit_wait:.0f}s: {msg[:200]}', flush=True)
                    await asyncio.sleep(self.rate_limit_wait)
                    waited += self.rate_limit_wait
                    continue
                attempt += 1
                if attempt > self.retries:
                    e.meta = {'failed_attempts': failed}
                    raise
                rec['waited'] = min(5 * 2 ** attempt, 120)
                await asyncio.sleep(rec['waited'])
                continue
            reply.latency = round(time.monotonic() - start, 2)
            meta = {'argv': self.last_call.get('argv'), 'raw': self.last_call.get('raw'),
                    'session_before': before, 'session_after': self.session}
            if failed:
                meta['failed_attempts'] = failed
            if self.rebuilding:
                meta['rebuilt_from_turns'] = len(self.turns)
                self.rebuilding = False
            reply.meta = {**meta, **reply.meta}
            reply.meta.setdefault('audit', self.AUDIT)
            if self.last_call.get('cli'):
                reply.meta['cli'] = self.last_call['cli']
            models = reply.meta.get('observed_models') or ([reply.meta['observed_model']]
                                                           if reply.meta.get('observed_model') else [])
            reply.meta.update(self.identity(reply.meta.get('observed_model'), models))
            if reply.meta.get('raw') is None:   # no CLI ran (an in-process backend): what it returned is the record
                reply.meta['raw'] = {'synthetic': {'text': reply.text, 'tools': reply.meta.get('tool_use') or [],
                                                   'models': models, 'audit': reply.meta['audit']}}
            self.turns.append([text, reply.text])
            return reply

    async def _call(self, text: str) -> Reply:
        raise NotImplementedError

    def identity(self, observed: str | None = None, models: list | None = None) -> dict:
        """Who was asked and who answered, for one call: `requested_model` (the configured model, None for the
        CLI's default), `observed_model` (the model the backend's own output names, None when it names none),
        `identity_reported` (whether the output named a model at all) and `identity_match`: 'match' when the named
        model is the requested one (ratings.same_model), 'mismatch' when it is another model (or nothing was
        requested), 'unattested' when the output named none. `identity` ('attested' | 'unattested') only says
        whether a model was named; consumers judge identity by `identity_match`. `models` is every model the output
        named for the turn: identity_match is 'match' only when all of them are the requested one."""
        models = sorted({m for m in (models or [observed]) if m})
        out = {'requested_model': self.model, 'observed_model': observed or None,
               **identity_fields(self.model, models), 'identity': 'attested' if models else 'unattested'}
        if models:
            out['observed_models'] = models
        if getattr(self, 'effort', None) is not None:
            out['requested_effort'] = self.effort
        if observed:
            out['served_model'] = observed
        return out

    @classmethod
    def evidence(cls, raw) -> dict | None:
        """What a call's retained raw output alone says: {text, tools, models, audit}; None when it can't say.
        In-process backends (no CLI) keep their reply as `synthetic` evidence."""
        if isinstance(raw, dict) and isinstance(raw.get('synthetic'), dict):
            return raw['synthetic']
        return None

    # ---- running a CLI -----------------------------------------------------------------------
    def folder(self) -> str:
        """This seat's work folder: empty apart from files the backend puts there (no project files or
        instructions), and kept for the match, since CLIs key their sessions by folder."""
        if not self.workdir:
            self.workdir = tempfile.mkdtemp(prefix='poker-arena-')
        return self.workdir

    async def run(self, cmd: list[str], stdin: str | None = None, cwd: str | None = None) -> str:
        """Run a CLI; its exit status, stdout and stderr are kept in last_call['raw'] even when it fails or
        times out (what it printed before the kill)."""
        self.last_call['argv'] = [('<system prompt>' if a == self.system else f'<{len(a)} chars>' if len(a) > 500
                                   else a) for a in cmd]
        exe = shutil.which(cmd[0])
        if not exe:
            raise BackendError(f'{cmd[0]!r} is not installed or not on PATH')
        art = await asyncio.to_thread(cli_artifact, cmd[0])
        self.last_call['cli'] = art
        if self.cli_pin is None:
            self.cli_pin = art
        elif artifact_changed(self.cli_pin, art):
            self.cli_transitions.append({'t': round(time.time()), 'exe': cmd[0], 'from': self.cli_pin, 'to': art})
            self.cli_pin = art
        exe = art['path'] or exe
        workdir = cwd or self.folder()
        proc = await asyncio.create_subprocess_exec(
            exe, *cmd[1:], cwd=workdir, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        self.last_call['launched'] = True

        def kill():   # the CLI and anything it started
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                if proc.returncode is None:
                    proc.kill()
        bufs = ([], [])

        async def pump(stream, buf):
            while chunk := await stream.read(65536):
                buf.append(chunk)

        async def feed():
            if stdin is not None:
                try:
                    proc.stdin.write(stdin.encode())
                    await proc.stdin.drain()
                    proc.stdin.close()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        io = asyncio.ensure_future(asyncio.gather(pump(proc.stdout, bufs[0]), pump(proc.stderr, bufs[1]), feed()))
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(io), self.timeout)
            await proc.wait()
        except asyncio.CancelledError:
            kill()
            raise
        except asyncio.TimeoutError:
            timed_out = True
            kill()
            try:
                await asyncio.wait_for(asyncio.shield(io), 5)
            except asyncio.TimeoutError:
                io.cancel()
            await proc.wait()
        out_s, err_s = (b''.join(b).decode(errors='replace') for b in bufs)
        self.last_call['raw'] = {'returncode': proc.returncode, 'stdout': out_s, 'stderr': err_s,
                                 **({'timed_out': True} if timed_out else {})}
        if timed_out:
            raise BackendError(f'{cmd[0]} timed out after {self.timeout:.0f}s')
        if proc.returncode != 0:
            raise BackendError(f'{cmd[0]} exited {proc.returncode}: {(err_s or out_s).strip()[-1500:]}')
        return out_s


UNREADABLE_OUTPUT = 'unreadable output line'
UNREADABLE_TRANSCRIPT = 'unreadable transcript line'


def _unreadable(line) -> bool:
    """A non-blank line of a JSONL stream that isn't a JSON object."""
    if not line.strip():
        return False
    try:
        return not isinstance(json.loads(line), dict)
    except ValueError:
        return True


def _audit_of(parsed) -> dict:
    """{'audit': 'unverified', 'audit_reason': ...} when a parsed stream had lines that couldn't be read (what they
    held, a tool call included, is unknown), else {}."""
    return {'audit': 'unverified', 'audit_reason': UNREADABLE_OUTPUT} if parsed.get('unreadable') else {}


def _ids(stdout: str, key: str, types=None) -> str | None:
    """The last `key` in a JSONL stream (optionally only from events of the given types)."""
    found = None
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get(key) and (types is None or ev.get('type') in types):
            found = ev[key]
    return found


class ClaudeCLI(Backend):
    """Claude Code in print mode. The first turn sets the system prompt and opens a session; later turns
    resume it, so the model keeps its own earlier replies (and the CLI handles caching and compaction).
    Tools and MCP servers are switched off, and only project settings load (the folder is empty), so your
    own hooks and settings don't run inside the eval. Set `isolate = false` on the player to keep them.
    Thinking is shown as summaries (`--thinking-display summarized`) so the viewer can show it."""
    stateful = True
    exe = 'claude'
    TOOLS = "--tools '' (no built-in tools) and --strict-mcp-config with no servers"

    def __init__(self, system, *, isolate: bool = True, thinking_display: str | None = 'summarized',
                 effort: str | None = None, **kw):
        super().__init__(system, **kw)
        self.isolate, self.thinking_display, self.effort = isolate, thinking_display, effort

    async def _call(self, text):
        cmd = ['claude', '-p', '--output-format', 'stream-json', '--verbose', '--tools', '',
               '--strict-mcp-config', '--system-prompt', self.system]
        if self.isolate:
            cmd += ['--setting-sources', 'project']
        if self.thinking_display:  # otherwise thinking blocks arrive with empty text
            cmd += ['--thinking-display', self.thinking_display]
        if self.model:
            cmd += ['--model', self.model]
        if self.effort:  # low, medium, high, xhigh or max (players.toml); unset, Claude may not think at all
            cmd += ['--effort', self.effort]
        if self.session:
            cmd += ['--resume', self.session]
        out = await self.run(cmd + self.extra_args, stdin=text)
        p = self.parse(out)
        if p['result'] is None or p['is_error']:
            raise BackendError(f'claude returned no result: {out.strip()[-1500:]}')
        self.session = p['session'] or self.session
        models = p['models']
        meta = {'session': self.session, **_audit_of(p), 'observed_model': p['answered'][0] if len(p['answered']) == 1 else None,
                **({'observed_models': models} if models else {}),
                **({'answered_by': p['answered']} if p['answered'] else {}),
                **({'init_model': p['init_model']} if p['init_model'] else {}),
                **({'tool_use': p['tools']} if p['tools'] else {})}
        return Reply(p['result'], '\n'.join(p['thinking']), p['usage'], meta=meta)

    @staticmethod
    def parse(out: str) -> dict:
        """The stream-json output of one turn. `answered`: the model of every assistant message; `models`: those and
        the init event's model (supplementary: a turn whose init names another model than its messages is mixed),
        empty when no assistant message names one."""
        thinking, result, usage, session, is_error, tools = [], None, {}, None, False, []
        init, answered = None, []
        unreadable = [line for line in out.splitlines() if _unreadable(line)]
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get('type') == 'system' and ev.get('subtype') == 'init':
                init = ev.get('model') or init
            if ev.get('type') == 'assistant':
                if (ev.get('message') or {}).get('model'):
                    answered.append(ev['message']['model'])
                for block in (ev.get('message') or {}).get('content') or []:
                    if block.get('type') == 'thinking' and block.get('thinking'):
                        thinking.append(block['thinking'])
                    elif str(block.get('type', '')).endswith('tool_use'):
                        tools.append({'type': block['type'], 'name': block.get('name'),
                                      'input': json.dumps(block.get('input'), default=str)[:500]})
            elif ev.get('type') == 'result':
                result, usage = ev.get('result'), ev.get('usage') or {}
                session, is_error = ev.get('session_id'), ev.get('is_error', False)
                usage = {**usage, 'cost_usd_equivalent': ev.get('total_cost_usd')}
            if ev.get('session_id'):
                session = session or ev['session_id']
        models = sorted(set(answered) | ({init} if init and answered else set()))
        return {'thinking': thinking, 'result': result, 'usage': usage, 'session': session, 'is_error': is_error,
                'tools': tools, 'init_model': init, 'answered': sorted(set(answered)), 'models': models,
                'unreadable': unreadable}

    @classmethod
    def evidence(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get('stdout'), str):
            return None
        p = cls.parse(raw['stdout'])
        return {'text': p['result'], 'tools': p['tools'], 'models': p['models'],
                'audit': _audit_of(p).get('audit', cls.AUDIT)}

    def _session_from(self, stdout):
        return _ids(stdout, 'session_id')


CODEX_NO_TOOLS = ['-c', 'web_search="disabled"', '-c', 'mcp_servers={}', '-c', 'agents.enabled=false',
                  *[f for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'browser_use',
                                      'browser_use_external', 'in_app_browser', 'computer_use', 'image_generation',
                                      'view_image', 'multi_agent', 'goals', 'hooks', 'skill_search', 'tool_suggest',
                                      'sleep_tool', 'code_mode_host') for f in ('--disable', feature)]]
CODEX_SAFE_ITEMS = {'agent_message', 'assistant_message', 'reasoning', 'error', 'todo_list'}


class CodexCLI(Backend):
    """Codex CLI (`codex exec`), read-only sandbox in an empty folder. The game's system prompt replaces Codex's
    built-in coding instructions (`model_instructions_file`); later turns resume the same thread. Every tool
    Codex lets a flag switch off is off (CODEX_NO_TOOLS: shell, web search, MCP servers, subagents, apps,
    browser, images, ...), and with `isolate` (the default) your own config.toml and exec rules don't load.
    A turn that still ran a tool is rejected by the seat. Reasoning comes back as short summary titles."""
    stateful = True
    exe = 'codex'
    TOOLS = 'read-only sandbox; disabled: ' + ' '.join(CODEX_NO_TOOLS)

    def __init__(self, system, *, effort: str | None = 'high', summary: str | None = 'auto',
                 replace_instructions: bool = True, isolate: bool = True, **kw):
        super().__init__(system, **kw)
        self.effort, self.summary, self.replace_instructions = effort, summary, replace_instructions
        self.isolate = isolate

    async def _call(self, text):
        flags = ['--json', '--skip-git-repo-check', *CODEX_NO_TOOLS]
        if self.isolate:
            flags += ['--ignore-user-config', '--ignore-rules']
        if self.replace_instructions:
            path = os.path.join(self.folder(), 'instructions.md')
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(self.system)
            flags += ['-c', f'model_instructions_file="{path}"']
        if self.model:
            flags += ['-m', self.model]
        if self.effort:
            flags += ['-c', f'model_reasoning_effort="{self.effort}"']
        if self.summary:  # ask for reasoning summaries, so the viewer can show them
            flags += ['-c', f'model_reasoning_summary="{self.summary}"']
        flags += self.extra_args
        if self.session:
            cmd = ['codex', 'exec', 'resume', self.session, '-c', 'sandbox_mode="read-only"', *flags, '-']
            prompt = text
        else:
            cmd = ['codex', 'exec', '--sandbox', 'read-only', *flags, '-']
            prompt = text if self.replace_instructions else self.system + '\n\n=== THE MATCH BEGINS ===\n' + text
        out = await self.run(cmd, stdin=prompt)
        p = self.parse(out)
        self.session = p['session'] or self.session
        if p['failed']:
            raise BackendError(f'codex: {p["failed"][:1500]}')
        if not p['messages']:
            raise BackendError(f'codex returned no message: {out.strip()[-1500:]}')
        meta = {'session': self.session, **_audit_of(p), **({'tool_use': p['tools']} if p['tools'] else {})}
        return Reply(p['messages'][-1], '\n'.join(p['thinking']), p['usage'], meta=meta)

    @staticmethod
    def parse(out: str) -> dict:
        messages, thinking, usage, tools, session, failed = [], [], {}, {}, None, None
        unreadable = [line for line in out.splitlines() if _unreadable(line)]
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            t = ev.get('type', '')
            if t == 'thread.started':
                session = ev.get('thread_id') or session
            elif t in ('item.started', 'item.completed'):
                item = ev.get('item') or {}
                kind = item.get('type') or item.get('item_type')
                if kind not in CODEX_SAFE_ITEMS:
                    tools[item.get('id') or len(tools)] = {'type': kind, 'name': str(
                        item.get('command') or item.get('query') or item.get('tool') or item.get('server') or '')[:500]}
                if t == 'item.started' or kind == 'error':  # config warnings arrive as items; they don't fail the turn
                    continue
                if kind in ('agent_message', 'assistant_message'):
                    messages.append(item.get('text', ''))
                elif kind == 'reasoning':
                    thinking.append(item.get('text', ''))
            elif t == 'turn.completed':
                usage = ev.get('usage') or {}
            elif t in ('turn.failed', 'error') and failed is None:
                failed = json.dumps(ev)
        return {'messages': messages, 'thinking': thinking, 'usage': usage, 'tools': list(tools.values()),
                'session': session, 'failed': failed, 'unreadable': unreadable}

    @classmethod
    def evidence(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get('stdout'), str):
            return None
        p = cls.parse(raw['stdout'])
        return {'text': p['messages'][-1] if p['messages'] and not p['failed'] else None, 'tools': p['tools'],
                'models': [], 'audit': _audit_of(p).get('audit', cls.AUDIT)}

    def _session_from(self, stdout):
        return _ids(stdout, 'thread_id', {'thread.started'})


AGY_AGENT = """---
name: poker
description: A player in a poker match. Answers with one JSON object.
tools: []
subagent: false
---

"""
UNREADABLE_KEY = 'unreadable_line'
AGY_DENY = {'permissions': {'deny': ['mcp(*)', 'command(*)', 'read_url(*)', 'execute_url(*)', 'read_file(*)',
                                     'write_file(*)']}}


class AgyCLI(Backend):
    """Google Antigravity CLI (`agy -p`), signed in with a Google account: Gemini models.

    The game's system prompt is an agent definition in the seat's folder (`.agents/agents/poker/agent.md`,
    with `tools: []`), selected with `--agent poker`; a deny list in `--settings` blocks MCP servers, commands,
    files and URLs, and `--sandbox` limits the terminal. Later turns continue the conversation by id. stdin is
    closed (scripted output can go missing otherwise). The model is passed on every call: agy's default comes
    from your own settings and can be a non-Gemini model. Thinking isn't in agy's output, so it is read from
    the conversation's transcript, along with the model that actually answered."""
    stateful = True
    exe = 'agy'
    TOOLS = "agent with tools: [], settings deny " + ' '.join(AGY_DENY['permissions']['deny']) + ', --sandbox'
    BRAIN = os.path.expanduser('~/.gemini/antigravity-cli/brain')

    def __init__(self, system, *, effort=None, print_timeout='15m', lockdown: bool = True, **kw):
        super().__init__(system, **kw)
        self.effort, self.print_timeout, self.lockdown = effort, print_timeout, lockdown

    def _agent(self):
        d = os.path.join(self.folder(), '.agents', 'agents', 'poker')
        path = os.path.join(d, 'agent.md')
        if not os.path.exists(path):
            os.makedirs(d, exist_ok=True)
            with open(path, 'w') as f:
                f.write(AGY_AGENT + self.system + '\n')

    AUDIT_SOURCE = 'the conversation transcript agy writes under ' + BRAIN

    def _transcript(self):
        """The rows of this turn (the steps after the last user input) in the conversation's transcript, the user
        input included; None when the transcript can't be read."""
        path = os.path.join(self.BRAIN, self.session or '', '.system_generated', 'logs', 'transcript.jsonl')
        try:
            with open(path, 'rb') as f:
                lines = f.read().splitlines()
        except OSError:
            return None
        rows = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            rows.append(row if isinstance(row, dict) else {UNREADABLE_KEY: line.decode('utf-8', 'surrogateescape')})
        start = max((i for i, r in enumerate(rows) if r.get('type') == 'USER_INPUT'), default=-1)
        return rows[start:] if start >= 0 else [r for r in rows if UNREADABLE_KEY in r]

    @staticmethod
    def turn_problem(rows) -> str | None:
        """Why a turn's transcript rows can't verify it: None when they can."""
        if rows is None:
            return 'no readable transcript'
        if any(not isinstance(r, dict) or UNREADABLE_KEY in r for r in rows):
            return UNREADABLE_TRANSCRIPT
        if not (rows and rows[0].get('type') == 'USER_INPUT'):
            return 'no user input for this turn in the transcript'
        if not any(r.get('type') == 'PLANNER_RESPONSE' for r in rows[1:]):
            return 'no response to this turn in the transcript'
        return None

    @staticmethod
    def read_turn(rows) -> tuple:
        """(thinking, models, tool calls, audit) of a turn's transcript rows. audit is 'unverified' when there are
        none, one of them couldn't be read, or they hold no response for this turn: then nothing is known about
        the tools the turn ran."""
        if rows is None:
            return '', [], [], 'unverified'
        thinking, models, tools, seen = '', [], [], False
        started = bool(rows) and isinstance(rows[0], dict) and rows[0].get('type') == 'USER_INPUT'
        for rec in rows[1:] if started else []:
            if isinstance(rec, dict) and rec.get('type') == 'PLANNER_RESPONSE':
                seen = True
                thinking = rec.get('thinking') or thinking
                if rec.get('model'):
                    models.append(rec['model'])
                tools += [{'type': 'tool_call', 'name': c.get('name'), 'input': json.dumps(c.get('args'))[:500]}
                          for c in rec.get('tool_calls') or []]
        ok = started and seen and AgyCLI.turn_problem(rows) is None
        return thinking, sorted(set(models)), tools, 'verified' if ok else 'unverified'

    @staticmethod
    def _response(out: str) -> dict | None:
        try:
            data = json.loads(out[out.find('{'):])
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    @classmethod
    def evidence(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get('stdout'), str):
            return None
        data = cls._response(raw['stdout']) or {}
        _, models, tools, audit = cls.read_turn(raw.get('transcript'))
        return {'text': (data.get('response') or '') if data.get('status') == 'SUCCESS' else None, 'tools': tools,
                'models': models, 'audit': audit}

    async def _call(self, text):
        cmd = ['agy', '--output-format', 'json', '--sandbox', '--print-timeout', self.print_timeout]
        if self.lockdown:
            self._agent()
            cmd += ['--agent', 'poker', '--settings', json.dumps(AGY_DENY)]
            prompt = text
        else:
            prompt = text if self.session else self.system + '\n\n=== THE MATCH BEGINS ===\n' + text
        if self.model:
            cmd += ['--model', self.model]
        if self.effort:
            cmd += ['--effort', self.effort]
        if self.session:
            cmd += ['--conversation', self.session]
        cmd += self.extra_args + ['-p', prompt]  # flags must come before -p
        out = await self.run(cmd)
        data = self._response(out)
        if data is None:
            raise BackendError(f'agy returned non-JSON output: {out.strip()[-1500:] or "(empty)"}')
        if data.get('status') != 'SUCCESS':
            raise BackendError(f'agy {data.get("status")}: {json.dumps(data.get("error") or data)[:1500]}')
        self.session = data.get('conversation_id') or self.session
        rows = self._transcript()
        if rows is not None:
            self.last_call['raw']['transcript'] = rows
        thinking, models, tools, audit = self.read_turn(rows)
        meta = {'session': self.session, 'audit': audit,
                **({'audit_reason': self.turn_problem(rows)} if audit == 'unverified' else {}), 'observed_model': models[0] if len(models) == 1 else None,
                **({'observed_models': models} if models else {}), **({'tool_use': tools} if tools else {})}
        return Reply(data.get('response') or '', thinking, data.get('usage') or {}, meta=meta)

    def _session_from(self, stdout):
        try:
            return json.loads(stdout[stdout.find('{'):]).get('conversation_id')
        except (ValueError, AttributeError):
            return None


class GeminiCLI(Backend):
    """Gemini CLI in headless mode; the transcript is re-sent every turn (as an argument, so it is capped
    lower than the other stateless backends; the oldest turns go first and the notes carry what matters)."""
    exe = 'gemini'
    AUDIT_SOURCE = "the tool counts in gemini's JSON stats (missing counts: unverified)"

    def __init__(self, system, **kw):
        kw.setdefault('max_history_chars', 200_000)
        super().__init__(system, **kw)

    async def _call(self, text):
        cmd = ['gemini', '--output-format', 'json']
        if self.model:
            cmd += ['--model', self.model]
        cmd += self.extra_args + ['-p', self.transcript(text)]
        out = await self.run(cmd)
        p = self.parse(out)
        if p is None:
            raise BackendError(f'gemini returned non-JSON output: {out.strip()[-1500:]}')
        if p['error']:
            raise BackendError(f'gemini: {json.dumps(p["error"])[:1500]}')
        models = p['models']
        meta = {'audit': p['audit'], 'observed_model': models[0] if len(models) == 1 else None,
                **({'observed_models': models} if models else {}), **({'tool_use': p['tools']} if p['tools'] else {})}
        return Reply(p['text'], '', p['usage'], meta=meta)

    @staticmethod
    def parse(out: str) -> dict | None:
        try:
            data = json.loads(out[out.find('{'):])
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        stats = data.get('stats') if isinstance(data.get('stats'), dict) else {}
        tool_stats = stats.get('tools')
        audit = 'verified' if isinstance(tool_stats, dict) and isinstance(tool_stats.get('byName'), dict) else 'unverified'
        by_name = tool_stats.get('byName') if audit == 'verified' else {}
        tools = [{'type': 'tool_call', 'name': k} for k, v in by_name.items() if (v or {}).get('count', 1)]
        if audit == 'verified' and not tools and (tool_stats.get('totalCalls') or 0) > 0:
            tools = [{'type': 'tool_call', 'name': 'unnamed tool'}]
        models = sorted((stats.get('models') or {}) if isinstance(stats.get('models'), dict) else {})
        return {'text': data.get('response') or '', 'error': data.get('error'), 'audit': audit, 'tools': tools,
                'models': models, 'usage': data.get('usage') or stats}

    @classmethod
    def evidence(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get('stdout'), str):
            return None
        p = cls.parse(raw['stdout'])
        if p is None:
            return None
        return {'text': None if p['error'] else p['text'], 'tools': p['tools'], 'models': p['models'], 'audit': p['audit']}


class CommandCLI(Backend):
    """Any CLI: `command` is a list of arguments ({model} is filled in); the full transcript goes to stdin
    and stdout is the reply. For example ["pi", "-p", "--model", "{model}"] or ["ollama", "run", "{model}"].
    Its stdout is only the reply, so whether a turn ran a tool can't be checked (audit 'opaque')."""
    AUDIT = 'opaque'
    AUDIT_SOURCE = 'none: the CLI prints only its reply'

    def __init__(self, system, *, command: list[str], **kw):
        super().__init__(system, **kw)
        self.command = command
        self.exe = command[0] if command else None

    async def _call(self, text):
        cmd = [c.replace('{model}', self.model or '') for c in self.command] + self.extra_args
        return Reply((await self.run(cmd, stdin=self.transcript(text))).strip())

    @classmethod
    def evidence(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get('stdout'), str):
            return None
        return {'text': raw['stdout'].strip(), 'tools': [], 'models': [], 'audit': cls.AUDIT}


class OpenAICompat(Backend):
    """An OpenAI-compatible chat completions endpoint (OpenRouter, Ollama, LM Studio, vLLM...)."""

    def __init__(self, system, *, base_url='https://openrouter.ai/api/v1', api_key_env='OPENROUTER_API_KEY', **kw):
        super().__init__(system, **kw)
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.key = os.environ.get(api_key_env, '')

    def _post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method='POST', headers={
            'Content-Type': 'application/json', **({'Authorization': f'Bearer {self.key}'} if self.key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            raise BackendError(f'HTTP {e.code}: {e.read().decode(errors="replace")[:1500]}')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BackendError(f'transport error: {e}')

    async def _call(self, text):
        messages = [{'role': 'system', 'content': self.system}]
        for u, a in self.turns[-200:]:
            messages += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': a}]
        messages.append({'role': 'user', 'content': text})
        raw = await asyncio.to_thread(self._post, {'model': self.model, 'messages': messages})
        self.last_call['raw'] = {'response': raw}
        p = self.parse(raw)
        return Reply(p['text'], p['thinking'], raw.get('usage') or {},
                     meta={'observed_model': raw.get('model'), **({'tool_use': p['tools']} if p['tools'] else {})})

    @staticmethod
    def parse(raw: dict) -> dict:
        msg = ((raw.get('choices') or [{}])[0]).get('message') or {}
        tools = [{'type': 'tool_call', 'name': (c.get('function') or {}).get('name')} for c in msg.get('tool_calls') or []]
        return {'text': msg.get('content') or '', 'thinking': msg.get('reasoning') or '', 'tools': tools,
                'models': [raw['model']] if raw.get('model') else []}

    @classmethod
    def evidence(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get('response'), dict):
            return None
        p = cls.parse(raw['response'])
        return {'text': p['text'], 'tools': p['tools'], 'models': p['models'], 'audit': cls.AUDIT}


BACKENDS = {'claude': ClaudeCLI, 'codex': CodexCLI, 'agy': AgyCLI, 'gemini': GeminiCLI, 'command': CommandCLI,
            'openai_compat': OpenAICompat}


def verify_call(kind: str | None, call: dict, raw) -> dict:
    """A recorded reply checked against its retained raw output, by the backend's own parser: {status, problems}.
    status: the audit the raw output supports ('verified' | 'opaque' | 'unverified') when the record agrees with it,
    'inconsistent' when the record says another text, tool use, identity or audit, 'unverifiable' when there is no
    raw output the backend can read."""
    cls = BACKENDS.get(kind) or Backend
    ev = cls.evidence(raw) if raw is not None else None
    if ev is None:
        return {'status': 'unverifiable', 'problems': ['no raw output this backend can read']}
    problems = []
    if ev.get('text') != call.get('output'):
        problems.append('the reply text is not what the raw output says')
    if bool(ev.get('tools')) != bool(call.get('tool_use')):
        problems.append('tool use differs from the raw output')
    match = identity_fields(call.get('requested_model'), ev.get('models') or [])['identity_match']
    if call.get('identity_match') and call['identity_match'] != match:
        problems.append(f'identity {call["identity_match"]!r} but the raw output says {match!r}')
    if call.get('audit') != ev.get('audit'):
        problems.append(f'audit {call.get("audit")!r} but the raw output supports {ev.get("audit")!r}')
    return {'status': 'inconsistent' if problems else ev.get('audit'), 'problems': problems,
            'identity_match': match}


def make_backend(kind: str, system: str, **kw) -> Backend:
    if kind not in BACKENDS:
        raise KeyError(f'unknown backend {kind!r}; known: {", ".join(BACKENDS)}')
    return BACKENDS[kind](system, **kw)
