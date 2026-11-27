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
reasoning the tool exposes, usage and timing; `state()` / `restore()` let a stopped match resume.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field

RATE_LIMIT = re.compile(r'usage limit|rate.?limit|quota|too many requests|429|resource.?exhausted|limit reached'
                        r'|try again (?:at|in|later)', re.I)


class BackendError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    thinking: str = ''
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    meta: dict = field(default_factory=dict)


class Backend:
    stateful = False

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

    # ---- state for resuming a stopped match -------------------------------------------------
    def state(self):
        return {'turns': self.turns, 'session': self.session}

    def restore(self, st):
        self.turns, self.session = st.get('turns', []), st.get('session')

    # ---- the transcript a stateless backend re-sends ----------------------------------------
    def transcript(self, text: str) -> str:
        """System prompt, earlier turns (oldest dropped first when too long) and the new turn, as one prompt."""
        turns = list(self.turns)
        def build(ts):
            parts = [self.system, '', '=== THE MATCH SO FAR (your earlier turns and replies) ===']
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
        waited, attempt = 0.0, 0
        while True:
            start = time.monotonic()
            try:
                reply = await self._call(text)
                reply.latency = round(time.monotonic() - start, 2)
                self.turns.append([text, reply.text])
                return reply
            except BackendError as e:
                msg = str(e)
                if RATE_LIMIT.search(msg) and waited < self.max_wait:
                    print(f'[{self.label}] rate limited; waiting {self.rate_limit_wait:.0f}s: {msg[:200]}', flush=True)
                    await asyncio.sleep(self.rate_limit_wait)
                    waited += self.rate_limit_wait
                    continue
                attempt += 1
                if attempt > self.retries:
                    raise
                await asyncio.sleep(min(5 * 2 ** attempt, 120))

    async def _call(self, text: str) -> Reply:
        raise NotImplementedError

    # ---- running a CLI -----------------------------------------------------------------------
    def folder(self) -> str:
        """This seat's work folder: empty apart from files the backend puts there (no project files or
        instructions), and kept for the match, since CLIs key their sessions by folder."""
        if not self.workdir:
            self.workdir = tempfile.mkdtemp(prefix='poker-arena-')
        return self.workdir

    async def run(self, cmd: list[str], stdin: str | None = None, cwd: str | None = None) -> str:
        exe = shutil.which(cmd[0])
        if not exe:
            raise BackendError(f'{cmd[0]!r} is not installed or not on PATH')
        workdir = cwd or self.folder()
        proc = await asyncio.create_subprocess_exec(
            exe, *cmd[1:], cwd=workdir, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin.encode() if stdin is not None else None),
                                              self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise BackendError(f'{cmd[0]} timed out after {self.timeout:.0f}s')
        out_s, err_s = out.decode(errors='replace'), err.decode(errors='replace')
        if proc.returncode != 0:
            raise BackendError(f'{cmd[0]} exited {proc.returncode}: {(err_s or out_s).strip()[-1500:]}')
        return out_s


class ClaudeCLI(Backend):
    """Claude Code in print mode. The first turn sets the system prompt and opens a session; later turns
    resume it, so the model keeps its own earlier replies (and the CLI handles caching and compaction).
    Tools and MCP servers are switched off, and only project settings load (the folder is empty), so your
    own hooks and settings don't run inside the eval. Set `isolate = false` on the player to keep them.
    Thinking is shown as summaries (`--thinking-display summarized`) so the viewer can show it."""
    stateful = True

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
        thinking, result, usage, session, is_error = [], None, {}, None, False
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get('type') == 'assistant':
                for block in (ev.get('message') or {}).get('content') or []:
                    if block.get('type') == 'thinking' and block.get('thinking'):
                        thinking.append(block['thinking'])
            elif ev.get('type') == 'result':
                result, usage = ev.get('result'), ev.get('usage') or {}
                session, is_error = ev.get('session_id'), ev.get('is_error', False)
                usage = {**usage, 'cost_usd_equivalent': ev.get('total_cost_usd')}
            if ev.get('session_id'):
                session = session or ev['session_id']
        if result is None or is_error:
            raise BackendError(f'claude returned no result: {out.strip()[-1500:]}')
        self.session = session or self.session
        return Reply(result, '\n'.join(thinking), usage, meta={'session': self.session})


class CodexCLI(Backend):
    """Codex CLI (`codex exec`), read-only sandbox in an empty folder. The system prompt goes in the
    The game's system prompt replaces Codex's built-in coding instructions (`model_instructions_file`); later
    turns resume the same thread. Codex's tools can't be switched off, so the read-only sandbox is the guard.
    Reasoning comes back as short summary titles."""
    stateful = True

    def __init__(self, system, *, effort: str | None = 'high', summary: str | None = 'auto',
                 replace_instructions: bool = True, **kw):
        super().__init__(system, **kw)
        self.effort, self.summary, self.replace_instructions = effort, summary, replace_instructions

    async def _call(self, text):
        flags = ['--json', '--skip-git-repo-check']
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
            cmd = ['codex', 'exec', 'resume', self.session, *flags, '-']
            prompt = text
        else:
            cmd = ['codex', 'exec', '--sandbox', 'read-only', *flags, '-']
            prompt = text if self.replace_instructions else self.system + '\n\n=== THE MATCH BEGINS ===\n' + text
        out = await self.run(cmd, stdin=prompt)
        messages, thinking, usage = [], [], {}
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            t = ev.get('type', '')
            if t == 'thread.started':
                self.session = ev.get('thread_id') or self.session
            elif t == 'item.completed':
                item = ev.get('item') or {}
                kind = item.get('type') or item.get('item_type')
                if kind == 'error':  # config warnings arrive as items; they don't fail the turn
                    continue
                if kind in ('agent_message', 'assistant_message'):
                    messages.append(item.get('text', ''))
                elif kind == 'reasoning':
                    thinking.append(item.get('text', ''))
            elif t == 'turn.completed':
                usage = ev.get('usage') or {}
            elif t in ('turn.failed', 'error'):
                raise BackendError(f'codex: {json.dumps(ev)[:1500]}')
        if not messages:
            raise BackendError(f'codex returned no message: {out.strip()[-1500:]}')
        return Reply(messages[-1], '\n'.join(thinking), usage, meta={'session': self.session})


AGY_AGENT = """---
name: poker
description: A player in a poker match. Answers with one JSON object.
tools: []
subagent: false
---

"""
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

    def _transcript(self):
        """(thinking, model) of the latest model step in this conversation's transcript, if agy wrote one."""
        path = os.path.join(self.BRAIN, self.session or '', '.system_generated', 'logs', 'transcript.jsonl')
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            return '', None
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get('type') == 'PLANNER_RESPONSE':
                return rec.get('thinking') or '', rec.get('model')
        return '', None

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
        try:
            data = json.loads(out[out.find('{'):])
        except ValueError:
            raise BackendError(f'agy returned non-JSON output: {out.strip()[-1500:] or "(empty)"}')
        if data.get('status') != 'SUCCESS':
            raise BackendError(f'agy {data.get("status")}: {json.dumps(data.get("error") or data)[:1500]}')
        self.session = data.get('conversation_id') or self.session
        thinking, served = self._transcript()
        meta = {'session': self.session, **({'served_model': served} if served else {})}
        return Reply(data.get('response') or '', thinking, data.get('usage') or {}, meta=meta)


class GeminiCLI(Backend):
    """Gemini CLI in headless mode; the transcript is re-sent every turn (as an argument, so it is capped
    lower than the other stateless backends; the oldest turns go first and the notes carry what matters)."""

    def __init__(self, system, **kw):
        kw.setdefault('max_history_chars', 200_000)
        super().__init__(system, **kw)

    async def _call(self, text):
        cmd = ['gemini', '--output-format', 'json']
        if self.model:
            cmd += ['--model', self.model]
        cmd += self.extra_args + ['-p', self.transcript(text)]
        out = await self.run(cmd)
        try:
            data = json.loads(out[out.find('{'):])
        except ValueError:
            raise BackendError(f'gemini returned non-JSON output: {out.strip()[-1500:]}')
        if data.get('error'):
            raise BackendError(f'gemini: {json.dumps(data["error"])[:1500]}')
        return Reply(data.get('response') or '', '', data.get('usage') or data.get('stats') or {})


class CommandCLI(Backend):
    """Any CLI: `command` is a list of arguments ({model} is filled in); the full transcript goes to stdin
    and stdout is the reply. For example ["pi", "-p", "--model", "{model}"] or ["ollama", "run", "{model}"]."""

    def __init__(self, system, *, command: list[str], **kw):
        super().__init__(system, **kw)
        self.command = command

    async def _call(self, text):
        cmd = [c.replace('{model}', self.model or '') for c in self.command] + self.extra_args
        return Reply((await self.run(cmd, stdin=self.transcript(text))).strip())


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
        msg = ((raw.get('choices') or [{}])[0]).get('message') or {}
        return Reply(msg.get('content') or '', msg.get('reasoning') or '', raw.get('usage') or {})


BACKENDS = {'claude': ClaudeCLI, 'codex': CodexCLI, 'agy': AgyCLI, 'gemini': GeminiCLI, 'command': CommandCLI,
            'openai_compat': OpenAICompat}


def make_backend(kind: str, system: str, **kw) -> Backend:
    if kind not in BACKENDS:
        raise KeyError(f'unknown backend {kind!r}; known: {", ".join(BACKENDS)}')
    return BACKENDS[kind](system, **kw)
