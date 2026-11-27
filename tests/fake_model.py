"""A stand-in 'model' CLI for tests: reads the prompt on stdin, prints a reply.

It answers in the game's JSON from the last 'Your options:' line: raises with a strong-looking
hand, otherwise checks or calls. Every 7th reply is malformed, to exercise the retry path.
"""
import json, random, re, sys

prompt = sys.stdin.read()
rng = random.Random(len(prompt))
now = prompt.rsplit('=== NOW ===', 1)[-1]
if rng.random() < 1 / 7:
    print('I think I will call here.')  # not JSON: the agent must ask again
    sys.exit(0)
opts = re.findall(r'Your options: (.*)', now)
opts = opts[-1] if opts else 'check'
cards = re.findall(r'Your cards: (\S\S) (\S\S)', now)
strong = bool(cards) and cards[-1][0][0] in 'AKQ'
m = re.search(r'to between (\d+) and (\d+)', opts)
if strong and m:
    act = {'action': 'raise', 'amount': int(m.group(1)) * 2}
elif 'check' in opts:
    act = {'action': 'check', 'amount': None}
else:
    act = {'action': 'call', 'amount': None}
reply = {'reasoning': f'fake model: {"strong" if strong else "meh"} hand', **act,
         'say': rng.choice(['', 'nice hand', 'I have it', '']), 'notes': f'seen {prompt.count(">>> GAME:")} turns'}
text = json.dumps(reply)
print(f'```json\n{text}\n```' if rng.random() < 0.3 else text)
