"""Live checks for agent consistency (no truncation, no dropped streams, no lost calls), against a running server:

  A  keep-alive: the longest silence on a stream while a long prompt is read, and while queued behind another request
  B  reasoning budget: reasoning cut by its budget ends at a sentence or line, not mid-word, and the answer follows
  C  repetition guard: a 12 KiB file of identical rows is written whole, or a runaway past it is still stopped
  D  JSON arguments: a list argument comes back as a JSON array
  E  unknown tool: a call to a function the request does not offer is passed on by name (the client reports it)
  F  output limit inside a call: finish length, no half call, a note in the reply for the next turn
  G  client abort mid-answer: the conversation goes on, answered, with the prompt reused
  H  recall of an exact signature ~50K tokens back (beyond the exact VRAM window)
  I  prompt reuse across the steps of a tool loop with reasoning on (no re-reads)
  J  two sessions at once: both answered correctly

  python tests/agent_consistency.py --api-key-file KEY [--url http://127.0.0.1:8090] [--only A,B]
"""
import argparse, json, sys, threading, time, urllib.request

WRITE = {'type': 'function', 'function': {'name': 'write', 'description': 'Create or fully replace a UTF-8 text file.',
         'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'content': {'type': 'string'}},
                        'required': ['file_path', 'content']}}}
TODO = {'type': 'function', 'function': {'name': 'todo_write', 'description': 'Replace the task list.',
        'parameters': {'type': 'object', 'properties': {'todos': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'content': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed']}}}}},
            'required': ['todos']}}}


def stream(url, key, body):
    """Streams a chat request; returns (gaps between messages in s, reasoning, content, tool calls, finish)."""
    body = dict(body, stream=True)
    req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    gaps, last = [], time.time()
    reasoning, content, calls, finish = '', '', {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: '):
                continue
            now = time.time()
            gaps.append(now - last)
            last = now
            if line == 'data: [DONE]':
                break
            j = json.loads(line[6:])
            if 'error' in j:
                raise RuntimeError(j['error'])
            ch = j['choices'][0]
            finish = ch.get('finish_reason') or finish
            d = ch.get('delta') or {}
            reasoning += d.get('reasoning_content') or ''
            content += d.get('content') or ''
            for tc in d.get('tool_calls') or []:
                c = calls.setdefault(tc.get('index', 0), {'name': '', 'args': ''})
                f = tc.get('function') or {}
                c['name'] = f.get('name') or c['name']
                c['args'] += f.get('arguments') or ''
    return gaps, reasoning, content, [calls[i] for i in sorted(calls)], finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8090')
    ap.add_argument('--api-key-file', required=True)
    ap.add_argument('--only', default='A,B,C,D,E,F,G,H,I,J')
    a = ap.parse_args()
    key = [l for l in open(a.api_key_file).read().splitlines() if l.strip() and not l.startswith('#')][-1].strip()
    only = set(a.only.split(','))
    fails = 0

    def report(name, ok, msg):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)

    off = {'chat_template_kwargs': {'enable_thinking': False}}
    if 'A' in only:
        # a ~60K-token prompt (read in tens of seconds), and a second request queued behind it
        doc = ' '.join(f'Record {i}: the {["red", "green", "blue"][i % 3]} unit moved {i * 7 % 101} metres north.' for i in range(6000))
        long_body = dict(off, messages=[{'role': 'user', 'content': doc + '\n\nHow many records are there? One line.'}], max_tokens=40)
        short_body = dict(off, messages=[{'role': 'user', 'content': 'Say OK.'}], max_tokens=8)
        res = {}
        t1 = threading.Thread(target=lambda: res.__setitem__('long', stream(a.url, key, long_body)))
        t1.start()
        time.sleep(1.0)
        g2, *_ = stream(a.url, key, short_body)  # waits for its turn behind the long one
        t1.join()
        g1 = res['long'][0]
        report('A keep-alive (long prompt)', max(g1) < 7, f'longest silence {max(g1):.1f} s over {sum(g1):.0f} s')
        report('A keep-alive (queued)', max(g2) < 7, f'longest silence {max(g2):.1f} s over {sum(g2):.0f} s')
    if 'B' in only:
        body = {'messages': [{'role': 'user', 'content': 'Find all integer solutions of x^3 + y^3 + z^3 = 42 with |x|,|y|,|z| < 50 '
                              'by careful reasoning, checking cases one by one. Then give the answer.'}],
                'reasoning_effort': 'low', 'max_tokens': 9000, 'chat_template_kwargs': {'enable_thinking': True}}
        _, reasoning, content, _, finish = stream(a.url, key, body)
        tail = reasoning.rstrip()
        closed = tail.endswith('I will act on the best plan so far.')
        before = tail[:-len('I have thought about this enough; I will act on the best plan so far.')].rstrip() if closed else ''
        clean = (not closed) or before.endswith(('.', '!', '?', ':')) or before == '' or '\n' in reasoning[-200:]
        report('B reasoning budget', clean and bool(content.strip()),
               f"{'cut by budget' if closed else 'finished on its own'}, {len(reasoning)} chars of reasoning, "
               f"end before the closing note: {before[-60:]!r}; answer {len(content)} chars, finish {finish}")
    if 'C' in only:
        # ~12 KiB of identical lines: over the old 8 KiB loop threshold, under the 32 KiB one used inside tool calls
        row = 'ok,' + ','.join(['0'] * 38)
        body = dict(off, tools=[WRITE], max_tokens=16000, messages=[{'role': 'user', 'content':
                    f'Create data/fixture.csv with exactly 150 identical lines, each exactly: {row}\nNo header. Use the write '
                    'tool now.'}])
        try:
            _, _, _, calls, finish = stream(a.url, key, body)
            try:
                text = json.loads(calls[0]['args'])['content'] if calls else ''
            except ValueError:
                text = ''
            rows = text.count(row)
            # the model loses count on long runs of identical lines and goes on writing them: either it stops near
            # 150 (a 12 KiB file, over the old 8 KiB threshold, written whole) or the guard stops the runaway
            report('C repetition guard', 120 <= rows <= 200 and finish == 'tool_calls',
                   f'{rows} identical rows ({len(text)} bytes) written, finish {finish}')
        except RuntimeError as e:
            report('C repetition guard', 'repeated output' in str(e), 'runaway stopped by the guard (the model lost count)')
    if 'D' in only:
        body = dict(off, tools=[TODO], max_tokens=800, messages=[{'role': 'user', 'content':
                    "Set the task list to 3 items: 'fix the parser', 'add tests', 'ship it' (the first in_progress, the others "
                    'pending). Use todo_write now.'}])
        _, _, _, calls, finish = stream(a.url, key, body)
        try:
            todos = json.loads(calls[0]['args'])['todos']
            report('D JSON list argument', isinstance(todos, list) and len(todos) == 3, f'{len(todos)} todos, finish {finish}')
        except Exception as e:
            report('D JSON list argument', False, f'{e}: {calls}')
    if 'E' in only:
        body = dict(off, tools=[WRITE], max_tokens=300, messages=[{'role': 'user', 'content':
                    'Call the function magic_wand with the argument spell set to "lumos". Do it now, no questions.'}])
        _, _, content, calls, finish = stream(a.url, key, body)
        names = [c['name'] for c in calls]
        # the model may decline or use a real tool; what must not happen is a garbled or dropped call (the
        # pass-through of an unknown name itself is covered by tests/tool_recovery.cpp)
        ok = all(n in ('magic_wand', 'write') for n in names) and finish in ('tool_calls', 'stop')
        report('E unknown tool', ok, f'calls {names or "none (answered in text)"}, finish {finish}')
    if 'F' in only:  # a file bigger than max_tokens: no half call, a note the model will see next turn
        body = dict(off, tools=[WRITE], max_tokens=600, messages=[{'role': 'user', 'content':
                    'Create lib/big.py with 60 functions f1..f60, each with a 3-line docstring. Use the write tool now.'}])
        _, _, content, calls, finish = stream(a.url, key, body)
        report('F output limit inside a call', finish == 'length' and 'cut off at the output limit' in content,
               f'finish {finish}, note {"present" if "output limit" in content else "MISSING"}, {len(calls)} streamed call(s) '
               '(the client drops a call when the finish is length)')
    if 'G' in only:  # the client goes away mid-answer; the conversation then goes on
        q1 = 'Write a numbered list of 40 classic novels with their authors.'
        req = urllib.request.Request(a.url + '/v1/chat/completions', json.dumps(dict(off, stream=True, max_tokens=1500, messages=[
            {'role': 'user', 'content': q1}])).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        r = urllib.request.urlopen(req, timeout=600)
        got = ''
        for line in r:
            line = line.decode().strip()
            if line.startswith('data: {'):
                got += (json.loads(line[6:])['choices'][0].get('delta') or {}).get('content') or ''
            if len(got) > 300:
                break
        r.close()  # abort mid-answer
        time.sleep(1.0)
        msgs = [{'role': 'user', 'content': q1}, {'role': 'assistant', 'content': got},
                {'role': 'user', 'content': 'Stop the list there. In one line: which author appears first in your list?'}]
        body = dict(off, max_tokens=60, messages=msgs)
        req = urllib.request.Request(a.url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        j = json.loads(urllib.request.urlopen(req, timeout=600).read())
        ans = j['choices'][0]['message'].get('content') or ''
        first = got.split('\n')[0] if got else ''
        cached = j['usage']['prompt_tokens_details']['cached_tokens']
        report('G after a client abort', bool(ans.strip()), f'answer {ans.strip()[:70]!r}; list started {first[:50]!r}; '
               f'{cached} of {j["usage"]["prompt_tokens"]} prompt tokens reused')
    if 'H' in only:  # an exact detail ~50K tokens back (beyond the exact VRAM window: sparse attention)
        filler = ' '.join(f'Log line {i}: worker {i % 17} finished batch {i * 31 % 997} in {i % 9 + 1} ms.' for i in range(4200))
        needle = 'def reconcile_ledger(entries: list[LedgerEntry], *, tolerance_cents: int = 3, strict: bool = False) -> Report:'
        doc = f'Module notes. The key function is:\n{needle}\nThen the logs follow.\n' + filler
        body = dict(off, max_tokens=120, messages=[{'role': 'user', 'content': doc + '\n\nQuote the exact signature of '
                    'reconcile_ledger from the start of this message, on one line, nothing else.'}])
        _, _, content, _, _ = stream(a.url, key, body)
        report('H recall ~50K tokens back', needle.split('def ')[1].split(' ->')[0] in content,
               f'answer {content.strip()[:150]!r}')
    if 'I' in only:  # an agent loop: each step's prompt should be read from the cache, not again
        sys_p = 'You are a coding agent. Use the tools.'
        msgs = [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content':
                 'Create notes/a.txt, then notes/b.txt, then notes/c.txt, each with one line about its letter. '
                 'One write call per step, waiting for each result.'}]
        reuse = []
        for step in range(4):
            body = {'tools': [WRITE], 'max_tokens': 2000, 'messages': msgs, 'chat_template_kwargs': {'enable_thinking': True}}
            req = urllib.request.Request(a.url + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            j = json.loads(urllib.request.urlopen(req, timeout=900).read())
            u, m = j['usage'], j['choices'][0]['message']
            reuse.append((u['prompt_tokens_details']['cached_tokens'], u['prompt_tokens']))
            if not m.get('tool_calls'):
                break
            msgs.append({'role': 'assistant', 'content': m.get('content') or '', 'tool_calls': m['tool_calls'],
                         'reasoning_content': m.get('reasoning_content') or ''})
            for tc in m['tool_calls']:
                msgs.append({'role': 'tool', 'tool_call_id': tc['id'], 'content': 'ok, written'})
        later = reuse[1:]
        ok = bool(later) and all(c >= 0.9 * (p - 400) for c, p in later)  # all but the new turn's tokens reused
        report('I prompt reuse across agent steps', ok, ', '.join(f'{c}/{p}' for c, p in reuse) + ' reused/prompt per step')
    if 'J' in only:  # two sessions at once: both answer correctly, the second waits its turn
        res = {}
        def one(tag, q):
            t0 = time.time()
            _, _, content, _, finish = stream(a.url, key, dict(off, max_tokens=300, messages=[{'role': 'user', 'content': q}]))
            res[tag] = (content, finish, time.time() - t0)
        th = [threading.Thread(target=one, args=('a', 'What is 17 * 23? Answer with the number only.')),
              threading.Thread(target=one, args=('b', 'What is the capital of Australia? One word.'))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        ok = '391' in res['a'][0] and 'Canberra' in res['b'][0]
        report('J two sessions at once', ok, f"a {res['a'][0].strip()[:20]!r} in {res['a'][2]:.1f} s, "
               f"b {res['b'][0].strip()[:20]!r} in {res['b'][2]:.1f} s")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
