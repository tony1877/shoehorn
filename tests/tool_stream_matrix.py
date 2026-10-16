"""Does a streamed tool call's arguments arrive as they are generated (so dsh can show a file growing), for every kind
of file-writing call? Runs a matrix of requests against a live server with dsh's file tools (write, edit,
str_replace_editor) and classifies each run:

  streamed  the arguments arrived spread over the generation (at least half their characters before the last 25%
            of the tool call's time span, in 3+ pieces)
  burst     they arrived in one or two pieces at the end (nothing to preview while the model writes)
  no-call   the model answered without a tool call
  bad-json  the joined arguments do not parse

  python tests/tool_stream_matrix.py --api-key-file KEY [--url http://127.0.0.1:8090] [--only NAME] [--repeat N]
                                     [--out results.jsonl]
"""
import argparse, json, sys, time, urllib.request

WRITE = {'type': 'function', 'function': {
    'name': 'write', 'description': 'Create or fully replace a UTF-8 text file.',
    'parameters': {'type': 'object', 'properties': {
        'file_path': {'type': 'string', 'description': 'Path to write, resolved by the filesystem backend.'},
        'content': {'type': 'string', 'description': 'Full UTF-8 text content to write.'}},
        'required': ['file_path', 'content']}}}
EDIT = {'type': 'function', 'function': {
    'name': 'edit', 'description': 'Edit an existing UTF-8 text file by replacing literal text.',
    'parameters': {'type': 'object', 'properties': {
        'file_path': {'type': 'string', 'description': 'Path to edit, resolved by the filesystem backend.'},
        'old_string': {'type': 'string', 'description': 'Literal text to replace. Must match exactly.'},
        'new_string': {'type': 'string', 'description': 'Literal replacement text. Use an empty string to delete the match.'},
        'replace_all': {'type': 'boolean', 'description': 'Replace all matches. Defaults to false.'}},
        'required': ['file_path', 'old_string', 'new_string']}}}
SRE = {'type': 'function', 'function': {
    'name': 'str_replace_editor', 'description': 'Custom editing tool for viewing, creating and editing files.',
    'parameters': {'type': 'object', 'properties': {
        'command': {'type': 'string', 'enum': ['view', 'create', 'str_replace', 'insert', 'undo_edit']},
        'path': {'type': 'string'}, 'file_text': {'type': 'string'}, 'old_str': {'type': 'string'},
        'new_str': {'type': 'string'}, 'insert_line': {'type': 'integer'}},
        'required': ['command', 'path']}}}
TOOLS = [WRITE, EDIT, SRE]

EXISTING = '''def load(path):
    with open(path) as f:
        return f.read()


def main():
    data = load("input.txt")
    print(len(data))
'''

# name -> (user message, thinking, extra messages before it)
CASES = {
    'py-small': ('Create util/hello.py with a hello(name) function and a docstring (about 8 lines). Use the write tool now.', False),
    'py-large': ('Create lib/shapes.py: classes Circle, Rectangle, Triangle with area, perimeter, __repr__, docstrings, '
                 'type hints, and a __main__ demo (about 150 lines). Use the write tool now.', False),
    'py-think': ('Create lib/stack.py: a Stack class with push, pop, peek, is_empty, __len__, docstrings and a demo '
                 '(about 50 lines). Use the write tool.', True),
    'json': ('Create config/settings.json: a realistic app config with nested objects, arrays, strings containing '
             'double quotes and backslashes (Windows paths like C:\\\\data\\\\in), about 60 lines. Use the write tool now.', False),
    'markdown-fences': ('Create docs/USAGE.md: a usage guide with 4 sections, each with a ```bash and a ```python code '
                        'block, about 80 lines. Use the write tool now.', False),
    'js-regex': ('Create src/validate.js: functions validating emails, URLs and phone numbers with regular '
                 'expressions full of backslashes, plus JSDoc, about 70 lines. Use the write tool now.', False),
    'shell': ('Create scripts/backup.sh: a bash backup script with variables, "$@" handling, quotes inside quotes, '
              'a here-doc and a case statement, about 60 lines. Use the write tool now.', False),
    'unicode': ('Create i18n/greetings.txt: greetings in 20 languages with their native scripts and an emoji each, '
                'one per line. Use the write tool now.', False),
    'markup-lookalike': ('Create notes/qwen-format.md: explain the tool-call format with literal examples that '
                         'contain the exact text <tool_call>, <function=write>, <parameter=content>, </parameter>, '
                         '</function> and </tool_call> inside code blocks, about 40 lines. Use the write tool now.', False),
    'text-first': ('First explain in two sentences what you will write, then create lib/queue.py with a Queue class '
                   '(about 40 lines) using the write tool.', False),
    'two-files': ('Create two files with two write calls in this one reply: lib/a.py (a function add, 15 lines) and '
                  'lib/b.py (a function mul, 15 lines).', False),
    'edit': ('In app.py, change load() so it opens the file with encoding="utf-8" and add a docstring to both '
             'functions. Use the edit tool (one call that replaces the whole current text).', False),
    'edit-think': ('In app.py, add error handling to load() for a missing file (print a message and return "") and '
                   'docstrings. Use the edit tool.', True),
    'sre-create': ('Create web/index.html (about 60 lines: a page with a form, inline CSS and a small script) using '
                   'str_replace_editor with command create.', False),
    # text that looks like markup the parser or the chat template gives meaning to
    'think-tags': ('Create notes/reasoning.md: explain how a model marks its reasoning, with literal examples containing '
                   '<think> and </think> tags in code blocks, about 30 lines. Use the write tool now.', False),
    'template-tokens': ('Create notes/chatml.md: describe the ChatML format with literal examples containing '
                        '<|im_start|>user, <|im_start|>assistant and <|im_end|>, about 30 lines. Use the write tool now.', False),
    'xml-file': ('Create config/pom.xml: a Maven pom with 6 dependencies, properties and a build section, about 80 lines. '
                 'Use the write tool now.', False),
    'parameter-only': ('Create notes/template.txt containing exactly these 3 lines and nothing else:\n'
                       '<parameter=name>\nvalue\n</parameter>\nUse the write tool now.', False),
    'empty-file': ('Create an empty file pkg/__init__.py (content is the empty string). Use the write tool now.', False),
    'crlf-tabs': ('Create build/Makefile with 5 targets; recipe lines must start with a real tab character, about 30 '
                  'lines. Use the write tool now.', False),
    'odd-path': ('Create "docs/résumé notes/Ünïcödé file (draft).md" with 10 lines of notes. Use the write tool now.', False),
    'very-long': ('Create data/countries.py: a list of dicts for 120 countries, each with name, capital, ISO code, '
                  'continent and population. Use the write tool now, all in one file.', False),
    'three-files': ('Create three files with three write calls in this one reply: x/one.txt, x/two.txt, x/three.txt, '
                    'each with 5 lines about the number.', False),
}
# text that must survive in the written content (the markup cases: the parser must not take it for structure)
EXPECT = {
    'markup-lookalike': ['</parameter>', '</function>', '</tool_call>'],
    'think-tags': ['<think>', '</think>'],
    'template-tokens': ['<|im_end|>'],
    'parameter-only': ['<parameter=name>', '</parameter>'],
    'crlf-tabs': ['\t'],
    'xml-file': ['</project>'],
}
TRUNCATE = ('Create lib/big.py with 40 small functions f1..f40, each with a docstring. Use the write tool now.', False)
EDIT_CONTEXT = [  # app.py was read in an earlier turn
    {'role': 'user', 'content': 'Show me app.py.'},
    {'role': 'assistant', 'content': 'Here is app.py:\n\n```python\n' + EXISTING + '```'},
]


def run(url, key, name, user, think, max_tokens=8000):
    msgs = (EDIT_CONTEXT if name.startswith('edit') else []) + [{'role': 'user', 'content': user}]
    body = {'stream': True, 'max_tokens': max_tokens, 'temperature': 0.6, 'tools': TOOLS, 'messages': msgs,
            'chat_template_kwargs': {'enable_thinking': think}}
    req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    t0 = time.time()
    calls = {}  # index -> {'name', 'pieces': [(t, chars)], 'args'}
    finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            ch = json.loads(line[6:])['choices'][0]
            finish = ch.get('finish_reason') or finish
            d = ch.get('delta', {})
            for tc in d.get('tool_calls') or []:
                c = calls.setdefault(tc.get('index', 0), {'name': '', 'pieces': [], 'args': ''})
                f = tc.get('function') or {}
                if f.get('name'):
                    c['name'] = f['name']
                a = f.get('arguments') or ''
                if a:
                    c['pieces'].append((time.time() - t0, len(a)))
                    c['args'] += a
    total = time.time() - t0
    out = []
    for i, c in sorted(calls.items()):
        p = c['pieces']
        n = sum(x[1] for x in p)
        verdict = 'streamed'
        if len(p) < 3:
            verdict = 'burst'
        else:
            t_first, t_last = p[0][0], p[-1][0]
            cut = t_first + 0.75 * (t_last - t_first)
            early = sum(x[1] for x in p if x[0] <= cut)
            if n and early / n < 0.5:
                verdict = 'burst'
        try:
            a = json.loads(c['args'])
            text = a.get('content', a.get('file_text', a.get('new_string', ''))) if isinstance(a, dict) else ''
            missing = [x for x in EXPECT.get(name, []) if x not in (text or '')]
            if missing:
                verdict = 'lost-text'
        except Exception:
            verdict = 'bad-json'
        out.append({'call': i, 'tool': c['name'], 'pieces': len(p), 'chars': n,
                    'first_s': round(p[0][0], 1) if p else None, 'last_s': round(p[-1][0], 1) if p else None,
                    'finish': finish, 'verdict': verdict})
    if not out:
        out.append({'call': None, 'finish': finish, 'verdict': 'no-call'})
    return round(total, 1), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8090')
    ap.add_argument('--api-key-file', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    key = [l for l in open(a.api_key_file).read().splitlines() if l.strip() and not l.startswith('#')][-1].strip()
    bad = 0
    for rep in range(a.repeat):
        for name, (user, think) in CASES.items():
            if a.only and a.only not in name:
                continue
            total, calls = run(a.url, key, name, user, think)
            if name == 'two-files' and len(calls) != 2 or name == 'three-files' and len(calls) != 3:
                for c in calls:
                    c['verdict'] = c['verdict'] if c['verdict'] != 'streamed' else 'call-count'
            for c in calls:
                rec = dict(case=name, rep=rep, think=think, total_s=total, **c)
                print(json.dumps(rec), flush=True)
                if a.out:
                    with open(a.out, 'a') as f:
                        f.write(json.dumps(rec) + '\n')
                bad += c['verdict'] != 'streamed'
    if not a.only or 'truncate' in a.only:  # cut off by max_tokens inside the call: never a finished call
        total, calls = run(a.url, key, 'truncate', TRUNCATE[0], False, max_tokens=300)
        for c in calls:
            ok = c.get('finish') == 'length'
            rec = dict(case='truncate', rep=0, think=False, total_s=total, **c)
            rec['verdict'] = 'ok (length)' if ok else 'BAD: finish ' + str(c.get('finish'))
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
            bad += not ok
    print(f'{bad} runs not streamed', file=sys.stderr)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
