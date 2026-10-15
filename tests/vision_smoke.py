"""Image input through `shoehorn serve` (docs/VISION.md): one image, a follow-up turn that reuses the prompt, a
same-size image with the same question and a different image in the same place (neither may reuse the first one's
state), two images in one message, and streaming.

  python tests/vision_smoke.py --url http://127.0.0.1:8099 --image a.png --image2 b.png [--image3 c.png] [--api-key-file F]
Each check prints the answer; the expected facts are asserted case-insensitively when --expect / --expect2 are given.
"""
import argparse, base64, json, sys, time, urllib.request


def data_url(path):
    mime = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(open(path, 'rb').read()).decode()


def post(url, body, key, stream=False):
    req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + key} if key else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream:
            j = json.loads(r.read())
            return j['choices'][0]['message'].get('content') or '', j.get('usage', {}), time.time() - t0
        text = ''
        for line in r:
            line = line.decode().strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                d = json.loads(line[6:])['choices'][0].get('delta', {})
                text += d.get('content') or ''
        return text, {}, time.time() - t0


def user(text, *images):
    return {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_url(p)}} for p in images] +
            [{'type': 'text', 'text': text}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8099')
    ap.add_argument('--image', required=True)
    ap.add_argument('--image2', required=True)
    ap.add_argument('--expect', default='', help='comma-separated facts the first image answer must contain')
    ap.add_argument('--expect2', default='', help='... and the second image answer')
    ap.add_argument('--image3', default='', help='same size as --image, different content: the same question must not '
                                                 'reuse the first image (its placeholders must differ)')
    ap.add_argument('--expect3', default='')
    ap.add_argument('--api-key-file', default='')
    a = ap.parse_args()
    key = open(a.api_key_file).read().split()[-1] if a.api_key_file else ''
    base = {'model': 'x', 'temperature': 0, 'max_tokens': 300, 'chat_template_kwargs': {'enable_thinking': False}}
    fails = 0

    def check(name, text, expect):
        nonlocal fails
        miss = [e for e in expect.split(',') if e and e.lower() not in text.lower()]
        print(f'--- {name}: {"OK" if not miss else "MISSING " + str(miss)}\n{text.strip()[:600]}\n')
        fails += bool(miss)

    q = 'What text is written in this image? Quote it exactly, then name the shapes and colors.'
    msgs = [user(q, a.image)]
    t, u, s = post(a.url, {**base, 'messages': msgs}, key)
    print(f'(prompt {u.get("prompt_tokens")} tokens, {s:.1f} s)')
    check('image 1', t, a.expect)
    msgs += [{'role': 'assistant', 'content': t}, {'role': 'user', 'content': 'Which shape is on the far left? One word.'}]
    t2, u, s = post(a.url, {**base, 'messages': msgs}, key)
    print(f'(follow-up {s:.1f} s: should reuse the image, no re-encoding)')
    check('follow-up', t2, '')
    if a.image3:
        post(a.url, {**base, 'messages': [user(q, a.image)]}, key)
        t6, u, s = post(a.url, {**base, 'messages': [user(q, a.image3)]}, key)
        check('same-size image, same question (no stale reuse)', t6, a.expect3)
    t3, u, s = post(a.url, {**base, 'messages': [user(q, a.image2)]}, key)
    check('image 2 in the same place (no stale reuse)', t3, a.expect2)
    t4, u, s = post(a.url, {**base, 'messages': [user('Two images. Describe each in one sentence, first then second.', a.image, a.image2)]}, key)
    check('two images', t4, '')
    t5, _, s = post(a.url, {**base, 'messages': [user(q, a.image)], 'stream': True}, key, stream=True)
    check('streaming', t5, a.expect)
    print('PASS' if not fails else f'FAIL ({fails})')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
