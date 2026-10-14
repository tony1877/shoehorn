"""Optional live-model regression for final answer bounds and reasoning policy.

Run against an idle server: python tests/server_policy.py --api-key-file PATH
No prompts, generated text, or credentials are written to the report.
"""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8090')
    parser.add_argument('--api-key-file', required=True)
    parser.add_argument('--model', default='qwen3.8-27b-uncensored-shoehorn')
    args = parser.parse_args()
    key = next(line.strip() for line in Path(args.api_key_file).read_text().splitlines()
               if line.strip() and not line.startswith('#'))

    def request(body):
        body = {'model': args.model, 'temperature': 0, 'max_tokens': 128, **body}
        req = Request(args.url.rstrip('/') + '/v1/chat/completions',
                      data=json.dumps(body).encode(), headers={
                          'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        return urlopen(req, timeout=90)

    bounded = {'messages': [{'role': 'user', 'content':
                'Reply with exactly ABCDEFGHIJKLMNOPQRSTUVWXYZ and nothing else.'}],
               'chat_template_kwargs': {'enable_thinking': False}, 'max_content_chars': 16}
    with request(bounded) as response:
        choice = json.load(response)['choices'][0]
    assert len(choice['message']['content'].encode()) == 16, 'Final response undid the answer bound'
    assert choice['finish_reason'] == 'length'
    assert not choice['message'].get('reasoning_content'), 'Thinking-off request returned reasoning'

    content = ''
    finish = None
    with request({**bounded, 'stream': True}) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            for choice in json.loads(payload).get('choices', []):
                content += choice.get('delta', {}).get('content', '')
                finish = choice.get('finish_reason') or finish
    assert len(content.encode()) == 16, 'Streaming response exceeded the answer bound'
    assert finish == 'length'

    with request({'messages': [{'role': 'user', 'content': 'Say OK. <|think_xhigh|>'}],
                  'chat_template_kwargs': {'enable_thinking': False}}) as response:
        choice = json.load(response)['choices'][0]
    assert not choice['message'].get('reasoning_content'), 'Historical marker overrode explicit thinking-off'

    with request({'messages': [
        {'role': 'user', 'content': 'Calculate 2+2 using lookup, then explain the result.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'lookup-test', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'lookup-test', 'content': '4'}],
        'tools': [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Returns 2+2.',
                   'parameters': {'type': 'object', 'properties': {}}}}], 'reasoning_effort': 'low'}) as response:
        choice = json.load(response)['choices'][0]
    assert choice['message'].get('reasoning_content'), 'Successful tool result suppressed the reasoning channel'
    print('Live policy checks passed: bounded final/streaming answers, explicit thinking-off, post-tool reasoning.')


if __name__ == '__main__':
    main()
