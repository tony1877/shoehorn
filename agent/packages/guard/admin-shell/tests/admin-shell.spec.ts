import { createServer } from 'node:net'
import type { AddressInfo, Server } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'
import { bridgeCommand, renderAdminResult, runOnBridge } from '../src/index.ts'

const TOKEN = 'x'.repeat(40)
let server: Server | undefined
afterEach(() => { server?.close(); server = undefined })

/** A fake bridge: checks the request, then answers with the given frames. */
function fakeBridge(frames: object[], seen: (request: Record<string, unknown>) => void = () => {}): Promise<number> {
  server = createServer(socket => {
    let input = ''
    socket.setEncoding('utf8')
    socket.on('data', (chunk: string) => {
      input += chunk
      if (!input.includes('\n')) return
      seen(JSON.parse(input.slice(0, input.indexOf('\n'))))
      for (const frame of frames) socket.write(`${JSON.stringify(frame)}\n`)
      socket.end()
    })
  })
  return new Promise(resolve => server!.listen(0, '127.0.0.1', () => resolve((server!.address() as AddressInfo).port)))
}

describe('admin_shell bridge client', () => {
  it('sends protocol v2 with the code base64-encoded and collects output and the exit code', async () => {
    let request: Record<string, unknown> = {}
    const port = await fakeBridge([
      { kind: 'output', channel: 'stdout', data: 'line 1\n' },
      { kind: 'output', channel: 'stderr', data: 'warn\n' },
      { kind: 'output', channel: 'stdout', data: 'line 2\n' },
      { kind: 'exit', code: 0 },
    ], r => { request = r })
    const code = `Get-Service 'x' | % { "$($_.Name)" }\nexit 0`
    const result = await runOnBridge({ host: '127.0.0.1', port, token: TOKEN }, code, 'C:\work', new AbortController().signal)
    expect(request).toEqual({ version: 2, token: TOKEN, command: bridgeCommand(code), cwd: 'C:\work' })
    expect(Buffer.from(String(request.command).slice(4), 'base64').toString('utf8')).toBe(code)
    expect(result).toEqual({ exitCode: 0, stdout: 'line 1\nline 2\n', stderr: 'warn\n' })
  })

  it('maps unsigned Windows exit codes and reports a bridge that is down', async () => {
    const port = await fakeBridge([{ kind: 'exit', code: 4294967295 }])
    expect((await runOnBridge({ host: '127.0.0.1', port, token: TOKEN }, 'x', '.', new AbortController().signal)).exitCode).toBe(-1)
    server!.close(); server = undefined
    const down = await runOnBridge({ host: '127.0.0.1', port, token: TOKEN }, 'x', '.', new AbortController().signal)
    expect(down.exitCode).toBeNull()
    expect(down.failure).toMatch(/not running|connection failed/)
  })

  it('renders output like the pwsh tool, with the exit marker last', () => {
    expect(renderAdminResult('ok\n', '', 0, 100)).toBe('ok')
    expect(renderAdminResult('', 'Access denied\n', 1, 100)).toBe('[stderr]\nAccess denied\n[exit code: 1]')
    expect(renderAdminResult('', '', 0, 100)).toBe('(no output)')
    expect(renderAdminResult('abcdefghij', '', 0, 4)).toBe('[... 6 earlier characters cut ...]\nghij')
    expect(renderAdminResult('a\r\nb\r\n', '\x1b[31;1mParserError: \x1b[0m\r\nbad\r\n', 1, 100))
      .toBe('a\nb\n[stderr]\nParserError: \nbad\n[exit code: 1]')
  })
})
