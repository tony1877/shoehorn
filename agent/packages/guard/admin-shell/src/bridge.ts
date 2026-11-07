/**
 * Client for the exec bridge (`C:\Tools\bin\dsh-bridge.js`, protocol v2): one TCP connection per command, a
 * JSON request line `{version, token, command, cwd}`, then `output` frames and one `exit` frame. Closing the
 * connection makes the bridge kill the command's process tree, so aborting a call stops the command.
 */
import { Socket } from 'node:net'

export interface BridgeTarget {
  host: string
  port: number
  token: string
}

export interface BridgeResult {
  /** Exit code of the command; null when the bridge could not be reached or closed without one. */
  exitCode: number | null
  stdout: string
  stderr: string
  /** Why there is no exit code: connection refused, closed early, aborted. */
  failure?: string
}

/** The bridge runs PowerShell from base64, so quotes, `$` and newlines reach it unchanged. */
export function bridgeCommand(code: string): string {
  return `B64|${Buffer.from(code, 'utf8').toString('base64')}`
}

/** Run one PowerShell command through the bridge. Never rejects: failures come back in `failure`. */
export function runOnBridge(target: BridgeTarget, code: string, cwd: string, signal: AbortSignal,
  maxChars = 1 << 20): Promise<BridgeResult> {
  return new Promise((resolve) => {
    const out = { stdout: '', stderr: '' }
    let buffer = ''
    let settled = false
    const socket = new Socket()
    const finish = (exitCode: number | null, failure?: string) => {
      if (settled) return
      settled = true
      signal.removeEventListener('abort', onAbort)
      socket.destroy()
      resolve({ exitCode, ...out, ...failure !== undefined ? { failure } : {} })
    }
    const onAbort = () => finish(null, 'aborted')
    if (signal.aborted) return finish(null, 'aborted')
    signal.addEventListener('abort', onAbort, { once: true })

    socket.setEncoding('utf8')
    socket.on('error', error => finish(null, (error as NodeJS.ErrnoException).code === 'ECONNREFUSED'
      ? `the exec bridge is not running (nothing listens on ${target.host}:${target.port})`
      : `bridge connection failed: ${error.message}`))
    socket.on('close', () => finish(null, 'the bridge closed the connection without an exit status'))
    socket.on('data', (chunk: string) => {
      buffer += chunk
      let newline: number
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline)
        buffer = buffer.slice(newline + 1)
        if (line.trim() === '') continue
        let frame: { kind?: string; channel?: string; data?: unknown; code?: unknown }
        try { frame = JSON.parse(line) } catch { return finish(null, 'the bridge sent a malformed frame') }
        if (frame.kind === 'output' && (frame.channel === 'stdout' || frame.channel === 'stderr')) {
          const stream = frame.channel
          if (out[stream].length < maxChars) out[stream] += String(frame.data ?? '')
        } else if (frame.kind === 'exit') {
          // Windows exit codes arrive unsigned (e.g. 4294967295 for -1).
          const code = Number(frame.code)
          return finish(code > 0x7fffffff ? code - 0x100000000 : code)
        }
      }
    })
    socket.connect(target.port, target.host, () => {
      socket.write(`${JSON.stringify({ version: 2, token: target.token, command: bridgeCommand(code), cwd })}\n`)
    })
  })
}
