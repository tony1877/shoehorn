/**
 * `admin_shell`: run a PowerShell command ELEVATED on this machine, through the exec bridge.
 *
 * dsh's own shell runs with the user's normal rights, and Windows cannot raise a running process. The bridge
 * (`C:\Tools\bin\dsh-bridge.js` on 127.0.0.1:9999, started elevated by the `dsh-bridge` scheduled task) is the
 * elevated side; this plugin is the model-facing tool for it, in place of shelling out to `brg.ps1`. Every
 * command goes through `ctx.approval` first (the human sees and allows the exact command), unless
 * `approve: false`.
 *
 * @module @deepseek-ai/dsh-admin-shell
 */

import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { isAbsolute, join, resolve as resolvePath } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolExecution, ToolResult, ToolResultView } from '@deepseek-ai/dsh-tools'
import type {} from '@deepseek-ai/dsh-user-approval'
import { runOnBridge } from './bridge.ts'

export { bridgeCommand, runOnBridge } from './bridge.ts'

export const name = 'admin-shell'
export const inject = ['tools']

export interface Config {
  host?: string
  port?: number
  /** File holding the bridge token (default `~/.dsh/bridge-token.txt`, or `DSH_BRIDGE_TOKEN_FILE`). */
  tokenFile?: string
  /** Ask the human before every command (default true). */
  approve?: boolean
  /** Output kept per stream, in characters; the tail is kept. */
  maxOutputChars?: number
}

export const Config: z<Config> = z.object({
  host: z.string().default('127.0.0.1'),
  port: z.natural().default(9999),
  tokenFile: z.string().default(''),
  approve: z.boolean().default(true),
  maxOutputChars: z.natural().default(30000),
})

interface AdminShellArgs {
  description: string
  command: string
  workdir?: string
}

/** Keep the tail of long output; the end of a command's output is usually what matters. */
function tail(text: string, max: number): string {
  if (text.length <= max) return text
  return `[... ${text.length - max} earlier characters cut ...]\n${text.slice(text.length - max)}`
}

/**
 * Plain text: CRLF to LF, and no ANSI colors. The bridge's preamble turns PowerShell's colors off, but a parse
 * error is printed before the preamble runs, in color.
 */
function plain(text: string): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '').replace(/\r\n?/g, '\n').replace(/\s+$/, '')
}

/** The text the model reads: output, stderr, then the same exit marker the pwsh tool uses. */
export function renderAdminResult(stdout: string, stderr: string, exitCode: number, max: number): string {
  const parts: string[] = []
  if (stdout.trim() !== '') parts.push(tail(plain(stdout), max))
  if (stderr.trim() !== '') parts.push(`[stderr]\n${tail(plain(stderr), max)}`)
  if (parts.length === 0) parts.push('(no output)')
  if (exitCode === 124) parts.push('[the bridge stopped the command after its 120 s limit]')
  return exitCode === 0 ? parts.join('\n') : `${parts.join('\n')}\n[exit code: ${exitCode}]`
}

export function apply(ctx: Context, config: Config): void {
  const tokenFile = config.tokenFile || process.env.DSH_BRIDGE_TOKEN_FILE || join(homedir(), '.dsh', 'bridge-token.txt')
  const host = config.host ?? '127.0.0.1'
  const port = config.port ?? 9999
  const max = config.maxOutputChars ?? 30000

  const readToken = (): string => {
    let token: string
    try { token = readFileSync(tokenFile, 'utf8').trim() } catch { throw new Error(`admin_shell: cannot read the bridge token file ${tokenFile}`) }
    if (token.length < 32) throw new Error(`admin_shell: the bridge token in ${tokenFile} is missing or too short`)
    return token
  }

  const approve = async (args: AdminShellArgs, exec: Pick<ToolExecution, 'agent' | 'callId' | 'signal'>): Promise<void> => {
    if (config.approve === false) return
    const approval = ctx.get('approval')
    if (approval === undefined || exec.agent === undefined) {
      throw new Error('admin_shell needs approval and none can be asked in this session; ask the user to run the command in an admin shell')
    }
    const outcome = await approval.request({
      agent: exec.agent,
      toolName: 'admin_shell',
      callId: exec.callId,
      reason: `Run ELEVATED (administrator): ${args.description}`,
      signal: exec.signal,
    })
    if (outcome !== 'allowed-once') {
      throw new Error(outcome === 'rejected'
        ? 'The user declined this elevated command. Do not retry it another way; ask the user how to proceed.'
        : `Approval was not given (${outcome}); the command did not run.`)
    }
  }

  ctx.tools.register(defineTool({
    name: 'admin_shell',
    description: 'Run a PowerShell 7 command on this machine with ADMINISTRATOR rights. Use it only for what needs '
      + 'elevation (services, drivers, firewall, scheduled tasks, protected folders, machine-wide settings); '
      + 'use pwsh for everything else. The user approves each command before it runs. Each call is a fresh, '
      + 'non-interactive pwsh process (no prompts; pass -Force / -Confirm:$false where a cmdlet would ask), '
      + 'limited to 120 s; output comes back when it finishes. Pass `workdir` instead of using cd.',
    parameters: {
      description: {
        type: 'string',
        required: true,
        description: 'What the command does and why it needs admin rights, in a few words (shown to the user when they approve).',
      },
      command: { type: 'string', required: true, description: 'The PowerShell code to run elevated. Quotes, $ and multiple lines are fine.' },
      workdir: { type: 'string', description: 'Working directory (absolute, or relative to the session directory).' },
    },
    output: { schema: { type: 'string' as const }, render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }] },
    execute: async (args, exec) => {
      if (args.command.trim() === '') throw new Error('invalid command: expected a non-empty string')
      const sessionCwd = exec.agent?.session.header.cwd ?? process.cwd()
      const cwd = args.workdir === undefined ? sessionCwd
        : isAbsolute(args.workdir) ? args.workdir : resolvePath(sessionCwd, args.workdir)
      const token = readToken()
      await approve(args, exec)
      const result = await runOnBridge({ host, port, token }, args.command, cwd, exec.signal, max * 4)
      if (result.exitCode === null) {
        if (result.failure === 'aborted') throw new Error('admin_shell: the call was cancelled; the bridge stopped the command')
        throw new Error(`admin_shell: ${result.failure}. Tell the user; the bridge starts with dsh (C:\\Tools\\bin\\start-dsh.ps1).`)
      }
      return renderAdminResult(result.stdout, result.stderr, result.exitCode, max)
    },
    presentCall: args => ({
      card: 'terminal',
      title: args.command,
      description: `[admin] ${args.description}`,
      ...args.workdir !== undefined ? { cwd: args.workdir } : {},
    }),
    presentResult: (_args: unknown, result: ToolResult): ToolResultView | undefined => {
      const block = result.content.length === 1 ? result.content[0] : undefined
      if (block === undefined || block.type !== 'text') return undefined
      if (result.isError) return { card: 'generic', content: [{ type: 'text', text: block.text }] }
      const exit = /\n\[exit code: (-?\d+)\]$/.exec(block.text)
      return exit?.[1] !== undefined
        ? { card: 'terminal', output: block.text.slice(0, exit.index), exitCode: Number(exit[1]) }
        : { card: 'terminal', output: block.text, exitCode: 0 }
    },
  }))
}
