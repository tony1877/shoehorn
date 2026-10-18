/**
 * Model-facing PowerShell Consumer of the `ctx.shell` capability seam. Intended for
 * Windows compositions where a PowerShell executor (e.g.
 * `@deepseek-ai/dsh-pwsh-local`) backs `ctx.shell`; the tool contract is
 * PowerShell-dialect: native `C:\...` paths and `$env:NAME` variables.
 *
 * Behavior mirrors `dsh-tool-bash` call-for-call: foreground and
 * `run_in_background` execution (with a job registry composed, every call
 * registers its process with `ctx.jobs` as it starts, and a foreground call
 * waits on its job until the timeout passes), the managed `DSH_*` environment through the
 * shared `shell-env` registry, the per-call sandbox policy resolution (the
 * calling session's mode and cwd travel to the confining executor), the
 * sandbox-denial rendering with the same-turn escalation surface
 * (`sandbox_permissions` + `justification` resolved through
 * `ctx.approval`), and the bash marker/truncation rendering story. UI
 * presentation mirrors the bash tool's too: a completed foreground call is
 * a terminal card with the parsed exit-status pill, using the shared
 * exit-status parse from `@deepseek-ai/dsh-shell`.
 *
 * @module @deepseek-ai/dsh-tool-pwsh
 */

import { isAbsolute, resolve as resolvePath } from 'node:path'
import type {} from '@deepseek-ai/dsh-working-directory'
import { FiberState } from '@deepseek-ai/cordis'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { defineTool, TOOL_ABORTED } from '@deepseek-ai/dsh-tools'
import type { GenericCallView, TerminalCallView, ToolDefinition, ToolExecution, ToolResult, ToolResultView } from '@deepseek-ai/dsh-tools'
import { HarnessError } from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { JobRegistry } from '@deepseek-ai/dsh-jobs'
import type {} from '@deepseek-ai/dsh-shell-env'
import type {} from '@deepseek-ai/dsh-user-approval'
import type { SandboxExecutionPolicy, SandboxMode } from '@deepseek-ai/dsh-sandbox'
import { ESCALATION_TARGETS, approveEscalation, sandboxPermissionsDescription, validateEscalationArgs } from '@deepseek-ai/dsh-sandbox'
import type { SandboxPolicyService } from '@deepseek-ai/dsh-sandbox-policy'
import type { ShellExecRequest, ShellProcess, ShellRunResult } from '@deepseek-ai/dsh-shell'
import { parseExitStatus } from '@deepseek-ai/dsh-shell'
import { processJob, processOutcome, processSources, ringDelta } from './background.ts'
import { renderPwshJobRead, renderPwshPromoted, renderPwshResult } from './render.ts'
import type { RenderablePwshResult } from './render.ts'

declare module '@deepseek-ai/dsh-jobs' {
  interface JobKindMap {
    pwsh: 'pwsh'
  }
}

export const name = 'tool-pwsh'
export const inject = ['tools', 'shell', 'systemPrompt', 'shellEnv', 'workingDirectory']

/* jscpd:ignore-start -- the pwsh Config mirrors tool-bash's by design, like render/background. */
/** Configuration for the pwsh tool. */
export interface Config {
  /**
   * Expose `run_in_background` while a job registry is composed (default
   * true); disabled calls are also rejected. Without a registry the tool is
   * foreground-only regardless.
   */
  enableRunInBackground?: boolean
  /**
   * Auto-background: a foreground command still running after this many
   * milliseconds keeps running as a background job (the call returns its job
   * id and output so far) instead of being killed at the executor timeout.
   * `0` (default) keeps plain foreground runs. Needs `enableRunInBackground`
   * and a job runtime (`ctx.jobs`); without one the call runs as before.
   */
  autoBackgroundAfterMs?: number
  /** Upper bound on the foreground wait a call's own `timeoutMs` may request under auto-background (default 600000). */
  maxForegroundMs?: number
}

/** Runtime configuration schema for the pwsh tool plugin. */
export const Config: z<Config> = z.object({
  enableRunInBackground: z.boolean().default(true),
  autoBackgroundAfterMs: z.number().min(0).default(0),
  maxForegroundMs: z.number().min(1).default(600_000),
})
/* jscpd:ignore-end */

/** Parsed tool args; execute validates value constraints absent from ParameterSchemaSpec. */
interface PwshToolArgs {
  command: string
  description: string
  timeoutMs?: number
  workdir?: string
  run_in_background?: boolean
  sandbox_permissions?: string
  justification?: string
}

/** The canonical foreground result of one pwsh call (the `output.schema` value shape). */
interface PwshForegroundResult {
  cwd: string
  kind: 'foreground'
  exitCode: number | null
  signal: NodeJS.Signals | null
  timedOut: boolean
  aborted: boolean
  /** The reason the command was stopped from outside this call, such as a human kill of its job. */
  stopped?: string
  timeoutMs: number
  stdout: { text: string; truncated: boolean; spillPath?: string }
  stderr: { text: string; truncated: boolean; spillPath?: string }
  sandbox?: { mode: string; denied: boolean; enforcement?: string; runnerFailed?: boolean }
}

/** A foreground call that outlived its wait and continues as a background job. */
interface PwshPromotedResult {
  kind: 'promoted'
  jobId: string
  afterMs: number
  output: string
}

/** First line of a promoted result; presentResult keys the generic card on it. */
const PROMOTED_PREFIX = '[still running after '

/** Model-facing text for a command that moved to the background mid-run. */
function renderPromoted(value: PwshPromotedResult): string {
  const seconds = Math.round(value.afterMs / 1000)
  return [
    `${PROMOTED_PREFIX}${seconds}s - moved to background job ${value.jobId}; it keeps running]`,
    value.output.length > 0 ? value.output.replace(/\n+$/, '') : '(no output yet)',
    `[you are notified in-session when job ${value.jobId} finishes; read new output with job_output, stop it with job_kill]`,
  ].join('\n')
}

/* jscpd:ignore-start -- minimal mirror of dsh-tool-bash's validation and execute plumbing (Agent Note). */
function validatePwshArgs(args: PwshToolArgs): void {
  if (args.command.trim().length === 0) {
    throw new Error('invalid command: expected a non-empty string')
  }
  if (args.description.trim().length === 0) {
    throw new Error('invalid description: expected a non-empty string')
  }
  if (args.timeoutMs !== undefined && (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0)) {
    throw new Error(`invalid timeoutMs: expected a positive number, got ${JSON.stringify(args.timeoutMs)}`)
  }
  // The escalation pairing (sandbox_permissions ⇔ justification, non-empty) is
  // the shared rule both enforcing families validate identically.
  validateEscalationArgs(args.sandbox_permissions, args.justification)
}
/* jscpd:ignore-end */

function pwshDescription(
  backgroundEnabled: boolean,
  escalationModes: readonly SandboxMode[],
  autoBackgroundAfterMs = 0,
  maxForegroundMs = 600_000,
): string {
  const autoBackground = backgroundEnabled && autoBackgroundAfterMs > 0
    ? ` A foreground command still running after ${Math.round(autoBackgroundAfterMs / 1000)}s (or after \`timeoutMs\`, at most ${Math.round(maxForegroundMs / 1000)}s) is NOT killed: it keeps running as a background job, the call returns its job id and the output so far, and you are notified when it finishes.`
    : ''
  const background = backgroundEnabled
    ? 'Set `run_in_background: true` for long-running commands: the call returns a job id immediately; read its output with `job_output` and stop it with `job_kill`.' + autoBackground
    : 'Background execution is not available; long-running commands must finish within the timeout.'
  const base = 'Execute a PowerShell command (`pwsh -Command`) and return its stdout/stderr. '
    + 'Each call runs in a fresh pwsh process; pass `workdir` instead of using `cd`. Paths use native Windows form (`C:\\...`); read environment '
    + 'variables with `$env:NAME`. '
    + 'Managed `$env:DSH_*` variables expose current harness environment facts. '
    + 'Long output is truncated to its tail; the full output is saved to a file whose path is reported when available. '
    + 'On Windows a force-killed command settles as `[exit code: 1]` without a signal marker — treat it as an interruption, not a command failure. '
    + 'Provide `description` before `command` in the arguments. '
    + 'Before any delete or move, verify that the resolved absolute target path is the intended one; never run it against a computed path you have not checked. '
    + 'Do not assign to automatic variables such as `$HOME`; variable names are case-insensitive, so `$home` is the same read-only variable. '
    + 'Commands may run under a file sandbox; a blocked file operation is reported as `[sandbox: file access denied under <mode> mode]`, a policy denial: do not retry another way.'
  if (!windowsSandbox) return base
  // The language-mode and named-pipe contracts below are Windows-restricted-token
  // behavior, but the gate is 'any confining executor is mounted'
  // (escalation fields advertised). Every shipped composition pairing tool-pwsh
  // with a confining executor is win32-only, so the gate is equivalent. A POSIX
  // pwsh-sandbox composition must gate both sentences on the platform instead
  // (tracked in the pwsh-tool-and-executor Agent Note).
  return base + ' Under the Windows sandbox, read-only pwsh runs in PowerShell ConstrainedLanguage mode, while '
    + 'workspace-write stays in FullLanguage unless host policy says otherwise. In read-only, prefer cmdlets and core types (`[string]`, `[datetime]`, `[regex]`, `[guid]`); '
    + '.NET static calls (`[System.IO.*]::`, `[math]::`), `Add-Type`, COM objects, and reflection fail '
    + 'with "only core types" errors. `-f` formatting, property access, and core cmdlets work. '
    + 'In both confined modes, programs cannot open named pipes, so a command that captures another '
    + 'program\'s output through piped stdio (Node.js `child_process.spawn`/`exec` with the default '
    + '`stdio: \'pipe\'`) fails with EPERM, while `stdio: \'inherit\'` and `stdio: \'ignore\'` spawns '
    + 'work and PowerShell\'s own pipelines are unaffected. That EPERM is the documented boundary: '
    + 'do not retry the command another way — escalate the exact command once or restructure it to '
    + 'avoid capturing output.'
}

/** Resolve a relative per-call override against the Session's current directory. */
function resolveWorkdir(modelWorkdir: string | undefined, sessionCwd: string | undefined): string | undefined {
  if (modelWorkdir === undefined) return sessionCwd
  if (sessionCwd !== undefined && !isAbsolute(modelWorkdir)) {
    return resolvePath(sessionCwd, modelWorkdir)
  }
  return modelWorkdir
}

/** Detach the executor DTO from readonly Service Definition types into plain JSON data. */
function canonicalPwshResult(result: ShellRunResult, cwd: string): PwshForegroundResult {
  const output = (stream: ShellRunResult['stdout']) => ({
    text: stream.text,
    truncated: stream.truncated,
    ...stream.spillPath !== undefined ? { spillPath: stream.spillPath } : {},
  })
  return {
    cwd,
    kind: 'foreground',
    exitCode: result.exitCode,
    signal: result.signal,
    timedOut: result.timedOut,
    aborted: result.aborted,
    timeoutMs: result.timeoutMs,
    /* jscpd:ignore-start -- the canonical projection and background-handle shape mirror dsh-tool-bash's by design (Agent Note). */
    stdout: output(result.stdout),
    stderr: output(result.stderr),
    ...result.sandbox !== undefined ? {
      sandbox: {
        mode: result.sandbox.mode,
        denied: result.sandbox.denied,
        ...result.sandbox.enforcement !== undefined ? { enforcement: result.sandbox.enforcement } : {},
        ...result.sandbox.runnerFailed !== undefined ? { runnerFailed: result.sandbox.runnerFailed } : {},
      },
    } : {},
  }
}

/**
 * Wait for a started process up to `waitMs`, or until the call is cancelled.
 * The timer and the abort listener are always released, so neither outlives
 * the wait (the process itself keeps running on `timeout`).
 */
async function waitForProcess(proc: ShellProcess, waitMs: number, signal: AbortSignal): Promise<'done' | 'timeout' | 'aborted'> {
  let timer: ReturnType<typeof setTimeout> | undefined
  let onAbort: (() => void) | undefined
  try {
    return await Promise.race([
      proc.done.then(() => 'done' as const),
      new Promise<'timeout'>((resolve) => { timer = setTimeout(() => resolve('timeout'), waitMs) }),
      new Promise<'aborted'>((resolve) => {
        onAbort = () => resolve('aborted')
        signal.addEventListener('abort', onAbort, { once: true })
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
    if (onAbort !== undefined) signal.removeEventListener('abort', onAbort)
  }
}

/**
 * The foreground result of a process that settled within its wait. A handle
 * reports one merged stream (stdout, then a `[stderr]` section - the same
 * layout the foreground renderer produces), so it lands in `stdout`; loss
 * and spill facts map onto the truncation notice.
 */
function foregroundFromProcess(proc: ShellProcess, waitMs: number): PwshForegroundResult {
  const read = proc.readOutput()
  const spillPath = read.stdoutSpillPath ?? read.stderrSpillPath
  return {
    kind: 'foreground',
    exitCode: proc.exitCode,
    signal: proc.signal,
    timedOut: false,
    aborted: false,
    timeoutMs: waitMs,
    stdout: { text: read.delta, truncated: read.lossy, ...spillPath !== undefined ? { spillPath } : {} },
    stderr: { text: '', truncated: false },
    ...proc.sandbox !== undefined ? {
      sandbox: {
        mode: proc.sandbox.mode,
        denied: proc.sandbox.denied,
        ...proc.sandbox.enforcement !== undefined ? { enforcement: proc.sandbox.enforcement } : {},
        ...proc.sandbox.runnerFailed !== undefined ? { runnerFailed: proc.sandbox.runnerFailed } : {},
      },
    } : {},
  }
}

/** Canonical background-handle properties shared by the pwsh output union. */
const BACKGROUND_OUTPUT_PROPERTIES = {
  kind: { type: 'string', required: true, const: 'background' },
  jobId: { type: 'string', required: true },
  cwd: { type: 'string', required: true },
} as const
/* jscpd:ignore-end */

/* jscpd:ignore-start -- deliberate mirror of dsh-tool-bash's apply() preamble (pwsh-tool-and-executor Agent Note). */
export function apply(ctx: Context, config: Config = {}): void {
  const backgroundEnabled = config.enableRunInBackground ?? true
  const autoBackgroundAfterMs = config.autoBackgroundAfterMs ?? 0
  const maxForegroundMs = config.maxForegroundMs ?? 600_000
  if (!Number.isFinite(autoBackgroundAfterMs) || autoBackgroundAfterMs < 0) {
    throw new Error('tool-pwsh: autoBackgroundAfterMs must be a non-negative number')
  }
  if (!Number.isFinite(maxForegroundMs) || maxForegroundMs <= 0) {
    throw new Error('tool-pwsh: maxForegroundMs must be a positive number')
  }
  const autoBackground = backgroundEnabled && autoBackgroundAfterMs > 0
  const defaultMode = ctx.shell.sandboxMode
  const escalationModes: readonly SandboxMode[] = defaultMode === undefined ? [] : ESCALATION_TARGETS
  const sandboxPolicy: SandboxPolicyService | undefined = defaultMode === undefined ? undefined : ctx.get('sandboxPolicy')
  if (defaultMode !== undefined && sandboxPolicy === undefined) {
    throw new Error('tool-pwsh: the mounted bash executor confines but ctx.sandboxPolicy is missing')
  }
  /* jscpd:ignore-end */
  /** Resolve the complete standing policy for this call when a confining executor is mounted. */
  const resolveSandboxPolicy = (exec: ToolExecution): SandboxExecutionPolicy | undefined =>
    sandboxPolicy?.resolve(exec.agent === undefined ? {} : { session: exec.agent.session })

  /* jscpd:ignore-start -- deliberate mirror of dsh-tool-bash's escalation resolver (pwsh-tool-and-executor Agent Note). */
  /**
   * Resolve a sandbox-escalation request through `ctx.approval` BEFORE
   * anything executes, delegating the shared fail-closed sequence (strict
   * widening, channel resolution, outcome mapping) to
   * {@link approveEscalation}. This tool contributes only the composition
   * guard (the fields are unadvertised without a sandboxing executor, yet
   * schema validation checks advertised keys only, so an unadvertised
   * `sandbox_permissions` still reaches execute) and the approval
   * ingredients. The shared policy resolver is required whenever the
   * executor advertises confinement, so a split composition fails at
   * tool-plugin load.
   */
  const approvePwshEscalation = (
    mode: string,
    justification: string,
    exec: ToolExecution,
    standingPolicy: SandboxExecutionPolicy | undefined,
  ): Promise<SandboxMode> => {
    if (escalationModes.length === 0) {
      throw new Error('sandbox_permissions is not available in this composition (no sandboxing executor to escalate)')
    }
    const effectiveMode = (standingPolicy as SandboxExecutionPolicy).mode
    return approveEscalation(
      { requestedMode: mode, justification, effectiveMode, subject: 'command' },
      {
        approver: ctx.get('approval'),
        agent: exec.agent,
        callId: exec.callId,
        toolName: 'pwsh',
        signal: exec.signal,
      },
    )
  }
  /* jscpd:ignore-end */

  /** Abort translation shared by both foreground paths. */
  const abortedCall = (): HarnessError => {
    const error = new HarnessError('tool call aborted', TOOL_ABORTED)
    error.name = 'AbortError'
    return error
  }

  /**
   * Foreground run under auto-background. The command starts as a process
   * handle (so it can outlive this call) and is awaited for the foreground
   * wait. Finishing in time yields the ordinary foreground result; still
   * running hands the SAME process to `ctx.jobs` - nothing restarts, no output
   * is repeated (the "so far" read consumes the handle's cursor) - and the call
   * returns the job id. A cancelled call kills the command before promotion;
   * after promotion the job owns cancellation. If the job runtime refuses the
   * registration (e.g. its per-owner limit), the command is stopped and the
   * call fails rather than leaving an untracked process behind.
   */
  const runWithAutoBackground = async (
    request: ShellExecRequest,
    args: PwshToolArgs,
    exec: ToolExecution,
    jobs: JobRegistry,
  ): Promise<PwshForegroundResult | PwshPromotedResult> => {
    const waitMs = Math.min(args.timeoutMs ?? autoBackgroundAfterMs, maxForegroundMs)
    if (exec.signal.aborted) throw abortedCall()
    const proc = ctx.shell.start(ctx.shell.resolve(request))
    const outcome = await waitForProcess(proc, waitMs, exec.signal)
    if (outcome === 'aborted') {
      proc.kill()
      await proc.done
      throw abortedCall()
    }
    if (outcome === 'done' || proc.status !== 'running') {
      await proc.done
      return foregroundFromProcess(proc, waitMs)
    }
    const soFar = renderPwshProcessRead(proc.readOutput(), undefined, escalationModes)
    let id: string
    try {
      id = jobs.start({
        kind: 'pwsh',
        label: args.command,
        ...exec.agent ? { owner: exec.agent } : {},
        run: () => ({
          cancel: () => void proc.kill(),
          done: proc.done.then(() => processOutcome(proc)),
          readOutput: () => renderPwshProcessRead(proc.readOutput(), proc.sandbox, escalationModes),
        }),
      })
    } catch (error: unknown) {
      proc.kill()
      await proc.done
      const reason = error instanceof Error ? error.message : String(error)
      throw new Error(`command still running after ${Math.round(waitMs / 1000)}s could not move to the background (${reason}); it was stopped`)
    }
    return { kind: 'promoted', jobId: id, afterMs: waitMs, output: soFar }
  }

  ctx.systemPrompt.section({
    name: 'tool:pwsh',
    order: ctx.systemPrompt.getSectionOrder('TOOL_PWSH'),
    text: 'Non-zero exits are reported as `[exit code: N]` markers; investigate failures before moving on. '
      + 'On Windows a killed process settles as `[exit code: 1]` without a signal marker; treat a bare exit 1 after an interruption as a termination, not a command failure.',
  })

  ctx.tools.register(defineTool({
    name: 'pwsh',
    description: pwshDescription(backgroundEnabled, escalationModes, autoBackgroundAfterMs, maxForegroundMs),
    /* jscpd:ignore-start -- deliberate mirror of dsh-tool-bash's parameter surface (pwsh-tool-and-executor Agent Note). */
    parameters: {
      command: { type: 'string', required: true, description: 'The PowerShell command to execute.' },
      description: {
        type: 'string',
        required: true,
        description: 'Clear, concise description of what this command does in active voice, '
          + '5-10 words (shown in the UI). Examples: "ls" → "List files in current directory"; '
          + '"git status" → "Show working tree status"; "Get-Process" → "List running processes".',
      },
      timeoutMs: {
        type: 'number',
        description: autoBackground
          ? `How long to wait in the foreground, in milliseconds (default ${autoBackgroundAfterMs}, max ${maxForegroundMs}). A command still running then continues as a background job instead of being killed.`
          : 'Timeout in milliseconds. The executor applies its configured default and cap, and kills the command on expiry.',
      },
      workdir: { type: 'string', description: 'Working directory for this command. Defaults to the session workspace; a relative path is resolved against it.' },
      ...backgroundEnabled ? {
        run_in_background: { type: 'boolean' as const, description: 'Run in the background and return a job id immediately (collect with job_output, stop with job_kill). No timeout applies.' },
      } : {},
      ...escalationModes.length > 0 ? {
        sandbox_permissions: {
          type: 'string' as const,
          enum: [...escalationModes],
          description: 'The wider sandbox mode this command needs. Only valid as a one-shot retry of a command the sandbox just denied; requires justification and user approval.',
        },
      })
      return { id, process: () => proc, stopped: () => stopped }
    }
    /** Wait on a registered foreground command until it settles or the timeout passes. */
    const waitOnJob = async (registry: JobRegistry, attached: StartedJob, exec: ToolExecution, spec: ShellExecSpec) => {
      const owner = exec.agent?.id
      const timeoutMs = spec.timeoutMs
      /**
       * Stop the job on this call's own account and stay on it until it
       * settles, so the settlement is `awaited` and no completion notice
       * follows a result this call already carries; the record then leaves
       * with the call, as the model never saw the id.
       */
      const stop = async (reason: string): Promise<JobView> => {
        registry.kill(attached.id, owner, reason)
        const settled = await registry.wait(attached.id, timeoutMs, owner)
        if (settled.status !== 'running' && settled.status !== 'stopping') registry.remove(attached.id, owner)
        return settled
      }
      let view: JobView
      try {
        view = await registry.wait(attached.id, timeoutMs, owner, exec.signal)
      } catch {
        // A wait on the caller's own live job rejects only for the caller's
        // abort: the model's call is over, so the command goes with it, as
        // under the deadline kill.
        await stop('tool call aborted')
        throw toolAborted()
      }
      if ((view.status === 'running' || view.status === 'stopping') && attached.process() === undefined) {
        // The wait expired while preparation still ran: nothing is running
        // to keep, so this is the deadline's preparation timeout, with the
        // settled empty result the executor's own deadline would have given.
        await stop('timed out during preparation')
        return {
          kind: 'foreground' as const,
          cwd: spec.workdir,
          exitCode: null,
          signal: null,
          timedOut: true,
          aborted: false,
          timeoutMs,
          stdout: { text: '', truncated: false },
          stderr: { text: '', truncated: false },
          ...spec.sandboxPolicy !== undefined ? { sandbox: { mode: spec.sandboxPolicy.mode, denied: false } } : {},
        }
      }
      if (view.status === 'running' || view.status === 'stopping') {
        // The wait timed out: the job keeps running. One consuming read seeds
        // the result with the output so far, so `job_output` continues exactly
        // where this result stops.
        const read = registry.read(attached.id, owner)
        return {
          kind: 'promoted' as const,
          cwd: spec.workdir,
          jobId: attached.id,
          timeoutMs,
          output: renderPwshJobRead(
            ringDelta(read.chunks), read.lossy, read.job.output.spillPaths ?? [], attached.process()?.sandbox, escalationModes,
          ),
        }
      }
      // Settled while the call waited: the model never saw the id, so the
      // record leaves the registry with this result. The process handle keeps
      // the split streams and classification the foreground result needs.
      registry.remove(attached.id, owner)
      const process = attached.process()
      // No process was published: the starter failed before the spawn, and the
      // job's terminal detail is the failure the foreground projection would
      // have rejected with.
      if (process === undefined) throw new Error(view.detail)
      const result = await process.result()
      const stopped = attached.stopped()
      return { ...canonicalPwshResult(result, spec.workdir), ...stopped !== undefined ? { stopped } : {} }
    }
    /* jscpd:ignore-end */
    return defineTool({
      name: 'pwsh',
      description: pwshDescription(escalationModes.length > 0),
      /* jscpd:ignore-start -- deliberate mirror of dsh-tool-bash's parameter surface (pwsh-tool-and-executor Agent Note). */
      parameters: {
        description: {
          type: 'string',
          required: true,
          description: 'Clear, concise description of what this command does in active voice, '
            + '5-10 words (shown in the UI). Examples: "ls" → "List files in current directory"; '
            + '"git status" → "Show working tree status"; "Get-Process" → "List running processes".',
        },
        command: { type: 'string', required: true, description: 'The PowerShell command to execute.' },
        timeoutMs: {
          type: 'number',
          description: promote
            ? 'Timeout in milliseconds. The executor applies its configured default and cap; on expiry the command moves to the background as a job instead of being killed.'
            : 'Timeout in milliseconds. The executor applies its configured default and cap, and kills the command on expiry.',
        },
        workdir: { type: 'string', description: 'Working directory for this command. Defaults to the session workspace; a relative path is resolved against it.' },
        ...background ? {
          run_in_background: { type: 'boolean' as const, description: 'Run in the background and return a job id immediately (collect with job_output, stop with job_kill). No timeout applies.' },
        } : {},
        ...escalationModes.length > 0 ? {
          sandbox_permissions: {
            type: 'string' as const,
            enum: [...escalationModes],
            description: sandboxPermissionsDescription('command'),
          },
          {
            type: 'object',
            additionalProperties: false,
            properties: {
              kind: { type: 'string', required: true, const: 'promoted' },
              jobId: { type: 'string', required: true },
              afterMs: { type: 'number', required: true },
              output: { type: 'string', required: true },
            },
          },
          {
            type: 'object',
            additionalProperties: false,
            properties: {
              kind: { type: 'string', required: true, const: 'foreground' },
              exitCode: { required: true, oneOf: [{ type: 'integer' }, { type: 'null' }] },
              signal: { required: true, oneOf: [{ type: 'string' }, { type: 'null' }] },
              timedOut: { type: 'boolean', required: true },
              aborted: { type: 'boolean', required: true },
              timeoutMs: { type: 'number', required: true },
              stdout: {
                type: 'object',
                additionalProperties: false,
                required: true,
                properties: {
                  text: { type: 'string', required: true },
                  truncated: { type: 'boolean', required: true },
                  spillPath: { type: 'string' },
                },
              },
            },
            {
              type: 'object',
              additionalProperties: false,
              properties: {
                kind: { type: 'string', required: true, const: 'foreground' },
                cwd: { type: 'string', required: true },
                exitCode: { required: true, oneOf: [{ type: 'integer' }, { type: 'null' }] },
                signal: { required: true, oneOf: [{ type: 'string' }, { type: 'null' }] },
                timedOut: { type: 'boolean', required: true },
                aborted: { type: 'boolean', required: true },
                stopped: { type: 'string' },
                timeoutMs: { type: 'number', required: true },
                stdout: {
                  type: 'object',
                  additionalProperties: false,
                  required: true,
                  properties: {
                    text: { type: 'string', required: true },
                    truncated: { type: 'boolean', required: true },
                    spillPath: { type: 'string' },
                  },
                },
                stderr: {
                  type: 'object',
                  additionalProperties: false,
                  required: true,
                  properties: {
                    text: { type: 'string', required: true },
                    truncated: { type: 'boolean', required: true },
                    spillPath: { type: 'string' },
                  },
                },
                sandbox: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    mode: { type: 'string', required: true },
                    denied: { type: 'boolean', required: true },
                    enforcement: { type: 'string' },
                    runnerFailed: { type: 'boolean' },
                  },
                },
              },
            },
          ],
        },
        /* jscpd:ignore-end */
        render: (_args, value) => [{
          type: 'text',
          text: value.kind === 'background'
            ? `started background job ${value.jobId}`
            : value.kind === 'promoted'
              ? renderPwshPromoted(value)
              : renderPwshResult(value as RenderablePwshResult, escalationModes),
        }],
        presentationMeta: (_args, value) => ({ cwd: value.cwd }),
      },
      /* jscpd:ignore-start -- the execute path mirrors dsh-tool-bash's by design (see the pwsh-tool-and-executor Agent Note). */
      async execute(args: PwshToolArgs, exec) {
        validatePwshArgs(args)
        // Description is display metadata; workdir defaults to the caller's session.
        const standingPolicy = resolveSandboxPolicy(exec)
        const approvedMode = args.sandbox_permissions !== undefined && args.justification !== undefined
          ? await approvePwshEscalation(args.sandbox_permissions, args.justification, exec, standingPolicy)
          : undefined
        const policy = approvedMode === undefined
          ? standingPolicy
          : { ...(standingPolicy as SandboxExecutionPolicy), mode: approvedMode }
        const cwd = exec.agent === undefined ? undefined : await ctx.workingDirectory.ensure(exec.agent, exec.signal)
        const workdir = resolveWorkdir(args.workdir, cwd)
        const request: ShellExecRequest = {
          command: args.command,
          ...workdir !== undefined ? { workdir } : {},
          ...args.timeoutMs !== undefined ? { timeoutMs: args.timeoutMs } : {},
          dshEnv: ctx.shellEnv.collect(exec),
          ...policy !== undefined ? { sandboxPolicy: policy } : {},
        }
        if (args.run_in_background === true) {
          // Undeclared keys are allowed, so schema omission also needs enforcement.
          if (!backgroundEnabled) {
            throw new Error('run_in_background is disabled for this deployment (enableRunInBackground: false)')
          }
          if (jobs === undefined) {
            throw new Error('background jobs unavailable: load @deepseek-ai/dsh-jobs and @deepseek-ai/dsh-tool-jobs')
          }
          // The caller owns cancellation until ctx.jobs commits detached ownership.
          if (exec.signal.aborted) throw toolAborted()
          const spec = ctx.shell.resolve({ ...request, onExpiry: 'none' })
          return { kind: 'background' as const, jobId: startJob(jobs, args, exec, spec).id, cwd: spec.workdir }
        }
        // A foreground call is a job the tool waits on, so the command is
        // visible and killable from the moment it starts and outlives the wait
        // when the timeout passes. Admission refused at the start (the owner's
        // job limit, no controller) runs the command under the deadline kill
        // instead, exactly as a composition without a registry does.
        if (jobs !== undefined && promote) {
          const spec = ctx.shell.resolve({ ...request, onExpiry: 'none' })
          let attached: StartedJob | undefined
          try {
            attached = startJob(jobs, args, exec, spec)
          } catch (error) {
            ctx.logger.warn(`pwsh: job registration refused, running in the foreground with the timeout kill instead: ${String(error)}`)
          }
          if (attached !== undefined) return waitOnJob(jobs, attached, exec, spec)
        }
        const spec = ctx.shell.resolve({ ...request, signal: exec.signal })
        const foreground = await ctx.shell.execute(spec)
        const result = await foreground.result()
        if (result.aborted) throw toolAborted()
        return canonicalPwshResult(result, spec.workdir)
      },
      /* jscpd:ignore-end */
      render: (_args, value) => [{
        type: 'text',
        text: value.kind === 'background'
          ? `started background job ${value.jobId}`
          : value.kind === 'promoted'
            ? renderPromoted(value as PwshPromotedResult)
            : renderPwshResult(value as RenderablePwshResult, escalationModes),
      }],
    },
    /* jscpd:ignore-start -- the execute path mirrors dsh-tool-bash's by design (see the pwsh-tool-and-executor Agent Note). */
    async execute(args: PwshToolArgs, exec) {
      validatePwshArgs(args)
      // Description is display metadata; workdir defaults to the caller's session.
      const standingPolicy = resolveSandboxPolicy(exec)
      const approvedMode = args.sandbox_permissions !== undefined && args.justification !== undefined
        ? await approvePwshEscalation(args.sandbox_permissions, args.justification, exec, standingPolicy)
        : undefined
      const policy = approvedMode === undefined
        ? standingPolicy
        : { ...(standingPolicy as SandboxExecutionPolicy), mode: approvedMode }
      const workdir = resolveWorkdir(args.workdir, exec)
      const request = {
        command: args.command,
        ...workdir !== undefined ? { workdir } : {},
        ...args.timeoutMs !== undefined ? { timeoutMs: args.timeoutMs } : {},
        dshEnv: ctx.shellEnv.collect(exec),
        ...policy !== undefined ? { sandboxPolicy: policy } : {},
      }
      if (args.run_in_background === true) {
        // Undeclared keys are allowed, so schema omission also needs enforcement.
        if (!backgroundEnabled) {
          throw new Error('run_in_background is disabled for this deployment (enableRunInBackground: false)')
        }
        const jobs = ctx.get('jobs')
        if (jobs === undefined) {
          throw new Error('background jobs unavailable: load @deepseek-ai/dsh-jobs and @deepseek-ai/dsh-tool-jobs')
        }
        // The caller owns cancellation until ctx.jobs commits detached ownership.
        if (exec.signal.aborted) {
          const error = new HarnessError('tool call aborted', TOOL_ABORTED)
          error.name = 'AbortError'
          throw error
        }
        // Task preflight finishes before the starter can spawn a process.
        const id = jobs.start({
          kind: 'pwsh',
          label: args.command,
          ...exec.agent ? { owner: exec.agent } : {},
          run: () => {
            const proc = ctx.shell.start(ctx.shell.resolve(request))
            return {
              cancel: () => void proc.kill(),
              done: proc.done.then(() => processOutcome(proc)),
              readOutput: () => renderPwshProcessRead(proc.readOutput(), proc.sandbox, escalationModes),
            }
          },
        })
        return { kind: 'background' as const, jobId: id }
      }
      const jobs = autoBackground ? ctx.get('jobs') : undefined
      if (jobs !== undefined) {
        return runWithAutoBackground(request, args, exec, jobs)
      }
      const result = await ctx.shell.run(ctx.shell.resolve({
        ...request,
        signal: exec.signal,
      }))
      if (result.aborted) {
        const error = new HarnessError('tool call aborted', TOOL_ABORTED)
        error.name = 'AbortError'
        throw error
      }
      return canonicalPwshResult(result)
    },
    /* jscpd:ignore-end */
    /* jscpd:ignore-start -- the background call card mirrors presentBashCall's by design (Agent Note). */
    presentCall: (args: PwshToolArgs): TerminalCallView | GenericCallView => {
      // Background acknowledgements carry no terminal exit status; the generic
      // card mirrors the bash tool's background presentation.
      if (args.run_in_background === true) {
        return {
          card: 'terminal',
          title: args.command,
          description: args.description,
          ...args.workdir !== undefined ? { cwd: args.workdir } : {},
        }
      }
      return {
        card: 'terminal',
        title: args.command,
        description: args.description,
        ...args.workdir !== undefined ? { cwd: args.workdir } : {},
      }
    },
    /* jscpd:ignore-end */
    /* jscpd:ignore-start -- the completed-result presentation mirrors presentBashResult's by design (Agent Note). */
    presentResult: (args: unknown, result: ToolResult): ToolResultView | undefined => {
      const block = result.content.length === 1 ? result.content[0] : undefined
      if (block === undefined || block.type !== 'text') return undefined
      const raw = block.text
      const isBackground = (typeof args === 'object' && args !== null && (args as { run_in_background?: unknown }).run_in_background === true)
        || raw.startsWith(PROMOTED_PREFIX)
      // Background acknowledgements (including auto-backgrounded runs) and errors have no terminal exit status.
      if (isBackground || result.isError) {
        return { card: 'generic', content: [{ type: 'text', text: `\`\`\`console\n${raw.replace(/\n+$/, '')}\n\`\`\`` }] }
      }
      // The exit marker becomes the card's exit pill, so it leaves the output body.
      const { body, ...exit } = parseExitStatus(raw)
      return { card: 'terminal', output: body, ...exit }
    },
    /* jscpd:ignore-end */
  }))
}
