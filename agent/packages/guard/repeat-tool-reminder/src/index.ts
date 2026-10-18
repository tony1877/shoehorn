/**
 * Advisory per-agent repeat-call detector. It enriches post-execute decisions
 * with logged model context without vetoing or rewriting calls. Configuration
 * and chain semantics live in the package README; rationale lives in the
 * repeat-tool-reminder Agent Note.
 * @module @deepseek-ai/dsh-repeat-tool-reminder
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent, PreStepDecision } from '@deepseek-ai/dsh-agent'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContextFormed } from '@deepseek-ai/dsh-llm'
declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'repeat-tool-reminder': { kind: 'repeat-tool-reminder' } & ContextFormed
  }
}

import type { ContentBlock, MessageSource } from '@deepseek-ai/dsh-llm'
import type { UserMessage } from '@deepseek-ai/dsh-session'
import type { PostToolDecision, ToolExecution, ToolExecutionResult } from '@deepseek-ai/dsh-tools'

export const name = 'repeat-tool-reminder'

/**
 * Plugin config, validated by the same-named schemastery schema plus the
 * load-time checks in `apply` (misconfiguration fails loud: an empty
 * `thresholds` list, a non-integer, a value below 2, or a duplicate throws at
 * plugin load, never a silent fall-back). `include`/`exclude` entries are
 * `*`-wildcard predicates over tool names at call time, not references to
 * registry entries — a pattern matching no currently registered tool is valid
 * (`exclude: [mcp_*]` must stay legal in a deployment that loads no MCP tools).
 */
export interface Config {
  /** Consecutive-repeat counts that trigger a reminder (default `[3, 5, 8]`). */
  thresholds?: number[]
  /** Tool-name patterns to track; empty means every tool is tracked. */
  include?: string[]
  /** Tool-name patterns transparent to the chain (neither count nor reset). */
  exclude?: string[]
  /**
   * Maximum characters of canonical arguments quoted in the DETAILED reminder
   * (default 500). Large payloads (a `write` body, a long command) would
   * otherwise ride into the next request unbounded — precisely in a loop
   * scenario; the cap bounds the reminder, never the detection (the chain key
   * always compares the FULL canonical string).
   */
  argumentsPreviewChars?: number
  /**
   * Same-result counts that trigger a reminder (default `[]`: off). A tracked
   * call counts toward them when its model-facing result equals another
   * result of the same tool within the agent's last {@link resultWindow}
   * result-tracked calls, whatever the arguments: a loop that varies file
   * names or wording but keeps getting the same answer.
   */
  resultThresholds?: number[]
  /** Tool-name patterns whose results are compared; empty means every tracked tool. */
  resultInclude?: string[]
  /** How many of the agent's most recent result-tracked calls are searched for an equal result (default 12). */
  resultWindow?: number
  /** Maximum characters of the repeated result quoted in the detailed same-result reminder (default 500). */
  resultPreviewChars?: number
  /**
   * Text appended to every reminder (default `''`), e.g. a chat-template
   * control tag that raises the model's reasoning effort for the rest of the
   * conversation.
   */
  reminderSuffix?: string
  /** Advisory counts of tracked calls since a successful reset tool; empty disables it. */
  investigationThresholds?: number[]
  /** Tool names whose successful result resets the investigation count. */
  investigationResetTools?: string[]
}

export const Config: z<Config> = z.object({
  thresholds: z.array(z.number()).default([3, 5, 8]),
  include: z.array(z.string()).default([]),
  exclude: z.array(z.string()).default([]),
  argumentsPreviewChars: z.number().default(500),
  resultThresholds: z.array(z.number()).default([]),
  resultInclude: z.array(z.string()).default([]),
  resultWindow: z.number().default(12),
  resultPreviewChars: z.number().default(500),
  reminderSuffix: z.string().default(''),
  investigationThresholds: z.array(z.number()).default([]),
  investigationResetTools: z.array(z.string()).default(['write', 'edit', 'str_replace_editor']),
})

/**
 * The `{kind:'repeat-tool-reminder'}` producer source stamped on every reminder
 * this guard injects — the label is load-bearing (an unlabeled context would
 * render as a user prompt in derived history).
 */
const REMINDER_SOURCE: MessageSource = { kind: 'repeat-tool-reminder' }

/**
 * The gentle first-threshold reminder. Keyed to `thresholds[0]`, not a literal
 * count, so a custom first threshold keeps the gentle-then-detailed escalation.
 */
const GENTLE_REMINDER =
  'You are repeating the exact same tool call with identical arguments. '
  + 'Carefully analyze the previous result before calling again: if the task is '
  + 'not complete, try a different approach or different arguments instead of '
  + 'repeating the call.'

/** The detailed later-threshold reminder naming the tool, the run length, and the canonical arguments. */
function detailedReminder(toolName: string, count: number, canonicalArguments: string): string {
  return 'Repeated tool call detected:\n'
    + `- tool: ${toolName}\n`
    + `- consecutive_calls: ${count}\n`
    + `- arguments: ${canonicalArguments}\n`
    + 'The repeated calls are not making progress. Do not call this tool with '
    + 'these exact arguments again. Inspect the latest result and choose a '
    + 'different action, different arguments, or finish the task if enough '
    + 'evidence has been gathered.'
}

/** The gentle first same-result reminder, keyed to `resultThresholds[0]`. */
const GENTLE_RESULT_REMINDER =
  'Your recent calls to this tool keep returning the same result. Running more '
  + 'variations of the same check will not change it. Before calling again, work '
  + 'out why the result differs from what you expected: re-check your assumptions, '
  + 'including any expectation stated in the task, which may itself be wrong. Then '
  + 'change approach, or report the discrepancy and finish.'

/** The detailed later same-result reminder naming the tool, the count, the window, and the repeated result. */
function detailedResultReminder(toolName: string, count: number, window: number, result: string): string {
  return 'Repeated tool result detected:\n'
    + `- tool: ${toolName}\n`
    + `- same_result_count: ${count} of the last ${window} calls\n`
    + `- result: ${result}\n`
    + 'These calls are not making progress: the result is not going to change. '
    + 'Do not run another variation of this check. State what you expected and '
    + 'why the observed result differs, verify that expectation itself, then take '
    + 'a different action or finish the task and report the discrepancy.'
}

/**
 * Deep key-sort of a parsed-JSON value so two argument objects that differ
 * only in property order canonicalize identically. Arguments reach the guard
 * as the loop's `JSON.parse` output (or its raw-string fallback for malformed
 * argument JSON), so JSON's value domain is the whole input domain — no
 * bigint, cycle, or `undefined` handling exists because no input path can
 * produce them.
 */
function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJsonValue)
  if (value !== null && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const sorted: Record<string, unknown> = {}
    for (const key of Object.keys(record).sort()) {
      sorted[key] = sortJsonValue(record[key])
    }
    return sorted
  }
  return value
}

/** Canonical string form of a call's arguments: deep key-sort, then stringify. */
function canonicalize(argumentsValue: unknown): string {
  return JSON.stringify(sortJsonValue(argumentsValue))
}

/** Compile one `*`-wildcard pattern to an anchored RegExp (every other regex metacharacter is matched literally). */
function wildcardToRegExp(pattern: string): RegExp {
  const escaped = pattern.replace(/[|\\{}()[\]^$+?.]/g, String.raw`\$&`)
  return new RegExp(`^${escaped.replaceAll('*', '.*')}$`)
}

/**
 * Head-truncate the canonical arguments for quoting in the detailed reminder,
 * marking how much was omitted. Bounds only the model-visible text — the
 * chain key always uses the full canonical string.
 */
function previewArguments(canonical: string, cap: number): string {
  if (canonical.length <= cap) return canonical
  return `${canonical.slice(0, cap)}… (+${canonical.length - cap} more chars)`
}

/** The model-facing text of a result's content blocks, non-text blocks named by type. */
function resultText(content: readonly ContentBlock[]): string {
  return content.map(block => block.type === 'text' ? block.text : `[${block.type} content]`).join('\n')
}

/**
 * Validate a threshold list per the fail-loud contract and return it sorted
 * ascending (the escalation rule reads element 0 as the gentle tier, so order
 * is normalized here, once). `field` names the config field in errors;
 * `allowEmpty` admits `[]` for the opt-in same-result list.
 */
function validateThresholds(values: number[], field: string, allowEmpty: boolean): number[] {
  if (values.length === 0 && !allowEmpty) {
    throw new Error(`repeat-tool-reminder: \`${field}\` must not be empty`)
  }
  for (const value of values) {
    if (!Number.isInteger(value) || value < 2) {
      throw new Error(`repeat-tool-reminder: invalid ${field} entry ${value} — every threshold must be an integer >= 2`)
    }
  }
  if (new Set(values).size !== values.length) {
    throw new Error(`repeat-tool-reminder: \`${field}\` must not contain duplicates`)
  }
  return [...values].sort((a, b) => a - b)
}

/** Validate a positive-integer config field per the fail-loud contract. */
function validatePositive(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`repeat-tool-reminder: invalid ${field} ${value} — must be an integer >= 1`)
  }
  return value
}

/**
 * Prepend the guard's reminder while preserving every downstream context's
 * source and metadata.
 */
function prependContext(ours: UserMessage, theirs: UserMessage[] | undefined): UserMessage[] {
  return [ours, ...theirs ?? []]
}

/** One agent's consecutive-repeat chain: the last tracked call's identity key and its run length. */
interface Chain {
  key: string
  count: number
}

/**
 * One agent's same-result history: the result keys of its most recent
 * result-tracked calls (oldest first, at most `resultWindow`), and per key the
 * highest same-result threshold already reminded, so a count that dips and
 * climbs back as the window slides reminds only once per threshold.
 */
interface ResultHistory {
  keys: string[]
  reminded: Map<string, number>
}

/**
 * Install the guard's listeners.
 * @param ctx - plugin context; listeners are scoped to it and disposed with it.
 * @param config - validated {@link Config}; thresholds and sizes are re-checked fail-loud here.
 */
export function apply(ctx: Context, config: Config): void {
  // schemastery's .default() guarantees the fields are set after validation.
  const thresholds = validateThresholds(config.thresholds as number[], 'thresholds', false)
  const thresholdSet = new Set(thresholds)
  const includePatterns = (config.include as string[]).map(wildcardToRegExp)
  const excludePatterns = (config.exclude as string[]).map(wildcardToRegExp)
  const argumentsPreviewChars = validatePositive(config.argumentsPreviewChars as number, 'argumentsPreviewChars')
  const resultThresholds = validateThresholds(config.resultThresholds as number[], 'resultThresholds', true)
  const resultThresholdSet = new Set(resultThresholds)
  const resultIncludePatterns = (config.resultInclude as string[]).map(wildcardToRegExp)
  const resultWindow = validatePositive(config.resultWindow as number, 'resultWindow')
  const resultPreviewChars = validatePositive(config.resultPreviewChars as number, 'resultPreviewChars')
  const reminderSuffix = config.reminderSuffix as string
  const investigationThresholds = new Set(validateThresholds(config.investigationThresholds as number[], 'investigationThresholds', true))
  const investigationResetTools = new Set(config.investigationResetTools as string[])
  const investigationCounts = new WeakMap<Agent, number>()
  const highestResultThreshold = resultThresholds.at(-1)
  if (highestResultThreshold !== undefined && highestResultThreshold > resultWindow) {
    throw new Error(`repeat-tool-reminder: resultWindow ${resultWindow} is smaller than resultThresholds entry ${highestResultThreshold}, which could never be reached`)
  }

  const chains = new WeakMap<Agent, Chain>()
  const histories = new WeakMap<Agent, ResultHistory>()

  /** Whether a tool participates in the chain (untracked calls are transparent: they neither count nor reset). */
  function tracked(toolName: string): boolean {
    if (includePatterns.length > 0 && !includePatterns.some(pattern => pattern.test(toolName))) return false
    return !excludePatterns.some(pattern => pattern.test(toolName))
  }

  /** Whether a tracked tool's results are compared: same-result detection is on and the tool matches `resultInclude`. */
  function resultTracked(toolName: string): boolean {
    if (resultThresholds.length === 0) return false
    return resultIncludePatterns.length === 0 || resultIncludePatterns.some(pattern => pattern.test(toolName))
  }

  /** A reminder message: the text plus the configured suffix, under the guard's notice source. */
  function reminderMessage(text: string, summary: string): UserMessage {
    return createUserMessage({
      content: [{ type: 'text', text: reminderSuffix === '' ? text : `${text}\n\n${reminderSuffix}` }],
      source: { ...REMINDER_SOURCE, form: 'notice', summary },
    })
  }

  /** Advance the agent's consecutive-repeat chain and return a reminder if this run length hits a threshold. */
  function observeArguments(agent: Agent, exec: ToolExecution): UserMessage | undefined {
    const canonical = canonicalize(exec.arguments)
    const key = JSON.stringify([exec.name, canonical])
    const chain = chains.get(agent)
    const count = chain !== undefined && chain.key === key ? chain.count + 1 : 1
    chains.set(agent, { key, count })
    if (!thresholdSet.has(count)) return undefined
    const text = count === thresholds[0]
      ? GENTLE_REMINDER
      : detailedReminder(exec.name, count, previewArguments(canonical, argumentsPreviewChars))
    return reminderMessage(text, `${exec.name} × ${count}`)
  }

  /**
   * Record this call's result in the agent's same-result history and return a
   * reminder if the number of equal results of this tool in the window reaches
   * a threshold not yet reminded for that result.
   */
  function observeResult(agent: Agent, exec: ToolExecution, result: Readonly<ToolExecutionResult>): UserMessage | undefined {
    if (!resultTracked(exec.name)) return undefined
    // Key on the full blocks (two different images are different results); quote only their text rendering.
    const key = JSON.stringify([exec.name, result.content])
    let history = histories.get(agent)
    if (history === undefined) {
      history = { keys: [], reminded: new Map<string, number>() }
      histories.set(agent, history)
    }
    history.keys.push(key)
    if (history.keys.length > resultWindow) history.keys.shift()
    const count = history.keys.filter(k => k === key).length
    if (!resultThresholdSet.has(count) || count <= (history.reminded.get(key) ?? 0)) return undefined
    history.reminded.set(key, count)
    const body = count === resultThresholds[0]
      ? GENTLE_RESULT_REMINDER
      : detailedResultReminder(exec.name, count, history.keys.length, previewArguments(resultText(result.content), resultPreviewChars))
    return reminderMessage(body, `${exec.name} same result × ${count}`)
  }

  /**
   * Advance both detectors for one attempt and return the reminder to
   * deliver. Counting happens here — in post-execute — because denied calls
   * also flow through this waterfall (`ToolRuntime.execute` routes a deny
   * through the same pipeline), and a model hammering a denied call is exactly
   * the loop worth breaking. When both detectors fire on one call, the
   * exact-repeat reminder wins: it names the repeated arguments.
   */
  function observe(exec: ToolExecution, result: Readonly<ToolExecutionResult>): UserMessage | undefined {
    // A direct `ctx.tools.execute()` caller has no model to remind and no id
    // to key on; only agent-loop calls participate.
    if (!exec.agent) return undefined
    if (!tracked(exec.name)) return undefined
    const successful = !result.isError
    if (successful && investigationResetTools.has(exec.name)) investigationCounts.delete(exec.agent)
    else investigationCounts.set(exec.agent, (investigationCounts.get(exec.agent) ?? 0) + 1)
    const byArguments = observeArguments(exec.agent, exec)
    const byResult = observeResult(exec.agent, exec, result)
    const count = investigationCounts.get(exec.agent) ?? 0
    const investigation = investigationThresholds.has(count)
      ? reminderMessage(`You have made ${count} tool calls since a successful file mutation. Consolidate the facts already verified, identify the unresolved question, and check an authoritative reference or a runnable assertion before continuing. This is an advisory checkpoint, not evidence that the investigation is wrong.`, `Investigation checkpoint × ${count}`)
      : undefined
    return byArguments ?? byResult ?? investigation
  }

  // Observe-and-enrich, never veto: count first (state advances regardless of
  // the downstream outcome), DELEGATE so a later listener can still block or
  // replace, then fold the reminder onto whatever came back — additionalContexts
  // rides both decision variants, so a blocked call still gets the nudge.
  ctx.on('tools/post-execute', async (exec, result, next): Promise<PostToolDecision> => {
    const reminder = observe(exec, result)
    const downstream = await next()
    if (!reminder) return downstream
    if (downstream.kind === 'block') {
      return { kind: 'block', feedback: downstream.feedback, additionalContexts: prependContext(reminder, downstream.additionalContexts) }
    }
    return {
      ...downstream,
      additionalContexts: prependContext(reminder, downstream.additionalContexts),
    }
  })

  // A user interjection changes the context; repetition across it is not a
  // loop. Pure reset hook: always delegates (attaching nothing, vetoing
  // nothing).
  ctx.on('agent/pre-step', ({ agent, messages }, next): Promise<PreStepDecision> => {
    if (messages.some(message => message.source.kind === 'user')) {
      chains.delete(agent)
      histories.delete(agent)
      investigationCounts.delete(agent)
    }
    return next()
  })
}
