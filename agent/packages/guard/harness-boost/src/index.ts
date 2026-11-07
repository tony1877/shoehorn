/**
 * shoehorn harness boost: harness-side help for a small local model, each part switched by its own config key so it
 * can be measured on its own (tests/agent_eval).
 *
 * - verify: when a turn is about to end, block it (at most `maxContinuations` times per user turn) if files were
 *   edited after the last test/build run, if the last run failed while the final message claims success, or if the
 *   task list still has open items.
 * - todoRestore: after a compaction, put the open task list back into context.
 * - effort: after a failure signal (failed edit, failed command or check), run the next steps at a higher
 *   reasoning effort. The effort reaches the server as the request's budget; the prompt (and its cache) is unchanged.
 * - recall: at the start of a user turn, attach the memories that match the prompt; after a failed call, attach
 *   lessons that match the error.
 * - lessons: when a failed call is later retried successfully, have the summarizer route write a short lesson and
 *   save it as a `feedback` memory tagged `lesson`. Lessons shown before a failure that was not fixed lose standing.
 * - skillTriggers: when a failed result matches a skill's front-matter trigger, attach that skill section.
 *
 * @module @deepseek-ai/dsh-harness-boost
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent, PreStepDecision } from '@deepseek-ai/dsh-agent'
import { BlockAssembler, ReasoningEffortId, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, ContextFormed, LlmCallConfig } from '@deepseek-ai/dsh-llm'
import type { Session, UserMessage } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-session-projection'
import type { PostToolDecision, ToolExecution, ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import { GLOBAL_SCOPE, MemoryStore, ftsQuery, normalizeScope } from '@deepseek-ai/dsh-tool-memory'
import type { Memory } from '@deepseek-ai/dsh-tool-memory'
import {
  argsPreview, asksUser, attemptKey, changesExisting, checkFailed, claimsSuccess, disclosesTestChange, editedPath,
  exitCode, isCheckCommand, isTestPath, shellCommand, specTask, verdictLine,
} from './signals.ts'
import { loadTriggers, type SkillTrigger } from './skills.ts'
import { NotesStore, applyUpdate, renderNotes, type Notes } from './notes.ts'
import { buildIndex, codeFileCount, findDefinitions, findReferences, renderMap, renderSymbol } from './repomap.ts'
import { defineTool } from '@deepseek-ai/dsh-tools'

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'harness-boost': { kind: 'harness-boost' } & ContextFormed
  }
}

export { parseTriggers, sectionText } from './skills.ts'
export * as signals from './signals.ts'

export const name = 'harness-boost'
export const inject = ['tools', 'sessionProjections', 'llm', 'systemPrompt']

export interface Config {
  /** The `notes` tool, its runtime-context view, the goal check before finishing, and nudges to keep notes. */
  notes?: { enabled?: boolean; nudgeEvery?: number }
  /** The `repo_map` / `find_symbol` tools and the large-workspace hint. */
  repoMap?: { enabled?: boolean; largeRepoFiles?: number }
  verify?: boolean
  /** Before a spec task may finish, have the agent check its work against the spec rule by rule (once per turn). */
  specCheck?: boolean
  maxContinuations?: number
  todoRestore?: boolean
  effort?: { enabled?: boolean; level?: string; steps?: number }
  recall?: { enabled?: boolean; topK?: number; onError?: boolean }
  lessons?: { enabled?: boolean; provider?: string; model?: string; maxPerTurn?: number }
  skillTriggers?: boolean
  /** Memory database (default: the tool-memory default, `$DSH_HOME/memory/memory.db`). */
  memoryPath?: string
}

export const Config: z<Config> = z.object({
  verify: z.boolean().default(true),
  specCheck: z.boolean().default(true),
  maxContinuations: z.number().step(1).min(0).default(3),
  notes: z.object({
    enabled: z.boolean().default(true),
    nudgeEvery: z.number().step(1).min(4).default(12),
  }).default({}),
  repoMap: z.object({
    enabled: z.boolean().default(true),
    largeRepoFiles: z.number().step(1).min(0).default(150),
  }).default({}),
  todoRestore: z.boolean().default(true),
  effort: z.object({
    enabled: z.boolean().default(true),
    level: z.string().default('xhigh'),
    steps: z.number().step(1).min(1).default(2),
  }).default({}),
  recall: z.object({
    enabled: z.boolean().default(true),
    topK: z.number().step(1).min(1).default(3),
    onError: z.boolean().default(true),
  }).default({}),
  lessons: z.object({
    enabled: z.boolean().default(false),
    provider: z.string().default(''),
    model: z.string().default(''),
    maxPerTurn: z.number().step(1).min(1).default(2),
  }).default({}),
  skillTriggers: z.boolean().default(true),
  memoryPath: z.string(),
})

/** A failed call waiting for a successful retry (a lesson candidate). */
interface Failure {
  readonly key: string
  readonly tool: string
  readonly args: string
  readonly error: string
  /** Lessons attached to this failure, credited when the retry succeeds. */
  readonly shown: number[]
  /** Successful file edits in the session when it failed: a retry after more edits fixed the code, not the call. */
  readonly editsAtFailure: number
}

/** A failure that a later call fixed. */
interface Episode extends Failure {
  readonly fixArgs: string
  readonly fixResult: string
}

interface State {
  edited: Set<string>
  lastCheck: { failed: boolean; verdict: string } | undefined
  continuations: number
  lastReason: string | undefined
  finalText: string
  restoreTodos: boolean
  boost: number
  failures: Map<string, Failure>
  episodes: Episode[]
  /** Memories already put in this session's context. */
  shownMemories: Set<number>
  /** Skill sections already attached in this session. */
  shownSections: Set<string>
  /** Successful file edits so far. */
  edits: number
  /** Lessons written from this session. */
  lessonsWritten: number
  /** Implementation (non-test) edits since the last failed check. */
  implEditsSinceFailure: number
  /** Test files changed right after a failed check, before any implementation edit: in this user turn. */
  weakenedTests: Set<string>
  /** The user prompt that opened this turn. */
  prompt: string
  /** User turns so far (the turn a goal was recorded in is compared with it). */
  turn: number
  /** Tool calls since the notes were last updated, and notes nudges given, in this turn. */
  callsSinceNotes: number
  notesNudges: number
  goalNudged: boolean
  /** The open-goal check already ran in this turn. */
  goalChecked: boolean
  /** File edits when the turn opened. */
  editsAtTurnStart: number
  /** The spec check already ran in this turn. */
  specChecked: boolean
}

/**
 * Full-text search ranks anything sharing one word, so a recalled memory must share this many distinct
 * meaningful words (5+ letters, not common filler) with the prompt or error text.
 */
const MIN_SHARED_TERMS = 3

/** Lessons one session may write. */
const MAX_LESSONS_PER_SESSION = 2
const FILLER = new Set(['about', 'after', 'again', 'before', 'being', 'could', 'every', 'first', 'their', 'there',
  'these', 'thing', 'those', 'through', 'under', 'until', 'which', 'while', 'would', 'should', 'other', 'where',
  'tests', 'files', 'change', 'changed', 'report', 'result', 'workspace', 'python', 'unittest', 'under'])

function terms(text: string): Set<string> {
  return new Set((text.toLowerCase().match(/[a-z_][a-z0-9_]{4,}/g) ?? []).filter(w => !FILLER.has(w)))
}

/** Distinct meaningful words two texts share. */
export function sharedTerms(a: string, b: string): number {
  const right = terms(b)
  let n = 0
  for (const w of terms(a)) if (right.has(w)) n += 1
  return n
}

const DSH_HOME = (): string => process.env.DSH_HOME?.trim() || join(homedir(), '.dsh')

/** A harness notice the model sees as a user-role context message. */
function notice(text: string, summary: string): UserMessage {
  return createUserMessage({
    content: [{ type: 'text', text: `[harness] ${text}` }],
    source: { kind: 'harness-boost', form: 'notice', summary },
  })
}

function resultText(content: readonly ContentBlock[]): string {
  return content.map(block => block.type === 'text' ? block.text : '').join('\n')
}

/** Lesson usefulness, kept beside the memory database: how often a lesson was shown and how often a fix followed. */
class LessonStats {
  private stats: Record<string, { shown: number; helped: number }> = {}

  constructor(private readonly file: string) {
    try {
      if (existsSync(file)) this.stats = JSON.parse(readFileSync(file, 'utf8')) as typeof this.stats
    } catch {
      this.stats = {}
    }
  }

  private save(): void {
    try {
      mkdirSync(dirname(this.file), { recursive: true })
      writeFileSync(this.file, JSON.stringify(this.stats, null, 1))
    } catch {
      // statistics are best effort
    }
  }

  private entry(id: number): { shown: number; helped: number } {
    return this.stats[id] ??= { shown: 0, helped: 0 }
  }

  shown(ids: readonly number[]): void {
    for (const id of ids) this.entry(id).shown += 1
    if (ids.length > 0) this.save()
  }

  helped(ids: readonly number[]): void {
    for (const id of ids) this.entry(id).helped += 1
    if (ids.length > 0) this.save()
  }

  /** Shown at least 5 times and followed by a fix less than one time in five. */
  demoted(id: number): boolean {
    const s = this.stats[id]
    return s !== undefined && s.shown >= 5 && s.helped / s.shown < 0.2
  }
}

export function apply(ctx: Context, config: Config): void {
  // DSH_HARNESS_BOOST=0 turns every part off (the A/B control arm in tests/agent_eval)
  if (process.env.DSH_HARNESS_BOOST === '0') return
  const states = new WeakMap<Session, State>()
  const state = (session: Session): State => {
    let s = states.get(session)
    if (s === undefined) {
      s = {
        edited: new Set(), lastCheck: undefined, continuations: 0, lastReason: undefined, finalText: '', restoreTodos: false, boost: 0,
        failures: new Map(), episodes: [], shownMemories: new Set(), shownSections: new Set(), edits: 0, lessonsWritten: 0, implEditsSinceFailure: 0, weakenedTests: new Set(), prompt: '', editsAtTurnStart: 0, specChecked: false, turn: 0, callsSinceNotes: 0, notesNudges: 0, goalNudged: false, goalChecked: false,
      }
      states.set(session, s)
    }
    return s
  }

  const memoryPath = config.memoryPath?.trim() || join(DSH_HOME(), 'memory', 'memory.db')
  let store: MemoryStore | undefined
  const memories = (): MemoryStore | undefined => {
    try {
      return store ??= new MemoryStore(memoryPath)
    } catch (error: unknown) {
      ctx.logger.warn(`harness-boost: memory store unavailable: ${String(error)}`)
      return undefined
    }
  }
  ctx.effect(() => () => { store?.close() }, 'harness-boost.memory')
  const stats = new LessonStats(join(dirname(memoryPath), 'lesson-stats.json'))
  const triggers: SkillTrigger[] = config.skillTriggers === false ? [] : loadTriggers([join(DSH_HOME(), 'skills')])
  const notesOn = config.notes?.enabled !== false
  const notesStore = new NotesStore(join(DSH_HOME(), 'harness-boost', 'notes'))
  const sessionKey = (session: Session): string => String(session.id)
  const textOutput = { schema: { type: 'string' as const }, render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }] }

  if (notesOn) {
    ctx.systemPrompt.context({
      name: 'harness:notes',
      order: 135,
      text: (context) => {
        const session = context.agent?.session
        if (session === undefined) return ''
        const body = renderNotes(notesStore.get(sessionKey(session)))
        return body === '' ? '' : `Your working notes (kept for you across steps and compaction; update them with the notes tool):\n${body}`
      },
    })
    ctx.tools.register(defineTool({
      name: 'notes',
      description: 'Your working notes, kept by the harness and shown back to you at the start of every turn and after '
        + 'the context is compacted, so the plan and what you learned are not lost in a long task. Record what the '
        + 'user asked as a goal checklist (one item per requirement, in their words) before you change files; keep '
        + 'the plan current; add decisions with their reason and findings (verified facts, dead ends you ruled out). '
        + 'Mark a goal item done only with evidence: the command and the result line, the test, or the file:line that '
        + 'shows it. Before finishing, every goal item must be done or explained in your report. Every field is '
        + 'optional; send only what changes. Returns the current notes.',
      parameters: {
        goal: { type: 'array', items: { type: 'string' }, description: 'Replace the goal checklist: what the user asked, one requirement per item.' },
        done: {
          type: 'array',
          description: 'Mark goal items done: item number (1-based) and the evidence that shows it.',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              n: { type: 'integer', required: true, description: 'Goal item number, 1-based.' },
              evidence: { type: 'string', required: true, description: 'Command and result, test name, or file:line showing the item is done.' },
            },
          },
        },
        undone: { type: 'array', items: { type: 'integer' }, description: 'Goal item numbers to reopen.' },
        plan: { type: 'string', description: 'Replace the plan: the next steps, short.' },
        decision: { type: 'string', description: 'Append one decision and why.' },
        finding: { type: 'string', description: 'Append one verified fact or ruled-out approach (so it is not re-checked).' },
      },
      output: textOutput,
      execute: async (args, exec) => {
        const session = exec.agent?.session
        if (session === undefined) throw new Error('notes: no agent session')
        const notes = notesStore.get(sessionKey(session))
        const problems = applyUpdate(notes, args as Parameters<typeof applyUpdate>[1], state(session).turn)
        notesStore.save(sessionKey(session))
        state(session).callsSinceNotes = 0
        const body = renderNotes(notes) || '(notes are empty)'
        return problems.length === 0 ? body : `${problems.map(p => `Not applied: ${p}`).join('\n')}\n\n${body}`
      },
      presentCall: args => ({ card: 'generic', kind: 'edit', title: 'Update notes', rawInput: JSON.stringify(args).slice(0, 200) }),
    }))
  }

  if (config.repoMap?.enabled !== false) {
    const rootOf = (exec: ToolExecution): string => exec.agent?.session.header.cwd ?? process.cwd()
    ctx.systemPrompt.context({
      name: 'harness:repo',
      order: 136,
      text: (context) => {
        const cwd = context.agent?.session.header.cwd
        if (cwd === undefined) return ''
        const n = codeFileCount(cwd)
        return n < (config.repoMap?.largeRepoFiles ?? 150) ? ''
          : `This workspace is large (${n} code files). To find code, use repo_map (overview; path=DIR for one directory `
            + 'file by file) and find_symbol NAME (definitions and uses) before reading files; read only the regions they point to.'
      },
    })
    ctx.tools.register(defineTool({
      name: 'repo_map',
      description: 'Map of the workspace code: directories with file counts and their main symbols, or with path one '
        + 'directory file by file with the functions, classes and types of each file. Use it to find where things are '
        + 'before reading files in a large codebase. Cached; refreshed for changed files.',
      parameters: {
        path: { type: 'string', description: 'Directory relative to the workspace to show file by file. Omit for the overview.' },
      },
      output: textOutput,
      isConcurrencySafe: () => true,
      execute: async (args, exec) => renderMap(buildIndex(rootOf(exec)), args.path),
      presentCall: args => ({ card: 'generic', kind: 'search', title: 'Repository map', rawInput: args.path ?? '.' }),
    }))
    ctx.tools.register(defineTool({
      name: 'find_symbol',
      description: 'Where a function, class, method or type is defined (file:line and the defining line) and where its '
        + 'name is used (whole-word matches). Faster and more precise than grepping for a definition.',
      parameters: {
        name: { type: 'string', required: true, description: 'Symbol name, e.g. parse_config or Parser.parse_expr.' },
      },
      output: textOutput,
      isConcurrencySafe: () => true,
      execute: async (args, exec) => {
        const root = rootOf(exec)
        const index = buildIndex(root)
        const tail = args.name.split(/[.:]+/).pop() ?? args.name
        return renderSymbol(args.name, findDefinitions(index, args.name), await findReferences(root, tail))
      },
      presentCall: args => ({ card: 'generic', kind: 'search', title: 'Find symbol', rawInput: args.name }),
    }))
  }

  const scopesOf = (agent: Agent): string[] => {
    const cwd = agent.session.header.cwd
    return cwd === undefined ? [GLOBAL_SCOPE] : [GLOBAL_SCOPE, normalizeScope(cwd)]
  }

  /** Memories matching `text`, not demoted, not yet shown in this session. */
  const recall = (agent: Agent, text: string, limit: number, lessonsOnly: boolean): Memory[] => {
    const db = memories()
    if (db === undefined || ftsQuery(text) === undefined) return []
    const s = state(agent.session)
    try {
      return db.search(text.slice(0, 2000), { scopes: scopesOf(agent), limit: limit * 4, ...lessonsOnly ? { kind: 'feedback' as const } : {} })
        .filter(m => !s.shownMemories.has(m.id) && !stats.demoted(m.id) && (!lessonsOnly || m.tags.includes('lesson'))
          && sharedTerms(text, `${m.title} ${m.body} ${m.tags.join(' ')}`) >= MIN_SHARED_TERMS)
        .slice(0, limit)
    } catch {
      return []
    }
  }

  const renderMemories = (list: readonly Memory[]): string =>
    list.map(m => `- #${m.id} [${m.kind}] ${m.title}\n  ${m.body.replace(/\n/g, '\n  ')}`).join('\n')

  // ---- observe the session: final text, compactions
  ctx.on('session/event', (session, event) => {
    if (event.type === 'assistant/message') {
      const text = event.data.message.content.filter(b => b.type === 'text').map(b => (b as { text: string }).text).join('\n').trim()
      if (text.length > 0) state(session).finalText = text
    } else if ((event.type as string) === 'compaction/summary') {
      state(session).restoreTodos = true
    }
  })

  // ---- tool results: edits, checks, failures, recall on error, skill triggers, lesson episodes
  ctx.on('tools/post-execute', async (exec: ToolExecution, result: Readonly<ToolExecutionResult>, next): Promise<PostToolDecision> => {
    const downstream = await next()
    const agent = exec.agent
    if (agent === undefined) return downstream
    const s = state(agent.session)
    const text = resultText(result.content)
    const command = shellCommand(exec.name, exec.arguments)
    const path = editedPath(exec.name, exec.arguments)
    const isCheck = command !== undefined && isCheckCommand(command)
    const failed = result.isError === true
      || (command !== undefined && (isCheck ? checkFailed(text) : exitCode(text) !== 0))

    const contexts: UserMessage[] = []
    if (path !== undefined && !failed) {
      s.edited.add(path)
      s.edits += 1
      // Test weakening: a test changed right after a run failed, with no implementation change in between, is
      // usually the model making its test agree with broken code.
      if (!isTestPath(path)) {
        s.implEditsSinceFailure += 1
      } else if (s.lastCheck?.failed === true && s.implEditsSinceFailure === 0 && changesExisting(exec.name, exec.arguments)
        && !s.weakenedTests.has(path)) {
        s.weakenedTests.add(path)
        const file = path.split(/[\\/]/).pop()
        contexts.push(notice(`You changed a test (${file}) right after a test run failed, without changing any implementation `
          + 'file since. If the failing test states what the task or spec requires, the code is wrong: undo this test '
          + 'change and fix the implementation. Change a test only when the test itself contradicts the spec, and say '
          + 'which test you changed and why in your final report.', 'test changed after failure'))
      }
    }
    if (isCheck) {
      s.lastCheck = { failed, verdict: verdictLine(text) }
      if (failed) s.implEditsSinceFailure = 0
      else s.edited.clear()
    }

    const key = attemptKey(exec.name, exec.arguments)
    if (failed) {
      if (config.effort?.enabled !== false) s.boost = config.effort?.steps ?? 2
      const shownIds: number[] = []
      if (config.recall?.enabled !== false && config.recall?.onError !== false) {
        const hits = recall(agent, `${exec.name} ${text.slice(-600)}`, 2, true)
        if (hits.length > 0) {
          for (const m of hits) s.shownMemories.add(m.id)
          shownIds.push(...hits.map(m => m.id))
          stats.shown(shownIds)
          contexts.push(notice(`Lessons from earlier sessions that match this failure:\n${renderMemories(hits)}`, `lessons × ${hits.length}`))
        }
      }
      for (const trigger of triggers) {
        const id = `${trigger.skill}#${trigger.heading}`
        if (s.shownSections.has(id) || !trigger.pattern.test(text)) continue
        s.shownSections.add(id)
        contexts.push(notice(`This failure matches the \`${trigger.skill}\` skill. The relevant section:\n\n${trigger.text}`, `skill ${trigger.skill}`))
        break
      }
      if (key !== undefined && !s.failures.has(key)) {
        s.failures.set(key, { key, tool: exec.name, args: argsPreview(exec.arguments), error: text.slice(-800), shown: shownIds, editsAtFailure: s.edits })
      }
    } else if (key !== undefined && s.failures.has(key)) {
      const failure = s.failures.get(key)!
      s.failures.delete(key)
      stats.helped(failure.shown)
      // A lesson is about using a tool, so the fix must be a different call: an edit retried on the same file, or a
      // changed command with no file edits in between (a check that passes after code edits taught nothing).
      const fixArgs = argsPreview(exec.arguments)
      const toolFix = path !== undefined ? s.edits === failure.editsAtFailure + 1 : s.edits === failure.editsAtFailure && fixArgs !== failure.args
      if (toolFix) s.episodes.push({ ...failure, fixArgs, fixResult: text.slice(-300) })
    }

    if (notesOn && exec.name !== 'notes') {
      const notes = notesStore.get(sessionKey(agent.session))
      s.callsSinceNotes += 1
      const substantial = s.prompt.length >= 60 || specTask(s.prompt)
      if (path !== undefined && !failed && substantial && !s.goalNudged && (notes.goal.length === 0 || notes.goalTurn < s.turn)) {
        s.goalNudged = true
        contexts.push(notice('Record what the user asked before you change more: call notes with goal = one item per '
          + 'requirement, in their words (including details like exact formats, limits and edge cases they named). If the '
          + 'goal already in your notes still describes this request, send it again unchanged. The goal is your checklist '
          + 'for "done".', 'notes: record goal'))
      } else if (s.callsSinceNotes >= (config.notes?.nudgeEvery ?? 12) && s.notesNudges < 3) {
        s.notesNudges += 1
        s.callsSinceNotes = 0
        contexts.push(notice('Update your notes (notes tool): the plan as it stands now, and findings: what you verified '
          + 'and what you ruled out, so you do not re-check it later.', 'notes: update'))
      }
    }

    if (contexts.length === 0) return downstream
    return { ...downstream, additionalContexts: [...contexts, ...downstream.additionalContexts ?? []] } as PostToolDecision
  })

  // ---- turn start: reset per-turn state, recall memories for the prompt; after a compaction: restore todos
  ctx.on('agent/pre-step', async ({ agent, messages }, next): Promise<PreStepDecision> => {
    const decision = await next()
    if (decision.kind !== 'enter') return decision
    const s = state(agent.session)
    const extra: UserMessage[] = []
    const prompt = messages.filter(m => m.source.kind === 'user')
      .map(m => m.content.filter(b => b.type === 'text').map(b => (b as { text: string }).text).join('\n')).join('\n')
    if (prompt.length > 0 && !prompt.trimStart().startsWith('[harness]')) {
      s.continuations = 0
      s.lastReason = undefined
      s.edited.clear()
      s.lastCheck = undefined
      s.failures.clear()
      s.weakenedTests.clear()
      s.implEditsSinceFailure = 0
      s.prompt = prompt
      s.editsAtTurnStart = s.edits
      s.specChecked = false
      s.turn += 1
      s.callsSinceNotes = 0
      s.notesNudges = 0
      s.goalNudged = false
      s.goalChecked = false
      if (config.recall?.enabled !== false) {
        const hits = recall(agent, prompt, config.recall?.topK ?? 3, false)
        if (hits.length > 0) {
          for (const m of hits) s.shownMemories.add(m.id)
          extra.push(notice(`Memories from earlier sessions that may apply to this task (verify before relying on them):\n${renderMemories(hits)}`, `recall × ${hits.length}`))
        }
      }
    }
    if (s.restoreTodos && config.todoRestore !== false) {
      s.restoreTodos = false
      const todos = ctx.sessionProjections.stateOf(agent.session, 'todos' as never) as { content: string; status: string }[] | null | undefined
      const open = (todos ?? []).filter(t => t.status !== 'completed')
      if (open.length > 0) {
        const list = (todos ?? []).map(t => `- [${t.status}] ${t.content}`).join('\n')
        extra.push(notice(`Your task list, restored after the context was compacted (${open.length} open):\n${list}`, 'todos restored'))
      }
    }
    return extra.length === 0 ? decision : { ...decision, messages: [...decision.messages, ...extra] }
  })

  // ---- adaptive effort: the next steps after a failure think harder
  ctx.on('agent/request', async ({ agent }, next): Promise<LlmCallConfig> => {
    const proposed = await next()
    const s = state(agent.session)
    if (s.boost <= 0 || config.effort?.enabled === false) return proposed
    s.boost -= 1
    return { ...proposed, reasoningEffort: ReasoningEffortId(config.effort?.level ?? 'xhigh') }
  })

  // ---- verify guard and lesson writing at the end of a turn
  ctx.on('agent/turn-stopping', async ({ agent, signal }) => {
    const s = state(agent.session)
    if (config.verify !== false && s.continuations < (config.maxContinuations ?? 2)) {
      const reason = verifyReason(ctx, agent, s, config.specCheck !== false, notesOn ? notesStore.get(sessionKey(agent.session)) : undefined)
      // the same reason twice in a row means the model already answered it (e.g. "nothing to run"): let it stop
      if (reason !== undefined && reason !== s.lastReason) {
        s.lastReason = reason
        s.continuations += 1
        agent.steer(notice(reason, 'verify'))
        return
      }
    }
    if (config.lessons?.enabled === true && s.episodes.length > 0 && s.lessonsWritten < MAX_LESSONS_PER_SESSION) {
      const episodes = s.episodes.splice(0, Math.min(config.lessons.maxPerTurn ?? 2, MAX_LESSONS_PER_SESSION - s.lessonsWritten))
      s.episodes.length = 0
      s.lessonsWritten += episodes.length
      void writeLessons(ctx, config, agent, episodes, memories, signal).catch((error: unknown) => {
        ctx.logger.warn(`harness-boost: lesson writing failed: ${String(error)}`)
      })
    }
  })
}

/** Why the turn must not end yet, or undefined when it may. */
function verifyReason(ctx: Context, agent: Agent, s: State, specCheck: boolean, notes?: Notes): string | undefined {
  const report = 'Then write your COMPLETE final report again from the start: only that last message is shown as your answer.'
  if (s.lastCheck?.failed === true && s.edited.size === 0 && claimsSuccess(s.finalText)) {
    return `Your last test/build run FAILED (${s.lastCheck.verdict || 'non-zero exit'}), but your final message says the work passes or is fixed. `
      + `Fix the failure and run it again, or state plainly that it still fails and why. ${report}`
  }
  if (s.edited.size > 0) {
    const files = [...s.edited].slice(-5).map(p => p.split(/[\\/]/).pop()).join(', ')
    const last = s.lastCheck === undefined ? 'You did not run any test or build in this turn' : 'You edited files after your last test or build run'
    return `${last} (edited: ${files}). Run the tests or build that cover these files now and read the result. `
      + `If there is nothing to run, say plainly in the report that the change is untested. ${report}`
  }
  if (s.weakenedTests.size > 0 && !disclosesTestChange(s.finalText)) {
    const files = [...s.weakenedTests].map(p => p.split(/[\\/]/).pop()).join(', ')
    return `You changed tests after they failed (${files}), and your final message does not say so. Check each changed `
      + 'assertion against the task and spec: if the original expectation was right, restore it and fix the code. '
      + `Your report must list every test you changed and why. ${report}`
  }
  const todos = ctx.sessionProjections.stateOf(agent.session, 'todos' as never) as { content: string; status: string }[] | null | undefined
  const open = (todos ?? []).filter(t => t.status !== 'completed')
  if (open.length > 0 && !asksUser(s.finalText)) {
    return `Your task list still has ${open.length} open item(s): ${open.map(t => `"${t.content}"`).join(', ')}. `
      + `Finish them, or update the list if they are done or no longer needed, and say so. ${report}`
  }
  // A spec task that passes its visible tests is not done yet: runs that stop here miss the spec's edge cases
  // (tests/agent_eval: boundaries, whitespace, literals, error cases). Once per turn.
  if (notes !== undefined && !s.goalChecked && !asksUser(s.finalText)) {
    const open = notes.goal.map((g, i) => ({ g, n: i + 1 })).filter(({ g }) => !g.done)
    if (open.length > 0) {
      s.goalChecked = true
      return `Your goal list still has ${open.length} open item(s): ${open.map(({ g, n }) => `${n}. ${g.text}`).join('; ')}. `
        + 'For each: finish it, or mark it done with notes(done=[{n, evidence}]) citing the command and result, test or '
        + `file:line that shows it, or say plainly in your report that it is not done and why. ${report}`
    }
  }
  if (specCheck && !s.specChecked && specTask(s.prompt) && s.edits > s.editsAtTurnStart && s.lastCheck?.failed === false) {
    s.specChecked = true
    return 'Before you finish: this task states required behaviour (a spec, a docstring, or the task text itself), and '
      + 'the visible tests cover only a little of it. Go through every stated rule one by one: every bullet, every example, and the edge cases each rule '
      + 'implies (exact boundaries, empty input, extra whitespace, every literal and keyword, every error case). For '
      + 'each rule, name the test or the check you ran that exercises it. For any rule nothing exercises yet, add a '
      + 'test or run a check now, and fix the code where it fails. Do not change the spec or the given tests. '
      + `${report} Include a short list of the rules you checked.`
  }
  return undefined
}

const LESSON_PROMPT = `You write short lessons for an AI coding agent, from one episode where a tool call failed and a later call fixed it.
Write the lesson only if it would help in a DIFFERENT future task: a tool, language, shell or environment pitfall and how to avoid it. Do not write lessons about this task's specific bug, file names or values.
Reply with exactly one JSON object and nothing else:
{"title": "<one-line hook, under 80 chars>", "lesson": "<1-3 sentences: when this happens, do this>", "scope": "global" | "workspace"}
or the single word NONE when there is no reusable lesson.`

/** Ask the summarizer route for a lesson per episode and save the useful ones as `lesson` memories. */
async function writeLessons(
  ctx: Context,
  config: Config,
  agent: Agent,
  episodes: readonly Episode[],
  memories: () => MemoryStore | undefined,
  signal: AbortSignal,
): Promise<void> {
  const provider = config.lessons?.provider ?? ''
  const model = config.lessons?.model ?? ''
  const db = memories()
  if (provider === '' || model === '' || db === undefined) return
  for (const episode of episodes) {
    const user = [
      `Failed call: ${episode.tool} ${episode.args}`,
      `Its result:\n${episode.error}`,
      `The call that worked: ${episode.tool} ${episode.fixArgs}`,
      `Its result:\n${episode.fixResult}`,
    ].join('\n\n')
    const assembler = new BlockAssembler()
    for await (const chunk of ctx.llm.stream({
      provider, model, system: LESSON_PROMPT, maxTokens: 400, signal,
      messages: [{ role: 'user', content: [{ type: 'text', text: user }] }],
    })) {
      assembler.push(chunk)
    }
    const text = assembler.blocks().filter(b => b.type === 'text').map(b => (b as { text: string }).text).join('').trim()
    const json = /\{[\s\S]*\}/.exec(text)?.[0]
    if (json === undefined) continue
    let parsed: { title?: unknown; lesson?: unknown; scope?: unknown }
    try {
      parsed = JSON.parse(json) as typeof parsed
    } catch {
      continue
    }
    if (typeof parsed.title !== 'string' || typeof parsed.lesson !== 'string') continue
    const scope = parsed.scope === 'workspace' && agent.session.header.cwd !== undefined ? agent.session.header.cwd : GLOBAL_SCOPE
    // an existing lesson on the same subject (or one contradicting it) wins: lessons are kept few and settled
    const about = `${parsed.title} ${parsed.lesson}`
    const duplicate = ftsQuery(about) !== undefined
      && db.search(about, { kind: 'feedback', limit: 10 }).some(m => m.tags.includes('lesson') && sharedTerms(about, `${m.title} ${m.body}`) >= 3)
    if (duplicate) continue
    db.save({
      kind: 'feedback',
      scope,
      title: parsed.title.slice(0, 120),
      body: parsed.lesson.slice(0, 600),
      tags: ['lesson', episode.tool],
      sessionId: String(agent.session.id),
    })
  }
}
