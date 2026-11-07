/**
 * Working notes: what the user asked (a goal checklist), the current plan, decisions and findings. The agent writes
 * them with the `notes` tool; the harness shows them back in the runtime context at every turn start and after a
 * compaction, and checks the goal list before the agent may finish. The server drops earlier reasoning and
 * compaction drops detail, so this is where the agent's plan and conclusions survive a long session.
 * @module @deepseek-ai/dsh-harness-boost/notes
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export interface GoalItem {
  text: string
  done: boolean
  /** What shows it is done: a command and its result, a test name, a file:line. */
  evidence?: string
}

export interface Notes {
  goal: GoalItem[]
  plan: string
  decisions: string[]
  findings: string[]
  /** Turn in which the goal list was last replaced. */
  goalTurn: number
  updatedAt: string
}

export const emptyNotes = (): Notes => ({ goal: [], plan: '', decisions: [], findings: [], goalTurn: 0, updatedAt: '' })

/** Most entries kept in each append-only list; older ones fall off (the latest matter most). */
const MAX_LIST = 15
const MAX_ITEM_CHARS = 400

export interface NotesUpdate {
  goal?: string[]
  done?: { n: number; evidence: string }[]
  undone?: number[]
  plan?: string
  decision?: string
  finding?: string
}

const clip = (s: string): string => (s.length > MAX_ITEM_CHARS ? `${s.slice(0, MAX_ITEM_CHARS)}…` : s).trim()

/**
 * Apply one update. Errors (a `done` number that does not exist, evidence missing) are returned, not thrown, so the
 * rest of the update still lands.
 * @returns the problems found, empty when everything applied.
 */
export function applyUpdate(notes: Notes, update: NotesUpdate, turn: number): string[] {
  const problems: string[] = []
  if (update.goal !== undefined) {
    const items = update.goal.map(clip).filter(t => t.length > 0)
    // keep the done state of items that are repeated word for word
    const before = new Map(notes.goal.map(g => [g.text, g]))
    notes.goal = items.map(text => before.get(text) ?? { text, done: false })
    notes.goalTurn = turn
  }
  for (const d of update.done ?? []) {
    const item = notes.goal[d.n - 1]
    if (item === undefined) problems.push(`done: there is no goal item ${d.n} (the goal has ${notes.goal.length} items)`)
    else if (d.evidence === undefined || d.evidence.trim().length === 0) problems.push(`done: item ${d.n} needs evidence`)
    else {
      item.done = true
      item.evidence = clip(d.evidence)
    }
  }
  for (const n of update.undone ?? []) {
    const item = notes.goal[n - 1]
    if (item === undefined) problems.push(`undone: there is no goal item ${n}`)
    else {
      item.done = false
      delete item.evidence
    }
  }
  if (update.plan !== undefined) notes.plan = update.plan.trim().slice(0, 2000)
  if (update.decision !== undefined && update.decision.trim() !== '') notes.decisions = [...notes.decisions, clip(update.decision)].slice(-MAX_LIST)
  if (update.finding !== undefined && update.finding.trim() !== '') notes.findings = [...notes.findings, clip(update.finding)].slice(-MAX_LIST)
  notes.updatedAt = new Date().toISOString()
  return problems
}

/** The notes as the model sees them, or '' when empty. */
export function renderNotes(notes: Notes): string {
  const parts: string[] = []
  if (notes.goal.length > 0) {
    const open = notes.goal.filter(g => !g.done).length
    parts.push(`Goal (what the user asked; ${open} of ${notes.goal.length} open):`)
    notes.goal.forEach((g, i) => parts.push(`  ${i + 1}. [${g.done ? 'x' : ' '}] ${g.text}${g.done && g.evidence ? `  — evidence: ${g.evidence}` : ''}`))
  }
  if (notes.plan !== '') parts.push(`Plan:\n${notes.plan.split('\n').map(l => `  ${l}`).join('\n')}`)
  if (notes.decisions.length > 0) parts.push(`Decisions:\n${notes.decisions.map(d => `  - ${d}`).join('\n')}`)
  if (notes.findings.length > 0) parts.push(`Findings (verified facts, dead ends):\n${notes.findings.map(f => `  - ${f}`).join('\n')}`)
  return parts.join('\n')
}

/** Notes persisted per session under `dir`, so a restart or resume keeps them. */
export class NotesStore {
  private readonly loaded = new Map<string, Notes>()

  constructor(private readonly dir: string) {}

  private file(sessionId: string): string {
    return join(this.dir, `${sessionId.replace(/[^\w.-]/g, '_')}.json`)
  }

  get(sessionId: string): Notes {
    let notes = this.loaded.get(sessionId)
    if (notes === undefined) {
      notes = emptyNotes()
      try {
        const file = this.file(sessionId)
        if (existsSync(file)) notes = { ...notes, ...JSON.parse(readFileSync(file, 'utf8')) as Partial<Notes> }
      } catch {
        // unreadable notes start empty
      }
      this.loaded.set(sessionId, notes)
    }
    return notes
  }

  save(sessionId: string): void {
    const notes = this.loaded.get(sessionId)
    if (notes === undefined) return
    try {
      mkdirSync(this.dir, { recursive: true })
      writeFileSync(this.file(sessionId), JSON.stringify(notes, null, 1))
    } catch {
      // persistence is best effort; the in-memory notes still work
    }
  }
}
