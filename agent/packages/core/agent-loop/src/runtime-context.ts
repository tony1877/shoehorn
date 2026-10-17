/**
 * Durable projection state for dynamic runtime context.
 * @module @deepseek-ai/dsh-agent-loop/runtime-context
 */

import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContextSnapshotSection } from '@deepseek-ai/dsh-llm'
import type { Session, UserMessage } from '@deepseek-ai/dsh-session'
import { isReplacementSurfaceEvent, SessionSeq } from '@deepseek-ai/dsh-session'
import type { Context } from '@deepseek-ai/cordis'

const SOURCE = '@deepseek-ai/dsh-system-prompt'
const CLEARED = 'Current runtime context: none. Earlier runtime-context snapshots no longer apply.'

function isOwned(message: UserMessage): boolean {
  return message.source.kind === 'plugin' && message.source.plugin === SOURCE
}

/**
 * Sections whose change alone does not justify a new snapshot in the middle of a turn: the memory index changes when
 * the agent saves a memory itself, and a whole snapshot appended mid-turn (as a user message) for that only
 * distracts it. The next turn's first step brings the snapshot up to date.
 */
const DEFERRABLE: ReadonlySet<string> = new Set(['memory:index'])

function sectionsOf(message: UserMessage): readonly ContextSnapshotSection[] | undefined {
  const sections = (message.source as { sections?: unknown }).sections
  return Array.isArray(sections) ? sections as readonly ContextSnapshotSection[] : undefined
}

function textOf(message: UserMessage): string | undefined {
  const [block] = message.content
  return message.content.length === 1 && block?.type === 'text' ? block.text : undefined
}

/** Tracks the last retained runtime-context snapshot without owning its commit. */
export class RuntimeContextProjection {
  /** `undefined` means no snapshot ever existed; `null` means none is retained. */
  private retained: {
    seq: SessionSeq
    text: string | undefined
    sections: readonly ContextSnapshotSection[] | undefined
  } | null | undefined

  /**
   * Restore projection state once, then follow authoritative session events.
   * @param ctx - agent-scoped event context.
   * @param session - session receiving projected messages.
   */
  constructor(ctx: Context, session: Session) {
    const surface = new Set(session.surface.nodes)
    for (let index = session.seq - 1; index >= 0; index -= 1) {
      const event = session.eventAt(SessionSeq(index))
      if (event?.type !== 'user/message' || !isOwned(event.data)) continue
      this.retained ??= null
      if (surface.has(event.seq)) {
        this.retained = { seq: event.seq, text: textOf(event.data), sections: sectionsOf(event.data) }
        break
      }
    }

    ctx.on('session/event', (subject, event) => {
      if (subject !== session) return
      if (event.type === 'user/message' && isOwned(event.data)) {
        this.retained = { seq: event.seq, text: textOf(event.data), sections: sectionsOf(event.data) }
      } else if (this.retained
        && isReplacementSurfaceEvent(event)
        && event.sourceEventSeqs?.includes(this.retained.seq) === true) {
        this.retained = null
      }
    })
  }

  /**
   * Create an uncommitted snapshot only when the retained value differs.
   * @param current - fully rendered dynamic context.
   * @param sections - named contributions that formed the current snapshot.
   * @param midTurn - a later step of a running turn: changes of deferrable sections alone wait for the next turn.
   * @returns a candidate user message, or `undefined` when no update is needed.
   */
  project(current: string, sections: readonly ContextSnapshotSection[], midTurn = false): UserMessage | undefined {
    if (this.retained === undefined && current.length === 0) return
    const snapshot = current.length === 0 ? CLEARED : current
    if (this.retained?.text === snapshot) return
    const kept = this.retained?.sections
    if (midTurn && kept !== undefined && current.length > 0) {
      const before = new Map(kept.map(section => [section.name, section.text]))
      const after = new Map(sections.map(section => [section.name, section.text]))
      const changed = [...new Set([...before.keys(), ...after.keys()])]
        .filter(name => before.get(name) !== after.get(name))
      if (changed.length > 0 && changed.every(name => DEFERRABLE.has(name))) return
    }
    return createUserMessage({
      content: [{ type: 'text', text: snapshot }],
      // The cleared marker has no contributions left to attribute.
      source: sections.length === 0
        ? { kind: 'plugin', plugin: SOURCE }
        : { kind: 'plugin', plugin: SOURCE, form: 'snapshot', sections },
    })
  }
}
