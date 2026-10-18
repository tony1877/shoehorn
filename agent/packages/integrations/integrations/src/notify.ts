/**
 * Outbound notifications: when a top-level chat finishes a turn, waits for an
 * approval, or asks the user a question, POST a message to each subscribed
 * target in the target's format. Sends are signed with the target's id-scoped
 * HMAC when it is a plain JSON hook, retried with backoff, and never block the
 * agent.
 */

import { createHmac } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { readSecrets } from './secrets.ts'
import type { NotificationEvent, NotificationSettings } from './types.ts'

/** Attempts per send, including the first. */
const ATTEMPTS = 3

/** First retry delay; doubles per attempt. */
const RETRY_BASE_MS = 2_000

/** Per-attempt request timeout. */
const SEND_TIMEOUT_MS = 10_000

/** Longest excerpt of the last answer included in a message. */
const EXCERPT_CHARS = 600

/** What one notification says, before formatting. */
export interface NotificationMessage {
  event: NotificationEvent
  sessionId: string
  title: string
  text: string
  /** Link that opens the chat, when the public base URL is known. */
  link?: string
}

/** Format one message for a target. */
export function formatNotification(format: NotificationSettings['format'], message: NotificationMessage): { body: string; headers: Record<string, string> } {
  const heading = {
    finished: 'Chat finished',
    approval: 'Approval needed',
    question: 'Question for you',
  }[message.event]
  const lines = [`${heading}: ${message.title}`, message.text, message.link].filter((line): line is string => line !== undefined && line !== '')
  switch (format) {
    case 'json':
      return { body: JSON.stringify(message), headers: { 'content-type': 'application/json' } }
    case 'discord':
      return { body: JSON.stringify({ content: lines.join('\n').slice(0, 1_900) }), headers: { 'content-type': 'application/json' } }
    case 'slack':
      return { body: JSON.stringify({ text: lines.join('\n') }), headers: { 'content-type': 'application/json' } }
    case 'ntfy':
      return {
        body: [message.text, message.link].filter(Boolean).join('\n') || heading,
        headers: {
          'content-type': 'text/plain; charset=utf-8',
          // ntfy reads the title from a header; HTTP headers must be Latin-1.
          title: `${heading}: ${message.title}`.replace(/[^\x20-\x7e]/g, '?').slice(0, 200),
          tags: message.event === 'finished' ? 'white_check_mark' : 'warning',
          ...message.link === undefined ? {} : { click: message.link },
        },
      }
  }
}

function excerpt(agent: Agent): string {
  const events = agent.session.snapshotEvents()
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'assistant/message') continue
    const text = event.data.message.content
      .map(block => (block.type === 'text' ? block.text : ''))
      .join('')
      .trim()
    if (text === '') continue
    return text.length > EXCERPT_CHARS ? `${text.slice(0, EXCERPT_CHARS)}…` : text
  }
  return ''
}

/** Owner of the notification listeners and sender. */
export class Notifier {
  private targets: readonly NotificationSettings[] = []
  private readonly status = new Map<string, { lastStatus: string; lastSentAt: number }>()
  private publicBaseUrl = ''

  constructor(private readonly ctx: Context) {
    ctx.on('agent/status', ({ agent, status }) => {
      if (status !== 'idle' || !this.topLevel(agent)) return
      this.emit({ event: 'finished', ...this.describe(agent), text: excerpt(agent) })
    })
    ctx.on('approval/request', async (request, next) => {
      if (request.agent !== undefined && this.topLevel(request.agent)) {
        this.emit({ event: 'approval', ...this.describe(request.agent), text: `${request.toolName}${request.reason === undefined ? '' : ` — ${request.reason}`}` })
      }
      return next()
    })
    ctx.on('user-questions/request', async (request, next) => {
      if (request.agent !== undefined && this.topLevel(request.agent)) {
        this.emit({ event: 'question', ...this.describe(request.agent), text: request.questions.map(question => question.question).join('\n') })
      }
      return next()
    })
  }

  apply(targets: readonly NotificationSettings[], publicBaseUrl: string): void {
    this.targets = targets
    this.publicBaseUrl = publicBaseUrl
  }

  get list(): readonly NotificationSettings[] {
    return this.targets
  }

  lastStatus(id: string): { lastStatus: string; lastSentAt: number } | undefined {
    return this.status.get(id)
  }

  /** Send a sample message to one target now, once, and report the result. */
  async test(id: string): Promise<{ ok: boolean; detail: string }> {
    const target = this.targets.find(row => row.id === id)
    if (target === undefined) return { ok: false, detail: `no notification target "${id}"` }
    try {
      const detail = await this.send(target, { event: 'finished', sessionId: 'test', title: 'Test notification', text: 'Notifications from the agent reach this target.' }, 1)
      return { ok: true, detail }
    } catch (error) {
      return { ok: false, detail: error instanceof Error ? error.message : String(error) }
    }
  }

  private topLevel(agent: Agent): boolean {
    return agent.session.header.parentSession === undefined
  }

  private describe(agent: Agent): { sessionId: string; title: string; link?: string } {
    const sessionId = String(agent.session.id)
    const title = this.ctx.get('sessionTitle')?.get(agent.session)?.title ?? 'Untitled chat'
    const base = this.publicBaseUrl.replace(/\/+$/, '')
    return { sessionId, title, ...base === '' ? {} : { link: `${base}/?session=${encodeURIComponent(sessionId)}` } }
  }

  private emit(message: NotificationMessage): void {
    for (const target of this.targets) {
      if (!target.enabled || !target.events.includes(message.event)) continue
      void this.send(target, message, ATTEMPTS).catch((error: unknown) => {
        this.ctx.logger.warn(`integrations: notification to ${target.id} failed: ${error instanceof Error ? error.message : String(error)}`)
      })
    }
  }

  private async send(target: NotificationSettings, message: NotificationMessage, attempts: number): Promise<string> {
    const secrets = await readSecrets(this.ctx, 'notify', target.id)
    if (secrets.url === undefined) throw new Error('no URL is set for this target')
    const { body, headers } = formatNotification(target.format, message)
    if (target.format === 'json' && secrets.secret !== undefined) {
      headers['x-signature-256'] = `sha256=${createHmac('sha256', secrets.secret).update(body).digest('hex')}`
    }
    let lastError: unknown
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      try {
        const response = await fetch(secrets.url, { method: 'POST', headers, body, signal: AbortSignal.timeout(SEND_TIMEOUT_MS) })
        if (response.ok) {
          const detail = `HTTP ${response.status}`
          this.status.set(target.id, { lastStatus: detail, lastSentAt: Date.now() })
          return detail
        }
        lastError = new Error(`HTTP ${response.status} ${response.statusText}`)
        // A client error will not fix itself on retry, except rate limiting.
        if (response.status < 500 && response.status !== 429) break
      } catch (error) {
        lastError = error
      }
      if (attempt < attempts) await new Promise(resolve => setTimeout(resolve, RETRY_BASE_MS * 2 ** (attempt - 1)).unref())
    }
    const detail = lastError instanceof Error ? lastError.message : String(lastError)
    this.status.set(target.id, { lastStatus: `failed: ${detail}`, lastSentAt: Date.now() })
    throw new Error(detail)
  }
}
