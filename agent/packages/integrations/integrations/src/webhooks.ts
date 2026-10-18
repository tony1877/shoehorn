/**
 * Inbound webhooks configured as data. Each endpoint is `POST /hooks/<id>`,
 * authenticated with its own secret (an HMAC-SHA256 signature of the body, or
 * the secret as a bearer token for senders that cannot sign), rate limited,
 * and bounded in size. A delivery that matches the endpoint's filter starts a
 * chat with the prompt filled from the body, or continues the endpoint's chat.
 * Deliveries are logged (kept across restarts) and can be run again.
 */

import { createHmac, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Context } from '@deepseek-ai/cordis'
import {
  WebhookDeliveryId,
  WebhookRuleId,
  WebhookSourceId,
  webhookSource,
  type VerifiedWebhookDelivery,
} from '@deepseek-ai/dsh-webhook'
import { snapshotJsonValue, type JsonValue } from '@deepseek-ai/dsh-util-values'
import { findChat, sendToChat, startChat } from './chats.ts'
import { modifySecrets, readSecrets } from './secrets.ts'
import { StateFile } from './state.ts'
import type { WebhookDeliveryView, WebhookOutcome, WebhookPreset, WebhookSettings } from './types.ts'

/** Path prefix every endpoint lives under. */
export const HOOK_PREFIX = '/hooks/'

/** Deliveries the log keeps. */
const LOG_LIMIT = 100

/** Body bytes kept per logged delivery for display. */
const PREVIEW_BYTES = 2_000

/** Body bytes kept per logged delivery for "Run again"; larger bodies cannot be replayed after a restart. */
const REPLAY_BYTES = 64 * 1024

/** Longest value a placeholder expands to, so one field cannot flood the prompt. */
const PLACEHOLDER_LIMIT = 20_000

/** Defaults every new endpoint starts from. */
export const WEBHOOK_DEFAULTS: Omit<WebhookSettings, 'id' | 'workspacePath'> = {
  enabled: true,
  title: '',
  authMode: 'hmac',
  signatureHeader: 'X-Signature-256',
  chatMode: 'new',
  matchPath: '',
  matchValues: [],
  promptTemplate: 'A webhook delivered this payload:\n\n{{body}}\n\nHandle it.',
  titleTemplate: '',
  agentPreset: 'minimal-compact',
  permissionPreset: 'workspace-write',
  maxBodyBytes: 256_000,
  ratePerMinute: 30,
}

/** Ready-made endpoints. */
export const WEBHOOK_PRESETS: readonly WebhookPreset[] = [
  {
    id: 'github-issues',
    title: 'GitHub: new issues',
    description: 'In the repository: Settings → Webhooks → add this URL, content type application/json, this secret, event "Issues". The agent triages each new issue.',
    settings: {
      ...WEBHOOK_DEFAULTS,
      title: 'GitHub issues',
      signatureHeader: 'X-Hub-Signature-256',
      matchPath: 'action',
      matchValues: ['opened'],
      titleTemplate: 'Issue #{{issue.number}}: {{issue.title}}',
      promptTemplate: 'A new issue was opened in {{repository.full_name}} by {{issue.user.login}}:\n\n# {{issue.title}}\n\n{{issue.body}}\n\nLink: {{issue.html_url}}\n\nRead the relevant code in this folder, then write a short triage: what the issue is about, where in the code it likely lives, and a suggested next step. Do not change any files.',
    },
  },
  {
    id: 'github-ci',
    title: 'GitHub: failed CI runs',
    description: 'Event "Workflow runs". Starts a chat when a workflow run completes with a failure, to find the cause.',
    settings: {
      ...WEBHOOK_DEFAULTS,
      title: 'GitHub CI failures',
      signatureHeader: 'X-Hub-Signature-256',
      matchPath: 'workflow_run.conclusion',
      matchValues: ['failure'],
      titleTemplate: 'CI failed: {{workflow_run.name}} on {{workflow_run.head_branch}}',
      promptTemplate: 'The workflow "{{workflow_run.name}}" failed on branch {{workflow_run.head_branch}} (commit {{workflow_run.head_sha}}) in {{repository.full_name}}.\nRun: {{workflow_run.html_url}}\n\nFind the likely cause from the code in this folder and the recent changes, and propose a fix. Do not push anything.',
    },
  },
  {
    id: 'home-assistant',
    title: 'Home Assistant / Zapier / IFTTT',
    description: 'For senders that cannot sign requests: they send the secret as "Authorization: Bearer <secret>" or ?token=<secret>. Every delivery continues one chat.',
    settings: {
      ...WEBHOOK_DEFAULTS,
      title: 'Automation events',
      authMode: 'token',
      signatureHeader: 'Authorization',
      chatMode: 'same',
      promptTemplate: 'Event received:\n\n{{body}}',
    },
  },
  {
    id: 'custom',
    title: 'Custom (signed JSON)',
    description: 'Any sender that can sign the raw body with HMAC-SHA256 and send "sha256=<hex>" in X-Signature-256.',
    settings: { ...WEBHOOK_DEFAULTS },
  },
]

/** Read a dot path (`a.b.0.c`) from parsed JSON. */
export function readPath(value: unknown, path: string): unknown {
  if (path === '') return value
  let current: unknown = value
  for (const segment of path.split('.')) {
    if (current === null || typeof current !== 'object') return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return current
}

function stringify(value: unknown): string {
  if (value === undefined || value === null) return ''
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  return text.length > PLACEHOLDER_LIMIT ? `${text.slice(0, PLACEHOLDER_LIMIT)}\n[cut at ${PLACEHOLDER_LIMIT} characters]` : text
}

/** Fill `{{path}}` placeholders; `{{body}}` is the whole payload. */
export function fillTemplate(template: string, body: unknown): string {
  return template.replace(/\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}/g, (_match, path: string) =>
    stringify(path === 'body' ? body : readPath(body, path)))
}

/** Whether a delivery passes the endpoint's filter. */
export function matches(settings: WebhookSettings, body: unknown): boolean {
  if (settings.matchPath === '') return true
  const value = readPath(body, settings.matchPath)
  if (value === undefined || value === null) return false
  if (settings.matchValues.length === 0) return true
  return settings.matchValues.includes(typeof value === 'string' ? value : JSON.stringify(value))
}

/** Expected signature header value for a body. */
export function sign(secret: string, body: Buffer): string {
  return `sha256=${createHmac('sha256', secret).update(body).digest('hex')}`
}

function sameSecret(a: string, b: string): boolean {
  const left = Buffer.from(a)
  const right = Buffer.from(b)
  return left.length === right.length && timingSafeEqual(left, right)
}

/**
 * Check one request against an endpoint's secret.
 * @returns `undefined` when authentic, else why not (for the log).
 */
export function authenticate(settings: WebhookSettings, secret: string, body: Buffer, headers: IncomingMessage['headersDistinct'], url: string): string | undefined {
  const header = headers[settings.signatureHeader.toLowerCase()]?.[0]
  if (settings.authMode === 'token') {
    const query = new URL(url, 'http://host').searchParams.get('token')
    const given = header !== undefined ? header.replace(/^Bearer\s+/i, '') : query
    if (given === null || given === undefined) return `missing ${settings.signatureHeader} header or token parameter`
    return sameSecret(given, secret) ? undefined : 'token does not match'
  }
  if (header === undefined) return `missing ${settings.signatureHeader} header`
  return sameSecret(header.startsWith('sha256=') ? header : `sha256=${header}`, sign(secret, body)) ? undefined : 'signature does not match'
}

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
  }
}

async function readBody(request: IncomingMessage, limit: number): Promise<Buffer> {
  const declared = Number(request.headers['content-length'])
  if (Number.isFinite(declared) && declared > limit) {
    request.resume()
    throw new HttpError(413, 'request body is too large')
  }
  const chunks: Buffer[] = []
  let size = 0
  for await (const raw of request) {
    const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as string)
    size += chunk.byteLength
    if (size > limit) {
      request.resume()
      throw new HttpError(413, 'request body is too large')
    }
    chunks.push(chunk)
  }
  return Buffer.concat(chunks, size)
}

interface PersistedDelivery extends WebhookDeliveryView {
  /** Body for replay, base64; absent when it was larger than REPLAY_BYTES. */
  body?: string
}

interface WebhookState {
  deliveries: PersistedDelivery[]
  /** webhook id -> the chat "same chat" deliveries continue */
  chats: Record<string, { sessionId: string; agentPreset: string }>
}

/** Owner of the `/hooks/` route, the delivery log, and endpoint secrets. */
export class WebhookEndpoints {
  private settings: readonly WebhookSettings[] = []
  private readonly state: StateFile<WebhookState>
  private readonly hits = new Map<string, number[]>()
  private readonly controller = new AbortController()
  /** Context carrying the session-creation services; set while they are present. */
  private sessionCtx: Context | undefined

  constructor(private readonly ctx: Context, stateDirectory?: string) {
    this.state = new StateFile<WebhookState>('webhooks.json', { deliveries: [], chats: {} }, stateDirectory)
    ctx.effect(() => () => this.controller.abort(), 'integrations.webhooks.abort')
  }

  /** Install the route and take the session services (call inside their injection). */
  mount(webCtx: Context): void {
    this.sessionCtx = webCtx
    webCtx.effect(() => () => {
      this.sessionCtx = undefined
    }, 'integrations.webhooks.services')
    webCtx.effect(() => webCtx.webServer.register({
      kind: 'prefix',
      // Prefix routes are registered without the trailing slash; they match `/hooks/<id>`.
      path: HOOK_PREFIX.slice(0, -1),
      handler: (request, response) => this.handle(request, response),
    }), 'integrations.webhooks.route')
  }

  apply(settings: readonly WebhookSettings[]): void {
    this.settings = settings
  }

  deliveries(): WebhookDeliveryView[] {
    return this.state.get().deliveries.map(({ body: _body, ...view }) => view)
  }

  /** Generate, store, and return a new secret for one endpoint. */
  async rotateSecret(id: string): Promise<string> {
    const secret = randomBytes(24).toString('base64url')
    await modifySecrets(this.ctx, 'hook', id, current => ({ ...current, secret }))
    return secret
  }

  /** Forget an endpoint's chat binding and log entries (it was removed). */
  async forget(id: string): Promise<void> {
    await this.state.update(current => {
      const chats = { ...current.chats }
      delete chats[id]
      return { deliveries: current.deliveries.filter(entry => entry.webhookId !== id), chats }
    })
  }

  /** Run a logged delivery's body through its endpoint again. */
  async replay(deliveryId: string): Promise<WebhookDeliveryView> {
    const original = this.state.get().deliveries.find(entry => entry.id === deliveryId)
    if (original === undefined) throw new Error(`no logged delivery "${deliveryId}"`)
    if (original.body === undefined) throw new Error('this delivery\'s body was too large to keep, so it cannot be run again')
    const settings = this.settings.find(row => row.id === original.webhookId)
    if (settings === undefined) throw new Error(`webhook "${original.webhookId}" no longer exists`)
    const id = await this.record(settings.id, Buffer.from(original.body, 'base64'), 202, 'ignored', 'run again from the log')
    return await this.process(settings, id, Buffer.from(original.body, 'base64'))
  }

  private async record(webhookId: string, body: Buffer, status: number, outcome: WebhookOutcome, detail: string): Promise<string> {
    const entry: PersistedDelivery = {
      id: randomUUID(),
      webhookId,
      receivedAt: Date.now(),
      outcome,
      status,
      detail,
      bodyPreview: body.subarray(0, PREVIEW_BYTES).toString('utf8'),
      ...body.byteLength > 0 && body.byteLength <= REPLAY_BYTES ? { body: body.toString('base64') } : {},
    }
    await this.state.update(current => ({ ...current, deliveries: [entry, ...current.deliveries].slice(0, LOG_LIMIT) }))
    return entry.id
  }

  private async settle(id: string, patch: Partial<PersistedDelivery>): Promise<WebhookDeliveryView> {
    await this.state.update(current => ({
      ...current,
      deliveries: current.deliveries.map(entry => (entry.id === id ? { ...entry, ...patch } : entry)),
    }))
    const { body: _body, ...view } = this.state.get().deliveries.find(entry => entry.id === id)!
    return view
  }

  private limited(settings: WebhookSettings): boolean {
    const now = Date.now()
    const recent = (this.hits.get(settings.id) ?? []).filter(at => now - at < 60_000)
    if (recent.length >= settings.ratePerMinute) {
      this.hits.set(settings.id, recent)
      return true
    }
    recent.push(now)
    this.hits.set(settings.id, recent)
    return false
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const id = decodeURIComponent((request.url ?? '').split('?')[0]!.slice(HOOK_PREFIX.length))
    const settings = this.settings.find(row => row.id === id && row.enabled)
    const reply = (status: number, message: string): void => {
      response.writeHead(status, { 'content-type': 'text/plain; charset=utf-8' })
      response.end(message)
    }
    if (settings === undefined) return reply(404, 'no such webhook')
    if (request.method !== 'POST') {
      response.setHeader('allow', 'POST')
      return reply(405, 'method not allowed')
    }
    let body: Buffer
    try {
      body = await readBody(request, settings.maxBodyBytes)
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 400
      const message = error instanceof Error ? error.message : 'unreadable body'
      await this.record(settings.id, Buffer.alloc(0), status, 'rejected', message)
      return reply(status, message)
    }
    if (this.limited(settings)) {
      await this.record(settings.id, body, 429, 'rejected', 'rate limit reached')
      return reply(429, 'too many deliveries; try again in a minute')
    }
    const secret = (await readSecrets(this.ctx, 'hook', settings.id)).secret
    if (secret === undefined) {
      await this.record(settings.id, body, 503, 'rejected', 'no secret is set for this webhook')
      return reply(503, 'webhook is not ready')
    }
    const refused = authenticate(settings, secret, body, request.headersDistinct, request.url ?? '')
    if (refused !== undefined) {
      // A failed check is logged without the body: it may not be the sender's data at all.
      await this.record(settings.id, Buffer.alloc(0), 401, 'rejected', refused)
      return reply(401, settings.authMode === 'token' ? 'invalid token' : 'invalid signature')
    }
    const deliveryId = await this.record(settings.id, body, 202, 'ignored', 'accepted')
    reply(202, 'accepted')
    void this.process(settings, deliveryId, body)
  }

  /** Match, fill, and start or continue the chat for one accepted delivery; settles its log entry. */
  private async process(settings: WebhookSettings, deliveryId: string, body: Buffer): Promise<WebhookDeliveryView> {
    const text = body.toString('utf8')
    let parsed: unknown = text
    try {
      parsed = JSON.parse(text)
    } catch {
      // Non-JSON bodies are passed through as text; path filters then match nothing.
    }
    if (!matches(settings, parsed)) {
      return this.settle(deliveryId, { outcome: 'ignored', detail: `filter ${settings.matchPath} did not match` })
    }
    const sessionCtx = this.sessionCtx
    if (sessionCtx === undefined) return this.settle(deliveryId, { outcome: 'failed', detail: 'chat creation is unavailable on this Host' })
    const delivery: VerifiedWebhookDelivery = {
      kind: 'integrations',
      source: WebhookSourceId(settings.id),
      deliveryId: WebhookDeliveryId(deliveryId),
      event: (snapshotJsonValue(parsed as JsonValue) ?? text) as JsonValue,
      receivedAt: Date.now(),
    }
    const source = webhookSource(delivery, WebhookRuleId(`integrations:${settings.id}`))
    const prompt = fillTemplate(settings.promptTemplate, parsed)
    if (prompt.trim() === '') return this.settle(deliveryId, { outcome: 'failed', detail: 'the prompt template produced an empty prompt' })
    try {
      if (settings.chatMode === 'same') {
        const bound = this.state.get().chats[settings.id]
        const agent = bound === undefined ? undefined : await findChat(sessionCtx, bound.sessionId, bound.agentPreset)
        if (agent !== undefined && bound !== undefined) {
          sendToChat(agent, [{ type: 'text', text: prompt }], source)
          return await this.settle(deliveryId, { outcome: 'started', detail: 'sent to the endpoint\'s chat', sessionId: bound.sessionId })
        }
      }
      const title = fillTemplate(settings.titleTemplate, parsed).trim() || settings.title.trim() || `Webhook ${settings.id}`
      const created = await startChat(sessionCtx, settings, title, [{ type: 'text', text: prompt }], source, this.controller.signal)
      if (settings.chatMode === 'same') {
        await this.state.update(current => ({ ...current, chats: { ...current.chats, [settings.id]: { sessionId: String(created.sessionId), agentPreset: settings.agentPreset } } }))
      }
      return await this.settle(deliveryId, { outcome: 'started', detail: 'chat started', sessionId: String(created.sessionId) })
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      this.ctx.logger.warn(`integrations: webhook ${settings.id} could not start a chat: ${detail}`)
      return this.settle(deliveryId, { outcome: 'failed', detail })
    }
  }
}
