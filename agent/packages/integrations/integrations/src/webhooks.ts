/**
 * Inbound webhooks configured as data. Each endpoint is `POST /hooks/<id>`,
 * signed with its own HMAC-SHA256 secret, rate limited, and bounded in size.
 * A delivery that matches the endpoint's filter starts a chat through the
 * webhook runtime's session creation, with the prompt filled from the body.
 * Every delivery lands in an in-memory log the settings page shows.
 */

import { createHmac, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Context } from '@deepseek-ai/cordis'
import {
  createWebhookSession,
  WebhookDeliveryId,
  WebhookRuleId,
  WebhookSourceId,
  type VerifiedWebhookDelivery,
} from '@deepseek-ai/dsh-webhook'
import { snapshotJsonValue, type JsonValue } from '@deepseek-ai/dsh-util-values'
import { modifySecrets, readSecrets } from './secrets.ts'
import type { WebhookDeliveryView, WebhookOutcome, WebhookSettings } from './types.ts'

/** Path prefix every endpoint lives under. */
export const HOOK_PREFIX = '/hooks/'

/** Deliveries the log keeps. */
const LOG_LIMIT = 100

/** Body bytes kept per logged delivery for display. */
const PREVIEW_BYTES = 2_000

/** Longest value a placeholder expands to, so one field cannot flood the prompt. */
const PLACEHOLDER_LIMIT = 20_000

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
  if (value === undefined) return ''
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

function signatureValid(secret: string, body: Buffer, header: string | undefined): boolean {
  if (header === undefined) return false
  const expected = Buffer.from(sign(secret, body))
  const given = Buffer.from(header.startsWith('sha256=') ? header : `sha256=${header}`)
  return expected.length === given.length && timingSafeEqual(expected, given)
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

interface StoredDelivery extends WebhookDeliveryView {
  body: Buffer
}

/** Owner of the `/hooks/` route, the delivery log, and endpoint secrets. */
export class WebhookEndpoints {
  private settings: readonly WebhookSettings[] = []
  private readonly log: StoredDelivery[] = []
  private readonly hits = new Map<string, number[]>()
  private readonly controller = new AbortController()
  /** Context carrying the session-creation services; set while they are present. */
  private sessionCtx: Context | undefined

  constructor(private readonly ctx: Context) {
    ctx.effect(() => () => this.controller.abort(), 'integrations.webhooks.abort')
  }

  /** Install the route on the web server (call inside an injection of `webServer`). */
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

  get endpoints(): readonly WebhookSettings[] {
    return this.settings
  }

  deliveries(): WebhookDeliveryView[] {
    return this.log.map(({ body: _body, ...view }) => view)
  }

  /** Generate, store, and return a new secret for one endpoint. */
  async rotateSecret(id: string): Promise<string> {
    const secret = randomBytes(24).toString('base64url')
    await modifySecrets(this.ctx, 'hook', id, current => ({ ...current, secret }))
    return secret
  }

  /** Run a logged delivery's body through its endpoint again. */
  async replay(deliveryId: string): Promise<WebhookDeliveryView> {
    const original = this.log.find(entry => entry.id === deliveryId)
    if (original === undefined) throw new Error(`no logged delivery "${deliveryId}"`)
    const settings = this.settings.find(row => row.id === original.webhookId)
    if (settings === undefined) throw new Error(`webhook "${original.webhookId}" no longer exists`)
    const entry = this.record(settings.id, original.body, 202, 'ignored', 'replayed')
    await this.process(settings, entry)
    return this.view(entry)
  }

  private view(entry: StoredDelivery): WebhookDeliveryView {
    const { body: _body, ...view } = entry
    return view
  }

  private record(webhookId: string, body: Buffer, status: number, outcome: WebhookOutcome, detail: string): StoredDelivery {
    const entry: StoredDelivery = {
      id: randomUUID(),
      webhookId,
      receivedAt: Date.now(),
      outcome,
      status,
      detail,
      bodyPreview: body.subarray(0, PREVIEW_BYTES).toString('utf8'),
      body,
    }
    this.log.unshift(entry)
    this.log.length = Math.min(this.log.length, LOG_LIMIT)
    return entry
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
      this.record(settings.id, Buffer.alloc(0), status, 'rejected', error instanceof Error ? error.message : 'unreadable body')
      return reply(status, error instanceof Error ? error.message : 'unreadable body')
    }
    if (this.limited(settings)) {
      this.record(settings.id, body, 429, 'rejected', 'rate limit reached')
      return reply(429, 'too many deliveries; try again in a minute')
    }
    const secret = (await readSecrets(this.ctx, 'hook', settings.id)).secret
    if (secret === undefined) {
      this.record(settings.id, body, 503, 'rejected', 'no secret is set for this webhook')
      return reply(503, 'webhook is not ready')
    }
    const header = request.headersDistinct[settings.signatureHeader.toLowerCase()]?.[0]
    if (!signatureValid(secret, body, header)) {
      this.record(settings.id, body, 401, 'rejected', header === undefined ? `missing ${settings.signatureHeader} header` : 'signature does not match')
      return reply(401, 'invalid signature')
    }
    const entry = this.record(settings.id, body, 202, 'ignored', 'accepted')
    reply(202, 'accepted')
    void this.process(settings, entry)
  }

  /** Match, fill, and start the chat for one accepted delivery; updates its log entry. */
  private async process(settings: WebhookSettings, entry: StoredDelivery): Promise<void> {
    const text = entry.body.toString('utf8')
    let parsed: unknown = text
    try {
      parsed = JSON.parse(text)
    } catch {
      // Non-JSON bodies are passed through as text; path filters then match nothing.
    }
    if (!matches(settings, parsed)) {
      entry.outcome = 'ignored'
      entry.detail = `filter ${settings.matchPath} did not match`
      return
    }
    const event = snapshotJsonValue(parsed as JsonValue) ?? text
    const delivery: VerifiedWebhookDelivery = {
      kind: 'integrations',
      source: WebhookSourceId(settings.id),
      deliveryId: WebhookDeliveryId(entry.id),
      event: event as JsonValue,
      receivedAt: entry.receivedAt,
    }
    const sessionCtx = this.sessionCtx
    if (sessionCtx === undefined) {
      entry.outcome = 'failed'
      entry.detail = 'chat creation is unavailable on this Host'
      return
    }
    try {
      const sessionId = await createWebhookSession(sessionCtx, delivery, WebhookRuleId(`integrations:${settings.id}`), {
        workspacePath: settings.workspacePath,
        title: fillTemplate(settings.titleTemplate, parsed).slice(0, 120) || settings.title,
        prompt: fillTemplate(settings.promptTemplate, parsed),
        agentPreset: settings.agentPreset,
        permissionPreset: settings.permissionPreset,
      }, this.controller.signal)
      entry.outcome = 'started'
      entry.detail = 'chat started'
      entry.sessionId = sessionId
    } catch (error) {
      entry.outcome = 'failed'
      entry.detail = error instanceof Error ? error.message : String(error)
      this.ctx.logger.warn(`integrations: webhook ${settings.id} could not start a chat: ${entry.detail}`)
    }
  }
}
