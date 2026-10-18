/**
 * Integrations: MCP servers, inbound webhooks, and outbound notifications,
 * all configured at runtime from the `integrations` settings namespace with
 * their secrets in the credential store, and exposed to the Web settings page
 * through the `integrations` Remote.
 *
 * @module @deepseek-ai/dsh-integrations
 */

import { randomBytes } from 'node:crypto'
import { isAbsolute } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { TypertRemoteService, Remote } from '@deepseek-ai/dsh-typert-protocol'
import type {} from '@deepseek-ai/dsh-credentials'
import type {} from '@deepseek-ai/dsh-host-webserver'
import type {} from '@deepseek-ai/dsh-settings'
import type {} from '@deepseek-ai/dsh-tools'
import type {} from '@deepseek-ai/dsh-user-approval'
import type {} from '@deepseek-ai/dsh-user-questions'
import type {} from '@deepseek-ai/dsh-session-title'
import type { SettingsScope } from '@deepseek-ai/dsh-settings'
// Typert-generated ./typert and ./remote artifacts import Zod at runtime.
import type {} from 'zod'
import { McpManager, assertServerSettings, serverView } from './mcp.ts'
import { Notifier } from './notify.ts'
import { OAUTH_CALLBACK_PATH, OAuthSignIns } from './oauth.ts'
import { MCP_PRESETS, SERVER_DEFAULTS } from './presets.ts'
import { deleteSecrets, modifySecrets, readSecrets } from './secrets.ts'
import { TelegramBot } from './telegram.ts'
import { HOOK_PREFIX, WEBHOOK_DEFAULTS, WEBHOOK_PRESETS, WebhookEndpoints } from './webhooks.ts'
import type {
  AuthorizeServerRequest,
  ChatIdRequest,
  SaveTelegramRequest,
  TelegramPairingResult,
  TelegramSettings,
  AuthorizeServerResult,
  IdRequest,
  IntegrationsOverview,
  McpServerSettings,
  McpTestResult,
  NotificationSettings,
  NotificationView,
  SaveNotificationRequest,
  SaveServerRequest,
  SaveWebhookRequest,
  SetPublicBaseUrlRequest,
  TestNotificationResult,
  TestServerRequest,
  WebhookDeliveryView,
  WebhookSecretResult,
  WebhookSettings,
  WebhookView,
} from './types.ts'

export type * from './types.ts'
export { MCP_PRESETS } from './presets.ts'
export { authenticate, fillTemplate, matches, readPath, sign, WEBHOOK_PRESETS } from './webhooks.ts'
export { markdownToHtml, splitMarkdown } from './telegram-api.ts'
export { sniffImage } from './telegram.ts'
export { formatNotification } from './notify.ts'
export { approvalFor } from './mcp.ts'

/** Settings namespace owned by this plugin. */
const NAMESPACE = 'integrations'

const ID = /^[a-z][a-z0-9-]{0,31}$/

/** Resolved settings value. */
export interface IntegrationsSettings {
  servers: McpServerSettings[]
  webhooks: WebhookSettings[]
  notifications: NotificationSettings[]
  telegram: TelegramSettings
  /** Base URL webhook senders and notification links use to reach this Host. */
  publicBaseUrl: string
}

const ServerSchema: z<McpServerSettings> = z.object({
  id: z.string().required(),
  enabled: z.boolean().default(SERVER_DEFAULTS.enabled),
  transport: z.union(['stdio', 'streamable-http', 'sse'] as const).default(SERVER_DEFAULTS.transport),
  command: z.string().default(''),
  args: z.array(String).default([]),
  cwd: z.string().default(''),
  url: z.string().default(''),
  includeTools: z.array(String).default([]),
  exposure: z.union(['native', 'deferred'] as const).default(SERVER_DEFAULTS.exposure),
  approval: z.union(['ask', 'ask-writes', 'allow'] as const).default(SERVER_DEFAULTS.approval),
  auth: z.union(['none', 'oauth'] as const).default('none'),
  toolCallTimeoutMs: z.number().min(1_000).max(3_600_000).default(SERVER_DEFAULTS.toolCallTimeoutMs),
}) as z<McpServerSettings>

const WebhookSchema: z<WebhookSettings> = z.object({
  id: z.string().required(),
  enabled: z.boolean().default(true),
  title: z.string().default(''),
  authMode: z.union(['hmac', 'token'] as const).default(WEBHOOK_DEFAULTS.authMode),
  signatureHeader: z.string().default(WEBHOOK_DEFAULTS.signatureHeader),
  chatMode: z.union(['new', 'same'] as const).default(WEBHOOK_DEFAULTS.chatMode),
  matchPath: z.string().default(''),
  matchValues: z.array(String).default([]),
  workspacePath: z.string().required(),
  promptTemplate: z.string().required(),
  titleTemplate: z.string().default(''),
  agentPreset: z.string().default('minimal-compact'),
  permissionPreset: z.string().default('workspace-write'),
  maxBodyBytes: z.number().min(1).max(10_000_000).default(256_000),
  ratePerMinute: z.number().min(1).max(10_000).default(30),
}) as z<WebhookSettings>

const NotificationSchema: z<NotificationSettings> = z.object({
  id: z.string().required(),
  enabled: z.boolean().default(true),
  title: z.string().default(''),
  format: z.union(['json', 'discord', 'slack', 'ntfy'] as const).default('json'),
  events: z.array(z.union(['finished', 'approval', 'question'] as const)).default(['finished', 'approval', 'question']),
  minTurnSeconds: z.number().min(0).max(86_400).default(20),
}) as z<NotificationSettings>

const TelegramSchema: z<TelegramSettings> = z.object({
  enabled: z.boolean().default(false),
  allowedChats: z.array(String).default([]),
  workspacePath: z.string().default(''),
  agentPreset: z.string().default('minimal-compact'),
  permissionPreset: z.string().default('workspace-write'),
  answerInTelegram: z.boolean().default(true),
  notifyEvents: z.array(z.union(['finished', 'approval', 'question'] as const)).default(['approval', 'question']),
  minTurnSeconds: z.number().min(0).max(86_400).default(20),
}) as z<TelegramSettings>

export const SettingsSchema: z<IntegrationsSettings> = z.object({
  servers: z.array(ServerSchema).default([]),
  webhooks: z.array(WebhookSchema).default([]),
  notifications: z.array(NotificationSchema).default([]),
  telegram: TelegramSchema,
  publicBaseUrl: z.string().default(''),
}) as z<IntegrationsSettings>

/** Composition config: the deployment's starting rows, below the user's edits. */
export type Config = Partial<IntegrationsSettings> & {
  /** Folder for the bot's and the webhook log's state files; default `$DSH_HOME/integrations`. */
  stateDirectory?: string
  /** Telegram Bot API server; default `https://api.telegram.org` (a self-hosted Bot API server also works). */
  telegramApiBase?: string
}
export const Config: z<Config> = z.object({
  servers: z.array(ServerSchema),
  webhooks: z.array(WebhookSchema),
  notifications: z.array(NotificationSchema),
  telegram: TelegramSchema,
  publicBaseUrl: z.string(),
  stateDirectory: z.string(),
  telegramApiBase: z.string(),
}) as z<Config>

function assertUniqueIds(kind: string, rows: ReadonlyArray<{ id: string }>): void {
  const seen = new Set<string>()
  for (const row of rows) {
    if (!ID.test(row.id)) throw new Error(`${kind} id "${row.id}" must start with a lowercase letter and use only a-z, 0-9 and "-" (at most 32 characters)`)
    if (seen.has(row.id)) throw new Error(`${kind} id "${row.id}" is used twice`)
    seen.add(row.id)
  }
}

function validate(value: IntegrationsSettings): void {
  assertUniqueIds('MCP server', value.servers)
  assertUniqueIds('webhook', value.webhooks)
  assertUniqueIds('notification', value.notifications)
}

function upsert<T extends { id: string }>(rows: readonly T[], row: T): T[] {
  const index = rows.findIndex(existing => existing.id === row.id)
  if (index === -1) return [...rows, row]
  return rows.map((existing, at) => (at === index ? row : existing))
}

function mergeMap(current: Record<string, string> | undefined, next: Record<string, string> | null | undefined): Record<string, string> | undefined {
  if (next === null) return undefined
  if (next === undefined) return current
  // A blank value keeps the stored one, so the page never needs to read secrets back.
  const merged: Record<string, string> = {}
  for (const [name, value] of Object.entries(next)) {
    const kept = value === '' ? current?.[name] : value
    if (kept !== undefined) merged[name] = kept
  }
  return Object.keys(merged).length === 0 ? undefined : merged
}

function oauthPage(title: string, detail: string): string {
  const escape = (text: string): string => text.replace(/[&<>"]/g, ch => `&#${ch.charCodeAt(0)};`)
  return `<!doctype html><meta charset="utf-8"><title>${escape(title)}</title>`
    + `<body style="font:16px system-ui;margin:3rem auto;max-width:32rem;line-height:1.5">`
    + `<h2>${escape(title)}</h2><p>${escape(detail)}</p><p>You can close this tab.</p>`
    + `<script>try{window.opener&&window.opener.postMessage({type:'integrations-oauth'},'*')}catch{}</script></body>`
}

/** Host service behind the integrations settings page. */
export class Integrations extends TypertRemoteService {
  static inject = ['tools', 'credentials']
  static Config = Config

  private scope: SettingsScope<IntegrationsSettings> | undefined
  private composed: IntegrationsSettings
  private readonly mcp: McpManager
  private readonly webhooks: WebhookEndpoints
  private readonly notifier: Notifier
  private readonly signIns: OAuthSignIns
  private readonly telegram: TelegramBot

  constructor(ctx: Context, config: Config = {}) {
    super(ctx, 'integrations')
    const { stateDirectory, telegramApiBase, ...base } = config
    this.composed = SettingsSchema(base as IntegrationsSettings)
    validate(this.composed)
    this.mcp = new McpManager(ctx)
    this.webhooks = new WebhookEndpoints(ctx, stateDirectory)
    this.notifier = new Notifier(ctx)
    this.signIns = new OAuthSignIns(ctx)
    this.telegram = new TelegramBot(ctx, chatId => this.pairTelegramChat(chatId), stateDirectory, telegramApiBase)
    ctx.effect(() => this.notifier.addSink(message => this.telegram.notify(message)), 'integrations.telegram.notify')

    ctx.inject(['settings'], (settingsCtx) => {
      const scope = settingsCtx.settings.register(NAMESPACE, SettingsSchema, { base: base as IntegrationsSettings, validate })
      this.scope = scope
      settingsCtx.effect(() => scope.watch(next => this.applyAll(next)), 'integrations.settings.watch')
      settingsCtx.effect(() => () => {
        this.scope = undefined
      }, 'integrations.settings')
      void this.applyAll(scope.get())
    })

    ctx.inject(['webServer'], (webCtx) => {
      webCtx.effect(() => webCtx.webServer.register({
        kind: 'exact',
        path: OAUTH_CALLBACK_PATH,
        handler: (request, response) => this.oauthCallback(request.url ?? '', response),
      }), 'integrations.oauth.route')
    })

    ctx.inject(['webServer', 'agents', 'agentDefaultModel', 'agentPresets', 'permissionPresets', 'sessionTitle', 'workspaceRegistry'], (hookCtx) => {
      this.webhooks.mount(hookCtx)
      this.telegram.mount(hookCtx)
    })
  }

  protected async [Service.init](): Promise<void> {
    if (this.scope === undefined) await this.applyAll(this.composed)
  }

  private get value(): IntegrationsSettings {
    return this.scope?.get() ?? this.composed
  }

  private async applyAll(value: IntegrationsSettings): Promise<void> {
    this.webhooks.apply(value.webhooks)
    this.notifier.apply(value.notifications, value.publicBaseUrl)
    await this.telegram.apply(value.telegram)
    await this.mcp.apply(value.servers)
  }

  private async pairTelegramChat(chatId: string): Promise<void> {
    const telegram = this.value.telegram
    if (telegram.allowedChats.includes(chatId)) return
    await this.write({ ...this.value, telegram: { ...telegram, allowedChats: [...telegram.allowedChats, chatId] } })
  }

  private async write(next: IntegrationsSettings): Promise<void> {
    validate(next)
    if (this.scope === undefined) throw new Error('settings storage is unavailable; integrations cannot be changed at runtime')
    await this.scope.update(next)
  }

  private async oauthCallback(url: string, response: import('node:http').ServerResponse): Promise<void> {
    const params = new URL(url, 'http://host').searchParams
    const send = (status: number, title: string, detail: string): void => {
      response.writeHead(status, { 'content-type': 'text/html; charset=utf-8' })
      response.end(oauthPage(title, detail))
    }
    const error = params.get('error')
    if (error !== null) return send(400, 'Sign-in was not completed', params.get('error_description') ?? error)
    const code = params.get('code')
    const state = params.get('state')
    if (code === null || state === null) return send(400, 'Sign-in was not completed', 'The response had no authorization code.')
    try {
      const serverId = await this.signIns.finish(state, code)
      await this.mcp.restart(serverId)
      send(200, 'Signed in', `The agent can now use "${serverId}".`)
    } catch (failure) {
      send(400, 'Sign-in failed', failure instanceof Error ? failure.message : String(failure))
    }
  }

  /** @returns everything the settings page renders. */
  @Remote('overview')
  async overview(): Promise<IntegrationsOverview> {
    const value = this.value
    const servers = await Promise.all(value.servers.map(async settings => serverView(settings, await readSecrets(this.ctx, 'mcp', settings.id), this.mcp.statsOf(settings.id))))
    const webhooks: WebhookView[] = await Promise.all(value.webhooks.map(async settings => ({
      settings,
      path: `${HOOK_PREFIX}${settings.id}`,
      secretSet: (await readSecrets(this.ctx, 'hook', settings.id)).secret !== undefined,
    })))
    const notifications: NotificationView[] = await Promise.all(value.notifications.map(async (settings) => {
      const url = (await readSecrets(this.ctx, 'notify', settings.id)).url
      let urlHost: string | undefined
      try {
        if (url !== undefined) urlHost = new URL(url).host
      } catch { /* stored URL is unparsable; show none */ }
      const status = this.notifier.lastStatus(settings.id)
      return {
        settings,
        urlSet: url !== undefined,
        ...urlHost === undefined ? {} : { urlHost },
        ...status === undefined ? {} : status,
      }
    }))
    const bot = this.telegram.view()
    return {
      servers,
      presets: [...MCP_PRESETS],
      webhooks,
      webhookPresets: [...WEBHOOK_PRESETS],
      deliveries: this.webhooks.deliveries(),
      notifications,
      telegram: {
        settings: value.telegram,
        tokenSet: (await readSecrets(this.ctx, 'channel', 'telegram')).secret !== undefined,
        status: bot.status,
        chats: bot.chats,
        ...bot.pairing === undefined ? {} : { pairing: bot.pairing },
      },
      publicBaseUrl: value.publicBaseUrl,
    }
  }

  /** Add or replace one MCP server row and its secrets. */
  @Remote('saveServer')
  async saveServer(request: SaveServerRequest): Promise<void> {
    const settings = { ...SERVER_DEFAULTS, ...request.settings }
    assertServerSettings(settings)
    if (request.secrets !== undefined) {
      const secrets = request.secrets
      await modifySecrets(this.ctx, 'mcp', settings.id, current => ({
        ...current,
        env: mergeMap(current.env, secrets.env),
        headers: mergeMap(current.headers, secrets.headers),
      }))
    }
    await this.write({ ...this.value, servers: upsert(this.value.servers, settings) })
  }

  /** Remove one MCP server row and its secrets, including any sign-in. */
  @Remote('removeServer')
  async removeServer(request: IdRequest): Promise<void> {
    await this.write({ ...this.value, servers: this.value.servers.filter(row => row.id !== request.id) })
    await deleteSecrets(this.ctx, 'mcp', request.id)
  }

  /** Connect once and list tools without installing the server. */
  @Remote('testServer')
  async testServer(request: TestServerRequest): Promise<McpTestResult> {
    const settings = { ...SERVER_DEFAULTS, ...request.settings }
    const stored = await readSecrets(this.ctx, 'mcp', settings.id)
    return this.mcp.test(settings, {
      ...stored,
      env: mergeMap(stored.env, request.secrets?.env),
      headers: mergeMap(stored.headers, request.secrets?.headers),
    })
  }

  /** Reconnect one server now. */
  @Remote('restartServer')
  async restartServer(request: IdRequest): Promise<void> {
    await this.mcp.restart(request.id)
  }

  /** Start an OAuth sign-in for one HTTP server. */
  @Remote('authorizeServer')
  async authorizeServer(request: AuthorizeServerRequest): Promise<AuthorizeServerResult> {
    const settings = this.value.servers.find(row => row.id === request.id)
    if (settings === undefined) throw new Error(`no MCP server "${request.id}"`)
    if (settings.transport === 'stdio') throw new Error('sign-in applies only to HTTP servers')
    const result = await this.signIns.start(settings.id, settings.url, request.origin)
    if (result.status === 'authorized') await this.mcp.restart(settings.id)
    return result
  }

  /** Forget one server's sign-in. */
  @Remote('signOutServer')
  async signOutServer(request: IdRequest): Promise<void> {
    await modifySecrets(this.ctx, 'mcp', request.id, current => ({ ...current, oauth: undefined }))
    await this.mcp.restart(request.id)
  }

  /** Add or replace one webhook endpoint; a new endpoint gets a secret, returned once. */
  @Remote('saveWebhook')
  async saveWebhook(request: SaveWebhookRequest): Promise<WebhookSecretResult | null> {
    const settings = WebhookSchema(request.settings)
    await this.write({ ...this.value, webhooks: upsert(this.value.webhooks, settings) })
    if ((await readSecrets(this.ctx, 'hook', settings.id)).secret !== undefined) return null
    return { secret: await this.webhooks.rotateSecret(settings.id) }
  }

  /** Remove one webhook endpoint and its secret. */
  @Remote('removeWebhook')
  async removeWebhook(request: IdRequest): Promise<void> {
    await this.write({ ...this.value, webhooks: this.value.webhooks.filter(row => row.id !== request.id) })
    await deleteSecrets(this.ctx, 'hook', request.id)
    await this.webhooks.forget(request.id)
  }

  /** Replace one endpoint's secret; the new one is returned once. */
  @Remote('rotateWebhookSecret')
  async rotateWebhookSecret(request: IdRequest): Promise<WebhookSecretResult> {
    if (!this.value.webhooks.some(row => row.id === request.id)) throw new Error(`no webhook "${request.id}"`)
    return { secret: await this.webhooks.rotateSecret(request.id) }
  }

  /** Run a logged delivery again. */
  @Remote('replayDelivery')
  async replayDelivery(request: IdRequest): Promise<WebhookDeliveryView> {
    return this.webhooks.replay(request.id)
  }

  /** Add or replace one notification target; a new JSON target gets a signing secret, returned once. */
  @Remote('saveNotification')
  async saveNotification(request: SaveNotificationRequest): Promise<WebhookSecretResult | null> {
    const settings = NotificationSchema(request.settings)
    if (request.url !== undefined) {
      const parsed = new URL(request.url)
      if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') throw new Error('notification URL must be http or https')
    }
    await this.write({ ...this.value, notifications: upsert(this.value.notifications, settings) })
    let created: string | undefined
    await modifySecrets(this.ctx, 'notify', settings.id, (current) => {
      const secret = current.secret ?? (settings.format === 'json' ? (created = randomBytes(24).toString('base64url')) : undefined)
      return { ...current, ...request.url === undefined ? {} : { url: request.url }, secret }
    })
    return created === undefined ? null : { secret: created }
  }

  /** Remove one notification target and its URL. */
  @Remote('removeNotification')
  async removeNotification(request: IdRequest): Promise<void> {
    await this.write({ ...this.value, notifications: this.value.notifications.filter(row => row.id !== request.id) })
    await deleteSecrets(this.ctx, 'notify', request.id)
  }

  /** Send a sample notification to one target. */
  @Remote('testNotification')
  async testNotification(request: IdRequest): Promise<TestNotificationResult> {
    return this.notifier.test(request.id)
  }

  /** Save the Telegram bot's settings and, optionally, its token. */
  @Remote('saveTelegram')
  async saveTelegram(request: SaveTelegramRequest): Promise<void> {
    const settings = TelegramSchema(request.settings)
    if (settings.workspacePath.trim() !== '' && !isAbsolute(settings.workspacePath)) {
      throw new Error('the Telegram chat folder must be an absolute path')
    }
    if (request.token !== undefined) {
      const token = request.token === null ? null : request.token.trim()
      if (token !== null && !/^\d+:[\w-]{30,}$/.test(token)) {
        throw new Error('that does not look like a bot token from @BotFather (digits, a colon, then a long code)')
      }
      await this.telegram.setToken(token)
    }
    const previous = this.value.telegram.allowedChats
    await this.write({ ...this.value, telegram: settings })
    for (const chatId of previous) if (!settings.allowedChats.includes(chatId)) await this.telegram.forgetChat(chatId)
  }

  /** A one-time code that pairs the Telegram chat it is sent from. */
  @Remote('telegramPairingCode')
  async telegramPairingCode(): Promise<TelegramPairingResult> {
    return this.telegram.newPairingCode()
  }

  /** Unpair one Telegram chat. */
  @Remote('removeTelegramChat')
  async removeTelegramChat(request: ChatIdRequest): Promise<void> {
    const telegram = this.value.telegram
    await this.write({ ...this.value, telegram: { ...telegram, allowedChats: telegram.allowedChats.filter(id => id !== request.chatId) } })
    await this.telegram.forgetChat(request.chatId)
  }

  /** Set the base URL senders and links use. */
  @Remote('setPublicBaseUrl')
  async setPublicBaseUrl(request: SetPublicBaseUrlRequest): Promise<void> {
    const url = request.url.trim()
    if (url !== '') {
      const parsed = new URL(url)
      if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') throw new Error('base URL must be http or https')
    }
    await this.write({ ...this.value, publicBaseUrl: url.replace(/\/+$/, '') })
  }
}

export default Integrations
