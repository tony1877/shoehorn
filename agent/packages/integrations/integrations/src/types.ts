/** Wire types for the integrations Remote: MCP servers, webhook endpoints, notification targets. */

/** How the harness reaches one MCP server. */
export type McpTransportKind = 'stdio' | 'streamable-http' | 'sse'

/**
 * When an MCP tool call needs the user's approval: always, only for tools the
 * server does not mark read-only, or never.
 */
export type McpApprovalMode = 'ask' | 'ask-writes' | 'allow'

/** Connection phase reported for one MCP server. */
export type McpPhase = 'disabled' | 'connecting' | 'connected' | 'reconnecting' | 'failed' | 'needs-auth' | 'stopped'

/** Non-secret settings of one MCP server. */
export interface McpServerSettings {
  /** Stable id; also the tool namespace (`mcp__<id>__<tool>`). */
  id: string
  enabled: boolean
  transport: McpTransportKind
  /** stdio: executable. */
  command: string
  /** stdio: arguments, passed without a shell. */
  args: string[]
  /** stdio: working directory; empty inherits the Host's. */
  cwd: string
  /** HTTP/SSE: endpoint URL. */
  url: string
  /** Raw tool names to expose; empty exposes all. */
  includeTools: string[]
  /** `deferred` keeps tool schemas out of every request until the model looks them up. */
  exposure: 'native' | 'deferred'
  approval: McpApprovalMode
  /** HTTP/SSE: whether the server is reached through an OAuth sign-in. */
  auth: 'none' | 'oauth'
  toolCallTimeoutMs: number
}

/**
 * Secret values of one MCP server, kept in the credential store. On save,
 * an omitted map keeps the stored one and `null` clears it.
 */
export interface McpServerSecrets {
  /** stdio: environment variables for the server process. */
  env?: Record<string, string> | null
  /** HTTP/SSE: request headers such as `Authorization`. */
  headers?: Record<string, string> | null
}

/** One discovered tool. */
export interface McpToolView {
  rawName: string
  publicName: string
  description: string
  readOnly?: boolean
  destructive?: boolean
  excluded: boolean
}

/** One MCP server as the settings page shows it. Secret values never cross the wire. */
export interface McpServerView {
  settings: McpServerSettings
  /** Names of the stored environment variables. */
  envNames: string[]
  /** Names of the stored headers. */
  headerNames: string[]
  /** Whether an OAuth grant is stored. */
  authorized: boolean
  phase: McpPhase
  error?: string
  tools: McpToolView[]
  /** Last lines the server process wrote to stderr (stdio servers). */
  log: string[]
  /** Tool calls since the Host started. */
  calls: number
  /** Failed tool calls since the Host started. */
  failures: number
  /** When the last tool call finished (Unix ms). */
  lastCallAt?: number
}

/** A ready-made server row the page offers with one click. */
export interface McpPreset {
  id: string
  title: string
  description: string
  settings: McpServerSettings
  /** Secrets the user must fill in before the server can work. */
  requiredEnv: string[]
  requiredHeaders: string[]
}

/** Result of connecting to a server without installing it. */
export type McpTestResult =
  | { ok: true; tools: McpToolView[]; serverName?: string; serverVersion?: string }
  | { ok: false; error: string; needsAuth: boolean }

/**
 * How a sender proves a delivery is genuine: an HMAC-SHA256 signature of the
 * raw body (GitHub-style), or the secret itself as a bearer token (for senders
 * that cannot sign: Home Assistant, Zapier, IFTTT, curl).
 */
export type WebhookAuthMode = 'hmac' | 'token'

/** Whether each delivery starts a new chat or continues one chat per endpoint. */
export type WebhookChatMode = 'new' | 'same'

/** One configured inbound webhook endpoint. */
export interface WebhookSettings {
  /** Stable id; also the URL segment (`/hooks/<id>`). */
  id: string
  enabled: boolean
  title: string
  authMode: WebhookAuthMode
  /**
   * `hmac`: request header carrying `sha256=<hex HMAC>` of the raw body.
   * `token`: request header carrying `Bearer <secret>` (or the secret alone);
   * `?token=<secret>` in the URL is accepted too.
   */
  signatureHeader: string
  chatMode: WebhookChatMode
  /** Dot path into the JSON body; empty matches every delivery. */
  matchPath: string
  /** Values `matchPath` may hold; empty matches any present value. */
  matchValues: string[]
  /** Folder the created chat works in. */
  workspacePath: string
  /** Prompt with `{{path.to.field}}` placeholders and `{{body}}` for the whole payload. */
  promptTemplate: string
  /** Chat title with the same placeholders. */
  titleTemplate: string
  agentPreset: string
  permissionPreset: string
  /** Largest accepted request body in bytes. */
  maxBodyBytes: number
  /** Most deliveries accepted per minute. */
  ratePerMinute: number
}

/** What happened to one inbound delivery. */
export type WebhookOutcome = 'started' | 'ignored' | 'rejected' | 'failed'

/** One logged inbound delivery. */
export interface WebhookDeliveryView {
  id: string
  webhookId: string
  receivedAt: number
  outcome: WebhookOutcome
  /** HTTP status returned to the sender. */
  status: number
  detail: string
  sessionId?: string
  /** Body as received, cut to the log's size bound. */
  bodyPreview: string
}

/** One webhook endpoint as the settings page shows it. */
export interface WebhookView {
  settings: WebhookSettings
  /** Path on this Host; prefix it with the public base URL. */
  path: string
  secretSet: boolean
}

/** Event a notification target can subscribe to. */
export type NotificationEvent = 'finished' | 'approval' | 'question'

/** Payload shape a notification target expects. */
export type NotificationFormat = 'json' | 'discord' | 'slack' | 'ntfy'

/** Non-secret settings of one notification target; its URL is a secret. */
export interface NotificationSettings {
  id: string
  enabled: boolean
  title: string
  format: NotificationFormat
  events: NotificationEvent[]
  /** "Chat finished" only after turns that ran at least this long (seconds), so quick replies you watched do not ping you. */
  minTurnSeconds: number
}

/** The Telegram bot's settings; its token is a secret. */
export interface TelegramSettings {
  enabled: boolean
  /** Paired chat ids; only these are served. */
  allowedChats: string[]
  /** Folder new Telegram chats work in. */
  workspacePath: string
  agentPreset: string
  permissionPreset: string
  /** Answer approvals and questions of Telegram chats with buttons in Telegram (off: in the web page). */
  answerInTelegram: boolean
  /** Also tell paired chats about other chats: when they finish, need approval, or ask a question. */
  notifyEvents: NotificationEvent[]
  /** "Chat finished" notifications about other chats only after turns at least this long (seconds). */
  minTurnSeconds: number
}

/** Connection state of the Telegram bot. */
export interface TelegramStatus {
  phase: 'off' | 'needs-token' | 'connecting' | 'running' | 'error'
  username?: string
  error?: string
  lastPollAt?: number
}

/** One paired Telegram chat. */
export interface TelegramChatView {
  chatId: string
  name: string
  /** Title of the agent chat it currently talks to. */
  chatTitle?: string
  sessionId?: string
}

/** The Telegram section of the page. */
export interface TelegramView {
  settings: TelegramSettings
  tokenSet: boolean
  status: TelegramStatus
  chats: TelegramChatView[]
  pairing?: { code: string; expiresAt: number }
}

/** One notification target as the settings page shows it. */
export interface NotificationView {
  settings: NotificationSettings
  urlSet: boolean
  /** Host of the stored URL, for recognition without revealing paths or tokens. */
  urlHost?: string
  lastStatus?: string
  lastSentAt?: number
}

/** Everything the integrations page renders. */
export interface IntegrationsOverview {
  servers: McpServerView[]
  presets: McpPreset[]
  webhooks: WebhookView[]
  deliveries: WebhookDeliveryView[]
  notifications: NotificationView[]
  telegram: TelegramView
  webhookPresets: WebhookPreset[]
  /** Base URL senders reach this Host at; empty when unset. */
  publicBaseUrl: string
}

export interface SaveServerRequest {
  settings: McpServerSettings
  secrets?: McpServerSecrets
}

export interface TestServerRequest {
  settings: McpServerSettings
  secrets?: McpServerSecrets
}

export interface IdRequest {
  id: string
}

export interface AuthorizeServerRequest {
  id: string
  /** Origin the browser reaches this Host at; the OAuth callback returns there. */
  origin: string
}

export type AuthorizeServerResult =
  | { status: 'authorized' }
  | { status: 'redirect'; url: string }

export interface SaveWebhookRequest {
  settings: WebhookSettings
}

/** A ready-made webhook row the page offers with one click. */
export interface WebhookPreset {
  id: string
  title: string
  description: string
  settings: Omit<WebhookSettings, 'id' | 'workspacePath'>
}

/** A newly generated webhook secret, shown once. */
export interface WebhookSecretResult {
  secret: string
}

export interface SaveNotificationRequest {
  settings: NotificationSettings
  /** New URL; omitted keeps the stored one. */
  url?: string
}

export interface TestNotificationResult {
  ok: boolean
  detail: string
}

export interface SetPublicBaseUrlRequest {
  url: string
}

export interface SaveTelegramRequest {
  settings: TelegramSettings
  /** New bot token; `null` removes it; omitted keeps it. */
  token?: string | null
}

export interface TelegramPairingResult {
  code: string
  /** t.me deep link that sends the code, once the bot's username is known. */
  link?: string
}

export interface ChatIdRequest {
  chatId: string
}
