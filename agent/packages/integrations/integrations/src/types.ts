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

/** One configured inbound webhook endpoint. */
export interface WebhookSettings {
  /** Stable id; also the URL segment (`/hooks/<id>`). */
  id: string
  enabled: boolean
  title: string
  /** Request header carrying `sha256=<hex HMAC>` of the raw body. */
  signatureHeader: string
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
