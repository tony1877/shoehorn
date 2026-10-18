/**
 * MCP servers from settings: one `dsh-mcp-client` child fiber per enabled
 * server, restarted when its settings or secrets change; two proxy tools that
 * reach `deferred` servers without putting their schemas in every request;
 * and the per-server approval policy, enforced on every MCP tool call.
 */

import type { Context, Fiber } from '@deepseek-ai/cordis'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import * as McpClient from '@deepseek-ai/dsh-mcp-client'
import {
  createTransport,
  mcpAuthProviders,
  mcpServerState,
  mcpServerStates,
  watchMcpServers,
  type McpServerState,
} from '@deepseek-ai/dsh-mcp-client'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { PreToolDecision, ToolDefinition, ToolExecution } from '@deepseek-ai/dsh-tools'
import type { JsonValue } from '@deepseek-ai/dsh-util-values'
import { StoredOAuthProvider } from './oauth.ts'
import { readSecrets, type IntegrationSecrets } from './secrets.ts'
import type { McpServerSettings, McpServerView, McpTestResult, McpToolView } from './types.ts'

/** Model-facing proxy tool names. */
export const SEARCH_TOOL = 'mcp_search'
export const CALL_TOOL = 'mcp_call'

/** Most matches one search returns. */
const SEARCH_LIMIT = 8

interface RunningServer {
  /** Settings + secrets fingerprint the fiber was started from. */
  key: string
  fiber: Fiber
}

/** Build the mcp-client config for one server row. */
function clientConfig(settings: McpServerSettings, secrets: IntegrationSecrets): McpClient.Config {
  const common = {
    serverName: settings.id,
    toolCallTimeoutMs: settings.toolCallTimeoutMs,
    failOnStartupError: false,
    includeTools: settings.includeTools,
    exposure: settings.exposure,
  }
  if (settings.transport === 'stdio') {
    return {
      ...common,
      transport: 'stdio',
      command: settings.command,
      args: settings.args,
      env: secrets.env ?? {},
      cwd: settings.cwd,
    }
  }
  return {
    ...common,
    transport: settings.transport,
    url: settings.url,
    headers: secrets.headers ?? {},
  }
}

/** Fail loud on a row that cannot start, before a fiber is spawned for it. */
export function assertServerSettings(settings: McpServerSettings): void {
  if (!/^[a-z][a-z0-9-]{0,31}$/.test(settings.id)) {
    throw new Error(`MCP server id "${settings.id}" must start with a lowercase letter and use only a-z, 0-9 and "-" (at most 32 characters)`)
  }
  if (settings.transport === 'stdio') {
    if (settings.command.trim() === '') throw new Error(`MCP server "${settings.id}" needs a command`)
    if (settings.auth === 'oauth') throw new Error(`MCP server "${settings.id}": sign-in applies only to HTTP servers`)
    return
  }
  let url: URL
  try {
    url = new URL(settings.url)
  } catch {
    throw new Error(`MCP server "${settings.id}" needs a valid URL`)
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') throw new Error(`MCP server "${settings.id}" URL must be http or https`)
}

function toolViews(state: McpServerState | undefined): McpToolView[] {
  return (state?.tools ?? []).map(tool => ({ ...tool }))
}

/** Per-server tool-call counters since the Host started. */
export interface CallStats {
  calls: number
  failures: number
  lastCallAt?: number
}

/** One server's view, given its stored secrets and call counters. */
export function serverView(settings: McpServerSettings, secrets: IntegrationSecrets, stats?: CallStats): McpServerView {
  const state = settings.enabled ? mcpServerState(settings.id) : undefined
  return {
    settings,
    envNames: Object.keys(secrets.env ?? {}),
    headerNames: Object.keys(secrets.headers ?? {}),
    authorized: secrets.oauth?.tokens !== undefined,
    phase: !settings.enabled
      ? 'disabled'
      : state?.phase ?? (settings.auth === 'oauth' && settings.transport !== 'stdio' && secrets.oauth?.tokens === undefined ? 'needs-auth' : 'stopped'),
    ...state?.error === undefined ? {} : { error: state.error },
    tools: toolViews(state),
    log: [...state?.log ?? []],
    calls: stats?.calls ?? 0,
    failures: stats?.failures ?? 0,
    ...stats?.lastCallAt === undefined ? {} : { lastCallAt: stats.lastCallAt },
  }
}

/** Decide one MCP call under its server's approval mode. */
export function approvalFor(settings: McpServerSettings, tool: McpToolView | undefined): 'allow' | 'ask' {
  switch (settings.approval) {
    case 'allow': return 'allow'
    case 'ask': return 'ask'
    case 'ask-writes': return tool?.readOnly === true && tool.destructive !== true ? 'allow' : 'ask'
  }
}

/** Find the server and tool behind a model-facing MCP name. */
function locate(publicName: string): { state: McpServerState; tool: McpToolView | undefined } | undefined {
  for (const state of mcpServerStates()) {
    const tool = state.tools.find(candidate => candidate.publicName === publicName)
    if (tool !== undefined) return { state, tool }
    if (state.deferred.has(publicName)) return { state, tool: undefined }
  }
  return undefined
}

/** Score a deferred tool against search words: name hits weigh double. */
function score(words: string[], name: string, description: string): number {
  const haystackName = name.toLowerCase()
  const haystackDescription = description.toLowerCase()
  let total = 0
  for (const word of words) {
    if (haystackName.includes(word)) total += 2
    if (haystackDescription.includes(word)) total += 1
  }
  return total
}

/** The deferred tools of every connected server, in a stable order. */
function deferredTools(): Array<{ server: string; definition: ToolDefinition }> {
  const out: Array<{ server: string; definition: ToolDefinition }> = []
  for (const state of [...mcpServerStates()].sort((a, b) => a.serverName.localeCompare(b.serverName))) {
    for (const definition of state.deferred.values()) out.push({ server: state.serverName, definition })
  }
  return out
}

function searchDescription(): string {
  const byServer = new Map<string, string[]>()
  for (const { server, definition } of deferredTools()) {
    const raw = definition.name.slice(`mcp__${server}__`.length) || definition.name
    byServer.set(server, [...byServer.get(server) ?? [], raw])
  }
  const lines = [...byServer].map(([server, names]) => `- ${server}: ${names.join(', ')}`)
  return [
    'Look up tools from connected services and get their exact names and input schemas. Search before calling one with mcp_call; the schema tells you which arguments it takes.',
    'Connected services and their tools:',
    ...lines,
  ].join('\n')
}

function render(content: readonly ContentBlock[]): string {
  return content.map(block => (block.type === 'text' ? block.text : `[${block.type}]`)).join('\n')
}

/** Owner of every MCP server fiber and of the MCP-facing tools and policy. */
export class McpManager {
  private readonly running = new Map<string, RunningServer>()
  private settings: readonly McpServerSettings[] = []
  private proxyDispose: (() => void) | undefined
  private proxySignature = ''
  private reconciling: Promise<void> = Promise.resolve()
  private readonly stats = new Map<string, CallStats>()

  constructor(private readonly ctx: Context) {
    ctx.effect(() => watchMcpServers(() => this.refreshProxies()), 'integrations.mcp.watch')
    ctx.effect(() => () => {
      this.proxyDispose?.()
      this.proxyDispose = undefined
    }, 'integrations.mcp.proxies')
    ctx.effect(() => () => {
      for (const [id, server] of this.running) {
        void server.fiber.dispose()
        mcpAuthProviders.delete(id)
      }
      this.running.clear()
    }, 'integrations.mcp.servers')
    ctx.on('tools/pre-execute', (exec, next) => this.gate(exec, next))
    ctx.on('tools/post-execute', async (exec, result, next) => {
      const server = this.serverOf(exec)
      if (server !== undefined) {
        const current = this.stats.get(server) ?? { calls: 0, failures: 0 }
        this.stats.set(server, { calls: current.calls + 1, failures: current.failures + (result.isError ? 1 : 0), lastCallAt: Date.now() })
      }
      return next()
    })
  }

  /** Call counters of one server. */
  statsOf(id: string): CallStats | undefined {
    return this.stats.get(id)
  }

  /** The server a call goes to, direct or through the proxy. */
  private serverOf(exec: ToolExecution): string | undefined {
    let publicName: string | undefined
    if (exec.name.startsWith('mcp__')) publicName = exec.name
    else if (exec.name === CALL_TOOL) {
      const tool = (exec.arguments as { tool?: unknown } | null)?.tool
      if (typeof tool === 'string') publicName = tool
    }
    return publicName === undefined ? undefined : locate(publicName)?.state.serverName
  }

  /** Current settings rows. */
  get servers(): readonly McpServerSettings[] {
    return this.settings
  }

  /**
   * Bring the running fibers in line with the settings: start new or changed
   * enabled servers, stop removed or disabled ones. Serialized.
   */
  apply(servers: readonly McpServerSettings[]): Promise<void> {
    this.settings = servers
    const run = this.reconciling.then(() => this.reconcile())
    this.reconciling = run.catch((error: unknown) => {
      this.ctx.logger.warn(`integrations: MCP reconcile failed: ${String(error)}`)
    })
    return run
  }

  /** Restart one server, e.g. after its sign-in finished. */
  restart(id: string): Promise<void> {
    const running = this.running.get(id)
    if (running !== undefined) running.key = ''
    return this.apply(this.settings)
  }

  private async reconcile(): Promise<void> {
    const wanted = new Map<string, { config: McpClient.Config; key: string; oauthRedirect?: string; oauth: boolean }>()
    for (const settings of this.settings) {
      if (!settings.enabled) continue
      try {
        assertServerSettings(settings)
      } catch (error) {
        this.ctx.logger.warn(`integrations: ${String(error instanceof Error ? error.message : error)}`)
        continue
      }
      const secrets = await readSecrets(this.ctx, 'mcp', settings.id)
      const config = clientConfig(settings, secrets)
      const oauth = settings.auth === 'oauth' && settings.transport !== 'stdio'
      // Without a grant a connection can only fail; it waits for the sign-in.
      if (oauth && secrets.oauth?.tokens === undefined) continue
      wanted.set(settings.id, {
        config,
        oauth,
        ...secrets.oauth?.redirectUrl === undefined ? {} : { oauthRedirect: secrets.oauth.redirectUrl },
        key: JSON.stringify([config, oauth, secrets.oauth?.tokens?.access_token ?? null]),
      })
    }
    for (const [id, server] of [...this.running]) {
      if (wanted.get(id)?.key === server.key) continue
      this.running.delete(id)
      await server.fiber.dispose()
      mcpAuthProviders.delete(id)
    }
    for (const [id, want] of wanted) {
      if (this.running.has(id)) continue
      if (want.oauth) mcpAuthProviders.set(id, new StoredOAuthProvider(this.ctx, id, want.oauthRedirect))
      const fiber = this.ctx.plugin(McpClient, want.config)
      this.running.set(id, { key: want.key, fiber })
    }
  }

  /**
   * Connect to a server once, list its tools, and disconnect, without
   * installing anything.
   */
  async test(settings: McpServerSettings, secrets: IntegrationSecrets): Promise<McpTestResult> {
    try {
      assertServerSettings(settings)
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error), needsAuth: false }
    }
    const config = McpClient.Config(clientConfig({ ...settings, id: settings.id }, secrets) as never) as McpClient.Config
    const provider = settings.auth === 'oauth' && settings.transport !== 'stdio'
      ? new StoredOAuthProvider(this.ctx, settings.id, secrets.oauth?.redirectUrl)
      : undefined
    const previousProvider = mcpAuthProviders.get(settings.id)
    if (provider !== undefined) mcpAuthProviders.set(settings.id, provider)
    const client = new Client({ name: 'dsh-integrations-test', version: '0.0.1' }, { capabilities: {} })
    const stderr: string[] = []
    try {
      const transport = createTransport(config) as Awaited<ReturnType<typeof createTransport>> & { stderr?: NodeJS.ReadableStream | null }
      transport.stderr?.on('data', (chunk: Buffer | string) => {
        stderr.push(...String(chunk).split(/\r?\n/).filter(line => line.trim() !== ''))
        if (stderr.length > 20) stderr.splice(0, stderr.length - 20)
      })
      await client.connect(transport)
      const tools: McpToolView[] = []
      let cursor: string | undefined
      do {
        const page = await client.listTools(cursor === undefined ? {} : { cursor })
        for (const tool of page.tools) {
          tools.push({
            rawName: tool.name,
            publicName: McpClient.publicToolName(settings.id, tool.name),
            description: tool.description ?? '',
            ...typeof tool.annotations?.readOnlyHint === 'boolean' ? { readOnly: tool.annotations.readOnlyHint } : {},
            ...typeof tool.annotations?.destructiveHint === 'boolean' ? { destructive: tool.annotations.destructiveHint } : {},
            excluded: settings.includeTools.length > 0 && !settings.includeTools.includes(tool.name),
          })
        }
        cursor = page.nextCursor
      } while (cursor !== undefined)
      const info = client.getServerVersion()
      return {
        ok: true,
        tools,
        ...info?.name === undefined ? {} : { serverName: info.name },
        ...info?.version === undefined ? {} : { serverVersion: info.version },
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      const message = stderr.length === 0 ? detail : [detail, ...stderr.slice(-8)].join('\n')
      return { ok: false, error: message, needsAuth: /unauthori[sz]ed|\b401\b/i.test(message) || (error as Error)?.name === 'UnauthorizedError' }
    } finally {
      try { await client.close() } catch { /* already closed */ }
      if (provider !== undefined) {
        if (previousProvider === undefined) mcpAuthProviders.delete(settings.id)
        else mcpAuthProviders.set(settings.id, previousProvider)
      }
    }
  }

  /** Register, replace, or drop the proxy tools to match the deferred tool set. */
  private refreshProxies(): void {
    const tools = deferredTools()
    const signature = tools.map(({ definition }) => `${definition.name}\n${definition.description}`).join('\n\n')
    if (signature === this.proxySignature) return
    this.proxySignature = signature
    this.proxyDispose?.()
    this.proxyDispose = undefined
    if (tools.length === 0) return
    const disposeSearch = this.ctx.tools.register(this.searchTool())
    const disposeCall = this.ctx.tools.register(this.callTool())
    this.proxyDispose = () => {
      disposeSearch()
      disposeCall()
    }
  }

  private searchTool(): ToolDefinition {
    return {
      name: SEARCH_TOOL,
      description: searchDescription(),
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Words describing what you want to do, or a tool name. Empty lists every tool.' },
        },
        required: ['query'],
        additionalProperties: false,
      },
      output: {
        schema: { type: 'string' },
        render: (_args, value) => [{ type: 'text', text: value as string }],
      },
      isConcurrencySafe: () => true,
      execute: async (args) => {
        const query = String((args as { query?: unknown }).query ?? '').toLowerCase()
        const words = query.split(/[^a-z0-9_-]+/).filter(word => word !== '')
        const ranked = deferredTools()
          .map(entry => ({ ...entry, score: words.length === 0 ? 1 : score(words, entry.definition.name, entry.definition.description) }))
          .filter(entry => entry.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, SEARCH_LIMIT)
        if (ranked.length === 0) return `No tool matches "${query}". Try other words, or an empty query to list every tool.`
        return ranked.map(({ definition }) => [
          `## ${definition.name}`,
          definition.description.trim(),
          `Input schema: ${JSON.stringify(definition.parameters)}`,
        ].filter(line => line !== '').join('\n')).join('\n\n')
      },
    }
  }

  private callTool(): ToolDefinition {
    const target = (args: unknown): ToolDefinition | undefined => {
      const name = (args as { tool?: unknown } | null)?.tool
      if (typeof name !== 'string') return undefined
      return locate(name)?.state.deferred.get(name)
    }
    return {
      name: CALL_TOOL,
      description: 'Call a tool from a connected service by its exact name, with arguments matching its input schema. Look the tool up with mcp_search first unless its schema is already in this chat.',
      parameters: {
        type: 'object',
        properties: {
          tool: { type: 'string', description: 'Exact tool name, such as mcp__browser__browser_navigate.' },
          arguments: { type: 'object', description: 'Arguments for that tool.' },
        },
        required: ['tool', 'arguments'],
        additionalProperties: false,
      },
      output: {
        schema: { type: 'object', properties: { content: { type: 'array', items: {} }, structuredContent: {} }, required: ['content'], additionalProperties: false },
        render: (args, value) => {
          const definition = target(args)
          if (definition === undefined) return [{ type: 'text', text: JSON.stringify(value) }]
          return definition.output.render((args as { arguments?: unknown }).arguments ?? {}, value)
        },
      },
      timeoutMs: 120_000,
      execute: async (args, exec) => {
        const name = (args as { tool?: unknown }).tool
        const definition = target(args)
        if (definition === undefined) {
          throw new Error(`No connected tool is named "${String(name)}". Use mcp_search to find the exact name.`)
        }
        const toolArgs = (args as { arguments?: unknown }).arguments ?? {}
        try {
          return await definition.execute(toolArgs, exec)
        } catch (error) {
          const text = error instanceof Error ? error.message : String(error)
          throw new Error(`${text}
(Check the arguments against the input schema: mcp_search with the query "${String(name)}" returns it.)`, { cause: error })
        }
      },
      finalizeContent: (exec, result) => target(exec.arguments)?.finalizeContent?.(exec, result),
    }
  }

  /** Approval policy for every MCP tool, direct or through the proxy. */
  private async gate(exec: ToolExecution, next: () => Promise<PreToolDecision>): Promise<PreToolDecision> {
    const decision = await next()
    if (decision.kind !== 'allow') return decision
    let publicName: string | undefined
    if (exec.name.startsWith('mcp__')) publicName = exec.name
    else if (exec.name === CALL_TOOL) {
      const tool = (exec.arguments as { tool?: unknown } | null)?.tool
      if (typeof tool === 'string') publicName = tool
    }
    if (publicName === undefined) return decision
    const found = locate(publicName)
    if (found === undefined) return decision
    const settings = this.settings.find(row => row.id === found.state.serverName)
    if (settings === undefined) return decision
    if (approvalFor(settings, found.tool) === 'allow') return decision
    // A session whose approval policy is `never` (full access) runs without
    // prompts; an ask there would be refused, not skipped.
    const agent = exec.agent
    if (agent !== undefined && this.ctx.get('approval')?.overrideOf(agent.session) === 'never') return decision
    return { kind: 'ask', reason: `"${found.tool?.rawName ?? publicName}" from the connected service "${settings.id}" needs your approval` }
  }
}

/** Plain-text projection of MCP content, for logs and the delivery view. */
export function contentText(content: readonly ContentBlock[]): string {
  return render(content)
}

/** Narrow an arbitrary JSON value to an object. */
export function isObject(value: JsonValue | unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
