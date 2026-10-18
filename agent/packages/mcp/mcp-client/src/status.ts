/**
 * Live per-server state that a management surface reads: connection phase,
 * last error, the discovered tools with their MCP hints, and — for servers
 * exposed `deferred` — the built tool definitions that are kept out of the
 * model's tool list until a proxy tool dispatches them.
 *
 * Keyed by `serverName`. Only the most recent instance for a name is tracked;
 * an instance clears its own entry on disposal.
 *
 * @module
 */

import type { OAuthClientProvider } from '@modelcontextprotocol/client'
import type { ToolDefinition } from '@deepseek-ai/dsh-tools'

/** Connection phase of one MCP server. */
export type McpServerPhase = 'connecting' | 'connected' | 'reconnecting' | 'failed' | 'needs-auth' | 'stopped'

/** One discovered tool, as the management surface and policy layer see it. */
export interface McpToolInfo {
  /** Name on the MCP wire. */
  readonly rawName: string
  /** Model-facing name (`mcp__<server>__<raw>`). */
  readonly publicName: string
  readonly description: string
  /** Server's `readOnlyHint` annotation; absent when the server did not say. */
  readonly readOnly?: boolean
  /** Server's `destructiveHint` annotation; absent when the server did not say. */
  readonly destructive?: boolean
  /** Whether `includeTools` filtered this tool out. */
  readonly excluded: boolean
}

/** Snapshot of one server. */
export interface McpServerState {
  readonly serverName: string
  readonly phase: McpServerPhase
  readonly exposure: 'native' | 'deferred'
  readonly error?: string
  readonly tools: readonly McpToolInfo[]
  /** Definitions of the included tools when exposure is `deferred`. */
  readonly deferred: ReadonlyMap<string, ToolDefinition>
  /** Last lines the server process wrote to stderr (stdio servers), oldest first. */
  readonly log: readonly string[]
  readonly updatedAt: number
}

/** Lines of server stderr kept per server. */
const LOG_LINES = 60

type Listener = (state: McpServerState | undefined, serverName: string) => void

const states = new Map<string, McpServerState>()
const owners = new Map<string, symbol>()
const listeners = new Set<Listener>()

/**
 * OAuth providers installed by a management plugin, keyed by `serverName`.
 * HTTP and SSE transports pick theirs up when the connection is built.
 */
export const mcpAuthProviders = new Map<string, OAuthClientProvider>()

/** @returns the current state of every tracked server. */
export function mcpServerStates(): McpServerState[] {
  return [...states.values()]
}

/** @returns the current state of one server, if tracked. */
export function mcpServerState(serverName: string): McpServerState | undefined {
  return states.get(serverName)
}

/**
 * Observe state changes.
 * @returns a disposer.
 */
export function watchMcpServers(listener: Listener): () => void {
  listeners.add(listener)
  return () => void listeners.delete(listener)
}

function publish(serverName: string): void {
  const state = states.get(serverName)
  for (const listener of listeners) {
    try { listener(state, serverName) } catch { /* observers are best-effort */ }
  }
}

/** Writer handle owned by one connection supervisor. */
export interface McpStateWriter {
  update(patch: Partial<Omit<McpServerState, 'serverName' | 'updatedAt' | 'log'>>): void
  /** Append stderr text; split into lines, the oldest dropped beyond the bound. */
  appendLog(text: string): void
  clear(): void
}

/**
 * Claim the state slot for one server name. A newer claim supersedes an older
 * one; writes and clears from the superseded owner are ignored.
 */
export function claimMcpState(serverName: string, exposure: 'native' | 'deferred'): McpStateWriter {
  const owner = Symbol(serverName)
  owners.set(serverName, owner)
  states.set(serverName, {
    serverName,
    phase: 'connecting',
    exposure,
    tools: [],
    deferred: new Map(),
    log: [],
    updatedAt: Date.now(),
  })
  publish(serverName)
  return {
    update(patch) {
      if (owners.get(serverName) !== owner) return
      const previous = states.get(serverName)
      if (previous === undefined) return
      const next: McpServerState = { ...previous, ...patch, updatedAt: Date.now() }
      // An explicit phase change without an error clears the stale one.
      if (patch.phase !== undefined && patch.phase !== 'failed' && patch.phase !== 'needs-auth' && !('error' in patch)) {
        delete (next as { error?: string }).error
      }
      states.set(serverName, next)
      publish(serverName)
    },
    appendLog(text) {
      if (owners.get(serverName) !== owner) return
      const previous = states.get(serverName)
      if (previous === undefined) return
      const lines = text.split(/\r?\n/).map(line => line.trimEnd()).filter(line => line !== '')
      if (lines.length === 0) return
      states.set(serverName, { ...previous, log: [...previous.log, ...lines].slice(-LOG_LINES), updatedAt: Date.now() })
      publish(serverName)
    },
    clear() {
      if (owners.get(serverName) !== owner) return
      owners.delete(serverName)
      states.delete(serverName)
      publish(serverName)
    },
  }
}
