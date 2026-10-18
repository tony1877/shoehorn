/**
 * Transport factory: creates the appropriate MCP transport based on the
 * plugin's resolved config. Stdio spawns a child process (with credential
 * scrubbing); Streamable HTTP connects to a URL.
 *
 * @module
 */

import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'
import { scrubbedParentEnv } from '@deepseek-ai/dsh-subprocess'
import type { Config } from './index.ts'
import { mcpAuthProviders } from './status.ts'

/**
 * The subprocess seam's scrubbed parent env (credential-shaped and stale
 * `DSH_*` names dropped), plus the spec's explicit env. The MCP SDK owns the
 * actual spawn, so this transport shares the scrub definition rather than the
 * spawn path.
 */
function buildChildEnv(extra: Record<string, string>): Record<string, string> {
  return { ...scrubbedParentEnv(), ...extra }
}

/**
 * Create an MCP transport from the resolved plugin config.
 *
 * @param config - Resolved plugin config discriminated on `transport`.
 * @returns A connected-ready MCP Transport (stdio, Streamable HTTP, or HTTP+SSE).
 */
export function createTransport(config: Config): Transport {
  switch (config.transport) {
    case 'stdio':
      return new StdioClientTransport({
        command: config.command,
        args: config.args,
        env: buildChildEnv(config.env),
        cwd: config.cwd,
        // Piped, not inherited: the supervisor keeps the tail for diagnostics (and must read it, or the child blocks).
        stderr: 'pipe',
      })
    case 'streamable-http': {
      const authProvider = mcpAuthProviders.get(config.serverName)
      // The MCP SDK's StreamableHTTPClientTransport has optional callback
      // properties typed without `| undefined` (exactOptionalPropertyTypes
      // mismatch with the Transport interface); the SDK constructed the
      // object, so the cast records only that widening.
      return new StreamableHTTPClientTransport(
        new URL(config.url),
        { requestInit: { headers: config.headers }, ...authProvider === undefined ? {} : { authProvider } },
      ) as Transport
    }
    case 'sse': {
      const authProvider = mcpAuthProviders.get(config.serverName)
      return new SSEClientTransport(
        new URL(config.url),
        {
          requestInit: { headers: config.headers },
          eventSourceInit: { fetch: (url, init) => fetch(url, { ...init, headers: { ...init?.headers as Record<string, string>, ...config.headers } }) },
          ...authProvider === undefined ? {} : { authProvider },
        },
      ) as Transport
    }
  }
}
