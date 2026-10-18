/** Ready-made MCP server rows offered on the integrations page. */

import { homedir } from 'node:os'
import { join } from 'node:path'
import type { McpPreset, McpServerSettings } from './types.ts'

/** Where the browser preset writes page snapshots and screenshots, as absolute paths the agent can read. */
const BROWSER_OUTPUT_DIR = join(process.env.DSH_HOME ?? join(homedir(), '.dsh'), 'browser-output')

/** Defaults every new server row starts from. */
export const SERVER_DEFAULTS: Omit<McpServerSettings, 'id'> = {
  enabled: true,
  transport: 'stdio',
  command: '',
  args: [],
  cwd: '',
  url: '',
  includeTools: [],
  exposure: 'deferred',
  approval: 'ask-writes',
  auth: 'none',
  toolCallTimeoutMs: 60_000,
}

/**
 * The shipped presets. Versions are pinned so a preset installs the same
 * server it was tested with; `npx` resolves the package on first start.
 */
export const MCP_PRESETS: readonly McpPreset[] = [
  {
    id: 'playwright',
    title: 'Browser (Playwright)',
    description: 'A real browser: open pages, click, type, fill forms, read the page and take screenshots. Handles sites that need JavaScript or a login.',
    settings: {
      ...SERVER_DEFAULTS,
      id: 'browser',
      command: 'npx',
      args: ['-y', '@playwright/mcp@0.0.83', '--isolated', '--headless', '--file-paths', 'absolute', '--output-dir', BROWSER_OUTPUT_DIR],
    },
    requiredEnv: [],
    requiredHeaders: [],
  },
  {
    id: 'context7',
    title: 'Library docs (Context7)',
    description: 'Up-to-date documentation and code examples for libraries and frameworks. Remote server, no account needed.',
    settings: {
      ...SERVER_DEFAULTS,
      id: 'docs',
      transport: 'streamable-http',
      url: 'https://mcp.context7.com/mcp',
      approval: 'allow',
    },
    requiredEnv: [],
    requiredHeaders: [],
  },
  {
    id: 'github',
    title: 'GitHub',
    description: 'Issues, pull requests, code search and repository files through GitHub\'s hosted MCP server. Needs a personal access token.',
    settings: {
      ...SERVER_DEFAULTS,
      id: 'github',
      transport: 'streamable-http',
      url: 'https://api.githubcopilot.com/mcp/',
    },
    requiredEnv: [],
    requiredHeaders: ['Authorization'],
  },
  {
    id: 'notion',
    title: 'Notion',
    description: 'Search, read and edit your Notion pages and databases. Signs in with your Notion account.',
    settings: {
      ...SERVER_DEFAULTS,
      id: 'notion',
      transport: 'streamable-http',
      url: 'https://mcp.notion.com/mcp',
      auth: 'oauth',
    },
    requiredEnv: [],
    requiredHeaders: [],
  },
  {
    id: 'filesystem',
    title: 'Folder access',
    description: 'Read and write files in folders you list as arguments, outside the chat\'s own workspace.',
    settings: {
      ...SERVER_DEFAULTS,
      id: 'files',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem@1.0.0'],
    },
    requiredEnv: [],
    requiredHeaders: [],
  },
]
