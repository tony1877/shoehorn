import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LocalCredentialProvider from '@deepseek-ai/dsh-credentials-local'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import FileSettingsProvider from '@deepseek-ai/dsh-settings-file'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import Integrations, {
  approvalFor,
  authenticate,
  markdownToHtml,
  sniffImage,
  splitMarkdown,
  WEBHOOK_PRESETS,
  fillTemplate,
  formatNotification,
  matches,
  MCP_PRESETS,
  sign,
  type McpServerSettings,
  type WebhookSettings,
} from '@deepseek-ai/dsh-integrations'

const here = dirname(fileURLToPath(import.meta.url))
const fixtureServer = resolve(here, '../../../mcp/mcp-client/tests/fixture-server.ts')

const dirs: string[] = []
const contexts: Context[] = []
afterEach(async () => {
  for (const ctx of contexts.splice(0)) await ctx.fiber.dispose()
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function server(patch: Partial<McpServerSettings>): McpServerSettings {
  return {
    id: 'fixture',
    enabled: true,
    transport: 'stdio',
    command: process.execPath,
    args: [fixtureServer],
    cwd: '',
    url: '',
    includeTools: [],
    exposure: 'native',
    approval: 'allow',
    auth: 'none',
    toolCallTimeoutMs: 15_000,
    ...patch,
  }
}

async function mount(): Promise<{ ctx: Context; service: Integrations }> {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-integrations-'))
  dirs.push(dir)
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(LocalCredentialProvider, { path: join(dir, '.credentials.yaml'), watch: false })
  await ctx.plugin(FileSettingsProvider, { path: join(dir, 'settings.yaml'), watch: false })
  await ctx.plugin(Integrations, { stateDirectory: join(dir, 'state') })
  const service = ctx.get('integrations') as Integrations
  return { ctx, service }
}

async function until(check: () => boolean | Promise<boolean>, ms = 15_000): Promise<void> {
  const deadline = Date.now() + ms
  while (!await check()) {
    if (Date.now() > deadline) throw new Error('timed out waiting')
    await new Promise(resolve => setTimeout(resolve, 50))
  }
}

let seq = 0
function call(ctx: Context, name: string, args: unknown) {
  return ctx.tools.execute({ signal: new AbortController().signal, callId: ToolCallId(`t${++seq}`), name, arguments: args })
}

describe('webhook helpers', () => {
  const hook: WebhookSettings = {
    id: 'gh',
    enabled: true,
    title: '',
    authMode: 'hmac',
    signatureHeader: 'X-Signature-256',
    chatMode: 'new',
    matchPath: 'action',
    matchValues: ['opened'],
    workspacePath: '/w',
    promptTemplate: 'Issue {{issue.title}} by {{issue.user.login}}',
    titleTemplate: '',
    agentPreset: 'p',
    permissionPreset: 'q',
    maxBodyBytes: 1000,
    ratePerMinute: 5,
  }

  it('fills dot-path placeholders and the whole body', () => {
    const body = { issue: { title: 'Crash', user: { login: 'ana' } }, n: [1, 2] }
    expect(fillTemplate(hook.promptTemplate, body)).toBe('Issue Crash by ana')
    expect(fillTemplate('{{missing}}|{{n.1}}', body)).toBe('|2')
    expect(fillTemplate('{{body}}', { a: 1 })).toBe('{\n  "a": 1\n}')
  })

  it('matches on a field value and ignores the rest', () => {
    expect(matches(hook, { action: 'opened' })).toBe(true)
    expect(matches(hook, { action: 'closed' })).toBe(false)
    expect(matches(hook, {})).toBe(false)
    expect(matches({ ...hook, matchPath: '' }, 'anything')).toBe(true)
  })

  it('signs like GitHub', () => {
    expect(sign('s3cret', Buffer.from('{}'))).toMatch(/^sha256=[0-9a-f]{64}$/)
  })

  it('accepts a matching HMAC signature and refuses a wrong or missing one', () => {
    const body = Buffer.from('{"a":1}')
    expect(authenticate(hook, 'k', body, { 'x-signature-256': [sign('k', body)] }, '/hooks/gh')).toBeUndefined()
    expect(authenticate(hook, 'k', body, { 'x-signature-256': [sign('other', body)] }, '/hooks/gh')).toBe('signature does not match')
    expect(authenticate(hook, 'k', body, {}, '/hooks/gh')).toBe('missing X-Signature-256 header')
  })

  it('accepts the secret as a bearer token or a token parameter in token mode', () => {
    const token = { ...hook, authMode: 'token' as const, signatureHeader: 'Authorization' }
    const body = Buffer.from('x')
    expect(authenticate(token, 'tok', body, { authorization: ['Bearer tok'] }, '/hooks/gh')).toBeUndefined()
    expect(authenticate(token, 'tok', body, {}, '/hooks/gh?token=tok')).toBeUndefined()
    expect(authenticate(token, 'tok', body, { authorization: ['Bearer nope'] }, '/hooks/gh')).toBe('token does not match')
    expect(authenticate(token, 'tok', body, {}, '/hooks/gh')).toMatch(/missing Authorization/)
  })

  it('ships webhook presets whose templates fill from a sample payload', () => {
    const github = WEBHOOK_PRESETS.find(preset => preset.id === 'github-issues')!
    const payload = { action: 'opened', issue: { number: 7, title: 'Crash', body: 'It crashes', user: { login: 'ana' }, html_url: 'u' }, repository: { full_name: 'o/r' } }
    expect(matches({ ...hook, ...github.settings }, payload)).toBe(true)
    expect(fillTemplate(github.settings.titleTemplate, payload)).toBe('Issue #7: Crash')
    expect(fillTemplate(github.settings.promptTemplate, payload)).toContain('opened in o/r by ana')
  })
})

describe('Telegram formatting', () => {
  it('converts the Markdown models write into Telegram HTML, escaping the rest', () => {
    expect(markdownToHtml('**Bold** and `a<b>` and [link](https://x.y/?a=1&b=2)'))
      .toBe('<b>Bold</b> and <code>a&lt;b&gt;</code> and <a href="https://x.y/?a=1&amp;b=2">link</a>')
    expect(markdownToHtml('# Title\n- item <x>')).toBe('<b>Title</b>\n• item &lt;x&gt;')
    expect(markdownToHtml('```ts\nconst a = 1 < 2\n```')).toBe('<pre><code class="language-ts">const a = 1 &lt; 2</code></pre>')
  })

  it('splits long answers under the limit and keeps code blocks balanced in every piece', () => {
    const fence = '```'
    const code = Array.from({ length: 400 }, (_, i) => `line ${i} of the code block`).join('\n')
    const pieces = splitMarkdown(`Intro\n\n${fence}js\n${code}\n${fence}\nOutro`)
    expect(pieces.length).toBeGreaterThan(1)
    for (const piece of pieces) {
      expect(piece.length).toBeLessThanOrEqual(3600)
      expect((piece.match(/^\s*```/gm) ?? []).length % 2).toBe(0)
      expect(markdownToHtml(piece).length).toBeLessThanOrEqual(4096)
    }
    expect(pieces.join('\n')).toContain('line 399 of the code block')
    expect(pieces.at(-1)).toContain('Outro')
  })

  it('tells image types from their bytes, not their labels', () => {
    expect(sniffImage(Uint8Array.from([0xFF, 0xD8, 0xFF, 0xE0]))).toBe('image/jpeg')
    expect(sniffImage(Uint8Array.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))).toBe('image/png')
    expect(sniffImage(new TextEncoder().encode('GIF89a'))).toBe('image/gif')
    expect(sniffImage(new TextEncoder().encode('RIFF\0\0\0\0WEBPVP8 '))).toBe('image/webp')
    expect(sniffImage(new TextEncoder().encode('%PDF-1.7'))).toBeUndefined()
  })

  it('cuts a single overlong line', () => {
    const pieces = splitMarkdown('x'.repeat(9000))
    expect(pieces.length).toBe(3)
    expect(pieces.join('')).toBe('x'.repeat(9000))
  })
})

describe('approval and notification formats', () => {
  it('asks for writes unless the tool is marked read-only and not destructive', () => {
    const settings = server({ approval: 'ask-writes' })
    expect(approvalFor(settings, { rawName: 'r', publicName: 'p', description: '', readOnly: true, excluded: false })).toBe('allow')
    expect(approvalFor(settings, { rawName: 'w', publicName: 'p', description: '', excluded: false })).toBe('ask')
    expect(approvalFor(settings, { rawName: 'x', publicName: 'p', description: '', readOnly: true, destructive: true, excluded: false })).toBe('ask')
    expect(approvalFor(server({ approval: 'ask' }), undefined)).toBe('ask')
    expect(approvalFor(server({ approval: 'allow' }), undefined)).toBe('allow')
  })

  it('formats each target kind', () => {
    const message = { event: 'approval' as const, sessionId: 's', title: 'Fix bug', text: 'pwsh — rm', link: 'http://h/?session=s' }
    expect(JSON.parse(formatNotification('discord', message).body)).toEqual({ content: 'Approval needed: Fix bug\npwsh — rm\nhttp://h/?session=s' })
    expect(JSON.parse(formatNotification('slack', message).body).text).toContain('Approval needed')
    const ntfy = formatNotification('ntfy', message)
    expect(ntfy.headers.title).toBe('Approval needed: Fix bug')
    expect(ntfy.headers.click).toBe('http://h/?session=s')
    expect(JSON.parse(formatNotification('json', message).body)).toEqual(message)
  })

  it('ships presets that pass validation', () => {
    for (const preset of MCP_PRESETS) expect(preset.settings.id).toMatch(/^[a-z][a-z0-9-]*$/)
  })
})

describe('MCP servers from settings', () => {
  it('starts a native server, exposes its tools, and removes them with the row', async () => {
    const { ctx, service } = await mount()
    await service.saveServer({ settings: server({}) })
    // Status is read through the service: it owns the mcp-client module instance.
    await until(async () => (await service.overview()).servers[0]?.phase === 'connected')
    expect(ctx.tools.get('mcp__fixture__add')).toBeDefined()
    const result = await call(ctx, 'mcp__fixture__add', { a: 2, b: 3 })
    expect(result.content[0]).toEqual({ type: 'text', text: '5' })

    const overview = await service.overview()
    expect(overview.servers[0]!.phase).toBe('connected')
    expect(overview.servers[0]!.tools.map(tool => tool.rawName)).toContain('greet')

    await service.removeServer({ id: 'fixture' })
    await until(() => ctx.tools.get('mcp__fixture__add') === undefined)
    expect((await service.overview()).servers).toEqual([])
  }, 30_000)

  it('keeps deferred tools behind mcp_search / mcp_call and honors the allowlist', async () => {
    const { ctx, service } = await mount()
    await service.saveServer({ settings: server({ exposure: 'deferred', includeTools: ['add', 'greet'] }) })
    await until(() => ctx.tools.get('mcp_search') !== undefined)
    expect(ctx.tools.get('mcp__fixture__add')).toBeUndefined()
    expect(ctx.tools.get('mcp_search')!.description).toContain('- fixture: add, greet')

    const found = await call(ctx, 'mcp_search', { query: 'greet' })
    expect(JSON.stringify(found.content)).toContain('mcp__fixture__greet')
    const greeted = await call(ctx, 'mcp_call', { tool: 'mcp__fixture__greet', arguments: { name: 'Ana' } })
    expect(greeted.content[0]).toEqual({ type: 'text', text: 'Hello, Ana!' })
    const excluded = await call(ctx, 'mcp_call', { tool: 'mcp__fixture__fail', arguments: {} })
    expect(excluded.isError).toBe(true)
  }, 30_000)

  it('requires approval for a write under ask-writes, through the proxy too', async () => {
    const { ctx, service } = await mount()
    await service.saveServer({ settings: server({ approval: 'ask-writes', exposure: 'deferred' }) })
    await until(() => ctx.tools.get('mcp_call') !== undefined)
    // No approval service is mounted, so an ask is refused.
    const result = await call(ctx, 'mcp_call', { tool: 'mcp__fixture__add', arguments: { a: 1, b: 1 } })
    expect(result.isError).toBe(true)
    expect(JSON.stringify(result.content)).toMatch(/needs your approval/)
  }, 30_000)

  it('keeps secrets out of settings and reports only their names', async () => {
    const { service } = await mount()
    await service.saveServer({ settings: server({ enabled: false }), secrets: { env: { TOKEN: 'abc' } } })
    const view = (await service.overview()).servers[0]!
    expect(view.envNames).toEqual(['TOKEN'])
    expect(view.phase).toBe('disabled')
    expect(JSON.stringify(view)).not.toContain('abc')
    // A blank value keeps the stored one.
    await service.saveServer({ settings: server({ enabled: false }), secrets: { env: { TOKEN: '', OTHER: 'x' } } })
    expect((await service.overview()).servers[0]!.envNames).toEqual(['TOKEN', 'OTHER'])
  }, 30_000)

  it('rejects an invalid server row before saving it', async () => {
    const { service } = await mount()
    await expect(service.saveServer({ settings: server({ id: 'Bad Name' }) })).rejects.toThrow(/lowercase/)
    await expect(service.saveServer({ settings: server({ transport: 'streamable-http', url: 'nope' }) })).rejects.toThrow(/valid URL/)
  })
})
