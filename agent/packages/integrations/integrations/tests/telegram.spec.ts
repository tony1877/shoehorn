import { createServer, type Server } from 'node:http'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import type { AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LocalCredentialProvider from '@deepseek-ai/dsh-credentials-local'
import StubSettings from './settings-stub.ts'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import Integrations, { type TelegramSettings } from '@deepseek-ai/dsh-integrations'

/** A fake Bot API: queued updates for getUpdates, every other call recorded. */
class FakeTelegram {
  readonly calls: Array<{ method: string; body: Record<string, unknown> }> = []
  private updates: Array<Record<string, unknown>> = []
  private nextId = 1
  server: Server

  constructor() {
    this.server = createServer((request, response) => {
      let text = ''
      request.on('data', (chunk: Buffer) => { text += String(chunk) })
      request.on('end', () => {
        const method = (request.url ?? '').split('/').at(-1)!
        const body = text === '' ? {} : JSON.parse(text) as Record<string, unknown>
        const send = (result: unknown): void => {
          response.writeHead(200, { 'content-type': 'application/json' })
          response.end(JSON.stringify({ ok: true, result }))
        }
        if (method === 'getUpdates') {
          const offset = Number(body.offset ?? 0)
          const ready = (): Array<Record<string, unknown>> => this.updates.filter(update => Number(update.update_id) >= offset)
          const deadline = Date.now() + 300
          const wait = (): void => {
            if (ready().length > 0 || Date.now() > deadline) send(ready())
            else setTimeout(wait, 20)
          }
          wait()
          return
        }
        this.calls.push({ method, body })
        if (method === 'getMe') return send({ id: 1, is_bot: true, first_name: 'Bot', username: 'shoehorn_test_bot' })
        if (method === 'sendMessage') return send({ message_id: this.calls.length, chat: { id: body.chat_id, type: 'private' } })
        send(true)
      })
    })
  }

  async listen(): Promise<string> {
    await new Promise<void>(resolve => this.server.listen(0, '127.0.0.1', resolve))
    return `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`
  }

  message(chatId: number, text: string): void {
    this.updates.push({ update_id: this.nextId++, message: { message_id: this.nextId, chat: { id: chatId, type: 'private', first_name: 'Ana' }, from: { id: chatId, is_bot: false, first_name: 'Ana' }, text } })
  }

  button(chatId: number, data: string): void {
    this.updates.push({ update_id: this.nextId++, callback_query: { id: `cb${this.nextId}`, from: { id: chatId, is_bot: false, first_name: 'Ana' }, data, message: { message_id: 1, chat: { id: chatId, type: 'private' } } } })
  }

  sent(chatId: number): string[] {
    return this.calls.filter(call => call.method === 'sendMessage' && String(call.body.chat_id) === String(chatId)).map(call => String(call.body.text))
  }
}

const cleanups: Array<() => Promise<void> | void> = []
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup()
})

async function until(check: () => boolean | Promise<boolean>, ms = 10_000): Promise<void> {
  const deadline = Date.now() + ms
  while (!await check()) {
    if (Date.now() > deadline) throw new Error('timed out waiting')
    await new Promise(resolve => setTimeout(resolve, 25))
  }
}

const TOKEN = `123456:${'a'.repeat(35)}`

function settings(patch: Partial<TelegramSettings> = {}): TelegramSettings {
  return {
    enabled: true,
    allowedChats: [],
    workspacePath: '',
    agentPreset: 'minimal-compact',
    permissionPreset: 'workspace-write',
    answerInTelegram: true,
    notifyEvents: ['approval', 'question'],
    minTurnSeconds: 20,
    ...patch,
  }
}

async function mount(): Promise<{ service: Integrations; fake: FakeTelegram; state: string }> {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-telegram-'))
  const fake = new FakeTelegram()
  const base = await fake.listen()
  const ctx = new Context()
  cleanups.push(() => rmSync(dir, { recursive: true, force: true }))
  cleanups.push(() => new Promise<void>(resolve => fake.server.close(() => resolve())))
  cleanups.push(() => ctx.fiber.dispose())
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(LocalCredentialProvider, { path: join(dir, '.credentials.yaml'), watch: false })
  await ctx.plugin(StubSettings)
  await ctx.plugin(Integrations, { stateDirectory: join(dir, 'state'), telegramApiBase: base } as never)
  return { service: ctx.get('integrations') as Integrations, fake, state: join(dir, 'state', 'telegram.json') }
}

describe('Telegram bot', () => {
  it('connects, refuses unknown chats once, pairs with a code, and answers commands', async () => {
    const { service, fake, state } = await mount()
    await expect(service.saveTelegram({ settings: settings(), token: 'not a token' })).rejects.toThrow(/BotFather/)
    await service.saveTelegram({ settings: settings(), token: TOKEN })
    await until(async () => (await service.overview()).telegram.status.phase === 'running')
    const overview = await service.overview()
    expect(overview.telegram.tokenSet).toBe(true)
    expect(overview.telegram.status.username).toBe('shoehorn_test_bot')
    expect(fake.calls.map(call => call.method)).toEqual(expect.arrayContaining(['getMe', 'deleteWebhook', 'setMyCommands']))

    fake.message(42, 'hello')
    await until(() => fake.sent(42).length === 1)
    expect(fake.sent(42)[0]).toContain('This bot is private')
    fake.message(42, 'hello again')
    fake.message(42, '/start 000000')
    await new Promise(resolve => setTimeout(resolve, 500))
    expect(fake.sent(42)).toHaveLength(1)  // told once, and a wrong code pairs nothing

    const pairing = await service.telegramPairingCode()
    expect(pairing.code).toMatch(/^\d{6}$/)
    expect(pairing.link).toBe(`https://t.me/shoehorn_test_bot?start=${pairing.code}`)
    fake.message(42, `/start ${pairing.code}`)
    await until(async () => (await service.overview()).telegram.settings.allowedChats.includes('42'))
    await until(() => fake.sent(42).some(text => text.startsWith('Paired.')))
    expect((await service.overview()).telegram.chats).toEqual([{ chatId: '42', name: 'Ana' }])

    fake.message(42, '/status')
    await until(() => fake.sent(42).some(text => text.includes('No chat yet')))
    fake.message(42, '/stop')
    await until(() => fake.sent(42).some(text => text.includes('Nothing is running')))
    // No folder is set, so a plain message cannot start a chat; the bot says what is missing.
    fake.message(42, 'do something')
    await until(() => fake.sent(42).some(text => text.includes('folder')))

    // A button from a chat that is not paired is refused.
    fake.button(99, 'a:xyz:y')
    await until(() => fake.calls.some(call => call.method === 'answerCallbackQuery' && call.body.text === 'Not allowed.'))

    // The offset survives in the state file, so a restart does not replay these updates.
    expect(JSON.parse(readFileSync(state, 'utf8')).offset).toBeGreaterThan(5)

    await service.removeTelegramChat({ chatId: '42' })
    expect((await service.overview()).telegram.settings.allowedChats).toEqual([])
  }, 30_000)

  it('removes a stored token', async () => {
    const { service } = await mount()
    await service.saveTelegram({ settings: settings(), token: TOKEN })
    expect((await service.overview()).telegram.tokenSet).toBe(true)
    await service.saveTelegram({ settings: settings({ enabled: false }), token: null })
    const overview = await service.overview()
    expect(overview.telegram.tokenSet).toBe(false)
    expect(overview.telegram.status.phase).toBe('off')
  })

  it('stays off without a token and reports it', async () => {
    const { service } = await mount()
    await service.saveTelegram({ settings: settings() })
    expect((await service.overview()).telegram.status.phase).toBe('needs-token')
    await service.saveTelegram({ settings: settings({ enabled: false }) })
    expect((await service.overview()).telegram.status.phase).toBe('off')
  })
})
