/**
 * Telegram bot: talk to the agent from Telegram.
 *
 * - Long polling (no public URL needed). The update offset is persisted, so a
 *   restart neither loses nor repeats messages.
 * - Only paired chats are served: the settings page shows a one-time code,
 *   and `/start <code>` (or the deep link) from a chat pairs it.
 * - Each paired chat has its own agent chat (`/new` starts another), resumed
 *   from its log after a restart. Answers come back as formatted messages;
 *   "typing…" shows while the agent works. Photos reach a vision model.
 * - Approvals and questions for a Telegram chat are answered with buttons in
 *   Telegram. Other chats' notifications can be sent to paired chats too.
 */

import { randomBytes, randomInt } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { ApprovalOutcome } from '@deepseek-ai/dsh-user-approval'
import type { AskUserQuestionAnswer, AskUserQuestionItem } from '@deepseek-ai/dsh-user-questions'
import { answerSince, eventCount, findChat, sendToChat, startChat, toolCallArguments } from './chats.ts'
import { modifySecrets, readSecrets } from './secrets.ts'
import { StateFile } from './state.ts'
import { escapeHtml, TelegramApi, TelegramError, type InlineButton, type TelegramCallbackQuery, type TelegramMessage, type TelegramUpdate, type TelegramUser } from './telegram-api.ts'
import type { NotificationMessage } from './notify.ts'
import type { TelegramChatView, TelegramSettings, TelegramStatus } from './types.ts'

/** Long-poll wait per getUpdates call (seconds). */
const POLL_SECONDS = 30

/** How long a pairing code stays valid. */
const PAIRING_TTL_MS = 10 * 60_000

/** Largest photo or image document accepted. */
const MAX_IMAGE_BYTES = 10 * 1024 * 1024

/** Telegram shows "typing…" for about 5 s per chat action. */
const TYPING_EVERY_MS = 4_500

const HELP = [
  '<b>Talk to the agent</b>: just send a message (text or a photo).',
  '/new - start a new chat',
  '/stop - stop what the agent is doing',
  '/status - show the current chat',
  '/help - this message',
].join('\n')

interface ChatBinding {
  sessionId: string
  agentPreset: string
  title: string
}

interface TelegramState {
  /** getUpdates offset: the next update id to fetch. */
  offset: number
  /** chat id -> the agent chat it talks to */
  chats: Record<string, ChatBinding>
  /** chat id -> how it is shown on the settings page */
  names: Record<string, string>
}

interface PendingApproval {
  chatId: string
  messageId: number
  resolve: (outcome: ApprovalOutcome) => void
  text: string
}

interface PendingQuestion {
  chatId: string
  messageId: number
  question: AskUserQuestionItem
  selected: Set<string>
  resolve: (answer: { selected: string[]; custom?: string }) => void
  text: string
}

/** Image type from the file's first bytes; `undefined` for anything but JPEG, PNG, GIF or WebP. */
export function sniffImage(bytes: Uint8Array): ImageMediaType | undefined {
  const at = (offset: number, ...values: number[]): boolean => values.every((value, index) => bytes[offset + index] === value)
  if (at(0, 0xFF, 0xD8, 0xFF)) return 'image/jpeg'
  if (at(0, 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)) return 'image/png'
  if (at(0, 0x47, 0x49, 0x46, 0x38)) return 'image/gif'
  if (at(0, 0x52, 0x49, 0x46, 0x46) && at(8, 0x57, 0x45, 0x42, 0x50)) return 'image/webp'
  return undefined
}

function chatName(chat: TelegramMessage['chat'], from?: TelegramUser): string {
  if (chat.title !== undefined) return chat.title
  if (chat.username !== undefined) return `@${chat.username}`
  return chat.first_name ?? from?.first_name ?? String(chat.id)
}

/** The Telegram bot service. */
export class TelegramBot {
  private settings: TelegramSettings | undefined
  private sessionCtx: Context | undefined
  private api: TelegramApi | undefined
  private loop: AbortController | undefined
  private readonly state: StateFile<TelegramState>
  private status: TelegramStatus = { phase: 'off' }
  private pairing: { code: string; expiresAt: number } | undefined
  /** session id -> chat id, for every bound chat */
  private readonly bySession = new Map<string, string>()
  /** chats whose message is being handled, so they run in order */
  private readonly chatQueues = new Map<string, Promise<void>>()
  private readonly typing = new Map<string, NodeJS.Timeout>()
  /** session id -> log length when its current turn started, so only that turn's answer is relayed */
  private readonly turnStart = new Map<string, number>()
  private readonly approvals = new Map<string, PendingApproval>()
  private readonly questions = new Map<string, PendingQuestion>()
  /** chat id -> question awaiting a typed answer */
  private readonly typedAnswer = new Map<string, string>()
  /** unknown chats already told how to pair (once per hour) */
  private readonly toldUnknown = new Map<string, number>()

  constructor(
    private readonly ctx: Context,
    private readonly pairChat: (chatId: string) => Promise<void>,
    stateDirectory?: string,
    private readonly apiBase?: string,
  ) {
    this.state = new StateFile<TelegramState>('telegram.json', { offset: 0, chats: {}, names: {} }, stateDirectory)
    for (const [chatId, binding] of Object.entries(this.state.get().chats)) this.bySession.set(binding.sessionId, chatId)
    ctx.effect(() => () => this.stop(), 'integrations.telegram.stop')
    ctx.on('agent/status', ({ agent, status }) => this.onStatus(agent, status))
    ctx.on('approval/request', (request, next) => {
      const chatId = this.bySession.get(String(request.agent.session.id))
      if (chatId === undefined || !this.answersHere()) return next()
      return this.askApproval(chatId, request.agent, request.toolName, request.reason, request.callId, request.signal)
    }, { prepend: true })
    ctx.on('user-questions/request', (request, next) => {
      const chatId = request.agent === undefined ? undefined : this.bySession.get(String(request.agent.session.id))
      if (chatId === undefined || !this.answersHere()) return next()
      return this.askQuestions(chatId, request.questions, request.signal)
    }, { prepend: true })
  }

  /** Provide the session services (inside their injection). */
  mount(sessionCtx: Context): void {
    this.sessionCtx = sessionCtx
    sessionCtx.effect(() => () => {
      this.sessionCtx = undefined
    }, 'integrations.telegram.services')
  }

  /** Apply settings: start, restart, or stop polling. */
  async apply(settings: TelegramSettings): Promise<void> {
    const previous = this.settings
    this.settings = settings
    const token = (await readSecrets(this.ctx, 'channel', 'telegram')).secret
    const wanted = settings.enabled && token !== undefined
    const restart = wanted && (this.loop === undefined || previous?.enabled !== settings.enabled || this.api === undefined)
    if (!wanted) {
      this.stop()
      this.status = { phase: settings.enabled ? 'needs-token' : 'off' }
      return
    }
    if (restart) this.start(token)
  }

  /** Store a new bot token and restart with it. */
  async setToken(token: string | null): Promise<void> {
    await modifySecrets(this.ctx, 'channel', 'telegram', current => ({ ...current, secret: token ?? undefined }))
    // Update ids belong to one bot: a new token starts from that bot's own queue.
    await this.state.update(current => ({ ...current, offset: 0 }))
    this.stop()
    if (this.settings !== undefined) await this.apply(this.settings)
  }

  /** A fresh pairing code (shown on the settings page), valid for ten minutes. */
  newPairingCode(): { code: string; link?: string } {
    this.pairing = { code: String(randomInt(100_000, 1_000_000)), expiresAt: Date.now() + PAIRING_TTL_MS }
    const username = this.status.username
    return { code: this.pairing.code, ...username === undefined ? {} : { link: `https://t.me/${username}?start=${this.pairing.code}` } }
  }

  /** Forget a paired chat's binding (the settings page removes it from the allow list). */
  async forgetChat(chatId: string): Promise<void> {
    const binding = this.state.get().chats[chatId]
    if (binding !== undefined) this.bySession.delete(binding.sessionId)
    await this.state.update(current => {
      const chats = { ...current.chats }
      const names = { ...current.names }
      delete chats[chatId]
      delete names[chatId]
      return { ...current, chats, names }
    })
  }

  view(): { status: TelegramStatus; chats: TelegramChatView[]; pairing?: { code: string; expiresAt: number } } {
    const state = this.state.get()
    const chats = (this.settings?.allowedChats ?? []).map(chatId => ({
      chatId,
      name: state.names[chatId] ?? chatId,
      ...state.chats[chatId] === undefined ? {} : { chatTitle: state.chats[chatId]!.title, sessionId: state.chats[chatId]!.sessionId },
    }))
    const pairing = this.pairing !== undefined && this.pairing.expiresAt > Date.now() ? this.pairing : undefined
    return { status: this.status, chats, ...pairing === undefined ? {} : { pairing } }
  }

  /** Send a notification about another chat to every paired chat that wants it. */
  async notify(message: NotificationMessage): Promise<void> {
    const settings = this.settings
    if (this.api === undefined || settings === undefined || !settings.notifyEvents.includes(message.event)) return
    if (message.event === 'finished' && (message.turnSeconds ?? 0) < settings.minTurnSeconds) return
    // A chat that lives in Telegram already gets its answers and buttons there.
    if (this.bySession.has(message.sessionId)) return
    const heading = { finished: '✅ Chat finished', approval: '⚠️ Approval needed', question: '❓ Question for you' }[message.event]
    const html = [`<b>${heading}: ${escapeHtml(message.title)}</b>`, escapeHtml(message.text), message.link === undefined ? '' : `<a href="${message.link.replace(/"/g, '&quot;')}">Open in dsh</a>`]
      .filter(line => line !== '').join('\n')
    for (const chatId of settings.allowedChats) {
      try {
        await this.api.call('sendMessage', { chat_id: chatId, text: html.slice(0, 4000), parse_mode: 'HTML', link_preview_options: { is_disabled: true } })
      } catch (error) {
        this.ctx.logger.warn(`integrations: Telegram notification to ${chatId} failed: ${String(error)}`)
      }
    }
  }

  private answersHere(): boolean {
    return this.api !== undefined && this.settings?.answerInTelegram !== false
  }

  private stop(): void {
    this.loop?.abort()
    this.loop = undefined
    this.api = undefined
    for (const timer of this.typing.values()) clearInterval(timer)
    this.typing.clear()
    for (const pending of this.approvals.values()) pending.resolve('unavailable')
    this.approvals.clear()
  }

  private start(token: string): void {
    this.stop()
    const api = new TelegramApi(token, this.apiBase)
    const loop = new AbortController()
    this.api = api
    this.loop = loop
    this.status = { phase: 'connecting' }
    void this.poll(api, loop.signal)
  }

  private async poll(api: TelegramApi, signal: AbortSignal): Promise<void> {
    let backoff = 2_000
    try {
      const me = await api.call<TelegramUser>('getMe', {}, signal)
      this.status = { phase: 'running', ...me.username === undefined ? {} : { username: me.username } }
      // Long polling and a webhook exclude each other; a leftover webhook would make every poll fail.
      await api.call('deleteWebhook', { drop_pending_updates: false }, signal)
      await api.call('setMyCommands', {
        commands: [
          { command: 'new', description: 'Start a new chat' },
          { command: 'stop', description: 'Stop what the agent is doing' },
          { command: 'status', description: 'Show the current chat' },
          { command: 'help', description: 'How to use this bot' },
        ],
      }, signal)
    } catch (error) {
      if (signal.aborted) return
      this.status = { phase: 'error', error: error instanceof TelegramError && error.code === 401 ? 'Telegram rejected the bot token.' : String(error instanceof Error ? error.message : error) }
      if (error instanceof TelegramError && error.code === 401) return
      // A network failure at start: retry the whole start later.
      await new Promise(resolve => setTimeout(resolve, 15_000).unref())
      if (!signal.aborted) void this.poll(api, signal)
      return
    }
    while (!signal.aborted) {
      try {
        const updates = await api.call<TelegramUpdate[]>('getUpdates', {
          offset: this.state.get().offset,
          timeout: POLL_SECONDS,
          allowed_updates: ['message', 'callback_query'],
        }, AbortSignal.any([signal, AbortSignal.timeout((POLL_SECONDS + 15) * 1000)]))
        backoff = 2_000
        if (this.status.phase !== 'running') this.status = { ...this.status, phase: 'running' }
        for (const update of updates) {
          await this.state.update(current => ({ ...current, offset: update.update_id + 1 }))
          this.dispatch(update)
        }
        this.status = { ...this.status, lastPollAt: Date.now() }
      } catch (error) {
        if (signal.aborted) return
        if (error instanceof TelegramError && error.code === 401) {
          this.status = { phase: 'error', error: 'Telegram rejected the bot token.' }
          return
        }
        if (error instanceof TelegramError && error.code === 409) {
          this.status = { ...this.status, phase: 'error', error: 'Another program is reading this bot\'s messages (only one may poll a bot at a time).' }
        }
        const wait = error instanceof TelegramError && error.retryAfter !== undefined ? error.retryAfter * 1000 : backoff
        backoff = Math.min(backoff * 2, 60_000)
        await new Promise(resolve => setTimeout(resolve, wait).unref())
      }
    }
  }

  private dispatch(update: TelegramUpdate): void {
    if (update.callback_query !== undefined) {
      void this.onCallback(update.callback_query).catch(error => this.ctx.logger.warn(`integrations: Telegram button failed: ${String(error)}`))
      return
    }
    const message = update.message
    if (message === undefined) return
    const chatId = String(message.chat.id)
    // One message at a time per chat, in order; chats do not wait for each other.
    const previous = this.chatQueues.get(chatId) ?? Promise.resolve()
    const next = previous.then(() => this.onMessage(message)).catch((error: unknown) => {
      this.ctx.logger.warn(`integrations: Telegram message failed: ${String(error)}`)
      void this.reply(chatId, `Sorry, that failed: ${escapeHtml(error instanceof Error ? error.message : String(error))}`)
    })
    this.chatQueues.set(chatId, next)
    void next.finally(() => {
      if (this.chatQueues.get(chatId) === next) this.chatQueues.delete(chatId)
    })
  }

  private allowed(chatId: string): boolean {
    return this.settings?.allowedChats.includes(chatId) === true
  }

  private async reply(chatId: string, html: string, extra: Record<string, unknown> = {}): Promise<number | undefined> {
    if (this.api === undefined) return undefined
    const sent = await this.api.call<TelegramMessage>('sendMessage', { chat_id: chatId, text: html, parse_mode: 'HTML', link_preview_options: { is_disabled: true }, ...extra })
    return sent.message_id
  }

  private async onMessage(message: TelegramMessage): Promise<void> {
    const chatId = String(message.chat.id)
    const text = (message.text ?? message.caption ?? '').trim()
    const command = /^\/(\w+)(?:@\w+)?(?:\s+(.*))?$/s.exec(text)

    if (!this.allowed(chatId)) {
      const code = command !== null && (command[1] === 'start' || command[1] === 'pair') ? command[2]?.trim() : undefined
      if (code !== undefined && this.pairing !== undefined && this.pairing.expiresAt > Date.now() && code === this.pairing.code) {
        this.pairing = undefined
        await this.state.update(current => ({ ...current, names: { ...current.names, [chatId]: chatName(message.chat, message.from) } }))
        await this.pairChat(chatId)
        await this.reply(chatId, `Paired. ${HELP}`)
        return
      }
      const told = this.toldUnknown.get(chatId)
      if (told === undefined || Date.now() - told > 3_600_000) {
        this.toldUnknown.set(chatId, Date.now())
        await this.reply(chatId, 'This bot is private. To connect this chat, open dsh → Settings → Integrations → Telegram, create a pairing code, and send <code>/start CODE</code> here.')
      }
      return
    }

    // A typed answer to a pending question.
    const waiting = this.typedAnswer.get(chatId)
    if (waiting !== undefined && command === null && text !== '') {
      const pending = this.questions.get(waiting)
      if (pending !== undefined) {
        this.typedAnswer.delete(chatId)
        this.questions.delete(waiting)
        await this.editText(pending.chatId, pending.messageId, `${pending.text}\n\n<i>Answered: ${escapeHtml(text)}</i>`)
        pending.resolve({ selected: [], custom: text })
        return
      }
    }

    if (command !== null) {
      switch (command[1]) {
        case 'start':
        case 'help':
          await this.reply(chatId, HELP)
          return
        case 'new':
          await this.unbind(chatId)
          await this.reply(chatId, 'OK. Your next message starts a new chat.')
          return
        case 'stop': {
          const agent = await this.boundAgent(chatId)
          if (agent === undefined || agent.status !== 'running') {
            await this.reply(chatId, 'Nothing is running.')
            return
          }
          agent.cancel({ kind: 'user' }, { keepInbox: true })
          await this.reply(chatId, 'Stopped.')
          return
        }
        case 'status': {
          const binding = this.state.get().chats[chatId]
          if (binding === undefined) {
            await this.reply(chatId, 'No chat yet: send a message to start one.')
            return
          }
          const agent = this.sessionCtx?.agents.get(binding.sessionId as never)
          await this.reply(chatId, `<b>${escapeHtml(this.sessionCtx?.get('sessionTitle')?.get(agent?.session as never)?.title ?? binding.title)}</b>\n${agent?.status === 'running' ? 'Working…' : 'Idle'}`)
          return
        }
        default:
          // Unknown commands go to the agent as ordinary text.
          break
      }
    }

    const content: ContentBlock[] = []
    if (text !== '') content.push({ type: 'text', text })
    const image = await this.image(chatId, message)
    if (image === 'refused') return
    if (image !== undefined) content.push(image)
    if (content.length === 0) {
      await this.reply(chatId, 'I can read text and photos. Voice notes, stickers and other files are not supported yet.')
      return
    }

    const settings = this.settings
    const bound = this.state.get().chats[chatId] !== undefined
    if (settings === undefined || (!bound && settings.workspacePath.trim() === '')) {
      await this.reply(chatId, 'Set the folder Telegram chats work in first: dsh → Settings → Integrations → Telegram.')
      return
    }
    const sessionCtx = this.sessionCtx
    if (sessionCtx === undefined) {
      await this.reply(chatId, 'The agent is not available right now.')
      return
    }
    const source = { kind: 'user' as const }
    const existing = await this.boundAgent(chatId)
    if (existing !== undefined) {
      sendToChat(existing, content, source)
      return
    }
    if (settings.workspacePath.trim() === '') {
      await this.reply(chatId, 'Set the folder Telegram chats work in first: dsh → Settings → Integrations → Telegram.')
      return
    }
    const title = text === '' ? 'Telegram photo' : text.split('\n')[0]!.slice(0, 60)
    const created = await startChat(sessionCtx, settings, title, content, source, AbortSignal.timeout(120_000))
    this.bySession.set(String(created.sessionId), chatId)
    // The first turn may have started before this chat was bound.
    if (created.agent.status === 'running') {
      this.turnStart.set(String(created.sessionId), 0)
      this.startTyping(chatId)
    }
    await this.state.update(current => ({
      ...current,
      chats: { ...current.chats, [chatId]: { sessionId: String(created.sessionId), agentPreset: settings.agentPreset, title } },
    }))
  }

  /** A photo (or image document) as an image block; `refused` when it cannot be used (the user was told). */
  private async image(chatId: string, message: TelegramMessage): Promise<ContentBlock | 'refused' | undefined> {
    const photo = message.photo?.at(-1)
    const document = message.document?.mime_type?.startsWith('image/') === true ? message.document : undefined
    const fileId = photo?.file_id ?? document?.file_id
    if (fileId === undefined || this.api === undefined) return undefined
    const attachments = this.ctx.get('attachments')
    if (attachments === undefined) {
      await this.reply(chatId, 'Images are not enabled on this agent.')
      return 'refused'
    }
    try {
      const data = await this.api.download(fileId, MAX_IMAGE_BYTES)
      // The bytes decide the type: Telegram's own label is missing for photos and can be wrong for documents.
      const mediaType = sniffImage(data)
      if (mediaType === undefined) {
        await this.reply(chatId, 'Send images as JPEG, PNG, WebP or GIF.')
        return 'refused'
      }
      const [attachment] = await attachments.saveImages([{ data, mediaType, name: document?.file_name ?? 'telegram-photo' }])
      return { type: 'image', attachment: attachment! }
    } catch (error) {
      await this.reply(chatId, `Could not use that image: ${escapeHtml(error instanceof Error ? error.message : String(error))}`)
      return 'refused'
    }
  }

  private async boundAgent(chatId: string): Promise<Agent | undefined> {
    const binding = this.state.get().chats[chatId]
    if (binding === undefined || this.sessionCtx === undefined) return undefined
    const agent = await findChat(this.sessionCtx, binding.sessionId, binding.agentPreset)
    if (agent === undefined) await this.unbind(chatId)
    return agent
  }

  private async unbind(chatId: string): Promise<void> {
    const binding = this.state.get().chats[chatId]
    if (binding === undefined) return
    this.bySession.delete(binding.sessionId)
    this.stopTyping(chatId)
    await this.state.update(current => {
      const chats = { ...current.chats }
      delete chats[chatId]
      return { ...current, chats }
    })
  }

  private onStatus(agent: Agent, status: 'idle' | 'running'): void {
    const chatId = this.bySession.get(String(agent.session.id))
    if (chatId === undefined || this.api === undefined) return
    const sessionId = String(agent.session.id)
    if (status === 'running') {
      this.turnStart.set(sessionId, eventCount(agent))
      this.startTyping(chatId)
      return
    }
    this.stopTyping(chatId)
    const answer = answerSince(agent, this.turnStart.get(sessionId) ?? 0)
    this.turnStart.delete(sessionId)
    // A turn that wrote nothing new (stopped before answering, or only ran tools) relays nothing.
    if (answer === undefined) return
    void this.api.sendMarkdown(chatId, answer === '' ? '_(finished without a written answer)_' : answer)
      .catch(error => this.ctx.logger.warn(`integrations: Telegram answer to ${chatId} failed: ${String(error)}`))
  }

  private startTyping(chatId: string): void {
    if (this.typing.has(chatId) || this.api === undefined) return
    const api = this.api
    const send = (): void => {
      void api.call('sendChatAction', { chat_id: chatId, action: 'typing' }).catch(() => {})
    }
    send()
    const timer = setInterval(send, TYPING_EVERY_MS)
    timer.unref()
    this.typing.set(chatId, timer)
  }

  private stopTyping(chatId: string): void {
    const timer = this.typing.get(chatId)
    if (timer !== undefined) clearInterval(timer)
    this.typing.delete(chatId)
  }

  private async editText(chatId: string, messageId: number, html: string): Promise<void> {
    try {
      await this.api?.call('editMessageText', { chat_id: chatId, message_id: messageId, text: html, parse_mode: 'HTML', reply_markup: { inline_keyboard: [] } })
    } catch {
      // The message may be too old to edit; the answer still counts.
    }
  }

  /** Ask in Telegram whether a tool may run; resolves with the button pressed. */
  private async askApproval(chatId: string, agent: Agent, toolName: string, reason: string | undefined, callId: string | undefined, signal: AbortSignal | undefined): Promise<ApprovalOutcome> {
    if (this.api === undefined) return 'unavailable'
    const token = randomBytes(6).toString('base64url')
    const args = toolCallArguments(agent, callId)
    // A call through the connected-services proxy is shown as the tool it runs.
    let shownName = toolName
    if (toolName === 'mcp_call' && args !== undefined) {
      try {
        const target = (JSON.parse(args) as { tool?: unknown }).tool
        if (typeof target === 'string') shownName = target
      } catch { /* malformed arguments: keep the proxy's name */ }
    }
    const text = [
      `⚠️ <b>Allow ${escapeHtml(shownName)}?</b>`,
      reason === undefined ? '' : escapeHtml(reason),
      args === undefined ? '' : `<pre>${escapeHtml(args.length > 1500 ? `${args.slice(0, 1500)}…` : args)}</pre>`,
    ].filter(line => line !== '').join('\n')
    const buttons: InlineButton[][] = [[
      { text: '✅ Allow', callback_data: `a:${token}:y` },
      { text: '❌ Deny', callback_data: `a:${token}:n` },
    ]]
    let messageId: number
    try {
      messageId = (await this.api.call<TelegramMessage>('sendMessage', { chat_id: chatId, text, parse_mode: 'HTML', reply_markup: { inline_keyboard: buttons } })).message_id
    } catch {
      return 'unavailable'
    }
    return await new Promise<ApprovalOutcome>((resolve) => {
      const finish = (outcome: ApprovalOutcome): void => {
        if (!this.approvals.delete(token)) return
        signal?.removeEventListener('abort', onAbort)
        resolve(outcome)
      }
      const onAbort = (): void => {
        void this.editText(chatId, messageId, `${text}\n\n<i>No longer needed.</i>`)
        finish('cancelled')
      }
      this.approvals.set(token, { chatId, messageId, text, resolve: finish })
      if (signal?.aborted === true) onAbort()
      else signal?.addEventListener('abort', onAbort, { once: true })
    })
  }

  /** Ask each question in Telegram, one after another. */
  private async askQuestions(chatId: string, items: readonly AskUserQuestionItem[], signal: AbortSignal | undefined): Promise<AskUserQuestionAnswer> {
    const answers: AskUserQuestionAnswer['answers'] = []
    for (const question of items) {
      const answer = await this.askQuestion(chatId, question, signal)
      answers.push({ id: question.id, selected: answer.selected, ...answer.custom === undefined ? {} : { custom: answer.custom } })
    }
    return { answers }
  }

  private async askQuestion(chatId: string, question: AskUserQuestionItem, signal: AbortSignal | undefined): Promise<{ selected: string[]; custom?: string }> {
    if (this.api === undefined) throw new Error('Telegram is not connected')
    const token = randomBytes(6).toString('base64url')
    const options = question.options ?? []
    const text = [
      `❓ <b>${escapeHtml(question.header ?? 'Question')}</b>`,
      escapeHtml(question.question),
      question.detail === undefined ? '' : escapeHtml(question.detail.length > 2500 ? `${question.detail.slice(0, 2500)}…` : question.detail),
      options.length === 0 ? '<i>Reply with your answer.</i>' : '<i>Pick an option, or reply with your own answer.</i>',
    ].filter(line => line !== '').join('\n\n')
    const keyboard = (selected: Set<string>): InlineButton[][] => [
      ...options.map((option, index) => [{ text: `${question.multiSelect === true && selected.has(option.label) ? '☑️ ' : ''}${option.label}`.slice(0, 60), callback_data: `q:${token}:${index}` }]),
      ...question.multiSelect === true ? [[{ text: 'Done', callback_data: `q:${token}:done` }]] : [],
    ]
    const messageId = (await this.api.call<TelegramMessage>('sendMessage', {
      chat_id: chatId,
      text: text.slice(0, 4000),
      parse_mode: 'HTML',
      ...options.length === 0 ? {} : { reply_markup: { inline_keyboard: keyboard(new Set()) } },
    })).message_id
    return await new Promise((resolve, reject) => {
      const onAbort = (): void => {
        this.questions.delete(token)
        if (this.typedAnswer.get(chatId) === token) this.typedAnswer.delete(chatId)
        void this.editText(chatId, messageId, `${text}\n\n<i>No longer needed.</i>`)
        reject(new Error('the question was withdrawn'))
      }
      this.questions.set(token, {
        chatId,
        messageId,
        question,
        selected: new Set(),
        text,
        resolve: (answer) => {
          signal?.removeEventListener('abort', onAbort)
          resolve(answer)
        },
      })
      this.typedAnswer.set(chatId, token)
      if (signal?.aborted === true) onAbort()
      else signal?.addEventListener('abort', onAbort, { once: true })
      // Keep the keyboard renderer for multi-select toggles.
      this.keyboards.set(token, keyboard)
    })
  }

  private readonly keyboards = new Map<string, (selected: Set<string>) => InlineButton[][]>()

  private async onCallback(query: TelegramCallbackQuery): Promise<void> {
    const api = this.api
    if (api === undefined) return
    const chatId = query.message === undefined ? undefined : String(query.message.chat.id)
    const ack = (text?: string): Promise<unknown> => api.call('answerCallbackQuery', { callback_query_id: query.id, ...text === undefined ? {} : { text } }).catch(() => {})
    if (chatId === undefined || !this.allowed(chatId)) {
      await ack('Not allowed.')
      return
    }
    const [kind, token, value] = (query.data ?? '').split(':')
    if (kind === 'a' && token !== undefined) {
      const pending = this.approvals.get(token)
      if (pending === undefined || pending.chatId !== chatId) {
        await ack('This request is no longer open.')
        return
      }
      const allow = value === 'y'
      await this.editText(chatId, pending.messageId, `${pending.text}\n\n<i>${allow ? '✅ Allowed' : '❌ Denied'} by ${escapeHtml(query.from.first_name)}</i>`)
      pending.resolve(allow ? 'allowed-once' : 'rejected')
      await ack(allow ? 'Allowed' : 'Denied')
      return
    }
    if (kind === 'q' && token !== undefined) {
      const pending = this.questions.get(token)
      if (pending === undefined || pending.chatId !== chatId) {
        await ack('This question is no longer open.')
        return
      }
      const options = pending.question.options ?? []
      if (value === 'done') {
        this.complete(token, pending, [...pending.selected])
        await ack()
        return
      }
      const option = options[Number(value)]
      if (option === undefined) {
        await ack()
        return
      }
      if (pending.question.multiSelect === true) {
        if (pending.selected.has(option.label)) pending.selected.delete(option.label)
        else pending.selected.add(option.label)
        const keyboard = this.keyboards.get(token)
        if (keyboard !== undefined) {
          await api.call('editMessageReplyMarkup', { chat_id: chatId, message_id: pending.messageId, reply_markup: { inline_keyboard: keyboard(pending.selected) } }).catch(() => {})
        }
        await ack()
        return
      }
      this.complete(token, pending, [option.label])
      await ack(option.label.slice(0, 180))
    }
  }

  private complete(token: string, pending: PendingQuestion, selected: string[]): void {
    this.questions.delete(token)
    this.keyboards.delete(token)
    if (this.typedAnswer.get(pending.chatId) === token) this.typedAnswer.delete(pending.chatId)
    void this.editText(pending.chatId, pending.messageId, `${pending.text}\n\n<i>Answered: ${escapeHtml(selected.join(', ') || '(nothing selected)')}</i>`)
    pending.resolve({ selected })
  }
}
