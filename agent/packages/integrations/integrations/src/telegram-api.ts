/**
 * Telegram Bot API client (the subset the bot uses) and the conversion of the
 * agent's Markdown answers into Telegram's HTML subset, split under the
 * message size limit without breaking code blocks.
 */

/** Telegram rejects messages longer than this (characters after entity parsing). */
export const MESSAGE_LIMIT = 4096

/** Source characters per piece: headroom for the HTML tags the conversion adds. */
const PIECE_LIMIT = 3500

/** One error answer from the Bot API. */
export class TelegramError extends Error {
  constructor(readonly code: number, readonly description: string, readonly retryAfter?: number) {
    super(`Telegram ${code}: ${description}`)
    this.name = 'TelegramError'
  }
}

export interface TelegramUser {
  id: number
  is_bot: boolean
  first_name: string
  username?: string
}

export interface TelegramChat {
  id: number
  type: 'private' | 'group' | 'supergroup' | 'channel'
  title?: string
  username?: string
  first_name?: string
}

export interface TelegramPhotoSize {
  file_id: string
  width: number
  height: number
  file_size?: number
}

export interface TelegramMessage {
  message_id: number
  chat: TelegramChat
  from?: TelegramUser
  text?: string
  caption?: string
  photo?: TelegramPhotoSize[]
  document?: { file_id: string; file_name?: string; mime_type?: string; file_size?: number }
  voice?: unknown
  sticker?: unknown
}

export interface TelegramCallbackQuery {
  id: string
  from: TelegramUser
  data?: string
  message?: TelegramMessage
}

export interface TelegramUpdate {
  update_id: number
  message?: TelegramMessage
  callback_query?: TelegramCallbackQuery
}

export interface InlineButton {
  text: string
  callback_data: string
}

/** Minimal Bot API client over fetch. */
export class TelegramApi {
  constructor(private readonly token: string, private readonly base: string = 'https://api.telegram.org') {}

  /**
   * Call one method.
   * @throws {TelegramError} when Telegram answers `ok: false`.
   */
  async call<T>(method: string, params: Record<string, unknown> = {}, signal?: AbortSignal): Promise<T> {
    const response = await fetch(`${this.base}/bot${this.token}/${method}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(params),
      ...signal === undefined ? {} : { signal },
    })
    let body: { ok: boolean; result?: T; error_code?: number; description?: string; parameters?: { retry_after?: number } }
    try {
      body = await response.json() as typeof body
    } catch {
      throw new TelegramError(response.status, `unreadable answer (HTTP ${response.status})`)
    }
    if (!body.ok) throw new TelegramError(body.error_code ?? response.status, body.description ?? 'unknown error', body.parameters?.retry_after)
    return body.result as T
  }

  /** Download a file the bot received. */
  async download(fileId: string, maxBytes: number, signal?: AbortSignal): Promise<Uint8Array> {
    const file = await this.call<{ file_path?: string; file_size?: number }>('getFile', { file_id: fileId }, signal)
    if (file.file_path === undefined) throw new Error('Telegram returned no file path')
    if (file.file_size !== undefined && file.file_size > maxBytes) throw new Error('the file is too large')
    const response = await fetch(`${this.base}/file/bot${this.token}/${file.file_path}`, signal === undefined ? {} : { signal })
    if (!response.ok) throw new Error(`file download failed (HTTP ${response.status})`)
    const bytes = new Uint8Array(await response.arrayBuffer())
    if (bytes.byteLength > maxBytes) throw new Error('the file is too large')
    return bytes
  }

  /**
   * Send Markdown text as one or more HTML messages; a piece Telegram cannot
   * parse is sent again as plain text.
   * @returns the id of the last message sent.
   */
  async sendMarkdown(chatId: number | string, markdown: string, extra: Record<string, unknown> = {}): Promise<number> {
    let last = 0
    const pieces = splitMarkdown(markdown)
    for (const [index, piece] of pieces.entries()) {
      const tail = index === pieces.length - 1 ? extra : {}
      try {
        last = (await this.call<TelegramMessage>('sendMessage', {
          chat_id: chatId,
          text: markdownToHtml(piece),
          parse_mode: 'HTML',
          link_preview_options: { is_disabled: true },
          ...tail,
        })).message_id
      } catch (error) {
        if (!(error instanceof TelegramError) || error.code !== 400) throw error
        last = (await this.call<TelegramMessage>('sendMessage', { chat_id: chatId, text: piece.slice(0, MESSAGE_LIMIT), ...tail })).message_id
      }
    }
    return last
  }
}

/** Escape text for Telegram HTML. */
export function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/** Inline Markdown (outside code blocks) to Telegram HTML. */
function inlineToHtml(text: string): string {
  // Inline code first, so its content is not treated as emphasis.
  const parts = text.split(/(`[^`\n]+`)/)
  return parts.map((part) => {
    if (part.length > 1 && part.startsWith('`') && part.endsWith('`')) return `<code>${escapeHtml(part.slice(1, -1))}</code>`
    let html = escapeHtml(part)
    html = html.replace(/^#{1,6}\s+(.+)$/gm, '<b>$1</b>')
    html = html.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
    html = html.replace(/(^|[^\w*])\*([^*\n]+)\*(?=[^\w*]|$)/g, '$1<i>$2</i>')
    html = html.replace(/~~([^~\n]+)~~/g, '<s>$1</s>')
    html = html.replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, (_m, label: string, url: string) => `<a href="${url.replace(/"/g, '&quot;')}">${label}</a>`)
    html = html.replace(/^(\s*)[-*]\s+/gm, '$1• ')
    return html
  }).join('')
}

/** Markdown (the subset models write) to Telegram's HTML. */
export function markdownToHtml(markdown: string): string {
  const out: string[] = []
  const fence = /```([\w+-]*)\n?([\s\S]*?)(?:```|$)/g
  let at = 0
  for (let match = fence.exec(markdown); match !== null; match = fence.exec(markdown)) {
    out.push(inlineToHtml(markdown.slice(at, match.index)))
    const language = match[1] ?? ''
    const code = (match[2] ?? '').replace(/\n$/, '')
    out.push(language === ''
      ? `<pre>${escapeHtml(code)}</pre>`
      : `<pre><code class="language-${escapeHtml(language)}">${escapeHtml(code)}</code></pre>`)
    at = fence.lastIndex
    if (match[0].length === 0) break
  }
  out.push(inlineToHtml(markdown.slice(at)))
  return out.join('').trim()
}

/**
 * Split Markdown into pieces of at most PIECE_LIMIT characters at line
 * boundaries; a code block cut in two is closed and reopened, so every piece
 * converts on its own.
 */
export function splitMarkdown(markdown: string): string[] {
  const text = markdown.trim() === '' ? '(empty answer)' : markdown
  const pieces: string[] = []
  let current = ''
  let open: string | undefined  // the fence line of a code block open at the end of `current`
  const flush = (): void => {
    if (current === '') return
    pieces.push(open === undefined ? current : `${current}\n\`\`\``)
    current = open === undefined ? '' : `${open}\n`
  }
  for (const rawLine of text.split('\n')) {
    // A single overlong line is cut hard.
    const lines = rawLine.length > PIECE_LIMIT - 20 ? rawLine.match(new RegExp(`.{1,${PIECE_LIMIT - 20}}`, 'g')) ?? [rawLine] : [rawLine]
    for (const line of lines) {
      if (current.length + line.length + 1 > PIECE_LIMIT) flush()
      current += current === '' || current.endsWith('\n') ? line : `\n${line}`
      if (/^\s*```/.test(line)) open = open === undefined ? line.trim() : undefined
    }
  }
  if (current.trim() !== '' && current.trim() !== open) pieces.push(current)
  return pieces
}
