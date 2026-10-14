/**
 * dsh plugin: a keyless `WebSearchProvider` backed by a local SearXNG instance
 * (GET /search?format=json). Mounted host-wide from ~/.dsh/cordis.patch.yml and
 * selected there with `web.searchProvider: searxng`.
 *
 * Deliberately dependency-free (plain JS, no workspace imports) so it loads by
 * absolute file URL from outside the dsh repo. Errors carry the same `code`
 * values as dsh's WebError so tool-web reports them the same way.
 *
 * SearXNG itself: C:\Tools\searxng, started by C:\Tools\bin\start-searxng.ps1.
 */

export const name = 'web-search-searxng'
export const inject = ['web']

const PROVIDER_ID = 'searxng'
const DEFAULT_BASE_URL = 'http://127.0.0.1:8888'
const START_HINT = 'start it with C:\\Tools\\bin\\start-searxng.ps1'

function webError(message, code, cause) {
  const error = new Error(message, cause === undefined ? undefined : { cause })
  error.code = code
  return error
}

function isAbortError(error) {
  return error?.name === 'AbortError'
}

/** Map one SearXNG result to dsh's WebSearchSource, or undefined without a URL. */
function mapResult(result) {
  if (typeof result?.url !== 'string' || result.url.length === 0) return undefined
  const source = { url: result.url }
  if (typeof result.title === 'string' && result.title.trim()) source.title = result.title.trim()
  if (typeof result.content === 'string' && result.content.trim()) source.snippet = result.content.trim()
  if (typeof result.publishedDate === 'string' && result.publishedDate) source.publishedAt = result.publishedDate
  return source
}

/** Direct answers / infobox text SearXNG sometimes returns, as optional `content`. */
function answerText(payload) {
  const parts = []
  for (const answer of payload.answers ?? []) {
    const text = typeof answer === 'string' ? answer : answer?.answer
    if (typeof text === 'string' && text.trim()) parts.push(text.trim())
  }
  for (const box of payload.infoboxes ?? []) {
    if (typeof box?.content === 'string' && box.content.trim()) {
      parts.push(box.infobox ? `${box.infobox}: ${box.content.trim()}` : box.content.trim())
    }
  }
  return parts.length > 0 ? parts.join('\n\n') : undefined
}

class SearxngSearchProvider {
  id = PROVIDER_ID

  constructor(options) {
    this.baseURL = options.baseURL.replace(/\/+$/, '')
    this.language = options.language
    this.categories = options.categories
    this.numResults = options.numResults
  }

  available() {
    return URL.canParse(this.baseURL)
  }

  async search(request, signal) {
    const url = new URL(`${this.baseURL}/search`)
    url.searchParams.set('q', request.query)
    url.searchParams.set('format', 'json')
    if (this.language) url.searchParams.set('language', this.language)
    if (this.categories) url.searchParams.set('categories', this.categories)

    let response
    try {
      response = await fetch(url, {
        headers: { accept: 'application/json' },
        ...signal !== undefined ? { signal } : {},
      })
    } catch (error) {
      if (isAbortError(error)) throw webError('SearXNG search aborted', 'WEB_ABORTED', error)
      throw webError(`SearXNG at ${this.baseURL} is not reachable (${START_HINT}): ${String(error)}`, 'WEB_PROVIDER_ERROR', error)
    }

    if (!response.ok) {
      const hint = response.status === 403 ? ' - JSON output must be enabled in settings.yml (search.formats: json)' : ''
      throw webError(`SearXNG returned HTTP ${response.status}${hint}`, 'WEB_PROVIDER_ERROR')
    }

    let payload
    try {
      payload = await response.json()
    } catch (error) {
      if (isAbortError(error)) throw webError('SearXNG search aborted', 'WEB_ABORTED', error)
      throw webError(`SearXNG returned an unreadable response: ${String(error)}`, 'WEB_PROVIDER_ERROR', error)
    }

    const seen = new Set()
    const sources = []
    for (const result of payload.results ?? []) {
      const source = mapResult(result)
      if (source === undefined || seen.has(source.url)) continue
      seen.add(source.url)
      sources.push(source)
    }
    const limit = request.maxResults ?? this.numResults
    const content = answerText(payload)
    // The web seam enforces maxResults and sets `truncated`; report false here.
    return {
      ...content !== undefined ? { content } : {},
      sources: limit !== undefined ? sources.slice(0, limit) : sources,
      truncated: limit !== undefined && sources.length > limit,
    }
  }
}

export function apply(ctx, config) {
  const options = config ?? {}
  ctx.web.registerSearchProvider(new SearxngSearchProvider({
    baseURL: options.baseURL ?? DEFAULT_BASE_URL,
    language: options.language,
    categories: options.categories,
    numResults: options.numResults,
  }))
}
