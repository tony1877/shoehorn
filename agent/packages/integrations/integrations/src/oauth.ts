/**
 * OAuth for remote MCP servers. The provider persists client registration,
 * tokens, and the PKCE verifier in the server's credential record, so a
 * sign-in survives restarts and the MCP SDK refreshes tokens on its own.
 *
 * A sign-in runs in two halves: `start` returns the authorization URL the
 * browser opens; the provider's callback route finishes it with the code.
 */

import { randomBytes } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import { auth, type OAuthClientProvider } from '@modelcontextprotocol/client'
import type { OAuthClientInformationMixed, OAuthClientMetadata, OAuthTokens } from '@modelcontextprotocol/client'
import { modifySecrets, readSecrets } from './secrets.ts'

/** Callback path on this Host. */
export const OAUTH_CALLBACK_PATH = '/integrations/oauth/callback'

/** How long a started sign-in waits for its callback. */
const PENDING_TTL_MS = 10 * 60_000

/** Provider whose state lives in one server's credential record. */
export class StoredOAuthProvider implements OAuthClientProvider {
  /** Authorization URL captured when the SDK asks to redirect. */
  authorizationUrl: URL | undefined

  constructor(
    private readonly ctx: Context,
    private readonly serverId: string,
    private readonly redirect: string | undefined,
    private readonly stateValue?: string,
  ) {}

  get redirectUrl(): string | undefined {
    return this.redirect
  }

  get clientMetadata(): OAuthClientMetadata {
    return {
      client_name: 'shoehorn agent',
      redirect_uris: this.redirect === undefined ? [] : [this.redirect],
      grant_types: ['authorization_code', 'refresh_token'],
      response_types: ['code'],
      token_endpoint_auth_method: 'none',
    }
  }

  state(): string {
    return this.stateValue ?? randomBytes(16).toString('hex')
  }

  async clientInformation(): Promise<OAuthClientInformationMixed | undefined> {
    const grant = (await readSecrets(this.ctx, 'mcp', this.serverId)).oauth
    // A registration made for a different redirect URL cannot complete this one.
    if (grant?.redirectUrl !== undefined && this.redirect !== undefined && grant.redirectUrl !== this.redirect) return undefined
    return grant?.clientInformation
  }

  async saveClientInformation(clientInformation: OAuthClientInformationMixed): Promise<void> {
    await this.patch({ clientInformation, ...this.redirect === undefined ? {} : { redirectUrl: this.redirect } })
  }

  async tokens(): Promise<OAuthTokens | undefined> {
    return (await readSecrets(this.ctx, 'mcp', this.serverId)).oauth?.tokens
  }

  async saveTokens(tokens: OAuthTokens): Promise<void> {
    await this.patch({ tokens })
  }

  redirectToAuthorization(authorizationUrl: URL): void {
    this.authorizationUrl = authorizationUrl
  }

  async saveCodeVerifier(codeVerifier: string): Promise<void> {
    await this.patch({ codeVerifier })
  }

  async codeVerifier(): Promise<string> {
    const verifier = (await readSecrets(this.ctx, 'mcp', this.serverId)).oauth?.codeVerifier
    if (verifier === undefined) throw new Error(`no sign-in is in progress for MCP server "${this.serverId}"`)
    return verifier
  }

  async invalidateCredentials(scope: 'all' | 'client' | 'tokens' | 'verifier' | 'discovery'): Promise<void> {
    await modifySecrets(this.ctx, 'mcp', this.serverId, (current) => {
      const oauth = { ...current.oauth }
      if (scope === 'all' || scope === 'client') delete oauth.clientInformation
      if (scope === 'all' || scope === 'tokens') delete oauth.tokens
      if (scope === 'all' || scope === 'verifier') delete oauth.codeVerifier
      return { ...current, oauth }
    })
  }

  private async patch(fields: Record<string, unknown>): Promise<void> {
    await modifySecrets(this.ctx, 'mcp', this.serverId, current => ({ ...current, oauth: { ...current.oauth, ...fields } }))
  }
}

interface PendingSignIn {
  serverId: string
  serverUrl: string
  redirectUrl: string
  expiresAt: number
}

/** Sign-ins started and not yet finished, keyed by OAuth `state`. */
export class OAuthSignIns {
  private readonly pending = new Map<string, PendingSignIn>()

  constructor(private readonly ctx: Context) {}

  /**
   * Start (or silently complete) a sign-in.
   * @param serverId - server whose record holds the grant.
   * @param serverUrl - the MCP endpoint.
   * @param origin - origin the browser reaches this Host at.
   * @returns `authorized` when stored tokens already work, else the URL to open.
   */
  async start(serverId: string, serverUrl: string, origin: string): Promise<{ status: 'authorized' } | { status: 'redirect'; url: string }> {
    const redirectUrl = new URL(OAUTH_CALLBACK_PATH, origin).href
    const state = randomBytes(16).toString('hex')
    const provider = new StoredOAuthProvider(this.ctx, serverId, redirectUrl, state)
    const result = await auth(provider, { serverUrl })
    if (result === 'AUTHORIZED') return { status: 'authorized' }
    if (provider.authorizationUrl === undefined) throw new Error('the server did not provide an authorization URL')
    this.sweep()
    this.pending.set(state, { serverId, serverUrl, redirectUrl, expiresAt: Date.now() + PENDING_TTL_MS })
    return { status: 'redirect', url: provider.authorizationUrl.href }
  }

  /**
   * Finish a sign-in from its callback parameters.
   * @returns the server id whose grant is now stored.
   */
  async finish(state: string, code: string): Promise<string> {
    this.sweep()
    const pending = this.pending.get(state)
    if (pending === undefined) throw new Error('this sign-in link has expired or was already used; start it again from Settings')
    this.pending.delete(state)
    const provider = new StoredOAuthProvider(this.ctx, pending.serverId, pending.redirectUrl, state)
    const result = await auth(provider, { serverUrl: pending.serverUrl, authorizationCode: code })
    if (result !== 'AUTHORIZED') throw new Error('the server did not accept the authorization code')
    return pending.serverId
  }

  private sweep(): void {
    const now = Date.now()
    for (const [state, entry] of this.pending) if (entry.expiresAt < now) this.pending.delete(state)
  }
}
