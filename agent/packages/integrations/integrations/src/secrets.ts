/**
 * Secret halves of integrations, kept as credential-store grant records under
 * the `integrations` scope: one record per MCP server, webhook endpoint, or
 * notification target. Settings hold everything else.
 */

import type { Context } from '@deepseek-ai/cordis'
import { credentialKey, type CredentialKey } from '@deepseek-ai/dsh-credentials'
import type { OAuthClientInformationMixed, OAuthTokens } from '@modelcontextprotocol/sdk/shared/auth.js'

/** OAuth state for one MCP server. */
export interface OAuthGrant {
  clientInformation?: OAuthClientInformationMixed
  tokens?: OAuthTokens
  codeVerifier?: string
  /** Redirect URL the client registered with; refreshes must present the same one. */
  redirectUrl?: string
}

/** Every secret an integration may hold; each kind uses its own subset. */
export interface IntegrationSecrets {
  env?: Record<string, string> | undefined
  headers?: Record<string, string> | undefined
  oauth?: OAuthGrant | undefined
  /** Webhook HMAC secret, notification signing secret, or a channel's bot token. */
  secret?: string | undefined
  /** Notification target URL. */
  url?: string | undefined
}

/** Which integration list a record belongs to. */
export type SecretKind = 'mcp' | 'hook' | 'notify' | 'channel'

function keyOf(kind: SecretKind, id: string): CredentialKey {
  return credentialKey('integrations', `${kind}-${id}`)
}

/** Read one integration's secrets; empty when none are stored. */
export async function readSecrets(ctx: Context, kind: SecretKind, id: string): Promise<IntegrationSecrets> {
  const record = await ctx.credentials.readRecord(keyOf(kind, id))
  if (record?.kind !== 'grant') return {}
  return (record.payload ?? {}) as IntegrationSecrets
}

/** Atomically rewrite one integration's secrets; an empty result deletes the record. */
export async function modifySecrets(
  ctx: Context,
  kind: SecretKind,
  id: string,
  mutate: (current: IntegrationSecrets) => IntegrationSecrets,
): Promise<IntegrationSecrets> {
  let next: IntegrationSecrets = {}
  let empty = false
  await ctx.credentials.modifyRecord(keyOf(kind, id), async (current) => {
    const payload = current?.kind === 'grant' ? (current.payload ?? {}) as IntegrationSecrets : {}
    next = mutate(payload)
    empty = Object.values(next).every(value => value === undefined)
    // `undefined` leaves the record as it is; an emptied record is deleted below instead.
    return empty ? undefined : { kind: 'grant', payload: JSON.parse(JSON.stringify(next)) as unknown }
  })
  if (empty) await ctx.credentials.deleteRecord(keyOf(kind, id))
  return next
}

/** Delete one integration's secrets. */
export async function deleteSecrets(ctx: Context, kind: SecretKind, id: string): Promise<void> {
  await ctx.credentials.deleteRecord(keyOf(kind, id))
}
