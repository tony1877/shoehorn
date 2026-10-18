/** Integrations settings section: MCP servers, inbound webhooks, notification targets. */

import type {} from '@deepseek-ai/dsh-client-locale/client'
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type {} from '@deepseek-ai/dsh-client-ui-settings/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import { IntegrationsSection, type IntegrationsApi, type IntegrationsSectionInjected } from './IntegrationsSection.tsx'
import { en, zh, type IntegrationsLocaleKey } from './locales.ts'

export type { IntegrationsApi, IntegrationsSectionInjected, IntegrationsSectionProps } from './IntegrationsSection.tsx'
export type { IntegrationsLocaleKey } from './locales.ts'

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap {
    /** Integrations settings copy. */
    'settings.integrations': IntegrationsLocaleKey
  }
}

/** Dictionary namespace owned by this plugin. */
export const NS = 'settings.integrations'

/** Services required by the section registration and its Remote face. */
export const inject = ['slots', 'locale', 'remote', 'remote.integrations']

type Result<T> = { ok: true; value: T } | { ok: false; error: { code: string; message: string } }

/** Unwrap one Remote result into its value or a readable error. */
async function unwrap<T>(call: Promise<Result<T>>): Promise<T> {
  const result = await call
  if (!result.ok) throw new Error(result.error.message)
  return result.value
}

/** Contribute the Integrations section to Settings. */
export function apply(ctx: ClientContext): void {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'ui-integrations: dictionaries')
  const t = ctx.locale.bind(NS)
  const remote = ctx.remote.integrations
  const api: IntegrationsApi = {
    overview: () => unwrap(remote.overview()),
    saveServer: (settings, secrets) => unwrap(remote.saveServer({ settings, ...secrets === undefined ? {} : { secrets } })),
    removeServer: id => unwrap(remote.removeServer({ id })),
    testServer: (settings, secrets) => unwrap(remote.testServer({ settings, ...secrets === undefined ? {} : { secrets } })),
    restartServer: id => unwrap(remote.restartServer({ id })),
    authorizeServer: (id, origin) => unwrap(remote.authorizeServer({ id, origin })),
    signOutServer: id => unwrap(remote.signOutServer({ id })),
    saveWebhook: settings => unwrap(remote.saveWebhook({ settings })),
    removeWebhook: id => unwrap(remote.removeWebhook({ id })),
    rotateWebhookSecret: id => unwrap(remote.rotateWebhookSecret({ id })),
    replayDelivery: id => unwrap(remote.replayDelivery({ id })),
    saveNotification: (settings, url) => unwrap(remote.saveNotification({ settings, ...url === undefined ? {} : { url } })),
    removeNotification: id => unwrap(remote.removeNotification({ id })),
    testNotification: id => unwrap(remote.testNotification({ id })),
    setPublicBaseUrl: url => unwrap(remote.setPublicBaseUrl({ url })),
  }
  const injected = (): IntegrationsSectionInjected => ({ api })

  // Ordered after Models (10) and Agent presets (20), before Plugins.
  ctx.slots.inject('settings.section', () => ctx.slots.register({
    name: 'settings.section',
    id: 'integrations',
    order: 25,
    label: () => t('nav'),
    locale: NS,
    inject: injected,
  }, IntegrationsSection))
}
