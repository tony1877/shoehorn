/**
 * shoehorn status plugin, node half: embeds the configured shoehorn address in product pages for the browser half
 * (which ships through `exports["./client"]`).
 */
import type { Context } from '@deepseek-ai/cordis'
import { CONFIG_GLOBAL, shoehornBaseUrl, type Config } from './config.ts'

export type { Config } from './config.ts'

/**
 * The one dsh-host-webserver event this plugin uses, typed locally: the package's single client tsconfig cannot
 * reference the host webserver's types (see packages/client/shortcuts for the split-tsconfig form).
 */
interface IndexInjectTable {
  push(entry: { kind: 'global'; name: string; value: unknown }): void
}

/**
 * Embed the validated shoehorn base URL in product pages.
 * @param ctx - Host context serving browser pages.
 * @param config - where shoehorn runs; empty keeps the same-host default.
 */
export function apply(ctx: Context, config: Config = {}): void {
  const url = shoehornBaseUrl(config.url)
  const host = ctx as unknown as { on(name: 'webserver/index-inject', listener: (table: IndexInjectTable) => void): void }
  host.on('webserver/index-inject', (table) => {
    table.push({ kind: 'global', name: CONFIG_GLOBAL, value: { url } })
  })
}
