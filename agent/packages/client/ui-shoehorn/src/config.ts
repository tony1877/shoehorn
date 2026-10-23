/** Where the dsh web client reaches shoehorn (status chip, dashboard, sensors). */

/** Host configuration for the shoehorn panel. */
export interface Config {
  /**
   * Base URL of the shoehorn server, for example `http://192.168.1.50:8090` when shoehorn runs on a dedicated
   * machine. Empty: port 8090 on the machine that served the dsh page (shoehorn and dsh on one PC).
   */
  url?: string
}

/** Name of the page global the host writes and the browser half reads. */
export const CONFIG_GLOBAL = '__DSH_SHOEHORN_CONFIG__'

/**
 * Validated base URL: an absolute http(s) URL without a trailing slash, or '' for the same-host default.
 * @param value - configured URL, possibly absent or malformed.
 * @returns the normalized URL or ''.
 */
export function shoehornBaseUrl(value: unknown): string {
  if (typeof value !== 'string' || value.trim() === '') return ''
  try {
    const url = new URL(value.trim())
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return ''
    return url.origin + url.pathname.replace(/\/+$/, '')
  } catch {
    return ''
  }
}
