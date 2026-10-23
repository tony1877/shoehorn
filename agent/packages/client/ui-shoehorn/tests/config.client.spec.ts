import { describe, expect, it } from 'vitest'
import { CONFIG_GLOBAL, shoehornBaseUrl } from '../src/config.ts'
import { apply } from '../src/index.ts'

describe('shoehorn base URL', () => {
  it('keeps the same-host default for empty, absent or invalid values', () => {
    for (const value of [undefined, null, '', '   ', 42, 'not a url', 'ftp://host:8090', 'file:///C:/x']) {
      expect(shoehornBaseUrl(value)).toBe('')
    }
  })

  it('normalizes http(s) URLs: trims, drops trailing slashes and query', () => {
    expect(shoehornBaseUrl(' http://192.168.1.50:8090/ ')).toBe('http://192.168.1.50:8090')
    expect(shoehornBaseUrl('https://engine.lan:8443/shoehorn//?x=1')).toBe('https://engine.lan:8443/shoehorn')
  })
})

describe('host injection', () => {
  function inject(config?: { url?: string }) {
    let listener: ((table: { push(entry: unknown): void }) => void) | undefined
    const ctx = { on: (_name: string, cb: typeof listener) => { listener = cb } }
    apply(ctx as never, config)
    const entries: unknown[] = []
    listener?.({ push: (entry) => entries.push(entry) })
    return entries
  }

  it('publishes the validated URL as a page global', () => {
    expect(inject({ url: 'http://10.0.0.5:8090/' })).toEqual([
      { kind: 'global', name: CONFIG_GLOBAL, value: { url: 'http://10.0.0.5:8090' } },
    ])
  })

  it('publishes an empty URL when unconfigured, keeping the same-host default', () => {
    expect(inject()).toEqual([{ kind: 'global', name: CONFIG_GLOBAL, value: { url: '' } }])
  })
})
