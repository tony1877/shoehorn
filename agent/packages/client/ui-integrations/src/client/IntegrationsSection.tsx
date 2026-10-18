import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import type {
  IntegrationsOverview,
  McpPreset,
  McpServerSettings,
  McpServerView,
  McpTestResult,
  NotificationEvent,
  NotificationSettings,
  NotificationView,
  TelegramSettings,
  WebhookDeliveryView,
  WebhookPreset,
  WebhookSettings,
  WebhookView,
} from '@deepseek-ai/dsh-integrations/types'
import { Button, Input, StateDot, type StateDotState } from '@deepseek-ai/dsh-client-ui-primitives'
import type { InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import css from './IntegrationsSection.module.css'

/** Remote operations the section calls; each rejects with a readable message. */
export interface IntegrationsApi {
  overview(): Promise<IntegrationsOverview>
  saveServer(settings: McpServerSettings, secrets?: { env?: Record<string, string> | null; headers?: Record<string, string> | null }): Promise<void>
  removeServer(id: string): Promise<void>
  testServer(settings: McpServerSettings, secrets?: { env?: Record<string, string> | null; headers?: Record<string, string> | null }): Promise<McpTestResult>
  restartServer(id: string): Promise<void>
  authorizeServer(id: string, origin: string): Promise<{ status: 'authorized' } | { status: 'redirect'; url: string }>
  signOutServer(id: string): Promise<void>
  saveWebhook(settings: WebhookSettings): Promise<{ secret: string } | null>
  removeWebhook(id: string): Promise<void>
  rotateWebhookSecret(id: string): Promise<{ secret: string }>
  replayDelivery(id: string): Promise<WebhookDeliveryView>
  saveNotification(settings: NotificationSettings, url?: string): Promise<{ secret: string } | null>
  removeNotification(id: string): Promise<void>
  testNotification(id: string): Promise<{ ok: boolean; detail: string }>
  setPublicBaseUrl(url: string): Promise<void>
  saveTelegram(settings: TelegramSettings, token?: string | null): Promise<void>
  telegramPairingCode(): Promise<{ code: string; link?: string }>
  removeTelegramChat(chatId: string): Promise<void>
}

/** Registration-side face. */
export interface IntegrationsSectionInjected {
  api: IntegrationsApi
}

/** Full component props assembled by the Settings slot renderer. */
export type IntegrationsSectionProps =
  PropsRuntime<'settings.section'>
  & PropsLocale<'settings.integrations'>
  & InjectFace<IntegrationsSectionInjected>

type Translate = IntegrationsSectionProps['t']

const PHASE_DOT: Record<McpServerView['phase'], StateDotState | undefined> = {
  disabled: undefined,
  connecting: 'ongoing',
  connected: 'done',
  reconnecting: 'warning',
  failed: 'error',
  'needs-auth': 'warning',
  stopped: undefined,
}

const SERVER_DEFAULTS: Omit<McpServerSettings, 'id'> = {
  enabled: true,
  transport: 'stdio',
  command: '',
  args: [],
  cwd: '',
  url: '',
  includeTools: [],
  exposure: 'deferred',
  approval: 'ask-writes',
  auth: 'none',
  toolCallTimeoutMs: 60_000,
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function lines(text: string): string[] {
  return text.split('\n').map(line => line.trim()).filter(line => line !== '')
}

/** Parse `NAME=value` or `Name: value` lines into a map; `null` when the box was left empty. */
function pairs(text: string, separator: '=' | ':'): Record<string, string> | undefined {
  const out: Record<string, string> = {}
  for (const line of lines(text)) {
    const at = line.indexOf(separator)
    const name = (at === -1 ? line : line.slice(0, at)).trim()
    if (name === '') continue
    out[name] = at === -1 ? '' : line.slice(at + 1).trim()
  }
  return Object.keys(out).length === 0 ? undefined : out
}

function Field({ label, hint, children }: { label: string; hint?: string | undefined; children: ReactNode }): ReactNode {
  return (
    <label className={css.field}>
      <span className={css.label}>{label}</span>
      {children}
      {hint === undefined ? null : <span className={css.hint}>{hint}</span>}
    </label>
  )
}

function Select<T extends string>({ value, options, onChange, label }: {
  value: T
  options: ReadonlyArray<{ value: T; label: string }>
  onChange: (value: T) => void
  label: string
}): ReactNode {
  return (
    <select className={css.select} aria-label={label} value={value} onChange={event => onChange(event.target.value as T)}>
      {options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
    </select>
  )
}

function TextArea({ value, onChange, rows = 3, placeholder }: { value: string; onChange: (value: string) => void; rows?: number; placeholder?: string }): ReactNode {
  return <textarea className={css.textarea} rows={rows} value={value} placeholder={placeholder} onChange={event => onChange(event.target.value)} spellCheck={false} />
}

// ---- MCP server editor ----

interface ServerDraft {
  settings: McpServerSettings
  argsText: string
  envText: string
  headersText: string
  isNew: boolean
  requiredEnv: string[]
  requiredHeaders: string[]
}

function draftFrom(settings: McpServerSettings, isNew: boolean, preset?: McpPreset): ServerDraft {
  return {
    settings,
    argsText: settings.args.join('\n'),
    envText: (preset?.requiredEnv ?? []).map(name => `${name}=`).join('\n'),
    headersText: (preset?.requiredHeaders ?? []).map(name => `${name}: ${name === 'Authorization' ? 'Bearer ' : ''}`).join('\n'),
    isNew,
    requiredEnv: preset?.requiredEnv ?? [],
    requiredHeaders: preset?.requiredHeaders ?? [],
  }
}

function ServerEditor({ draft, view, api, t, onDone }: {
  draft: ServerDraft
  view: McpServerView | undefined
  api: IntegrationsApi
  t: Translate
  onDone: (saved: boolean) => void
}): ReactNode {
  const [state, setState] = useState(draft)
  const [busy, setBusy] = useState<'save' | 'test' | undefined>()
  const [result, setResult] = useState<{ tone: 'ok' | 'error'; text: string } | undefined>()
  const [tested, setTested] = useState<McpTestResult | undefined>()
  const settings = state.settings
  const set = (patch: Partial<McpServerSettings>): void => setState(current => ({ ...current, settings: { ...current.settings, ...patch } }))
  const stdio = settings.transport === 'stdio'

  const collect = (): { settings: McpServerSettings; secrets: { env?: Record<string, string>; headers?: Record<string, string> } } => {
    const env = stdio ? pairs(state.envText, '=') : undefined
    const headers = stdio ? undefined : pairs(state.headersText, ':')
    return {
      settings: { ...settings, args: stdio ? lines(state.argsText) : [], id: settings.id.trim() },
      secrets: { ...env === undefined ? {} : { env }, ...headers === undefined ? {} : { headers } },
    }
  }

  const test = async (): Promise<void> => {
    setBusy('test')
    setResult(undefined)
    try {
      const { settings: next, secrets } = collect()
      const outcome = await api.testServer(next, secrets)
      setTested(outcome)
      if (outcome.ok) {
        const server = outcome.serverName === undefined ? '' : ` to ${outcome.serverName}${outcome.serverVersion === undefined ? '' : ` ${outcome.serverVersion}`}`
        setResult({ tone: 'ok', text: t('testOk', { server, count: outcome.tools.length }) })
      } else {
        setResult({ tone: 'error', text: outcome.needsAuth ? t('testNeedsAuth') : t('testFailed', { message: outcome.error }) })
      }
    } catch (error) {
      setResult({ tone: 'error', text: t('testFailed', { message: message(error) }) })
    } finally {
      setBusy(undefined)
    }
  }

  const save = async (): Promise<void> => {
    setBusy('save')
    setResult(undefined)
    try {
      const { settings: next, secrets } = collect()
      await api.saveServer(next, secrets)
      onDone(true)
    } catch (error) {
      setResult({ tone: 'error', text: message(error) })
      setBusy(undefined)
    }
  }

  const toolNames = (tested?.ok === true ? tested.tools : view?.tools ?? []).map(tool => tool.rawName)
  const include = new Set(settings.includeTools)
  const missing = [
    ...state.requiredEnv.filter(name => !(view?.envNames ?? []).includes(name) && (pairs(state.envText, '=')?.[name] ?? '') === ''),
    ...state.requiredHeaders.filter(name => !(view?.headerNames ?? []).includes(name) && (pairs(state.headersText, ':')?.[name] ?? '').replace(/^Bearer\s*$/, '') === ''),
  ]

  return (
    <div className={css.editor}>
      <div className={css.grid}>
        <Field label={t('fieldId')} hint={t('fieldIdHint')}>
          <Input value={settings.id} disabled={!state.isNew} onChange={event => set({ id: event.target.value.toLowerCase() })} />
        </Field>
        <Field label={t('fieldTransport')}>
          <Select
            label={t('fieldTransport')}
            value={settings.transport}
            onChange={transport => set({ transport, ...transport === 'stdio' ? { auth: 'none' } : {} })}
            options={(['stdio', 'streamable-http', 'sse'] as const).map(value => ({ value, label: t(`transport_${value}`) }))}
          />
        </Field>
      </div>
      {stdio
        ? (
            <>
              <Field label={t('fieldCommand')}>
                <Input value={settings.command} placeholder="npx" onChange={event => set({ command: event.target.value })} />
              </Field>
              <Field label={t('fieldArgs')}>
                <TextArea value={state.argsText} onChange={argsText => setState(current => ({ ...current, argsText }))} />
              </Field>
              <Field label={t('fieldCwd')}>
                <Input value={settings.cwd} onChange={event => set({ cwd: event.target.value })} />
              </Field>
              <Field label={t('fieldEnv')} hint={[t('secretHint'), view !== undefined && view.envNames.length > 0 ? t('storedNames', { names: view.envNames.join(', ') }) : ''].join(' ')}>
                <TextArea value={state.envText} placeholder="API_KEY=" onChange={envText => setState(current => ({ ...current, envText }))} />
              </Field>
            </>
          )
        : (
            <>
              <Field label={t('fieldUrl')}>
                <Input value={settings.url} placeholder="https://example.com/mcp" onChange={event => set({ url: event.target.value })} />
              </Field>
              <Field label={t('fieldHeaders')} hint={[t('secretHint'), view !== undefined && view.headerNames.length > 0 ? t('storedNames', { names: view.headerNames.join(', ') }) : ''].join(' ')}>
                <TextArea value={state.headersText} placeholder="Authorization: Bearer …" onChange={headersText => setState(current => ({ ...current, headersText }))} />
              </Field>
              <Field label={t('fieldAuth')}>
                <Select
                  label={t('fieldAuth')}
                  value={settings.auth}
                  onChange={auth => set({ auth })}
                  options={(['none', 'oauth'] as const).map(value => ({ value, label: t(`auth_${value}`) }))}
                />
              </Field>
            </>
          )}
      <div className={css.grid}>
        <Field label={t('fieldExposure')}>
          <Select
            label={t('fieldExposure')}
            value={settings.exposure}
            onChange={exposure => set({ exposure })}
            options={(['deferred', 'native'] as const).map(value => ({ value, label: t(`exposure_${value}`) }))}
          />
        </Field>
        <Field label={t('fieldApproval')}>
          <Select
            label={t('fieldApproval')}
            value={settings.approval}
            onChange={approval => set({ approval })}
            options={(['ask-writes', 'ask', 'allow'] as const).map(value => ({ value, label: t(`approval_${value}`) }))}
          />
        </Field>
      </div>
      {toolNames.length === 0
        ? null
        : (
            <Field label={t('fieldTools')}>
              <div className={css.toolPicker}>
                <label className={css.check}>
                  <input type="checkbox" checked={include.size === 0} onChange={() => set({ includeTools: [] })} />
                  {t('allTools')}
                </label>
                {toolNames.map(name => (
                  <label key={name} className={css.check}>
                    <input
                      type="checkbox"
                      checked={include.size === 0 || include.has(name)}
                      onChange={(event) => {
                        const base = include.size === 0 ? new Set(toolNames) : new Set(include)
                        if (event.target.checked) base.add(name)
                        else base.delete(name)
                        set({ includeTools: base.size === toolNames.length ? [] : toolNames.filter(tool => base.has(tool)) })
                      }}
                    />
                    <code>{name}</code>
                  </label>
                ))}
              </div>
            </Field>
          )}
      {missing.length === 0 ? null : <p className={css.warn}>{t('requiredSecrets', { names: missing.join(', ') })}</p>}
      {result === undefined ? null : <p className={result.tone === 'ok' ? css.ok : css.error}>{result.text}</p>}
      <div className={css.actions}>
        <Button variant="primary" size="sm" disabled={busy !== undefined || settings.id.trim() === ''} onClick={() => { void save() }}>
          {busy === 'save' ? t('saving') : t('save')}
        </Button>
        <Button variant="outline" size="sm" disabled={busy !== undefined} onClick={() => { void test() }}>
          {busy === 'test' ? t('testing') : t('test')}
        </Button>
        <Button variant="ghost" size="sm" disabled={busy !== undefined} onClick={() => onDone(false)}>{t('cancel')}</Button>
      </div>
    </div>
  )
}

function ServerCard({ view, api, t, onChanged, onEdit }: {
  view: McpServerView
  api: IntegrationsApi
  t: Translate
  onChanged: () => void
  onEdit: () => void
}): ReactNode {
  const [open, setOpen] = useState(false)
  const [error, setError] = useState<string | undefined>()
  const settings = view.settings
  const run = async (action: () => Promise<unknown>): Promise<void> => {
    setError(undefined)
    try {
      await action()
    } catch (failure) {
      setError(message(failure))
    }
    onChanged()
  }
  const signIn = async (): Promise<void> => {
    const popup = window.open('', '_blank')
    try {
      const result = await api.authorizeServer(settings.id, window.location.origin)
      if (result.status === 'redirect') {
        if (popup === null) window.location.assign(result.url)
        else popup.location.href = result.url
      } else {
        popup?.close()
      }
    } catch (failure) {
      popup?.close()
      setError(message(failure))
    }
    onChanged()
  }
  const dot = PHASE_DOT[view.phase]
  const offered = view.tools.filter(tool => !tool.excluded)
  return (
    <li className={css.card}>
      <div className={css.cardHead}>
        <button type="button" className={css.cardTitle} onClick={() => setOpen(!open)} aria-expanded={open}>
          {dot === undefined ? <span className={css.dotOff} /> : <StateDot state={dot} size={8} />}
          <strong>{settings.id}</strong>
          <span className={css.meta}>
            {t(`phase_${view.phase}`)}
            {view.tools.length === 0 ? '' : ` · ${t('toolsCount', { count: offered.length })}`}
            {settings.auth === 'oauth' && view.authorized ? ` · ${t('signedIn')}` : ''}
          </span>
        </button>
        <div className={css.cardActions}>
          <label className={css.check}>
            <input type="checkbox" checked={settings.enabled} onChange={event => { void run(() => api.saveServer({ ...settings, enabled: event.target.checked })) }} />
            {t('enabled')}
          </label>
          {settings.auth === 'oauth' && settings.transport !== 'stdio'
            ? view.authorized
              ? <Button size="sm" variant="ghost" onClick={() => { void run(() => api.signOutServer(settings.id)) }}>{t('signOut')}</Button>
              : <Button size="sm" variant="outline" onClick={() => { void signIn() }}>{t('signIn')}</Button>
            : null}
          <Button size="sm" variant="ghost" onClick={() => { void run(() => api.restartServer(settings.id)) }}>{t('reconnect')}</Button>
          <Button size="sm" variant="ghost" onClick={onEdit}>{t('edit')}</Button>
          <Button size="sm" variant="ghost" onClick={() => { void run(() => api.removeServer(settings.id)) }}>{t('remove')}</Button>
        </div>
      </div>
      {view.error === undefined ? null : <p className={css.error}>{view.error}</p>}
      {error === undefined ? null : <p className={css.error}>{error}</p>}
      {open
        ? (
            <div className={css.cardBody}>
              <p className={css.meta}>{settings.transport === 'stdio' ? [settings.command, ...settings.args].join(' ') : settings.url}</p>
              {view.calls === 0
                ? null
                : (
                    <p className={css.meta}>
                      {t('mcpStats', { calls: view.calls, failures: view.failures })}
                      {view.lastCallAt === undefined ? '' : ` · ${t('mcpLastCall', { time: new Date(view.lastCallAt).toLocaleTimeString() })}`}
                    </p>
                  )}
              {view.log.length === 0
                ? null
                : (
                    <details>
                      <summary className={css.meta}>{t('showLog', { count: view.log.length })}</summary>
                      <pre className={css.pre}>{view.log.join('\n')}</pre>
                    </details>
                  )}
              {view.tools.length === 0
                ? null
                : (
                    <>
                      <p className={css.meta}>{t('toolsShown', { shown: offered.length, count: view.tools.length })}</p>
                      <ul className={css.tools}>
                        {view.tools.map(tool => (
                          <li key={tool.rawName} className={tool.excluded ? css.toolExcluded : undefined}>
                            <code>{tool.rawName}</code>
                            {tool.readOnly === true ? <span className={css.badge}>{t('readOnly')}</span> : null}
                            {tool.destructive === true ? <span className={css.badgeWarn}>{t('destructive')}</span> : null}
                            {tool.description === '' ? null : <span className={css.toolDesc}>{tool.description.split('\n')[0]}</span>}
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
            </div>
          )
        : null}
    </li>
  )
}

function McpPanel({ overview, api, t, reload }: { overview: IntegrationsOverview; api: IntegrationsApi; t: Translate; reload: () => void }): ReactNode {
  const [editing, setEditing] = useState<{ draft: ServerDraft; key: number } | undefined>()
  const start = (draft: ServerDraft): void => setEditing({ draft, key: Date.now() })
  const taken = new Set(overview.servers.map(view => view.settings.id))
  const freeId = (base: string): string => {
    if (!taken.has(base)) return base
    for (let index = 2; ; index += 1) if (!taken.has(`${base}-${index}`)) return `${base}-${index}`
  }
  return (
    <section className={css.group}>
      <h3 className={css.groupTitle}>{t('mcpTitle')}</h3>
      <p className={css.intro}>{t('mcpIntro')}</p>
      {overview.servers.length === 0 ? <p className={css.meta}>{t('mcpEmpty')}</p> : null}
      <ul className={css.cards}>
        {overview.servers.map(view => (
          editing !== undefined && !editing.draft.isNew && editing.draft.settings.id === view.settings.id
            ? (
                <li key={view.settings.id} className={css.card}>
                  <ServerEditor key={editing.key} draft={editing.draft} view={view} api={api} t={t} onDone={() => { setEditing(undefined); reload() }} />
                </li>
              )
            : <ServerCard key={view.settings.id} view={view} api={api} t={t} onChanged={reload} onEdit={() => start(draftFrom(view.settings, false))} />
        ))}
      </ul>
      {editing?.draft.isNew === true
        ? (
            <div className={css.card}>
              <ServerEditor key={editing.key} draft={editing.draft} view={undefined} api={api} t={t} onDone={() => { setEditing(undefined); reload() }} />
            </div>
          )
        : (
            <div className={css.addRow}>
              <span className={css.label}>{t('addService')}</span>
              <div className={css.presets}>
                {overview.presets.map(preset => (
                  <button
                    key={preset.id}
                    type="button"
                    className={css.preset}
                    title={preset.description}
                    onClick={() => start(draftFrom({ ...preset.settings, id: freeId(preset.settings.id) }, true, preset))}
                  >
                    <strong>{preset.title}</strong>
                    <span>{preset.description}</span>
                  </button>
                ))}
                <button type="button" className={css.preset} onClick={() => start(draftFrom({ ...SERVER_DEFAULTS, id: freeId('server') }, true))}>
                  <strong>{t('custom')}</strong>
                </button>
              </div>
            </div>
          )}
    </section>
  )
}

// ---- Webhooks ----

function emptyHook(id: string, preset?: WebhookPreset): WebhookSettings {
  if (preset !== undefined) return { ...preset.settings, id, workspacePath: '' }
  return {
    id,
    enabled: true,
    title: '',
    authMode: 'hmac',
    signatureHeader: 'X-Signature-256',
    chatMode: 'new',
    matchPath: '',
    matchValues: [],
    workspacePath: '',
    promptTemplate: 'A webhook delivered this payload:\n\n{{body}}\n\nHandle it.',
    titleTemplate: '',
    agentPreset: 'minimal-compact',
    permissionPreset: 'workspace-write',
    maxBodyBytes: 256_000,
    ratePerMinute: 30,
  }
}

function HookEditor({ initial, isNew, api, t, onDone }: {
  initial: WebhookSettings
  isNew: boolean
  api: IntegrationsApi
  t: Translate
  onDone: (secret?: { id: string; secret: string }) => void
}): ReactNode {
  const [hook, setHook] = useState(initial)
  const [valuesText, setValuesText] = useState(initial.matchValues.join('\n'))
  const [error, setError] = useState<string | undefined>()
  const [busy, setBusy] = useState(false)
  const set = (patch: Partial<WebhookSettings>): void => setHook(current => ({ ...current, ...patch }))
  const save = async (): Promise<void> => {
    setBusy(true)
    setError(undefined)
    try {
      const next = { ...hook, id: hook.id.trim(), matchValues: lines(valuesText) }
      const created = await api.saveWebhook(next)
      onDone(created === null ? undefined : { id: next.id, secret: created.secret })
    } catch (failure) {
      setError(message(failure))
      setBusy(false)
    }
  }
  return (
    <div className={css.editor}>
      <div className={css.grid}>
        <Field label={t('fieldId')}>
          <Input value={hook.id} disabled={!isNew} onChange={event => set({ id: event.target.value.toLowerCase() })} />
        </Field>
        <Field label={t('fieldTitle')}>
          <Input value={hook.title} onChange={event => set({ title: event.target.value })} />
        </Field>
      </div>
      <Field label={t('fieldWorkspace')}>
        <Input value={hook.workspacePath} placeholder={'C:\\src\\project'} onChange={event => set({ workspacePath: event.target.value })} />
      </Field>
      <Field label={t('fieldPrompt')}>
        <TextArea rows={5} value={hook.promptTemplate} onChange={promptTemplate => set({ promptTemplate })} />
      </Field>
      <Field label={t('fieldChatTitle')}>
        <Input value={hook.titleTemplate} onChange={event => set({ titleTemplate: event.target.value })} />
      </Field>
      <div className={css.grid}>
        <Field label={t('fieldMatchPath')}>
          <Input value={hook.matchPath} placeholder="action" onChange={event => set({ matchPath: event.target.value })} />
        </Field>
        <Field label={t('fieldMatchValues')}>
          <TextArea rows={2} value={valuesText} onChange={setValuesText} />
        </Field>
      </div>
      <div className={css.grid}>
        <Field label={t('hookAuth')}>
          <Select
            label={t('hookAuth')}
            value={hook.authMode}
            onChange={authMode => set({ authMode, signatureHeader: authMode === 'token' ? 'Authorization' : 'X-Signature-256' })}
            options={(['hmac', 'token'] as const).map(value => ({ value, label: t(`hookAuth_${value}`) }))}
          />
        </Field>
        <Field label={t(`hookAuthHeader_${hook.authMode}`)}>
          <Input value={hook.signatureHeader} onChange={event => set({ signatureHeader: event.target.value })} />
        </Field>
        <Field label={t('hookChat')}>
          <Select
            label={t('hookChat')}
            value={hook.chatMode}
            onChange={chatMode => set({ chatMode })}
            options={(['new', 'same'] as const).map(value => ({ value, label: t(`hookChat_${value}`) }))}
          />
        </Field>
        <Field label={t('fieldRate')}>
          <Input type="number" value={String(hook.ratePerMinute)} onChange={event => set({ ratePerMinute: Math.max(1, Number(event.target.value) || 1) })} />
        </Field>
        <Field label={t('fieldAgentPreset')}>
          <Input value={hook.agentPreset} onChange={event => set({ agentPreset: event.target.value })} />
        </Field>
        <Field label={t('fieldPermissionPreset')}>
          <Input value={hook.permissionPreset} onChange={event => set({ permissionPreset: event.target.value })} />
        </Field>
      </div>
      {error === undefined ? null : <p className={css.error}>{error}</p>}
      <div className={css.actions}>
        <Button variant="primary" size="sm" disabled={busy || hook.id.trim() === '' || hook.workspacePath.trim() === ''} onClick={() => { void save() }}>{busy ? t('saving') : t('save')}</Button>
        <Button variant="ghost" size="sm" disabled={busy} onClick={() => onDone()}>{t('cancel')}</Button>
      </div>
    </div>
  )
}

function HooksPanel({ overview, api, t, reload }: { overview: IntegrationsOverview; api: IntegrationsApi; t: Translate; reload: () => void }): ReactNode {
  const [editing, setEditing] = useState<{ hook: WebhookSettings; isNew: boolean } | undefined>()
  const [secret, setSecret] = useState<{ id: string; secret: string } | undefined>()
  const [baseUrl, setBaseUrl] = useState(overview.publicBaseUrl)
  const [error, setError] = useState<string | undefined>()
  const base = (overview.publicBaseUrl || window.location.origin).replace(/\/+$/, '')
  const run = async (action: () => Promise<unknown>): Promise<void> => {
    setError(undefined)
    try {
      await action()
    } catch (failure) {
      setError(message(failure))
    }
    reload()
  }
  const header = (id: string): string => overview.webhooks.find(view => view.settings.id === id)?.settings.signatureHeader ?? 'X-Signature-256'
  return (
    <section className={css.group}>
      <h3 className={css.groupTitle}>{t('hooksTitle')}</h3>
      <p className={css.intro}>{t('hooksIntro')}</p>
      <Field label={t('baseUrl')} hint={t('baseUrlHint')}>
        <div className={css.inline}>
          <Input value={baseUrl} placeholder={window.location.origin} onChange={event => setBaseUrl(event.target.value)} />
          <Button size="sm" variant="outline" disabled={baseUrl === overview.publicBaseUrl} onClick={() => { void run(() => api.setPublicBaseUrl(baseUrl)) }}>{t('save')}</Button>
        </div>
      </Field>
      {secret === undefined
        ? null
        : (
            <div className={css.secretBox}>
              <p>{t(overview.webhooks.find(view => view.settings.id === secret.id)?.settings.authMode === 'token' ? 'newSecretShownToken' : 'newSecretShown', { header: header(secret.id) })}</p>
              <code className={css.secret}>{secret.secret}</code>
            </div>
          )}
      {error === undefined ? null : <p className={css.error}>{error}</p>}
      {overview.webhooks.length === 0 ? <p className={css.meta}>{t('hooksEmpty')}</p> : null}
      <ul className={css.cards}>
        {overview.webhooks.map((view: WebhookView) => (
          editing !== undefined && !editing.isNew && editing.hook.id === view.settings.id
            ? (
                <li key={view.settings.id} className={css.card}>
                  <HookEditor initial={view.settings} isNew={false} api={api} t={t} onDone={(created) => { setEditing(undefined); if (created !== undefined) setSecret(created); reload() }} />
                </li>
              )
            : (
                <li key={view.settings.id} className={css.card}>
                  <div className={css.cardHead}>
                    <div className={css.cardTitle}>
                      {view.settings.enabled ? <StateDot state="done" size={8} /> : <span className={css.dotOff} />}
                      <strong>{view.settings.title || view.settings.id}</strong>
                      <span className={css.meta}>{view.secretSet ? t('hookSecretSet') : t('hookSecretMissing')}</span>
                    </div>
                    <div className={css.cardActions}>
                      <label className={css.check}>
                        <input type="checkbox" checked={view.settings.enabled} onChange={event => { void run(() => api.saveWebhook({ ...view.settings, enabled: event.target.checked })) }} />
                        {t('enabled')}
                      </label>
                      <Button size="sm" variant="ghost" onClick={() => { void run(async () => setSecret({ id: view.settings.id, ...await api.rotateWebhookSecret(view.settings.id) })) }}>{t('newSecret')}</Button>
                      <Button size="sm" variant="ghost" onClick={() => setEditing({ hook: view.settings, isNew: false })}>{t('edit')}</Button>
                      <Button size="sm" variant="ghost" onClick={() => { void run(() => api.removeWebhook(view.settings.id)) }}>{t('remove')}</Button>
                    </div>
                  </div>
                  <p className={css.meta}>{t('hookUrl')}: <code>{`${base}${view.path}`}</code></p>
                </li>
              )
        ))}
      </ul>
      {editing?.isNew === true
        ? (
            <div className={css.card}>
              <HookEditor initial={editing.hook} isNew api={api} t={t} onDone={(created) => { setEditing(undefined); if (created !== undefined) setSecret(created); reload() }} />
            </div>
          )
        : (
            <div className={css.addRow}>
              <span className={css.label}>{t('addHook')}</span>
              <div className={css.presets}>
                {overview.webhookPresets.map(preset => (
                  <button
                    key={preset.id}
                    type="button"
                    className={css.preset}
                    title={preset.description}
                    onClick={() => {
                      const taken = new Set(overview.webhooks.map(view => view.settings.id))
                      let id = preset.id === 'custom' ? 'hook' : preset.id
                      for (let index = 2; taken.has(id); index += 1) id = `${preset.id === 'custom' ? 'hook' : preset.id}-${index}`
                      setEditing({ hook: emptyHook(id, preset), isNew: true })
                    }}
                  >
                    <strong>{preset.title}</strong>
                    <span>{preset.description}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
      <h4 className={css.subTitle}>{t('deliveries')}</h4>
      {overview.deliveries.length === 0
        ? <p className={css.meta}>{t('deliveriesEmpty')}</p>
        : (
            <ul className={css.deliveries}>
              {overview.deliveries.slice(0, 30).map(delivery => (
                <li key={delivery.id}>
                  <details>
                    <summary>
                      <span className={delivery.outcome === 'started' ? css.ok : delivery.outcome === 'ignored' ? css.meta : css.error}>{t(`outcome_${delivery.outcome}`)}</span>
                      {' '}
                      <span className={css.meta}>{`${new Date(delivery.receivedAt).toLocaleString()} · ${delivery.webhookId} · HTTP ${delivery.status} · ${delivery.detail}`}</span>
                    </summary>
                    <pre className={css.pre}>{delivery.bodyPreview}</pre>
                    {delivery.status === 202 ? <Button size="sm" variant="ghost" onClick={() => { void run(() => api.replayDelivery(delivery.id)) }}>{t('replay')}</Button> : null}
                  </details>
                </li>
              ))}
            </ul>
          )}
    </section>
  )
}

// ---- Notifications ----

function NotifyEditor({ initial, isNew, view, api, t, onDone }: {
  initial: NotificationSettings
  isNew: boolean
  view: NotificationView | undefined
  api: IntegrationsApi
  t: Translate
  onDone: (secret?: string) => void
}): ReactNode {
  const [target, setTarget] = useState(initial)
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | undefined>()
  const toggle = (event: NotificationEvent, on: boolean): void =>
    setTarget(current => ({ ...current, events: on ? [...new Set([...current.events, event])] : current.events.filter(item => item !== event) }))
  const save = async (): Promise<void> => {
    setBusy(true)
    setError(undefined)
    try {
      const created = await api.saveNotification({ ...target, id: target.id.trim() }, url.trim() === '' ? undefined : url.trim())
      onDone(created?.secret)
    } catch (failure) {
      setError(message(failure))
      setBusy(false)
    }
  }
  return (
    <div className={css.editor}>
      <div className={css.grid}>
        <Field label={t('fieldId')}>
          <Input value={target.id} disabled={!isNew} onChange={event => setTarget({ ...target, id: event.target.value.toLowerCase() })} />
        </Field>
        <Field label={t('fieldFormat')}>
          <Select
            label={t('fieldFormat')}
            value={target.format}
            onChange={format => setTarget({ ...target, format })}
            options={(['ntfy', 'discord', 'slack', 'json'] as const).map(value => ({ value, label: t(`format_${value}`) }))}
          />
        </Field>
      </div>
      <Field label={t('fieldNotifyUrl')} hint={view?.urlSet === true ? t('urlStored', { host: view.urlHost ?? '' }) : t('secretHint')}>
        <Input value={url} placeholder={target.format === 'ntfy' ? 'https://ntfy.sh/your-topic' : 'https://…'} onChange={event => setUrl(event.target.value)} />
      </Field>
      <Field label={t('fieldEvents')}>
        <div className={css.inline}>
          {(['finished', 'approval', 'question'] as const).map(event => (
            <label key={event} className={css.check}>
              <input type="checkbox" checked={target.events.includes(event)} onChange={change => toggle(event, change.target.checked)} />
              {t(`event_${event}`)}
            </label>
          ))}
        </div>
      </Field>
      {target.events.includes('finished')
        ? (
            <Field label={t('fieldMinTurn')} hint={t('fieldMinTurnHint')}>
              <Input type="number" value={String(target.minTurnSeconds)} onChange={event => setTarget({ ...target, minTurnSeconds: Math.max(0, Number(event.target.value) || 0) })} />
            </Field>
          )
        : null}
      {error === undefined ? null : <p className={css.error}>{error}</p>}
      <div className={css.actions}>
        <Button variant="primary" size="sm" disabled={busy || target.id.trim() === '' || (isNew && url.trim() === '')} onClick={() => { void save() }}>{busy ? t('saving') : t('save')}</Button>
        <Button variant="ghost" size="sm" disabled={busy} onClick={() => onDone()}>{t('cancel')}</Button>
      </div>
    </div>
  )
}

function NotifyPanel({ overview, api, t, reload }: { overview: IntegrationsOverview; api: IntegrationsApi; t: Translate; reload: () => void }): ReactNode {
  const [editing, setEditing] = useState<{ target: NotificationSettings; isNew: boolean } | undefined>()
  const [note, setNote] = useState<{ tone: 'ok' | 'error'; text: string } | undefined>()
  const run = async (action: () => Promise<unknown>): Promise<void> => {
    setNote(undefined)
    try {
      await action()
    } catch (failure) {
      setNote({ tone: 'error', text: message(failure) })
    }
    reload()
  }
  const done = (secret?: string): void => {
    setEditing(undefined)
    if (secret !== undefined) setNote({ tone: 'ok', text: t('signingSecret', { secret }) })
    reload()
  }
  return (
    <section className={css.group}>
      <h3 className={css.groupTitle}>{t('notifyTitle')}</h3>
      <p className={css.intro}>{t('notifyIntro')}</p>
      {note === undefined ? null : <p className={note.tone === 'ok' ? css.ok : css.error}>{note.text}</p>}
      {overview.notifications.length === 0 ? <p className={css.meta}>{t('notifyEmpty')}</p> : null}
      <ul className={css.cards}>
        {overview.notifications.map(view => (
          editing !== undefined && !editing.isNew && editing.target.id === view.settings.id
            ? (
                <li key={view.settings.id} className={css.card}>
                  <NotifyEditor initial={view.settings} isNew={false} view={view} api={api} t={t} onDone={done} />
                </li>
              )
            : (
                <li key={view.settings.id} className={css.card}>
                  <div className={css.cardHead}>
                    <div className={css.cardTitle}>
                      {view.settings.enabled ? <StateDot state="done" size={8} /> : <span className={css.dotOff} />}
                      <strong>{view.settings.id}</strong>
                      <span className={css.meta}>{`${t(`format_${view.settings.format}`)}${view.urlHost === undefined ? '' : ` · ${view.urlHost}`}`}</span>
                    </div>
                    <div className={css.cardActions}>
                      <label className={css.check}>
                        <input type="checkbox" checked={view.settings.enabled} onChange={event => { void run(() => api.saveNotification({ ...view.settings, enabled: event.target.checked })) }} />
                        {t('enabled')}
                      </label>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          void run(async () => {
                            const result = await api.testNotification(view.settings.id)
                            setNote({ tone: result.ok ? 'ok' : 'error', text: result.ok ? t('testSent', { detail: result.detail }) : t('testSendFailed', { detail: result.detail }) })
                          })
                        }}
                      >
                        {t('test')}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setEditing({ target: view.settings, isNew: false })}>{t('edit')}</Button>
                      <Button size="sm" variant="ghost" onClick={() => { void run(() => api.removeNotification(view.settings.id)) }}>{t('remove')}</Button>
                    </div>
                  </div>
                  <p className={css.meta}>
                    {view.settings.events.map(event => t(`event_${event}`)).join(', ')}
                    {view.lastStatus === undefined ? '' : ` · ${t('lastStatus', { status: view.lastStatus })}`}
                  </p>
                </li>
              )
        ))}
      </ul>
      {editing?.isNew === true
        ? (
            <div className={css.card}>
              <NotifyEditor initial={editing.target} isNew view={undefined} api={api} t={t} onDone={done} />
            </div>
          )
        : (
            <Button
              size="sm"
              variant="outline"
              onClick={() => setEditing({ target: { id: `notify-${overview.notifications.length + 1}`, enabled: true, title: '', format: 'ntfy', events: ['finished', 'approval', 'question'], minTurnSeconds: 20 }, isNew: true })}
            >
              {t('addNotify')}
            </Button>
          )}
    </section>
  )
}

// ---- Telegram ----

const TELEGRAM_DOT: Record<IntegrationsOverview['telegram']['status']['phase'], StateDotState | undefined> = {
  off: undefined,
  'needs-token': 'warning',
  connecting: 'ongoing',
  running: 'done',
  error: 'error',
}

function TelegramPanel({ overview, api, t, reload }: { overview: IntegrationsOverview; api: IntegrationsApi; t: Translate; reload: () => void }): ReactNode {
  const view = overview.telegram
  const [draft, setDraft] = useState(view.settings)
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<{ tone: 'ok' | 'error'; text: string } | undefined>()
  const [pairing, setPairing] = useState<{ code: string; link?: string } | undefined>()
  const dirty = token.trim() !== '' || JSON.stringify(draft) !== JSON.stringify(view.settings)
  // Server-side changes (a chat just paired) flow in while nothing is being edited here.
  const settingsKey = JSON.stringify(view.settings)
  useEffect(() => {
    if (!dirty) setDraft(view.settings)
  }, [settingsKey])
  const run = async (action: () => Promise<unknown>): Promise<void> => {
    setBusy(true)
    setNote(undefined)
    try {
      await action()
    } catch (failure) {
      setNote({ tone: 'error', text: message(failure) })
    } finally {
      setBusy(false)
      reload()
    }
  }
  const status = view.status
  const statusText = status.phase === 'running'
    ? t('tgPhase_running', { username: status.username ?? '?' })
    : status.phase === 'error' ? t('tgPhase_error', { error: status.error ?? '' }) : t(`tgPhase_${status.phase}`)
  const dot = TELEGRAM_DOT[status.phase]
  const toggleEvent = (event: NotificationEvent, on: boolean): void =>
    setDraft(current => ({ ...current, notifyEvents: on ? [...new Set([...current.notifyEvents, event])] : current.notifyEvents.filter(item => item !== event) }))
  return (
    <section className={css.group}>
      <h3 className={css.groupTitle}>{t('tgTitle')}</h3>
      <p className={css.intro}>{t('tgIntro')}</p>
      <div className={css.cardTitle}>
        {dot === undefined ? <span className={css.dotOff} /> : <StateDot state={dot} size={8} />}
        <span className={status.phase === 'error' ? css.error : css.meta}>{statusText}</span>
      </div>
      {view.tokenSet ? null : <p className={css.meta}>{t('tgSteps')}</p>}
      <div className={css.card}>
        <div className={css.editor}>
          <label className={css.check}>
            <input type="checkbox" checked={draft.enabled} onChange={event => setDraft({ ...draft, enabled: event.target.checked })} />
            {t('tgEnabled')}
          </label>
          <Field label={t('tgToken')} hint={view.tokenSet ? t('tgTokenStored') : t('secretHint')}>
            <Input type="password" value={token} placeholder="123456789:AA…" onChange={event => setToken(event.target.value)} autoComplete="off" />
          </Field>
          <Field label={t('tgFolder')}>
            <Input value={draft.workspacePath} placeholder={'C:\\src\\project'} onChange={event => setDraft({ ...draft, workspacePath: event.target.value })} />
          </Field>
          <div className={css.grid}>
            <Field label={t('fieldAgentPreset')}>
              <Input value={draft.agentPreset} onChange={event => setDraft({ ...draft, agentPreset: event.target.value })} />
            </Field>
            <Field label={t('fieldPermissionPreset')}>
              <Input value={draft.permissionPreset} onChange={event => setDraft({ ...draft, permissionPreset: event.target.value })} />
            </Field>
          </div>
          <label className={css.check}>
            <input type="checkbox" checked={draft.answerInTelegram} onChange={event => setDraft({ ...draft, answerInTelegram: event.target.checked })} />
            {t('tgAnswerHere')}
          </label>
          <Field label={t('tgNotify')}>
            <div className={css.inline}>
              {(['finished', 'approval', 'question'] as const).map(event => (
                <label key={event} className={css.check}>
                  <input type="checkbox" checked={draft.notifyEvents.includes(event)} onChange={change => toggleEvent(event, change.target.checked)} />
                  {t(`event_${event}`)}
                </label>
              ))}
            </div>
          </Field>
          {draft.notifyEvents.includes('finished')
            ? (
                <Field label={t('fieldMinTurn')} hint={t('fieldMinTurnHint')}>
                  <Input type="number" value={String(draft.minTurnSeconds)} onChange={event => setDraft({ ...draft, minTurnSeconds: Math.max(0, Number(event.target.value) || 0) })} />
                </Field>
              )
            : null}
          {note === undefined ? null : <p className={note.tone === 'ok' ? css.ok : css.error}>{note.text}</p>}
          <div className={css.actions}>
            <Button
              variant="primary"
              size="sm"
              disabled={busy || !dirty}
              onClick={() => {
                void run(async () => {
                  await api.saveTelegram(draft, token.trim() === '' ? undefined : token.trim())
                  setToken('')
                })
              }}
            >
              {busy ? t('saving') : t('save')}
            </Button>
            {view.tokenSet
              ? <Button variant="ghost" size="sm" disabled={busy} onClick={() => { void run(() => api.saveTelegram({ ...draft, enabled: false }, null)) }}>{t('tgRemoveToken')}</Button>
              : null}
          </div>
        </div>
      </div>
      <h4 className={css.subTitle}>{t('tgChats')}</h4>
      {view.chats.length === 0
        ? <p className={css.meta}>{t('tgNoChats')}</p>
        : (
            <ul className={css.cards}>
              {view.chats.map(chat => (
                <li key={chat.chatId} className={css.card}>
                  <div className={css.cardHead}>
                    <div className={css.cardTitle}>
                      <strong>{chat.name}</strong>
                      <span className={css.meta}>{chat.chatTitle === undefined ? chat.chatId : t('tgChatTalksTo', { title: chat.chatTitle })}</span>
                    </div>
                    <Button size="sm" variant="ghost" onClick={() => { void run(() => api.removeTelegramChat(chat.chatId)) }}>{t('tgUnpair')}</Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
      {status.phase === 'running'
        ? (
            <div className={css.inline}>
              <Button size="sm" variant="outline" disabled={busy} onClick={() => { void run(async () => setPairing(await api.telegramPairingCode())) }}>{t('tgPair')}</Button>
            </div>
          )
        : null}
      {pairing === undefined
        ? null
        : (
            <div className={css.secretBox}>
              <p>
                {pairing.link === undefined
                  ? t('tgPairingNoLink', { code: pairing.code })
                  : t('tgPairing', { code: pairing.code })}
              </p>
              {pairing.link === undefined ? null : <a href={pairing.link} target="_blank" rel="noreferrer">{pairing.link}</a>}
            </div>
          )}
    </section>
  )
}

/** The Integrations settings section. */
export function IntegrationsSection({ t, api }: IntegrationsSectionProps): ReactNode {
  const [overview, setOverview] = useState<IntegrationsOverview | undefined>()
  const [error, setError] = useState<string | undefined>()
  const reload = useCallback(() => {
    api.overview().then((next) => {
      setOverview(next)
      setError(undefined)
    }, (failure: unknown) => setError(message(failure)))
  }, [api])

  useEffect(() => {
    reload()
    // Connection phases change on their own; poll while the section is open.
    const timer = window.setInterval(reload, 3_000)
    const onMessage = (event: MessageEvent): void => {
      if ((event.data as { type?: unknown } | null)?.type === 'integrations-oauth') reload()
    }
    window.addEventListener('message', onMessage)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('message', onMessage)
    }
  }, [reload])

  const body = useMemo(() => {
    if (overview === undefined) return null
    return (
      <>
        <McpPanel overview={overview} api={api} t={t} reload={reload} />
        <TelegramPanel overview={overview} api={api} t={t} reload={reload} />
        <HooksPanel overview={overview} api={api} t={t} reload={reload} />
        <NotifyPanel overview={overview} api={api} t={t} reload={reload} />
      </>
    )
  }, [overview, api, t, reload])

  return (
    <div className={css.section}>
      <h2 className={css.title}>{t('title')}</h2>
      <p className={css.intro}>{t('intro')}</p>
      {error === undefined
        ? null
        : (
            <p className={css.error}>
              {t('loadError', { message: error })}
              {' '}
              <Button size="sm" variant="ghost" onClick={reload}>{t('retry')}</Button>
            </p>
          )}
      {overview === undefined && error === undefined ? <p className={css.meta}>{t('loading')}</p> : body}
    </div>
  )
}
