/**
 * A stand-in for `@deepseek-ai/dsh-settings` in unit tests: `update` writes the patch into the integrations service's
 * volatile config references and emits `loader/volatile-update`, as the Loader does after a profile write.
 */
import { Service, type Context } from '@deepseek-ai/cordis'
import { createVolatile, updateVolatile, type Volatile } from '@deepseek-ai/cosmokit'

export default class StubSettings extends Service {
  constructor(ctx: Context) {
    super(ctx, 'settings')
  }

  update(_ns: string, patch: object): Promise<void> {
    const service = this.ctx.get('integrations') as unknown as { config: Record<string, Volatile<unknown>> }
    const paths: string[][] = []
    for (const [key, value] of Object.entries(patch)) {
      const ref = service.config[key]
      if (ref === undefined) continue
      updateVolatile(ref, createVolatile(value))
      paths.push([key])
    }
    this.ctx.root.emit('loader/volatile-update', paths)
    return Promise.resolve()
  }
}
