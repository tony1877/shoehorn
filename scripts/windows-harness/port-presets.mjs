// Convert the old per-directory agent presets ($DSH_HOME/.agent-presets/<id>/{preset.yml,agent.cordis.yml}, dsh 0.1)
// into dsh 0.2 preset rows: one `@deepseek-ai/dsh-agent-preset` insert per preset (its plugin list is the old
// agent.cordis.yml unchanged), plus the registry's default.
//
//   node scripts/windows-harness/port-presets.mjs [--home DIR] [--default minimal-compact] [--out FILE]
//
// Without --out the patch is printed. `!!js` expressions in the presets are kept as written.
import fs from 'node:fs'
import path from 'node:path'
import yaml from '../../agent/node_modules/js-yaml/index.js'

class JsExpression {
  constructor(source) { this.source = source }
}
const jsType = new yaml.Type('tag:yaml.org,2002:js', {
  kind: 'scalar',
  construct: source => new JsExpression(source),
  instanceOf: JsExpression,
  represent: expression => expression.source,
})
const schema = yaml.DEFAULT_SCHEMA.extend([jsType])

function argument(name, fallback) {
  const at = process.argv.indexOf(name)
  return at === -1 ? fallback : process.argv[at + 1]
}

/** Plugin config fields renamed in dsh 0.2: [plugin name, old field, new field]. */
const RENAMED = [
  ['@deepseek-ai/dsh-persona', 'text', 'prefix'],
]

/** Plugins replaced in dsh 0.2: old name -> [new name, new row id]. */
const REPLACED = {
  '@deepseek-ai/dsh-workflow-worker-thread': ['@deepseek-ai/dsh-workflow-ptc', 'workflow-ptc'],
}

function migrate(rows) {
  for (const row of rows) {
    if (row === null || typeof row !== 'object') continue
    if (row.name in REPLACED) [row.name, row.id] = REPLACED[row.name]
    for (const [plugin, from, to] of RENAMED) {
      if (row.name === plugin && row.config && from in row.config && !(to in row.config)) {
        row.config[to] = row.config[from]
        delete row.config[from]
      }
    }
    if (Array.isArray(row.config)) migrate(row.config)  // cordis:group rows list their children as config
  }
  return rows
}

export function presetRows(home, defaultId) {
  const root = path.join(home, '.agent-presets')
  const rows = []
  for (const id of fs.readdirSync(root).sort()) {
    const dir = path.join(root, id)
    const compositionFile = path.join(dir, 'agent.cordis.yml')
    if (!fs.statSync(dir).isDirectory() || !fs.existsSync(compositionFile)) continue
    const meta = fs.existsSync(path.join(dir, 'preset.yml'))
      ? yaml.load(fs.readFileSync(path.join(dir, 'preset.yml'), 'utf8'), { schema }) ?? {}
      : {}
    const plugins = migrate(yaml.load(fs.readFileSync(compositionFile, 'utf8'), { schema }))
    if (!Array.isArray(plugins)) throw new Error(`${compositionFile}: expected a list of plugin rows`)
    rows.push({
      id: `preset-${id}`,
      name: '@deepseek-ai/dsh-agent-preset',
      config: {
        id,
        ...meta.name === undefined ? {} : { name: meta.name },
        ...meta.description === undefined ? {} : { description: meta.description },
        ...meta.order === undefined ? {} : { order: meta.order },
        plugins,
      },
    })
  }
  if (rows.length === 0) throw new Error(`no presets found in ${root}`)
  const patch = [{ insert: rows }]
  if (defaultId !== undefined) {
    if (!rows.some(row => row.config.id === defaultId)) throw new Error(`--default ${defaultId}: no such preset`)
    patch.push({ id: 'agent-preset-registry', config: { default: defaultId } })
  }
  return patch
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname.replace(/^\/(?=[A-Za-z]:)/, ''))) {
  const home = argument('--home', process.env.DSH_HOME || path.join(process.env.USERPROFILE ?? process.env.HOME, '.dsh'))
  const patch = presetRows(home, argument('--default', undefined))
  const text = '# shoehorn agent presets for dsh 0.2 (from $DSH_HOME/.agent-presets by port-presets.mjs)\n'
    + yaml.dump(patch, { schema, lineWidth: 120, noRefs: true })
  const out = argument('--out', undefined)
  if (out === undefined) process.stdout.write(text)
  else {
    fs.writeFileSync(out, text)
    console.log(`wrote ${patch[0].insert.length} presets to ${out}`)
  }
}
