/**
 * Error-triggered skill sections. A skill may list, in its front matter, output patterns that mean "this skill
 * applies right now", each with the heading of the section to show:
 *
 * ```yaml
 * triggers:
 *   - pattern: ConstrainedLanguage|Method invocation is supported only
 *     section: 2. ConstrainedLanguage means the dsh sandbox, not a bug
 *   - pattern: is not recognized as (the )?name of a cmdlet
 *     section: 12. Bash habits that break here
 * ```
 *
 * When a failed tool result matches a pattern, that section is attached to the result, so the model gets the fix
 * without having to decide to load the skill. Patterns are case-insensitive regular expressions.
 * @module @deepseek-ai/dsh-harness-boost/skills
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

/** One trigger: the skill, its compiled pattern and the section text it attaches. */
export interface SkillTrigger {
  readonly skill: string
  readonly pattern: RegExp
  readonly heading: string
  readonly text: string
}

/** Most characters of one attached section. */
const SECTION_MAX_CHARS = 3000

/** Parse the `triggers:` list of a SKILL.md front matter (the small YAML subset shown above). */
export function parseTriggers(skill: string, markdown: string): SkillTrigger[] {
  const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(markdown)?.[1]
  if (front === undefined) return []
  const lines = front.split(/\r?\n/)
  const start = lines.findIndex(l => /^triggers:\s*$/.test(l))
  if (start < 0) return []
  const items: { pattern?: string; section?: string }[] = []
  for (const line of lines.slice(start + 1)) {
    if (/^\S/.test(line)) break
    const item = /^\s*-\s+(pattern|section):\s*(.*)$/.exec(line)
    const field = item ?? /^\s+(pattern|section):\s*(.*)$/.exec(line)
    if (field === null) continue
    if (item !== null) items.push({})
    const value = field[2]!.trim().replace(/^(['"])(.*)\1$/, '$2')
    const current = items.at(-1)
    if (current !== undefined) current[field[1] as 'pattern' | 'section'] = value
  }
  const body = markdown.slice(markdown.indexOf('---', 3) + 3)
  const out: SkillTrigger[] = []
  for (const { pattern, section } of items) {
    if (pattern === undefined || pattern.length === 0) continue
    let compiled: RegExp
    try {
      compiled = new RegExp(pattern, 'i')
    } catch {
      continue
    }
    const text = section === undefined ? body.trim() : sectionText(body, section)
    if (text.length === 0) continue
    out.push({ skill, pattern: compiled, heading: section ?? skill, text: text.slice(0, SECTION_MAX_CHARS) })
  }
  return out
}

/** The text of the Markdown section whose heading contains `heading`, up to the next heading of the same or a higher level. */
export function sectionText(markdown: string, heading: string): string {
  const lines = markdown.split(/\r?\n/)
  const at = lines.findIndex(l => /^#{1,6}\s/.test(l) && l.toLowerCase().includes(heading.toLowerCase()))
  if (at < 0) return ''
  const level = /^(#+)/.exec(lines[at]!)![1]!.length
  let end = lines.length
  for (let i = at + 1; i < lines.length; i += 1) {
    const m = /^(#+)\s/.exec(lines[i]!)
    if (m !== null && m[1]!.length <= level) {
      end = i
      break
    }
  }
  return lines.slice(at, end).join('\n').trim()
}

/** Load every trigger of the skills under `roots` (each a directory of `<name>/SKILL.md`). Unreadable skills are skipped. */
export function loadTriggers(roots: readonly string[]): SkillTrigger[] {
  const out: SkillTrigger[] = []
  for (const root of roots) {
    if (!existsSync(root)) continue
    for (const name of readdirSync(root)) {
      const file = join(root, name, 'SKILL.md')
      try {
        out.push(...parseTriggers(name, readFileSync(file, 'utf8')))
      } catch {
        // a skill directory without a readable SKILL.md has no triggers
      }
    }
  }
  return out
}
