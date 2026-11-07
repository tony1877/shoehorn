/**
 * Repository map: a cached per-workspace index of code files and their symbols (functions, classes, types, methods),
 * so the agent can ask where something is instead of reading files broadly. File lists come from git when the
 * workspace is a repository (ignored files stay out), otherwise from a bounded walk. Symbols come from per-language
 * line patterns: cheap, no parser, good enough to point at the right file and line.
 * @module @deepseek-ai/dsh-harness-boost/repomap
 */

import { execFile, execFileSync } from 'node:child_process'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { extname, join, relative, sep } from 'node:path'

export interface RepoSymbol {
  readonly name: string
  readonly kind: string
  /** 1-based. */
  readonly line: number
  /** The defining line, trimmed and clipped. */
  readonly sig: string
}

interface FileEntry {
  readonly mtime: number
  readonly lines: number
  readonly symbols: readonly RepoSymbol[]
}

export interface RepoIndex {
  readonly root: string
  readonly git: boolean
  /** Every listed file, relative, with `/` separators. */
  readonly files: readonly string[]
  /** Code files with their symbols. */
  readonly code: ReadonlyMap<string, FileEntry>
}

/** Directories never walked (outside git). */
const SKIP_DIRS = new Set(['node_modules', '.git', '.hg', '.svn', 'dist', 'build', 'out', 'target', 'bin', 'obj',
  '__pycache__', '.venv', 'venv', '.mypy_cache', '.pytest_cache', '.next', '.cache', 'coverage', 'vendor', 'third_party'])
const MAX_FILES = 30000
const MAX_FILE_BYTES = 512 * 1024
const SIG_MAX = 140

type Pattern = readonly [kind: string, regex: RegExp, group: number]

const CONTROL = new Set(['if', 'for', 'while', 'switch', 'catch', 'return', 'else', 'do', 'try', 'sizeof', 'new', 'delete', 'throw', 'case', 'using', 'typeof', 'await'])

const PY: Pattern[] = [['def', /^\s*(?:async\s+)?def\s+(\w+)/, 1], ['class', /^\s*class\s+(\w+)/, 1]]
const JS: Pattern[] = [
  ['function', /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+(\w+)/, 1],
  ['class', /^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)/, 1],
  ['type', /^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+(\w+)/, 1],
  ['const', /^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)\s*(?::[^=]+)?=>|function|\w+\s*=>)/, 1],
  ['method', /^\s+(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*(#?\w+)\s*(?:<[^>]*>)?\([^)]*\)\s*(?::\s*[^{=]+)?\{\s*$/, 1],
]
const GO: Pattern[] = [['func', /^func\s+(?:\([^)]*\)\s*)?(\w+)/, 1], ['type', /^type\s+(\w+)/, 1]]
const RS: Pattern[] = [
  ['fn', /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)/, 1],
  ['type', /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union|type|mod)\s+(\w+)/, 1],
  ['impl', /^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?(\w+)/, 1],
]
const JVM: Pattern[] = [
  ['type', /^\s*(?:(?:public|private|protected|internal|static|abstract|sealed|final|partial|data|open)\s+)*(?:class|interface|enum|record|struct|object)\s+(\w+)/, 1],
  ['method', /^\s+(?:(?:public|private|protected|internal|static|override|async|virtual|abstract|final|suspend|fun)\s+)+[\w<>[\],.?\s]*?\b(\w+)\s*\([^;]*\)\s*(?:[:\w<>?,\s]*)\{?\s*$/, 1],
]
const C: Pattern[] = [
  ['type', /^\s*(?:typedef\s+)?(?:struct|class|union|enum(?:\s+class)?|namespace)\s+(\w+)\s*(?:[:{]|$)/, 1],
  ['define', /^#\s*define\s+(\w+)/, 1],
  ['function', /^(?:[\w:<>,*&~]+\s+)*?[*&]*(~?[\w:]+)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{?\s*$/, 1],
]
const PS: Pattern[] = [['function', /^\s*function\s+([\w-]+)/i, 1], ['class', /^\s*class\s+(\w+)/i, 1]]
const SH: Pattern[] = [['function', /^\s*(?:function\s+)?([\w-]+)\s*\(\)\s*\{?/, 1]]
const RB: Pattern[] = [['def', /^\s*def\s+([\w.?!]+)/, 1], ['class', /^\s*(?:class|module)\s+([\w:]+)/, 1]]
const PHP: Pattern[] = [['function', /^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+(\w+)/, 1], ['class', /^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+(\w+)/, 1]]
const LUA: Pattern[] = [['function', /^\s*(?:local\s+)?function\s+([\w.:]+)/, 1]]

const LANGUAGES: Record<string, readonly Pattern[]> = {
  '.py': PY, '.pyi': PY,
  '.js': JS, '.jsx': JS, '.mjs': JS, '.cjs': JS, '.ts': JS, '.tsx': JS, '.mts': JS, '.cts': JS, '.vue': JS, '.svelte': JS,
  '.go': GO, '.rs': RS,
  '.java': JVM, '.kt': JVM, '.kts': JVM, '.cs': JVM, '.scala': JVM, '.swift': JVM,
  '.c': C, '.h': C, '.cc': C, '.cpp': C, '.cxx': C, '.hpp': C, '.hh': C, '.hxx': C, '.cu': C, '.cuh': C, '.m': C, '.mm': C,
  '.ps1': PS, '.psm1': PS, '.sh': SH, '.bash': SH, '.zsh': SH, '.rb': RB, '.php': PHP, '.lua': LUA,
}

export function isCodeFile(path: string): boolean {
  return extname(path).toLowerCase() in LANGUAGES
}

/** The symbols of one file's text. Exported for tests. */
export function extractSymbols(path: string, text: string): RepoSymbol[] {
  const patterns = LANGUAGES[extname(path).toLowerCase()]
  if (patterns === undefined) return []
  const out: RepoSymbol[] = []
  const lines = text.split(/\r?\n/)
  let inBlockComment = false
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!
    if (line.length > 400) continue
    const trimmed = line.trim()
    if (inBlockComment) {
      if (trimmed.includes('*/')) inBlockComment = false
      continue
    }
    if (trimmed.startsWith('/*') && !trimmed.includes('*/')) {
      inBlockComment = true
      continue
    }
    if (trimmed.startsWith('//') || trimmed.startsWith('#') && !trimmed.startsWith('#define') && !trimmed.startsWith('# define') || trimmed.startsWith('*')) continue
    for (const [kind, regex, group] of patterns) {
      const m = regex.exec(line)
      const name = m?.[group]
      if (name === undefined || CONTROL.has(name) || name.length < 2) continue
      out.push({ name, kind, line: i + 1, sig: trimmed.length > SIG_MAX ? `${trimmed.slice(0, SIG_MAX)}…` : trimmed })
      break
    }
  }
  return out
}

function listGit(root: string): string[] | undefined {
  try {
    const out = execFileSync('git', ['-C', root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
      { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, timeout: 20000, stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true })
    return out.split('\0').filter(f => f.length > 0).slice(0, MAX_FILES)
  } catch {
    return undefined
  }
}

function walk(root: string): string[] {
  const out: string[] = []
  const stack = ['']
  while (stack.length > 0 && out.length < MAX_FILES) {
    const rel = stack.pop()!
    let entries
    try {
      entries = readdirSync(join(root, rel), { withFileTypes: true })
    } catch {
      continue
    }
    for (const e of entries) {
      const child = rel === '' ? e.name : `${rel}/${e.name}`
      if (e.isDirectory()) {
        if (!SKIP_DIRS.has(e.name) && !e.name.startsWith('.')) stack.push(child)
      } else if (e.isFile()) {
        out.push(child)
      }
    }
  }
  return out.sort()
}

/** Per-root cache: files are re-read only when their mtime changes. */
const cache = new Map<string, Map<string, FileEntry>>()

/**
 * Build (or refresh) the index of `root`.
 * @param root - the workspace directory.
 * @returns the index; code files whose mtime is unchanged reuse their cached symbols.
 */
export function buildIndex(root: string): RepoIndex {
  const listed = listGit(root)
  const git = listed !== undefined
  const files = (listed ?? walk(root)).map(f => f.split(sep).join('/'))
  const previous = cache.get(root) ?? new Map<string, FileEntry>()
  const code = new Map<string, FileEntry>()
  for (const file of files) {
    if (!isCodeFile(file)) continue
    const abs = join(root, file)
    let st
    try {
      st = statSync(abs)
    } catch {
      continue
    }
    if (!st.isFile() || st.size > MAX_FILE_BYTES) continue
    const old = previous.get(file)
    if (old !== undefined && old.mtime === st.mtimeMs) {
      code.set(file, old)
      continue
    }
    let text: string
    try {
      text = readFileSync(abs, 'utf8')
    } catch {
      continue
    }
    code.set(file, { mtime: st.mtimeMs, lines: text.split('\n').length, symbols: extractSymbols(file, text) })
  }
  cache.set(root, code)
  return { root, git, files, code }
}

function extCounts(files: readonly string[]): string {
  const counts = new Map<string, number>()
  for (const f of files) {
    const e = extname(f).toLowerCase() || '(none)'
    counts.set(e, (counts.get(e) ?? 0) + 1)
  }
  return [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([e, n]) => `${e} ${n}`).join(', ')
}

function symbolList(entry: FileEntry, max: number): string {
  const names = entry.symbols.filter(s => s.kind !== 'define').map(s => s.name)
  const unique = [...new Set(names)]
  return unique.length <= max ? unique.join(', ') : `${unique.slice(0, max).join(', ')}, … +${unique.length - max}`
}

/**
 * Render the map: a directory overview of the whole workspace, or every file of one directory with its symbols.
 * @param index - the index.
 * @param dir - optional directory (relative to the root) to show in detail.
 * @param maxChars - output budget.
 */
export function renderMap(index: RepoIndex, dir: string | undefined, maxChars = 9000): string {
  const prefix = dir === undefined || dir === '' || dir === '.' ? '' : `${dir.replace(/\\/g, '/').replace(/^\.\/|\/+$/g, '')}/`
  const inScope = (f: string): boolean => f.startsWith(prefix)
  const files = index.files.filter(inScope)
  const codeFiles = [...index.code.keys()].filter(inScope)
  const head = `Repository map of ${join(index.root, prefix)} (${index.git ? 'git' : 'walked'}: ${files.length} files, `
    + `${codeFiles.length} code files indexed). find_symbol NAME shows where a symbol is defined and used; `
    + 'repo_map path=DIR shows one directory file by file.'
  const lines: string[] = [head]
  let used = head.length
  const push = (line: string): boolean => {
    if (used + line.length + 1 > maxChars) return false
    lines.push(line)
    used += line.length + 1
    return true
  }
  const detailed = prefix !== '' || codeFiles.length <= 80
  if (detailed) {
    for (const f of codeFiles) {
      const entry = index.code.get(f)!
      if (!push(`${f.slice(prefix.length)} (${entry.lines} lines): ${symbolList(entry, 14) || '-'}`)) {
        push(`… ${codeFiles.length - (lines.length - 1)} more files: narrow with repo_map path=DIR`)
        break
      }
    }
    const other = files.filter(f => !index.code.has(f))
    if (other.length > 0) push(`other files: ${other.length} (${extCounts(other)})`)
    return lines.join('\n')
  }
  // overview: directories two levels deep, with counts and their biggest code files' main symbols
  const groups = new Map<string, string[]>()
  for (const f of files) {
    const parts = f.slice(prefix.length).split('/')
    const key = parts.length <= 1 ? '.' : parts.slice(0, Math.min(2, parts.length - 1)).join('/')
    const list = groups.get(key) ?? []
    list.push(f)
    groups.set(key, list)
  }
  for (const [key, list] of [...groups].sort((a, b) => a[0].localeCompare(b[0]))) {
    const code = list.filter(f => index.code.has(f))
      .sort((a, b) => index.code.get(b)!.symbols.length - index.code.get(a)!.symbols.length)
    const top = code.slice(0, 3).map(f => `${f.split('/').pop()}: ${symbolList(index.code.get(f)!, 5)}`).join(' | ')
    if (!push(`${key}/ (${list.length} files: ${extCounts(list)})${top === '' ? '' : ` — ${top}`}`)) {
      push('… more directories: narrow with repo_map path=DIR')
      break
    }
  }
  return lines.join('\n')
}

export interface SymbolHit {
  readonly file: string
  readonly symbol: RepoSymbol
}

/** Definitions of `name`: exact matches first, else case-insensitive, else prefix matches. */
export function findDefinitions(index: RepoIndex, name: string, limit = 30): SymbolHit[] {
  const all: SymbolHit[] = []
  for (const [file, entry] of index.code) for (const symbol of entry.symbols) all.push({ file, symbol })
  const tail = name.split(/[.:]+/).pop() ?? name
  const exact = all.filter(h => h.symbol.name === tail)
  if (exact.length > 0) return exact.slice(0, limit)
  const lower = tail.toLowerCase()
  const ci = all.filter(h => h.symbol.name.toLowerCase() === lower)
  if (ci.length > 0) return ci.slice(0, limit)
  return all.filter(h => h.symbol.name.toLowerCase().startsWith(lower)).slice(0, limit)
}

/** Whole-word references of `name` under `root` via ripgrep (respects .gitignore); [] when rg is missing. */
export function findReferences(root: string, name: string, limit = 60): Promise<string[]> {
  return new Promise((resolve) => {
    execFile('rg', ['-n', '-w', '-F', '--no-heading', '--max-columns', '200', '--max-count', '20', '-g', '!node_modules', '--', name, '.'],
      { cwd: root, maxBuffer: 16 * 1024 * 1024, timeout: 20000, windowsHide: true },
      (_error, stdout) => {
        const lines = (stdout ?? '').split(/\r?\n/).filter(l => l.length > 0).map(l => l.replace(/^\.[\\/]/, '').replace(/\\/g, '/'))
        resolve(lines.slice(0, limit))
      })
  })
}

/** Render definitions and references for `find_symbol`. */
export function renderSymbol(name: string, defs: readonly SymbolHit[], refs: readonly string[]): string {
  const out: string[] = []
  if (defs.length === 0) out.push(`No definition of "${name}" found in the index.`)
  else {
    out.push(`Definitions of "${name}":`)
    for (const d of defs) out.push(`  ${d.file}:${d.symbol.line}  [${d.symbol.kind}] ${d.symbol.sig}`)
  }
  const defKeys = new Set(defs.map(d => `${d.file}:${d.symbol.line}:`))
  const uses = refs.filter(r => ![...defKeys].some(k => r.startsWith(k)))
  if (uses.length > 0) {
    out.push(`References (${uses.length}${uses.length >= 60 ? '+' : ''}):`)
    for (const r of uses) out.push(`  ${r}`)
  } else if (refs.length === 0) {
    out.push('No references found (or ripgrep is not available).')
  }
  return out.join('\n')
}

/** Number of code files in `root`, cheaply (cached per root), for the large-workspace hint. */
const sizeCache = new Map<string, number>()
export function codeFileCount(root: string): number {
  const cached = sizeCache.get(root)
  if (cached !== undefined) return cached
  const files = listGit(root) ?? []
  const n = files.filter(isCodeFile).length
  sizeCache.set(root, n)
  return n
}

export function relativeTo(root: string, path: string): string {
  return relative(root, path).split(sep).join('/')
}
