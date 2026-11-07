/**
 * Reading tool calls and results: which calls edit files, which shell commands check the work, and whether a call
 * failed. Pure functions over tool names, arguments and result text.
 * @module @deepseek-ai/dsh-harness-boost/signals
 */

/** Tools that change files. `str_replace_editor` counts only for its mutating commands. */
const EDIT_TOOLS = new Set(['edit', 'write', 'str_replace_editor', 'apply_patch', 'multi_edit'])

/** Shell tools whose results carry an exit-code marker. */
const SHELL_TOOLS = new Set(['pwsh', 'bash', 'shell', 'exec'])

/** Commands that check the work: test runners and builds. */
const CHECK = /\b(pytest|unittest|tox|nox|npm (run )?test|pnpm (run )?test|yarn test|vitest|jest|cargo (test|build|check)|go (test|build|vet)|ctest|cmake --build|ninja|msbuild|dotnet (test|build)|make( |$)|gradle|mvn|tsc\b|build\.ps1|build\.sh|run_tests|test_\w+\.py|\w+_test\.py)/i

/** Lines that state a run failed, whatever the exit code says. */
const FAILED_OUTPUT = /^(FAILED|FAIL\b|ERROR:|Traceback \(most recent call last\))|\b[1-9]\d* (failed|errors?)\b|\bBUILD FAILED\b|error (C|LNK)\d{4}|: error TS\d+/m

export function isShellTool(name: string): boolean {
  return SHELL_TOOLS.has(name)
}

/** The file a call edits, or undefined when it is not an edit. */
export function editedPath(name: string, args: unknown): string | undefined {
  if (!EDIT_TOOLS.has(name) || args === null || typeof args !== 'object') return undefined
  const a = args as Record<string, unknown>
  if (name === 'str_replace_editor' && (a['command'] === 'view' || a['command'] === undefined)) return undefined
  for (const key of ['file_path', 'path', 'filePath', 'file']) {
    if (typeof a[key] === 'string') return a[key]
  }
  return '?'
}

/** A test file by its path: under tests/ or test/, or named test_*.py, *_test.py, *.test.ts, *.spec.js and the like. */
export function isTestPath(path: string): boolean {
  return /(^|[\\/])tests?[\\/]|(^|[\\/])test_[^\\/]*\.py$|_test\.(py|go)$|\.(test|spec)\.[cm]?[jt]sx?$/i.test(path)
}

/**
 * Whether an edit changes or removes existing text (rather than only adding to it): `new_string` does not contain
 * `old_string`. A `write` always counts, it replaces the file.
 */
export function changesExisting(name: string, args: unknown): boolean {
  if (args === null || typeof args !== 'object') return false
  const a = args as Record<string, unknown>
  if (name === 'write') return true
  const oldText = a['old_string'] ?? a['old_str']
  const newText = a['new_string'] ?? a['new_str']
  return typeof oldText === 'string' && typeof newText === 'string' && !newText.includes(oldText)
}

/** The final report says that tests were changed (so a test change is disclosed, not hidden). */
export function disclosesTestChange(text: string): boolean {
  return /\btest\w*[^.\n]{0,80}\b(changed|adjusted|updated|modified|relaxed|corrected|fixed|rewrote|removed)\b|\b(changed|adjusted|updated|modified|relaxed|corrected|rewrote|removed)\b[^.\n]{0,80}\btest\w*/i.test(text)
}

/**
 * The task states behaviour the work must match rule by rule: a spec file, a docstring, stated requirements, or new
 * behaviour described in the task itself ("implement …", "add a … parameter").
 */
export function specTask(prompt: string): boolean {
  return /\bspec(ification)?s?\b|SPEC\.md|\bdocstrings?\b|\brequirements?\b|\bacceptance criteria\b|\bimplement\b|\badd\b[^.\n]{0,60}\b(parameter|option|function|method|flag|feature|argument)\b/i.test(prompt)
}

/** The shell command of a call, or undefined when it is not a shell call. */
export function shellCommand(name: string, args: unknown): string | undefined {
  if (!SHELL_TOOLS.has(name) || args === null || typeof args !== 'object') return undefined
  const command = (args as Record<string, unknown>)['command']
  return typeof command === 'string' ? command : undefined
}

export function isCheckCommand(command: string): boolean {
  return CHECK.test(command)
}

/** The exit code a shell result reports: `[exit code: N]` / `[Command finished with exit code N]`, 0 when absent. */
export function exitCode(text: string): number {
  const m = /\[(?:exit code:|Command finished with exit code|shell exited: code) (-?\d+)\]/.exec(text)
  return m === null ? 0 : Number(m[1])
}

/** Whether a check run failed: a non-zero exit, or output that says so. */
export function checkFailed(text: string): boolean {
  return exitCode(text) !== 0 || FAILED_OUTPUT.test(text)
}

/** The line that best summarizes a test or build run (its verdict), for messages to the model. */
export function verdictLine(text: string): string {
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0 && !/^\[(exit code|Command finished)/.test(l))
  const verdict = lines.findLast(l => /\b(passed|failed|FAILED|OK|errors?|Ran \d+ tests?)\b/.test(l))
  return (verdict ?? lines.at(-1) ?? '').slice(0, 200)
}

/** The model's final text claims the work is verified or done. */
export function claimsSuccess(text: string): boolean {
  return /\b(all (\d+ )?tests? (now )?pass|tests? (now )?pass(es|ing)?|\d+ passed|suite passes|everything passes|build (succeeds|passes)|verified|works now|is fixed|fixed it)\b/i.test(text)
}

/** The model's final text ends by asking the user something (then open todos may legitimately wait). */
export function asksUser(text: string): boolean {
  return /\?\s*$/.test(text.trim())
}

/**
 * A key under which a failed call and a later successful retry are the same attempt: the edited file, or the
 * command's first word plus the check it runs.
 */
export function attemptKey(name: string, args: unknown): string | undefined {
  const path = editedPath(name, args)
  if (path !== undefined) return `edit:${path.toLowerCase()}`
  const command = shellCommand(name, args)
  if (command === undefined) return undefined
  const first = /^\s*(?:cd [^;&]+[;&]+\s*)?([^\s;|&]+)/.exec(command)?.[1] ?? command.slice(0, 20)
  return `shell:${first.toLowerCase()}`
}

/** Short one-line form of call arguments for prompts and messages. */
export function argsPreview(args: unknown, max = 300): string {
  const text = typeof args === 'string' ? args : JSON.stringify(args) ?? ''
  return text.length > max ? `${text.slice(0, max)}…` : text
}
