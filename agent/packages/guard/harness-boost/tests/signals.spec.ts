import { describe, expect, it } from 'vitest'
import {
  attemptKey, checkFailed, claimsSuccess, editedPath, exitCode, isCheckCommand, verdictLine,
} from '../src/signals.ts'
import { parseTriggers, sectionText } from '../src/skills.ts'

describe('signals', () => {
  it('reads shell exit codes in both tool formats', () => {
    expect(exitCode('Ran 3 tests\n\nFAILED (failures=1)\n[exit code: 1]')).toBe(1)
    expect(exitCode('ok\n[Command finished with exit code 2]')).toBe(2)
    expect(exitCode('all fine')).toBe(0)
  })

  it('treats failing test output as a failed check even with exit code 0', () => {
    expect(checkFailed('3 passed, 1 failed in 0.2s')).toBe(true)
    expect(checkFailed('Ran 4 tests in 0.001s\n\nOK')).toBe(false)
    expect(checkFailed('0 failed, 5 passed')).toBe(false)
    expect(checkFailed('src/a.c(3): error C2065: x undeclared')).toBe(true)
  })

  it('recognizes edits and checks', () => {
    expect(editedPath('edit', { file_path: 'a.py', old_string: 'x', new_string: 'y' })).toBe('a.py')
    expect(editedPath('str_replace_editor', { command: 'view', path: 'a.py' })).toBeUndefined()
    expect(editedPath('str_replace_editor', { command: 'str_replace', path: 'a.py' })).toBe('a.py')
    expect(editedPath('read', { file_path: 'a.py' })).toBeUndefined()
    expect(isCheckCommand('cd C:\\x; python -m unittest 2>&1')).toBe(true)
    expect(isCheckCommand('Get-ChildItem')).toBe(false)
  })

  it('keys a retry of the same file or command to its failure', () => {
    expect(attemptKey('edit', { file_path: 'C:\\A.py' })).toBe('edit:c:\\a.py')
    expect(attemptKey('pwsh', { command: 'cd C:\\x; python -m unittest' })).toBe('shell:python')
    expect(attemptKey('read', { file_path: 'a' })).toBeUndefined()
  })

  it('finds the verdict line and success claims', () => {
    expect(verdictLine('...\nRan 46 tests in 0.003s\n\nFAILED (failures=3, errors=9)\n[exit code: 1]')).toBe('FAILED (failures=3, errors=9)')
    expect(claimsSuccess('Test result: all 12 tests pass.')).toBe(true)
    expect(claimsSuccess('The suite still fails on test_x.')).toBe(false)
  })
})

describe('skill triggers', () => {
  const skill = [
    '---',
    'name: powershell-expert',
    'description: shell help',
    'triggers:',
    '  - pattern: ConstrainedLanguage',
    '    section: 2. ConstrainedLanguage',
    "  - pattern: '[bad'",
    '---',
    '# PowerShell',
    '## 1. Check the shell',
    'one',
    '## 2. ConstrainedLanguage means the sandbox',
    'two',
    '### detail',
    'more',
    '## 3. Native executables',
    'three',
  ].join('\n')

  it('parses triggers and skips invalid patterns', () => {
    const triggers = parseTriggers('powershell-expert', skill)
    expect(triggers).toHaveLength(1)
    expect(triggers[0]!.pattern.test('in constrainedlanguage mode')).toBe(true)
    expect(triggers[0]!.text).toBe('## 2. ConstrainedLanguage means the sandbox\ntwo\n### detail\nmore')
  })

  it('returns no triggers without a triggers key', () => {
    expect(parseTriggers('x', '---\nname: x\n---\nbody')).toEqual([])
  })

  it('cuts a section at the next heading of its level', () => {
    expect(sectionText('# A\na\n## B\nb\n# C\nc', 'B')).toBe('## B\nb')
  })
})

describe('recall relevance', () => {
  it('counts distinct meaningful shared words only', async () => {
    const { sharedTerms } = await import('../src/index.ts')
    expect(sharedTerms('Add a keyword-only max_len parameter to slugify', 'Machines available: desktop host vs laptop')).toBe(0)
    expect(sharedTerms('pwsh Get-Content fails on paths with brackets', 'Use -LiteralPath: Get-Content treats brackets in paths as wildcards')).toBe(3)
  })
})

describe('test weakening', () => {
  it('recognizes test files', async () => {
    const { isTestPath } = await import('../src/signals.ts')
    expect(isTestPath('C:\\w\\tests\\test_x.py')).toBe(true)
    expect(isTestPath('C:\\w\\test_mine.py')).toBe(true)
    expect(isTestPath('src/a.spec.ts')).toBe(true)
    expect(isTestPath('C:\\w\\calc\\interp.py')).toBe(false)
    expect(isTestPath('C:\\w\\contest.py')).toBe(false)
  })

  it('tells changing existing test text from adding to it', async () => {
    const { changesExisting } = await import('../src/signals.ts')
    expect(changesExisting('edit', { old_string: 'self.ok("print nil", ["nil"])', new_string: 'self.ok("print 1", ["1"])' })).toBe(true)
    expect(changesExisting('edit', { old_string: '    self.ok("print -2 ^ 2", ["-4"])', new_string: '    self.ok("print -2 ^ 2", ["4"])' })).toBe(true)
    expect(changesExisting('edit', { old_string: 'class T:\n', new_string: 'class T:\n    def test_new(self): ...\n' })).toBe(false)
    expect(changesExisting('write', { file_path: 't.py', content: 'x' })).toBe(true)
  })

  it('notices when a report discloses a test change', async () => {
    const { disclosesTestChange } = await import('../src/signals.ts')
    expect(disclosesTestChange('I changed test_eq_ne because the spec says nil is a literal.')).toBe(true)
    expect(disclosesTestChange('All 30 tests pass.')).toBe(false)
  })
})

describe('spec tasks', () => {
  it('recognizes tasks defined by a spec', async () => {
    const { specTask } = await import('../src/signals.ts')
    expect(specTask('Implement the language described in SPEC.md in calc/interp.py')).toBe(true)
    expect(specTask('Implement semver.match.satisfies exactly to the spec in its docstring.')).toBe(true)
    expect(specTask('Implement every function in textutil/core.py according to its docstring')).toBe(true)
    expect(specTask('Add a keyword-only max_len parameter to textkit.slug.slugify: the slug is cut to at most max_len')).toBe(true)
    expect(specTask('The inventory report crashes with a KeyError. Find the real cause and fix it.')).toBe(false)
    expect(specTask('In pricing.py, change the EU tax rate from 0.2 to 0.23.')).toBe(false)
  })
})
