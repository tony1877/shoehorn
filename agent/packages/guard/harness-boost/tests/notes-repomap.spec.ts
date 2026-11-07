import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { NotesStore, applyUpdate, emptyNotes, renderNotes } from '../src/notes.ts'
import { buildIndex, extractSymbols, findDefinitions, renderMap, renderSymbol } from '../src/repomap.ts'

describe('notes', () => {
  it('records a goal, marks items done only with evidence, and renders a checklist', () => {
    const notes = emptyNotes()
    expect(applyUpdate(notes, { goal: ['add max_len', 'cut at a word boundary'], plan: '1. code\n2. test' }, 1)).toEqual([])
    expect(applyUpdate(notes, { done: [{ n: 1, evidence: 'test_max_len passed' }, { n: 2, evidence: ' ' }, { n: 9, evidence: 'x' }] }, 1))
      .toEqual(['done: item 2 needs evidence', 'done: there is no goal item 9 (the goal has 2 items)'])
    expect(renderNotes(notes)).toBe([
      'Goal (what the user asked; 1 of 2 open):',
      '  1. [x] add max_len  — evidence: test_max_len passed',
      '  2. [ ] cut at a word boundary',
      'Plan:\n  1. code\n  2. test',
    ].join('\n'))
  })

  it('keeps done state for goal items repeated word for word, and caps appended lists', () => {
    const notes = emptyNotes()
    applyUpdate(notes, { goal: ['a', 'b'] }, 1)
    applyUpdate(notes, { done: [{ n: 1, evidence: 'ok' }] }, 1)
    applyUpdate(notes, { goal: ['a', 'c'] }, 2)
    expect(notes.goal).toEqual([{ text: 'a', done: true, evidence: 'ok' }, { text: 'c', done: false }])
    expect(notes.goalTurn).toBe(2)
    for (let i = 0; i < 20; i += 1) applyUpdate(notes, { finding: `f${i}` }, 2)
    expect(notes.findings).toHaveLength(15)
    expect(notes.findings[0]).toBe('f5')
  })

  it('persists per session', () => {
    const dir = mkdtempSync(join(tmpdir(), 'notes-'))
    const a = new NotesStore(dir)
    applyUpdate(a.get('s-1'), { decision: 'use LiteralPath: names contain [ ]' }, 1)
    a.save('s-1')
    expect(new NotesStore(dir).get('s-1').decisions).toEqual(['use LiteralPath: names contain [ ]'])
    expect(new NotesStore(dir).get('s-2').decisions).toEqual([])
  })
})

describe('repo map', () => {
  it('extracts symbols per language', () => {
    expect(extractSymbols('a.py', 'class Parser:\n    def parse_expr(self):\n        pass\n# def not_this():\nasync def main():\n').map(s => `${s.kind}:${s.name}:${s.line}`))
      .toEqual(['class:Parser:1', 'def:parse_expr:2', 'def:main:5'])
    expect(extractSymbols('a.ts', 'export async function load(x: string) {\nexport class Store {\n  async get(id: number): Promise<void> {\n    if (x) {\nexport const run = async (a) => {\ninterface Opts {\n').map(s => `${s.kind}:${s.name}`))
      .toEqual(['function:load', 'class:Store', 'method:get', 'const:run', 'type:Opts'])
    expect(extractSymbols('a.go', 'func (s *Server) Handle(w http.ResponseWriter) {\ntype Config struct {\n').map(s => s.name)).toEqual(['Handle', 'Config'])
    expect(extractSymbols('a.cpp', 'static int apply_template(const char * s) {\nstruct Slot {\n#define MAX_SLOTS 8\n    if (x) {\n').map(s => s.name))
      .toEqual(['apply_template', 'Slot', 'MAX_SLOTS'])
    expect(extractSymbols('b.ps1', 'function Get-Thing {\n').map(s => s.name)).toEqual(['Get-Thing'])
  })

  it('indexes a workspace, maps it, and finds definitions', () => {
    const root = mkdtempSync(join(tmpdir(), 'repomap-'))
    mkdirSync(join(root, 'pkg', 'sub'), { recursive: true })
    mkdirSync(join(root, 'node_modules', 'dep'), { recursive: true })
    writeFileSync(join(root, 'pkg', 'cache.py'), 'def memo(fn):\n    pass\n\ndef memo_kw(fn):\n    pass\n')
    writeFileSync(join(root, 'pkg', 'sub', 'fx.py'), 'from ..cache import memo\n\n@memo\ndef rate(c):\n    return 1\n')
    writeFileSync(join(root, 'node_modules', 'dep', 'x.js'), 'function hidden() {}\n')
    writeFileSync(join(root, 'README.md'), '# r\n')
    const index = buildIndex(root)
    expect([...index.code.keys()].sort()).toEqual(['pkg/cache.py', 'pkg/sub/fx.py'])
    const map = renderMap(index, undefined)
    expect(map).toContain('pkg/cache.py (6 lines): memo, memo_kw')
    expect(map).toContain('pkg/sub/fx.py (6 lines): rate')
    expect(map).toContain('other files: 1 (.md 1)')
    const defs = findDefinitions(index, 'memo')
    expect(defs.map(d => `${d.file}:${d.symbol.line}`)).toEqual(['pkg/cache.py:1'])
    expect(renderSymbol('memo', defs, ['pkg/cache.py:1:def memo(fn):', 'pkg/sub/fx.py:3:@memo']))
      .toBe('Definitions of "memo":\n  pkg/cache.py:1  [def] def memo(fn):\nReferences (1):\n  pkg/sub/fx.py:3:@memo')
    expect(renderMap(index, 'pkg/sub')).toContain('fx.py (6 lines): rate')
  })
})
