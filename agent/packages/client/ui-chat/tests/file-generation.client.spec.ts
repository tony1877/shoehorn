import { describe, expect, it } from 'vitest'
import { fileGeneration } from '../src/client/chat/file-generation.ts'

describe('streamed file generation', () => {
  it('decodes content before the tool call closes', () => {
    expect(fileGeneration('write', '{"file_path":"a.py","content":"print(1)\\nnext'))
      .toEqual({ path: 'a.py', text: 'print(1)\nnext' })
  })
  it('holds incomplete escapes and decodes quotes and Unicode', () => {
    expect(fileGeneration('write', '{"content":"a\\')).toEqual({ path: '', text: 'a' })
    expect(fileGeneration('write', '{"content":"a\\u12')).toEqual({ path: '', text: 'a' })
    expect(fileGeneration('write', '{"content":"\\u00e9\\"x')).toEqual({ path: '', text: 'é"x' })
  })
  it('does not mistake field-like file content for root fields', () => {
    expect(fileGeneration('write', JSON.stringify({ content: '"file_path":"wrong"', file_path: 'right' })))
      .toEqual({ path: 'right', text: '"file_path":"wrong"' })
  })
  it('previews replacement text and bounds large writes', () => {
    expect(fileGeneration('edit', '{"file_path":"a","old_string":"old","new_string":"new'))
      .toEqual({ path: 'a', text: 'new' })
    expect(fileGeneration('write', JSON.stringify({ content: 'x'.repeat(20000) }))?.text).toHaveLength(16384)
    expect(fileGeneration('pwsh', '{"content":"no"}')).toBeUndefined()
  })
  it('does not promote nested values or invalid prefixes into file content', () => {
    expect(fileGeneration('write', '{"metadata":{"content":"not file"},"content":"actual'))
      .toEqual({ path: '', text: 'actual' })
    expect(fileGeneration('write', 'garbage {"content":"no"}')).toBeUndefined()
  })
})
