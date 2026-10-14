import { mkdtemp, readFile, rmdir, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium, type Browser } from 'playwright'
import { expect, it } from 'vitest'
import { ToolCallId, type StreamChunk } from '@deepseek-ai/dsh-llm'
import { launchWebScaffold, type WebScaffold } from './scaffold.ts'
import { connectFreshWorkspace, newEnglishPage } from './support.ts'

it('shows generated file content before executing the completed write', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-file-generation-'))
  const override = join(directory, 'replay.override.json')
  const id = ToolCallId('streamed-file')
  const fragments = ['{"file_path":"preview.py","content":"STREAM_FILE_FIRST ', ...Array.from({ length: 80 }, () => 'next '), '"}']
  const argumentsJson = fragments.join('')
  const chunks: StreamChunk[] = [
    { type: 'block-start', index: 0, blockType: 'tool-call' },
    ...fragments.map(argumentsDelta => ({ type: 'tool-call-delta' as const, index: 0, id, name: 'write', argumentsDelta })),
    { type: 'block-end', index: 0, block: { type: 'tool-call', id, name: 'write', arguments: argumentsJson } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ]
  await writeFile(override, JSON.stringify([
    { kind: 'chunks', chunks: [
      { type: 'tool-call-delta', index: 0, id: ToolCallId('observe-file'), name: 'read', argumentsDelta: '{"file_path":"preview.py"}' },
      { type: 'block-end', index: 0, block: { type: 'tool-call', id: ToolCallId('observe-file'), name: 'read', arguments: '{"file_path":"preview.py"}' } },
      { type: 'finish', reason: { kind: 'tool-calls' } },
    ] },
    { kind: 'chunks', chunks },
    { kind: 'chunks', chunks: [
      { type: 'text-delta', index: 0, text: 'FILE_DONE' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'FILE_DONE' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ] },
  ]))
  let scaffold: WebScaffold | undefined
  let browser: Browser | undefined
  try {
    scaffold = await launchWebScaffold({ replayFixture: join(directory, 'override-only.jsonl'), replayOverride: override, paceMs: 75 })
    browser = await chromium.launch({ channel: process.env.DSH_TEST_BROWSER_CHANNEL })
    let executed = false
    let toolError = ''
    scaffold.ctx.on('session/event', (_session, event) => {
      if (event.type === 'tool/call' && event.data.name === 'write') executed = true
      if (event.type === 'tool/result' && event.data.message.source.callId === id && event.data.message.content[0].isError) {
        toolError = event.data.message.content[0].content.map(block => block.type === 'text' ? block.text : '').join('\n')
      }
    })
    const page = await newEnglishPage(browser, 900)
    await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
    await connectFreshWorkspace(page, scaffold.workspaceCwd, 'file-generation-e2e')
    const composer = page.locator('[data-composer-input][contenteditable="true"]').last()
    await composer.fill('Write preview.py with the streamed fixture content.')
    const settled = scaffold.whenTurnSettled(30_000)
    await page.getByRole('button', { name: 'Send message', exact: true }).click()
    const preview = page.locator('[data-file-generation]')
    await expect.poll(() => preview.textContent(), { timeout: 15_000 }).toContain('STREAM_FILE_FIRST')
    expect(executed).toBe(false)
    const target = join(scaffold.workspaceCwd, 'file-generation-e2e', 'preview.py')
    await expect(readFile(target)).rejects.toMatchObject({ code: 'ENOENT' })
    await settled
    expect(toolError).toBe('')
    expect(executed).toBe(true)
    expect(await readFile(target, 'utf8')).toBe('STREAM_FILE_FIRST ' + 'next '.repeat(80))
    await expect.poll(() => preview.count()).toBe(0)
  } finally {
    await browser?.close()
    await scaffold?.close()
    await unlink(override)
    await rmdir(directory)
  }
}, 120_000)
