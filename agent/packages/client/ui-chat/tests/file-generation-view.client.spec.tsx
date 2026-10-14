// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest'
import { cleanup, render } from '@testing-library/react'
import { makeTranslate } from '@deepseek-ai/dsh-client-test-runtime'
import { zh as commonZh } from '@deepseek-ai/dsh-client-locale/src/locales/zh.ts'
import { zh } from '../src/client/locale.ts'
import { AssistantMarkdown } from '../src/client/chat/AssistantMarkdown.tsx'

afterEach(cleanup)
it('renders the actual file prefix and updates it during generation', () => {
  const t = makeTranslate(zh, commonZh)
  const block = (content: string) => [{ kind: 'tool-call' as const, callId: 'file', name: 'write', argsRaw: `{"file_path":"a.py","content":"${content}` }]
  const view = render(<AssistantMarkdown t={t} blocks={block('print(1)')} streaming renderMessageImages={() => null} />)
  expect(view.container.querySelector('[data-file-generation]')?.textContent).toContain('print(1)')
  view.rerender(<AssistantMarkdown t={t} blocks={block('print(1)\\nprint(2)')} streaming renderMessageImages={() => null} />)
  expect(view.container.querySelector('[data-file-generation]')?.textContent).toContain('print(2)')
  view.rerender(<AssistantMarkdown t={t} blocks={block('print(1)')} streaming={false} renderMessageImages={() => null} />)
  expect(view.container.querySelector('[data-file-generation]')).toBeNull()
})
