import { describe, expect, it } from 'vitest'
import { stripCallMarkup } from '../src/summarizer.ts'

describe('stripCallMarkup', () => {
  it('removes the closing tags a summary written as a tool call leaves at its end', () => {
    const out = stripCallMarkup([{ type: 'text', text: '## Primary Request\n- a\n\n</parameter>\n</function>\n</tool_call>' }])
    expect(out[0]!.text).toBe('## Primary Request\n- a\n')
  })

  it('removes the opening tags at its start', () => {
    const out = stripCallMarkup([{ type: 'text', text: '<tool_call>\n<function=write>\n<parameter=content>\n## Primary Request' }])
    expect(out[0]!.text).toBe('## Primary Request')
  })

  it('leaves an ordinary summary alone', () => {
    const text = '## Primary Request\n- quotes `</tool_call>` in the middle\n- done'
    expect(stripCallMarkup([{ type: 'text', text }])[0]!.text).toBe(text)
  })
})

describe('stripCallMarkup and the checkpoint tags', () => {
  it('removes an echoed closing checkpoint tag', () => {
    const out = stripCallMarkup([{ type: 'text', text: '## Primary Request\n- a\n</compacted-summary>\n' }])
    expect(out[0]!.text).toBe('## Primary Request\n- a\n')
  })
})
