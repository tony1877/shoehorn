import { describe, expect, it } from 'vitest'
import { announcesAction } from '../src/agent.ts'

describe('announcesAction (auto-continue)', () => {
  it('flags a reply that stops right before the action it announced', () => {
    expect(announcesAction('The tests pass. Now I\'ll update the README:')).toBe(true)
    expect(announcesAction('Done with the parser.\n\nLet me run the tests next.')).toBe(true)
    expect(announcesAction('Next, I will add the migration.')).toBe(true)
    expect(announcesAction('I’m going to rebuild the project')).toBe(true)
  })

  it('leaves finished answers, questions and labelled summaries alone', () => {
    expect(announcesAction('')).toBe(false)
    expect(announcesAction('All 12 tests pass and the bug is fixed.')).toBe(false)
    expect(announcesAction('Should I also update the docs?')).toBe(false)
    expect(announcesAction('Everything is in place.\n\n**Summary:**')).toBe(false)
    expect(announcesAction('Files:')).toBe(false)
  })
})
