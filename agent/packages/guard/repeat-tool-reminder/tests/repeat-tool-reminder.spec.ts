import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { AttachmentId } from '@deepseek-ai/dsh-attachment'
import { createUserMessage, ToolCallId  } from '@deepseek-ai/dsh-llm'
import type { ContextFormed } from '@deepseek-ai/dsh-llm'
import { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import { defineContentToolFixture } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import * as RepeatToolGuard from '@deepseek-ai/dsh-repeat-tool-reminder'
import type { Config } from '@deepseek-ai/dsh-repeat-tool-reminder'
import { MockAdapter, textResponse, toolCallResponse } from '../../../core/agent-loop/tests/mock-adapter.ts'

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    'test': { kind: 'test' } & ContextFormed
  }
}

const testToolSignal = new AbortController().signal

it('reminds across changing investigation calls and resets on successful configured tools', async () => {
  const ctx = await harness({ thresholds: [99], investigationThresholds: [2], investigationResetTools: ['other'] })
  const adapter = new MockAdapter([
    toolCallResponse('c1', 'probe', { q: 'one' }),
    toolCallResponse('c2', 'probe', { q: 'two' }),
    toolCallResponse('c3', 'other', {}),
    toolCallResponse('c4', 'probe', { q: 'three' }),
    toolCallResponse('c5', 'probe', { q: 'four' }),
    textResponse('done'),
  ])
  ctx.llm.registerAdapter(['mock'], adapter)
  const agent = await ctx.agentLoop.create(SessionId('investigation'), { provider: 'mock', model: 'mock' })
  agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
  await waitForIdle(ctx, agent)
  expect(reminders(agent).map(item => item.text)).toMatchInlineSnapshot(`
    [
      "You have made 2 tool calls since a successful file mutation. Consolidate the facts already verified, identify the unresolved question, and check an authoritative reference or a runnable assertion before continuing. This is an advisory checkpoint, not evidence that the investigation is wrong.",
      "You have made 2 tool calls since a successful file mutation. Consolidate the facts already verified, identify the unresolved question, and check an authoritative reference or a runnable assertion before continuing. This is an advisory checkpoint, not evidence that the investigation is wrong.",
    ]
  `)
})

/**
 * Behavior suite for the repeat-tool-call guard: chain semantics (identical /
 * different-tracked / untracked-transparent / per-agent / resets), threshold
 * escalation incl. the `thresholds[0]` gentle-text rule, canonicalization,
 * fold-onto-downstream-decision, and fail-loud config validation — all driven
 * through a real agent loop against a scripted mock adapter (no network).
 */

/** Boot the core spine + the guard; the caller registers adapters and extra listeners. */
async function harness(config: Config = {}): Promise<Context> {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(RepeatToolGuard, config)
  ctx.tools.register(defineContentToolFixture({ name: 'probe', description: 'p', parameters: {}, async execute() { return [{ type: 'text', text: 'ok' }] } }))
  ctx.tools.register(defineContentToolFixture({ name: 'other', description: 'o', parameters: {}, async execute() { return [{ type: 'text', text: 'ok' }] } }))
  return ctx
}

function waitForIdle(ctx: Context, agent: Agent): Promise<void> {
  return new Promise((resolve) => { const d = ctx.on('agent/status', ({ agent: s, status: st }) => { if (s === agent && st === 'idle') { d(); resolve() } }) })
}

/** Every injected-context user message in the agent's log, flattened to joined text + source for terse assertions. */
function reminders(agent: Agent): { text: string; source: unknown }[] {
  return agent.session.snapshotEvents()
    .filter((e): e is SessionEvent<'user/message'> => e.type === 'user/message' && e.data.source.kind !== 'user')
    .map(e => ({
      text: e.data.content.map(block => block.type === 'text' ? block.text : '').join('|'),
      source: e.data.source,
    }))
}

// The reminder is a `notice`-form context; its summary names the repeated
// call so a reader sees it without expanding the row.
const guardSource = (tool: string, count: number) => ({
  kind: 'repeat-tool-reminder',
  form: 'notice',
  summary: `${tool} × ${count}`,
})

describe('threshold escalation', () => {
  it('reminds gently at the first default threshold (3) and in detail at the second (5)', async () => {
    const ctx = await harness()
    const adapter = new MockAdapter([
      ...Array.from({ length: 5 }, (_, i) => toolCallResponse(`c${i}`, 'probe', { q: 'same' })),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(2)
    expect(found[0]!.text).toContain('repeating the exact same tool call')
    expect(found[0]!.source).toEqual(guardSource('probe', 3))
    expect(found[1]!.text).toContain('consecutive_calls: 5')
    expect(found[1]!.text).toContain('- tool: probe')
    expect(found[1]!.text).toContain('{"q":"same"}')
    expect(found[1]!.source).toEqual(guardSource('probe', 5))
  })

  it('keys the gentle text to thresholds[0], not the literal 3', async () => {
    const ctx = await harness({ thresholds: [4, 2] }) // unsorted on purpose: normalized ascending
    const adapter = new MockAdapter([
      ...Array.from({ length: 4 }, (_, i) => toolCallResponse(`c${i}`, 'probe', {})),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(2)
    expect(found[0]!.text).toContain('repeating the exact same tool call') // gentle at 2
    expect(found[1]!.text).toContain('consecutive_calls: 4') // detailed at 4
  })
})

describe('chain semantics', () => {
  it('caps the detailed reminder arguments at argumentsPreviewChars (detection still keys on the full string)', async () => {
    const ctx = await harness({ thresholds: [2, 3], argumentsPreviewChars: 24 })
    const bigPayload = 'x'.repeat(400)
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { body: bigPayload }),
      toolCallResponse('c2', 'probe', { body: bigPayload }),
      toolCallResponse('c3', 'probe', { body: bigPayload }),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(2) // gentle at 2, detailed at 3 — full-key matching survived the cap
    const detailed = found[1]!.text
    expect(detailed).toContain('- arguments: {"body":"xxxxxxxxxxxxxx') // 24-char head
    expect(detailed).toContain('… (+387 more chars)')
    expect(detailed).not.toContain(bigPayload)
  })

  it('a different tracked call resets the chain', async () => {
    const ctx = await harness()
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      toolCallResponse('c2', 'probe', { q: 1 }),
      toolCallResponse('c3', 'other', {}), // tracked, different → reset
      toolCallResponse('c4', 'probe', { q: 1 }),
      toolCallResponse('c5', 'probe', { q: 1 }),
      toolCallResponse('c6', 'probe', { q: 1 }), // 3rd consecutive AFTER the reset
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(1)
  })

  it('excluded calls are transparent: they neither count nor reset', async () => {
    const ctx = await harness({ exclude: ['other'] })
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      toolCallResponse('c2', 'other', {}), // excluded → invisible to the chain
      toolCallResponse('c3', 'probe', { q: 1 }),
      toolCallResponse('c4', 'other', {}),
      toolCallResponse('c5', 'probe', { q: 1 }), // 3rd consecutive probe
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(1)
    expect(found[0]!.text).toContain('repeating the exact same tool call')
  })

  it('include patterns track only matching tools (wildcard star)', async () => {
    const ctx = await harness({ include: ['pro*'] })
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'other', {}),
      toolCallResponse('c2', 'other', {}),
      toolCallResponse('c3', 'other', {}), // 3 identical, but untracked
      toolCallResponse('c4', 'probe', {}),
      toolCallResponse('c5', 'probe', {}),
      toolCallResponse('c6', 'probe', {}), // 3 identical, tracked
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(1)
    expect(found[0]!.text).toContain('repeating the exact same tool call')
  })

  it('escapes regex metacharacters in patterns (a dot matches only a literal dot)', async () => {
    const ctx = await harness({ exclude: ['pr.be'] }) // would match 'probe' as a regex; must not as a wildcard
    const adapter = new MockAdapter([
      ...Array.from({ length: 3 }, (_, i) => toolCallResponse(`c${i}`, 'probe', {})),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(1) // probe was NOT excluded
  })

  it('canonicalization ignores property order, deeply', async () => {
    const ctx = await harness()
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { a: 1, nested: { x: [1, 2], y: null } }),
      toolCallResponse('c2', 'probe', { nested: { y: null, x: [1, 2] }, a: 1 }),
      toolCallResponse('c3', 'probe', { a: 1, nested: { x: [1, 2], y: null } }),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(1) // all three canonicalize identically
  })

  it('keys chains per agent: one agent repeating never trips another', async () => {
    const ctx = await harness()
    ctx.llm.registerAdapter(['mock-a'], new MockAdapter([
      toolCallResponse('a1', 'probe', { q: 1 }),
      toolCallResponse('a2', 'probe', { q: 1 }),
      textResponse('done'),
    ]))
    ctx.llm.registerAdapter(['mock-b'], new MockAdapter([
      toolCallResponse('b1', 'probe', { q: 1 }),
      toolCallResponse('b2', 'probe', { q: 1 }),
      toolCallResponse('b3', 'probe', { q: 1 }),
      textResponse('done'),
    ]))
    const agentA = await ctx.agentLoop.create(SessionId('a'), { provider: 'mock-a', model: 'model-a' })
    const agentB = await ctx.agentLoop.create(SessionId('b'), { provider: 'mock-b', model: 'model-b' })
    agentA.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    agentB.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await Promise.all([waitForIdle(ctx, agentA), waitForIdle(ctx, agentB)])

    expect(reminders(agentA)).toHaveLength(0) // 2 repeats < 3, despite B's 3 in the same registry
    expect(reminders(agentB)).toHaveLength(1)
  })

  it('a new user prompt resets the chain', async () => {
    const ctx = await harness()
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      toolCallResponse('c2', 'probe', { q: 1 }),
      textResponse('turn one done'),
      toolCallResponse('c3', 'probe', { q: 1 }), // without the reset this would be the 3rd
      textResponse('turn two done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'again' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(0)
  })

  it('drops an agent chain on disposal', async () => {
    const ctx = await harness({ thresholds: [2] })
    ctx.llm.registerAdapter(['mock'], new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      textResponse('done'),
      toolCallResponse('c2', 'probe', { q: 1 }), // same id, fresh agent: count 1, not 2
      textResponse('done'),
    ]))
    // Loop agents are torn down by disposing the scope that created them
    // (the loop.spec pattern): a child plugin fiber owns `first`.
    let first!: Agent
    const fiber = await ctx.plugin(Object.assign(async (inner: Context) => {
      first = await inner.agentLoop.create(SessionId('reused'), { provider: 'mock', model: 'mock' })
    }, { inject: ['agentLoop'] }))
    first.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, first)
    await fiber.dispose()
    await first.whenIdle()

    const second = await ctx.agentLoop.create(SessionId('reused'), { provider: 'mock', model: 'mock' })
    second.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, second)

    expect(reminders(second)).toHaveLength(0)
  })

  it('counts denied calls: hammering a denied tool still draws the reminder', async () => {
    const ctx = await harness({ thresholds: [2] })
    ctx.on('tools/pre-execute', async () => ({ kind: 'deny' as const, reason: 'sealed' }))
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      toolCallResponse('c2', 'probe', { q: 1 }),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(1)
  })

  it('ignores direct executes with no agent (they neither crash nor advance any chain)', async () => {
    const ctx = await harness({ thresholds: [2] })
    const direct = await ctx.tools.execute({ signal: testToolSignal, callId: ToolCallId('d1'), name: 'probe', arguments: { q: 1 } })
    expect(direct.isError).toBe(false)

    ctx.llm.registerAdapter(['mock'], new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }), // if the direct call had counted, this would be #2
      textResponse('done'),
    ]))
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(0)
  })
})

describe('fold onto the downstream decision', () => {
  it('folds the reminder onto a downstream block and keeps its feedback', async () => {
    const ctx = await harness({ thresholds: [2] })
    ctx.on('tools/post-execute', async () => ({
      kind: 'block' as const,
      feedback: [{ type: 'text' as const, text: 'nope' }],
      additionalContexts: [createUserMessage({
        content: [{ type: 'text' as const, text: 'downstream-ctx' }], source: { kind: 'test' as const },
      })],
    }))
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      toolCallResponse('c2', 'probe', { q: 1 }),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(3)
    // Only the repeated call adds guard context; downstream source fields survive.
    expect(found[0]!.text).toBe('downstream-ctx')
    expect(found[0]!.source).toEqual({ kind: 'test' })
    expect(found[1]!.text).toContain('repeating the exact same tool call')
    expect(found[1]!.source).toEqual(guardSource('probe', 2))
    expect(found[2]).toEqual({ text: 'downstream-ctx', source: { kind: 'test' } })
    // The block's feedback reached the tool result unchanged.
    const results = agent.session.snapshotEvents().filter((e): e is SessionEvent<'tool/result'> => e.type === 'tool/result')
    expect(results.every(r => r.data.message.isError)).toBe(true)
    expect(results[1]!.data.message.content).toEqual([{ type: 'text', text: 'nope' }])
  })

  it('preserves a downstream canonical value replacement while folding', async () => {
    const ctx = await harness({ thresholds: [2] })
    ctx.on('tools/post-execute', async () => ({
      kind: 'accept' as const,
      value: [{ type: 'text' as const, text: 'replaced' }],
    }))
    const adapter = new MockAdapter([
      toolCallResponse('c1', 'probe', { q: 1 }),
      toolCallResponse('c2', 'probe', { q: 1 }),
      textResponse('done'),
    ])
    ctx.llm.registerAdapter(['mock'], adapter)
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(1)
    expect(found[0]!.text).toContain('repeating the exact same tool call')
    const results = agent.session.snapshotEvents().filter((e): e is SessionEvent<'tool/result'> => e.type === 'tool/result')
    expect(results[1]!.data.message.content).toEqual([{ type: 'text', text: 'replaced' }])
  })
})

describe('same-result detection', () => {
  /** The core harness plus `run`, a tool whose output is its `out` argument (the `file` argument only varies the call). */
  async function resultHarness(config: Config): Promise<Context> {
    const ctx = await harness(config)
    ctx.tools.register(defineContentToolFixture({
      name: 'run',
      description: 'r',
      parameters: { file: { type: 'string' }, out: { type: 'string' } },
      async execute(args) { return [{ type: 'text', text: args.out ?? '' }] },
    }))
    return ctx
  }

  /** Run one turn of scripted `run` calls, one per output, each with a fresh `file` argument; then finish. */
  async function runTurn(ctx: Context, outputs: string[]): Promise<Agent> {
    ctx.llm.registerAdapter(['mock'], new MockAdapter([
      ...outputs.map((out, i) => toolCallResponse(`c${i}`, 'run', { file: `rtest${i}.py`, out })),
      textResponse('done'),
    ]))
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)
    return agent
  }

  it('is off by default: varied calls returning one result draw no reminder', async () => {
    const ctx = await resultHarness({})
    expect(reminders(await runTurn(ctx, Array.from({ length: 8 }, () => '10000')))).toHaveLength(0)
  })

  it('reminds gently at resultThresholds[0] and in detail at the next, though every call differs', async () => {
    const ctx = await resultHarness({ resultThresholds: [3, 5] })
    const found = reminders(await runTurn(ctx, ['10000', '10000', 'other', '10000', '10000', '10000']))
    expect(found).toHaveLength(2)
    expect(found[0]!.text).toContain('keep returning the same result')
    expect(found[0]!.source).toEqual({ kind: 'repeat-tool-reminder', form: 'notice', summary: 'run same result × 3' })
    expect(found[1]!.text).toContain('- tool: run')
    expect(found[1]!.text).toContain('- same_result_count: 5 of the last 6 calls')
    expect(found[1]!.text).toContain('- result: 10000')
  })

  it('counts only results inside the window', async () => {
    const ctx = await resultHarness({ resultThresholds: [3], resultWindow: 3 })
    // the same result 4 times, but never 3 times within any 3 consecutive calls
    expect(reminders(await runTurn(ctx, ['10000', 'a', '10000', 'b', '10000', 'c', '10000']))).toHaveLength(0)
  })

  it('reminds once per threshold when the window slides away and back', async () => {
    const ctx = await resultHarness({ resultThresholds: [3], resultWindow: 4 })
    // count reaches 3 at call 3, drops to 2 when call 1 leaves the window, and reaches 3 again at call 5
    expect(reminders(await runTurn(ctx, ['10000', '10000', '10000', 'x', '10000']))).toHaveLength(1)
  })

  it('compares only tools matching resultInclude', async () => {
    const ctx = await resultHarness({ resultThresholds: [2], resultInclude: ['pwsh', 'bash'] })
    expect(reminders(await runTurn(ctx, ['10000', '10000', '10000']))).toHaveLength(0)
  })

  it('delivers the exact-repeat reminder alone when both detectors fire on one call', async () => {
    const ctx = await harness({ thresholds: [3], resultThresholds: [3] })
    ctx.llm.registerAdapter(['mock'], new MockAdapter([
      ...Array.from({ length: 3 }, (_, i) => toolCallResponse(`c${i}`, 'probe', { q: 'same' })),
      textResponse('done'),
    ]))
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found).toHaveLength(1)
    expect(found[0]!.text).toContain('repeating the exact same tool call')
  })

  it('a new user prompt clears the result history', async () => {
    const ctx = await resultHarness({ resultThresholds: [3] })
    ctx.llm.registerAdapter(['mock'], new MockAdapter([
      toolCallResponse('c1', 'run', { file: 'a.py', out: '10000' }),
      toolCallResponse('c2', 'run', { file: 'b.py', out: '10000' }),
      textResponse('turn one done'),
      toolCallResponse('c3', 'run', { file: 'c.py', out: '10000' }), // without the reset this would be the 3rd
      textResponse('turn two done'),
    ]))
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'again' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    expect(reminders(agent)).toHaveLength(0)
  })

  it('appends reminderSuffix to every reminder', async () => {
    const ctx = await resultHarness({ thresholds: [2], resultThresholds: [2], reminderSuffix: '<|think_xhigh|>' })
    const found = reminders(await runTurn(ctx, ['10000', '10000']))
    expect(found).toHaveLength(1)
    expect(found[0]!.text.endsWith('\n\n<|think_xhigh|>')).toBe(true)
  })

  it('compares whole blocks: equal images repeat, different images do not, and the quote names the block type', async () => {
    const ctx = await harness({ resultThresholds: [2, 3] })
    ctx.tools.register(defineContentToolFixture({
      name: 'shot',
      description: 's',
      parameters: { data: { type: 'string' } },
      async execute(args) { return [{ type: 'image', attachment: {
        attachmentId: AttachmentId(args.data ?? ''), mediaType: 'image/png', bytes: 1, width: 1, height: 1,
      } }] },
    }))
    ctx.llm.registerAdapter(['mock'], new MockAdapter([
      toolCallResponse('c1', 'shot', { data: 'AQ==' }),
      toolCallResponse('c2', 'shot', { data: 'Ag==' }), // a different image: no repeat
      toolCallResponse('c3', 'shot', { data: 'AQ==' }), // 2nd equal → gentle
      toolCallResponse('c4', 'shot', { data: 'AQ==' }), // 3rd equal → detailed
      textResponse('done'),
    ]))
    const agent = await ctx.agentLoop.create(SessionId('a1'), { provider: 'mock', model: 'mock' })
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }], source: { kind: 'user' } }))
    await waitForIdle(ctx, agent)

    const found = reminders(agent)
    expect(found.map(r => r.source)).toEqual([
      { kind: 'repeat-tool-reminder', form: 'notice', summary: 'shot same result × 2' },
      { kind: 'repeat-tool-reminder', form: 'notice', summary: 'shot same result × 3' },
    ])
    expect(found[1]!.text).toContain('- result: [image content]')
  })

  it('caps the quoted result at resultPreviewChars', async () => {
    const ctx = await resultHarness({ resultThresholds: [2, 3], resultPreviewChars: 10 })
    const big = 'y'.repeat(100)
    const found = reminders(await runTurn(ctx, [big, big, big]))
    expect(found[1]!.text).toContain('- result: yyyyyyyyyy… (+90 more chars)')
    expect(found[1]!.text).not.toContain(big)
  })
})

describe('config validation fails loud', () => {
  async function spine(): Promise<Context> {
    const ctx = new Context()
    await mountAgentLoopTestDependencies(ctx)
    await ctx.plugin(AgentLoop, { agents: [] })
    return ctx
  }

  it('rejects an empty thresholds list', async () => {
    const ctx = await spine()
    await expect(ctx.plugin(RepeatToolGuard, { thresholds: [] })).rejects.toThrow(/must not be empty/)
  })

  it('rejects a threshold below 2', async () => {
    const ctx = await spine()
    await expect(ctx.plugin(RepeatToolGuard, { thresholds: [1, 3] })).rejects.toThrow(/integer >= 2/)
  })

  it('rejects a non-integer threshold', async () => {
    const ctx = await spine()
    await expect(ctx.plugin(RepeatToolGuard, { thresholds: [2.5] })).rejects.toThrow(/integer >= 2/)
  })

  it('rejects duplicate thresholds', async () => {
    const ctx = await spine()
    await expect(ctx.plugin(RepeatToolGuard, { thresholds: [3, 3] })).rejects.toThrow(/duplicates/)
  })

  it('rejects a non-positive or fractional argumentsPreviewChars', async () => {
    const ctx = await spine()
    await expect(ctx.plugin(RepeatToolGuard, { argumentsPreviewChars: 0 })).rejects.toThrow(/argumentsPreviewChars/)
    const ctx2 = await spine()
    await expect(ctx2.plugin(RepeatToolGuard, { argumentsPreviewChars: 12.5 })).rejects.toThrow(/argumentsPreviewChars/)
  })

  it('rejects resultThresholds entries below 2 or duplicated, and accepts an empty list', async () => {
    await expect((await spine()).plugin(RepeatToolGuard, { resultThresholds: [1] })).rejects.toThrow(/resultThresholds entry 1/)
    await expect((await spine()).plugin(RepeatToolGuard, { resultThresholds: [3, 3] }))
      .rejects.toThrow(/`resultThresholds` must not contain duplicates/)
    await expect((await spine()).plugin(RepeatToolGuard, { resultThresholds: [] })).resolves.toBeDefined()
  })

  it('rejects a resultWindow smaller than the highest resultThresholds entry', async () => {
    await expect((await spine()).plugin(RepeatToolGuard, { resultThresholds: [3, 8], resultWindow: 5 }))
      .rejects.toThrow(/could never be reached/)
  })

  it('rejects a non-positive resultWindow or resultPreviewChars', async () => {
    await expect((await spine()).plugin(RepeatToolGuard, { resultWindow: 0 })).rejects.toThrow(/resultWindow/)
    await expect((await spine()).plugin(RepeatToolGuard, { resultPreviewChars: 1.5 })).rejects.toThrow(/resultPreviewChars/)
  })
})
