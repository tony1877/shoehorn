/**
 * Chats owned by an integration (a Telegram chat, a webhook in "same chat"
 * mode): start one, find it again while it is live, or resume it from the
 * session log after a restart, then hand it the next message.
 */

import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { ContentBlock, MessageSource } from '@deepseek-ai/dsh-llm'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { createPromptedSession } from '@deepseek-ai/dsh-webhook'
import type {} from '@deepseek-ai/dsh-agent-presets'
import type {} from '@deepseek-ai/dsh-agent-default-model'

/** Where a new chat works and how it is composed. */
export interface ChatPlacement {
  workspacePath: string
  agentPreset: string
  permissionPreset: string
}

/**
 * Start a chat whose first message is `content`.
 * @param ctx - context carrying the session services.
 * @returns the new chat's id and live agent.
 */
export async function startChat(
  ctx: Context,
  placement: ChatPlacement,
  title: string,
  content: ContentBlock[],
  source: MessageSource,
  signal: AbortSignal,
): Promise<{ sessionId: SessionId; agent: Agent }> {
  const text = content.map(block => (block.type === 'text' ? block.text : '')).join('\n').trim()
  const created = await createPromptedSession(ctx, {
    workspacePath: placement.workspacePath,
    title: title.trim() === '' ? 'Untitled chat' : title.trim().slice(0, 120),
    // The helper admits one text prompt; images follow right after as part of the same turn's inbox.
    prompt: text === '' ? '(see the attached image)' : text,
    agentPreset: placement.agentPreset,
    permissionPreset: placement.permissionPreset,
  }, signal, source)
  const extra = content.filter(block => block.type !== 'text')
  if (extra.length > 0) created.agent.followup(createUserMessage({ content: extra, source }))
  return created
}

/**
 * The live agent of a chat, resuming it from its log when it is not live.
 * @returns the agent, or `undefined` when the chat no longer exists.
 */
export async function findChat(ctx: Context, sessionId: string, agentPreset: string): Promise<Agent | undefined> {
  const id = brandString<SessionId>(sessionId)
  const live = ctx.agents.get(id)
  if (live !== undefined) return live
  try {
    const preset = await ctx.agentPresets.resolve(agentPreset)
    const selected = ctx.agentDefaultModel.currentSelection()
    const handle = await ctx.agents.resume({
      resumeSessionId: id,
      agentOptions: { provider: selected.provider, model: selected.model },
      setup: async (agentCtx) => {
        await ctx.agentPresets.mount(agentCtx, preset.id)
      },
    })
    return handle.agent
  } catch {
    // Deleted, unreadable, or composed with a preset that is gone: the caller starts a new chat.
    return undefined
  }
}

/** Hand a chat its next message. */
export function sendToChat(agent: Agent, content: ContentBlock[], source: MessageSource): void {
  agent.followup(createUserMessage({ content, source }))
}

/** Text of the newest assistant message with any text, or `''`. */
export function lastAnswer(agent: Agent): string {
  const events = agent.session.snapshotEvents()
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'assistant/message') continue
    const text = event.data.message.content
      .map(block => (block.type === 'text' ? block.text : ''))
      .join('')
      .trim()
    if (text !== '') return text
  }
  return ''
}

/**
 * The newest assistant text written at or after event `from`.
 * @returns the text; `''` when the turn wrote assistant messages without text; `undefined` when it wrote none.
 */
export function answerSince(agent: Agent, from: number): string | undefined {
  const events = agent.session.snapshotEvents()
  let sawAssistant = false
  for (let index = events.length - 1; index >= from; index -= 1) {
    const event = events[index]!
    if (event.type !== 'assistant/message') continue
    sawAssistant = true
    const text = event.data.message.content
      .map(block => (block.type === 'text' ? block.text : ''))
      .join('')
      .trim()
    if (text !== '') return text
  }
  return sawAssistant ? '' : undefined
}

/** Number of events in a chat's log, used to tell new answers from ones already relayed. */
export function eventCount(agent: Agent): number {
  return agent.session.snapshotEvents().length
}

/** Arguments of one tool call in a chat, for approval prompts shown elsewhere. */
export function toolCallArguments(agent: Agent, callId: string | undefined): string | undefined {
  if (callId === undefined) return undefined
  const events = agent.session.snapshotEvents()
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'assistant/message') continue
    for (const block of event.data.message.content) {
      if (block.type === 'tool-call' && block.id === callId) return block.arguments
    }
  }
  return undefined
}
