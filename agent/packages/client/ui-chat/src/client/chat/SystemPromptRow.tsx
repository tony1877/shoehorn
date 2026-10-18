import { memo, useState } from 'react'
import type { ChatNodeViewProps, ChatViewSlotProps } from '../contract/slots.ts'
import { DisclosureRow, IconBrowseOutlineRegular } from '@deepseek-ai/dsh-client-ui-primitives'
import { OpaqueBody } from './ContextBody.tsx'
import css from './ContextInjectionRow.module.css'

/** Props for one complete system prompt disclosure. */
export interface SystemPromptRowProps {
  /** Complete model-visible prompt text. */
  text: string
  /** The same prompt again at a rewritten conversation's new request series. */
  repeat?: boolean
  /** The owning view's locale seat. */
  t: ChatViewSlotProps['t']
}

/**
 * Render one complete system prompt as a collapsed disclosure whose expanded
 * body is the same opaque context chrome: 141px code-block scrollport and
 * model-facing text with its real line breaks. An in-history update uses the
 * same row under its own title.
 * @param props - Complete prompt text, whether it is an update, and the locale seat.
 * @returns The system-prompt disclosure row.
 */
export function SystemPromptRow({ text, repeat = false, t }: SystemPromptRowProps) {
  const [open, setOpen] = useState(false)
  return (
    <DisclosureRow
      className={css.root}
      icon={<IconBrowseOutlineRegular size={14} />}
      chevronClassName={css.chevron}
      title={t(repeat ? 'message.systemPromptRepeat' : 'message.systemPrompt')}
      open={open}
      expandable
      expandOnRowClick
      onToggle={() => { setOpen(value => !value) }}
    >
      <div className={css.body} data-system-prompt-body>
        <OpaqueBody content={[{ type: 'text', text }]} source={null} t={t} />
      </div>
    </DisclosureRow>
  )
}

/** System-prompt keyed Chat renderer. */
export const SystemPromptNodeView = memo(function SystemPromptNodeView({
  node, t,
}: Pick<ChatNodeViewProps<'system-prompt'>, 'node' | 't'>) {
  return <SystemPromptRow text={node.data.text} repeat={node.data.repeat === true} t={t} />
})
