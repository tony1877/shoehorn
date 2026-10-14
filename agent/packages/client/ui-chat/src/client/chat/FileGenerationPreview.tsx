import { CodeBlock } from '@deepseek-ai/dsh-client-ui-primitives'
import { fileGeneration } from './file-generation.ts'

/**
 * Display decoded file content while the assistant is still generating arguments.
 * @param props.name - tool wire name.
 * @param props.raw - partial argument JSON.
 * @param props.copyLabel - localized copy action.
 * @param props.copiedLabel - localized copy confirmation.
 * @returns a bounded code preview; no filesystem side effects.
 */
export function FileGenerationPreview({ name, raw, copyLabel, copiedLabel }: {
  name: string
  raw: string
  copyLabel: string
  copiedLabel: string
}) {
  const preview = fileGeneration(name, raw)
  if (preview === undefined) return null
  return <div data-file-generation>
    <code>{preview.path}</code>
    <CodeBlock code={preview.text} streaming copyLabel={copyLabel} copiedLabel={copiedLabel} />
  </div>
}
