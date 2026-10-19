import css from './live-preview.module.css'

/** Lines kept in view while an argument streams; the box is sized for them, so the newest line is always visible. */
const TAIL_LINES = 12
/** Characters kept from the end, so a long single-line argument stays cheap to render. */
const TAIL_CHARS = 4000

/**
 * shoehorn: the text of a tool argument while the model is still writing it (a shell command, a file's content, an
 * edit's replacement). Shows the last lines; renders nothing for an empty or absent value.
 * @param props - text: the decoded argument so far; label: accessible name of the preview.
 */
export function LivePreview({ text, label }: { readonly text: string | undefined; readonly label: string }) {
  const tail = (text ?? '').slice(-TAIL_CHARS).split('\n').slice(-TAIL_LINES).join('\n')
  if (tail.trim() === '') return null
  return (
    <pre className={css.preview} aria-label={label} data-live-preview>
      {tail}
    </pre>
  )
}
