/** Decoded file text from a tool argument stream; never executes or writes it. */
export interface FileGeneration {
  readonly path: string
  readonly text: string
}

/**
 * Read complete JSON string units, holding an unfinished escape until it arrives.
 * @param raw - streamed argument JSON.
 * @param start - opening quote offset.
 * @returns decoded prefix, next offset and whether the closing quote arrived.
 */
function stringPrefix(raw: string, start: number): { value: string; end: number; complete: boolean } {
  let end = start + 1
  while (end < raw.length) {
    if (raw[end] === '"') return { value: JSON.parse(raw.slice(start, end + 1)) as string, end: end + 1, complete: true }
    if (raw[end] === '\\') {
      const width = raw[end + 1] === 'u' ? 6 : 2
      if (end + width > raw.length) break
      end += width
    } else end++
  }
  return { value: JSON.parse(`${raw.slice(start, end)}"`) as string, end, complete: false }
}

/**
 * Decode root string fields of write/edit arguments before the JSON closes.
 * @param name - streamed tool name.
 * @param raw - complete or partial argument JSON.
 * @returns a bounded display-only file preview, or absence for other tools.
 */
export function fileGeneration(name: string, raw: string): FileGeneration | undefined {
  if (name !== 'write' && name !== 'edit' && name !== 'str_replace_editor') return undefined
  const fields: Record<string, string> = {}
  if (!raw.trimStart().startsWith('{')) return undefined
  let p = raw.indexOf('{') + 1
  if (p === 0) return undefined
  try {
    while (p < raw.length) {
      while (/\s|,/.test(raw[p] ?? '') && p < raw.length) p++
      if (raw[p] !== '"') break
      const key = stringPrefix(raw, p)
      if (!key.complete) break
      p = key.end
      while (/\s/.test(raw[p] ?? '') && p < raw.length) p++
      if (raw[p++] !== ':') break
      while (/\s/.test(raw[p] ?? '') && p < raw.length) p++
      if (raw[p] !== '"') {
        let depth = 0
        let comma = -1
        for (; p < raw.length; p++) {
          const char = raw[p]
          if (char === '"') {
            const nested = stringPrefix(raw, p)
            if (!nested.complete) break
            p = nested.end - 1
          } else if (char === '{' || char === '[') depth++
          else if (char === '}' || char === ']') {
            if (depth === 0) break
            depth--
          } else if (char === ',' && depth === 0) { comma = p; break }
        }
        if (comma < 0) break
        p = comma + 1
        continue
      }
      const value = stringPrefix(raw, p)
      fields[key.value] = value.value
      p = value.end
      if (!value.complete) break
    }
  } catch {
    // Invalid argument JSON remains the executor's error; it is not file content.
    return undefined
  }
  const text = name === 'write' ? fields.content : name === 'edit' ? fields.new_string : fields.file_text ?? fields.new_str
  if (text === undefined) return undefined
  return { path: fields.file_path ?? fields.path ?? '', text: text.slice(-16384) }
}
