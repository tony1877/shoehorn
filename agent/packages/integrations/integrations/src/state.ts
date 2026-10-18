/**
 * Small JSON state files under `$DSH_HOME/integrations/`: Telegram chat
 * bindings and update offset, webhook chat bindings, the delivery log. Writes
 * go through a temp file and a rename, one at a time per file, so a crash
 * leaves the previous version intact.
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { dshHomePath } from '@deepseek-ai/dsh-home-paths'

/** One state file with an in-memory copy. */
export class StateFile<T extends object> {
  private value: T
  private readonly path: string
  private writing: Promise<void> = Promise.resolve()

  /**
   * @param name - file name inside the integrations state folder.
   * @param initial - value used when the file is missing or unreadable.
   * @param directory - folder override (tests); defaults to `$DSH_HOME/integrations`.
   */
  constructor(name: string, initial: T, directory?: string) {
    this.path = join(directory ?? dshHomePath('integrations'), name)
    try {
      this.value = { ...initial, ...JSON.parse(readFileSync(this.path, 'utf8')) as T }
    } catch {
      this.value = initial
    }
  }

  get(): T {
    return this.value
  }

  /** Replace the value and persist it; returns once this write is on disk. */
  set(next: T): Promise<void> {
    this.value = next
    const text = JSON.stringify(next, null, 2)
    this.writing = this.writing.then(() => {
      try {
        mkdirSync(dirname(this.path), { recursive: true })
        writeFileSync(`${this.path}.tmp`, text)
        renameSync(`${this.path}.tmp`, this.path)
      } catch {
        // State is a convenience (bindings, offsets, logs); the in-memory copy stays authoritative.
      }
    })
    return this.writing
  }

  update(mutate: (current: T) => T): Promise<void> {
    return this.set(mutate(this.value))
  }
}
