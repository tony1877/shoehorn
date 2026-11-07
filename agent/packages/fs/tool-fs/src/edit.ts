/**
 * Model-facing literal edit, unique-match by default. It obtains an optional guard from the
 * single intent slot, calls `ctx.fs.editText` without a separate stat, then records the observed
 * version; no policy means an unconditional atomic edit.
 * @module @deepseek-ai/dsh-tool-fs/src/edit
 */

import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { DiffCallView, DiffResultView, ToolResult } from '@deepseek-ai/dsh-tools'
import { FsError } from '@deepseek-ai/dsh-fs'
import { computeHunkDiffs, diffsFromMeta } from './diff.ts'
import { remediateFsError } from './error.ts'
import { mutationResult } from './mutation-result.ts'
import { sessionResolveOptions } from './session-cwd.ts'
import type { FsSandboxController } from './sandbox.ts'

/** Validated `edit` arguments after defaulting. */
interface EditInput {
  filePath: string
  oldString: string
  newString: string
  replaceAll: boolean
}

/**
 * The `edit` tool's validated arguments: the base parameters plus the two
 * escalation fields, advertised only under a confining `ctx.fs` (absent from
 * the schema otherwise, so the validator rejects them before `execute`).
 */
interface EditToolArgs {
  file_path: string
  old_string: string
  new_string: string
  replace_all?: boolean
  sandbox_permissions?: string
  justification?: string
}

/**
 * Validate value constraints the schema DSL can't express: a non-blank
 * `file_path`, a non-empty `old_string`, and `old_string !== new_string`
 * (an equal pair would be a guaranteed no-op edit).
 * @param args - the schema-validated raw tool arguments.
 * @returns the camelCased input with `replace_all` defaulted to false.
 */
export function parseEditArgs(args: { file_path: string; old_string: string; new_string: string; replace_all?: boolean }): EditInput {
  if (args.file_path.trim().length === 0) throw new Error('file_path must be a non-empty string')
  if (args.old_string.length === 0) throw new Error('old_string must be a non-empty string')
  if (args.old_string === args.new_string) throw new Error('old_string and new_string must differ')
  return {
    filePath: args.file_path,
    oldString: args.old_string,
    newString: args.new_string,
    replaceAll: args.replace_all ?? false,
  }
}

/**
 * Format an edit success (single-match or replace-all) as a Claude-style model-facing message.
 * @param displayPath - the backend-resolved path shown to the model.
 * @param replaceAll - selects the all-occurrences wording over the single-replacement one.
 * @returns the confirmation sentence the model sees as the tool result.
 */
export function formatEditOutput(displayPath: string, replaceAll: boolean, before?: string, after?: string): string {
  const head = replaceAll
    ? `The file ${displayPath} has been updated. All occurrences were successfully replaced.`
    : `The file ${displayPath} has been updated successfully.`
  const snippet = before === undefined || after === undefined || !harnessBoost() ? '' : editedSnippet(before, after)
  return snippet === '' ? head : `${head} The edited region now reads:\n${snippet}`
}

/** `DSH_HARNESS_BOOST=0` restores the plain edit messages (the control arm of tests/agent_eval A/B runs). */
function harnessBoost(): boolean {
  return process.env.DSH_HARNESS_BOOST !== '0'
}

/**
 * Turn a stale or unread-file refusal into one the model can act on at once. Models answer "file changed since it was
 * read — re-read the file" by resending the same edit, over and over (typically after changing the file through the
 * shell). An edit is a literal match, so it cannot clobber text the model has not seen: record the file's current
 * version as observed and say whether `old_string` matches the current content, so the retry succeeds or the model
 * knows to copy fresh text. Any other error, or a failed re-read, is returned unchanged.
 * @returns the error to throw.
 */
async function refreshObservation(
  ctx: Context,
  target: Parameters<Context['fs']['stat']>[0],
  input: EditInput,
  error: unknown,
  exec: Parameters<Context['emit']>[3],
): Promise<unknown> {
  if (!harnessBoost() || !(error instanceof FsError) || (error.code !== 'FS_STALE_VERSION' && error.code !== 'FS_NOT_OBSERVED')) return error
  try {
    const info = await ctx.fs.stat(target)
    if (info === undefined || info.type !== 'file') return error
    const content = (await ctx.fs.readText(target)).replace(/\r\n/g, '\n')
    ctx.emit('fs/observed', target, { kind: 'present', version: info.version }, exec)
    const needle = input.oldString.replace(/\r\n/g, '\n')
    const lines: number[] = []
    for (let at = content.indexOf(needle); at !== -1; at = content.indexOf(needle, at + needle.length)) {
      lines.push(content.slice(0, at).split('\n').length)
    }
    const status = lines.length === 1
      ? `old_string matches the current content once, at line ${lines[0]}: send the same edit again.`
      : lines.length > 1
        ? `old_string matches the current content ${lines.length} times (lines ${lines.join(', ')}): add context to make it unique, or set replace_all.`
        : 'old_string does not match the current content: read the region you want to change and copy old_string from it.'
    return new FsError(`${error.message}\nThe file was re-read for you and is now observed at its current version. ${status}`, error.code, { cause: error })
  } catch {
    return error
  }
}

/** Lines of context shown around an edit, and the most lines a success snippet shows. */
const SNIPPET_CONTEXT = 3
const SNIPPET_MAX_LINES = 30

/**
 * The changed region of `after` with a few lines of context and 1-based line numbers (as `read` shows them), so the
 * model can check indentation and neighbours without reading the file again. Empty when nothing differs.
 * @param before - the content before the edit.
 * @param after - the content after the edit.
 * @returns the numbered snippet, cut at {@link SNIPPET_MAX_LINES}.
 */
export function editedSnippet(before: string, after: string): string {
  if (before === after) return ''
  const a = before.replace(/\r\n/g, '\n').split('\n')
  const b = after.replace(/\r\n/g, '\n').split('\n')
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1
  let endA = a.length - 1
  let endB = b.length - 1
  while (endA >= start && endB >= start && a[endA] === b[endB]) {
    endA -= 1
    endB -= 1
  }
  const from = Math.max(0, start - SNIPPET_CONTEXT)
  const to = Math.min(b.length - 1, Math.max(endB, start) + SNIPPET_CONTEXT)
  const last = Math.min(to, from + SNIPPET_MAX_LINES - 1)
  const lines = b.slice(from, last + 1).map((line, i) => `${from + i + 1}: ${line}`)
  if (last < to) lines.push(`[… ${to - last} more lines; replace_all or a large edit: read the file to see the rest]`)
  return lines.join('\n')
}

/**
 * Register the `edit` tool and its scope-aware system-prompt guidance.
 * @param ctx - the plugin context; registrations are effects scoped to it, and execution uses its `fs` service.
 * @param sandbox - the shared sandbox-escalation API (advertisement, mode stamping, denial mapping).
 */
export function applyEditTool(ctx: Context, sandbox: FsSandboxController): void {
  ctx.systemPrompt.section({
    name: 'tool:edit',
    order: ctx.systemPrompt.getSectionOrder('TOOL_EDIT'),
    text: ({ scope }) => ctx.tools.get('edit', scope) === undefined
      ? ''
      : 'Read a file before editing it (the default fs-observation-policy requires it), unless you just created or edited it in this session.',
  })

  ctx.tools.register(defineTool({
    name: 'edit',
    description: 'Edit an existing UTF-8 text file by replacing literal text.',
    parameters: {
      file_path: {
        type: 'string', required: true,
        description: 'Path to edit, resolved by the filesystem backend. '
          + 'Provide `file_path` before `old_string` and `new_string` in the arguments.',
      },
      old_string: { type: 'string', required: true, description: 'Literal text to replace.' },
      new_string: { type: 'string', required: true, description: 'Literal replacement text. Use an empty string to delete the match.' },
      replace_all: { type: 'boolean', description: 'Replace all matches. Defaults to false; when false, old_string must appear exactly once.' },
      ...sandbox.escalationModes.length > 0 ? sandbox.schemaFields() : {},
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          path: { type: 'string', required: true, description: 'Canonical absolute path in the filesystem execution world.' },
          before: { type: 'string', required: true },
          after: { type: 'string', required: true },
        },
      },
      render: (args, value) => [{
        type: 'text',
        text: formatEditOutput(value.path, args.replace_all ?? false, value.before, value.after),
      }],
      presentationMeta: (_args, value) => ({
        path: value.path,
        diffs: computeHunkDiffs(value.path, value.before, value.after)
          .map(({ path, oldText, newText }) => ({ path, oldText, newText })),
      }),
    },
    async execute(args: EditToolArgs, exec) {
      const input = parseEditArgs(args)
      // Resolve the per-call sandbox policy (approved mode > session override
      // > backend default, plus the session cwd root) BEFORE anything executes.
      const sandboxPolicy = await sandbox.resolvePolicy('edit', args, exec)
      const target = await ctx.fs.resolve(input.filePath, await sessionResolveOptions(ctx, exec))
      // Single-slot decision: the policy plugin returns { version: vObserved } or
      // throws FS_NOT_OBSERVED; the bare default is undefined (unconditional edit).
      // No stat — the bare default never manufactures a version basis. The intent
      // slot itself can throw FS_NOT_OBSERVED for an unread target, so it sits
      // inside the try: both that refusal and the provider's guarded-mutation
      // failure get the model-facing remedy below.
      let outcome
      try {
        const intent = await ctx.waterfall('fs/edit-intent', target, exec, () => undefined)
        outcome = await ctx.fs.editText(
          target,
          { oldString: input.oldString, newString: input.newString, replaceAll: input.replaceAll },
          intent,
          exec.signal,
          sandboxPolicy,
        )
      } catch (error: unknown) {
        // A sandbox denial becomes the shared [sandbox: …] marker (the model
        // recognizes it from bash); guarded mutation failures receive their
        // stable model-facing diagnostic; anything else passes through.
        const remediated = remediateFsError(sandbox.mapError(error, sandboxPolicy), target.displayPath)
        throw await refreshObservation(ctx, target, input, remediated, exec)
      }
      return mutationResult(ctx, target, outcome, exec)
    },
    // Pure display: a diff card of the literal replacement (old_string → new_string), derived
    // from the call args. `oldText: old_string || null` matches claude-agent-acp's Edit arm;
    // new_string is a required arg here, so it maps straight to newText.
    presentCall(args): DiffCallView {
      return {
        card: 'diff',
        title: `Edit ${args.file_path}`,
        diffs: [{ path: args.file_path, oldText: args.old_string || null, newText: args.new_string }],
        locations: [{ path: args.file_path }],
      }
    },
    // Applied metadata replaces the call-time snippet; errors or malformed replay metadata use
    // the generic result rendering.
    presentResult(args, result: ToolResult): DiffResultView | undefined {
      if (result.isError) return undefined
      const diffs = diffsFromMeta(result.meta)
      if (diffs === undefined) return undefined
      return { card: 'diff', title: `Edit ${args.file_path}`, diffs }
    },
  }))
}
