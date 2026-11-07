# dsh upgrade: v0.2.1-rc.2 → v0.2.2-rc.1 (planned 2026-11-07)

Same method as the 0.2.1 upgrade (`docs/DSH-UPGRADE.md`): work on branch `dsh-upgrade-0.2.2` in worktree
`C:\src\shoehorn-upgrade`. The running dsh (`C:\src\shoehorn\agent`, pointed at the rig) stays untouched until the
live checks pass.

## Facts

| Item | Value |
|---|---|
| Our base upstream | `dsh-v0.2.1-rc.2` (Oct 17) |
| Target | `dsh-v0.2.2-rc.1` (Nov 4), remote `dsh-upstream` (fetched 2026-11-07) |
| Upstream commits in between | ~3,450; 8,278 files changed |
| Our customizations in `agent/` | 192 files (+18.6k lines), committed, plus 13 uncommitted files |
| Text conflicts (dry run, `git merge-tree`, committed side only) | **18**, listed below |
| Uncommitted files upstream also changed | 10 of 13 (they go in on the branch first, so they add to the conflicts) |
| Toolchain | unchanged: pnpm 11.28.5, Node ≥ 24.9 (we have 24.20) |
| Upstream deletions or renames touching our files | none |

Our churned packages are small this time (agent-loop, compaction, fs, shell, llm: 456 files, +11k / −2k). Most of
upstream's work is in liveshare, the UI primitives, forge, CUA (macOS) and schedule, which we don't patch.

### The 18 conflicts

- **Generated or config files:** take upstream's version, then re-add our entries. These are `pnpm-lock.yaml`, `apps/cli/package.json`, `packages/bundle/web-app/package.json`, `packages/api/remotes/package.json`, `packages/shell/tool-pwsh-persistent/package.json` and `packages/client/ui-chat/README.i18n.yaml`.
- **Engine-side:** replay our hunks onto upstream's code.
  - `core/agent-loop/src/agent.ts`: auto-continue.
  - `compaction/compaction-basic/src/index.ts`: condensing, templates, call-markup stripping.
  - `core/tools/src/index.ts`
  - `fs/fs-local/src/index.ts`: keeping the UTF-8 BOM.
  - `fs/tool-fs/src/index.ts`
  - `skill/tool-skill/src/index.ts`
  - `experimental/webhook/src/session.ts`: the preset registry.
  - `api/remotes/src/client/index.ts`: integrationsRemote.
- **UI:**
  - `ui-theme/src/styles/design-platform.css` and `tests/elevation-styles.client.spec.ts`: our soft dark and Solarized themes.
  - `ui-tool/.../GenericToolCard.tsx` and `file-mutation-row.tsx`: our streamed tool arguments. Upstream now names and opens the files an edit row touches; prefer theirs and keep our live argument streaming only where theirs lacks it.

### Upstream changes that need action on our side

- **`llm-request-purpose-declarations`:** `compaction` and `session-title` are no longer in the shared purpose union. Our compaction summarizer and condensing calls must import the producer's declarations.
- **`fs-observation-policy-default-off`, plus `apply_patch` now on in dsh-base and the Web `standard` preset:**
  - `write` and `edit` no longer require reading the file first. Decide whether to keep that check on for our presets: it guards against the local model overwriting files blind.
  - Decide whether to give our 27B `apply_patch` or keep it on `edit` only. The patch format is a known failure mode for small models; the fallback is to set `applyPatch: false` in `minimal-compact` and `standard-compact`.
- **`cordis-loader-entry-trees`:** Loader rc.8 and Include 1.1.1 change how patch files are persisted. Check that `~/.dsh/profiles/web/cordis.patch.yml`, which holds our preset rows, loads and that UI edits still save to it.
- **`model-listing-refresh`:** custom model-listing owners need `settings` and `changed` callbacks. Check the shoehorn provider (the remote at 192.168.68.54:8090) still lists its model.
- **`attachment-read-access`:** image reads now carry the full reference. Check vision (`read_image`, pasted images) end to end.

## Steps

1. [ ] **Snapshot the uncommitted work.** Commit the 13 uncommitted `agent/` files, the rest of the working tree
   (engine sensors, YaRN, spec auto, the rig changes) and the `experiments/` deletions onto a branch, so the merge has
   a clean base. *Needs the owner's OK: nothing has been committed so far.*
2. [ ] Create branch `dsh-upgrade-0.2.2` and worktree `C:\src\shoehorn-upgrade`. Merge `dsh-v0.2.2-rc.1` into
   `agent/` (base = `dsh-v0.2.1-rc.2`, `-X ours` as before, then review all 18 files plus the 10 uncommitted ones).
3. [ ] Run `pnpm install` and `pnpm run build:lib`; fix the host and client typecheck errors, package by package,
   logging the counts here.
4. [ ] Handle the five upstream changes listed above; settle the `apply_patch` and read-before-write choices in our
   presets.
5. [ ] Run the tests of every package we patch: agent-loop, compaction-basic, the pruner, fs-local, tool-fs, the
   shells, mcp-client, repeat-tool-reminder, integrations and the UI style rules.
6. [ ] Run the full build (`pnpm run build`, 9 gates). Start a test dsh on port 3090 with a copy of `~/.dsh`
   (`C:\Users\aless\dsh-test-home`, refreshed), pointed at the rig.
7. [ ] Live checks on the test dsh against the rig. These are the five not run last time, plus the new ones:
   - [ ] Write, edit and pwsh tools; a 75 s command; auto-background with a command longer than its wait.
   - [ ] Vision: `read_image`, and an image pasted into the chat.
   - [ ] A long session through compaction. This uses the summarizer route and the 512k context on the rig.
   - [ ] The integrations page and the browser MCP server.
   - [ ] Telegram against the fake Bot API; one webhook delivery.
   - [ ] The streamed file preview on a large write; our themes (light, soft dark, Solarized).
   - [ ] Preset edits from the UI persist after a restart (the loader change).
8. [ ] **Switch:**
   - Back up `~/.dsh` to `~/.dsh/backups/pre-dsh-0.2.2-*`.
   - Merge the branch to main, rebuild the agent and restart dsh. `use-remote-shoehorn.ps1` stays pointed at the rig.
   - Rollback: restore the backup, check out the pre-merge commit, rebuild.

## Risks

- **apply_patch** with the local model: test it before it reaches the presets.
- **The loader persistence change** could silently stop UI preset edits saving; step 7 covers it.
- **Size:** about a day of agent time, mostly steps 3 and 7. The previous upgrade (7.4k commits, 48 conflicts) took
  one long session.

## Log

- 2026-11-07: plan written. `dsh-upstream` fetched; dry-run merge shows 18 conflicts.
