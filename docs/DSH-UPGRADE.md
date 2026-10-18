# dsh upgrade: v0.1.2-rc.1 fork → v0.2.1-rc.2 (started 2026-10-18)

Work happens on branch `dsh-upgrade-0.2.1`, worktree `C:\src\shoehorn-upgrade`. The running dsh (`C:\src\shoehorn\agent`) stays untouched until step 6 passes.

## Facts

| Item | Value |
|---|---|
| Our fork point upstream | `76fda72979` (Sept 3, v0.1.2-rc.1 era) |
| Our customizations | 9 fork commits plus 16 shoehorn commits, about 140 files |
| Target | `dsh-v0.2.1-rc.2` (Oct 17), remote `dsh-upstream` |
| Upstream commits since the fork | 7,414 |
| Text conflicts | 48. Resolved in favor of our hunks (`merge-tree -X ours`, base = the fork point); each is reviewed as the build reaches it |

## Known breaking changes upstream

- **Presets.** `packages/preset/agent-presets` and the `minimal-compact` preset are gone; the replacements are `agent-preset`, `agent-preset-registry` and `persona`. These need porting: `~/.dsh` settings, the integrations defaults (`agentPreset: minimal-compact`), and the webhook and Telegram defaults.
- **Webhooks moved.** `packages/webhook/*` moved to `packages/experimental/webhook*`; our integrations import `@deepseek-ai/dsh-webhook`.
- **Heavy churn in patched packages:** agent-loop (187 commits), llm-pi-ai (122), compaction-basic (78), mcp-client (57). Keep our version only where it is better; otherwise take upstream's.

## Steps

1. [x] Create the branch and worktree, merge rc.2 into `agent/` with our side kept on conflicts.
2. [x] Run `pnpm install` (done), then `pnpm run build:lib`; fix the compile errors. **In progress:** the host typecheck is down from 184 to 18 errors; the full list is in `docs/dsh-upgrade-host-errors.txt`.
   - [x] tool-pwsh (63), tool-pwsh-persistent (30), tool-bash-persistent (24). tool-pwsh: upstream's version taken (it now moves a timed-out command to a background job itself). The persistent shells: upstream's files plus our auto-background patch replayed (3-way), ported to the new jobs API (output pushed with `job.append`, owner = session id, the prompt check by `waitReason === 'stdin_read'`); tests ported (async `agent()` helper, the working-directory fixture, random job ids, bash's new "[Command finished ...]" line). All 201 shell tests pass, including the real-PowerShell auto-background test.
   - [x] agent-loop (24): upstream's files plus our two changes replayed: auto-continue (now its own message source `auto-continue`, as upstream's producers declare theirs) and the mid-turn memory-index deferral in the runtime-context projection. 451 upstream tests and our auto-continue test pass.
   - [x] compaction-basic (19) and compaction-tool-result-pruner (6): upstream's files plus our combined patch (fork + shoehorn) replayed 3-way. Conflicts: upstream's `thresholdRatio` / new `headroomTokens` kept; the summarizer input merged (upstream's `RequestMessage` plus our template choice and condensing); the pruner's selection (unread results, open turn, protected tail, target) kept on upstream's message access. `condense.ts` rewritten for the new message model (tool results are tool-role messages). Our hard-coded summarizer defaults (old heretic / orca routes, gone since 2026-10-09) removed: the plugin now has upstream's defaults plus our two options off. 247 compaction tests pass.
   - [x] mcp-client: upstream moved to the new MCP SDK (`@modelcontextprotocol/client` 2.2.0). Our connection / transport / index changes replayed (SSE transport, OAuth provider, server stderr log, allowlist `includeTools`, `exposure: deferred`); `includeTools` / `exposure` made optional so upstream's tests stand; integrations moved to the same SDK. 131 tests pass.
   - [x] repeat-tool-reminder (upstream's new message source plus our same-result and investigation detectors; 34 tests), fs-local (our UTF-8 BOM keeping on upstream's imports), tool-memory (upstream test). fs-local's symlink tests fail on this Windows account without developer mode, with or without our change.
   - [x] apps/cli `image-offload.expected.e2e.ts`: upstream's version for now; recheck in step 4 (its expected text depends on our image-hint wording).
   - [x] integrations: upstream replaced settings namespaces (`settings.yaml`, `SettingsScope`) with volatile plugin config. The five integration settings are now `.volatile()` Config fields: the service reads them live, re-applies on `loader/volatile-update`, and the page's writes go through `settings.update(entryId, ...)` into the profile. Tests use a stand-in Settings service (`tests/settings-stub.ts`) that updates the references and emits the same event. 21 tests pass.
   - [x] **Host typecheck: 0 errors.**
   - [x] Client side: `pnpm run build:lib` passes. Our UI packages' tsdown configs take the package directory, and their tsconfigs reference the client builds (`tsconfig.client.json`); the integrations page opens a chat through `uiWorkspace.openSession` (the sessions service lost `open`); the remotes list keeps `integrationsRemote`. Taken from upstream: the chat's system-prompt node and strings (our "repeat" label on an unchanged prompt after pruning is dropped) and the untouched test-support files.
   - [x] **Streamed file preview: upstream's wins.** Upstream now shows a write tool in a `preparing` phase with its streamed arguments and live content, the same feature as ours, better integrated. Our preview (`FileGenerationPreview`, `file-generation.ts`, the assistant-block argument accumulation) and its tests are removed, including `apps/web/tests/file-generation.e2e.ts`. Check upstream's preview live in step 6.
3. [ ] Port presets: our presets to the new preset system, and the defaults in integrations, webhooks and Telegram. The compaction plugin no longer carries shoehorn defaults, so shoehorn's presets must set: `summarizationProvider: local-27b-shoehorn-summarizer`, `summarizationModel: qwen3.8-27b-uncensored-shoehorn`, `thresholdRatio: 0.7`, `retainRatio: 0.16`, `maxTokens: 8192`, `compactionRetries: 3`, `condenseToolOutput: true`, `summaryTemplate: standard` (the values in today's `minimal-compact` preset and `scripts/windows-harness/configure-local.mjs`).
4. [ ] Run the tests for every package we changed. Review each of the 48 conflict files: keep ours or take upstream's.
5. [ ] Run the full build (`pnpm run build`) and start a test dsh on another port with a copy of `~/.dsh`.
6. [ ] Live checks: a long session with compaction, tool edge cases, vision, the browser MCP service, Telegram (fake API), webhooks, a streamed file write (upstream's live preview), the auto-background shell (a slow command becomes a job).
7. [ ] Switch: merge to main, update `~/.dsh` (backup first), restart dsh. Roll back with `git checkout <old>` plus a rebuild.

## Log

- 2026-10-18: step 1 done. The merge produced 0 remaining conflicts after `-X ours`.
- 2026-10-18: conflicted config files resolved as upstream's version plus our additions (package.json files, tsconfig references). Generated files taken from upstream: pnpm-lock, the docs catalogs, tsconfig.base.json (aliases regenerated, plus hand-written ones for ui-integrations, ui-shoehorn and integrations/types).
- 2026-10-18: integrations and the webhook session now point at `dsh-agent-preset-registry` (same `ctx.agentPresets` service) and `packages/experimental/webhook`. `pnpm install` is OK.
- 2026-10-18: first host typecheck: 184 errors (see step 2). Stopped here for the quota; resume from step 2.
- TODO: `dsh-settings-file` is gone upstream, and the integrations tests (settings-yaml based) still need porting to the new settings provider.
- 2026-10-18: shells done (184 -> 67 errors). Next: agent-loop (24), then compaction (25).
- 2026-10-18: agent-loop done (67 -> 43). Next: compaction-basic (19), the pruner (6), then the small ones.
- 2026-10-18: compaction done (43 -> 18). Next: integrations (5), tool-memory (3), mcp-client (3), repeat-tool-reminder (3), fs-local (1).
- 2026-10-18: mcp-client, reminder, fs-local done (18 -> 5, all in integrations).
- 2026-10-18: integrations ported to volatile config; host typecheck clean (184 -> 0). Next: client typecheck and build:lib.
- Note for step 7: upstream imports an old `~/.dsh/settings.yaml` into the active profile once at start (renamed to `settings.yaml.imported`); our providers and models live there, so check the import result on the test copy first.
- 2026-10-18: step 2 done: host and client typecheck and `build:lib` pass. Next: step 3 (presets).
