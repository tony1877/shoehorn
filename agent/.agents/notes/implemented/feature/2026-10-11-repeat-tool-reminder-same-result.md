# Agent Note: repeat-tool-reminder — same-result detection and a reminder suffix

Status: implemented

English | [中文](2026-10-11-repeat-tool-reminder-same-result.zh.md)

## Problem

A local Qwen3.8 27B session on a three-task coding test looped for about eight minutes. Task 1 asked for 100000 from 10 threads × 1000 increments, which is 10000. The model's first run printed 10000, so the program was already right. In its only real reasoning pass the model computed 10 × 1000 = 100,000, then decided that the shell was rewriting its output. It wrote `rtest1.py` … `rtestZ.py` and `rtest111.py` … `rtest999.py` with identical bodies and ran each one; every `pwsh` call returned `10000`, and it reasoned on 4 of 120 steps. `repeat-tool-reminder` never fired: no two consecutive calls had identical arguments, because each `write` named a new file and each run named that file.

## Decision

The guard gains an opt-in second detector keyed on results. With `resultThresholds` non-empty, every tracked call to a tool matching `resultInclude` adds `(tool name, result content blocks)` to the agent's history of its last `resultWindow` such calls. When the number of equal entries reaches a threshold not yet reminded for that result, the guard delivers a reminder: the first threshold gets a short text telling the model to stop running variations and to re-check its assumptions, including any expectation stated in the task, and later thresholds get a detailed text quoting the result (capped at `resultPreviewChars`). The key is the whole blocks, so two different images differ. When the exact-repeat chain fires on the same call, its reminder wins, since it names the repeated arguments. A user prompt clears the history with the chain.

`reminderSuffix` appends deployment text to every reminder. The local deployment sets the shoehorn chat template's `<|think_xhigh|>` tag there: the template strips the tag from the rendered user message and switches the request to reasoning effort xhigh, whose system instruction asks the model to validate key assumptions.

The base bundle enables same-result detection for `pwsh` and `bash` at `[3, 5, 8]` over 12 calls. It is off by default in the plugin: `edit` returns the same success text on every legitimate edit of one file, so comparing every tool's results would nag during normal work.

## Alternatives considered

**Fuzzy argument matching** (masking digits or file names before comparing). It depends on the shape of each tool's arguments, and the loop above would still evade it whenever the model changed the script body. The result is the signal that stays equal however the calls vary.

**Forcing a minimum of reasoning tokens on the server.** It costs latency on every step, including trivial ones, and does not tell the model what is wrong. It remains a separate option if reminders alone do not restore thinking.

**Counting equal results only among consecutive calls.** The loop above alternated `write` and `pwsh`, so a consecutive rule would need the same tool-transparency lists as the chain; a window over the tool's own recent calls needs none.

## Consequences

An equal result is not always a loop: a test run that keeps passing, or a command that keeps printing nothing, draws reminders; `resultInclude` and `resultThresholds` limit that. Output that embeds a timestamp or a temporary path evades the detector. A suffix that a chat template interprets can change text rendered before the reminder and so invalidates the cached prompt prefix once per conversation.

## Verification

`packages/guard/repeat-tool-reminder/tests/repeat-tool-reminder.spec.ts` covers: off by default, the gentle and detailed reminders on calls that all differ, the window, one reminder per threshold as the window slides, `resultInclude`, exact-repeat precedence, the user-prompt reset, the suffix, the result cap, whole-block comparison of images, and fail-loud validation of the new fields. `src/index.ts` keeps 100% line, branch, function and statement coverage.
