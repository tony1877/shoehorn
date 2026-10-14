# Agent Note: Streamed file previews and investigation checkpoints

Status: implemented

English | [中文](2026-10-14-streamed-files-and-investigation-checkpoints.zh.md)

## Problem

File arguments arrive in assistant chunks, but Chat tool rows begin at completed tool calls. A file-only assistant therefore hides minutes of available generation. Exact-repeat detectors also miss investigations whose commands and outputs keep changing without a successful mutation.

## Decision

Chat publishes file-tool argument blocks during generation and renders decoded content with the existing localized code primitive. The root JSON string reader holds incomplete escapes and keeps only the latest 16,384 code units for display. Execution remains owned by the complete validated tool call; the preview creates no new Session events and cannot write files.

The repeat-tool-reminder adds optional investigation thresholds and configurable reset tools. Tracked calls increment a per-agent count; successful reset tools and user interjections clear it. Exact-call and same-result reminders take precedence. Investigation messages ask for an authoritative reference or runnable assertion without asserting that productive research is a failure. Defaults keep this detector disabled.

## Alternatives considered

**Writing partial files during generation** loses atomic validation and exposes malformed or abandoned output. A display-only preview keeps the filesystem executor unchanged.

**Classifying analysis-looking answer text in the UI** cannot reliably distinguish analysis from a requested explanation. The inference policy owns thinking separation; Chat preserves the received block types.

**Vetoing long investigations** interrupts legitimate research. Advisory counts give the model a checkpoint without blocking calls or claiming semantic progress detection.

## Consequences

The preview is bounded and disappears at settlement so applied tool rows remain authoritative. Very large files show their tail while generating. Investigation counts measure successful reset tools, not the truth or usefulness of arbitrary shell operations; deployments choose thresholds appropriate to their tools.

## Testing

The Conversation assembler regression records a file-only partial before any tool/call event. The renderer test observes decoded text changing and disappearing on settlement. Decoder cases cover escapes, Unicode, nested fields and bounds. Agent-loop regressions pin advisory text and resets alongside the existing repeat detector suite.
