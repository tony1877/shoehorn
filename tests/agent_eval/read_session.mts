// Print one dsh session log (session.v4.jsonl.zstd: concatenated zstd frames) as plain JSONL on stdout, using the
// persistence backend's own frame scanner. A torn final frame (the session is still being written) is skipped.
//
//   node --import <agent>/node_modules/tsx/dist/esm/index.mjs tests/agent_eval/read_session.mts FILE
import { readFileSync } from 'node:fs'
import { zstdDecompressSync } from 'node:zlib'
import { scanZstdFrames } from '../../agent/packages/session/session-persistence-jsonl/src/zstd.ts'

const buffer = readFileSync(process.argv[2]!)
const { frames } = scanZstdFrames(buffer)
for (const { start, end } of frames) process.stdout.write(zstdDecompressSync(buffer.subarray(start, end)))
