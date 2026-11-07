// Agent eval driver: paste into the devtools console of the dsh page (http://127.0.0.1:3080) while
// `python tests/agent_eval/run_eval.py run ...` is running. It asks the runner for one task at a time, creates a
// session in the task's folder with the task's preset, sends the prompt, and reports the session id back. The page
// is already signed in, so the sessions are ordinary dsh sessions. Stop it with `window.__evalStop = true`.
(async () => {
  const RUNNER = 'http://127.0.0.1:3099'
  const call = async (method, request) => {
    const r = await fetch('/api/' + method, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ type: 'client-request', rpcId: crypto.randomUUID(), method, payload: { args: { request } } }),
    })
    const body = await r.json()
    if (!body.result?.ok) throw new Error(`${method}: ${JSON.stringify(body.result?.error ?? body)}`)
    return body.result.value
  }
  const post = (path, obj) => fetch(RUNNER + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(obj) })
  window.__evalStop = false
  console.log('[agent-eval] driver started')
  while (!window.__evalStop) {
    let next
    try {
      next = await (await fetch(RUNNER + '/next')).json()
    } catch (error) {
      console.warn('[agent-eval] runner not reachable, retrying', error)
      await new Promise(r => setTimeout(r, 15000))
      continue
    }
    // a finished runner may be followed by the next A/B arm (ab.py): keep polling
    if (next.finished) { console.log('[agent-eval] run finished, waiting for the next one'); await new Promise(r => setTimeout(r, 30000)); continue }
    if (next.cancel) {
      try { await call('session/cancel', { sessionId: next.cancel }) } catch (error) { console.warn('[agent-eval] cancel failed', error) }
    } else if (next.task) {
      const { id, cwd, prompt, preset } = next.task
      try {
        const { sessionId } = await call('session/create', { cwd, agentPreset: preset })
        await call('session/prompt', {
          requestId: crypto.randomUUID(), sessionId, mode: 'queue', content: [{ type: 'text', text: prompt }],
        })
        console.log(`[agent-eval] ${id} -> ${sessionId}`)
        await post('/started', { id, sessionId })
      } catch (error) {
        console.error(`[agent-eval] ${id} failed to start`, error)
        await post('/started', { id, error: String(error) })
      }
    }
    await new Promise(r => setTimeout(r, 10000))
  }
})()
