// Forwards the LAN address's port 3080 to dsh on 127.0.0.1:3080 (dsh itself only listens on loopback).
// Plain TCP pass-through, so HTTP, streaming and WebSockets work unchanged; dsh's own token still guards every session.
// Usage: node dsh-lan-proxy.js <lan-ip>
const net = require('net')
const lan = process.argv[2]
if (!lan) { console.error('usage: node dsh-lan-proxy.js <lan-ip>'); process.exit(1) }
const server = net.createServer(client => {
  const upstream = net.connect(3080, '127.0.0.1')
  client.pipe(upstream).pipe(client)
  const close = () => { client.destroy(); upstream.destroy() }
  client.on('error', close)
  upstream.on('error', close)
})
server.on('error', e => { console.error(`dsh-lan-proxy: ${e.message}`); process.exit(1) })
server.listen(3080, lan, () => console.log(`dsh-lan-proxy: ${lan}:3080 -> 127.0.0.1:3080`))
