// Authenticated, bounded JSON-line exec bridge. Run on loopback only.
const net = require('net');
const cp = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { StringDecoder } = require('string_decoder');

function tokenize(text) {
  const result = []; let value = '', quote = null, started = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === quote) { quote = null; started = true; }
    else if (quote) { value += c; started = true; }
    else if (c === '"' || c === "'") { quote = c; started = true; }
    else if (/\s/.test(c)) {
      if (started) { result.push(value); value = ''; started = false; }
    } else { value += c; started = true; }
  }
  if (quote) throw new Error('Unterminated EXE quote; use an EXE JSON array for literal quotes');
  if (started) result.push(value);
  return result;
}

function execution(command, pwsh) {
  if (command.startsWith('CMD|')) return ['cmd.exe', ['/d', '/s', '/c', command.slice(4)]];
  if (command.startsWith('EXE|')) {
    const value = command.slice(4);
    const args = value.trimStart().startsWith('[') ? JSON.parse(value) : tokenize(value);
    if (!Array.isArray(args) || !args.length || args.some(x => typeof x !== 'string') || !args[0]) {
      throw new Error('EXE requires a program and string arguments');
    }
    return [args[0], args.slice(1)];
  }
  const code = command.startsWith('B64|') ? Buffer.from(command.slice(4), 'base64').toString('utf8') : command;
  const preamble = "$ProgressPreference='SilentlyContinue'; if ($PSStyle) { $PSStyle.OutputRendering='PlainText' }; " +
    "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $OutputEncoding=[Text.UTF8Encoding]::new($false);\n";
  return [pwsh, ['-NoLogo', '-NoProfile', '-NonInteractive', '-OutputFormat', 'Text',
    '-EncodedCommand', Buffer.from(preamble + code, 'utf16le').toString('base64')]];
}

function createBridge({ token, pwsh = 'pwsh', timeoutMs = 120000, maxOutputBytes = 8 * 1024 * 1024,
                        maxRequestBytes = 1024 * 1024, spawn = cp.spawn } = {}) {
  if (typeof token !== 'string' || token.length < 32) throw new Error('A bridge token of at least 32 characters is required');
  const secret = Buffer.from(token);
  const server = net.createServer(socket => {
    let input = '', bytes = 0, outputBytes = 0, child, finished = false, accepted = false;
    const decoders = { stdout: new StringDecoder('utf8'), stderr: new StringDecoder('utf8') };
    const requestDecoder = new StringDecoder('utf8');
    const timer = setTimeout(() => {
      stop(); end(124, 'Bridge command timed out');
    }, timeoutMs);
    timer.unref();
    function send(frame) {
      if (!socket.destroyed && !socket.writableEnded) return socket.write(JSON.stringify(frame) + '\n');
      return false;
    }
    function stop() {
      if (!child || child.exitCode !== null) return;
      if (process.platform === 'win32' && child.pid) {
        const killer = cp.spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
        killer.on('error', () => child.kill());
      } else child.kill('SIGKILL');
    }
    function end(code, error) {
      if (finished) return;
      finished = true; clearTimeout(timer);
      for (const channel of ['stdout', 'stderr']) {
        const data = decoders[channel].end();
        if (data) send({ kind: 'output', channel, data });
      }
      if (error) send({ kind: 'output', channel: 'stderr', data: error + '\n' });
      send({ kind: 'exit', code });
      socket.end();
    }
    socket.on('close', () => { clearTimeout(timer); if (!finished) { finished = true; stop(); } });
    socket.on('error', () => socket.destroy());
    socket.on('drain', () => { child?.stdout?.resume(); child?.stderr?.resume(); });
    socket.on('data', buffer => {
      if (accepted) { stop(); end(1, 'One command is allowed per connection'); return; }
      bytes += buffer.length;
      if (bytes > maxRequestBytes) { end(1, 'Bridge request exceeds size limit'); return; }
      input += requestDecoder.write(buffer);
      const newline = input.indexOf('\n');
      if (newline < 0) return;
      accepted = true;
      try {
        const request = JSON.parse(input.slice(0, newline));
        const supplied = Buffer.from(typeof request.token === 'string' ? request.token : '');
        if (supplied.length !== secret.length || !crypto.timingSafeEqual(supplied, secret)) {
          end(1, 'Bridge authentication failed'); return;
        }
        if (request.version !== 2 || typeof request.command !== 'string' || typeof request.cwd !== 'string') {
          throw new Error('Expected bridge protocol v2, command and cwd');
        }
        if (!fs.statSync(request.cwd).isDirectory()) throw new Error('Working directory is not a directory');
        const [program, args] = execution(request.command, pwsh);
        child = spawn(program, args, { cwd: request.cwd, windowsHide: true, shell: false });
        for (const channel of ['stdout', 'stderr']) child[channel].on('data', data => {
          if (finished) return;
          outputBytes += data.length;
          if (outputBytes > maxOutputBytes) { stop(); end(1, 'Bridge output exceeds size limit'); return; }
          const text = decoders[channel].write(data);
          if (text && !send({ kind: 'output', channel, data: text })) child[channel].pause();
        });
        child.on('error', error => end(1, error.message));
        child.on('close', code => end(code ?? 1));
      } catch (error) { end(1, error.message); }
    });
  });
  return server;
}

if (require.main === module) {
  try {
    const keyPath = process.env.DSH_BRIDGE_TOKEN_FILE || path.join(process.env.USERPROFILE, '.dsh', 'bridge-token.txt');
    const token = fs.readFileSync(keyPath, 'utf8').trim();
    const server = createBridge({ token, pwsh: fs.existsSync('C:\\Program Files\\PowerShell\\7\\pwsh.exe')
      ? 'C:\\Program Files\\PowerShell\\7\\pwsh.exe' : 'pwsh' });
    server.on('error', error => { console.error(error.message); process.exitCode = 1; });
    server.listen(9999, '127.0.0.1', () => console.log('dsh-bridge v2 listening on 127.0.0.1:9999'));
  } catch (error) { console.error('Bridge startup failed: ' + error.message); process.exitCode = 1; }
}
module.exports = { createBridge, execution, tokenize };
