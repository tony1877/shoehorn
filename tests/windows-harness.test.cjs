const { test } = require('node:test');
const assert = require('node:assert/strict');
const net = require('node:net');
const { spawn } = require('node:child_process');
const { createBridge, tokenize } = require('../scripts/windows-harness/dsh-bridge.js');
const token = 'test-token-'.repeat(8);

async function fixture(options, fn) {
  const server = createBridge({ token, ...options });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try { await fn(server.address().port); }
  finally { await new Promise(resolve => server.close(resolve)); }
}
function request(port, command, options = {}) {
  const socket = net.connect(port, '127.0.0.1');
  const frames = []; let buffer = '';
  const finished = new Promise((resolve, reject) => {
    socket.on('error', reject);
    socket.on('data', data => {
      buffer += data.toString();
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const frame = JSON.parse(buffer.slice(0,newline)); buffer = buffer.slice(newline+1);
        frames.push(frame);
        options.onFrame?.(frame, socket);
      }
    });
    socket.on('end', () => resolve(frames));
    socket.on('close', () => resolve(frames));
  });
  socket.on('connect', () => socket.write(JSON.stringify({ version:2, token, command, cwd:process.cwd(), ...options.fields })+'\n'));
  return { socket, frames, finished };
}
const command = code => 'EXE|'+JSON.stringify([process.execPath, '-e', code]);

test('bridge streams before exit and marker-like output stays stdout', async () => {
  await fixture({}, async port => {
    const started = Date.now(); let first, ended;
    const result = request(port,command("console.log('@@ERR@@');setTimeout(()=>console.error('late'),400)"), {
      onFrame(frame) { if (frame.channel === 'stdout') first=Date.now()-started; if(frame.kind==='exit')ended=Date.now()-started; }
    });
    const frames = await result.finished;
    assert(first !== undefined && ended - first >= 200);
    assert(frames.some(x=>x.channel==='stdout' && x.data.includes('@@ERR@@')));
    assert(frames.some(x=>x.channel==='stderr' && x.data.includes('late')));
    assert.equal(frames.at(-1).code,0);
  });
});
test('bridge rejects unauthenticated execution', async () => {
  await fixture({}, async port => {
    const frames = await request(port,command("console.log('MUST NOT RUN')"),{fields:{token:'wrong'}}).finished;
    assert.equal(frames.at(-1).code,1);
    assert(!frames.some(x=>x.data?.includes('MUST NOT RUN')));
  });
});
test('output and duration are bounded', async () => {
  await fixture({maxOutputBytes:100},async port=>{
    const frames=await request(port,command("console.log('x'.repeat(1000))")).finished;
    assert.equal(frames.at(-1).code,1);
  });
  await fixture({timeoutMs:150},async port=>{
    const frames=await request(port,command('setInterval(()=>{},1000)')).finished;
    assert.equal(frames.at(-1).code,124);
  });
});
test('disconnect terminates the spawned command', async () => {
  let child;
  await fixture({spawn(program,args,options){child=spawn(program,args,options);return child;}},async port=>{
    const result=request(port,command("console.log('ready');setInterval(()=>{},1000)"),{onFrame(frame,socket){if(frame.channel==='stdout')socket.destroy();}});
    await result.finished;
    if (child.exitCode === null) await new Promise(resolve=>child.once('close',resolve));
    assert.notEqual(child.exitCode,null);
  });
});
test('client cwd, nonzero exit and split UTF-8 are preserved',async()=>{
  await fixture({},async port=>{
    const frames=await request(port,command("console.log(process.cwd());const b=Buffer.from('é');process.stdout.write(b.subarray(0,1));setTimeout(()=>{process.stdout.write(b.subarray(1));process.exit(7)},30)")).finished;
    const output=frames.filter(x=>x.channel==='stdout').map(x=>x.data).join('');
    assert(output.includes(process.cwd()));assert(output.endsWith('é'));
    assert.equal(frames.at(-1).code,7);
  });
  assert.deepEqual(tokenize('program "" tail'),['program','','tail']);
});
test('search reports that results were discarded',async()=>{
  const { apply }=await import('../scripts/windows-harness/web-search-searxng.mjs');
  let provider;apply({web:{registerSearchProvider(value){provider=value;}}});
  const original=global.fetch;
  global.fetch=async()=>({ok:true,json:async()=>({results:[{url:'https://a'},{url:'https://b'}]})});
  try { const result=await provider.search({query:'fixture',maxResults:1});assert.equal(result.sources.length,1);assert.equal(result.truncated,true); }
  finally {global.fetch=original;}
});
test('PowerShell client forwards live streams, cwd and exit status', {skip:process.platform!=='win32'}, async()=>{
  const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
  const directory=await fs.mkdtemp(path.join(os.tmpdir(),'bridge-client-test-'));
  const tokenFile=path.join(directory,'token.txt');
  try {
    await fs.writeFile(tokenFile,token);
    await fixture({},async port=>{
      const child=spawn('pwsh',['-NoProfile','-File',path.resolve(__dirname,'../scripts/windows-harness/brg.ps1'),
        '-Command',"[Console]::Out.WriteLine((Get-Location).Path); [Console]::Out.Flush(); Start-Sleep -Milliseconds 600; [Console]::Error.WriteLine('late-error'); exit 7",
        '-Port',String(port),'-TokenFile',tokenFile,'-WorkingDirectory',directory]);
      let output='',error='',receivedBeforeExit=false,closed=false;
      child.stdout.on('data',data=>{output+=data;receivedBeforeExit=!closed;});
      child.stderr.on('data',data=>{error+=data;});
      const code=await new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',code=>{closed=true;resolve(code);});});
      assert.equal(code,7);assert.equal(receivedBeforeExit,true);
      assert(output.includes(directory));assert(error.includes('late-error'));
    });
  } finally {await fs.unlink(tokenFile);await fs.rmdir(directory);}
});
test('local policy configuration is valid and idempotent',async()=>{
  const {configure}=await import('../scripts/windows-harness/configure-local.mjs');
  const input=[
    'llm-pi-ai:\n  providers:\n    local-27b-shoehorn:\n      sampling:\n        temperature: 0.6\n    local-flash-shoehorn:\n      model: flash\n',
    '- id: compaction\n  config:\n    maxTokens: 48000\n    summaryTemplate: detailed\n',
    '- id: repeat-tool-reminder\n  config:\n    reminderSuffix: "<|think_xhigh|>"\n',
  ];
  const out=configure(...input);
  assert(out.settings.includes('max_content_chars: 16384'));
  assert(out.preset.includes('maxTokens: 8192'));
  assert(out.preset.includes('summaryTemplate: standard'));
  assert(!out.patch.includes('<|think_xhigh|>'));
  assert.deepEqual(configure(out.settings,out.preset,out.patch),out);
});
