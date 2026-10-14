// Apply only the reviewed policy settings, preserving unrelated machine configuration.
import fs from 'node:fs';
import path from 'node:path';
import yaml from '../../agent/node_modules/js-yaml/index.js';
const schema=yaml.DEFAULT_SCHEMA.extend([new yaml.Type('tag:yaml.org,2002:js',{kind:'scalar',construct:value=>value})]);
function parseYaml(text){try{return yaml.load(text,{schema});}catch(error){throw new Error('Invalid YAML configuration: '+error.reason);}}

export function configure(settings, preset, patch) {
  const newline=settings.includes('\r\n')?'\r\n':'\n';
  settings=settings.replace(/^    (local-27b-shoehorn|local-flash-shoehorn):\r?\n([\s\S]*?)(?=^    [\w-]+:|$(?![\s\S]))/gm,(whole,name)=>{
    if (/^      sampling:/m.test(whole)) {
      return whole.includes('max_content_chars:') ? whole.replace(/max_content_chars: \d+/,'max_content_chars: 16384')
        : whole.replace(/^(      sampling:)\r?\n/m,'$1'+newline+'        max_content_chars: 16384'+newline);
    }
    return whole.replace(newline,newline+'      sampling:'+newline+'        max_content_chars: 16384'+newline);
  });
  preset=preset.replace(/^(\s*)maxTokens: 48000\s*$/gm,'$1maxTokens: 8192')
    .replace(/summaryTemplate: detailed/g,'summaryTemplate: standard');
  preset=preset.replace(/        # Generation cap[\s\S]*?        maxTokens: \d+/,
    '        # Bounded thinking-off checkpoints preserve facts without generating a long report.\n        maxTokens: 8192')
    .replace(/        # dsh patch dsh-compaction-summary-quality[\s\S]*?        summaryTemplate: \w+/,
    '        # Condense tool traffic and use the standard checkpoint structure.\n        condenseToolOutput: true\n        summaryTemplate: standard');
  patch=patch.replace(/# Loop guard[\s\S]*?(?=^- id: repeat-tool-reminder)/m,
    '# Loop guards use ordinary advisory messages; historical control tags do not change request policy.\n');
  // Structured parsing validates the preserved YAML before any caller writes it.
  const rows=parseYaml(patch);
  const reminder=rows.find(row=>row.id==='repeat-tool-reminder');
  if (!reminder) throw new Error('repeat-tool-reminder patch is required');
  reminder.config.reminderSuffix='';
  reminder.config.investigationThresholds=[12,24,48];
  reminder.config.investigationResetTools=['write','edit','str_replace_editor'];
  // Replace only this row, leaving the rest of the annotated patch intact.
  const start=patch.search(/^- id: repeat-tool-reminder\s*$/m);
  const tail=patch.slice(start).search(/\n- (?:id:|insert:)/);
  const end=tail<0?patch.length:start+tail+1;
  patch=patch.slice(0,start)+yaml.dump([reminder],{lineWidth:120})+patch.slice(end);
  parseYaml(settings);parseYaml(preset);parseYaml(patch);
  const checked=parseYaml(settings)['llm-pi-ai']?.providers;
  for(const name of ['local-27b-shoehorn','local-flash-shoehorn']){
    if(checked?.[name]?.sampling?.max_content_chars!==16384)throw new Error('Missing conversation route: '+name);
  }
  return {settings,preset,patch};
}
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname.replace(/^\/(?=[A-Za-z]:)/,''))) {
  const home=process.env.DSH_HOME||path.join(process.env.USERPROFILE,'.dsh');
  const paths={settings:path.join(home,'settings.yaml'),preset:path.join(home,'.agent-presets','minimal-compact','agent.cordis.yml'),patch:path.join(home,'cordis.patch.yml')};
  const modified=configure(...['settings','preset','patch'].map(key=>fs.readFileSync(paths[key],'utf8')));
  const backup=path.join(home,'backups','agent-harness-'+Date.now());fs.mkdirSync(backup,{recursive:true});
  for(const key of Object.keys(paths))fs.copyFileSync(paths[key],path.join(backup,key+'.yml'));
  for(const key of Object.keys(paths))fs.writeFileSync(paths[key],modified[key]);
  console.log('Applied answer limits, bounded compaction and advisory investigation checkpoints. Backups saved.');
}
