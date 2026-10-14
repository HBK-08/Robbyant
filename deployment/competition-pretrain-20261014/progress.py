"""Report verified downloads and CPU conversion progress."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
manifest=json.loads((OUT/'input-manifest.json').read_text())
for kind in ['model','dataset']:
    jobs=[j for j in manifest['files'] if j['kind']==kind]
    have=0; complete=0
    for j in jobs:
        p=Path(j['target'])
        if p.exists():
            have+=min(p.stat().st_size,j['bytes']); complete+=1
        else:
            segments=p.parent/(p.name+'.segments')
            if segments.exists(): have+=min(sum(f.stat().st_size for f in segments.iterdir() if f.is_file()),j['bytes'])
    print(f'{kind}: {complete}/{len(jobs)} files, {have/1e9:.2f}/{sum(j["bytes"] for j in jobs)/1e9:.2f} GB')
try:
    state=json.loads((OUT/'conversion-progress.json').read_text())
    episode_progress={}
    for task in state['active']:
        p=OUT.parents[1]/'datasets/competition-clean50/lerobot'/task/'meta/info.json'
        try: episode_progress[task]=json.loads(p.read_text())['total_episodes']
        except (FileNotFoundError,json.JSONDecodeError): episode_progress[task]=0
    print('converted:',state['completed_count'],'/50; active:',episode_progress,'failures:',state['failures'])
except FileNotFoundError: pass
