import json
from pathlib import Path
import time
import re

out = Path(__file__).resolve().parent
root = out.parent.parent
model = root / 'models/lingbot-vla-v2-6b-robotwin'
parts = list(model.rglob('*.segments/*'))
done = list(model.rglob('*.safetensors'))
total = 25503889772
downloaded = sum(path.stat().st_size for path in parts + done)
state = {'timestamp': time.time(), 'downloaded_bytes': downloaded,
         'downloaded_GB': round(downloaded / 1e9, 3),
         'weight_total_GB': round(total / 1e9, 3),
         'percent': round(downloaded / total * 100, 1), 'verified_shards': len(done),
         'scene_passed': (out / 'scene-verification.json').exists(),
         'expert_passed': (out / 'expert-verification.json').exists(),
         'policy_passed': (out / 'policy-verification.json').exists()}
previous_file = out / 'download-progress.json'
if previous_file.exists():
    previous = json.loads(previous_file.read_text())
    elapsed = state['timestamp'] - previous['timestamp']
    if elapsed > 10:
        speed = (downloaded - previous['downloaded_bytes']) / elapsed
        state['recent_MB_per_second'] = round(speed / 1e6, 2)
        if speed > 0:
            state['estimated_remaining_download_minutes'] = round((total - downloaded) / speed / 60, 1)
previous_file.write_text(json.dumps(state, indent=2))
print(json.dumps(state))
for name in ['verify-policy.log', 'evaluate-launcher.log']:
    path = out / name
    if path.exists():
        lines = path.read_text(errors='replace').splitlines()
        if lines:
            print(name + ': ' + lines[-1][:250])
run_dirs = sorted((out / 'results').glob('lingbot-vla-v2-6b-robotwin_*'))
if run_dirs:
    run = run_dirs[-1]
    print('current_run: ' + str(run))
    eval_log = run / 'eval_logs/lift_pot.log'
    if eval_log.exists():
        log = re.sub(r'\x1b\[[0-9;]*m', '', eval_log.read_text(errors='replace'))
        rates = re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%', log)
        if rates:
            print('episodes: ' + '/'.join(rates[-1][:2]) + ' rate=' + rates[-1][2] + '%')
        print('eval_log_tail: ' + log.splitlines()[-1][:250] if log.splitlines() else 'eval_log empty')
    for path in (run / 'inference_logs').glob('*.log'):
        lines = path.read_text(errors='replace').splitlines()
        if lines:
            print('inference_tail: ' + lines[-1][:250])
