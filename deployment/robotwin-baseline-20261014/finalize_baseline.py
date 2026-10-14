"""Require a real, complete 100-episode result and consolidate reproducibility evidence."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

out = Path(__file__).resolve().parent
root = out.parent.parent
run = sorted((out / 'results').glob('lingbot-vla-v2-6b-robotwin_*'))[-1]
stats = (run / 'stats.txt').read_text()
assert 'Result: 1 done, 0 skipped' in stats, 'Official launcher did not finish the task'
assert 'All tasks fully executed 100 episodes' in stats
assert 'use_bf16=False, use_fp32=True, use_compile=True' in stats
log = re.sub(r'\x1b\[[0-9;]*m', '', (run / 'eval_logs/lift_pot.log').read_text(errors='replace'))
rates = re.findall(r'Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%', log)
success, episodes, rate = int(rates[-1][0]), int(rates[-1][1]), float(rates[-1][2])
assert episodes == 100
result_file = run / 'eval_results/lift_pot/_result.txt'
canonical = float(result_file.read_text().strip().splitlines()[-1])
assert abs(canonical - success / episodes) < 1e-9
assert abs(rate - success / episodes * 100) < 0.051
inputs = json.loads((out / 'input-checksum-audit.json').read_text())
policy = json.loads((out / 'policy-verification.json').read_text())
scene = json.loads((out / 'scene-verification.json').read_text())
expert = json.loads((out / 'expert-verification.json').read_text())
manifest = json.loads((out / 'download-manifest.json').read_text())
assert inputs['complete'] and policy['passed'] and policy['strict_load'] and policy['all_finite']
assert scene['passed'] and expert['plan_success'] and expert['task_success']
repos = {}
for name in ['lingbot-vla-v2', 'RoboTwin']:
    path = root / name
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    tracked_changes = subprocess.check_output(['git', '-C', str(path), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
    assert not tracked_changes, f'Tracked changes in {name}: {tracked_changes}'
    repos[name] = {'commit': commit, 'tracked_changes': tracked_changes}
copies = [('experiment/robotwin/eval_policy_client_lingbotvla.py', 'script/eval_policy_client_lingbotvla.py')]
copies += [('deploy/' + name, 'script/deploy/' + name) for name in ['__init__.py', 'websocket_client_policy.py', 'msgpack_numpy.py']]
for source, target in copies:
    assert (root / 'lingbot-vla-v2' / source).read_bytes() == (root / 'RoboTwin' / target).read_bytes()
summary = {'status': 'single-task baseline complete', 'task': 'lift_pot', 'task_config': 'demo_clean',
           'episodes': episodes, 'successes': success, 'success_rate_percent': rate,
           'instruction_type': 'unseen', 'seed': 0, 'gpu_physical_index': 0,
           'precision_flags': {'use_fp32': True, 'use_bf16': False, 'use_compile': True},
           'inference_env': 'lingbotvla', 'sim_env': 'RoboTwin', 'run_directory': str(run),
           'stats_file': str(run / 'stats.txt'), 'result_file': str(result_file),
           'input_checksum_files_matched': inputs['matched_files'],
           'model_revisions': {model['repo']: model['revision'] for model in manifest['models']},
           'asset_revision': manifest['assets'][0]['revision'], 'repositories': repos,
           'official_copied_client_files_match': True, 'model_validation': policy,
           'scene_validation': scene, 'expert_validation': expert,
           'norm_stats_sha256': hashlib.sha256((root / 'lingbot-vla-v2/assets/norm_stats/robotwin.json').read_bytes()).hexdigest(),
           'limitations': ['One clean task only; not a 50-task benchmark or official competition score'],
           'training_started': False}
(out / 'baseline-summary.json').write_text(json.dumps(summary, indent=2))
(out / 'progress-report.json').write_text(json.dumps(summary, indent=2))
manifest['status'] = 'approved inputs downloaded and verified; single-task baseline complete'
manifest['runtime_checks_completed'] = manifest.pop('pending_runtime_checks', [])
(out / 'download-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(summary, indent=2))
