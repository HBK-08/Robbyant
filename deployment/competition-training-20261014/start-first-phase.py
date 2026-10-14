"""Gate the first 250-step clean50 phase on action diagnostics and bounded benchmarks."""
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

diagnostic = Path(sys.argv[1])
directory = Path(__file__).resolve().parent
root = directory.parents[1]
state = {"status": "waiting_for_diagnostics_and_benchmarks", "diagnostic_directory": str(diagnostic),
         "first_phase_steps": 250, "planned_training_limit": 1000, "gpu_indices": list(range(6))}

def save():
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    (diagnostic / 'first-phase-status.json').write_text(json.dumps(state, indent=2) + '\n')

save()
try:
    deadline = time.monotonic() + 14400
    while True:
        assert time.monotonic() < deadline, 'Diagnostics and benchmarks exceeded four hours'
        benchmark = json.loads((diagnostic / 'benchmark-report.json').read_text())
        assert benchmark['status'] != 'failed', benchmark.get('error')
        if benchmark['status'] == 'completed':
            break
        time.sleep(15)
    summary = json.loads((diagnostic / 'openloop-summary.json').read_text())
    assert summary['smoke100']['mean_arm_mae_radians'] < summary['pretrained']['mean_arm_mae_radians']
    smoke = json.loads((diagnostic / 'smoke100-diagnostic.json').read_text())
    assert all(r['normalization_roundtrip_max_error'] < 1e-4 for r in smoke['records'])
    simulation = json.loads((diagnostic / 'simulation-diagnostic.json').read_text())
    assert simulation['status'] == 'completed' and simulation['episodes'] == 10
    pretrain = json.loads((root / 'deployment/competition-pretrain-20261014/pretrain-report.json').read_text())
    assert pretrain['ready_to_train'] and not pretrain['errors']
    selected = benchmark['recommended_trial']
    trial = next(t for t in benchmark['trials'] if t['label'] == selected)
    assert trial['passed']
    assert shutil.disk_usage(root).free > 180 * 1024**3
    run = directory / ('clean50-phase1-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir()
    config = yaml.safe_load((root / 'deployment/competition-pretrain-20261014/six-gpu-clean50.yaml').read_text())
    candidate = yaml.safe_load((diagnostic / selected / 'config.yaml').read_text())
    for key in ['micro_batch_size', 'gradient_accumulation_steps', 'global_batch_size', 'optimizer', 'use_compile', 'use_moe_expert_lr']:
        config['train'][key] = candidate['train'][key]
    for key, value in candidate['train'].items():
        if key.startswith('muon_') or key.startswith('dist_muon_'):
            config['train'][key] = value
    config['train'].update(output_dir=str(root / 'outputs' / run.name), max_steps=250,
                           save_steps=250, save_epochs=29000, enable_resume=False,
                           save_hf_weights=True, async_save_hf_weights=True,
                           resume_dataloader_state=True)
    manifest = {'scope': 'First bounded 250-step phase of clean50 training from designated pretrained weights',
                'authorized_at_utc': datetime.now(timezone.utc).isoformat(),
                'diagnostic_directory': str(diagnostic), 'benchmark_selected': selected,
                'lr_schedule_steps': 1000, 'phase_target_steps': 250, 'planned_training_limit': 1000,
                'gpu_indices': list(range(6)), 'forbidden_gpu_indices': [6, 7],
                'output_dir': config['train']['output_dir'], 'task_count': 50, 'demonstration_count': 2500,
                'pretrain_report_sha256': hashlib.sha256((root / 'deployment/competition-pretrain-20261014/pretrain-report.json').read_bytes()).hexdigest(),
                'base_history_sha256_before': hashlib.sha256(Path('/root/miniconda3/conda-meta/history').read_bytes()).hexdigest()}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (directory / 'current-training-phase.txt').write_text(str(run) + '\n')
    state.update(status='starting', run_directory=str(run), selected_trial=selected,
                 estimated_training_seconds=250 * trial['median_step_seconds_after_two'])
    save()
    with (run / 'launcher.log').open('w') as log:
        process = subprocess.Popen(['bash', str(directory / 'run-phase.sh'), str(run)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (run / 'launcher-pid.txt').write_text(str(process.pid) + '\n')
        state.update(status='running', launcher_pid=process.pid)
        save()
        while process.poll() is None:
            log_file = run / 'log.txt'
            text = log_file.read_text(errors='replace') if log_file.exists() else ''
            steps = re.findall(r'Step (\d+)/\d+, Epoch \d+, Loss ([\d.e+\-naninf]+),.*?GradNorm ([\d.e+\-naninf]+),.*?StepTime ([\d.]+)s', text)
            if steps:
                s, loss, grad, seconds = steps[-1]
                assert math.isfinite(float(loss)) and math.isfinite(float(grad)), 'Non-finite loss or gradient'
                state.update(completed_steps=int(s), last_loss=float(loss), last_gradient_norm=float(grad), last_step_seconds=float(seconds))
                times = [float(step[3]) for step in steps[2:]]
                if times:
                    state['estimated_remaining_training_seconds'] = statistics.median(times) * max(0, 250-int(s))
            assert shutil.disk_usage(root).free > 95 * 1024**3, 'Low disk space during training'
            save()
            time.sleep(30)
        assert process.returncode == 0, f'Training exited {process.returncode}; see launcher.log'
    final_text = (run / 'log.txt').read_text(errors='replace')
    final_steps = re.findall(r'Step (\d+)/\d+, Epoch \d+, Loss ([\d.e+\-naninf]+),.*?GradNorm ([\d.e+\-naninf]+),.*?StepTime ([\d.]+)s', final_text)
    assert final_steps and int(final_steps[-1][0]) == 250
    state['completed_steps'] = 250
    assert state.get('completed_steps') == 250
    hf = Path(config['train']['output_dir']) / 'checkpoints/global_step_250/hf_ckpt'
    assert (hf / 'model.safetensors.index.json').exists(), 'HF checkpoint export missing'
    assert manifest['base_history_sha256_before'] == hashlib.sha256(Path('/root/miniconda3/conda-meta/history').read_bytes()).hexdigest()
    state.update(status='checkpoint_ready_for_development_evaluation', checkpoint=str(hf), training_exit_code=0)
    save()
except BaseException as error:
    if 'process' in globals() and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    state.update(status='failed', error=repr(error))
    save()
    raise
