"""Resume authorized CPU preparation in independent background sessions; no training."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PYTHON = '/root/miniconda3/envs/lingbotvla/bin/python'
JOBS = ('download_inputs.py', 'convert_all.py', 'finish_preparation.py')

if __name__ == '__main__':
    # Refuse duplicate workers, including subprocesses outside this terminal session.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if args and args[0].endswith(b'/python') and any(
            a.decode(errors='replace').endswith('/' + name) for a in args[1:] for name in JOBS
        ):
            raise RuntimeError(f'Preparation worker already running: PID {proc.name}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    saved = []
    tasks = json.loads((OUT / 'input-manifest.json').read_text())['tasks']
    for task in tasks:
        target = ROOT / 'datasets/competition-clean50/lerobot' / task
        if not target.exists():
            continue
        info_file = target / 'meta/info.json'
        info = json.loads(info_file.read_text()) if info_file.exists() else {}
        if info.get('total_episodes') == 50 and (
            info.get('codebase_version') == 'v3.0' or (OUT / f'{task}-conversion-audit.json').exists()
        ):
            continue
        # Keep interrupted output for inspection; the official converter needs an empty target.
        quarantine = OUT / 'interrupted-conversions' / stamp / task
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        target.rename(quarantine)
        saved.append({'task': task, 'saved_path': str(quarantine), 'episodes': info.get('total_episodes')})

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    commands = [
        ('download_inputs.py', ['all-clean', '--workers', '4'], f'download-all-clean-{stamp}.log'),
        ('convert_all.py', ['--workers', '4'], f'convert-all-{stamp}.log'),
        ('finish_preparation.py', [], f'finish-preparation-{stamp}.log'),
    ]
    running = []
    children = []
    for name, args, log in commands:
        with (OUT / log).open('w') as stream:
            child = subprocess.Popen(
                [PYTHON, str(OUT / name), *args], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        running.append({'script': name, 'pid': child.pid, 'log': str(OUT / log)})
        children.append(child)
    record = {'resumed_at_utc': stamp, 'training_started': False, 'cuda_visible_devices': '',
              'saved_interrupted_outputs': saved, 'jobs': running}
    (OUT / 'resumed-background-jobs.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    # Keep the app execution session open so its process cleanup does not stop workers.
    while any(child.poll() is None for child in children):
        time.sleep(5)
    record['exit_codes'] = {job['script']: child.returncode for job, child in zip(running, children)}
    (OUT / 'resumed-background-jobs.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record['exit_codes']), flush=True)
    raise SystemExit(int(any(child.returncode for child in children)))
