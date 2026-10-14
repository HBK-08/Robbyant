"""Convert clean50 with bounded CPU workers; no model or training execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
DATA = ROOT / 'datasets/competition-clean50'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''

def read_json(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None

def complete(task):
    info = read_json(DATA / 'lerobot' / task / 'meta/info.json')
    return bool(info and info['codebase_version']=='v3.0' and info['total_episodes']==50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4
    tasks = read_json(OUT / 'input-manifest.json')['tasks']
    allowed = sorted(os.sched_getaffinity(0))
    assert len(allowed) >= args.workers * 8
    cpu_groups = [allowed[i*8:(i+1)*8] for i in range(args.workers)]
    lock = threading.Lock()
    slots = list(range(args.workers))
    active = {}
    failures = {}
    started = time.time()
    logs = OUT / 'conversion-logs'
    logs.mkdir(exist_ok=True)

    def convert(task):
        with lock:
            slot = slots.pop()
            active[task] = {'slot':slot,'cpu_affinity':cpu_groups[slot],'started':time.time()}
        try:
            target = DATA / 'lerobot' / task
            info = read_json(target / 'meta/info.json')
            if info and info['codebase_version']=='v2.1' and info['total_episodes']==50 and (OUT / f'{task}-conversion-audit.json').exists():
                env = dict(os.environ, HF_HOME=str(OUT/'cache/huggingface'), XDG_CACHE_HOME=str(OUT/'cache'),
                    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    LD_LIBRARY_PATH='/root/miniconda3/envs/lingbotvla/lib', MPLCONFIGDIR=str(OUT/'cache/matplotlib'),
                    PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                command = ['/root/miniconda3/envs/lingbotvla/bin/python','-m','lerobot.datasets.v30.convert_dataset_v21_to_v30',
                    '--repo-id',task,'--root',str(DATA/'lerobot'),'--push-to-hub','false']
            else:
                env = dict(os.environ)
                command = ['bash', str(OUT/'convert-clean.sh'),task]
            command = ['taskset','-c',','.join(map(str,cpu_groups[slot]))] + command
            with (logs/f'{task}.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                with lock: active[task]['pid']=child.pid
                code = child.wait()
            assert code == 0 and complete(task), f'{task} conversion failed with exit {code}'
            print('CONVERTED',task,flush=True)
        except Exception as error:
            with lock: failures[task]=str(error)
            print('FAILED',task,str(error),flush=True)
        finally:
            with lock:
                active.pop(task,None)
                slots.append(slot)

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            done = [t for t in tasks if complete(t)]
            with lock: active_snapshot = {k:dict(v) for k,v in active.items()}
            for task in tasks:
                if task in done or task in futures or task in failures: continue
                if len([f for f in futures.values() if not f.done()]) >= args.workers: break
                if (DATA / 'raw' / task / '.extraction-complete').exists():
                    futures[task]=pool.submit(convert,task)
            state={'training_started':False,'completed_tasks':done,'completed_count':len(done),
                'total_tasks':50,'active':active_snapshot,'failures':dict(failures),'elapsed_seconds':int(time.time()-started),
                'cpu_workers':args.workers,'waiting_for_download':[t for t in tasks if t not in done and t not in futures and t not in failures]}
            tmp=OUT/'conversion-progress.json.tmp'
            tmp.write_text(json.dumps(state,indent=2)); tmp.replace(OUT/'conversion-progress.json')
            if len(done)+len(failures)==50: break
            time.sleep(15)
    if failures: raise SystemExit(1)
    print('ALL 50 CLEAN TASKS CONVERTED; TRAINING NOT STARTED',flush=True)
