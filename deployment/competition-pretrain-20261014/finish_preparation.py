"""Finish CPU normalization and checks after approved inputs arrive; never launch training."""
import json
import os
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def run(mode,log):
    with (OUT/log).open('w') as stream:
        subprocess.run(['bash',str(OUT/'prepare.sh'),mode],stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    print('Waiting for full clean50 data and approved model downloads; training is disabled',flush=True)
    norm_done=False
    checkpoints_done=False
    while True:
        conv=read(OUT/'conversion-progress.json') or {}
        if conv.get('failures'):
            raise RuntimeError(f'Conversion failures need repair: {conv["failures"]}')
        if conv.get('completed_count')==50 and not norm_done:
            print('Running official clean50 CPU normalization',flush=True)
            run('norm-all','norm-clean50.log')
            norm_done=True
            print('CLEAN50 NORMALIZATION COMPLETE',flush=True)
        models=read(OUT/'models-verification.json')
        if models and len(models)==31 and not checkpoints_done:
            print('Checking checkpoint indices and loading frozen teachers on CPU',flush=True)
            run('checkpoints','verify-checkpoints.log')
            checkpoints_done=True
            print('CHECKPOINT CPU VALIDATION COMPLETE',flush=True)
        if norm_done and checkpoints_done and (OUT/'all-clean-verification.json').exists():
            print('Running complete data and configuration preflight',flush=True)
            run('verify','verify-full.log')
            report=read(OUT/'pretrain-report.json')
            assert report['ready_to_train'] and not report['training_started'],report['errors']
            print('TRAINING PREPARATION COMPLETE; TRAINING NOT STARTED',flush=True)
            break
        time.sleep(20)
