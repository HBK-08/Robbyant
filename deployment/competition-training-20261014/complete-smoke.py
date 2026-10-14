"""Wait for training, validate its checkpoint, and persist the final audit."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

directory = Path(__file__).resolve().parent
run = Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
stage = "waiting_for_training"
report = {"scope": manifest["scope"], "run_directory": str(run), "status": stage}

def save():
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")

try:
    save()
    pid = int((run / "launcher-pid.txt").read_text())
    while not (run / "exit-code.txt").exists():
        os.kill(pid, 0)
        subprocess.run([sys.executable, str(directory / "progress.py")], stdout=subprocess.DEVNULL, check=True)
        time.sleep(30)
    report["training_exit_code"] = int((run / "exit-code.txt").read_text())
    assert report["training_exit_code"] == 0, 'Training exited unsuccessfully; see log.txt'
    subprocess.run([sys.executable, str(directory / "progress.py")], stdout=subprocess.DEVNULL, check=True)
    progress = json.loads((run / "progress.json").read_text())
    report["training"] = progress
    assert progress["completed_steps"] == 100 and progress["finite_losses_and_gradients"], 'Training validation failed'
    assert progress["hf_checkpoint_present"], 'HF export missing; see checkpoint export logs'
    stage = "checkpoint_reload_and_simulation"
    report["status"] = stage
    save()
    free_memory = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True)
    assert int(free_memory.strip()) < 512, 'GPU 0 occupied after training; validation deferred'
    validation = subprocess.Popen(['bash', str(directory / 'run-validation.sh'), str(run)], start_new_session=True)
    try:
        validation_code = validation.wait(timeout=4500)
    except BaseException:
        os.killpg(validation.pid, signal.SIGTERM)
        try:
            validation.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(validation.pid, signal.SIGKILL)
            validation.wait()
        raise
    assert validation_code == 0, f'Checkpoint/simulation validation exited {validation_code}; see validation logs'
    report["policy"] = json.loads((run / "policy-verification.json").read_text())
    report["simulation"] = json.loads((run / "simulation-verification.json").read_text())
    report["base_history_unchanged"] = hashlib.sha256(Path('/root/miniconda3/conda-meta/history').read_bytes()).hexdigest() == manifest['base_history_sha256_before']
    report["gpu_0_to_5_after_validation"] = subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4,5','--query-gpu=index,memory.used,utilization.gpu','--format=csv'], text=True)
    report["status"] = "smoke_complete"
    report["full_50_task_training_started"] = False
    save()
except Exception as error:
    report["status"] = "failed"
    report["failed_stage"] = stage
    report["error"] = repr(error)
    save()
    raise
