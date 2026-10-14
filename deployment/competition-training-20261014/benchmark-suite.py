"""Serialize bounded benchmarks after diagnostics release the six authorized GPUs."""
import csv
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

run = Path(sys.argv[1])
directory = Path(__file__).resolve().parent
report = {"status": "waiting_for_diagnostics", "gpu_indices": list(range(6)),
          "forbidden_gpu_indices": [6, 7], "max_steps_per_trial": 8,
          "effective_batch_size": 48, "trials": []}

def save():
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run / "benchmark-report.json").write_text(json.dumps(report, indent=2) + "\n")

def parse_trial(label, code):
    trial = run / label
    log = (trial / "log.txt").read_text(errors="replace") if (trial / "log.txt").exists() else ""
    pattern = r"Step (\d+)/\d+, Epoch \d+, Loss ([\d.e+\-naninf]+),.*?GradNorm ([\d.e+\-naninf]+),.*?StepTime ([\d.]+)s"
    steps = [{"step": int(s), "loss": float(l), "gradient_norm": float(g), "seconds": float(t)}
             for s, l, g, t in re.findall(pattern, log)]
    median = statistics.median([s["seconds"] for s in steps[2:]]) if len(steps) > 2 else None
    optimizer = [json.loads(line) for line in re.findall(r"\[benchmark_optimizer\] (\{[^\n]+\})", log)]
    peaks = {}
    metrics = trial / "gpu-metrics.csv"
    if metrics.exists():
        for row in csv.reader(metrics.read_text().splitlines()):
            if len(row) == 6 and row[1].strip().isdigit():
                gpu, memory = row[1].strip(), int(row[3].split()[0])
                peaks[gpu] = max(peaks.get(gpu, 0), memory)
    passed = code == 0 and len(steps) == 8 and all(math.isfinite(s["loss"]) and math.isfinite(s["gradient_norm"]) for s in steps)
    return {"label": label, "exit_code": code, "passed": passed,
            "median_step_seconds_after_two": median,
            "samples_per_second": 48 / median if median else None,
            "peak_gpu_memory_mib": peaks, "steps": steps,
            "optimizer_step_timings": optimizer,
            "checkpoint_io_excluded": True,
            "error_tail": log[-2500:] if not passed else None}

save()
try:
    deadline = time.monotonic() + 7200
    markers = [run / "openloop-exit-code.txt", run / "simulation-exit-code.txt"]
    while not all(path.exists() for path in markers):
        assert time.monotonic() < deadline, 'Diagnostics did not finish within two hours'
        time.sleep(10)
    assert all(int(path.read_text()) == 0 for path in markers), 'A diagnostic failed; refusing automatic benchmark start'
    # Child exit markers are written after their inference servers have terminated.
    for label in ["muon-m1", "muon-m2", "dist-m1"]:
        report.update(status="running", current_trial=label)
        save()
        with (run / label / "launcher.log").open("w") as log:
            process = subprocess.Popen(["bash", str(directory / "run-benchmark.sh"), str(run / label)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (run / label / "launcher-pid.txt").write_text(str(process.pid) + "\n")
            try:
                code = process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        trial = parse_trial(label, code)
        report["trials"].append(trial)
        save()
        print(json.dumps({k: v for k, v in trial.items() if k not in ['steps', 'optimizer_step_timings', 'error_tail']}), flush=True)
        # Ensure OOM/failed torchrun children have actually released GPU memory.
        release_deadline = time.monotonic() + 120
        while True:
            memory = subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4,5','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if all(int(value) < 512 for value in memory.splitlines()):
                break
            assert time.monotonic() < release_deadline, 'GPU memory still occupied after benchmark; stop here'
            time.sleep(5)
    valid = [trial for trial in report['trials'] if trial['passed']]
    assert valid, 'All benchmark candidates failed'
    best = min(valid, key=lambda trial: trial['median_step_seconds_after_two'])
    report.update(status='completed', recommended_trial=best['label'])
    report.pop('current_trial', None)
    save()
except BaseException as error:
    report.update(status='failed', error=repr(error))
    save()
    raise
