"""Report this training run from its logs without allocating CUDA memory."""
import csv
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
run = Path((ROOT / "current-run.txt").read_text().strip())
manifest = json.loads((run / "manifest.json").read_text())
log_path = run / "log.txt"
log = log_path.read_text(errors="replace") if log_path.exists() else ""
pattern = re.compile(r"Step (\d+)/\d+, Epoch \d+, Loss ([\d.e+\-naninf]+),.*?GradNorm ([\d.e+\-naninf]+),.*?StepTime ([\d.]+)s")
steps = [{"step": int(s), "loss": float(l), "grad_norm": float(g), "step_seconds": float(t)} for s, l, g, t in pattern.findall(log)]
exit_path = run / "exit-code.txt"
exit_code = int(exit_path.read_text()) if exit_path.exists() else None
steady_times = [s["step_seconds"] for s in steps[2:]]
median_time = statistics.median(steady_times) if steady_times else None
peaks = {}
metrics = run / "gpu-metrics.csv"
if metrics.exists():
    for row in csv.reader(metrics.read_text().splitlines()):
        if len(row) == 6 and row[1].strip().isdigit():
            gpu = row[1].strip()
            peaks[gpu] = max(peaks.get(gpu, 0), int(row[3].split()[0]))
result = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_directory": str(run),
    "output_directory": manifest["output_dir"],
    "status": "completed" if exit_code == 0 else "failed" if exit_code is not None else "running",
    "exit_code": exit_code,
    "target_steps": 100,
    "completed_steps": steps[-1]["step"] if steps else 0,
    "last_step": steps[-1] if steps else None,
    "finite_losses_and_gradients": all(math.isfinite(s["loss"]) and math.isfinite(s["grad_norm"]) for s in steps) if steps else None,
    "median_step_seconds_after_first_two": median_time,
    "estimated_remaining_training_seconds": median_time * max(0, 100-steps[-1]["step"]) if median_time is not None else None,
    "peak_sampled_gpu_memory_mib": peaks,
    "hf_checkpoint_present": (Path(manifest["output_dir"]) / "checkpoints/global_step_100/hf_ckpt/config.json").exists(),
}
(run / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
