"""Use the official training loop for <=10 steps, excluding disposable checkpoint IO."""
import json
import os
import runpy
import sys
import time
from pathlib import Path

import yaml
import torch
import lingbotvla.checkpoint as checkpoint

config = yaml.safe_load(Path(sys.argv[1]).read_text())
assert 1 <= config["train"]["max_steps"] <= 10, "Benchmark wrapper must never run long training"
assert config["train"]["save_hf_weights"] is False
assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5"
original_build = checkpoint.build_checkpointer

class BenchmarkCheckpointer:
    def __init__(self, original):
        self.original = original

    def load(self, *args, **kwargs):
        raise RuntimeError("Benchmarks start from the same pretrained weights; resume is disabled")

    def save(self, *args, **kwargs):
        print("[benchmark] checkpoint IO intentionally skipped; weights are disposable", flush=True)

def build_benchmark_checkpointer(*args, **kwargs):
    return BenchmarkCheckpointer(original_build(*args, **kwargs))

checkpoint.build_checkpointer = build_benchmark_checkpointer

# Observe optimizer duration via PyTorch's public step hooks, including CUDA completion.
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
timers = {}
def before_step(optimizer, args, kwargs):
    torch.cuda.synchronize()
    timers[id(optimizer)] = time.monotonic()
def after_step(optimizer, args, kwargs):
    torch.cuda.synchronize()
    elapsed = time.monotonic() - timers.pop(id(optimizer))
    if os.environ.get("RANK", "0") == "0":
        print("[benchmark_optimizer] " + json.dumps({"class": type(optimizer).__name__, "seconds": elapsed}), flush=True)
register_optimizer_step_pre_hook(before_step)
register_optimizer_step_post_hook(after_step)
runpy.run_path(str(Path(__file__).resolve().parents[2] / "lingbot-vla-v2/tasks/vla/train_lingbotvla.py"), run_name="__main__")
