"""Strictly reload the smoke checkpoint and infer from a saved real simulation observation."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

root = Path(__file__).resolve().parents[2]
run = Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
checkpoint = Path(manifest["output_dir"]) / "checkpoints/global_step_100/hf_ckpt"
base = root / "models/lingbot-vla-v2-6b"

def read_tensor(directory, key):
    for filename in sorted(directory.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as archive:
            if key in archive.keys():
                return archive.get_tensor(key)
    raise KeyError(key)

changes = {}
for key in ["model.state_proj.weight", "model.action_in_proj.weight", "model.action_out_proj.weight"]:
    before = read_tensor(base, key).float()
    after = read_tensor(checkpoint, key).float()
    assert torch.isfinite(after).all(), key
    delta = (after - before).abs()
    changes[key] = {"changed_values": int(torch.count_nonzero(delta)), "max_abs_delta": delta.max().item()}
    assert changes[key]["changed_values"] > 0, f"Parameter not updated: {key}"

os.environ["QWEN3VL_PATH"] = str(root / "models/Qwen3-VL-4B-Instruct")
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

started = time.monotonic()
torch.cuda.reset_peak_memory_stats()
server = LingbotVLAv2Server(str(checkpoint), use_length=50, use_bf16=False, use_fp32=True, use_compile=False)
loaded = time.monotonic()
server.infer({"reset": True, "robo_name": "robotwin"})
with np.load(root / "deployment/robotwin-baseline-20261014/real-scene-observation.npz") as archive:
    observation = {key: archive[key] for key in archive.files}
observation["task"] = "Lift the pot with both arms."
action = np.asarray(server.infer(observation)["action"])
assert action.shape[-1] == 14 and action.ndim in [1, 2], action.shape
assert np.isfinite(action).all()
np.save(run / "reloaded-policy-action.npy", action)
result = {"passed": True, "checkpoint": str(checkpoint), "strict_load": True,
          "parameter_updates": changes, "precision": "FP32", "load_seconds": loaded-started,
          "first_inference_seconds": time.monotonic()-loaded, "action_shape": list(action.shape),
          "action_min": float(action.min()), "action_max": float(action.max()),
          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
          "norm_stats_file": server.robot_norm_path}
(run / "policy-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
