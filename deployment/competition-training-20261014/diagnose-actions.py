"""Compare actual 14-D action chunks on fixed demonstration frames, on GPU 0 only."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("label", choices=["pretrained", "smoke100", "official_robotwin"])
args = parser.parse_args()
run = Path(args.run)
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "lingbot-vla-v2"))
os.environ["QWEN3VL_PATH"] = str(root / "models/Qwen3-VL-4B-Instruct")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import matplotlib.pyplot as plt
from scripts.open_loop_eval import prepare_eval_observation, LeRobotDataset, LeRobotDatasetMetadata
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

torch.set_num_threads(2)
assert torch.cuda.device_count() == 1
manifest = json.loads((run / "manifest.json").read_text())
checkpoint = manifest["models"][args.label]
started = time.monotonic()
policy = LingbotVLAv2Server(checkpoint, use_length=50, chunk_ret=True,
                           use_bf16=False, use_fp32=True, use_compile=False)
policy.reset("robotwin")
load_seconds = time.monotonic() - started
data_root = root / "datasets/competition-clean50/lerobot/lift_pot"
meta = LeRobotDatasetMetadata("lift_pot", root=data_root)
delta = {key: [i / meta.fps for i in range(50)]
         for key in policy.vla.feature_transform.org_features["actions"]}
dataset = LeRobotDataset("lift_pot", root=data_root, delta_timestamps=delta)
arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
gripper = [6, 13]
records, arrays = [], {}
for episode in [0, 17, 34, 49]:
    info = meta.episodes[episode]
    start, end = int(info["dataset_from_index"]), int(info["dataset_to_index"])
    for fraction in [0.0, 0.25, 0.5, 0.75]:
        index = min(end - 1, start + int((end - start) * fraction))
        raw = dataset[index]
        observation, restored = prepare_eval_observation(policy, raw)
        ground_truth = np.asarray(restored["action"], dtype=np.float32)
        state = np.asarray(raw["observation.state"], dtype=np.float32)
        # Reset every sampled frame so no cached chunk from a different frame is reused.
        policy.reset("robotwin")
        torch.manual_seed(20261015 + index)
        before = time.monotonic()
        predicted = np.asarray(policy.infer(observation)["action"], dtype=np.float32)
        horizon = min(50, end - index)
        predicted, ground_truth = predicted[:horizon], ground_truth[:horizon]
        assert predicted.shape == ground_truth.shape == (horizon, 14)
        assert np.isfinite(predicted).all()
        original = np.asarray(raw["action"], dtype=np.float32)[:horizon]
        roundtrip = float(np.max(np.abs(original - ground_truth)))
        assert roundtrip < 1e-4, f"Normalization roundtrip mismatch: {roundtrip}"
        error = np.abs(predicted - ground_truth)
        hold_error = np.abs(state[None, :] - ground_truth)
        key = f"episode_{episode}_frame_{index}"
        arrays[key + "_predicted"] = predicted
        arrays[key + "_target"] = ground_truth
        arrays[key + "_state"] = state
        record = {"episode": episode, "frame": index, "fraction": fraction,
                  "horizon": horizon, "instruction": str(observation.get("task", "")),
                  "arm_mae_radians": float(error[:, arm].mean()),
                  "gripper_mae": float(error[:, gripper].mean()),
                  "first_action_arm_mae": float(error[0, arm].mean()),
                  "hold_state_arm_mae": float(hold_error[:, arm].mean()),
                  "hold_state_gripper_mae": float(hold_error[:, gripper].mean()),
                  "normalization_roundtrip_max_error": roundtrip,
                  "predicted_min": float(predicted.min()), "predicted_max": float(predicted.max()),
                  "inference_seconds": time.monotonic() - before}
        records.append(record)
        print(json.dumps(record), flush=True)
        if fraction == 0.5:
            fig, axes = plt.subplots(2, 2, figsize=(11, 6))
            for ax, dim in zip(axes.flat, [1, 8, 6, 13]):
                ax.plot(ground_truth[:, dim], label="demonstration")
                ax.plot(predicted[:, dim], label=args.label)
                ax.axhline(state[dim], color="gray", linestyle=":", label="current state")
                ax.set_title(["left joint 2", "right joint 2", "left gripper", "right gripper"][[1, 8, 6, 13].index(dim)])
                ax.set_xlabel("Action horizon (50 Hz)")
                ax.legend()
            fig.suptitle(f"{args.label}: episode {episode}, frame {index}")
            fig.tight_layout()
            fig.savefig(run / f"{args.label}-episode-{episode}.png", dpi=130)
            plt.close(fig)
np.savez_compressed(run / f"{args.label}-actions.npz", **arrays)
result = {"status": "completed", "label": args.label, "checkpoint": checkpoint,
          "load_seconds": load_seconds, "samples": len(records), "episodes": [0, 17, 34, 49],
          "norm_stats_file": policy.robot_norm_path, "precision": "FP32", "use_compile": False,
          "purpose": "training-demonstration diagnostic, not held-out evaluation",
          "mean_arm_mae_radians": float(np.mean([r["arm_mae_radians"] for r in records])),
          "mean_gripper_mae": float(np.mean([r["gripper_mae"] for r in records])),
          "mean_hold_state_arm_mae": float(np.mean([r["hold_state_arm_mae"] for r in records])),
          "mean_first_action_arm_mae": float(np.mean([r["first_action_arm_mae"] for r in records])),
          "records": records}
(run / f"{args.label}-diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
