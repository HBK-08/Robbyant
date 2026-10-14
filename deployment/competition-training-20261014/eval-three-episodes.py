"""Use the official evaluation loop for three smoke episodes; report the actual denominator."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run = Path(sys.argv[1])
port = int(sys.argv[2])
sys.path[:0] = [str(root / "RoboTwin"), str(root / "RoboTwin/policy"),
                str(root / "RoboTwin/script"), str(root / "RoboTwin/description/utils")]

import yaml
import eval_policy_client_lingbotvla as official

original_eval_policy = official.eval_policy

def smoke_eval(*args, **kwargs):
    kwargs["test_num"] = 3
    next_seed, successes = original_eval_policy(*args, **kwargs)
    result = {"status": "completed", "task": "lift_pot", "setting": "demo_clean",
              "episodes": 3, "successes": int(successes), "success_rate": successes / 3,
              "seed": 0, "next_seed": int(next_seed), "instruction_type": "unseen",
              "official_competition_score": False,
              "purpose": "checkpoint-to-simulation smoke validation"}
    (run / "simulation-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    # The official main assumes 100 episodes when writing its result; exit before
    # that section so this smoke never creates a result with the wrong denominator.
    raise SystemExit(0)

official.eval_policy = smoke_eval
config = yaml.safe_load((root / "RoboTwin/policy/ACT/deploy_policy.yml").read_text())
config.update(task_name="lift_pot", task_config="demo_clean", train_config_name="smoke",
              ckpt_setting="smoke100", seed=0, policy_name="ACT", port=port,
              robo_name="robotwin", instruction_type="unseen", video_fps=10,
              eval_video_log=False, output_dir=str(run / "simulation-results"))
official.main(config)
