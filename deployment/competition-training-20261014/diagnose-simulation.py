"""Instrument the unchanged official evaluation loop for ten lift_pot episodes."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

root = Path(__file__).resolve().parents[2]
run, port = Path(sys.argv[1]), int(sys.argv[2])
sys.path[:0] = [str(root / "RoboTwin"), str(root / "RoboTwin/policy"),
                str(root / "RoboTwin/script"), str(root / "RoboTwin/description/utils")]
import eval_policy_client_lingbotvla as official

original_eval = official.eval_policy
episodes = []

def instrumented_eval(*args, **kwargs):
    kwargs["test_num"] = 10
    env = args[1] if len(args) > 1 else kwargs["TASK_ENV"]
    original_setup, original_instruction = env.setup_demo, env.set_instruction
    original_take, original_close = env.take_action, env.close_env
    current = {"active": False}

    def setup(*setup_args, **setup_kwargs):
        current["active"] = False
        current["seed"] = int(setup_kwargs["seed"])
        return original_setup(*setup_args, **setup_kwargs)

    def instruction(*instruction_args, **instruction_kwargs):
        result = original_instruction(*instruction_args, **instruction_kwargs)
        current.update(active=True, records=[], screenshots=[], instruction=env.get_instruction(),
                       initial_height=float(env.pot.get_pose().p[2]))
        return result

    def take(action, *take_args, **take_kwargs):
        result = original_take(action, *take_args, **take_kwargs)
        if not current["active"]:
            return result
        left = np.array(env.robot.get_left_tcp_pose()[:3])
        right = np.array(env.robot.get_right_tcp_pose()[:3])
        contacts = [np.array(env.pot.get_contact_point(i)[:3]) for i in [0, 1]]
        current["records"].append({"step": int(env.take_action_cnt),
            "pot_height": float(env.pot.get_pose().p[2]),
            "left_distance_to_handle_m": float(np.linalg.norm(left - contacts[0])),
            "right_distance_to_handle_m": float(np.linalg.norm(right - contacts[1])),
            "left_gripper": float(env.robot.get_left_gripper_val()),
            "right_gripper": float(env.robot.get_right_gripper_val()),
            "command": np.asarray(action).tolist(), "success": bool(env.eval_success)})
        if len(episodes) < 3 and (env.take_action_cnt == 1 or env.take_action_cnt % 50 == 0):
            observation = env.get_obs()
            rgb = observation["observation"]["head_camera"]["rgb"]
            image_path = run / f"simulation-episode-{len(episodes)}-step-{env.take_action_cnt}.png"
            Image.fromarray(rgb).save(image_path)
            current["screenshots"].append(str(image_path))
        return result

    def close(*close_args, **close_kwargs):
        if current["active"]:
            records = current["records"]
            assert records, "No policy actions were recorded"
            max_height = max(r["pot_height"] for r in records)
            best_both_distance = min(max(r["left_distance_to_handle_m"], r["right_distance_to_handle_m"]) for r in records)
            success = any(r["success"] for r in records)
            if success:
                classification = "success"
            elif best_both_distance > 0.05:
                classification = "both arms did not simultaneously approach handles within 5 cm"
            elif max_height < current["initial_height"] + 0.03:
                classification = "approached handles but did not lift pot by 3 cm"
            elif max_height <= 0.82:
                classification = "partial lift below task height threshold"
            else:
                classification = "height reached but final grasp/orientation condition failed"
            episode = {"episode": len(episodes), "seed": current["seed"],
                       "instruction": current["instruction"], "success": success,
                       "steps": len(records), "initial_pot_height": current["initial_height"],
                       "max_pot_height": max_height, "best_simultaneous_handle_distance_m": best_both_distance,
                       "classification": classification, "screenshots": current["screenshots"]}
            (run / f"simulation-episode-{len(episodes)}-trace.json").write_text(json.dumps(records) + "\n")
            episodes.append(episode)
            (run / "simulation-progress.json").write_text(json.dumps({"episodes_completed": len(episodes), "episodes": episodes}, indent=2) + "\n")
            print("[diagnostic_episode] " + json.dumps(episode), flush=True)
            current["active"] = False
        return original_close(*close_args, **close_kwargs)

    env.setup_demo, env.set_instruction, env.take_action, env.close_env = setup, instruction, take, close
    next_seed, successes = original_eval(*args, **kwargs)
    assert len(episodes) == 10
    result = {"status": "completed", "task": "lift_pot", "setting": "demo_clean",
              "episodes": 10, "successes": int(successes), "success_rate": successes / 10,
              "next_seed": int(next_seed), "instruction_type": "unseen",
              "official_competition_score": False, "episode_details": episodes}
    (run / "simulation-diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0)

official.eval_policy = instrumented_eval
config = yaml.safe_load((root / "RoboTwin/policy/ACT/deploy_policy.yml").read_text())
config.update(task_name="lift_pot", task_config="demo_clean", train_config_name="smoke",
              ckpt_setting="smoke100-diagnostic", seed=0, policy_name="ACT", port=port,
              robo_name="robotwin", instruction_type="unseen", eval_video_log=False,
              output_dir=str(run / "simulation-results"))
official.main(config)
