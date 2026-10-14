"""Initialize the genuine lift_pot scene and save an observation for policy checks."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from envs.lift_pot import lift_pot

OUT = Path(__file__).resolve().parent
args = yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
camera = yaml.safe_load(Path('task_config/_camera_config.yml').read_text())[args['camera']['head_camera_type']]
robot_file = './assets/embodiments/aloha-agilex/'
args.update(task_name='lift_pot', task_config='demo_clean', ckpt_setting='official_50k',
            policy_name='lingbotvla', left_robot_file=robot_file, right_robot_file=robot_file,
            left_embodiment_config=yaml.safe_load(Path(robot_file, 'config.yml').read_text()),
            right_embodiment_config=yaml.safe_load(Path(robot_file, 'config.yml').read_text()),
            dual_arm_embodied=True, head_camera_h=camera['h'], head_camera_w=camera['w'],
            eval_mode=True, eval_video_log=False, save_data=False)
task = lift_pot()
try:
    task.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
    task.set_instruction('Lift the pot with both arms.')
    obs = task.get_obs()
    formatted = {'observation.images.cam_high': obs['observation']['head_camera']['rgb'],
                 'observation.images.cam_left_wrist': obs['observation']['left_camera']['rgb'],
                 'observation.images.cam_right_wrist': obs['observation']['right_camera']['rgb'],
                 'observation.state': obs['joint_action']['vector']}
    assert formatted['observation.state'].shape == (14,)
    assert all(np.isfinite(value).all() for value in formatted.values())
    np.savez_compressed(OUT / 'real-scene-observation.npz', **formatted)
    Image.fromarray(formatted['observation.images.cam_high']).save(OUT / 'lift-pot-scene.png')
    result = {'passed': True, 'task': 'lift_pot', 'seed': 100000,
              'renderer_device': str(task.scene.render_system.device),
              'renderer_pci': task.scene.render_system.device.pci_string,
              'pot_variant': int(task.model_id),
              'observations': {key: list(value.shape) for key, value in formatted.items()}}
    (OUT / 'scene-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if '--expert' in sys.argv:
        started_info = task.play_once()
        expert = {'seed': 100000, 'plan_success': bool(task.plan_success),
                  'task_success': bool(task.check_success()), 'episode_info': started_info}
        (OUT / 'expert-verification.json').write_text(json.dumps(expert, indent=2))
        print('EXPERT', json.dumps(expert), flush=True)
finally:
    task.close_env()
