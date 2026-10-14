"""Generate independent official-derived configs without importing a trainer."""
from copy import deepcopy
import json
from pathlib import Path
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
DATA = ROOT / 'datasets/competition-clean50'
official = ROOT / 'lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml'
config = yaml.safe_load(official.read_text())
config['model']['model_path'] = str(ROOT / 'models/lingbot-vla-v2-6b')
config['model']['tokenizer_path'] = str(ROOT / 'models/Qwen3-VL-4B-Instruct')
config['data'].update(robot_config_root=str(OUT / 'robot_configs'), num_workers=2)
config['train'].update(enable_gradient_checkpointing=True, enable_full_shard=True,
    micro_batch_size=1, global_batch_size=48, gradient_accumulation_steps=8,
    data_parallel_replicate_size=1, data_parallel_shard_size=6,
    enable_resume=False, save_steps=100, max_steps=100, use_compile=False,
    output_dir=str(ROOT / 'outputs/competition-six-gpu-smoke'))
align = config['train']['align_params']
align['depth']['moge_path'] = str(ROOT / 'models/moge-2-vitb-normal/model.pt')
align['depth']['morgbd_path'] = str(ROOT / 'models/lingbot-vla-v2-6b/depth/model.pt')
align['video']['ckpt_path'] = str(ROOT / 'models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth')
align['video']['config_path'] = str(ROOT / 'models/lingbot-vla-v2-6b/dino_video/config.yaml')
config['data']['train_path'] = str(OUT / 'train-lift-pot-clean50.txt')
config['data']['norm_stats_file'] = str(OUT / 'norm-lift-pot-clean50.json')
smoke = deepcopy(config)
smoke['data']['robot_config_root'] = str(OUT / 'robot_configs/smoke')
(OUT / 'six-gpu-smoke.yaml').write_text(yaml.safe_dump(smoke, sort_keys=False))
config['data']['train_path'] = str(OUT / 'train-clean50.txt')
config['data']['norm_stats_file'] = str(OUT / 'norm-clean50.json')
config['train'].update(max_steps=1000, save_steps=250, use_compile=True,
    output_dir=str(ROOT / 'outputs/competition-clean50-first-segment'))
(OUT / 'six-gpu-clean50.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
tasks = json.loads((OUT / 'input-manifest.json').read_text())['tasks']
(OUT / 'train-clean50.txt').write_text(''.join(f'robotwin {DATA / "lerobot" / task}\n' for task in tasks))
(OUT / 'train-lift-pot-clean50.txt').write_text(f'robotwin {DATA / "lerobot/lift_pot"}\n')
for dirname, stats in [('robot_configs', 'norm-clean50.json'), ('robot_configs/smoke', 'norm-lift-pot-clean50.json')]:
    dst = OUT / dirname
    dst.mkdir(parents=True, exist_ok=True)
    robot = yaml.safe_load((ROOT / 'lingbot-vla-v2/configs/robot_configs/robotwin.yaml').read_text())
    robot['norm_stats'] = str(OUT / stats)
    (dst / 'robotwin.yaml').write_text(yaml.safe_dump(robot, sort_keys=False))
norm = yaml.safe_load((ROOT / 'lingbot-vla-v2/configs/vla/norm_compute/post_data.yaml').read_text())
norm['data'].update(data_name='multi', robot_name='robotwin', train_path=str(OUT / 'train-lift-pot-clean50.txt'),
    robot_config_root=str(OUT / 'robot_configs/smoke'), norm_path=str(OUT / 'norm-lift-pot-clean50.json'), num_workers=2)
norm['train']['micro_batch_size'] = 48
(OUT / 'norm-lift-pot.yaml').write_text(yaml.safe_dump(norm, sort_keys=False))
norm['data'].update(train_path=str(OUT / 'train-clean50.txt'), robot_config_root=str(OUT / 'robot_configs'),
    norm_path=str(OUT / 'norm-clean50.json'))
(OUT / 'norm-clean50.yaml').write_text(yaml.safe_dump(norm, sort_keys=False))
print('Generated configs; training not started. Full-data first segment is provisional 1000 steps.')
