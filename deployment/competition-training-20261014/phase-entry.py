"""Official training loop, with a 1000-step LR horizon shared by bounded phases."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
manifest = json.loads((path.parent / "manifest.json").read_text())
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5'
assert config['train']['max_steps'] in [250, 500, 1000]
assert manifest['lr_schedule_steps'] == 1000
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('competition_phase_training', root / 'lingbot-vla-v2/tasks/vla/train_lingbotvla.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original_scheduler = module.build_lr_scheduler

def shared_horizon_scheduler(*args, **kwargs):
    kwargs['train_steps'] = manifest['lr_schedule_steps']
    return original_scheduler(*args, **kwargs)

module.build_lr_scheduler = shared_horizon_scheduler
module.main()
