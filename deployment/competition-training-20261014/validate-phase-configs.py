"""Parse all benchmark and first-phase candidates with the official dataclasses on CPU."""
import json
import os
import sys
from pathlib import Path

import yaml

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / 'lingbot-vla-v2'), str(root / 'deployment/competition-pretrain-20261014')]
from cpu_arguments import load_argument_classes
from lingbotvla.utils.arguments import parse_args
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

assert callable(register_optimizer_step_pre_hook) and callable(register_optimizer_step_post_hook)
Arguments = load_argument_classes().Arguments
os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='6')
run = Path(sys.argv[1])
results = []
for label in ['muon-m1', 'muon-m2', 'dist-m1']:
    path = run / label / 'config.yaml'
    sys.argv = ['validate', str(path)]
    args = parse_args(Arguments)
    assert args.train.global_batch_size == 48 and args.train.world_size == 6 and args.train.max_steps == 8
    assert args.train.save_hf_weights is False
    results.append({'label': label, 'passed': True, 'micro_batch': args.train.micro_batch_size,
                    'accumulation': args.train.gradient_accumulation_steps, 'optimizer': args.train.optimizer})
    candidate = yaml.safe_load(path.read_text())
    final = yaml.safe_load((root / 'deployment/competition-pretrain-20261014/six-gpu-clean50.yaml').read_text())
    for key in ['micro_batch_size', 'gradient_accumulation_steps', 'global_batch_size', 'optimizer', 'use_compile', 'use_moe_expert_lr']:
        final['train'][key] = candidate['train'][key]
    for key, value in candidate['train'].items():
        if key.startswith('muon_') or key.startswith('dist_muon_'):
            final['train'][key] = value
    final['train'].update(max_steps=250, save_steps=250, save_epochs=29000, enable_resume=False,
                          save_hf_weights=True, async_save_hf_weights=True, resume_dataloader_state=True)
    final_path = run / label / 'first-phase-config-preflight.yaml'
    final_path.write_text(yaml.safe_dump(final, sort_keys=False))
    sys.argv = ['validate', str(final_path)]
    args = parse_args(Arguments)
    assert args.train.max_steps == 250 and args.train.resume_dataloader_state and args.train.save_hf_weights
    results.append({'label': label + '-first-phase', 'passed': True})
(run / 'config-preflight.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results))
