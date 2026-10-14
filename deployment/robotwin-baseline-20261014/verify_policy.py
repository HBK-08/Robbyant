"""Strictly load the real checkpoint, infer on a real scene, and verify action output."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

OUT = Path(__file__).resolve().parent
MODEL = OUT.parent.parent / 'models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt'
started = time.monotonic()
server = LingbotVLAv2Server(str(MODEL), use_length=50, use_bf16=False, use_fp32=True, use_compile=False)
loaded = time.monotonic()
server.infer({'reset': True, 'robo_name': 'robotwin'})
with np.load(OUT / 'real-scene-observation.npz') as archive:
    observation = {key: archive[key] for key in archive.files}
observation['task'] = 'Lift the pot with both arms.'
response = server.infer(observation)
action = np.asarray(response['action'])
assert action.shape[-1] == 14 and action.ndim in [1, 2], action.shape
assert np.isfinite(action).all()
np.save(OUT / 'real-scene-action.npy', action)
result = {'passed': True, 'checkpoint': str(MODEL), 'strict_load': True,
          'precision': 'FP32', 'load_seconds': loaded - started,
          'first_inference_seconds': time.monotonic() - loaded,
          'action_shape': list(action.shape), 'action_min': float(action.min()),
          'action_max': float(action.max()), 'all_finite': True,
          'gpu': torch.cuda.get_device_name(0),
          'gpu_memory_total_bytes': torch.cuda.get_device_properties(0).total_memory,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'qwen_config_processor_only': True}
(OUT / 'policy-verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
