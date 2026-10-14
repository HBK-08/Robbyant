"""Call official RoboTwin data conversion functions in an isolated CPU environment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
DATA = ROOT / 'datasets/competition-clean50'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Conversion must be CPU only'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='lift_pot')
    args = parser.parse_args()
    task = args.task
    assert task in json.loads((OUT / 'input-manifest.json').read_text())['tasks']
    raw = DATA / 'raw' / task / 'aloha-agilex_clean_50'
    processed = DATA / 'processed' / task
    target = DATA / 'lerobot' / task
    if target.exists():
        raise RuntimeError('Refusing to overwrite an existing LeRobot dataset')
    np.random.seed(0)
    audit = {'task': task, 'setting': 'aloha-agilex_clean_50', 'training_started': False, 'episodes': []}
    for i in range(50):
        h5 = raw / 'data' / f'episode{i}.hdf5'
        instruction = json.loads((raw / 'instructions' / f'episode{i}.json').read_text())
        assert instruction['seen'] and all(isinstance(x, str) for x in instruction['seen'])
        with h5py.File(h5) as f:
            n = len(f['joint_action/left_arm'])
            assert n > 1
            for key in ['left_arm', 'right_arm', 'left_gripper', 'right_gripper']:
                arr = f['joint_action/' + key][:]
                assert len(arr) == n and np.isfinite(arr).all()
            for cam in ['head_camera', 'left_camera', 'right_camera']:
                assert len(f[f'observation/{cam}/rgb']) == n
        audit['episodes'].append({'raw_episode': i, 'raw_frames': n, 'training_frames': n - 1,
            'raw_sha256': hashlib.sha256(h5.read_bytes()).hexdigest(), 'seen_instruction_count': len(instruction['seen'])})
    process_path = ROOT / 'RoboTwin/policy/pi0/scripts/process_data.py'
    process = module('official_process_data', process_path)
    if not (processed / '.processed-complete').exists():
        assert process.data_transform(str(raw), 50, str(processed)) == 50
        (processed / '.processed-complete').write_text('official data_transform, state[t] -> action[t+1]')
    converter_path = ROOT / 'RoboTwin/policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py'
    converter = module('official_lerobot_converter', converter_path)
    converter.HF_LEROBOT_HOME = DATA / 'lerobot'
    # Sort episode paths numerically, retaining every official frame and seen instruction.
    files = [processed / f'episode_{i}/episode_{i}.hdf5' for i in range(50)]
    settings = converter.DatasetConfig(image_writer_processes=2, image_writer_threads=2, video_backend='pyav')
    dataset = converter.create_empty_dataset(task, robot_type='aloha', mode='video', dataset_config=settings)
    assert dataset.root.resolve() == target.resolve(), 'LeRobot cache root must equal the requested output directory'
    try:
        converter.populate_dataset(dataset, files, task, episodes=list(range(50)))
    finally:
        dataset.stop_image_writer()
    audit['output_root'] = str(target)
    audit['total_training_frames'] = sum(e['training_frames'] for e in audit['episodes'])
    audit['preprocessing_script_sha256'] = hashlib.sha256(process_path.read_bytes()).hexdigest()
    audit['conversion_script_sha256'] = hashlib.sha256(converter_path.read_bytes()).hexdigest()
    audit['lerobot_revision'] = 'a445d9c9da6bea99a8972daa4fe1fdd053d711d2'
    audit['output_format'] = 'v2.1; convert to v3.0 with official LeRobot utility before training environment reads'
    (OUT / f'{task}-conversion-audit.json').write_text(json.dumps(audit, indent=2))
    print('Converted', task, 'episodes=50', 'frames=', audit['total_training_frames'], flush=True)
