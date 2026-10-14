"""Validate arguments, real CPU data and input availability without starting training."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preflight must be CPU only'
report = {'training_started': False, 'optimizer_created': False, 'model_forward_executed': False,
    'backward_executed': False, 'checks': {}, 'errors': [], 'pending': []}
assert not torch.cuda.is_initialized()
from cpu_arguments import load_argument_classes
official_args = load_argument_classes()
report['argument_definition_source_sha256'] = official_args.source_sha256

def check(name, fn):
    try:
        value = fn()
        report['checks'][name] = {'passed': True, 'result': value}
        print('PASS', name, flush=True)
        return value
    except Exception as e:
        report['checks'][name] = {'passed': False, 'error': str(e)}
        report['errors'].append(name)
        traceback.print_exc()
        return None

def arguments():
    from tasks.vla.train_lingbotvla import Arguments
    from lingbotvla.utils.arguments import parse_args
    result = {}
    os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='6')
    for name in ['six-gpu-smoke.yaml', 'six-gpu-clean50.yaml']:
        sys.argv = [__file__, str(OUT / name)]
        args = parse_args(Arguments)
        assert args.train.data_parallel_size == 6
        assert args.train.global_batch_size == 48
        assert args.train.gradient_accumulation_steps == 8
        assert args.train.micro_batch_size == 1
        assert args.train.enable_full_shard and args.train.enable_gradient_checkpointing
        assert args.train.enable_fp32
        assert args.train.align_params['use_future_video']
        assert args.model.model_path == str(ROOT / 'models/lingbot-vla-v2-6b')
        result[name] = {'world_size': 6, 'batch_size': 48, 'gradient_accumulation_steps': 8,
            'optimizer': args.train.optimizer, 'max_steps': args.train.max_steps,
            'data_path': args.data.train_path, 'teacher_paths': [args.train.align_params['depth']['moge_path'],
                args.train.align_params['depth']['morgbd_path'], args.train.align_params['video']['ckpt_path']]}
    os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
    return result

def input_availability():
    manifest = json.loads((OUT / 'input-manifest.json').read_text())
    missing = [j for j in manifest['files'] if not Path(j['target']).is_file()]
    for j in missing:
        report['pending'].append({'kind': j['kind'], 'path': j['target'], 'bytes': j['bytes']})
    unverified = []
    for job in manifest['files']:
        if not Path(job['target']).is_file():
            continue
        key = hashlib.sha256(job['target'].encode()).hexdigest()
        proof = OUT / 'download-records' / (key + '.json')
        if not proof.exists():
            unverified.append(job['target'])
        else:
            record = json.loads(proof.read_text())
            assert record['verified'] and record['revision'] == job['revision']
            assert Path(job['target']).stat().st_size == job['bytes']
    report['pending'].extend({'kind': 'download_verification', 'path': p} for p in unverified)
    assert len(manifest['tasks']) == 50
    lines = (OUT / 'train-clean50.txt').read_text().splitlines()
    assert len(lines) == 50 and len(set(lines)) == 50
    assert all('randomized' not in j['path'] for j in manifest['files'] if j['kind'] == 'dataset')
    missing_datasets = [str(ROOT / 'datasets/competition-clean50/lerobot' / t) for t in manifest['tasks']
        if not (ROOT / 'datasets/competition-clean50/lerobot' / t / 'meta/info.json').exists()]
    report['pending'].extend({'kind': 'converted_dataset', 'path': p} for p in missing_datasets)
    if not (OUT / 'norm-clean50.json').exists():
        report['pending'].append({'kind': 'norm_stats', 'path': str(OUT / 'norm-clean50.json')})
    checkpoint_report = OUT / 'checkpoint-verification.json'
    if not checkpoint_report.exists():
        report['pending'].append({'kind': 'checkpoint_cpu_verification', 'path': str(checkpoint_report)})
    else:
        assert json.loads(checkpoint_report.read_text())['passed']
    return {'tasks': 50, 'manifest_files': len(manifest['files']), 'missing_files': len(missing),
        'missing_converted_datasets': len(missing_datasets), 'randomized_training_files': 0,
        'existing_files_without_this_stage_verification': len(unverified),
        'base_model_revision': next(m['revision'] for m in json.loads((OUT / 'input-metadata.json').read_text())['models']
            if m['repo'] == 'robbyant/lingbot-vla-v2-6b')}

def dataset_read():
    from lingbotvla.data import build_vla_dataset, VLADataCollatorWithPacking
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.utils.arguments import parse_args
    from tasks.vla.train_lingbotvla import Arguments
    os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='6')
    sys.argv = [__file__, str(OUT / 'six-gpu-smoke.yaml')]
    args = parse_args(Arguments)
    args.data.chunk_size = args.train.chunk_size
    config = LingbotVLAV2Config(**{**vars(args.model), **vars(args.train)})
    processor = build_processor(args.model.tokenizer_path)
    dataset = build_vla_dataset(args.data, args.model, config, processor, use_depth_align=True)
    assert len(dataset._datasets) == 1
    child = dataset._datasets[0]
    assert child.dataset_meta.total_episodes == 50
    samples = []
    for idx in [0, len(dataset)//2, len(dataset)-1]:
        sample = dataset[idx]
        for key, val in sample.items():
            if torch.is_tensor(val) and val.is_floating_point():
                assert torch.isfinite(val).all(), key
                assert val.device.type == 'cpu'
        shapes = {k: list(v.shape) for k, v in sample.items() if torch.is_tensor(v)}
        samples.append({'index': idx, 'shapes': shapes, 'keys': list(sample)})
    batch = VLADataCollatorWithPacking()([dataset[0], dataset[min(1, len(dataset)-1)]])
    def describe(v):
        if torch.is_tensor(v):
            assert v.device.type == 'cpu'
            return {'shape': list(v.shape), 'dtype': str(v.dtype)}
        if isinstance(v, dict): return {k: describe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [describe(x) for x in v]
        return str(type(v).__name__)
    result = {'episodes': 50, 'frames': len(dataset), 'fps': child.dataset_meta.fps,
        'format': child.dataset_meta.info['codebase_version'], 'samples': samples, 'collated_batch': describe(batch)}
    os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
    return result

def time_alignment():
    import h5py
    import pyarrow.parquet as pq
    data = ROOT / 'datasets/competition-clean50'
    audited = 0
    tasks = json.loads((OUT / 'input-manifest.json').read_text())['tasks']
    completed = []
    for task in tasks:
        root=data/'lerobot'/task
        if not (root/'meta/info.json').exists(): continue
        info=json.loads((root/'meta/info.json').read_text())
        if info['codebase_version']!='v3.0': continue
        assert info['total_episodes']==50 and info['fps']==50
        files=sorted((root/'data').rglob('*.parquet'))
        frames=pq.read_table(files).to_pandas()
        task_frames=0
        for ep in range(50):
            episode = frames[frames.episode_index == ep]
            with h5py.File(data / f'raw/{task}/aloha-agilex_clean_50/data/episode{ep}.hdf5') as raw:
                joint = raw['joint_action']
                state = np.concatenate([joint['left_arm'][:], joint['left_gripper'][:].reshape(-1,1),
                    joint['right_arm'][:], joint['right_gripper'][:].reshape(-1,1)], axis=-1).astype(np.float32)
            assert len(episode) == len(state)-1
            np.testing.assert_allclose(np.stack(episode['observation.state']), state[:-1], rtol=0, atol=0)
            np.testing.assert_allclose(np.stack(episode['action']), state[1:], rtol=0, atol=0)
            np.testing.assert_allclose(episode.timestamp, np.arange(len(episode))/50, atol=1e-6)
            task_frames += len(episode)
        assert task_frames==info['total_frames']
        audited += task_frames
        completed.append(task)
        print('ALIGNED',task,'50 episodes',task_frames,'frames',flush=True)
    assert completed, 'No converted dataset available'
    return {'tasks':len(completed),'raw_episodes':50*len(completed), 'frames_checked': audited,
        'state_action_relation': 'observation=raw_state[t], action=raw_state[t+1]', 'fps': 50}

def full_dataset_read():
    from lingbotvla.data import build_vla_dataset, VLADataCollatorWithPacking
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.utils.arguments import parse_args
    from tasks.vla.train_lingbotvla import Arguments
    os.environ.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='6')
    sys.argv=[__file__,str(OUT/'six-gpu-clean50.yaml')]
    args=parse_args(Arguments)
    args.data.chunk_size=args.train.chunk_size
    cfg=LingbotVLAV2Config(**{**vars(args.model),**vars(args.train)})
    processor=build_processor(args.model.tokenizer_path)
    ds=build_vla_dataset(args.data,args.model,cfg,processor,use_depth_align=True)
    assert len(ds._datasets)==50
    samples=[]
    frames=0
    for child in ds._datasets:
        assert child.dataset_meta.total_episodes==50
        for idx in [0,len(child)-1]:
            sample=child[idx]
            assert sample['state'].shape==(55,) and sample['actions'].shape==(50,55)
            assert int(sample['state_joint_mask'].sum())==14
            assert sample['pil_images'].shape==(3,3,256,256)
            assert sample['future_pil_images'].shape==(3,3,256,256)
            assert sample['lang_tokens'].shape==(72,)
            assert sample['action_is_pad'].shape==(50,)
            for key,val in sample.items():
                if torch.is_tensor(val):
                    assert val.device.type=='cpu'
                    if val.is_floating_point(): assert torch.isfinite(val).all(),key
            samples.append({'task':Path(child.dataset.root).name,'frame':idx,'finite':True,'device':'cpu'})
        frames+=len(child)
        print('DATA READ',Path(child.dataset.root).name,len(child),'frames',flush=True)
    batch=VLADataCollatorWithPacking()([ds._datasets[0][0],ds._datasets[1][0]])
    assert batch['actions'].shape==(2,50,55) and batch['state'].shape==(2,55)
    assert frames==len(ds)
    os.environ.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
    return {'tasks':50,'episodes':2500,'frames':frames,'samples':samples,
        'batch_state_shape':list(batch['state'].shape),'batch_action_shape':list(batch['actions'].shape)}

def provenance():
    base = hashlib.sha256(Path('/root/miniconda3/conda-meta/history').read_bytes()).hexdigest()
    assert base == 'df562164c3983d83b08776c530349731099dca0ce2e5213cda63c5935a1ffb52'
    return {'base_conda_history_sha256': base, 'versions': {n: importlib.metadata.version(n)
        for n in ['torch','torchvision','lerobot','flash_attn','transformers','numpy','scipy']}}

check('six_gpu_argument_validation', arguments)
check('input_manifest_and_availability', input_availability)
check('real_dataset_time_alignment', time_alignment)
check('official_loader_and_collator_cpu', dataset_read)
if (OUT/'norm-clean50.json').exists():
    check('full_50_task_loader_and_collator_cpu', full_dataset_read)
else:
    report['pending'].append({'kind':'full_50_task_loader_check','reason':'Awaiting clean50 data and normalization'})
check('base_and_training_environment', provenance)
report['cuda_initialized'] = torch.cuda.is_initialized()
assert not report['cuda_initialized']
report['ready_to_train'] = not report['errors'] and not report['pending']
report['training_runtime_checks_deferred']=['NCCL/FSDP runtime','first model forward and backward','optimizer update','peak training VRAM','training throughput']
report['status'] = 'ready; training not started' if report['ready_to_train'] else 'partial preparation; see pending inputs and checks'
(OUT / 'pretrain-report.json').write_text(json.dumps(report, indent=2))
print(report['status'], 'pending=', len(report['pending']), 'errors=', report['errors'])
