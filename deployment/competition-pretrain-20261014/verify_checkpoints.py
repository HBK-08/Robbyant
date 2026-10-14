"""Check official checkpoint structure and load teachers on CPU; no forward or training."""
import gc
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import torch
import yaml

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not torch.cuda.is_initialized()

def tensors(repo):
    root=ROOT/'models'/repo
    index=json.loads((root/'model.safetensors.index.json').read_text())
    result={}
    dtype_bytes={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
    total=0
    for name in sorted(set(index['weight_map'].values())):
        path=root/name
        with path.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]
            assert n<32*1024**2
            header=json.loads(f.read(n))
        for key, value in header.items():
            if key=='__metadata__': continue
            assert key not in result and index['weight_map'][key]==name
            size=math.prod(value['shape'])*dtype_bytes[value['dtype']]
            begin,end=value['data_offsets']
            assert end-begin==size and end<=path.stat().st_size-8-n
            total+=size
            result[key]={'shape':value['shape'],'dtype':value['dtype']}
    assert set(result)==set(index['weight_map'])
    assert total==index['metadata']['total_size']
    output={'shards':len(set(index['weight_map'].values())),'tensors':len(result),'tensor_bytes':total,
        'dtypes':sorted(set(v['dtype'] for v in result.values()))}
    if repo=='lingbot-vla-v2-6b':
        shapes={key:result[key]['shape'] for key in ['model.state_proj.weight','model.action_in_proj.weight','model.action_out_proj.weight']}
        assert shapes['model.state_proj.weight'][1]==55
        assert shapes['model.action_in_proj.weight'][1]==55
        assert shapes['model.action_out_proj.weight'][0]==55
        output['state_action_projection_shapes']=shapes
    return output

def teacher(name,load):
    sources={'MoGe':ROOT/'models/moge-2-vitb-normal/model.pt',
        'LingBot-Depth':ROOT/'models/lingbot-vla-v2-6b/depth/model.pt',
        'DINO-VIDEO':ROOT/'models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth'}
    source=sources[name]
    manifest=json.loads((OUT/'input-manifest.json').read_text())
    job=next(j for j in manifest['files'] if Path(j['target'])==source)
    record_path=OUT/'download-records'/(hashlib.sha256(str(source).encode()).hexdigest()+'.json')
    proof=json.loads(record_path.read_text())
    assert proof['verified'] and proof['actual_sha256']==job['sha256']
    cache=OUT/f'teacher-{name}-verification.json'
    if cache.exists():
        cached=json.loads(cache.read_text())
        if cached['source_sha256']==job['sha256'] and cached['source_bytes']==source.stat().st_size:
            print('REUSED CPU TEACHER CHECK',name,flush=True)
            return cached
    start=time.time()
    model=load()
    model.eval().requires_grad_(False)
    assert all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
    result={'class':type(model).__name__,'parameters':sum(p.numel() for p in model.parameters()),
        'load_seconds':round(time.time()-start,2),'device':'cpu','forward_executed':False,
        'source_sha256':job['sha256'],'source_bytes':source.stat().st_size}
    del model
    gc.collect()
    print('LOADED',name,result,flush=True)
    cache.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher-only',choices=['MoGe','LingBot-Depth','DINO-VIDEO'])
    args=parser.parse_args()
    report={'training_started':False,'forward_executed':False,'backward_executed':False,'checkpoints':{},'teachers':{}}
    for name in ([] if args.teacher_only else ['lingbot-vla-v2-6b','Qwen3-VL-4B-Instruct']):
        report['checkpoints'][name]=tensors(name)
        print('CHECKPOINT INDEX VERIFIED',name,report['checkpoints'][name],flush=True)
    from lingbotvla.models.vla.vision_models.module_utils import v2,v2_morgbd
    if args.teacher_only in [None,'MoGe']:
        report['teachers']['MoGe']=teacher('MoGe',lambda:v2.from_pretrained(ROOT/'models/moge-2-vitb-normal/model.pt'))
    if args.teacher_only in [None,'LingBot-Depth']:
        report['teachers']['LingBot-Depth']=teacher('LingBot-Depth',lambda:v2_morgbd.from_pretrained(ROOT/'models/lingbot-vla-v2-6b/depth/model.pt'))
    from lingbotvla.models.vla.vision_models.dino_video.teacher import build_dino_video_teacher
    cfg=yaml.safe_load((OUT/'six-gpu-clean50.yaml').read_text())['train']['align_params']['video']
    cfg['device']='cpu'
    if args.teacher_only in [None,'DINO-VIDEO']:
        report['teachers']['DINO-VIDEO']=teacher('DINO-VIDEO',lambda:build_dino_video_teacher(cfg))
    report['cuda_initialized']=torch.cuda.is_initialized()
    assert not report['cuda_initialized']
    report['passed']=True
    if not args.teacher_only:
        (OUT/'checkpoint-verification.json').write_text(json.dumps(report,indent=2))
    print('Checkpoint and CPU teacher checks passed; no forward or training executed',flush=True)
