"""Download revision-pinned public training inputs; never invoke training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import zipfile
import requests
import urllib.parse
import mirror_transfer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
spec = importlib.util.spec_from_file_location('baseline_transfer', ROOT / 'deployment/robotwin-baseline-20261014/download_inputs.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
USE_PUBLISHER_MIRROR = False

def publisher_mirror(job):
    if not USE_PUBLISHER_MIRROR or not job['sha256']:return None
    repos={'robbyant/lingbot-vla-v2-6b':'Robbyant/lingbot-vla-v2-6b',
        'Qwen/Qwen3-VL-4B-Instruct':'Qwen/Qwen3-VL-4B-Instruct'}
    if job['repo'] not in repos:return None
    metadata=json.loads((OUT/(job['repo'].split('/')[-1]+'-modelscope-metadata.json')).read_text())
    f=next((f for f in metadata['Data']['Files'] if f['Path']==job['path']),None)
    if not f or f['Sha256']!=job['sha256'] or f['Size']!=job['bytes']:return None
    repo=repos[job['repo']]
    params=urllib.parse.urlencode({'Revision':f['Revision'],'FilePath':job['path']})
    return {'repo':repo,'revision':f['Revision'],'url':f'https://modelscope.cn/api/v1/models/{repo}/repo?{params}'}

def manifest():
    metadata = json.loads((OUT / 'input-metadata.json').read_text())
    jobs = []
    for model in metadata['models']:
        for f in model['files']:
            if f['rfilename'] in ['README.md', '.gitattributes'] or f['rfilename'].startswith('assets/'):
                continue
            jobs.append({'kind': 'model', 'repo': model['repo'], 'revision': model['revision'],
                'path': f['rfilename'], 'bytes': f['size'], 'sha256': f.get('lfs', {}).get('sha256'),
                'git_blob_sha1': f.get('blobId') if not f.get('lfs') else None,
                'target': str(ROOT / 'models' / model['repo'].split('/')[-1] / f['rfilename'])})
    tree = json.loads((OUT / 'dataset-tree.json').read_text())
    clean = [f for f in tree if f['type'] == 'file' and f['path'].endswith('/aloha-agilex_clean_50.zip')]
    launcher = (ROOT / 'lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh').read_text()
    tasks = re.findall(r'"([a-z0-9_]+)"', re.search(r'task_list_all=\(([^\n]+)\)', launcher).group(1))
    assert len(clean) == 50 and set(tasks) == {f['path'].split('/')[1] for f in clean}
    for f in sorted(clean, key=lambda f: (f['path'].split('/')[1] != 'lift_pot', f['path'])):
        jobs.append({'kind': 'dataset', 'repo': 'TianxingChen/RoboTwin2.0',
            'revision': metadata['dataset_revision'], 'path': f['path'], 'bytes': f['size'],
            'sha256': f['lfs']['oid'], 'git_blob_sha1': None,
            'target': str(ROOT / 'datasets/competition-clean50/archives' / f['path'].split('/')[1] / 'aloha-agilex_clean_50.zip')})
    doc = {'training_started': False, 'forbidden_gpu_indices': [6, 7], 'tasks': tasks,
        'files': jobs, 'total_transfer_bytes': sum(j['bytes'] for j in jobs),
        'clean_archive_bytes': sum(j['bytes'] for j in jobs if j['kind'] == 'dataset'),
        'disk_free_bytes': shutil.disk_usage(ROOT).free}
    (OUT / 'input-manifest.json').write_text(json.dumps(doc, indent=2))
    return doc

def fetch(job):
    target = Path(job['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'datasets/' if job['kind'] == 'dataset' else ''
    url = f"https://huggingface.co/{prefix}{job['repo']}/resolve/{job['revision']}/{job['path']}"
    mirror=publisher_mirror(job)
    if mirror:url=mirror['url']
    if not target.exists():
        if job['bytes'] > 64 * 1024 * 1024:
            transport=mirror_transfer.segmented_file if mirror else transfer.segmented_file
            assembled, segments = transport(url, target, target.with_name(target.name + '.partial'), job['bytes'])
            if transfer.digest(assembled) != job['sha256']:
                raise RuntimeError(f'Checksum failed: {target}')
            assembled.replace(target)
            for p in segments:
                p.unlink()
            segments[0].parent.rmdir()
        else:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            content = response.content
            if len(content) != job['bytes']:
                raise RuntimeError(f'Size mismatch: {target}')
            if job['git_blob_sha1']:
                assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == job['git_blob_sha1']
            partial = target.with_name(target.name + '.partial')
            partial.write_bytes(content)
            partial.replace(target)
    actual = transfer.digest(target)
    if target.stat().st_size != job['bytes'] or (job['sha256'] and actual != job['sha256']):
        raise RuntimeError(f'Existing file incompatible: {target}')
    if job['git_blob_sha1']:
        content = target.read_bytes()
        assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == job['git_blob_sha1']
    record = {**job, 'actual_sha256': actual, 'verified': True}
    if mirror:record['publisher_mirror']={'repo':mirror['repo'],'revision':mirror['revision'],'sha256_matched':True}
    print('VERIFIED', job['kind'], job['path'], job['bytes'], flush=True)
    if job['kind'] == 'dataset':
        task = job['path'].split('/')[1]
        raw = ROOT / 'datasets/competition-clean50/raw' / task
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            for entry in archive.infolist():
                dest = (raw / entry.filename).resolve()
                if not dest.is_relative_to(raw.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Unsafe archive member')
            required = sum(f.file_size for f in archive.infolist())
            if shutil.disk_usage(raw).free < required + 30 * 1024**3:
                raise RuntimeError('Insufficient free disk for extraction')
            if not (raw / '.extraction-complete').exists():
                archive.extractall(raw)
                (raw / '.extraction-complete').write_text(actual)
            else:
                assert (raw / '.extraction-complete').read_text() == actual
            record['uncompressed_bytes'] = required
            record['raw_root'] = str(raw)
    records_dir = OUT / 'download-records'
    records_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(str(target).encode()).hexdigest()
    (records_dir / (key + '.json')).write_text(json.dumps(record, indent=2))
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['manifest', 'models', 'single-task', 'all-clean'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--publisher-mirror', action='store_true')
    args = parser.parse_args()
    USE_PUBLISHER_MIRROR=args.publisher_mirror
    assert 1 <= args.workers <= 6
    doc = manifest()
    print('TOTAL BYTES', doc['total_transfer_bytes'], 'CLEAN BYTES', doc['clean_archive_bytes'], flush=True)
    jobs = doc['files']
    if args.mode == 'manifest':
        raise SystemExit(0)
    if args.mode == 'models':
        jobs = sorted([j for j in jobs if j['kind'] == 'model'], key=lambda j: j['bytes'])
    else:
        jobs = [j for j in jobs if j['kind'] == 'dataset' and (args.mode == 'all-clean' or '/lift_pot/' in j['path'])]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(fetch, jobs))
    (OUT / (args.mode + '-verification.json')).write_text(json.dumps(records, indent=2))
