"""Read public metadata only; no training or weight downloads."""
import json
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor

OUT = Path(__file__).resolve().parent
SESSION = requests.Session()

def read_model(repo):
    r = requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60)
    r.raise_for_status()
    data = r.json()
    (OUT / (repo.split('/')[-1] + '-metadata.json')).write_text(json.dumps(data, indent=2))
    return {'repo': repo, 'revision': data['sha'], 'files': data['siblings']}

def tree(path):
    url = 'https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/' + path
    result = []
    while url:
        r = SESSION.get(url, params={'recursive': 'true', 'limit': 1000}, timeout=60)
        r.raise_for_status()
        result += r.json()
        url = r.links.get('next', {}).get('url')
    return result

if __name__ == '__main__':
    repos = ['robbyant/lingbot-vla-v2-6b', 'Qwen/Qwen3-VL-4B-Instruct', 'Ruicheng/moge-2-vitb-normal']
    with ThreadPoolExecutor(max_workers=3) as pool:
        models = list(pool.map(read_model, repos))
    data = tree('dataset')
    (OUT / 'dataset-tree.json').write_text(json.dumps(data, indent=2))
    (OUT / 'input-metadata.json').write_text(json.dumps({'models': models, 'dataset_revision': '3dc3b798668feb99ac61cc9086d84cbcc3d79186'}, indent=2))
    for m in models:
        print(m['repo'], m['revision'])
        for f in m['files']:
            print(f['rfilename'], f.get('size', 0))
    print('DATASET')
    for f in data:
        if f['type'] == 'file' and ('clean' in f['path'] or 'lift_pot' in f['path']):
            print(f['path'], f.get('size', 0))
