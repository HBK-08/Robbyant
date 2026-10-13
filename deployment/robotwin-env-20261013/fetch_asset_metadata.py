"""Fetch two genuine JSON members via HTTP ranges, never download objects.zip."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import requests

out = Path(__file__).parent
repo = Path('/home/lmy/Robbyant/RoboTwin')
dataset = 'TianxingChen/RoboTwin2.0'
session = requests.Session()
info_response = session.get('https://huggingface.co/api/datasets/' + dataset, timeout=40)
info_response.raise_for_status()
revision = info_response.json()['sha']
tree_response = session.get(f'https://huggingface.co/api/datasets/{dataset}/tree/{revision}?limit=100', timeout=40)
tree_response.raise_for_status()
item = next(i for i in tree_response.json() if i['path'] == 'objects.zip')
url = f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/objects.zip'

class RemoteZip(io.RawIOBase):
    def __init__(self, size):
        self.size = size
        self.position = 0
        self.downloaded = 0
        self.ranges = []
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position + offset if whence == 1 else self.size + offset)
        if self.position < 0:
            raise ValueError('Negative file position')
        return self.position
    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        count = min(count, self.size - self.position)
        if count <= 0:
            return b''
        if count > 32 * 1024 * 1024 or self.downloaded + count > 32 * 1024 * 1024:
            raise RuntimeError('Refusing a large transfer; metadata-only budget is 32 MiB')
        start, end = self.position, self.position + count - 1
        response = session.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                               stream=True, timeout=60)
        try:
            if response.status_code != 206:
                raise RuntimeError(f'Range request not honored (HTTP {response.status_code}); full download refused')
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if not match or tuple(map(int, match.groups())) != (start, end, self.size):
                raise RuntimeError('Unexpected HTTP range; refusing to consume response body')
            data = response.raw.read(count + 1, decode_content=False)
            if len(data) != count:
                raise RuntimeError('Range response length mismatch')
        finally:
            response.close()
        self.downloaded += count
        self.position += count
        self.ranges.append({'start': start, 'end': end, 'bytes': count})
        print('Fetched ZIP range:', start, end, count, flush=True)
        return data

remote = RemoteZip(item['size'])
saved = []
with zipfile.ZipFile(remote) as archive:
    names = archive.namelist()
    list_candidates = [name for name in names if name.endswith('objaverse/list.json')]
    same_candidates = [name for name in names if name in ['objects/same.json', 'same.json']]
    assert len(list_candidates) == 1 and len(same_candidates) == 1, (list_candidates, same_candidates)
    for member, target in [(list_candidates[0], repo / 'assets/objects/objaverse/list.json'),
                           (same_candidates[0], repo / 'assets/objects/same.json')]:
        assert archive.getinfo(member).file_size < 4 * 1024 * 1024
        data = archive.read(member)  # zipfile checks the member CRC.
        document = json.loads(data)
        if member.endswith('objaverse/list.json'):
            assert all(key in document for key in ['item_names', 'list_of_items', 'z_max', 'radius', 'z_offset'])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        saved.append({'zip_member': member, 'target': str(target), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()})
        print('Saved official asset metadata:', member, len(data), 'bytes', flush=True)
report = {'dataset': dataset, 'revision': revision, 'canonical_source_url': url,
          'full_zip_bytes': item['size'], 'zip_lfs_sha256': item.get('lfs', {}).get('oid'),
          'downloaded_bytes': remote.downloaded, 'full_zip_downloaded': False,
          'http_ranges': remote.ranges, 'metadata_files': saved}
(out / 'asset-metadata-download.json').write_text(json.dumps(report, indent=2))
print('Metadata-only bytes transferred:', remote.downloaded, 'of full ZIP', item['size'], flush=True)
