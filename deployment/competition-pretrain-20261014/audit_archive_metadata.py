"""Read only remote ZIP central directories to budget clean50 disk space."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import zipfile
import requests

OUT = Path(__file__).resolve().parent

class RemoteZip(io.RawIOBase):
    def __init__(self, job):
        self.size = job['bytes']
        self.url = f"https://huggingface.co/datasets/{job['repo']}/resolve/{job['revision']}/{job['path']}"
        self.position = 0
        self.transferred = 0
        self.session = requests.Session()
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, n=-1):
        n = min(n if n >= 0 else self.size-self.position, self.size-self.position)
        if n <= 0: return b''
        if self.transferred+n > 2*1024**2: raise RuntimeError('ZIP metadata byte limit exceeded')
        start, end = self.position, self.position+n-1
        r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
            stream=True, timeout=60)
        try:
            assert r.status_code == 206 and r.headers.get('Content-Range') == f'bytes {start}-{end}/{self.size}'
            data = r.raw.read(n+1, decode_content=False)
            assert len(data) == n
        finally: r.close()
        self.position += n
        self.transferred += n
        return data

def audit(job):
    remote = RemoteZip(job)
    with zipfile.ZipFile(remote) as archive:
        files = archive.infolist()
        episodes = [f.filename for f in files if re.search(r'/data/episode\d+\.hdf5$',f.filename)]
        instructions = [f.filename for f in files if re.search(r'/instructions/episode\d+\.json$',f.filename)]
        assert len(episodes) == len(instructions) == 50
        raw_ids = sorted(int(re.search(r'episode(\d+)\.hdf5$',f).group(1)) for f in episodes)
        assert raw_ids == list(range(50))
        assert all('randomized' not in f.filename for f in files)
        result = {'archive': job['path'], 'archive_bytes': job['bytes'],
            'uncompressed_bytes': sum(f.file_size for f in files), 'raw_episodes': 50,
            'instruction_files': 50, 'zip_directory_transfer_bytes': remote.transferred}
    print('AUDITED',job['path'],result['uncompressed_bytes'],flush=True)
    return result

if __name__ == '__main__':
    manifest = json.loads((OUT/'input-manifest.json').read_text())
    jobs = [j for j in manifest['files'] if j['kind']=='dataset']
    with ThreadPoolExecutor(max_workers=3) as pool: records=list(pool.map(audit,jobs))
    report={'archives':records,'tasks':len(records),'raw_episodes':sum(r['raw_episodes'] for r in records),
        'compressed_bytes':sum(r['archive_bytes'] for r in records),
        'uncompressed_bytes':sum(r['uncompressed_bytes'] for r in records),
        'metadata_transfer_bytes':sum(r['zip_directory_transfer_bytes'] for r in records),
        'trajectory_payload_downloaded_by_this_script':False}
    (OUT/'archive-metadata-audit.json').write_text(json.dumps(report,indent=2))
    print('TOTAL',report['compressed_bytes'],report['uncompressed_bytes'],report['metadata_transfer_bytes'])
