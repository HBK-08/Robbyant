"""Download only approved, revision-pinned inputs; retain checksums and resume files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import threading
import zipfile
import zlib
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
MANIFEST = json.loads((OUT / 'download-manifest.json').read_text())
TRANSPORT = threading.local()


def segmented_file(url, target, partial, size):
    segment_size = 64 * 1024 * 1024
    segment_dir = target.parent / (target.name + '.segments')
    segment_dir.mkdir(exist_ok=True)
    prefix_size = partial.stat().st_size if partial.exists() else 0

    def fetch_segment(start):
        if not hasattr(TRANSPORT, 'session'):
            TRANSPORT.session = requests.Session()
        session = TRANSPORT.session
        end = min(start + segment_size, size)
        path = segment_dir / str(start)
        if not path.exists() and start < prefix_size:
            with partial.open('rb') as source, path.open('wb') as output:
                source.seek(start)
                output.write(source.read(min(end, prefix_size) - start))
        attempt = 0
        while True:
            have = path.stat().st_size if path.exists() else 0
            if have == end - start:
                return path
            if have > end - start:
                raise RuntimeError('Oversized segment')
            try:
                request_end = min(end, start + have + 8 * 1024 * 1024)
                response = session.get(url, headers={'Range': f'bytes={start + have}-{request_end - 1}',
                    'Accept-Encoding': 'identity'}, stream=True, timeout=(40, 120))
                try:
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start + have}-{request_end - 1}/{size}':
                        raise RuntimeError('Segment range not honored')
                    with path.open('ab') as output:
                        for chunk in response.iter_content(512 * 1024):
                            if output.tell() + len(chunk) > request_end - start:
                                raise RuntimeError('Oversized segment response')
                            output.write(chunk)
                    if path.stat().st_size != request_end - start:
                        raise RuntimeError('Incomplete segment')
                    attempt = 0
                    if request_end == end:
                        print(f'SEGMENT {target.name} {start}-{end}', flush=True)
                        return path
                finally:
                    response.close()
            except (requests.RequestException, RuntimeError) as error:
                print(f'SEGMENT RETRY {target.name} {start} {error}', flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 20))
                attempt += 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(fetch_segment, range(0, size, segment_size)))
    assembled = target.with_name(target.name + '.assembled')
    with assembled.open('wb') as output:
        for path in paths:
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    output.write(chunk)
    return assembled, paths


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_file(job):
    model, item = job
    target = ROOT / 'models' / model['repo'].split('/')[-1] / item['rfilename']
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('lfs', {}).get('sha256')
    if target.exists():
        actual = digest(target)
        if target.stat().st_size != item['size'] or (expected and actual != expected):
            raise RuntimeError(f'Existing file failed verification: {target}')
        return {'file': str(target), 'bytes': item['size'], 'sha256': actual, 'cached': True}
    partial = target.with_name(target.name + '.partial')
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{item['rfilename']}"
    if item['size'] > 64 * 1024 * 1024:
        assembled, segments = segmented_file(url, target, partial, item['size'])
        actual = digest(assembled)
        if expected != actual:
            raise RuntimeError(f'SHA256 mismatch: {assembled}')
        assembled.rename(target)
        for segment in segments:
            segment.unlink()
        segments[0].parent.rmdir()
        partial.unlink(missing_ok=True)
        print(f'VERIFIED {target.name} sha256={actual}', flush=True)
        return {'file': str(target), 'bytes': item['size'], 'sha256': actual, 'cached': False}
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item['size']:
            break
        if offset > item['size']:
            raise RuntimeError(f'Oversized partial: {partial}')
        try:
            response = requests.get(url, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'},
                                    stream=True, timeout=(40, 120))
            try:
                response.raise_for_status()
                if offset and (response.status_code != 206 or
                               not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise RuntimeError('Server did not honor resume range')
                print(f'DOWNLOAD {target.name} offset={offset} size={item["size"]}', flush=True)
                written, last_report = offset, time.monotonic()
                with partial.open('ab') as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if written + len(chunk) > item['size']:
                            raise RuntimeError('Response exceeds manifest size')
                        stream.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report > 30:
                            print(f'PROGRESS {target.name} {written}/{item["size"]}', flush=True)
                            last_report = time.monotonic()
                if written != item['size']:
                    raise RuntimeError('Incomplete response')
            finally:
                response.close()
            break
        except (requests.RequestException, RuntimeError) as error:
            print(f'RETRY {target.name} attempt={attempt + 1}: {error}', flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 20))
    actual = digest(partial)
    if expected and expected != actual:
        raise RuntimeError(f'SHA256 mismatch: {partial}')
    partial.rename(target)
    print(f'VERIFIED {target.name} sha256={actual}', flush=True)
    return {'file': str(target), 'bytes': item['size'], 'sha256': actual, 'cached': False}


def download_models():
    jobs = sorted([(model, item) for model in MANIFEST['models'] for item in model['files']],
                  key=lambda job: job[1]['size'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(model_file, jobs))
    (OUT / 'model-download-verification.json').write_text(json.dumps(records, indent=2))


class RemoteZip(io.RawIOBase):
    def __init__(self, spec):
        self.spec, self.position, self.downloaded, self.blocks = spec, 0, 0, []
        self.session = requests.Session()
        self.cache = OUT / 'asset-range-cache' / spec['revision'] / spec['archive']
        self.cache.mkdir(parents=True, exist_ok=True)
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.spec['full_archive_bytes'] + offset
        return self.position
    def fetch(self, start, count):
        for block_start, data in self.blocks:
            if block_start <= start and start + count <= block_start + len(data):
                return data[start - block_start:start - block_start + count]
        cursor, chunks = start, []
        for block_start, data in sorted(self.blocks, key=lambda item: item[0]):
            if block_start <= cursor < block_start + len(data):
                end = min(start + count, block_start + len(data))
                chunks.append(data[cursor - block_start:end - block_start])
                cursor = end
                if cursor == start + count:
                    return b''.join(chunks)
        if self.downloaded + count > self.spec['selected_compressed_bytes'] + 16 * 1024 * 1024:
            raise RuntimeError('Asset transfer exceeds selected-content budget')
        cached = self.cache / f'{start}-{count}'
        if cached.exists() and cached.stat().st_size == count:
            data = cached.read_bytes()
            self.blocks.append((start, data))
            return data
        end = start + count - 1
        for attempt in range(6):
            try:
                response = self.session.get(self.spec['url'],
                    headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                    stream=True, timeout=(40, 120))
                try:
                    expected = f"bytes {start}-{end}/{self.spec['full_archive_bytes']}"
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('Range not honored; refusing full archive')
                    data = response.raw.read(count + 1, decode_content=False)
                    if len(data) != count:
                        raise RuntimeError('Incomplete range')
                finally:
                    response.close()
                self.blocks.append((start, data))
                cached.write_bytes(data)
                self.downloaded += count
                print(f'ASSET RANGE {self.spec["archive"]} {start}-{end} bytes={count}', flush=True)
                return data
            except (requests.RequestException, RuntimeError):
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))
    def read(self, count=-1):
        count = min(count if count >= 0 else self.spec['full_archive_bytes'] - self.position,
                    self.spec['full_archive_bytes'] - self.position)
        if count <= 0:
            return b''
        data = self.fetch(self.position, count)
        self.position += count
        return data


def download_assets():
    records = []
    for spec in MANIFEST['assets']:
        remote = RemoteZip(spec)
        with zipfile.ZipFile(remote) as archive:
            all_entries = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
            selected = {f['path'] for f in spec['files']}
            spans = []
            for index, entry in enumerate(all_entries):
                if entry.filename not in selected:
                    continue
                end = all_entries[index + 1].header_offset if index + 1 < len(all_entries) else archive.start_dir
                if spans and spans[-1][1] == entry.header_offset:
                    spans[-1][1] = end
                else:
                    spans.append([entry.header_offset, end])
            for start, end in spans:
                while start < end:
                    count = min(8 * 1024 * 1024, end - start)
                    remote.fetch(start, count)
                    start += count
            verified = []
            for member in spec['files']:
                path = Path(member['path'])
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError('Unsafe ZIP member')
                target = ROOT / 'RoboTwin/assets' / path
                data = archive.read(member['path'])
                assert len(data) == member['uncompressed_bytes']
                assert f'{zlib.crc32(data):08x}' == member['crc32']
                if target.exists() and target.read_bytes() != data:
                    raise RuntimeError(f'Refusing to replace differing existing asset: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                verified.append({'file': str(target), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        records.append({'archive': spec['archive'], 'revision': spec['revision'],
                        'downloaded_bytes': remote.downloaded, 'full_archive_downloaded': False,
                        'files': verified})
        (OUT / 'asset-download-verification.json').write_text(json.dumps(records, indent=2))
        print('VERIFIED ASSETS', spec['archive'], len(verified), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['models', 'assets'])
    args = parser.parse_args()
    download_models() if args.mode == 'models' else download_assets()
