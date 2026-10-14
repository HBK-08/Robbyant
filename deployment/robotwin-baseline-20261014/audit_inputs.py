"""Verify small files against official Git blob hashes; verify LFS records against manifest."""
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parent.parent
manifest = json.loads((out / 'download-manifest.json').read_text())
download_records = {}
if (out / 'model-download-verification.json').exists():
    download_records = {r['file']: r for r in json.loads((out / 'model-download-verification.json').read_text())}
results = []
for model in manifest['models']:
    for item in model['files']:
        target = root / 'models' / model['repo'].split('/')[-1] / item['rfilename']
        result = {'file': str(target), 'exists': target.exists()}
        if target.exists():
            result['size_matches'] = target.stat().st_size == item['size']
            if item.get('lfs'):
                record = download_records.get(str(target))
                result['hash_matches'] = bool(record and record['sha256'] == item['lfs']['sha256'])
                result['hash_method'] = 'SHA256 record produced after actual full-file hashing by downloader'
            else:
                data = target.read_bytes()
                blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                result['hash_matches'] = blob == item['blobId']
                result['hash_method'] = 'Git blob SHA1 against official metadata'
                result['actual_blob_id'] = blob
        results.append(result)
complete = all(r.get('size_matches') and r.get('hash_matches') for r in results)
report = {'complete': complete, 'matched_files': sum(bool(r.get('hash_matches')) for r in results),
          'expected_files': len(results), 'files': results}
(out / 'input-checksum-audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'files'}))
