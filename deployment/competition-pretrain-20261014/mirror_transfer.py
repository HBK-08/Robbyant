"""Transfer only publisher mirror files whose size and SHA256 match the pinned HF source."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import threading
import time
import requests

def segmented_file(url,target,partial,size):
    block_size=64*1024**2
    folder=target.parent/(target.name+'.segments')
    folder.mkdir(exist_ok=True)
    local=threading.local()
    def fetch(start):
        if not hasattr(local,'session'):
            local.session=requests.Session(); local.session.trust_env=False
        end=min(start+block_size,size)
        path=folder/str(start)
        failures=0
        while True:
            have=path.stat().st_size if path.exists() else 0
            if have==end-start:return path
            assert have<end-start
            request_end=min(end,start+have+32*1024**2)
            try:
                r=local.session.get(url,headers={'Range':f'bytes={start+have}-{request_end-1}',
                    'Accept-Encoding':'identity'},stream=True,timeout=(30,120))
                try:
                    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start+have}-{request_end-1}/{size}'
                    with path.open('ab') as out:
                        for chunk in r.iter_content(512*1024):
                            assert out.tell()+len(chunk)<=request_end-start
                            out.write(chunk)
                    assert path.stat().st_size==request_end-start
                finally:r.close()
                failures=0
            except (requests.RequestException,AssertionError) as e:
                print('MIRROR RETRY',target.name,start,type(e).__name__,flush=True)
                failures+=1
                if failures>7:raise
                time.sleep(min(2**failures,15))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths=list(pool.map(fetch,range(0,size,block_size)))
    assembled=target.with_name(target.name+'.assembled')
    with assembled.open('wb') as dst:
        for p in paths:
            with p.open('rb') as src:shutil.copyfileobj(src,dst,8*1024**2)
    return assembled,paths
