"""Cache already-resolved large official dependency wheels; never install packages."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

out = Path(__file__).parent
env = dict(os.environ, PIP_INDEX_URL='https://pypi.org/simple',
           PIP_CACHE_DIR=str(out / 'pip-cache'), PYTHONNOUSERSITE='1',
           TMPDIR=str(out / 'prefetch-tmp'), PIP_NO_INPUT='1')
(out / 'prefetch-tmp').mkdir(exist_ok=True)
(out / 'wheels').mkdir(exist_ok=True)
packages = ['open3d==0.18.0', 'nvidia-cudnn-cu12==9.1.0.70',
            'nvidia-cublas-cu12==12.1.3.1', 'triton==3.0.0',
            'nvidia-cufft-cu12==11.0.2.54', 'nvidia-cusolver-cu12==11.4.5.107',
            'nvidia-cusparse-cu12==12.1.0.106', 'nvidia-nccl-cu12==2.20.5']

def download(package):
    log = out / ('prefetch-' + package.split('==')[0] + '.log')
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                                 '--dest', str(out / 'wheels'), package], env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    print(package, 'cached' if result.returncode == 0 else 'failed; see ' + str(log), flush=True)
    return result.returncode

with ThreadPoolExecutor(max_workers=2) as pool:
    codes = list(pool.map(download, packages))
sys.exit(0 if all(c == 0 for c in codes) else 1)
