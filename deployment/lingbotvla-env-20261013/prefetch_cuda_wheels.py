"""Download only the exact CUDA dependencies of the official PyTorch 2.8 wheel."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import urllib.request

root = Path(__file__).parent / "wheels"
root.mkdir(exist_ok=True)

def metadata(name, version):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=45) as response:
        return json.load(response)

torch_metadata = metadata("torch", "2.8.0")
packages = []
for requirement in torch_metadata["info"]["requires_dist"]:
    match = re.match(r"(nvidia-[a-z0-9-]+|triton)==([0-9.]+)", requirement)
    if match:
        packages.append(match.groups())

def fetch(package):
    name, version = package
    data = metadata(name, version)
    assets = [asset for asset in data["urls"] if asset["filename"].endswith(".whl")
              and "x86_64" in asset["filename"] and "manylinux" in asset["filename"]
              and ("-py3-" in asset["filename"] or "-cp312-" in asset["filename"])]
    if len(assets) != 1:
        raise RuntimeError((name, [a["filename"] for a in assets]))
    asset = assets[0]
    destination = root / asset["filename"]
    expected = asset["digests"]["sha256"]
    if destination.exists() and hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() == expected:
        return f"CACHED {destination.name}"
    temporary = destination.with_suffix(".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(asset["url"], timeout=90) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"SHA256 mismatch: {destination}")
    temporary.replace(destination)
    return f"VERIFIED {destination.name} {asset['size']} bytes sha256={expected}"

print(f"Prefetching {len(packages)} exact-version wheels, 4 downloads at a time", flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(fetch, package) for package in packages]):
        print(future.result(), flush=True)
