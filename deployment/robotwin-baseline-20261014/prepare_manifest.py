"""Read public metadata and ZIP directories; never download model weights/assets."""
import io
import json
from pathlib import Path
import re
import shutil
import zipfile
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
session = requests.Session()


def get_json(url):
    response = session.get(url, timeout=40)
    response.raise_for_status()
    return response.json()


models = []
for repo in ["robbyant/lingbot-vla-v2-6b-robotwin", "Qwen/Qwen3-VL-4B-Instruct"]:
    metadata = get_json("https://huggingface.co/api/models/" + repo + "?blobs=true")
    (OUT / (repo.split("/")[-1] + "-metadata.json")).write_text(json.dumps(metadata, indent=2))
    files = metadata["siblings"]
    if repo.startswith("robbyant/"):
        selected = [f for f in files if f["rfilename"].startswith("checkpoints/global_step_50000/hf_ckpt/")
                    or f["rfilename"] == "lingbotvla_cli.yaml"]
        purpose = "RoboTwin post-training checkpoint; all six weight shards plus official config/tokenizer"
    else:
        selected = [f for f in files if f["rfilename"] not in [".gitattributes", "README.md"]
                    and not f["rfilename"].endswith(".safetensors")
                    and f["rfilename"] != "model.safetensors.index.json"]
        purpose = "Config and processor only; inference constructs Qwen from config and loads VLA weights"
    models.append({"repo": repo, "revision": metadata["sha"], "purpose": purpose,
                   "selected_bytes": sum(f["size"] for f in selected), "files": selected})
    print(repo, metadata["sha"], models[-1]["selected_bytes"], flush=True)

previous = json.loads((ROOT / "deployment/robotwin-env-20261013/asset-metadata-download.json").read_text())
asset_tree = json.loads((ROOT / "deployment/robotwin-env-20261013/hf-asset-tree.json").read_text())


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position, self.downloaded = url, size, 0, 0
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("negative offset")
        return self.position
    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size - self.position)
        if count <= 0:
            return b""
        if self.downloaded + count > 8 * 1024 * 1024:
            raise RuntimeError("metadata transfer limit exceeded")
        start, end = self.position, self.position + count - 1
        response = session.get(self.url, headers={"Range": f"bytes={start}-{end}",
                               "Accept-Encoding": "identity"}, stream=True, timeout=45)
        try:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match or tuple(map(int, match.groups())) != (start, end, self.size):
                raise RuntimeError("HTTP range not honored; full download refused")
            data = response.raw.read(count + 1, decode_content=False)
            if len(data) != count:
                raise RuntimeError("range length mismatch")
        finally:
            response.close()
        self.position += count
        self.downloaded += count
        return data


assets = []
for archive_name, prefix in [("embodiments.zip", "embodiments/aloha-agilex/"),
                             ("objects.zip", "objects/060_kitchenpot/")]:
    metadata = next(f for f in asset_tree if f["path"] == archive_name)
    url = f"https://huggingface.co/datasets/{previous['dataset']}/resolve/{previous['revision']}/{archive_name}"
    remote = RemoteZip(url, metadata["size"])
    with zipfile.ZipFile(remote) as archive:
        entries = [{"path": f.filename, "compressed_bytes": f.compress_size,
                    "uncompressed_bytes": f.file_size, "crc32": f"{f.CRC:08x}"}
                   for f in archive.infolist() if f.filename.startswith(prefix) and not f.is_dir()]
        if not entries:
            raise RuntimeError(f"no members under {prefix}")
    assets.append({"archive": archive_name, "revision": previous["revision"],
                   "url": url, "full_archive_bytes": metadata["size"],
                   "selected_prefix": prefix, "files": entries,
                   "selected_compressed_bytes": sum(f["compressed_bytes"] for f in entries),
                   "selected_uncompressed_bytes": sum(f["uncompressed_bytes"] for f in entries),
                   "metadata_bytes_transferred": remote.downloaded})
    print(prefix, len(entries), assets[-1]["selected_compressed_bytes"], flush=True)

manifest = {"status": "metadata preflight only; weights and geometry not downloaded",
            "task": "lift_pot", "task_config": "demo_clean", "episodes": 100,
            "gpu_physical_index": 0, "forbidden_gpu_indices": [6, 7],
            "precision": "BF16 for initial pipeline check on GPU 0 RTX 4090 (24 GB)",
            "precision_flags": ["--use_bf16", "True", "--use_fp32", "False"],
            "published_benchmark_precision": "FP32; official guide estimates ~32 GB with simulator",
            "gpu_hardware_evidence": "deployment/lingbotvla-env-20261013/verification.json",
            "inference_env": "lingbotvla", "sim_env": "RoboTwin",
            "models": models, "assets": assets,
            "free_disk_bytes": shutil.disk_usage(ROOT).free,
            "not_required_for_this_stage": ["full trajectories", "background_texture.zip",
                                             "Qwen weight shards", "depth/DINO teacher weights"],
            "pending_runtime_checks": ["legacy SAPIEN renderer must select GPU 0",
                "strict checkpoint load with config/processor-only Qwen directory",
                "real scene initialization and action response", "100-episode closed-loop result"]}
(OUT / "download-manifest.json").write_text(json.dumps(manifest, indent=2))
print("Manifest saved", OUT / "download-manifest.json", flush=True)
