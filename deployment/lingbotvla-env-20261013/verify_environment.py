"""Offline environment checks; tiny CUDA operations on physical GPU 0 only."""
import importlib
import importlib.metadata as metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Expose physical GPU 0 only"
assert Path(sys.prefix).name == "lingbotvla", sys.prefix
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/home/lmy/Robbyant/lingbot-vla-v2")
report = {"python": platform.python_version(), "prefix": sys.prefix,
          "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
          "versions": {}, "checks": {}}

def check(name, function):
    try:
        detail = function()
        report["checks"][name] = {"passed": True, "detail": detail}
        print(f"PASS {name}: {detail}", flush=True)
    except Exception:
        detail = traceback.format_exc()
        report["checks"][name] = {"passed": False, "detail": detail}
        print(f"FAIL {name}:\n{detail}", flush=True)

for package in ["torch", "torchvision", "torchaudio", "torchdata", "torchcodec",
                "flash-attn", "transformers", "numpy", "huggingface-hub",
                "lerobot", "lingbotvla", "moge", "mdm", "utils3d", "accelerate",
                "pyarrow", "protobuf", "packaging", "databricks-sdk", "mlflow",
                "numpydantic", "plyfile", "setuptools"]:
    try:
        report["versions"][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        report["versions"][package] = None

def torch_cuda():
    import torch
    assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
    assert sys.version_info[:2] == (3, 12), sys.version
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch.manual_seed(42)
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    torch.testing.assert_close((a @ b).cpu(), a.cpu() @ b.cpu(), rtol=1e-4, atol=1e-4)
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "runtime_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}

def flash_attention():
    import torch
    import torch.nn.functional as functional
    import flash_attn
    from flash_attn import flash_attn_func
    assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
    errors = {}
    for dtype in (torch.float16, torch.bfloat16):
        q, k, v = [torch.randn(1, 32, 2, 64, device="cuda", dtype=dtype,
                              requires_grad=True) for _ in range(3)]
        actual = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        reference = functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(),
            v.transpose(1, 2).float(), is_causal=True).transpose(1, 2)
        torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.03)
        actual.float().square().mean().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
        errors[str(dtype)] = float((actual.float() - reference).detach().abs().max())
    torch.cuda.synchronize()
    return {"version": flash_attn.__version__, "forward_backward": True,
            "max_abs_error_vs_fp32_sdpa": errors}

check("torch_cuda_matmul", torch_cuda)
check("flash_attention_fp16_bf16", flash_attention)
for module in [
    "torchvision", "torchaudio", "torchdata", "torchcodec", "transformers",
    "peft", "datasets", "accelerate", "cv2", "mlflow", "trimesh",
    "lerobot.datasets.lerobot_dataset", "utils3d", "moge.model.v2", "mdm.model.v2",
    "lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla",
    "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2",
    "lingbotvla.data.vla_data.utils", "lingbotvla.data", "lingbotvla.optim",
    "lingbotvla.models.vla.vision_models.module_utils", "deploy.lingbot_vla_v2_policy",
    "tasks.vla.train_lingbotvla",
]:
    check(f"import:{module}", lambda module=module: str(importlib.import_module(module).__file__))

def depth_symbols():
    from lingbotvla.models.vla.vision_models import module_utils
    assert module_utils.v2 is not None
    assert module_utils.v2_morgbd is not None
    assert callable(module_utils.colorize_depth)
    return "MoGe and MDM symbols available; no swallowed depth import failure"

check("depth_model_symbols", depth_symbols)

def video_decode():
    import av
    import numpy as np
    import torch
    from lingbotvla.data.vla_data.video_utils import decode_video_frames_torchcodec
    with tempfile.TemporaryDirectory(prefix="video-check-", dir=Path(__file__).parent) as temporary:
        path = Path(temporary) / "synthetic.mp4"
        with av.open(str(path), mode="w") as output:
            stream = output.add_stream("libx264", rate=10)
            stream.width = 32
            stream.height = 32
            stream.pix_fmt = "yuv420p"
            for index in range(4):
                array = np.full((32, 32, 3), index * 60, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        frames = decode_video_frames_torchcodec(path, [0.0, 0.1, 0.2, 0.3], tolerance_s=0.01)
        assert frames.shape == (4, 3, 32, 32), frames.shape
        assert frames.dtype == torch.uint8, frames.dtype
        means = frames.float().mean(dim=(1, 2, 3))
        assert torch.all(means[1:] > means[:-1] + 40), means
        return {"shape": list(frames.shape), "dtype": str(frames.dtype), "frame_means": means.tolist()}

check("project_video_decode", video_decode)

def numpy_auxiliary_compatibility():
    import numpy as np
    from plyfile import PlyData, PlyElement
    from numpydantic import NDArray, Shape
    from pydantic import BaseModel

    class ArrayProbe(BaseModel):
        values: NDArray[Shape["2, 3"], np.float32]

    probe = ArrayProbe(values=np.zeros((2, 3), dtype=np.float32))
    assert probe.values.shape == (2, 3)
    vertices = np.array([(1.0, 2.0, 3.0)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    stream = BytesIO()
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(stream)
    stream.seek(0)
    restored = PlyData.read(stream)["vertex"].data
    np.testing.assert_array_equal(vertices, restored)
    return {"numpy": np.__version__, "numpydantic_validation": True, "plyfile_roundtrip": True}

check("numpy_auxiliary_compatibility", numpy_auxiliary_compatibility)
report["passed"] = all(value["passed"] for value in report["checks"].values())
Path(__file__).with_name("verification.json").write_text(json.dumps(report, indent=2))
print(f"Overall passed: {report['passed']}", flush=True)
raise SystemExit(0 if report["passed"] else 1)
