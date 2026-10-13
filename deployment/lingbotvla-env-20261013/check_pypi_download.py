import json
import time
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/torch/2.8.0/json", timeout=15) as response:
    metadata = json.load(response)
asset = next(item for item in metadata["urls"]
             if "cp312-cp312-manylinux_2_28_x86_64" in item["filename"])
start = time.monotonic()
request = urllib.request.Request(asset["url"], headers={"Range": "bytes=0-4194303"})
with urllib.request.urlopen(request, timeout=15) as response:
    received = len(response.read(4194304))
seconds = time.monotonic() - start
print(json.dumps({"source": "files.pythonhosted.org", "filename": asset["filename"],
                  "received_bytes": received, "seconds": seconds,
                  "mib_per_second": received / 1048576 / seconds}))
