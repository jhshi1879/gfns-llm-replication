import hashlib
import json
from pathlib import Path
import subprocess
import sys


def validate():
    project = Path(__file__).resolve().parent
    marker = Path(sys.prefix) / "preflight-passed.json"
    if sys.version_info[:2] != (3, 11) or not marker.is_file():
        raise RuntimeError("Run run_preflight.sh on this pod first; a validated Python 3.11 environment is required.")
    record = json.loads(marker.read_text())
    digest = hashlib.sha256((project / "requirements.txt").read_bytes()).hexdigest()
    if digest != record["requirements_sha256"]:
        raise RuntimeError("Requirements changed since preflight; rerun run_preflight.sh.")
    lock = Path(record["result_directory"]) / "requirements.lock.txt"
    installed = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    if sorted(installed.splitlines()) != sorted(lock.read_text().splitlines()):
        raise RuntimeError("Installed packages changed since preflight; rerun run_preflight.sh.")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")
    if torch.__version__ != record["torch"] or torch.version.cuda != record["cuda"]:
        raise RuntimeError("PyTorch/CUDA build changed since preflight; rerun run_preflight.sh.")
    print("Using preflight-validated environment:", sys.prefix)
    print("CUDA GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))


if __name__ == "__main__":
    validate()
