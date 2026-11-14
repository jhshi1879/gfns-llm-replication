import atexit
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def load_runpod_env():
    env_path = Path(os.environ.get(
        "RUNPOD_ENV_FILE", str(Path(__file__).resolve().parents[1] / ".env")
    ))
    if not env_path.exists():
        return
    for line_number, line in enumerate(env_path.read_text().splitlines(), start=1):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in ("RUNPOD_API_KEY", "RUNPOD_POD_ID"):
            continue
        try:
            tokens = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError(f"Invalid RunPod entry in .env at line {line_number}.") from None
        if len(tokens) != 1:
            raise ValueError(f"Invalid RunPod entry in .env at line {line_number}.")
        os.environ.setdefault(name, tokens[0])


def validate_credentials(pod_id, api_key):
    if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise ValueError("Set RUNPOD_POD_ID to the specific pod to stop.")
    if not isinstance(api_key, str) or not api_key.strip() or "\n" in api_key or "\r" in api_key:
        raise ValueError("Set RUNPOD_API_KEY locally to enable automatic pod shutdown.")


def validate_runpod_config():
    load_runpod_env()
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    validate_credentials(pod_id, api_key)
    return pod_id, api_key


def current_runpod_id():
    environment_id = os.environ.get("RUNPOD_POD_ID")
    try:
        entries = Path("/proc/1/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError):
        entries = []
    pod_ids = [entry.partition(b"=")[2].decode("utf-8") for entry in entries
               if entry.startswith(b"RUNPOD_POD_ID=")]
    if len(pod_ids) > 1:
        raise ValueError("Current pod identity is ambiguous.")
    pod_id = pod_ids[0] if pod_ids else environment_id
    if not pod_id or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise ValueError("Cannot verify the current RunPod pod ID; automatic shutdown is unavailable.")
    if environment_id and environment_id != pod_id:
        raise ValueError("Shutdown pod ID does not match the current pod.")
    return pod_id


def register_pod_shutdown():
    if os.environ.get("STOP_RUNPOD_ON_EXIT") != "1" or os.environ.get("RANK", "0") != "0":
        return
    credentials_path = os.environ.get("RUNPOD_CREDENTIALS_FILE")
    if credentials_path:
        path = Path(credentials_path)
        try:
            pod_id, api_key = json.loads(path.read_text())
        finally:
            path.unlink()
        validate_credentials(pod_id, api_key)
    else:
        pod_id, api_key = validate_runpod_config()
    current_pod = os.environ.get("RUNPOD_POD_ID")
    if current_pod and current_pod != pod_id:
        raise ValueError("Shutdown pod ID does not match the current pod.")
    sdk_python = Path(__file__).resolve().parent / ".venv-runpod-sdk/bin/python"
    if not sdk_python.is_file():
        raise RuntimeError("RunPod SDK environment missing; use the updated remote launcher.")

    def stop_runpod_on_exit():
        print(f"Requesting stop for pod {pod_id} using RunPod SDK...", flush=True)
        result = subprocess.run(
            [str(sdk_python), str(Path(__file__).resolve()), "--stop"],
            input=json.dumps([pod_id, api_key]), text=True, timeout=90,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError("RunPod SDK stop failed; stop the pod manually and verify billing.")
        print("RunPod accepted the stop request; verify pod status in the console.", flush=True)

    atexit.register(stop_runpod_on_exit)
    print("Pod-side shutdown registered for training exit (success or failure).", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--supervise":
        register_pod_shutdown()
        child_env = os.environ.copy()
        child_env["STOP_RUNPOD_ON_EXIT"] = "0"
        result = subprocess.call(sys.argv[2:], env=child_env)
        sys.exit(result if result >= 0 else 128 - result)
    elif sys.argv[1:] == ["--credentials"]:
        print(json.dumps(validate_runpod_config()))
    elif len(sys.argv) == 3 and sys.argv[1] == "--credentials-for-pod":
        load_runpod_env()
        pod_id = sys.argv[2]
        configured_id = os.environ.get("RUNPOD_POD_ID")
        if configured_id and configured_id != pod_id:
            raise ValueError("Configured RunPod pod ID does not match the connected pod.")
        api_key = os.environ.get("RUNPOD_API_KEY", "")
        validate_credentials(pod_id, api_key)
        print(json.dumps([pod_id, api_key]))
    elif sys.argv[1:] == ["--current-pod-id"]:
        print(current_runpod_id())
    elif sys.argv[1:] in (["--receive"], ["--receive-current"]):
        credentials = json.load(sys.stdin)
        validate_credentials(*credentials)
        current_pod = current_runpod_id() if sys.argv[1] == "--receive-current" else os.environ.get("RUNPOD_POD_ID")
        if current_pod and current_pod != credentials[0]:
            raise ValueError("Shutdown pod ID does not match the current pod.")
        with tempfile.NamedTemporaryFile(mode="w", prefix="gfn-runpod-", dir="/tmp", delete=False) as file:
            json.dump(credentials, file)
        print(file.name)
    elif sys.argv[1:] == ["--stop"]:
        import runpod

        pod_id, api_key = json.load(sys.stdin)
        validate_credentials(pod_id, api_key)
        runpod.api_key = api_key
        runpod.stop_pod(pod_id)
    else:
        raise SystemExit("Unsupported RunPod helper command.")
