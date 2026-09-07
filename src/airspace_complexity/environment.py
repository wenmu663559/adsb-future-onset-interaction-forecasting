from datetime import datetime
import os
import platform as platform_module
import shutil
import subprocess
import sys
from typing import Any


def _run_version_command(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _collect_nvidia_info() -> tuple[bool, list[str], str | None]:
    output = _run_version_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if output is None:
        return False, [], None

    gpu_names: list[str] = []
    driver_versions: list[str] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if parts and parts[0]:
            gpu_names.append(parts[0])
        if len(parts) == 2 and parts[1]:
            driver_versions.append(parts[1])
    driver = driver_versions[0] if driver_versions else None
    return True, gpu_names, driver


def collect_environment_info() -> dict[str, Any]:
    """Collect Python, OS, CPU, memory, Git, GPU and CUDA information without changing the system."""
    nvidia_available, gpu_names, driver_version = _collect_nvidia_info()
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform_module.system(),
        "platform_release": platform_module.release(),
        "platform_version": platform_module.version(),
        "machine": platform_module.machine(),
        "processor": platform_module.processor(),
        "python_version": platform_module.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "git_version": _run_version_command(["git", "--version"]),
        "nvidia_smi_available": nvidia_available,
        "gpu_names": gpu_names,
        "cuda_driver_version": driver_version,
    }

