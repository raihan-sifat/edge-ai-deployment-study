"""Small, dependency-light helpers shared across the package."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LOGGER_NAME = "edgebench"
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger configured exactly once."""
    logger = logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
    if not logging.getLogger(LOGGER_NAME).handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root = logging.getLogger(LOGGER_NAME)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


def configure_logging(verbose: bool = False) -> None:
    """Raise or lower the level of the package-wide logger."""
    get_logger()
    logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG if verbose else logging.INFO)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that influences training.

    ``deterministic=True`` additionally asks PyTorch to avoid nondeterministic
    kernels. This can make some CPU convolutions fall back to slow paths, so it
    is off by default and only used by the reproducibility tests.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency, kept defensive
        pass

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def human_bytes(num_bytes: float) -> str:
    """Format a byte count with a binary prefix, e.g. ``4.21 MiB``."""
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < step or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= step
    return f"{value:.2f} TiB"  # pragma: no cover - unreachable


def slugify(text: str) -> str:
    """Turn an arbitrary label into a filesystem- and URL-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return slug.strip("-")


def json_dump(payload: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write ``payload`` as JSON, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=False, default=_json_default)
        handle.write("\n")
    return destination


def json_load(path: str | Path) -> Any:
    """Read a JSON document produced by :func:`json_dump`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    """Coerce a handful of common non-JSON types instead of raising."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def git_commit_hash(short: bool = True) -> str | None:
    """Return the current commit hash, or ``None`` outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short" if short else "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def environment_fingerprint() -> dict[str, Any]:
    """Capture everything needed to interpret a latency measurement.

    Latency on a CPU is only meaningful together with the thread count, the chip
    and the software stack. Every result file embeds this block.
    """
    import torch

    fingerprint: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads_default": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_threads_available": os.cpu_count(),
    }

    try:
        import numpy as np

        fingerprint["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass

    try:
        import onnxruntime

        fingerprint["onnxruntime"] = onnxruntime.__version__
        fingerprint["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        fingerprint["onnxruntime"] = None

    try:
        import psutil

        fingerprint["ram_total_bytes"] = psutil.virtual_memory().total
        fingerprint["cpu_freq_mhz_max"] = getattr(psutil.cpu_freq(), "max", None)
    except Exception:  # pragma: no cover - psutil is best effort
        pass

    fingerprint["git_commit"] = git_commit_hash()
    return fingerprint


def describe_cpu() -> str:
    """Best-effort human-readable CPU model string for reports."""
    for probe in _cpu_name_probes():
        if probe:
            return probe
    return platform.processor() or platform.machine() or "unknown CPU"


def _cpu_name_probes() -> list[str]:
    """Platform-specific ways to recover a marketing CPU name."""
    names: list[str] = []

    try:
        import psutil

        names.append(platform.processor() or "")
        del psutil  # only imported to confirm availability
    except Exception:  # pragma: no cover
        names.append("")

    if sys.platform.startswith("win"):
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            names.insert(0, str(value).strip())
        except Exception:  # pragma: no cover - registry access is best effort
            pass
    elif sys.platform.startswith("linux"):
        try:
            with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        names.insert(0, line.split(":", 1)[1].strip())
                        break
        except OSError:  # pragma: no cover
            pass
    elif sys.platform == "darwin":  # pragma: no cover - not a target platform
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0:
                names.insert(0, completed.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass

    return [name for name in names if name]
