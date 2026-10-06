import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if "extends" in config:
        parent = read_config(Path(path).parent / config.pop("extends"))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(parent.get(key), dict):
                parent[key].update(value)
            else:
                parent[key] = value
        return parent
    return config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed, threads=4):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment():
    import scipy
    import sklearn

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "threads": torch.get_num_threads(),
        "commit": commit,
    }


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a fresh directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state(model, path):
    """Only numeric tensors, never Python object serialization."""
    np.savez_compressed(
        path, **{name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    )


def load_state(model, path, device="cpu"):
    with np.load(path, allow_pickle=False) as arrays:
        state = {name: torch.from_numpy(arrays[name].copy()).to(device) for name in arrays.files}
    model.load_state_dict(state, strict=True)
