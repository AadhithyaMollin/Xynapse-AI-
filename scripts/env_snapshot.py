#!/usr/bin/env python
"""Phase 0: snapshot the installed environment to reports/env_snapshot.json."""
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DISTS = [
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch",
    "transformers", "biopython", "biotite", "freesasa", "pydssp", "streamlit",
    "plotly", "py3Dmol", "requests", "pyarrow", "pyyaml", "tqdm", "joblib",
    "matplotlib", "pytest",
]


def dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    import torch

    snapshot = {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: dist_version(name) for name in DISTS},
        "torch": {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
    }
    out = ROOT / "reports" / "env_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
