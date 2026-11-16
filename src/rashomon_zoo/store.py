"""Storage layout of rashomon-zoo (see docs/design.md).

<zoo_dir>/<task>/task.json, split.npz
<zoo_dir>/<task>/<model_class>/<run>/manifest.json, models.parquet, outputs_<split>.npy, models/, explanations/<explainer>/
Arrays are stacked over model rows (row order = models.parquet) and opened memory-mapped.
"""
from __future__ import annotations

import datetime as _dt
import json
import pickle
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SPLITS = ("train", "holdout_1", "holdout_2")  # see tasks.py: train fits the black boxes; held-out sets have no fixed role
_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "captum", "shap", "rashomon-zoo")


def environment() -> dict:
    """Python, package versions and git commit of this repo (for manifests)."""
    versions = {}
    for p in _PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "packages": versions, "git_commit": commit, "git_dirty": dirty,
            "created": _dt.datetime.now().isoformat(timespec="seconds")}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr))


def load_array(path: Path, mmap: bool = True) -> np.ndarray:
    """Open a stacked array; with mmap=True only the slices that are indexed are read from disk."""
    return np.load(path, mmap_mode="r" if mmap else None)


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Task:
    """Fixed split of one dataset (standardised features)."""
    name: str
    X: dict  # split -> (n, d) float32
    y: dict  # split -> (n,) int64
    n_classes: int
    info: dict

    @property
    def n_features(self) -> int:
        return self.X["train"].shape[1]


def task_dir(zoo_dir: Path, task: str) -> Path:
    return Path(zoo_dir) / task


def task_hash(task: Task) -> str:
    """content hash of a task's data (arrays of all splits), recorded by runs and checked when they load the task"""
    import hashlib
    h = hashlib.sha256()
    for s in SPLITS:
        for a in (task.X[s], task.y[s]):
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_task(zoo_dir: Path, task: Task, overwrite: bool = False) -> Path:
    """Write split.npz and task.json. Refuses to replace an existing task unless overwrite=True, since runs below
    the task folder were trained on its data."""
    d = task_dir(zoo_dir, task.name)
    if (d / "split.npz").exists() and not overwrite:
        raise FileExistsError(f"task exists: {d} (runs may depend on it; pass overwrite=True to replace)")
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "split.npz", **{f"X_{s}": task.X[s] for s in SPLITS}, **{f"y_{s}": task.y[s] for s in SPLITS})
    write_json(d / "task.json", {"name": task.name, "n_classes": task.n_classes, "n_features": task.n_features,
                                 "n_samples": {s: int(len(task.y[s])) for s in SPLITS}, **task.info,
                                 "task_hash": task_hash(task), "environment": environment()})
    return d


def load_task(zoo_dir: Path, name: str, expected_hash: str | None = None) -> Task:
    """expected_hash (from a run manifest): raise if the task's data changed since the run was trained"""
    d = task_dir(zoo_dir, name)
    z = np.load(d / "split.npz")
    info = read_json(d / "task.json")
    t = Task(name=name, X={s: z[f"X_{s}"] for s in SPLITS}, y={s: z[f"y_{s}"] for s in SPLITS},
             n_classes=int(info["n_classes"]), info=info)
    if expected_hash is not None and task_hash(t) != expected_hash:
        raise ValueError(f"task {name!r} changed since the run was trained (task_hash mismatch)")
    return t


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    """One model class + configuration on one task, many model rows."""
    zoo_dir: Path
    task: str
    model_class: str
    name: str

    @property
    def dir(self) -> Path:
        return task_dir(self.zoo_dir, self.task) / self.model_class / self.name

    def load_task(self) -> Task:
        """the run's task; checks the task hash recorded at training time (runs from before the hash: no check)"""
        return load_task(self.zoo_dir, self.task, self.manifest().get("task_hash"))

    def manifest(self) -> dict:
        return read_json(self.dir / "manifest.json")

    def models(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "models.parquet")

    def outputs(self, split: str, mmap: bool = True) -> np.ndarray:
        return load_array(self.dir / f"outputs_{split}.npy", mmap)

    def explanations(self, explainer: str, split: str, mmap: bool = True) -> np.ndarray:
        return load_array(self.dir / "explanations" / explainer / f"{split}.npy", mmap)

    def explainer_manifest(self, explainer: str) -> dict:
        return read_json(self.dir / "explanations" / explainer / "manifest.json")

    def list_explainers(self) -> list[str]:
        d = self.dir / "explanations"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def load_model(self, model_id: int) -> Any:
        """the trained model of row `model_id` (architecture / estimator from the manifest's params)"""
        from . import modelzoo
        manifest = self.manifest()
        mc = modelzoo.get(manifest["model_class"])
        return load_model(self.model_path(model_id), mc.kind,
                          lambda: mc.build(manifest["params"], manifest["n_features"], manifest["n_classes"]))

    def model_path(self, model_id: int) -> Path:
        hits = sorted((self.dir / "models").glob(f"{model_id}.*"))
        if not hits:
            raise FileNotFoundError(f"no stored model {model_id} in {self.dir / 'models'}")
        return hits[0]


# ---------------------------------------------------------------------------------------------------------------
# model files: torch -> state dict (.pt), xgboost -> native (.ubj), scikit-learn -> pickle (.pkl, version in manifest)
def save_model(model: Any, path_stem: Path, kind: str) -> Path:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if kind == "torch":
        import torch
        p = path_stem.with_suffix(".pt")
        torch.save(model.state_dict(), p)
    elif kind == "xgboost":
        p = path_stem.with_suffix(".ubj")
        model.save_model(p)
    elif kind == "sklearn":
        p = path_stem.with_suffix(".pkl")
        with open(p, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(kind)
    return p


def load_model(path: Path, kind: str, build: Callable[[], Any] | None = None) -> Any:
    """`build`: callable returning an untrained model (torch architecture, xgboost estimator)."""
    path = Path(path)
    if kind == "torch":
        import torch
        model = build()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    if kind == "xgboost":
        model = build()
        model.load_model(path)
        return model
    if kind == "sklearn":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(kind)
