"""Stage `task`: fix the data every later stage works on (which samples exist and which split they are in).

Splits: `train` (the black boxes are fitted on it, nothing else) and two held-out sets `holdout_1`, `holdout_2`
without a built-in role: every later stage names the held-out set it uses and for what (recorded in its manifest).

    make_task(zoo_dir, "beans", splits=(0.6, 0.2, 0.2))                        # fractions (train, holdout_1, holdout_2)
    make_task(zoo_dir, "beans", splits={"holdout_1": 1000, "holdout_2": 1000}) # absolute held-out sizes, train = rest
    make_task(zoo_dir, "beans", splits=(0.6, 0.2, 0.2), max_samples=3000)      # cap the dataset first
    make_task(zoo_dir, "heloc", splits={"holdout_2": 1000})                    # provider test file kept as holdout_1
    make_task(zoo_dir, "breastcancer", splits="legacy")                        # splits of the earlier studies

- max_samples caps only the total number of samples (stratified by class, before splitting). With a provider
  test file, it caps the provider's training data; the test file is not capped (give holdout_1 = n for that).
- splits sets the split sizes: fractions (train, holdout_1, holdout_2) of the (capped) data, or absolute sizes for
  the held-out sets with train taking the rest. Draw order: holdout_1, then holdout_2, then train (the rest).
  Datasets with a provider test file: it is holdout_1 (an absolute holdout_1 size draws from it); fractions
  (train, holdout_2) of the provider's training data.
- All draws are stratified by class and use `seed`. Standardisation: fit on the training split.
- splits="legacy": the legacy loaders as they are (splits of papers R and C, data_seed, the loaders' own caps);
  loader test -> holdout_1, loader val -> holdout_2 (the loaders' own order; papers R and C used the loader test
  split as their reference set / CFIRE input set and the loader val split for the rule evaluation).
Everything is recorded in task.json (specification, sizes, indices into the raw data, legacy name mapping).
"features": column description where the loader gives one (feature names, "numeric" / "categorical" per column,
category names per categorical column, in code order; categorical columns are integer codes before
standardisation, which is applied to all columns), else None.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from . import datasets
from .store import SPLITS, Task, save_task

H1, H2 = "holdout_1", "holdout_2"
LEGACY_NAMES = {H1: "loader test (meta X)", H2: "loader val (meta X_val)"}

DEFAULT_DATA_SEED = 11880  # data seed of the earlier Rashomon and CFIRE studies


def available_tasks() -> list[str]:
    return sorted(datasets.dataset_callables)


def _loader(name):
    if name not in datasets.dataset_callables:
        raise KeyError(f"unknown task {name!r}; available: {available_tasks()}")
    return datasets.dataset_callables[name]


# ---------------------------------------------------------------------------------------------------------------
# legacy splits
def _val_cap(loader) -> int | None:
    """The legacy `_return_dataset` cuts its validation split (holdout_2) to the loader's default batch_sizes[-1] samples
    (e.g. Dry Bean: 1050, paper R's reference-set size). Dataverse loaders do not cut. Returned for the record."""
    if "_return_dataset(" not in inspect.getsource(loader):
        return None
    default = inspect.signature(loader).parameters.get("batch_sizes")
    return None if default is None or default.default is inspect.Parameter.empty else int(default.default[-1])


def load_dataset(name: str, data_seed: int = DEFAULT_DATA_SEED) -> Task:
    """Legacy splits: exactly as the legacy loaders split (default batch sizes matter: they cap the loader's val
    split). Loader test -> holdout_1, loader val -> holdout_2."""
    loader = _loader(name)
    datasets.last_class_names = None
    datasets.last_feature_info = None
    (X_tr, y_tr), (X_te, y_te), (X_val, y_val), n_dims, n_classes = loader(random_state=data_seed, as_torch=False)
    class_names = datasets.last_class_names
    features = datasets.last_feature_info
    X = {"train": X_tr, H1: X_te, H2: X_val}
    y = {"train": y_tr, H1: y_te, H2: y_val}
    X = {k: np.asarray(v, dtype=np.float32) for k, v in X.items()}
    y = {k: np.asarray(v).astype(np.int64) for k, v in y.items()}
    cap = _val_cap(loader)
    return Task(name=name, X=X, y=y, n_classes=int(n_classes),
                info={"splits": "legacy", "data_seed": data_seed, "loader": f"rashomon_zoo.datasets.{loader.__name__}",
                      "standardised": "legacy loader: StandardScaler fit on the training split",
                      "legacy_names": LEGACY_NAMES, "class_names": class_names, "features": features,
                      "holdout_2_cap_by_loader": cap, "holdout_2_capped": cap is not None and len(y[H2]) == cap})


# ---------------------------------------------------------------------------------------------------------------
# raw data and split specification
class _Captured(Exception):
    pass


def raw_data(name: str) -> dict:
    """The dataset before any split or standardisation, read by the unchanged legacy loader: the loader is run with
    its split helper (`_return_dataset` / `_return_dataverse`) replaced by one that hands back its inputs, so file
    reading and label encoding stay exactly the legacy code. Returns {"X", "y"} and, for datasets with a provider
    test file, {"X_test_fixed", "y_test_fixed"}."""
    loader = _loader(name)
    out = {}

    def grab_dataset(data, target, *a, **k):
        out.update(X=data, y=target)
        raise _Captured

    def grab_dataverse(X_tr, Y_tr, X_te, Y_te, *a, **k):
        out.update(X=X_tr, y=Y_tr, X_test_fixed=X_te, y_test_fixed=Y_te)
        raise _Captured

    saved = datasets._return_dataset, datasets._return_dataverse
    datasets._return_dataset, datasets._return_dataverse = grab_dataset, grab_dataverse
    datasets.last_class_names = None
    datasets.last_feature_info = None
    try:
        loader(random_state=None, as_torch=False)
    except _Captured:
        pass
    finally:
        datasets._return_dataset, datasets._return_dataverse = saved
    if not out:
        raise RuntimeError(f"loader of {name!r} does not use a known split helper")
    out["class_names"] = datasets.last_class_names
    out["features"] = datasets.last_feature_info
    for k in [k for k in out if k not in ("class_names", "features")]:
        out[k] = np.asarray(out[k], dtype=np.float32 if k.startswith("X") else np.int64)
    classes = np.unique(np.concatenate([out["y"], out.get("y_test_fixed", out["y"][:0])]))
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(f"{name}: labels are not 0..n_classes-1: {classes}")
    return out


def _draw(idx, y, n, seed, what):
    """n of idx, stratified by y (y aligned with idx); returns (drawn, rest), both sorted. The caller has checked
    0 < n < len(idx); stratification needs every class at least once in the drawn part and in the rest."""
    from sklearn.model_selection import train_test_split
    try:
        a, b = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    except ValueError as e:
        counts = dict(zip(*np.unique(y, return_counts=True)))
        raise ValueError(f"cannot draw {n} of {len(idx)} samples for {what} stratified by class "
                         f"(class counts {counts}): {e}") from None
    return np.sort(a), np.sort(b)


def _check_n(n, available, what, rest_needed=True):
    """requested sizes must be positive integers and leave at least one sample where a rest is needed"""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError(f"{what}: size must be a positive integer, got {n!r}")
    limit = available - 1 if rest_needed else available
    if n > limit:
        raise ValueError(f"{what}: {n} requested, {available} available"
                         + (" (at least one sample must remain for the other splits)" if rest_needed else ""))


def _sizes(n, splits, fixed_test):
    """held-out sizes (holdout_1, holdout_2) from the specification; train is the rest. With a provider test file,
    holdout_1 = None means the whole file."""
    if isinstance(splits, (tuple, list)):
        f = np.asarray(splits, dtype=float)
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            raise ValueError(f"fractions must be positive, got {splits!r}")
        if fixed_test:
            if len(f) != 2:
                raise ValueError("dataset has a provider test file (= holdout_1): give fractions (train, holdout_2), "
                                 "or absolute sizes")
            return None, int(round(f[1] / f.sum() * n))
        if len(f) != 3:
            raise ValueError("fractions must be (train, holdout_1, holdout_2)")
        f = f / f.sum()
        return int(round(f[1] * n)), int(round(f[2] * n))
    if isinstance(splits, dict):
        bad = set(splits) - {H1, H2}
        if bad:
            raise ValueError(f"absolute sizes are given for {H1} and {H2} only (train is the rest), got {sorted(bad)}")
        if not fixed_test and H1 not in splits:
            raise ValueError(f"give a {H1} size (this dataset has no provider test file)")
        if H2 not in splits:
            raise ValueError(f"give a {H2} size")
        return splits.get(H1), splits[H2]
    raise TypeError(f"splits must be 'legacy', a tuple of fractions or a dict of sizes, got {splits!r}")


def split_dataset(name: str, splits: tuple[float, ...] | dict[str, int], max_samples: int | None = None, seed: int = 0) -> Task:
    """Task from the raw data and a split specification (see module docstring)."""
    from sklearn.preprocessing import StandardScaler
    raw = raw_data(name)
    X, y = raw["X"], raw["y"]
    fixed_test = "X_test_fixed" in raw
    pool = np.arange(len(y))
    what_pool = "provider training data" if fixed_test else "dataset"
    if max_samples is not None:
        _check_n(max_samples, len(pool), f"max_samples ({what_pool} has {len(pool)})", rest_needed=False)
        if max_samples < len(pool):
            pool, _ = _draw(pool, y, max_samples, seed, "max_samples")
    n1, n2 = _sizes(len(pool), splits, fixed_test)
    if fixed_test:
        h1_src = "provider test file"
        n_file = len(raw["y_test_fixed"])
        i1 = np.arange(n_file)
        if n1 is not None:
            _check_n(n1, n_file, f"{H1} (provider test file)", rest_needed=False)
            if n1 < n_file:
                i1, _ = _draw(i1, raw["y_test_fixed"], n1, seed, H1)
        _check_n(n2, len(pool), H2)
        i2, itr = _draw(pool, y[pool], n2, seed, H2)
        Xs = {"train": X[itr], H1: raw["X_test_fixed"][i1], H2: X[i2]}
        ys = {"train": y[itr], H1: raw["y_test_fixed"][i1], H2: y[i2]}
    else:
        h1_src = "drawn from the dataset"
        _check_n(n1, len(pool), H1)
        _check_n(n2, len(pool) - n1, f"{H2} (after {H1}: {len(pool) - n1} left)")
        i1, rest = _draw(pool, y[pool], n1, seed, H1)
        i2, itr = _draw(rest, y[rest], n2, seed, H2)
        Xs = {"train": X[itr], H1: X[i1], H2: X[i2]}
        ys = {"train": y[itr], H1: y[i1], H2: y[i2]}
    indices = {"train": itr, H1: i1, H2: i2}
    missing = {s: sorted(set(range(int(max(ys[q].max() for q in SPLITS)) + 1)) - set(ys[s].tolist())) for s in SPLITS}
    if any(missing.values()):
        raise ValueError(f"classes missing from a split: { {s: m for s, m in missing.items() if m} }")
    scaler = StandardScaler().fit(Xs["train"])
    Xs = {s: scaler.transform(Xs[s]).astype(np.float32) for s in SPLITS}
    n_classes = int(max(ys[s].max() for s in SPLITS) + 1)
    return Task(name=name, X=Xs, y=ys, n_classes=n_classes, info={
        "splits": list(splits) if isinstance(splits, tuple) else splits, "max_samples": max_samples, "seed": seed,
        "stratified": "by class, every draw", H1: h1_src, "class_names": raw["class_names"],
        "features": raw["features"],
        "loader": f"rashomon_zoo.datasets.{_loader(name).__name__} (raw data)",
        "standardised": "StandardScaler fit on the training split",
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "n_raw": int(len(y)) + (int(len(raw["y_test_fixed"])) if fixed_test else 0),
        # indices into the raw data (holdout_1 with a provider test file: into that file)
        "indices": {s: v.tolist() for s, v in indices.items()}})


def make_task(zoo_dir: Path, name: str, splits: str | tuple[float, ...] | dict[str, int], max_samples: int | None = None, seed: int = 0,
              data_seed: int = DEFAULT_DATA_SEED, task_name: str | None = None, overwrite: bool = False) -> Task:
    """Write <zoo_dir>/<task_name or name>/ (see module docstring). `splits` has no default: the split is a decision.
    data_seed applies to splits="legacy" only; seed and max_samples to the specification only."""
    if isinstance(splits, str):
        if splits != "legacy":
            raise ValueError(f"splits={splits!r}: only 'legacy' is a valid string")
        if max_samples is not None:
            raise ValueError("max_samples does not apply to the legacy splits")
        task = load_dataset(name, data_seed)
    else:
        task = split_dataset(name, splits, max_samples, seed)
    if task_name:
        task = Task(name=task_name, X=task.X, y=task.y, n_classes=task.n_classes,
                    info={**task.info, "dataset": name})
    save_task(zoo_dir, task, overwrite=overwrite)
    return task
