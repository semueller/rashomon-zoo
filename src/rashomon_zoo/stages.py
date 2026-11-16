"""Stages `tune`, `train`, `select` (see docs/design.md). Each reads and writes stored artefacts only."""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import modelzoo
from .store import SPLITS, Run, environment, load_task, read_json, save_array, save_model, task_hash, write_json  # noqa: F401

# what differs between the models of a run (see train). Default: all models get the same training data; the
# variation comes from the learner's own seeded randomness (nn: initialisation; rf: bootstrap and feature sampling;
# xgb: its own row subsampling, e.g. params subsample=0.9 - without it xgboost is deterministic and all models are
# identical, which train reports). SVC has no such randomness. "subsample" / "resample" vary the data instead.
DEFAULT_VARIATION = {"xgb": "seed", "nn": "seed", "rf": "seed", "svm": "seed"}

# paper R: all models within 5 percentage points of the best model's accuracy on the held-out reference set
# (paper R's X_ref = loader test split = holdout_1 under the legacy splits)
DEFAULT_CRITERION = {"metric": "accuracy", "split": "holdout_1", "epsilon": 0.05, "relative_to": "best"}
HELD_OUT = SPLITS[1:]  # outputs and accuracies are stored for both held-out sets


def _stopping_data(t, params):
    """held-out data for early stopping (nn, stopping="early_stopping": params stopping_split), else (None, None)"""
    split = params.get("stopping_split")
    return (t.X[split], t.y[split]) if split else (None, None)


def seed_everything(seed: int) -> None:
    """model seed before a model is built: torch (initialisation), numpy, random"""
    import random
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def tune(zoo_dir: Path, task: str, model_class: str, grid: dict, *, score_split: str, seeds: Iterable[int] = (0,),
         name: str = "tuning") -> dict:
    """Grid search scored by accuracy on the held-out set `score_split` (mean over `seeds`; no default: that set
    then informs a choice, recorded). Writes <task>/<model_class>/_tune/<name>/. Returns the best configuration."""
    if score_split not in HELD_OUT:
        raise ValueError(f"score_split must be one of {HELD_OUT}")
    t = load_task(zoo_dir, task)
    mc = modelzoo.get(model_class)
    Xtr, ytr = t.X["train"], t.y["train"]
    keys = sorted(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        mc.resolve(params, t.n_features, t.n_classes)  # unknown / inapplicable keys raise before training
        accs = []
        for s in seeds:
            seed_everything(int(s))
            m = mc.build(params, t.n_features, t.n_classes)
            m, _ = mc.fit(m, Xtr, ytr, *_stopping_data(t, mc.resolve(params, t.n_features, t.n_classes)), s, params)
            accs.append(float((mc.proba(m, t.X[score_split]).argmax(1) == t.y[score_split]).mean()))
        rows.append({**{f"param_{k}": str(v) for k, v in params.items()}, "score_mean": np.mean(accs),
                     "score_std": np.std(accs), "params": params})
    tuning = pd.DataFrame(rows).sort_values("score_mean", ascending=False)  # one row per configuration
    best = tuning.iloc[0]["params"]
    tune_dir = Path(zoo_dir) / task / model_class / "_tune" / name
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuning.drop(columns="params").to_csv(tune_dir / "tuning.csv", index=False)
    write_json(tune_dir / "manifest.json", {"task": task, "model_class": model_class, "grid": grid, "seeds": list(seeds),
                                     "score": f"accuracy on {score_split} (mean over seeds)", "score_split": score_split, "best": best,
                                     "task_hash": task_hash(t), "environment": environment()})
    return best


def train(zoo_dir: Path, task: str, model_class: str, run: str, seeds: Iterable[int], params: dict | None = None,
          tuned: str | None = None, variation: str | None = None,
          resample_fraction: float = 0.9) -> Run:
    """Train one model per seed with fixed hyperparameters: `params`, or the best of a `tune` result (`tuned`).

    variation: what differs between the models of a run
      "seed"     same training data for all models; only the model's own randomness depends on the seed
                 (nn: initialisation; rf: bootstrap and feature sampling; xgb: subsample / colsample_* < 1 only)
      "resample" additionally, each model is trained on `resample_fraction` of the training rows drawn with
                 replacement under the model seed (bootstrap; as in the earlier CFIRE study's tree models)
      "subsample" additionally, each model is trained on `resample_fraction` of the training rows drawn without
                 replacement under the model seed (a slightly different snapshot of the data)
    Note: xgboost without subsample / colsample_* < 1 is deterministic and invariant to row order, so with
    variation="seed" all its models are identical (train warns and records n_distinct_models).
    """
    if variation is None:
        variation = DEFAULT_VARIATION.get(model_class, "seed")
    if variation not in ("seed", "resample", "subsample"):
        raise ValueError(variation)
    if (params is None) == (tuned is None):
        raise ValueError("give exactly one of params / tuned")
    if tuned is not None:
        params = read_json(Path(zoo_dir) / task / model_class / "_tune" / tuned / "manifest.json")["best"]
        params_source = f"tune:{tuned}"
    else:
        params_source = "fixed"
    t = load_task(zoo_dir, task)
    mc = modelzoo.get(model_class)
    params_effective = mc.resolve(params, t.n_features, t.n_classes)  # raises on unknown / inapplicable keys
    r = Run(Path(zoo_dir), task, model_class, run)
    if (r.dir / "models.parquet").exists():
        raise FileExistsError(f"run exists: {r.dir}")
    Xtr, ytr = t.X["train"], t.y["train"]  # data size is fixed by the task stage, not here
    rows, outs = [], {s: [] for s in HELD_OUT}
    stop_X, stop_y = _stopping_data(t, params_effective)
    for model_id, seed in enumerate(seeds):
        seed_everything(int(seed))
        if variation in ("resample", "subsample"):
            ridx = np.random.default_rng(int(seed)).choice(len(Xtr), int(resample_fraction * len(Xtr)),
                                                            replace=variation == "resample")
            Xs, ys = Xtr[ridx], ytr[ridx]
        else:
            Xs, ys = Xtr, ytr
        m = mc.build(params, t.n_features, t.n_classes)
        m, info = mc.fit(m, Xs, ys, stop_X, stop_y, int(seed), params)
        save_model(m, r.dir / "models" / str(model_id), mc.kind)
        row = {"model_id": model_id, "seed": int(seed), "step": -1, "parent_id": -1,
               "param_fingerprint": modelzoo.fingerprint(m, mc.kind), **info}
        for s in outs:
            p = mc.proba(m, t.X[s]).astype(np.float32)
            outs[s].append(p)
            row[f"{s}_acc"] = float((p.argmax(1) == t.y[s]).mean())
        rows.append(row)
    # identical models are an error, identical behaviour is not (equally good models with different parameters are
    # what a Rashomon set is made of). Identical models: same learned parameters for deterministic learners
    # (param_fingerprint; xgboost without subsampling gives the same model for every seed), repeated seeds for torch.
    def mark(col, keys):
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
            rows[i][col] = first[k] if first[k] != i else -1
        return len(first)

    ident = [row["param_fingerprint"] for row in rows] if rows[0]["param_fingerprint"] is not None \
        else [(row["seed"], variation) for row in rows]
    n_distinct_models = mark("duplicate_of", ident)
    # information only: models with identical outputs on both held-out sets
    n_distinct_outputs = mark("same_outputs_as",
                              [hash(np.concatenate([outs[s][i] for s in HELD_OUT]).tobytes()) for i in range(len(rows))])
    if n_distinct_models < len(rows):
        import warnings
        warnings.warn(f"{model_class} run {run}: only {n_distinct_models} of {len(rows)} models are distinct "
                      f"(identical {'parameters' if rows[0]['param_fingerprint'] is not None else 'seeds'}; "
                      f"variation='{variation}'); see duplicate_of in models.parquet.")
    for s, arr in outs.items():
        save_array(r.dir / f"outputs_{s}.npy", np.stack(arr))
    pd.DataFrame(rows).to_parquet(r.dir / "models.parquet", index=False)
    write_json(r.dir / "manifest.json", {
        "task": task, "model_class": model_class, "run": run, "kind": mc.kind, "params": params_effective,
        "defaults_used": sorted(set(params_effective) - set(params)), "params_source": params_source, "seeds": [int(s) for s in seeds],
        "variation": variation, "resample_fraction": resample_fraction if variation != "seed" else None,
        "n_distinct_models": n_distinct_models, "n_distinct_outputs": n_distinct_outputs,
        "task_hash": task_hash(t),
        "n_features": t.n_features, "n_classes": t.n_classes, "environment": environment(),
        "created_by": "rashomon_zoo.stages.train", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    describe(r)
    return r


def _jsd(p, q):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1); q = np.clip(np.asarray(q, np.float64), 1e-12, 1)
    p /= p.sum(-1, keepdims=True); q /= q.sum(-1, keepdims=True); m = (p + q) / 2
    kl = lambda a, b: np.sum(a * np.log2(a / b), -1)  # noqa: E731
    return np.sqrt(np.maximum(0.0, (kl(p, m) + kl(q, m)) / 2))


def describe(run: Run) -> dict:
    """Basic statistics of the models of a run (all models, and the Rashomon set if selected); writes stats.json.
    prediction disagreement: share of samples on which two models predict different classes (over model pairs);
    output JSD: mean Jensen-Shannon distance (base 2) between the models' class distributions (over model pairs)."""
    models = run.models()
    task = run.load_task()
    groups = {"all": models}
    if "in_rashomon_set" in models:
        groups["rashomon_set"] = models[models["in_rashomon_set"]]
    out = {"run": run.name, "task": run.task, "model_class": run.model_class,
           "variation": run.manifest().get("variation")}
    for group, members in groups.items():
        ids = members["model_id"].to_numpy()
        st = {"n_models": int(len(members)),
              "n_distinct_models": int((members["duplicate_of"] == -1).sum()) if "duplicate_of" in members
              else int(len(members)),
              "n_distinct_outputs": int((members["same_outputs_as"] == -1).sum()) if "same_outputs_as" in members
              else None}
        for split in HELD_OUT:
            if len(task.y[split]) == 0:
                continue
            P = np.asarray(run.outputs(split))[ids]
            pred = P.argmax(-1)
            acc = (pred == task.y[split][None]).mean(1)
            pairs = list(itertools.combinations(range(len(ids)), 2))
            dis = [float(np.mean(pred[a] != pred[b])) for a, b in pairs]
            jsd = [float(_jsd(P[a], P[b]).mean()) for a, b in pairs]
            st[split] = {"accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std()),
                         "accuracy_min": float(acc.min()), "accuracy_max": float(acc.max()),
                         "prediction_disagreement_mean": float(np.mean(dis)) if dis else None,
                         "prediction_disagreement_max": float(np.max(dis)) if dis else None,
                         "output_jsd_mean": float(np.mean(jsd)) if jsd else None}
        out[group] = st
    write_json(run.dir / "stats.json", out)
    return out


def select(run: Run, criterion: dict | None = None) -> pd.DataFrame:
    """Mark the Rashomon set in models.parquet (column `in_rashomon_set`) and write selection.json."""
    c = {**DEFAULT_CRITERION, **(criterion or {})}
    if c["metric"] != "accuracy" or c["relative_to"] != "best":
        raise NotImplementedError(c)
    models = run.models()
    acc = models[f"{c['split']}_acc"]
    models["in_rashomon_set"] = (acc.max() - acc) <= c["epsilon"] + 1e-12
    models.to_parquet(run.dir / "models.parquet", index=False)
    write_json(run.dir / "selection.json", {"criterion": c, "best_accuracy": float(acc.max()),
                                            "n_models": int(len(models)), "n_selected": int(models["in_rashomon_set"].sum()),
                                            "selected_model_ids": models.loc[models["in_rashomon_set"], "model_id"].tolist(),
                                            "environment": environment()})
    describe(run)
    return models
