"""Stage `explain`: local (and global) explanations for the models of a run.

Local explainers write explanations/<name>/<split>.npy with shape (n_models, n_samples, n_features);
global importances write explanations/<name>/global.npy with shape (n_models, n_features).
Each explainer folder has a manifest with hyperparameters, model ids, sample indices and the target rule.
Targets: the class each model predicts for the sample.
Explained output: class probabilities by default (`output="proba"`); `output="logits"` for
torch models. ks_shap / lime_tabular / shap_tree always use probabilities.
Quality measures (explanations/<name>/<split>_info.npz, (n_models, n_samples) per key; manifest key "info"):
f_x, f_baseline (explained output at the target, at x and at the baseline / background mean) and the completeness
gap f_x - f_baseline - sum of contributions for ig, ks_captum, ks_shap, lime_captum, shap_tree; lime_tabular: intercept,
local_r2 (weighted R^2 of the surrogate), local_pred, local_pred_error = local_pred - f_x. vg, sg: none.
lime_tabular returns LIME's coefficients (slopes of the local linear model in standardised units), not contributions
relative to a baseline like the other explainers: compare signs only after multiplying by (x - training mean) / std.
"""
from __future__ import annotations

import time
from typing import Any, Sequence
from pathlib import Path

import numpy as np

from . import modelzoo
from .store import Run, Task, environment, save_array, write_json

# captum explainers: settings of the earlier CFIRE study (IG: 200 steps, KS / LIME: 300 samples); IG now with the
# (x - baseline) factor. Package explainers: see the notes at ks_shap / lime_tabular below.
DEFAULTS = {
    "vg": {},
    "sg": {"n_samples": 100, "noise_level": 0.1, "seed": 0},
    "ig": {"n_samples": 200},
    # captum KernelShap: single zero baseline (= training mean after standardisation); empty / full coalitions get
    # weight 1e6 (soft efficiency), other coalitions drawn i.i.d. from the Shapley size distribution, weight 1
    "ks_captum": {"n_samples": 300, "seed": 0},
    # captum Lime: binary on / off space against the zero baseline, Bernoulli(0.5), exp kernel on cosine distance,
    # Lasso(alpha=0.01) surrogate
    "lime_captum": {"n_samples": 300, "seed": 0},
    # shap KernelExplainer: background = one baseline (training mean), i.e. KernelSHAP as defined; exact efficiency;
    # baseline="train": all training rows as background (observed values only, e.g. for category codes);
    # subset sizes enumerated where the budget allows, complements paired. l1_reg=False: shap 0.54 defaults to
    # "num_features(10)", which would zero all but 10 attributions.
    "ks_shap": {"n_samples": "auto", "baseline": "train_mean", "l1_reg": False, "seed": 0},
    # lime package: regression on perturbed samples around the input (both flags differ from the package defaults:
    # discretize_continuous=True would give a binary "same quartile" space, sample_around_instance=False samples around
    # the training mean); all features (num_features=d; package default 10); otherwise package defaults
    # (5000 samples, kernel width 0.75 * sqrt(d), Ridge)
    "lime_tabular": {"n_samples": 5000, "discretize_continuous": False, "sample_around_instance": True,
                     "kernel_width": None, "seed": 0},
    # background: `background_size` training samples drawn with `seed` (default), or baseline="train_mean": the
    # training mean as the only background point. For random forests this is the exact Shapley value KernelSHAP
    # (ks_shap, same baseline) estimates. For xgboost with model_output="probability" it is not (shap 0.54.1: checked
    # against brute-force Shapley values of the probability, which ks_shap matches); there the logistic transform
    # is handled approximately, efficiency still holds.
    "shap_tree": {"model_output": "probability", "feature_perturbation": "interventional", "background_size": 100,
                  "baseline": None, "seed": 0},
    "builtin": {},
    "permutation": {"n_repeats": 10, "split": None, "seed": 0},  # split: held-out set to permute on (required)
}
# every explainer with randomness has a `seed` (recorded in the manifest); repeated runs with different seeds give
# the run-to-run variation for stability studies. Seeds are set locally (global random states are restored).
STOCHASTIC = {k for k, v in DEFAULTS.items() if "seed" in v}
GRADIENT = {"vg", "sg", "ig"}
GLOBAL = {"builtin", "permutation"}


def _torch_module(model, mc):
    """Torch view of any model (captum-based explainers need nn.Module with parameters)."""
    import torch
    if mc.kind == "torch":
        return model

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, x):
            return torch.from_numpy(mc.proba(model, x.detach().cpu().numpy()).astype(np.float32))

    return Wrapped().eval()


def _gap(fx, fb, contributions):
    """completeness gap: explained output at x minus output at the baseline minus the summed contributions"""
    return (fx - fb - contributions.sum(1)).astype(np.float32)


def _local(name, model, mc, X, targets, params, X_background):
    """returns (attributions (n, d), info {key: (n,) array}); info holds the quality measures the method offers:
    completeness gap for baseline methods (ig, ks_*, lime_captum, shap_tree), surrogate fit for lime_tabular"""
    import torch
    from . import attribution as A
    if name in GRADIENT and mc.kind != "torch":
        raise ValueError(f"{name} needs a differentiable (torch) model")
    m = _torch_module(model, mc)
    # explained output: class probabilities (default) or logits (torch models only)
    # or logits (torch models only)
    if params.get("output", "proba") == "logits":
        if mc.kind != "torch":
            raise ValueError("output='logits' needs a torch model")
        infer = m
    else:
        infer = (lambda x: torch.softmax(m(x), 1)) if mc.kind == "torch" else m  # wrapper already returns proba
    Xt, yt = torch.from_numpy(X), torch.from_numpy(targets)
    rows = np.arange(len(X))

    def f(Z):  # explained output at the targets, (n,)
        with torch.no_grad():
            out = infer(torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)))
        return out.detach().cpu().numpy()[rows, targets].astype(np.float32)

    fx = f(X)  # explained output (proba or logits) for captum / gradient methods
    fxp = mc.proba(model, X)[rows, targets].astype(np.float32)  # package explainers always explain probabilities
    zero = np.zeros_like(X)
    if name == "vg":
        out = A.vanilla_grad(m, Xt, targets=yt, inference_fn=infer)
    elif name == "sg":
        out = A.smooth_grad(m, Xt, targets=yt, inference_fn=infer, n_samples=params["n_samples"],
                            noise_level=params["noise_level"], force_random=False, random_state=params["seed"])
    elif name == "ig":  # zero baseline
        out = A.integrated_gradients(m, Xt, targets=yt, inference_fn=infer, n_samples=params["n_samples"],
                                     subtract_baseline=True)
        a = _np(out)
        return a, {"f_x": fx, "f_baseline": f(zero), "gap": _gap(fx, f(zero), a)}
    elif name == "ks_captum":  # zero baseline
        out = A.kernelshap(m, Xt, targets=yt, inference_fn=infer, n_samples=params["n_samples"],
                           masks=torch.arange(X.shape[1]))
        a = _np(out)
        return a, {"f_x": fx, "f_baseline": f(zero), "gap": _gap(fx, f(zero), a)}
    elif name == "lime_captum":  # on / off space against the zero baseline; captum returns no fit statistics
        out = A.lime(m, Xt, targets=yt, inference_fn=infer, n_samples=params["n_samples"])
        a = _np(out)
        return a, {"f_x": fx, "f_baseline": f(zero), "gap": _gap(fx, f(zero), a)}
    elif name == "ks_shap":
        import shap
        # baseline "train_mean": the training mean as the only background point; "train": all training rows as the
        # background distribution (only observed values, e.g. valid category codes; cost grows with the training size)
        if isinstance(params["baseline"], str):
            if params["baseline"] == "train_mean":
                bg = X_background.mean(0, keepdims=True)
            elif params["baseline"] == "train":
                bg = np.asarray(X_background, dtype=np.float32)
            else:
                raise ValueError(params["baseline"])
        else:
            bg = np.asarray(params["baseline"], dtype=np.float32).reshape(-1, X.shape[1])
        fp = lambda Z: mc.proba(model, np.asarray(Z, dtype=np.float32))  # noqa: E731
        ex = shap.KernelExplainer(fp, bg)
        sv = ex.shap_values(X, nsamples=params["n_samples"], l1_reg=params["l1_reg"], silent=True)
        sv = np.asarray(sv)  # (n, d, n_classes) in shap >= 0.45
        if sv.shape[0] != len(X):  # older shap: list over classes
            sv = np.moveaxis(sv, 0, -1)
        a = sv[rows, :, targets].astype(np.float32)
        fb = np.asarray(ex.expected_value, dtype=np.float32).reshape(-1)[targets]  # mean output over the background
        return a, {"f_x": fxp, "f_baseline": fb, "gap": _gap(fxp, fb, a)}
    elif name == "lime_tabular":
        from lime.lime_tabular import LimeTabularExplainer
        # sample_scale (optional, default 1, not in DEFAULTS so that existing manifests stay valid): width of the
        # perturbations in multiples of the training standard deviation (lime samples x + N(0, 1) * std). Passing
        # the training data scaled by it changes lime's std (sampling) but not the kernel weights: lime computes
        # distances in units of that same std.
        # categorical_features (optional, default none, not in DEFAULTS for the same reason): column indices of
        # category codes; lime samples them from their training frequencies (not x + N(0, std)) and its local model
        # uses the binary "same value as x" for them, so the slope is the effect of x's category
        scale = float(params.get("sample_scale", 1.0))
        cat = [int(c) for c in (params.get("categorical_features") or [])]
        if cat and scale != 1.0:
            raise ValueError("sample_scale != 1 would change the category codes in the training data")
        ex = LimeTabularExplainer(X_background * scale if scale != 1.0 else X_background, mode="classification",
                                  discretize_continuous=params["discretize_continuous"],
                                  sample_around_instance=params["sample_around_instance"],
                                  kernel_width=params["kernel_width"], random_state=params["seed"],
                                  categorical_features=cat or None)
        fp = lambda Z: mc.proba(model, np.asarray(Z, dtype=np.float32))  # noqa: E731
        out = np.zeros_like(X, dtype=np.float32)
        info = {k: np.zeros(len(X), np.float32) for k in ("intercept", "local_r2", "local_pred")}
        for i, (x, t) in enumerate(zip(X, targets)):
            e = ex.explain_instance(x, fp, labels=(int(t),), num_features=X.shape[1], num_samples=params["n_samples"])
            for j, w in e.as_map()[int(t)]:
                out[i, j] = w / scale  # lime's slopes are per unit of its std (= scale * training std): per training std
            # surrogate fit: intercept, weighted R^2 on the perturbed samples, surrogate prediction at x
            info["intercept"][i], info["local_r2"][i] = e.intercept[int(t)], e.score[int(t)] \
                if isinstance(e.score, dict) else e.score
            info["local_pred"][i] = np.asarray(e.local_pred[int(t)] if isinstance(e.local_pred, dict)
                                               else e.local_pred).reshape(-1)[0]
        return out, {"f_x": fxp, **info, "local_pred_error": (info["local_pred"] - fxp).astype(np.float32)}
    elif name == "shap_tree":
        import shap
        if mc.kind == "torch":
            raise ValueError("shap_tree needs a tree model (rf, xgb)")
        if mc.kind == "xgboost" and params["model_output"] == "probability":
            import json
            objective = json.loads(model.get_booster().save_config())["learner"]["objective"]["name"]
            if objective.startswith("multi:"):
                raise ValueError(f"shap_tree: shap's TreeExplainer has no probability output for multi-class "
                                 f"xgboost ({objective}); use model_output='raw' (margins, other units) or ks_shap")
        if params.get("baseline") is None:
            bg = X_background[np.random.default_rng(params["seed"]).choice(
                len(X_background), min(params["background_size"], len(X_background)), replace=False)]
        elif params["baseline"] == "train_mean":
            bg = X_background.mean(0, keepdims=True)
        else:
            raise ValueError(f"shap_tree baseline: None or 'train_mean', got {params['baseline']!r}")
        ex = shap.TreeExplainer(model, data=bg, model_output=params["model_output"],
                                feature_perturbation=params["feature_perturbation"])
        sv = np.asarray(ex.shap_values(X))  # (n, d, n_classes) or (n, d) for binary log-odds
        ev = np.asarray(ex.expected_value, dtype=np.float64).reshape(-1)  # mean output over the background
        if sv.ndim == 3:  # one value per class: take the predicted class
            sv = sv[np.arange(len(X)), :, targets] if sv.shape[0] == len(X) else sv[targets, np.arange(len(X)), :]
        else:  # binary models give values for class 1 only: flip the sign where class 0 is predicted
            sv = np.where((targets == 0)[:, None], -sv, sv)
            ev = np.array([1.0 - ev[0], ev[0]])  # probability output: class 0 = 1 - class 1
        a = sv.astype(np.float32)
        fb = ev[targets].astype(np.float32)
        return a, {"f_x": fxp, "f_baseline": fb, "gap": _gap(fxp, fb, a)}
    else:
        raise KeyError(name)
    return _np(out), {}  # vg, sg: no quality measure


def _np(out):
    if isinstance(out, tuple):
        out = out[0]
    return out.detach().cpu().numpy().astype(np.float32)


class _seeded:
    """set torch / numpy / random seeds for a block and restore the previous states afterwards"""
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        if self.seed is None:
            return
        import random
        import torch
        self.states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        random.seed(self.seed); np.random.seed(self.seed % 2**32); torch.manual_seed(self.seed)

    def __exit__(self, *exc):
        if self.seed is None:
            return
        import random
        import torch
        random.setstate(self.states[0]); np.random.set_state(self.states[1]); torch.set_rng_state(self.states[2])


def compute(model: Any, model_class: str, explainer: str, X: np.ndarray, params: dict | None = None,
            targets: np.ndarray | None = None, X_background: np.ndarray | None = None, task: Task | None = None,
            return_info: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Explanations without storing them (for experiments, e.g. repeated runs or budget studies).

    model: a trained model of `model_class` ("nn", "rf", "xgb", "svm"), e.g. from store.load_model
    explainer: a key of DEFAULTS; params override its defaults (same semantics as in `explain`)
    X: (n, d) samples; targets: class per sample (default: the model's predicted class)
    X_background: training data (needed by ks_shap with baseline "train_mean", lime_tabular, shap_tree)
    task: store.Task (needed by the global importances "permutation")
    Returns (n, d) local attributions or (d,) global importances as float32; with return_info also the quality
    measures {key: (n,)} (local explainers: see _local; f_* are outputs at the target, as explained).
    """
    params = {**DEFAULTS[explainer], **(params or {})}
    mc = modelzoo.get(model_class)
    if explainer in GLOBAL:
        g = _global(explainer, model, mc, task, params)
        return (g, {}) if return_info else g
    X = np.ascontiguousarray(X, dtype=np.float32)
    if targets is None:
        targets = mc.proba(model, X).argmax(1)
    with _seeded(params.get("seed")):  # captum (torch) and shap (numpy) draw from the global random states
        a, info = _local(explainer, model, mc, X, np.asarray(targets, dtype=np.int64), params, X_background)
    return (a, info) if return_info else a


def _global(name, model, mc, task, params):
    if name == "builtin":
        if not hasattr(model, "feature_importances_"):
            raise ValueError("model has no built-in feature importances")
        return np.asarray(model.feature_importances_, dtype=np.float32)
    if name == "permutation":
        from sklearn.inspection import permutation_importance
        s = params["split"]
        if s not in ("holdout_1", "holdout_2"):
            raise ValueError("permutation importance needs params split='holdout_1' or 'holdout_2'")

        class _Est:  # minimal estimator interface around proba
            def fit(self, *a): return self
            def predict(self, X): return mc.proba(model, X).argmax(1)
            def score(self, X, y): return float((self.predict(X) == y).mean())

        r = permutation_importance(_Est(), task.X[s], task.y[s], n_repeats=params["n_repeats"], random_state=params["seed"],
                                   scoring=lambda est, X, y: est.score(X, y))
        return r.importances_mean.astype(np.float32)
    raise KeyError(name)


def sample_order(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """An order of all sample positions 0..len(y)-1 whose every prefix is stratified by class (up to rounding):
    within each class the samples are shuffled with `seed`, and the j-th sample of class c (n_c samples) gets the
    key (j + 0.5) / n_c; positions are sorted by key (ties: by class). The first n positions are the subset of
    size n, so subsets of different sizes are nested."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    keys = np.empty(len(y))
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        pos = pos[rng.permutation(len(pos))]
        keys[pos] = (np.arange(len(pos)) + 0.5) / len(pos)
    return np.lexsort((y, keys))


def _sample_indices(task: Task, split: str, samples, sample_seed: int) -> np.ndarray:
    n_split = len(task.y[split])
    if samples is None:
        return np.arange(n_split)
    if isinstance(samples, (int, np.integer)) and not isinstance(samples, bool):
        if not 0 < samples <= n_split:
            raise ValueError(f"samples={samples}: {split} has {n_split} samples")
        return np.sort(sample_order(task.y[split], sample_seed)[:samples])
    idx = np.asarray(samples, dtype=np.int64)
    if idx.ndim != 1 or len(np.unique(idx)) != len(idx) or idx.min() < 0 or idx.max() >= n_split:
        raise ValueError(f"samples: distinct positions in 0..{n_split - 1} expected")
    return np.sort(idx)


def explain(run: Run, name: str, split: str | None = None, params: dict | None = None,
            models: str | Sequence[int] = "all",
            out_name: str | None = None, samples: int | Sequence[int] | None = None, sample_seed: int = 0) -> Path:
    """Explain the samples of `split` (a held-out set; which samples it holds is fixed by the task stage): all of
    them (samples=None), the first `samples` of `sample_order(y_split, sample_seed)` (a class-stratified subset;
    smaller sizes give nested subsets) or the given positions. Stored in position order; the positions are
    recorded as sample_indices. models: 'all', 'rashomon' (requires `select`) or a list of model ids. split:
    'holdout_1', 'holdout_2' or 'train' (recorded in the manifest)."""
    if name not in GLOBAL and split is None:
        raise ValueError("give the split to explain: 'holdout_1', 'holdout_2' or 'train'")
    params = {**DEFAULTS[name], **(params or {})}
    run_manifest = run.manifest()
    task = run.load_task()
    models_table = run.models()
    if isinstance(models, str):
        if models == "rashomon":
            models_table = models_table[models_table["in_rashomon_set"]]
        elif models != "all":
            raise ValueError(f"models: 'all', 'rashomon' or a list of ids, got {models!r}")
        ids = models_table["model_id"].tolist()
    else:
        ids = [int(i) for i in models]
        unknown = set(ids) - set(models_table["model_id"])
        if unknown:
            raise ValueError(f"unknown model ids {sorted(unknown)}")
    if name in GLOBAL:
        if samples is not None:
            raise ValueError("samples: only for local explainers")
        X, idx = task.X["train"][:0], np.arange(0)  # global importances: no samples explained
    else:
        X, idx = task.X[split], _sample_indices(task, split, samples, sample_seed)
    X = np.ascontiguousarray(X[idx], dtype=np.float32)
    out_dir = run.dir / "explanations" / (out_name or name)
    results, infos, seconds = [], [], []
    for mid in ids:
        model = run.load_model(mid)
        t0 = time.time()
        a, info = compute(model, run_manifest["model_class"], name, X, params, X_background=task.X["train"], task=task,
                          return_info=True)
        results.append(a); infos.append(info)
        seconds.append(time.time() - t0)
    save_array(out_dir / ("global.npy" if name in GLOBAL else f"{split}.npy"), np.stack(results))
    if infos[0]:  # quality measures, (n_models, n_samples) per key
        np.savez(out_dir / f"{split}_info.npz", **{k: np.stack([i[k] for i in infos]) for k in infos[0]})
    write_json(out_dir / "manifest.json", {
        "explainer": name, "params": params, "split": None if name in GLOBAL else split,
        "model_ids": ids, "models": models, "sample_indices": None if name in GLOBAL else idx.tolist(),
        "samples": None if samples is None else (int(samples) if isinstance(samples, (int, np.integer)) else "given"),
        "sample_seed": sample_seed if isinstance(samples, (int, np.integer)) else None,
        "targets": "predicted class of each model", "seconds_per_model": seconds,
        "info": sorted(infos[0]) if infos[0] else None, "environment": environment()})
    return out_dir


DERIVED = {"times_input": "attribution * x (standardised input, i.e. (x - training mean) / training std); for "
                          "lime_tabular: the local linear model's contribution relative to the training mean"}


def derive(run: Run, source: str, out_name: str, op: str = "times_input") -> Path:
    """A new explanation folder computed from stored explanations, without recomputing them (e.g. lime_tabular
    slopes -> slope * x). The manifest copies the source's settings and records the derivation; quality measures
    are not carried over (they refer to the source)."""
    if op not in DERIVED:
        raise KeyError(f"op: one of {sorted(DERIVED)}")
    src = run.explainer_manifest(source)
    if src["split"] is None:
        raise ValueError("derive: local explanations only")
    idx = np.asarray(src["sample_indices"], dtype=np.int64)
    X = run.load_task().X[src["split"]][idx]
    a = np.asarray(run.explanations(source, src["split"]), dtype=np.float32) * X[None].astype(np.float32)
    out_dir = run.dir / "explanations" / out_name
    if out_dir.exists():
        raise FileExistsError(out_dir)
    save_array(out_dir / f"{src['split']}.npy", a)
    write_json(out_dir / "manifest.json", {**{k: v for k, v in src.items() if k not in ("info", "environment")},
                                           "info": None, "derived_from": source, "derivation": op,
                                           "derivation_note": DERIVED[op], "environment": environment()})
    return out_dir
