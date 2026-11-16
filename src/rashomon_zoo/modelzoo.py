"""Model classes: construction, training and prediction behind one interface.

kind   : storage kind (see store.save_model)
build  : params -> untrained model
fit    : (model, X_train, y_train, X_stop, y_stop, seed, params) -> fitted model, info dict
         (X_stop / y_stop: held-out set for early stopping, nn with stopping="early_stopping" only; else None)
proba  : (model, X) -> (n, n_classes) probabilities
resolve: params -> the values actually used (defaults filled in); raises on unknown or inapplicable keys
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ModelClass:
    name: str
    kind: str
    build: Callable
    fit: Callable
    proba: Callable
    resolve: Callable


# ---------------------------------------------------------------------------------------------------------------
# scikit-learn / xgboost
def _sk_fit(model, X, y, Xv, yv, seed, params):
    model.set_params(random_state=seed)
    t0 = time.time()
    model.fit(X, y)
    return model, {"fit_seconds": time.time() - t0}


def _sk_proba(model, X):
    return model.predict_proba(X)


def _rf_build(params, n_features=None, n_classes=None):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**{"n_jobs": -1, **params})


def _svm_build(params, n_features=None, n_classes=None):
    from sklearn.svm import SVC
    return SVC(**{"probability": True, **params})


def _xgb_build(params, n_features=None, n_classes=None):
    from xgboost import XGBClassifier
    p = dict(params)
    if n_classes is not None and n_classes > 2:
        p.setdefault("objective", "multi:softprob")
        p.setdefault("eval_metric", "mlogloss")
    else:
        p.setdefault("objective", "binary:logistic")
        p.setdefault("eval_metric", "logloss")
    return XGBClassifier(**p)


def _sk_resolve(build):
    """given parameters plus the ones this module sets on top of the library defaults (e.g. xgboost objective);
    unknown keys raise. Library defaults are not repeated: the stored model files (xgboost config, pickles) carry
    the full configuration. random_state is set per model from the seed (models.parquet)."""
    def resolve(params, n_features=None, n_classes=None):
        est = build({}, n_features, n_classes)
        unknown = set(params) - set(est.get_params())
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}")
        lib = type(est)().get_params()
        full = build(params, n_features, n_classes).get_params()
        return {k: v for k, v in full.items()
                if k != "random_state" and (k in params or not _same(v, lib.get(k)))}
    return resolve


def _same(a, b):
    try:
        return bool(a == b) or (a != a and b != b)  # NaN == NaN (xgboost `missing`)
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------
# torch feed-forward networks (Linear + ReLU, logits out)
NN_DEFAULTS = {"hidden": [64, 64], "lr": 1e-3, "weight_decay": 0.0, "batch_size": 64, "order_seed": 0,
               "stopping": "fixed_steps", "n_steps": 2000, "max_epochs": 100, "patience": 10, "stopping_split": None}
_NN_ONLY_FOR = {"n_steps": "fixed_steps", "max_epochs": "early_stopping", "patience": "early_stopping",
                "stopping_split": "early_stopping"}


def _nn_resolve(params, n_features=None, n_classes=None):
    """NN_DEFAULTS updated by params. Raises on unknown keys and on keys that do not apply to the chosen stopping
    rule (e.g. max_epochs with fixed_steps), so a typo or a misplaced setting cannot be silently ignored."""
    unknown = set(params) - set(NN_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown nn parameters {sorted(unknown)}; valid: {sorted(NN_DEFAULTS)}")
    p = {**NN_DEFAULTS, **params}
    if p["stopping"] not in ("fixed_steps", "early_stopping"):
        raise ValueError(f"stopping must be 'fixed_steps' or 'early_stopping', got {p['stopping']!r}")
    inapplicable = [k for k in params if _NN_ONLY_FOR.get(k, p["stopping"]) != p["stopping"]]
    if inapplicable:
        raise ValueError(f"{inapplicable} do not apply with stopping={p['stopping']!r}")
    if p["stopping"] == "early_stopping" and p["stopping_split"] not in ("holdout_1", "holdout_2"):
        raise ValueError("early_stopping needs stopping_split='holdout_1' or 'holdout_2' (the set informs the stop)")
    return {k: v for k, v in p.items() if _NN_ONLY_FOR.get(k, p["stopping"]) == p["stopping"]}


def _nn_build(params, n_features, n_classes):
    from .models import make_ff
    return make_ff([n_features, *_nn_resolve(params)["hidden"], n_classes])


def _nn_fit(model, X, y, Xv, yv, seed, params):  # Xv, yv: early-stopping set (stopping_split)
    """Adam + cross-entropy on mini-batches.

    Rashomon setup (default, as in Müller et al. 2023): models differ only in their initialisation (model seed, set
    before the model is built, see stages.seed_everything); the data order is the same for all models (`order_seed`)
    and every model is trained for exactly `n_steps` batches (`stopping="fixed_steps"`).
    Option `stopping="early_stopping"`: stop when the accuracy on `stopping_split` has not improved for `patience` epochs and keep
    the best epoch (training length then differs between models)."""
    import torch
    p = {**NN_DEFAULTS, **_nn_resolve(params)}
    lr, batch_size, stopping = p["lr"], p["batch_size"], p["stopping"]
    n_steps, max_epochs, patience = p["n_steps"], p["max_epochs"], p["patience"]
    order_rng = np.random.default_rng(p["order_seed"])  # same batch order for every model
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=p["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(np.asarray(X, np.float32)), torch.from_numpy(np.asarray(y, np.int64))
    Xvt = None if Xv is None else torch.from_numpy(np.asarray(Xv, np.float32))

    def stop_acc():
        model.eval()
        with torch.no_grad():
            return float((model(Xvt).argmax(1).numpy() == yv).mean())

    if stopping == "fixed_steps" and n_steps * batch_size < len(X):
        raise ValueError(f"n_steps * batch_size = {n_steps * batch_size} < {len(X)} training samples: the network "
                         f"would not see every training sample (increase n_steps or reduce the training split)")
    t0 = time.time()
    steps, epoch, best, best_state, best_epoch = 0, 0, -1.0, None, -1
    while True:
        model.train()
        perm = order_rng.permutation(len(Xt))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
            steps += 1
            if stopping == "fixed_steps" and steps >= n_steps:
                break
        epoch += 1
        if stopping == "fixed_steps":
            if steps >= n_steps:
                break
            continue
        acc = stop_acc()
        if acc > best:
            best, best_epoch = acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience or epoch >= max_epochs:
            break
        if epoch >= max_epochs:
            break
    if stopping == "early_stopping" and best_state is not None:
        model.load_state_dict(best_state)
    elif stopping not in ("fixed_steps", "early_stopping"):
        raise ValueError(stopping)
    model.eval()
    # data_passes < 1: the model saw only the first steps * batch_size samples of the first permutation
    return model, {"fit_seconds": time.time() - t0, "steps": steps, "epochs": epoch,
                   "data_passes": steps * batch_size / len(X),
                   "best_epoch": best_epoch if stopping == "early_stopping" else None}


def _nn_proba(model, X):
    import torch
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(np.asarray(X, np.float32))), 1).numpy()


# ---------------------------------------------------------------------------------------------------------------
# parameter fingerprints for tree / kernel models (deterministic learners: identical parameters are a real risk,
# e.g. xgboost without subsampling gives the same model for every seed). Not computed for torch models, whose
# initialisation always differs between seeds; the output check in stages.train covers all model classes.
def fingerprint(model: Any, kind: str) -> str | None:
    import hashlib
    import json
    if kind == "torch":
        return None
    h = hashlib.sha256()
    if kind == "xgboost":
        learner = json.loads(model.get_booster().save_raw("json"))["learner"]
        learner.pop("attributes", None)  # metadata, not parameters
        h.update(json.dumps(learner, sort_keys=True).encode())
    elif hasattr(model, "estimators_"):  # forests
        from .treeio import export_forest
        for k, v in sorted(export_forest(model).items()):
            h.update(k.encode()); h.update(np.asarray(v).tobytes())
    elif hasattr(model, "support_vectors_"):  # SVC
        for v in (model.support_vectors_, model.dual_coef_, model.intercept_):
            h.update(np.asarray(v).tobytes())
    else:
        import pickle
        h.update(pickle.dumps(model))
    return h.hexdigest()


MODEL_CLASSES = {
    "rf": ModelClass("rf", "sklearn", _rf_build, _sk_fit, _sk_proba, _sk_resolve(_rf_build)),
    "svm": ModelClass("svm", "sklearn", _svm_build, _sk_fit, _sk_proba, _sk_resolve(_svm_build)),
    "xgb": ModelClass("xgb", "xgboost", _xgb_build, _sk_fit, _sk_proba, _sk_resolve(_xgb_build)),
    "nn": ModelClass("nn", "torch", _nn_build, _nn_fit, _nn_proba, _nn_resolve),
}


def get(name: str) -> ModelClass:
    if name not in MODEL_CLASSES:
        raise KeyError(f"unknown model class {name!r}; available: {sorted(MODEL_CLASSES)}")
    return MODEL_CLASSES[name]
