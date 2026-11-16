"""Version-independent storage of scikit-learn tree models (numpy only).

export_forest / save_forest: tree arrays of a fitted RandomForestClassifier -> .npz
load_forest: rebuild a RandomForestClassifier in the installed scikit-learn version.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

import numpy as np

_FIELDS = ("left_child", "right_child", "feature", "threshold", "impurity", "n_node_samples",
           "weighted_n_node_samples")


def export_forest(rf: Any) -> dict:
    out = {"classes": np.asarray(rf.classes_), "n_features": np.int64(rf.n_features_in_),
           "n_estimators": np.int64(len(rf.estimators_))}
    for i, est in enumerate(rf.estimators_):
        st = est.tree_.__getstate__()
        nodes = st["nodes"]
        for f in _FIELDS:
            out[f"t{i}_{f}"] = np.asarray(nodes[f])
        if "missing_go_to_left" in nodes.dtype.names:
            out[f"t{i}_missing_go_to_left"] = np.asarray(nodes["missing_go_to_left"])
        out[f"t{i}_value"] = np.asarray(st["values"], dtype=np.float64)
        out[f"t{i}_max_depth"] = np.int64(st["max_depth"])
    return out


def save_forest(rf: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **export_forest(rf))


def load_forest(path: Path) -> RandomForestClassifier:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree._tree import Tree
    z = np.load(path)
    classes = z["classes"]
    n_classes = np.array([len(classes)], dtype=np.intp)
    n_features = int(z["n_features"])
    rf = RandomForestClassifier(n_estimators=int(z["n_estimators"]))
    ests = []
    for i in range(int(z["n_estimators"])):
        tree = Tree(n_features, n_classes, 1)
        dtype = tree.__getstate__()["nodes"].dtype  # node layout of the installed version
        n = len(z[f"t{i}_left_child"])
        nodes = np.zeros(n, dtype=dtype)
        for f in dtype.names:
            key = f"t{i}_{f}"
            if key in z:
                nodes[f] = z[key]
            elif f == "missing_go_to_left":
                nodes[f] = 0
            else:
                raise KeyError(f"tree field {f} not in {path}")
        value = z[f"t{i}_value"]
        # scikit-learn < 1.4 stores weighted class counts, >= 1.4 class fractions (and predict_proba no longer
        # normalises): store fractions, which give the same predict_proba in both
        tot = value.sum(axis=-1, keepdims=True)
        value = np.divide(value, tot, out=np.zeros_like(value), where=tot > 0)
        tree.__setstate__({"max_depth": int(z[f"t{i}_max_depth"]), "node_count": n, "nodes": nodes,
                           "values": np.ascontiguousarray(value)})
        est = DecisionTreeClassifier()
        est.tree_ = tree
        est.n_outputs_ = 1
        est.n_classes_ = int(n_classes[0])
        est.classes_ = classes
        est.n_features_in_ = n_features
        ests.append(est)
    rf.estimators_ = ests
    rf.classes_ = classes
    rf.n_classes_ = int(n_classes[0])
    rf.n_outputs_ = 1
    rf.n_features_in_ = n_features
    return rf
