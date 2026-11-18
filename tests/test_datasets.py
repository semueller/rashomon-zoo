"""Loaders that keep nominal columns (abalone_full, autouniv_full). Need the OpenML cache in RASHOMON_ZOO_DATASETS
(skipped otherwise; the loaders would fetch from the network)."""
from pathlib import Path

import numpy as np
import pytest

from rashomon_zoo import datasets, tasks

CACHE = Path(datasets.DATA_ROOT) / "openml" / "openml.org" / "api" / "v1" / "json" / "data"
needs = lambda *ids: pytest.mark.skipif(not all((CACHE / f"{i}.gz").exists() for i in ids),
                                        reason="OpenML cache not found in RASHOMON_ZOO_DATASETS")


def _check_features(raw, n_cols):
    f = raw["features"]
    assert len(f["feature_names"]) == len(f["feature_types"]) == len(f["category_names"]) == n_cols
    for i, (t, cats) in enumerate(zip(f["feature_types"], f["category_names"])):
        if t == "categorical":  # codes 0..k-1 in the order of the category names, every value occurring
            assert np.array_equal(np.unique(raw["X"][:, i]), np.arange(len(cats)))
        else:
            assert cats is None


@needs(1557)
def test_abalone_full():
    raw = tasks.raw_data("abalone_full")
    assert raw["X"].shape == (4177, 8) and raw["class_names"] == ["1", "2", "3"]
    _check_features(raw, 8)
    f = raw["features"]
    assert f["feature_types"] == ["categorical"] + ["numeric"] * 7 and f["category_names"][0] == ["F", "I", "M"]


@needs(1553)
def test_autouniv_full_extends_autouniv():
    full, cut = tasks.raw_data("autouniv_full"), tasks.raw_data("autouniv")
    assert full["X"].shape == (700, 12)
    _check_features(full, 12)
    cat = [i for i, t in enumerate(full["features"]["feature_types"]) if t == "categorical"]
    assert cat == [4, 5, 7, 9] and [len(full["features"]["category_names"][i]) for i in cat] == [2, 2, 3, 3]
    # same samples, labels and class names; the truncated loader's columns are a subset
    assert np.array_equal(full["X"][:, [0, 1, 2, 6, 10]], cut["X"]) and np.array_equal(full["y"], cut["y"])
    assert full["class_names"] == cut["class_names"] and cut["features"] is None


@needs(1553)
def test_features_recorded_in_task(tmp_path):
    t = tasks.make_task(tmp_path, "autouniv_full", splits=(0.6, 0.2, 0.2))
    from rashomon_zoo.store import load_task
    back = load_task(tmp_path, "autouniv_full")
    assert back.info["features"] == t.info["features"] and back.info["features"]["feature_types"][4] == "categorical"


@needs(31, 21, 1461)
@pytest.mark.parametrize("name,shape,n_cat,classes", [
    ("creditg", (1000, 20), 13, ["bad", "good"]),
    ("car", (1728, 6), 6, ["acc", "good", "unacc", "vgood"]),
    ("bankmarketing", (45211, 16), 9, ["1", "2"])])
def test_nominal_openml_loaders(name, shape, n_cat, classes):
    raw = tasks.raw_data(name)
    assert raw["X"].shape == shape and raw["class_names"] == classes
    _check_features(raw, shape[1])
    assert raw["features"]["feature_types"].count("categorical") == n_cat
