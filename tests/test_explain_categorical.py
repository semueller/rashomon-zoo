"""Explainers on inputs with a categorical column (category codes): ks_shap with all training rows as background,
lime_tabular with categorical_features."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from rashomon_zoo.explain import compute


@pytest.fixture(scope="module")
def coded():
    rng = np.random.default_rng(0)
    n = 300
    X = np.c_[rng.normal(size=n), rng.integers(0, 3, n), rng.normal(size=n)].astype(np.float32)  # column 1: codes
    y = (X[:, 1] == 2).astype(int)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return X, rf


def test_ks_shap_train_background(coded):
    X, rf = coded
    a, info = compute(rf, "rf", "ks_shap", X[:10], {"baseline": "train", "n_samples": 200}, X_background=X[:100],
                      return_info=True)
    assert a.shape == (10, 3)
    assert np.allclose(info["gap"], 0, atol=1e-5)  # efficiency against the mean output over the background rows
    assert np.all(np.abs(a[:, 1]) >= np.abs(a[:, [0, 2]]).max(1))  # the code column decides the class
    with pytest.raises(ValueError):
        compute(rf, "rf", "ks_shap", X[:2], {"baseline": "median"}, X_background=X)


def test_lime_categorical_features(coded):
    X, rf = coded
    a = compute(rf, "rf", "lime_tabular", X[:10], {"categorical_features": [1], "n_samples": 1000}, X_background=X)
    assert a.shape == (10, 3)
    assert np.all(np.abs(a[:, 1]) > np.abs(a[:, [0, 2]]).max(1))
    with pytest.raises(ValueError):
        compute(rf, "rf", "lime_tabular", X[:2], {"categorical_features": [1], "sample_scale": 2.0}, X_background=X)
