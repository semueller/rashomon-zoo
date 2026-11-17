"""Small end-to-end tests on a synthetic task (no dataset files needed)."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import rashomon_zoo  # noqa: F401  (sets OMP_NUM_THREADS on macOS)
from rashomon_zoo import stages
from rashomon_zoo.explain import explain
from rashomon_zoo.store import Task, save_task
from rashomon_zoo.treeio import load_forest, save_forest


@pytest.fixture()
def zoo_dir(tmp_path):
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    idx = {"train": slice(0, 400), "holdout_2": slice(400, 500), "holdout_1": slice(500, 600)}
    save_task(tmp_path, Task("toy", {k: X[v] for k, v in idx.items()}, {k: y[v] for k, v in idx.items()}, 2, {}))
    return tmp_path


@pytest.mark.parametrize("mc,params", [("xgb", {"n_estimators": 20}), ("rf", {"n_estimators": 20}),
                                       ("nn", {"hidden": [16], "n_steps": 200})])
def test_train_select_explain(zoo_dir, mc, params):
    r = stages.train(zoo_dir, "toy", mc, "t", seeds=range(3), params=params)
    df = stages.select(r)
    assert len(df) == 3 and df["in_rashomon_set"].any()
    assert r.outputs("holdout_1").shape == (3, 100, 2)
    names = ["ks_captum", "ks_shap", "lime_tabular"] + (["ig"] if mc == "nn" else ["builtin"])
    for n in names:
        explain(r, n, split="holdout_1", params={"n_samples": 500} if n == "lime_tabular" else None)
    for n in ("ks_captum", "ks_shap", "lime_tabular"):
        assert r.explanations(n, "holdout_1").shape == (3, 100, 6)


def test_ig_completeness(zoo_dir):
    """True IG (zero baseline): attributions sum to f(x) - f(0) for the predicted class (logits)."""
    import torch
    from rashomon_zoo import modelzoo
    from rashomon_zoo.store import load_model
    r = stages.train(zoo_dir, "toy", "nn", "t", seeds=[0], params={"hidden": [16], "n_steps": 200})
    explain(r, "ig", split="holdout_1", params={"n_samples": 400, "output": "logits"})
    a = r.explanations("ig", "holdout_1")[0]
    man, ex = r.manifest(), r.explainer_manifest("ig")
    mc = modelzoo.get("nn")
    m = load_model(r.model_path(0), "torch", lambda: mc.build(man["params"], 6, 2))
    from rashomon_zoo.store import load_task
    X = torch.from_numpy(load_task(zoo_dir, "toy").X["holdout_1"][ex["sample_indices"]])
    with torch.no_grad():
        lg, lg0 = m(X), m(torch.zeros_like(X))
    t = lg.argmax(1)
    diff = (lg.gather(1, t[:, None]) - lg0.gather(1, t[:, None])).squeeze(1).numpy()
    assert np.allclose(a.sum(1), diff, atol=0.05 * np.abs(diff).max() + 1e-3)


def test_forest_roundtrip(tmp_path):
    X, y = make_classification(n_samples=300, n_features=5, random_state=1)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    save_forest(rf, tmp_path / "rf.npz")
    assert np.allclose(load_forest(tmp_path / "rf.npz").predict_proba(X), rf.predict_proba(X))


def test_ks_shap_efficiency(zoo_dir):
    """shap KernelSHAP with one baseline: attributions sum exactly to f(x) - f(baseline) for the predicted class."""
    from rashomon_zoo import modelzoo
    from rashomon_zoo.store import load_model, load_task
    r = stages.train(zoo_dir, "toy", "xgb", "t", seeds=[0], params={"n_estimators": 20})
    explain(r, "ks_shap", split="holdout_1")
    a = r.explanations("ks_shap", "holdout_1")[0]
    t = load_task(zoo_dir, "toy")
    mc = modelzoo.get("xgb")
    m = load_model(r.model_path(0), "xgboost", lambda: mc.build(r.manifest()["params"], 6, 2))
    X = t.X["holdout_1"][r.explainer_manifest("ks_shap")["sample_indices"]]
    p, p0 = mc.proba(m, X), mc.proba(m, t.X["train"].mean(0, keepdims=True))
    k = p.argmax(1)
    assert np.allclose(a.sum(1), p[np.arange(len(X)), k] - p0[0, k], atol=1e-5)


def test_nn_rashomon_setup(zoo_dir):
    """same seed -> identical model; different seeds -> different models with the same data order and length"""
    import torch
    r1 = stages.train(zoo_dir, "toy", "nn", "a", seeds=[0, 1], params={"hidden": [8], "n_steps": 50})
    r2 = stages.train(zoo_dir, "toy", "nn", "b", seeds=[0], params={"hidden": [8], "n_steps": 50})
    o1, o2 = np.asarray(r1.outputs("holdout_1")), np.asarray(r2.outputs("holdout_1"))
    assert np.array_equal(o1[0], o2[0])            # reproducible from the seed
    assert not np.array_equal(o1[0], o1[1])        # seeds differ in initialisation
    assert (r1.models()["steps"] == 50).all()      # same training length


def test_duplicate_detection_and_variation(zoo_dir):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = stages.train(zoo_dir, "toy", "xgb", "det", seeds=range(3), params={"n_estimators": 10}, variation="seed")
    assert r.manifest()["n_distinct_models"] == 1 and r.manifest()["n_distinct_outputs"] == 1
    assert any("distinct" in str(x.message) for x in w)
    m = r.models()
    assert list(m["duplicate_of"]) == [-1, 0, 0] and list(m["same_outputs_as"]) == [-1, 0, 0]
    assert m["param_fingerprint"].nunique() == 1
    rn = stages.train(zoo_dir, "toy", "nn", "fp", seeds=[0, 0, 1], params={"hidden": [8], "n_steps": 20})
    mn = rn.models()
    assert list(mn["duplicate_of"]) == [-1, 0, -1] and list(mn["same_outputs_as"]) == [-1, 0, -1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r2 = stages.train(zoo_dir, "toy", "nn", "fp2", seeds=[0, 1], params={"hidden": [8], "n_steps": 20})
    assert r2.manifest()["n_distinct_models"] == 2 and not any("distinct" in str(x.message) for x in w)
    for v in ("resample", "subsample"):
        rv = stages.train(zoo_dir, "toy", "xgb", v, seeds=range(3), params={"n_estimators": 10}, variation=v)
        assert rv.manifest()["n_distinct_models"] == 3


def test_describe_and_default_variation(zoo_dir):
    # default: same data for all models; xgboost's own row subsampling makes the models differ
    r = stages.train(zoo_dir, "toy", "xgb", "dv", seeds=range(3), params={"n_estimators": 10, "subsample": 0.9})
    m = r.manifest()
    assert m["variation"] == "seed" and m["resample_fraction"] is None and m["n_distinct_models"] == 3
    stages.select(r)
    st = __import__("json").loads((r.dir / "stats.json").read_text())
    assert st["all"]["n_models"] == 3 and "rashomon_set" in st
    assert 0 <= st["all"]["holdout_1"]["prediction_disagreement_mean"] <= 1 and st["all"]["holdout_1"]["output_jsd_mean"] >= 0


def test_compute_matches_explain(zoo_dir):
    from rashomon_zoo import explain as EX, modelzoo
    from rashomon_zoo.store import load_model, load_task
    r = stages.train(zoo_dir, "toy", "xgb", "cmp", seeds=[0], params={"n_estimators": 10})
    explain(r, "shap_tree", split="holdout_1")
    t, man, ex = load_task(zoo_dir, "toy"), r.manifest(), r.explainer_manifest("shap_tree")
    mc = modelzoo.get("xgb")
    m = load_model(r.model_path(0), "xgboost", lambda: mc.build(man["params"], 6, 2))
    a = EX.compute(m, "xgb", "shap_tree", t.X["holdout_1"][ex["sample_indices"]], X_background=t.X["train"])
    assert np.allclose(a, r.explanations("shap_tree", "holdout_1")[0])


def test_nn_settings_take_effect(zoo_dir):
    """non-default nn settings are recorded (with the defaults actually used) and take effect in training"""
    import torch
    from unittest import mock
    from rashomon_zoo import modelzoo
    from rashomon_zoo.store import load_model
    params = {"hidden": [7, 5], "n_steps": 13, "batch_size": 60, "lr": 0.0123, "order_seed": 3}
    seen = {"lr": [], "batches": []}
    orig_adam = torch.optim.Adam

    def adam(p, lr, weight_decay):
        seen["lr"].append(lr)
        return orig_adam(p, lr=lr, weight_decay=weight_decay)

    orig_ce = torch.nn.CrossEntropyLoss

    class CE(orig_ce):
        def forward(self, a, b):
            seen["batches"].append(len(b))
            return super().forward(a, b)

    with mock.patch("torch.optim.Adam", adam), mock.patch("torch.nn.CrossEntropyLoss", CE):
        r = stages.train(zoo_dir, "toy", "nn", "t", seeds=[0, 1], params=params)
    man = r.manifest()
    # values used: given + defaults, minus the early-stopping-only keys; defaults_used names the filled-in keys
    assert man["params"] == {k: v for k, v in {**modelzoo.NN_DEFAULTS, **params}.items()
                             if k not in ("max_epochs", "patience", "stopping_split")}
    assert man["defaults_used"] == ["stopping", "weight_decay"]
    assert seen["lr"] == [0.0123, 0.0123]
    assert len(seen["batches"]) == 2 * 13
    assert seen["batches"][:7] == [60] * 6 + [40]  # 400 training rows per epoch: 6 full batches + the rest
    assert (r.models()["steps"] == 13).all()
    m = load_model(r.model_path(0), "torch", lambda: modelzoo.get("nn").build(man["params"], 6, 2))
    shapes = [tuple(v.shape) for k, v in m.state_dict().items() if k.endswith("weight")]
    assert shapes == [(7, 6), (5, 7), (2, 5)]


def test_same_batch_order_across_models(zoo_dir):
    """Rashomon setup: models of a run see the same batches in the same order (order_seed), whatever their seed"""
    import torch
    from unittest import mock
    batches = []
    orig_ce = torch.nn.CrossEntropyLoss

    class CE(orig_ce):
        def forward(self, a, b):
            batches.append(b.numpy().copy())
            return super().forward(a, b)

    with mock.patch("torch.nn.CrossEntropyLoss", CE):
        stages.train(zoo_dir, "toy", "nn", "t", seeds=[0, 1], params={"hidden": [4], "n_steps": 20})
    a, b = batches[:20], batches[20:]
    assert len(a) == len(b) == 20 and all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("mc,params", [("nn", {"n_step": 10}), ("nn", {"max_epochs": 5}),
                                       ("nn", {"stopping": "early_stopping", "n_steps": 10}),
                                       ("nn", {"stopping": "early"}), ("xgb", {"n_estimator": 10}),
                                       ("rf", {"max_dept": 3})])
def test_bad_params_raise(zoo_dir, mc, params):
    with pytest.raises(ValueError):
        stages.train(zoo_dir, "toy", mc, "t", seeds=[0], params=params)


def test_effective_params_recorded_sklearn(zoo_dir):
    r = stages.train(zoo_dir, "toy", "xgb", "t", seeds=[0], params={"n_estimators": 5, "max_depth": 2})
    man = r.manifest()
    # given + what modelzoo sets on top of xgboost's defaults; library defaults live in the model file
    assert man["params"] == {"n_estimators": 5, "max_depth": 2, "eval_metric": "logloss"}
    assert man["defaults_used"] == ["eval_metric"]


def test_task_overwrite_guard_and_hash(zoo_dir):
    from rashomon_zoo.store import Run, load_task
    t = load_task(zoo_dir, "toy")
    with pytest.raises(FileExistsError):
        save_task(zoo_dir, t)
    r = stages.train(zoo_dir, "toy", "xgb", "h", seeds=[0], params={"n_estimators": 5})
    r.load_task()  # unchanged: passes
    t.X["train"] = t.X["train"] + 1.0
    save_task(zoo_dir, t, overwrite=True)
    with pytest.raises(ValueError, match="task_hash"):
        Run(zoo_dir, "toy", "xgb", "h").load_task()


def test_split_specification_breast_cancer(tmp_path):
    """fractions, absolute sizes and the cap: sizes as specified, stratified, recorded, disjoint, standardised on train"""
    from rashomon_zoo.tasks import make_task, raw_data
    raw = raw_data("breastcancer")
    p_raw = np.bincount(raw["y"]) / len(raw["y"])
    t = make_task(tmp_path, "breastcancer", splits=(0.6, 0.2, 0.2), seed=1)
    assert {s: len(t.y[s]) for s in ("train", "holdout_2", "holdout_1")} == {"train": 341, "holdout_2": 114, "holdout_1": 114}
    ind = t.info["indices"]
    assert not (set(ind["train"]) & set(ind["holdout_2"]) or set(ind["train"]) & set(ind["holdout_1"]) or set(ind["holdout_2"]) & set(ind["holdout_1"]))
    for s in ("train", "holdout_2", "holdout_1"):
        assert np.allclose(np.bincount(t.y[s]) / len(t.y[s]), p_raw, atol=0.01)
        assert np.array_equal(t.y[s], raw["y"][ind[s]])
    assert np.allclose(t.X["train"].mean(0), 0, atol=1e-5)
    t2 = make_task(tmp_path, "breastcancer", splits={"holdout_2": 50, "holdout_1": 60}, max_samples=300, task_name="bc_small")
    assert {s: len(t2.y[s]) for s in ("train", "holdout_2", "holdout_1")} == {"train": 190, "holdout_2": 50, "holdout_1": 60}
    assert t2.info["max_samples"] == 300 and t2.info["dataset"] == "breastcancer"
    with pytest.raises(FileExistsError):
        make_task(tmp_path, "breastcancer", splits=(0.6, 0.2, 0.2))
    for bad in ({"train": 100, "holdout_1": 50}, {"holdout_2": 50}, "random", (0.5, 0.5)):
        with pytest.raises((ValueError, TypeError)):
            make_task(tmp_path, "breastcancer", splits=bad, task_name="bad")


def test_legacy_splits_breast_cancer(tmp_path):
    from rashomon_zoo.tasks import make_task
    t = make_task(tmp_path, "breastcancer", splits="legacy")
    assert {s: len(t.y[s]) for s in ("train", "holdout_2", "holdout_1")} == {"train": 285, "holdout_2": 170, "holdout_1": 114}
    assert t.info["holdout_2_cap_by_loader"] == 300 and t.info["holdout_2_capped"] is False
    with pytest.raises(ValueError):
        make_task(tmp_path, "breastcancer", splits="legacy", max_samples=100, task_name="x")


def test_explain_train_split(zoo_dir):
    r = stages.train(zoo_dir, "toy", "xgb", "tr", seeds=[0], params={"n_estimators": 5})
    explain(r, "shap_tree", split="train")
    assert r.explainer_manifest("shap_tree")["split"] == "train"


@pytest.mark.parametrize("kw,msg", [
    ({"splits": {"holdout_2": 100, "holdout_1": 600}}, "holdout_1: 600 requested, 569 available"),
    ({"splits": {"holdout_2": 500, "holdout_1": 69}}, "holdout_2 .after holdout_1: 500 left.: 500 requested"),
    ({"splits": {"holdout_2": 0, "holdout_1": 50}}, "positive integer"),
    ({"splits": {"holdout_2": 10.5, "holdout_1": 50}}, "positive integer"),
    ({"splits": (0.6, 0.4, 0.0)}, "positive"),
    ({"splits": (0.6, 0.2, 0.2), "max_samples": 1000}, "max_samples"),
    ({"splits": {"holdout_2": 1, "holdout_1": 50}}, "stratified by class"),
])
def test_split_sizes_out_of_range(tmp_path, kw, msg):
    from rashomon_zoo.tasks import make_task
    with pytest.raises(ValueError, match=msg):
        make_task(tmp_path, "breastcancer", **kw)
    assert not (tmp_path / "breastcancer").exists()  # nothing written


def test_nn_must_see_every_training_sample(zoo_dir):
    with pytest.raises(ValueError, match="every training sample"):
        stages.train(zoo_dir, "toy", "nn", "short", seeds=[0], params={"hidden": [4], "n_steps": 5, "batch_size": 64})
    r = stages.train(zoo_dir, "toy", "nn", "ok", seeds=[0], params={"hidden": [4], "n_steps": 7, "batch_size": 64})
    assert r.models()["data_passes"].iloc[0] == 7 * 64 / 400


def test_explainer_quality_measures(zoo_dir):
    """gaps: ~0 for exact-efficiency methods (ks_shap, shap_tree), small for IG; LIME fit statistics stored"""
    rx = stages.train(zoo_dir, "toy", "xgb", "q", seeds=[0], params={"n_estimators": 20, "subsample": 0.9})
    rn = stages.train(zoo_dir, "toy", "nn", "q", seeds=[0], params={"hidden": [16], "n_steps": 200})
    for r, names in ((rx, ["shap_tree", "ks_shap", "lime_tabular"]), (rn, ["ig", "ks_captum", "vg"])):
        for n in names:
            explain(r, n, split="holdout_1", params={"n_samples": 500} if n == "lime_tabular" else None)
    info = lambda r, n: np.load(r.dir / "explanations" / n / "holdout_1_info.npz")  # noqa: E731
    assert np.abs(info(rx, "shap_tree")["gap"]).max() < 1e-4
    assert np.abs(info(rx, "ks_shap")["gap"]).max() < 1e-4
    ig = info(rn, "ig")
    assert np.abs(ig["gap"]).max() < 0.05 and ig["gap"].shape == (1, 100)
    li = info(rx, "lime_tabular")
    assert set(li.files) == {"f_x", "intercept", "local_r2", "local_pred", "local_pred_error"}
    assert np.all(li["local_r2"] <= 1.0)
    assert rn.explainer_manifest("ks_captum")["info"] == ["f_baseline", "f_x", "gap"]
    assert rn.explainer_manifest("vg")["info"] is None
    assert not (rn.dir / "explanations" / "vg" / "holdout_1_info.npz").exists()


def test_explainer_seeds_and_model_subsets(zoo_dir):
    """same seed -> identical explanations, different seed -> different; global random state untouched; model ids"""
    import torch
    from rashomon_zoo.explain import STOCHASTIC, compute
    r = stages.train(zoo_dir, "toy", "nn", "s", seeds=[0, 1, 2], params={"hidden": [8], "n_steps": 50})
    m, t = r.load_model(1), r.load_task()
    X = t.X["holdout_1"][:5]
    assert {"sg", "ks_captum", "lime_captum", "ks_shap", "lime_tabular", "shap_tree", "permutation"} <= STOCHASTIC
    for e in ("sg", "ks_captum", "lime_captum", "ks_shap"):
        state = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
        a = compute(m, "nn", e, X, {"seed": 3}, X_background=t.X["train"])
        b = compute(m, "nn", e, X, {"seed": 3}, X_background=t.X["train"])
        c = compute(m, "nn", e, X, {"seed": 4}, X_background=t.X["train"])
        assert np.array_equal(a, b), e
        if e != "ks_shap":  # shap enumerates all coalitions for d=6 at the default budget: deterministic
            assert not np.array_equal(a, c), e
        assert torch.equal(torch.get_rng_state(), state[0]) and np.array_equal(np.random.get_state()[1], state[1])
    explain(r, "ig", split="holdout_1", models=[0, 2])
    assert r.explainer_manifest("ig")["model_ids"] == [0, 2] and r.explanations("ig", "holdout_1").shape[0] == 2
    with pytest.raises(ValueError, match="unknown model ids"):
        explain(r, "ig", split="holdout_1", models=[7], out_name="x")


def test_held_out_roles_explicit(zoo_dir):
    """held-out sets have no built-in role: early stopping, tune, explain and permutation importance name theirs"""
    with pytest.raises(ValueError, match="stopping_split"):
        stages.train(zoo_dir, "toy", "nn", "es", seeds=[0], params={"hidden": [4], "stopping": "early_stopping"})
    r = stages.train(zoo_dir, "toy", "nn", "es", seeds=[0], params={"hidden": [4], "stopping": "early_stopping",
                                                                 "stopping_split": "holdout_2", "max_epochs": 3})
    assert r.manifest()["params"]["stopping_split"] == "holdout_2"
    with pytest.raises(ValueError, match="score_split / cv"):  # no default: the scoring data are named
        stages.tune(zoo_dir, "toy", "xgb", {"n_estimators": [5]})
    best = stages.tune(zoo_dir, "toy", "xgb", {"n_estimators": [5, 10]}, score_split="holdout_2")
    assert best["n_estimators"] in (5, 10)
    with pytest.raises(ValueError, match="split"):
        explain(r, "ig")
    rx = stages.train(zoo_dir, "toy", "xgb", "perm", seeds=[0], params={"n_estimators": 5})
    with pytest.raises(ValueError, match="permutation"):
        explain(rx, "permutation")
    explain(rx, "permutation", params={"split": "holdout_2"})
    explain(rx, "builtin")
    assert (rx.dir / "explanations" / "builtin" / "global.npy").exists()


def test_class_names_recorded(tmp_path):
    from rashomon_zoo.tasks import make_task
    t = make_task(tmp_path, "breastcancer", splits="legacy")
    assert t.info["class_names"] == ["malignant", "benign"]  # label id -> name (scikit-learn's target_names)
    t2 = make_task(tmp_path, "breastcancer", splits=(0.6, 0.2, 0.2), task_name="bc_spec")
    assert t2.info["class_names"] == ["malignant", "benign"]


def test_tune_cv(zoo_dir):
    """cv: folds inside the training split, held-out sets unused; exactly one of score_split / cv"""
    import pandas as pd
    from rashomon_zoo.store import read_json
    grid = {"n_estimators": [5, 10], "max_depth": [2]}
    best = stages.tune(zoo_dir, "toy", "xgb", grid, cv=3, name="cv")
    d = zoo_dir / "toy" / "xgb" / "_tune" / "cv"
    folds = pd.read_csv(d / "folds.csv")
    assert len(folds) == 2 * 3 and set(folds["n_fit"] + folds["n_score"]) == {400}
    man = read_json(d / "manifest.json")
    assert man["cv"] == 3 and man["score_split"] is None and man["best"] == best
    with pytest.raises(ValueError):
        stages.tune(zoo_dir, "toy", "xgb", grid)
    with pytest.raises(ValueError):
        stages.tune(zoo_dir, "toy", "xgb", grid, cv=3, score_split="holdout_1")


def test_sample_order_nested_stratified():
    from rashomon_zoo.explain import sample_order
    y = np.array([0] * 70 + [1] * 30)
    o = sample_order(y, seed=1)
    assert sorted(o) == list(range(100))
    for n in (10, 20, 50):
        assert abs((y[o[:n]] == 1).sum() - 0.3 * n) <= 1
    assert not np.array_equal(o, sample_order(y, seed=2))


def test_explain_samples_and_derive(zoo_dir):
    from rashomon_zoo.explain import derive, sample_order
    from rashomon_zoo.store import load_task
    r = stages.train(zoo_dir, "toy", "xgb", "t", seeds=range(2), params={"n_estimators": 10, "subsample": 0.9})
    explain(r, "lime_tabular", split="holdout_1", params={"n_samples": 300}, samples=30, out_name="l30")
    explain(r, "shap_tree", split="holdout_1", samples=15, out_name="t15")
    i30 = r.explainer_manifest("l30")["sample_indices"]
    i15 = r.explainer_manifest("t15")["sample_indices"]
    t = load_task(zoo_dir, "toy")
    assert i30 == sorted(sample_order(t.y["holdout_1"], 0)[:30].tolist()) and set(i15) <= set(i30)
    assert r.explanations("l30", "holdout_1").shape == (2, 30, 6)
    derive(r, "l30", "l30_x")
    X = t.X["holdout_1"][i30]
    assert np.allclose(r.explanations("l30_x", "holdout_1"), r.explanations("l30", "holdout_1") * X[None])
    assert r.explainer_manifest("l30_x")["derived_from"] == "l30"
    with pytest.raises(ValueError):
        explain(r, "shap_tree", split="holdout_1", samples=101, out_name="too_many")


def test_shap_tree_train_mean_equals_exact_ks(zoo_dir):
    """random forest (probability = mean of leaf values, no output transform): interventional TreeSHAP with the
    training mean as the only background point = exact KernelSHAP with that baseline (d = 6: shap enumerates all
    coalitions). Not so for xgboost with model_output="probability" (shap 0.54.1; see explain.DEFAULTS)."""
    r = stages.train(zoo_dir, "toy", "rf", "t", seeds=[0], params={"n_estimators": 20})
    explain(r, "shap_tree", split="holdout_1", params={"baseline": "train_mean"}, samples=20, out_name="tree_mean")
    explain(r, "ks_shap", split="holdout_1", params={"n_samples": 1000}, samples=20, out_name="ks_exact")
    a, b = r.explanations("tree_mean", "holdout_1"), r.explanations("ks_exact", "holdout_1")
    assert np.allclose(a, b, atol=1e-4)


def test_heloc_fico_loader(tmp_path, monkeypatch):
    """drops rows that are -9 everywhere, keeps the other special values, class 0 = Bad"""
    import pandas as pd
    from rashomon_zoo import tasks
    cols = ["RiskPerformance"] + [f"f{i}" for i in range(23)]
    rows = [["Bad"] + [-9] * 23, ["Bad"] + [-9] * 23]
    rng = np.random.default_rng(0)
    for i in range(40):
        r = list(rng.integers(0, 100, 23)); r[3] = -7 if i % 3 == 0 else r[3]; r[5] = -8 if i % 4 == 0 else r[5]
        rows.append(["Good" if i % 2 else "Bad"] + r)
    (tmp_path / "HELOC_FICO").mkdir()
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "HELOC_FICO" / "heloc_dataset_v1.csv", index=False)
    monkeypatch.setenv("RASHOMON_ZOO_DATASETS", str(tmp_path))
    r = tasks.raw_data("heloc_fico")
    assert r["X"].shape == (40, 23) and r["class_names"] == ["Bad", "Good"]
    assert (r["X"][:, 3] == -7).sum() == 14 and (r["X"][:, 5] == -8).sum() == 10
    assert list(r["y"][:2]) == [0, 1] and "X_test_fixed" not in r


def test_shap_tree_multiclass_xgb_clear_error(tmp_path):
    X, y = make_classification(n_samples=300, n_features=5, n_informative=4, n_redundant=0, n_classes=3, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    idx = {"train": slice(0, 200), "holdout_2": slice(200, 250), "holdout_1": slice(250, 300)}
    save_task(tmp_path, Task("toy3", {k: X[v] for k, v in idx.items()}, {k: y[v] for k, v in idx.items()}, 3, {}))
    r = stages.train(tmp_path, "toy3", "xgb", "t", seeds=[0], params={"n_estimators": 5})
    with pytest.raises(ValueError, match="multi-class"):
        explain(r, "shap_tree", split="holdout_1")
    explain(r, "shap_tree", split="holdout_1", params={"model_output": "raw"}, out_name="raw")
    assert r.explanations("raw", "holdout_1").shape == (1, 50, 5)


def test_lime_sample_scale(zoo_dir):
    """sample_scale narrows lime's perturbations; recorded in the manifest; 1.0 = package behaviour"""
    r = stages.train(zoo_dir, "toy", "nn", "t", seeds=[0], params={"hidden": [16], "n_steps": 200})
    explain(r, "lime_tabular", split="holdout_1", params={"n_samples": 500}, samples=10, out_name="w1")
    explain(r, "lime_tabular", split="holdout_1", params={"n_samples": 500, "sample_scale": 1.0}, samples=10,
            out_name="w1b")
    explain(r, "lime_tabular", split="holdout_1", params={"n_samples": 500, "sample_scale": 0.1}, samples=10,
            out_name="w01")
    assert np.allclose(r.explanations("w1", "holdout_1"), r.explanations("w1b", "holdout_1"))
    assert not np.allclose(r.explanations("w1", "holdout_1"), r.explanations("w01", "holdout_1"))
    assert r.explainer_manifest("w01")["params"]["sample_scale"] == 0.1
    # units: slopes per training standard deviation for every scale (a smooth model: similar magnitudes)
    a1, a01 = r.explanations("w1", "holdout_1"), r.explanations("w01", "holdout_1")
    ratio = np.abs(a01).mean() / np.abs(a1).mean()
    assert 0.2 < ratio < 5


def test_smoothgrad_targets_follow_samples(zoo_dir):
    """with (almost) no noise SmoothGrad equals vanilla gradients for every sample, also in batches with several
    predicted classes (the legacy code assigned other samples' targets to the noisy copies)"""
    import torch
    from rashomon_zoo import attribution as A
    r = stages.train(zoo_dir, "toy", "nn", "t", seeds=[0], params={"hidden": [16], "n_steps": 200})
    m = r.load_model(0)
    from rashomon_zoo.store import load_task
    X = torch.from_numpy(load_task(zoo_dir, "toy").X["holdout_1"][:30])
    infer = lambda x: torch.softmax(m(x), 1)  # noqa: E731
    with torch.no_grad():
        y = infer(X).argmax(1)
    assert len(torch.unique(y)) == 2
    vg = A.vanilla_grad(m, X, targets=y, inference_fn=infer)
    sg = A.smooth_grad(m, X, targets=y, inference_fn=infer, n_samples=7, noise_level=1e-7, random_state=0)
    vg, sg = (v[0] if isinstance(v, tuple) else v for v in (vg, sg))
    assert np.allclose(sg.detach().numpy(), vg.detach().numpy(), atol=1e-5)


def test_smoothgrad_noise_per_sample(zoo_dir):
    """identical samples in one batch get independent noise (different SmoothGrad values); same seed, same values"""
    import torch
    from rashomon_zoo import attribution as A
    r = stages.train(zoo_dir, "toy", "nn", "t", seeds=[0], params={"hidden": [16], "n_steps": 200})
    m = r.load_model(0)
    from rashomon_zoo.store import load_task
    x = torch.from_numpy(load_task(zoo_dir, "toy").X["holdout_1"][:1])
    X = x.repeat(2, 1)
    infer = lambda z: torch.softmax(m(z), 1)  # noqa: E731
    y = infer(X).argmax(1).detach()

    def sg(seed):
        out = A.smooth_grad(m, X, targets=y, inference_fn=infer, n_samples=20, noise_level=0.5, random_state=seed,
                            force_random=False)
        return (out[0] if isinstance(out, tuple) else out).detach().numpy()

    a = sg(0)
    assert not np.allclose(a[0], a[1])
    assert np.allclose(a, sg(0))
