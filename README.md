# rashomon-zoo

Many equally good black boxes per task (Rashomon sets) and their local explanations, stored for analysis.
A model and explanation factory; analyses live in separate projects (e.g. rashomon-analysis) that read the stored data.
Design and storage layout: `docs/design.md`.

## Use (Python API)
```python
import rashomon_zoo
from rashomon_zoo import tasks, stages
from rashomon_zoo.explain import explain

zoo_dir = "zoo"                                   # output folder
tasks.make_task(zoo_dir, "heloc", splits={"holdout_1": 500, "holdout_2": 300})  # or splits="legacy" (papers R / C)
best = stages.tune(zoo_dir, "heloc", "nn", grid={"hidden": [[32], [64, 64]], "lr": [1e-3, 3e-3]},
                   score_split="holdout_2")   # optional
run = stages.train(zoo_dir, "heloc", "nn", "tuned", seeds=range(50), tuned="tuning")
stages.select(run)                             # Rashomon set: paper R criterion, recorded in selection.json
explain(run, "ig", split="holdout_1", models="rashomon")
E = run.explanations("ig", "holdout_1")        # (n_models, n_samples, n_features), memory-mapped
```
Raw dataset files: set `RASHOMON_ZOO_DATASETS` (folders HELOC/, GMSC/, ...).

Model classes: `nn` (torch MLP), `rf`, `xgb`, `svm`.
For experiments: `rashomon_zoo.explain.compute(model, model_class, explainer, X, params)` returns explanations
without storing them.
Explainers: `vg`, `sg`, `ig` (Integrated Gradients with the (x - baseline) factor);
`ks_captum`, `lime_captum` (captum); `ks_shap` (shap KernelExplainer, one baseline),
`lime_tabular` (lime package, sampling around the input), `shap_tree` (extras `shap`, `lime`); `builtin`,
`permutation` (global). Parameter variants: `explain(run, "ks_shap", params={...}, out_name="ks_shap_variant")`;
see the notes at `DEFAULTS` in `explain.py` for how the implementations differ.

## macOS
torch and xgboost load different OpenMP runtimes; importing `rashomon_zoo` sets `OMP_NUM_THREADS=1` unless set.
