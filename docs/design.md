# rashomon-zoo: design

rashomon-zoo produces many equally good black boxes per task (Rashomon sets) and their local explanations,
stores them, and serves them to analyses.
It is a model and explanation factory: it does not analyse. Analyses (e.g. rashomon-analysis) read the stored
artefacts; the storage layout below is the interface.

## Stages
Each stage reads and writes stored artefacts only, so stages can be rerun independently.

1. `task`: fixes which samples exist and which split they are in; writes `split.npz` and `task.json` (incl. a
   content hash). Splits: `train` (fits the black boxes, nothing else) and two held-out sets `holdout_1`,
   `holdout_2` without a built-in role; every later stage names the held-out set it uses and for what, and records
   it. `splits` has no default: fractions `(train, holdout_1, holdout_2)`, or absolute sizes
   `{"holdout_1": n, "holdout_2": m}` with train taking the rest; `max_samples` caps the dataset before splitting.
   Every draw is stratified by class and seeded (order holdout_1, holdout_2, train); standardisation is fit on the
   training split. A provider test file is kept as `holdout_1` (an absolute holdout_1 size draws from it).
   `splits="legacy"` reproduces the splits of the earlier studies (legacy loaders, data seed): loader test ->
   `holdout_1` (paper R's reference set, paper C's CFIRE input set), loader val -> `holdout_2` (paper C's rule
   evaluation set); the loader's cap on its val split is recorded. Existing tasks are not replaced unless asked.
   This is the only place where data sizes are decided: `train` uses the training split as is, `explain` explains
   whole splits (any split, recorded). Runs record the task hash and check it on load.
2. `tune` (optional, all model classes): hyperparameter search once per task and model class, scored on a named
   held-out set (`score_split`, no default); writes `tuning.csv` and the chosen configuration.
3. `train`: one model class with fixed hyperparameters (given, or taken from `tune`), many seeds; writes models,
   class probabilities on both held-out sets and accuracies (`holdout_1_acc`, `holdout_2_acc`).
4. `select`: marks the Rashomon set with a recorded criterion (default: paper R, all models within 5 percentage
   points of the best model's accuracy on `holdout_1`); writes `selection.json`.
5. `explain`: an explainer with its hyperparameters on a named split for (a subset of) the models.
Analyses are separate projects that read stored artefacts only and record the Rashomon criterion they relied on.

## Storage layout
Few large files per run instead of one file per model and explainer: dense arrays are stacked over model rows and
opened memory-mapped, tables are Parquet, models are one file per model.

```
<zoo_dir>/<task>/
  task.json, split.npz
  <model_class>/<run>/                 # e.g. xgb/default, nn/tuned
    manifest.json                      # configuration, seeds, package versions, git commit, date
    tuning.csv                         # optional (tune stage)
    models/<model_id>.<ext>            # .ubj (xgboost native), .pt (torch state dict), .pkl (scikit-learn)
    models.parquet                     # one row per model: model_id, seed, step (final = -1), parent_id, accuracies,
                                       # in_rashomon_set, ...
    outputs_<split>.npy                # (n_models, n_samples, n_classes) per held-out set, row order = models.parquet
    selection.json                     # Rashomon criterion and selected models
    stats.json                         # basic statistics of the run (all models and Rashomon set; see below)
    explanations/<name>/manifest.json  # explainer, hyperparameters, model ids, sample indices
    explanations/<name>/<split>.npy    # (n_models, n_samples, n_features); global.npy for global importances
```

- Arrays are indexed by model row, not by seed, so intermediate checkpoints and fine-tuned models fit as extra rows.
- Memory-mapped arrays: only the slices an analysis touches are read from disk; nothing has to fit in RAM.
- Tables can be queried across tasks and runs with pandas or DuckDB (`SELECT ... FROM '<zoo_dir>/*/*/*/models.parquet'`).

## Run statistics
`stages.describe` (called by `train` and `select`) writes `stats.json`: number of models and distinct models, accuracy
(mean, std, min, max), pairwise prediction disagreement (share of samples on which two models predict different
classes; mean, max) and mean pairwise Jensen-Shannon distance between output distributions, on both held-out sets, for all
models and for the Rashomon set. Explanation-based statistics belong to analyses.

## Model classes
- `nn`: feed-forward ReLU networks (logits out; probabilities via softmax), Adam, cross-entropy. Default Rashomon
  setup: models differ only in their initialisation (model seed); same data, same batch order (`order_seed`), same
  number of training steps (`n_steps`). Early stopping is an option (`stopping="early_stopping"`, with the held-out
  set it monitors given as `stopping_split`; that set then informs the models).
- `rf`, `xgb`, `svm`: scikit-learn / xgboost estimators; the seed is the estimator's `random_state`. Default
  (`variation="seed"`, all model classes): every model gets the same training data, the variation comes from the
  learner's own seeded randomness (rf: bootstrap and feature sampling; xgb: its row subsampling, e.g.
  `subsample=0.9`, each tree sees 90 %, the model all rows). xgboost without `subsample` / `colsample_* < 1` and
  SVC are deterministic: their models are identical across seeds. `train(variation="subsample" | "resample")`
  varies the data per model instead (a relaxation: models no longer see the same data). `train` warns when models
  of a run are identical: same learned parameters for tree and kernel models (`param_fingerprint`; xgboost: trees
  without metadata), repeated seeds for torch models (`duplicate_of`, `n_distinct_models`). Identical behaviour of
  models with different parameters is not an error (it is what a Rashomon set consists of); it is recorded as
  information (`same_outputs_as`, `n_distinct_outputs`).
- `nn` with fixed steps must see every training sample at least once (`n_steps * batch_size >= n_train`, else
  `train` raises); `data_passes` in models.parquet records how often.
- Parameters: the run manifest records the values used (`params`: given values plus the defaults filled in by
  rashomon-zoo; `defaults_used` names the filled-in keys). Library defaults of xgboost / scikit-learn are not
  repeated, the model files carry them. Unknown keys, and nn keys that do not apply to the chosen stopping rule,
  raise before training.
- Model files: torch state dict (architecture in the manifest), xgboost native format; scikit-learn pickles plus
  the scikit-learn version in the manifest. Pickles load reliably only with the same scikit-learn version, so models
  can be retrained from their manifest, and forests can be stored as version-independent tree arrays (`treeio`).

## Explainers
Explanations refer to the class each model predicts and, by default, to class probabilities (`output="proba"`),
which keeps attribution scales comparable across explainers and model classes; `output="logits"` for torch models.

- Gradient-based (torch models): `vg`, `sg`, `ig` (Integrated Gradients with the (x - baseline) factor).
- KernelSHAP in two implementations:
  - `ks_captum` (captum): single baseline; empty and full coalitions first with a large weight (approximate
    efficiency); other coalitions drawn i.i.d. from the Shapley size distribution, unpaired, weight 1.
  - `ks_shap` (shap `KernelExplainer`): one baseline (training mean) by default; exact efficiency; subset sizes
    enumerated where the budget allows, complements paired. `l1_reg=False` (shap's default keeps 10 features).
- LIME in two implementations:
  - `lime_captum` (captum): binary on / off representation against a baseline, exponential kernel on cosine
    distance, Lasso surrogate.
  - `lime_tabular` (lime): regression on perturbed samples around the input (`discretize_continuous=False`,
    `sample_around_instance=True`; both differ from lime's defaults), all features. Returns the coefficients of
    the local linear model (slopes in standardised units), not contributions relative to a baseline.
- Tree models: `shap_tree` (TreeSHAP), `builtin` (impurity / gain importance, global), `permutation` (global).
- Every explainer with randomness has a `seed` parameter (set locally, recorded in the manifest); repeated runs
  with different seeds give the run-to-run variation for stability studies.
- Quality measures stored next to the attributions (`<split>_info.npz`): completeness gap for baseline methods
  (ig, KernelSHAP, TreeSHAP, captum LIME), surrogate fit for lime_tabular. KernelSHAP enforces efficiency, so its gap
  checks the implementation, not convergence.
- `explain(models=...)`: "all", "rashomon" or a list of model ids; `Run.load_model(id)` for use with `compute`.
- Parameter variants need no new code: `explain(run, "ks_shap", params={...}, out_name="ks_shap_variant")`.
- `explain.compute(model, model_class, explainer, X, params)` computes explanations without storing them, for
  experiments (repeated runs, budget or baseline studies) in analysis projects; results of such studies come back to
  the zoo only as parameters of `explain` (recorded in the explainer manifest).

## Planned extensions
- Checkpoints: keep intermediate checkpoints as model rows (`step`) and explain all or a filtered subset
  (explanations during training).
- Fine-tuning and continual learning: a `derive` stage trains further from stored models; derived models are rows
  with `parent_id` (lineage) and a reference to their training data.
- Size: with many checkpoints the arrays grow (e.g. 100 models x 200 checkpoints x 2000 samples x 23 features is
  about 3.6 GB per explainer, read in slices); options: reference subsets of samples, selected checkpoints, chunked
  storage (Zarr), float16 for rank-based analyses.
- Text tasks with current packages, as an optional extra.
- Decision-boundary reference points: the closest point on the decision boundary as a well-defined notion of
  "local" for perturbation-based explainers; exact for ReLU networks (piecewise linear) and tree ensembles via
  mixed-integer programming, search otherwise.
