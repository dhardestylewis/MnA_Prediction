# TODO: Compile-once / Train-many M&A Backtest + Stability Harness Redesign

**Status**: IN PROGRESS  
**Created**: 2026-01-30  
**Last Updated**: 2026-01-30

---

## 0) Mission

Redesign the pipeline so that **iteration-1 "one-shot AutoML" is net positive** and the **development inner loop is ≤60 seconds** for a single (quarter, horizon) fit, while preserving the existing end-to-end artifacts (per-quarter metrics, rankings, portfolio returns, benchmark comparisons) and adding a **stability certification harness** that covers:

1. rank stability,
2. **raw probability stability + calibration stability**, and
3. **returns stability** (per quarter + path-level risk).

The redesign must enforce **compile once, train many** so that **no hyperparameter trial repeats feature engineering or dataset construction**.

---

## 1) Non-negotiable requirements

### R1. Two-phase architecture (hard separation)

- [ ] **Phase A: COMPILE (slow, amortized, cached)**
  - [ ] Load raw data (Compustat quarterly/annual + deals).
  - [ ] Clean/filter (STD keyset, PIT-risk flag, safe lag/snapshot timestamp).
  - [ ] Feature engineering.
  - [ ] Multi-horizon labeling.
  - [ ] Produce a **global numeric design matrix** and **row indices by quarter**.

- [ ] **Phase B: TRAIN/EVAL (fast, repeated)**
  - [ ] For a given (quarter, horizon, seed/repeat, hp_config):
    - [ ] Slice rows by precomputed indices.
    - [ ] Optional training-only sampling.
    - [ ] Train with early stopping.
    - [ ] Predict on full unsampled test.
    - [ ] Compute metrics, selections, portfolio returns, calibration outputs.
  - [ ] **No Pandas joins. No dtype inference. No string/object columns. No feature engineering.**

> **CRITICAL**: If any Phase A work happens inside Phase B, the redesign is considered failed.

---

### R2. Caching contract (compile artifacts are reusable and versioned)

The redesign must produce and reuse these compile artifacts:

**Required artifacts**

- [ ] `X_global`: float32 matrix (NumPy memmap or Arrow) with stable column order.
- [ ] `feature_names`: list aligned to `X_global`.
- [ ] `y_by_horizon`: uint8 arrays (or packed booleans) aligned to `X_global` rows.
- [ ] `quarter_id`: int16/int32 array per row (or mapping to quarter labels).
- [ ] `rows_train_by_quarter[q]`, `rows_test_by_quarter[q]`: int32 index arrays.
- [ ] `metadata_global`: row keys (gvkey/cik/datadate/snapshot_ts) needed for attribution.

**Optional but recommended**

- [ ] Learner-native caches:
  - [ ] LightGBM `Dataset` binary
  - [ ] XGBoost `DMatrix` binary

**Cache keying**

- [ ] Cache must be keyed on a deterministic fingerprint of:
  - [ ] input file checksums or modification timestamps,
  - [ ] config parameters affecting compile (safe_lag_days, feature set, labeling rule),
  - [ ] code version (git commit hash).
- [ ] Cache reuse must be automatic and logged.

---

### R3. One-shot AutoML break-even constraints (iteration-1 net positive)

AutoML must be constrained so that iteration-1 is net positive **without relying on "warm caches from a previous run."**

**Hard rule**

- [ ] If any trial would trigger rebuild of X/y/splits, cap trials at **≤5** and treat as a failure condition to fix.

**Target implementation**

- [ ] Enforce "compile once, train many" such that:
  - [ ] **≤10–12 trials** (typical) is safe for iteration-1.
  - [ ] Trials differ only in learner hyperparameters and training seed.
  - [ ] Trials reuse identical sliced arrays/DMatrix/Dataset.

**Scheduling**

- [ ] Implement a halving-style schedule:
  - [ ] Start ~8 configs at small budget (e.g., 50–100 trees/iterations).
  - [ ] Keep top 2.
  - [ ] Train top 2 to early stopping.

---

### R4. Sampling policy (training speed without material ranking regression)

Sampling must be built-in and configurable:

**Default training sampling (dev mode on)**

- [ ] Keep all positives.
- [ ] Sample negatives to ratio `neg_pos_ratio`:
  - [ ] default `10:1` (dev), `20:1` (standard), configurable.
- [ ] Apply **sample weights** to correct the effective base rate.
- [ ] Evaluate on **full unsampled test split** always.

**Calibration handling**

- [ ] Output:
  - [ ] `p_raw`
  - [ ] `p_calibrated` (post-calibration; method configurable)
  - [ ] `p_rescaled` (empirical prior rescaling aligned with realized deal rates under the strategy)

---

### R5. Stability certification must cover outputs + strategy + data integrity

Stability must be defined and computed for:

**A) Rank stability**

- [ ] Spearman/Kendall of `p_raw` across repeats (per quarter/horizon).
- [ ] Jaccard@K overlap for top-K selections (K configurable).

**B) Probability stability**

- [ ] Per-name mean/std of predicted probabilities across repeats.
- [ ] Distribution stability across repeats (KS / PSI).
- [ ] Calibration stability across repeats (Brier, log loss, ECE).

**C) Strategy stability**

- [ ] Per-quarter returns distribution across repeats (mean/std + percentile band).
- [ ] Path metrics distribution across repeats:
  - [ ] Sharpe/Sortino (if used), max drawdown, hit rate, turnover, concentration (HHI).
- [ ] Benchmark integrity checks (SPY returns must not be NaN; fail-fast if missing).

**D) Data/coverage stability**

- [ ] Dropped-deal counts due to missing CIK mapping must be tracked per year/quarter.
- [ ] Coverage rates become first-class metrics with thresholds.

---

## 2) Deliverables

### D1. New module layout (single source of truth)

Implement a minimal, explicit module structure in the repo:

- [ ] `src/compile/compile_panel.py`
  - [ ] `compile_dataset(config) -> CompileArtifacts`
- [ ] `src/train/train_eval.py`
  - [ ] `train_eval_one(task: TrainTask, artifacts: CompileArtifacts) -> FitResult`
- [ ] `src/automl/one_shot.py`
  - [ ] `run_one_shot_automl(task, artifacts) -> BestConfig + trial table`
- [ ] `src/stability/stability.py`
  - [ ] `run_stability_suite(artifacts, suite_config) -> StabilityReport`
- [ ] `src/portfolio/portfolio.py`
  - [ ] `simulate_portfolio(preds, strategy_config) -> returns + exposures + turnover`
- [ ] `src/benchmarks/spy.py`
  - [ ] robust SPY quarterly return computation + validation
- [ ] `src/viz/plots.py`
  - [ ] no-GIF-by-default; optional animation must not break runs

### D2. Artifact contract (files written to Drive run folder)

In `ARTIFACT_DIR`, write:

- [ ] `compile_manifest.json` (cache keys + row/feature counts + dtypes)
- [ ] `X_global.*` (memmap/arrow)
- [ ] `feature_names.json`
- [ ] `splits/rows_train_by_quarter.npz`, `splits/rows_test_by_quarter.npz`
- [ ] `labels/y_{horizon}m.npz` (or combined)
- [ ] `predictions/`:
  - [ ] per (quarter, horizon, repeat, config_id) predictions in Parquet
- [ ] `metrics/`:
  - [ ] per quarter/horizon metrics table
  - [ ] stability summary tables
  - [ ] portfolio return series and summary stats
- [ ] `logs/timing.csv` (required; see Logging spec)

### D3. Explicit "inner loop" entrypoint

- [ ] Provide a function and a Colab cell path that runs:
  - [ ] 1 quarter
  - [ ] 1 horizon
  - [ ] 1 fit (or one-shot AutoML with halving)
  - [ ] sampling on
  - [ ] early stopping on
  - [ ] total wall time target **≤60 seconds**
- [ ] This must not require the full 93-quarter backtest.

---

## 3) Performance targets and acceptance tests

### A) Compile-time acceptance

- [ ] Compile artifacts are created once per run and cached.
- [ ] Re-running train-only tasks does not re-run feature engineering.
- [ ] A run log must show:
  - [ ] `compile_seconds`
  - [ ] `cache_hit` boolean
  - [ ] compile artifact sizes in GB

### B) Train/Eval inner-loop acceptance

- [ ] `train_eval_one()` must not allocate Pandas DataFrames larger than trivial metadata tables.
- [ ] Must log:
  - [ ] `slice_seconds`
  - [ ] `fit_seconds`
  - [ ] `predict_seconds`
  - [ ] `eval_seconds`
  - [ ] RSS before/after (process memory)
- [ ] Target: **≤60 seconds** for dev-inner-loop on cached artifacts with sampling.

### C) AutoML iteration-1 acceptance

- [ ] One-shot AutoML with halving must complete within:
  - [ ] **≤2×** the time of a single fit in dev mode, for the same task.
- [ ] Must prove via timing table that:
  - [ ] X/splits are reused across trials (no compile calls).

### D) Stability acceptance (minimum viable certification)

- [ ] Implement sequential stopping for repeats:
  - [ ] Start `R=5`.
  - [ ] Compute CI half-width for:
    - [ ] mean Jaccard@K,
    - [ ] mean quarterly return,
    - [ ] mean top-K probability SD.
  - [ ] If thresholds met, stop early; else increase to `R=10`, then `R=20`.
- [ ] This must be implemented; fixed-R-only is not acceptable.

---

## 4) Parallelization and memory constraints

### Parallel execution model

- [ ] Unit task: `(quarter, horizon, repeat, config_id)`
- [ ] Parallelize across tasks **only in Phase B**.
- [ ] Enforce no oversubscription:
  - [ ] If using multi-process parallelism, set learner threads to 1.
  - [ ] If using multi-threaded learner, keep process count small.

### Memory constraints

- [ ] `X_global` must be float32.
- [ ] Prefer memmap/Arrow so workers can share read-only pages.
- [ ] Avoid per-worker full copies of `X_global`.

### Required instrumentation

- [ ] Per task, log:
  - [ ] `n_train_rows`, `n_test_rows`, `n_features`
  - [ ] `rss_before_fit_mb`, `rss_after_fit_mb`
  - [ ] `peak_rss_mb` if available
  - [ ] GPU memory only if GPU training is enabled

---

## 5) Model set and defaults

### Default learner set (must implement Tier 1 first)

**Tier 1 (primary): LightGBM**

- [ ] Objective: binary
- [ ] Early stopping
- [ ] Parameters to tune in one-shot:
  - [ ] `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l2`, `learning_rate`, `max_bin`

**Tier 2 (optional): XGBoost**

- [ ] `hist` CPU; optional `gpu_hist` if GPU present
- [ ] Tune: `max_depth`/`min_child_weight`, `subsample`, `colsample_bytree`, `lambda`, `eta`, `max_bin`

**Tier 3 (baseline): SGD logistic**

- [ ] For sanity checks and fast debugging

---

## 6) Benchmark and data integrity fixes

### SPY quarterly returns must be correct or the run fails

- [ ] Implement robust SPY return retrieval and quarterly aggregation.
- [ ] If SPY returns missing/NaN for any evaluated quarter:
  - [ ] log error with quarter list,
  - [ ] mark benchmark comparison invalid,
  - [ ] fail-fast unless user explicitly disables benchmark checks.

### GIF/animation must not break runs

Current error: `FigureCanvasAgg has no attribute tostring_rgb`.

- [ ] Fix by using a backend-safe extraction:
  - [ ] `fig.canvas.draw()`
  - [ ] `buf = np.asarray(fig.canvas.buffer_rgba())`
  - [ ] Convert to frames from `buf`
- [ ] Additionally:
  - [ ] Default: animations off.
  - [ ] Animations must be isolated so failure cannot break training runs.

---

## 7) Logging spec (must be implemented exactly)

- [ ] Write `logs/timing.csv` with one row per phase/task, columns:
  - [ ] `phase` in {`compile`, `train_eval`, `automl_trial`, `stability_aggregate`}
  - [ ] `run_id`
  - [ ] `quarter`
  - [ ] `horizon_mo`
  - [ ] `repeat_id`
  - [ ] `config_id`
  - [ ] `cache_hit_compile` (boolean; only for compile)
  - [ ] `n_train_rows`, `n_test_rows`, `n_features`
  - [ ] `slice_seconds`, `fit_seconds`, `predict_seconds`, `eval_seconds`, `total_seconds`
  - [ ] `rss_before_mb`, `rss_after_mb`
  - [ ] `notes` (free text)

- [ ] Also write a concise `logs/summary.txt` with:
  - [ ] compile time
  - [ ] backtest time
  - [ ] stability suite time
  - [ ] counts of cache hits/misses
  - [ ] counts of dropped deals due to missing CIK

---

## 8) Minimal implementation plan (ordered)

1. [ ] Implement `compile_dataset()` producing `X_global` float32 + indices + labels + manifest.
2. [ ] Implement `train_eval_one()` using only array slicing + learner fit/predict + metrics.
3. [ ] Implement one-shot AutoML halving on top of `train_eval_one()` without rebuild.
4. [ ] Implement stability suite (rank + prob + returns + calibration) with sequential stopping.
5. [ ] Add parallel task runner for Phase B with safe threading settings.
6. [ ] Add SPY benchmark fix + fail-fast checks.
7. [ ] Add optional animation fix, isolated from core pipeline.

---

## 9) What to remove or de-emphasize from the current notebook

- [ ] Any training-time dependence on Pandas merges/sorts inside the per-quarter loop.
- [ ] Any repeated construction of per-quarter DataFrames for each trial/repeat.
- [ ] Any default behavior that runs the full 93-quarter suite when the user wants inner-loop iteration.

---

## 10) Definition of done

The redesign is complete when the agent can demonstrate:

1. [ ] A second run hits compile cache and immediately enters train/eval.
2. [ ] A single quarter/horizon dev-inner-loop run completes in ≤60 seconds with sampling.
3. [ ] One-shot AutoML halving runs without any feature rebuild and completes within ≤2× single-fit time.
4. [ ] Stability suite produces a report for rank + probability + returns with sequential stopping.
5. [ ] Timing and memory logs are written and interpretable.
6. [ ] SPY benchmark is either valid or the run fails with an explicit quarter list.

---

## Budget Table (Numeric Thresholds)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Inner-loop wall time | ≤60 seconds | Single (quarter, horizon) with sampling |
| AutoML overhead | ≤2× single fit | 8-12 trials with halving |
| Compile cache hit | 100% | Second run must not re-compile |
| SPY NaN rate | 0% | Fail-fast if any quarter missing |
| Jaccard@50 CI half-width | ≤0.10 | For stability acceptance |
| Quarterly return CI half-width | ≤2% | For stability acceptance |
| Top-K probability SD | ≤0.05 | For stability acceptance |
| Max memory (X_global) | float32 only | No float64 allowed |

---

### Calibration footer

**Persona used:** Senior Quant ML Engineer.  
**Difficulty tuning:** 70% execution-level spec (interfaces, artifacts, acceptance tests), 20% architectural constraints (compile/train split, cache keying, parallel safety), 10% research-lead framing (stability certification gates, sequential stopping).
