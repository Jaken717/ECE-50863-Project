"""Experiment 2 — PCA and classification.

Protocol (see README):
  * same group-aware 70/15/15 split as Experiment 1 (preprocessing.load_splits)
  * pipeline = StandardScaler -> PCA(k) -> classifier, all fit on TRAIN only
  * k in {5, 10, 15, 20, 25}; for EACH k every classifier is re-tuned on the
    VALIDATION set with the same grids as Experiment 1 (selection: phishing F1)
  * baseline = Experiment 1 setup (30 original features, no PCA), re-run here so
    timings come from the same session
  * the chosen configuration for each (model, k) is refit on TRAIN and scored on
    TEST; "best k" per model is chosen by validation F1, never by test scores
  * fit/predict times are the median of TIMING_REPEATS runs and include the
    scaling + PCA steps (the real cost of using the pipeline)

Outputs (results/):
  exp2_explained_variance.csv / .png   PCA explained variance (fit on train)
  exp2_validation_tuning.csv           every configuration tried
  exp2_results.csv                     val + test metrics and timings, every (model, k)
  exp2_best_k.md                       best k per model (by val F1) vs. baseline
  exp2_metrics_vs_components.png       test F1 and recall vs. k, baseline dashed
  exp2_timing_fixed_params.csv         controlled timing (see below)
  exp2_time_vs_components.png          fit / predict time vs. k (controlled)

Timing note: re-tuning picks different hyperparameters at each k (e.g. 100 vs
300 trees), which would dominate any time comparison. The time figure therefore
uses a CONTROLLED run: each model keeps its no-PCA (Experiment 1) hyperparameters
at every k, so only the input representation changes.

Run:  python src/pca_experiments.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classification import (GRID, INK, INK_2, MODEL_COLORS, RESULTS, metrics,  # noqa: E402
                            model_specs, style_axes, tune)
from preprocessing import RANDOM_STATE, load_splits  # noqa: E402

N_COMPONENTS = [5, 10, 15, 20, 25]
TIMING_REPEATS = 5
BASELINE = 30  # label used for the no-PCA, 30-original-feature configuration


def pca_pipeline(base, grid, k):
    """Wrap an Experiment 1 estimator as StandardScaler -> PCA(k) -> classifier."""
    if isinstance(base, Pipeline):  # LogReg / KNN already have a scaler step
        clf = base.named_steps["clf"]
        grid = {key.replace("clf__", "", 1): v for key, v in grid.items()}
    else:
        clf = base
    pipe = Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=k, random_state=RANDOM_STATE)),
                     ("clf", clone(clf))])
    return pipe, {f"clf__{key}": v for key, v in grid.items()}


def timed_eval(base, params, s):
    fit_t, pred_t = [], []
    for _ in range(TIMING_REPEATS):
        est = clone(base).set_params(**params)
        t0 = time.perf_counter()
        est.fit(s["X_train"], s["y_train"])
        fit_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        y_pred = est.predict(s["X_test"])
        pred_t.append(time.perf_counter() - t0)
    val = metrics(s["y_val"], est.predict(s["X_val"]))
    test = metrics(s["y_test"], y_pred)
    return {**{f"test_{m}": v for m, v in test.items()},
            **{f"val_{m}": v for m, v in val.items()},
            "fit_time_s": float(np.median(fit_t)),
            "predict_time_s": float(np.median(pred_t))}


def controlled_timing(name, base, grid, baseline_params, s):
    """Time every k with the model's no-PCA hyperparameters held fixed."""
    rows = []
    for k in [BASELINE] + N_COMPONENTS:
        if k == BASELINE:
            est, params = base, baseline_params
        else:
            est, _ = pca_pipeline(base, grid, k)
            params = {"clf__" + key.replace("clf__", "", 1): v
                      for key, v in baseline_params.items()}
        r = timed_eval(est, params, s)
        rows.append({"model": name, "n_components": k, "fit_time_s": r["fit_time_s"],
                     "predict_time_s": r["predict_time_s"], "test_f1": r["test_f1"]})
    return rows


def explained_variance(s):
    X = StandardScaler().fit_transform(s["X_train"])
    pca = PCA(random_state=RANDOM_STATE).fit(X)
    ev = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                       "explained_variance_ratio": pca.explained_variance_ratio_,
                       "cumulative": np.cumsum(pca.explained_variance_ratio_)})
    return ev


def plot_explained_variance(ev, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(ev.component, ev.explained_variance_ratio, color="#c9ddf5",
           edgecolor="white", linewidth=1, label="Individual component")
    ax.plot(ev.component, ev.cumulative, color=MODEL_COLORS[0], linewidth=2,
            label="Cumulative")
    for k in N_COMPONENTS:
        c = ev.cumulative.iloc[k - 1]
        ax.plot(k, c, "o", color=MODEL_COLORS[0], markersize=7,
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{c:.0%}", (k, c), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK)
    ax.set_xlim(0.3, 30.7)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Number of principal components", color=INK_2, fontsize=9)
    ax.set_ylabel("Explained variance ratio", color=INK_2, fontsize=9)
    ax.set_title("PCA explained variance (standardized training features)", fontsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _model_lines(ax, res, col, models, log=False):
    for name, color in zip(models, MODEL_COLORS):
        d = res[(res.model == name) & (res.n_components != BASELINE)].sort_values("n_components")
        base = res[(res.model == name) & (res.n_components == BASELINE)][col].iloc[0]
        ax.plot(d.n_components, d[col], "-o", color=color, linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
        ax.axhline(base, color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.9)
    ax.set_xticks(N_COMPONENTS)
    ax.set_xlabel("Number of principal components", color=INK_2, fontsize=9)
    if log:
        ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    style_axes(ax)


def plot_metrics(res, models, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, title in zip(axes, ["test_f1", "test_recall"], ["F1-score", "Recall"]):
        _model_lines(ax, res, col, models)
        ax.set_title(f"Test {title} (phishing = positive)", fontsize=11)
    axes[0].set_ylabel("Score", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, linestyle=(0, (4, 3))))
    labels.append("No-PCA baseline (30 features)")
    fig.legend(handles, labels, frameon=False, ncol=5, fontsize=9, labelcolor=INK_2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_times(timing, models, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, title in zip(axes, ["fit_time_s", "predict_time_s"],
                              ["Training time", "Prediction time (test set)"]):
        _model_lines(ax, timing, col, models, log=True)
        ax.set_title(f"{title}, median of {TIMING_REPEATS} runs", fontsize=11)
        ax.set_ylabel("Seconds (log scale)", color=INK_2, fontsize=9)
    fig.text(0.5, 0.005, "Hyperparameters fixed at each model's no-PCA (Experiment 1) "
             "setting; times include standardization + PCA.", ha="center",
             fontsize=8, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, linestyle=(0, (4, 3))))
    labels.append("No-PCA baseline (30 features)")
    fig.legend(handles, labels, frameon=False, ncol=5, fontsize=9, labelcolor=INK_2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    s = load_splits()

    ev = explained_variance(s)
    ev.to_csv(RESULTS / "exp2_explained_variance.csv", index=False)
    plot_explained_variance(ev, RESULTS / "exp2_explained_variance.png")
    print("Cumulative explained variance:",
          {k: round(ev.cumulative.iloc[k - 1], 4) for k in N_COMPONENTS})

    specs = model_specs()
    tuning, rows, timing = [], [], []
    for name, (base, grid) in specs.items():
        baseline_params = None
        configs = [(BASELINE, base, grid)] + [(k, *pca_pipeline(base, grid, k)) for k in N_COMPONENTS]
        for k, est, g in configs:
            df, params = tune(name, est, g, s)
            tuning.append(df.assign(n_components=k))
            if k == BASELINE:
                baseline_params = params
            row = {"model": name, "n_components": k, "pca": k != BASELINE,
                   "params": json.dumps(params, default=str), **timed_eval(est, params, s)}
            rows.append(row)
            print(f"{name:<20} k={'none' if k == BASELINE else k:>4}  "
                  f"val F1={row['val_f1']:.4f}  test F1={row['test_f1']:.4f}  "
                  f"recall={row['test_recall']:.4f}  fit={row['fit_time_s']:.4f}s")
        timing += controlled_timing(name, base, grid, baseline_params, s)

    res = pd.DataFrame(rows)
    pd.concat(tuning).to_csv(RESULTS / "exp2_validation_tuning.csv", index=False)
    res.to_csv(RESULTS / "exp2_results.csv", index=False)
    timing = pd.DataFrame(timing)
    timing.to_csv(RESULTS / "exp2_timing_fixed_params.csv", index=False)
    print("\nControlled timing (no-PCA hyperparameters at every k):")
    print(timing.pivot(index="n_components", columns="model",
                       values="fit_time_s").round(4).to_string())

    # Best k per model, chosen on VALIDATION F1 among the PCA settings
    pca_only = res[res.pca]
    best = pca_only.loc[pca_only.groupby("model", sort=False)["val_f1"].idxmax()]
    base = res[~res.pca].set_index("model")
    lines = ["| Model | Best k (val F1) | Var. explained | Test F1 (PCA) | Test F1 (no PCA) | ΔF1 "
             "| Test recall (PCA) | Test recall (no PCA) | Fit time PCA / no PCA (s) |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in best.itertuples():
        b = base.loc[r.model]
        lines.append(
            f"| {r.model} | {r.n_components} | {ev.cumulative.iloc[r.n_components - 1]:.1%} "
            f"| {r.test_f1:.4f} | {b.test_f1:.4f} | {r.test_f1 - b.test_f1:+.4f} "
            f"| {r.test_recall:.4f} | {b.test_recall:.4f} "
            f"| {r.fit_time_s:.4f} / {b.fit_time_s:.4f} |")
    (RESULTS / "exp2_best_k.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    models = list(specs)
    plot_metrics(res, models, RESULTS / "exp2_metrics_vs_components.png")
    plot_times(timing, models, RESULTS / "exp2_time_vs_components.png")


if __name__ == "__main__":
    main()
