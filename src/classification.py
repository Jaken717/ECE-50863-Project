"""Experiment 1 — Classification algorithm comparison (original 30 features, no PCA).

Protocol (see README):
  * shared group-aware 70/15/15 split from preprocessing.py
  * each model's hyperparameters are tuned on the VALIDATION set; the selection
    criterion is F1 for the phishing (positive) class
  * the chosen configuration is refit on TRAIN only and evaluated once on TEST
  * scaling (LogReg, KNN) lives inside a Pipeline, so it is fit on train only

Outputs (results/):
  exp1_validation_tuning.csv   every configuration tried, with validation metrics
  exp1_best_params.json        chosen hyperparameters per model
  exp1_test_metrics.csv        final comparison table (test set)
  exp1_test_metrics.md         same table in Markdown, for the report/README
  exp1_classification_reports.txt  per-class precision/recall/F1 (test)
  exp1_confusion_matrices.png  2x2 grid of test confusion matrices
  exp1_metric_comparison.png   grouped bar chart of test metrics
  exp1_rf_feature_importance.png  top-15 Random Forest feature importances

Run:  python src/classification.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

# NumPy 2.0 + macOS Accelerate BLAS emits spurious "encountered in matmul"
# RuntimeWarnings; results were verified to match a non-BLAS computation.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import RANDOM_STATE, ROOT, load_splits  # noqa: E402

RESULTS = ROOT / "results"

# Categorical slots 1-4 of the reference palette (fixed order, one per model).
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def model_specs():
    """name -> (base estimator, hyperparameter grid). Scaled models use a Pipeline."""
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"criterion": ["gini", "entropy"],
             "max_depth": [5, 10, 15, 20, None],
             "min_samples_leaf": [1, 2, 5, 10]},
        ),
        "K-Nearest Neighbors": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 300],
             "max_depth": [10, 20, None],
             "max_features": ["sqrt", "log2"],
             "min_samples_leaf": [1, 2]},
        ),
    }


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }


def tune(name, base, grid, s):
    rows = []
    for params in ParameterGrid(grid):
        est = clone(base).set_params(**params)
        est.fit(s["X_train"], s["y_train"])
        rows.append({"model": name, "params": json.dumps(params, default=str),
                     **metrics(s["y_val"], est.predict(s["X_val"]))})
    df = pd.DataFrame(rows)
    # best validation F1; ties broken by accuracy, then by grid order (simpler first)
    best = df.sort_values(["f1", "accuracy"], ascending=False, kind="stable").iloc[0]
    return df, json.loads(best["params"])


def final_eval(name, base, params, s):
    est = clone(base).set_params(**params)
    t0 = time.perf_counter()
    est.fit(s["X_train"], s["y_train"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = est.predict(s["X_test"])
    pred_s = time.perf_counter() - t0
    val = metrics(s["y_val"], est.predict(s["X_val"]))
    test = metrics(s["y_test"], y_pred)
    row = {"model": name, **test,
           "val_f1": val["f1"], "fit_time_s": fit_s, "predict_time_s": pred_s}
    return est, y_pred, row


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.title.set_color(INK)


def plot_confusion(cms, path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.2))
    labels = ["Legitimate", "Phishing"]
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / cm[i].sum()
                dark = cm[i, j] > 0.5 * cm.sum(axis=1).max()
                ax.text(j, i, f"{cm[i, j]}\n({frac:.1%})", ha="center", va="center",
                        fontsize=10, color="white" if dark else INK)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted", color=INK_2)
        ax.set_ylabel("Actual", color=INK_2)
        ax.set_title(name, fontsize=11)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Experiment 1 — test-set confusion matrices (row % = share of actual class)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_bars(table, path):
    cols = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(cols))
    n = len(table)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, (row, color) in enumerate(zip(table.itertuples(), MODEL_COLORS)):
        vals = [getattr(row, c) for c in cols]
        # 2px-ish surface gap between adjacent bars via white edges
        ax.bar(x + (k - (n - 1) / 2) * w, vals, w, color=color, label=row.model,
               edgecolor="white", linewidth=1.5)
    lo = np.floor(table[cols].to_numpy().min() * 50) / 50 - 0.02
    ax.set_ylim(max(lo, 0), 1.0)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylabel("Score (test set, phishing = positive)", color=INK_2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), labelcolor=INK_2)
    ax.text(0, 1.18, "Note: y-axis does not start at 0", transform=ax.transAxes,
            fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rf_importance(rf, columns, path, top=15):
    imp = pd.Series(rf.feature_importances_, index=columns).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=MODEL_COLORS[0], edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Mean decrease in impurity", color=INK_2, fontsize=9)
    ax.set_title(f"Random Forest — top {top} feature importances", fontsize=11)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    s = load_splits()
    print({k: v.shape for k, v in s.items()})

    tuning, best_params, rows, cms, reports, fitted = [], {}, [], {}, [], {}
    for name, (base, grid) in model_specs().items():
        df, params = tune(name, base, grid, s)
        tuning.append(df)
        best_params[name] = params
        est, y_pred, row = final_eval(name, base, params, s)
        fitted[name] = est
        rows.append(row)
        cms[name] = confusion_matrix(s["y_test"], y_pred, labels=[0, 1])
        reports.append(f"=== {name} ===  params: {params}\n" + classification_report(
            s["y_test"], y_pred, labels=[0, 1], target_names=["legitimate", "phishing"], digits=4))
        print(f"{name:<20} val F1={row['val_f1']:.4f}  test F1={row['f1']:.4f}  {params}")

    table = pd.DataFrame(rows)
    pd.concat(tuning).to_csv(RESULTS / "exp1_validation_tuning.csv", index=False)
    (RESULTS / "exp1_best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    table.to_csv(RESULTS / "exp1_test_metrics.csv", index=False)
    (RESULTS / "exp1_classification_reports.txt").write_text("\n".join(reports))

    md = table.copy()
    for c in ["accuracy", "precision", "recall", "f1", "val_f1"]:
        md[c] = md[c].map("{:.4f}".format)
    md["fit_time_s"] = md["fit_time_s"].map("{:.3f}".format)
    md["predict_time_s"] = md["predict_time_s"].map("{:.4f}".format)
    md.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "Val F1",
                  "Fit time (s)", "Predict time (s)"]
    (RESULTS / "exp1_test_metrics.md").write_text(md.to_markdown(index=False) + "\n"
                                                    if hasattr(md, "to_markdown") else md.to_string(index=False))

    plot_confusion(cms, RESULTS / "exp1_confusion_matrices.png")
    plot_metric_bars(table, RESULTS / "exp1_metric_comparison.png")
    plot_rf_importance(fitted["Random Forest"], s["X_train"].columns,
                       RESULTS / "exp1_rf_feature_importance.png")

    print("\nTest-set results (phishing = positive class):")
    print(table.round(4).to_string(index=False))
    for name, cm in cms.items():
        tn, fp, fn, tp = cm.ravel()
        print(f"  {name:<20} TN={tn} FP={fp} FN={fn} (missed phishing) TP={tp}")


if __name__ == "__main__":
    main()
