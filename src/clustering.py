"""Experiment 3 — K-Means clustering.

Protocol (see README):
  * same group-aware split as Experiments 1-2 (preprocessing.load_splits)
  * StandardScaler + K-Means (k = 2) are fit on TRAIN features only, labels unused
  * validation/test rows are assigned to the nearest learned centroid
  * labels are used ONLY afterwards, to evaluate agreement:
      - cluster x label contingency tables
      - Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
      - purity / "cluster-as-classifier": each cluster is mapped to its majority
        TRAIN label, and that fixed mapping is applied to the test set
    Cluster IDs are arbitrary; no cluster is assumed to mean phishing.
  * 2D PCA projection (fit on train) for visualisation only — K-Means is run on
    all 30 standardized features, not on the projection
  * supporting analyses: stability over random seeds, a k = 2..10 sweep
    (inertia, silhouette, ARI, NMI), and per-cluster feature profiles

Outputs (results/):
  exp3_summary.md               headline metrics + contingency tables
  exp3_metrics.csv              ARI / NMI / purity / mapped scores per split
  exp3_contingency.csv          cluster x label counts per split
  exp3_contingency_k4.csv       exploratory k = 4 composition
  exp3_seed_stability.csv       k = 2 over 20 seeds
  exp3_k_sweep.csv / .png       k = 2..10
  exp3_cluster_profiles.csv     mean (raw) feature value per cluster
  exp3_cluster_profiles.png     features that differ most between the 2 clusters
  exp3_pca_clusters.png         2D PCA: coloured by cluster vs. by true label

Run:  python src/clustering.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classification import GRID, INK, INK_2, MODEL_COLORS, RESULTS, style_axes  # noqa: E402
from preprocessing import RANDOM_STATE, load_splits  # noqa: E402

K = 2
N_INIT = 10
K_SWEEP = range(2, 11)
N_SEEDS = 20

# Scatter colours: clusters use categorical slots 1 and 3 (validated all-pairs);
# true labels use slot 2 (phishing) vs. a neutral grey, so no cluster colour is
# reused for a label and the plot does not imply a cluster->label mapping.
CLUSTER_COLORS = [MODEL_COLORS[0], MODEL_COLORS[2]]
LABEL_COLORS = {0: "#8a8984", 1: MODEL_COLORS[1]}  # 0 legitimate, 1 phishing
LABEL_NAMES = {0: "Legitimate", 1: "Phishing"}


def fit_kmeans(Xs, k, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(Xs)


def majority_mapping(clusters, y):
    """cluster id -> majority label, learned from TRAIN only."""
    df = pd.DataFrame({"c": clusters, "y": np.asarray(y)})
    return df.groupby("c")["y"].agg(lambda s: s.value_counts().idxmax()).to_dict()


def agreement(clusters, y, mapping):
    y = np.asarray(y)
    y_hat = pd.Series(clusters).map(mapping).fillna(0).astype(int).to_numpy()
    ct = pd.crosstab(clusters, y)
    return {
        "ARI": adjusted_rand_score(y, clusters),
        "NMI": normalized_mutual_info_score(y, clusters),
        "purity": ct.max(axis=1).sum() / len(y),
        "mapped_accuracy": accuracy_score(y, y_hat),
        "mapped_precision": precision_score(y, y_hat, zero_division=0),
        "mapped_recall": recall_score(y, y_hat, zero_division=0),
        "mapped_f1": f1_score(y, y_hat, zero_division=0),
    }


def contingency(clusters, y, split):
    ct = pd.crosstab(pd.Series(clusters, name="cluster"),
                     pd.Series(np.asarray(y), name="label").map(LABEL_NAMES))
    ct = ct.reindex(columns=["Legitimate", "Phishing"], fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    ct["% phishing"] = (100 * ct["Phishing"] / ct["Total"]).round(1)
    ct.insert(0, "split", split)
    return ct.reset_index()


def seed_stability(Xs, y, ref_labels):
    rows = []
    for seed in range(N_SEEDS):
        km = fit_kmeans(Xs, K, seed)
        rows.append({"seed": seed, "inertia": km.inertia_,
                     "ARI_vs_labels": adjusted_rand_score(y, km.labels_),
                     "NMI_vs_labels": normalized_mutual_info_score(y, km.labels_),
                     "ARI_vs_reference_clustering": adjusted_rand_score(ref_labels, km.labels_)})
    return pd.DataFrame(rows)


def k_sweep(Xs, y):
    rows = []
    rng = np.random.RandomState(RANDOM_STATE)
    sample = rng.choice(len(Xs), size=min(5000, len(Xs)), replace=False)
    for k in K_SWEEP:
        km = fit_kmeans(Xs, k)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Xs[sample], km.labels_[sample]),
                     "ARI": adjusted_rand_score(y, km.labels_),
                     "NMI": normalized_mutual_info_score(y, km.labels_),
                     "purity": pd.crosstab(km.labels_, y).max(axis=1).sum() / len(y)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- plotting
def _jitter(a, rng, scale=0.08):
    return a + rng.normal(0, scale, size=a.shape)


def plot_pca(Z, clusters, y, centers_2d, ev, path):
    """Two panels over the same 2D PCA projection (training set)."""
    rng = np.random.RandomState(0)
    Zj = np.column_stack([_jitter(Z[:, 0], rng), _jitter(Z[:, 1], rng)])
    order = rng.permutation(len(Z))  # random draw order so no group is always on top
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    # one scatter per panel with a random draw order, so neither group is always on top
    ax = axes[0]
    ax.scatter(Zj[order, 0], Zj[order, 1], s=9, alpha=0.35, linewidths=0,
               c=[CLUSTER_COLORS[c] for c in clusters[order]])
    for c in range(K):
        ax.scatter([], [], s=30, color=CLUSTER_COLORS[c],
                   label=f"Cluster {c} (n={int((clusters == c).sum())})")
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="X", s=160, color=INK,
               edgecolors="white", linewidths=2, zorder=5)
    ax.scatter([], [], marker="X", s=60, color=INK, label="Centroid")
    ax.set_title("Coloured by K-Means cluster (k = 2)", fontsize=11)

    ax = axes[1]
    yy = np.asarray(y)
    ax.scatter(Zj[order, 0], Zj[order, 1], s=9, alpha=0.35, linewidths=0,
               c=[LABEL_COLORS[v] for v in yy[order]])
    for lab in (0, 1):
        ax.scatter([], [], s=30, color=LABEL_COLORS[lab],
                   label=f"{LABEL_NAMES[lab]} (n={int((yy == lab).sum())})")
    ax.set_title("Coloured by true label (not used for clustering)", fontsize=11)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} of variance)", color=INK_2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    axes[0].set_ylabel(f"PC2 ({ev[1]:.1%} of variance)", color=INK_2, fontsize=9)
    fig.text(0.5, 0.005, "Training set, standardized features projected to 2D with PCA "
             "(visualisation only). Small jitter added because many rows are identical.",
             ha="center", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_k_sweep(sweep, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("inertia", "Inertia (within-cluster SS)", "Elbow: inertia"),
              ("silhouette", "Silhouette score", "Silhouette (5,000-row sample)"),
              (None, "Score", "Agreement with true labels")]
    for ax, (col, ylab, title) in zip(axes, panels):
        if col:
            ax.plot(sweep.k, sweep[col], "-o", color=MODEL_COLORS[0], linewidth=2,
                    markersize=6, markeredgecolor="white", markeredgewidth=1.5)
        else:
            for c, color in zip(["ARI", "NMI"], [MODEL_COLORS[0], MODEL_COLORS[1]]):
                ax.plot(sweep.k, sweep[c], "-o", color=color, linewidth=2, markersize=6,
                        markeredgecolor="white", markeredgewidth=1.5, label=c)
                last = sweep.iloc[-1]
                ax.annotate(c, (last.k, last[c]), textcoords="offset points",
                            xytext=(6, 0), va="center", fontsize=9, color=INK)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
            ax.set_ylim(0, None)
        ax.set_xticks(list(sweep.k))
        ax.set_xlabel("Number of clusters k", color=INK_2, fontsize=9)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_profiles(profiles, path, top=12):
    diff = (profiles.loc[1] - profiles.loc[0])
    feats = diff.abs().sort_values(ascending=True).tail(top).index
    y = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    h = 0.38
    for c, off in ((0, h / 2), (1, -h / 2)):
        ax.barh(y + off, profiles.loc[c, feats], h, color=CLUSTER_COLORS[c],
                edgecolor="white", linewidth=1, label=f"Cluster {c}")
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, feats)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Mean raw feature value (−1 = phishing-like, +1 = legitimate-like)",
                  color=INK_2, fontsize=9)
    ax.set_title(f"Features that differ most between the two clusters (top {top})",
                 fontsize=11)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def md_table(df):
    return df.to_markdown(index=False)


def main():
    RESULTS.mkdir(exist_ok=True)
    s = load_splits()
    scaler = StandardScaler().fit(s["X_train"])
    Xs = {n: scaler.transform(s[f"X_{n}"]) for n in ("train", "val", "test")}
    y = {n: s[f"y_{n}"].to_numpy() for n in ("train", "val", "test")}

    # --- main experiment: k = 2, fit on train only, no labels
    km = fit_kmeans(Xs["train"], K)
    clusters = {"train": km.labels_, **{n: km.predict(Xs[n]) for n in ("val", "test")}}
    mapping = majority_mapping(clusters["train"], y["train"])

    metrics = pd.DataFrame([{"split": n, "n": len(y[n]), **agreement(clusters[n], y[n], mapping)}
                            for n in ("train", "val", "test")])
    cont = pd.concat([contingency(clusters[n], y[n], n) for n in ("train", "test")])
    metrics.to_csv(RESULTS / "exp3_metrics.csv", index=False)
    cont.to_csv(RESULTS / "exp3_contingency.csv", index=False)

    # --- seed stability, k sweep, profiles
    stab = seed_stability(Xs["train"], y["train"], km.labels_)
    stab.to_csv(RESULTS / "exp3_seed_stability.csv", index=False)
    sweep = k_sweep(Xs["train"], y["train"])
    sweep.to_csv(RESULTS / "exp3_k_sweep.csv", index=False)
    km4 = fit_kmeans(Xs["train"], 4)  # exploratory: k = 4 had the highest ARI in the sweep
    cont4 = pd.concat([contingency(km4.labels_, y["train"], "train"),
                       contingency(km4.predict(Xs["test"]), y["test"], "test")])
    cont4.to_csv(RESULTS / "exp3_contingency_k4.csv", index=False)
    profiles = s["X_train"].groupby(km.labels_).mean()
    profiles.to_csv(RESULTS / "exp3_cluster_profiles.csv")

    # --- plots
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs["train"])
    plot_pca(pca.transform(Xs["train"]), km.labels_, y["train"],
             pca.transform(km.cluster_centers_), pca.explained_variance_ratio_,
             RESULTS / "exp3_pca_clusters.png")
    plot_k_sweep(sweep, RESULTS / "exp3_k_sweep.png")
    plot_profiles(profiles, RESULTS / "exp3_cluster_profiles.png")

    # --- summary
    mshow = metrics.copy()
    for c in mshow.columns[2:]:
        mshow[c] = mshow[c].map("{:.4f}".format)
    sweep_show = sweep.copy()
    sweep_show["inertia"] = sweep_show["inertia"].round(0).astype(int)
    for c in ["silhouette", "ARI", "NMI", "purity"]:
        sweep_show[c] = sweep_show[c].map("{:.4f}".format)
    mapping_txt = ", ".join(f"cluster {c} → {LABEL_NAMES[v].lower()}" for c, v in mapping.items())
    summary = "\n\n".join([
        "# Experiment 3 — K-Means (k = 2) results",
        "Scaler and K-Means fit on the training set only, without labels. "
        f"Majority-label mapping learned on train: {mapping_txt}.",
        "## Agreement with true labels\n\n" + md_table(mshow),
        "## Cluster × label contingency\n\n" + md_table(cont),
        f"## Seed stability (k = 2, {N_SEEDS} seeds)\n\n"
        f"ARI vs labels: mean {stab.ARI_vs_labels.mean():.4f}, sd {stab.ARI_vs_labels.std():.4f}, "
        f"range {stab.ARI_vs_labels.min():.4f}–{stab.ARI_vs_labels.max():.4f}  \n"
        f"ARI vs reference clustering: min {stab.ARI_vs_reference_clustering.min():.4f}",
        "## k sweep (training set)\n\n" + md_table(sweep_show),
        "## Exploratory: k = 4 composition\n\nk = 4 was not chosen a priori; it is "
        "shown because it had the highest ARI in the sweep (choosing k by ARI uses "
        "labels, so treat this as exploratory).\n\n" + md_table(cont4),
    ]) + "\n"
    (RESULTS / "exp3_summary.md").write_text(summary)
    print(summary)
    diff = (profiles.loc[1] - profiles.loc[0]).sort_values(key=abs, ascending=False)
    print("Largest cluster-profile differences (cluster1 - cluster0):")
    print(diff.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
