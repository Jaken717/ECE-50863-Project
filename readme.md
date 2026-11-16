# Phishing Website Detection Using Data Mining Techniques

**Fall 2026 Data Mining Course Project**  
**Team:** Jinkai Liu and Bryan Zhang  
**Report deadline:** December 19, 2026, 11:59 PM

## Project Overview

Phishing websites imitate legitimate sites to trick visitors into disclosing sensitive information. This project investigates whether classical data mining techniques can distinguish phishing websites from legitimate ones using structured website characteristics.

Our study focuses on three experiments: (1) comparing classification algorithms, (2) studying the effect of principal component analysis (PCA) on classification, and (3) exploring whether unsupervised clustering discovers groups that correspond to phishing and legitimate websites.

### Research Questions

1. **Classification:** Which classical classifier performs best on the phishing website dataset?
2. **PCA:** How does dimensionality reduction affect classification accuracy, F1-score, and computational cost?
3. **Clustering:** Can K-Means identify meaningful website groups without using class labels during training?

## Dataset

**Source:** [UCI Machine Learning Repository — Phishing Websites (Dataset 327)](https://archive.ics.uci.edu/dataset/327/phishing+websites)

| Property | Description |
| --- | --- |
| Number of instances | 11,055 |
| Number of input features | 30 |
| Task | Binary classification |
| Labels | Phishing vs. legitimate |
| Feature format | Pre-extracted, mostly discrete/coded website attributes |

We will inspect the original feature descriptions and class encoding before modeling. We will also check class balance, duplicate records, potential conflicting labels, and data quality.

## Methodology

### Data Preprocessing

1. Load the dataset and inspect the label distribution and feature types.
2. Check for missing values, duplicates, and suspicious records; document any cleaning decisions.
3. Create a consistent, stratified **70% training / 15% validation / 15% test** split, keeping identical feature vectors in the same split where practical to reduce leakage.
4. Standardize features for distance-based or linear methods; fit all preprocessing transformations using **training data only**.
5. Reuse the same data partitions for fair model comparisons.

> **Evaluation protocol:** Tune hyperparameters and choose PCA dimensions using the validation set. Reserve the test set for final evaluation after decisions are fixed.

## Experiment 1 — Classification Algorithm Comparison

**Objective:** Compare four classical classification algorithms using the original 30 website features, without PCA.

**Models:**

- Logistic Regression (linear baseline)
- Decision Tree (interpretable tree-based model)
- K-Nearest Neighbors (distance-based model)
- Random Forest (ensemble of decision trees)

**Evaluation metrics:** Accuracy, precision, recall, F1-score, and confusion matrix. We will explicitly designate *phishing* as the positive class, since false negatives (missed phishing websites) are particularly important.

**Expected outputs:**

- Classification performance comparison table
- Confusion matrices for selected models
- Discussion of the strengths and limitations of each algorithm

## Experiment 2 — PCA and Classification

**Objective:** Investigate whether PCA can reduce feature dimensionality while preserving classification performance.

**Procedure:**

1. Fit a standardization step and PCA on the training data.
2. Evaluate **5, 10, 15, 20, and 25** principal components.
3. Train the same four classifiers on each PCA representation.
4. Compare each configuration against its **30-original-feature, no-PCA baseline** from Experiment 1.
5. Analyze explained variance, F1-score, recall, and training/prediction time.

**Expected outputs:**

- PCA cumulative explained variance plot
- Classification performance versus number of principal components
- Discussion of when dimensionality reduction helps or hurts each classifier

**Note:** The dataset contains coded/discrete features, so PCA may not improve predictive performance. This is itself a useful experimental finding.

## Experiment 3 — K-Means Clustering

**Objective:** Examine whether unlabeled website features naturally form groups aligned with the known classes.

**Procedure:**

1. Apply K-Means with **k = 2** to standardized features, without using labels to fit clusters.
2. Compare cluster memberships with the true phishing/legitimate labels only for evaluation.
3. Report cluster composition and metrics such as **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**.
4. Visualize the data with a 2D PCA projection, coloring points by cluster membership and, separately, by true label.
5. Optionally explore additional values of k as a supporting analysis.

**Expected outputs:**

- PCA-based 2D cluster visualization
- Cluster-versus-label contingency table
- Interpretation of whether unsupervised clusters align with the classification labels

**Important:** Cluster IDs are arbitrary; cluster 0 should not automatically be treated as legitimate or cluster 1 as phishing. PCA is used for visualization unless an additional PCA-plus-K-Means experiment is explicitly reported.

## Project Schedule

| Period | Milestone | Deliverables |
| --- | --- | --- |
| **Nov 14–20** | Dataset exploration and preprocessing | EDA, cleaned dataset, shared splits |
| **Nov 21–27** | Classification; PCA and clustering development | Four baseline classifiers, initial PCA and K-Means runs |
| **Nov 28–Dec 4** | Combined experiments and evaluation | Final metrics, figures, model comparisons |
| **Dec 5–11** | Report drafting | Complete 2–4 page draft |
| **Dec 12–19** | Reproducibility checks and submission | Final `project.pdf` submitted by Dec 19, 11:59 PM |

## Team Responsibilities

| Member 1 | Member 2 | Shared |
| --- | --- | --- |
| Dataset loading and preprocessing | PCA experiments | Experimental design and evaluation protocol |
| Classification models | K-Means clustering | Result interpretation |
| Baseline performance evaluation | Dimensionality-reduction evaluation | Report writing and final review |

Both members will review each other's results. Tasks may be adjusted according to availability.

## Planned Repository Structure

```text
phishing-website-detection/
├── README.md
├── requirements.txt
├── data/                  # Dataset files or download instructions
├── notebooks/             # Exploration and visualization
├── src/
│   ├── preprocessing.py
│   ├── classification.py
│   ├── pca_experiments.py
│   └── clustering.py
├── results/               # Tables and generated figures
└── report/
    └── project.pdf
```

> This directory structure is planned. Files will be added as experiments are completed.

## Tools and Libraries

- Python
- pandas and NumPy
- scikit-learn
- matplotlib (optionally seaborn for plots)

## Final Deliverables

- Reproducible code for preprocessing and the three experiments
- Classification comparison table and confusion matrices
- PCA explained variance and performance plots
- K-Means clustering evaluation and visualization
- A **2–4 page report** containing introduction, problem formulation, dataset, algorithms, experiments, discussion, and the required AI usage disclosure

## Results

**Work in progress.** Experimental results and conclusions will be added after training and evaluation. No performance values are claimed at this stage.

## References

- UCI Machine Learning Repository. [Phishing Websites (Dataset 327)](https://archive.ics.uci.edu/dataset/327/phishing+websites).

## AI Usage Disclosure

The final course report will disclose any AI tools actually used and how they were used, as required by the assignment. This README was drafted with assistance from ChatGPT and should be reviewed by the project team before publication.
