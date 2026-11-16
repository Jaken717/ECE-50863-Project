# Experiment 3 — K-Means (k = 2) results

Scaler and K-Means fit on the training set only, without labels. Majority-label mapping learned on train: cluster 0 → legitimate, cluster 1 → legitimate.

## Agreement with true labels

| split   |    n |    ARI |    NMI |   purity |   mapped_accuracy |   mapped_precision |   mapped_recall |   mapped_f1 |
|:--------|-----:|-------:|-------:|---------:|------------------:|-------------------:|----------------:|------------:|
| train   | 7794 | 0.0015 | 0.0001 |   0.5625 |            0.5625 |                  0 |               0 |           0 |
| val     | 1653 | 0.008  | 0.0046 |   0.5535 |            0.5402 |                  0 |               0 |           0 |
| test    | 1608 | 0.001  | 0.0002 |   0.5473 |            0.5473 |                  0 |               0 |           0 |

## Cluster × label contingency

|   cluster | split   |   Legitimate |   Phishing |   Total |   % phishing |
|----------:|:--------|-------------:|-----------:|--------:|-------------:|
|         0 | train   |         3651 |       2809 |    6460 |         43.5 |
|         1 | train   |          733 |        601 |    1334 |         45.1 |
|         0 | test    |          725 |        592 |    1317 |         45   |
|         1 | test    |          155 |        136 |     291 |         46.7 |

## Seed stability (k = 2, 20 seeds)

ARI vs labels: mean 0.0019, sd 0.0006, range 0.0015–0.0027  
ARI vs reference clustering: min 0.9601

## k sweep (training set)

|   k |   inertia |   silhouette |     ARI |    NMI |   purity |
|----:|----------:|-------------:|--------:|-------:|---------:|
|   2 |    199101 |       0.2786 |  0.0015 | 0.0001 |   0.5625 |
|   3 |    173457 |       0.2888 | -0.0067 | 0.0042 |   0.5625 |
|   4 |    161400 |       0.115  |  0.3019 | 0.2356 |   0.8057 |
|   5 |    154419 |       0.117  |  0.291  | 0.2218 |   0.8038 |
|   6 |    150062 |       0.0981 |  0.196  | 0.1849 |   0.7774 |
|   7 |    145476 |       0.1051 |  0.2022 | 0.1927 |   0.8154 |
|   8 |    142039 |       0.1043 |  0.1251 | 0.1697 |   0.781  |
|   9 |    138059 |       0.1013 |  0.1221 | 0.1908 |   0.8196 |
|  10 |    135808 |       0.1008 |  0.1122 | 0.1564 |   0.7988 |

## Exploratory: k = 4 composition

k = 4 was not chosen a priori; it is shown because it had the highest ARI in the sweep (choosing k by ARI uses labels, so treat this as exploratory).

|   cluster | split   |   Legitimate |   Phishing |   Total |   % phishing |
|----------:|:--------|-------------:|-----------:|--------:|-------------:|
|         0 | train   |          614 |        429 |    1043 |         41.1 |
|         1 | train   |          563 |        291 |     854 |         34.1 |
|         2 | train   |          451 |       2347 |    2798 |         83.9 |
|         3 | train   |         2756 |        343 |    3099 |         11.1 |
|         0 | test    |          136 |        109 |     245 |         44.5 |
|         1 | test    |          131 |         47 |     178 |         26.4 |
|         2 | test    |           74 |        497 |     571 |         87   |
|         3 | test    |          539 |         75 |     614 |         12.2 |
