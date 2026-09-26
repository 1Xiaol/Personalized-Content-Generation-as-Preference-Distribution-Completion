# Evaluation Protocol

## Overview

Both datasets use the same normalized embedding geometry, chordal Gaussian-RBF kernel, per-user metric definitions, and user-level aggregation rules.

## Shared Preprocessing

History captions, held-out targets, and generated captions must be encoded with the same frozen text encoder and L2-normalized. The number of generated samples `K` is declared by the input and must be identical across compared methods for every user.

## Embedding Metrics

Let `G={g_i}_{i=1}^K` be generated samples, `T={t_m}_{m=1}^M` held-out targets, `H={h_j}` observed history, and `sim(x,y)=x^T y`.

**History-nearest** is `mean_i max_j sim(g_i,h_j)`. It is diagnostic and has no preferred direction.

**Target-best** is `mean_m max_i sim(g_i,t_m)`. Higher is better.

**Coverage** is `mean_m min_i arccos(sim(g_i,t_m))`, measured in radians. Lower is better.

**Future-mode coverage** clusters targets only, using average linkage over cosine distance with cut distance `1-tau_clust`. Each generated sample is assigned only to its globally nearest target:

```text
m*(i) = argmax_m sim(g_i, t_m)
```

The assigned mode is supported only when `max_m sim(g_i,t_m) >= tau_hit`. Future-mode coverage is the fraction of target modes supported by at least one generated sample. `tau_clust` and `tau_hit` are recorded separately even when configured to the same value.

**Energy** uses chordal distance:

```text
mean_{i,m} ||g_i-t_m||_2
- [1 / (2K(K-1))] sum_{i != j} ||g_i-g_j||_2
```

Lower is better.

**Unbiased MMD** uses:

```text
k(x,y) = exp(-||x-y||_2^2 / (2 sigma^2))
MMD2-U = offdiag_mean(k(G,G)) + offdiag_mean(k(T,T)) - 2 mean(k(G,T))
```

For each dataset, `sigma` is the median nonzero chord distance from uniformly sampled distinct Train item pairs. Negative unbiased estimates are retained. Users with fewer than two held-out targets are excluded from MMD aggregation only.

## Statistical Aggregation

Metrics are computed per user and macro-averaged. Method comparisons use paired user-level bootstrap resampling. Each reported metric records its effective user count.

## Incremental Valid Coverage

P1 builds target aspects. P2 aggregates direct, related, and uncertain evidence over the complete observed history. Aspect states are assigned in this order:

1. `observed`: at least two direct observations.
2. `uncertain`: not observed and at least one uncertain observation.
3. `weak_observed`: exactly one direct observation.
4. `weak_partial`: no direct observation and at least one related observation.
5. `not_observed`: none of the above.

Incremental targets belong to `weak_observed`, `weak_partial`, or `not_observed` aspects. A target is validly covered when at least one generation has P5 semantic score at least 2 and P3 generation quality equal to 2. Values are averaged over targets and presentation orders within each user, then macro-averaged over users. Users without a comparable incremental target are treated as missing rather than zero.

