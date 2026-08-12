"""
verify_semantic_align.py - TEST THE USER'S HYPOTHESIS (2026-08-13):

  "这些法律数据集实际上做的是语义对齐的任务，而对于法理不要求？"
  (Are the legal classification datasets really SEMANTIC-ALIGNMENT tasks
   that do NOT require doctrinal/legal reasoning?)

Motivation. MDI-phi gives a strong explanation/isometry path but no class
discrimination gain (see verify_subspace.py: phi class subspaces overlap even
vs dim-matched PCA64). One clean explanation: the classification datasets
(SCOTUS / LEDGAR / CUAD / MAUD) are solvable by SEMANTIC ALIGNMENT alone --
labels ≈ "same-class texts are semantically similar" -- so they never need the
doctrinal structure phi provides. This script tests that directly with three
independent pieces of evidence:

  E1  within/between-class cohesion : mean cosine similarity of same-class
      pairs minus different-class pairs, in mpnet AND phi space. A large gap
      => labels are decided by semantic similarity (alignment), not doctrine.

  E2  UNSUPERVISED alignment classifier : KMeans over the embeddings to K
      clusters, then Hungarian-match clusters to the true labels. This uses
      ZERO label supervision and ZERO doctrine -- pure semantic alignment.
      If E2 accuracy is close to supervised LR => the task is essentially
      an alignment task (labels live in semantic geometry).

  E3  doctrinal-axis vs label alignment : learn the doctrinal axis from the
      Type-A E/N/C triples (the direction that separates E from C in phi),
      project each task's samples onto that axis, and measure how much of the
      label structure is explained by the projection. If labels are NOT
      aligned to the doctrinal axis => the task does not need doctrine.

Verdict logic:
  - E2(unsup) ≈ LR   AND   E1 large   => hypothesis CONFIRMED: tasks are
    semantic-alignment benchmarks, not doctrinal ones.
  - E3 small          => reinforces: the doctrinal axis phi carries is
    orthogonal to what these labels need.

Usage:
  python code/verify_semantic_align.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: semantic_align.txt
"""
import argparse
import collections
import os
import time

import numpy as np

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud)
from verify_rigor import st_encode
from eval_unified import load_scotus, load_ledgar
from mdi_version import W_MDI_MPNET, header


def load_type_a(dirpath, max_n):
    """Type-A NLI triples used to define the DOCTRINAL axis."""
    rows = []
    for fn in (load_contractnli, load_willsnli, load_sara):
        try:
            rows += fn(dirpath, max_n)
        except FileNotFoundError:
            pass
    return rows


def cosine_pairs(X, same, diff, n_sample=4000):
    """Mean cosine similarity of random same/different-label pairs."""
    rng = np.random.RandomState(0)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cs, cd = [], []
    n = len(X)
    idx = np.arange(n)
    for _ in range(n_sample):
        i, j = rng.randint(0, n, 2)
        if same[i] == same[j]:
            cs.append(Xn[i] @ Xn[j])
        else:
            cd.append(Xn[i] @ Xn[j])
    return float(np.mean(cs)), float(np.mean(cd))


def unsup_align_acc(X, y, k=None, seed=0):
    """KMeans over embeddings -> Hungarian-match clusters to true labels.

    Pure semantic alignment: NO label supervision, NO classifier training.
    """
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    X = np.asarray(X, dtype=float)
    y = np.asarray(LabelEncoder().fit_transform(np.asarray(y)))  # int labels
    K = len(set(y)) if k is None else k
    if len(set(y)) < 2 or len(X) < K * 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xs)
    # cost matrix: cluster -> label counts (maximize correct = minimize neg)
    C = np.zeros((K, K))
    for a, b in zip(km.labels_, y):
        C[a, b] += 1
    ri, ci = linear_sum_assignment(-C)
    total = max(1, len(y))
    return float(sum(C[ri[k], ci[k]] for k in range(K)) / total)


def doctrinal_axis(phi, pairs, n):
    """Direction in phi separating E (entail) from C (contradict) triples.

    Uses the difference vector d = h - p: the doctrinal axis is the mean of
    (d_C - d_E) -- the direction along which contradiction vs entailment
    separates. Returns a unit vector in phi space.
    """
    P = phi[:n]; H = phi[n:]
    d = H - P
    E = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "E"])
    C = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "C"])
    if len(E) == 0 or len(C) == 0:
        return None
    ax = d[C].mean(0) - d[E].mean(0)
    nrm = np.linalg.norm(ax)
    if nrm < 1e-12:
        return None
    return ax / nrm


def label_sep_along_axis(X, y, ax):
    """How much of the label structure is explained by projection onto ax.

    Measured as the ratio of between-class variance to total variance of the
    projections (eta^2 of the projection w.r.t. labels). 0 => labels are
    orthogonal to the axis; 1 => labels perfectly live on the axis.
    """
    p = X @ ax
    tot = np.var(p)
    if tot < 1e-12:
        return 0.0
    gmean = p.mean()
    between = 0.0
    for c in set(y):
        pc = p[y == c]
        between += len(pc) * (pc.mean() - gmean) ** 2
    between /= len(p)
    return float(between / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default=W_MDI_MPNET)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    log = open("semantic_align.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(header(f"W={args.W}"))
    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## are legal classification tasks really SEMANTIC-ALIGNMENT tasks "
         "(no doctrine needed)?")
    emit("   E1 cohesion | E2 unsupervised-align acc | E3 doctrinal-axis fit")

    # ---- learn the doctrinal axis from Type-A triples (in phi space) ----
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    vecs = st_encode(allt, "all-mpnet-base-v2")
    phi = vecs @ W
    ax = doctrinal_axis(phi, pairs, n)
    if ax is not None:
        cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
        emit(f"  doctrinal axis from {n} type-A triples "
             f"(E={cnt('E')} N={cnt('N')} C={cnt('C')}) in phi space")
    else:
        emit("  [doctrinal axis unavailable -- E or C empty]")

    # ---- per-dataset evidence ----
    for name, fn in [("SCOTUS", load_scotus), ("LEDGAR", load_ledgar),
                     ("CUAD", load_cuad), ("MAUD", load_maud)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(f"  [{name}] missing: {e}")
            continue
        n_lab = len(set(l for _, l in rows))
        per = min(50, max(1, int(1000 / max(1, n_lab))))
        g = collections.defaultdict(list)
        for t, l in rows:
            g[l].append(t)
        big = {l: v[:per] for l, v in g.items() if len(v) >= 2}
        texts = [t for v in big.values() for t in v]
        lbls = np.array([l for l, v in big.items() for _ in v])
        emit(f"  [{name}] n={len(texts)} classes={len(set(lbls))}")

        mp = st_encode(texts, "all-mpnet-base-v2")
        ph = mp @ W

        # E1 cohesion
        cs_mp, cd_mp = cosine_pairs(mp, lbls, lbls)
        cs_ph, cd_ph = cosine_pairs(ph, lbls, lbls)
        emit(f"    [E1 cohesion] mpnet same={cs_mp:.3f} diff={cd_mp:.3f} "
             f"gap={cs_mp-cd_mp:+.3f} | phi same={cs_ph:.3f} diff={cd_ph:.3f} "
             f"gap={cs_ph-cd_ph:+.3f}")

        # E2 unsupervised alignment classifier vs supervised LR
        from eval_downstream import linear_acc
        unsup_mp = unsup_align_acc(mp, lbls)
        unsup_ph = unsup_align_acc(ph, lbls)
        lr_mp = linear_acc(mp, lbls, cv=args.cv)
        lr_ph = linear_acc(ph, lbls, cv=args.cv)
        emit(f"    [E2 unsup-align] mpnet kmeans={unsup_mp:.3f} "
             f"vs LR={lr_mp:.3f} (ratio {unsup_mp/max(1e-9,lr_mp):.2f})")
        emit(f"        phi       kmeans={unsup_ph:.3f} "
             f"vs LR={lr_ph:.3f} (ratio {unsup_ph/max(1e-9,lr_ph):.2f})")

        # E3 doctrinal-axis fit
        if ax is not None:
            eta_ph = label_sep_along_axis(ph, lbls, ax)
            # null axis (random unit vector) for reference in the same space
            rng = np.random.RandomState(1)
            rv = rng.randn(ph.shape[1]); rv /= np.linalg.norm(rv)
            eta_null = label_sep_along_axis(ph, lbls, rv)
            emit(f"    [E3 doctrinal-axis fit] phi eta2={eta_ph:.3f} "
                 f"(null={eta_null:.3f}) "
                 f"(0=labels orthogonal to doctrinal axis)")
        else:
            emit("    [E3 skipped: no doctrinal axis]")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
