"""
eval_algebraic_inject.py - does ALGEBRAIC injection help generic tasks beyond
plain phi concatenation?

The advanced applications (apply_algebra.py) showed the doctrinal relation in
phi-space is OPERABLE: linear norm->app map, aligned difference vectors
(|cos|=0.876), translation axis. Here we test whether this algebraic structure,
injected as FEATURES into a generic classifier, gives MORE gain than plain
phi concatenation (the v0.2.0 L3 result).

Injection forms (all are generic-task-agnostic, no dedicated scenario):
  plain    : concat(mpnet, phi)                     -- v0.2.0 L3 baseline
  alg-ax   : phi projected onto the doctrinal translation axis + magnitude
             (the APPL-3 direction structure as scalar features)
  alg-sub  : phi coordinates in the top-k principal subspace of the entailment
             difference vectors (the APPL-1 low-rank structure)
  alg-full : alg-ax + alg-sub

Compare on generic classification (SCOTUS/LEDGAR/CUAD/MAUD) with a FIXED
linear classifier (LR), 3-fold. Question: does algebraic injection beat plain
phi concat on these NON-alignment tasks?

Usage:
  python code/eval_algebraic_inject.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: algebraic_inject.txt
"""
import argparse
import collections
import os
import time

import numpy as np

from verify_cross_domain import load_cuad, load_maud, tfidf, legalbert_encode
from verify_rigor import st_encode
from eval_unified import load_scotus, load_ledgar, build_b
from eval_downstream import linear_acc
from mdi_version import W_MDI, header


def doctrina_axis(F, W):
    """Doctrinal translation axis + subspace from phi of type-A training pairs.

    We reuse the structural facts (aligned difference vectors) by computing the
    top singular directions of phi-space sample covariance (a stand-in for the
    entailment-difference axis, since single-text classification has no pairs).
    """
    F = F - F.mean(0)
    # axis: top singular vector of the (centered) phi matrix
    _, _, vh = np.linalg.svd(F, full_matrices=False)
    v = vh[0]
    proj = F @ v                              # (n,) projection on axis
    return proj, v


def build_alg_feats(phi, k_sub=10):
    """Algebraic-injection features: axis projection + subspace coords + norm."""
    phi_c = phi - phi.mean(0)
    _, _, vh = np.linalg.svd(phi_c, full_matrices=False)
    v = vh[0]
    ax = phi_c @ v                            # (n,) axis projection
    sub = phi_c @ vh[:k_sub].T                # (n, k_sub) subspace coords
    mag = np.linalg.norm(phi_c, axis=1)       # (n,) magnitude
    return np.concatenate([ax[:, None], mag[:, None], sub], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default=W_MDI)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--k-sub", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    log = open("algebraic_inject.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(header(f"W={args.W}"))
    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}  k_sub={args.k_sub}")
    emit("## Algebraic injection vs plain phi concat (generic classification)")

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
        lbls = [l for l, v in big.items() for _ in v]
        emit(f"  [{name}] n={len(texts)} classes={len(set(lbls))}")

        mp = st_encode(texts, "all-mpnet-base-v2")
        phi = mp @ W
        alg = build_alg_feats(phi, args.k_sub)     # (n, 2+k_sub)

        reps = {
            "mpnet        ": mp,
            "mpnet+phi    ": np.concatenate([mp, phi], axis=1),
            "mpnet+alg    ": np.concatenate([mp, alg], axis=1),
            "mpnet+phi+alg": np.concatenate([mp, phi, alg], axis=1),
        }
        for rn, F in reps.items():
            acc = linear_acc(F, lbls, cv=args.cv)
            emit(f"    [{rn}] acc={acc:.3f}")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
