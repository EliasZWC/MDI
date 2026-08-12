"""
verify_space_ops.py - systematically verify OTHER mathematical operations on
the unified MDI representation space (beyond alignment).

MDI constructs a unified computable representation space; alignment is one
verified use. Here we probe the space with several independent math operations
on held-out / cross-domain data (CUAD / MAUD real contract clauses, which the
alignment-trained phi did NOT see in this pairing form):

  OP-1 Projection (subspace geometry)
      - doctrinal vectors of CUAD clause categories: do same-category vectors
        project closer to their category centroid (prototype alignment)?
        metric: same-centroid cosine > different-centroid cosine (alignment
        margin), in phi vs raw mpnet.
  OP-2 Mapping (linear transform fit)
      - fit a linear map from phi space back to mpnet (invertibility /
        reconstruction): can phi-space (64-d) reconstruct mpnet (768-d)?
        metric: R2 of ridge reconstruction (lossy? informative?).
  OP-3 Vector algebra (composition)
      - category centroids in phi: is the BETWEEN-centroid structure (pairwise
        distances) stable / discriminative? metric: centroid separation ratio
        (avg intra vs inter centroid distance) — is phi a usable "space" for
        algebra (not just pairwise text distances)?
  OP-4 Transform invariance (fit stability)
      - does a simple linear probe (ridge) fit better / more stably on phi
        than on raw mpnet at equal effective dimension? (space usability for
        fitting)

The point: these are operations ON THE SPACE, not new "alignment tasks". If
they work, MDI-phi is a general computable space, not an alignment-only tool.

Usage:
  python code/verify_space_ops.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: space_ops.txt
"""
import argparse
import collections
import time

import numpy as np

from verify_cross_domain import load_cuad, load_maud
from verify_rigor import st_encode


def op1_projection_align(phi, lbls, mp):
    """OP-1: category-prototype alignment margin (phi vs mpnet)."""
    out = {}
    for tag, F in (("mpnet", mp), ("phi", phi)):
        F = F - F.mean(0)
        cents = {c: F[lbls == c].mean(0) for c in set(lbls)}
        same, diff = [], []
        for i in range(len(F)):
            cs = {c: float(F[i] @ cents[c] /
                           (np.linalg.norm(F[i]) * np.linalg.norm(cents[c]) + 1e-12))
                  for c in cents}
            same.append(cs[lbls[i]])
            others = [v for k, v in cs.items() if k != lbls[i]]
            diff.append(max(others))
        out[tag] = float(np.mean(same) - np.mean(diff))
    return out


def op2_reconstruct(phi, mp):
    """OP-2: linear reconstruction phi -> mpnet (R2)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    Xtr, Xte, Ytr, Yte = train_test_split(phi, mp, test_size=0.3, random_state=0)
    r = Ridge(alpha=1.0).fit(Xtr, Ytr)
    pred = r.predict(Xte)
    ss_res = np.sum((Yte - pred) ** 2)
    ss_tot = np.sum((Yte - Yte.mean(0)) ** 2)
    return 1.0 - ss_res / ss_tot


def op3_centroid_space(phi, lbls, mp):
    """OP-3: centroid separation (intra vs inter), usable space for algebra?"""
    out = {}
    for tag, F in (("mpnet", mp), ("phi", phi)):
        F = F - F.mean(0)
        cents = {c: F[lbls == c].mean(0) for c in set(lbls)}
        cs = np.array([cents[c] for c in sorted(cents)])
        D = np.linalg.norm(cs[:, None] - cs[None, :], axis=2)
        intra = np.mean([D[i, i] for i in range(len(cs))])  # 0 by def
        iu = np.triu_indices(len(cs), 1)
        inter = D[iu].mean()
        out[tag] = float(inter)  # separation of category centroids
    return out


def op4_probe_fit(phi, lbls, mp):
    """OP-4: linear probe fit stability (R2 of predicting a category basis)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(np.asarray(lbls))
    out = {}
    for tag, F in (("mpnet", mp), ("phi", phi)):
        # one-hot targets
        K = len(set(y))
        Y = np.zeros((len(y), K))
        Y[np.arange(len(y)), y] = 1
        scores = cross_val_score(Ridge(alpha=1.0), F, Y, cv=3,
                                 scoring="neg_mean_squared_error")
        out[tag] = float(-scores.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default="mdi_W_v2b_mpnet.npy")
    args = ap.parse_args()

    t0 = time.time()
    log = open("space_ops.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## math operations on the unified MDI space (cross-domain clauses)")

    for name, fn in [("CUAD", load_cuad), ("MAUD", load_maud)]:
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
        phi = mp @ W

        p = op1_projection_align(phi, lbls, mp)
        emit(f"  [OP-1 projection] same-vs-diff centroid cosine: "
             f"mpnet={p['mpnet']:.4f} phi={p['phi']:.4f} "
             f"(higher = phi projects classes more cleanly)")

        r2 = op2_reconstruct(phi, mp)
        emit(f"  [OP-2 mapping] phi->mpnet reconstruction R2 = {r2:.3f} "
             f"(lossy/lossless?)")

        s = op3_centroid_space(phi, lbls, mp)
        emit(f"  [OP-3 vector algebra] centroid separation: "
             f"mpnet={s['mpnet']:.4f} phi={s['phi']:.4f} "
             f"(phi usable as algebra space?)")

        f = op4_probe_fit(phi, lbls, mp)
        emit(f"  [OP-4 transform fit] one-hot probe MSE: "
             f"mpnet={f['mpnet']:.4f} phi={f['phi']:.4f} "
             f"(lower = space easier to fit)")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
