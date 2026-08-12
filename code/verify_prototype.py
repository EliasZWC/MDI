"""
verify_prototype.py - test the user's insight: classification IS alignment.

If classification is viewed as "sample-to-class-prototype alignment"
(nearest-centroid: each class is a prototype/centroid, a sample is classified
by the prototype it aligns best with), then MDI-phi's isometry guarantee
(what should be close IS close) should transfer: in phi-space the sample-to-
its-class-centroid distance should be smaller than to other centroids.

Previous algebraic-injection experiments used LogisticRegression (a
discriminative boundary), which does NOT use the alignment structure -- that
may be why phi showed no gain on classification. A prototype (nearest-centroid)
classifier DOES use it. This experiment tests:

  does MDI-phi improve nearest-centroid classification over raw mpnet?

Checks (generic classification tasks, nearest-centroid, stratified k-fold):
  mpnet          : centroid classifier on 768-d mpnet
  mpnet+phi      : centroid classifier on concat(mpnet, phi)
  phi            : centroid classifier on 64-d phi alone
  mpnet+alg      : centroid classifier on concat(mpnet, algebraic features)

Also report the "alignment margin": distance to own-centroid vs nearest other
centroid (in phi vs mpnet) -- the direct evidence of "classification=alignment".

Usage:
  python code/verify_prototype.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: prototype_verify.txt
"""
import argparse
import collections
import os
import time

import numpy as np

from verify_cross_domain import load_cuad, load_maud
from verify_rigor import st_encode
from eval_unified import load_scotus, load_ledgar


def centroid_acc(X, y, cv=3, seed=0):
    """Nearest-centroid classifier, stratified k-fold accuracy."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    from collections import Counter
    min_cls = min(Counter(y).values())
    k = cv
    while k > 1 and min_cls < k:
        k -= 1
    if k < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        cents = {}
        for c in set(y[tr]):
            cents[c] = Xtr[y[tr] == c].mean(0)
        preds = []
        for x in Xte:
            dists = {c: np.linalg.norm(x - cents[c]) for c in cents}
            preds.append(min(dists, key=dists.get))
        accs.append(np.mean(np.array(preds) == y[te]))
    return float(np.mean(accs))


def alignment_margin(X, y):
    """Distance to own centroid vs nearest other centroid (mean ratio)."""
    X = X - X.mean(0)
    cents = {c: X[y == c].mean(0) for c in set(y)}
    own = []; other = []
    for i in range(len(X)):
        ds = {c: np.linalg.norm(X[i] - cents[c]) for c in cents}
        own.append(ds[y[i]])
        other.append(min(v for k, v in ds.items() if k != y[i]))
    return float(np.mean(other) / (np.mean(own) + 1e-12))


def build_alg_feats(phi, k_sub=10):
    phi_c = phi - phi.mean(0)
    _, _, vh = np.linalg.svd(phi_c, full_matrices=False)
    v = vh[0]
    ax = phi_c @ v
    sub = phi_c @ vh[:k_sub].T
    mag = np.linalg.norm(phi_c, axis=1)
    return np.concatenate([ax[:, None], mag[:, None], sub], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default="mdi_W_v2b_mpnet.npy")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--k-sub", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    log = open("prototype_verify.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## classification-as-alignment: nearest-centroid classifier")
    emit("   (does MDI-phi help? own-centroid vs other-centroid distance)")

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
        phi = mp @ W
        alg = build_alg_feats(phi, args.k_sub)
        reps = {
            "mpnet        ": mp,
            "phi          ": phi,
            "mpnet+phi    ": np.concatenate([mp, phi], axis=1),
            "mpnet+alg    ": np.concatenate([mp, alg], axis=1),
        }
        for rn, F in reps.items():
            acc = centroid_acc(F, lbls, cv=args.cv)
            emit(f"    [{rn}] centroid-acc={acc:.3f}")
        # alignment margin: own-centroid distance vs other-centroid distance
        m_mp = alignment_margin(mp, lbls)
        m_ph = alignment_margin(phi, lbls)
        emit(f"    [align-margin] mpnet={m_mp:.3f}  phi={m_ph:.3f}  "
             f"(higher = clearer own-vs-other alignment)")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
