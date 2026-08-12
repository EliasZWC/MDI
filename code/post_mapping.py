"""
post_mapping.py - MDI Post-Mapping: channel library + fit-to-select mechanism.

MDI = the mathematization step (mapping texts -> Doctrinal Space). Everything
after is MDI Post-Mapping: operations on the space. This module implements the
Post-Mapping core mechanism -- a CHANNEL LIBRARY plus a FIT-TO-SELECT chooser:

  channels   : different mathematical treatments of the same φ space
               (alignment/distance, centroid, linear, poly-kernel, subspace)
  select     : for a given task, fit each channel (CV), score it, and pick the
               best one -- like fitting data to several distributions and
               choosing the one that fits.

Demo on the benchmark tasks shows WHICH channel wins WHERE, i.e. that the
unified space serves different tasks via different channels (not one-fits-all).

Channels (all on φ, or on the raw base for contrast):
  dist  : distance/nearest-centroid (alignment-style, φ native)
  lin   : linear LR
  poly2 : polynomial-2 kernel SVM
  rbf   : RBF kernel SVM
  sub   : PCA subspace + linear
  base  : raw mpnet + linear (non-φ reference)

Usage:
  python code/post_mapping.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: post_mapping.txt
"""
import argparse
import collections
import os
import time

import numpy as np

from verify_cross_domain import load_cuad, load_maud, load_contractnli
from verify_rigor import st_encode
from eval_unified import load_scotus, load_ledgar, build_a


# ---------------- channel definitions ----------------

def ch_centroid(X, y, cv=3, seed=0):
    """Nearest-centroid (prototype alignment)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        cents = {c: Xtr[y[tr] == c].mean(0) for c in set(y[tr])}
        preds = [min(cents, key=lambda c: np.linalg.norm(x - cents[c])) for x in Xte]
        accs.append(np.mean(np.array(preds) == y[te]))
    return float(np.mean(accs))


def _svm_kernel(X, y, kernel, degree=None, cv=3, seed=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        kw = {"kernel": kernel}
        if kernel == "poly":
            kw["degree"] = degree
        clf = SVC(C=1.0, **kw)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def ch_linear(X, y, cv=3, seed=0):
    return _svm_kernel(X, y, "linear", cv=cv, seed=seed)


def ch_poly2(X, y, cv=3, seed=0):
    return _svm_kernel(X, y, "poly", degree=2, cv=cv, seed=seed)


def ch_rbf(X, y, cv=3, seed=0):
    return _svm_kernel(X, y, "rbf", cv=cv, seed=seed)


def ch_subspace(X, y, cv=3, seed=0, k=16):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    k = min(k, X.shape[1], len(set(y)) * 2)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k), SVC(kernel="linear"))
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(pipe, X, y, cv=cv)
    return float(s.mean())


CHANNELS = {
    "centroid": ch_centroid,
    "linear": ch_linear,
    "poly2": ch_poly2,
    "rbf": ch_rbf,
    "subspace": ch_subspace,
}


# ---------------- fit-to-select ----------------

def select_channel(X, y, cv=3):
    """Fit every channel, score by CV, return {name: score} + best."""
    scores = {}
    for name, fn in CHANNELS.items():
        try:
            scores[name] = fn(X, y, cv=cv)
        except Exception as e:
            scores[name] = float("nan")
    best = max(scores, key=lambda k: scores[k])
    return scores, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default="mdi_W_v2b_mpnet.npy")
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    log = open("post_mapping.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## MDI Post-Mapping: channel library + fit-to-select")
    emit("   (which channel wins WHERE on the unified space)")

    # ---- classification tasks (Type B) ----
    emit("## Classification (fit each channel, pick best)")
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
        phi = st_encode(texts, "all-mpnet-base-v2") @ W
        mp = st_encode(texts, "all-mpnet-base-v2")
        scores, best = select_channel(phi, lbls, args.cv)
        base = ch_linear(mp, lbls, args.cv)
        desc = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
        emit(f"    [phi channels] {desc}")
        emit(f"    [best={best}]  raw-mpnet-linear={base:.3f}")

    # ---- alignment task (Type A) : the "dist" channel is native ----
    emit("## Alignment (dist channel on φ; isometry AUC lower=better)")
    for name, fn in [("ContractNLI", load_contractnli)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(f"  [{name}] missing: {e}")
            continue
        pairs, allt = build_a(rows)
        n = len(pairs)
        phi = st_encode(allt, "all-mpnet-base-v2") @ W
        P = phi[:n]; H = phi[n:]
        dd = np.sqrt(np.sum((P - H) ** 2, axis=1))
        pos = dd[np.array([l == "E" for _, _, l in pairs])]
        neg = dd[np.array([l == "C" for _, _, l in pairs])]
        from verify_rigor import auc_effect
        auc, dcohen = auc_effect(pos, neg)
        emit(f"  [{name}] dist-channel isometry AUC={auc:.3f} (d={dcohen:.2f}) "
             f"-- alignment native channel")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
