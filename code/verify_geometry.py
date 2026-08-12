"""
verify_geometry.py - test the GEOMETRICITY of the MDI-phi space, the
prerequisite for applying algebraic-geometric (algebraization) methods.

A MDI space is "geometric" if the doctrinal relation (entailment vs
contradiction) is (a) carried by distance structure (isometry), (b) lives on a
low-dimensional / spectral-compressed subspace (effective rank, spectral
entropy -- Vista-style), and (c) has a detectable algebraic form: linear vs
polynomial separability tells whether the relation is affine (linear algebra)
or a non-linear algebraic variety (full algebraic geometry).

Checks (in phi-space vs raw mpnet-space, same data):
  A. spectra    : effective rank, spectral entropy, PCA-95% dims
  B. geometry   : isometry AUC (entailment closer than contradiction)
  C. form       : linear / poly-2 / poly-3 / RBF kernel separability
                  (3-fold, entailment vs contradiction)
Interpretation:
  - if phi has lower effective rank than mpnet -> info compressed to a
    low-dim subspace (variety-approximable)
  - if poly/RBF >> linear in phi -> non-linear algebraic structure present
  - if linear ~ poly in phi -> affine/linear geometric structure

Usage:
  python code/verify_geometry.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: stdout + geometry_verify.txt
"""
import argparse
import time
import collections

import numpy as np

from verify_cross_domain import load_contractnli, load_sara, load_willsnli
from verify_rigor import st_encode, auc_effect
from mdi_version import W_MDI, header


def load_type_a(dirpath, max_n):
    rows = []
    for fn in (load_contractnli, load_willsnli, load_sara):
        try:
            rows += fn(dirpath, max_n)
        except FileNotFoundError:
            pass
    return rows


def spectral(X):
    """Effective rank + spectral entropy of the embedding matrix."""
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    er = float(np.exp(ent))
    return er, ent, s


def pca_dims(X, thr=0.95):
    """# principal components to explain thr of the variance."""
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    c = 0.0
    for i, v in enumerate(ev):
        c += v
        if c >= thr:
            return i + 1
    return len(ev)


def sep_acc(X, y, cv=3, seed=0):
    """Kernel-form separability of E vs C. Returns {name: mean acc}."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    res = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for name, kern, deg in (("linear", "linear", 2), ("poly2", "poly", 2),
                            ("poly3", "poly", 3), ("rbf", "rbf", 3)):
        accs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
            if kern == "linear":
                clf = SVC(kernel="linear", C=1.0)
            elif kern == "poly":
                clf = SVC(kernel="poly", degree=deg, C=1.0)
            else:
                clf = SVC(kernel="rbf", C=1.0, gamma="scale")
            clf.fit(Xtr, y[tr])
            accs.append(accuracy_score(y[te], clf.predict(Xte)))
        res[name] = float(np.mean(accs))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default=W_MDI)
    args = ap.parse_args()

    t0 = time.time()
    log = open("geometry_verify.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(header(f"W={args.W}"))
    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")

    # raw mpnet embeddings
    vecs = st_encode(allt, "all-mpnet-base-v2")   # (2n, 768)
    raw = vecs

    # phi embedding
    phi = vecs @ W                                # (2n, 64)

    for tag, F in (("mpnet(768)", raw), ("phi(64)", phi)):
        er, ent, _ = spectral(F)
        d95 = pca_dims(F)
        # isometry on E vs C pairs
        P = F[:n]; H = F[n:]
        dd = np.sqrt(np.sum((P - H) ** 2, axis=1))
        pos = dd[np.array([l == "E" for _, _, l in pairs])]
        neg = dd[np.array([l == "C" for _, _, l in pairs])]
        auc, dcohen = auc_effect(pos, neg)
        emit(f"[{tag}] eff_rank={er:.2f} spec_ent={ent:.2f} PCA95={d95} "
             f"| isometry AUC={auc:.3f} d={dcohen:.2f}")

    # separability: E vs C (binary), both spaces
    ybin = np.array([1 if l == "E" else (0 if l == "C" else -1)
                     for _, _, l in pairs])
    mask = ybin >= 0
    emit("## E vs C separability (3-fold SVM acc)")
    for tag, F in (("mpnet(768)", raw), ("phi(64)", phi)):
        P = F[:n][mask]; H = F[n:][mask]
        y = ybin[mask]
        # pair rep: concat or diff. Use difference (relation vector).
        Xdiff = P - H
        r = sep_acc(Xdiff, y)
        r2 = sep_acc(np.concatenate([P, H], axis=1), y)
        emit(f"[{tag}] diff-rep: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))
        emit(f"[{tag}] concat-rep: " + "  ".join(f"{k}={v:.3f}" for k, v in r2.items()))

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
