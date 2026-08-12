"""
verify_algebra.py - test whether algebraic-geometry methods can ALGEBRAIZE
the MDI-phi doctrinal space.

Geometricity (verify_geometry.py) established: phi is low-dim, isometry-
preserving, dominantly affine/linear. Here we ask the algebraization question:
is the doctrinal relation captured by POLYNOMIAL (algebraic) structure beyond
linear? This decides WHICH algebraic tools apply:

  (a) if a polynomial map p -> h (norm -> application) beats linear -> the
      relation is a non-linear algebraic map -> full algebraic geometry
  (b) if E/C separability gains from polynomial features (vs linear) -> the
      classes live on an algebraic hypersurface, not a flat subspace
  (c) if the entailment-difference vectors d = h - p span a LOW-rank subspace
      -> the relation is a low-dim linear sub-variety (variety-approximable)

Checks (in phi-space, v2b projection, Type-A pairs):
  A. polynomial mapping   : Ridge on p->h with deg 1/2/3 features (RMSE)
  B. relation subspace    : PCA of d = h-p (entailment) -> explained variance
  C. polynomial separability: E vs C with linear vs poly-2 vs poly-3 features
                              (logistic regression, 5-fold)

Interpretation:
  - A: poly < linear RMSE  -> non-linear algebraic map exists
  - B: top-k PCs explain ~all variance -> relation = low-rank sub-variety
  - C: poly >> linear acc  -> classes on algebraic hypersurface

Usage:
  python code/verify_algebra.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: stdout + algebra_verify.txt
"""
import argparse
import time

import numpy as np

from verify_cross_domain import load_contractnli, load_sara, load_willsnli
from verify_rigor import st_encode


def load_type_a(dirpath, max_n):
    rows = []
    for fn in (load_contractnli, load_willsnli, load_sara):
        try:
            rows += fn(dirpath, max_n)
        except FileNotFoundError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default="mdi_W_v2b_mpnet.npy")
    args = ap.parse_args()

    t0 = time.time()
    log = open("algebra_verify.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")
    vecs = st_encode(allt, "all-mpnet-base-v2")
    phi = vecs @ W
    P = phi[:n]; H = phi[n:]
    E_idx = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "E"])
    C_idx = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "C"])

    # ---- A. polynomial mapping p -> h (Ridge, deg 1/2/3) ----
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    emit("## A. polynomial map norm->application  (Ridge RMSE, 70/30 split)")
    for tag, idx in (("entail", E_idx), ("contrd", C_idx)):
        X = P[idx]; Y = H[idx]
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3,
                                              random_state=0)
        for deg in (1, 2, 3):
            if deg > 2 and len(idx) < 400:
                continue   # too few samples for deg-3 on 64d
            pipe = make_pipeline(StandardScaler(),
                                 PolynomialFeatures(degree=deg),
                                 Ridge(alpha=1.0))
            pipe.fit(Xtr, Ytr)
            rmse = float(np.sqrt(np.mean((pipe.predict(Xte) - Yte) ** 2)))
            emit(f"  [{tag}] deg-{deg} RMSE={rmse:.4f}")

    # ---- B. relation subspace: PCA of d = h-p (entailment) ----
    from sklearn.decomposition import PCA
    emit("## B. relation-vector subspace  (PCA of d=h-p, entailment)")
    dE = H[E_idx] - P[E_idx]
    dC = H[C_idx] - P[C_idx]
    for tag, D in (("entail-d", dE), ("contrd-d", dC)):
        pca = PCA().fit(D)
        evr = pca.explained_variance_ratio_
        for k in (5, 10, 20, 64):
            emit(f"  [{tag}] top-{k} PCs explain {evr[:k].sum():.3f}")
    # subspace alignment: do entail/contrd difference vectors share direction?
    vE = PCA(n_components=1).fit(dE).components_[0]
    vC = PCA(n_components=1).fit(dC).components_[0]
    cos = float(np.abs(vE @ vC) / (np.linalg.norm(vE) * np.linalg.norm(vC)))
    emit(f"  [dir-align] |cos(vE, vC)| = {cos:.3f}")

    # ---- C. polynomial separability E vs C (logistic, 5-fold) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    emit("## C. E vs C polynomial separability (logistic, 5-fold acc)")
    y = np.concatenate([np.ones(len(E_idx)), np.zeros(len(C_idx))])
    X = np.concatenate([dE, dC])   # use difference vectors
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for deg in (1, 2, 3):
        accs = []
        for tr, te in skf.split(X, y):
            pipe = make_pipeline(StandardScaler(),
                                 PolynomialFeatures(degree=deg),
                                 LogisticRegression(max_iter=2000))
            pipe.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], pipe.predict(X[te])))
        emit(f"  [poly-{deg}] E-vs-C acc={np.mean(accs):.3f}")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
