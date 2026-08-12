"""
verify_subspace.py - CATEGORY SUBSPACE SPECTRAL ANALYSIS.

Question this experiment answers (user 2026-08-13):
  MDI-phi gives a strong explanation path (isometry, traceability) but only a
  small accuracy gain on classification. Why? Two competing hypotheses:

  H1 (downstream blind): phi DOES give classes a geometric structure (each
     class is a low-dim subspace), but centroid / LR / concat classifiers are
     "subspace-blind" -- they average or fit boundaries, discarding it.
  H2 (no structure): phi genuinely has no class structure (classes are not
     aligned with the doctrinal relation), so the small gain is real.

We adjudicate with SUBSPACE spectral analysis, not centroid averaging:

  A. per-class spectrum : PCA per class -> effective rank (subspace dim),
     spectral radius, top principal directions. Classes that live on low-dim
     subspaces have small effective rank; overlapping classes share directions.
  B. subspace principal angles : Grassmann angles between class subspaces
     (via SVD of the principal-basis overlap matrix). Near-0 angle =
     overlapping, near-90 = separated. We report the SMALLEST principal angle
     (worst overlap) and the mean.
  C. subspace-distance classifier : assign a sample to the class whose
     subspace best explains it (smallest projection residual). This is
     "subspace-aware" and uses the structure that H1 claims exists. Compare
     against LR and nearest-centroid baselines in phi vs mpnet.

Interpretation:
  - If phi classes have SMALLER effective rank and LARGER separation angles
    than mpnet, AND the subspace classifier beats centroid/LR in phi -> H1:
    phi has usable class structure, downstream was blind -> keep optimizing.
  - If phi classes show LOW rank but OVERLAPPING (small angles) and the
    subspace classifier is no better -> H2 boundary -> document it honestly.

Usage:
  python code/verify_subspace.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: subspace_verify.txt
"""
import argparse
import collections
import os
import time

import numpy as np

from verify_cross_domain import load_cuad, load_maud
from verify_rigor import st_encode
from eval_unified import load_scotus, load_ledgar
from mdi_version import W_MDI_MPNET, header


# ---------------------------------------------------------------------------
# per-class spectral analysis
# ---------------------------------------------------------------------------
def class_spectrum(X, y):
    """Per-class effective rank + spectral radius. Returns dict c -> dict."""
    out = {}
    for c in sorted(set(y)):
        F = X[y == c]
        F = F - F.mean(0)
        s = np.linalg.svd(F, compute_uv=False)
        s = s[s > 1e-9]
        p = s / (s.sum() + 1e-12)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        er = float(np.exp(ent))
        out[c] = {"eff_rank": er, "entropy": ent, "radius": float(s.max()),
                  "n": len(F)}
    return out


def subspace_angles(X, y, k=8):
    """Grassmann principal angles between class subspaces (via top-k PCs).

    For each class we take its top-k PCA directions; the principal angle
    between two subspaces A, B is arccos of the largest singular value of
    A^T B (smallest angle = worst overlap). Returns mean + min over pairs.
    """
    cs = sorted(set(y))
    bases = {}
    for c in cs:
        F = X[y == c] - X[y == c].mean(0)
        _, _, vh = np.linalg.svd(F, full_matrices=False)
        kk = min(k, vh.shape[0], F.shape[0])
        bases[c] = vh[:kk].T if kk > 0 else np.zeros((X.shape[1], 1))
    angles = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            A = bases[cs[i]]; B = bases[cs[j]]
            # principal angle = arccos(largest singular value of A^T B)
            sv = np.linalg.svd(A.T @ B, compute_uv=False)
            th = float(np.rad2deg(np.arccos(np.clip(sv.max(), -1, 1))))
            angles.append(th)
    if not angles:
        return 0.0, 0.0
    return float(np.mean(angles)), float(np.min(angles))


# ---------------------------------------------------------------------------
# subspace-distance classifier (subspace-aware)
# ---------------------------------------------------------------------------
def subspace_dist_classify(X, y, cv=3, seed=0, k=8):
    """Assign each test sample to the class subspace that best explains it
    (smallest projection residual onto the class's top-k PCA basis).
    CV only on the test fold; class subspaces learned on the training fold.
    """
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    from collections import Counter
    min_cls = min(Counter(y).values())
    kk = cv
    while kk > 1 and min_cls < kk:
        kk -= 1
    if kk < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        bases = {}
        for c in set(y[tr]):
            F = X[tr][y[tr] == c]
            F = F - F.mean(0)
            if F.shape[0] < 2:
                bases[c] = np.zeros((X.shape[1], 1))
                continue
            _, _, vh = np.linalg.svd(F, full_matrices=False)
            kk2 = min(k, vh.shape[0], F.shape[0])
            bases[c] = vh[:kk2].T if kk2 > 0 else np.zeros((X.shape[1], 1))
        preds = []
        for x in X[te]:
            # projection residual onto each class subspace
            best_c, best_r = None, 1e30
            for c, B in bases.items():
                r = x - B @ (B.T @ x)
                res = float(np.linalg.norm(r))
                if res < best_r:
                    best_r, best_c = res, c
            preds.append(best_c)
        accs.append(np.mean(np.array(preds) == y[te]))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default=W_MDI_MPNET)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    log = open("subspace_verify.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(header(f"W={args.W}"))
    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## category-subspace spectral analysis: does phi give classes "
         "usable low-dim structure?")
    emit("   (H1: downstream was blind vs H2: no structure)")

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

        # FAIR CONTROL: mpnet projected to the SAME dimension as phi (64-d via
        # PCA). Removes the dimension confound -- if phi's smaller subspace
        # angles are just a low-dim artifact, PCA64 will show the same.
        mp_c = mp - mp.mean(0)
        _, _, vh = np.linalg.svd(mp_c, full_matrices=False)
        mp_pca64 = mp_c @ vh[:64].T

        # A. per-class effective rank (mean over classes)
        sp_mp = class_spectrum(mp, lbls)
        sp_ph = class_spectrum(phi, lbls)
        er_mp = np.mean([v["eff_rank"] for v in sp_mp.values()])
        er_ph = np.mean([v["eff_rank"] for v in sp_ph.values()])
        emit(f"    [A per-class eff-rank] mpnet={er_mp:.1f} phi={er_ph:.1f} "
             f"(lower = class lives on lower-dim subspace)")

        # B. Grassmann principal angles between class subspaces
        am_mp, amin_mp = subspace_angles(mp, lbls, k=args.k)
        am_ph, amin_ph = subspace_angles(phi, lbls, k=args.k)
        am_p64, amin_p64 = subspace_angles(mp_pca64, lbls, k=args.k)
        emit(f"    [B subspace angles] mpnet={am_mp:.1f}°({amin_mp:.1f}) | "
             f"phi={am_ph:.1f}°({amin_ph:.1f}) | "
             f"mpnet-PCA64={am_p64:.1f}°({amin_p64:.1f}) "
             f"(higher = classes more separated; PCA64 is the dim-matched control)")

        # C. subspace-distance classifier vs baselines (LR / centroid)
        from eval_downstream import linear_acc
        from verify_prototype import centroid_acc
        sub_mp = subspace_dist_classify(mp, lbls, cv=args.cv, k=args.k)
        sub_ph = subspace_dist_classify(phi, lbls, cv=args.cv, k=args.k)
        sub_p64 = subspace_dist_classify(mp_pca64, lbls, cv=args.cv, k=args.k)
        lr_mp = linear_acc(mp, lbls, cv=args.cv)
        lr_ph = linear_acc(phi, lbls, cv=args.cv)
        lr_p64 = linear_acc(mp_pca64, lbls, cv=args.cv)
        ce_mp = centroid_acc(mp, lbls, cv=args.cv)
        ce_ph = centroid_acc(phi, lbls, cv=args.cv)
        emit(f"    [C subspace-classifier] phi: sub={sub_ph:.3f} "
             f"LR={lr_ph:.3f} centroid={ce_ph:.3f}")
        emit(f"        mpnet: sub={sub_mp:.3f} LR={lr_mp:.3f} "
             f"centroid={ce_mp:.3f}")
        emit(f"        mpnet-PCA64: sub={sub_p64:.3f} LR={lr_p64:.3f} "
             f"(dim-matched control)")
        # verdict
        sub_gain = sub_ph - lr_ph
        emit(f"    [C verdict] phi subspace-classifier over phi-LR: "
             f"{sub_gain:+.3f} | phi sub over mpnet sub: {sub_ph-sub_mp:+.3f}")

    emit(f"total time {time.time()-t0:.1f}s")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
