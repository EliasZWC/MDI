"""
verify_phi_v2.py - verify that the theory-loaded v2 projection actually
satisfies the P3 (monotonicity) and P5 (Lipschitz) axioms on the training
domain, and compare P5 against the v1 (bare-hinge) projection.

Reads mdi_W_mpnet.npy (v1) and mdi_W_v2_mpnet.npy (v2), encodes the Type-A
pairs, and reports:
  dE / dN / dC  (mean doctrina distances by class)   -> P3: dE < dN < dC
  spectral norm ||W||_2 (<= ||W||_F)                 -> P5: v2 should be smaller
  train-domain isometry AUC for both                 -> check P2 not destroyed
Usage:
  python code/verify_phi_v2.py --data-dir <dir> --max 600
Output: stdout + phi_v2_verify.txt
"""
import argparse
import time

import numpy as np

from verify_cross_domain import load_contractnli, load_sara, load_willsnli
from verify_rigor import st_encode, auc_effect


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
    ap.add_argument("--W1", default="mdi_W_mpnet.npy")
    ap.add_argument("--W2", default="mdi_W_v2_mpnet.npy")
    args = ap.parse_args()

    t0 = time.time()
    log = open("phi_v2_verify.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    W1 = np.load(args.W1)
    W2 = np.load(args.W2)
    emit(f"v1 {args.W1} shape={W1.shape}  v2 {args.W2} shape={W2.shape}")

    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    vecs = st_encode(allt, "all-mpnet-base-v2")
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")

    for tag, W in (("v1", W1), ("v2", W2)):
        P = vecs[:n] @ W
        H = vecs[n:] @ W
        dd = np.sqrt(np.sum((P - H) ** 2, axis=1))
        mE = dd[np.array([l == "E" for _, _, l in pairs])].mean()
        mN = dd[np.array([l == "N" for _, _, l in pairs])].mean()
        mC = dd[np.array([l == "C" for _, _, l in pairs])].mean()
        pos = dd[np.array([l == "E" for _, _, l in pairs])]
        neg = dd[np.array([l == "C" for _, _, l in pairs])]
        auc, dcohen = auc_effect(pos, neg)
        lip2 = np.linalg.norm(W, ord=2)
        lipF = np.linalg.norm(W, ord="fro")
        mono = "OK (E<N<C)" if mE < mN < mC else ("PARTIAL" if mE < mC else "VIOLATED")
        emit(f"[{tag}] dE={mE:.3f} dN={mN:.3f} dC={mC:.3f}  P3={mono}  "
             f"|AUC={auc:.3f} d={dcohen:.2f} | ||W||_2={lip2:.3f} ||W||_F={lipF:.3f}")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
