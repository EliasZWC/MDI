"""
mdi_phi_v2.py - MDI-phi with THEORY-LOADED regularization (P3 + P5).

Depth application of MDI: instead of a bare contrastive hinge, the loss now
implements two of the theory's own axioms so the trained projection is a
faithful instantiation of the isomorphism, not just an embedding head.

  L = L_hinge(ent, con)                       # P2 isometry (alignment)
    + lam3 * L_monotonicity(ent, neu, con)    # P3 monotonicity
    + lam5 * ||W||_F^2                         # P5 Lipschitz (via weight decay)

P3 monotonicity (doctrinal signal never decreases structural value):
    distance must order E < N < C on average:
      L3 = max(0, dE - dN + m3) + max(0, dN - dC + m3)
  where dX = mean over class X of squared distance ||g(p)W - g(h)W||^2.

P5 Lipschitz (micro perturbations -> bounded drift): for the linear map
phi(x)=xW the Lipschitz constant is bounded by ||W||_2 <= ||W||_F, so
weight decay on W directly enforces the property (trainable, differentiable).

Usage:
  python code/mdi_phi_v2.py --data-dir <dir> [--epochs 20] [--k 64]
      [--model all-mpnet-base-v2] [--out mdi_W_v2_mpnet.npy]
      [--lam3 1.0] [--lam5 1e-4] [--m3 0.02]
Output: <out>.npy, mdi_train_log.txt (v2 block appended)
"""
import argparse
import os
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


def make_batch3(pairs, n, rng, b=192):
    """Sample balanced E/N/C batches; return (p_indices, class_flags)."""
    ent = [i for i, (p, h, l) in enumerate(pairs) if l == "E"]
    neu = [i for i, (p, h, l) in enumerate(pairs) if l == "N"]
    con = [i for i, (p, h, l) in enumerate(pairs) if l == "C"]
    be = int(min(len(ent), b)); bn = int(min(len(neu), b)); bc = int(min(len(con), b))
    e = rng.choice(ent, be, replace=False) if be else np.array([], dtype=int)
    x = rng.choice(neu, bn, replace=False) if bn else np.array([], dtype=int)
    c = rng.choice(con, bc, replace=False) if bc else np.array([], dtype=int)
    return e, x, c


def train_v2(pairs, vecs, d, k, epochs, seed=0, lr=0.05, margin=0.6,
             lam3=1.0, lam5=1e-4, m3=0.02, lip=2.0, b=192, log=None):
    """Theory-loaded training, CORRECTED gradient scale + explicit P5.

    P2 isometry   : hinge on (E,C) distance, same as v1.
    P3 monotonicity: per-triplet constraint d(E)_i + m3 < d(N)_i < d(C)_i - m3,
                     gradient NOT normalized by class size (v2.0 bug: /nn diluted
                     P3 ~200x below hinge, so it never moved N). Triplets are
                     random E/N/C tuples; penalty is pairwise hinge.
    P5 Lipschitz  : explicit spectral normalization W <- W * min(1, lip/||W||_2)
                     after each step -> hard upper bound ||W||_2 <= lip, i.e.
                     ||phi(x+d)-phi(x)|| <= lip*||d|| (true Lipschitz constant
                     of the linear map). Stronger and more faithful than wd.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, k)) * 0.1
    n = len(pairs)
    for ep in range(epochs):
        e, x, c = make_batch3(pairs, n, rng, b)
        B = min(len(e), len(x), len(c))
        if B == 0:
            break
        e, x, c = e[:B], x[:B], c[:B]
        idx = np.concatenate([e, x, c])
        # forward
        p = vecs[idx] @ W
        h = vecs[idx + n] @ W
        d2 = np.sum((p - h) ** 2, axis=1)   # (3B,)
        # --- P2 hinge coefficients ---
        grad_d2 = np.zeros(3 * B)
        grad_d2[:B] = 1.0                                    # E: push closer
        grad_d2[2*B:] = -1.0 * (d2[2*B:] < margin * margin).astype(float)  # C: push far
        # --- P3 per-triplet monotonicity (correct scale, no /class-size) ---
        dE = d2[:B]; dN = d2[B:2*B]; dC = d2[2*B:]
        t1 = (dE - dN + m3 > 0).astype(float)   # penalize dE >= dN
        t2 = (dN - dC + m3 > 0).astype(float)   # penalize dN >= dC
        grad_d2[:B] += lam3 * t1
        grad_d2[B:2*B] += lam3 * (t2 - t1)
        grad_d2[2*B:] += lam3 * (-t2)
        # accumulate gradient dL/dW = sum_i grad_d2_i * 2 * outer(delta_g, r)
        gW = np.zeros_like(W)
        for i in range(len(idx)):
            delta_g = vecs[idx[i]] - vecs[idx[i] + n]
            r = p[i] - h[i]
            gW += grad_d2[i] * 2.0 * np.outer(delta_g, r)
        gW /= max(1, len(idx))
        # optional soft weight decay (auxiliary, small)
        if lam5 > 0:
            gW += lam5 * 2.0 * W
        W -= lr * gW
        # P5: explicit Lipschitz bound via spectral normalization
        s2 = np.linalg.norm(W, ord=2)
        if s2 > lip:
            W *= lip / s2
        # report last epoch
        if ep == epochs - 1:
            Pp = vecs[np.arange(n)] @ W
            Hh = vecs[np.arange(n) + n] @ W
            dd = np.sqrt(np.sum((Pp - Hh) ** 2, axis=1))
            pos = dd[np.array([l == "E" for _, _, l in pairs])]
            neg = dd[np.array([l == "C" for _, _, l in pairs])]
            auc, dcohen = auc_effect(pos, neg)
            dEf = dd[np.array([l == "E" for _, _, l in pairs])].mean()
            dNf = dd[np.array([l == "N" for _, _, l in pairs])].mean()
            dCf = dd[np.array([l == "C" for _, _, l in pairs])].mean()
            lip2 = np.linalg.norm(W, ord=2)
            line = (f"  epoch {ep}: train AUC={auc:.3f} d={dcohen:.2f} "
                    f"| dE={dEf:.3f} dN={dNf:.3f} dC={dCf:.3f} | ||W||_2={lip2:.3f}")
            print(line)
            if log is not None:
                log.write(line + "\n")
                log.flush()
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--model", default="all-mpnet-base-v2")
    ap.add_argument("--out", default="mdi_W_v2_mpnet.npy")
    ap.add_argument("--lam3", type=float, default=1.0)
    ap.add_argument("--lam5", type=float, default=0.0)
    ap.add_argument("--m3", type=float, default=0.02)
    ap.add_argument("--lip", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.6)
    args = ap.parse_args()

    t0 = time.time()
    log = open("mdi_train_log.txt", "a", encoding="utf-8")
    emit = lambda s: (print(s), log.write(s + "\n"), log.flush())[1]

    emit("=" * 70)
    emit("MDI-phi v2: theory-loaded training (P3 monotonicity + P5 Lipschitz)")
    emit(f"lam3={args.lam3} lam5={args.lam5} m3={args.m3} lip={args.lip} "
         f"lr={args.lr} margin={args.margin}")
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")
    vecs = st_encode(allt, args.model)
    d = vecs.shape[1]
    emit(f"base embeddings dim={d} model={args.model}")
    W = train_v2(pairs, vecs, d, args.k, args.epochs,
                 lr=args.lr, margin=args.margin,
                 lam3=args.lam3, lam5=args.lam5, m3=args.m3,
                 lip=args.lip, log=log)
    np.save(args.out, W)
    emit(f"saved {args.out} shape={W.shape} total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
