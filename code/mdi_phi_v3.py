"""
mdi_phi_v3.py - MDI-phi with ALGEBRAIC-STRUCTURE loading (translation
consistency), built on the advanced-application finding.

v2b loaded P2 (isometry), P3 (monotonicity E<N<C), P5 (Lipschitz).
The advanced applications (apply_algebra.py) then revealed that the doctrinal
relation is OPERABLE: APPL-3 showed entailment is translation-like
(d_E = h - p is a stable "entailment translation"), APPL-1 that norm->app is a
linear map, APPL-2 that the difference vector linearly discriminates E/N/C.

v3 makes that structure a TRAINING TARGET: entailment difference vectors must
concentrate on a single "doctrinal translation axis". This should make the
space more algebraic -> stronger advanced applications (positive feedback).

  L = L_hinge(P2) + lam3*L_mono(P3) + lam_t*L_trans   [+ P5 spectral norm]

P_trans (translation consistency):
    v = top singular vector of entailment difference vectors (per-epoch)
    L_trans = -mean_i (d_i . v)^2 / mean_j ||d_j||^2     (projection share)
  pushing entailment differences onto one axis -> "entailment = translation".

Gradient: d_i = delta_i W (delta_i = vec[p]-vec[h]); d_i . v = delta_i . (W v);
  dL/dW += -2 * (d_i . v) / mean||d||^2 * outer(delta_i, v).

Usage:
  python code/mdi_phi_v3.py --data-dir <dir> [--epochs 80] [--k 64]
      [--model all-mpnet-base-v2] [--out mdi_W_v3_mpnet.npy]
      [--lam3 1.0] [--lam_t 0.5] [--lip 2.0]
Output: <out>.npy, mdi_train_log.txt (v3 block appended)
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


def make_batch3(pairs, n, rng, b=192):
    ent = [i for i, (p, h, l) in enumerate(pairs) if l == "E"]
    neu = [i for i, (p, h, l) in enumerate(pairs) if l == "N"]
    con = [i for i, (p, h, l) in enumerate(pairs) if l == "C"]
    be = int(min(len(ent), b)); bn = int(min(len(neu), b)); bc = int(min(len(con), b))
    e = rng.choice(ent, be, replace=False) if be else np.array([], dtype=int)
    x = rng.choice(neu, bn, replace=False) if bn else np.array([], dtype=int)
    c = rng.choice(con, bc, replace=False) if bc else np.array([], dtype=int)
    return e, x, c


def train_v3(pairs, vecs, d, k, epochs, seed=0, lr=0.05, margin=0.6,
             lam3=1.0, lam_t=0.5, m3=0.02, lip=2.0, b=192, log=None):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, k)) * 0.1
    n = len(pairs)
    E_idx = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "E"])
    for ep in range(epochs):
        # --- per-epoch translation axis v from entailment differences ---
        dE = (vecs[E_idx] - vecs[E_idx + n]) @ W      # (|E|, k)
        scale2 = float(np.mean(np.sum(dE ** 2, axis=1))) or 1.0
        _, _, vh = np.linalg.svd(dE, full_matrices=False)
        v = vh[0]                                      # top right singular vector
        e, x, c = make_batch3(pairs, n, rng, b)
        B = min(len(e), len(x), len(c))
        if B == 0:
            break
        e, x, c = e[:B], x[:B], c[:B]
        idx = np.concatenate([e, x, c])
        p = vecs[idx] @ W
        h = vecs[idx + n] @ W
        d2 = np.sum((p - h) ** 2, axis=1)
        # --- P2 hinge ---
        grad_d2 = np.zeros(3 * B)
        grad_d2[:B] = 1.0
        grad_d2[2*B:] = -1.0 * (d2[2*B:] < margin * margin).astype(float)
        # --- P3 per-triplet monotonicity ---
        dE3 = d2[:B]; dN3 = d2[B:2*B]; dC3 = d2[2*B:]
        t1 = (dE3 - dN3 + m3 > 0).astype(float)
        t2 = (dN3 - dC3 + m3 > 0).astype(float)
        grad_d2[:B] += lam3 * t1
        grad_d2[B:2*B] += lam3 * (t2 - t1)
        grad_d2[2*B:] += lam3 * (-t2)
        # --- accumulate hinge/mono gradient ---
        gW = np.zeros_like(W)
        for i in range(len(idx)):
            delta_g = vecs[idx[i]] - vecs[idx[i] + n]
            r = p[i] - h[i]
            gW += grad_d2[i] * 2.0 * np.outer(delta_g, r)
        gW /= max(1, len(idx))
        # --- P_trans translation consistency (on entailment batch) ---
        if lam_t > 0 and len(e) > 0:
            dE_b = (vecs[e] - vecs[e + n]) @ W        # (B, k)
            proj = dE_b @ v                           # (B,)
            gW_t = np.zeros_like(W)
            for i in range(len(e)):
                delta_i = vecs[e[i]] - vecs[e[i] + n]
                gW_t += -2.0 * (proj[i] / scale2) * np.outer(delta_i, v)
            gW += lam_t * gW_t / max(1, len(e))
        W -= lr * gW
        # --- P5 spectral normalization ---
        s2 = np.linalg.norm(W, ord=2)
        if s2 > lip:
            W *= lip / s2
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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--model", default="all-mpnet-base-v2")
    ap.add_argument("--out", default="mdi_W_v3_mpnet.npy")
    ap.add_argument("--lam3", type=float, default=1.0)
    ap.add_argument("--lam_t", type=float, default=0.5)
    ap.add_argument("--m3", type=float, default=0.05)
    ap.add_argument("--lip", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.6)
    args = ap.parse_args()

    t0 = time.time()
    log = open("mdi_train_log.txt", "a", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit("=" * 70)
    emit("MDI-phi v3: algebraic-structure loading (translation consistency)")
    emit(f"lam3={args.lam3} lam_t={args.lam_t} m3={args.m3} lip={args.lip} "
         f"lr={args.lr} margin={args.margin}")
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")
    vecs = st_encode(allt, args.model)
    d = vecs.shape[1]
    emit(f"base embeddings dim={d} model={args.model}")
    W = train_v3(pairs, vecs, d, args.k, args.epochs,
                 lr=args.lr, margin=args.margin, lam3=args.lam3,
                 lam_t=args.lam_t, m3=args.m3, lip=args.lip, log=log)
    np.save(args.out, W)
    emit(f"saved {args.out} shape={W.shape} total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
