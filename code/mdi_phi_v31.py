"""
mdi_phi_v31.py - ORTHOGONALIZED translation-consistency loading.

v3's P_trans used raw projection share  -mean (d.v)^2 / mean||d||^2, whose
gradient compresses the amplitude structure -> broke P3 (dN<dE) and degraded
isometry (AUC 0.285 -> 0.395), even though APPL-1 retrieval improved 3.2x.

v3.1 fixes this by aligning DIRECTIONS ONLY (cosine alignment, amplitude-
invariant): the entailment difference vectors are pushed onto the doctrinal
translation axis v while their magnitudes (the E<N<C ordering) are untouched:

    L_trans = -mean_i cos^2(d_i, v) = -mean_i (d_i.v)^2 / (||d_i||^2 ||v||^2)

Cosine is scale-invariant: gradient rotates direction, does NOT change ||d_i||,
so P3 monotonicity and P2 isometry (amplitude-driven) are preserved while the
direction structure (which powers APPL-1 mapping retrieval) is strengthened.

Exact gradient (d_i = delta_i W; a = d_i.v = delta_i.(Wv); b = ||d_i||^2):
    d(cos^2)/dW = 2/(||v||^2 b^2) * [ a*b*outer(delta_i, v)
                                      - a^2*outer(delta_i, d_i) ]

Usage:
  python code/mdi_phi_v31.py --data-dir <dir> [--epochs 80] [--k 64]
      [--model all-mpnet-base-v2] [--out mdi_W_v31_mpnet.npy]
      [--lam3 1.0] [--lam_t 1.0] [--m3 0.05] [--lip 2.0]
Output: <out>.npy, mdi_train_log.txt (v3.1 block appended)
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


def train_v31(pairs, vecs, d, k, epochs, seed=0, lr=0.05, margin=0.6,
              lam3=1.0, lam_t=1.0, m3=0.05, lip=2.0, b=192, log=None):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, k)) * 0.1
    n = len(pairs)
    E_idx = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "E"])
    for ep in range(epochs):
        dE = (vecs[E_idx] - vecs[E_idx + n]) @ W
        _, _, vh = np.linalg.svd(dE, full_matrices=False)
        v = vh[0]
        nv2 = float(np.sum(v * v)) or 1.0
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
        gW = np.zeros_like(W)
        for i in range(len(idx)):
            delta_g = vecs[idx[i]] - vecs[idx[i] + n]
            r = p[i] - h[i]
            gW += grad_d2[i] * 2.0 * np.outer(delta_g, r)
        gW /= max(1, len(idx))
        # --- P_trans (v3.1): cosine alignment, amplitude-invariant ---
        if lam_t > 0 and len(e) > 0:
            gWt = np.zeros_like(W)
            for i in range(len(e)):
                delta_i = vecs[e[i]] - vecs[e[i] + n]
                di = delta_i @ W
                a = float(di @ v)
                b = float(np.sum(di * di))
                if b < 1e-12:
                    continue
                # d(cos^2)/dW = 2/(nv2 b^2) * [a b outer(d,v) - a^2 outer(d,di)]
                gWt += (2.0 / (nv2 * b * b)) * (
                    a * b * np.outer(delta_i, v) - a * a * np.outer(delta_i, di))
            gW += lam_t * gWt / max(1, len(e))
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
    ap.add_argument("--out", default="mdi_W_v31_mpnet.npy")
    ap.add_argument("--lam3", type=float, default=1.0)
    ap.add_argument("--lam_t", type=float, default=1.0)
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
    emit("MDI-phi v3.1: orthogonalized translation consistency (cosine align)")
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
    W = train_v31(pairs, vecs, d, args.k, args.epochs,
                  lr=args.lr, margin=args.margin, lam3=args.lam3,
                  lam_t=args.lam_t, m3=args.m3, lip=args.lip, log=log)
    np.save(args.out, W)
    emit(f"saved {args.out} shape={W.shape} total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
