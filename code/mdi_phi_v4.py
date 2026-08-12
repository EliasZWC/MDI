"""
mdi_phi_v4.py - MDI-phi with MULTI-TASK supervision (alignment + category).

The space-operation probe (v0.2.5) confirmed: the unified MDI space is
computable, but the current φ's alignment-only supervision biases its
coordinates toward alignment (class structure weak). v4 trains φ with BOTH:

  L = L_align(P2 hinge + P3 monotonicity)     # Type-A: normative-application
    + lam_c * L_cat(structure)                 # Type-B: same-class close
    + [P5 spectral norm]

L_cat (category structure): sample same-class and different-class pairs from
Type-B labeled instances (CUAD/MAUD clause categories); push same-class
phi-distances down, different-class up:

    L_cat = mean d^2(same) - mean d^2(diff)  (as a hinge: max(0, ms - md + m_c))

Goal: the space keeps its alignment strength AND gains category structure —
so that it is a GENERAL computable space, not alignment-only.

Usage:
  python code/mdi_phi_v4.py --data-dir <dir> [--epochs 80] [--k 64]
      [--model all-mpnet-base-v2] [--out mdi_W_v4_mpnet.npy]
      [--lam3 1.0] [--lam_c 0.5] [--m3 0.05] [--lip 2.0]
Output: <out>.npy, mdi_train_log.txt (v4 block appended)
"""
import argparse
import collections
import time

import numpy as np

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud)
from verify_rigor import st_encode, auc_effect


def load_type_a(dirpath, max_n):
    rows = []
    for fn in (load_contractnli, load_willsnli, load_sara):
        try:
            rows += fn(dirpath, max_n)
        except FileNotFoundError:
            pass
    return rows


def load_type_b(dirpath, max_n):
    """Type-B labeled instances: (text, label) for category supervision."""
    rows = []
    for fn in (load_cuad, load_maud):
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


def make_cat_batch(bcat, nb, rng):
    """Same-class and different-class index pairs from Type-B instances."""
    g = collections.defaultdict(list)
    for i, (t, l) in enumerate(bcat):
        g[l].append(i)
    big = {l: v for l, v in g.items() if len(v) >= 2}
    keys = list(big.keys())
    same, diff = [], []
    for _ in range(nb):
        k = rng.choice(keys)
        i, j = rng.choice(big[k], 2, replace=False)
        same.append((i, j))
        a, b2 = rng.choice(keys, 2, replace=False)
        i2 = rng.choice(big[a]); j2 = rng.choice(big[b2])
        diff.append((i2, j2))
    return same, diff


def train_v4(pairs, bcat, vecs_a, vecs_b, d, k, epochs, seed=0, lr=0.05,
             margin=0.6, lam3=1.0, lam_c=0.5, m3=0.05, m_c=0.2,
             lip=2.0, b=192, nb=128, log=None):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, k)) * 0.1
    n = len(pairs)
    for ep in range(epochs):
        # --- alignment batch (Type A) ---
        e, x, c = make_batch3(pairs, n, rng, b)
        B = min(len(e), len(x), len(c))
        if B > 0:
            e, x, c = e[:B], x[:B], c[:B]
            idx = np.concatenate([e, x, c])
            p = vecs_a[idx] @ W
            h = vecs_a[idx + n] @ W
            d2 = np.sum((p - h) ** 2, axis=1)
            grad_d2 = np.zeros(3 * B)
            grad_d2[:B] = 1.0
            grad_d2[2*B:] = -1.0 * (d2[2*B:] < margin * margin).astype(float)
            dE3 = d2[:B]; dN3 = d2[B:2*B]; dC3 = d2[2*B:]
            t1 = (dE3 - dN3 + m3 > 0).astype(float)
            t2 = (dN3 - dC3 + m3 > 0).astype(float)
            grad_d2[:B] += lam3 * t1
            grad_d2[B:2*B] += lam3 * (t2 - t1)
            grad_d2[2*B:] += lam3 * (-t2)
            gW = np.zeros_like(W)
            for i in range(len(idx)):
                delta_g = vecs_a[idx[i]] - vecs_a[idx[i] + n]
                r = p[i] - h[i]
                gW += grad_d2[i] * 2.0 * np.outer(delta_g, r)
            gW /= max(1, len(idx))
        else:
            gW = np.zeros_like(W)

        # --- category batch (Type B) ---
        if lam_c > 0 and len(bcat) >= 2:
            same, diff = make_cat_batch(bcat, nb, rng)
            # concatenate: first nb are same, next nb are diff
            pairs_idx = same + diff
            f0 = vecs_b[np.array([a for a, b2 in pairs_idx])] @ W
            f1 = vecs_b[np.array([b2 for a, b2 in pairs_idx])] @ W
            d2c = np.sum((f0 - f1) ** 2, axis=1)
            gd = np.zeros(len(pairs_idx))
            gd[:len(same)] = 1.0
            # hinge: max(0, ms - md + m_c): grad for diff is -1 when active
            md = d2c[len(same):].mean() if len(same) else 0.0
            ms = d2c[:len(same)].mean() if len(same) else 0.0
            if len(same) and len(diff):
                gd[len(same):] = -1.0 * (ms - md + m_c > 0)
            gWc = np.zeros_like(W)
            for i in range(len(pairs_idx)):
                delta = vecs_b[pairs_idx[i][0]] - vecs_b[pairs_idx[i][1]]
                r2 = f0[i] - f1[i]
                gWc += gd[i] * 2.0 * np.outer(delta, r2)
            gW += lam_c * gWc / max(1, len(pairs_idx))

        W -= lr * gW
        s2 = np.linalg.norm(W, ord=2)
        if s2 > lip:
            W *= lip / s2
        if ep == epochs - 1:
            Pp = vecs_a[np.arange(n)] @ W
            Hh = vecs_a[np.arange(n) + n] @ W
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
    ap.add_argument("--out", default="mdi_W_v4_mpnet.npy")
    ap.add_argument("--lam3", type=float, default=1.0)
    ap.add_argument("--lam_c", type=float, default=0.5)
    ap.add_argument("--m3", type=float, default=0.05)
    ap.add_argument("--m_c", type=float, default=0.2)
    ap.add_argument("--lip", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.6)
    args = ap.parse_args()

    t0 = time.time()
    log = open("mdi_train_log.txt", "a", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit("=" * 70)
    emit("MDI-phi v4: multi-task supervision (alignment + category)")
    emit(f"lam3={args.lam3} lam_c={args.lam_c} m3={args.m3} m_c={args.m_c} "
         f"lip={args.lip} lr={args.lr}")
    pairs = load_type_a(args.data_dir, args.max)
    bcat = load_type_b(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")
    emit(f"type-B labeled instances n={len(bcat)} (for category supervision)")
    vecs_a = st_encode(allt, args.model)
    vecs_b = st_encode([t for t, l in bcat], args.model)
    d = vecs_a.shape[1]
    emit(f"base embeddings dim={d} model={args.model}")
    W = train_v4(pairs, bcat, vecs_a, vecs_b, d, args.k, args.epochs,
                 lr=args.lr, margin=args.margin, lam3=args.lam3,
                 lam_c=args.lam_c, m3=args.m3, m_c=args.m_c,
                 lip=args.lip, log=log)
    np.save(args.out, W)
    emit(f"saved {args.out} shape={W.shape} total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
