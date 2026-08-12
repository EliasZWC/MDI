"""
mdi_unified.py - MDI unified representation: contrastive alignment of
normative-application pairs (the framework's own method)

Trains a linear projection phi(x)=xW (R^d -> R^k) with a contrastive hinge loss
so that doctrinal support pairs (entailment) map close and contradiction pairs
map far. This is the "mathematical method" of MDI — learned from the dataset's
own normative-application relation, not an off-the-shelf representation.

  Input features : minilm/mpnet embeddings of texts
  Supervision    : Type-A entailment pairs (ContractNLI, WillsNLI, SARA)
  Loss           : y*d^2 + (1-y)*max(0, margin^2 - d^2), d = ||g(p)W - g(h)W||
  Output         : W (d x k) saved to mdi_W.npy; train AUC reported

Usage:
  python code/mdi_unified.py --data-dir <dir> [--epochs 20] [--k 64]
Output: mdi_W.npy, mdi_train_log.txt
"""
import argparse
import os
import time

import numpy as np

from verify_cross_domain import load_contractnli, load_sara, load_willsnli
from verify_rigor import st_encode, auc_effect

MODEL = "all-MiniLM-L6-v2"


def load_type_a(dirpath, max_n):
    rows = []
    for fn in (load_contractnli, load_willsnli, load_sara):
        try:
            rows += fn(dirpath, max_n)
        except FileNotFoundError:
            pass
    return rows


def make_batch(pairs, vecs, n, rng):
    """Sample a balanced batch of (p_idx, h_idx, y=1 entail / 0 contradict)."""
    ent = [(i, n + i, 1.0) for i, (p, h, l) in enumerate(pairs) if l == "E"]
    con = [(i, n + i, 0.0) for i, (p, h, l) in enumerate(pairs) if l == "C"]
    b = int(min(len(ent), len(con), 256))
    e = rng.choice(len(ent), b, replace=False)
    c = rng.choice(len(con), b, replace=False)
    return [ent[i] for i in e] + [con[i] for i in c], vecs


def train(pairs, vecs, d, k, epochs, seed=0, lr=0.05, margin=0.6):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, k)) * 0.1
    n = len(pairs)
    n_batch = max(1, int(256 * 2))
    for ep in range(epochs):
        batch, _ = make_batch(pairs, vecs, n, rng)
        idx = np.array([x[0] for x in batch])
        y = np.array([x[2] for x in batch])
        p = vecs[idx] @ W           # (B, k)
        h = vecs[idx + n] @ W
        diff = p - h
        d2 = np.sum(diff * diff, axis=1)
        # hinge: y*d2 + (1-y)*max(0, m^2 - d2)
        grad_d2 = np.where(y > 0.5, 1.0, -1.0 * (d2 < margin * margin).astype(float))
        # dL/dW = sum_i grad_i * 2 * outer(delta_g, r),  delta_g=(d), r=(k)
        gW = np.zeros_like(W)
        for i in range(len(idx)):
            delta_g = vecs[idx[i]] - vecs[idx[i] + n]   # (d,)
            r = p[i] - h[i]                              # (k,)
            gW += grad_d2[i] * 2.0 * np.outer(delta_g, r)
        W -= lr * gW / len(idx)
        # report
        if ep == epochs - 1:
            Pp = vecs[np.arange(n)] @ W
            Hh = vecs[np.arange(n) + n] @ W
            dd = np.sqrt(np.sum((Pp - Hh) ** 2, axis=1))
            pos = dd[np.array([l == "E" for _, _, l in pairs])]
            neg = dd[np.array([l == "C" for _, _, l in pairs])]
            auc, dcohen = auc_effect(pos, neg)
            print(f"  epoch {ep}: train AUC={auc:.3f} d={dcohen:.2f}")
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="mdi_W.npy", help="output weight file")
    args = ap.parse_args()

    t0 = time.time()
    log = open("mdi_train_log.txt", "w", encoding="utf-8")

    def emit(s):
        print(s)
        log.write(s + "\n")
        log.flush()

    emit("MDI unified representation (contrastive normative-application alignment)")
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    emit(f"type-A pairs n={n} (entail={sum(1 for *_ , l in pairs if l=='E')}, "
         f"contrad={sum(1 for *_ , l in pairs if l=='C')})")
    vecs = st_encode(allt, args.model)
    d = vecs.shape[1]
    emit(f"base embeddings dim={d}")
    W = train(pairs, vecs, d, args.k, args.epochs)
    np.save(args.out, W)
    emit(f"saved {args.out} shape={W.shape} total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
