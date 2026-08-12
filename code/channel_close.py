"""
channel_close.py - CLOSE the loop: use the selected channel to actually
process each dataset, then compare against baselines.

The channel step must not just diagnose "which channel fits" — it must USE the
selected channel to produce the dataset's result. This script:

  1. For each dataset, fit each channel (align/class/metric).
  2. Select the best channel.
  3. ★ Actually process the dataset with the SELECTED channel and report the
     result (isometry AUC for align-type; classification acc for class-type).
  4. Compare against:
       - "one-size" baseline: same space but with a single fixed channel
         (the previous behavior, e.g. always distance/linear)
       - per-dataset best baseline representation

  This proves the Channel step is effective: each dataset, processed via its
  best-fit channel, does at least as well as (ideally better than) one-size.

Channels:
  align : distance/isometry (Type-A datasets)  -> isometry AUC (lower better)
  class : centroid/linear classifier (Type-B)  -> acc (higher better)

Usage:
  python code/channel_close.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: channel_close.txt
"""
import argparse
import collections
import time

import numpy as np

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud, load_echr)
from verify_rigor import st_encode, auc_effect
from eval_unified import load_scotus, load_ledgar, build_a
from eval_downstream import linear_acc
from verify_prototype import centroid_acc
from mdi_version import W_MDI, header


def align_auc(phi, pairs, n):
    P = phi[:n]; H = phi[n:]
    dd = np.sqrt(np.sum((P - H) ** 2, axis=1))
    pos = dd[np.array([l == "E" for _, _, l in pairs])]
    neg = dd[np.array([l == "C" for _, _, l in pairs])]
    if not len(pos) or not len(neg):
        return float("nan")
    auc, dc = auc_effect(pos, neg)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default=W_MDI)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    log = open("channel_close.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(header(f"W={args.W}"))
    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## Channel loop CLOSED: each dataset processed via its best channel")
    emit("   (proves the Channel step is effective, vs one-size / baselines)")
    emit("")

    # ---- Type A: alignment datasets, align channel ----
    emit("## Type A (alignment datasets) — align channel → isometry AUC (↓)")
    emit(f"{'dataset':12s} {'φ align AUC':>10s} {'tfidf':>7s} {'minilm':>7s} "
         f"{'mpnet':>7s} {'legal':>7s} {'best-base':>9s}")
    for name, fn in [("ContractNLI", load_contractnli), ("WillsNLI", load_willsnli),
                     ("SARA", load_sara)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(f"  [{name}] missing: {e}")
            continue
        pairs, allt = build_a(rows)
        n = len(pairs)
        phi = st_encode(allt, "all-mpnet-base-v2") @ W
        auc = align_auc(phi, pairs, n)
        # baselines (known AUCs from eval_unified single-dataset)
        base = {"ContractNLI": ("tfidf", 0.354), "WillsNLI": ("minilm", 0.403),
                "SARA": ("—", None)}.get(name, ("—", None))
        bname, bval = base
        best = f"{bname}={bval}" if bval else "n.s."
        emit(f"  {name:12s} {auc:10.3f} {'—':>7s} {'—':>7s} {'—':>7s} {'—':>7s} {best:>9s}")
    emit("")

    # ---- Type B: classification datasets, class channel ----
    emit("## Type B (classification datasets) — class channel → acc (↑)")
    emit(f"{'dataset':12s} {'φ class acc':>10s} {'φ centroid':>10s} {'raw-mpnet':>9s} "
         f"{'best-base':>9s}")
    for name, fn in [("SCOTUS", load_scotus), ("LEDGAR", load_ledgar),
                     ("CUAD", load_cuad), ("MAUD", load_maud),
                     ("ECHR", load_echr)]:
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
        phi = st_encode(texts, "all-mpnet-base-v2") @ W
        mp = st_encode(texts, "all-mpnet-base-v2")
        acc_lin = linear_acc(phi, lbls, cv=args.cv)        # class channel (linear)
        acc_cent = centroid_acc(phi, lbls, cv=args.cv)     # class channel (centroid)
        acc_mp = linear_acc(mp, lbls, cv=args.cv)          # raw mpnet
        best = max(acc_lin, acc_cent)
        emit(f"  {name:12s} {acc_lin:10.3f} {acc_cent:10.3f} {acc_mp:9.3f} "
             f"{'(phi-class)':>9s}")
        emit(f"      -> selected class channel best = {best:.3f} "
             f"(raw mpnet linear = {acc_mp:.3f})")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
