"""
post_mapping.py - MDI Post-Mapping: channel library + fit-to-select (DATASET level)

MDI paradigm step 2 (Channel): after Mapping builds the Doctrinal Space, each
DATASET is processed by the mathematical treatment (channel) that fits its
structure best — like fitting data to several distributions and choosing the
best-fit family. A channel is a dataset-level operation, NOT a downstream
classifier choice.

Channels (dataset-level mathematical treatments on the φ space):
  align  : alignment structure — is the dataset's pairwise relation
           (E vs C) isometry-preserved?  [dist AUC, lower=better]
  class  : category structure — are the dataset's class centroids well-separated
           in φ?  [own-vs-other centroid distance ratio]
  metric : metric structure — within-class vs between-class cosine margin?

For each dataset we score every channel and report the best-fit one, showing
that different datasets select different channels (the space is general; the
channel adapts to the dataset).

Usage:
  python code/post_mapping.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: post_mapping.txt
"""
import argparse
import collections
import time

import numpy as np

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud, load_echr)
from verify_rigor import st_encode, auc_effect
from eval_unified import load_scotus, load_ledgar, build_a
from mdi_version import W_MDI, header


def ch_align_dist(phi, pairs, n):
    """ALIGN channel: isometry fidelity, normalized vs shuffle-label baseline."""
    P = phi[:n]; H = phi[n:]
    dd = np.sqrt(np.sum((P - H) ** 2, axis=1))
    pos = dd[np.array([l == "E" for _, _, l in pairs])]
    neg = dd[np.array([l == "C" for _, _, l in pairs])]
    if not len(pos) or not len(neg):
        return float("nan"), float("nan")
    auc, dcohen = auc_effect(pos, neg)
    # shuffle baseline: expected AUC ~ 0.5 with spread; gain over 0.5
    gain = (0.5 - auc) * 2        # 0..1, 1 = perfect alignment
    return gain, auc


def _shuffle_ref(F, lbls, metric_fn, n_shuf=20, seed=0):
    """Mean of a metric under shuffled labels (no-structure baseline)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_shuf):
        sh = rng.permutation(lbls)
        vals.append(metric_fn(F, sh))
    return float(np.mean(vals))


def ch_class_sep(phi, lbls):
    """CLASS channel: own vs other centroid ratio, vs shuffle baseline."""
    def _m(F, l):
        F = F - F.mean(0)
        cents = {c: F[l == c].mean(0) for c in set(l)}
        same, diff = [], []
        for i in range(len(F)):
            ds = {c: np.linalg.norm(F[i] - cents[c]) for c in cents}
            same.append(ds[l[i]])
            diff.append(min(v for k, v in ds.items() if k != l[i]))
        return float(np.mean(diff) / (np.mean(same) + 1e-12))
    real = _m(phi, lbls)
    ref = _shuffle_ref(phi, lbls, _m)
    # gain: how much above the no-structure baseline (ratio ref)
    return max(0.0, min(1.0, (real - ref) / max(ref, 1e-9)))


def ch_metric_retr(phi, lbls):
    """METRIC channel: within/between cosine margin, vs shuffle baseline."""
    def _m(F, l):
        F = F - F.mean(0)
        Pn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        cls = collections.defaultdict(list)
        for i, ll in enumerate(l):
            cls[ll].append(i)
        rng = np.random.default_rng(0)
        same, diff = [], []
        for i in range(len(F)):
            own = cls[l[i]]
            other = [j for j in range(len(F)) if l[j] != l[i]]
            if not own or not other:
                continue
            same.append(Pn[i] @ Pn[rng.choice(own)])
            diff.append(Pn[i] @ Pn[rng.choice(other)])
        return float(np.mean(same) - np.mean(diff))
    real = _m(phi, lbls)
    ref = _shuffle_ref(phi, lbls, _m)
    return max(0.0, min(1.0, (real - ref) / max(abs(ref), 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default=W_MDI)
    args = ap.parse_args()

    t0 = time.time()
    log = open("post_mapping.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    emit(header(f"W={args.W}"))
    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    emit("## MDI Post-Mapping: channel fit-and-select (DATASET level)")
    emit("   each dataset picks the channel that fits its structure best")

    for name, fn in [("ContractNLI", load_contractnli), ("WillsNLI", load_willsnli),
                     ("SARA", load_sara)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(f"  [{name}] missing: {e}")
            continue
        pairs, allt = build_a(rows)
        n = len(pairs)
        if n == 0:
            continue
        phi = st_encode(allt, "all-mpnet-base-v2") @ W
        align_fit, auc = ch_align_dist(phi, pairs, n)
        lbls = np.array([{"E": 0, "N": 1, "C": 2}[l] for _, _, l in pairs])
        cls_fit = ch_class_sep(phi[:n], lbls)
        met_fit = ch_metric_retr(phi[:n], lbls)
        scores = {"align": align_fit, "class": cls_fit, "metric": met_fit}
        best = max(scores, key=lambda k: scores[k])
        emit(f"  [{name}] align={align_fit:.3f}(auc={auc:.3f}) "
             f"class={cls_fit:.3f} metric={met_fit:.3f} -> BEST={best}")

    for name, fn in [("CUAD", load_cuad), ("MAUD", load_maud),
                     ("ECHR", load_echr), ("SCOTUS", load_scotus),
                     ("LEDGAR", load_ledgar)]:
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
        cls_fit = ch_class_sep(phi, lbls)
        met_fit = ch_metric_retr(phi, lbls)
        scores = {"class": cls_fit, "metric": met_fit}
        best = max(scores, key=lambda k: scores[k])
        emit(f"  [{name}] class={cls_fit:.3f} metric={met_fit:.3f} -> BEST={best}")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
