"""
verify_welldefined.py - MDI P1 well-definedness across domains

Validates P1 (well-definedness / paraphrase invariance) on all 6 datasets:
a doctrinally-neutral rewrite of a legal text should map to a near-identical
point in the representation space.

  Invariant: dist(orig, rewrite) << dist(orig, random)   (rewrite-stability)
  Report   : rewrite-pair dist vs random-pair dist, AUC/d, permutation null
  Progress : tqdm on encoding / permutation.

Usage:
  python code/verify_welldefined.py --data-dir <dir> [--max 300]
                                     [--models tfidf,minilm] [--perm 100]
Output: welldefined_log.txt
"""
import argparse
import os
import random
import re
import time

import numpy as np
from scipy import stats
from tqdm import tqdm

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud, load_echr, tfidf)
from verify_rigor import st_encode, auc_effect, null_perm

# Doctrinally-neutral rewrites (no doctrinal content changes)
NEUTRAL = [
    (r"\bpursuant to\b", "under"),
    (r"\bin accordance with\b", "under"),
    (r"\bnotwithstanding\b", "despite"),
    (r"\bprovided that\b", "provided"),
    (r"\bincluding but not limited to\b", "including"),
    (r"\bsuch\b", "that"),
    (r"\bwithout prejudice to\b", "subject to"),
    (r"\bshall be\b", "is"),
]
# Semantic-level rewrites (touch surface, may or may not change doctrine)
SEMANTIC = [
    (r"\bshall\b", "must"),
    (r"\bterminate\b", "cancel"),
    (r"\bdamages\b", "compensation"),
    (r"\bindemnify\b", "hold harmless"),
]

REPS = {
    "tfidf": tfidf,
    "minilm": lambda t: st_encode(t, "all-MiniLM-L6-v2"),
}


def paraphrase(t, pairs):
    out = t
    for pat, rep in pairs:
        out = re.sub(pat, rep, out, flags=re.I)
    return out if out != t else t


def texts_of(rows, max_n):
    """Type A: premise side only; Type B: instance text."""
    out = []
    for r in rows[:max_n]:
        if isinstance(r[0], str) and isinstance(r[1], str) and len(r) == 2:
            out.append(r[0]) if r[0] and len(r[0]) > 20 else None
    return out


def run_dataset(name, rows, log, models, perm, max_n):
    # collect a text pool (premise for Type A, text for Type B)
    pool = []
    for r in rows[:max_n]:
        t = r[0] if isinstance(r[0], str) else str(r[0])
        if len(t) > 20:
            pool.append(t)
    if not pool:
        return
    # build rewrite pairs per table
    rng = random.Random(5)
    for pname, pairs in [("neutral", NEUTRAL), ("semantic", SEMANTIC)]:
        rw_idx, rw_txt = [], []
        for i, t in enumerate(pool):
            tp = paraphrase(t, pairs)
            if tp != t:
                rw_idx.append(i)
                rw_txt.append(tp)
        if not rw_idx:
            emit(log, f"  [{name}] {pname}: no rewrites hit")
            continue
        # random pairs from pool
        rand = [(rng.randrange(len(pool)), rng.randrange(len(pool)))
                for _ in range(len(rw_idx) * 2)]
        allt = pool + rw_txt
        n0 = len(pool)
        for m in models:
            vecs = REPS[m](allt)
            dr = []
            for i, tp in zip(rw_idx, range(n0, n0 + len(rw_idx))):
                a, b = vecs[i], vecs[tp]
                dr.append(1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            dr = np.array(dr)
            dd = []
            for i, j in rand:
                a, b = vecs[i], vecs[j]
                dd.append(1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            dd = np.array(dd)
            auc, d = auc_effect(dr, dd)   # rewrite should be closer (AUC<0.5)
            nm, ns, pct = null_perm(dr, dd, auc, perm, seed=0)
            emit(log, f"  [{name}] {pname:8s} [{m:6s}] n={len(dr)} "
                      f"rw={dr.mean():.4f} rand={dd.mean():.4f} "
                      f"AUC={auc:.3f} d={d:.2f} null={nm:.3f}±{ns:.3f} pctile={pct:.3f}")


def emit(log, line):
    print(line)
    log.write(line + "\n")
    log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--models", default="tfidf,minilm")
    ap.add_argument("--perm", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    log = open("welldefined_log.txt", "w", encoding="utf-8")
    models = [x for x in args.models.split(",") if x in REPS]
    print("=" * 72)
    print("MDI P1 well-definedness (rewrite-stability) across 6 datasets")
    print("=" * 72)
    log.write("MDI P1 well-definedness\n")

    for name, fn in [("ContractNLI", load_contractnli), ("SARA", load_sara),
                     ("WillsNLI", load_willsnli), ("CUAD", load_cuad),
                     ("MAUD", load_maud), ("ECHR", load_echr)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        run_dataset(name, rows, log, models, args.perm, args.max)

    emit(log, f"total time {time.time() - t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
