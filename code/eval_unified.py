"""
eval_unified.py - evaluate the MDI unified representation phi(x)=xW
(the framework's own method) on 8 datasets, vs off-the-shelf baselines.

For every dataset it reports isometry / structure with effect size + null:
  Type A (pairs): entailment closer than contradiction (E vs C)
  Type B (labels): same-label closer than different-label
Representations compared per dataset: tfidf (lexical), minilm (universal),
mpnet (universal-strong), and MDI phi (learned on type-A pairs of
ContractNLI/WillsNLI/SARA).

Usage:
  python code/eval_unified.py --data-dir <dir> [--W mdi_W.npy] [--max 400]
Output: eval_unified_log.txt
"""
import argparse
import json
import os
import random
import time

import numpy as np
from scipy import stats
from tqdm import tqdm

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud, load_echr, tfidf,
                                 tfidf_bigram, legalbert_encode,
                                 pair_cosine_dists)
from verify_rigor import st_encode, auc_effect, null_perm

# ---- LexGLUE loaders (Type B) ----
def load_scotus(dirpath, max_n):
    rows = json.load(open(os.path.join(dirpath, "LexGLUE", "scotus.json"),
                          encoding="utf-8"))
    rng = random.Random(7); rng.shuffle(rows)
    return [(r["text"], r["label"]) for r in rows[:max_n]]


def load_ledgar(dirpath, max_n):
    rows = json.load(open(os.path.join(dirpath, "LexGLUE", "ledgar.json"),
                          encoding="utf-8"))
    rng = random.Random(7); rng.shuffle(rows)
    return [(r["text"], r["label"]) for r in rows[:max_n]]


def build_a(rows, per=400):
    import collections
    g = collections.defaultdict(list)
    for p, h, l in rows:
        g[l].append((p, h))
    sel = {l: v[:per] for l, v in g.items()}
    pairs = [(p, h, l) for l, v in sel.items() for p, h in v]
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    return pairs, allt


def build_b(rows):
    import collections
    n_lab = len(set(l for _, l in rows))
    per = min(50, max(1, int(1000 / max(1, n_lab))))
    g = collections.defaultdict(list)
    for t, l in rows:
        g[l].append(t)
    big = {l: v[:per] for l, v in g.items() if len(v) >= 2}
    texts = [t for v in big.values() for t in v]
    lbls = [l for l, v in big.items() for _ in v]
    idx = list(range(len(texts)))
    rng = random.Random(3)
    same, diff = [], []
    for _ in range(2000):
        i, j = rng.sample(idx, 2)
        (same if lbls[i] == lbls[j] else diff).append((i, j))
    return texts, same, diff


def reps_of(feats_minilm, mdi_feats, texts, W, models):
    """Build representation dict for a dataset per --models."""
    reps = {}
    if "tfidf" in models:
        reps["tfidf"] = tfidf(texts)
    if "bigram" in models:
        reps["bigram"] = tfidf_bigram(texts)
    if "minilm" in models:
        reps["minilm"] = feats_minilm
    if "mpnet" in models:
        reps["mpnet"] = st_encode(texts, "all-mpnet-base-v2")
    if "legalbert" in models:
        reps["legalbert"] = legalbert_encode(texts)
    if W is not None:
        reps["MDI-phi"] = mdi_feats @ W
    return reps


def run_a(name, rows, log, W, models, perm=100, mdi_base="all-MiniLM-L6-v2"):
    pairs, allt = build_a(rows)
    n = len(pairs)
    emit(log, f"  [{name}] Type A n={n}")
    feats = st_encode(allt, "all-MiniLM-L6-v2")
    mdi_feats = feats if mdi_base == "all-MiniLM-L6-v2" else st_encode(allt, mdi_base)
    reps = reps_of(feats, mdi_feats, allt, W, models)
    for rname, F in reps.items():
        d = {}
        for i, (p, h, l) in enumerate(pairs):
            a, b = F[i], F[n + i]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                continue
            d.setdefault(l, []).append(1.0 - float(a @ b / (na * nb)))
        pos = np.array(d.get("E", [])); neg = np.array(d.get("C", []))
        if not len(pos) or not len(neg):
            emit(log, f"    [{rname:8s}] insufficient")
            continue
        auc, dc = auc_effect(pos, neg)
        nm, ns, pct = null_perm(pos, neg, auc, perm)
        emit(log, f"    [{rname:8s}] AUC={auc:.3f} d={dc:.2f} "
                  f"null={nm:.3f}±{ns:.3f} pctile={pct:.3f}")


def run_b(name, rows, log, W, models, perm=100, mdi_base="all-MiniLM-L6-v2"):
    texts, same, diff = build_b(rows)
    emit(log, f"  [{name}] Type B n={len(texts)}")
    feats = st_encode(texts, "all-MiniLM-L6-v2")
    mdi_feats = feats if mdi_base == "all-MiniLM-L6-v2" else st_encode(texts, mdi_base)
    reps = reps_of(feats, mdi_feats, texts, W, models)
    for rname, F in reps.items():
        ds = pair_cosine_dists(F, same)
        dd = pair_cosine_dists(F, diff)
        if not len(ds) or not len(dd):
            continue
        auc, dc = auc_effect(ds, dd)
        nm, ns, pct = null_perm(ds, dd, auc, perm)
        emit(log, f"    [{rname:8s}] AUC={auc:.3f} d={dc:.2f} "
                  f"null={nm:.3f}±{ns:.3f} pctile={pct:.3f}")


def emit(log, line):
    print(line)
    log.write(line + "\n")
    log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--W", default="mdi_W.npy")
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--models", default="tfidf,bigram,minilm,mpnet,legalbert")
    ap.add_argument("--mdi-base", default="all-MiniLM-L6-v2",
                    help="base features for MDI-phi (all-MiniLM-L6-v2 | all-mpnet-base-v2)")
    args = ap.parse_args()

    t0 = time.time()
    log = open("eval_unified_log.txt", "w", encoding="utf-8")
    W = np.load(args.W) if os.path.exists(args.W) else None
    print("=" * 72)
    print("MDI unified phi evaluation (8 datasets, vs minilm baseline)")
    print(f"W={'loaded '+str(W.shape) if W is not None else 'NONE (identity)'}")
    print("=" * 72)
    log.write("MDI unified phi eval\n")

    emit(log, "Type A (isometry)")
    for name, fn in [("ContractNLI", load_contractnli), ("SARA", load_sara),
                     ("WillsNLI", load_willsnli)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        run_a(name, rows, log, W, args.models.split(","), args.perm, args.mdi_base)

    emit(log, "Type B (structure)")
    for name, fn in [("CUAD", load_cuad), ("MAUD", load_maud),
                     ("ECHR", load_echr), ("SCOTUS", load_scotus),
                     ("LEDGAR", load_ledgar)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        run_b(name, rows, log, W, args.models.split(","), args.perm, args.mdi_base)

    emit(log, f"total time {time.time() - t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
