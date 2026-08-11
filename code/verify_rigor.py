"""
verify_rigor.py - MDI rigor checks: effect size + null control + stability

Strengthens the 5×6 matrix from significance lists to quantifiable,
chance-excluded evidence:

  Effect size : AUC (Mann-Whitney U / n1n2) + Cohen's d for the "closer" invariant
  Null control: permutation of the same pairs with shuffled labels -> null AUC
                distribution; report null mean±std and the real AUC's percentile
  Stability   : bootstrap of the same distance arrays -> AUC mean ± std

All long steps (encoding / permutation / bootstrap) show a tqdm progress bar.

Usage:
  python code/verify_rigor.py --data-dir <dir> [--max 400]
                               [--models tfidf,minilm] [--perm 100] [--boot 200]
Output: rigor_log.txt
"""
import argparse
import collections
import os
import random
import time

import numpy as np
from scipy import stats
from tqdm import tqdm

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud, load_echr, tfidf,
                                 pair_cosine_dists)


# ---------- representations (with progress bars) ----------
_ENC = {}


def st_encode(texts, model="all-MiniLM-L6-v2", bs=64):
    if model not in _ENC:
        from sentence_transformers import SentenceTransformer
        _ENC[model] = SentenceTransformer(model)
    return _ENC[model].encode(texts, batch_size=bs, show_progress_bar=True,
                              convert_to_numpy=True)


REPS = {
    "tfidf": tfidf,
    "minilm": lambda t: st_encode(t, "all-MiniLM-L6-v2"),
    "mpnet": lambda t: st_encode(t, "all-mpnet-base-v2"),
}


# ---------- rigor stats ----------
def auc_effect(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    u, _ = stats.mannwhitneyu(pos, neg, alternative="less")
    auc = float(u / (len(pos) * len(neg)))
    sp = float(np.sqrt(((len(pos) - 1) * pos.std(ddof=1) ** 2 +
                        (len(neg) - 1) * neg.std(ddof=1) ** 2) /
                       (len(pos) + len(neg) - 2)))
    d = float((neg.mean() - pos.mean()) / sp) if sp > 0 else 0.0
    return auc, d


def null_perm(pos, neg, auc_real, n_perm=100, seed=0):
    allv = np.concatenate([pos, neg])
    n1, n2 = len(pos), len(neg)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in tqdm(range(n_perm), desc="null-perm", ncols=70):
        idx = rng.permutation(len(allv))
        p2, n2v = allv[idx[:n1]], allv[idx[n1:]]
        u, _ = stats.mannwhitneyu(p2, n2v, alternative="less")
        nulls.append(float(u / (n1 * max(n2, 1))))
    na = np.array(nulls)
    return float(na.mean()), float(na.std()), float(np.mean(na >= auc_real))


def bootstrap_auc(pos, neg, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in tqdm(range(n_boot), desc="bootstrap", ncols=70):
        p2 = rng.choice(pos, size=len(pos), replace=True)
        n2 = rng.choice(neg, size=len(neg), replace=True)
        u, _ = stats.mannwhitneyu(p2, n2, alternative="less")
        aucs.append(float(u / (len(p2) * len(n2))))
    a = np.array(aucs)
    return float(a.mean()), float(a.std())


def dists_of_a(pairs, vecs):
    n = len(pairs)
    d = collections.defaultdict(list)
    for i, (p, h, l) in enumerate(pairs):
        a, b = vecs[i], vecs[n + i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            continue
        d[l].append(1.0 - float(a @ b / (na * nb)))
    return np.array(d.get("E", [])), np.array(d.get("C", []))


def build_pairs_a(rows, per=400):
    grouped = collections.defaultdict(list)
    for p, h, l in rows:
        grouped[l].append((p, h))
    sel = {l: v[:per] for l, v in grouped.items()}
    pairs = [(p, h, l) for l, v in sel.items() for p, h in v]
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    return pairs, allt


def build_pairs_b(rows):
    n_labels = len(set(l for _, l in rows))
    per = min(50, max(1, int(1000 / max(1, n_labels))))
    grouped = collections.defaultdict(list)
    for t, l in rows:
        grouped[l].append(t)
    big = {l: v[:per] for l, v in grouped.items() if len(v) >= 2}
    texts = [t for v in big.values() for t in v]
    lbls = [l for l, v in big.items() for _ in v]
    idx = list(range(len(texts)))
    rng = random.Random(3)
    same, diff = [], []
    for _ in range(2000):
        i, j = rng.sample(idx, 2)
        (same if lbls[i] == lbls[j] else diff).append((i, j))
    return texts, same, diff


def emit(log, line):
    print(line)
    log.write(line + "\n")
    log.flush()


def run_a(name, rows, log, models, perm, boot):
    pairs, allt = build_pairs_a(rows)
    emit(log, f"  [{name}] Type A n={len(pairs)}")
    for m in models:
        vecs = REPS[m](allt)
        pos, neg = dists_of_a(pairs, vecs)
        if not len(pos) or not len(neg):
            emit(log, f"    [{m:6s}] insufficient E/C")
            continue
        auc, d = auc_effect(pos, neg)
        nm, ns, pct = null_perm(pos, neg, auc, perm)
        bm, bs = bootstrap_auc(pos, neg, boot)
        emit(log, f"    [{m:6s}] AUC={auc:.3f} d={d:.2f} "
                  f"null={nm:.3f}±{ns:.3f} pctile={pct:.3f} boot={bm:.3f}±{bs:.3f}")


def run_b(name, rows, log, models, perm, boot):
    texts, same, diff = build_pairs_b(rows)
    emit(log, f"  [{name}] Type B n={len(texts)}")
    for m in models:
        vecs = REPS[m](texts)
        ds = pair_cosine_dists(vecs, same)
        dd = pair_cosine_dists(vecs, diff)
        if not len(ds) or not len(dd):
            continue
        auc, d = auc_effect(ds, dd)
        nm, ns, pct = null_perm(ds, dd, auc, perm)
        bm, bs = bootstrap_auc(ds, dd, boot)
        emit(log, f"    [{m:6s}] AUC={auc:.3f} d={d:.2f} "
                  f"null={nm:.3f}±{ns:.3f} pctile={pct:.3f} boot={bm:.3f}±{bs:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--models", default="tfidf,minilm")
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    log = open("rigor_log.txt", "w", encoding="utf-8")
    models = [x for x in args.models.split(",") if x in REPS]
    print("=" * 72)
    print(f"MDI rigor: AUC/d + permutation-null + bootstrap  (perm={args.perm}, boot={args.boot})")
    print("=" * 72)
    log.write("MDI rigor\n")

    emit(log, "Type A (isometry)")
    for name, fn in [("ContractNLI", load_contractnli), ("SARA", load_sara),
                     ("WillsNLI", load_willsnli)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        run_a(name, rows, log, models, args.perm, args.boot)

    emit(log, "Type B (structure)")
    for name, fn in [("CUAD", load_cuad), ("MAUD", load_maud),
                     ("ECHR", load_echr)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        run_b(name, rows, log, models, args.perm, args.boot)

    emit(log, f"total time {time.time() - t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
