"""
eval_downstream.py - validate the MDI geometric representation (phi) with SIMPLE
downstream models on real legal tasks.

Theoretical claim -> geometric info (MDI-phi) -> simple models verify task value:
  Classification : SCOTUS / LEDGAR / CUAD / MAUD  (linear classifier / KNN)
  NLI/entailment : ContractNLI / WillsNLI          (3-way on pair representation)
  Retrieval      : premise -> hypothesis search    (cosine Top-k hit)

Compares representations: tfidf, minilm, mpnet, MDI-phi(minilm), MDI-phi(mpnet),
legalbert (optional). Simple models only (sklearn LogisticRegression / KNN).

Usage:
  python code/eval_downstream.py --data-dir <dir> [--max 600]
      [--W mdi_W.npy --W-mpnet mdi_W_mpnet.npy]
      [--models tfidf,minilm,mpnet,phi-minilm,phi-mpnet] [--cv 3]
Output: downstream_log.txt
"""
import argparse
import collections
import os
import random
import time

import numpy as np
from tqdm import tqdm

from verify_cross_domain import (load_contractnli, load_sara, load_willsnli,
                                 load_cuad, load_maud, load_echr, tfidf,
                                 legalbert_encode)
from verify_rigor import st_encode, auc_effect
from eval_unified import load_scotus, load_ledgar, build_b, build_a


def linear_acc(X, y, cv=3, seed=0):
    """Simple linear classifier (LR), stratified k-fold accuracy.

    cv auto-adapts down when the rarest class has too few members for a
    stratified split (silences sklearn's UserWarning).
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    from collections import Counter
    min_cls = min(Counter(y).values())
    k = cv
    while k > 1 and min_cls < k:
        k -= 1
    if k < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def knn_acc(X, y, cv=3, seed=0, k=5):
    import warnings
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if len(set(y)) < 2 or len(y) < 6:
        return float("nan")
    from collections import Counter
    min_cls = min(Counter(y).values())
    kk = cv
    while kk > 1 and min_cls < kk:
        kk -= 1
    if kk < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = KNeighborsClassifier(n_neighbors=min(k, len(tr)))
            clf.fit(sc.transform(X[tr]), y[tr])
            accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def ret_topk(prem_feats, hyp_feats, k=5):
    """Retrieval: for each premise, rank hypotheses by cosine; Top-k hit rate."""
    P = prem_feats / (np.linalg.norm(prem_feats, axis=1, keepdims=True) + 1e-12)
    H = hyp_feats / (np.linalg.norm(hyp_feats, axis=1, keepdims=True) + 1e-12)
    S = P @ H.T
    n = len(P)
    hits = 0
    for i in range(n):
        rank = np.argsort(-S[i])
        if i in rank[:k]:
            hits += 1
    return hits / n


BASE_KEYS = ("minilm", "mpnet", "legalbert")


def _needs_minilm(models):
    return any(m in ("minilm", "phi-minilm", "minilm+phi") for m in models)


def _needs_mpnet(models):
    return any(m in ("mpnet", "phi-mpnet", "mpnet+phi", "legalbert+phi") for m in models)


def _needs_legalbert(models):
    return any(m in ("legalbert", "legalbert+phi") for m in models)


def build_reps(texts, Wm, Wm_mp, models):
    """Single reps + feature-augmentation combos (strong rep + MDI-phi).

    Guarantees base representations exist whenever a combo needs them, so
    requesting just 'mpnet+phi' still builds mpnet + phi-mpnet internally.
    """
    reps = {}
    base = {}
    if "tfidf" in models:
        reps["tfidf"] = tfidf(texts)
    if _needs_minilm(models):
        base["minilm"] = st_encode(texts, "all-MiniLM-L6-v2")
    if _needs_mpnet(models):
        base["mpnet"] = st_encode(texts, "all-mpnet-base-v2")
    if _needs_legalbert(models):
        base["legalbert"] = legalbert_encode(texts)
    if any(m in ("phi-minilm", "minilm+phi") for m in models) and Wm is not None:
        base["phi-minilm"] = base["minilm"] @ Wm
    if any(m in ("phi-mpnet", "mpnet+phi", "legalbert+phi") for m in models) and Wm_mp is not None:
        base["phi-mpnet"] = base["mpnet"] @ Wm_mp
    for k in BASE_KEYS + ("phi-minilm", "phi-mpnet"):
        if k in models and k in base:
            reps[k] = base[k]
    # feature augmentation: strong rep + MDI-phi
    if "mpnet+phi" in models and "mpnet" in base and "phi-mpnet" in base:
        reps["mpnet+phi"] = np.concatenate([base["mpnet"], base["phi-mpnet"]], axis=1)
    if "minilm+phi" in models and "minilm" in base and "phi-minilm" in base:
        reps["minilm+phi"] = np.concatenate([base["minilm"], base["phi-minilm"]], axis=1)
    if "legalbert+phi" in models and "legalbert" in base and "phi-mpnet" in base:
        reps["legalbert+phi"] = np.concatenate([base["legalbert"], base["phi-mpnet"]], axis=1)
    return reps


# representations used for pair tasks (NLI / retrieval): the lightweight
# semantic family + MDI-phi + feature augmentation (excludes slow legalbert)
PAIR_MODELS = ("minilm", "mpnet", "phi-minilm", "phi-mpnet",
               "minilm+phi", "mpnet+phi")


def emit(log, line):
    print(line)
    log.write(line + "\n")
    log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default="mdi_W.npy")
    ap.add_argument("--W-mpnet", default="mdi_W_mpnet.npy")
    ap.add_argument("--models",
                    default="tfidf,minilm,mpnet,legalbert,phi-minilm,phi-mpnet,"
                            "minilm+phi,mpnet+phi,legalbert+phi")
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    log = open("downstream_log.txt", "w", encoding="utf-8")
    models = [x for x in args.models.split(",")]
    Wm = np.load(args.W) if os.path.exists(args.W) else None
    Wm_mp = np.load(args.W_mpnet) if os.path.exists(args.W_mpnet) else None
    print("=" * 74)
    print("MDI downstream validation: geometric rep (phi) + SIMPLE models on tasks")
    print(f"W-minilm={'yes' if Wm is not None else 'no'} W-mpnet={'yes' if Wm_mp is not None else 'no'}")
    print("=" * 74)
    log.write("MDI downstream validation\n")

    # ---- Classification tasks ----
    emit(log, "## Classification (linear acc)  [LR mean acc over CV]")
    for name, fn in [("SCOTUS", load_scotus), ("LEDGAR", load_ledgar),
                     ("CUAD", load_cuad), ("MAUD", load_maud)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        n_lab = len(set(l for _, l in rows))
        per = min(50, max(1, int(1000 / max(1, n_lab))))
        g = collections.defaultdict(list)
        for t, l in rows:
            g[l].append(t)
        big = {l: v[:per] for l, v in g.items() if len(v) >= 2}
        texts = [t for v in big.values() for t in v]
        lbls = [l for l, v in big.items() for _ in v]
        reps = build_reps(texts, Wm, Wm_mp, models)
        emit(log, f"  [{name}] n={len(texts)} classes={len(set(lbls))}")
        for rn, F in reps.items():
            acc = linear_acc(F, lbls, cv=args.cv)
            emit(log, f"    [{rn:11s}] acc={acc:.3f}")

    # ---- NLI / entailment tasks ----
    emit(log, "## NLI (3-way on pair rep)  [LR acc]")
    for name, fn in [("ContractNLI", load_contractnli), ("WillsNLI", load_willsnli)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            emit(log, f"  [{name}] missing: {e}")
            continue
        pairs, allt = build_a(rows)
        n = len(pairs)
        lbls = np.array([{"E": 0, "N": 1, "C": 2}[l] for _, _, l in pairs])
        for rn in [m for m in models if m in PAIR_MODELS]:
            text_reps = build_reps(allt, Wm, Wm_mp, [rn])
            F = text_reps[rn]
            pairX = np.concatenate([F[:n], F[n:]], axis=1)  # [g(p), g(h)]
            acc = linear_acc(pairX, lbls, cv=args.cv)
            emit(log, f"  [{name}] [{rn:11s}] NLI-acc={acc:.3f}")

    # ---- Retrieval (premise -> hypothesis) ----
    emit(log, "## Retrieval (premise -> hypothesis Top-k hit)")
    for name, fn in [("ContractNLI", load_contractnli), ("WillsNLI", load_willsnli)]:
        try:
            rows = fn(args.data_dir, 300)
        except FileNotFoundError:
            continue
        pairs, allt = build_a(rows)
        n = len(pairs)
        for rn in [m for m in models if m in PAIR_MODELS]:
            text_reps = build_reps(allt, Wm, Wm_mp, [rn])
            F = text_reps[rn]
            hk = ret_topk(F[:n], F[n:], k=5)
            emit(log, f"  [{name}] [{rn:11s}] Top5-hit={hk:.3f}")

    emit(log, f"total time {time.time() - t0:.1f}s")
    log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import sys
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:
            with open("downstream_log.txt", "a", encoding="utf-8") as f:
                f.write("\n[TRACEBACK]\n" + tb + "\n")
        except Exception:
            pass
        sys.exit(1)
