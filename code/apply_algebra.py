"""
apply_algebra.py - ADVANCED APPLICATIONS of the algebraized MDI-phi space.

The algebraization conclusion (verify_algebra.py): the doctrinal relation in
phi-space is linear/affine (norm->application is a linear map RMSE 0.0317;
difference vectors aligned cos=0.876; E/C linear-separable). Here we turn those
structural facts into OPERATIONS (not just distance reading):

  APPL-1  cross-space mapping -> recommendation/retrieval
          train linear map f: p -> h on entailment pairs; for a held-out
          norm p, predict h_hat = f(p); retrieve nearest real applications.
          Hit-rate vs random at top-k. (uses the linear-map fact)
  APPL-2  doctrinal vector algebra -> 3-way relation classifier
          the difference vector d = h - p encodes the doctrinal order E<N<C;
          a linear classifier on d predicts E/N/C. If linear works, the
          relation is *operable* by vector algebra, not just measurable.
          (uses the aligned-difference-vectors fact)
  APPL-3  translation invariant -> entailment completion
          if d_E is a stable "entailment translation", then given (p, h) with
          h - p ~ d_E, the pair is entailment; test consistency.

Usage:
  python code/apply_algebra.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
Output: stdout + apply_algebra.txt
"""
import argparse
import time

import numpy as np

from verify_cross_domain import load_contractnli, load_sara, load_willsnli
from verify_rigor import st_encode


def load_type_a(dirpath, max_n):
    rows = []
    for fn in (load_contractnli, load_willsnli, load_sara):
        try:
            rows += fn(dirpath, max_n)
        except FileNotFoundError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--W", default="mdi_W_v2b_mpnet.npy")
    args = ap.parse_args()

    t0 = time.time()
    log = open("apply_algebra.txt", "w", encoding="utf-8")
    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    W = np.load(args.W)
    emit(f"W={args.W} shape={W.shape}")
    pairs = load_type_a(args.data_dir, args.max)
    n = len(pairs)
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    cnt = lambda L: sum(1 for *_ , l in pairs if l == L)
    emit(f"type-A pairs n={n} (E={cnt('E')} N={cnt('N')} C={cnt('C')})")
    vecs = st_encode(allt, "all-mpnet-base-v2")
    phi = vecs @ W
    P = phi[:n]; H = phi[n:]
    E = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "E"])
    N = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "N"])
    C = np.array([i for i, (p, h, l) in enumerate(pairs) if l == "C"])

    # ---- APPL-1 cross-space map -> recommendation/retrieval ----
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    emit("## APPL-1  norm->application map: recommendation/retrieval (top-k)")
    X = P[E]; Y = H[E]
    rng = np.random.default_rng(0)
    tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0)
    f = Ridge(alpha=1.0).fit(X[tr], Y[tr])
    Hhat = f.predict(X[te])
    # gold: the true application H[te]; candidates: all applications
    Hcand = H  # all application vectors as candidate pool
    Hnorm = Hcand / (np.linalg.norm(Hcand, axis=1, keepdims=True) + 1e-12)
    hits = {1: 0, 5: 0, 10: 0}
    tot = len(te)
    for i, ti in enumerate(te):
        q = Hhat[i] / (np.linalg.norm(Hhat[i]) + 1e-12)
        sim = Hnorm @ q
        rank = np.argsort(-sim)
        gold_i = E[ti]  # index of the true application text in phi
        for k in hits:
            if gold_i in rank[:k]:
                hits[k] += 1
    for k in hits:
        emit(f"  [map->retrieve] top-{k} hit = {hits[k]/tot:.3f} (random {k}/{len(Hcand):.3f})")

    # ---- APPL-2 doctrinal vector algebra -> 3-way relation classifier ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    emit("## APPL-2  difference-vector algebra: E/N/C linear classifier")
    idx = np.concatenate([E, N, C])
    lab = np.concatenate([np.zeros(len(E)), np.ones(len(N)), np.full(len(C), 2)])
    D = H[idx] - P[idx]   # difference vector (relation vector)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for a, b in skf.split(D, lab):
        sc = StandardScaler().fit(D[a])
        clf = LogisticRegression(max_iter=3000)
        clf.fit(sc.transform(D[a]), lab[a])
        accs.append(accuracy_score(lab[b], clf.predict(sc.transform(D[b]))))
    acc = np.mean(accs)
    labi = lab.astype(int)
    base = max(np.bincount(labi).max() / len(labi), 1 / 3)
    emit(f"  [d-vector 3-way] acc={acc:.3f} (majority={base:.3f})")
    # binary E vs C for reference
    idx2 = np.concatenate([E, C]); lab2 = np.concatenate([np.zeros(len(E)), np.ones(len(C))])
    D2 = H[idx2] - P[idx2]
    accs2 = []
    for a, b in skf.split(D2, lab2):
        sc = StandardScaler().fit(D2[a])
        clf = LogisticRegression(max_iter=3000)
        clf.fit(sc.transform(D2[a]), lab2[a])
        accs2.append(accuracy_score(lab2[b], clf.predict(sc.transform(D2[b]))))
    emit(f"  [d-vector E-vs-C] acc={np.mean(accs2):.3f}")

    # ---- APPL-3 translation-consistency of entailment ----
    emit("## APPL-3  entailment-translation consistency")
    dE = H[E] - P[E]
    mu = dE.mean(0)
    # does subtracting the mean translation from an entailment h bring it near p?
    n_s = min(300, len(E))
    sE = rng.choice(len(E), n_s, replace=False)
    drift = np.linalg.norm((H[E[sE]] - mu) - P[E[sE]], axis=1).mean()
    # contrast: subtract mean translation from a random non-E h
    sX = rng.choice(np.concatenate([N, C]), n_s, replace=False)
    driftX = np.linalg.norm((H[sX] - mu) - P[sX], axis=1).mean()
    emit(f"  [translation] entail drift={drift:.4f} vs non-entail drift={driftX:.4f}")
    emit(f"  (smaller entail drift => d_E is a real 'entailment translation')")

    emit(f"total time {time.time()-t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
