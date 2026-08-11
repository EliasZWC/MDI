"""
verify_cross_domain.py - MDI cross-domain validation (self-contained)

Validates MDI across legal sub-domains using ONLY public datasets and generic
representations (no task-specific scorer):

  Type A (isometry / P2): entailment pairs are significantly closer than
      contradiction pairs   [ContractNLI, SARA, WillsNLI]
  Type B (structure / E2): same-label pairs are significantly closer than
      different-label pairs [CUAD, MAUD, ECHR]

Representations (baselines): lexical TF-IDF, universal semantic embeddings
(all-MiniLM-L6-v2, optional all-mpnet-base-v2).

Datasets (expected under --data-dir):
  Type A:
    ContractNLI  <dir>/ContractNLI/contract_nli_v1.jsonl
    SARA         <dir>/SARA/sara_entailment/test.tsv
    WillsNLI     <dir>/WillsNLI/processed/willsnli_processed.json
  Type B:
    CUAD         <dir>/CUAD/test.json          (41 clause categories)
    MAUD         <dir>/MAUD/MAUD_train.csv     (category column)
    ECHR         <dir>/ECHR/dev.jsonl          (violated True/False)

Usage:
  python code/verify_cross_domain.py [--data-dir ./data] [--max 600]
"""
import argparse
import collections
import json
import math
import os
import random
import re
import time

import numpy as np

# ---------------------------------------------------------------------------
# Loaders — Type A: [(premise, hypothesis, label)]  label in E/N/C
# ---------------------------------------------------------------------------


def load_contractnli(dirpath, max_n):
    rows = []
    for line in open(os.path.join(dirpath, "ContractNLI", "contract_nli_v1.jsonl"),
                     encoding="utf-8"):
        o = json.loads(line)
        rows.append((o["premise"], o["hypothesis"], o["label"][0].upper()))
    return rows[:max_n]


def load_sara(dirpath, max_n):
    import pandas as pd
    df = pd.read_csv(os.path.join(dirpath, "SARA", "sara_entailment", "test.tsv"),
                     sep="\t")
    rows = []
    for _, r in df.iterrows():
        ans = str(r["answer"]).strip()
        lab = "E" if ans.startswith("E") else ("C" if ans.startswith("C") else "N")
        rows.append((str(r["statute"]), str(r["question"]), lab))
    return rows[:max_n]


def load_willsnli(dirpath, max_n):
    d = json.load(open(os.path.join(dirpath, "WillsNLI", "processed",
                                    "willsnli_processed.json"), encoding="utf-8"))
    rows = []
    for i in range(len(d["E_texts"])):
        law = d["R_texts"][d["E_law_indices"][i]]
        c = str(d["E_classifications"][i])
        rows.append((law, d["E_texts"][i], {"1": "E", "0": "C", "2": "N"}[c]))
    return rows[:max_n]


# ---------------------------------------------------------------------------
# Loaders — Type B: [(text, label)]
# ---------------------------------------------------------------------------


def load_cuad(dirpath, max_n):
    obj = json.load(open(os.path.join(dirpath, "CUAD", "test.json"),
                         encoding="utf-8"))
    rows = []
    for doc in obj["data"]:
        for para in doc.get("paragraphs", []):
            for qa in para.get("qas", []):
                m = re.search(r'related to "([^"]+)"', qa.get("question", ""))
                if not m or qa.get("is_impossible"):
                    continue
                for a in qa.get("answers", [])[:1]:
                    t = (a.get("text") or "").strip()
                    if t:
                        rows.append((t, m.group(1)))
    rng = random.Random(7)
    rng.shuffle(rows)
    return rows[:max_n]


def load_maud(dirpath, max_n):
    import pandas as pd
    df = pd.read_csv(os.path.join(dirpath, "MAUD", "MAUD_train.csv"),
                     usecols=["category", "text"]).dropna(subset=["text"])
    rows = [(str(t).strip(), str(c)) for t, c in zip(df["text"], df["category"])
            if str(t).strip()]
    rng = random.Random(7)
    rng.shuffle(rows)
    return rows[:max_n]


def load_echr(dirpath, max_n):
    import ast
    rows = []
    with open(os.path.join(dirpath, "ECHR", "dev.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            txt = o.get("text") or ""
            if isinstance(txt, list):
                txt = " ".join(str(x) for x in txt)
            elif isinstance(txt, str) and txt.startswith("["):
                try:
                    txt = " ".join(ast.literal_eval(txt))
                except Exception:  # noqa: BLE001
                    pass
            txt = (txt.strip() if isinstance(txt, str) else str(txt)).strip()
            if not txt:
                continue
            rows.append((txt, str(o.get("violated"))))
    rng = random.Random(7)
    rng.shuffle(rows)
    return rows[:max_n]


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


def tokenize(t):
    return re.findall(r"[a-z']+", t.lower())


def tfidf(texts):
    docs = [tokenize(t) for t in texts]
    df = collections.Counter()
    for d in docs:
        df.update(set(d))
    N = max(len(docs), 1)
    vecs = []
    keys = set()
    for d in docs:
        tf = collections.Counter(d)
        v = {w: c * math.log((N + 1) / (df[w] + 1)) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        v = {w: x / norm for w, x in v.items()}
        vecs.append(v)
        keys.update(v)
    ki = {k: i for i, k in enumerate(sorted(keys))}
    dense = np.zeros((len(vecs), len(ki)))
    for i, v in enumerate(vecs):
        for k, val in v.items():
            dense[i, ki[k]] = val
    return dense


_ENC = {}


def semantic_encode(texts, model_name="all-MiniLM-L6-v2"):
    if model_name not in _ENC:
        from sentence_transformers import SentenceTransformer
        _ENC[model_name] = SentenceTransformer(model_name)
    return _ENC[model_name].encode(texts, batch_size=64,
                                   show_progress_bar=False,
                                   convert_to_numpy=True)


def pair_cosine_dists(vecs, pairs):
    out = []
    for i, j in pairs:
        a, b = vecs[i], vecs[j]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            continue
        out.append(1.0 - float(a @ b / (na * nb)))
    return np.array(out)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def report_a(name, rows, log, reps):
    """Type A: isometry — E pairs closer than C pairs."""
    per = min(400, len(rows))
    grouped = collections.defaultdict(list)
    for p, h, l in rows:
        grouped[l].append((p, h))
    sel = {l: v[:per] for l, v in grouped.items()}
    pairs = [(p, h, l) for l, v in sel.items() for p, h in v]
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    n = len(pairs)
    print(f"  [{name}] Type A isometry  n={n}  E={len(sel.get('E', []))} "
          f"C={len(sel.get('C', []))} N={len(sel.get('N', []))}")
    log.write(f"  [{name}] Type A n={n}\n")
    from scipy import stats
    for rname, get_vecs in reps.items():
        vecs = get_vecs(allt)
        d = collections.defaultdict(list)
        for i, (p, h, l) in enumerate(pairs):
            d[l].append(1.0 - float(vecs[i] @ vecs[n + i] /
                                    (np.linalg.norm(vecs[i]) *
                                     np.linalg.norm(vecs[n + i]) + 1e-12)))
        pv = None
        if d["E"] and d["C"]:
            pv = float(stats.mannwhitneyu(d["E"], d["C"],
                                          alternative="less").pvalue)
        m = {l: float(np.mean(d[l])) for l in d if d[l]}
        ok = m.get("E", 1e9) < m.get("C", 1e9) and (pv is not None and pv < 0.05)
        line = (f"    [{rname:8s}] E={m.get('E', float('nan')):.4f} "
                f"N={m.get('N', float('nan')):.4f} C={m.get('C', float('nan')):.4f} "
                f"| E<C: {ok}")
        if pv is not None:
            line += f" | p={pv:.2e}"
        print(line)
        log.write(line + "\n")


def report_b(name, rows, log, reps):
    """Type B: structure — same-label pairs closer than different-label pairs."""
    n_labels = len(set(l for _, l in rows))
    per = min(50, max(1, int(1000 / max(1, n_labels))))
    grouped = collections.defaultdict(list)
    for t, l in rows:
        grouped[l].append(t)
    big = {l: v[:per] for l, v in grouped.items() if len(v) >= 2}
    if len(big) < 2:
        print(f"  [{name}] Type B skipped (labels={n_labels})")
        return
    texts = [t for v in big.values() for t in v]
    lbls = [l for l, v in big.items() for _ in v]
    idx = list(range(len(texts)))
    rng = random.Random(3)
    same, diff = [], []
    for _ in range(2000):
        i, j = rng.sample(idx, 2)
        (same if lbls[i] == lbls[j] else diff).append((i, j))
    print(f"  [{name}] Type B structure  n={len(texts)} classes={len(big)} "
          f"same={len(same)} diff={len(diff)}")
    log.write(f"  [{name}] Type B n={len(texts)} classes={len(big)}\n")
    from scipy import stats
    for rname, get_vecs in reps.items():
        vecs = get_vecs(texts)
        ds = pair_cosine_dists(vecs, same)
        dd = pair_cosine_dists(vecs, diff)
        if not len(ds) or not len(dd):
            continue
        pv = float(stats.mannwhitneyu(ds, dd, alternative="less").pvalue)
        ok = float(ds.mean()) < float(dd.mean()) and pv < 0.05
        line = (f"    [{rname:8s}] same={ds.mean():.4f} diff={dd.mean():.4f} "
                f"| same<diff: {ok} | p={pv:.2e}")
        print(line)
        log.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--models", default="tfidf,minilm",
                    help="comma list: tfidf,minilm,mpnet")
    args = ap.parse_args()

    t0 = time.time()
    log = open("validation_log.txt", "w", encoding="utf-8")
    print("=" * 68)
    print("MDI cross-domain validation (6 public datasets, generic representations)")
    print("=" * 68)
    log.write("MDI cross-domain validation\n")

    reps = {}
    if "tfidf" in args.models:
        reps["tfidf"] = tfidf
    if "minilm" in args.models:
        reps["minilm"] = lambda t: semantic_encode(t, "all-MiniLM-L6-v2")
    if "mpnet" in args.models:
        reps["mpnet"] = lambda t: semantic_encode(t, "all-mpnet-base-v2")

    print("Type A (isometry: entailment closer than contradiction)")
    log.write("Type A\n")
    for name, fn in [("ContractNLI (contract)", load_contractnli),
                     ("SARA (tax)", load_sara),
                     ("WillsNLI (wills)", load_willsnli)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            print(f"  [{name}] data not found: {e}")
            continue
        report_a(name, rows, log, reps)

    print("Type B (structure: same-label closer than different-label)")
    log.write("Type B\n")
    for name, fn in [("CUAD (contract)", load_cuad),
                     ("MAUD (contract)", load_maud),
                     ("ECHR (human rights)", load_echr)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            print(f"  [{name}] data not found: {e}")
            continue
        report_b(name, rows, log, reps)

    print(f"total time {time.time() - t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
