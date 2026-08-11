"""
verify_cross_domain.py - MDI cross-domain validation (self-contained)

Validates the MDI invariant — entailment pairs are significantly closer than
contradiction pairs in the representation space — on public legal NLI datasets,
using generic representations only. No task-specific scorer is involved.

Datasets (expected under --data-dir):
  ContractNLI  <dir>/ContractNLI/contract_nli_v1.jsonl     (premise/hypothesis/label)
  SARA         <dir>/SARA/sara_entailment/test.tsv         (statute/question/answer)
  WillsNLI     <dir>/WillsNLI/processed/willsnli_processed.json  (R_texts/E_texts/classification)

Usage:
  python code/verify_cross_domain.py [--data-dir ./data] [--max 600]
"""
import argparse
import collections
import json
import math
import os
import re
import time

import numpy as np

# ---------------------------------------------------------------------------
# Loaders -> [(premise, hypothesis, label)]  label in E/N/C
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
    for d in docs:
        tf = collections.Counter(d)
        v = {w: c * math.log((N + 1) / (df[w] + 1)) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def cos(a, b):
    return sum(a.get(w, 0) * b.get(w, 0) for w in set(a) | set(b))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run(name, rows, log):
    per = min(400, len(rows))
    grouped = collections.defaultdict(list)
    for p, h, l in rows:
        grouped[l].append((p, h))
    sel = {l: v[:per] for l, v in grouped.items()}
    pairs = [(p, h, l) for l, v in sel.items() for p, h in v]
    allt = [p for p, _, _ in pairs] + [h for _, h, _ in pairs]
    n = len(pairs)

    def report(tag, vecs):
        d_by = collections.defaultdict(list)
        for i, (p, h, l) in enumerate(pairs):
            a, b = vecs[i], vecs[n + i]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                continue
            d_by[l].append(1.0 - float(a @ b / (na * nb)))
        from scipy import stats
        pv = None
        if d_by["E"] and d_by["C"]:
            pv = float(stats.mannwhitneyu(d_by["E"], d_by["C"],
                                          alternative="less").pvalue)
        m = {l: sum(d_by[l]) / len(d_by[l]) for l in d_by if d_by[l]}
        ok = m.get("E", 1e9) < m.get("C", 1e9) and (pv is not None and pv < 0.05)
        line = (f"    [{tag}] E={m.get('E', float('nan')):.4f} "
                f"N={m.get('N', float('nan')):.4f} C={m.get('C', float('nan')):.4f} "
                f"| E<C: {ok}")
        if pv is not None:
            line += f" | p={pv:.2e}"
        print(line)
        log.write(line + "\n")

    print(f"  [{name}] n={n} pairs per class: " + ", ".join(
        f"{l}={len([x for x in pairs if x[2] == l])}" for l in ["E", "N", "C"]))
    log.write(f"  [{name}] n={n}\n")

    # lexical TF-IDF (dense for embedding code path)
    vecs = tfidf(allt)
    lex = []
    keys = sorted(set(k for v in vecs for k in v))
    ki = {k: i for i, k in enumerate(keys)}
    for v in vecs:
        x = np.zeros(len(keys))
        for k, val in v.items():
            x[ki[k]] = val
        lex.append(x)
    report("lexical", lex)

    # semantic embedding
    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        sem = enc.encode(allt, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True)
        report("semantic", sem)
    except Exception as e:  # noqa: BLE001
        print(f"    [semantic] unavailable: {e}")
        log.write(f"    [semantic] unavailable: {e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=600)
    args = ap.parse_args()

    t0 = time.time()
    log = open("validation_log.txt", "w", encoding="utf-8")

    print("=" * 64)
    print("MDI cross-domain validation (independent of any task-specific scorer)")
    print("=" * 64)
    log.write("MDI cross-domain validation\n")

    for name, fn in [("ContractNLI (contract)", load_contractnli),
                     ("SARA (tax)", load_sara),
                     ("WillsNLI (wills)", load_willsnli)]:
        try:
            rows = fn(args.data_dir, args.max)
        except FileNotFoundError as e:
            print(f"  [{name}] data not found: {e} (use --data-dir)")
            continue
        run(name, rows, log)

    print(f"total time {time.time() - t0:.1f}s")
    log.close()


if __name__ == "__main__":
    main()
