"""fetch_legal_datasets.py - download SCOTUS + CaseHOLD (LexGLUE) to JSON

Expands MDI cross-domain validation to more legal tasks (Type B structure):
  SCOTUS   : supreme-court case text -> issue label (14 classes)
  CaseHOLD : case citation context -> cited source (classification)

Output: <data-dir>/LexGLUE/{scotus,casehold}.json  [{text,label}]
"""
import argparse
import json
import os
import time


def fetch(cfg, name, outdir, max_rows=None):
    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", cfg, split="train")
    rows = []
    for ex in ds:
        text = ex.get("text") or ex.get("facts") or ""
        label = ex.get("label")
        if isinstance(label, int):
            label = str(label)
        if text and label is not None and len(text) > 20:
            rows.append({"text": text, "label": str(label)})
        if max_rows and len(rows) >= max_rows:
            break
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    labels = set(r["label"] for r in rows)
    print(f"[{name}] n={len(rows)} classes={len(labels)} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max", type=int, default=0, help="0 = all train")
    args = ap.parse_args()
    t0 = time.time()
    outdir = os.path.join(args.data_dir, "LexGLUE")
    os.makedirs(outdir, exist_ok=True)
    mx = args.max or None
    if not os.path.exists(os.path.join(outdir, "scotus.json")):
        fetch("scotus", "scotus", outdir, mx)
    else:
        print("[scotus] exists, skip")
    if not os.path.exists(os.path.join(outdir, "case_hold.json")):
        fetch("case_hold", "case_hold", outdir, mx)
    else:
        print("[case_hold] exists, skip")
    if not os.path.exists(os.path.join(outdir, "ledgar.json")):
        fetch("ledgar", "ledgar", outdir, mx)
    else:
        print("[ledgar] exists, skip")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
