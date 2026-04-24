import argparse
import json
from pathlib import Path

from infact.dataset import load_split
from infact.metrics import summarize


FIXED_OPS = ("gau", "mb", "cmp", "cap", "sub", "shu", "rev")


def _preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r.get("pred_idx")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--split", choices=["faithfulness", "factuality"], required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args()

    items = load_split(args.data_root, args.split)
    ids = [i.id for i in items]
    gold = [i.answer_idx for i in items]
    order = [i.order_sensitive for i in items]

    base_map = _preds(args.runs / "base.jsonl")
    base = [base_map.get(i) for i in ids]

    adv_files = sorted(args.runs.glob("adv_*.jsonl"))
    all_ops = list(FIXED_OPS) + [f.stem for f in adv_files]

    perturbed = {}
    for op in all_ops:
        f = args.runs / f"{op}.jsonl"
        if not f.exists():
            continue
        preds = _preds(f)
        if op in ("shu", "rev"):
            perturbed[op] = [preds.get(i, base_map.get(i)) for i in ids]
        else:
            perturbed[op] = [preds.get(i) for i in ids]

    result = summarize(base, gold, perturbed, order)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
