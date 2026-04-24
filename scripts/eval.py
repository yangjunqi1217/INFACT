import argparse
import json
from pathlib import Path

from tqdm import tqdm

from infact.attack import PROXY_NAMES
from infact.dataset import load_split
from infact.models import MODELS, answer


ADV_CONDS = tuple(f"adv_{p}" for p in PROXY_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--split", choices=["faithfulness", "factuality"])
    ap.add_argument("--conditions", default="base,gau,mb,cmp,cap,sub,adv,shu,rev")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--perturbed-root", type=Path, default=Path("data/perturbed"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for m in MODELS:
            print(m)
        return

    if not (args.model and args.split and args.out):
        raise SystemExit("--model, --split, --out are required")
    if args.model not in MODELS:
        raise SystemExit(f"unknown model: {args.model}")

    items = load_split(args.data_root, args.split)
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    raw = [c.strip() for c in args.conditions.split(",") if c.strip()]
    conditions = []
    for c in raw:
        conditions.extend(ADV_CONDS if c == "adv" else [c])

    for cond in conditions:
        path = args.out / f"{cond}.jsonl"
        subset = [i for i in items if i.order_sensitive] if cond in ("shu", "rev") else items
        n_written = 0
        with path.open("w") as f:
            for it in tqdm(subset, desc=cond):
                vp = _video_path(it, cond, args.data_root, args.perturbed_root, args.split)
                if not vp.exists():
                    continue
                try:
                    out = answer(args.model, vp, it.question, it.options,
                                 n_frames=args.n_frames, device=args.device)
                except Exception as e:
                    out = {"response": "", "pred_idx": None, "error": str(e)}
                rec = {"id": it.id, "answer_idx": it.answer_idx, **out}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
        if n_written == 0 and subset:
            raise SystemExit(
                f"0 videos found for condition '{cond}' — check --data-root / --perturbed-root "
                "or configs/data_sources.yaml"
            )


def _video_path(item, cond, data_root, pert_root, split):
    if cond == "base":
        return data_root / item.video
    return pert_root / split / cond / f"{item.id}.mp4"


if __name__ == "__main__":
    main()
