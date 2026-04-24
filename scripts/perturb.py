import argparse
import gc
import random
from pathlib import Path

import yaml
from tqdm import tqdm

from infact.dataset import load_split
from infact.perturb import OPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["faithfulness", "factuality"], required=True)
    ap.add_argument("--ops", nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/perturbed"))
    ap.add_argument("--config", type=Path, default=Path("configs/operators.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    items = load_split(args.data_root, args.split)
    if args.limit:
        items = items[: args.limit]

    base_rng = random.Random(args.seed)
    item_seeds = {it.id: base_rng.randrange(2**32) for it in items}

    pool = None
    if "cap" in args.ops:
        pool_path = Path("data/meta/irrelevant_captions.txt")
        if not pool_path.exists():
            pool_path = Path("data/meta/irrelevant_captions.sample.txt")
        if not pool_path.exists():
            raise FileNotFoundError("cap needs data/meta/irrelevant_captions.txt")
        pool = [s.strip() for s in pool_path.read_text().splitlines() if s.strip()]

    for op in args.ops:
        if op == "adv":
            _run_adv(args, cfg["adv"], items)
            continue
        if op not in OPS:
            raise SystemExit(f"unknown op: {op}")

        out_dir = args.out / args.split / op
        out_dir.mkdir(parents=True, exist_ok=True)
        params = dict(cfg.get(op, {}))

        for it in tqdm(items, desc=op):
            if op in ("shu", "rev") and not it.order_sensitive:
                continue
            dst = out_dir / f"{it.id}.mp4"
            if dst.exists():
                continue
            src = args.data_root / it.video
            p = {**params, "seed": item_seeds[it.id]}
            if op == "cap":
                OPS[op](src, dst, misleading=it.meta.get("misleading_caption"),
                        irrelevant_pool=pool, **p)
            elif op == "sub":
                OPS[op](src, dst, transcript=it.meta.get("transcript", []), **p)
            else:
                OPS[op](src, dst, **p)


def _run_adv(args, adv_cfg, items):
    import torch
    from infact.attack import MIFGSMConfig, PROXY_NAMES, load_proxy, mi_fgsm

    cfg = MIFGSMConfig(
        epsilon=adv_cfg["epsilon"],
        alpha=adv_cfg["alpha"],
        steps=adv_cfg["steps"],
        mu=adv_cfg.get("mu", 1.0),
    )
    for name in PROXY_NAMES:
        out_dir = args.out / args.split / f"adv_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [it for it in items if not (out_dir / f"{it.id}.mp4").exists()]
        if not todo:
            continue

        proxy = load_proxy(name)
        try:
            for it in tqdm(todo, desc=f"adv_{name}"):
                mi_fgsm(args.data_root / it.video, out_dir / f"{it.id}.mp4", proxy, cfg)
        finally:
            del proxy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
