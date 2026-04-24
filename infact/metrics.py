from statistics import mean


VIS_OPS = ("gau", "mb", "cmp")
COR_OPS = ("cap", "sub")
TEMP_OPS = ("shu", "rev")


def accuracy(preds, gold):
    if not gold:
        return 0.0
    return sum(p is not None and p == g for p, g in zip(preds, gold)) / len(gold)


def resist_rate(base, pert, gold):
    num = den = 0
    for b, p, g in zip(base, pert, gold):
        if b is not None and b == g:
            den += 1
            if p is not None and p == g:
                num += 1
    return num / den if den else 0.0


def tss(base, pert, gold, order_mask):
    num = den = 0
    for b, p, g, o in zip(base, pert, gold, order_mask):
        if not o or b is None or b != g:
            continue
        den += 1
        if not (p is not None and p == g):
            num += 1
    return num / den if den else 0.0


def summarize(base, gold, perturbed, order_mask):
    def _rr(keys):
        return {k: resist_rate(base, perturbed[k], gold) for k in keys if k in perturbed}

    rr_vis = _rr(VIS_OPS)
    rr_cor_base = _rr(COR_OPS)
    rr_adv_per = {k: resist_rate(base, v, gold)
                  for k, v in perturbed.items() if k.startswith("adv_")}
    ts = {k: tss(base, perturbed[k], gold, order_mask)
          for k in TEMP_OPS if k in perturbed}

    def _m(vals):
        vals = list(vals)
        return mean(vals) if vals else None

    rr_adv = _m(rr_adv_per.values())
    rr_deg = _m(rr_vis.values())
    rr_cor_vals = list(rr_cor_base.values()) + ([rr_adv] if rr_adv is not None else [])
    rr_cor = _m(rr_cor_vals)
    tss_bar = _m(ts.values())

    base_acc = accuracy(base, gold)
    parts = [x for x in (base_acc, rr_cor, rr_deg, tss_bar) if x is not None]
    return {
        "base": base_acc,
        "rr": {**rr_vis, **rr_cor_base, **rr_adv_per, "adv": rr_adv},
        "rr_deg": rr_deg,
        "rr_cor": rr_cor,
        "tss": ts,
        "tss_bar": tss_bar,
        "avg": sum(parts) / len(parts) if parts else None,
    }
