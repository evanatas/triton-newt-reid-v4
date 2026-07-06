"""Часть 4 — A/B вариантов кропа на DEV TK temporal (test ЗАПЕЧАТАН).
yellow (baseline) vs birefnet vs +label vs +label+belly. SIFT-recall + McNemar + Wilson.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
for _p in ("spike_lab", "registry", "segment"):
    sys.path.insert(0, str(HERE.parent / _p))
sys.path.insert(0, str(HERE))
from common import ROOT                       # noqa: E402
from sift_eval import score_matrix, recall_from_scores   # noqa: E402
from embed_eval import wilson, KS             # noqa: E402
from sealed import assert_unsealed            # noqa: E402
from birefnet import body_mask, _yellow_mask  # noqa: E402
from label_mask import detect_label, apply_label_mask   # noqa: E402
from crop import tight_crop, belly_only       # noqa: E402
from preprocess import to_sift_gray           # noqa: E402

METHODS = ["yellow", "birefnet", "birefnet_label", "birefnet_label_belly"]


def _sift_feats(grays):
    sift = cv2.SIFT_create(nfeatures=1500)
    feats = []
    for g in grays:
        kp, des = sift.detectAndCompute(g, None)
        feats.append((np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des))
    return feats


def _crop_from_cache(img, mask, method):
    img = img.copy(); m = mask.copy()
    has = False
    if method != "yellow":
        if "label" in method:
            lbl, has = detect_label(img, m)
            img, m = apply_label_mask(img, m, lbl)
        if "belly" in method:
            m = belly_only(m)
    return to_sift_gray(tight_crop(img, m)), has


def mcnemar(a_hits, b_hits):
    """b=challenger vs a=baseline. -> (b_only, a_only, p двусторонний)."""
    b = int((~a_hits & b_hits).sum())   # challenger выиграл
    c = int((a_hits & ~b_hits).sum())   # baseline выиграл
    n = b + c
    if n == 0:
        return b, c, 1.0
    # точный биномиальный двусторонний
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, p)


def main():
    assert_unsealed("dev", unseal=False)
    spl = pd.read_csv(HERE.parent / "registry" / "splits.csv")
    reg = pd.read_csv(HERE.parent / "registry" / "registry.csv")
    df = spl.merge(reg[["frame_id", "path_rel"]], on="frame_id")
    tk = df[df.cohort == "TK"]
    gallery = tk[((tk.split_fold == "dev") & (tk.split_role == "gallery") & (~tk.is_open_new))
                 | (tk.split_fold == "distractor")].reset_index(drop=True)
    probe = tk[(tk.split_fold == "dev") & (tk.split_role == "probe") & (~tk.is_open_new)].reset_index(drop=True)
    pool = pd.concat([gallery, probe], ignore_index=True)
    gal_idx = list(range(len(gallery)))
    prb_idx = list(range(len(gallery), len(pool)))
    g_ids = gallery.individual_id.to_numpy()
    p_ids = probe.individual_id.to_numpy()
    p_bins = probe.interval_months.to_numpy()
    print(f"A/B кропа на DEV: gallery {len(gallery)} | probe {len(probe)} | test ЗАПЕЧАТАН")

    # кэш BiRefNet-маски на кадр (1 раз)
    print("Считаю BiRefNet-маски (1 раз на кадр)...")
    cache = []
    n_lbl = 0
    for p in pool.path_rel:
        img = np.array(Image.open(ROOT / p).convert("RGB"))
        m, _, _ = body_mask(img)
        cache.append((img, m))

    results = {}
    hits_by_method = {}
    for method in METHODS:
        grays = []
        for (img, m), p in zip(cache, pool.path_rel):
            if method == "yellow":
                ym = _yellow_mask(img)
                grays.append(to_sift_gray(tight_crop(img, ym)))
            else:
                g, has = _crop_from_cache(img, m, method)
                grays.append(g); n_lbl += int(has and method == "birefnet_label")
        feats = _sift_feats(grays)
        S = score_matrix(feats, prb_idx, gal_idx)
        hits = recall_from_scores(S, p_ids, g_ids)
        hits_by_method[method] = hits
        row = {}
        for k in KS:
            pv, lo, hi = wilson(int(hits[k].sum()), len(p_ids))
            row[f"recall@{k}"] = {"value": round(pv, 3), "ci": [round(lo, 3), round(hi, 3)]}
        # adult-срез приблизим интервалом ≤2мес (молодь даёт длинные/мусорные); по интервалу
        row["by_interval"] = {}
        for bI in sorted(set(p_bins)):
            mI = p_bins == bI
            row["by_interval"][f"{int(bI)}мес"] = {k: round(float(hits[k][mI].mean()), 3) for k in KS}
        results[method] = row

    # McNemar vs yellow (baseline) по @1 и @5
    base = hits_by_method["yellow"]
    print(f"\n{'метод':<24}" + "  ".join(f"R@{k}" for k in KS) + "   McNemar@1(b,c,p)")
    summary = {"gallery": len(gallery), "probe": len(probe), "methods": {}}
    for method in METHODS:
        h = hits_by_method[method]
        cells = []
        for k in KS:
            v = results[method][f"recall@{k}"]
            cells.append(f"{v['value']:.3f}")
        if method == "yellow":
            mc = "(baseline)"
        else:
            b, c, p = mcnemar(base[1], h[1])
            mc = f"b={b} c={c} p={p:.3f}"
        print(f"{method:<24}" + "  ".join(cells) + f"   {mc}")
        summary["methods"][method] = results[method]

    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "ab_crop_dev.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n✅ artifacts/ab_crop_dev.json (test не вскрывался)")


if __name__ == "__main__":
    main()
