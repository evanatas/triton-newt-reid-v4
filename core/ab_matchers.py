"""Часть 7 — A/B деформ-верификации на DEV TK temporal (test ЗАПЕЧАТАН).

Гейт выбора деформ-модели на SIFT: homography (baseline) vs affine vs partial vs soft.
identity-level recall + Wilson CI + McNemar vs homography (сериализуется в JSON — аудит-след, F-8). Победитель → affine.
Learned-матчеры (DISK/ALIKED+LightGlue) оценены ОТДЕЛЬНО в learned_probe.py.
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
for _p in ("..", "../segment", "../registry", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
from sift_eval import recall_from_scores          # noqa: E402
from embed_eval import wilson, KS                 # noqa: E402
from sealed import assert_unsealed                # noqa: E402
import matchers as M                              # noqa: E402
import deform as D                                # noqa: E402

CROPS = (HERE / ".." / "segment").resolve()


def mcnemar(a_hits, b_hits):
    b = int((~a_hits & b_hits).sum()); c = int((a_hits & ~b_hits).sum()); n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def load_pool():
    spl = pd.read_csv(HERE / ".." / "registry" / "splits.csv")
    man = pd.read_csv(CROPS / "crops_manifest.csv")[["frame_id", "crop_path"]]
    df = spl.merge(man, on="frame_id")
    tk = df[df.cohort == "TK"]
    gallery = tk[((tk.split_fold == "dev") & (tk.split_role == "gallery") & (~tk.is_open_new))
                 | (tk.split_fold == "distractor")].reset_index(drop=True)
    probe = tk[(tk.split_fold == "dev") & (tk.split_role == "probe") & (~tk.is_open_new)].reset_index(drop=True)
    return gallery, probe


def score_matrix(matcher, feats_p, feats_g, method):
    S = np.zeros((len(feats_p), len(feats_g)), np.float32)
    for i, fp in enumerate(feats_p):
        for j, fg in enumerate(feats_g):
            pa, pb = matcher.matched(fp, fg)
            S[i, j] = D.verify(pa, pb, method)
    return S


def main(configs):
    assert_unsealed("dev", unseal=False)
    gallery, probe = load_pool()
    g_ids, p_ids = gallery.individual_id.to_numpy(), probe.individual_id.to_numpy()
    p_bins = probe.interval_months.to_numpy()
    print(f"A/B матчеров на DEV: gallery {len(gallery)} | probe {len(probe)} | test ЗАПЕЧАТАН\n")

    def imgs(rows):
        return [cv2.cvtColor(cv2.imread(str(CROPS / r.crop_path)), cv2.COLOR_BGR2RGB) for _, r in rows.iterrows()]
    g_imgs, p_imgs = imgs(gallery), imgs(probe)

    out = {}; hits_by = {}
    print(f"{'конфиг':<28}{'R@1':>16}{'R@5':>8}{'R@10':>7}  McNemar@1(b,c,p)")
    for cfg in configs:
        name, matcher_name, method = cfg["label"], cfg["matcher"], cfg["deform"]
        t0 = time.time()
        m = M.build(matcher_name)
        fg = [m.extract(im) for im in g_imgs]
        fp = [m.extract(im) for im in p_imgs]
        S = score_matrix(m, fp, fg, method)
        hits = recall_from_scores(S, p_ids, g_ids)
        hits_by[name] = hits
        row = {"matcher": matcher_name, "deform": method, "sec": round(time.time() - t0, 1)}
        cells = []
        for k in KS:
            v, lo, hi = wilson(int(hits[k].sum()), len(p_ids))
            row[f"recall@{k}"] = {"value": round(v, 3), "ci": [round(lo, 2), round(hi, 2)]}
            cells.append(f"{v:.3f}")
        row["by_interval"] = {f"{int(bI)}мес": {k: round(float(hits[k][p_bins == bI].mean()), 3) for k in KS}
                              for bI in sorted(set(p_bins))}
        if name == "sift_homography":
            mc = "(baseline)"
        else:
            b, c, p = mcnemar(hits_by["sift_homography"][1], hits[1])
            row["mcnemar_vs_homography@1"] = {"b": b, "c": c, "p": round(p, 4)}   # F-8: сериализуем аудит-след
            mc = f"b={b} c={c} p={p:.3f}"
        print(f"{name:<28}{cells[0]:>10}[{row['recall@1']['ci'][0]:.2f}-{row['recall@1']['ci'][1]:.2f}]"
              f"{cells[1]:>8}{cells[2]:>7}  {mc}")
        out[name] = row

    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "ab_matchers_dev.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n— по интервалам (R@1) —")
    for name, row in out.items():
        bi = row["by_interval"]
        print(f"  {name:<28} 1мес {bi.get('1мес',{}).get(1,'-')}  2мес {bi.get('2мес',{}).get(1,'-')}")
    print("\n✅ artifacts/ab_matchers_dev.json (test не вскрывался)")


CONFIGS = [   # F-8: деформ-A/B (совпадает с ab_matchers_dev.json + цитируемыми числами). Learned — в learned_probe.py.
    {"label": "sift_homography", "matcher": "sift", "deform": "homography"},   # baseline (= спайк)
    {"label": "sift_affine",     "matcher": "sift", "deform": "affine"},       # деформ R4.1 (победитель)
    {"label": "sift_partial",    "matcher": "sift", "deform": "partial"},      # 4-DOF similarity
    {"label": "sift_soft",       "matcher": "sift", "deform": "soft"},         # мягкая гео-консистентность
]

if __name__ == "__main__":
    main(CONFIGS)
