"""
Часть 6 (спайк LAB) — под-шаг 3b: local-feature matching (SIFT + геом. верификация).

Принципиально другой класс методов, чем global-эмбеддинг (подход AmphIdent / I3S / Wild-ID для
амфибий): локальные дескрипторы пятен + геометрическая согласованность (RANSAC-homography inliers).
Гипотеза: устойчивее к возрастному дрейфу узора, чем глобальный вектор.

score(a,b) = число RANSAC-inlier матчей. Identity-level recall, протоколы earliest + adjacent.
Запуск:  python sift_eval.py
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from embed_eval import yellow_crop_pil, identity_recall, wilson, KS, ART, BENCH
from PIL import Image

MAX_SIDE = 600   # ресайз кропа (скорость SIFT)
LOWE = 0.75
MIN_GOOD = 4


def prep(path: str) -> np.ndarray:
    img = yellow_crop_pil(Image.open(path).convert("RGB"))
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    s = MAX_SIDE / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)))
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)   # выровнять контраст между сессиями


def sift_all(df):
    sift = cv2.SIFT_create(nfeatures=1500)
    feats = []
    for p in df.abs_path:
        kp, des = sift.detectAndCompute(prep(p), None)
        feats.append((np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), des))
    return feats


def score(fa, fb) -> int:
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good_a, good_b = [], []
    for m_n in bf.knnMatch(da, db, k=2):
        if len(m_n) == 2 and m_n[0].distance < LOWE * m_n[1].distance:
            good_a.append(pa[m_n[0].queryIdx]); good_b.append(pb[m_n[0].trainIdx])
    if len(good_a) < MIN_GOOD:
        return len(good_a)
    _, mask = cv2.findHomography(np.float32(good_a), np.float32(good_b), cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else len(good_a)


def score_matrix(feats, prb_idx, gal_idx):
    S = np.zeros((len(prb_idx), len(gal_idx)), np.float32)
    for i, pi in enumerate(prb_idx):
        for j, gj in enumerate(gal_idx):
            S[i, j] = score(feats[pi], feats[gj])
    return S


def recall_from_scores(S, p_ids, g_ids, ks=KS):
    uniq = np.array(sorted(set(g_ids)))
    agg = np.full((S.shape[0], len(uniq)), -1.0)
    for j, gid in enumerate(uniq):
        agg[:, j] = S[:, g_ids == gid].max(axis=1)
    ranked = uniq[np.argsort(-agg, axis=1)]
    return {k: np.array([p_ids[i] in ranked[i, :k] for i in range(len(p_ids))]) for k in ks}


def tbl(hits, n, label, store):
    cells = []
    for k in KS:
        p, lo, hi = wilson(int(hits[k].sum()), n)
        cells.append(f"{p:.3f}[{lo:.2f}-{hi:.2f}]")
        store.setdefault(label, {})[f"recall@{k}"] = {"value": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": n}
    print(f"{label:<16}{n:>4}  " + "  ".join(cells))


def main():
    df = pd.read_csv(BENCH).reset_index(drop=True)
    print(f"SIFT по {len(df)} кадрам (CLAHE, RANSAC-inliers)…")
    cache = ART / "sift_feats.npz"
    feats = sift_all(df)   # SIFT-объекты не пиклятся тривиально → считаем каждый раз (быстро на 99)
    res = {"method": "SIFT+RANSAC", "random_top1": round(1 / df.individual_id.nunique(), 3), "earliest": {}, "adjacent": {}}
    print("=" * 60)
    print(f"SIFT local-feature matching | random top-1 = {res['random_top1']}")
    print("=" * 60)

    # earliest
    gal = df[df.role == "gallery"]; prb = df[df.role == "probe"]
    S = score_matrix(feats, prb.index.to_numpy(), gal.index.to_numpy())
    hits = recall_from_scores(S, prb.individual_id.to_numpy(), gal.individual_id.to_numpy())
    print("\n— EARLIEST —"); print(f"{'срез':<16}{'n':>4}  " + "  ".join(f'R@{k:<2}' for k in KS))
    tbl(hits, len(prb), "ВСЕ", res["earliest"])

    # adjacent
    sessions = sorted(df.session_date.unique())
    all_hits = {k: [] for k in KS}; per_pair = {}
    for d_prev, d_next in zip(sessions, sessions[1:]):
        g = df[df.session_date == d_prev]; p = df[df.session_date == d_next]
        gset = set(g.individual_id); p_use = p[p.individual_id.isin(gset)]
        if len(p_use) == 0:
            continue
        S = score_matrix(feats, p_use.index.to_numpy(), g.index.to_numpy())
        h = recall_from_scores(S, p_use.individual_id.to_numpy(), g.individual_id.to_numpy())
        per_pair[f"{d_prev}->{d_next}"] = {"n": int(len(p_use)), **{f"recall@{k}": round(float(h[k].mean()), 3) for k in KS}}
        for k in KS:
            all_hits[k].append(h[k])
    agg = {k: np.concatenate(all_hits[k]) for k in KS}
    print("\n— ADJACENT —"); print(f"{'срез':<16}{'n':>4}  " + "  ".join(f'R@{k:<2}' for k in KS))
    tbl(agg, len(agg[1]), "ВСЕ соседние", res["adjacent"])
    res["adjacent"]["per_pair"] = per_pair
    for name, d in per_pair.items():
        print(f"    {name} (n={d['n']}): R@1 {d['recall@1']:.2f}  R@5 {d['recall@5']:.2f}  R@10 {d['recall@10']:.2f}")

    (ART / "metrics_sift.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n✅ metrics_sift.json")


if __name__ == "__main__":
    main()
