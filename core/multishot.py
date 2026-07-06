"""Часть 7 — multi-reference / rolling gallery (R4.3 из ревью) на SIFT+affine (победитель деформа).

single: gallery = только earliest-сессия (1 реф/особь, как сейчас). Нижняя честная граница.
rolling: для probe из сессии T — gallery = ВСЕ кадры особей из сессий < T (растущая база; прод-режим).
Gallery строго РАНЬШЕ probe → утечки времени нет. Репортим оба раздельно (не смешивать).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
for _p in ("..", "../segment", "../registry", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
from embed_eval import wilson, KS                 # noqa: E402
from sealed import assert_unsealed                # noqa: E402
import matchers as M                              # noqa: E402
import deform as D                                # noqa: E402

CROPS = (HERE / ".." / "segment").resolve()
DEFORM = "affine"


def main():
    assert_unsealed("dev", unseal=False)
    spl = pd.read_csv(HERE / ".." / "registry" / "splits.csv")
    man = pd.read_csv(CROPS / "crops_manifest.csv")[["frame_id", "crop_path"]]
    df = spl.merge(man, on="frame_id")
    tk = df[df.cohort == "TK"]
    # все dev-кадры (gallery earliest + probe поздние) + distractor; не open_new
    dev = tk[((tk.split_fold == "dev") & (~tk.is_open_new)) | (tk.split_fold == "distractor")].reset_index(drop=True)
    # сессия как сравнимое число (дата YYYY-MM → int; distractor=Dec=минимум)
    def skey(r):
        d = r.date
        return int(str(d).replace("-", "")) if pd.notna(d) and d != "" else 202412
    dev["skey"] = dev.apply(skey, axis=1)
    probe = dev[(dev.split_fold == "dev") & (dev.split_role == "probe")].reset_index(drop=True)
    print(f"multi-ref на DEV (SIFT+{DEFORM}): кадров {len(dev)} | probe {len(probe)} | test ЗАПЕЧАТАН")

    m = M.build("sift")
    feats = {r.frame_id: m.extract(cv2.cvtColor(cv2.imread(str(CROPS / r.crop_path)), cv2.COLOR_BGR2RGB))
             for _, r in dev.iterrows()}

    def eval_mode(mode):
        hits = {k: [] for k in KS}
        bins = []
        for _, p in probe.iterrows():
            if mode == "single":
                refs = dev[(dev.split_role == "gallery") | (dev.split_fold == "distractor")]   # earliest + distr
            else:  # rolling: все кадры из сессий РАНЬШЕ probe (+ distractor)
                refs = dev[(dev.skey < p.skey) | (dev.split_fold == "distractor")]
            refs = refs[refs.frame_id != p.frame_id]
            scores = {}
            for _, g in refs.iterrows():
                s = D.verify(*m.matched(feats[p.frame_id], feats[g.frame_id]), DEFORM)
                scores[g.individual_id] = max(scores.get(g.individual_id, -1), s)   # max-pool по особи
            ranked = [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]
            for k in KS:
                hits[k].append(p.individual_id in ranked[:k])
            bins.append(p.interval_months)
        bins = np.array(bins)
        res = {}
        for k in KS:
            h = np.array(hits[k]); v, lo, hi = wilson(int(h.sum()), len(h))
            res[f"recall@{k}"] = {"value": round(v, 3), "ci": [round(lo, 2), round(hi, 2)]}
        res["by_interval"] = {f"{int(b)}мес": {k: round(float(np.array(hits[k])[bins == b].mean()), 3) for k in KS}
                              for b in sorted(set(bins))}
        return res, hits

    out = {}
    print(f"\n{'режим':<16}{'R@1':>16}{'R@5':>8}{'R@10':>7}   по интервалам R@1")
    for mode in ["single", "rolling"]:
        res, _ = eval_mode(mode)
        out[mode] = res
        bi = res["by_interval"]
        c = res["recall@1"]
        print(f"{mode:<16}{c['value']:>10.3f}[{c['ci'][0]:.2f}-{c['ci'][1]:.2f}]"
              f"{res['recall@5']['value']:>8.3f}{res['recall@10']['value']:>7.3f}"
              f"   1мес {bi.get('1мес',{}).get(1,'-')}  2мес {bi.get('2мес',{}).get(1,'-')}")

    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "multishot_dev.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n✅ artifacts/multishot_dev.json (test не вскрывался)")


if __name__ == "__main__":
    main()
