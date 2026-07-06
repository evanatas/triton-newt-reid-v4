"""Часть 3 — sanity на DEV (НЕ test). Проверка корректности протокола: SIFT temporal-recall на dev TK.
Ожидаемо ≈0.28 (близко к общему sanity). Test ЗАПЕЧАТАН — не вскрывается.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "spike_lab"))
from sift_eval import sift_all, score_matrix, recall_from_scores   # noqa: E402
from embed_eval import wilson, KS   # noqa: E402
from sealed import assert_unsealed   # noqa: E402
from common import ROOT   # noqa: E402


def main():
    assert_unsealed("dev", unseal=False)   # dev открыт — гейт пропускает
    spl = pd.read_csv(HERE / "splits.csv")
    reg = pd.read_csv(HERE / "registry.csv")
    df = spl.merge(reg[["frame_id", "path_rel"]], on="frame_id")
    df["abs_path"] = df.path_rel.map(lambda r: str(ROOT / r))

    tk = df[df.cohort == "TK"]
    # галерея dev = dev gallery (closed, не open_new) + distractor особи (реализм)
    gallery = tk[((tk.split_fold == "dev") & (tk.split_role == "gallery") & (~tk.is_open_new))
                 | (tk.split_fold == "distractor")].reset_index(drop=True)
    probe = tk[(tk.split_fold == "dev") & (tk.split_role == "probe") & (~tk.is_open_new)].reset_index(drop=True)
    pool = pd.concat([gallery, probe], ignore_index=True)
    print(f"DEV sanity (SIFT): gallery {len(gallery)} кадров / {gallery.individual_id.nunique()} особей "
          f"(вкл. {(gallery.split_fold=='distractor').sum()} дистракторов) | probe {len(probe)}")
    print(f"random top-1 = {1/gallery.individual_id.nunique():.3f}")

    feats = sift_all(pool)
    gal_idx = list(range(len(gallery)))
    prb_idx = list(range(len(gallery), len(pool)))
    S = score_matrix(feats, prb_idx, gal_idx)
    hits = recall_from_scores(S, probe.individual_id.to_numpy(), gallery.individual_id.to_numpy())

    p_bins = probe.interval_months.to_numpy()
    n = len(probe)
    print("\n— SIFT на DEV — (identity-level)")
    print(f"{'срез':<12}{'n':>4}  " + "  ".join(f"R@{k}" for k in KS))
    res = {"fold": "dev", "n": int(n), "random_top1": round(1/gallery.individual_id.nunique(), 3), "overall": {}, "by_interval": {}}
    cells = []
    for k in KS:
        p, lo, hi = wilson(int(hits[k].sum()), n)
        cells.append(f"{p:.3f}[{lo:.2f}-{hi:.2f}]")
        res["overall"][f"recall@{k}"] = {"value": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}
    print(f"{'ВСЕ':<12}{n:>4}  " + "  ".join(cells))
    for b in sorted(set(p_bins)):
        m = p_bins == b
        nb = int(m.sum())
        cb = [f"{wilson(int(hits[k][m].sum()), nb)[0]:.3f}" for k in KS]
        res["by_interval"][f"{int(b)}мес"] = {"n": nb, **{f"recall@{k}": round(float(hits[k][m].mean()), 3) for k in KS}}
        print(f"  {int(b)}мес{'':<6}{nb:>4}  " + "  ".join(cb))

    (HERE / "artifacts" / "sanity_dev.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n✅ artifacts/sanity_dev.json  (test ЗАПЕЧАТАН, не вскрывался)")


if __name__ == "__main__":
    (HERE / "artifacts").mkdir(exist_ok=True)
    main()
