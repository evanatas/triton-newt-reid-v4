"""Часть 2 — под-шаг 2.8: sanity-проверка переноса local-feature метода на ЦЕЛЕВОЙ вид TK (Карелина).

Спайк был на LAB (другой вид). Здесь проверяем на TK: держится ли local-feature ≫ global на Карелине.
gallery = ранняя дата особи, probe = поздние; identity-level recall@1/5/10 по интервалам.
Сравниваем SIFT (local) против MegaDescriptor (global) на одних и тех же TK-перепоимках.

Запуск:  python sanity_tk.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "spike_lab"))   # переиспуем НАШ код спайка (не 3.0)
from sift_eval import sift_all, score_matrix, recall_from_scores   # noqa: E402
from embed_eval import wilson, KS, embed_paths, get_megadescriptor, identity_recall   # noqa: E402
from common import ROOT   # noqa: E402

ART = HERE / "artifacts"


def build_tk_split():
    reg = pd.read_csv(HERE / "registry.csv")
    tk = reg[(reg.cohort == "TK") & (reg.has_recapture) & (reg.date.notna())].copy().reset_index(drop=True)
    tk["abs_path"] = tk.path_rel.map(lambda r: str(ROOT / r))
    roles, intervals = [], []
    for ind, g in tk.groupby("individual_id"):
        dates = sorted(g.date.unique())
        early = dates[0]
        for _, r in g.iterrows():
            if r.date == early:
                roles.append("gallery"); intervals.append("")
            else:
                roles.append("probe")
                ym0 = [int(x) for x in early.split("-")]; ym1 = [int(x) for x in r.date.split("-")]
                m = (ym1[0]-ym0[0])*12 + (ym1[1]-ym0[1])
                intervals.append(f"{m}мес")
    tk = tk.assign(_role=np.array(roles, dtype=object), _interval=np.array(intervals, dtype=object))
    return tk


def recall_table(hits, ids_bins, label, store):
    p_ids, p_bins = ids_bins
    n = len(p_ids)
    cells = []
    for k in KS:
        p, lo, hi = wilson(int(hits[k].sum()), n)
        cells.append(f"{p:.3f}[{lo:.2f}-{hi:.2f}]")
        store.setdefault(label, {})[f"recall@{k}"] = {"value": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": n}
    print(f"{label:<16}{n:>4}  " + "  ".join(cells))
    for b in sorted(set(p_bins)):
        m = p_bins == b
        nb = int(m.sum())
        cb = [f"{wilson(int(hits[k][m].sum()), nb)[0]:.3f}" for k in KS]
        print(f"  {b:<14}{nb:>4}  " + "  ".join(cb))


def main():
    ART.mkdir(exist_ok=True)
    tk = build_tk_split()
    gal = tk[tk._role == "gallery"]; prb = tk[tk._role == "probe"]
    n_ind = tk.individual_id.nunique()
    print(f"TK sanity: {n_ind} особей с перепоимкой | gallery {len(gal)} | probe {len(prb)} | "
          f"random top-1 = {1/n_ind:.3f}")
    p_ids = prb.individual_id.to_numpy(); g_ids = gal.individual_id.to_numpy()
    p_bins = prb._interval.to_numpy()
    res = {"n_individuals": int(n_ind), "random_top1": round(1/n_ind, 3), "sift": {}, "megadescriptor": {}}

    # --- SIFT (local-feature) ---
    print("\nSIFT (local-feature + RANSAC)…")
    feats = sift_all(tk)
    S = score_matrix(feats, prb.index.to_numpy(), gal.index.to_numpy())
    hits_sift = recall_from_scores(S, p_ids, g_ids)
    print("— SIFT — (R@1 / R@5 / R@10, identity-level)")
    recall_table(hits_sift, (p_ids, p_bins), "ВСЕ", res["sift"])

    # --- MegaDescriptor (global) для сравнения ---
    print("\nMegaDescriptor (global embedding)…")
    model, transform = get_megadescriptor()
    cache = ART / "tk_emb_mega.npz"
    if cache.exists():
        emb = np.load(cache)["emb"]
    else:
        emb = embed_paths(tk.abs_path.tolist(), model, transform, crop="yellow")
        np.savez(cache, emb=emb)
    hits_mega = identity_recall(emb[prb.index.to_numpy()], p_ids, emb[gal.index.to_numpy()], g_ids)
    print("— MegaDescriptor — (R@1 / R@5 / R@10, identity-level)")
    recall_table(hits_mega, (p_ids, p_bins), "ВСЕ", res["megadescriptor"])

    (ART / "sanity_tk.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n✅ artifacts/sanity_tk.json")
    s1, m1 = res["sift"]["ВСЕ"]["recall@1"]["value"], res["megadescriptor"]["ВСЕ"]["recall@1"]["value"]
    print(f"\nВЫВОД: SIFT R@1 {s1} vs MegaDescriptor R@1 {m1} на TK — "
          f"{'local ≫ global подтверждён на Карелине' if s1 > m1 else 'на TK картина иная (разобрать)'}")


if __name__ == "__main__":
    main()
