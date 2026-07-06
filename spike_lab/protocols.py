"""
Часть 6 (спайк LAB) — под-шаг 4: протоколы оценки на одних эмбеддингах.

Два честных temporal-протокола (оба identity-level, time-aware, анти-утечка):
  - earliest : gallery = самая ранняя сессия особи, probe = все поздние (МАКС дрейф; текущий бенчмарк).
  - adjacent : для каждой пары соседних сессий gallery = prev, probe = next (дрейф за ОДИН шаг ~1.2 мес).

Эмбеддит все 99 кадров один раз (кэш), считает оба протокола, печатает таблицы.
Запуск:  python protocols.py --model megadescriptor --crop yellow
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from embed_eval import EMBEDDERS, embed_paths, identity_recall, wilson, KS, ART, BENCH


def embed_all(model_name: str, crop: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(BENCH).reset_index(drop=True)
    cache = ART / f"emb_all_{model_name}_{crop}.npz"
    if cache.exists():
        emb = np.load(cache)["emb"]
        print(f"эмбеддинги всех кадров из кэша: {cache.name}")
    else:
        print(f"Загрузка {model_name}…")
        model, transform = EMBEDDERS[model_name]()
        print(f"эмбеддинг всех {len(df)} кадров (crop={crop})…")
        emb = embed_paths(df.abs_path.tolist(), model, transform, crop)
        np.savez(cache, emb=emb)
        print(f"сохранено: {cache.name} (dim={emb.shape[1]})")
    return emb, df


def _table(hits: dict, n: int, label: str, res: dict):
    cells = []
    for k in KS:
        p, lo, hi = wilson(int(hits[k].sum()), n)
        cells.append(f"{p:.3f}[{lo:.2f}-{hi:.2f}]")
        res.setdefault(label, {})[f"recall@{k}"] = {"value": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "n": n}
    print(f"{label:<22}{n:>4}  " + "  ".join(cells))


def protocol_earliest(emb, df, res):
    gal = df[df.role == "gallery"]; prb = df[df.role == "probe"]
    g_emb, g_ids = emb[gal.index.to_numpy()], gal.individual_id.to_numpy()
    p_emb, p_ids = emb[prb.index.to_numpy()], prb.individual_id.to_numpy()
    hits = identity_recall(p_emb, p_ids, g_emb, g_ids)
    print("\n— Протокол EARLIEST (gallery=ранняя сессия, probe=поздние; макс. дрейф) —")
    print(f"{'срез':<22}{'n':>4}  " + "  ".join(f'R@{k:<2}[CI]' for k in KS))
    _table(hits, len(p_ids), "ВСЕ", res["earliest"])


def protocol_adjacent(emb, df, res):
    sessions = sorted(df.session_date.unique())
    all_hits = {k: [] for k in KS}
    per_pair = {}
    for d_prev, d_next in zip(sessions, sessions[1:]):
        g = df[df.session_date == d_prev]; p = df[df.session_date == d_next]
        g_ids = g.individual_id.to_numpy()
        # probe учитывается только если его особь есть в предыдущей сессии (closed-set по паре)
        mask = p.individual_id.isin(set(g_ids)).to_numpy()
        p_use = p[mask]
        if len(p_use) == 0:
            continue
        gap = (pd.Timestamp(d_next) - pd.Timestamp(d_prev)).days
        hits = identity_recall(emb[p_use.index.to_numpy()], p_use.individual_id.to_numpy(),
                               emb[g.index.to_numpy()], g_ids)
        per_pair[f"{d_prev}->{d_next}"] = {"n": int(len(p_use)), "gap_days": int(gap),
                                           **{f"recall@{k}": round(float(hits[k].mean()), 3) for k in KS}}
        for k in KS:
            all_hits[k].append(hits[k])
    agg = {k: np.concatenate(all_hits[k]) for k in KS}
    n = len(agg[KS[0]])
    print("\n— Протокол ADJACENT (gallery=пред. сессия, probe=след.; дрейф за один шаг ~1.2 мес) —")
    print(f"{'срез':<22}{'n':>4}  " + "  ".join(f'R@{k:<2}[CI]' for k in KS))
    _table(agg, n, "ВСЕ соседние", res["adjacent"])
    res["adjacent"]["per_pair"] = per_pair
    print("  по парам сессий:")
    for name, d in per_pair.items():
        print(f"    {name} ({d['gap_days']}д, n={d['n']}): R@1 {d['recall@1']:.2f}  R@5 {d['recall@5']:.2f}  R@10 {d['recall@10']:.2f}")


def main(model_name: str, crop: str):
    emb, df = embed_all(model_name, crop)
    res = {"model": model_name, "crop": crop, "n_frames": len(df),
           "n_individuals": int(df.individual_id.nunique()),
           "random_top1": round(1 / df.individual_id.nunique(), 3),
           "earliest": {}, "adjacent": {}}
    print("=" * 66)
    print(f"ПРОТОКОЛЫ — {model_name} (crop={crop}) | random top-1 = {res['random_top1']}")
    print("=" * 66)
    protocol_earliest(emb, df, res)
    protocol_adjacent(emb, df, res)
    out = ART / f"protocols_{model_name}_{crop}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n✅ {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="megadescriptor")
    ap.add_argument("--crop", default="yellow")
    a = ap.parse_args()
    main(a.model, a.crop)
