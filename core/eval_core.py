"""Часть 7 / задача #22 — честный оценщик ЗАМОРОЖЕННОГО ядра (SIFT+affine+rolling), параметризован по когорте.

Обобщает multishot.py на TK/PW/LAB: ось времени = date (TK/LAB) либо session (PW, дат нет). Режимы single + rolling.
Метрики честности через core/metrics.py: macro/micro/FRR; для TK — open-set (known vs open_new, Youden→BAKS/BAUS/G).

TK — на dev (test ЗАПЕЧАТАН). PW/LAB — fold=aux (не sealed): РАЗОВАЯ проверка обобщения, НЕ тюнинг.
Запуск: python eval_core.py --cohort TK|PW|LAB
"""
from __future__ import annotations
import argparse
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
import metrics as MET                             # noqa: E402

CROPS = (HERE / ".." / "segment").resolve()
DEFORM = "affine"                                 # замороженное ядро


def tkey_of(r):
    """Сравнимый ключ времени: дата YYYYMM (TK/LAB) или номер сессии (PW). Меньше = раньше."""
    if pd.notna(r.date) and str(r.date) != "":
        return int(str(r.date).replace("-", "")) * 100
    return int(r.session) if str(r.session).isdigit() else 0


def load_cohort(cohort, fold="dev"):
    """-> (refs_all, gallery_single, known_probe, new_probe) с колонкой tkey и crop_path. fold ∈ {dev,test} для TK."""
    spl = pd.read_csv(HERE / ".." / "registry" / "splits.csv")
    man = pd.read_csv(CROPS / "crops_manifest.csv")[["frame_id", "crop_path"]]
    df = spl[spl.cohort == cohort].merge(man, on="frame_id")
    df["tkey"] = df.apply(tkey_of, axis=1)
    if cohort == "TK":
        base = df[(df.split_fold == fold) | (df.split_fold == "distractor")]
        refs_all = base[~base.is_open_new.fillna(False)].copy()          # референсы без неизвестных особей
        refs_all.loc[refs_all.split_fold == "distractor", "tkey"] = -1   # distractor — всегда «раньше» (как multishot)
        gallery_single = refs_all[(refs_all.split_role == "gallery") | (refs_all.split_fold == "distractor")]
        known_probe = df[(df.split_fold == fold) & (df.split_role == "probe") & (~df.is_open_new.fillna(False))]
        new_probe = df[(df.split_fold == fold) & (df.split_role == "probe") & (df.is_open_new.fillna(False))]
    else:                                                                # PW/LAB: всё aux, без distractor/open-set
        refs_all = df
        gallery_single = df[df.split_role == "gallery"]
        known_probe = df[df.split_role == "probe"]
        new_probe = df.iloc[0:0]
    return (refs_all.reset_index(drop=True), gallery_single.reset_index(drop=True),
            known_probe.reset_index(drop=True), new_probe.reset_index(drop=True))


def score_probe(score_fn, feats, p, refs):
    """max-pool score по особи; возврат (ranked особи, top_score). score_fn(feat_p, feat_g) -> float."""
    refs = refs[refs.frame_id != p.frame_id]
    scores = {}
    for _, g in refs.iterrows():
        s = score_fn(feats[p.frame_id], feats[g.frame_id])
        scores[g.individual_id] = max(scores.get(g.individual_id, -1), s)
    if not scores:
        return [], 0.0
    ranked = sorted(scores, key=lambda i: -scores[i])
    return ranked, float(scores[ranked[0]])


def eval_mode(mode, score_fn, feats, known_probe, refs_all, gallery_single):
    hits = {k: [] for k in KS}; tops = []; bins = []
    for _, p in known_probe.iterrows():
        refs = gallery_single if mode == "single" else refs_all[refs_all.tkey < p.tkey]
        ranked, top = score_probe(score_fn, feats, p, refs)
        for k in KS:
            hits[k].append(p.individual_id in ranked[:k])
        tops.append(top); bins.append(p.interval_months)
    hits = {k: np.array(v) for k, v in hits.items()}
    return hits, np.array(tops), np.array(bins)


def by_interval(hits, bins):
    return {f"{int(b)}мес": {f"r@{k}": round(float(hits[k][bins == b].mean()), 3) for k in KS}
            for b in sorted(set(bins[~np.isnan(bins)]))} if len(bins) else {}


def main(cohort, fold="dev", unseal=False):
    assert_unsealed(fold if cohort == "TK" else "dev", unseal)   # TK fold может быть sealed test; PW/LAB всегда aux
    refs_all, gallery_single, known_probe, new_probe = load_cohort(cohort, fold)
    sift = M.build("sift")
    score_fn = lambda a, b: D.verify(*sift.matched(a, b), DEFORM)   # ядро: SIFT+affine
    all_rows = pd.concat([refs_all, known_probe, new_probe]).drop_duplicates("frame_id")
    feats = {r.frame_id: sift.extract(cv2.cvtColor(cv2.imread(str(CROPS / r.crop_path)), cv2.COLOR_BGR2RGB))
             for _, r in all_rows.iterrows()}
    n_ident = refs_all.individual_id.nunique()
    p_ids = known_probe.individual_id.to_numpy()
    print(f"eval {cohort}: refs {len(refs_all)} ({n_ident} особей) | probe known {len(known_probe)} | "
          f"open_new {len(new_probe)} | random top-1 {1/max(n_ident,1):.3f}")

    out = {"cohort": cohort, "n_ref_identities": int(n_ident), "random_top1": round(1 / max(n_ident, 1), 3),
           "modes": {}}
    rolling_known_tops = None
    for mode in ["single", "rolling"]:
        hits, tops, bins = eval_mode(mode, score_fn, feats, known_probe, refs_all, gallery_single)
        rep = MET.report(hits, p_ids)
        rep["wilson@1"] = [round(x, 3) for x in wilson(int(hits[1].sum()), len(p_ids))]
        rep["by_interval"] = by_interval(hits, bins)
        out["modes"][mode] = rep
        if mode == "rolling":
            rolling_known_tops = tops
        print(f"  {mode:<8} R@1 {rep['micro_recall'][1]:.3f} (macro {rep['macro_recall'][1]:.3f}) "
              f"R@5 {rep['micro_recall'][5]:.3f} R@10 {rep['micro_recall'][10]:.3f}")

    # open-set (только TK: есть open_new) — на rolling-режиме; n=5 → НЕДОМОЩНО, dry-run машинерии
    if len(new_probe) > 0:
        new_tops = []
        for _, p in new_probe.iterrows():
            _, top = score_probe(score_fn, feats, p, refs_all[refs_all.tkey < p.tkey])
            new_tops.append(top)
        new_tops = np.array(new_tops)
        thr = MET.youden_threshold(rolling_known_tops, new_tops)
        os_res = MET.open_set(rolling_known_tops, new_tops, thr)
        os_res["n_known"] = int(len(rolling_known_tops)); os_res["n_new"] = int(len(new_tops))
        os_res["WARN"] = "n_new=%d — недомощно, dry-run; реальный open-set на sealed (Часть 11)" % len(new_tops)
        out["open_set"] = os_res
        print(f"  open-set(n_new={len(new_tops)}, НЕДОМОЩНО): thr {thr:.0f} BAKS {os_res['BAKS']} "
              f"BAUS {os_res['BAUS']} G {os_res['G']}")

    suffix = "" if fold == "dev" else f"_{fold}"
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / f"eval_{cohort.lower()}{suffix}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ artifacts/eval_{cohort.lower()}{suffix}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="TK", choices=["TK", "PW", "LAB"])
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    main(a.cohort, a.fold, a.unseal)
