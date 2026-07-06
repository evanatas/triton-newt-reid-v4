"""Часть 8 — формализация open-set на DEV: калибровка порога + tri-state (known / на проверку / new).

DEV-ONLY, аддитивно (как interpret.py): ЗАМОРОЖЕННОЕ ядро и sealed НЕ трогает. Финальный open-set на
sealed-тесте (BAKS 0.67 / BAUS 1.0 / G 0.819, n_new=17) заморожен в `artifacts/final_sealed_test.json`
и здесь НЕ пересчитывается — этот скрипт лишь калибрует порог на DEV и определяет tri-state-зоны.

Ключевой факт честности: open-new особи — ВСЕ TK (Карелина), значит open-set 4.0 = **within-TK**
(known и new — один вид), БЕЗ конфаунда состава по виду (в 3.0 new были другого вида PW → конфаунд).
within-PW чистый срез невычислим (у PW нет open-new особей) и не нужен — within-TK его замещает и сильнее.

Порог: индекс Юдена на DEV (max TAR−FAR), как в metrics.youden_threshold. Tri-state:
  τ_known = наименьший порог с FAR=0 на dev (уверенно known: ни один new не принят);
  τ_new   = порог Юдена (уверенно new ниже него);
  зона [τ_new, τ_known) → «на проверку» (human-in-the-loop, ФТ-4 / комментарий заказчика №2 known/new/other).

Запуск:  python openset.py   →  artifacts/openset_dev.json
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
import eval_core as EC          # noqa: E402  load_cohort/score_probe/CROPS/tkey — переиспуем dev-путь
import metrics as MET           # noqa: E402  open_set/youden_threshold
import matchers as M            # noqa: E402
import deform as D              # noqa: E402

DEFORM = "affine"               # замороженное ядро


def rolling_tops(score_fn, feats, probe, refs_all):
    """max-pool top-score каждой probe против rolling-референсов (строго ранее по времени)."""
    tops = []
    for _, p in probe.iterrows():
        _, top = EC.score_probe(score_fn, feats, p, refs_all[refs_all.tkey < p.tkey])
        tops.append(top)
    return np.array(tops, dtype=float)


def tri_state_thresholds(known_tops, new_tops):
    """(τ_known, τ_new): τ_known = min порог с FAR=0; τ_new = Юден. Гарантируем τ_new ≤ τ_known."""
    cand = np.unique(np.concatenate([known_tops, new_tops])) if len(new_tops) else np.unique(known_tops)
    far0 = [float(t) for t in cand if (new_tops >= t).mean() == 0.0] if len(new_tops) else [float(cand.min())]
    tau_known = min(far0) if far0 else float(cand.max()) + 1.0
    tau_new = MET.youden_threshold(known_tops, new_tops) if len(new_tops) else tau_known
    return tau_known, min(tau_new, tau_known)


def zone(score, tau_known, tau_new):
    return "known" if score >= tau_known else ("new" if score < tau_new else "review")


def _fracs(scores, tau_known, tau_new):
    zs = [zone(float(s), tau_known, tau_new) for s in scores]
    n = max(len(zs), 1)
    return {k: round(sum(z == k for z in zs) / n, 3) for k in ("known", "review", "new")}


def main():
    sift = M.build("sift")
    score_fn = lambda a, b: D.verify(*sift.matched(a, b), DEFORM)   # SIFT+affine, ядро

    refs_all, _gal, known_probe, new_probe = EC.load_cohort("TK", "dev")   # DEV (test запечатан, не трогаем)
    rows = pd.concat([refs_all, known_probe, new_probe]).drop_duplicates("frame_id")
    feats = {r.frame_id: sift.extract(cv2.cvtColor(cv2.imread(str(EC.CROPS / r.crop_path)), cv2.COLOR_BGR2RGB))
             for _, r in rows.iterrows()}

    known_tops = rolling_tops(score_fn, feats, known_probe, refs_all)
    new_tops = rolling_tops(score_fn, feats, new_probe, refs_all)

    thr_youden = MET.youden_threshold(known_tops, new_tops)
    os_dev = MET.open_set(known_tops, new_tops, thr_youden)
    tau_known, tau_new = tri_state_thresholds(known_tops, new_tops)

    known_species = sorted(known_probe.cohort.unique().tolist())
    new_species = sorted(new_probe.cohort.unique().tolist())
    clean_within = known_species == new_species == ["TK"]

    out = {
        "scope": "DEV-only калибровка open-set + tri-state; ядро и sealed заморожены (final_sealed_test.json).",
        "within_cohort": {
            "known_cohorts": known_species, "new_cohorts": new_species, "clean_within_TK": bool(clean_within),
            "note": ("open-set = within-TK (known и new — Карелина) → конфаунда ВИДА нет; сильнее 3.0, "
                     "где new были другого вида (PW) и AUROC=0.446 был спутан составом."),
        },
        "within_PW": "невычислим — у PW нет open-new особей (все open-new = TK); within-TK замещает и чище.",
        "n_known": int(len(known_tops)), "n_new": int(len(new_tops)),
        "youden_dev": {"threshold": round(float(thr_youden), 3), **os_dev},
        "tri_state": {
            "tau_known_FAR0": round(float(tau_known), 3),
            "tau_new_youden": round(float(tau_new), 3),
            "review_zone": [round(float(tau_new), 3), round(float(tau_known), 3)],
            "review_degenerate_on_dev": bool(tau_new == tau_known),
            "known_probe_distribution": _fracs(known_tops, tau_known, tau_new),
            "new_probe_distribution": _fracs(new_tops, tau_known, tau_new),
            "legend": "score = affine-inliers top-1; known ≥ τ_known; new < τ_new; между → на проверку (human-in-the-loop).",
            "operational_note": ("на dev within-TK порог 9 inliers разделяет known/new ЧИСТО (FAR=0) → метрически "
                                 "review-зона вырождается. В демо «на проверку» задаётся в шкале КАЛИБРОВАННОЙ "
                                 "УВЕРЕННОСТИ (не сырые inliers) как human-in-the-loop подушка вокруг порога — "
                                 "см. backend/reid_service.verdict (conf_thr/review_thr). Две разные шкалы, не смешивать."),
        },
        "sealed_frozen_reference": {
            "note": "НЕ пересчитывается здесь — цитата из final_sealed_test.json (вскрыт 2026-07-02).",
            "threshold": 9.0, "BAKS": 0.67, "BAUS": 1.0, "FAR": 0.0, "G": 0.819, "n_new": 17,
        },
        "WARN": ("dev n мал → калибровочный индикатор, не финальная метрика; финальный open-set — sealed "
                 "(within-TK, 17 new, заморожен). Порог Юдена dev сверить с core_config.open_set_DRYRUN.threshold."),
    }

    print(f"open-set DEV (within-TK={clean_within}): known {len(known_tops)} / new {len(new_tops)}")
    print(f"  Юден-порог dev = {thr_youden:.1f} | BAKS {os_dev['BAKS']} BAUS {os_dev['BAUS']} G {os_dev['G']} "
          f"(сверить с config open_set_DRYRUN.threshold=9.0)")
    print(f"  tri-state: τ_new(Юден)={tau_new:.1f} ≤ τ_known(FAR=0)={tau_known:.1f} | "
          f"review-зона [{tau_new:.1f}, {tau_known:.1f})")
    print(f"  known-пробы: {out['tri_state']['known_probe_distribution']}")
    print(f"  new-пробы:   {out['tri_state']['new_probe_distribution']}")

    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "openset_dev.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("✅ artifacts/openset_dev.json (sealed НЕ тронут)")


if __name__ == "__main__":
    main()
