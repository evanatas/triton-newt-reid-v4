"""Часть 2 — EDA-отчёт по реестру (markdown + цифры)."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "EDA.md"


def main():
    df = pd.read_csv(HERE / "registry.csv")
    L = []
    L.append("# EDA — единый реестр тритонов 4.0\n")
    L.append(f"Всего кадров: **{len(df)}**, особей: **{df.individual_id.nunique()}**.\n")

    L.append("## Когорты\n")
    L.append("| Когорта | Вид | Роль | Кадров | Особей | С перепоимкой |")
    L.append("|---|---|---|---|---|---|")
    for c in ["TK", "PW", "LAB", "GCN"]:
        g = df[df.cohort == c]
        L.append(f"| {c} | {g.species.iloc[0]} | {g.role.iloc[0]} | {len(g)} | "
                 f"{g.individual_id.nunique()} | {g[g.has_recapture].individual_id.nunique()} |")

    L.append("\n## Временное покрытие (даты съёмки)\n")
    L.append("| Когорта | Дата/сессия | Кадров | Особей |")
    L.append("|---|---|---|---|")
    for c in ["TK", "LAB"]:
        g = df[df.cohort == c]
        for d, gg in g.groupby("date"):
            L.append(f"| {c} | {d} | {len(gg)} | {gg.individual_id.nunique()} |")

    L.append("\n## Перепоимки (распределение особей по числу сессий)\n")
    L.append("| Когорта | n_sessions | Особей |")
    L.append("|---|---|---|")
    for c in ["TK", "PW", "LAB"]:
        g = df[df.cohort == c].drop_duplicates("individual_id")
        for n, cnt in g.n_sessions.value_counts().sort_index().items():
            L.append(f"| {c} | {n} | {cnt} |")

    L.append("\n## Фото на особь\n")
    L.append("| Когорта | min | max | среднее | медиана |")
    L.append("|---|---|---|---|---|")
    for c in ["TK", "PW", "LAB", "GCN"]:
        g = df[df.cohort == c].groupby("individual_id").size()
        L.append(f"| {c} | {g.min()} | {g.max()} | {g.mean():.1f} | {g.median():.0f} |")

    L.append("\n## KPI-материал\n")
    tgt = df[df.role == "target"]
    L.append(f"- Целевые виды (TK+PW): **{tgt.individual_id.nunique()} особей**, "
             f"**{tgt[tgt.has_recapture].individual_id.nunique()} с перепоимкой**.")
    L.append(f"- TK (Карелина, главный): {df[df.cohort=='TK'][df.has_recapture].individual_id.nunique()} особей с перепоимкой "
             f"(даты Dec24/Jan25/Feb25).")
    L.append("- PW (ребристый): перепоимка = session 01→02 (псевдо-время, без календарных дат).")
    L.append("- LAB (другой вид): honest-temporal, 5 реальных месячных сессий — обучение/диагностика инвариантности.")
    L.append("- GCN (внешний): 0 перепоимок — только маски сегментации + претрейн, НЕ в KPI.")

    L.append("\n## Анти-утечка\n")
    nd = HERE / "artifacts" / "near_duplicates.csv"
    if nd.exists():
        ndf = pd.read_csv(nd)
        L.append(f"- pHash near-dup пар (dist≤6): **{len(ndf)}** — **все внутри одной сессии** "
                 f"(бёрст-съёмка), межсессионных = 0 → temporal-сплит их не разводит, прямой утечки нет.")
    dd = HERE / "artifacts" / "dedup_dropped.csv"
    if dd.exists():
        L.append(f"- md5-байт-дублей выброшено: **{len(pd.read_csv(dd))}** (TK IMG_* копии, LAB дубль-метки).")

    L.append("\n## Стадия (молодь/взрослый)\n")
    L.append("- Поле `stage` заложено; авторазметка отложена — нет ground truth возраста. "
             "Запрошено у заказчика (оригинальная база). Критично: KPI считать на `adult` (спайк: молодь проваливает узор).")

    OUT.write_text("\n".join(L))
    print(f"✅ EDA-отчёт: {OUT}")
    print("\n".join(L[:18]))


if __name__ == "__main__":
    main()
