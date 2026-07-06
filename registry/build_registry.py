"""Часть 2 — оркестрация сборки единого реестра всех 4 когорт.

Запуск:  python build_registry.py
Выход:   registry.csv + печать сводки и near-dup отчёта.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from common import ROOT, COLUMNS, SPECIES, ROLE
from parsers import ALL_PARSERS
from dedup import dedup_md5, phash_audit

OUT = Path(__file__).parent / "registry.csv"
DROPPED = Path(__file__).parent / "artifacts" / "dedup_dropped.csv"
NEARDUP = Path(__file__).parent / "artifacts" / "near_duplicates.csv"


def assign_individual_meta(df: pd.DataFrame) -> pd.DataFrame:
    df["individual_id"] = df["cohort"] + "-" + df["local_id"].map(lambda x: f"{int(x):03d}")
    df["species"] = df["cohort"].map(SPECIES)
    df["role"] = df["cohort"].map(ROLE)
    # временная метка особи: date если есть, иначе session (для PW — псевдо-время)
    tkey = df["date"].where(df["date"].notna() & (df["date"] != ""), df["session"].astype(str))
    n_sessions, has_recap = {}, {}
    for ind, grp in df.assign(_tkey=tkey).groupby("individual_id"):
        n = grp["_tkey"].nunique()
        n_sessions[ind] = n
        has_recap[ind] = n >= 2
    df["n_sessions"] = df["individual_id"].map(n_sessions)
    df["has_recapture"] = df["individual_id"].map(has_recap)
    return df


def main():
    (Path(__file__).parent / "artifacts").mkdir(exist_ok=True)
    records = []
    for cohort, fn in ALL_PARSERS.items():
        r = fn()
        print(f"  {cohort}: {len(r)} кадров, {len({x['local_id'] for x in r})} особей")
        records += r
    df = pd.DataFrame(records)
    print(f"\nВсего сырых кадров: {len(df)}")

    df = assign_individual_meta(df)

    # md5-дедуп
    df, dropped = dedup_md5(df)
    print(f"После md5-дедупа: {len(df)} (выброшено {len(dropped)} байт-дублей)")
    if dropped:
        pd.DataFrame(dropped).to_csv(DROPPED, index=False)
        for d in dropped[:8]:
            print(f"    DROP {d['drop']}  ≡  KEEP {d['keep']}")
    # пересчёт recapture после дедупа
    df = assign_individual_meta(df)

    # phash near-duplicate audit
    print("\npHash near-duplicate audit…")
    df, pairs = phash_audit(df.reset_index(drop=True), ROOT, threshold=6)
    real_pairs = [p for p in pairs if "error" not in p]
    if pairs and "error" in pairs[0]:
        print(f"  ⚠️ {pairs[0]['error']}")
    else:
        print(f"  near-dup пар (dist≤6, разный md5): {len(real_pairs)}")
        if real_pairs:
            pd.DataFrame(real_pairs).to_csv(NEARDUP, index=False)
            same = sum(1 for p in real_pairs if p["same_individual"])
            print(f"    из них той же особи: {same} (риск утечки gallery↔probe — разобрать)")

    df["frame_id"] = [f"F{i:05d}" for i in range(len(df))]
    df = df[COLUMNS].sort_values(["cohort", "individual_id", "date", "path_rel"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 60)
    print("СВОДКА РЕЕСТРА")
    print("=" * 60)
    for cohort in ["TK", "PW", "LAB", "GCN"]:
        g = df[df.cohort == cohort]
        recap = g[g.has_recapture].individual_id.nunique()
        print(f"  {cohort:4s}: {len(g):4d} кадров | {g.individual_id.nunique():3d} особей | "
              f"{recap:3d} с перепоимкой | роль={ROLE[cohort]}")
    print(f"\n  ВСЕГО: {len(df)} кадров, {df.individual_id.nunique()} особей")
    tgt = df[df.role == "target"]
    print(f"  Целевые (TK+PW): {tgt.individual_id.nunique()} особей, "
          f"{tgt[tgt.has_recapture].individual_id.nunique()} с перепоимкой (KPI-материал)")
    print(f"\n✅ реестр: {OUT}")


if __name__ == "__main__":
    main()
