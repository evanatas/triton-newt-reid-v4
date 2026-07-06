"""Часть 3 — оркестрация сплитов: registry.csv → splits.csv + sealed_manifest.json."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from splits import build, SEED
from sealed import seal

HERE = Path(__file__).parent
OUT = HERE / "splits.csv"
COLS = ["frame_id", "cohort", "individual_id", "date", "session", "split_role",
        "split_fold", "interval_months", "is_open_new", "in_train_pool"]


def main():
    reg = pd.read_csv(HERE / "registry.csv")
    df = build(reg, test_frac=0.5, open_frac=0.15, seed=SEED)
    out = df[COLS].copy()
    out.to_csv(OUT, index=False)

    print("=" * 64)
    print(f"СПЛИТЫ (seed={SEED}, test_frac=0.5, open_frac=0.15)")
    print("=" * 64)

    # сводка по fold × роль (только TK eval)
    tk = df[df.cohort == "TK"]
    print("\nTK по fold (особей / кадров):")
    for f in ["dev", "test", "distractor"]:
        g = tk[tk.split_fold == f]
        print(f"  {f:11s}: {g.individual_id.nunique():3d} особей | {len(g):3d} кадров")

    print("\nTK eval-роли (dev+test):")
    ev = tk[tk.split_fold.isin(["dev", "test"])]
    for f in ["dev", "test"]:
        g = ev[ev.split_fold == f]
        gal = g[g.split_role == "gallery"]
        prb = g[g.split_role == "probe"]
        opn = g[g.is_open_new]
        print(f"  {f}: gallery {len(gal)} кадров / {gal.individual_id.nunique()} особей | "
              f"probe {len(prb)} | open_new особей {opn.individual_id.nunique()}")

    print("\nProbe по интервалу (dev/test):")
    for f in ["dev", "test"]:
        g = ev[(ev.split_fold == f) & (ev.split_role == "probe")]
        bins = g.interval_months.value_counts().sort_index().to_dict()
        print(f"  {f}: {bins}")

    print("\nTrain-pool (для обучаемых, disjoint от TK-test):")
    tp = df[df.in_train_pool]
    print(f"  {len(tp)} кадров, {tp.individual_id.nunique()} особей; по когортам: "
          f"{tp.cohort.value_counts().to_dict()}")

    # disjoint-проверка (быстрая)
    test_ids = set(tk[tk.split_fold == "test"].individual_id)
    train_ids = set(tp.individual_id)
    leak = test_ids & train_ids
    print(f"\n  TK-test особей: {len(test_ids)} | train∩test: {len(leak)} {'✅' if not leak else '❌ УТЕЧКА'}")

    # запечатать test + open_test
    manifest = seal(df)
    print(f"\n🔒 sealed: {manifest['n_sealed']} кадров запечатано → sealed_manifest.json")
    print(f"\n✅ сплиты: {OUT}")


if __name__ == "__main__":
    main()
