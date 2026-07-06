"""Часть 3 — гейты честности сплита G-S1…G-S7."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from splits import build
from sealed import verify_manifest

HERE = Path(__file__).parent


def validate() -> list[tuple[str, bool, str]]:
    spl = pd.read_csv(HERE / "splits.csv")          # уже содержит date, session, cohort, individual_id
    reg = pd.read_csv(HERE / "registry.csv")
    df = spl.merge(reg[["frame_id", "md5"]], on="frame_id", how="left")
    res = []

    # G-S1 каждая особь ровно в одном fold
    fpi = spl.groupby("individual_id")["split_fold"].nunique()
    res.append(("G-S1 особь в одном fold", (fpi == 1).all(), f"нарушений: {(fpi != 1).sum()}"))

    # G-S2 train-pool ∩ TK-test по individual_id = ∅
    train_ids = set(spl[spl.in_train_pool].individual_id)
    test_ids = set(spl[spl.split_fold == "test"].individual_id)
    res.append(("G-S2 train ∩ test = ∅", len(train_ids & test_ids) == 0, f"пересечений: {len(train_ids & test_ids)}"))

    # G-S3 dev ∩ test по individual_id = ∅
    dev_ids = set(spl[spl.split_fold == "dev"].individual_id)
    res.append(("G-S3 dev ∩ test = ∅", len(dev_ids & test_ids) == 0, f"пересечений: {len(dev_ids & test_ids)}"))

    # G-S4 gallery и probe одной особи — разные сессии (нет одной фотографии в обеих ролях)
    bad = 0
    for ind, g in df[df.split_role.isin(["gallery", "probe"])].groupby("individual_id"):
        gs = set(g[g.split_role == "gallery"]["date"].fillna(g["session"].astype(str)))
        ps = set(g[g.split_role == "probe"]["date"].fillna(g["session"].astype(str)))
        if gs & ps:
            bad += 1
    res.append(("G-S4 gallery/probe разные сессии", bad == 0, f"особей с пересечением сессий: {bad}"))

    # G-S5 open-new особи не в галерее (их gallery → open_excluded)
    onv = spl[spl.is_open_new]
    in_gal = (onv.split_role == "gallery").sum()
    res.append(("G-S5 open-new нет в галерее", in_gal == 0, f"open-new в gallery: {in_gal}"))

    # G-S6 детерминизм: повторный build с тем же seed = тот же сплит
    df2 = build(reg, test_frac=0.5, open_frac=0.15, seed=42)
    same = (df2.sort_values("frame_id")["split_fold"].values == spl.sort_values("frame_id")["split_fold"].values).all()
    res.append(("G-S6 детерминизм (seed=42)", bool(same), ""))

    # G-S7 sealed-манифест согласован с текущим test-fold
    ok7 = verify_manifest(spl.merge(reg[["frame_id", "md5"]], on="frame_id"))
    res.append(("G-S7 sealed-манифест согласован", bool(ok7), ""))

    return res


def main():
    print("ГЕЙТЫ ЧЕСТНОСТИ СПЛИТА")
    print("=" * 52)
    all_ok = True
    for name, ok, detail in validate():
        all_ok &= ok
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  ({detail})" if detail else ""))
    print("=" * 52)
    print("ИТОГ:", "✅ все гейты пройдены" if all_ok else "❌ ЕСТЬ НАРУШЕНИЯ")
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
