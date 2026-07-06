"""Часть 4 — сборка кропов выбранным методом (birefnet_label) → crops/ + crops_manifest.csv.

По умолчанию KPI-когорты (TK+PW+LAB); GCN (--gcn) отдельно (объёмный, для претрейна).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
for _p in ("spike_lab", "registry", "segment"):
    sys.path.insert(0, str(HERE.parent / _p))
sys.path.insert(0, str(HERE))
from common import ROOT                       # noqa: E402
from birefnet import body_mask                # noqa: E402
from label_mask import detect_label, apply_label_mask   # noqa: E402
from crop import tight_crop                   # noqa: E402

METHOD = "birefnet_label"
CROPS = HERE / "crops"


def build(cohorts):
    reg = pd.read_csv(HERE.parent / "registry" / "registry.csv")
    spl = pd.read_csv(HERE.parent / "registry" / "splits.csv")[["frame_id", "split_fold"]]
    sub = reg[reg.cohort.isin(cohorts)].merge(spl, on="frame_id", how="left")
    # R1.1: НЕ материализуем кропы sealed test (вскрытие один раз в Части 11). Чистим уже созданные.
    test_ids = set(sub.loc[sub.split_fold == "test", "frame_id"])
    for fid in test_ids:
        (CROPS / f"{fid}.png").unlink(missing_ok=True)
    sub = sub[sub.split_fold != "test"].reset_index(drop=True)
    print(f"  R1.1: исключён sealed test — {len(test_ids)} кадров не материализуются до Части 11")
    CROPS.mkdir(exist_ok=True)
    rows = []
    n = len(sub)
    for i, r in sub.iterrows():
        img = np.array(Image.open(ROOT / r.path_rel).convert("RGB"))
        m, seg, frac = body_mask(img)
        lbl, has = detect_label(img, m)
        img2, m2 = apply_label_mask(img, m, lbl)
        crop = tight_crop(img2, m2)                         # RGB канон-кроп (gray для SIFT — на лету)
        rel = f"crops/{r.frame_id}.png"
        Image.fromarray(crop).save(HERE / "crops" / f"{r.frame_id}.png")
        rows.append({"frame_id": r.frame_id, "cohort": r.cohort, "crop_path": rel,
                     "seg_method": seg, "mask_area_frac": round(frac, 4), "has_label": bool(has)})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}...")
    man = pd.DataFrame(rows)
    man.to_csv(HERE / "crops_manifest.csv", index=False)
    print(f"\n✅ {len(man)} кропов → crops/ ; манифест crops_manifest.csv")
    print(f"   seg_method: {man.seg_method.value_counts().to_dict()}")
    print(f"   с меткой (исключена): {int(man.has_label.sum())} кадров")


def build_test_only():
    """Часть 11 (вскрытие sealed): материализовать кропы ТОЛЬКО TK-test ТЕМ ЖЕ пайплайном birefnet_label.
    Дописать в crops_manifest.csv. Вызывается РОВНО один раз из seal_open.py после verify sha256."""
    reg = pd.read_csv(HERE.parent / "registry" / "registry.csv")
    spl = pd.read_csv(HERE.parent / "registry" / "splits.csv")[["frame_id", "split_fold"]]
    sub = reg.merge(spl, on="frame_id", how="left")
    sub = sub[sub.split_fold == "test"].reset_index(drop=True)
    CROPS.mkdir(exist_ok=True)
    rows = []
    n = len(sub)
    print(f"  материализация {n} sealed test-кропов (тот же пайплайн birefnet_label)...")
    for i, r in sub.iterrows():
        img = np.array(Image.open(ROOT / r.path_rel).convert("RGB"))
        m, seg, frac = body_mask(img)
        lbl, has = detect_label(img, m)
        img2, m2 = apply_label_mask(img, m, lbl)
        crop = tight_crop(img2, m2)
        Image.fromarray(crop).save(CROPS / f"{r.frame_id}.png")
        rows.append({"frame_id": r.frame_id, "cohort": r.cohort, "crop_path": f"crops/{r.frame_id}.png",
                     "seg_method": seg, "mask_area_frac": round(frac, 4), "has_label": bool(has)})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}...")
    add = pd.DataFrame(rows)
    man_path = HERE / "crops_manifest.csv"
    man = pd.read_csv(man_path)
    man = pd.concat([man[~man.frame_id.isin(add.frame_id)], add], ignore_index=True)   # дописать test
    man.to_csv(man_path, index=False)
    print(f"  ✅ {len(add)} test-кропов материализованы; манифест дополнен ({len(man)} всего)")
    return add


if __name__ == "__main__":
    cohorts = ["GCN"] if "--gcn" in sys.argv else ["TK", "PW", "LAB"]
    print(f"Сборка кропов методом {METHOD}: когорты {cohorts}")
    build(cohorts)
