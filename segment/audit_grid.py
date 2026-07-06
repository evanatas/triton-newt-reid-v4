"""R0.2 — визуальный контактный лист TK-кропов для оценки доли дорсальных (спина) кадров.

Автоклассификатор стороны по цвету ненадёжен (спина/брюшко Карелина похожи) → проверяем гипотезу
ревью «~40% дорсальных» визуально: монтаж кропов в один PNG, смотрим глазами.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CELL = 150
COLS = 8


def main(n=48, seed=1):
    man = pd.read_csv(HERE / "crops_manifest.csv")
    tk = man[man.cohort == "TK"].reset_index(drop=True)
    # репрезентативная выборка: по одному кадру от разных особей где возможно
    reg = pd.read_csv(HERE.parent / "registry" / "registry.csv")[["frame_id", "individual_id"]]
    tk = tk.merge(reg, on="frame_id")
    sample = tk.groupby("individual_id").first().reset_index()
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed)
    sample = sample.sort_values("frame_id").reset_index(drop=True)

    rows = (len(sample) + COLS - 1) // COLS
    grid = np.full((rows * CELL, COLS * CELL, 3), 240, np.uint8)
    for i, r in sample.iterrows():
        c = cv2.imread(str(HERE / r.crop_path))
        c = cv2.resize(c, (CELL - 6, CELL - 18))
        ry, cx = (i // COLS) * CELL, (i % COLS) * CELL
        grid[ry + 16:ry + CELL - 2, cx + 3:cx + CELL - 3] = c
        cv2.putText(grid, r.frame_id, (cx + 4, ry + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    out = HERE / "artifacts" / "tk_audit_grid.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"✅ {len(sample)} TK-кропов (по особям) → {out}")


if __name__ == "__main__":
    main()
