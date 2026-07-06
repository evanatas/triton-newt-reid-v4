"""R0 — классификатор стороны тела (вентраль / дорсаль / бок) для TK-кропов.

Гипотеза ревью: часть TK снята со СПИНЫ (дорсальная жёлтая полоса), узор брюшка не виден → засоряет KPI.
Сигнатура: на ВЕНТРАЛИ жёлто-оранжевое распределено по ВСЕЙ ширине тела (брюшко); на ДОРСАЛИ — осевая
жёлтая полоса по хребту, бока тёмные. Меряем долю жёлтого в центре оси vs на краях (поперёк главной оси PCA).
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FILL = 127            # нейтральная заливка фона в кропе


def side_signature(crop_rgb: np.ndarray) -> dict:
    """-> {yellow_frac, ventral_score, side}. ventral_score = yellow_край / yellow_центр (поперёк оси тела)."""
    body = ~np.all(np.abs(crop_rgb.astype(int) - FILL) < 6, axis=2)   # тело = не нейтральный фон
    if body.sum() < 500:
        return {"yellow_frac": 0.0, "ventral_score": 0.0, "side": "unknown", "axis": "?"}
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    yellow = (cv2.inRange(hsv, (10, 55, 45), (45, 255, 255)) > 0) & body
    yellow_frac = yellow.sum() / body.sum()

    ys, xs = np.where(body)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    minor = evecs[:, 0]                      # поперёк тела (наименьшая дисперсия)
    proj = (pts - mean) @ minor              # координата поперёк оси
    pmax = np.abs(proj).max() + 1e-6
    t = np.abs(proj) / pmax                  # 0=центр оси, 1=край (бок)

    yv = yellow[ys, xs]
    center = t < 0.40
    edge = t > 0.62
    yc = yv[center].mean() if center.sum() else 0.0     # доля жёлтого вдоль центральной оси
    ye = yv[edge].mean() if edge.sum() else 0.0         # доля жёлтого на боках
    ventral = ye / (yc + 1e-6)                          # вентраль→≈1+ (бока жёлтые); дорсаль→≪1 (бока тёмные)

    # решение: вентраль = бока тоже жёлтые И достаточно жёлтого всего
    if yellow_frac < 0.06:
        side = "dorsal"                                  # почти нет жёлтого = спина/молодь
    elif ventral >= 0.55:
        side = "ventral"
    else:
        side = "dorsal"                                  # жёлтое полосой по центру, бока тёмные
    return {"yellow_frac": round(float(yellow_frac), 3), "ventral_score": round(float(ventral), 3), "side": side}


def main():
    man = pd.read_csv(HERE / "crops_manifest.csv")
    tk = man[man.cohort == "TK"].reset_index(drop=True)
    rows = []
    for _, r in tk.iterrows():
        crop = cv2.cvtColor(cv2.imread(str(HERE / r.crop_path)), cv2.COLOR_BGR2RGB)
        sig = side_signature(crop)
        rows.append({"frame_id": r.frame_id, **sig})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "tk_side.csv", index=False)

    print("=== эталоны (калибровка) ===")
    for fid, exp in [("F00268", "ventral"), ("F00270", "dorsal"), ("F00060", "dorsal")]:
        row = out[out.frame_id == fid]
        if len(row):
            r = row.iloc[0]
            ok = "✅" if r.side == exp else "❌"
            print(f"  {fid}: side={r.side} (ожид. {exp}) {ok}  yellow_frac={r.yellow_frac} ventral_score={r.ventral_score}")

    n = len(out)
    nd = (out.side == "dorsal").sum()
    print(f"\n=== TK ({n} кропов, без sealed test) ===")
    print(f"  вентраль: {(out.side=='ventral').sum()} ({(out.side=='ventral').mean()*100:.0f}%)")
    print(f"  дорсаль/бок: {nd} ({nd/n*100:.0f}%)   ← ревью заявлял ~40%")
    print(f"  → tk_side.csv")


if __name__ == "__main__":
    main()
