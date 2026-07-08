#!/usr/bin/env python3
"""Пересчёт шкалы «уверенности» демо → backend/calib.json.
Калибрует НА ТОЙ ЖЕ величине, что показывается в карточке: MAX-POOL скор особи под демо-протоколом
(сессия запроса исключена). lo=P90(impostor max-pool), hi=P80(genuine max-pool). Иначе (попарные пары)
hi занижен и «уверенность» липнет к потолку 99 % — см. QA-ревью. НЕ влияет на ранг/KPI 0.79 (только %).
Запуск: cd 'triton 4.0' && /opt/anaconda3/bin/python backend/build_calibration.py"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reid_service import ReIDService

def main():
    svc = ReIDService()
    cat = svc.cat
    gen, imp = [], []
    for iid, g in cat.groupby("individual_id"):
        dates = sorted(d for d in g.date.dropna().unique())
        if len(dates) < 2:                                  # нужна перепоимка (межсессионный матч)
            continue
        fr = g[g.date == dates[-1]].iloc[0].frame_id        # поздняя сессия как запрос
        ranked = svc.rank(svc.feats[fr], 5, exclude_frame=svc.session_of(fr))
        if not ranked:
            continue
        tru = [r for r in ranked if r["individual_id"] == iid]
        wrong = [r for r in ranked if r["individual_id"] != iid]
        if tru:
            gen.append(tru[0]["score"])                     # max-pool скор истинной особи
        if wrong:
            imp.append(wrong[0]["score"])                   # max-pool лучшей чужой
    gen, imp = np.array(gen), np.array(imp)
    lo = float(round(np.percentile(imp, 90)))
    hi = float(round(np.percentile(gen, 80)))
    calib = {"lo": lo, "hi": hi, "method": "maxpool_temporal_protocol",
             "n_genuine": int(len(gen)), "n_impostor": int(len(imp)),
             "note": "lo=P90(impostor max-pool), hi=P80(genuine max-pool) под демо-протоколом (сессия запроса исключена)"}
    with open(os.path.join(HERE, "calib.json"), "w") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print("backend/calib.json:", calib)

if __name__ == "__main__":
    main()
