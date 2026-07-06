"""Часть 7 / задача #22 — тюнинг ядра (SIFT+affine) КРУПНЫМИ шагами на dev single-ref.

dev мал (n=97 known) → грид намеренно грубый, защита от переобучения на шум. Адопция нового дефолта ТОЛЬКО при
ЯСНОМ приросте над текущим (1500/2.0/5): CI-разделение ИЛИ McNemar p<0.05. Иначе держим текущие параметры.

Оптимизация: матчинг SIFT (дорого) считаем раз на (nfeatures,clip), кэшируем точки; порог affine (дёшево) гоняем
поверх кэша. Всё на DEV (test ЗАПЕЧАТАН). Запуск: python tune.py
"""
from __future__ import annotations
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
for _p in ("..", "../segment", "../registry", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
from sift_eval import recall_from_scores          # noqa: E402
from embed_eval import wilson, KS                 # noqa: E402
from sealed import assert_unsealed                # noqa: E402
import matchers as M                              # noqa: E402
import deform as D                                # noqa: E402
from ab_matchers import load_pool, mcnemar        # noqa: E402

CROPS = (HERE / ".." / "segment").resolve()
ART = HERE / "artifacts"
DEFORM = "affine"

NFEAT = [1000, 1500, 2000]      # крупные шаги
CLIP = [2.0, 3.0]
THR = [4.0, 5.0, 6.0]
DEFAULT = (1500, 2.0, 5.0)      # текущее ядро — эталон адопции


def make_extract(nfeatures, clip):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    clahe = cv2.createCLAHE(clip, (8, 8))

    def extract(rgb):
        g = clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        kp, des = sift.detectAndCompute(g, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return {"pts": pts, "des": des}
    return extract


def main():
    assert_unsealed("dev", unseal=False)
    gallery, probe = load_pool()
    g_ids, p_ids = gallery.individual_id.to_numpy(), probe.individual_id.to_numpy()

    def imgs(rows):
        return [cv2.cvtColor(cv2.imread(str(CROPS / r.crop_path)), cv2.COLOR_BGR2RGB) for _, r in rows.iterrows()]
    g_imgs, p_imgs = imgs(gallery), imgs(probe)
    P, G = len(p_imgs), len(g_imgs)
    sm = M.SiftMatcher()   # только .matched (Lowe 0.75) — не зависит от параметров экстракта
    print(f"тюнинг на DEV single-ref: gallery {G} | probe {P} | grid {len(NFEAT)*len(CLIP)*len(THR)} конфигов | test ЗАПЕЧАТАН\n")

    rows = {}; hits1 = {}
    print(f"{'config (nfeat/clip/thr)':<26}{'R@1':>16}{'R@5':>8}{'R@10':>7}")
    for nf, clip in itertools.product(NFEAT, CLIP):
        ext = make_extract(nf, clip)
        t0 = time.time()
        fg = [ext(im) for im in g_imgs]; fp = [ext(im) for im in p_imgs]
        cache = [[sm.matched(fp[i], fg[j]) for j in range(G)] for i in range(P)]   # матчинг раз
        for thr in THR:
            S = np.array([[D.verify(cache[i][j][0], cache[i][j][1], DEFORM, thr) for j in range(G)]
                          for i in range(P)], np.float32)
            hits = recall_from_scores(S, p_ids, g_ids)
            key = (nf, clip, thr)
            r = {}
            for k in KS:
                v, lo, hi = wilson(int(hits[k].sum()), P)
                r[f"recall@{k}"] = {"value": round(v, 3), "ci": [round(lo, 2), round(hi, 2)]}
            rows[str(key)] = r; hits1[str(key)] = hits[1]
            tag = "  ← текущее ядро" if key == DEFAULT else ""
            print(f"{f'{nf}/{clip}/{thr}':<26}{r['recall@1']['value']:>10.3f}"
                  f"[{r['recall@1']['ci'][0]:.2f}-{r['recall@1']['ci'][1]:.2f}]"
                  f"{r['recall@5']['value']:>8.3f}{r['recall@10']['value']:>7.3f}{tag}")
        print(f"    ({nf}/{clip}: матчинг+скоринг {time.time()-t0:.0f}s)")

    # выбор и адопция
    dkey = str(DEFAULT)
    dv = rows[dkey]["recall@1"]["value"]
    best = max(rows, key=lambda k: rows[k]["recall@1"]["value"])
    bv = rows[best]["recall@1"]["value"]
    b, c, p = mcnemar(hits1[dkey], hits1[best])
    default_ci = rows[dkey]["recall@1"]["ci"]; best_ci = rows[best]["recall@1"]["ci"]
    ci_separated = best_ci[0] > default_ci[1]
    adopt = (best != dkey) and (bv > dv) and (ci_separated or p < 0.05)
    decision = {
        "default": {"config": DEFAULT, "recall@1": dv},
        "best": {"config": best, "recall@1": bv},
        "mcnemar_best_vs_default@1": {"b": b, "c": c, "p": round(p, 4)},
        "multiplicity_note_F11": "18 конфигов (3x2x3), поправка Holm/Bonferroni НЕ применялась; решение консервативно "
                                 "(adopt только при p<0.05 ИЛИ CI-разделении → дефолт держится, ничего лишнего не принято)",
        "ci_separated": bool(ci_separated),
        "adopt_best": bool(adopt),
        "frozen_config": list(eval(best)) if adopt else list(DEFAULT),
        "rationale": (f"best {best} R@1 {bv} > default {dv}, McNemar p={p:.3f}/CI-sep={ci_separated} → принять"
                      if adopt else
                      f"лучший {best} ({bv}) не превосходит текущий {dv} значимо (McNemar p={p:.3f}, "
                      f"CI-sep={ci_separated}) → держим текущее ядро {DEFAULT} (защита от переобучения dev)"),
    }
    ART.mkdir(exist_ok=True)
    (ART / "tune_dev.json").write_text(json.dumps({"grid": rows, "decision": decision}, ensure_ascii=False, indent=2))
    print("\n" + "=" * 64)
    print(f"ЗАМОРОЗИТЬ параметры: {decision['frozen_config']}")
    print(decision["rationale"])
    print("=" * 64)
    print("✅ artifacts/tune_dev.json (test не вскрывался)")


if __name__ == "__main__":
    main()
