"""R0.5 — влияет ли «чистота кадра» на dev-метрику? (test ЗАПЕЧАТАН)

Точной разметки стороны нет (авто-классификатор по цвету не разделяет спину/брюшко Карелина надёжно).
Грубый прокси: yellow_frac (доля жёлтого в теле) — у явной спины/молоди ниже. Исключаем кадры ниже порога
и смотрим, растёт ли SIFT-recall. Это НЕ доказательство, а индикатор: стоит ли добиваться экспертной разметки.
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
from sift_eval import score_matrix, recall_from_scores   # noqa: E402
from embed_eval import wilson, KS             # noqa: E402
from sealed import assert_unsealed            # noqa: E402
from birefnet import body_mask                # noqa: E402
from ab_crop import _sift_feats, _crop_from_cache   # noqa: E402


def main():
    assert_unsealed("dev", unseal=False)
    spl = pd.read_csv(HERE.parent / "registry" / "splits.csv")
    reg = pd.read_csv(HERE.parent / "registry" / "registry.csv")
    side = pd.read_csv(HERE / "tk_side.csv")[["frame_id", "yellow_frac"]]
    df = spl.merge(reg[["frame_id", "path_rel"]], on="frame_id").merge(side, on="frame_id", how="left")
    tk = df[df.cohort == "TK"]
    gallery = tk[((tk.split_fold == "dev") & (tk.split_role == "gallery") & (~tk.is_open_new))
                 | (tk.split_fold == "distractor")].reset_index(drop=True)
    probe = tk[(tk.split_fold == "dev") & (tk.split_role == "probe") & (~tk.is_open_new)].reset_index(drop=True)
    pool = pd.concat([gallery, probe], ignore_index=True)
    gal_idx = list(range(len(gallery))); prb_idx = list(range(len(gallery), len(pool)))

    # birefnet_label кропы (победитель Части 4) → SIFT (1 раз на весь pool)
    grays = []
    for p in pool.path_rel:
        img = np.array(Image.open(ROOT / p).convert("RGB"))
        m, _, _ = body_mask(img)
        g, _ = _crop_from_cache(img, m, "birefnet_label")
        grays.append(g)
    S = score_matrix(_sift_feats(grays), prb_idx, gal_idx)
    g_yf = gallery.yellow_frac.to_numpy()
    p_yf = probe.yellow_frac.to_numpy()
    g_ids = gallery.individual_id.to_numpy(); p_ids = probe.individual_id.to_numpy()

    print(f"R0.5 dev (birefnet_label SIFT): gallery {len(gallery)} | probe {len(probe)} | test ЗАПЕЧАТАН")
    print(f"yellow_frac probe: медиана {np.nanmedian(p_yf):.2f}, мин {np.nanmin(p_yf):.2f}")
    print(f"\n{'фильтр yellow_frac':<22}{'n_probe':>8}  " + "  ".join(f"R@{k}" for k in KS))
    for thr in [0.0, 0.5, 0.6, 0.7, 0.8]:
        mp = (p_yf >= thr) | np.isnan(p_yf)
        mg = (g_yf >= thr) | np.isnan(g_yf)
        if mp.sum() < 5 or mg.sum() < 5:
            continue
        Ssub = S[np.ix_(mp, mg)]
        hits = recall_from_scores(Ssub, p_ids[mp], g_ids[mg])
        cells = []
        for k in KS:
            v, lo, hi = wilson(int(hits[k].sum()), int(mp.sum()))
            cells.append(f"{v:.3f}[{lo:.2f}-{hi:.2f}]")
        tag = "все (baseline)" if thr == 0 else f"≥{thr}"
        print(f"{tag:<22}{int(mp.sum()):>8}  " + "  ".join(cells))
    print("\n(Рост R@1 с порогом = «грязные» кадры тянут метрику вниз → экспертная разметка стороны оправдана.)")


if __name__ == "__main__":
    main()
