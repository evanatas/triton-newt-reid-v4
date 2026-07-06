"""Часть 10 — global-эмбеддинг БЕЙЗЛАЙНЫ на TK dev тем же протоколом, что ядро (eval_core).

Очная ставка «local-feature (ядро SIFT+affine+rolling) vs global-embedding». Бейзлайны на ТЕХ ЖЕ birefnet_label-кропах
(crop="none"), что и ядро → единственная разница = матчер (apples-to-apples; global в ЛУЧШЕМ виде, не crippled-yellow 3.0).
Протокол идентичен: eval_core.eval_mode (single+rolling, identity-disjoint, gallery/probe разные сессии).

Анти-fishing: все методы фиксируются ДО вскрытия sealed; на тесте (Часть 11) репортятся ВСЕ. sealed НЕ вскрывается.
Запуск: python baselines.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
for _p in ("..", "../segment", "../registry", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
from embed_eval import get_megadescriptor, get_dinov2, embed_paths, wilson, KS   # noqa: E402
from sealed import assert_unsealed                                              # noqa: E402
from metrics import report                                                      # noqa: E402
from ab_matchers import mcnemar                                                 # noqa: E402
import matchers as M                                                           # noqa: E402
import deform as D                                                             # noqa: E402
import eval_core as EC                                                         # noqa: E402

CROPS = EC.CROPS
ART = HERE / "artifacts"
DEFORM = "affine"
EMBEDDERS = {"megadescriptor": get_megadescriptor, "dinov2": get_dinov2}


def embed_feats(model_name, rows, fold):
    """frame_id -> L2-норм эмбеддинг (cosine=dot). Кэш в artifacts/emb_{model}_{fold}.npz."""
    cache = ART / f"emb_{model_name}_{fold}.npz"
    fids = [r.frame_id for _, r in rows.iterrows()]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(zip(z["fids"].tolist(), z["embs"]))
    model, tf = EMBEDDERS[model_name]()
    paths = [str(CROPS / r.crop_path) for _, r in rows.iterrows()]
    embs = embed_paths(paths, model, tf, crop="none")            # (N,D), L2-норм; birefnet-кроп как есть
    np.savez(cache, fids=np.array(fids), embs=embs)
    return dict(zip(fids, embs))


def run_method(score_fn, feats, known_probe, refs_all, gallery_single):
    out = {}; hits1 = {}
    p_ids = known_probe.individual_id.to_numpy()
    for mode in ["single", "rolling"]:
        hits, _, bins = EC.eval_mode(mode, score_fn, feats, known_probe, refs_all, gallery_single)
        rep = report(hits, p_ids)
        rep["wilson@1"] = [round(x, 3) for x in wilson(int(hits[1].sum()), len(hits[1]))]
        rep["by_interval"] = EC.by_interval(hits, bins)
        out[mode] = rep; hits1[mode] = hits[1]
    return out, hits1


def main(fold="dev", unseal=False):
    assert_unsealed(fold, unseal)
    refs_all, gallery_single, known_probe, new_probe = EC.load_cohort("TK", fold)
    all_rows = pd.concat([refs_all, known_probe, new_probe]).drop_duplicates("frame_id")
    print(f"бейзлайны на TK {fold}: refs {len(refs_all)} | probe known {len(known_probe)} | "
          f"random top-1 {1/refs_all.individual_id.nunique():.3f}\n")

    # ЯДРО (SIFT+affine) — для McNemar-эталона, тот же probe-порядок
    sift = M.build("sift")
    sift_score = lambda a, b: D.verify(*sift.matched(a, b), DEFORM)
    sift_feats = {r.frame_id: sift.extract(cv2.cvtColor(cv2.imread(str(CROPS / r.crop_path)), cv2.COLOR_BGR2RGB))
                  for _, r in all_rows.iterrows()}
    core_out, core_hits = run_method(sift_score, sift_feats, known_probe, refs_all, gallery_single)

    results = {"core_sift_affine": core_out}

    def save():
        out = {"cohort": "TK", "protocol": "identity-disjoint, gallery/probe разные сессии, crop=birefnet_label(none)",
               "random_top1": round(1 / refs_all.individual_id.nunique(), 3),
               "note": "global-бейзлайны на тех же кропах, что ядро; McNemar c>b => ядро бьёт бейзлайн",
               "methods": results}
        ART.mkdir(exist_ok=True)
        (ART / f"baselines_{fold}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    save()   # core записан сразу — устойчивость к зависанию загрузки бейзлайна
    print(f"{'метод':<22}{'single R@1':>14}{'rolling R@1':>14}{'macro@1':>10}  McNemar(b,c,p) vs ядро", flush=True)
    print(f"{'core SIFT+affine':<22}{core_out['single']['micro_recall'][1]:>14.3f}"
          f"{core_out['rolling']['micro_recall'][1]:>14.3f}{core_out['rolling']['macro_recall'][1]:>10.3f}  (эталон)", flush=True)

    for mname in ["megadescriptor", "dinov2"]:
        try:
            feats = embed_feats(mname, all_rows, fold)
            cos = lambda a, b: float(np.dot(a, b))
            res, hits = run_method(cos, feats, known_probe, refs_all, gallery_single)
            for mode in ["single", "rolling"]:
                b, c, p = mcnemar(core_hits[mode], hits[mode])   # b=ядро-miss&base-hit, c=ядро-hit&base-miss
                res[mode]["mcnemar_vs_core@1"] = {"b": b, "c": c, "p": round(p, 4)}
            results[mname] = res
            save()
            mc = res["rolling"]["mcnemar_vs_core@1"]
            print(f"{mname:<22}{res['single']['micro_recall'][1]:>14.3f}{res['rolling']['micro_recall'][1]:>14.3f}"
                  f"{res['rolling']['macro_recall'][1]:>10.3f}  b={mc['b']} c={mc['c']} p={mc['p']:.3f}", flush=True)
        except Exception as e:
            results[mname] = {"skipped": f"{type(e).__name__}: {e}"}
            save()
            print(f"{mname:<22}  ПРОПУЩЕН: {e}", flush=True)

    print(f"\n✅ artifacts/baselines_{fold}.json", flush=True)


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    main(a.fold, a.unseal)
