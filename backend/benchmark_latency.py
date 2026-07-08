"""Latency-бенчмарк пайплайна re-id тритонов (НФТ-2: ≤2 с/фото).

Ядро НЕ трогается — только замер времени вызовов замороженного пайплайна:
  сегментация (BiRefNet → маска метки → tight_crop) · SIFT-extract · матч+affine-verify на 1 реф ·
  полный запрос identify по каталогу (276 кадров).
Вход — готовые фото (демо_фото/01_известные, иначе examples/). Результат → core/artifacts/latency_bench.json.
Запуск: cd "triton 4.0" && PYTORCH_ENABLE_MPS_FALLBACK=1 /opt/anaconda3/bin/python backend/benchmark_latency.py
"""
from __future__ import annotations
import sys, time, json, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))            # backend/  (reid_service добавит segment/core/registry в path)
import reid_service as RS                 # noqa: E402
import deform as D                        # noqa: E402  (доступен после импорта reid_service)


def stat(a):
    a = list(a)
    return {"mean": round(float(np.mean(a)), 1), "median": round(float(np.median(a)), 1),
            "p90": round(float(np.percentile(a, 90)), 1)} if a else {}


def main():
    try:
        import torch
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"

    svc = RS.ReIDService("TK")
    n_refs = len(svc.cat)

    src = glob.glob(str(ROOT / "демо_фото" / "01_известные" / "*")) or glob.glob(str(ROOT / "examples" / "*"))
    raws = [r for r in sorted(src) if r.lower().endswith((".jpg", ".jpeg", ".png"))][:30]
    if not raws:
        raise SystemExit("нет фото в демо_фото/01_известные/ или examples/")

    # разогрев (загрузка весов BiRefNet — не входит в статистику)
    with open(raws[0], "rb") as f:
        _ = svc.segment_upload(f.read())

    seg_ms, ext_ms, idc_ms, full_ms = [], [], [], []
    for rp in raws:
        with open(rp, "rb") as f:
            b = f.read()
        t0 = time.perf_counter(); crop, _meta = svc.segment_upload(b); t1 = time.perf_counter()
        fp = svc.sift.extract(crop);                                    t2 = time.perf_counter()
        _out = svc.identify_crop(crop);                                 t3 = time.perf_counter()
        seg_ms.append((t1 - t0) * 1000); ext_ms.append((t2 - t1) * 1000)
        idc_ms.append((t3 - t2) * 1000); full_ms.append((t3 - t0) * 1000)

    # чистый замер одного матча (SIFT knn + Lowe + affine-RANSAC) на паре кропов
    fid0 = svc.cat.iloc[0].frame_id
    fp0 = svc.feats[fid0]
    match_ms = []
    for _, r in svc.cat.head(80).iterrows():
        t0 = time.perf_counter()
        pa, pb = svc.sift.matched(fp0, svc.feats[r.frame_id]); D.verify(pa, pb, RS.DEFORM)
        match_ms.append((time.perf_counter() - t0) * 1000)

    res = {
        "device": device,
        "n_refs_catalog": n_refs,
        "n_photos_benched": len(raws),
        "segmentation_birefnet_ms": stat(seg_ms),
        "sift_extract_ms": stat(ext_ms),
        "match_verify_per_ref_ms": stat(match_ms),
        "identify_vs_catalog_ms": stat(idc_ms),
        "full_query_ms": stat(full_ms),
        "note": ("full_query = сегментация + SIFT-extract + матч по всему каталогу + оверлеи top-5; "
                 "матчинг линеен по размеру каталога (match_verify_per_ref × n_refs)."),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    outp = ROOT / "core" / "artifacts" / "latency_bench.json"
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved:", outp)


if __name__ == "__main__":
    main()
