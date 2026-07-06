"""
Часть 6 (спайк LAB) — под-шаг 2-3: zero-shot эмбеддинги + cross-session recall.

Изолированный код (без triton 3.0). Грузит LAB temporal-бенчмарк, считает эмбеддинги
указанной моделью, меряет identity-level recall@k на перепоимках с разбивкой по интервалу.

Препроцессинг един для gallery и probe (урок H1 из 3.0: разный препроцессинг врёт ±0.20).
Identity-level: фото галереи агрегируются до особи (max cosine), ранжируются особи.

Запуск:  python embed_eval.py --model megadescriptor [--crop none|center]
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

HERE = Path(__file__).parent
BENCH = HERE / "data" / "lab_benchmark.csv"
ART = HERE / "artifacts"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KS = (1, 5, 10)


# ---------- метрики ----------
def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def identity_recall(probe_emb, probe_ids, gallery_emb, gallery_ids, ks=KS):
    """Identity-level CMC: агрегируем галерею до особи (max cosine), ранжируем особей.
    Возврат: {k: bool-массив hit по пробам}, uniq_ids."""
    uniq = np.array(sorted(set(gallery_ids)))
    sim = probe_emb @ gallery_emb.T                      # (P, G), эмбеддинги L2-норм
    # агрегация по особи (max)
    agg = np.full((probe_emb.shape[0], len(uniq)), -np.inf)
    for j, gid in enumerate(uniq):
        cols = np.where(gallery_ids == gid)[0]
        agg[:, j] = sim[:, cols].max(axis=1)
    order = np.argsort(-agg, axis=1)                     # ранжирование особей
    ranked_ids = uniq[order]
    hits = {}
    for k in ks:
        topk = ranked_ids[:, :k]
        hits[k] = np.array([probe_ids[i] in topk[i] for i in range(len(probe_ids))])
    return hits


# ---------- эмбеддеры ----------
def get_megadescriptor():
    import timm
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
    model = model.eval().to(DEVICE)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


def get_dinov2():
    import timm
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    model = model.eval().to(DEVICE)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


EMBEDDERS = {"megadescriptor": get_megadescriptor, "dinov2": get_dinov2}


def center_crop_pil(img: Image.Image, frac: float = 0.7) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def yellow_crop_pil(img: Image.Image, pad: float = 0.18) -> Image.Image:
    """Грубая сегментация тритона по цвету: жёлто-оранжевое тело на бежево-белом фоне.
    Для спайка (не продакшн): HSV-маска -> крупнейший компонент -> bbox + поля.
    Fallback на center-crop, если тело не найдено."""
    import cv2
    rgb = np.array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # жёлто-оранжевый: H 12..45 (из 180), S>60, V>50; узор включает и тёмные пятна — добираем их dilate
    ymask = cv2.inRange(hsv, (12, 60, 50), (45, 255, 255))
    ymask = cv2.morphologyEx(ymask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ymask, connectivity=8)
    if n <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < 0.01 * rgb.shape[0] * rgb.shape[1]:
        return center_crop_pil(img, 0.6)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[big, cv2.CC_STAT_LEFT], stats[big, cv2.CC_STAT_TOP], stats[big, cv2.CC_STAT_WIDTH], stats[big, cv2.CC_STAT_HEIGHT]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(rgb.shape[1], x + w + px), min(rgb.shape[0], y + h + py)
    return img.crop((x0, y0, x1, y1))


CROPPERS = {"none": lambda im: im, "center": lambda im: center_crop_pil(im, 0.7), "yellow": yellow_crop_pil}


@torch.no_grad()
def embed_paths(paths, model, transform, crop="none", batch_size=16) -> np.ndarray:
    embs = []
    batch = []
    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(DEVICE)
        feat = model(x)
        feat = torch.nn.functional.normalize(feat, dim=1)
        embs.append(feat.cpu().numpy())
        batch.clear()
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = CROPPERS[crop](img)
        batch.append(transform(img))
        if len(batch) >= batch_size:
            flush()
    flush()
    return np.concatenate(embs, axis=0).astype(np.float32)


# ---------- прогон ----------
def run(model_name: str, crop: str):
    df = pd.read_csv(BENCH)
    gal = df[df.role == "gallery"].reset_index(drop=True)
    prb = df[df.role == "probe"].reset_index(drop=True)
    print(f"Модель: {model_name} | crop={crop} | device={DEVICE}")
    print(f"Gallery: {len(gal)} (по особи) | Probe: {len(prb)} перепоимок | особей: {gal.individual_id.nunique()}")

    builder = EMBEDDERS[model_name]
    print("Загрузка модели (первый раз — скачивание весов с HuggingFace)…")
    model, transform = builder()

    ART.mkdir(exist_ok=True)
    cache = ART / f"emb_{model_name}_{crop}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        g_emb, p_emb = z["g_emb"], z["p_emb"]
        print(f"эмбеддинги из кэша: {cache.name}")
    else:
        print("эмбеддинг gallery…"); g_emb = embed_paths(gal.abs_path.tolist(), model, transform, crop)
        print("эмбеддинг probe…");   p_emb = embed_paths(prb.abs_path.tolist(), model, transform, crop)
        np.savez(cache, g_emb=g_emb, p_emb=p_emb)
        print(f"эмбеддинги сохранены: {cache.name} (dim={g_emb.shape[1]})")

    g_ids = gal.individual_id.to_numpy()
    p_ids = prb.individual_id.to_numpy()
    p_bins = prb.interval_bin.to_numpy()

    hits = identity_recall(p_emb, p_ids, g_emb, g_ids)
    n = len(p_ids)
    rnd = 1 / gal.individual_id.nunique()

    print("\n" + "=" * 64)
    print(f"РЕЗУЛЬТАТ — {model_name} (crop={crop}), n={n} перепоимок")
    print("=" * 64)
    print(f"{'срез':<14}{'n':>5}  " + "  ".join(f"R@{k:<2}[95% CI]" for k in KS))
    res = {"model": model_name, "crop": crop, "n": int(n), "random_top1": round(rnd, 3), "overall": {}, "by_interval": {}}
    # overall
    cells = []
    for k in KS:
        p, lo, hi = wilson(int(hits[k].sum()), n)
        cells.append(f"{p:.3f}[{lo:.2f}-{hi:.2f}]")
        res["overall"][f"recall@{k}"] = {"value": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "hits": int(hits[k].sum())}
    print(f"{'ВСЕ':<14}{n:>5}  " + "  ".join(cells))
    # по интервалам
    for b in ("~1мес", "2-3мес", "4-5мес"):
        m = p_bins == b
        nb = int(m.sum())
        if nb == 0:
            continue
        cells = []
        res["by_interval"][b] = {"n": nb}
        for k in KS:
            p, lo, hi = wilson(int(hits[k][m].sum()), nb)
            cells.append(f"{p:.3f}[{lo:.2f}-{hi:.2f}]")
            res["by_interval"][b][f"recall@{k}"] = {"value": round(p, 3), "ci": [round(lo, 3), round(hi, 3)]}
        print(f"{b:<14}{nb:>5}  " + "  ".join(cells))
    print(f"\nслучайный baseline top-1 = {rnd:.3f}  (21 особь)")

    out = ART / f"metrics_{model_name}_{crop}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"✅ метрики: {out.name}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="megadescriptor", choices=list(EMBEDDERS))
    ap.add_argument("--crop", default="none", choices=list(CROPPERS))
    a = ap.parse_args()
    run(a.model, a.crop)
