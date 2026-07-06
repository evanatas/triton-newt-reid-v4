"""Часть 4 — тугой кроп брюшка + нормализация масштаба по высоте маски тела.

Нормализация масштаба критична для RANSAC-гомографии: узор одного физического размера между сессиями.
"""
from __future__ import annotations
import cv2
import numpy as np

CANVAS = 512        # размер канон-кропа
TARGET_H = 440      # целевая высота маски тела внутри канваса (нормализация масштаба)


def tight_crop(img_rgb: np.ndarray, body: np.ndarray, pad: float = 0.08,
               canvas: int = CANVAS, target_h: int = TARGET_H, fill=(127, 127, 127)) -> np.ndarray:
    """Кроп по bbox маски, фон вне маски залит нейтральным, нормализация масштаба → canvas×canvas.
    Высота bbox маски приводится к target_h (одинаковый физический масштаб узора между кадрами)."""
    ys, xs = np.where(body > 0)
    if len(xs) == 0:                                  # пустая маска → центр-кроп
        h, w = img_rgb.shape[:2]
        y0, y1, x0, x1 = int(h*0.2), int(h*0.8), int(w*0.2), int(w*0.8)
        sub = img_rgb[y0:y1, x0:x1]
        return _fit_canvas(sub, canvas, fill)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad), int(bh * pad)
    H, W = img_rgb.shape[:2]
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1 + px), min(H, y1 + py)

    sub = img_rgb[cy0:cy1, cx0:cx1].copy()
    sub_mask = body[cy0:cy1, cx0:cx1]
    sub[sub_mask == 0] = fill                          # фон залить нейтральным (вне тела)

    # нормализация масштаба: высота МАСКИ (bh) → target_h
    scale = target_h / max(bh, 1)
    nh, nw = int(sub.shape[0] * scale), int(sub.shape[1] * scale)
    if nh > 0 and nw > 0:
        sub = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    return _fit_canvas(sub, canvas, fill)


def _fit_canvas(sub: np.ndarray, canvas: int, fill) -> np.ndarray:
    """Вписать в canvas×canvas по центру (letterbox, без растяжения)."""
    h, w = sub.shape[:2]
    s = min(canvas / max(h, 1), canvas / max(w, 1), 1.0)
    if s < 1:
        sub = cv2.resize(sub, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        h, w = sub.shape[:2]
    out = np.full((canvas, canvas, 3), fill, np.uint8)
    oy, ox = (canvas - h) // 2, (canvas - w) // 2
    out[oy:oy+h, ox:ox+w] = sub
    return out


def belly_only(body: np.ndarray, erode_frac: float = 0.12) -> np.ndarray:
    """НЕ ИСПОЛЬЗУЕТСЯ в продукте: A/B подтвердил вред (R@1 0.289 vs 0.485). Оставлен как проверенный негатив.
    R1.5-фикс (ревью §6 #5): эрозия от высоты bbox МАСКИ, а не от размера всего кадра (была эрозия в сотни px)."""
    ys, xs = np.where(body > 0)
    if len(xs) == 0:
        return body
    bh = int(ys.max() - ys.min() + 1)
    k = max(3, int(bh * erode_frac) | 1)
    er = cv2.erode(body, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(er, connectivity=8)
    if n <= 1:
        return body
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    core = (lab == big).astype(np.uint8)
    return cv2.dilate(core, np.ones((k//2 | 1, k//2 | 1), np.uint8))   # вернуть часть толщины
