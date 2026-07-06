"""Часть 4 — детект и маскирование метки (бумажка с номером особи). Анти-утечка идентичности.

Метка = яркий низконасыщенный (белый) прямоугольный регион, обычно у края / вне тела.
"""
from __future__ import annotations
import cv2
import numpy as np


def detect_label(img_rgb: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, bool]:
    """Найти белую бумажку-метку. -> (label_mask {0,1}, has_label)."""
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    white = ((hsv[..., 2] > 200) & (hsv[..., 1] < 40)).astype(np.uint8)   # яркое + низкая насыщенность
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    out = np.zeros((H, W), np.uint8)
    found = False
    body_cy, body_cx = _centroid(body)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0008 * H * W or area > 0.15 * H * W:        # не слишком мелкое/большое
            continue
        extent = area / (w * h + 1e-6)                          # прямоугольность бумажки
        if extent < 0.55:
            continue
        comp = (lab == i)
        overlap = (comp & (body > 0)).sum() / (area + 1e-6)     # метка в основном ВНЕ тела
        if overlap > 0.5:
            continue
        out |= comp.astype(np.uint8)
        found = True
    return out, found


def _centroid(m: np.ndarray):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return m.shape[0] / 2, m.shape[1] / 2
    return ys.mean(), xs.mean()


def apply_label_mask(img_rgb: np.ndarray, body: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Исключить метку из маски тела и закрасить её нейтральным (медиана тела), чтобы не дать keypoints.
    -> (img_clean, body_clean)."""
    body_clean = (body.astype(bool) & ~label.astype(bool)).astype(np.uint8)
    img = img_rgb.copy()
    if label.any():
        med = np.median(img_rgb[body > 0], axis=0) if (body > 0).any() else np.array([127, 127, 127])
        img[label > 0] = med.astype(np.uint8)
    return img, body_clean
