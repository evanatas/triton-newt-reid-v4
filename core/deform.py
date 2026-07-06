"""Часть 7 — деформ-устойчивая геометрическая верификация соответствий (R4.1 из ревью).

Жёсткая findHomography (8-DOF проективная) штрафует растяжение узора у растущих особей.
Замена: affine (6-DOF, допускает анизотропное растяжение) / partial (4-DOF similarity) / soft (без жёсткой фильтрации).
score = число геометрически согласованных матчей (или мягкая сумма).
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_GOOD = 4


def verify(pts_a: np.ndarray, pts_b: np.ndarray, method: str = "affine", thr: float = 5.0) -> int:
    """Сырые соответствия (Nx2, Nx2) → число геом-согласованных (inliers)."""
    n = len(pts_a)
    if n < MIN_GOOD:
        return n
    a = np.float32(pts_a)
    b = np.float32(pts_b)
    if method == "homography":                       # baseline (как спайк), жёсткая проективная
        _, mask = cv2.findHomography(a, b, cv2.RANSAC, thr)
    elif method == "affine":                         # 6-DOF: поворот+масштаб+СДВИГ+растяжение по осям
        _, mask = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=thr)
    elif method == "partial":                        # 4-DOF: similarity (поворот+единый масштаб+сдвиг)
        _, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=thr)
    elif method == "soft":                           # мягкая: similarity-модель, но score = Σ gauss(остаток)
        return _soft_score(a, b, thr)
    else:
        raise ValueError(method)
    return int(mask.sum()) if mask is not None else 0   # F-4: RANSAC вырожден (mask None) → 0 инлайеров, не сырой n


def inlier_mask(pts_a: np.ndarray, pts_b: np.ndarray, method: str = "affine", thr: float = 5.0) -> np.ndarray:
    """Часть 9 (интерпретируемость): булева маска геом-inlier'ов ТОЙ ЖЕ RANSAC-моделью, что verify.
    Для окраски матчей на оверлее (inlier vs отклонённый Lowe-матч). n<MIN_GOOD → все False. verify() НЕ меняем."""
    n = len(pts_a)
    if n < MIN_GOOD:
        return np.zeros(n, bool)
    a = np.float32(pts_a)
    b = np.float32(pts_b)
    if method == "homography":
        _, mask = cv2.findHomography(a, b, cv2.RANSAC, thr)
    elif method == "affine":
        _, mask = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=thr)
    elif method == "partial":
        _, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=thr)
    else:
        raise ValueError(f"inlier_mask: {method}")
    return mask.ravel().astype(bool) if mask is not None else np.zeros(n, bool)


def _soft_score(a: np.ndarray, b: np.ndarray, thr: float) -> int:
    """Гео-консистентность домножает, а не отсекает: score = Σ exp(-(r/thr)^2) по similarity-модели."""
    M, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=thr * 2)
    if M is None:
        return len(a)
    proj = (a @ M[:, :2].T) + M[:, 2]
    r = np.linalg.norm(proj - b, axis=1)
    return int(round(float(np.exp(-(r / thr) ** 2).sum())))
