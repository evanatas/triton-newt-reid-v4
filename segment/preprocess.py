"""Часть 4 — единый препроцессинг кропа для SIFT. Один и тот же для gallery и probe (урок H1)."""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
for _p in ("spike_lab", "registry"):
    sys.path.insert(0, str(HERE.parent / _p))
from common import ROOT   # noqa: E402

from birefnet import body_mask          # noqa: E402
from label_mask import detect_label, apply_label_mask   # noqa: E402
from crop import tight_crop, belly_only   # noqa: E402


def to_sift_gray(crop_rgb: np.ndarray) -> np.ndarray:
    """Канон-кроп RGB → серый + CLAHE (как спайк). Единый контракт препроцессинга."""
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)


def make_crop(path_rel: str, method: str = "birefnet_label") -> tuple[np.ndarray, dict]:
    """Полный кроп-пайплайн для одного кадра по выбранному методу.
    method ∈ {yellow, birefnet, birefnet_label, birefnet_label_belly}.
    -> (sift_gray uint8, meta)."""
    img = np.array(Image.open(ROOT / path_rel).convert("RGB"))
    meta = {"method": method, "has_label": False}

    if method == "yellow":
        from birefnet import _yellow_mask
        m = _yellow_mask(img)
        meta["seg_method"] = "yellow"
    else:
        m, seg, frac = body_mask(img)
        meta["seg_method"] = seg
        meta["mask_area_frac"] = round(frac, 4)
        if "label" in method:
            lbl, has = detect_label(img, m)
            meta["has_label"] = bool(has)
            img, m = apply_label_mask(img, m, lbl)
        if "belly" in method:
            m = belly_only(m)

    crop = tight_crop(img, m)
    return to_sift_gray(crop), meta
