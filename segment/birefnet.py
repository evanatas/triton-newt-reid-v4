"""Часть 4 — сегментация тела тритона. BiRefNet zero-shot matting + лестница fallback.

Лестница (никогда не падаем): BiRefNet → HSV-yellow (как спайк) → center-crop.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "spike_lab"))

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_TF = None
HF_ID = "ZhengPeng7/BiRefNet"


def _load():
    """Ленивая загрузка BiRefNet (transformers, trust_remote_code). fp16→.float() (грабли 3.0)."""
    global _MODEL, _TF
    if _MODEL is not None:
        return
    from transformers import AutoModelForImageSegmentation
    import torchvision.transforms as T
    m = AutoModelForImageSegmentation.from_pretrained(HF_ID, trust_remote_code=True)
    m.float().to(DEVICE).eval()                      # .float(): защита от fp16-bias-mismatch
    _MODEL = m
    _TF = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def body_mask_birefnet(img_rgb: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """RGB uint8 → бинарная маска тела (H,W) uint8 {0,1}. Пустая маска → нули."""
    _load()
    h, w = img_rgb.shape[:2]
    x = _TF(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _MODEL(x)
        pred = out[-1] if isinstance(out, (list, tuple)) else out.logits
        pred = pred.sigmoid().cpu().numpy()[0, 0]    # (1024,1024) в [0,1]
    m = (cv2.resize(pred, (w, h)) > thr).astype(np.uint8)
    # оставить крупнейшую связную компоненту (убрать мелкий шум matting)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8)
    return m


def _yellow_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Fallback из спайка: HSV жёлто-оранжевый → крупнейшая компонента."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (12, 60, 50), (45, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(img_rgb.shape[:2], np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def body_mask(img_rgb: np.ndarray, min_frac: float = 0.01) -> tuple[np.ndarray, str, float]:
    """Главный вход: маска тела с лестницей fallback.
    -> (mask {0,1}, method ∈ {birefnet,yellow,center}, area_frac)."""
    H, W = img_rgb.shape[:2]
    area = H * W
    try:
        m = body_mask_birefnet(img_rgb)
        if m.sum() >= min_frac * area:
            return m, "birefnet", float(m.sum() / area)
    except Exception as e:
        print(f"  [birefnet fail → yellow] {e}", file=sys.stderr)
    m = _yellow_mask(img_rgb)
    if m.sum() >= min_frac * area:
        return m, "yellow", float(m.sum() / area)
    # center-crop как маска (последний fallback)
    m = np.zeros((H, W), np.uint8)
    y0, y1 = int(H * 0.2), int(H * 0.8)
    x0, x1 = int(W * 0.2), int(W * 0.8)
    m[y0:y1, x0:x1] = 1
    return m, "center", float(m.sum() / area)


if __name__ == "__main__":
    # smoke: загрузка + маска на одном LAB-кадре
    import pandas as pd
    sys.path.insert(0, str(HERE.parent / "registry"))
    from common import ROOT  # noqa
    reg = pd.read_csv(HERE.parent / "registry" / "registry.csv")
    row = reg[reg.cohort == "LAB"].iloc[0]
    img = np.array(Image.open(ROOT / row.path_rel).convert("RGB"))
    m, method, frac = body_mask(img)
    print(f"smoke: {row.path_rel}")
    print(f"  device={DEVICE} method={method} area_frac={frac:.3f} mask_sum={int(m.sum())}")
