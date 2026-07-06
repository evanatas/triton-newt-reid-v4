"""Часть 7 — единый интерфейс локальных матчеров: SIFT (baseline) + learned (DISK/ALIKED+LightGlue, LoFTR).

Контракт: matcher.extract(rgb_crop) -> feat (кэшируемо); matcher.matched(feat_a, feat_b) -> (pts_a, pts_b).
score = deform.verify(pts_a, pts_b, method). Единый вход — RGB-кроп birefnet_label (gallery=probe, H1).
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # kornia ops без MPS → CPU-fallback

import cv2
import numpy as np
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_LOWE = 0.75


def _gray_clahe(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)


# ───────────────────── SIFT (baseline) ─────────────────────
class SiftMatcher:
    name = "sift"

    def __init__(self, nfeatures=1500):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)

    def extract(self, rgb):
        kp, des = self.sift.detectAndCompute(_gray_clahe(rgb), None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return {"pts": pts, "des": des}

    def matched(self, fa, fb):
        da, db = fa["des"], fb["des"]
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        ga, gb = [], []
        for mn in bf.knnMatch(da, db, k=2):
            if len(mn) == 2 and mn[0].distance < _LOWE * mn[1].distance:
                ga.append(fa["pts"][mn[0].queryIdx]); gb.append(fb["pts"][mn[0].trainIdx])
        return np.float32(ga).reshape(-1, 2), np.float32(gb).reshape(-1, 2)


# ───────────────────── learned: DISK / ALIKED + LightGlue ─────────────────────
class LightGlueLearned:
    """detector ∈ {disk, aliked}. Извлечение фич kornia + матчинг LightGlueMatcher."""

    def __init__(self, detector="disk", n_features=1024):
        import kornia.feature as KF
        self.name = f"{detector}_lightglue"
        self.detector = detector
        self.n = n_features
        self._KF = KF
        if detector == "disk":
            self.model = KF.DISK.from_pretrained("depth").to(DEVICE).eval()
        elif detector == "aliked":
            self.model = KF.ALIKED(model_name="aliked-n16", detection_threshold=0.0).to(DEVICE).eval()
        else:
            raise ValueError(detector)
        self.lg = KF.LightGlueMatcher(detector).to(DEVICE).eval()

    def _tensor(self, rgb):
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
        return t.to(DEVICE)

    @torch.inference_mode()
    def extract(self, rgb):
        img = self._tensor(rgb)
        if self.detector == "disk":
            f = self.model(img, self.n, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
            kp, desc = f.keypoints, f.descriptors
        else:  # aliked — kornia 0.8 возвращает list[ALIKEDFeatures] (dataclass), не dict; топ-N по score
            feat = self.model(img)[0]
            order = feat.keypoint_scores.argsort(descending=True)
            kp, desc = feat.keypoints[order], feat.descriptors[order]
        hw = (rgb.shape[0], rgb.shape[1])
        return {"kp": kp.float(), "desc": desc, "hw": hw}

    @torch.inference_mode()
    def matched(self, fa, fb):
        if len(fa["kp"]) < 2 or len(fb["kp"]) < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        laf_a = self._KF.laf_from_center_scale_ori(fa["kp"][None])
        laf_b = self._KF.laf_from_center_scale_ori(fb["kp"][None])
        _, idxs = self.lg(fa["desc"], fb["desc"], laf_a, laf_b)
        if idxs is None or len(idxs) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        ia, ib = idxs[:, 0].cpu().numpy(), idxs[:, 1].cpu().numpy()
        return fa["kp"].cpu().numpy()[ia], fb["kp"].cpu().numpy()[ib]


# ───────────────────── LoFTR (detector-free, dense) ─────────────────────
class LoftrMatcher:
    name = "loftr"

    def __init__(self, conf_thr=0.5):
        import kornia.feature as KF
        self.loftr = KF.LoFTR(pretrained="outdoor").to(DEVICE).eval()
        self.conf = conf_thr

    def extract(self, rgb):                              # LoFTR — image-pair, «фича» = подготовленный gray-тензор
        g = _gray_clahe(rgb)
        t = torch.from_numpy(g).float()[None, None] / 255.0
        return {"img": t.to(DEVICE)}

    @torch.inference_mode()
    def matched(self, fa, fb):
        out = self.loftr({"image0": fa["img"], "image1": fb["img"]})
        m = out["confidence"] >= self.conf
        ka = out["keypoints0"][m].cpu().numpy()
        kb = out["keypoints1"][m].cpu().numpy()
        return ka.astype(np.float32), kb.astype(np.float32)


def build(name: str):
    if name == "sift":
        return SiftMatcher()
    if name in ("disk_lightglue", "aliked_lightglue"):
        return LightGlueLearned("disk" if name.startswith("disk") else "aliked")
    if name == "loftr":
        return LoftrMatcher()
    raise ValueError(name)
