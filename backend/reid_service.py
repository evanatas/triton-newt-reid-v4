"""Бекенд-движок re-id тритонов — обёртка над ЗАМОРОЖЕННЫМ ядром (SIFT + affine + rolling max-pool).

Каталог известных особей = TK-кропы (`segment/crops`), SIFT-фичи предрасчитаны в памяти. Загруженное фото проходит
ТОТ ЖЕ frozen-пайплайн (BiRefNet → маскирование метки → tight_crop) и матчится по каталогу. Это ДЕМО, не замер KPI.
"""
from __future__ import annotations
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent                                   # triton 4.0/
for _p in ("segment", "core", "registry", "spike_lab"):
    sys.path.insert(0, str((ROOT / _p).resolve()))

from birefnet import body_mask                        # noqa: E402  segment/
from label_mask import detect_label, apply_label_mask # noqa: E402  segment/
from crop import tight_crop                            # noqa: E402  segment/
import matchers as M                                   # noqa: E402  core/
import deform as D                                     # noqa: E402  core/
from interpret import draw_pair                        # noqa: E402  core/

SEG_DIR = ROOT / "segment"                             # crop_path в манифесте относителен к segment/
MANIFEST = SEG_DIR / "crops_manifest.csv"
SPLITS = ROOT / "registry" / "splits.csv"
DEFORM = "affine"                                      # деформ-модель замороженного ядра


def _b64_jpg(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


class ReIDService:
    """Загружает каталог TK один раз; identify() матчит новое фото по каталогу (max-pool по особи)."""

    def __init__(self, cohort: str = "TK"):
        self.sift = M.build("sift")
        man = pd.read_csv(MANIFEST)
        spl = pd.read_csv(SPLITS)[["frame_id", "individual_id", "split_role", "date", "is_open_new"]]  # cohort уже в манифесте
        self.cat = man.merge(spl, on="frame_id")
        self.cat = self.cat[self.cat.cohort == cohort].reset_index(drop=True)
        # open-set: особи, помеченные «новыми» (is_open_new), НЕ входят в каталог известных — иначе загрузка их
        # фото ложно даёт known (в каталоге остаются ДРУГИЕ их кадры). Честная демонстрация «новой» требует их
        # отсутствия в галерее. Held-out фото таких особей — в демо_фото/02_новые_особи.
        self.n_open_new = int(self.cat.is_open_new.fillna(False).sum())
        self.cat = self.cat[~self.cat.is_open_new.fillna(False)].reset_index(drop=True)
        self.imgs, self.feats = {}, {}
        for _, r in self.cat.iterrows():
            im = self._load(r.crop_path)
            self.imgs[r.frame_id] = im
            self.feats[r.frame_id] = self.sift.extract(im)
        self.n_individuals = int(self.cat.individual_id.nunique())
        # семплы для быстрой демонстрации: по 1 probe-кадру от разных особей
        self.samples = []
        for _, r in self.cat[self.cat.split_role == "probe"].iterrows():
            if r.individual_id not in {s[1] for s in self.samples}:
                self.samples.append((r.frame_id, r.individual_id))
            if len(self.samples) >= 4:
                break
        self.probes = [(r.frame_id, r.individual_id)
                       for _, r in self.cat[self.cat.split_role == "probe"].iterrows()]
        reg = pd.read_csv(ROOT / "registry" / "registry.csv")[["frame_id", "md5"]]
        self.md5_to_frame = {m: f for f, m in zip(reg.frame_id, reg.md5) if f in self.imgs}  # для анти-самосовпадения
        self.cal_lo, self.cal_hi = self._calibrate()      # шкала «уверенности» из genuine/impostor распределений
        print(f"[ReID] каталог {cohort}: {len(self.cat)} кадров / {self.n_individuals} известных особей "
              f"(open_new исключены из галереи: {self.n_open_new} кадров) | проб {len(self.probes)} | "
              f"калибровка lo={self.cal_lo:.0f}/hi={self.cal_hi:.0f}")

    # ───────────── калибровка «уверенности» (как в 3.0-демо, но на inlier-скорах) ─────────────
    def _calibrate(self, max_pairs: int = 500):
        """(lo, hi) для шкалы уверенности: lo = 90-й перцентиль impostor-скоров (разные особи),
        hi = 75-й перцентиль genuine-скоров (одна особь). Своя → ~99 %, чужая/новая → низкая, без сатурации."""
        import random
        rng = random.Random(42)
        by_ind: dict = {}
        for _, r in self.cat.iterrows():
            by_ind.setdefault(r.individual_id, []).append(r.frame_id)
        gen, imp = [], []
        for fids in by_ind.values():                      # genuine: пары внутри особи
            for a in range(len(fids)):
                for b in range(a + 1, len(fids)):
                    gen.append(D.verify(*self.sift.matched(self.feats[fids[a]], self.feats[fids[b]]), DEFORM))
            if len(gen) >= max_pairs:
                break
        inds = list(by_ind)
        while len(imp) < max_pairs and len(inds) > 1:     # impostor: случайные пары разных особей
            i, j = rng.sample(inds, 2)
            imp.append(D.verify(*self.sift.matched(self.feats[rng.choice(by_ind[i])],
                                                   self.feats[rng.choice(by_ind[j])]), DEFORM))
        gen = np.array(gen or [1.0]); imp = np.array(imp or [0.0])
        # lo = высокий impostor (уровень «похожего чужого»); hi = типичный genuine (75-й перцентиль, НЕ хвост
        # near-dup пар с 200+ inlier — иначе умеренные истинные матчи читались бы заниженно).
        lo = float(np.percentile(imp, 90)); hi = float(np.percentile(gen, 75))
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    @staticmethod
    def calibrate_confidence(score: float, lo: float, hi: float) -> float:
        """inlier-скор → «уверенность» 0..99 % (монотонно, клип к [lo,hi]). Не вероятность."""
        x = max(0.0, min(1.0, (float(score) - lo) / max(hi - lo, 1e-6)))
        return min(x * 100.0, 99.0)

    def _score_all(self, qfeat, exclude_frame=None) -> dict:
        """individual_id -> (best_score, best_frame) max-pool по кадрам особи.
        exclude_frame — кадр ИЛИ множество кадров, исключаемых из каталога (напр. вся сессия запроса)."""
        excl = {exclude_frame} if isinstance(exclude_frame, str) else set(exclude_frame or ())
        best = {}
        for _, r in self.cat.iterrows():
            if r.frame_id in excl:
                continue
            s = D.verify(*self.sift.matched(qfeat, self.feats[r.frame_id]), DEFORM)
            if r.individual_id not in best or s > best[r.individual_id][0]:
                best[r.individual_id] = (s, r.frame_id)
        return best

    def rank(self, qfeat, topk: int = 5, exclude_frame=None) -> list:
        """top-K особей карточками: individual_id, score, confidence%, best_frame, cohort, n_photos."""
        best = self._score_all(qfeat, exclude_frame)
        rows = []
        for iid, (s, fr) in best.items():
            rows.append({"individual_id": iid, "score": int(s),
                         "confidence": round(self.calibrate_confidence(s, self.cal_lo, self.cal_hi), 1),
                         "best_frame": fr, "cohort": str(self.cat[self.cat.frame_id == fr].iloc[0].cohort),
                         "n_photos": int((self.cat.individual_id == iid).sum())})
        rows.sort(key=lambda r: -r["score"])
        return rows[:topk]

    @staticmethod
    def verdict(ranked: list, conf_thr: float = 70.0, review_thr: float = 40.0, margin_thr: float = 8.0) -> dict:
        """Tri-state (known / review «на проверку» / new) в шкале КАЛИБРОВАННОЙ УВЕРЕННОСТИ (не сырые inliers):
        known  — уверенность top-1 ≥ conf_thr И отрыв от top-2 ≥ margin_thr (однозначно известна);
        new    — уверенность top-1 < review_thr (низкая → кандидат в новую особь);
        review — промежуток: сигнал есть, но для однозначного known мало (human-in-the-loop, ФТ-4 / known-new-other).
        Метрический open-set меряется ОТДЕЛЬНО в affine-inliers (порог 9, sealed BAKS 0.67/G 0.819) — не смешивать."""
        if not ranked:
            return {"verdict": "new", "confidence": 0.0, "margin": 0.0}
        t1 = float(ranked[0]["confidence"]); t2 = float(ranked[1]["confidence"]) if len(ranked) > 1 else 0.0
        margin = round(t1 - t2, 1)
        if t1 >= conf_thr and margin >= margin_thr:
            v = "known"
        elif t1 < review_thr:
            v = "new"
        else:
            v = "review"
        return {"verdict": v, "confidence": t1, "margin": margin}

    def pair_score(self, frame_a: str, frame_b: str) -> int:
        """affine-inliers между двумя кадрами каталога — наглядность temporal-матча одной особи во времени."""
        pa, pb = self.sift.matched(self.feats[frame_a], self.feats[frame_b])
        return int(D.verify(pa, pb, DEFORM))

    def overlay_pair(self, query_crop, qfeat, gallery_frame):
        """BGR-оверлей совпавших пятен: запрос ↔ лучший кадр top-1 особи (зелёные = affine-inliers)."""
        pa, pb = self.sift.matched(qfeat, self.feats[gallery_frame])
        mask = D.inlier_mask(pa, pb, DEFORM)
        return draw_pair(self.imgs[gallery_frame], query_crop, pb, pa, mask, f"matched inliers: {int(mask.sum())}")

    def _load(self, crop_path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(SEG_DIR / crop_path)), cv2.COLOR_BGR2RGB)

    def segment_upload(self, image_bytes: bytes):
        """Сырое фото → RGB → BiRefNet → маскирование метки → tight_crop (frozen-пайплайн)."""
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        m, seg, frac = body_mask(img)
        lbl, has = detect_label(img, m)
        img2, m2 = apply_label_mask(img, m, lbl)
        crop = tight_crop(img2, m2)
        return crop, {"seg_method": seg, "mask_frac": round(float(frac), 3), "has_label": bool(has)}

    def frame_for_md5(self, image_bytes: bytes):
        """frame_id каталога с тем же md5, что загруженный файл (анти-самосовпадение при загрузке held-out) или None."""
        import hashlib
        return self.md5_to_frame.get(hashlib.md5(image_bytes).hexdigest())

    def session_of(self, frame_id) -> set:
        """Все кадры каталога той же особи и той же даты (съёмочной сессии), что frame_id.
        Честный temporal-тест: исключив сессию запроса, система матчит только по снимкам ДРУГИХ месяцев,
        а не по near-duplicate того же дня. frame_id=None → пусто; неизвестный кадр → сам кадр."""
        if not frame_id:
            return set()
        row = self.cat[self.cat.frame_id == frame_id]
        if row.empty:
            return {frame_id}
        iid, dt = row.iloc[0].individual_id, row.iloc[0].date
        if pd.isna(dt) or not str(dt):
            return {frame_id}
        return set(self.cat[(self.cat.individual_id == iid) & (self.cat.date == dt)].frame_id)

    def date_of(self, frame_id):
        """Дата (YYYY-MM) кадра каталога или None."""
        row = self.cat[self.cat.frame_id == frame_id]
        return None if row.empty or pd.isna(row.iloc[0].date) else str(row.iloc[0].date)

    def register(self, crop, feat, name: str = "") -> dict:
        """Демо флоу учёта: добавить особь/кадр в каталог В ПАМЯТИ (без БД) — становится «известной» до конца
        сессии. В проде здесь была бы запись кропа+фич+метаданных в БД (SQLite→PostgreSQL)."""
        n = sum(1 for f in self.imgs if str(f).startswith("REG-")) + 1
        fid, iid = f"REG-{n:03d}", (str(name).strip() or f"NEW-{n}")
        self.imgs[fid] = crop
        self.feats[fid] = feat
        row = {c: "" for c in self.cat.columns}
        row.update({"frame_id": fid, "individual_id": iid, "cohort": "TK",
                    "split_role": "registered", "date": "рег.", "is_open_new": False})
        self.cat = pd.concat([self.cat, pd.DataFrame([row])], ignore_index=True)
        self.n_individuals = int(self.cat.individual_id.nunique())
        return {"frame_id": fid, "individual_id": iid}

    def identify_crop(self, crop: np.ndarray, topk: int = 5, exclude_frame: str | None = None) -> dict:
        """Кроп (RGB) → топ-k особей каталога (max-pool score по особи) + оверлеи лучшего кадра."""
        fp = self.sift.extract(crop)
        excl = {exclude_frame} if isinstance(exclude_frame, str) else set(exclude_frame or ())
        best = {}                                     # individual_id -> (score, frame_id)
        for _, r in self.cat.iterrows():
            if r.frame_id in excl:
                continue
            pa, pb = self.sift.matched(fp, self.feats[r.frame_id])
            s = D.verify(pa, pb, DEFORM)
            if r.individual_id not in best or s > best[r.individual_id][0]:
                best[r.individual_id] = (s, r.frame_id)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:topk]
        results = []
        for iid, (score, gframe) in ranked:
            gimg = self.imgs[gframe]
            pa, pb = self.sift.matched(fp, self.feats[gframe])          # pa=probe, pb=gallery
            mask = D.inlier_mask(pa, pb, DEFORM)
            title = f"{iid} | score={int(score)} inliers={int(mask.sum())}"
            overlay = draw_pair(gimg, crop, pb, pa, mask, title)        # imgA=gallery(pb), imgB=probe(pa)
            results.append({"individual_id": iid, "score": int(score), "inliers": int(mask.sum()),
                            "gallery_frame": gframe, "overlay_b64": _b64_jpg(overlay)})
        return {"probe_crop_b64": _b64_jpg(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)),
                "n_catalog_individuals": self.n_individuals, "results": results}

    def identify(self, image_bytes: bytes, topk: int = 5) -> dict:
        crop, meta = self.segment_upload(image_bytes)
        out = self.identify_crop(crop, topk)
        out["meta"] = meta
        return out

    def sample_thumbs(self) -> list[dict]:
        return [{"frame_id": fid, "individual_id": iid, "thumb_b64": _b64_jpg(
                 cv2.cvtColor(cv2.resize(self.imgs[fid], (160, 160)), cv2.COLOR_RGB2BGR))}
                for fid, iid in self.samples]

    def identify_sample(self, frame_id: str, topk: int = 5) -> dict:
        """Демо-прогон: существующий probe-кадр как запрос (исключён из каталога), с раскрытием истинной особи."""
        row = self.cat[self.cat.frame_id == frame_id]
        if row.empty:
            raise KeyError(frame_id)
        true_iid = row.iloc[0].individual_id
        out = self.identify_crop(self.imgs[frame_id], topk, exclude_frame=frame_id)
        out["meta"] = {"sample": frame_id, "true_individual": true_iid,
                       "top1_correct": bool(out["results"] and out["results"][0]["individual_id"] == true_iid)}
        return out
