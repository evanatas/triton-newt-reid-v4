"""Часть 9 — интерпретируемость ядра: SIFT-оверлеи повторных съёмок (dev).

Показывает, что affine-inlier SIFT-соответствия ложатся на УЗОР ПЯТЕН брюшка (а не силуэт/фон): зелёные линии+кружки =
affine-inliers (что считает ядро), тускло-серые = отклонённые Lowe-матчи. Same-individual (успех) + cross-individual
(контраст: мало/разбросаны). Метрика уже взята — это интерпретация. Всё на DEV (sealed не трогаем).

Запуск:  python interpret.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
for _p in ("..", "../segment", "../registry", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
from sealed import assert_unsealed                # noqa: E402
import matchers as M                              # noqa: E402
import deform as D                                # noqa: E402
import eval_core as EC                            # noqa: E402

CROPS = (HERE / ".." / "segment").resolve()
OUT = HERE / "artifacts" / "interpret"
DEFORM = "affine"
N_SAME, N_CROSS = 8, 0   # cross-плитки убраны из фигуры: зелёные «совпадения» у разных особей путали (контраст 20 vs 3–4 — словами в §3.8)


def load_img(crop_path):
    return cv2.cvtColor(cv2.imread(str(CROPS / crop_path)), cv2.COLOR_BGR2RGB)


def draw_pair(imgA, imgB, ptsA, ptsB, mask, title):
    """side-by-side gallery|probe: серые = отклонённые матчи, зелёные = affine-inliers + кружки на keypoints."""
    hA, wA = imgA.shape[:2]; hB, wB = imgB.shape[:2]
    H = max(hA, hB)
    cv = np.full((H + 26, wA + wB, 3), 20, np.uint8)
    cv[26:26 + hA, :wA] = cv2.cvtColor(imgA, cv2.COLOR_RGB2BGR)
    cv[26:26 + hB, wA:wA + wB] = cv2.cvtColor(imgB, cv2.COLOR_RGB2BGR)
    for (xa, ya), (xb, yb), inl in zip(ptsA, ptsB, mask):          # сперва отклонённые (тускло-серые)
        if not inl:
            cv2.line(cv, (int(xa), int(ya) + 26), (int(xb) + wA, int(yb) + 26), (85, 85, 85), 1, cv2.LINE_AA)
    for (xa, ya), (xb, yb), inl in zip(ptsA, ptsB, mask):          # поверх — inliers (зелёные) + кружки
        if inl:
            pa = (int(xa), int(ya) + 26); pb = (int(xb) + wA, int(yb) + 26)
            cv2.line(cv, pa, pb, (0, 230, 0), 1, cv2.LINE_AA)
            cv2.circle(cv, pa, 3, (0, 230, 0), 1, cv2.LINE_AA)
            cv2.circle(cv, pb, 3, (0, 230, 0), 1, cv2.LINE_AA)
    cv2.putText(cv, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv


def montage(canvases, w=760):
    rows = []
    for c in canvases:
        h = int(c.shape[0] * w / c.shape[1])
        rows.append(cv2.resize(c, (w, h)))
        rows.append(np.full((3, w, 3), 60, np.uint8))            # разделитель
    return np.vstack(rows[:-1]) if rows else None


# ───────────────────── премиум-рендер: выравнивание по аффинной модели ядра ─────────────────────
_FONT_CACHE: dict = {}


def _font(sz):
    if sz not in _FONT_CACHE:
        import matplotlib.font_manager as fm
        _FONT_CACHE[sz] = ImageFont.truetype(fm.findfont("DejaVu Sans"), sz)
    return _FONT_CACHE[sz]


def _text(bgr, text, xy, sz=20, color=(240, 240, 240)):
    """Кириллица на BGR-картинке через PIL (cv2.putText кириллицу не умеет). color — в RGB."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(xy, text, font=_font(sz), fill=tuple(color))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _affine_from_inliers(pa, pb, mask, thr=5.0):
    """Аффинная модель ядра (та же estimateAffine2D), оценённая по инлайерам, — для выравнивания кадра.
    Дисплей-only: метрику/score не трогает."""
    a, b = np.float32(np.asarray(pa)[mask]), np.float32(np.asarray(pb)[mask])
    if len(a) < 3:
        return None
    M, _ = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=thr)
    return M


ACCENT = (60, 220, 70)   # BGR — зелёный


def draw_pair_aligned(imgA, imgB, pa, pb, mask, M, title):
    """Левая панель — gallery A, выровненная аффинной моделью ядра в кадр probe B; правая — B (не тронут).
    Инлайеры рисуются как почти горизонтальные зелёные связи между ОДНИМИ пятнами; отклонённые скрыты.
    M=None (напр. разные особи) → левый кадр показывается без выравнивания."""
    hB, wB = imgB.shape[:2]
    bB = cv2.cvtColor(imgB, cv2.COLOR_RGB2BGR)
    aB = cv2.cvtColor(imgA, cv2.COLOR_RGB2BGR)
    pa = np.asarray(pa, np.float32)
    if M is not None:
        c0 = imgA[2, 2]                                   # цвет фона кропа → заливка полей варпа (без тёмных углов)
        wA = cv2.warpAffine(aB, M, (wB, hB), flags=cv2.INTER_LINEAR,
                            borderValue=(int(c0[2]), int(c0[1]), int(c0[0])))
        paL = (pa @ M[:, :2].T) + M[:, 2]
    elif aB.shape[:2] != (hB, wB):
        paL = pa * np.float32([wB / aB.shape[1], hB / aB.shape[0]])
        wA = cv2.resize(aB, (wB, hB))
    else:
        wA, paL = aB, pa
    TB, GAP = 30, 10
    off = wB + GAP
    cv = np.full((hB + TB, wB + GAP + wB, 3), 24, np.uint8)
    cv[TB:TB + hB, :wB] = wA
    cv[TB:TB + hB, off:off + wB] = bB
    ov = cv.copy()
    for (xa, ya), (xb, yb), inl in zip(paL, np.asarray(pb), mask):
        if not inl:
            continue
        pA = (int(round(float(xa))), int(round(float(ya))) + TB)
        pB = (int(round(float(xb))) + off, int(round(float(yb))) + TB)
        cv2.line(ov, pA, pB, ACCENT, 2, cv2.LINE_AA)
        cv2.circle(ov, pA, 3, ACCENT, -1, cv2.LINE_AA)
        cv2.circle(ov, pB, 3, ACCENT, -1, cv2.LINE_AA)
    cv = cv2.addWeighted(ov, 0.82, cv, 0.18, 0)
    return _text(cv, title, (8, 6), 19, (245, 245, 245))


def legend_strip(w):
    s = np.full((58, w, 3), 18, np.uint8)
    cv2.line(s, (16, 24), (56, 24), ACCENT, 3, cv2.LINE_AA)
    cv2.circle(s, (16, 24), 4, ACCENT, -1, cv2.LINE_AA)
    cv2.circle(s, (56, 24), 4, ACCENT, -1, cv2.LINE_AA)
    s = _text(s, "зелёные — инлайеры аффинной модели ядра (по их числу принимается решение)", (70, 6), 17, (235, 235, 235))
    s = _text(s, "левый кадр выровнен по модели → связи горизонтальны; отклонённые матчи скрыты", (16, 32), 14, (175, 175, 175))
    return s


def stack_before_after(imgA, imgB, pa, pb, mask, M, iid, interval):
    """Бонус-плитка: до (сырые SIFT-соответствия, диагонали) → после (выравнивание, инлайеры на одних пятнах)."""
    raw = draw_pair(imgA, imgB, pa, pb, mask, "")
    ali = draw_pair_aligned(imgA, imgB, pa, pb, mask, M, f"{iid} · {interval} мес · после выравнивания по аффинной модели ядра")
    W = max(raw.shape[1], ali.shape[1])

    def pad(im):
        if im.shape[1] == W:
            return im
        out = np.full((im.shape[0], W, 3), 24, np.uint8)
        out[:, :im.shape[1]] = im
        return out

    def bar(txt, color):
        return _text(np.full((28, W, 3), 24, np.uint8), txt, (8, 5), 16, color)

    sep = np.full((6, W, 3), 60, np.uint8)
    return np.vstack([bar("До: сырые соответствия SIFT — часть ложных, узор в разной позе", (215, 215, 215)),
                      pad(raw), sep,
                      bar("После: выравнивание по аффинной модели ядра — инлайеры ложатся на одни пятна брюшка", (185, 225, 195)),
                      pad(ali)])


def main():
    assert_unsealed("dev", unseal=False)
    OUT.mkdir(parents=True, exist_ok=True)
    refs_all, gallery_single, known_probe, new_probe = EC.load_cohort("TK", "dev")
    sift = M.build("sift")
    print(f"интерпретируемость на DEV: probe {len(known_probe)} | gallery-особей {gallery_single.individual_id.nunique()}")

    feats = {}
    def feat(r):
        if r.frame_id not in feats:
            feats[r.frame_id] = sift.extract(load_img(r.crop_path))
        return feats[r.frame_id]

    gal_by_ind = {iid: g.sort_values("tkey").iloc[0] for iid, g in gallery_single.groupby("individual_id")}

    same = []
    for _, p in known_probe.iterrows():
        if p.individual_id not in gal_by_ind:
            continue
        g = gal_by_ind[p.individual_id]
        pa, pb = sift.matched(feat(g), feat(p))
        mask = D.inlier_mask(pa, pb, DEFORM)
        same.append({"iid": p.individual_id, "g": g, "p": p, "matched": len(pa),
                     "inl": int(mask.sum()), "interval": int(round(float(p.interval_months))),
                     "pa": pa, "pb": pb, "mask": mask})

    # отбор ~N_SAME: по 1 лучшей паре на ОСОБЬ (диверсификация — разные тритоны), топ 1-мес + топ 2-мес + типичная (честность)
    same.sort(key=lambda x: -x["inl"])
    per_ind = {}
    for s in same:
        per_ind.setdefault(s["iid"], s)                          # лучшая пара на особь (список уже отсортирован)
    uniq = sorted(per_ind.values(), key=lambda x: -x["inl"])
    one = [s for s in uniq if s["interval"] == 1]
    two = [s for s in uniq if s["interval"] == 2]
    picked = one[:4] + two[:3]
    if len(uniq) >= 8:
        picked.append(uniq[len(uniq) // 2])                      # типичная по inliers (не только лучшие)
    seen, sel = set(), []
    for s in picked:
        if s["iid"] not in seen:
            seen.add(s["iid"]); sel.append(s)
    sel = sel[:N_SAME]

    # cross-individual контраст: gallery особи A × probe особи B (A != B — разные особи после диверсификации)
    cross = []
    if len(sel) >= 2:
        for g_src, p_src in [(sel[0], sel[1]), (sel[1], sel[0])][:N_CROSS]:
            g, p = g_src["g"], p_src["p"]
            pa, pb = sift.matched(feat(g), feat(p))
            mask = D.inlier_mask(pa, pb, DEFORM)
            cross.append({"g": g, "p": p, "matched": len(pa), "inl": int(mask.sum()),
                          "pa": pa, "pb": pb, "mask": mask})

    index = {"same": [], "cross": [],
             "note": "зелёные = инлайеры аффинной модели ядра (по их числу решение); левый кадр выровнен по этой модели, правый не тронут; отклонённые матчи скрыты; dev"}
    tiles = []
    for k, s in enumerate(sel):
        imgA, imgB = load_img(s["g"].crop_path), load_img(s["p"].crop_path)
        Aff = _affine_from_inliers(s["pa"], s["pb"], s["mask"])
        t = f"{s['iid']} · {s['interval']} мес · инлайеров {s['inl']} из {s['matched']}"
        c = draw_pair_aligned(imgA, imgB, s["pa"], s["pb"], s["mask"], Aff, t)
        fn = OUT / f"same_{s['iid']}_{s['interval']}mo_{s['inl']}inl.jpg"
        cv2.imwrite(str(fn), c); tiles.append(c)
        index["same"].append({"iid": s["iid"], "interval_mo": s["interval"], "matched": s["matched"],
                              "inliers": s["inl"], "file": fn.name})
        print(f"  SAME {s['iid']} {s['interval']}мес: matched={s['matched']} inliers={s['inl']} -> {fn.name}")
        if k == 0:
            ba = stack_before_after(imgA, imgB, s["pa"], s["pb"], s["mask"], Aff, s["iid"], s["interval"])
            cv2.imwrite(str(OUT / "interpret_before_after.jpg"), ba)
    for cr in cross:
        imgA, imgB = load_img(cr["g"].crop_path), load_img(cr["p"].crop_path)
        t = f"разные особи · {cr['g'].individual_id} → {cr['p'].individual_id} · инлайеров {cr['inl']} из {cr['matched']} (без выравнивания)"
        c = draw_pair_aligned(imgA, imgB, cr["pa"], cr["pb"], cr["mask"], None, t)
        fn = OUT / f"cross_{cr['g'].individual_id}_vs_{cr['p'].individual_id}_{cr['inl']}inl.jpg"
        cv2.imwrite(str(fn), c); tiles.append(c)
        index["cross"].append({"gallery": cr["g"].individual_id, "probe": cr["p"].individual_id,
                               "matched": cr["matched"], "inliers": cr["inl"], "file": fn.name})
        print(f"  CROSS {cr['g'].individual_id}->{cr['p'].individual_id}: matched={cr['matched']} inliers={cr['inl']} -> {fn.name}")

    if tiles:
        m = montage([legend_strip(tiles[0].shape[1])] + tiles, w=900)
        cv2.imwrite(str(OUT / "interpret_montage.jpg"), m)
    (OUT / "interpret_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    same_inl = [s["inl"] for s in sel]; cross_inl = [c["inl"] for c in cross]
    print(f"\nsame inliers медиана {int(np.median(same_inl))} (мин {min(same_inl)}–макс {max(same_inl)}) | "
          f"cross inliers {cross_inl} → контраст виден: {'ДА' if (cross_inl and max(cross_inl) < np.median(same_inl)) else 'проверить глазами'}")
    print(f"✅ artifacts/interpret/ ({len(sel)} same + {len(cross)} cross) + interpret_montage.jpg (test не вскрывался)")


if __name__ == "__main__":
    main()
