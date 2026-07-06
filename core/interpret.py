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
N_SAME, N_CROSS = 8, 2


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

    index = {"same": [], "cross": [], "note": "зелёные=affine-inliers (ядро); серые=отклонённые Lowe-матчи; dev"}
    canv = []
    for s in sel:
        t = f"SAME {s['iid']} {s['interval']}mo | matched={s['matched']} inliers={s['inl']}"
        c = draw_pair(load_img(s["g"].crop_path), load_img(s["p"].crop_path), s["pa"], s["pb"], s["mask"], t)
        fn = OUT / f"same_{s['iid']}_{s['interval']}mo_{s['inl']}inl.jpg"
        cv2.imwrite(str(fn), c); canv.append(c)
        index["same"].append({"iid": s["iid"], "interval_mo": s["interval"], "matched": s["matched"],
                              "inliers": s["inl"], "file": fn.name})
        print(f"  SAME {s['iid']} {s['interval']}мес: matched={s['matched']} inliers={s['inl']} -> {fn.name}")
    for cr in cross:
        t = f"CROSS {cr['g'].individual_id}->{cr['p'].individual_id} | matched={cr['matched']} inliers={cr['inl']}"
        c = draw_pair(load_img(cr["g"].crop_path), load_img(cr["p"].crop_path), cr["pa"], cr["pb"], cr["mask"], t)
        fn = OUT / f"cross_{cr['g'].individual_id}_vs_{cr['p'].individual_id}_{cr['inl']}inl.jpg"
        cv2.imwrite(str(fn), c); canv.append(c)
        index["cross"].append({"gallery": cr["g"].individual_id, "probe": cr["p"].individual_id,
                               "matched": cr["matched"], "inliers": cr["inl"], "file": fn.name})
        print(f"  CROSS {cr['g'].individual_id}->{cr['p'].individual_id}: matched={cr['matched']} inliers={cr['inl']} -> {fn.name}")

    m = montage(canv)
    if m is not None:
        cv2.imwrite(str(OUT / "interpret_montage.jpg"), m)
    (OUT / "interpret_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    same_inl = [s["inl"] for s in sel]; cross_inl = [c["inl"] for c in cross]
    print(f"\nsame inliers медиана {int(np.median(same_inl))} (мин {min(same_inl)}–макс {max(same_inl)}) | "
          f"cross inliers {cross_inl} → контраст виден: {'ДА' if (cross_inl and max(cross_inl) < np.median(same_inl)) else 'проверить глазами'}")
    print(f"✅ artifacts/interpret/ ({len(sel)} same + {len(cross)} cross) + interpret_montage.jpg (test не вскрывался)")


if __name__ == "__main__":
    main()
