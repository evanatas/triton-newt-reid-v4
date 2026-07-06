"""Часть 7 — ДИАГНОСТИКА learned-матчеров (белое пятно №1 ядра).

Вердикт «learned отброшены» НЕ подтверждён dev-числами: ab_matchers_dev.json = только SIFT,
learned_log.txt завис на стадии скоринга (время/инженерия, не качество), «силуэт» — несохранённый смоук.
Здесь — один честный АДАПТИВНЫЙ прогон ALIKED/DISK+LightGlue на dev, чтобы решить по пред-зарегистрированному
правилу, остаётся ли ядром SIFT+affine или его бьёт learned.

Адаптивно (дёшево→строго):
  1) стоимость/девайс: замер сек/пара на MPS и CPU → выбрать быстрее, оценить полную сетку;
  2) оверлей: нарисовать ALIKED-матчи на 2 same-individual temporal + 1 cross-individual паре → «узор vs силуэт»;
  3) R@1: подвыборка probe под бюджет (или полная, если влезает) × gallery; McNemar vs SIFT+affine на ТОМ ЖЕ срезе.
Безопасность: фон, wall-clock-guard на матчер, чекпоинт частичной S каждые N строк (таймаут даёт частичные числа).
Сравнение строго single-ref (как исходный A/B), не rolling — apples-to-apples. Всё на DEV (test ЗАПЕЧАТАН).

Запуск (в фоне):  python learned_probe.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
for _p in ("..", "../segment", "../registry", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
from sift_eval import recall_from_scores          # noqa: E402  (S, p_ids, g_ids) -> {k: bool[]}
from embed_eval import wilson, KS                 # noqa: E402
from sealed import assert_unsealed                # noqa: E402
import matchers as M                              # noqa: E402
import deform as D                                # noqa: E402
from ab_matchers import load_pool, mcnemar        # noqa: E402  пул и парный тест — переиспуем как есть

CROPS = (HERE / ".." / "segment").resolve()
ART = HERE / "artifacts"
DEFORM = "affine"          # деформ-победитель SIFT-ветки — фиксируем для apples-to-apples
TOPK = 512                 # кэп keypoints против O(N^2) LightGlue (причина прошлого зависания)
DISK_N = 512               # бюджет DISK-детектора
WALL_PER_MATCHER = 3000    # сек на матчер: при превышении — стоп с частичными числами (DISK ~463мс/пара → полная 97×46 ~34мин)
SIFT_AFFINE_REF = 0.588    # эталон из ab_matchers_dev.json (single-ref dev)


# ───────────────────── утилиты ─────────────────────
def imgs_of(rows):
    return [cv2.cvtColor(cv2.imread(str(CROPS / r.crop_path)), cv2.COLOR_BGR2RGB) for _, r in rows.iterrows()]


def cap(feat):
    """Усечь фичи до TOPK (анти-зависание). ALIKED при detection_threshold=0 даёт тысячи точек."""
    if "kp" in feat:                                   # learned: {"kp","desc","hw"}
        k = feat["kp"]
        if len(k) > TOPK:
            feat = {**feat, "kp": k[:TOPK], "desc": feat["desc"][:TOPK]}
    return feat


def extract_all(m, images, cap_feats=True):
    out = []
    for im in images:
        f = m.extract(im)
        out.append(cap(f) if cap_feats else f)
    return out


def measure_pair_cost(m, fp, fg, n=10):
    """Средняя сек/пара matched()+verify на первых n парах — для оценки полной сетки и выбора девайса."""
    t0 = time.time()
    cnt = 0
    for i in range(min(n, len(fp))):
        pa, pb = m.matched(fp[i], fg[i % len(fg)])
        D.verify(pa, pb, DEFORM)
        cnt += 1
    return (time.time() - t0) / max(cnt, 1)


def score_matrix_guarded(m, fp, fg, tag, budget):
    """Как ab_matchers.score_matrix, но: прогресс, чекпоинт частичной S, wall-clock-guard.
    Возврат: (S, n_done) — заполнены строки [:n_done]; остальное = -1 (probe не успели)."""
    P, G = len(fp), len(fg)
    S = np.full((P, G), -1.0, np.float32)
    t0 = time.time()
    n_done = 0
    for i in range(P):
        for j in range(G):
            pa, pb = m.matched(fp[i], fg[j])
            S[i, j] = D.verify(pa, pb, DEFORM)
        n_done = i + 1
        if i % 5 == 0:
            np.save(ART / f"_partial_{tag}.npy", S)
            print(f"  [{tag}] строка {n_done}/{P}  ⏱{time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > budget:
            print(f"  [{tag}] ⏱ бюджет {budget}s исчерпан на {n_done}/{P} — стоп с частичными", flush=True)
            break
    return S, n_done


def recall_block(S, p_ids, g_ids, n_done):
    """recall@k + Wilson CI на заполненных строках [:n_done]."""
    Su, pu = S[:n_done], p_ids[:n_done]
    hits = recall_from_scores(Su, pu, g_ids)
    res = {"n": int(n_done)}
    for k in KS:
        v, lo, hi = wilson(int(hits[k].sum()), n_done)
        res[f"recall@{k}"] = {"value": round(v, 3), "ci": [round(lo, 2), round(hi, 2)]}
    return res, hits


def draw_overlay(imgA, imgB, ptsA, ptsB, inliers, title, path):
    hA, wA = imgA.shape[:2]; hB, wB = imgB.shape[:2]
    H = max(hA, hB)
    canvas = np.zeros((H + 22, wA + wB, 3), np.uint8)
    canvas[22:22+hA, :wA] = cv2.cvtColor(imgA, cv2.COLOR_RGB2BGR)
    canvas[22:22+hB, wA:wA+wB] = cv2.cvtColor(imgB, cv2.COLOR_RGB2BGR)
    rng = np.random.default_rng(0)
    for (xa, ya), (xb, yb) in zip(ptsA, ptsB):
        c = tuple(int(v) for v in rng.integers(60, 255, 3))
        cv2.line(canvas, (int(xa), int(ya)+22), (int(xb)+wA, int(yb)+22), c, 1)
    cv2.putText(canvas, f"{title} | matched={len(ptsA)} affine_inliers={inliers}",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(path), canvas)


# ───────────────────── основной прогон ─────────────────────
def run_matcher(name, gallery, probe, g_imgs, p_imgs, p_idx, sift_hits, t_overlay=False):
    """Полный диагностический прогон одного learned-матчера на срезе probe[p_idx]."""
    print(f"\n=== {name} ===", flush=True)
    if name == "aliked":
        m = M.LightGlueLearned("aliked")
    elif name == "disk":
        m = M.LightGlueLearned("disk", n_features=DISK_N)
    else:
        raise ValueError(name)

    # sanity-гейт: матчер обязан матчить кадр сам с собой (>0). 0 → интеграция сломана (ALIKED+LightGlue в kornia 0.8).
    f0 = cap(m.extract(g_imgs[0]))
    sm = len(m.matched(f0, f0)[0])
    if sm == 0:
        print(f"  ⚠ {name}: self-match=0 → интеграция сломана в kornia (НЕ вердикт о качестве), пропуск.", flush=True)
        return {"broken": "self-match=0: LightGlue не распознаёт дескрипторы (kornia 0.8 ALIKED-integration)",
                "self_match": 0}
    print(f"  sanity self-match={sm} ✓", flush=True)

    t0 = time.time()
    fg = extract_all(m, g_imgs)
    fp_full = extract_all(m, p_imgs)
    print(f"  extract: gallery {len(fg)} + probe {len(fp_full)} за {time.time()-t0:.0f}s "
          f"(kp≈{len(fg[0].get('kp', []))})", flush=True)

    # 1) стоимость/девайс
    cost = measure_pair_cost(m, fp_full, fg)
    est_full = cost * len(p_idx) * len(fg)
    print(f"  стоимость: {cost*1000:.0f} мс/пара ({M.DEVICE}) → полная сетка "
          f"{len(p_idx)}×{len(fg)} ≈ {est_full:.0f}s", flush=True)

    # 2) оверлей узор-vs-силуэт (только один раз, для ALIKED)
    overlays = []
    if t_overlay:
        overlays = make_overlays(name, m, gallery, probe, g_imgs, p_imgs, fg, fp_full)

    # 3) R@1 на срезе p_idx (подвыборка уже выбрана снаружи)
    fp = [fp_full[i] for i in p_idx]
    S, n_done = score_matrix_guarded(m, fp, fg, name, WALL_PER_MATCHER)
    p_ids = probe.individual_id.to_numpy()[p_idx]
    g_ids = gallery.individual_id.to_numpy()
    res, l_hits = recall_block(S, p_ids, g_ids, n_done)
    res["sec"] = round(time.time() - t0, 1)
    res["ms_per_pair"] = round(cost * 1000, 1)
    res["device"] = M.DEVICE
    res["topk"] = TOPK
    res["overlays"] = overlays
    # McNemar vs SIFT+affine на ТЕХ ЖЕ probe (выровнено по n_done)
    b, c, p = mcnemar(sift_hits[1][:n_done], l_hits[1][:n_done])
    res["mcnemar_vs_sift_affine@1"] = {"b": b, "c": c, "p": round(p, 4)}
    print(f"  R@1 {res['recall@1']['value']:.3f}{res['recall@1']['ci']}  "
          f"R@5 {res['recall@5']['value']:.3f}  R@10 {res['recall@10']['value']:.3f}  "
          f"McNemar vs SIFT b={b} c={c} p={p:.3f}", flush=True)
    return res


def make_overlays(label, m, gallery, probe, g_imgs, p_imgs, fg, fp):
    """2 same-individual temporal пары + 1 cross-individual: рисуем learned-матчи (label = матчер)."""
    g_ids = gallery.individual_id.to_numpy()
    p_ids = probe.individual_id.to_numpy()
    common = [u for u in dict.fromkeys(p_ids) if u in set(g_ids)]
    out = []
    pairs = []
    for u in common[:2]:                              # 2 same-individual
        gi = int(np.where(g_ids == u)[0][0]); pi = int(np.where(p_ids == u)[0][0])
        pairs.append(("SAME", u, u, gi, pi))
    if len(common) >= 2:                              # 1 cross-individual
        gi = int(np.where(g_ids == common[0])[0][0]); pi = int(np.where(p_ids == common[1])[0][0])
        pairs.append(("CROSS", common[0], common[1], gi, pi))
    for kind, ga, pb_id, gi, pi in pairs:
        pa, pbp = m.matched(fg[gi], fp[pi])
        inl = D.verify(pa, pbp, DEFORM)
        fn = ART / f"learned_overlay_{label}_{kind}_{ga}_vs_{pb_id}.jpg"
        draw_overlay(g_imgs[gi], p_imgs[pi], pa, pbp, inl, f"{label} {kind} {ga}->{pb_id}", fn)
        out.append({"kind": kind, "gallery": ga, "probe": pb_id, "matched": len(pa),
                    "affine_inliers": int(inl), "file": fn.name})
        print(f"  оверлей {kind} {ga}->{pb_id}: matched={len(pa)} inliers={inl} -> {fn.name}", flush=True)
    return out


def main():
    assert_unsealed("dev", unseal=False)
    ART.mkdir(exist_ok=True)
    gallery, probe = load_pool()
    g_imgs = imgs_of(gallery)
    p_imgs = imgs_of(probe)
    print(f"DIAГ learned на DEV: gallery {len(gallery)} | probe {len(probe)} | test ЗАПЕЧАТАН", flush=True)

    # SIFT+affine эталон на ТОМ ЖЕ пуле (для McNemar и сравнения) — быстро
    sift = M.build("sift")
    sfg = [sift.extract(im) for im in g_imgs]
    sfp = [sift.extract(im) for im in p_imgs]
    Ss = np.zeros((len(sfp), len(sfg)), np.float32)
    for i, fpi in enumerate(sfp):
        for j, fgj in enumerate(sfg):
            Ss[i, j] = D.verify(*sift.matched(fpi, fgj), DEFORM)
    g_ids = gallery.individual_id.to_numpy()
    p_ids = probe.individual_id.to_numpy()
    sift_hits = recall_from_scores(Ss, p_ids, g_ids)
    sift_r1 = float(sift_hits[1].mean())
    print(f"SIFT+affine эталон на пуле: R@1 {sift_r1:.3f} (артефакт ab_matchers = {SIFT_AFFINE_REF})", flush=True)

    # выбор среза probe под бюджет: оценим стоимость ALIKED на 10 парах ниже в run_matcher;
    # здесь — берём ПОЛНЫЙ probe; внутри score_matrix_guarded бюджет сам обрежет, если не влезет.
    p_idx = list(range(len(probe)))

    out = {"protocol": "single-ref dev (как ab_matchers)", "deform": DEFORM,
           "sift_affine_ref": {"recall@1_pool": round(sift_r1, 3), "artifact": SIFT_AFFINE_REF},
           "matchers": {}}

    # DISK+LightGlue — рабочий learned-путь (overlay узор-vs-силуэт здесь). ALIKED→LightGlue в kornia 0.8 даёт self-match=0.
    out["matchers"]["disk"] = run_matcher("disk", gallery, probe, g_imgs, p_imgs, p_idx, sift_hits, t_overlay=True)
    # ALIKED — sanity-гейт отметит сломанность интеграции (документируем, не блокирует вывод по DISK)
    out["matchers"]["aliked"] = run_matcher("aliked", gallery, probe, g_imgs, p_imgs, p_idx, sift_hits)

    # ── ВЕРДИКТ по пред-зарегистрированному правилу ──
    verdict = decide(out, sift_r1)
    out["verdict"] = verdict
    (ART / "learned_probe_dev.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n" + "=" * 64)
    print("ВЕРДИКТ:", verdict["decision"])
    print(verdict["rationale"])
    print("=" * 64)
    print("✅ artifacts/learned_probe_dev.json + learned_overlay_*.jpg (test не вскрывался)", flush=True)


def decide(out, sift_r1):
    """Правило: learned ядром ТОЛЬКО если бьёт SIFT по McNemar p<0.05 И качественно матчит узор.
    Иначе SIFT остаётся (проще, быстрее, rotation-invariant, валиден)."""
    best, best_r1, best_p = None, -1, 1.0
    for name, r in out["matchers"].items():
        if "recall@1" not in r:
            continue
        if r["recall@1"]["value"] > best_r1:
            best, best_r1 = name, r["recall@1"]["value"]
            best_p = r["mcnemar_vs_sift_affine@1"]["p"]
    if best is None:
        return {"decision": "SIFT остаётся (ядро)",
                "rationale": "learned не дали чисел в бюджете — задокументировать стоимость, ядро SIFT+affine+rolling."}
    beats = best_r1 > sift_r1 and best_p < 0.05
    if beats:
        return {"decision": f"КАНДИДАТ: {best}+LightGlue (требует подтверждения оверлеем: узор, не силуэт)",
                "rationale": f"{best} R@1 {best_r1:.3f} > SIFT {sift_r1:.3f}, McNemar p={best_p:.3f}<0.05. "
                             f"СВЕРИТЬ learned_overlay_*.jpg: матчи на узоре брюшка, а не на силуэте. "
                             f"Если узор — эскалировать (тюнинг learned). Если силуэт — отклонить."}
    return {"decision": "SIFT+affine+rolling остаётся ядром",
            "rationale": f"Лучший learned ({best}) R@1 {best_r1:.3f} vs SIFT {sift_r1:.3f}, McNemar p={best_p:.3f}. "
                         f"Не бьёт значимо → SIFT проще/быстрее/rotation-invariant. learned отклонён С ЧИСЛАМИ."}


if __name__ == "__main__":
    main()
