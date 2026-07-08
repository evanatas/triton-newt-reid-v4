"""Streamlit-демо «NewtID» (4.0) — опознание тритона по узору брюшка. UX как в 3.0-демо, ядро — SIFT+affine+rolling.

Загрузка/выбор фото → top-K похожих ОСОБЕЙ карточками (миниатюра + номер + «уверенность %» + бейдж «Лучшее
совпадение») → known/new. Дифференциатор: опциональный side-by-side ОВЕРЛЕЙ совпавших пятен (top-1).

Запуск:  streamlit run backend/demo.py   (или preview «triton-demo», порт 8501)
Логика — в reid_service.ReIDService (то же замороженное ядро, что дало sealed top-1 0.79).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from reid_service import ReIDService  # noqa: E402

st.set_page_config(page_title="NewtID — re-ID тритонов по узору брюшка", page_icon="🦎", layout="wide")


@st.cache_resource(show_spinner="Загрузка ядра и каталога особей…")
def get_svc():
    return ReIDService("TK")


svc = get_svc()

st.title("🦎 NewtID — реидентификация тритонов по узору брюшка")
st.caption("ВКР «Прикладной ИИ» (ТГУ) · open-set re-ID по узору пятен брюшка. Ядро — локальные признаки "
           "**SIFT + affine-деформ + rolling** (не global-эмбеддинг): top-K похожих особей карточками + «уверенность %». "
           "Финальный sealed-KPI top-1 **0.79**. Offline, локально.")

tab_id, tab_ind, tab_about = st.tabs(["🔎 Опознание", "🗂 Особи (галерея)", "ℹ️ О системе"])

# ═══════════════ ВКЛАДКА: ОПОЗНАНИЕ ═══════════════
with tab_id:
    c_ctrl, c_query = st.columns([2, 1])
    with c_ctrl:
        source = st.radio("Источник запроса", ["Выбрать пробу (известная особь)", "Загрузить фото"], horizontal=True)
        topk = st.slider("Сколько кандидатов показать (top-K)", 3, 10, 5)

    query_crop = None       # RGB uint8
    query_feat = None
    true_id = None
    exclude = None          # исключить кадр пробы из каталога (не матчить сам с собой)

    if source.startswith("Выбрать"):
        labels = [f"{iid} · {fid}" for fid, iid in svc.probes]
        i = st.selectbox("Проба (известная особь — проверяем, найдёт ли система по ДРУГОМУ снимку)",
                         range(len(svc.probes)), format_func=lambda k: labels[k])
        fid, true_id = svc.probes[i]
        query_crop = svc.imgs[fid]
        query_feat = svc.feats[fid]
        exclude = fid
    else:
        up = st.file_uploader("Фото брюшка (JPG/PNG) — held-out примеры в папке демо_фото/ "
                              "(01_известные / 02_новые_особи)", type=["jpg", "jpeg", "png"])
        is_crop = st.checkbox("Это уже готовый кроп брюшка (не запускать сегментацию)", value=False)
        if up is not None:
            import hashlib
            data = up.read()
            exclude = svc.frame_for_md5(data)      # если это фото уже в каталоге — исключить (анти-самосовпадение)
            key = hashlib.md5(data).hexdigest() + ("_crop" if is_crop else "_seg")
            cache = st.session_state.get("_upload_cache")
            if cache and cache.get("key") == key:
                query_crop = cache["crop"]
            else:
                buf = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if buf is None:
                    st.error("Не удалось прочитать файл — повреждён или не JPG/PNG.")
                else:
                    raw = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
                    if is_crop:
                        query_crop = raw
                    else:
                        with st.spinner("Сегментация брюшка (BiRefNet) + кроп…"):
                            query_crop, _ = svc.segment_upload(data)
                    st.session_state["_upload_cache"] = {"key": key, "crop": query_crop}
            if query_crop is not None:
                query_feat = svc.sift.extract(query_crop)

    with c_query:
        if query_crop is not None:
            cap = "Запрос (кроп брюшка)" + (f" · истинно {true_id}" if true_id else "")
            st.image(query_crop, caption=cap, width="stretch")

    if query_feat is not None:
        excl = svc.session_of(exclude)              # честный temporal-тест: исключаем ВСЮ сессию запроса
        ranked = svc.rank(query_feat, topk=topk, exclude_frame=excl)
        v = svc.verdict(ranked)
        if v["verdict"] == "known":
            st.success(f"✅ Особь в базе: **{ranked[0]['individual_id']}** · уверенность **{v['confidence']:.0f}%** "
                       f"(отрыв от №2: {v['margin']:.0f} п.п.)")
        elif v["verdict"] == "review":
            st.warning(f"🟡 На проверку оператору: похоже на **{ranked[0]['individual_id']}** · уверенность "
                       f"{v['confidence']:.0f}% (отрыв {v['margin']:.0f} п.п.) — сигнал есть, но для однозначного "
                       f"«известна» недостаточно (human-in-the-loop).")
        else:
            st.error(f"🔴 Вероятно НОВАЯ особь · макс. уверенность {v['confidence']:.0f}% — ни один кандидат "
                     f"не набрал достаточного сходства (кандидат на регистрацию).")
        st.caption("«Уверенность» — калиброванная монотонная шкала (НЕ вероятность): своя особь → высокая, "
                   "чужая/новая → низкая. Надёжный сигнал — РАНЖИР top-1/top-5 (на нём sealed-KPI 0.79). "
                   "Вердикт **известна / на проверку / новая** — в шкале уверенности (UX-подушка human-in-the-loop); "
                   "метрический open-set меряется отдельно в affine-inliers (порог 9, sealed BAKS 0.67 / G 0.819, "
                   "within-TK). Две шкалы не смешивать.")
        if exclude:
            st.caption(f"ℹ️ Честный temporal-тест: из каталога исключён не только сам кадр ({exclude}), но и "
                       f"ВСЯ его съёмочная сессия ({len(excl)} кадров того же дня) — узнавание идёт по снимкам "
                       f"ДРУГИХ месяцев, а не по почти-дубликату того же дня.")

        rev = svc.cat[svc.cat.frame_id == exclude] if exclude else None
        reveal_iid = true_id or (rev.iloc[0].individual_id if rev is not None and not rev.empty else None)
        if reveal_iid:                                # открытая сверка top-1 с истиной + temporal-интервал
            top = ranked[0]; ok = top["individual_id"] == reveal_iid
            qd, md = svc.date_of(exclude), svc.date_of(top["best_frame"])
            mo = ""
            try:
                if qd and md:
                    d = abs((int(qd[:4]) - int(md[:4])) * 12 + int(qd[5:7]) - int(md[5:7]))
                    mo = f" спустя {d} мес" if d else " (тот же месяц)"
            except Exception:
                pass
            (st.success if ok else st.error)(
                f"{'✓ Верно' if ok else '✗ Мимо'}: истинная особь {reveal_iid} → top-1 {top['individual_id']}"
                + (f"; узнана по снимку {md}{mo} — это и есть temporal re-id" if ok and md else ""))

        if source.startswith("Загрузить"):        # демо флоу учёта: регистрация новой особи (в сессии, без БД)
            with st.expander("➕ Зарегистрировать эту особь в базе (демо флоу учёта новых особей)"):
                st.caption("Так выглядит учёт: если особи нет в базе (вердикт «новая») — оператор регистрирует её. "
                           "Здесь особь добавляется в каталог на время сессии (в проде — запись в БД SQLite→PostgreSQL). "
                           "После регистрации загрузите ДРУГОЙ снимок этой особи — система её узнает.")
                nm = st.text_input("Имя / ID особи", value=f"NEW-{len(st.session_state.get('_reg', [])) + 1}",
                                   key="reg_nm")
                if st.button("Зарегистрировать в базе", key="reg_go"):
                    info = svc.register(query_crop, query_feat, nm)
                    st.session_state.setdefault("_reg", []).append(info["individual_id"])
                    st.success(f"✅ Особь **{info['individual_id']}** добавлена в каталог (известных особей: "
                               f"{svc.n_individuals}). Теперь загрузите её ДРУГОЙ снимок — система узнает её.")

        st.subheader(f"Top-{len(ranked)} похожих особей")
        per_row = 4
        for r0 in range(0, len(ranked), per_row):
            cols = st.columns(per_row)
            for col, rk in zip(cols, range(r0, min(r0 + per_row, len(ranked)))):
                r = ranked[rk]
                with col:
                    st.image(svc.imgs[r["best_frame"]], width="stretch")
                    badge = "🏆 Лучшее совпадение" if rk == 0 else f"№{rk + 1}"
                    hit = " ✓" if true_id and r["individual_id"] == true_id else ""
                    st.markdown(f"**{r['individual_id']}**{hit} · {badge}")
                    st.progress(min(int(r["confidence"]), 100), text=f"уверенность {r['confidence']:.0f}%")
                    st.caption(f"вид {r['cohort']} · фото в базе: {r['n_photos']} · inliers {r['score']}")

# ═══════════════ ВКЛАДКА: ОСОБИ ═══════════════
with tab_ind:
    st.subheader("База известных особей — кадры по сессиям (перепоимки во времени)")
    uniq = sorted(set(svc.cat.individual_id))
    date_counts = svc.cat.groupby("individual_id").date.nunique()
    recap_ids = sorted(date_counts[date_counts >= 2].index)
    only_recap = st.checkbox(f"Только перепойманные — сняты в ≥2 сессиях ({len(recap_ids)} из {len(uniq)} особей)",
                             value=True)
    sel = st.selectbox("Особь", recap_ids if only_recap else uniq)
    rows = svc.cat[svc.cat.individual_id == sel].sort_values("date")
    dates = list(sorted(rows.date.dropna().unique()))
    tag = " · 🔁 перепоймана (temporal re-id)" if len(dates) >= 2 else " · один снимок"
    st.markdown(f"**{sel}** — {len(rows)} фото в **{len(dates)} сессиях** ({', '.join(map(str, dates)) or '—'}){tag}")
    cols = st.columns(min(len(rows), 6) or 1)
    for n, (_, r) in enumerate(rows.iterrows()):
        d = r.date if (isinstance(r.date, str) and r.date) else "без даты"
        role = {"gallery": "галерея", "probe": "проба"}.get(r.split_role, str(r.split_role))
        cols[n % len(cols)].image(svc.imgs[r.frame_id], caption=f"{d} · {role}", width="stretch")

    if len(dates) >= 2:                                     # живая проверка temporal re-id
        st.markdown("##### 🔬 Проверка перепоимки: поздний снимок → ищем особь по РАННИМ")
        late = rows[rows.date == dates[-1]].iloc[0]
        early = rows[rows.date == dates[0]].iloc[0]
        ranked = svc.rank(svc.feats[late.frame_id], topk=3, exclude_frame=svc.session_of(late.frame_id))
        hit = bool(ranked) and ranked[0]["individual_id"] == sel
        inl = svc.pair_score(late.frame_id, early.frame_id)
        try:
            months = (int(dates[-1][:4]) - int(dates[0][:4])) * 12 + (int(dates[-1][5:7]) - int(dates[0][5:7]))
        except Exception:
            months = "?"
        msg = (f"Запрос — снимок **{dates[-1]}**; в галерее особи только более ранние. top-1 = "
               f"**{ranked[0]['individual_id']}** ({'✓ та же особь' if hit else '✗ мимо'}) · "
               f"уверенность {ranked[0]['confidence']:.0f}%. Совпавших пятен (affine-inliers) со снимком "
               f"**{dates[0]}** (спустя {months} мес): **{inl}**.")
        (st.success if hit else st.warning)(msg)
        c1, c2 = st.columns(2)
        c1.image(svc.imgs[early.frame_id], caption=f"ранний · {dates[0]}", width="stretch")
        c2.image(svc.imgs[late.frame_id], caption=f"поздний · {dates[-1]} (запрос)", width="stretch")
        st.caption("Это и есть temporal re-id: узор пятен растянулся между съёмками, но ядро (SIFT+affine) "
                   "матчит особь по совпавшим пятнам. На этом стоит sealed-KPI 0.79.")

# ═══════════════ ВКЛАДКА: О СИСТЕМЕ ═══════════════
with tab_about:
    st.subheader("Архитектура и честные метрики")
    st.markdown(
        "**Пайплайн:** фото → сегментация брюшка (**BiRefNet**) → кроп + нормализация масштаба + маскирование метки → "
        "gray+CLAHE → **SIFT** (локальные признаки) → **affine-деформ-верификация** (`estimateAffine2D`, допускает "
        "растяжение узора) → **rolling multi-reference** (max-pool по особи) → top-K + tri-state **известна / на проверку / новая**.\n\n"
        "Разворот против версии 3.0: не global-эмбеддинг (он провалился), а **локальный признаковый матчинг**."
    )
    st.markdown("**Финальные числа ВКР — sealed-test (held-out, вскрыт РОВНО один раз, 2026-07-02):**")
    st.table({
        "метод": ["ЯДРО SIFT+affine (rolling)", "ЯДРО single-ref (нижняя граница)",
                  "MegaDescriptor-L-384 (= 3.0)", "DINOv2 ViT-L", "версия 3.0 (финал)"],
        "top-1": ["0.790", "0.700", "0.30", "0.17", "0.188"],
        "top-5": ["0.920", "0.850", "—", "—", "0.386"],
    })
    st.success("🎯 **ЦЕЛЬ top-1 ≥ 0.75 ВЗЯТА** на запечатанном тесте (rolling 0.79 [Wilson 0.70–0.86]). "
               "×4.2 над финалом 3.0 (0.188).")
    st.info(
        "**Тезис ВКР (подтверждён на held-out):** local-feature ядро бьёт global-embedding ~×3 "
        "(0.79 vs Mega 0.30 / DINOv2 0.17, McNemar p<0.001) на ТЕХ ЖЕ кропах и протоколе. Провал 3.0 — "
        "от парадигмы global-эмбеддинга, а не от кропов. Рычаги: affine-деформ + rolling multi-reference. "
        "test ≈ dev (переобучения нет). KPI подаётся раздельно: 0.79 — режим накапливаемого каталога (rolling), "
        "0.70 — нижняя граница при единственном прежнем снимке."
    )
    st.caption("Стек: Python · OpenCV (SIFT/RANSAC) · BiRefNet (сегментация) · PyTorch/MPS · Streamlit. "
               "Каталог демо = TK-кропы известных особей в памяти (open_new исключены; БД — следующий этап); "
               "held-out фото — `демо_фото/`. Метрики — `core/core_config.json`, "
               "`РЕЗУЛЬТАТ_ФИНАЛ_sealed.md`.")
