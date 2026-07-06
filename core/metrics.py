"""Часть 7 / R2 — метрики честности: macro-average recall, verification (FRR/FAR/TAR), open-set (BAKS/BAUS/G).

Переиспуется в Части 11. Все на dev (test ЗАПЕЧАТАН до финала).
"""
from __future__ import annotations

import numpy as np

KS = (1, 5, 10)


def macro_recall(hits: dict, p_ids: np.ndarray, ks=KS) -> dict:
    """Per-identity macro-average: особь с 5 пробами весит 1 (среднее её проб), не ×5.
    hits[k] — bool-массив по probe-кадрам. -> {k: macro recall}."""
    uniq = np.unique(p_ids)
    return {k: float(np.mean([hits[k][p_ids == u].mean() for u in uniq])) for k in ks}


def micro_recall(hits: dict, ks=KS) -> dict:
    """Per-probe (как сейчас): каждый probe-кадр весит 1."""
    return {k: float(hits[k].mean()) for k in ks}


def frr_at_k(hits: dict, ks=KS) -> dict:
    """False Rejection Rate@k = доля known-проб, где истинная особь НЕ в top-k. = 1 - recall@k.
    Язык герпетологов/CMR (AmphIdent, PLOS One)."""
    return {k: float(1.0 - hits[k].mean()) for k in ks}


def open_set(scores_known: np.ndarray, scores_new: np.ndarray, thr: float) -> dict:
    """Open-set по порогу score (max-сходство с галереей).
    known принимается если score≥thr; new должен быть отвергнут (score<thr).
    TAR=доля known принятых; FAR=доля new ошибочно принятых; BAKS/BAUS/G."""
    tar = float((scores_known >= thr).mean()) if len(scores_known) else float("nan")   # = 1-FRR(threshold)
    far = float((scores_new >= thr).mean()) if len(scores_new) else float("nan")       # ложно-известные
    baks = tar                                  # balanced accuracy known = доля верно-удержанных known
    baus = 1.0 - far                            # balanced accuracy unknown = доля верно-отвергнутых new
    g = float(np.sqrt(max(baks, 0) * max(baus, 0)))
    return {"threshold": round(thr, 4), "TAR": round(tar, 3), "FAR": round(far, 3),
            "FRR": round(1 - tar, 3), "BAKS": round(baks, 3), "BAUS": round(baus, 3), "G": round(g, 3)}


def youden_threshold(scores_known: np.ndarray, scores_new: np.ndarray) -> float:
    """Порог по индексу Юдена (max TAR-FAR) на dev. В Части 11 применяется фиксированным."""
    cand = np.unique(np.concatenate([scores_known, scores_new]))
    best, bt = -1, float(np.median(cand))
    for t in cand:
        j = (scores_known >= t).mean() - (scores_new >= t).mean()
        if j > best:
            best, bt = j, t
    return float(bt)


def report(hits: dict, p_ids: np.ndarray, ks=KS) -> dict:
    """Сводка retrieval-метрик (micro+macro+FRR) для closed-set."""
    return {"micro_recall": micro_recall(hits, ks), "macro_recall": macro_recall(hits, p_ids, ks),
            "frr": frr_at_k(hits, ks), "n_probe": int(len(p_ids)), "n_identity": int(len(np.unique(p_ids)))}
