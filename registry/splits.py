"""Часть 3 — temporal-протокол и сплиты. Детерминированно по individual_id (не по кадру — долг 3.0)."""
from __future__ import annotations
import hashlib

import numpy as np
import pandas as pd

SEED = 42


def frac(key: str, seed: int = SEED) -> float:
    """Детерминированное число в [0,1) от строки-ключа (blake2b)."""
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def _tkey(row):
    """Временная метка сессии по когорте. None → кадр без валидного времени (исключается из temporal).
    TK/LAB/GCN — только календарная дата (session как время НЕ используем); PW — session-код.
    F-12 (осознанное допущение): гранулярность TK-времени = YYYY-MM (месяц). Защита от «та же съёмка в gallery и
    probe» опирается на распределение данных (gallery=2024-12, probe=2025-01+, фактических утечек 0), а НЕ на явный
    session-барьер. Если появятся особи с ≥2 съёмками в ОДИН месяц — вернуться к session-ключу внутри месяца."""
    cohort = row.get("cohort")
    if cohort in ("TK", "LAB", "GCN"):
        d = row.get("date")
        return str(d) if (pd.notna(d) and d != "") else None
    s = row.get("session")                      # PW: псевдо-время по session
    return str(s) if pd.notna(s) else None


def assign_temporal_roles(df: pd.DataFrame) -> pd.DataFrame:
    """gallery = самая ранняя сессия особи; probe = поздние. Кадры без валидного времени → excluded_nodate."""
    df = df.copy()
    df["tkey"] = df.apply(_tkey, axis=1)
    role = pd.Series("gallery", index=df.index)
    interval = pd.Series(0.0, index=df.index)
    for ind, g in df.groupby("individual_id"):
        valid = g[g["tkey"].notna()]
        if len(valid) == 0:
            role.loc[g.index] = "excluded_nodate"
            continue
        sessions = sorted(valid["tkey"].unique())
        early = sessions[0]
        ey, em = _ym(early)
        for idx, r in g.iterrows():
            if pd.isna(r["tkey"]):                      # NaN (pandas конвертит None→NaN) — кадр без даты
                role.loc[idx] = "excluded_nodate"      # опечатка в дате — не в temporal
                continue
            if r["tkey"] == early:
                continue                                # gallery
            role.loc[idx] = "probe"
            ry, rm = _ym(r["tkey"])
            if ey and ry:                               # обе — валидные даты
                interval.loc[idx] = (ry - ey) * 12 + (rm - em)
            else:                                       # PW session-коды
                interval.loc[idx] = _session_gap(early, r["tkey"])
    df["split_role"] = role
    df["interval_months"] = interval
    return df


def _ym(tkey: str):
    """('2025-01'|'2025-01-23') -> (year, month); session-код -> (0,0)."""
    parts = str(tkey).split("-")
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
        return int(parts[0]), int(parts[1])
    return 0, 0


def _session_gap(a: str, b: str) -> float:
    try:
        return float(int(b) - int(a))   # PW session 1->2
    except Exception:
        return 0.0


def assign_eval_folds(df: pd.DataFrame, test_frac: float = 0.5, seed: int = SEED) -> pd.DataFrame:
    """Fold по individual_id (целая особь в один fold). TK temporal→dev/test; non-temporal→distractor;
    GCN→train; PW/LAB→aux (раздельные срезы). Детерминированно."""
    df = df.copy()
    fold = pd.Series("", index=df.index)
    for ind, g in df.groupby("individual_id"):
        cohort = g["cohort"].iloc[0]
        has_recap = bool(g["has_recapture"].iloc[0])
        if cohort == "GCN":
            f = "train"
        elif cohort in ("PW", "LAB"):
            f = "aux"
        elif cohort == "TK":
            if not has_recap:
                f = "distractor"            # 1 дата → только дистрактор в обеих eval-галереях
            else:
                f = "test" if frac(f"fold:{ind}", seed) < test_frac else "dev"
        else:
            f = "train"
        fold.loc[g.index] = f
    df["split_fold"] = fold
    return df


def assign_open_set(df: pd.DataFrame, open_frac: float = 0.15, seed: int = SEED) -> pd.DataFrame:
    """~open_frac особей каждого eval-fold помечаются open_new: их gallery-кадры убираются из галереи
    (роль probe только) → их probe = «новые» запросы. Раздельно dev/test."""
    df = df.copy()
    is_open = pd.Series(False, index=df.index)
    for ind, g in df.groupby("individual_id"):
        if g["split_fold"].iloc[0] not in ("dev", "test"):
            continue
        if frac(f"open:{ind}", seed) < open_frac:
            is_open.loc[g.index] = True
    df["is_open_new"] = is_open
    # open_new: gallery-кадры исключаем из галереи (помечаем role=probe_open, чтобы не попали в gallery)
    mask = df["is_open_new"] & (df["split_role"] == "gallery")
    df.loc[mask, "split_role"] = "open_excluded"   # ни gallery, ни обычный probe
    return df


def mark_train_pool(df: pd.DataFrame) -> pd.DataFrame:
    """in_train_pool: identity-disjoint от TK-test. Для обучаемых моделей Части 7.
    Источники: GCN + LAB + PW + TK-dev (cross-fit). НИКОГДА TK-test/distractor."""
    df = df.copy()
    df["in_train_pool"] = df["split_fold"].isin(["train", "aux", "dev"]) & (df["cohort"] != "TK") \
        | ((df["cohort"] == "TK") & (df["split_fold"] == "dev"))
    return df


def build(df: pd.DataFrame, test_frac: float = 0.5, open_frac: float = 0.15, seed: int = SEED) -> pd.DataFrame:
    df = assign_temporal_roles(df)
    df = assign_eval_folds(df, test_frac, seed)
    df = assign_open_set(df, open_frac, seed)
    df = mark_train_pool(df)
    return df
