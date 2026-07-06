"""Часть 2 — дедуп: md5 (точные дубли) + perceptual-hash (near-duplicate audit). Анти-утечка ДО сплита."""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd


def _survivor_key(path_rel: str) -> tuple:
    """Каноническая метка выживает: без варианта (.k) лучше; 2-значный стем бьёт 1-значный
    (урок спайка: 1.jpg = ошибочная метка для 10.jpg). Меньший ключ — предпочтительнее."""
    stem = Path(path_rel).stem
    m = re.match(r"(\d+)", stem)
    n_digits = len(m.group(1)) if m else 9
    has_variant = bool(re.match(r"^\d+\.\d+$", stem))   # «03.1»
    return (has_variant, 0 if n_digits >= 2 else 1, stem)


def dedup_md5(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Оставить один кадр на md5 (канонический). Вернуть (очищенный df, список выброшенных)."""
    df = df.copy()
    df["_key"] = df["path_rel"].map(_survivor_key)
    df = df.sort_values("_key")
    keep = df.drop_duplicates("md5", keep="first")
    dropped_idx = df.index.difference(keep.index)
    dropped = []
    keep_by_md5 = keep.set_index("md5")
    for _, r in df.loc[dropped_idx].iterrows():
        k = keep_by_md5.loc[r["md5"]]
        dropped.append({"drop": r["path_rel"], "keep": k["path_rel"] if hasattr(k, "__getitem__") else None})
    return keep.drop(columns="_key").reset_index(drop=True), dropped


def phash_audit(df: pd.DataFrame, root: Path, threshold: int = 6) -> list[dict]:
    """Perceptual-hash near-duplicate audit (report-only): почти-одинаковые кадры,
    которые md5 не ловит. Опасны, если попадут в gallery и probe одной особи."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        # R1.4: единый контракт возврата (df, pairs) — иначе распаковка `df, pairs = ...` падает
        return df, [{"error": "imagehash не установлен — pip install imagehash"}]
    hashes = []
    for _, r in df.iterrows():
        try:
            with Image.open(root / r["path_rel"]) as im:
                hashes.append(imagehash.phash(im))
        except Exception:
            hashes.append(None)
    df = df.reset_index(drop=True)
    pairs = []
    # сравнение только внутри одной когорты (между когортами near-dup не интересны)
    for cohort, grp in df.groupby("cohort"):
        idx = grp.index.tolist()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ha, hb = hashes[idx[a]], hashes[idx[b]]
                if ha is None or hb is None:
                    continue
                dist = ha - hb
                if dist <= threshold and df.loc[idx[a], "md5"] != df.loc[idx[b], "md5"]:
                    pairs.append({"cohort": cohort, "dist": int(dist),
                                  "a": df.loc[idx[a], "path_rel"], "b": df.loc[idx[b], "path_rel"],
                                  "same_individual": df.loc[idx[a], "local_id"] == df.loc[idx[b], "local_id"]})
    df["phash"] = [str(h) if h is not None else "" for h in hashes]
    return df, pairs
