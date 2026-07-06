"""Часть 3 — sealed-механизм. Запечатать test-fold; гейт против вскрытия без явного флага.

Вскрыть РОВНО один раз в Части 11. Числа test после вскрытия руками не правятся.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
MANIFEST = HERE / "sealed_manifest.json"
SEALED_FOLDS = {"test"}          # open_test = подмножество test (is_open_new)


def seal(df: pd.DataFrame) -> dict:
    """Запечатать все кадры test-fold: сохранить хеш-манифест frame_id+md5."""
    test = df[df["split_fold"] == "test"].sort_values("frame_id")
    payload = "|".join(f"{r.frame_id}:{r.md5}" for _, r in test.iterrows())
    digest = hashlib.sha256(payload.encode()).hexdigest()
    manifest = {
        "sealed_folds": sorted(SEALED_FOLDS),
        "n_sealed": int(len(test)),
        "n_individuals": int(test["individual_id"].nunique()),
        "frame_ids": test["frame_id"].tolist(),
        "sha256": digest,
        "sealed": True,
        "opened": False,
        "note": "Вскрыть один раз в Части 11 через unseal=True. Не тюнить на test.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def assert_unsealed(stage: str, unseal: bool = False) -> None:
    """Гейт: запрещает доступ к запечатанным стадиям без явного unseal=True. Зовётся ДО загрузки данных."""
    if stage in SEALED_FOLDS and not unseal:
        raise RuntimeError(
            f"Стадия '{stage}' ЗАПЕЧАТАНА (sealed). Доступ только с unseal=True — "
            f"вскрытие выполняется РОВНО ОДИН РАЗ в Части 11 (финальный KPI). "
            f"Для разработки используйте 'dev'."
        )
    if stage in SEALED_FOLDS and unseal:            # F-3 (defense-in-depth): доступ к sealed разрешён, но
        try:                                        # на ЛЮБОМ пути сверяем целостность сплита (не только в церемонии)
            reg = pd.read_csv(HERE / "registry.csv")
            spl = pd.read_csv(HERE / "splits.csv")[["frame_id", "split_fold"]]
            if not verify_manifest(reg.merge(spl, on="frame_id", how="left")):
                raise RuntimeError(f"F-3: sealed-манифест НЕ совпал (sha) — '{stage}'-сплит изменён. Доступ запрещён.")
        except FileNotFoundError:
            pass                                    # реестр/сплит недоступны — не ломаем гейт


def mark_opened(opened_date: str) -> dict:
    """Часть 11: пометить манифест ВСКРЫТЫМ (одноразовый audit). opened_date передаётся снаружи (детерминизм)."""
    m = json.loads(MANIFEST.read_text())
    m["opened"] = True
    m["opened_date"] = opened_date
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    return m


def verify_manifest(df: pd.DataFrame) -> bool:
    """Проверить, что текущий test-fold совпадает с запечатанным манифестом (защита от подмены сплита)."""
    if not MANIFEST.exists():
        return False
    m = json.loads(MANIFEST.read_text())
    test = df[df["split_fold"] == "test"].sort_values("frame_id")
    payload = "|".join(f"{r.frame_id}:{r.md5}" for _, r in test.iterrows())
    return hashlib.sha256(payload.encode()).hexdigest() == m["sha256"]


def load_split(stage: str, unseal: bool = False):
    """Удобный загрузчик среза с гейтом. stage ∈ {train,dev,test,aux,distractor}."""
    assert_unsealed(stage, unseal)
    reg = pd.read_csv(HERE / "registry.csv")
    spl = pd.read_csv(HERE / "splits.csv")
    df = reg.merge(spl[["frame_id", "split_role", "split_fold", "interval_months", "is_open_new", "in_train_pool"]],
                   on="frame_id", how="left")
    return df[df["split_fold"] == stage]
