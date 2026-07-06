"""Часть 2 — жёсткие гейты честности реестра. Падает с понятным сообщением при нарушении."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from common import ROOT, ROLE

HERE = Path(__file__).parent


def validate(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    res = []

    # G1 individual_id не пересекаются между когортами (namespace по префиксу)
    bad = [iid for iid in df.individual_id.unique() if iid.split("-")[0] not in ROLE]
    res.append(("G1 namespace individual_id", len(bad) == 0, f"чужих префиксов: {len(bad)}"))

    # G2 даты валидны (YYYY-MM[-DD], год 2024-2026, месяц 01-12) где заданы
    dts = df.date.dropna()
    def ok_date(s):
        try:
            p = str(s).split("-"); y, m = int(p[0]), int(p[1])
            return 2024 <= y <= 2026 and 1 <= m <= 12
        except Exception:
            return False
    badd = dts[~dts.map(ok_date)]
    res.append(("G2 даты валидны", len(badd) == 0, f"невалидных дат: {len(badd)}"))

    # G3 нет md5-дубликатов среди выживших
    dup = df.md5.duplicated().sum()
    res.append(("G3 md5 уникальны", dup == 0, f"дубликатов md5: {dup}"))

    # G4 у каждой target-особи известен has_recapture (bool)
    tgt = df[df.role == "target"]
    res.append(("G4 has_recapture у target", tgt.has_recapture.notna().all(), ""))

    # G5 GCN строго external
    g5 = (df[df.cohort == "GCN"].role == "external").all()
    res.append(("G5 GCN=external", bool(g5), ""))

    # G6 пути относительные и файлы существуют (выборка для скорости — все)
    miss = sum(1 for r in df.path_rel if not (ROOT / r).exists())
    absol = sum(1 for r in df.path_rel if str(r).startswith("/"))
    res.append(("G6 пути относительны+существуют", miss == 0 and absol == 0, f"нет файла: {miss}, абсолютных: {absol}"))

    # G7 near-dup разобраны: все ли в одной сессии (тогда temporal-сплит их не разводит)
    nd = HERE / "artifacts" / "near_duplicates.csv"
    if nd.exists():
        ndf = pd.read_csv(nd)
        key = df.set_index("path_rel")
        cross = 0
        for _, p in ndf.iterrows():
            if p.a in key.index and p.b in key.index:
                ta = key.loc[p.a, "date"] if pd.notna(key.loc[p.a, "date"]) else key.loc[p.a, "session"]
                tb = key.loc[p.b, "date"] if pd.notna(key.loc[p.b, "date"]) else key.loc[p.b, "session"]
                if ta != tb:
                    cross += 1
        res.append(("G7 near-dup не межсессионные", cross == 0, f"межсессионных near-dup: {cross} (=утечка)"))
    else:
        # R1.2: нет отчёта → НЕ зелёный (ложно-зелёный гейт честности), а явный провал
        res.append(("G7 near-dup отчёт ОТСУТСТВУЕТ", False, "near_duplicates.csv не найден — прогнать phash_audit"))

    return res


def main():
    df = pd.read_csv(HERE / "registry.csv")
    print("ГЕЙТЫ ЧЕСТНОСТИ РЕЕСТРА")
    print("=" * 50)
    all_ok = True
    for name, ok, detail in validate(df):
        mark = "✅" if ok else "❌"
        all_ok &= ok
        print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))
    print("=" * 50)
    print("ИТОГ:", "✅ все гейты пройдены" if all_ok else "❌ ЕСТЬ НАРУШЕНИЯ")
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
