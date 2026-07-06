"""
Часть 6 (спайк LAB) — под-шаг 1: построение temporal-бенчмарка.

Изолированный код (не зависит от triton 3.0). Читает сырую LAB-папку, строит лёгкий
time-aware бенчмарк перепоимок: gallery = самая ранняя сессия особи, probe = последующие.

Антипаттерны 3.0, которых избегаем:
  - сплит по individual_id (особь целиком в одну роль), не по кадру;
  - time-aware (ранняя сессия → база, поздние → запросы), не random;
  - md5-дедуп дублей-вариантов ДО сплита.

Выход: data/lab_benchmark.csv + печать сводки.
"""
from __future__ import annotations
import csv
import hashlib
import re
from datetime import date
from pathlib import Path

LAB_DIR = Path("/Users/evanatas/дичь/тритоны/ тритоны датасетов, но спустя время. Тритоны лабораторные")
OUT_CSV = Path(__file__).parent / "data" / "lab_benchmark.csv"

SESSION_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")        # DD.MM.YYYY
STEM_RE = re.compile(r"^(\d+)(?:\.(\d+))?$")                    # "03" | "03.1"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_session(name: str) -> date | None:
    m = SESSION_RE.match(name)
    if not m:
        return None
    dd, mm, yyyy = map(int, m.groups())
    return date(yyyy, mm, dd)


def parse_stem(stem: str) -> tuple[int, int] | None:
    """'03' -> (3,0); '03.1' -> (3,1); '1' -> (1,0). None если не число."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    ind = int(m.group(1))
    variant = int(m.group(2)) if m.group(2) else 0
    return ind, variant


def collect_records() -> list[dict]:
    records = []
    for sub in sorted(LAB_DIR.iterdir()):
        if not sub.is_dir():
            continue
        sess = parse_session(sub.name)
        if sess is None:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"} or f.name.startswith("."):
                continue
            parsed = parse_stem(f.stem)
            if parsed is None:
                print(f"  ⚠️  пропуск (нечисловой стем): {sub.name}/{f.name}")
                continue
            ind, variant = parsed
            records.append({
                "individual_id": ind,
                "variant": variant,
                "session_date": sess.isoformat(),
                "filename": f.name,
                "abs_path": str(f),
                "md5": md5_of(f),
                "stem_raw": f.stem,
            })
    return records


def _survivor_key(r: dict) -> tuple:
    """Каноническая метка выживает при дубле: без варианта лучше; ДВУзначный стем
    бьёт ОДНОзначный (1.jpg = ошибочная метка для 10.jpg, urok ТЗ про регенерацию/ошибки).
    Меньший ключ = предпочтительнее."""
    n_digits = len(re.match(r"^(\d+)", r["stem_raw"]).group(1))
    return (r["variant"], 0 if n_digits >= 2 else 1, r["stem_raw"])


def dedup_by_md5(records: list[dict]) -> list[dict]:
    """Убрать байт-идентичные дубли (варианты .1, ошибочные метки) — оставить КАНОНИЧЕСКИЙ на md5."""
    seen: dict[str, dict] = {}
    dropped = []
    for r in sorted(records, key=_survivor_key):
        if r["md5"] in seen:
            dropped.append((r, seen[r["md5"]]))
        else:
            seen[r["md5"]] = r
    if dropped:
        print(f"\n  md5-дедуп: убрано {len(dropped)} байт-дублей (оставлен канонический):")
        for r, keep in dropped:
            print(f"    DROP {r['session_date']}/{r['filename']} (id {r['individual_id']})  ≡  KEEP {keep['session_date']}/{keep['filename']} (id {keep['individual_id']})")
    return list(seen.values())


def months_between(a: date, b: date) -> float:
    return round((b - a).days / 30.44, 1)


def interval_bin(days: int) -> str:
    if days <= 40:
        return "~1мес"
    if days <= 95:
        return "2-3мес"
    return "4-5мес"


def build_benchmark(records: list[dict]) -> list[dict]:
    """Time-aware сплит по особи: gallery=ранняя сессия, probe=поздние. Интервалы для probe."""
    by_ind: dict[int, list[dict]] = {}
    for r in records:
        by_ind.setdefault(r["individual_id"], []).append(r)

    rows = []
    for ind, recs in sorted(by_ind.items()):
        sessions = sorted({r["session_date"] for r in recs})
        gallery_sess = sessions[0]
        g_date = date.fromisoformat(gallery_sess)
        for r in recs:
            is_gallery = (r["session_date"] == gallery_sess)
            p_date = date.fromisoformat(r["session_date"])
            days = (p_date - g_date).days
            rows.append({
                **r,
                "n_sessions": len(sessions),
                "role": "gallery" if is_gallery else "probe",
                "gallery_session": gallery_sess,
                "interval_days": days,
                "interval_months": months_between(g_date, p_date) if not is_gallery else 0.0,
                "interval_bin": "" if is_gallery else interval_bin(days),
            })
    return rows


def summarize(rows: list[dict]) -> None:
    inds = sorted({r["individual_id"] for r in rows})
    gallery = [r for r in rows if r["role"] == "gallery"]
    probe = [r for r in rows if r["role"] == "probe"]
    recaptured = sorted({r["individual_id"] for r in probe})
    only_one = [i for i in inds if all(r["n_sessions"] == 1 for r in rows if r["individual_id"] == i)]

    print("\n" + "=" * 60)
    print("СВОДКА LAB temporal-бенчмарка")
    print("=" * 60)
    print(f"Фото всего (после дедупа): {len(rows)}")
    print(f"Особей всего:              {len(inds)}  -> {inds}")
    print(f"Gallery-кадров:            {len(gallery)}")
    print(f"Probe-кадров (перепоимки): {len(probe)}")
    print(f"Особей с перепоимкой:      {len(recaptured)}  -> {recaptured}")
    print(f"Особей только в 1 сессии:  {len(only_one)}  -> {only_one}  (только дистракторы в gallery)")
    print(f"Случайный baseline top-1:  1/{len(inds)} = {1/len(inds):.3f}")

    print("\nProbe по интервалам перепоимки:")
    bins: dict[str, int] = {}
    for r in probe:
        bins[r["interval_bin"]] = bins.get(r["interval_bin"], 0) + 1
    for b in ("~1мес", "2-3мес", "4-5мес"):
        print(f"  {b:8s}: {bins.get(b, 0)} проб")

    print("\nКадров по особям (n_sessions):")
    for i in inds:
        n = max(r["n_sessions"] for r in rows if r["individual_id"] == i)
        marker = "  ← все 5 сессий" if n == 5 else ""
        print(f"  особь {i:2d}: {n} сесс.{marker}")


def main() -> None:
    print(f"LAB-папка: {LAB_DIR}")
    print(f"Существует: {LAB_DIR.exists()}\n")
    records = collect_records()
    print(f"Сырых кадров прочитано: {len(records)}")
    records = dedup_by_md5(records)
    print(f"После md5-дедупа:       {len(records)}")
    rows = build_benchmark(records)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["individual_id", "variant", "session_date", "n_sessions", "role",
            "gallery_session", "interval_days", "interval_months", "interval_bin",
            "filename", "md5", "abs_path", "stem_raw"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    summarize(rows)
    print(f"\n✅ бенчмарк сохранён: {OUT_CSV}")


if __name__ == "__main__":
    main()
