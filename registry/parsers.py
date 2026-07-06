"""Часть 2 — парсеры 4 когорт. Обрабатывают все известные аномалии (с диска, Explore-разведка)."""
from __future__ import annotations
import csv
import re
from pathlib import Path

from common import COHORT_DIRS, iter_images, base_record, mmyy_to_date

TK_NAME = re.compile(r"^(\d{2})-(\d{1,2})-(\d+)$")            # id-session-MMYY (дата = ровно 4 цифры)
PW_NAME = re.compile(r"^(\d+)-(\d+)\s*\((\d+)\)$")            # id-session (frame)
LAB_SESS = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")        # DD.MM.YYYY
LAB_STEM = re.compile(r"^(\d+)(?:\.(\d+))?$")                # id | id.variant
MMYY_IN = re.compile(r"(\d{2})(\d{2})")                      # MMYY в имени подпапки


def _folder_local_id(name: str) -> int | None:
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else None


def parse_tk() -> list[dict]:
    """TK Карелина. Даты MMYY в имени; IMG_* — дата из подпапки; «21 (error)»→local_id 20; ID70 неполный формат."""
    root = COHORT_DIRS["TK"]
    recs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        local_id = _folder_local_id(d.name)
        if local_id is None:
            continue
        if "error" in d.name.lower():
            local_id = 20                       # папка «21 (error)» = особь ID20
        for f in iter_images(d):
            date, dsrc, session = None, "none", None
            nm = TK_NAME.match(f.stem)
            if nm:
                _id, sess, dd = nm.groups()
                session = int(sess)
                if len(dd) == 4:                # корректное MMYY
                    date, dsrc = mmyy_to_date(dd[:2], dd[2:]), "filename"
                else:                           # опечатка (напр. 74-02-01025) — дату НЕ выдумываем
                    date, dsrc = None, "filename_baddate"
            elif f.parent != d:                 # файл в подпапке «Доп фото от MMYY» (в т.ч. опечатка ID66)
                sm = MMYY_IN.search(f.parent.name)
                if sm:
                    date, dsrc = mmyy_to_date(*sm.groups()), "subfolder"
            recs.append(base_record("TK", local_id, f, session=session, date=date, date_source=dsrc))
    return recs


def parse_pw() -> list[dict]:
    """PW ребристый. Дат нет — только session 01/02 (псевдо-время)."""
    root = COHORT_DIRS["PW"]
    recs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        local_id = _folder_local_id(d.name)
        if local_id is None:
            continue
        for f in iter_images(d):
            nm = PW_NAME.match(f.stem)
            session = int(nm.group(2)) if nm else None
            recs.append(base_record("PW", local_id, f, session=session, date=None, date_source="session_code"))
    return recs


def parse_lab() -> list[dict]:
    """LAB «спустя время». Сессии-подпапки DD.MM.YYYY, стем=id, варианты .k. (логика спайка)."""
    root = COHORT_DIRS["LAB"]
    recs = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        sm = LAB_SESS.match(sub.name)
        if not sm:
            continue
        dd, mm, yyyy = sm.groups()
        date = f"{yyyy}-{mm}-{dd}"
        for f in iter_images(sub):
            st = LAB_STEM.match(f.stem)
            if not st:
                continue
            recs.append(base_record("LAB", int(st.group(1)), f, session=date, date=date, date_source="subfolder"))
    return recs


def parse_gcn() -> list[dict]:
    """GCN внешний. Из metadata.csv. Без temporal-перепоимок (survey ≠ время). bbox/RLE — для Части 4 отдельно."""
    root = COHORT_DIRS["GCN"]
    recs = []
    with open(root / "metadata.csv") as fh:
        for row in csv.DictReader(fh):
            ident, fn = row.get("identity"), row.get("file_name")
            if not ident or not fn:
                continue
            p = root / "Raw_Data" / ident / fn
            if not p.exists():
                continue
            recs.append(base_record("GCN", int(ident), p, session=row.get("survey"), date=None,
                                    date_source="none",
                                    notes=f"survey={row.get('survey')};recapture_id={row.get('recapture_id')}"))
    return recs


ALL_PARSERS = {"TK": parse_tk, "PW": parse_pw, "LAB": parse_lab, "GCN": parse_gcn}
