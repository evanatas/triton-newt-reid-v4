"""Часть 2 — общие утилиты реестра. Новый код, без зависимостей от triton 3.0."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from PIL import Image, ImageOps

# R1.3: путь к данным из env TRITON_ROOT (переносимость). По умолчанию — папка над репозиторием;
# для реальных данных задать TRITON_ROOT (см. README).
ROOT = Path(os.environ.get("TRITON_ROOT", str(Path(__file__).resolve().parents[2])))

COHORT_DIRS = {
    "TK":  ROOT / "TK data тритоны",
    "PW":  ROOT / "Ребристый тритон 1 и 2 датасеты",
    "LAB": ROOT / " тритоны датасетов, но спустя время. Тритоны лабораторные",
    "GCN": ROOT / "фото тритонов из похожего проекта",
}
SPECIES = {"TK": "Triturus karelinii", "PW": "Pleurodeles waltl",
           "LAB": "unknown (см. заказчик)", "GCN": "Triturus cristatus"}
ROLE = {"TK": "target", "PW": "target", "LAB": "temporal_aux", "GCN": "external"}

# F-13: has_label_in_frame убрано (было всегда пустым; фактические данные о метке — в segment/crops_manifest.csv).
COLUMNS = ["frame_id", "cohort", "species", "role", "individual_id", "local_id", "session",
           "date", "date_source", "has_recapture", "n_sessions", "path_rel", "md5", "phash",
           "width", "height", "stage", "stage_source", "notes"]

IMG_EXT = {".jpg", ".jpeg", ".png"}
SKIP_NAMES = {"пояснение.txt", "иллюстрация.jpg", ".ds_store"}


def is_image(p: Path) -> bool:
    return (p.is_file() and p.suffix.lower() in IMG_EXT
            and p.name.lower() not in SKIP_NAMES and not p.name.startswith("."))


def iter_images(folder: Path):
    """Рекурсивно по папке особи (включая подпапки «Доп фото»), только изображения."""
    for p in sorted(folder.rglob("*")):
        if is_image(p):
            yield p


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def image_dims(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:           # .size читается из заголовка, без декодирования пикселей
            return im.size                      # (width, height)
    except Exception:
        return (0, 0)


def mmyy_to_date(mm: str, yy: str) -> str:
    return f"{2000 + int(yy)}-{int(mm):02d}"   # YYYY-MM


def base_record(cohort: str, local_id: int, path: Path, *, session=None,
                date=None, date_source="none", notes="") -> dict:
    w, h = image_dims(path)
    return {
        "cohort": cohort, "local_id": local_id, "session": session,
        "date": date, "date_source": date_source,
        "path_rel": str(path.relative_to(ROOT)), "md5": md5_of(path),
        "width": w, "height": h, "notes": notes,
        "phash": "", "stage": "unknown", "stage_source": "none",
    }
