"""FastAPI-бекенд демо re-id тритонов — обёртка над ReIDService (замороженное ядро).

Запуск:  uvicorn app:app --app-dir backend --port 8000    (или через preview_start «triton-backend»)
Эндпоинты: GET / (UI), GET /samples, GET /sample/{id}, POST /identify (multipart file).
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from reid_service import ReIDService

HERE = Path(__file__).parent
_svc: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[app] загрузка ядра и каталога TK…")
    _svc["reid"] = ReIDService("TK")
    print("[app] готов.")
    yield
    _svc.clear()


app = FastAPI(title="Тритоны — демо re-id", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/samples")
def samples():
    return {"samples": _svc["reid"].sample_thumbs()}


@app.get("/sample/{frame_id}")
def sample(frame_id: str):
    try:
        return JSONResponse(_svc["reid"].identify_sample(frame_id))
    except KeyError:
        raise HTTPException(404, f"нет кадра {frame_id}")


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "пустой файл")
    try:
        return JSONResponse(_svc["reid"].identify(data, topk=5))
    except Exception as e:  # noqa: BLE001 — демо: вернуть понятную ошибку в UI
        raise HTTPException(500, f"ошибка обработки: {e}")
