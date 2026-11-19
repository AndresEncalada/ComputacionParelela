"""
main.py - Aplicacion web FastAPI para aplicar filtros de convolucion.

Ejecucion (desde la raiz del repositorio):
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.filtros import FILTROS, FILTROS_POR_ID, MASCARA_MAX, MASCARA_MIN, catalogo
from app.servicios.procesamiento import ProcesamientoNoImplementado, aplicar_filtro

BASE = Path(__file__).resolve().parent
TIPOS_PERMITIDOS = {"image/jpeg", "image/png", "image/bmp", "image/x-portable-graymap"}
TAMANO_MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="App de Filtros", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


@app.get("/", include_in_schema=False)
def inicio(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"filtros": FILTROS, "config": catalogo()},
    )


@app.get("/api/filtros")
def listar_filtros() -> dict:
    return catalogo()


@app.post(
    "/api/procesar",
    responses={200: {"content": {"image/png": {}}}, 501: {"description": "Lógica pendiente"}},
)
async def procesar(
    imagen: UploadFile = File(...),
    filtro: str = Form(...),
    mascara: int = Form(3),
) -> Response:
    if filtro not in FILTROS_POR_ID:
        raise HTTPException(422, f"Filtro desconocido: {filtro}")
    if not FILTROS_POR_ID[filtro].mascara_variable:
        mascara = 3
    elif not (MASCARA_MIN <= mascara <= MASCARA_MAX and mascara % 2 == 1):
        raise HTTPException(422, f"La máscara debe ser impar entre {MASCARA_MIN} y {MASCARA_MAX}.")
    if imagen.content_type not in TIPOS_PERMITIDOS:
        raise HTTPException(415, "Formato no soportado. Usa JPG, PNG, BMP o PGM.")

    contenido = await imagen.read()
    if len(contenido) > TAMANO_MAX_BYTES:
        raise HTTPException(413, "La imagen supera los 25 MB.")

    try:
        salida = aplicar_filtro(contenido, filtro, mascara)
    except ProcesamientoNoImplementado as exc:
        raise HTTPException(501, str(exc)) from exc

    return Response(content=salida, media_type="image/png")
