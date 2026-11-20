"""
main.py - Capa HTTP (FastAPI). No contiene logica de convolucion: valida la solicitud,
delega en servicios/procesamiento.py y traduce errores y tiempos a la respuesta.

Ejecucion (desde la raiz del repositorio, un solo proceso: cada proceso crearia su
propio contexto CUDA y competiria por la GPU):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from concurrent.futures import TimeoutError as FuturoTimeout
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import registro
from app.config import CONFIG
from app.filtros import FILTROS, ParametroInvalido, catalogo, resolver_parametros
from app.servicios.motor_gpu import (
    ColaLlena,
    ErrorGPU,
    GPUNoDisponible,
    MemoriaGPUInsuficiente,
    MotorGPU,
)
from app.servicios.procesamiento import SALIDAS, EntradaInvalida, procesar as procesar_imagen

BASE = Path(__file__).resolve().parent


registro.configurar(CONFIG.log_archivo)
log = logging.getLogger("app")
motor = MotorGPU(hilos_bloque=CONFIG.hilos_bloque, max_cola=CONFIG.max_cola,
                 retardo_prueba_ms=CONFIG.retardo_prueba_ms)
if CONFIG.retardo_prueba_ms:
    log.warning("RETARDO_PRUEBA_MS=%d activo: solo para pruebas de cola, no medir con esto.", CONFIG.retardo_prueba_ms)

LIMITES_PUBLICOS = {
    "max_mb": CONFIG.max_bytes // 2**20,
    "max_lado": CONFIG.max_lado,
    "max_megapixeles": CONFIG.max_pixeles // 1_000_000,
    "max_en_espera": CONFIG.max_cola,
}


@asynccontextmanager
async def ciclo_de_vida(_: FastAPI):
    # Crear contexto CUDA y compilar el kernel antes de aceptar solicitudes
    await run_in_threadpool(motor.iniciar)
    registro.arranque(motor.info, LIMITES_PUBLICOS, CONFIG.hilos_bloque, motor.error_inicio)
    yield
    motor.detener()


app = FastAPI(title="App de Filtros", version="1.0.0", lifespan=ciclo_de_vida)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

# Cabeceras de tiempo: nombre interno -> cabecera HTTP (todas en milisegundos)
CABECERAS_TIEMPO = {
    "cola": "X-Tiempo-Cola-ms",
    "decodificacion": "X-Tiempo-Decodificacion-ms",
    "reserva": "X-Tiempo-Reserva-ms",
    "h2d": "X-Tiempo-H2D-ms",
    "kernel": "X-Tiempo-Kernel-ms",
    "d2h": "X-Tiempo-D2H-ms",
    "liberacion": "X-Tiempo-Liberacion-ms",
    "gpu_total": "X-Tiempo-GPU-Total-ms",
    "codificacion": "X-Tiempo-Codificacion-ms",
    "servidor": "X-Tiempo-Servidor-ms",
}


def _error(estado: int, codigo: str, mensaje: str, request_id: str | None, extra: dict | None = None):
    cabeceras = {"X-Request-ID": request_id} if request_id else {}
    if extra:
        cabeceras.update(extra)
    return JSONResponse(
        status_code=estado,
        content={"detail": mensaje, "codigo": codigo, "request_id": request_id},
        headers=cabeceras,
    )


@app.exception_handler(RequestValidationError)
async def campos_faltantes(_: Request, exc: RequestValidationError):
    faltan = [str(e["loc"][-1]) for e in exc.errors()]
    detalle = f"Faltan o son inválidos los campos: {', '.join(faltan)}."
    registro.solicitud_error(None, 422, "parametro_invalido", "-", "", 0.0, detalle)
    return _error(422, "parametro_invalido", detalle, None)


@app.middleware("http")
async def limitar_tamano(request: Request, call_next):
    """Rechaza cargas demasiado grandes antes de leer el cuerpo."""
    if request.url.path == "/api/procesar":
        largo = request.headers.get("content-length")
        margen = 64 * 1024  # cabeceras multipart y campos de texto
        if largo and largo.isdigit() and int(largo) > CONFIG.max_bytes + margen:
            detalle = f"El archivo supera el límite de {CONFIG.max_bytes // 2**20} MB."
            registro.solicitud_error(None, 413, "archivo_demasiado_grande", "-", "", 0.0,
                                     f"{detalle} (Content-Length {int(largo) / 2**20:.1f} MB)")
            return _error(413, "archivo_demasiado_grande", detalle, None)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def inicio(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"filtros": FILTROS, "config": catalogo(LIMITES_PUBLICOS)}
    )


@app.get("/api/filtros")
def listar_filtros() -> dict:
    return catalogo(LIMITES_PUBLICOS)


@app.get("/api/salud")
def salud():
    cuerpo = {
        "estado": "ok" if motor.disponible else "sin_gpu",
        "diagnostico": motor.error_inicio,
        **motor.info,
        "cola": motor.estado_cola(),
        "limites": LIMITES_PUBLICOS,
    }
    return JSONResponse(cuerpo, status_code=200 if motor.disponible else 503)


@app.post(
    "/api/procesar",
    responses={
        200: {"content": {"image/png": {}}, "description": "Imagen filtrada; tiempos en cabeceras X-Tiempo-*"},
        413: {"description": "Archivo o dimensiones superan el límite"},
        415: {"description": "El archivo no es una imagen JPG, PNG, BMP o PGM"},
        422: {"description": "Filtro, máscara o imagen inválidos"},
        503: {"description": "Cola llena o GPU no disponible"},
    },
)
async def procesar(
    imagen: UploadFile = File(...),
    filtro: str = Form(...),
    mascara: str = Form("3"),
    salida: str = Form("png"),
) -> Response:
    request_id = uuid.uuid4().hex
    t_llegada = time.perf_counter()
    extra: dict | None = None
    try:
        filtro_ok, k = resolver_parametros(filtro, mascara)
        if salida not in SALIDAS:
            raise ParametroInvalido(f"Salida no válida: '{salida}'. Usa 'png' o 'npy'.")

        contenido = await imagen.read(CONFIG.max_bytes + 1)
        resultado = await run_in_threadpool(procesar_imagen, contenido, filtro_ok, k, motor, CONFIG, salida)

        base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(imagen.filename or "imagen").stem)[:60] or "imagen"
        nombre = f"{base}_{filtro_ok}{k}_{request_id[:8]}.{salida}"
        cabeceras = {
            "X-Request-ID": request_id,
            "X-Imagen": f"{resultado.ancho}x{resultado.alto}",
            "X-Filtro": filtro_ok,
            "X-Mascara": f"{k}x{k}",
            "X-Hilos-Bloque": str(resultado.hilos_bloque),
            "X-Bloques": str(resultado.bloques),
            "Content-Disposition": f'attachment; filename="{nombre}"',
            "Server-Timing": ", ".join(
                f"{n};dur={resultado.tiempos_ms[n]:.3f}" for n in ("cola", "decodificacion", "h2d", "kernel", "d2h", "codificacion")
            ),
        }
        for clave, cabecera in CABECERAS_TIEMPO.items():
            cabeceras[cabecera] = f"{resultado.tiempos_ms[clave]:.4f}"
        registro.solicitud_ok(request_id, filtro_ok, k, resultado.ancho, resultado.alto, resultado.tiempos_ms)
        return Response(content=resultado.contenido, media_type=resultado.tipo_mime, headers=cabeceras)

    except ParametroInvalido as exc:
        estado, codigo, detalle = 422, "parametro_invalido", str(exc)
    except EntradaInvalida as exc:
        estado, codigo, detalle = exc.estado_http, exc.codigo, str(exc)
    except ColaLlena as exc:
        estado, codigo, detalle, extra = 503, exc.codigo, str(exc), {"Retry-After": "2"}
    except GPUNoDisponible as exc:
        estado, codigo, detalle = 503, exc.codigo, f"GPU no disponible: {exc}"
    except MemoriaGPUInsuficiente as exc:
        estado, codigo, detalle = 507, exc.codigo, str(exc)
    except FuturoTimeout:
        estado, codigo, detalle = 504, "tiempo_agotado", f"La solicitud superó {CONFIG.timeout_s} s de espera."
    except ErrorGPU as exc:
        estado, codigo, detalle = 500, exc.codigo, str(exc)
    except Exception:
        log.exception("[%s] error no controlado", request_id[:8])
        estado, codigo, detalle = 500, "error_interno", "Error interno del servidor."
    finally:
        await imagen.close()  # libera el archivo temporal de la carga (SpooledTemporaryFile)

    registro.solicitud_error(request_id, estado, codigo, filtro, mascara,
                             (time.perf_counter() - t_llegada) * 1000.0, detalle)
    return _error(estado, codigo, detalle, request_id, extra)
