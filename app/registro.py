"""
registro.py - Formato legible y uniforme para todos los logs del servicio.

Formato de cada linea (columnas de ancho fijo, separadas por " | "):

    2026-11-20 06:31:08.845 | INFO  | solicitud | [eda3d2cb] 200 OK     media      9x9   6000x4000 | ...
    fecha y hora (ms)         nivel   origen      mensaje

Origenes: "arranque" (GPU y configuracion), "gpu" (motor CUDA), "solicitud" (una linea por
solicitud procesada o rechazada) y "http" (acceso de uvicorn). Los logs de uvicorn se
redirigen a este mismo formato. Se usa solo ASCII para que funcione en cualquier consola,
y colores ANSI solo si la salida es una terminal (no en `docker logs -f` sin -t ni en archivo).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ORIGENES = {
    "app": "servicio",
    "app.arranque": "arranque",
    "app.gpu": "gpu",
    "app.solicitud": "solicitud",
    "uvicorn.access": "http",
    "uvicorn.error": "servidor",
    "uvicorn": "servidor",
}
NIVELES = {"WARNING": "WARN", "CRITICAL": "CRIT"}
COLORES = {"DEBUG": "\033[90m", "INFO": "\033[36m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[41m"}
RESET = "\033[0m"


class FormatoLegible(logging.Formatter):
    def __init__(self, color: bool = False) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        fecha = f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"
        nivel = NIVELES.get(record.levelname, record.levelname)
        origen = ORIGENES.get(record.name, record.name.split(".")[-1])[:9]
        mensaje = _mensaje_http(record) if record.name == "uvicorn.access" else record.getMessage()
        if self.color:
            c = COLORES.get(record.levelname, "")
            nivel_txt = f"{c}{nivel:<5}{RESET}"
        else:
            nivel_txt = f"{nivel:<5}"
        linea = f"{fecha} | {nivel_txt} | {origen:<9} | {mensaje}"
        if record.exc_info:
            linea += "\n" + self.formatException(record.exc_info)
        return linea


def _mensaje_http(record: logging.LogRecord) -> str:
    """uvicorn.access trae args = (cliente, metodo, ruta, version_http, estado)."""
    try:
        cliente, metodo, ruta, _, estado = record.args
        return f"{metodo:<4} {ruta:<22} -> {estado}  ({cliente})"
    except (TypeError, ValueError):
        return record.getMessage()


def configurar(log_archivo: str | None = None, flujo=None) -> None:
    """Configura la raiz y redirige los loggers de uvicorn al mismo formato."""
    flujo = flujo or sys.stdout
    if hasattr(flujo, "reconfigure"):  # Windows usa cp1252 al redirigir: forzar UTF-8
        flujo.reconfigure(encoding="utf-8", errors="replace")
    usar_color = flujo.isatty() and not os.environ.get("NO_COLOR")
    consola = logging.StreamHandler(flujo)
    consola.setFormatter(FormatoLegible(color=usar_color))
    manejadores: list[logging.Handler] = [consola]
    if log_archivo:
        Path(log_archivo).parent.mkdir(parents=True, exist_ok=True)
        archivo = logging.FileHandler(log_archivo, encoding="utf-8")
        archivo.setFormatter(FormatoLegible(color=False))
        manejadores.append(archivo)
    logging.basicConfig(level=logging.INFO, handlers=manejadores, force=True)

    # uvicorn configura sus loggers antes de importar la app: se les quitan sus
    # manejadores para que pasen por la raiz con el formato comun.
    for nombre in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(nombre)
        lg.handlers.clear()
        lg.propagate = True


# ---------- Mensajes de alto nivel ----------

def _ms(valor: float, ancho: int = 8) -> str:
    return f"{valor:>{ancho}.2f} ms"


def arranque(info: dict, limites: dict, hilos: int, error: str | None) -> None:
    log = logging.getLogger("app.arranque")
    raya = "=" * 72
    log.info(raya)
    log.info("App de Filtros - servicio de convolucion con PyCUDA")
    if error:
        log.error("GPU ............ NO DISPONIBLE")
        log.error("Diagnostico .... %s", error)
    else:
        g, v, a = info["gpu"], info["versiones"], info["arranque_ms"]
        log.info("GPU ............ %s (CC %s, %s MB)", g["nombre"], g["compute_capability"], g["memoria_total_mb"])
        log.info("Versiones ...... CUDA driver %s | PyCUDA %s (CUDA %s) | numpy %s | Python %s",
                 v["cuda_driver_api"], v["pycuda"], v["cuda_pycuda_compilado"], v["numpy"], v["python"])
        log.info("Arranque ....... contexto %.2f ms | compilacion kernel %.2f ms | primera ejecucion %.2f ms",
                 a["init_contexto"], a["compilacion_kernel"], a["primera_ejecucion"])
    log.info("Configuracion .. %d hilos/bloque | memoria host paginable | 1 solicitud en GPU + %d en cola",
             hilos, limites["max_en_espera"])
    log.info("Limites ........ %d MB | %d px por lado | %d megapixeles",
             limites["max_mb"], limites["max_lado"], limites["max_megapixeles"])
    log.info(raya)


def solicitud_ok(request_id: str, filtro: str, k: int, ancho: int, alto: int, t: dict) -> None:
    logging.getLogger("app.solicitud").info(
        "[%s] 200 OK     %-10s %5s %11s | cola%s | h2d%s | kernel%s | d2h%s | servidor%s",
        request_id[:8], filtro, f"{k}x{k}", f"{ancho}x{alto}",
        _ms(t["cola"]), _ms(t["h2d"], 7), _ms(t["kernel"]), _ms(t["d2h"], 7), _ms(t["servidor"], 9),
    )


def solicitud_error(request_id: str | None, estado: int, codigo: str, filtro: str, mascara: str,
                    duracion_ms: float, detalle: str) -> None:
    nivel = logging.ERROR if estado >= 500 and codigo not in ("cola_llena",) else logging.WARNING
    k = f"{mascara}x{mascara}" if mascara and mascara.isdigit() else (mascara or "-")
    logging.getLogger("app.solicitud").log(
        nivel, "[%s] %d %-24s %-10s %5s | %s | %s",
        (request_id or "--------")[:8], estado, codigo, filtro or "-", k, _ms(duracion_ms).strip(), detalle,
    )
