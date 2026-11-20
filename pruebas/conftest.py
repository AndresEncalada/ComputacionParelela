"""
conftest.py - Fixtures compartidas por las pruebas de validacion.

Por defecto cada fixture levanta un servidor uvicorn real (proceso aparte, con su
propio contexto CUDA) en un puerto libre. Si se define la variable URL_SERVICIO
(p. ej. http://localhost:8000 para el contenedor), las pruebas usan ese servicio.
"""

from __future__ import annotations

import csv
import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
# La evidencia se separa segun contra que servicio se ejecuto la prueba
_DESTINO = "contenedor" if os.environ.get("URL_SERVICIO") else "local"
RESULTADOS = Path(__file__).resolve().parent / "resultados" / _DESTINO
RESULTADOS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos-pycuda" / "src"))


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _levantar(entorno_extra: dict[str, str]):
    puerto = _puerto_libre()
    entorno = {**os.environ, **entorno_extra}
    log = open(RESULTADOS / f"servidor_{puerto}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(puerto),
         "--workers", "1"],
        cwd=RAIZ, env=entorno, stdout=log, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{puerto}"
    limite = time.time() + 120
    while time.time() < limite:
        if proc.poll() is not None:
            raise RuntimeError(f"El servidor terminó al arrancar; ver {log.name}")
        try:
            if httpx.get(f"{url}/api/salud", timeout=2).status_code == 200:
                return proc, url, log
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("El servidor no respondió /api/salud con 200 a tiempo")


@pytest.fixture(scope="session")
def url_servicio():
    externo = os.environ.get("URL_SERVICIO")
    if externo:
        yield externo.rstrip("/")
        return
    proc, url, log = _levantar({})
    yield url
    proc.terminate()
    proc.wait(timeout=15)
    log.close()


@pytest.fixture(scope="session")
def url_servicio_cola_corta():
    """Servidor con cola de 1 lugar y retardo artificial para forzar rechazos."""
    proc, url, log = _levantar({"MAX_COLA": "1", "RETARDO_PRUEBA_MS": "1500"})
    yield url
    proc.terminate()
    proc.wait(timeout=15)
    log.close()


# ---------- Utilidades ----------

def png_de(pixeles: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixeles.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def solicitar(url: str, contenido: bytes, filtro: str, mascara: int | str = 3,
              salida: str = "png", nombre: str = "prueba.png", timeout: float = 120.0) -> httpx.Response:
    return httpx.post(
        f"{url}/api/procesar",
        files={"imagen": (nombre, contenido, "image/png")},
        data={"filtro": filtro, "mascara": str(mascara), "salida": salida},
        timeout=timeout,
    )


def npy_de(resp: httpx.Response) -> np.ndarray:
    return np.load(io.BytesIO(resp.content), allow_pickle=False)


def convolucion_referencia(img: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Convolucion con zero-padding en float64 (referencia independiente, solo para pruebas)."""
    k = int(round(len(mascara) ** 0.5))
    m = mascara.reshape(k, k).astype(np.float64)
    r = k // 2
    alto, ancho = img.shape
    pad = np.zeros((alto + 2 * r, ancho + 2 * r), dtype=np.float64)
    pad[r:r + alto, r:r + ancho] = img
    salida = np.zeros((alto, ancho), dtype=np.float64)
    for ky in range(k):
        for kx in range(k):
            salida += m[ky, kx] * pad[ky:ky + alto, kx:kx + ancho]
    return salida


def escribir_csv(nombre: str, filas: list[dict]) -> Path:
    ruta = RESULTADOS / nombre
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return ruta
