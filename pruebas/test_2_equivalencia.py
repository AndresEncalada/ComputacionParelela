"""
Prueba 2 - Equivalencia entre PyCUDA directo (practica 02 validada) y el servicio.

Directo: experimentos-pycuda/src/gpu_conv.medir_tiempo_gpu (mismo codigo que los experimentos
de la practica 02). Servicio: POST /api/procesar con salida=npy (float32 SIN recortar).

Se registran error absoluto maximo y MAE antes del recorte. Ademas se compara la
salida PNG del servicio con el recorte [0,255] de la salida directa: PNG es sin perdida,
por lo que cualquier diferencia ahi seria atribuible a la codificacion (se espera 0).

Tambien genera la evidencia visual: original, los 3 filtros y el mismo recorte ampliado.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from conftest import RAIZ, RESULTADOS, escribir_csv, npy_de, png_de, solicitar

TOLERANCIA = 1e-3
CASOS = [("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3)]


@pytest.fixture(scope="module")
def imagen_2000():
    """entrada.jpg -> grises -> recorte cuadrado central -> 2000x2000 (igual que preparar_imagen)."""
    import cpu_conv

    base = cpu_conv.leer_imagen(str(RAIZ / "experimentos-pycuda" / "entrada.jpg"))
    img = cpu_conv.preparar_imagen(base, 2000, 2000)
    return np.frombuffer(bytes(img.datos), dtype=np.uint8).reshape(2000, 2000).copy()


def _directo(img: np.ndarray, filtro: str, k: int) -> np.ndarray:
    import cpu_conv
    import gpu_conv  # crea su propio contexto (pycuda.autoinit) en este proceso

    from app.filtros import construir_mascara

    ent = cpu_conv.Imagen(ancho=img.shape[1], alto=img.shape[0], datos=bytearray(img.tobytes()))
    crudo = [0.0] * img.size
    gpu_conv.medir_tiempo_gpu(ent, crudo, construir_mascara(filtro, k).tolist(), k,
                              gpu_conv.HILOS_BASE, filtro, None)
    return np.asarray(crudo, dtype=np.float32).reshape(img.shape)


def test_equivalencia_directo_vs_servicio(url_servicio, imagen_2000):
    filas = []
    salidas_png = {}
    for filtro, k in CASOS:
        directo = _directo(imagen_2000, filtro, k)

        r = solicitar(url_servicio, png_de(imagen_2000), filtro, k, salida="npy")
        assert r.status_code == 200, r.text
        servicio = npy_de(r)

        r_png = solicitar(url_servicio, png_de(imagen_2000), filtro, k)
        assert r_png.status_code == 200
        png = np.asarray(Image.open(io.BytesIO(r_png.content)))
        salidas_png[(filtro, k)] = png

        err = np.abs(servicio.astype(np.float64) - directo.astype(np.float64))
        recortado = np.clip(directo, 0, 255).astype(np.uint8)
        distintos = int(np.count_nonzero(png != recortado))
        filas.append({
            "filtro": filtro, "mascara": f"{k}x{k}", "resolucion": "2000x2000",
            "hilos_bloque": r.headers["X-Hilos-Bloque"],
            "error_abs_max": f"{err.max():.6e}", "mae": f"{err.mean():.6e}",
            "tolerancia": TOLERANCIA,
            "pixeles_png_distintos": distintos,
            "estado": "OK" if err.max() <= TOLERANCIA and distintos == 0 else "FALLA",
        })
        assert err.max() <= TOLERANCIA
        assert distintos == 0

    escribir_csv("equivalencia.csv", filas)
    _evidencia_visual(imagen_2000, salidas_png)


def _evidencia_visual(original: np.ndarray, salidas: dict) -> None:
    """Panel: fila 1 imagen completa, fila 2 el mismo recorte ampliado 3x."""
    paneles = [("Original", original)] + [
        (f"{f} {k}x{k}", salidas[(f, k)]) for f, k in (("media", 9), ("nitidez", 3), ("laplaciano", 3))
    ]
    lado = 400
    y0, x0, r = 1000, 1050, 200  # region con detalle (plumas del ave)
    lienzo = Image.new("L", (lado * len(paneles), lado * 2), 255)
    for i, (_, img) in enumerate(paneles):
        lienzo.paste(Image.fromarray(img).resize((lado, lado)), (i * lado, 0))
        recorte = img[y0:y0 + r, x0:x0 + r]
        lienzo.paste(Image.fromarray(recorte).resize((lado, lado), Image.NEAREST), (i * lado, lado))
    destino = RESULTADOS / "visual"
    destino.mkdir(exist_ok=True)
    lienzo.save(destino / "comparacion_filtros.png")
    for nombre, img in paneles:
        Image.fromarray(img).save(destino / f"{nombre.replace(' ', '_')}.png")
