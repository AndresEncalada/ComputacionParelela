"""
Prueba 1 - Matriz pequena y limites (a traves del servicio).

* Matriz 5x5 de la practica 02 con media 3x3: centro 50.0000 y esquina 8.8889 (byte 8).
  Con 256 hilos/bloque hay 251 hilos sin pixel: prueba el guard idx >= ancho*alto.
* 2050x2050 = 4 202 500 pixeles, que no es multiplo de 256: se comparan TODOS los
  pixeles contra una referencia independiente (cobertura completa y bordes con ceros).
"""

from __future__ import annotations

import numpy as np

from conftest import convolucion_referencia, escribir_csv, npy_de, png_de, solicitar
from app.filtros import construir_mascara

MATRIZ_5x5 = np.array([
    [10, 20, 30, 40, 50],
    [20, 30, 40, 50, 60],
    [30, 40, 50, 60, 70],
    [40, 50, 60, 70, 80],
    [50, 60, 70, 80, 90],
], dtype=np.uint8)

TOLERANCIA = 1e-3  # misma tolerancia que la practica 02 (float32)


def test_matriz_5x5_manual(url_servicio):
    crudo = npy_de(r := solicitar(url_servicio, png_de(MATRIZ_5x5), "media", 3, salida="npy"))
    assert r.status_code == 200
    assert abs(crudo[2, 2] - 50.0) < 1e-4
    assert abs(crudo[0, 0] - 80.0 / 9.0) < 1e-4

    r_png = solicitar(url_servicio, png_de(MATRIZ_5x5), "media", 3)
    from PIL import Image
    import io
    salida = np.asarray(Image.open(io.BytesIO(r_png.content)))
    assert salida[2, 2] == 50 and salida[0, 0] == 8
    assert r_png.headers["X-Bloques"] == "1"


def test_bordes_2050_no_multiplo_de_bloque(url_servicio):
    rng = np.random.default_rng(2050)
    img = rng.integers(0, 256, size=(2050, 2050), dtype=np.uint8)
    filas = []
    for filtro, k in (("media", 3), ("media", 21), ("nitidez", 3), ("laplaciano", 3)):
        r = solicitar(url_servicio, png_de(img), filtro, k, salida="npy")
        assert r.status_code == 200, r.text
        crudo = npy_de(r)
        ref = convolucion_referencia(img, construir_mascara(filtro, k))
        err = np.abs(crudo.astype(np.float64) - ref)
        hilos = int(r.headers["X-Hilos-Bloque"])
        bloques = int(r.headers["X-Bloques"])
        filas.append({
            "filtro": filtro, "mascara": f"{k}x{k}", "resolucion": "2050x2050",
            "pixeles": img.size, "hilos_bloque": hilos, "bloques": bloques,
            "hilos_sobrantes": hilos * bloques - img.size,
            "error_abs_max": f"{err.max():.6e}", "mae": f"{err.mean():.6e}",
            "esquina_ref": f"{ref[0, 0]:.4f}", "esquina_gpu": f"{crudo[0, 0]:.4f}",
            "ultimo_ref": f"{ref[-1, -1]:.4f}", "ultimo_gpu": f"{crudo[-1, -1]:.4f}",
            "estado": "OK" if err.max() <= TOLERANCIA else "FALLA",
        })
        assert img.size % hilos != 0
        assert bloques == -(-img.size // hilos)
        assert np.isfinite(crudo).all()
        assert err.max() <= TOLERANCIA
    escribir_csv("bordes_2050.csv", filas)
