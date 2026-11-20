"""
Prueba 4 - Aislamiento de solicitudes y politica de cola.

a) Dos clientes envian a la vez imagenes y filtros distintos e identificables. Cada uno
   debe recibir su propio resultado (comparado con su referencia) y un X-Request-ID unico.
b) Con MAX_COLA=1 y un retardo artificial de 1,5 s por solicitud (RETARDO_PRUEBA_MS, solo
   para esta prueba), se envian 4 solicitudes simultaneas: 1 se procesa, 1 espera en cola
   y el resto debe rechazarse con 503 "cola_llena". Luego el servicio debe recuperarse.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from conftest import convolucion_referencia, escribir_csv, npy_de, png_de, solicitar
from app.filtros import construir_mascara


def _imagen_cliente(semilla: int, alto: int, ancho: int) -> np.ndarray:
    return np.random.default_rng(semilla).integers(0, 256, size=(alto, ancho), dtype=np.uint8)


def test_dos_clientes_reciben_su_propio_resultado(url_servicio):
    clientes = [
        {"cliente": "A", "img": _imagen_cliente(1, 1500, 1200), "filtro": "media", "k": 9},
        {"cliente": "B", "img": _imagen_cliente(2, 1100, 1700), "filtro": "laplaciano", "k": 3},
    ]
    filas = []
    for ronda in range(1, 6):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futuros = [
                pool.submit(solicitar, url_servicio, png_de(c["img"]), c["filtro"], c["k"], "npy", f"cliente_{c['cliente']}.png")
                for c in clientes
            ]
            respuestas = [f.result() for f in futuros]

        ids = set()
        for c, r in zip(clientes, respuestas):
            assert r.status_code == 200, r.text
            crudo = npy_de(r)
            ref = convolucion_referencia(c["img"], construir_mascara(c["filtro"], c["k"]))
            propio = crudo.shape == ref.shape and float(np.abs(crudo - ref).max()) <= 1e-3
            ids.add(r.headers["X-Request-ID"])
            filas.append({
                "ronda": ronda, "cliente": c["cliente"], "filtro": c["filtro"], "mascara": f"{c['k']}x{c['k']}",
                "imagen_enviada": f"{c['img'].shape[1]}x{c['img'].shape[0]}",
                "imagen_recibida": r.headers["X-Imagen"], "request_id": r.headers["X-Request-ID"],
                "espera_cola_ms": r.headers["X-Tiempo-Cola-ms"],
                "resultado_propio": propio, "estado": "OK" if propio else "FALLA",
            })
            assert propio
        assert len(ids) == 2
    escribir_csv("aislamiento.csv", filas)


def test_cola_llena_rechaza_y_se_recupera(url_servicio_cola_corta):
    img = png_de(_imagen_cliente(3, 256, 256))
    with ThreadPoolExecutor(max_workers=4) as pool:
        respuestas = list(pool.map(
            lambda i: solicitar(url_servicio_cola_corta, img, "media", 3, nombre=f"concurrente_{i}.png"), range(4)
        ))

    filas = []
    for i, r in enumerate(respuestas):
        cuerpo = {} if r.status_code == 200 else r.json()
        filas.append({
            "solicitud": i, "http": r.status_code, "codigo": cuerpo.get("codigo", "ok"),
            "request_id": r.headers.get("X-Request-ID", ""),
            "espera_cola_ms": r.headers.get("X-Tiempo-Cola-ms", ""),
            "retry_after": r.headers.get("Retry-After", ""),
        })
    exitosas = [f for f in filas if f["http"] == 200]
    rechazadas = [f for f in filas if f["http"] == 503]

    r_despues = solicitar(url_servicio_cola_corta, img, "media", 3)
    filas.append({"solicitud": "despues", "http": r_despues.status_code, "codigo": "ok",
                  "request_id": r_despues.headers.get("X-Request-ID", ""),
                  "espera_cola_ms": r_despues.headers.get("X-Tiempo-Cola-ms", ""), "retry_after": ""})
    escribir_csv("cola_llena.csv", filas)

    assert len(exitosas) == 2, filas            # 1 en GPU + 1 en espera
    assert len(rechazadas) == 2, filas
    assert all(f["codigo"] == "cola_llena" and f["retry_after"] for f in rechazadas)
    assert r_despues.status_code == 200
