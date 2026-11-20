"""
Prueba 3 - Entradas invalidas y recuperacion del servicio.

Casos: archivo que no es imagen, imagen con extension falsa, archivo que supera el
limite configurado, filtro desconocido, mascara no permitida (par, intermedia 5, fuera de rango
y no numerica; solo se aceptan 3, 9 y 21) y campo faltante. Despues de todos los errores, una solicitud valida debe
responder 200 y /api/salud debe seguir en "ok".
"""

from __future__ import annotations

import httpx
import numpy as np

from conftest import escribir_csv, png_de, solicitar


def test_entradas_invalidas_y_recuperacion(url_servicio):
    valida = png_de(np.full((64, 64), 128, dtype=np.uint8))
    limite_mb = httpx.get(f"{url_servicio}/api/salud").json()["limites"]["max_mb"]
    grande = b"\x89PNG" + b"0" * ((limite_mb + 1) * 2**20)

    casos = [
        ("archivo_no_imagen", lambda: solicitar(url_servicio, b"esto no es una imagen", "media", 3, nombre="texto.png"), 415),
        ("pdf_renombrado_jpg", lambda: solicitar(url_servicio, b"%PDF-1.7\n%...", "media", 3, nombre="doc.jpg"), 415),
        ("archivo_supera_limite", lambda: solicitar(url_servicio, grande, "media", 3, nombre="grande.png"), 413),
        ("filtro_desconocido", lambda: solicitar(url_servicio, valida, "sepia", 3), 422),
        ("mascara_par", lambda: solicitar(url_servicio, valida, "media", 4), 422),
        ("mascara_intermedia_5", lambda: solicitar(url_servicio, valida, "media", 5), 422),
        ("mascara_fuera_de_rango", lambda: solicitar(url_servicio, valida, "media", 23), 422),
        ("mascara_no_numerica", lambda: solicitar(url_servicio, valida, "media", "abc"), 422),
        ("salida_no_valida", lambda: solicitar(url_servicio, valida, "media", 3, salida="gif"), 422),
        ("sin_archivo", lambda: httpx.post(f"{url_servicio}/api/procesar", data={"filtro": "media"}), 422),
    ]

    filas = []
    for nombre, enviar, esperado in casos:
        r = enviar()
        cuerpo = r.json()
        filas.append({
            "caso": nombre, "http_esperado": esperado, "http_obtenido": r.status_code,
            "codigo": cuerpo.get("codigo"), "mensaje": cuerpo.get("detail"),
            "request_id": cuerpo.get("request_id") or "",
            "estado": "OK" if r.status_code == esperado else "FALLA",
        })

    # Recuperacion: el servicio sigue atendiendo despues de los errores
    r_ok = solicitar(url_servicio, valida, "media", 3)
    salud = httpx.get(f"{url_servicio}/api/salud").json()
    filas.append({
        "caso": "recuperacion_solicitud_valida", "http_esperado": 200, "http_obtenido": r_ok.status_code,
        "codigo": salud["estado"], "mensaje": f"GPU {salud['gpu']['nombre']}",
        "request_id": r_ok.headers.get("X-Request-ID", ""),
        "estado": "OK" if r_ok.status_code == 200 and salud["estado"] == "ok" else "FALLA",
    })
    escribir_csv("entradas_invalidas.csv", filas)

    fallas = [f for f in filas if f["estado"] != "OK"]
    assert not fallas, fallas
