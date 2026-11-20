"""
filtros.py - Catalogo de filtros y construccion de mascaras.

Es la unica fuente de verdad para el frontend (carrusel) y para la validacion
de la API. Los coeficientes vienen de la practica 02 (app/practica02.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.practica02 import LAPLACIANO_3, NITIDEZ_3, mascara_media

# Tamanos de la media definidos en la practica 02 (unicos permitidos)
MASCARAS_MEDIA: tuple[int, ...] = (3, 9, 21)
MASCARA_DEFECTO = 9


@dataclass(frozen=True)
class Filtro:
    id: str
    nombre: str
    descripcion: str
    miniatura: str            # ruta relativa a /static
    mascara_variable: bool    # True -> el usuario elige k (k x k)


FILTROS: tuple[Filtro, ...] = (
    Filtro(
        id="media",
        nombre="Media",
        descripcion="Suavizado: promedia cada píxel con sus vecinos en una ventana k×k.",
        miniatura="img/filtros/media.jpg",
        mascara_variable=True,
    ),
    Filtro(
        id="nitidez",
        nombre="Nitidez",
        descripcion="Realce de bordes con la máscara 3×3 [0,-1,0; -1,5,-1; 0,-1,0].",
        miniatura="img/filtros/nitidez.jpg",
        mascara_variable=False,
    ),
    Filtro(
        id="laplaciano",
        nombre="Laplaciano",
        descripcion="Detección de bordes con la máscara 3×3 [0,1,0; 1,-4,1; 0,1,0].",
        miniatura="img/filtros/laplaciano.jpg",
        mascara_variable=False,
    ),
)

FILTROS_POR_ID = {f.id: f for f in FILTROS}


class ParametroInvalido(ValueError):
    """Filtro o tamano de mascara no permitidos."""


def mascara_permitida(k: int) -> bool:
    return k in MASCARAS_MEDIA


def resolver_parametros(filtro: str, mascara: str | int | None) -> tuple[str, int]:
    """Valida filtro y mascara recibidos como texto. Devuelve (filtro, k)."""
    if filtro not in FILTROS_POR_ID:
        validos = ", ".join(FILTROS_POR_ID)
        raise ParametroInvalido(f"Filtro desconocido: '{filtro}'. Valores permitidos: {validos}.")
    if not FILTROS_POR_ID[filtro].mascara_variable:
        return filtro, 3
    try:
        k = int(mascara) if mascara not in (None, "") else MASCARA_DEFECTO
    except (TypeError, ValueError):
        raise ParametroInvalido(f"La máscara debe ser un número entero, se recibió '{mascara}'.") from None
    if not mascara_permitida(k):
        raise ParametroInvalido(
            f"Tamaño de máscara no permitido: {k}. Valores permitidos para la media: "
            f"{', '.join(map(str, MASCARAS_MEDIA))}."
        )
    return filtro, k


def construir_mascara(filtro: str, k: int) -> np.ndarray:
    """Mascara k x k en float32 con los mismos coeficientes de la practica 02."""
    if filtro == "media":
        m = [0.0] * (k * k)
        mascara_media(m, k)
    elif filtro == "nitidez":
        m = NITIDEZ_3
    elif filtro == "laplaciano":
        m = LAPLACIANO_3
    else:
        raise ParametroInvalido(f"Filtro desconocido: '{filtro}'.")
    return np.asarray(m, dtype=np.float32)


def catalogo(limites: dict | None = None) -> dict:
    datos = {
        "filtros": [asdict(f) for f in FILTROS],
        "mascara": {"valores": list(MASCARAS_MEDIA), "defecto": MASCARA_DEFECTO},
    }
    if limites:
        datos["limites"] = limites
    return datos
