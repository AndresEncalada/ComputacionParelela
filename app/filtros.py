"""
filtros.py - Catalogo de filtros que expone la interfaz web.

Es la unica fuente de verdad para el frontend (carrusel) y para la validacion
de la API. Los nombres de `id` coinciden con los usados en src/main.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MASCARA_MIN = 3
MASCARA_MAX = 21
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


def catalogo() -> dict:
    return {
        "filtros": [asdict(f) for f in FILTROS],
        "mascara": {"min": MASCARA_MIN, "max": MASCARA_MAX, "defecto": MASCARA_DEFECTO},
    }
