"""
config.py - Limites y parametros del servicio, configurables por variables de entorno.

Los valores por defecto estan pensados para una GPU de 4 GB:
  6000 x 6000 px -> entrada uint8 36 MB + salida float32 144 MB en la GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _entero(nombre: str, defecto: int) -> int:
    valor = os.environ.get(nombre)
    return int(valor) if valor else defecto


@dataclass(frozen=True)
class Config:
    max_bytes: int            # tamano maximo del archivo subido
    max_lado: int             # ancho o alto maximo en pixeles
    max_pixeles: int          # ancho * alto maximo
    max_cola: int             # solicitudes en espera (sin contar la que esta en la GPU)
    hilos_bloque: int         # hilos por bloque del kernel (distribucion 1D en X)
    timeout_s: int            # tiempo maximo de espera de una solicitud
    log_archivo: str | None   # ruta opcional para conservar el registro
    retardo_prueba_ms: int    # SOLO PRUEBAS: retardo artificial por solicitud en la GPU (0 = desactivado)


def cargar() -> Config:
    return Config(
        max_bytes=_entero("MAX_MB", 25) * 1024 * 1024,
        max_lado=_entero("MAX_LADO", 8000),
        max_pixeles=_entero("MAX_MEGAPIXELES", 40) * 1_000_000,
        max_cola=_entero("MAX_COLA", 4),
        hilos_bloque=_entero("HILOS_BLOQUE", 256),
        timeout_s=_entero("TIMEOUT_S", 120),
        log_archivo=os.environ.get("LOG_ARCHIVO") or None,
        retardo_prueba_ms=_entero("RETARDO_PRUEBA_MS", 0),
    )


CONFIG = cargar()
