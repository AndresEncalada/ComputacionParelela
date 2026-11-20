"""
procesamiento.py - Procesamiento completo de una imagen, independiente de HTTP.

Lo usan igual el servicio web (app/main.py) y la ejecucion directa (app/directo.py),
de modo que ambos caminos ejecutan el mismo codigo:

    bytes -> validar/decodificar -> escala de grises uint8 -> GPU (kernel practica 02)
          -> salida float32 -> recorte [0, 255] -> PNG (o .npy crudo para validacion)

Todos los tiempos se miden con time.perf_counter (reloj monotonico) salvo H2D, kernel y
D2H, que vienen de eventos CUDA (ver motor_gpu.py).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Config
from app.filtros import construir_mascara
from app.servicios.motor_gpu import MotorGPU

# Formatos reales aceptados (los detecta Pillow por contenido, no por extension).
# PPM cubre los archivos .pgm (P5).
FORMATOS_PERMITIDOS = {"JPEG", "PNG", "BMP", "PPM"}
SALIDAS = {"png": "image/png", "npy": "application/octet-stream"}


class EntradaInvalida(ValueError):
    def __init__(self, mensaje: str, estado_http: int, codigo: str) -> None:
        super().__init__(mensaje)
        self.estado_http = estado_http
        self.codigo = codigo


@dataclass
class Resultado:
    contenido: bytes
    tipo_mime: str
    ancho: int
    alto: int
    formato_entrada: str
    hilos_bloque: int
    bloques: int
    tiempos_ms: dict[str, float]
    crudo: np.ndarray | None = None  # salida float32 sin recortar (solo si se pide)


def decodificar(contenido: bytes, config: Config) -> tuple[np.ndarray, str]:
    """Valida tamano, formato real y dimensiones; devuelve la imagen en grises (uint8)."""
    if len(contenido) == 0:
        raise EntradaInvalida("El archivo está vacío.", 422, "archivo_vacio")
    if len(contenido) > config.max_bytes:
        raise EntradaInvalida(
            f"El archivo pesa {len(contenido) / 2**20:.1f} MB y el límite es "
            f"{config.max_bytes / 2**20:.0f} MB.", 413, "archivo_demasiado_grande")
    try:
        img = Image.open(io.BytesIO(contenido))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise EntradaInvalida(
            "El archivo no es una imagen válida. Formatos aceptados: JPG, PNG, BMP o PGM.",
            415, "formato_no_valido") from None
    if img.format not in FORMATOS_PERMITIDOS:
        raise EntradaInvalida(
            f"Formato {img.format} no soportado. Formatos aceptados: JPG, PNG, BMP o PGM.",
            415, "formato_no_valido")

    ancho, alto = img.size  # se lee de la cabecera, antes de decodificar los pixeles
    if max(ancho, alto) > config.max_lado or ancho * alto > config.max_pixeles:
        raise EntradaInvalida(
            f"La imagen mide {ancho}×{alto} px. Límite: {config.max_lado} px por lado y "
            f"{config.max_pixeles / 1e6:.0f} megapíxeles.", 413, "dimensiones_excedidas")

    try:
        # Misma conversion a grises que la practica 02 (cpu_conv.leer_imagen)
        gris = img.convert("L")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise EntradaInvalida(f"La imagen está dañada o incompleta ({exc}).", 422, "imagen_corrupta") from None
    return np.asarray(gris, dtype=np.uint8), img.format


def a_uint8(crudo: np.ndarray) -> np.ndarray:
    """Recorte a [0, 255] y truncamiento, igual que cpu_conv.convertir_crudo_a_pgm."""
    return np.clip(crudo, 0.0, 255.0).astype(np.uint8)


def codificar_png(pixeles: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixeles, mode="L").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def codificar_npy(crudo: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, crudo, allow_pickle=False)
    return buf.getvalue()


def procesar(
    contenido: bytes,
    filtro: str,
    k: int,
    motor: MotorGPU,
    config: Config,
    salida: str = "png",
    conservar_crudo: bool = False,
) -> Resultado:
    """Procesa una imagen completa. Lanza EntradaInvalida o errores de motor_gpu."""
    t_ini = time.perf_counter()

    imagen, formato = decodificar(contenido, config)
    mascara = construir_mascara(filtro, k)
    t_dec = time.perf_counter()

    futuro = motor.enviar(imagen, mascara, k)
    salida_gpu = futuro.result(timeout=config.timeout_s)
    t_gpu = time.perf_counter()

    if salida == "npy":
        datos = codificar_npy(salida_gpu.crudo)
    else:
        datos = codificar_png(a_uint8(salida_gpu.crudo))
    t_fin = time.perf_counter()

    tiempos = {
        "decodificacion": (t_dec - t_ini) * 1000.0,
        **salida_gpu.tiempos_ms,                      # cola, reserva, h2d, kernel, d2h, liberacion
        "gpu_total": (t_gpu - t_dec) * 1000.0 - salida_gpu.tiempos_ms["cola"],
        "codificacion": (t_fin - t_gpu) * 1000.0,
    }
    # Procesamiento del servidor sin contar la espera en cola
    tiempos["servidor"] = (t_fin - t_ini) * 1000.0 - tiempos["cola"]

    return Resultado(
        contenido=datos,
        tipo_mime=SALIDAS.get(salida, "image/png"),
        ancho=imagen.shape[1],
        alto=imagen.shape[0],
        formato_entrada=formato,
        hilos_bloque=salida_gpu.hilos_bloque,
        bloques=salida_gpu.bloques,
        tiempos_ms=tiempos,
        crudo=salida_gpu.crudo if conservar_crudo else None,
    )
