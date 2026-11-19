"""
procesamiento.py - Punto de integracion con la logica de convolucion.

Aqui se conectaran experimentos/src/cpu_conv.py y experimentos/src/gpu_conv.py. La API ya llama a
`aplicar_filtro`; solo falta implementarla.
"""

from __future__ import annotations


class ProcesamientoNoImplementado(NotImplementedError):
    pass


def aplicar_filtro(contenido: bytes, filtro: str, mascara: int) -> bytes:
    """
    Recibe la imagen subida (bytes en JPG/PNG/BMP), el id del filtro y el
    tamano de mascara k (solo relevante para 'media'; 3 para los demas).

    Debe devolver la imagen filtrada en escala de grises codificada como PNG.

    TODO: leer bytes con Pillow -> Imagen, construir la mascara
    (mascara_media / NITIDEZ_3 / LAPLACIANO_3), ejecutar la convolucion en GPU
    y codificar el resultado.
    """
    raise ProcesamientoNoImplementado("La logica de convolucion aun no esta conectada.")
