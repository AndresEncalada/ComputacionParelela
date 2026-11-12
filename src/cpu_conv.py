"""
cpu_conv.py - Funciones de imagen, mascaras y convolucion 2D secuencial en CPU
Traduccion directa de src/cpu_conv.c a Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys
import time
import numpy as np
from PIL import Image

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

NUM_REPS = 5  # 5 repeticiones medidas (+ 1 previa de calentamiento)

_PGM_HEADER_RE = re.compile(
    rb"^P5\s+(?:#[^\r\n]*[\r\n]+\s*)*(\d+)\s+"
    rb"(?:#[^\r\n]*[\r\n]+\s*)*(\d+)\s+"
    rb"(?:#[^\r\n]*[\r\n]+\s*)*(\d+)\s"
)


# Estructura basica para manejar la imagen en un arreglo 1D
# El pixel (fila, col) se ubica en datos[fila * ancho + col]
@dataclass
class Imagen:
    ancho: int
    alto: int
    datos: np.ndarray


def crear_imagen(ancho: int, alto: int) -> Imagen:
    datos = np.zeros(ancho * alto, dtype=np.uint8)
    return Imagen(ancho=ancho, alto=alto, datos=datos)


def liberar_imagen(img: Imagen | None) -> None:
    if img is not None:
        img.datos = None


# Lee un archivo de grises sin compresion (formato interno P5)
def leer_grises(ruta: str) -> Imagen | None:
    if not os.path.isfile(ruta):
        return None
    try:
        with open(ruta, "rb") as f:
            contenido = f.read()
    except OSError:
        return None

    coincidencia = _PGM_HEADER_RE.match(contenido)
    if coincidencia is None:
        return None

    ancho = int(coincidencia.group(1))
    alto = int(coincidencia.group(2))
    inicio = coincidencia.end()
    esperados = ancho * alto
    bloque = contenido[inicio : inicio + esperados]
    if len(bloque) != esperados:
        return None

    datos = np.frombuffer(bloque, dtype=np.uint8).copy()
    return Imagen(ancho=ancho, alto=alto, datos=datos)


# Carga cualquier imagen (JPG, PNG, BMP) y la devuelve en escala de grises.
# Tambien acepta archivos de grises sin compresion (.pgm) para compatibilidad interna.
def leer_imagen(ruta: str) -> Imagen | None:
    if len(ruta) >= 4 and ruta[-4:].lower() == ".pgm":
        return leer_grises(ruta)

    try:
        with Image.open(ruta) as pil_img:
            gris = pil_img.convert("L")
            ancho, alto = gris.size
            datos = np.asarray(gris, dtype=np.uint8).reshape(-1).copy()
            return Imagen(ancho=ancho, alto=alto, datos=datos)
    except Exception as exc:
        print(f"No se pudo leer la imagen '{ruta}': {exc}", file=sys.stderr)
        return None


# Guarda la imagen de grises en disco en formato JPG (calidad 95)
def guardar_imagen(ruta: str, img: Imagen) -> None:
    arr_2d = img.datos.reshape((img.alto, img.ancho))
    pil_img = Image.fromarray(arr_2d, mode="L")
    pil_img.save(ruta, format="JPEG", quality=95)


# Si hay imagen base, toma un recorte cuadrado del centro y lo escala a (ancho x alto)
# usando vecino mas cercano. Si no hay imagen base, llena un gradiente diagonal.
def preparar_imagen(base: Imagen | None, ancho: int, alto: int) -> Imagen:
    img = crear_imagen(ancho, alto)
    if base is not None and base.ancho > 0 and base.alto > 0:
        lado = min(base.ancho, base.alto)
        ox = (base.ancho - lado) // 2
        oy = (base.alto - lado) // 2

        y = np.arange(alto, dtype=np.int64)
        x = np.arange(ancho, dtype=np.int64)
        src_y = oy + (y * lado) // alto
        src_x = ox + (x * lado) // ancho

        base_2d = base.datos.reshape((base.alto, base.ancho))
        recorte = base_2d[np.ix_(src_y, src_x)]
        img.datos[:] = recorte.reshape(-1)
    else:
        y = np.arange(alto, dtype=np.int64)[:, None]
        x = np.arange(ancho, dtype=np.int64)[None, :]
        grad = ((x + y) * 255) // (ancho + alto)
        img.datos[:] = grad.astype(np.uint8).reshape(-1)

    return img


# --- Definicion de mascaras ---

# Filtro de media 3x3: promedia los 9 vecinos para suavizar la imagen
MEDIA_3 = np.array(
    [
        1 / 9.0, 1 / 9.0, 1 / 9.0,
        1 / 9.0, 1 / 9.0, 1 / 9.0,
        1 / 9.0, 1 / 9.0, 1 / 9.0,
    ],
    dtype=np.float32,
)

# Filtro de nitidez 3x3: refuerza el pixel central y resta los lados
NITIDEZ_3 = np.array(
    [
         0.0, -1.0,  0.0,
        -1.0,  5.0, -1.0,
         0.0, -1.0,  0.0,
    ],
    dtype=np.float32,
)

# Filtro Laplaciano 3x3: detecta cambios bruscos de intensidad (bordes)
LAPLACIANO_3 = np.array(
    [
        0.0,  1.0, 0.0,
        1.0, -4.0, 1.0,
        0.0,  1.0, 0.0,
    ],
    dtype=np.float32,
)


# Llena una mascara de media de tamano k x k donde cada valor es 1/(k*k)
def mascara_media(m: np.ndarray | int, k: int | None = None) -> np.ndarray:
    if k is None and isinstance(m, int):
        k = m
        m = np.empty(k * k, dtype=np.float32)
    v = np.float32(1.0 / (k * k))
    m[: k * k] = v
    return m


# --- Funciones de convolucion ---

if _HAS_NUMBA:
    @njit(cache=True)
    def _calcular_pixel_conv(
        datos: np.ndarray,
        ancho: int,
        alto: int,
        fila: int,
        col: int,
        mascara: np.ndarray,
        tam: int,
        mitad: int,
    ) -> np.float32:
        suma = np.float32(0.0)
        for ky in range(tam):
            for kx in range(tam):
                y = fila + ky - mitad
                x = col + kx - mitad
                if 0 <= x < ancho and 0 <= y < alto:
                    pixel = np.float32(datos[y * ancho + x])
                    suma = np.float32(suma + pixel * mascara[ky * tam + kx])
        return suma

    @njit(cache=True)
    def _convolucion_cpu_crudo_kernel(
        datos: np.ndarray,
        ancho: int,
        alto: int,
        salida_cruda: np.ndarray,
        mascara: np.ndarray,
        tam: int,
    ) -> None:
        mitad = tam // 2
        for fila in range(alto):
            for col in range(ancho):
                salida_cruda[fila * ancho + col] = _calcular_pixel_conv(
                    datos, ancho, alto, fila, col, mascara, tam, mitad
                )
else:
    def _convolucion_cpu_crudo_kernel(
        datos: np.ndarray,
        ancho: int,
        alto: int,
        salida_cruda: np.ndarray,
        mascara: np.ndarray,
        tam: int,
    ) -> None:
        mitad = tam // 2
        ent_2d = datos.reshape((alto, ancho)).astype(np.float32)
        padded = np.pad(ent_2d, mitad, mode="constant", constant_values=0.0)
        sal_2d = salida_cruda.reshape((alto, ancho))
        sal_2d.fill(0.0)
        mask_f32 = mascara.astype(np.float32)

        for ky in range(tam):
            for kx in range(tam):
                peso = mask_f32[ky * tam + kx]
                sal_2d += padded[ky : ky + alto, kx : kx + ancho] * peso


# Aplica la convolucion y guarda el resultado en float32 sin recortar a [0, 255].
# Esto permite comparar los decimales exactos contra la GPU antes de guardar la imagen.
def convolucion_cpu_crudo(
    entrada: Imagen,
    salida_cruda: np.ndarray,
    mascara: np.ndarray,
    tam: int,
) -> None:
    mask_f32 = np.asarray(mascara, dtype=np.float32)
    _convolucion_cpu_crudo_kernel(
        entrada.datos,
        entrada.ancho,
        entrada.alto,
        salida_cruda,
        mask_f32,
        tam,
    )


# Limita los valores al rango [0, 255] (clamp) y los pasa a unsigned char (uint8)
def convertir_crudo_a_pgm(salida_cruda: np.ndarray, salida: Imagen) -> None:
    n = salida.ancho * salida.alto
    salida.datos[:n] = np.clip(salida_cruda[:n], 0.0, 255.0).astype(np.uint8)


def convolucion_cpu(
    entrada: Imagen,
    salida: Imagen,
    mascara: np.ndarray,
    tam: int,
) -> None:
    crudo = np.empty(entrada.ancho * entrada.alto, dtype=np.float32)
    convolucion_cpu_crudo(entrada, crudo, mascara, tam)
    convertir_crudo_a_pgm(crudo, salida)


# --- Medicion de tiempos ---

# Hace 1 corrida de calentamiento y luego mide NUM_REPS repeticiones.
# Devuelve la mediana en milisegundos.
def medir_tiempo(
    ent: Imagen,
    sal_cruda: np.ndarray,
    mask: np.ndarray,
    tam: int,
    tiempos_out: list[float] | np.ndarray | None = None,
    vmin: list[float] | None = None,
    vmax: list[float] | None = None,
) -> float:
    t = np.zeros(NUM_REPS, dtype=np.float64)
    convolucion_cpu_crudo(ent, sal_cruda, mask, tam)  # Calentamiento no medido

    for r in range(NUM_REPS):
        ini = time.perf_counter()
        convolucion_cpu_crudo(ent, sal_cruda, mask, tam)
        t[r] = (time.perf_counter() - ini) * 1000.0
        if tiempos_out is not None:
            tiempos_out[r] = float(t[r])

    t_ord = np.sort(t)
    if vmin:
        vmin[0] = float(t_ord[0])
    if vmax:
        vmax[0] = float(t_ord[NUM_REPS - 1])
    return float(t_ord[NUM_REPS // 2])


# --- Validacion con matriz pequena 5x5 ---

def validar_manual() -> None:
    print("-- Validacion manual (Matriz 5x5) --")

    # Matriz de prueba 5x5:
    #   10 20 30 40 50
    #   20 30 40 50 60
    #   30 40 50 60 70
    #   40 50 60 70 80
    #   50 60 70 80 90
    #
    # Calculo a mano con filtro de media 3x3 y zero-padding:
    #   Centro (2,2): (30+40+50 + 40+50+60 + 50+60+70) / 9 = 450 / 9 = 50.0
    #   Esquina (0,0): (0+0+0 + 0+10+20 + 0+20+30) / 9 = 80 / 9 = 8.8889 (en entero: 8)
    datos = np.array(
        [
            10, 20, 30, 40, 50,
            20, 30, 40, 50, 60,
            30, 40, 50, 60, 70,
            40, 50, 60, 70, 80,
            50, 60, 70, 80, 90,
        ],
        dtype=np.uint8,
    )
    ent = Imagen(ancho=5, alto=5, datos=datos)
    sal = crear_imagen(5, 5)
    crudo = np.zeros(25, dtype=np.float32)

    m = np.zeros(9, dtype=np.float32)
    mascara_media(m, 3)
    convolucion_cpu_crudo(ent, crudo, m, 3)
    convertir_crudo_a_pgm(crudo, sal)

    ok_centro = (abs(float(crudo[12]) - 50.0) < 1e-4) and (int(sal.datos[12]) == 50)
    ok_esquina = (abs(float(crudo[0]) - (80.0 / 9.0)) < 1e-4) and (int(sal.datos[0]) == 8)

    print(
        f"  Pixel central (2,2): esperado=50.0000 (50) | obtenido={crudo[12]:.4f} ({int(sal.datos[12])}) -> "
        f"{'OK' if ok_centro else 'FALLA'}"
    )
    print(
        f"  Pixel esquina (0,0): esperado=8.8889  (8)  | obtenido={crudo[0]:.4f} ({int(sal.datos[0])}) -> "
        f"{'OK' if ok_esquina else 'FALLA'}\n"
    )

    liberar_imagen(sal)


if __name__ == "__main__":
    validar_manual()
