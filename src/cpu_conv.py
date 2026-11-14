"""
cpu_conv.py - Funciones de imagen, mascaras y convolucion 2D secuencial en CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys
import time
from PIL import Image  # Equivalente a stb_image.h / stb_image_write.h solo para leer/guardar JPG

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
    datos: bytearray


def crear_imagen(ancho: int, alto: int) -> Imagen:
    return Imagen(ancho=ancho, alto=alto, datos=bytearray(ancho * alto))


def liberar_imagen(img: Imagen | None) -> None:
    if img is not None:
        img.datos = bytearray()


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

    img = crear_imagen(ancho, alto)
    for i in range(esperados):
        img.datos[i] = bloque[i]
    return img


# Carga cualquier imagen (JPG, PNG, BMP) y la devuelve en escala de grises.
# Tambien acepta archivos de grises sin compresion (.pgm) para compatibilidad interna.
def leer_imagen(ruta: str) -> Imagen | None:
    if len(ruta) >= 4 and ruta[-4:].lower() == ".pgm":
        return leer_grises(ruta)

    try:
        with Image.open(ruta) as pil_img:
            gris = pil_img.convert("L")
            ancho, alto = gris.size
            raw = gris.tobytes()
            img = crear_imagen(ancho, alto)
            n = ancho * alto
            for i in range(n):
                img.datos[i] = raw[i]
            return img
    except Exception as exc:
        print(f"No se pudo leer la imagen '{ruta}': {exc}", file=sys.stderr)
        return None


# Guarda la imagen de grises en disco en formato JPG (calidad 95)
def guardar_imagen(ruta: str, img: Imagen) -> None:
    pil_img = Image.frombytes("L", (img.ancho, img.alto), bytes(img.datos))
    pil_img.save(ruta, format="JPEG", quality=95)


def _escalar_recorte_central(base: Imagen, img: Imagen, ancho: int, alto: int) -> None:
    lado = base.ancho if base.ancho < base.alto else base.alto
    ox = (base.ancho - lado) // 2
    oy = (base.alto - lado) // 2

    for y in range(alto):
        src_y = oy + (y * lado) // alto
        for x in range(ancho):
            src_x = ox + (x * lado) // ancho
            img.datos[y * ancho + x] = base.datos[src_y * base.ancho + src_x]


def _llenar_gradiente_diagonal(img: Imagen, ancho: int, alto: int) -> None:
    for y in range(alto):
        for x in range(ancho):
            img.datos[y * ancho + x] = ((x + y) * 255) // (ancho + alto)


# Si hay imagen base, toma un recorte cuadrado del centro y lo escala a (ancho x alto)
# usando vecino mas cercano. Si no hay imagen base, llena un gradiente diagonal.
def preparar_imagen(base: Imagen | None, ancho: int, alto: int) -> Imagen:
    img = crear_imagen(ancho, alto)
    if base is not None and base.ancho > 0 and base.alto > 0:
        _escalar_recorte_central(base, img, ancho, alto)
    else:
        _llenar_gradiente_diagonal(img, ancho, alto)
    return img


# --- Definicion de mascaras ---

# Filtro de media 3x3: promedia los 9 vecinos para suavizar la imagen
MEDIA_3: list[float] = [
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
]

# Filtro de nitidez 3x3: refuerza el pixel central y resta los lados
NITIDEZ_3: list[float] = [
     0.0, -1.0,  0.0,
    -1.0,  5.0, -1.0,
     0.0, -1.0,  0.0,
]

# Filtro Laplaciano 3x3: detecta cambios bruscos de intensidad (bordes)
LAPLACIANO_3: list[float] = [
    0.0,  1.0, 0.0,
    1.0, -4.0, 1.0,
    0.0,  1.0, 0.0,
]


# Llena una mascara de media de tamano k x k donde cada valor es 1/(k*k)
def mascara_media(m: list[float], k: int) -> None:
    v = 1.0 / (k * k)
    for i in range(k * k):
        m[i] = v


# --- Funciones de convolucion ---

def _calcular_pixel_conv(
    datos: bytearray,
    ancho: int,
    alto: int,
    fila: int,
    col: int,
    mascara: list[float],
    tam: int,
    mitad: int,
) -> float:
    suma = 0.0
    # Recorrer la vecindad de tam x tam alrededor de (fila, col)
    for ky in range(tam):
        for kx in range(tam):
            y = fila + ky - mitad
            x = col + kx - mitad

            # Zero-padding: si el vecino cae fuera de la imagen, vale 0
            pixel = 0
            if 0 <= x < ancho and 0 <= y < alto:
                pixel = datos[y * ancho + x]

            suma += float(pixel) * mascara[ky * tam + kx]
    return suma


# Aplica la convolucion y guarda el resultado en float sin recortar a [0, 255].
# Esto permite comparar los decimales exactos contra la GPU antes de guardar la imagen.
def convolucion_cpu_crudo(
    entrada: Imagen,
    salida_cruda: list[float],
    mascara: list[float],
    tam: int,
) -> None:
    mitad = tam // 2  # Desplazamiento desde el centro de la mascara
    ancho = entrada.ancho
    alto = entrada.alto
    datos = entrada.datos

    for fila in range(alto):
        for col in range(ancho):
            salida_cruda[fila * ancho + col] = _calcular_pixel_conv(
                datos, ancho, alto, fila, col, mascara, tam, mitad
            )


# Limita los valores al rango [0, 255] (clamp) y los pasa a unsigned char para la salida
def convertir_crudo_a_pgm(salida_cruda: list[float], salida: Imagen) -> None:
    n = salida.ancho * salida.alto
    for i in range(n):
        v = salida_cruda[i]
        if v < 0.0:
            v = 0.0
        if v > 255.0:
            v = 255.0
        salida.datos[i] = int(v)


def convolucion_cpu(
    entrada: Imagen,
    salida: Imagen,
    mascara: list[float],
    tam: int,
) -> None:
    crudo = [0.0] * (entrada.ancho * entrada.alto)
    convolucion_cpu_crudo(entrada, crudo, mascara, tam)
    convertir_crudo_a_pgm(crudo, salida)


# --- Medicion de tiempos ---

# Hace 1 corrida de calentamiento y luego mide NUM_REPS repeticiones.
# Devuelve la mediana en milisegundos.
def medir_tiempo(
    ent: Imagen,
    sal_cruda: list[float],
    mask: list[float],
    tam: int,
    tiempos_out: list[float] | None = None,
    vmin: list[float] | None = None,
    vmax: list[float] | None = None,
) -> float:
    t = [0.0] * NUM_REPS
    convolucion_cpu_crudo(ent, sal_cruda, mask, tam)  # Calentamiento no medido

    for r in range(NUM_REPS):
        ini = time.perf_counter()
        convolucion_cpu_crudo(ent, sal_cruda, mask, tam)
        t[r] = (time.perf_counter() - ini) * 1000.0
        if tiempos_out is not None:
            tiempos_out[r] = t[r]

    # Ordenamos los 5 tiempos para sacar min, max y la mediana (posicion central)
    t.sort()
    if vmin:
        vmin[0] = t[0]
    if vmax:
        vmax[0] = t[NUM_REPS - 1]
    return t[NUM_REPS // 2]


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
    datos = bytearray(
        [
            10, 20, 30, 40, 50,
            20, 30, 40, 50, 60,
            30, 40, 50, 60, 70,
            40, 50, 60, 70, 80,
            50, 60, 70, 80, 90,
        ]
    )
    ent = Imagen(ancho=5, alto=5, datos=datos)
    sal = crear_imagen(5, 5)
    crudo = [0.0] * 25

    m = [0.0] * 9
    mascara_media(m, 3)
    convolucion_cpu_crudo(ent, crudo, m, 3)
    convertir_crudo_a_pgm(crudo, sal)

    ok_centro = (abs(crudo[12] - 50.0) < 1e-4) and (sal.datos[12] == 50)
    ok_esquina = (abs(crudo[0] - (80.0 / 9.0)) < 1e-4) and (sal.datos[0] == 8)

    print(
        f"  Pixel central (2,2): esperado=50.0000 (50) | obtenido={crudo[12]:.4f} ({sal.datos[12]}) -> "
        f"{'OK' if ok_centro else 'FALLA'}"
    )
    print(
        f"  Pixel esquina (0,0): esperado=8.8889  (8)  | obtenido={crudo[0]:.4f} ({sal.datos[0]}) -> "
        f"{'OK' if ok_esquina else 'FALLA'}\n"
    )

    liberar_imagen(sal)


if __name__ == "__main__":
    validar_manual()
