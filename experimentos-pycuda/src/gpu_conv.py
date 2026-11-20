"""
gpu_conv.py - Kernel CUDA 1D y medicion de tiempos en GPU (PyCUDA).
"""

from __future__ import annotations

import math
import time
import warnings
from typing import TextIO

import numpy as np

_t_ctx_ini = time.perf_counter()
import pycuda.autoinit  # Crea el contexto CUDA sobre la GPU 0 (equivale al contexto implicito de CUDA C)
TIEMPO_INIT_CTX_MS = (time.perf_counter() - _t_ctx_ini) * 1000.0

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from kernel_conv import KERNEL_SRC, NOMBRE_KERNEL
from cpu_conv import (
    NUM_REPS,
    Imagen,
    convolucion_cpu_crudo,
    mascara_media,
    validar_manual,
)

HILOS_BASE = 256  # Tamano de bloque fijo para los primeros 3 experimentos
# Con mascaras 9x9 (81 operaciones) o 21x21 (441), el error acumulado en float32
# puede superar 1e-4. 1e-3 es conservador y correcto para aritmetica de precision simple.
TOLERANCIA = 1e-3

# Nota: en PyCUDA no hace falta la macro CUDA_CHECK. Cada llamada (mem_alloc, memcpy,
# lanzamiento del kernel, eventos) lanza una excepcion pycuda.driver.Error si CUDA
# devuelve un codigo de error, mostrando el mensaje equivalente a cudaGetErrorString.



# PyCUDA compila el kernel con nvcc al importar el modulo.
# En Windows, cl.exe imprime el nombre del archivo ("kernel.cu") y PyCUDA lo muestra
# como advertencia aunque la compilacion fue correcta; solo se oculta ese aviso.
_t_comp_ini = time.perf_counter()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The CUDA compiler succeeded")
    _modulo = SourceModule(KERNEL_SRC)
kernelConvolucionGPU = _modulo.get_function(NOMBRE_KERNEL)
TIEMPO_COMPILACION_MS = (time.perf_counter() - _t_comp_ini) * 1000.0


# --- Informacion de la GPU ---

# Muestra el nombre, Compute Capability y VRAM de la GPU 0 (equivale a cudaGetDeviceProperties en main.cu)
def mostrar_info_gpu() -> None:
    prop = pycuda.autoinit.device
    major, minor = prop.compute_capability()
    print(
        f"GPU detectada: {prop.name()} | Compute Capability: {major}.{minor} | "
        f"VRAM: {prop.total_memory() / (1024.0 * 1024.0):.0f} MB"
    )
    print(
        f"Preparacion GPU: init_contexto={TIEMPO_INIT_CTX_MS:.2f} ms | "
        f"compilacion_kernel={TIEMPO_COMPILACION_MS:.2f} ms"
    )
    print("Nota: cudaMalloc (mem_alloc) y cudaFree (free) estan excluidos de la medicion de tiempos.\n")


# --- Validacion numerica ---

# Calcula el error absoluto maximo y el error absoluto medio (MAE) entre CPU y GPU
def comparar_cpu_vs_gpu(cpu_crudo, gpu_crudo, n: int) -> tuple[float, float]:
    max_e = 0.0
    suma_e = 0.0
    for i in range(n):
        diff = abs(float(cpu_crudo[i]) - float(gpu_crudo[i]))
        if diff > max_e:
            max_e = diff
        suma_e += diff
    max_err = max_e
    mae = suma_e / n
    return max_err, mae


# Prueba la matriz pequena de 5x5 en GPU y la compara con el resultado de CPU
def validar_manual_cpu_gpu() -> None:
    validar_manual()

    datos = bytearray([
        10, 20, 30, 40, 50,
        20, 30, 40, 50, 60,
        30, 40, 50, 60, 70,
        40, 50, 60, 70, 80,
        50, 60, 70, 80, 90,
    ])
    ent = Imagen(ancho=5, alto=5, datos=datos)
    crudo_cpu = [0.0] * 25
    crudo_gpu = np.zeros(25, dtype=np.float32)  # Arreglo float32 contiguo para recibir la copia D2H
    m = [0.0] * 9
    mascara_media(m, 3)

    convolucion_cpu_crudo(ent, crudo_cpu, m, 3)

    # Arreglos contiguos en host con los mismos tipos que en C (unsigned char y float)
    h_datos = np.frombuffer(datos, dtype=np.uint8)
    h_m = np.array(m, dtype=np.float32)

    d_ent = cuda.mem_alloc(25 * np.dtype(np.uint8).itemsize)
    d_sal = cuda.mem_alloc(25 * np.dtype(np.float32).itemsize)
    d_mask = cuda.mem_alloc(9 * np.dtype(np.float32).itemsize)

    cuda.memcpy_htod(d_ent, h_datos)
    cuda.memcpy_htod(d_mask, h_m)

    # Lanzamos 32 hilos para 25 pixeles (los 7 hilos sobrantes prueban el if idx >= ancho*alto)
    kernelConvolucionGPU(d_ent, d_sal, d_mask, np.int32(5), np.int32(5), np.int32(3),
                         block=(32, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(crudo_gpu, d_sal)

    max_err, mae = comparar_cpu_vs_gpu(crudo_cpu, crudo_gpu, 25)
    print(
        f"  [GPU 5x5 vs Manual/CPU] Central={crudo_gpu[12]:.4f} | Esquina={crudo_gpu[0]:.4f} | "
        f"MaxErr={max_err:.2e} | MAE={mae:.2e} -> {'OK' if max_err <= TOLERANCIA else 'FALLA'}\n"
    )

    d_ent.free()
    d_sal.free()
    d_mask.free()


# --- Medicion de tiempos en GPU ---

# Mide con cuda.Event los 3 pasos por separado:
# 1) Copia H2D (CPU a GPU), 2) Kernel, 3) Copia D2H (GPU a CPU), y la suma de los tres.
# La reserva de memoria con cuda.mem_alloc queda fuera de la medicion.
# Devuelve (med_h2d, med_ker, med_d2h, med_tot, iqr_tot, std_tot, iqr_ker, std_ker).
def medir_tiempo_gpu(
    ent: Imagen,
    h_salida_cruda: list[float],
    mask: list[float],
    tam: int,
    hilos_bloque: int,
    filtro: str,
    csv_crudo: TextIO | None,
) -> tuple[float, float, float, float, float, float, float, float]:
    n = ent.ancho * ent.alto
    bytes_img = n * np.dtype(np.uint8).itemsize
    bytes_sal = n * np.dtype(np.float32).itemsize
    bytes_mask = tam * tam * np.dtype(np.float32).itemsize

    # Arreglos contiguos en host con los mismos tipos que en C (unsigned char y float).
    # h_sal recibe la copia D2H; al final se pasa a h_salida_cruda (lista de Python).
    h_ent = np.frombuffer(ent.datos, dtype=np.uint8, count=n)
    h_mask = np.array(mask[: tam * tam], dtype=np.float32)
    h_sal = np.zeros(n, dtype=np.float32)

    # Cantidad de bloques redondeando hacia arriba para cubrir todos los pixeles
    bloques = (n + hilos_bloque - 1) // hilos_bloque

    d_ent = cuda.mem_alloc(bytes_img)
    d_sal = cuda.mem_alloc(bytes_sal)
    d_mask = cuda.mem_alloc(bytes_mask)

    ini = cuda.Event()
    fin = cuda.Event()

    # Ejecucion de calentamiento (no se mide)
    cuda.memcpy_htod(d_ent, h_ent)
    cuda.memcpy_htod(d_mask, h_mask)
    kernelConvolucionGPU(d_ent, d_sal, d_mask, np.int32(ent.ancho), np.int32(ent.alto), np.int32(tam),
                         block=(hilos_bloque, 1, 1), grid=(bloques, 1))
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(h_sal, d_sal)

    t_h2d = [0.0] * NUM_REPS
    t_ker = [0.0] * NUM_REPS
    t_d2h = [0.0] * NUM_REPS
    t_tot = [0.0] * NUM_REPS
    cfg = f"{hilos_bloque}_hilos_{bloques}_bloques"

    for r in range(NUM_REPS):
        # 1. Medir transferencia CPU -> GPU (imagen y mascara)
        ini.record()
        cuda.memcpy_htod(d_ent, h_ent)
        cuda.memcpy_htod(d_mask, h_mask)
        fin.record()
        fin.synchronize()
        ms_h2d = fin.time_since(ini)

        # 2. Medir solo la ejecucion del kernel
        ini.record()
        kernelConvolucionGPU(d_ent, d_sal, d_mask, np.int32(ent.ancho), np.int32(ent.alto), np.int32(tam),
                             block=(hilos_bloque, 1, 1), grid=(bloques, 1))
        fin.record()
        fin.synchronize()
        ms_ker = fin.time_since(ini)

        # 3. Medir transferencia GPU -> CPU (arreglo de salida)
        ini.record()
        cuda.memcpy_dtoh(h_sal, d_sal)
        fin.record()
        fin.synchronize()
        ms_d2h = fin.time_since(ini)

        t_h2d[r] = ms_h2d
        t_ker[r] = ms_ker
        t_d2h[r] = ms_d2h
        t_tot[r] = ms_h2d + ms_ker + ms_d2h

        if csv_crudo:
            csv_crudo.write(f"GPU_H2D,{filtro},{ent.ancho}x{ent.alto},{tam}x{tam},{cfg},{r + 1},{t_h2d[r]:.4f},ms,completada\n")
            csv_crudo.write(f"GPU_Kernel,{filtro},{ent.ancho}x{ent.alto},{tam}x{tam},{cfg},{r + 1},{t_ker[r]:.4f},ms,completada\n")
            csv_crudo.write(f"GPU_D2H,{filtro},{ent.ancho}x{ent.alto},{tam}x{tam},{cfg},{r + 1},{t_d2h[r]:.4f},ms,completada\n")
            csv_crudo.write(f"GPU_Conjunto,{filtro},{ent.ancho}x{ent.alto},{tam}x{tam},{cfg},{r + 1},{t_tot[r]:.4f},ms,completada\n")

    # Calcular desviacion estandar del tiempo total y del kernel
    suma_tot = 0.0
    suma_ker = 0.0
    for i in range(NUM_REPS):
        suma_tot += t_tot[i]
        suma_ker += t_ker[i]
    media_tot = suma_tot / NUM_REPS
    media_ker = suma_ker / NUM_REPS
    var_tot = 0.0
    var_ker = 0.0
    for i in range(NUM_REPS):
        var_tot += (t_tot[i] - media_tot) * (t_tot[i] - media_tot)
        var_ker += (t_ker[i] - media_ker) * (t_ker[i] - media_ker)
    std_tot = math.sqrt(var_tot / (NUM_REPS - 1))
    std_ker = math.sqrt(var_ker / (NUM_REPS - 1))

    # Ordenar las 5 repeticiones para obtener mediana (indice 2) y rango intercuartilico (Q3 - Q1)
    t_h2d.sort()
    t_ker.sort()
    t_d2h.sort()
    t_tot.sort()

    med_h2d = t_h2d[NUM_REPS // 2]
    med_ker = t_ker[NUM_REPS // 2]
    med_d2h = t_d2h[NUM_REPS // 2]
    med_tot = t_tot[NUM_REPS // 2]
    iqr_tot = t_tot[3] - t_tot[1]
    iqr_ker = t_ker[3] - t_ker[1]

    # Pasar la salida de la GPU a la lista del llamador (fuera de la medicion)
    for i in range(n):
        h_salida_cruda[i] = float(h_sal[i])

    d_ent.free()
    d_sal.free()
    d_mask.free()

    return med_h2d, med_ker, med_d2h, med_tot, iqr_tot, std_tot, iqr_ker, std_ker
