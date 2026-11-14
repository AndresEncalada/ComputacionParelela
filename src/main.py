"""
main.py - Programa principal que ejecuta CPU (cpu_conv.py) y GPU (gpu_conv.py)
Traduccion directa de src/main.cu a Python puro.

Uso:
    python3 src/main.py [imagen.jpg]
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from typing import Any, TextIO

from cpu_conv import (
    LAPLACIANO_3,
    MEDIA_3,
    NITIDEZ_3,
    NUM_REPS,
    Imagen,
    convertir_crudo_a_pgm,
    crear_imagen,
    guardar_imagen,
    leer_imagen,
    liberar_imagen,
    mascara_media,
    medir_tiempo,
    preparar_imagen,
    validar_manual,
)

try:
    _gpu_mod: Any = importlib.import_module("gpu_conv")
except ImportError:
    _gpu_mod = None

HILOS_BASE: int = getattr(_gpu_mod, "HILOS_BASE", 256)
TOLERANCIA: float = getattr(_gpu_mod, "TOLERANCIA", 1e-3)


def _comparar_cpu_vs_gpu_local(
    cpu_crudo: list[float],
    gpu_crudo: list[float],
    n: int,
) -> tuple[float, float]:
    max_e = 0.0
    suma_e = 0.0
    for i in range(n):
        diff = abs(float(cpu_crudo[i]) - float(gpu_crudo[i]))
        if diff > max_e:
            max_e = diff
        suma_e += diff
    mae = (suma_e / n) if n > 0 else 0.0
    return max_e, mae


def comparar_cpu_vs_gpu(
    cpu_crudo: list[float],
    gpu_crudo: list[float],
    n: int,
) -> tuple[float, float]:
    fn = getattr(_gpu_mod, "comparar_cpu_vs_gpu", None) or getattr(
        _gpu_mod, "compararCPUvsGPU", None
    )
    if fn is None:
        return _comparar_cpu_vs_gpu_local(cpu_crudo, gpu_crudo, n)

    res = fn(cpu_crudo, gpu_crudo, n)
    if isinstance(res, tuple) and len(res) == 2:
        return float(res[0]), float(res[1])
    return _comparar_cpu_vs_gpu_local(cpu_crudo, gpu_crudo, n)


def validar_manual_cpu_gpu() -> None:
    fn = getattr(_gpu_mod, "validar_manual_cpu_gpu", None) or getattr(
        _gpu_mod, "validarManualCPU_GPU", None
    )
    if fn is not None:
        fn()
        return

    validar_manual()
    print(
        "  [Aviso] gpu_conv.py no encontrado aun; "
        "se ejecuto solo la validacion manual en CPU.\n"
    )


def medir_tiempo_gpu(
    ent: Imagen,
    h_salida_cruda: list[float],
    mask: list[float],
    tam: int,
    hilos_bloque: int,
    filtro: str,
    csv_crudo: TextIO | None,
) -> tuple[float, float, float, float, float, float, float, float]:
    fn = getattr(_gpu_mod, "medir_tiempo_gpu", None) or getattr(
        _gpu_mod, "medirTiempoGPU", None
    )
    if fn is None:
        for i in range(len(h_salida_cruda)):
            h_salida_cruda[i] = 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    res = fn(ent, h_salida_cruda, mask, tam, hilos_bloque, filtro, csv_crudo)
    return tuple(float(v) for v in res)  # type: ignore[return-value]


def mostrar_info_gpu() -> None:
    """Imprime las propiedades de la GPU detectada tal como en main.cu."""
    fn = getattr(_gpu_mod, "mostrar_info_gpu", None) or getattr(
        _gpu_mod, "mostrarInfoGPU", None
    )
    if fn is not None:
        fn()
        return

    print("GPU detectada: (pendiente de gpu_conv.py)")
    print("Nota: cudaMalloc y cudaFree estan excluidos de la medicion de tiempos.\n")


def _guardar_imagenes_salida(
    archivo_base: str | None,
    crudo_cpu: list[float],
    crudo_gpu: list[float],
    sal_cpu: Imagen,
    sal_gpu: Imagen,
) -> None:
    if archivo_base is None:
        return
    convertir_crudo_a_pgm(crudo_cpu, sal_cpu)
    convertir_crudo_a_pgm(crudo_gpu, sal_gpu)
    guardar_imagen(f"resultados/{archivo_base}_cpu.jpg", sal_cpu)
    guardar_imagen(f"resultados/{archivo_base}_gpu.jpg", sal_gpu)


# Corre un experimento en CPU y en GPU usando la misma imagen en memoria,
# compara los arreglos en float para sacar el error y guarda todo en el CSV.
def experimento_unificado(
    img: Imagen,
    mask: list[float],
    tam: int,
    filtro: str,
    hilos_bloque: int,
    archivo_base: str | None,
    csv_crudo: TextIO,
    csv_resumen: TextIO,
) -> None:
    n = img.ancho * img.alto
    sal_cpu = crear_imagen(img.ancho, img.alto)
    sal_gpu = crear_imagen(img.ancho, img.alto)
    crudo_cpu = [0.0] * n
    crudo_gpu = [0.0] * n

    # 1. Medir en CPU (1 calentamiento + 5 repeticiones)
    t_cpu = [0.0] * NUM_REPS
    vmin_c = [0.0]
    vmax_c = [0.0]
    med_cpu = medir_tiempo(img, crudo_cpu, mask, tam, t_cpu, vmin_c, vmax_c)

    for r in range(NUM_REPS):
        csv_crudo.write(
            f"CPU,{filtro},{img.ancho}x{img.alto},{tam}x{tam},"
            f"1_hilo_secuencial,{r + 1},{t_cpu[r]:.4f},ms\n"
        )

    # Calcular desviacion estandar e IQR de CPU manualmente como en main.cu
    suma_c = 0.0
    for i in range(NUM_REPS):
        suma_c += t_cpu[i]
    media_c = suma_c / NUM_REPS

    var_c = 0.0
    for i in range(NUM_REPS):
        var_c += (t_cpu[i] - media_c) * (t_cpu[i] - media_c)
    std_cpu = math.sqrt(var_c / (NUM_REPS - 1))

    t_cpu.sort()
    iqr_cpu = t_cpu[3] - t_cpu[1]

    # 2. Medir en GPU por separado: H2D, Kernel, D2H y tiempo conjunto
    (
        med_h2d,
        med_ker,
        med_d2h,
        med_tot,
        iqr_tot,
        std_tot,
        iqr_ker,
        std_ker,
    ) = medir_tiempo_gpu(
        img, crudo_gpu, mask, tam, hilos_bloque, filtro, csv_crudo
    )

    # 3. Comparar el resultado en float de CPU contra el de GPU (antes del clamp a 0..255)
    max_err, mae = comparar_cpu_vs_gpu(crudo_cpu, crudo_gpu, n)

    # 4. Pasar de float a [0, 255] y guardar las imagenes de salida
    _guardar_imagenes_salida(archivo_base, crudo_cpu, crudo_gpu, sal_cpu, sal_gpu)

    # Calcular aceleracion solo del kernel y aceleracion contando transferencias
    speedup_computo = (med_cpu / med_ker) if med_ker > 0.0 else 0.0
    speedup_total = (med_cpu / med_tot) if med_tot > 0.0 else 0.0

    estado = "OK" if max_err <= TOLERANCIA else "FALLA"
    print(
        f"  {filtro:<10s} {tam:2d}x{tam:<2d} | {img.ancho:4d}x{img.alto:<4d} | "
        f"blq={hilos_bloque:3d} | CPU={med_cpu:6.1f} ms | GPU_Ker={med_ker:6.3f} ms | "
        f"GPU_Tot={med_tot:6.2f} ms | SpKer={speedup_computo:6.1f}x | "
        f"SpTot={speedup_total:5.1f}x | MaxErr={max_err:.1e} [{estado}]"
    )

    csv_resumen.write(
        f"{filtro},{img.ancho}x{img.alto},{tam}x{tam},{hilos_bloque},"
        f"{med_cpu:.4f},{iqr_cpu:.4f},{std_cpu:.4f},"
        f"{med_h2d:.4f},{med_ker:.4f},{iqr_ker:.4f},{std_ker:.4f},"
        f"{med_d2h:.4f},{med_tot:.4f},{iqr_tot:.4f},{std_tot:.4f},"
        f"{speedup_computo:.2f},{speedup_total:.2f},{max_err:.6e},{mae:.6e}\n"
    )

    liberar_imagen(sal_cpu)
    liberar_imagen(sal_gpu)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    os.makedirs("resultados", exist_ok=True)

    mostrar_info_gpu()

    ruta_entrada = argv[1] if len(argv) >= 2 else "entrada.pgm"
    base = leer_imagen(ruta_entrada)
    if base is not None:
        print(f"Imagen base leida: {base.ancho}x{base.alto} px\n")
    else:
        print("Sin imagen base; usando gradiente sintetico.\n")

    # Crear las mascaras de media para 3x3, 9x9 y 21x21
    m3 = [0.0] * 9
    m9 = [0.0] * 81
    m21 = [0.0] * 441
    mascara_media(m3, 3)
    mascara_media(m9, 9)
    mascara_media(m21, 21)

    # Prueba manual con matriz pequena 5x5 en CPU y GPU
    validar_manual_cpu_gpu()

    with (
        open("resultados/resultados.csv", "w", encoding="utf-8") as csv_crudo,
        open("resultados/resumen.csv", "w", encoding="utf-8") as csv_resumen,
    ):
        csv_crudo.write(
            "implementacion,filtro,resolucion,mascara,config_hilos_bloques,repeticion,tiempos,unidades\n"
        )
        csv_resumen.write(
            "filtro,resolucion,mascara,hilos_bloque,cpu_med_ms,cpu_iqr_ms,cpu_std_ms,"
            "gpu_h2d_ms,gpu_kernel_ms,gpu_ker_iqr_ms,gpu_ker_std_ms,gpu_d2h_ms,"
            "gpu_total_ms,gpu_tot_iqr_ms,gpu_tot_std_ms,aceleracion_computo,"
            "aceleracion_transferencias,error_abs_max,mae\n"
        )

        # Prueba con tamano que no es multiplo de 256 hilos por bloque (2050x2050)
        print("[Validacion Bordes] Dimension no multiplo de bloque (2050x2050)")
        i_irreg = preparar_imagen(base, 2050, 2050)
        experimento_unificado(i_irreg, m3, 3, "media", HILOS_BASE, "borde_2050", csv_crudo, csv_resumen)
        liberar_imagen(i_irreg)

        # Se genera la imagen 4000x4000 una sola vez para usarla en los experimentos 1, 3 y 4
        i4k = preparar_imagen(base, 4000, 4000)

        # Experimento 1: Comparar los 3 filtros 3x3 sobre 4000x4000
        print("\n[Exp 1] Filtros 3x3 sobre 4000x4000 (256 hilos/bloque)")
        experimento_unificado(i4k, m3,           3, "media",      HILOS_BASE, "media_4k",      csv_crudo, csv_resumen)
        experimento_unificado(i4k, NITIDEZ_3,    3, "nitidez",    HILOS_BASE, "nitidez_4k",    csv_crudo, csv_resumen)
        experimento_unificado(i4k, LAPLACIANO_3, 3, "laplaciano", HILOS_BASE, "laplaciano_4k", csv_crudo, csv_resumen)

        # Experimento 2: Cambiar la resolucion (2000, 4000 y 6000) dejando fija la media 9x9
        print("\n[Exp 2] Resolucion con media 9x9 (256 hilos/bloque)")
        i2k = preparar_imagen(base, 2000, 2000)
        experimento_unificado(i2k, m9, 9, "media", HILOS_BASE, "media9_2k", csv_crudo, csv_resumen)
        liberar_imagen(i2k)

        experimento_unificado(i4k, m9, 9, "media", HILOS_BASE, "media9_4k", csv_crudo, csv_resumen)

        i6k = preparar_imagen(base, 6000, 6000)
        experimento_unificado(i6k, m9, 9, "media", HILOS_BASE, "media9_6k", csv_crudo, csv_resumen)
        liberar_imagen(i6k)

        # Experimento 3: Cambiar el tamano de la mascara (3x3, 9x9, 21x21) sobre 4000x4000
        print("\n[Exp 3] Tamano de mascara sobre 4000x4000 (256 hilos/bloque)")
        experimento_unificado(i4k, m3,  3,  "media", HILOS_BASE, "media3_4k",   csv_crudo, csv_resumen)
        experimento_unificado(i4k, m9,  9,  "media", HILOS_BASE, "media9_4k_b", csv_crudo, csv_resumen)
        experimento_unificado(i4k, m21, 21, "media", HILOS_BASE, "media21_4k",  csv_crudo, csv_resumen)

        # Experimento 4: Probar 128, 256 y 512 hilos por bloque con media 9x9 sobre 4000x4000
        print("\n[Exp 4] Configuracion de hilos por bloque (128, 256, 512) sobre 4000x4000")
        experimento_unificado(i4k, m9, 9, "media", 128, None, csv_crudo, csv_resumen)
        experimento_unificado(i4k, m9, 9, "media", 256, None, csv_crudo, csv_resumen)
        experimento_unificado(i4k, m9, 9, "media", 512, None, csv_crudo, csv_resumen)

        liberar_imagen(i4k)

    if base is not None:
        liberar_imagen(base)

    print("\nListo. Resultados guardados en resultados/resultados.csv y resultados/resumen.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
