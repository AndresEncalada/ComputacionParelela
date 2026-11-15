"""
main.py - Programa principal que ejecuta CPU (cpu_conv.py) y GPU (gpu_conv.py).

Uso:
    python3 src/main.py [imagen.jpg]
"""

from __future__ import annotations

import math
import os
import sys
from typing import TextIO

from cpu_conv import (
    LAPLACIANO_3,
    LIMITE_CPU_SEG,
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
)
from gpu_conv import (
    HILOS_BASE,
    TOLERANCIA,
    comparar_cpu_vs_gpu,
    medir_tiempo_gpu,
    mostrar_info_gpu,
    validar_manual_cpu_gpu,
)


def _calcular_estadisticas_cpu(t_cpu: list[float]) -> tuple[float, float]:
    media_c = sum(t_cpu) / NUM_REPS
    var_c = sum((t - media_c) ** 2 for t in t_cpu)
    std_cpu = math.sqrt(var_c / (NUM_REPS - 1))
    t_ord = sorted(t_cpu)
    iqr_cpu = t_ord[3] - t_ord[1]
    return iqr_cpu, std_cpu


def _guardar_imagenes_salida(
    archivo_base: str | None,
    crudo_cpu: list[float],
    crudo_gpu: list[float],
    sal_cpu: Imagen,
    sal_gpu: Imagen,
    guardar_cpu: bool,
) -> None:
    if archivo_base is None:
        return
    if guardar_cpu:
        convertir_crudo_a_pgm(crudo_cpu, sal_cpu)
        guardar_imagen(f"resultados/{archivo_base}_cpu.jpg", sal_cpu)
    convertir_crudo_a_pgm(crudo_gpu, sal_gpu)
    guardar_imagen(f"resultados/{archivo_base}_gpu.jpg", sal_gpu)


def _ejecutar_cpu(
    img: Imagen,
    crudo_cpu: list[float],
    mask: list[float],
    tam: int,
    filtro: str,
    csv_crudo: TextIO,
) -> tuple[float | None, float, float, str]:
    t_cpu = [0.0] * NUM_REPS
    med_cpu, cpu_estado, transcurrido_ms = medir_tiempo(
        img, crudo_cpu, mask, tam, t_cpu, limite_seg=LIMITE_CPU_SEG
    )

    if cpu_estado == "completada":
        for r in range(NUM_REPS):
            csv_crudo.write(
                f"CPU,{filtro},{img.ancho}x{img.alto},{tam}x{tam},"
                f"1_hilo_secuencial,{r + 1},{t_cpu[r]:.4f},ms,completada\n"
            )
        iqr_cpu, std_cpu = _calcular_estadisticas_cpu(t_cpu)
        return med_cpu, iqr_cpu, std_cpu, cpu_estado

    csv_crudo.write(
        f"CPU,{filtro},{img.ancho}x{img.alto},{tam}x{tam},"
        f"1_hilo_secuencial,1,{transcurrido_ms:.4f},ms,no_completada\n"
    )
    return None, 0.0, 0.0, cpu_estado


# Corre un experimento en CPU (opcional) y en GPU usando la misma imagen en memoria,
# compara los arreglos en float cuando CPU termina y guarda todo en los CSV.
def experimento_unificado(
    img: Imagen,
    mask: list[float],
    tam: int,
    filtro: str,
    hilos_bloque: int,
    archivo_base: str | None,
    csv_crudo: TextIO,
    csv_resumen: TextIO,
    ejecutar_cpu: bool = True,
) -> None:
    n = img.ancho * img.alto
    sal_cpu = crear_imagen(img.ancho, img.alto)
    sal_gpu = crear_imagen(img.ancho, img.alto)
    crudo_cpu = [0.0] * n
    crudo_gpu = [0.0] * n

    # 1. Medir en CPU (1 calentamiento + 5 repeticiones, con limite de 30 min)
    if ejecutar_cpu:
        med_cpu, iqr_cpu, std_cpu, cpu_estado = _ejecutar_cpu(
            img, crudo_cpu, mask, tam, filtro, csv_crudo
        )
    else:
        med_cpu, iqr_cpu, std_cpu, cpu_estado = None, 0.0, 0.0, "omitida_exp4"

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

    # 3. Comparar CPU vs GPU solo si la ejecucion CPU se completo
    cpu_ok = cpu_estado == "completada" and med_cpu is not None
    if cpu_ok:
        max_err, mae = comparar_cpu_vs_gpu(crudo_cpu, crudo_gpu, n)
        speedup_computo = (med_cpu / med_ker) if med_ker > 0.0 else 0.0
        speedup_total = (med_cpu / med_tot) if med_tot > 0.0 else 0.0
        estado_num = "OK" if max_err <= TOLERANCIA else "FALLA"
    else:
        max_err, mae = None, None
        speedup_computo, speedup_total = None, None
        estado_num = cpu_estado.upper()

    # 4. Pasar de float a [0, 255] y guardar las imagenes de salida
    _guardar_imagenes_salida(
        archivo_base, crudo_cpu, crudo_gpu, sal_cpu, sal_gpu, guardar_cpu=cpu_ok
    )

    cpu_str = f"{med_cpu:8.1f} ms" if med_cpu is not None else f"{'N/A':>11s}"
    sp_ker_str = f"{speedup_computo:6.1f}x" if speedup_computo is not None else f"{'N/A':>7s}"
    sp_tot_str = f"{speedup_total:5.1f}x" if speedup_total is not None else f"{'N/A':>6s}"
    err_str = f"{max_err:.1e}" if max_err is not None else f"{'N/A':>7s}"

    print(
        f"  {filtro:<10s} {tam:2d}x{tam:<2d} | {img.ancho:4d}x{img.alto:<4d} | "
        f"blq={hilos_bloque:3d} | CPU={cpu_str} | GPU_Ker={med_ker:6.3f} ms | "
        f"GPU_Tot={med_tot:6.2f} ms | SpKer={sp_ker_str} | "
        f"SpTot={sp_tot_str} | MaxErr={err_str} [{estado_num}]"
    )

    cpu_med_csv = f"{med_cpu:.4f}" if med_cpu is not None else ""
    cpu_iqr_csv = f"{iqr_cpu:.4f}" if cpu_ok else ""
    cpu_std_csv = f"{std_cpu:.4f}" if cpu_ok else ""
    sp_ker_csv = f"{speedup_computo:.2f}" if speedup_computo is not None else ""
    sp_tot_csv = f"{speedup_total:.2f}" if speedup_total is not None else ""
    max_err_csv = f"{max_err:.6e}" if max_err is not None else ""
    mae_csv = f"{mae:.6e}" if mae is not None else ""

    csv_resumen.write(
        f"{filtro},{img.ancho}x{img.alto},{tam}x{tam},{hilos_bloque},{cpu_estado},"
        f"{cpu_med_csv},{cpu_iqr_csv},{cpu_std_csv},"
        f"{med_h2d:.4f},{med_ker:.4f},{iqr_ker:.4f},{std_ker:.4f},"
        f"{med_d2h:.4f},{med_tot:.4f},{iqr_tot:.4f},{std_tot:.4f},"
        f"{sp_ker_csv},{sp_tot_csv},{max_err_csv},{mae_csv}\n"
    )

    liberar_imagen(sal_cpu)
    liberar_imagen(sal_gpu)


def _cargar_imagen_base(argv: list[str]) -> Imagen | None:
    if len(argv) >= 2:
        return leer_imagen(argv[1])
    if os.path.isfile("entrada.jpg"):
        return leer_imagen("entrada.jpg")
    return leer_imagen("entrada.pgm")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    os.makedirs("resultados", exist_ok=True)

    mostrar_info_gpu()
    print(f"Limite configurado para Python secuencial: {LIMITE_CPU_SEG / 60.0:.0f} minutos por configuracion.\n")

    base = _cargar_imagen_base(argv)
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
            "implementacion,filtro,resolucion,mascara,config_hilos_bloques,repeticion,tiempos,unidades,estado\n"
        )
        csv_resumen.write(
            "filtro,resolucion,mascara,hilos_bloque,cpu_estado,cpu_med_ms,cpu_iqr_ms,cpu_std_ms,"
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

        # Experimento 4: Probar 128, 256 y 512 hilos por bloque con media 9x9 sobre 4000x4000 (solo GPU)
        print("\n[Exp 4] Configuracion de hilos por bloque (128, 256, 512) sobre 4000x4000 (PyCUDA)")
        experimento_unificado(i4k, m9, 9, "media", 128, None, csv_crudo, csv_resumen, ejecutar_cpu=False)
        experimento_unificado(i4k, m9, 9, "media", 256, None, csv_crudo, csv_resumen, ejecutar_cpu=False)
        experimento_unificado(i4k, m9, 9, "media", 512, None, csv_crudo, csv_resumen, ejecutar_cpu=False)

        liberar_imagen(i4k)

    if base is not None:
        liberar_imagen(base)

    print("\nListo. Resultados guardados en resultados/resultados.csv y resultados/resumen.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
