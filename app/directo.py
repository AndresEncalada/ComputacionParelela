"""
directo.py - Ejecuta el mismo procesamiento del servicio sin HTTP.

Sirve para verificar que el servicio y la ejecucion directa producen lo mismo, y como
"PyCUDA directo" en los experimentos (anfitrion o dentro del contenedor).

Uso (desde la raiz del repositorio):
    python -m app.directo IMAGEN --filtro media --mascara 9 [--reps 5] [--salida out.png]

Imprime una linea JSON por repeticion con los tiempos (ms). La repeticion 0 es el
calentamiento y no debe usarse en las estadisticas estables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import registro
from app.config import CONFIG
from app.filtros import resolver_parametros
from app.servicios.motor_gpu import MotorGPU
from app.servicios.procesamiento import procesar


def main(argv: list[str] | None = None) -> int:
    registro.configurar(flujo=sys.stderr)  # stdout queda solo para las lineas JSON
    p = argparse.ArgumentParser(description="Convolucion PyCUDA directa (sin HTTP)")
    p.add_argument("imagen", type=Path)
    p.add_argument("--filtro", default="media")
    p.add_argument("--mascara", default="9")
    p.add_argument("--reps", type=int, default=5, help="repeticiones medidas (ademas del calentamiento)")
    p.add_argument("--salida", type=Path, help="guarda el resultado (.png o .npy)")
    args = p.parse_args(argv)

    filtro, k = resolver_parametros(args.filtro, args.mascara)
    contenido = args.imagen.read_bytes()
    formato = "npy" if args.salida and args.salida.suffix == ".npy" else "png"

    motor = MotorGPU(hilos_bloque=CONFIG.hilos_bloque, max_cola=1)
    motor.iniciar()
    registro.arranque(motor.info, {"max_en_espera": 1, "max_mb": CONFIG.max_bytes // 2**20,
                      "max_lado": CONFIG.max_lado, "max_megapixeles": CONFIG.max_pixeles // 1_000_000},
                      CONFIG.hilos_bloque, motor.error_inicio)
    if not motor.disponible:
        print(f"GPU no disponible: {motor.error_inicio}", file=sys.stderr)
        return 2
    print(json.dumps({"tipo": "arranque", **motor.info}, ensure_ascii=False))

    try:
        for rep in range(args.reps + 1):
            res = procesar(contenido, filtro, k, motor, CONFIG, salida=formato)
            print(json.dumps({
                "tipo": "calentamiento" if rep == 0 else "medicion",
                "repeticion": rep,
                "filtro": filtro,
                "mascara": f"{k}x{k}",
                "imagen": f"{res.ancho}x{res.alto}",
                "hilos_bloque": res.hilos_bloque,
                "bloques": res.bloques,
                "tiempos_ms": {c: round(v, 4) for c, v in res.tiempos_ms.items()},
            }, ensure_ascii=False))
        if args.salida:
            args.salida.write_bytes(res.contenido)
    finally:
        motor.detener()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
