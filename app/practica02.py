"""
practica02.py - Unico punto de importacion del codigo reutilizado de la practica 02.

La app no copia el kernel ni las mascaras: los importa desde experimentos-pycuda/src para
que el servicio y los experimentos ejecuten exactamente el mismo codigo.
  - kernel_conv.KERNEL_SRC: kernel CUDA propio (1 pixel por hilo, distribucion en X).
  - cpu_conv: coeficientes de las mascaras (media k x k, nitidez 3x3, laplaciano 3x3).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "experimentos-pycuda" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cpu_conv import LAPLACIANO_3, NITIDEZ_3, mascara_media  # noqa: E402
from kernel_conv import KERNEL_SRC, NOMBRE_KERNEL  # noqa: E402

__all__ = ["KERNEL_SRC", "NOMBRE_KERNEL", "LAPLACIANO_3", "NITIDEZ_3", "mascara_media"]
