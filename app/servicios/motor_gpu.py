"""
motor_gpu.py - Ejecucion del kernel de convolucion con PyCUDA dentro del servicio.

Modelo de ejecucion
-------------------
* Un unico hilo trabajador ("gpu-worker") es dueno del contexto CUDA. Lo crea con
  cuda.init() + Device(0).make_context() (no se usa pycuda.autoinit, que ataria el
  contexto al hilo principal del servidor). Todas las llamadas CUDA ocurren en ese hilo.
* El kernel (app/practica02.py -> KERNEL_SRC) se compila una sola vez al arrancar.
* Las solicitudes entran a una cola FIFO acotada (MAX_COLA). Si esta llena, se rechaza
  la solicitud de inmediato (ColaLlena -> HTTP 503). La GPU atiende una solicitud a la vez.
* Por solicitud: cuda.mem_alloc de entrada/salida/mascara, copia H2D, kernel, copia D2H
  y free() en un bloque finally (la memoria se libera aun si hay error).
* Transferencias desde memoria paginable (arreglos numpy normales, sin page-locked).
* Tiempos de H2D, kernel y D2H con cuda.Event (record + synchronize). La reserva y la
  liberacion se miden aparte con time.perf_counter y no forman parte de H2D/kernel/D2H.
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
import time
import warnings
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from app.practica02 import KERNEL_SRC, NOMBRE_KERNEL

log = logging.getLogger("app.gpu")


# --- Errores con diagnostico claro ---

class ErrorGPU(RuntimeError):
    codigo = "error_gpu"


class GPUNoDisponible(ErrorGPU):
    codigo = "gpu_no_disponible"


class MemoriaGPUInsuficiente(ErrorGPU):
    codigo = "memoria_gpu_insuficiente"


class ColaLlena(ErrorGPU):
    codigo = "cola_llena"


# --- Estructuras ---

@dataclass
class SalidaGPU:
    crudo: np.ndarray                 # float32 (alto, ancho), sin recortar a [0, 255]
    tiempos_ms: dict[str, float]      # cola, reserva, h2d, kernel, d2h, liberacion
    hilos_bloque: int
    bloques: int


@dataclass
class _Trabajo:
    imagen: np.ndarray
    mascara: np.ndarray
    tam: int
    encolado: float = field(default_factory=time.perf_counter)
    futuro: Future = field(default_factory=Future)


class MotorGPU:
    def __init__(self, hilos_bloque: int = 256, max_cola: int = 4, retardo_prueba_ms: int = 0) -> None:
        self.hilos_bloque = hilos_bloque
        self.retardo_prueba_ms = retardo_prueba_ms  # solo para probar la politica de cola
        self.max_cola = max_cola
        self._cola: queue.Queue[_Trabajo | None] = queue.Queue(maxsize=max_cola)
        self._hilo = threading.Thread(target=self._bucle, name="gpu-worker", daemon=True)
        self._listo = threading.Event()
        self._ocupado = False
        self.error_inicio: str | None = None
        self.info: dict = {}

    # ---------- Ciclo de vida ----------

    def iniciar(self, timeout_s: float = 300.0) -> None:
        self._hilo.start()
        if not self._listo.wait(timeout_s):
            self.error_inicio = f"La inicializacion de CUDA no termino en {timeout_s:.0f} s."
        if self.error_inicio:
            log.error("No se pudo inicializar la GPU: %s", self.error_inicio)

    def detener(self) -> None:
        if self._hilo.is_alive():
            self._cola.put(None)
            self._hilo.join(timeout=10)

    @property
    def disponible(self) -> bool:
        return self._listo.is_set() and self.error_inicio is None

    def estado_cola(self) -> dict:
        return {"en_espera": self._cola.qsize(), "max_en_espera": self.max_cola, "gpu_ocupada": self._ocupado}

    # ---------- API para el servicio ----------

    def enviar(self, imagen: np.ndarray, mascara: np.ndarray, tam: int) -> Future:
        """Encola una convolucion. Devuelve un Future con SalidaGPU."""
        if not self.disponible:
            raise GPUNoDisponible(self.error_inicio or "La GPU aun se esta inicializando.")
        trabajo = _Trabajo(imagen=imagen, mascara=mascara, tam=tam)
        try:
            self._cola.put_nowait(trabajo)
        except queue.Full:
            raise ColaLlena(
                f"El servicio está ocupado: hay {self.max_cola} solicitudes en espera. Intenta de nuevo."
            ) from None
        return trabajo.futuro

    # ---------- Hilo trabajador ----------

    def _bucle(self) -> None:
        try:
            ctx = self._inicializar()
        except Exception as exc:  # noqa: BLE001 - se convierte en diagnostico
            self.error_inicio = _diagnostico_inicio(exc)
            self._listo.set()
            return
        self._listo.set()
        try:
            while True:
                trabajo = self._cola.get()
                if trabajo is None:
                    break
                if not trabajo.futuro.set_running_or_notify_cancel():
                    continue
                espera_ms = (time.perf_counter() - trabajo.encolado) * 1000.0
                self._ocupado = True
                try:
                    salida = self._convolucionar(trabajo.imagen, trabajo.mascara, trabajo.tam)
                    if self.retardo_prueba_ms:
                        time.sleep(self.retardo_prueba_ms / 1000.0)
                    salida.tiempos_ms["cola"] = espera_ms
                    trabajo.futuro.set_result(salida)
                except Exception as exc:  # noqa: BLE001 - se entrega al llamador
                    trabajo.futuro.set_exception(_traducir_error_cuda(exc))
                finally:
                    self._ocupado = False
        finally:
            ctx.pop()
            ctx.detach()

    def _inicializar(self):
        t0 = time.perf_counter()
        import pycuda.driver as cuda

        cuda.init()
        if cuda.Device.count() == 0:
            raise GPUNoDisponible("CUDA no encontró ninguna GPU.")
        dispositivo = cuda.Device(0)
        ctx = dispositivo.make_context()
        t_ctx = (time.perf_counter() - t0) * 1000.0
        try:
            self._preparar_kernel(dispositivo, t_ctx)
        except Exception:
            ctx.pop()
            ctx.detach()
            raise
        return ctx

    def _preparar_kernel(self, dispositivo, t_ctx: float) -> None:
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        t1 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The CUDA compiler succeeded")
            modulo = SourceModule(KERNEL_SRC)
        self._modulo = modulo  # mantener referencia viva
        self._kernel = modulo.get_function(NOMBRE_KERNEL)
        t_comp = (time.perf_counter() - t1) * 1000.0

        self._cuda = cuda

        # Primera ejecucion (arranque en frio) con una imagen pequena; se registra aparte
        t2 = time.perf_counter()
        prueba = np.zeros((64, 64), dtype=np.uint8)
        self._convolucionar(prueba, np.full(9, 1.0 / 9.0, dtype=np.float32), 3)
        t_primera = (time.perf_counter() - t2) * 1000.0

        import pycuda

        major, minor = dispositivo.compute_capability()
        dv = cuda.get_driver_version()
        cv = cuda.get_version()
        self.info = {
            "gpu": {
                "nombre": dispositivo.name(),
                "compute_capability": f"{major}.{minor}",
                "memoria_total_mb": round(dispositivo.total_memory() / 2**20),
            },
            "versiones": {
                "cuda_driver_api": f"{dv // 1000}.{(dv % 1000) // 10}",
                "cuda_pycuda_compilado": ".".join(map(str, cv)),
                "pycuda": pycuda.VERSION_TEXT,
                "numpy": np.__version__,
                "python": platform.python_version(),
            },
            "arranque_ms": {
                "init_contexto": round(t_ctx, 3),
                "compilacion_kernel": round(t_comp, 3),
                "primera_ejecucion": round(t_primera, 3),
            },
            "hilos_bloque": self.hilos_bloque,
            "memoria_host": "paginable",
        }

    def _convolucionar(self, imagen: np.ndarray, mascara: np.ndarray, tam: int) -> SalidaGPU:
        cuda = self._cuda
        alto, ancho = imagen.shape
        n = ancho * alto
        h_ent = np.ascontiguousarray(imagen, dtype=np.uint8).reshape(-1)
        h_mask = np.ascontiguousarray(mascara, dtype=np.float32)
        h_sal = np.empty(n, dtype=np.float32)
        hilos = self.hilos_bloque
        bloques = (n + hilos - 1) // hilos

        d_ent = d_sal = d_mask = None
        t0 = time.perf_counter()
        try:
            d_ent = cuda.mem_alloc(h_ent.nbytes)
            d_sal = cuda.mem_alloc(h_sal.nbytes)
            d_mask = cuda.mem_alloc(h_mask.nbytes)
            reserva_ms = (time.perf_counter() - t0) * 1000.0

            ini, fin = cuda.Event(), cuda.Event()

            ini.record()
            cuda.memcpy_htod(d_ent, h_ent)
            cuda.memcpy_htod(d_mask, h_mask)
            fin.record()
            fin.synchronize()
            h2d_ms = fin.time_since(ini)

            ini.record()
            self._kernel(d_ent, d_sal, d_mask, np.int32(ancho), np.int32(alto), np.int32(tam),
                         block=(hilos, 1, 1), grid=(bloques, 1))
            fin.record()
            fin.synchronize()
            kernel_ms = fin.time_since(ini)

            ini.record()
            cuda.memcpy_dtoh(h_sal, d_sal)
            fin.record()
            fin.synchronize()
            d2h_ms = fin.time_since(ini)
        finally:
            t_lib = time.perf_counter()
            for d in (d_ent, d_sal, d_mask):
                if d is not None:
                    d.free()
            liberacion_ms = (time.perf_counter() - t_lib) * 1000.0

        return SalidaGPU(
            crudo=h_sal.reshape(alto, ancho),
            tiempos_ms={
                "reserva": reserva_ms,
                "h2d": h2d_ms,
                "kernel": kernel_ms,
                "d2h": d2h_ms,
                "liberacion": liberacion_ms,
            },
            hilos_bloque=hilos,
            bloques=bloques,
        )


def _diagnostico_inicio(exc: Exception) -> str:
    texto = f"{type(exc).__name__}: {exc}"
    if "libcuda" in texto:
        return (f"No se encontró el controlador NVIDIA (libcuda) ({texto}). Si usas Docker, inicia el "
                "contenedor con --gpus all y verifica el runtime de GPU (NVIDIA Container Toolkit).")
    if isinstance(exc, ImportError):
        return f"PyCUDA no está instalado o no carga ({texto})."
    if "no CUDA-capable device" in texto or "CUDA_ERROR_NO_DEVICE" in texto:
        return (f"No hay GPU NVIDIA visible ({texto}). Si usas Docker, inicia el contenedor "
                "con --gpus all y verifica el NVIDIA Container Toolkit.")
    if "nvcc" in texto.lower():
        return f"No se pudo compilar el kernel: nvcc no está disponible ({texto})."
    return f"No se pudo inicializar CUDA ({texto})."


def _traducir_error_cuda(exc: Exception) -> Exception:
    try:
        import pycuda.driver as cuda
    except ImportError:
        return exc
    if isinstance(exc, cuda.MemoryError) or "out of memory" in str(exc).lower():
        return MemoriaGPUInsuficiente(f"Memoria de GPU insuficiente para esta imagen ({exc}).")
    if isinstance(exc, cuda.Error):
        return ErrorGPU(f"Error de CUDA durante el procesamiento: {exc}")
    return exc
