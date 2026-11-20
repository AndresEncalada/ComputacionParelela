# Pruebas de validación

Requieren una GPU NVIDIA. Desde la raíz del repositorio:

```bash
pip install -r app/requirements.txt -r pruebas/requirements.txt
python -m pytest pruebas -v
```

Cada prueba levanta servidores uvicorn reales en puertos libres. Para ejecutar las mismas pruebas contra el **contenedor** (u otro servidor), se indica su URL. La prueba de cola llena siempre usa un servidor local propio.

```bash
URL_SERVICIO=http://localhost:8000 python -m pytest pruebas -v
```

| Archivo | Qué verifica | Evidencia en `resultados/local/` o `resultados/contenedor/` |
|---------|--------------|----------------------------|
| `test_1_matriz_y_bordes.py` | Matriz 5×5 de la práctica 02 (centro 50.0000, esquina 8.8889 → 8) y entrada de 2050×2050 (4 202 500 píxeles, que no es múltiplo de 256): todos los píxeles se comparan con una referencia independiente | `bordes_2050.csv` |
| `test_2_equivalencia.py` | Servicio vs. PyCUDA directo validado (`experimentos-pycuda/src/gpu_conv.py`) para media 3/9/21, nitidez y laplaciano en 2000×2000. Error absoluto máximo y MAE **antes** del recorte (`salida=npy`), y píxeles distintos en el PNG final | `equivalencia.csv`, `visual/` |
| `test_3_entradas_invalidas.py` | Archivo que no es imagen, PDF renombrado, archivo sobre el límite, filtro desconocido, máscara par, intermedia (5), fuera de rango o no numérica (solo se aceptan 3, 9 y 21), salida no válida, campo faltante, y recuperación posterior | `entradas_invalidas.csv` |
| `test_4_aislamiento.py` | 2 clientes simultáneos (5 rondas) con imágenes y filtros distintos reciben su propio resultado y un ID único. Con `MAX_COLA=1`, 4 solicitudes simultáneas → 2 atendidas y 2 rechazadas con 503 `cola_llena`, y el servicio se recupera | `aislamiento.csv`, `cola_llena.csv` |

**Tolerancia:** 1e-3, la misma que en la práctica 02. Las sumas en float32 de máscaras de hasta 21×21 (441 términos) acumulan un error de redondeo del orden de 1e-4. El servicio y la ejecución directa usan el mismo kernel en la misma GPU, así que la diferencia esperada entre ambos es 0. Frente a la referencia en float64, el error queda por debajo de 2e-4.
