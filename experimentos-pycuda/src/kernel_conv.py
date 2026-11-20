"""
kernel_conv.py - Codigo fuente del kernel CUDA de convolucion (practica 02).

Se mantiene en un modulo sin efectos secundarios (no crea contexto CUDA ni compila)
para que lo reutilicen tanto gpu_conv.py (experimentos) como la aplicacion web.
El codigo del kernel es identico al de la practica 02.
"""

NOMBRE_KERNEL = "kernelConvolucionGPU"

# --- Kernel en CUDA ---

# Configuracion 1D (bloques e hilos solo en X).
# Cada hilo calcula 1 pixel de salida leyendo desde memoria global.
KERNEL_SRC = r"""
__global__ void kernelConvolucionGPU(const unsigned char *entrada,
                                     float *salida_cruda,
                                     const float *mascara,
                                     int ancho, int alto, int tam) {
    // 1. Calcular el indice global 1D del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Control de bordes: si hay mas hilos que pixeles en la imagen, no hacen nada
    if (idx >= ancho * alto) return;

    // 2. Convertir el indice 1D a coordenadas 2D (fila y columna)
    int fila  = idx / ancho;
    int col   = idx % ancho;

    int mitad = tam / 2;
    float suma = 0.0f;

    // 3. Recorrer la mascara centrada en (fila, col)
    for (int ky = 0; ky < tam; ky++) {
        for (int kx = 0; kx < tam; kx++) {
            int y = fila + ky - mitad;
            int x = col  + kx - mitad;

            // Zero-padding: fuera de los limites de la imagen se asume 0
            unsigned char pixel = 0;
            if (x >= 0 && x < ancho && y >= 0 && y < alto)
                pixel = entrada[y * ancho + x];

            suma += (float)pixel * mascara[ky * tam + kx];
        }
    }

    // Guardar el valor en float antes de truncar a [0, 255]
    salida_cruda[idx] = suma;
}
"""
