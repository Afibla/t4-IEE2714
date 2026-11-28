# Tarea 4 — Representación Multiescala y Compresión HDR (IEE2714)

## Estructura del Proyecto

* **`p1.py`**: Implementación de la Pirámide Laplaciana (decimada) y Transformada Starlet (*à trous*, no decimada).
* **`p2.py`**: Implementación de compresión HDR (Filtro Homomórfico y Método de Durand) y estrategias de recuperación de color (Razón $\beta$ y HSV).
* **`P1_PiramideLap.ipynb`**: Cuaderno experimental de la Pregunta 1 (invertibilidad, capas, umbral duro y ganancias por escala).
* **`P2_HDR.ipynb`**: Cuaderno experimental de la Pregunta 2 (compresión HDR, canal independiente, estrategias de color y métricas $C^*, \Delta H$). (INCOMPLETO)

## Requisitos y Ejecución

* **`archivos_t4.zip**:para correr los notebooks se necesita del zip entregado para la T4 en el mismo path que los notebooks e implementaciones, era muy pesado para subirlo

```bash
pip install numpy opencv-python scipy tifffile matplotlib