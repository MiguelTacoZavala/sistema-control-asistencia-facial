# Experimento 1: Detección Base — YOLOv8

## Objetivo

Establecer el punto de referencia del rendimiento del detector de rostros YOLO
**sin** reconocimiento facial ni tracking. Esto permite saber qué tan rápido es
el detector por sí solo y cuánto margen hay para agregarle los otros módulos.

**Preguntas que responde:**
- ¿El detector corre en tiempo real (>15 FPS)?
- ¿Qué tan estable es el FPS o hay caídas repentinas?
- ¿Cuántos rostros detecta en promedio por frame?
- ¿Qué tan seguro (confianza) está de sus detecciones?
- ¿Varía el rendimiento entre distintos videos?

---

## Procedimiento

1. Cargar `FaceDetector` con el modelo `yolov8n-face.pt` y umbral 0.5.
2. Para cada video en `dataset/test_videos/`:
   - Abrir el video con OpenCV (`cv2.VideoCapture`).
   - Leer frame por frame.
   - Por cada frame: medir el tiempo de `detector.detect()`, guardar cuántos
     rostros encontró y la confianza de cada uno.
3. Al terminar, calcular por video:
   - FPS promedio, mínimo y máximo.
   - Confianza promedio de las detecciones.
   - Cantidad promedio de rostros por frame.
   - Tiempo de inferencia promedio por frame.
4. Mostrar resultados en una tabla con pandas y un gráfico FPS vs. frame.
5. Exportar las métricas a `results/resultados_exp1.csv`.

---

## Glosario

### FPS (Frames Per Second)

Cantidad de frames (imágenes) que el sistema puede procesar en un segundo.
Mientras más alto, más fluido se ve el video.

- **30 FPS**: fluidez estándar de video (TV, YouTube).
- **15 FPS**: mínimo aceptable para considerarlo "tiempo real".
- **<10 FPS**: se nota lento, pero aún funcional.

En este experimento medimos el FPS **solo de la detección**, sin contar
reconocimiento ni dibujado en pantalla.

### Tiempo de inferencia

Tiempo que tarda YOLO en procesar un frame y devolver las detecciones.
Se mide en milisegundos (ms). Es lo opuesto al FPS:
- Si la inferencia tarda **33 ms** → ~30 FPS.
- Si tarda **66 ms** → ~15 FPS.
- Si tarda **100 ms** → ~10 FPS.

### Confianza

Valor entre 0.0 y 1.0 que indica qué tan seguro está el modelo de que
detectó un rostro. Un valor de **0.95** significa "95 % seguro". El umbral
de confianza (0.5 por defecto) descarta detecciones por debajo de ese valor.

### Bounding box

Rectángulo que dibuja el detector alrededor de cada rostro. Se define con
4 coordenadas en píxeles: `(x1, y1)` esquina superior izquierda,
`(x2, y2)` esquina inferior derecha.

### Umbral de confianza

Filtro mínimo para aceptar una detección. Si el modelo detecta algo con
confianza 0.3 y el umbral es 0.5, se descarta. Subir el umbral reduce
falsos positivos pero puede perder rostros reales mal iluminados.

---

## Cómo interpretar los resultados

### FPS

| Si el FPS promedio es... | Significa... |
|---|---|
| **>30** | El detector es muy rápido. Hay margen de sobra para agregar reconocimiento y tracking. |
| **15–30** | Corre en tiempo real. Al agregar los otros módulos podría bajar de 15, habría que optimizar. |
| **10–15** | Aceptable para tiempo real con tolerancia. Conviene optimizar antes de agregar más carga. |
| **<10** | El detector solo ya es lento. Revisar si el modelo es muy pesado o si hay problema de hardware. |

### Estabilidad del FPS

El gráfico de FPS a lo largo del tiempo muestra si hay caídas repentinas
(picos hacia abajo). Un FPS estable es preferible aunque sea más bajo,
porque la experiencia de usuario es predecible.

### Confianza promedio

Si la confianza promedio es alta (>0.8), el modelo está detectando rostros
con seguridad. Si es baja (~0.6), muchas detecciones apenas pasan el umbral
y podrían ser falsos positivos.

### Relación con el resto del sistema

Este experimento mide solo **YOLO puro**. Cuando se agreguen:
- **Reconocimiento** (`face_recognition`): sumará ~100–500 ms por rostro.
- **Tracking** (ByteTrack): sumará ~1–5 ms por frame.

Por eso es importante que el detector base tenga FPS holgados, para que
al final el sistema completo se mantenga sobre 10–15 FPS.

---

## Ejecución en Google Colab

Si no puedes correr el notebook localmente, usa Google Colab:

1. En la raíz de tu Google Drive, crea una carpeta llamada `IA_Experimentos`.
2. Sube el archivo `exp1_deteccion_base.ipynb` a esa carpeta.
3. Haz clic derecho sobre el archivo → "Abrir con" → "Google Colab".
4. Ejecuta las celdas en orden. Las primeras celdas montarán tu Drive,
   clonarán el repositorio (solo la primera vez) e instalarán las dependencias.
5. El resto del notebook se ejecuta normalmente. Los resultados (tabla, gráfico,
   CSV) se guardan dentro del repositorio clonado en tu Drive.

**Nota:** La primera ejecución tardará un poco más porque clona el repo
e instala PyTorch (~2 GB). Las siguientes veces solo ejecuta sin instalar.

---

## Requisitos

- `models/yolov8n-face.pt` (los pesos del modelo).
- Al menos 2 videos `.mp4` en `dataset/test_videos/`.
- Dependencias instaladas (`ultralytics`, `opencv-python`, `pandas`, `matplotlib`).
