# Estructura del Proyecto — Sistema de Reconocimiento Facial en Tiempo Real

```
faces-yolo-opencv/
├── README.md                 # Descripción general del proyecto
├── requirements.txt          # Dependencias de Python
├── .gitignore                # Archivos ignorados por git
├── ESTRUCTURA.md             # Este archivo
│
├── dataset/                  # Datos de entrada del sistema
│   ├── known_faces/          # Fotos de personas registradas (subcarpeta por persona)
│   └── test_videos/          # Videos de prueba para experimentos
│
├── models/                   # Pesos de modelos preentrenados (YOLO)
│
├── experiments/              # Notebooks de los 6 experimentos con resultados
│   └── graphs/               # Gráficos exportados de los experimentos
│
├── tests/                    # Notebooks de prueba unitaria de cada módulo
│
├── notebook_entrega/         # Notebook final (.ipynb) para entregar al profesor
│
├── screenshots/              # Capturas de pantalla de experimentos y demos
│
└── src/                      # Código fuente del sistema
    ├── __init__.py
    ├── main.py               # Punto de entrada: loop de video en tiempo real (realtime.py en jira)
    ├── configuracion.py      # Constantes y rutas centralizadas
    ├── detector.py           # Detección de rostros con YOLOv8
    ├── embedding_db.py       # Generación y almacenamiento de embeddings
    ├── recognizer.py         # Reconocimiento facial por comparación de embeddings
    └── tracker.py            # Seguimiento de rostros entre frames
```

## Descripción de directorios

### `dataset/`
Datos consumidos por el sistema para registro y evaluación.

| Subdirectorio | Propósito |
|---|---|
| `known_faces/` | Contiene una subcarpeta por persona registrada (`gerardo/`, `kevin/`, `miguel/`), cada una con 15-20 fotos en distintas condiciones (ángulos, iluminación, accesorios). `embedding_db.py` lee estas fotos para generar los encodings faciales. |
| `test_videos/` | Videos cortos (20-40s) para los experimentos: caso ideal, múltiples personas, poca luz, persona desconocida. Los videos pesados se almacenan en Drive; aquí solo se guarda un video corto de prueba. |

### `models/`
Almacena los pesos preentrenados del modelo YOLO para detección de rostros. No se sube a Git por su tamaño; cada integrante los descarga desde Drive. Los modelos de `face_recognition` se gestionan automáticamente en su propio caché interno.

| Archivo esperado | Propósito |
|---|---|
| `yolov8n-face.pt` | Pesos YOLOv8 nano entrenados para detección de rostros. |

### `experiments/`
Notebooks de Jupyter con los 6 experimentos del proyecto. Cada notebook documenta: pregunta del experimento, procedimiento, resultados (tablas y gráficos) e interpretación.

| Notebook | Propósito |
|---|---|
| `exp1_baseline_detection.ipynb` | Medir FPS y confianza promedio del detector YOLO sin reconocimiento. |
| `exp2_confidence_thresholds.ipynb` | Comparar umbrales 0.3, 0.5 y 0.7 para justificar el umbral seleccionado. |
| `exp3_embeddings_registration.ipynb` | Documentar el proceso de registro: tiempos, fotos necesarias, estabilidad de embeddings. |
| `exp4_recognition_accuracy.ipynb` | Medir precisión en 4 condiciones: frontal, perfil, desconocido, múltiples personas. |
| `exp5_real_conditions.ipynb` | Simular caso de estudio de control de acceso con distintas condiciones. |
| `exp6_performance_tradeoff.ipynb` | Cuantificar el costo en FPS de cada componente del sistema. |
| `graphs/` | Gráficos `.png` exportados para insertar en el notebook entregable y los slides. |

### `tests/`
Notebooks de prueba para validar cada módulo de forma aislada antes de la integración.

| Notebook | Propósito |
|---|---|
| `test_detector_images.ipynb` | Validar que el detector dibuja bounding boxes correctos sobre imágenes fijas. |
| `test_recognizer_images.ipynb` | Validar que el reconocedor identifica correctamente rostros conocidos y desconocidos. |
| `test_threshold_calibration.ipynb` | Probar múltiples umbrales de distancia y calcular FAR/FRR/EER. |
| `test_embeddings_validation.ipynb` | Verificar calidad de la base: distancias intra-persona vs inter-persona. |

### `notebook_entrega/`
Contiene únicamente el notebook final que se entrega al profesor. Es autocontenido y ejecutable: incluye marco teórico, código documentado, resultados de experimentos, análisis y conclusiones.

| Archivo | Formato de nombre |
|---|---|
| `solTC1_...ipynb` | `solTC1_León-Chacón_Gerardo-Bohorquez-Huaringa_Kevin-Taco-Zavala_Miguel.ipynb` |

### `screenshots/`
Capturas de pantalla del sistema funcionando en distintas condiciones. Se usan como evidencia en los experimentos y en el notebook entregable.

## `src/` — Módulos del sistema

| Módulo | Clase / Funciones | Responsabilidad |
|---|---|---|
| `main.py` | `main()` | Orquestar el loop de video: captura → detección → reconocimiento → visualización. Acepta webcam o archivo de video por argumento. Controles de teclado (`q`, `s`, `r`, `espacio`). |
| `configuracion.py` | Constantes (`YOLO_WEIGHTS`, `DETECTION_CONFIDENCE`, `RECOGNITION_THRESHOLD`, etc.) | Centralizar rutas, umbrales y parámetros configurables en un solo lugar. |
| `detector.py` | `FaceDetector` | Cargar el modelo YOLOv8 y ejecutar inferencia sobre cada frame para obtener bounding boxes de rostros con su confianza. |
| `embedding_db.py` | `generate_embeddings()` | Recorrer `dataset/known_faces/`, generar embeddings con `face_recognition`, y guardar el diccionario resultante en `embeddings.pkl`. Reportar fallidos en `fallidos.txt`. |
| `recognizer.py` | `FaceRecognizer` | Cargar `embeddings.pkl`, comparar un rostro recortado contra la base por distancia euclidiana, y devolver el nombre más cercano o `"Desconocido"` si supera el umbral. |
| `tracker.py` | `Tracker` | Asignar un ID único a cada rostro detectado y mantener consistencia entre frames para evitar reconocer el mismo rostro repetidamente. |

### `detector.py` — FaceDetector en detalle

La clase `FaceDetector` encapsula toda la lógica de detección de rostros usando YOLOv8 de Ultralytics.

**Inicialización:**
```python
detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)
```
El constructor carga el modelo inmediatamente. El `conf_threshold` define la confianza mínima para considerar una detección válida.

**Método `detect(frame, conf=None)`:**
1. Recibe un frame BGR de OpenCV (numpy array).
2. Llama a `self.model(frame, conf=conf, verbose=False)` que ejecuta la inferencia.
3. El resultado de Ultralytics contiene `boxes.xyxy` (tensores con coordenadas `[x1, y1, x2, y2]` en píxeles) y `boxes.conf` (confianza de cada detección).
4. Esos tensores (inicialmente en GPU) se pasan a CPU con `.cpu()`, se convierten a numpy con `.numpy()`, y se transforman a tuplas de Python.
5. Devuelve una lista de tuplas: `[(x1, y1, x2, y2, confidence), ...]`.

**Método `draw_detections(frame, detections)`:**
- Dibuja un rectángulo verde por cada detección usando `cv2.rectangle`.
- Escribe el valor de confianza arriba del rectángulo con `cv2.putText`.
- Modifica y devuelve el frame (in-place).

**Ejemplo de uso:**
```python
from src.detector import FaceDetector
import cv2

detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)
frame = cv2.imread("foto_con_rostros.jpg")
detecciones = detector.detect(frame)
frame_con_cajas = detector.draw_detections(frame, detecciones)
cv2.imshow("Detecciones", frame_con_cajas)
cv2.waitKey(0)
```

## Archivos de salida generados por el sistema

| Archivo | Generado por | Contenido |
|---|---|---|
| `embeddings.pkl` | `embedding_db.py` | Diccionario `{nombre: [embedding1, embedding2, ...]}` con los vectores faciales de cada persona registrada. |
| `fallidos.txt` | `embedding_db.py` | Lista de fotos donde no se detectó rostro durante la generación de embeddings. |
| `registro_acceso.csv` | `main.py --log` | Log de accesos: timestamp, nombre reconocido, confianza, número de frame. |

## Flujo del sistema

```
frame → FaceDetector (YOLO) → Tracker → FaceRecognizer (embeddings)
                                              ↓
                                     registro_acceso.csv
                                              ↓
                                     OpenCV → pantalla (con boxes, nombres, FPS)
```
