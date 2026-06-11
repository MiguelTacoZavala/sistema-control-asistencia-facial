# Estructura del Proyecto — Sistema de Control de Asistencia Facial

```
DESARROLLO/
├── README.md                 # Descripción general del proyecto
├── requirements.txt          # Dependencias de Python
├── .gitignore                # Archivos ignorados por git
├── ESTRUCTURA.md             # Este archivo
│
├── datos/                    # Datos persistentes del sistema
│   ├── empleados/            # Fotos de los empleados registrados
│   └── asistencias/          # Registros CSV o JSON de asistencia
│
├── modelos/                  # Pesos de modelos entrenados (YOLO, etc.)
│
└── src/                      # Código fuente del sistema
    ├── main.py               # Punto de entrada del programa
    ├── configuracion.py      # Constantes y rutas centralizadas
    │
    ├── deteccion/
    │   └── detector.py       # Detección de rostros con YOLO
    │
    ├── seguimiento/
    │   └── rastreador.py     # Seguimiento de personas con ByteTrack
    │
    ├── reconocimiento/
    │   └── identificador.py  # Reconocimiento facial (face_recognition) y
    │                         # estimación de edad/género (DeepFace)
    │
    ├── registro/
    │   └── registrador.py    # Escritura de asistencias en CSV
    │
    ├── interfaz/
    │   └── visualizador.py   # Visualización con OpenCV

```

## Descripción de directorios

### `datos/`
Datos generados y consumidos por el sistema.

| Subdirectorio | Propósito |
|---|---|
| `empleados/` | Contiene una imagen por cada empleado registrado (`{id_empleado}.jpg`). El sistema carga estas imágenes para generar los encodings faciales. |
| `asistencias/` | Almacena archivos CSV con el historial de asistencias (uno por día o uno único acumulativo). |

### `modelos/`
Almacena los pesos del modelo YOLO para detección de rostros. Los modelos de `face_recognition` y `DeepFace` se gestionan automáticamente en sus propios cachés internos.

| Archivo esperado | Propósito |
|---|---|
| `yolo-face.pt` | Pesos YOLO (ultralytics) fine-tuneado para detección de rostros. |

### `src/` — Módulos del sistema

| Módulo | Clase/Funciones principales | Responsabilidad |
|---|---|---|
| `main.py` | `main()` | Orquestar la captura de video, el pipeline de detección → seguimiento → reconocimiento → registro → visualización. |
| `configuracion.py` | Constantes (`RUTA_MODELOS`, `CONFIANZA_MINIMA`, `CAMARA_ID`, etc.) | Centralizar rutas y parámetros configurables. |
| `deteccion/detector.py` | `Detector` | Cargar el modelo YOLO y ejecutar inferencia sobre cada fotograma para obtener bounding boxes de rostros. |
| `seguimiento/rastreador.py` | `Rastreador` | Asignar un ID único a cada persona detectada usando ByteTrack, manteniendo consistencia entre fotogramas. |
| `reconocimiento/identificador.py` | `Identificador` | Comparar rostros contra la base de empleados (`face_recognition`) y estimar edad/género (`DeepFace`). |
| `registro/registrador.py` | `Registrador` | Escribir en CSV la información de cada persona reconocida (ID, nombre, fecha, hora). |
| `interfaz/visualizador.py` | `Visualizador` | Mostrar el video en una ventana OpenCV con rectángulos, etiquetas y métricas superpuestas. |

## Flujo del sistema

```
frame → Detector (YOLO) → Rastreador (ByteTrack) → Identificador (face_recognition + DeepFace)
                                                       ↓
                                              Registrador (CSV)
                                                       ↓
                                              Visualizador (OpenCV → pantalla)
```
