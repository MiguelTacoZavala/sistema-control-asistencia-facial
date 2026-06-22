# Face Recognition Attendance System

Sistema automatizado de registro de asistencia mediante reconocimiento facial en tiempo real.

## Integrantes

- Taco
- Rhamses
- León

## Tecnologías

- Python
- OpenCV
- YOLO (Ultralytics)
- face_recognition
- ByteTrack

## Requisitos previos

- Python 3.10 o superior
- pip y venv
- **Windows**: CMake y Visual Studio Build Tools (necesarios para `dlib`, dependencia de `face-recognition`)
  - CMake: https://cmake.org/download/
  - VS Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd DESARROLLO

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar el entorno
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

> **Nota**: La instalación descargará PyTorch (~2 GB) como dependencia de `ultralytics` y compilará `dlib` (requiere CMake + VS Build Tools en Windows). La primera instalación puede demorar varios minutos.

## Pesos del modelo YOLO

Los pesos de detección de rostros (`models/yolov8n-face.pt`, ~6 MB) **vienen incluidos en el repositorio**. Al clonar el proyecto ya los tienes; no hace falta descargarlos por separado.

| Archivo | Descripción |
|---------|-------------|
| `models/yolov8n-face.pt` | YOLOv8n entrenado para detección de rostros |

**Fuente:** [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face/releases) (referencia: [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face), entrenado en WIDER FACE).

Si el archivo falta por algún motivo, puedes recuperarlo con:

```bash
# Windows (PowerShell)
curl.exe -L -o models/yolov8n-face.pt "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
```

## Verificación

Ejecuta el script de verificación para confirmar que todas las dependencias se instalaron correctamente:

```bash
python test/check_instalacion.py
python test/verificar_yolo.py
```

`check_instalacion.py` confirma que las dependencias están instaladas. `verificar_yolo.py` comprueba que los pesos en `models/yolov8n-face.pt` cargan y detectan rostros en una imagen de prueba.

## Uso básico

```bash
python src/main.py
```

## Estructura del proyecto

```
DESARROLLO/
├── src/              # Código fuente
├── dataset/          # Datos de entrada (known_faces/, test_videos/)
├── models/           # Pesos del modelo YOLO
├── experiments/      # Notebooks de experimentos
├── test/             # Scripts de prueba
├── notebook_entrega/ # Notebook final para entregar
├── screenshots/      # Capturas del sistema
├── requirements.txt  # Dependencias
└── README.md         # Este archivo
```

## Advertencias

- `ultralytics` instala PyTorch automáticamente (~2 GB de descarga)
- `face-recognition` depende de `dlib`, que en Windows requiere CMake + Visual Studio Build Tools
- No instalar `opencv-contrib-python` junto con `opencv-python` (entran en conflicto)

## Dataset

Fotos capturadas el **19 de junio de 2026** en `dataset/known_faces/`, una subcarpeta por persona:

| Persona | Fotos | Formato | Condiciones |
|---------|-------|---------|-------------|
| gerardo | 15 | .jpg | Frontal, luz lateral, contraluz, perfiles, lentes, capucha, expresiones |
| miguel  | 15 | .jpeg | Variadas (frontal, ángulos, iluminación) |
| kevin   | 21 | .jpg | Frontal, expresiones, accesorios (lentes, capucha, gorra), perfiles, luz lateral |

**Videos:** el video de prueba corto está en `dataset/test_videos/video_corto_prueba.mp4`. Los videos de los experimentos (más pesados) están en Google Drive.
