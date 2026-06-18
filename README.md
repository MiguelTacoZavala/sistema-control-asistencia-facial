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

## Verificación

Ejecuta el script de verificación para confirmar que todas las dependencias se instalaron correctamente:

```bash
python test/check_instalacion.py
```

Debes ver un ✅ junto a cada dependencia.

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
