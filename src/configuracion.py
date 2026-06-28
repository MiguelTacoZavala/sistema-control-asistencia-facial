"""Configuracion centralizada del Sistema de Control de Asistencia Facial.

Este archivo es la **fuente unica de verdad** para rutas, umbrales y
parametros del proyecto. La idea es que cualquier valor que se quiera
cambiar este aqui y no disperso en el codigo.

Muchos de estos valores NO son arbitrarios: se definen a partir de los
experimentos y pruebas del proyecto. Junto a cada parametro se indica
que experimento/tarea lo justifica.

Nota: por ahora los modulos (detector.py, recognizer.py, main.py) siguen
usando sus propios valores por defecto. Este archivo queda como referencia
coherente; mas adelante se puede hacer que esos modulos importen de aqui.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
# RAIZ apunta a la carpeta del proyecto (un nivel arriba de src/).
RAIZ = Path(__file__).resolve().parent.parent

RUTA_MODELOS = RAIZ / "models"
RUTA_DATASET = RAIZ / "dataset"
RUTA_EXPERIMENTOS = RAIZ / "experiments"

# ---------------------------------------------------------------------------
# Modelos y datos de entrada
# ---------------------------------------------------------------------------
# Pesos de YOLOv8 para deteccion de rostros.
YOLO_WEIGHTS = RUTA_MODELOS / "yolov8n-face.pt"

# Carpeta con una subcarpeta por persona (sus fotos de registro).
KNOWN_FACES_DIR = RUTA_DATASET / "known_faces"

# Carpeta con los videos de prueba de los experimentos.
TEST_VIDEOS_DIR = RUTA_DATASET / "test_videos"

# ---------------------------------------------------------------------------
# Archivos de salida que genera el sistema
# ---------------------------------------------------------------------------
# Base de embeddings {nombre: [vector_128d, ...]} generada por embedding_db.py.
EMBEDDINGS_PATH = RAIZ / "embeddings.pkl"

# Lista de fotos donde no se pudo generar embedding (embedding_db.py).
FALLIDOS_PATH = RAIZ / "fallidos.txt"

# Registro de accesos (timestamp, nombre, confianza, frame) de realtime.py --log.
REGISTRO_ACCESO_PATH = RAIZ / "registro_acceso.csv"

# ---------------------------------------------------------------------------
# Parametros de DETECCION (YOLO)
# ---------------------------------------------------------------------------
# Umbral de confianza minimo para aceptar una deteccion de rostro.
# Definido en el Experimento 2 (umbrales de confianza): sobre un video dificil
# (distancia, contraluz, fotos/pantallas) 0.5 resulto el unico umbral con 0
# errores de detector: a 0.3 aparece 1 falso positivo de objeto y a 0.7 se
# pierde 1 rostro real a contraluz. Por eso se usa 0.5.
DETECTION_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Parametros de RECONOCIMIENTO (embeddings)
# ---------------------------------------------------------------------------
# Distancia euclidiana maxima para considerar una coincidencia valida.
# Valor por defecto recomendado por face_recognition (0.6).
# La tarea 5.2b (calibracion de umbral por FAR/FRR/EER) puede refinar
# este valor; al hacerlo, actualizar aqui.
RECOGNITION_THRESHOLD = 0.6

# Dimension del vector de embedding que produce face_recognition (dlib).
EMBEDDING_DIM = 128

# ---------------------------------------------------------------------------
# Parametros de REGISTRO (generacion de embeddings)
# ---------------------------------------------------------------------------
# Cantidad de fotos por persona recomendada para un registro suficiente.
# Justificado en el Experimento 3 (reconocimiento held-out): el reconocimiento
# de fotos no vistas se satura a partir de ~10 fotos (~99%); con ~5 ya se llega
# a ~96%. Mas fotos dan ganancia marginal. Ademas, NO hace falta forzar variedad
# (perfil/accesorios): el modelo de embeddings generaliza desde frontales limpias.
# (Medido con 3 personas; con muchas mas podria convenir algo mas de fotos.)
FOTOS_POR_PERSONA_RECOMENDADAS = 10
FOTOS_POR_PERSONA_MINIMAS = 5

# Extensiones de imagen aceptadas al recorrer el dataset.
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Parametros de TIEMPO REAL (video)
# ---------------------------------------------------------------------------
# Indice de la camara para cv2.VideoCapture (0 = webcam por defecto).
CAMARA_ID = 0

# FPS objetivo a mostrar / limitar en el loop de video.
FPS_MOSTRAR = 30

# Optimizacion: reconocer solo cada N frames (reusar el ultimo nombre en
# los intermedios) para subir los FPS. Se ajusta en la optimizacion 6.3
# y se mide en el Experimento 6. 1 = reconocer en todos los frames.
RECOGNIZE_EVERY_N_FRAMES = 1

# Segundos de espera antes de volver a registrar a la misma persona en el
# CSV, para no duplicar un acceso por cada frame (tarea 6.4).
CSV_COOLDOWN_SECONDS = 5

# ---------------------------------------------------------------------------
# Alias de compatibilidad
# ---------------------------------------------------------------------------
# Nombres que ya importa el codigo existente (test/verificar_yolo.py).
# Se mantienen para no romper esos imports.
MODELO_YOLO = str(YOLO_WEIGHTS)
CONFIANZA_MINIMA = DETECTION_CONFIDENCE
