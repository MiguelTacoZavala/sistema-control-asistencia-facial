from pathlib import Path

# Datos de prueba

RAIZ = Path(__file__).resolve().parent.parent

RUTA_MODELOS = RAIZ / "modelos"
RUTA_EMPLEADOS = RAIZ / "datos" / "empleados"
RUTA_ASISTENCIAS = RAIZ / "datos" / "asistencias"

MODELO_YOLO = str(RUTA_MODELOS / "yolo-face.pt")

CONFIANZA_MINIMA = 0.5
FPS_MOSTRAR = 30
CAMARA_ID = 0
