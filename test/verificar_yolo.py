"""Verifica que los pesos YOLO de detección de rostros cargan y detectan correctamente."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from ultralytics import YOLO

from src.configuracion import CONFIANZA_MINIMA, MODELO_YOLO

IMAGEN_PRUEBA = RAIZ / "dataset" / "known_faces" / "miguel" / "foto1.jpeg"


def main() -> None:
    weights = Path(MODELO_YOLO)

    if not weights.exists():
        print(f"[FAIL] No se encontraron los pesos en: {weights}")
        print("Descarga yolov8n-face.pt y colócalo en models/ (ver README).")
        sys.exit(1)

    print(f"Cargando modelo: {weights}")
    model = YOLO(str(weights))
    print("[OK] Modelo cargado sin errores")

    if not IMAGEN_PRUEBA.exists():
        print(f"[FAIL] Imagen de prueba no encontrada: {IMAGEN_PRUEBA}")
        sys.exit(1)

    print(f"Ejecutando predict sobre: {IMAGEN_PRUEBA.name}")
    results = model.predict(str(IMAGEN_PRUEBA), conf=CONFIANZA_MINIMA, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("[FAIL] No se detectaron rostros en la imagen de prueba")
        sys.exit(1)

    num_detections = len(results[0].boxes)
    print(f"[OK] {num_detections} rostro(s) detectado(s)")

    for i, box in enumerate(results[0].boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  Detección {i + 1}: conf={conf:.2f}, bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    print("\nVerificación YOLO completada correctamente.")


if __name__ == "__main__":
    main()
