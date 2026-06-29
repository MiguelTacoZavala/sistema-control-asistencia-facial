"""Punto de entrada del sistema de control de asistencia facial."""
import sys
from collections import Counter
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.tracker import Tracker


def main() -> None:
    print("Iniciando Sistema de Control de Asistencia Facial...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)
    recognizer = FaceRecognizer("embeddings.pkl", 0.4)
    tracker = Tracker(delay_frames=15, min_frames=15, min_ratio=0.8)
    frame_count = 0
    # Conteo de votos crudos del reconocedor, para verificar en consola
    # si hay confusion entre personas (ej. muchos "kevin" cuando es gerardo).
    votos = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: no se pudo capturar la imagen")
            break

        frame_count += 1
        detections = detector.track(frame)

        for det in detections:
            x1, y1, x2, y2, conf, track_id = det
            nombre_mostrar = "Desconocido"

            nombre = tracker.get_name(track_id)
            if nombre is not None:
                nombre_mostrar = nombre
            elif tracker.tick(track_id):
                if frame_count % 3 == 0:
                    crop = detector.crop_face(frame, det)
                    if crop is not None:
                        nombre_voto, _ = recognizer.recognize(crop)
                        votos[nombre_voto] += 1
                        nombre = tracker.vote(track_id, nombre_voto)
                nombre_mostrar = nombre if nombre else "Reconociendo..."
            else:
                nombre_mostrar = "Reconociendo..."

            if nombre_mostrar == "Desconocido":
                color = (0, 0, 255)
            elif nombre_mostrar == "Reconociendo...":
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, nombre_mostrar, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        if frame_count % 30 == 0 and votos:
            print(f"[frame {frame_count}] " +
                  " | ".join(f"{n}: {c}" for n, c in votos.most_common()))

        cv2.imshow("Camara", frame)

        if cv2.waitKey(1) == 27:
            break

    print("\n=== Resumen de reconocimientos (votos crudos) ===")
    total = sum(votos.values()) or 1
    for nombre, c in votos.most_common():
        print(f"  {nombre:15s}: {c:4d}  ({100 * c / total:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
