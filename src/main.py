"""Punto de entrada del sistema de control de asistencia facial."""
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer


def main() -> None:
    print("Iniciando Sistema de Control de Asistencia Facial...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)
    recognizer = FaceRecognizer("embeddings.pkl", threshold=0.6)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: no se pudo capturar la imagen")
            break

        detections = detector.detect(frame)

        for det in detections:
            crop = detector.crop_face(frame, det)
            if crop is not None:
                nombre, _ = recognizer.recognize(crop)
                x1, y1 = map(int, det[:2])
                cv2.putText(
                    frame, nombre, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        frame = detector.draw_detections(frame, detections)
        cv2.imshow("Camara", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
