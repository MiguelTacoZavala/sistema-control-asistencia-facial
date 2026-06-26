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

        height, width = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            margin_x = int((x2 - x1) * 0.3)
            margin_y = int((y2 - y1) * 0.3)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(width, x2 + margin_x)
            y2 = min(height, y2 + margin_y)

            if y2 > y1 and x2 > x1:
                crop = frame[y1:y2, x1:x2]
                nombre, _ = recognizer.recognize(crop)
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
