"""Punto de entrada del sistema de control de asistencia facial."""
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.tracker import Tracker
from src.asistencia import AsistenciaDB

COLORES = {"Reconociendo...": (0, 165, 255), "Desconocido": (0, 0, 255)}


def main() -> None:
    print("Iniciando Sistema de Control de Asistencia Facial...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)
    recognizer = FaceRecognizer("embeddings.pkl", threshold=0.6)
    tracker = Tracker(grace_frames=30)
    asistencia = AsistenciaDB()

    welcome_text = ""
    welcome_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: no se pudo capturar la imagen")
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2, _, track_id = det

            name = tracker.get_name(track_id)
            if name is None:
                if not tracker.is_grace(track_id):
                    crop = detector.crop_face(frame, det)
                    if crop is not None:
                        name, dist = recognizer.recognize(crop)
                        print(name , " " , dist)
                        tracker.confirm(track_id, name)
                        if name != "Desconocido":
                            completo = name.replace("_", " ")
                            asistencia.registrar(completo)
                            welcome_text = f"BIENVENIDO, {name.split('_')[0]}"
                            welcome_timer = 60
                nombre_mostrar = name or "Reconociendo..."
                nombre_mostrar = nombre_mostrar.split("_")[0] if nombre_mostrar not in COLORES else nombre_mostrar
            else:
                nombre_mostrar = name.split("_")[0]

            color = COLORES.get(nombre_mostrar, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, nombre_mostrar, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        if welcome_timer > 0:
            h, w = frame.shape[:2]
            cv2.putText(
                frame, welcome_text, (w // 2 - 150, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
            )
            welcome_timer -= 1

        cv2.imshow("Camara", frame)

        if cv2.waitKey(1) == 27:
            break

    asistencia.cerrar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
