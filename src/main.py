"""Punto de entrada del sistema de control de asistencia facial."""
import cv2
from detector import FaceDetector

def main() -> None:
    print("Iniciando Sistema de Control de Asistencia Facial...")
    # Abrir cámara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)    

    while True:
        # Leer frame de la cámara
        ret, frame = cap.read()

        # Verificar si la captura fue correcta
        if not ret:
            print("Error: no se pudo capturar la imagen")
            break
        
        detections = detector.detect(frame)        
        frame = detector.draw_detections(frame, detections)        

        # Mostrar imagen de la cámara
        cv2.imshow("Camara", frame)

        # Salir si presionas ESC
        if cv2.waitKey(1) == 27:
            break

    # Liberar cámara
    cap.release()

    # Cerrar ventanas
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
