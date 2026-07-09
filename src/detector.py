"""Módulo de detección de rostros usando YOLO."""

import cv2
from ultralytics import YOLO

from src import configuracion as cfg


class FaceDetector:

    def __init__(
        self,
        weights_path=cfg.YOLO_WEIGHTS,
        conf_threshold: float = cfg.DETECTION_CONFIDENCE,
    ) -> None:
        # Los defaults salen de configuracion.py (ruta de pesos y umbral de
        # confianza calibrado en el Experimento 2). Se pueden sobrescribir
        # pasando argumentos.
        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold

    def detect(self, frame, conf: float | None = None) -> list:
        """Detecta rostros usando YOLO sin tracking.

        Para cada frame ejecuta inferencia independiente (sin mantener
        IDs entre frames). Ideal para procesamiento de fotos individuales.

        Args:
            frame: Imagen numpy array BGR.
            conf: Umbral de confianza. Si es None, usa self.conf_threshold.

        Returns:
            Lista de tuplas (x1, y1, x2, y2, confidence).
        """
        if conf is None:
            conf = self.conf_threshold

        results = self.model(frame, conf=conf, verbose=False)

        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append((x1, y1, x2, y2, float(score)))

        return detections

    def track(self, frame, conf: float | None = None) -> list:
        """Detecta y rastrea rostros usando ByteTrack (Ultralytics).

        Mantiene IDs consistentes entre frames mediante persist=True.
        Si ByteTrack aun no esta inicializado (primeros frames),
        devuelve lista vacia.

        Args:
            frame: Imagen numpy array BGR.
            conf: Umbral de confianza. Si es None, usa self.conf_threshold.

        Returns:
            Lista de tuplas (x1, y1, x2, y2, confidence, track_id).
        """
        if conf is None:
            conf = self.conf_threshold

        results = self.model.track(frame, conf=conf, verbose=False, persist=True)

        if len(results) == 0 or results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        detections = []
        for box, score, tid in zip(boxes, scores, ids):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append((x1, y1, x2, y2, float(score), int(tid)))

        return detections

    def crop_face(self, frame, detection: tuple, margin: float = cfg.FACE_CROP_MARGIN):
        """Recorta un rostro del frame expandiendo el bounding box.

        Args:
            frame: Imagen numpy array BGR original.
            detection: Una deteccion individual (x1, y1, x2, y2, ...).
            margin: Fraccion del tamanio del bounding box para expandir
                el recorte (0.3 = 30%).

        Returns:
            Recorte BGR (numpy array) o None si el recorte no es valido.
        """
        x1, y1, x2, y2 = map(int, detection[:4])
        h, w = frame.shape[:2]

        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        if y2 > y1 and x2 > x1:
            return frame[y1:y2, x1:x2]
        return None
