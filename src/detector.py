"""Módulo de detección de rostros usando YOLO."""

import cv2
from ultralytics import YOLO


class FaceDetector:

    def __init__(self, weights_path: str, conf_threshold: float = 0.5) -> None:
        """Inicializa el detector cargando el modelo YOLO.
        Args:
            weights_path: Ruta al archivo .pt de los pesos del modelo.
            conf_threshold: Umbral de confianza mínimo para filtrar
                detecciones (0.0 a 1.0). Por defecto 0.5.
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(self.weights_path)
        self._local_next_id = 0

    def detect(self, frame, conf: float | None = None) -> list:
        """Ejecuta la detección de rostros sobre un frame.

        Args:
            frame: Imagen en formato numpy array BGR (como la que
                devuelve cv2.VideoCapture.read()).
            conf: Umbral de confianza para esta llamada. Si es None
                se usa el umbral guardado en el constructor.

        Returns:
            Lista de detecciones. Cada detección es una tupla:
            (x1, y1, x2, y2, confidence), donde (x1, y1) es la
            esquina superior izquierda y (x2, y2) la inferior
            derecha del bounding box, en píxeles.
        """
        if conf is None:
            conf = self.conf_threshold

        # results es una lista de resultados, uno por imagen.
        # Como procesamos un frame a la vez, tomamos el primero.
        results = self.model(frame, conf=conf, verbose=False)

        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        # results[0].boxes.xyxy contiene los bounding boxes en
        # formato [x1, y1, x2, y2] normalizado a píxeles. Cada
        # fila es un tensor de 4 elementos.
        
        # results[0].boxes.conf contiene la confianza de cada
        # detección. Ambas son tensores en GPU; los pasamos a
        # CPU y luego a numpy para trabajar con Python estándar.
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append((x1, y1, x2, y2, float(score)))

        return detections

    def draw_detections(self, frame, detections: list):
        """Dibuja los bounding boxes y la confianza sobre el frame.

        Args:
            frame: Imagen numpy array BGR original.
            detections: Lista de tuplas (x1, y1, x2, y2, confidence)
                devuelta por detect().

        Returns:
            El frame modificado con los rectángulos y etiquetas
            dibujados (se modifica in-place, también se devuelve
            para conveniencia).
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame

    def track(self, frame, conf: float | None = None) -> list:
        """Detecta y rastrea rostros usando ByteTrack (Ultralytics).

        Mantiene IDs consistentes entre frames mediante persist=True.
        Si ByteTrack aun no se inicializo (primer frame), asigna IDs
        locales secuenciales como fallback.

        Args:
            frame: Imagen numpy array BGR.
            conf: Umbral de confianza. Si es None, usa self.conf_threshold.

        Returns:
            Lista de tuplas (x1, y1, x2, y2, confidence, track_id).
        """
        if conf is None:
            conf = self.conf_threshold

        results = self.model.track(frame, conf=conf, verbose=False, persist=True)

        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        ids = results[0].boxes.id

        if ids is not None:
            ids = ids.cpu().numpy().astype(int)
        else:
            # ByteTrack aun no inicializado (primeros frames):
            # asignar IDs locales secuenciales para que el tracker
            # de nombres funcione desde el inicio
            ids = list(range(self._local_next_id, self._local_next_id + len(boxes)))
            self._local_next_id += len(boxes)

        for box, score, tid in zip(boxes, scores, ids):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append((x1, y1, x2, y2, float(score), int(tid)))

        return detections

    def crop_face(self, frame, detection: tuple, margin: float = 0.3):
        """Recorta un rostro del frame expandiendo el bounding box.

        Args:
            frame: Imagen numpy array BGR original.
            detection: Una deteccion individual (x1, y1, x2, y2, confidence)
                devuelta por detect().
            margin: Fraccion del tamanio del bounding box para expandir
                el recorte (0.3 = 30%). Ayuda a que face_recognition
                tenga contexto alrededor del rostro.

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
