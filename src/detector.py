"""Módulo de detección de rostros usando YOLO."""

from ultralytics import YOLO


class Detector:
    """Detecta rostros en fotogramas usando YOLO."""

    def __init__(self, modelo_path: str) -> None:
        self.modelo_path = modelo_path
        self.modelo: YOLO | None = None

    def cargar_modelo(self) -> None:
        self.modelo = YOLO(self.modelo_path)

    def detectar(self, frame) -> list:
        return []
