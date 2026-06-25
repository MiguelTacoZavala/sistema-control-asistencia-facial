"""Reconocimiento facial por comparacion de embeddings.

Carga embeddings.pkl generado por embedding_db.py, recibe un rostro
recortado (BGR), extrae su vector de 128 dimensiones con face_recognition,
y lo compara contra la base usando distancia euclidiana.
"""
import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np


class FaceRecognizer:
    """Compara un rostro contra la base de embeddings y devuelve el nombre.

    Args:
        db_path: Ruta al archivo embeddings.pkl.
        threshold: Distancia euclidiana maxima para considerar una
            coincidencia valida (0.6 es el valor recomendado por
            face_recognition).
    """

    def __init__(self, db_path: str = "embeddings.pkl", threshold: float = 0.6) -> None:
        self.db_path = db_path
        self.threshold = threshold
        self.embeddings_db: dict[str, list[np.ndarray]] = {}
        self.reload_db()

    def reload_db(self) -> None:
        """Recarga embeddings.pkl desde disco.

        Si el archivo no existe o esta corrupto, la base queda vacia
        y se emite una advertencia. Util para registrar nuevas personas
        sin reiniciar el sistema.
        """
        path = Path(self.db_path)
        if not path.exists():
            print(f"[ADVERTENCIA] No se encontro {self.db_path}. Base vacia.")
            self.embeddings_db = {}
            return

        try:
            with open(path, "rb") as f:
                self.embeddings_db = pickle.load(f)
            print(
                f"Base de embeddings cargada: "
                f"{len(self.embeddings_db)} persona(s)"
            )
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"[ERROR] No se pudo leer {self.db_path}: {e}")
            self.embeddings_db = {}

    def recognize(self, face_crop: np.ndarray) -> tuple[str, float]:
        """Reconoce un rostro recortado comparando contra la base.

        Args:
            face_crop: Rostro recortado en formato BGR (numpy array).

        Returns:
            Tupla (nombre, distancia):
                - ("Nombre persona", distancia) si coincide con el umbral.
                - ("Desconocido", dist_min) si no hay coincidencia.
                - ("Sin rostro", 0.0) si no se pudo extraer embedding.
                - ("Error", 0.0) si ocurre una excepcion inesperada.
        """
        try:
            # Convertir BGR -> RGB porque face_recognition usa RGB
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Extraer embedding del rostro recortado
            encodings = face_recognition.face_encodings(rgb)
            if not encodings:
                return "Sin rostro", 0.0

            encoding_input = encodings[0]

            # Si no hay personas registradas en la base
            if not self.embeddings_db:
                return "Desconocido", 0.0

            # Buscar la persona con menor distancia euclidiana
            mejor_persona = "Desconocido"
            mejor_distancia = float("inf")

            for persona, lista_embeddings in self.embeddings_db.items():
                for emb in lista_embeddings:
                    dist = np.linalg.norm(encoding_input - emb)
                    if dist < mejor_distancia:
                        mejor_distancia = dist
                        mejor_persona = persona

            if mejor_distancia < self.threshold:
                return mejor_persona, mejor_distancia
            return "Desconocido", mejor_distancia

        except Exception as e:
            print(f"[ERROR] Fallo en recognize(): {e}")
            return "Error", 0.0
