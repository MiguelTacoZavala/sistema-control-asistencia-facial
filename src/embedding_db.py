"""Generacion de embeddings faciales a partir del dataset de rostros conocidos.

Recorre dataset/known_faces/, detecta rostros con YOLO,
genera vectores de embedding con face_recognition (128 dimensiones),
y los guarda en embeddings.pkl para usarlos en el reconocimiento.

Que es un embedding?
    Es un vector numerico (lista de 128 floats) que representa las
    caracteristicas unicas de un rostro. La red neuronal de face_recognition
    (basada en dlib ResNet) convierte la foto en este vector. La idea es que
    rostros de la misma persona producen vectores cercanos (distancia chica),
    mientras que rostros distintos producen vectores lejanos.

Por que 128 dimensiones?
    face_recognition usa el modelo "dlib_face_recognition_resnet_model_v1"
    entrenado con Labeled Faces in the Wild (LFW). Produce vectores de 128
    componentes porque es un balance entre precision y velocidad de comparacion.
    DeepFace (alternativa) usa 512 dimensiones con su modelo VGG-Face.
"""

import pickle
import sys
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np

# Asegurar que la raíz del proyecto esté en sys.path para poder
# ejecutar el script con python src/embedding_db.py o con python -m
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.detector import FaceDetector

# Rutas por defecto
RUTA_DATASET = RAIZ / "dataset" / "known_faces"
RUTA_EMBEDDINGS = RAIZ / "embeddings.pkl"
RUTA_FALLIDOS = RAIZ / "fallidos.txt"
RUTA_MODELO_YOLO = RAIZ / "models" / "yolov8n-face.pt"


class EmbeddingDB:
    """Genera y almacena embeddings faciales a partir del dataset.

    Recibe la ruta al directorio con subcarpetas de personas.
    Cada subcarpeta debe tener el nombre de la persona y contener
    sus fotos (jpg, jpeg o png).
    """

    def __init__(self, ruta_dataset: str | Path = RUTA_DATASET) -> None:
        """Inicializa el generador de embeddings.

        Args:
            ruta_dataset: Ruta a la carpeta con subcarpetas de personas.
                Por defecto apunta a dataset/known_faces/.
        """
        self.ruta_dataset = Path(ruta_dataset)
        self.detector = FaceDetector(str(RUTA_MODELO_YOLO), conf_threshold=0.5)
        self.embeddings: dict[str, list[np.ndarray]] = {}
        self.fallidos: list[str] = []

    def generar(self) -> dict[str, list[np.ndarray]]:
        """Recorre el dataset y genera embeddings para cada persona.

        Por cada foto:
          1. Carga la imagen (BGR con OpenCV).
          2. Detecta rostros con FaceDetector (YOLO).
          3. Si detecta al menos un rostro, toma el de mayor confianza.
          4. Convierte coordenadas YOLO (x1,y1,x2,y2) al formato
             que espera face_recognition: (top, right, bottom, left).
          5. Convierte la imagen de BGR a RGB (face_recognition
             trabaja en RGB, no en BGR).
          6. Genera el embedding con face_recognition.face_encodings().
             Esta funcion usa dlib internamente para producir un vector
             de 128 floats por rostro.
          7. Si no se genera embedding, se registra como fallido.

        Returns:
            Diccionario {nombre_persona: [embedding_1, embedding_2, ...]}
        """
        t_inicio = time.perf_counter()
        total_fotos = 0
        total_embeddings = 0

        personas = sorted(
            [p for p in self.ruta_dataset.iterdir() if p.is_dir()]
        )

        if not personas:
            print(f"[ADVERTENCIA] No se encontraron subcarpetas en {self.ruta_dataset}")
            return self.embeddings

        for carpeta_persona in personas:
            nombre = carpeta_persona.name
            # Solo archivos de imagen
            fotos = sorted(
                f for f in carpeta_persona.glob("*")
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            embeddings_persona: list[np.ndarray] = []
            fallidos_persona = 0
            procesados_persona = 0

            for ruta_foto in fotos:
                total_fotos += 1
                procesados_persona += 1

                # 1. Cargar imagen en BGR (formato nativo de OpenCV)
                imagen_bgr = cv2.imread(str(ruta_foto))
                if imagen_bgr is None:
                    self.fallidos.append(str(ruta_foto.relative_to(RAIZ)))
                    fallidos_persona += 1
                    continue

                # 2. Detectar rostros con YOLO
                detecciones = self.detector.detect(imagen_bgr)
                if not detecciones:
                    self.fallidos.append(str(ruta_foto.relative_to(RAIZ)))
                    fallidos_persona += 1
                    continue

                # 3. Tomar la deteccion con mayor confianza
                mejor = max(detecciones, key=lambda d: d[4])
                x1, y1, x2, y2, _ = mejor

                # 4. Convertir formato de coordenadas:
                #    YOLO: (x1, y1, x2, y2)  -> esquina sup-izq e inf-der
                #    face_recognition: (top, right, bottom, left)
                top, right, bottom, left = y1, x2, y2, x1

                # 5. Convertir BGR -> RGB porque face_recognition
                #    usa dlib internamente y dlib espera RGB
                imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

                # 6. Generar embedding. face_encodings recibe la imagen
                #    y una lista de ubicaciones conocidas (opcional).
                #    Si le pasamos known_face_locations, dlib no vuelve
                #    a detectar el rostro, usa nuestras coordenadas.
                #    Devuelve una lista de arrays de 128 floats.
                encoding = face_recognition.face_encodings(
                    imagen_rgb,
                    known_face_locations=[(top, right, bottom, left)],
                )

                if not encoding:
                    # face_recognition no pudo extraer el embedding
                    # incluso con la ubicacion indicada (imagen muy
                    # oscura, borrosa, rostro muy pequenio, etc.)
                    self.fallidos.append(str(ruta_foto.relative_to(RAIZ)))
                    fallidos_persona += 1
                    continue

                embeddings_persona.append(encoding[0])
                total_embeddings += 1

            if embeddings_persona:
                self.embeddings[nombre] = embeddings_persona

            print(
                f"  {nombre}: {procesados_persona} fotos, "
                f"{fallidos_persona} fallidas, "
                f"{len(embeddings_persona)} embeddings"
            )

        t_total = time.perf_counter() - t_inicio
        t_promedio_ms = (t_total / total_fotos * 1000) if total_fotos else 0

        print(f"\nResumen:")
        print(f"  Personas registradas: {len(personas)}")
        print(f"  Total fotos procesadas: {total_fotos}")
        print(f"  Total embeddings generados: {total_embeddings}")
        print(f"  Total fallidos: {len(self.fallidos)}")
        print(f"  Tiempo total: {t_total:.2f} s")
        print(f"  Tiempo promedio por foto: {t_promedio_ms:.1f} ms")
        print(
            f"  Dimension del vector: "
            f"{len(encoding[0]) if total_embeddings else 'N/A'}"
        )

        return self.embeddings

    def guardar(self, ruta: str | Path = RUTA_EMBEDDINGS) -> None:
        """Guarda los embeddings en disco y la lista de fallidos.

        Args:
            ruta: Donde guardar embeddings.pkl.
                Por defecto en la raiz del proyecto.
        """
        # Guardar embeddings como pickle
        with open(ruta, "wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"\nEmbeddings guardados: {ruta}")
        file_size_kb = ruta.stat().st_size / 1024
        print(f"  Tamano: {file_size_kb:.1f} KB")

        # Guardar lista de fallidos
        with open(RUTA_FALLIDOS, "w", encoding="utf-8") as f:
            for fallido in self.fallidos:
                f.write(fallido + "\n")
        print(f"Fallidos guardados: {RUTA_FALLIDOS}")
        if self.fallidos:
            print(f"  {len(self.fallidos)} foto(s) con problemas")


if __name__ == "__main__":
    db = EmbeddingDB()
    db.generar()
    db.guardar()
