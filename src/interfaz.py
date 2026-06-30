"""Dashboard "Control de acceso - en vivo" dibujado con OpenCV.

Compone un panel completo sobre un lienzo numpy (no es una ventana de video
pelada como main.py): cabecera, panel de camara con cajas de colores, lista de
"Ultimos accesos", tarjetas de estadisticas y panel de controles.

El video, las cajas y la composicion se hacen con OpenCV. El texto se dibuja
con PIL/Pillow en una sola pasada por frame, para tener acentos y tipografia
limpia. Los 4 botones (espacio / F / A / E) por ahora SOLO se dibujan, no
ejecutan ninguna accion.

Ejecutar con:  python -m src.interfaz
"""
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.tracker import Tracker
from src.asistencia import AsistenciaDB

# --------------------------------------------------------------------------- #
# Paleta (en RGB; para OpenCV se invierte a BGR con bgr()).
# --------------------------------------------------------------------------- #
BG = (244, 246, 248)
PANEL = (255, 255, 255)
BORDER = (226, 230, 235)
INK = (31, 41, 55)
MUTED = (123, 131, 143)
CARD = (248, 249, 251)
KEYBG = (242, 243, 246)
GREEN = (34, 160, 92)
RED = (224, 78, 74)
ORANGE = (240, 154, 40)

# Avatares: colores que se reparten por nombre.
AVATAR_COLORS = [(80, 140, 220), (90, 175, 120), (210, 130, 70),
                 (160, 120, 210), (210, 100, 140), (95, 170, 190)]

W, H = 1000, 720


def bgr(rgb):
    """Convierte una tupla RGB a BGR para las funciones de OpenCV."""
    return (rgb[2], rgb[1], rgb[0])


# --------------------------------------------------------------------------- #
# Texto: se acumulan los textos del frame y se dibujan en una sola pasada PIL.
# --------------------------------------------------------------------------- #
_FONT_CACHE: dict = {}


def _font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidatos = (
        [r"C:\Windows\Fonts\segoeuib.ttf", "DejaVuSans-Bold.ttf"]
        if bold else
        [r"C:\Windows\Fonts\segoeui.ttf", "DejaVuSans.ttf"]
    )
    fuente = None
    for ruta in candidatos:
        try:
            fuente = ImageFont.truetype(ruta, size)
            break
        except OSError:
            continue
    if fuente is None:
        fuente = ImageFont.load_default()
    _FONT_CACHE[key] = fuente
    return fuente


class Pluma:
    """Cola de textos a dibujar con PIL al final del frame."""

    def __init__(self) -> None:
        self.items: list = []

    def add(self, texto, xy, size, color, bold=False, anchor="la") -> None:
        self.items.append((texto, xy, size, color, bold, anchor))

    def render(self, lienzo_bgr):
        img = Image.fromarray(cv2.cvtColor(lienzo_bgr, cv2.COLOR_BGR2RGB))
        dibujo = ImageDraw.Draw(img)
        for texto, xy, size, color, bold, anchor in self.items:
            dibujo.text(xy, texto, font=_font(size, bold), fill=color, anchor=anchor)
        self.items.clear()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Primitivas de dibujo (OpenCV).
# --------------------------------------------------------------------------- #
def _rounded(img, x, y, w, h, r, color) -> None:
    """Rectangulo relleno con esquinas redondeadas."""
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x + r, y), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x, y + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x + r, y + r), (x2 - r, y + r), (x + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1)


def panel(img, x, y, w, h, r=14, fill=PANEL, borde=BORDER) -> None:
    """Panel blanco con borde sutil de 1 px."""
    _rounded(img, x, y, w, h, r, bgr(borde))
    _rounded(img, x + 1, y + 1, w - 2, h - 2, r - 1, bgr(fill))


def check(img, cx, cy, color) -> None:
    """Marca de verificacion (visto)."""
    cv2.line(img, (cx - 6, cy), (cx - 2, cy + 5), bgr(color), 2, cv2.LINE_AA)
    cv2.line(img, (cx - 2, cy + 5), (cx + 7, cy - 6), bgr(color), 2, cv2.LINE_AA)


def cruz(img, cx, cy, color) -> None:
    """Marca de aspa (rechazado)."""
    cv2.line(img, (cx - 6, cy - 6), (cx + 6, cy + 6), bgr(color), 2, cv2.LINE_AA)
    cv2.line(img, (cx - 6, cy + 6), (cx + 6, cy - 6), bgr(color), 2, cv2.LINE_AA)


def color_avatar(nombre: str):
    return AVATAR_COLORS[sum(map(ord, nombre)) % len(AVATAR_COLORS)]


def iniciales(nombre: str) -> str:
    if nombre == "Desconocido":
        return "?"
    partes = [p for p in nombre.replace("_", " ").split() if p]
    if len(partes) >= 2:
        return (partes[0][0] + partes[1][0]).upper()
    return partes[0][:2].upper() if partes else "?"


def nombre_legible(nombre: str) -> str:
    partes = nombre.replace("_", " ").split()
    return " ".join(partes[:2]) if partes else nombre


# --------------------------------------------------------------------------- #
# Estado de la interfaz.
# --------------------------------------------------------------------------- #
class Dashboard:
    def __init__(self) -> None:
        self.accesos: deque = deque(maxlen=5)   # (nombre_legible, hora, ok)
        self.total_hoy = 0
        self.en_cuadro = 0
        self.fps = 0.0
        self.mensaje = ""        # aviso temporal en pantalla
        self.mensaje_timer = 0   # frames que quedan de mostrar el aviso
        self.ultimo_reconocido = ""   # nombre completo del ultimo conocido

    def registrar_acceso(self, nombre: str) -> None:
        ok = nombre != "Desconocido"
        hora = datetime.now().strftime("%H:%M:%S")
        self.accesos.appendleft((nombre_legible(nombre), hora, ok))
        if ok:
            self.total_hoy += 1
            self.ultimo_reconocido = nombre.replace("_", " ")

    def avisar(self, texto: str, frames: int = 70) -> None:
        """Muestra un aviso temporal en pantalla durante `frames` frames."""
        self.mensaje = texto
        self.mensaje_timer = frames


# --------------------------------------------------------------------------- #
# Secciones del dashboard.
# --------------------------------------------------------------------------- #
def dibujar_header(img, pen: Pluma, fps: float) -> None:
    panel(img, 16, 12, 968, 52)
    cy = 12 + 26
    cv2.circle(img, (40, cy), 6, bgr(GREEN), -1, cv2.LINE_AA)
    pen.add("Control de acceso - en vivo", (58, cy), 19, INK, bold=True, anchor="lm")
    # Reloj + badge de FPS a la derecha.
    pen.add(datetime.now().strftime("%H:%M:%S"), (812, cy), 16, MUTED, anchor="rm")
    _rounded(img, 900, cy - 14, 76, 28, 13, bgr((233, 245, 238)))
    pen.add(f"{fps:.0f} FPS", (938, cy), 14, GREEN, bold=True, anchor="mm")


def dibujar_camara(img, pen: Pluma, frame, detections, tracker) -> None:
    panel(img, 16, 76, 584, 372)
    pen.add("CAM 01", (36, 96), 14, MUTED, bold=True, anchor="lm")

    vx, vy, vw, vh = 32, 110, 552, 290
    _rounded(img, vx, vy, vw, vh, 8, (30, 32, 36))
    if frame is not None:
        fh, fw = frame.shape[:2]
        img[vy:vy + vh, vx:vx + vw] = cv2.resize(frame, (vw, vh))
        sx, sy = vw / fw, vh / fh
        for (x1, y1, x2, y2, conf, tid), etiqueta, col in _cajas(detections, tracker):
            rx1, ry1 = vx + int(x1 * sx), vy + int(y1 * sy)
            rx2, ry2 = vx + int(x2 * sx), vy + int(y2 * sy)
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), bgr(col), 2)
            texto = etiqueta if etiqueta == "procesando..." else f"{etiqueta} {conf:.2f}"
            tw = len(texto) * 8 + 10
            ty = ry1 if ry1 - 22 > vy else ry2 + 22
            cv2.rectangle(img, (rx1, ty - 22), (rx1 + tw, ty), bgr(col), -1)
            pen.add(texto, (rx1 + 5, ty - 11), 13, (255, 255, 255), anchor="lm")

    # Leyenda.
    lx, ly = 36, 424
    for txt, col in (("Reconocido", GREEN), ("Desconocido", RED), ("Procesando", ORANGE)):
        _rounded(img, lx, ly - 6, 12, 12, 3, bgr(col))
        pen.add(txt, (lx + 18, ly), 13, MUTED, anchor="lm")
        lx += 18 + len(txt) * 8 + 24


def _cajas(detections, tracker):
    """Genera (det, etiqueta_corta, color_rgb) para cada deteccion."""
    for det in detections:
        tid = det[5]
        name = tracker.get_name(tid)
        if name is None:
            etiqueta, col = "procesando...", ORANGE
        elif name == "Desconocido":
            etiqueta, col = "Desconocido", RED
        else:
            etiqueta, col = name.replace("_", " ").split()[0], GREEN
        yield det, etiqueta, col


def dibujar_accesos(img, pen: Pluma, accesos) -> None:
    panel(img, 616, 76, 368, 372)
    pen.add("Ultimos accesos", (636, 100), 17, INK, bold=True, anchor="lm")
    pen.add(str(len(accesos)), (964, 100), 14, MUTED, anchor="rm")
    cv2.line(img, (636, 116), (964, 116), bgr(BORDER), 1)

    y = 150
    if not accesos:
        pen.add("Sin accesos todavia", (636, y), 14, MUTED, anchor="lm")
        return
    for nombre, hora, ok in accesos:
        col = color_avatar(nombre) if ok else (150, 150, 156)
        cv2.circle(img, (656, y), 18, bgr(col), -1, cv2.LINE_AA)
        pen.add(iniciales(nombre), (656, y), 13, (255, 255, 255), bold=True, anchor="mm")
        pen.add(nombre, (686, y - 8), 15, INK, bold=True, anchor="lm")
        pen.add(hora, (686, y + 11), 12, MUTED, anchor="lm")
        if ok:
            check(img, 950, y, GREEN)
        else:
            cruz(img, 948, y, RED)
        y += 56


def dibujar_stats(img, pen: Pluma, dash: Dashboard) -> None:
    panel(img, 16, 464, 584, 240)
    cols = (("FPS", f"{dash.fps:.0f}"), ("En cuadro", str(dash.en_cuadro)),
            ("Total hoy", str(dash.total_hoy)))
    x = 44
    for etiqueta, valor in cols:
        pen.add(etiqueta, (x, 494), 13, MUTED, anchor="lm")
        pen.add(valor, (x, 530), 34, INK, bold=True, anchor="lm")
        x += 188
    cv2.line(img, (40, 566), (576, 566), bgr(BORDER), 1)

    # Panel reservado: confirmacion del ultimo reconocido (nombre completo).
    if dash.ultimo_reconocido:
        pen.add("Registrado correctamente:", (44, 594), 14, GREEN, bold=True, anchor="lm")
        col = color_avatar(dash.ultimo_reconocido)
        cv2.circle(img, (66, 644), 22, bgr(col), -1, cv2.LINE_AA)
        pen.add(iniciales(dash.ultimo_reconocido), (66, 644), 16,
                (255, 255, 255), bold=True, anchor="mm")
        pen.add(dash.ultimo_reconocido, (102, 644), 22, INK, bold=True, anchor="lm")
    else:
        pen.add("En espera de reconocimiento", (44, 640), 16, MUTED, anchor="lm")


def dibujar_controles(img, pen: Pluma) -> None:
    panel(img, 616, 464, 368, 240)
    filas = (("Iniciar / detener", "espacio"), ("Cambiar vista", "F"),
             ("Agregar persona", "A"), ("Exportar CSV", "E"))
    y = 502
    for etiqueta, tecla in filas:
        cv2.rectangle(img, (636, y - 9), (639, y + 9), bgr(MUTED), -1)
        pen.add(etiqueta, (652, y), 15, INK, anchor="lm")
        cap_w = len(tecla) * 9 + 18
        cap_x = 964 - cap_w
        _rounded(img, cap_x, y - 13, cap_w, 26, 6, bgr(KEYBG))
        cv2.rectangle(img, (cap_x, y - 13), (cap_x + cap_w, y + 13), bgr(BORDER), 1)
        pen.add(tecla, (cap_x + cap_w // 2, y), 13, MUTED, anchor="mm")
        y += 48


def dibujar_mensaje(img, pen: Pluma, texto: str) -> None:
    """Banner de aviso temporal centrado sobre el panel de camara."""
    cx, y = 308, 360
    w = len(texto) * 8 + 40
    _rounded(img, cx - w // 2, y, w, 34, 10, (30, 32, 36))
    pen.add(texto, (cx, y + 17), 14, (255, 255, 255), bold=True, anchor="mm")


def componer_dashboard(frame, detections, tracker, dash: Dashboard, corriendo: bool):
    """Arma el lienzo completo del dashboard y devuelve la imagen BGR."""
    lienzo = np.full((H, W, 3), bgr(BG), dtype=np.uint8)
    pen = Pluma()
    dibujar_header(lienzo, pen, dash.fps)
    dibujar_camara(lienzo, pen, frame, detections, tracker)
    dibujar_accesos(lienzo, pen, dash.accesos)
    dibujar_stats(lienzo, pen, dash)
    dibujar_controles(lienzo, pen)
    if not corriendo:
        pen.add("PAUSA", (584, 96), 13, RED, bold=True, anchor="rm")
    if dash.mensaje_timer > 0:
        dibujar_mensaje(lienzo, pen, dash.mensaje)
    return pen.render(lienzo)


def vista_camara_sola(frame, detections, tracker, dash: Dashboard, corriendo: bool):
    """Vista alternativa: solo el video con sus cajas (sin el dashboard)."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    for (x1, y1, x2, y2, conf, tid), etiqueta, col in _cajas(detections, tracker):
        c = bgr(col)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        texto = etiqueta if etiqueta == "procesando..." else f"{etiqueta} {conf:.2f}"
        cv2.putText(vis, texto, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    cv2.putText(vis, f"FPS: {dash.fps:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr(GREEN), 2)
    cv2.putText(vis, "F: dashboard", (w - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr(MUTED), 2)
    if not corriendo:
        cv2.putText(vis, "PAUSA", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr(RED), 2)
    if dash.mensaje_timer > 0:
        cv2.putText(vis, dash.mensaje, (10, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


# --------------------------------------------------------------------------- #
# Bucle principal.
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Iniciando dashboard de Control de Asistencia Facial...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    detector = FaceDetector("models/yolov8n-face.pt", conf_threshold=0.5)
    recognizer = FaceRecognizer("embeddings.pkl", threshold=0.6)
    tracker = Tracker(grace_frames=30)
    asistencia = AsistenciaDB()
    dash = Dashboard()
    prev = time.time()

    vista = "dashboard"   # "dashboard" o "camara"
    corriendo = True      # espacio pausa/reanuda la captura
    frame = None
    detections: list = []

    cv2.namedWindow("Control de acceso", cv2.WINDOW_AUTOSIZE)

    while True:
        if corriendo:
            ret, frame = cap.read()
            if not ret:
                print("Error: no se pudo capturar la imagen")
                break

            detections = detector.detect(frame)
            dash.en_cuadro = len(detections)

            for det in detections:
                tid = det[5]
                if tracker.get_name(tid) is None and not tracker.is_grace(tid):
                    crop = detector.crop_face(frame, det)
                    if crop is not None:
                        name, dist = recognizer.recognize(crop)
                        tracker.confirm(tid, name)
                        print(f"[reconocido] track {tid}: {name} ({dist:.3f})")
                        if name != "Desconocido":
                            asistencia.registrar(name.replace("_", " "))
                        dash.registrar_acceso(name)

            ahora = time.time()
            dash.fps = 1.0 / (ahora - prev) if ahora > prev else 0.0
            prev = ahora

        if frame is not None:
            if vista == "dashboard":
                salida = componer_dashboard(frame, detections, tracker, dash, corriendo)
            else:
                salida = vista_camara_sola(frame, detections, tracker, dash, corriendo)
            cv2.imshow("Control de acceso", salida)

        if dash.mensaje_timer > 0:
            dash.mensaje_timer -= 1

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):                      # ESC o q: salir
            break
        elif key == ord(" "):                          # espacio: pausar/reanudar
            corriendo = not corriendo
            if corriendo:
                prev = time.time()
        elif key in (ord("f"), ord("F")):              # F: alternar vista
            vista = "camara" if vista == "dashboard" else "dashboard"
        elif key in (ord("a"), ord("A")):              # A: agregar persona (pendiente)
            dash.avisar("Agregar persona: en construccion")
        elif key in (ord("e"), ord("E")):              # E: info de exportacion
            dash.avisar(f"Asistencia se guarda en data/{asistencia.ruta.name}")

    asistencia.cerrar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
