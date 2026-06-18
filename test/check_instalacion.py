"""Verifica que todas las dependencias del proyecto están instaladas."""

import importlib
import sys

DEPENDENCIAS = [
    ("ultralytics", "ultralytics"),
    ("cv2", "opencv-python"),
    ("face_recognition", "face-recognition"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("tqdm", "tqdm"),
]

errores = 0

print(f"Python {sys.version}")
print("Verificando dependencias...\n")

for mod, nombre in DEPENDENCIAS:
    try:
        importlib.import_module(mod)
        print(f"  [OK] {nombre}")
    except ImportError:
        print(f"  [FAIL] {nombre} — No instalado")
        errores += 1

print(f"\n{len(DEPENDENCIAS) - errores}/{len(DEPENDENCIAS)} dependencias OK")

if errores:
    print(f"Faltan {errores} dependencia(s). Ejecuta: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("Todo correcto.")
