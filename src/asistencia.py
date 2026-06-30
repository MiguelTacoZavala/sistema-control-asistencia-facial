"""Registro de asistencia en CSV diario.

Cada ejecucion del sistema crea o abre un archivo data/asistencia_YYYY-MM-DD.csv.
Por cada persona reconocida se escribe una fila con id, trabajador y hora.
Una misma persona solo se registra una vez por dia.
"""

import csv
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATA = RAIZ / "data"


class AsistenciaDB:

    def __init__(self) -> None:
        CARPETA_DATA.mkdir(exist_ok=True)
        hoy = datetime.now().strftime("%Y-%m-%d")
        self.ruta = CARPETA_DATA / f"asistencia_{hoy}.csv"
        self._archivo: Path | None = None
        self._ultimo_id = 0
        self._registrados_hoy: set[str] = set()
        self._iniciar()

    def _iniciar(self) -> None:
        if self.ruta.exists():
            with open(self.ruta, "r", encoding="utf-8") as f:
                lector = csv.reader(f)
                filas = list(lector)
            if len(filas) > 1:
                self._ultimo_id = int(filas[-1][0])
                for fila in filas[1:]:
                    self._registrados_hoy.add(fila[1].strip().lower())
        else:
            with open(self.ruta, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["id", "trabajador", "hora"])

    def registrar(self, nombre_completo: str) -> None:
        clave = nombre_completo.strip().lower()
        if clave in self._registrados_hoy:
            return
        self._ultimo_id += 1
        hora = datetime.now().strftime("%H:%M:%S")
        with open(self.ruta, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self._ultimo_id, nombre_completo.strip(), hora])
        self._registrados_hoy.add(clave)

    def cerrar(self) -> None:
        pass
