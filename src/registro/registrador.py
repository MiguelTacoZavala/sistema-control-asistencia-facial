"""Módulo de registro de asistencia."""

from datetime import datetime


class Registrador:
    """Registra la asistencia de empleados en un archivo CSV."""

    def __init__(self, ruta_salida: str) -> None:
        self.ruta_salida = ruta_salida

    def registrar_asistencia(self, empleado_id: str) -> None:
        pass

    def _obtener_fecha_hora(self) -> str:
        return datetime.now().isoformat()
