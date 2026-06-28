"""Cache de nombres por ID de tracking con periodo de gracia.

Cada rostro detectado recibe un track_id unico (asignado por ByteTrack
de Ultralytics). El proceso tiene dos fases:

1.  Periodo de gracia (grace_frames):
    No se reconoce, solo se muestra "Reconociendo...".

2.  Confirmado:
    Al expirar la gracia se reconoce una vez. Si el rostro coincide
    con alguien de la base, se cachea el nombre. Si no, queda como
    "Desconocido" permanentemente (para ese track_id).
"""


class Tracker:
    """Asocia nombres confirmados a IDs de tracking.

    Args:
        grace_frames: Cuantos frames esperar sin reconocer al aparecer
            un rostro nuevo (periodo de gracia).
    """

    def __init__(self, grace_frames: int = 15) -> None:
        self.grace_frames = grace_frames
        self._graces: dict[int, int] = {}
        self._confirmed: dict[int, str] = {}

    def get_name(self, track_id: int) -> str | None:
        """Devuelve el nombre confirmado para un track_id.

        Args:
            track_id: ID del track.

        Returns:
            Nombre confirmado o None si aun no se decidio.
        """
        return self._confirmed.get(track_id)

    def is_grace(self, track_id: int) -> bool:
        """Verifica si un track aun esta en periodo de gracia.

        La primera vez que se llama para un track_id nuevo, inicia el
        contador de grace_frames. Cada llamado posterior lo decrementa
        hasta llegar a 0.

        Args:
            track_id: ID del track.

        Returns:
            True  = aun en periodo de gracia.
            False = gracia expirada (listo para reconocer).
        """
        if track_id in self._confirmed:
            return False
        if track_id not in self._graces:
            self._graces[track_id] = self.grace_frames
        self._graces[track_id] -= 1
        return self._graces[track_id] > 0

    def confirm(self, track_id: int, name: str) -> None:
        """Cachea el nombre de un track_id como confirmado.

        Args:
            track_id: ID del track.
            name: Nombre a cachear.
        """
        self._confirmed[track_id] = name
