"""Cache de nombres por ID de tracking con periodo de gracia y votacion.

Cada rostro detectado recibe un track_id unico (asignado por ByteTrack
de Ultralytics). El proceso para confirmar un nombre tiene tres fases:

1.  Periodo de gracia (delay_frames):
    No se reconoce, solo se muestra "Reconociendo...".

2.  Votacion (min_frames):
    Se reconoce cada frame y se acumulan los resultados.

3.  Confirmado:
    Si la mayoria supera min_ratio, el nombre se cachea y ya no se
    reconoce mas. Si no hay consenso, se reinicia la votacion.
"""
from collections import Counter

# Estados internos de cada track
_DELAY = 0
_VOTING = 1
_CONFIRMED = 2


class Tracker:
    """Asocia nombres confirmados a IDs de tracking.

    Args:
        delay_frames: Cuantos frames esperar sin reconocer al aparecer
            un rostro nuevo (periodo de gracia).
        min_frames: Cuantos frames votar antes de decidir.
        min_ratio: Fraccion minima de acuerdo para confirmar (0.0 a 1.0).
    """

    def __init__(
        self,
        delay_frames: int = 15,
        min_frames: int = 15,
        min_ratio: float = 0.6,
    ) -> None:
        self.delay_frames = delay_frames
        self.min_frames = min_frames
        self.min_ratio = min_ratio
        self._states: dict[int, int] = {}
        self._delays: dict[int, int] = {}
        self._buffers: dict[int, list[str]] = {}
        self._confirmed: dict[int, str] = {}

    def get_name(self, track_id: int) -> str | None:
        """Devuelve el nombre confirmado para un track_id.

        Args:
            track_id: ID del track.

        Returns:
            Nombre confirmado o None si aun no se decidio.
        """
        return self._confirmed.get(track_id)

    def tick(self, track_id: int) -> bool:
        """Avanza el periodo de gracia de un track.

        La primera vez que se llama para un track_id nuevo, inicia el
        contador de delay_frames. Cada llamado posterior lo decrementa
        hasta llegar a 0, momento en que el track pasa a estado VOTING.

        Returns:
            True  = periodo de gracia terminado (listo para votar).
            False = aun en periodo de gracia.
        """
        estado = self._states.get(track_id)

        if estado == _CONFIRMED:
            return True

        if estado is None:
            self._states[track_id] = _DELAY
            self._delays[track_id] = self.delay_frames
            estado = _DELAY

        if estado == _DELAY:
            self._delays[track_id] -= 1
            if self._delays[track_id] <= 0:
                self._states[track_id] = _VOTING
                del self._delays[track_id]
                return True
            return False

        # Estado VOTING: delay ya termino antes
        return True

    def vote(self, track_id: int, name: str) -> str | None:
        """Agrega un voto al buffer y confirma si hay mayoria.

        Args:
            track_id: ID del track.
            name: Nombre predicho por el reconocedor en este frame.

        Returns:
            Nombre confirmado si se alcanzo la mayoria necesaria,
            None si la votacion aun no termina o no hay consenso.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = []
        self._buffers[track_id].append(name)

        if len(self._buffers[track_id]) >= self.min_frames:
            counter = Counter(self._buffers[track_id])
            candidate, count = counter.most_common(1)[0]
            ratio = count / len(self._buffers[track_id])

            if ratio >= self.min_ratio:
                self._confirmed[track_id] = candidate
                self._states[track_id] = _CONFIRMED
                del self._buffers[track_id]
                return candidate

            # Sin consenso suficiente: reiniciar buffer y seguir votando
            self._buffers[track_id] = []

        return None
