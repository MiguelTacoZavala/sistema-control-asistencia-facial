"""Validación de calidad de la base de embeddings (embeddings.pkl).

Reporta:
  - Cantidad de embeddings por persona.
  - Dimensión de cada embedding (verifica que todos sean 128).
  - Distancia promedio intra-persona (qué tan parecidos son entre sí
    los embeddings de la misma persona).
  - Distancia promedio inter-persona (qué tan distintos son los
    embeddings entre personas diferentes).
  - Evaluación automática: intra < inter para que el sistema sea confiable.

Uso:
    python test/validar_embeddings.py
    python test/validar_embeddings.py --pkl ruta/a/embeddings.pkl
"""

import argparse
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

RAIZ = Path(__file__).resolve().parent.parent
RUTA_EMBEDDINGS_DEFAULT = RAIZ / "embeddings.pkl"

DIMENSION_ESPERADA = 128      # dlib ResNet face_recognition produce 128 floats
UMBRAL_RECONOCIMIENTO = 0.6   # umbral estándar de face_recognition
SEPARACION_MINIMA = 0.10      # margen mínimo intra vs inter para considerar OK


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def cargar_embeddings(ruta: Path) -> dict[str, list[np.ndarray]]:
    """Carga embeddings.pkl y lo devuelve como diccionario."""
    if not ruta.exists():
        print(f"[ERROR] No se encontró el archivo: {ruta}")
        print("  Ejecuta primero: python src/embedding_db.py")
        sys.exit(1)

    with open(ruta, "rb") as f:
        db = pickle.load(f)

    if not isinstance(db, dict):
        print("[ERROR] El archivo no contiene un diccionario válido.")
        sys.exit(1)

    return db


def distancias_intra(embeddings: list[np.ndarray]) -> list[float]:
    """Calcula todas las distancias euclidianas entre pares de embeddings
    de la misma persona (combinaciones sin repetición)."""
    return [
        float(np.linalg.norm(e1 - e2))
        for e1, e2 in combinations(embeddings, 2)
    ]


def distancias_inter(embs_a: list[np.ndarray], embs_b: list[np.ndarray]) -> list[float]:
    """Calcula todas las distancias euclidianas entre embeddings de dos
    personas distintas (producto cartesiano)."""
    return [
        float(np.linalg.norm(e1 - e2))
        for e1 in embs_a
        for e2 in embs_b
    ]


def glosario() -> None:
    """Imprime una explicación de cada métrica usada en el reporte."""
    separador("GLOSARIO DE MÉTRICAS")
    print("""
  EMBEDDING
  ─────────
  Un embedding facial es un vector de 128 números (floats) que representa
  el "mapa biométrico" de un rostro. Lo genera la librería face_recognition
  usando una red neuronal ResNet entrenada por dlib. La idea central es que
  dos fotos de la misma persona producen vectores muy parecidos (cercanos),
  y fotos de personas distintas producen vectores muy diferentes (lejanos).

  DIMENSIÓN DEL EMBEDDING
  ───────────────────────
  Todos los embeddings deben tener exactamente 128 componentes. Si alguno
  tiene más o menos, significa que hubo un error al generarlo y ese embedding
  no es comparable con los demás. El script marca esto como ERROR.

  DISTANCIA EUCLIDIANA
  ────────────────────
  Es la forma en que comparamos dos embeddings. Se calcula como:

      distancia = sqrt( sum( (a_i - b_i)^2 ) )   para i = 1..128

  Un valor de 0.0 significaría que los dos vectores son idénticos (mismo
  rostro, misma imagen). Valores típicos:
    · < 0.4  →  muy parecidos (probablemente la misma persona)
    · 0.4 – 0.6  →  zona gris
    · > 0.6  →  personas distintas  (umbral recomendado por face_recognition)

  DISTANCIA INTRA-PERSONA
  ───────────────────────
  Promedio de distancias entre todos los pares de embeddings de UNA MISMA
  persona. Mide qué tan consistentes son las fotos de esa persona entre sí.

    · Valor BAJO (< 0.4)  →  las fotos son homogéneas y el modelo las
      agrupa correctamente. Esto es lo que queremos.
    · Valor ALTO (≥ 0.6)  →  las fotos de esa persona varían demasiado.
      Puede deberse a iluminación extrema, accesorios, ángulos, o fotos
      donde YOLO recortó mal el rostro.

  DISTANCIA INTER-PERSONA
  ───────────────────────
  Promedio de distancias entre todos los embeddings de DOS PERSONAS
  DISTINTAS. Mide qué tan separados están los grupos en el espacio de
  características.

    · Valor ALTO (> 0.6)  →  las personas son claramente distinguibles
      para el modelo. Esto es lo que queremos.
    · Valor BAJO (< 0.4)  →  el modelo confunde a estas dos personas.
      Puede indicar parecido físico, poca variedad de fotos, o un
      problema con el modelo.

  DESVIACIÓN ESTÁNDAR (std)
  ─────────────────────────
  Indica qué tan dispersas están las distancias respecto al promedio.
  Un std alto significa que hay fotos muy consistentes y otras muy
  problemáticas mezcladas. Un std bajo significa que todas las fotos
  se comportan de forma parecida.

  MARGEN DE SEPARACIÓN
  ────────────────────
  Es la diferencia:  margen = inter_promedio - intra_promedio

    · Margen POSITIVO y grande  →  el sistema distingue bien a la persona.
    · Margen POSITIVO pero pequeño (< 0.10)  →  funciona, pero con poco
      margen de seguridad. Podría fallar en condiciones difíciles.
    · Margen NEGATIVO  →  PROBLEMA GRAVE: los embeddings de la misma persona
      son más distintos entre sí que respecto a otra persona. El sistema
      reconocería mal a esa persona.

  UMBRAL DE RECONOCIMIENTO
  ────────────────────────
  Valor de corte (por defecto 0.6) usado en recognizer.py para decidir si
  un rostro detectado coincide con alguien de la base:
    · distancia < umbral  →  persona reconocida
    · distancia ≥ umbral  →  etiquetado como "Desconocido"
""")


def color_ok(texto: str) -> str:
    return f"\033[92m{texto}\033[0m"   # verde


def color_warn(texto: str) -> str:
    return f"\033[93m{texto}\033[0m"   # amarillo


def color_err(texto: str) -> str:
    return f"\033[91m{texto}\033[0m"   # rojo


def separador(titulo: str = "", ancho: int = 60) -> None:
    if titulo:
        lado = (ancho - len(titulo) - 2) // 2
        print(f"\n{'─' * lado} {titulo} {'─' * lado}")
    else:
        print("─" * ancho)


# ──────────────────────────────────────────────────────────────────────────────
# Validaciones
# ──────────────────────────────────────────────────────────────────────────────

def validar_dimensiones(db: dict) -> bool:
    """Verifica que todos los embeddings tengan la dimensión esperada."""
    separador("DIMENSIONES Y CONTEO")
    todo_ok = True

    for persona, embs in db.items():
        dims = [len(e) for e in embs]
        incorrectos = [d for d in dims if d != DIMENSION_ESPERADA]

        estado = color_ok("OK") if not incorrectos else color_err("ERROR")
        print(
            f"  [{estado}] {persona:<12} "
            f"embeddings: {len(embs):>3}   "
            f"dimensión: {set(dims)}"
        )

        if incorrectos:
            print(
                f"         {color_err('⚠')}  "
                f"{len(incorrectos)} embedding(s) no tienen {DIMENSION_ESPERADA} dimensiones."
            )
            todo_ok = False

    return todo_ok


def calcular_intra(db: dict) -> dict[str, dict]:
    """Calcula estadísticas de distancia intra-persona para cada persona."""
    separador("DISTANCIAS INTRA-PERSONA")
    print("  (qué tan parecidos son los embeddings de la misma persona)\n")

    resultados = {}
    for persona, embs in db.items():
        if len(embs) < 2:
            print(
                f"  {color_warn('SKIP')} {persona:<12} "
                f"— solo {len(embs)} embedding, se necesitan ≥2 para calcular."
            )
            resultados[persona] = None
            continue

        dists = distancias_intra(embs)
        stats = {
            "pares": len(dists),
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
        }
        resultados[persona] = stats

        alerta = ""
        if stats["mean"] >= UMBRAL_RECONOCIMIENTO:
            alerta = color_warn(f"  ⚠ promedio ≥ umbral ({UMBRAL_RECONOCIMIENTO})")

        print(
            f"  {persona:<12} "
            f"pares: {stats['pares']:>3}  "
            f"promedio: {stats['mean']:.4f}  "
            f"std: {stats['std']:.4f}  "
            f"[{stats['min']:.4f} – {stats['max']:.4f}]"
            f"{alerta}"
        )

    return resultados


def calcular_inter(db: dict) -> dict[tuple, dict]:
    """Calcula estadísticas de distancia inter-persona para cada par."""
    separador("DISTANCIAS INTER-PERSONA")
    print("  (qué tan distintos son los embeddings entre personas diferentes)\n")

    resultados = {}
    personas = list(db.keys())
    for p1, p2 in combinations(personas, 2):
        dists = distancias_inter(db[p1], db[p2])
        stats = {
            "comparaciones": len(dists),
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
        }
        resultados[(p1, p2)] = stats

        print(
            f"  {p1} vs {p2:<12} "
            f"comparaciones: {stats['comparaciones']:>4}  "
            f"promedio: {stats['mean']:.4f}  "
            f"std: {stats['std']:.4f}  "
            f"[{stats['min']:.4f} – {stats['max']:.4f}]"
        )

    return resultados


def evaluar(
    intra: dict[str, dict | None],
    inter: dict[tuple, dict],
) -> bool:
    """Evalúa si intra < inter para todos los integrantes y emite veredicto."""
    separador("EVALUACIÓN FINAL")
    todo_ok = True
    problemas = []

    for persona, stats_intra in intra.items():
        if stats_intra is None:
            continue

        # Recopilar todas las distancias inter donde aparece esta persona
        dists_inter_persona = []
        for (p1, p2), stats_i in inter.items():
            if persona in (p1, p2):
                dists_inter_persona.append(stats_i["mean"])

        if not dists_inter_persona:
            continue

        inter_promedio = float(np.mean(dists_inter_persona))
        intra_promedio = stats_intra["mean"]
        margen = inter_promedio - intra_promedio

        if margen < SEPARACION_MINIMA:
            estado = color_err("FALLO")
            todo_ok = False
            problemas.append(persona)
        else:
            estado = color_ok("OK")

        print(
            f"  [{estado}] {persona:<12} "
            f"intra={intra_promedio:.4f}  "
            f"inter={inter_promedio:.4f}  "
            f"margen={margen:+.4f}"
        )

    separador()
    if todo_ok:
        print(color_ok("\n  ✔  Validación exitosa."))
        print(
            "     Las distancias intra-persona son significativamente "
            "menores que las inter-persona."
        )
        print(
            "     El sistema debería reconocer correctamente a los "
            f"tres integrantes con umbral={UMBRAL_RECONOCIMIENTO}.\n"
        )
    else:
        print(color_err("\n  ✘  Validación con problemas."))
        print(
            f"     Las siguientes personas tienen margen insuficiente "
            f"(< {SEPARACION_MINIMA}): {', '.join(problemas)}"
        )
        print()
        print("  POSIBLES CAUSAS:")
        print("    · Fotos con ángulos muy extremos o iluminación deficiente.")
        print("    · Accesorios (gorras, lentes) que alteran mucho la cara.")
        print("    · Imágenes donde YOLO recortó mal el rostro → revisar fallidos.txt.")
        print("    · Poca variedad de fotos por persona (mínimo recomendado: 10).")
        print()
        print("  ACCIONES SUGERIDAS:")
        print("    1. Revisar fallidos.txt para ver qué fotos fueron problemáticas.")
        print("    2. Retomar fotos en condiciones más controladas.")
        print(
            "    3. Considerar bajar el umbral de reconocimiento "
            f"(actualmente: {UMBRAL_RECONOCIMIENTO})."
        )
        print("    4. Volver a ejecutar: python src/embedding_db.py\n")

    return todo_ok


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida la calidad de la base de embeddings faciales."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=RUTA_EMBEDDINGS_DEFAULT,
        help="Ruta al archivo embeddings.pkl (por defecto: raíz del proyecto).",
    )
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print("  VALIDACIÓN DE BASE DE EMBEDDINGS FACIALES")
    print(f"{'═' * 60}")
    print(f"  Archivo: {args.pkl}")

    # 0. Mostrar glosario de métricas
    glosario()

    # 1. Cargar
    db = cargar_embeddings(args.pkl)
    print(f"  Personas en la base: {len(db)} ({', '.join(db.keys())})")

    if len(db) < 2:
        print(
            color_warn(
                "\n[ADVERTENCIA] Se necesitan al menos 2 personas en la base "
                "para calcular distancias inter-persona."
            )
        )

    # 2. Validar dimensiones
    dims_ok = validar_dimensiones(db)

    if not dims_ok:
        print(color_err("\n[ERROR] Hay embeddings con dimensiones incorrectas. Deteniéndose."))
        sys.exit(1)

    # 3. Distancias intra-persona
    stats_intra = calcular_intra(db)

    # 4. Distancias inter-persona (solo si hay ≥2 personas)
    if len(db) >= 2:
        stats_inter = calcular_inter(db)
    else:
        stats_inter = {}
        print(color_warn("\n  Sin suficientes personas para calcular distancias inter."))

    # 5. Evaluación final
    if stats_inter:
        evaluar(stats_intra, stats_inter)
    else:
        print(color_warn("\n  Evaluación final omitida (se necesitan ≥2 personas)."))


if __name__ == "__main__":
    main()
