"""Configuración global, rutas, constantes y utilidades base del pipeline SECOP I."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend no interactivo para guardar figuras sin bloquear la ejecución

# Rutas del proyecto (config.py vive en `<RAIZ>/pipeline/config.py`)
RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
MUESTRAS = DATA / "muestras"
RESULTADOS = RAIZ / "resultados"
MODELOS = RESULTADOS / "modelos"
FIGURAS = RESULTADOS / "figuras"
TABLAS = RESULTADOS / "tablas"

BRONZE = DATA / "bronze" / "secop_i.parquet"
SILVER = DATA / "silver" / "secop_i_limpio.parquet"
GOLD = DATA / "gold" / "contratos_puntuados.parquet"

# Variables del estudio
NUMERICAS = [
    "log_valor_contrato",          # log(1 + valor del contrato)
    "log_ratio_contrato_proceso",  # log(valor del contrato / valor estimado del proceso)
    "ratio_adiciones",             # valor adicionado / valor del contrato
    "log_plazo_dias",              # log(1 + plazo en días)
    "log_adiciones_dias",          # log(1 + días adicionados)
    "slog_dias_firma_cargue",      # log "con signo" de los días entre la firma y la publicación
]
CATEGORICAS = [
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "orden_entidad",
    "nombre_grupo",
    "tipo_identifi_del_contratista",
    "destino_gasto",
    "estado_del_proceso",
]

# Parámetros globales compartidos entre etapas
TAMANO_MUESTRA = 50_000
GAMMA = 0.5
N_NUM = len(NUMERICAS)
MIN_SAMPLES = 2 * len(NUMERICAS + CATEGORICAS)


def asegurar_directorios() -> None:
    """Crea las carpetas de salida requeridas por el pipeline si no existen."""
    for carpeta in [MUESTRAS, MODELOS, FIGURAS, TABLAS]:
        carpeta.mkdir(parents=True, exist_ok=True)


def obtener_csv() -> Path:
    """Localiza el CSV original de SECOP I en `data/` (o en la raíz como respaldo)."""
    archivos = sorted(DATA.glob("SECOP_I_*.csv"))
    if not archivos:
        archivos = sorted(RAIZ.glob("SECOP_I_*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontró ningún archivo 'SECOP_I_*.csv' en {DATA} ni en {RAIZ}."
        )
    return archivos[0]


def tamano_gb(ruta: Path) -> float:
    """Devuelve el tamaño en gigabytes (base 10) de un archivo o directorio."""
    if ruta.is_file():
        return ruta.stat().st_size / 1e9
    return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file()) / 1e9
