"""Paquete modular con las 6 etapas del pipeline SECOP I."""

from pipeline.comparacion import comparacion_resultados
from pipeline.config import asegurar_directorios
from pipeline.dbscan import ejecutar_dbscan
from pipeline.ingesta import ingesta_y_limpieza_spark
from pipeline.kprototypes import ejecutar_kprototypes
from pipeline.poblacion import aplicar_modelos_poblacion_spark
from pipeline.preparacion import preparacion_datos

__all__ = [
    "asegurar_directorios",
    "ingesta_y_limpieza_spark",
    "preparacion_datos",
    "ejecutar_dbscan",
    "ejecutar_kprototypes",
    "aplicar_modelos_poblacion_spark",
    "comparacion_resultados",
]
