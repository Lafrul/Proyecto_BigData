"""Punto de entrada y flujo Prefect del pipeline de detección de atípicos en SECOP I.

Orquesta en el mismo orden las 6 tareas definidas originalmente en los notebooks:
    01. Ingesta y limpieza con PySpark       (`pipeline.ingesta`)
    02. Preparación de datos                 (`pipeline.preparacion`)
    03. DBSCAN                               (`pipeline.dbscan`)
    04. K-prototypes                         (`pipeline.kprototypes`)
    05. Aplicación a la población con Spark  (`pipeline.poblacion`)
    06. Comparación de resultados            (`pipeline.comparacion`)
"""

from __future__ import annotations

from pathlib import Path

from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

from pipeline import (
    aplicar_modelos_poblacion_spark,
    asegurar_directorios,
    comparacion_resultados,
    ejecutar_dbscan,
    ejecutar_kprototypes,
    ingesta_y_limpieza_spark,
    preparacion_datos,
)
from pipeline.config import BRONZE, GOLD, SILVER


def _publicar_resumen_artefactos(salidas: dict[str, Path]) -> None:
    """Publica un resumen Markdown de los artefactos generados en la UI de Prefect."""
    filas = "\n".join(f"| `{clave}` | `{ruta}` |" for clave, ruta in salidas.items())
    markdown = (
        "# Resumen del pipeline SECOP I\n\n"
        "| Artefacto | Ruta |\n"
        "| --- | --- |\n"
        f"{filas}\n"
    )
    create_markdown_artifact(
        key="resumen-pipeline-secop-i",
        markdown=markdown,
        description="Rutas de datos, modelos y tablas generados por el flujo SECOP I.",
    )


@flow(
    name="secop-i-deteccion-atipicos",
    description=(
        "Pipeline SECOP I: ingesta y limpieza Spark, preparación de datos, "
        "DBSCAN, K-prototypes, scoring sobre la población completa y comparación."
    ),
    log_prints=True,
)
def ejecutar_pipeline(
    csv_path: Path | None = None,
    bronze_path: Path = BRONZE,
    silver_path: Path = SILVER,
    gold_path: Path = GOLD,
) -> dict[str, Path]:
    """Ejecuta las 6 etapas del proyecto en el mismo orden que los notebooks."""
    logger = get_run_logger()
    logger.info("Iniciando flujo Prefect: detección de contratos atípicos SECOP I")
    asegurar_directorios()

    salidas_ingesta = ingesta_y_limpieza_spark(
        csv_path=csv_path,
        bronze_path=bronze_path,
        silver_path=silver_path,
    )
    salidas_preparacion = preparacion_datos(
        muestra_cruda_path=salidas_ingesta["muestra_cruda"],
        muestra_limpia_path=salidas_ingesta["muestra_limpia"],
    )
    salidas_dbscan = ejecutar_dbscan(
        prep_cruda_path=salidas_preparacion["preparacion_cruda"],
        prep_limpia_path=salidas_preparacion["preparacion_limpia"],
    )
    salidas_kprototypes = ejecutar_kprototypes(
        prep_cruda_path=salidas_preparacion["preparacion_cruda"],
        prep_limpia_path=salidas_preparacion["preparacion_limpia"],
    )
    salidas_poblacion = aplicar_modelos_poblacion_spark(
        silver_path=salidas_ingesta["silver"],
        prep_limpia_path=salidas_preparacion["preparacion_limpia"],
        kprototypes_limpio_path=salidas_kprototypes["modelo_limpio"],
        dbscan_limpio_path=salidas_dbscan["modelo_limpio"],
        gold_path=gold_path,
    )
    salidas_comparacion = comparacion_resultados(
        muestra_cruda_path=salidas_ingesta["muestra_cruda"],
        muestra_limpia_path=salidas_ingesta["muestra_limpia"],
        dbscan_cruda_path=salidas_dbscan["dbscan_cruda"],
        dbscan_limpia_path=salidas_dbscan["dbscan_limpia"],
        kprototypes_cruda_path=salidas_kprototypes["kprototypes_cruda"],
        kprototypes_limpia_path=salidas_kprototypes["kprototypes_limpia"],
        gold_path=salidas_poblacion["gold"],
    )

    salidas = {
        **salidas_ingesta,
        **salidas_preparacion,
        **salidas_dbscan,
        **salidas_kprototypes,
        **salidas_poblacion,
        **salidas_comparacion,
    }
    _publicar_resumen_artefactos(salidas)
    logger.info("Flujo completado correctamente con %d artefactos registrados.", len(salidas))
    return salidas


if __name__ == "__main__":
    ejecutar_pipeline()
