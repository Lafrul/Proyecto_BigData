"""01 · Ingesta y limpieza de SECOP I con PySpark (Prefect @task)."""

from __future__ import annotations

import csv
import re
import time
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd
from prefect import task
from pyspark.sql import Window
from pyspark.sql import functions as F

from pipeline.config import (
    BRONZE,
    CATEGORICAS,
    FIGURAS,
    MUESTRAS,
    NUMERICAS,
    SILVER,
    TABLAS,
    TAMANO_MUESTRA,
    asegurar_directorios,
    obtener_csv,
    tamano_gb,
)
from pipeline.spark_utils import crear_spark_session

import matplotlib.pyplot as plt


def nombre_columna(texto: str) -> str:
    """Normaliza un nombre de columna a snake_case sin tildes ni símbolos."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-zA-Z]+", "_", texto).strip("_").lower()


def a_numero(columna: str):
    return F.regexp_replace(F.col(columna), "[$,]", "").cast("double")


def a_fecha(columna: str):
    texto = F.trim(F.col(columna))
    return F.coalesce(
        F.to_date(texto, "dd/MM/yyyy"),
        F.to_date(F.substring(texto, 1, 10), "yyyy-MM-dd"),
    )


def log_con_signo(columna):
    return F.signum(columna) * F.log1p(F.abs(columna))


def limpiar_texto(columna: str):
    t = F.upper(F.trim(F.col(columna)))
    t = F.translate(t, "ÁÉÍÓÚÜÑ", "AEIOUUN")
    return F.when(
        t.isNull() | t.isin("", "NO DEFINIDO", "NO DEFINIDA", "NO APLICA"),
        "SIN DATO",
    ).otherwise(t)


def _parquet_valido(ruta: Path) -> bool:
    """Devuelve True si `ruta` contiene una salida Parquet completa y no vacía."""
    if not ruta.exists():
        return False
    if ruta.is_file():
        return ruta.stat().st_size > 0
    return (ruta / "_SUCCESS").exists() and tamano_gb(ruta) > 0


@task(
    name="01 · Ingesta y limpieza (Spark)",
    retries=1,
    retry_delay_seconds=10,
    tags=["secop", "spark", "etl"],
    log_prints=True,
)
def ingesta_y_limpieza_spark(
    csv_path: Path | None = None,
    bronze_path: Path = BRONZE,
    silver_path: Path = SILVER,
) -> dict[str, Path]:
    """Ejecuta el notebook 01: ingesta bronze, limpieza silver, muestras y benchmark."""
    asegurar_directorios()
    csv_path = csv_path or obtener_csv()

    print(f"{'CSV original':<16} {tamano_gb(csv_path):6.2f} GB   ({csv_path.name})")
    for nombre, ruta in [("Parquet bronze", bronze_path), ("Parquet silver", silver_path)]:
        if _parquet_valido(ruta):
            peso = tamano_gb(ruta)
            print(
                f"{nombre:<16} {peso:6.2f} GB   "
                f"({tamano_gb(csv_path) / peso:.1f} veces más pequeño que el CSV)"
            )
        else:
            print(f"{nombre:<16} todavía no existe (se crea más abajo)")

    # 1. Iniciar Spark
    spark = crear_spark_session(
        app_name="secop-limpieza",
        extra_configs={
            "spark.sql.ansi.enabled": "false",
            "spark.sql.shuffle.partitions": "48",
        },
    )

    muestra_cruda_path = MUESTRAS / "muestra_cruda.parquet"
    muestra_limpia_path = MUESTRAS / "muestra_limpia.parquet"

    try:
        # 2. Leer el CSV y guardarlo en Parquet (bronze)
        if not _parquet_valido(bronze_path):
            inicio = time.time()
            df_csv = (
                spark.read.option("header", True)
                .option("multiLine", True)
                .option("escape", '"')
                .csv(str(csv_path))
            )
            df_csv = df_csv.toDF(*[nombre_columna(c) for c in df_csv.columns])
            df_csv.repartition(48).write.mode("overwrite").parquet(str(bronze_path))
            print(f"CSV convertido a Parquet en {time.time() - inicio:.0f} segundos")
        else:
            print("El Parquet bronze ya existe, no se vuelve a leer el CSV")

        df = spark.read.parquet(str(bronze_path))
        n_total = df.count()
        print(f"{n_total:,} registros y {len(df.columns)} columnas")

        # 3. Exploración rápida
        df.select(
            "cuantia_contrato",
            "plazo_de_ejec_del_contrato",
            "rango_de_ejec_del_contrato",
            "fecha_de_firma_del_contrato",
            "moneda",
            "estado_del_proceso",
        ).show(5, truncate=False)

        df.groupBy("estado_del_proceso").count().orderBy(F.desc("count")).show(truncate=False)

        # El plazo viene en días (D) o en meses (M): "5" no significa lo mismo en los dos casos
        df.groupBy("rango_de_ejec_del_contrato").count().show()

        # Monedas distintas al peso colombiano
        df.groupBy("moneda").count().orderBy(F.desc("count")).show(6)

        # ¿Hay UIDs repetidos?
        n_uids = df.select("uid").distinct().count()
        print(f"UIDs distintos: {n_uids:,} de {n_total:,} registros -> {n_total - n_uids:,} repetidos")

        # Estimación de memoria con pandas
        muestra_pandas = pd.read_csv(csv_path, nrows=100_000, dtype=str)
        bytes_por_fila = muestra_pandas.memory_usage(deep=True).sum() / len(muestra_pandas)
        print(f"pandas usa {bytes_por_fila:,.0f} bytes por fila")
        print(f"Para {n_total:,} filas necesitaría ~{bytes_por_fila * n_total / 1e9:.1f} GB de RAM")
        del muestra_pandas

        # 4. Convertir textos a números y fechas
        df = (
            df.withColumn("row_id", F.monotonically_increasing_id())
            .withColumn("valor_contrato", a_numero("cuantia_contrato"))
            .withColumn("valor_proceso", a_numero("cuantia_proceso"))
            .withColumn("valor_adiciones", a_numero("valor_total_de_adiciones"))
            .withColumn("plazo", a_numero("plazo_de_ejec_del_contrato"))
            .withColumn("adiciones_dias_txt", a_numero("tiempo_adiciones_en_dias"))
            .withColumn("adiciones_meses_txt", a_numero("tiempo_adiciones_en_meses"))
            .withColumn("fecha_firma", a_fecha("fecha_de_firma_del_contrato"))
            .withColumn("fecha_cargue", a_fecha("fecha_de_cargue_en_el_secop"))
            .withColumn("fecha_actualizacion", a_fecha("ultima_actualizacion"))
        )
        df.select("valor_contrato", "valor_proceso", "plazo", "fecha_firma", "fecha_cargue").show(5)

        # 5. Variables del escenario CRUDO
        dias_firma_cargue_crudo = F.datediff("fecha_cargue", "fecha_firma")

        df = (
            df.withColumn("raw_log_valor_contrato", F.log1p("valor_contrato"))
            .withColumn(
                "raw_log_ratio_contrato_proceso",
                F.log((F.col("valor_contrato") + 1) / (F.col("valor_proceso") + 1)),
            )
            .withColumn("raw_ratio_adiciones", F.col("valor_adiciones") / F.col("valor_contrato"))
            .withColumn("raw_log_plazo_dias", F.log1p("plazo"))
            .withColumn("raw_log_adiciones_dias", F.log1p("adiciones_dias_txt"))
            .withColumn("raw_slog_dias_firma_cargue", log_con_signo(dias_firma_cargue_crudo))
        )
        df = df.fillna(0, subset=["raw_" + c for c in NUMERICAS])

        for c in CATEGORICAS:
            df = df.withColumn("raw_" + c, F.coalesce(F.col(c), F.lit("")))

        # 6. Limpieza y variables del escenario LIMPIO
        # 6.1 Unidades y fechas
        fecha_valida = (F.col("fecha_firma") >= F.lit("2000-01-01").cast("date")) & (
            F.col("fecha_firma") <= F.date_add("fecha_cargue", 365)
        )

        df = (
            df.withColumn(
                "plazo_dias",
                F.when(F.col("rango_de_ejec_del_contrato") == "M", F.col("plazo") * 30).when(
                    F.col("rango_de_ejec_del_contrato") == "D", F.col("plazo")
                ),
            )
            .withColumn(
                "adiciones_dias",
                F.coalesce("adiciones_dias_txt", F.lit(0))
                + 30 * F.coalesce("adiciones_meses_txt", F.lit(0)),
            )
            .withColumn(
                "dias_firma_cargue",
                F.when(fecha_valida, F.datediff("fecha_cargue", "fecha_firma")),
            )
        )
        df = (
            df.withColumn("log_valor_contrato", F.log1p("valor_contrato"))
            .withColumn(
                "log_ratio_contrato_proceso",
                F.log((F.col("valor_contrato") + 1) / (F.col("valor_proceso") + 1)),
            )
            .withColumn(
                "ratio_adiciones",
                F.coalesce("valor_adiciones", F.lit(0)) / F.col("valor_contrato"),
            )
            .withColumn("log_plazo_dias", F.log1p("plazo_dias"))
            .withColumn("log_adiciones_dias", F.log1p("adiciones_dias"))
            .withColumn("slog_dias_firma_cargue", log_con_signo(F.col("dias_firma_cargue")))
        )

        # 6.2 Textos de las categorías
        for c in CATEGORICAS + ["departamento_entidad"]:
            df = df.withColumn(c, limpiar_texto(c))

        # 6.3 Motivo de exclusión
        ventana = Window.partitionBy("uid").orderBy(F.desc("fecha_actualizacion"))
        df = df.withColumn("version", F.row_number().over(ventana))

        razon_contrato_proceso = F.col("valor_contrato") / F.col("valor_proceso")
        estados_con_contrato = ["CELEBRADO", "LIQUIDADO", "TERMINADO SIN LIQUIDAR"]

        df = df.withColumn(
            "motivo_exclusion",
            F.when(F.col("version") > 1, "duplicado")
            .when(
                ~F.col("estado_del_proceso").isin(estados_con_contrato)
                | F.col("fecha_firma").isNull(),
                "sin contrato firmado",
            )
            .when(F.upper(F.trim("moneda")) != "PESO COLOMBIANO", "otra moneda")
            .when(
                F.col("valor_contrato").isNull() | (F.col("valor_contrato") < 10_000),
                "error de captura: valor",
            )
            .when(
                (razon_contrato_proceso >= 100) | (razon_contrato_proceso <= 0.01),
                "error de captura: valor",
            )
            .when(
                F.col("plazo_dias").isNull() | (F.col("plazo_dias") > 365 * 20),
                "error de captura: plazo",
            )
            .when(F.col("ratio_adiciones") > 10, "error de captura: adiciones")
            .when(
                F.col("dias_firma_cargue").isNull()
                | (F.col("dias_firma_cargue") < -365 * 2)
                | (F.col("dias_firma_cargue") > 365 * 5),
                "error de captura: fecha",
            ),
        )

        # 7. Guardar la tabla silver
        if (
            "objeto_del_contrato_a_la_firma" not in df.columns
            and "objeto_del_contrato_a_la" in df.columns
        ):
            df = df.withColumn(
                "objeto_del_contrato_a_la_firma", F.col("objeto_del_contrato_a_la")
            )

        columnas = (
            [
                "row_id",
                "uid",
                "motivo_exclusion",
                "nombre_entidad",
                "departamento_entidad",
                "municipio_entidad",
                "anno_cargue_secop",
                "objeto_del_contrato_a_la_firma",
                "nom_razon_social_contratista",
                "ruta_proceso_en_secop_i",
                "valor_contrato",
                "valor_proceso",
                "valor_adiciones",
                "plazo_dias",
                "adiciones_dias",
                "dias_firma_cargue",
            ]
            + NUMERICAS
            + CATEGORICAS
            + ["raw_" + c for c in NUMERICAS + CATEGORICAS]
        )

        inicio = time.time()
        df.select(columnas).write.mode("overwrite").parquet(str(silver_path))
        print(f"Silver guardado en {time.time() - inicio:.0f} segundos")

        silver = spark.read.parquet(str(silver_path))

        motivos = silver.groupBy("motivo_exclusion").count().orderBy(F.desc("count")).toPandas()
        motivos["motivo_exclusion"] = motivos["motivo_exclusion"].fillna("(válido: escenario limpio)")
        motivos["porcentaje"] = (100 * motivos["count"] / motivos["count"].sum()).round(2)
        motivos.to_csv(TABLAS / "motivos_exclusion.csv", index=False)
        print(motivos)

        fig, ax = plt.subplots(figsize=(7, 3))
        otros = motivos[motivos["motivo_exclusion"] != "(válido: escenario limpio)"]
        ax.barh(otros["motivo_exclusion"], otros["porcentaje"], color="#c8553d")
        ax.invert_yaxis()
        ax.set_xlabel("% de los registros")
        ax.set_title("Registros que no entran al escenario limpio")
        for i, v in enumerate(otros["porcentaje"]):
            ax.text(v, i, f" {v:.2f}%", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURAS / "motivos_exclusion.png", dpi=130)
        plt.close(fig)

        # 8. Muestras para los modelos
        n_limpios = silver.filter(F.col("motivo_exclusion").isNull()).count()
        if n_limpios == 0:
            raise ValueError(
                "No quedaron registros válidos en el escenario limpio (n_limpios == 0). "
                "Revisa las reglas de limpieza y el formato de fechas/columnas."
            )

        muestra_cruda = (
            silver.sample(fraction=min(1.0, 1.2 * TAMANO_MUESTRA / n_total), seed=42)
            .limit(TAMANO_MUESTRA)
            .toPandas()
        )
        muestra_limpia = (
            silver.filter(F.col("motivo_exclusion").isNull())
            .sample(fraction=min(1.0, 1.2 * TAMANO_MUESTRA / n_limpios), seed=42)
            .limit(TAMANO_MUESTRA)
            .toPandas()
        )

        muestra_cruda.to_parquet(muestra_cruda_path)
        muestra_limpia.to_parquet(muestra_limpia_path)
        print(f"Registros válidos: {n_limpios:,} de {n_total:,} ({100 * n_limpios / n_total:.1f}%)")
        print("Muestras guardadas:", len(muestra_cruda), len(muestra_limpia))

        # 9. Comparación con DuckDB
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
            encabezados_csv = next(csv.reader(f))
        col_modalidad_csv = next(
            (c for c in encabezados_csv if nombre_columna(c) == "modalidad_de_contratacion"),
            "Modalidad de Contratacion",
        )
        col_modalidad_sql = col_modalidad_csv.replace('"', '""')

        duckdb.sql("SET enable_progress_bar = false")
        csv_sql = csv_path.as_posix().replace("'", "''")

        inicio = time.time()
        duckdb.sql(
            f"""SELECT "{col_modalidad_sql}", count(*) FROM read_csv('{csv_sql}', all_varchar=true)
                GROUP BY 1"""
        ).fetchall()
        t_duckdb = time.time() - inicio

        inicio = time.time()
        csv_spark = (
            spark.read.option("header", True)
            .option("multiLine", True)
            .option("escape", '"')
            .csv(str(csv_path))
        )
        csv_spark.groupBy(col_modalidad_csv).count().collect()
        t_spark_csv = time.time() - inicio

        inicio = time.time()
        spark.read.parquet(str(bronze_path)).groupBy("modalidad_de_contratacion").count().collect()
        t_spark_parquet = time.time() - inicio

        comparacion = pd.DataFrame(
            {
                "motor": ["DuckDB sobre CSV", "Spark sobre CSV (multiLine)", "Spark sobre Parquet"],
                "segundos": [round(t_duckdb, 1), round(t_spark_csv, 1), round(t_spark_parquet, 1)],
                "particiones": [
                    "-",
                    csv_spark.rdd.getNumPartitions(),
                    spark.read.parquet(str(bronze_path)).rdd.getNumPartitions(),
                ],
            }
        )
        comparacion.to_csv(TABLAS / "comparacion_motores.csv", index=False)
        print(comparacion)
    finally:
        spark.stop()

    return {
        "bronze": bronze_path,
        "silver": silver_path,
        "muestra_cruda": muestra_cruda_path,
        "muestra_limpia": muestra_limpia_path,
    }
