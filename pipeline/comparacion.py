"""06 · Comparación de resultados (Prefect @task)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from prefect import task
from pyspark.sql import functions as F
from sklearn.metrics import cohen_kappa_score

from pipeline.config import (
    FIGURAS,
    GOLD,
    MUESTRAS,
    RESULTADOS,
    TABLAS,
    asegurar_directorios,
)
from pipeline.spark_utils import crear_spark_session

import matplotlib.pyplot as plt


def comparar_metodos(a: pd.Series, b: pd.Series) -> dict:
    """Calcula conteos, índice de Jaccard y Kappa de Cohen entre dos banderas."""
    return {
        "DBSCAN": int(a.sum()),
        "K-prototypes": int(b.sum()),
        "ambos": int((a & b).sum()),
        "Jaccard": round((a & b).sum() / (a | b).sum(), 3),
        "Kappa": round(cohen_kappa_score(a, b), 3),
    }


@task(
    name="06 · Comparación de resultados",
    retries=1,
    retry_delay_seconds=10,
    tags=["secop", "spark", "resultados"],
    log_prints=True,
)
def comparacion_resultados(
    muestra_cruda_path: Path = MUESTRAS / "muestra_cruda.parquet",
    muestra_limpia_path: Path = MUESTRAS / "muestra_limpia.parquet",
    dbscan_cruda_path: Path = RESULTADOS / "dbscan_cruda.parquet",
    dbscan_limpia_path: Path = RESULTADOS / "dbscan_limpia.parquet",
    kprototypes_cruda_path: Path = RESULTADOS / "kprototypes_cruda.parquet",
    kprototypes_limpia_path: Path = RESULTADOS / "kprototypes_limpia.parquet",
    gold_path: Path = GOLD,
) -> dict[str, Path]:
    """Ejecuta el notebook 06: análisis comparativo, tablas finales y gráficas."""
    asegurar_directorios()

    muestras = {
        "cruda": pd.read_parquet(muestra_cruda_path),
        "limpia": pd.read_parquet(muestra_limpia_path),
    }
    rutas_dbscan = {"cruda": dbscan_cruda_path, "limpia": dbscan_limpia_path}
    rutas_kp = {"cruda": kprototypes_cruda_path, "limpia": kprototypes_limpia_path}

    res = {}
    for escenario, datos in muestras.items():
        datos = datos.merge(pd.read_parquet(rutas_dbscan[escenario]), on="row_id")
        datos = datos.merge(pd.read_parquet(rutas_kp[escenario]), on="row_id")
        res[escenario] = datos
    print({k: len(v) for k, v in res.items()})

    # 1. Crudo vs limpio
    filas = []
    for escenario in ["cruda", "limpia"]:
        r = res[escenario]
        filas.append(
            {
                "escenario": escenario,
                "k (K-prototypes)": r.kp_cluster.nunique(),
                "clusters DBSCAN": r.loc[r.db_cluster >= 0, "db_cluster"].nunique(),
                "% ruido DBSCAN": round(100 * r.db_ruido.mean(), 2),
                "% atípicos K-prototypes": round(100 * r.kp_atipico.mean(), 2),
            }
        )
    tabla_escenarios = pd.DataFrame(filas)
    tabla_escenarios.to_csv(TABLAS / "crudo_vs_limpio.csv", index=False)
    print(tabla_escenarios)

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(2)
    ax.bar(x - 0.2, tabla_escenarios["% ruido DBSCAN"], 0.4, label="DBSCAN (ruido)", color="#6a4c93")
    ax.bar(
        x + 0.2,
        tabla_escenarios["% atípicos K-prototypes"],
        0.4,
        label="K-prototypes",
        color="#f4a259",
    )
    ax.set_xticks(x, ["datos crudos", "datos limpios"])
    ax.set_ylabel("% de contratos atípicos")
    ax.set_title("Atípicos antes y después de limpiar")
    ax.legend()
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "atipicos_crudo_vs_limpio.png", dpi=130)
    plt.close(fig)

    # 1.1 ¿Cuántos atípicos de los datos crudos eran en realidad errores de los datos?
    cruda = res["cruda"].copy()
    cruda["motivo"] = cruda["motivo_exclusion"].fillna("válido (sigue después de limpiar)")

    origen = (
        pd.DataFrame(
            {
                "% de toda la muestra": cruda["motivo"].value_counts(normalize=True) * 100,
                "% de atípicos DBSCAN": cruda.loc[cruda.db_ruido, "motivo"].value_counts(
                    normalize=True
                )
                * 100,
                "% de atípicos K-prototypes": cruda.loc[cruda.kp_atipico, "motivo"].value_counts(
                    normalize=True
                )
                * 100,
            }
        )
        .fillna(0)
        .round(1)
        .sort_values("% de toda la muestra", ascending=False)
    )
    origen.to_csv(TABLAS / "origen_atipicos_crudos.csv")
    print(origen)

    for metodo, col in [("DBSCAN", "db_ruido"), ("K-prototypes", "kp_atipico")]:
        falsos = (cruda[col] & cruda.motivo_exclusion.notna()).sum()
        print(
            f"{metodo}: {100 * falsos / cruda[col].sum():.1f}% "
            "de los atípicos crudos eran registros que la limpieza elimina"
        )

    # 1.2 ¿Qué agrupa K-prototypes en los datos crudos?
    print(pd.crosstab(cruda["motivo"], cruda["kp_cluster"]))

    # 2. DBSCAN vs K-prototypes
    filas = []
    for escenario in ["cruda", "limpia"]:
        r = res[escenario]
        top_n = r.kp_puntaje.rank(ascending=False) <= r.db_ruido.sum()
        filas.append(
            {
                "escenario": escenario,
                "K-prototypes": "regla Q3 + 3·IQR",
                **comparar_metodos(r.db_ruido, r.kp_atipico),
            }
        )
        filas.append(
            {
                "escenario": escenario,
                "K-prototypes": "top-N (mismo N que DBSCAN)",
                **comparar_metodos(r.db_ruido, top_n),
            }
        )
    concordancia = pd.DataFrame(filas)
    concordancia.to_csv(TABLAS / "concordancia_metodos.csv", index=False)
    print(concordancia)

    # 3. Todos los contratos limpios (≈5,6 millones)
    spark = crear_spark_session(app_name="secop-resultados")

    try:
        gold = spark.read.parquet(str(gold_path))
        gold = gold.withColumn(
            "grupo",
            F.when(F.col("db_ruido") & F.col("kp_atipico"), "1. atípico en ambos")
            .when(F.col("db_ruido"), "2. solo DBSCAN")
            .when(F.col("kp_atipico"), "3. solo K-prototypes")
            .otherwise("4. normal"),
        )
        total = gold.count()

        conteo = gold.groupBy("db_ruido", "kp_atipico").count().toPandas()

        def contar(db: bool, kp: bool) -> int:
            return int(conteo.query("db_ruido == @db and kp_atipico == @kp")["count"].sum())

        ambos, solo_db, solo_kp = contar(True, True), contar(True, False), contar(False, True)
        print(f"Contratos: {total:,}")
        print(f"Ruido DBSCAN: {ambos + solo_db:,} ({100 * (ambos + solo_db) / total:.2f}%)")
        print(
            f"Atípicos K-prototypes: {ambos + solo_kp:,} "
            f"({100 * (ambos + solo_kp) / total:.2f}%)"
        )
        print(
            f"Atípicos en AMBOS: {ambos:,} ({100 * ambos / total:.2f}%)   "
            f"Jaccard = {ambos / (ambos + solo_db + solo_kp):.3f}"
        )

        # 3.1 ¿Cómo son los contratos atípicos?
        perfil = (
            gold.groupBy("grupo")
            .agg(
                F.count("*").alias("contratos"),
                F.round(F.percentile_approx("valor_contrato", 0.5) / 1e6, 1).alias(
                    "valor mediano (M COP)"
                ),
                F.percentile_approx("plazo_dias", 0.5).alias("plazo mediano (días)"),
                F.round(F.avg((F.col("ratio_adiciones") > 0.5).cast("int")) * 100, 2).alias(
                    "% con adiciones > 50%"
                ),
                F.round(F.avg((F.col("dias_firma_cargue") > 3).cast("int")) * 100, 1).alias(
                    "% publicado > 3 días después"
                ),
            )
            .orderBy("grupo")
            .toPandas()
        )
        perfil.to_csv(TABLAS / "perfil_atipicos.csv", index=False)
        print(perfil)

        def lift(variable: str, bandera: str) -> pd.DataFrame:
            t = (
                gold.groupBy(variable)
                .agg(
                    F.count("*").alias("contratos"),
                    F.sum(F.col(bandera).cast("int")).alias("atipicos"),
                )
                .toPandas()
            )
            t["% del total"] = 100 * t.contratos / t.contratos.sum()
            t["% de los atípicos"] = 100 * t.atipicos / t.atipicos.sum()
            t["lift"] = t["% de los atípicos"] / t["% del total"]
            return t[t.contratos >= 1000].sort_values("lift", ascending=False).round(2)

        print(lift("modalidad_de_contratacion", "db_ruido").head(6))
        print(lift("modalidad_de_contratacion", "kp_atipico").head(6))
        print(lift("tipo_de_contrato", "kp_atipico").head(6))

        # 3.2 Clusters de K-prototypes: ¿hay algún grupo pequeño de contratos raros?
        resumen_clusters = (
            gold.groupBy("kp_cluster")
            .agg(
                F.count("*").alias("contratos"),
                F.round(100 * F.avg(F.col("kp_atipico").cast("int")), 1).alias(
                    "% atípicos K-prototypes"
                ),
                F.round(100 * F.avg(F.col("db_ruido").cast("int")), 1).alias("% ruido DBSCAN"),
                F.round(F.percentile_approx("ratio_adiciones", 0.5), 2).alias(
                    "adiciones/valor (mediana)"
                ),
            )
            .orderBy("kp_cluster")
            .toPandas()
        )
        print(resumen_clusters)

        # 3.3 Distribución territorial
        departamentos = (
            gold.groupBy("departamento_entidad")
            .agg(
                F.count("*").alias("contratos"),
                F.sum(F.col("db_ruido").cast("int")).alias("ruido_dbscan"),
                F.sum(F.col("kp_atipico").cast("int")).alias("atipicos_kprototypes"),
                F.sum((F.col("db_ruido") & F.col("kp_atipico")).cast("int")).alias("atipicos_ambos"),
            )
            .toPandas()
        )
        for col in ["ruido_dbscan", "atipicos_kprototypes", "atipicos_ambos"]:
            departamentos["%_" + col] = (100 * departamentos[col] / departamentos.contratos).round(2)
        departamentos = departamentos.sort_values("%_atipicos_ambos", ascending=False)
        departamentos.to_csv(TABLAS / "atipicos_por_departamento.csv", index=False)  # para el mapa
        print(departamentos.head(10))

        d = departamentos[departamentos.contratos >= 1000].sort_values("%_atipicos_kprototypes")
        fig, ax = plt.subplots(figsize=(7, 8))
        y = np.arange(len(d))
        ax.barh(y - 0.2, d["%_ruido_dbscan"], 0.4, label="DBSCAN (ruido)", color="#6a4c93")
        ax.barh(y + 0.2, d["%_atipicos_kprototypes"], 0.4, label="K-prototypes", color="#f4a259")
        ax.set_yticks(y, d.departamento_entidad.str.title(), fontsize=8)
        ax.set_xlabel("% de contratos atípicos")
        ax.set_title(
            "Contratos atípicos por departamento\n(departamentos con al menos 1.000 contratos)"
        )
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGURAS / "atipicos_por_departamento.png", dpi=130)
        plt.close(fig)

        # 3.4 Lista de contratos para revisar (atípicos en ambos métodos)
        revisar = (
            gold.filter(F.col("db_ruido") & F.col("kp_atipico"))
            .orderBy(F.desc("kp_puntaje"))
            .select(
                "uid",
                "departamento_entidad",
                "nombre_entidad",
                "modalidad_de_contratacion",
                "tipo_de_contrato",
                "valor_contrato",
                "valor_adiciones",
                "plazo_dias",
                "dias_firma_cargue",
                "kp_puntaje",
                "objeto_del_contrato_a_la_firma",
                "ruta_proceso_en_secop_i",
            )
            .limit(200)
            .toPandas()
        )
        revisar.to_csv(TABLAS / "top200_atipicos_ambos_metodos.csv", index=False)
        print(
            revisar.head(10)[
                [
                    "departamento_entidad",
                    "tipo_de_contrato",
                    "valor_contrato",
                    "valor_adiciones",
                    "plazo_dias",
                    "kp_puntaje",
                ]
            ]
        )
    finally:
        spark.stop()

    return {
        "crudo_vs_limpio": TABLAS / "crudo_vs_limpio.csv",
        "origen_atipicos_crudos": TABLAS / "origen_atipicos_crudos.csv",
        "concordancia_metodos": TABLAS / "concordancia_metodos.csv",
        "perfil_atipicos": TABLAS / "perfil_atipicos.csv",
        "atipicos_por_departamento": TABLAS / "atipicos_por_departamento.csv",
        "top200_atipicos_ambos_metodos": TABLAS / "top200_atipicos_ambos_metodos.csv",
    }
