"""05 · Aplicar los modelos a todos los contratos con Spark (Prefect @task)."""

from __future__ import annotations

import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from prefect import task
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, FloatType, IntegerType, StructField, StructType
from sklearn.neighbors import NearestNeighbors

from pipeline.config import (
    CATEGORICAS,
    GOLD,
    MODELOS,
    N_NUM,
    NUMERICAS,
    SILVER,
    asegurar_directorios,
)
from pipeline.spark_utils import crear_spark_session


def clave_categorias(X_cat: np.ndarray) -> list[str]:
    return ["-".join(map(str, fila)) for fila in X_cat]


def transformar(lote: pd.DataFrame, m: dict, gamma: float):
    """Las mismas transformaciones del notebook 02."""
    p = m["prep"]
    X_num = p["escalador"].transform(p["imputador"].transform(lote[NUMERICAS]))
    X_cat = p["codificador"].transform(lote[CATEGORICAS].astype(str)).astype(int)
    Z = np.hstack([X_num, p["onehot"].transform(X_cat) * np.sqrt(gamma / 2)])
    return X_num, X_cat, Z


def aplicar_kprototypes(X_num: np.ndarray, X_cat: np.ndarray, m: dict, gamma: float):
    kp = m["kp"]
    parte_num = ((X_num[:, None, :] - kp["centros_num"][None]) ** 2).sum(axis=2)
    parte_cat = (X_cat[:, None, :] != kp["centros_cat"][None]).sum(axis=2)
    D = parte_num + gamma * parte_cat
    cluster = D.argmin(axis=1)
    distancia = D.min(axis=1)
    puntaje = distancia / kp["limites"][cluster]
    return cluster, distancia, puntaje


def aplicar_dbscan(X_num: np.ndarray, X_cat: np.ndarray, Z: np.ndarray, m: dict, eps: float):
    cluster = np.full(len(Z), -1)
    distancia = np.full(len(Z), np.inf)
    claves = pd.Series(clave_categorias(X_cat))
    for clave, filas in claves.groupby(claves).groups.items():
        if clave in m["buscadores"]:
            buscador, etiquetas = m["buscadores"][clave]
            filas = np.asarray(filas)
            d, i = buscador.kneighbors(X_num[filas])
            cerca = d[:, 0] <= eps
            cluster[filas[cerca]] = etiquetas[i[cerca, 0]]
            distancia[filas[cerca]] = d[cerca, 0]
    pendientes = np.where(cluster == -1)[0]
    if len(pendientes) > 0:
        d, i = m["buscador_general"].kneighbors(Z[pendientes])
        cerca = d[:, 0] <= eps
        cluster[pendientes[cerca]] = m["cluster_nucleos"][i[cerca, 0]]
        distancia[pendientes] = d[:, 0]
    return cluster, distancia


def _puntuar_archivo_parquet(
    src: Path,
    dst: Path,
    m: dict,
    gamma: float,
    eps: float,
    batch_size: int = 10_000,
) -> int:
    """Puntúa una partición Parquet de `silver` por lotes y escribe su partición en `gold`."""
    pf = pq.ParquetFile(src)
    drop_cols = {"raw_" + c for c in NUMERICAS + CATEGORICAS}
    cols_keep = [c for c in pf.schema_arrow.names if c not in drop_cols]
    cols_feat = NUMERICAS + CATEGORICAS

    out_fields = [pf.schema_arrow.field(c) for c in cols_keep] + [
        pa.field("kp_cluster", pa.int32()),
        pa.field("kp_distancia", pa.float32()),
        pa.field("kp_puntaje", pa.float32()),
        pa.field("kp_atipico", pa.bool_()),
        pa.field("db_cluster", pa.int32()),
        pa.field("db_distancia", pa.float32()),
        pa.field("db_ruido", pa.bool_()),
    ]
    out_schema = pa.schema(out_fields)

    total_filas = 0
    with pq.ParquetWriter(dst, out_schema, compression="snappy") as writer:
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols_keep):
            mask = pc.is_null(batch.column("motivo_exclusion"))
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                continue

            lote_feat = batch.select(cols_feat).to_pandas()
            X_num, X_cat, Z = transformar(lote_feat, m, gamma)
            kp_cluster, kp_distancia, kp_puntaje = aplicar_kprototypes(X_num, X_cat, m, gamma)
            db_cluster, db_distancia = aplicar_dbscan(X_num, X_cat, Z, m, eps)

            arrays = [batch.column(i) for i in range(batch.num_columns)] + [
                pa.array(kp_cluster.astype("int32"), type=pa.int32()),
                pa.array(kp_distancia.astype("float32"), type=pa.float32()),
                pa.array(kp_puntaje.astype("float32"), type=pa.float32()),
                pa.array((kp_puntaje > 1).astype("bool"), type=pa.bool_()),
                pa.array(db_cluster.astype("int32"), type=pa.int32()),
                pa.array(db_distancia.astype("float32"), type=pa.float32()),
                pa.array((db_cluster == -1).astype("bool"), type=pa.bool_()),
            ]
            writer.write_table(pa.Table.from_arrays(arrays, schema=out_schema))
            total_filas += batch.num_rows

    return total_filas


def _puntuar_particiones_windows(
    silver_path: Path,
    gold_path: Path,
    m: dict,
    gamma: float,
    eps: float,
    n_workers: int,
) -> int:
    """Puntúa en paralelo las particiones Parquet de `silver` evitando el IPC Arrow de PySpark en Windows."""
    archivos = sorted(silver_path.glob("*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {silver_path}")

    if gold_path.exists():
        shutil.rmtree(gold_path)
    gold_path.mkdir(parents=True, exist_ok=True)

    tareas = [
        (src, gold_path / f"part-{i:05d}.parquet")
        for i, src in enumerate(archivos)
    ]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futuros = [
            pool.submit(_puntuar_archivo_parquet, src, dst, m, gamma, eps)
            for src, dst in tareas
        ]
        total = sum(f.result() for f in futuros)

    (gold_path / "_SUCCESS").touch()
    return total


def _gold_valido(gold_path: Path) -> bool:
    """Devuelve True si la capa gold ya existe completa (`_SUCCESS` y archivos Parquet)."""
    return (gold_path / "_SUCCESS").exists() and any(gold_path.glob("*.parquet"))


@task(
    name="05 · Población completa (Spark)",
    retries=1,
    retry_delay_seconds=10,
    tags=["secop", "spark", "scoring"],
    log_prints=True,
)
def aplicar_modelos_poblacion_spark(
    silver_path: Path = SILVER,
    prep_limpia_path: Path = MODELOS / "preparacion_limpia.joblib",
    kprototypes_limpio_path: Path = MODELOS / "kprototypes_limpio.joblib",
    dbscan_limpio_path: Path = MODELOS / "dbscan_limpio.joblib",
    gold_path: Path = GOLD,
) -> dict[str, Path]:
    """Ejecuta el notebook 05: puntúa todos los contratos válidos en paralelo con Spark."""
    asegurar_directorios()

    for variable in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[variable] = "1"
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    n_workers = min(4, os.cpu_count() or 4)
    spark = None

    def _iniciar_spark():
        return crear_spark_session(
            app_name="secop-poblacion",
            master=f"local[{n_workers}]",
            extra_configs={
                "spark.sql.ansi.enabled": "false",
                "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
                "spark.sql.execution.pyspark.udf.faulthandler.enabled": "true",
                "spark.python.worker.faulthandler.enabled": "true",
            },
        )

    try:
        segundos = 0.0
        if _gold_valido(gold_path):
            print("El Parquet gold ya existe completo, se reutiliza sin recalcular.")
        else:
            # 1. Cargar los modelos entrenados (solo transformadores ligeros para broadcast)
            prep_completo = joblib.load(prep_limpia_path)
            prep = {
                k: prep_completo[k]
                for k in ("imputador", "escalador", "codificador", "onehot")
            }
            del prep_completo

            kp_completo = joblib.load(kprototypes_limpio_path)
            gamma = kp_completo["gamma"]
            kp = {
                k: kp_completo[k]
                for k in ("centros_num", "centros_cat", "limites")
            }
            del kp_completo

            db = joblib.load(dbscan_limpio_path)
            eps = db["eps"]
            print(
                f"K-prototypes: {len(kp['centros_num'])} prototipos | "
                f"DBSCAN: {len(db['Z_nucleos']):,} puntos núcleo, eps = {eps:.3f}"
            )

            # 2. Construir buscadores para aplicar DBSCAN a contratos nuevos
            claves_nucleos = pd.Series(clave_categorias(db["X_cat_nucleos"]))
            buscadores = {}
            for clave, posiciones in claves_nucleos.groupby(claves_nucleos).groups.items():
                pos = np.asarray(posiciones)
                buscadores[clave] = (
                    NearestNeighbors(n_neighbors=1).fit(db["Z_nucleos"][pos, :N_NUM]),
                    db["cluster_nucleos"][pos],
                )

            buscador_general = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                db["Z_nucleos"]
            )
            print(f"{len(buscadores):,} combinaciones de categorías distintas entre los núcleos")

            # 3. Puntuar todos los contratos válidos y guardar la capa gold
            inicio = time.time()
            if sys.platform == "win32":
                # En Windows, puntuamos primero en paralelo sin dejar una JVM ociosa
                # y abrimos Spark después solo para validar y resumir gold.
                m_local = {
                    "prep": prep,
                    "kp": kp,
                    "buscadores": buscadores,
                    "buscador_general": buscador_general,
                    "cluster_nucleos": db["cluster_nucleos"],
                }
                _puntuar_particiones_windows(
                    silver_path=silver_path,
                    gold_path=gold_path,
                    m=m_local,
                    gamma=gamma,
                    eps=eps,
                    n_workers=n_workers,
                )
            else:
                spark = _iniciar_spark()
                modelos = spark.sparkContext.broadcast(
                    {
                        "prep": prep,
                        "kp": kp,
                        "buscadores": buscadores,
                        "buscador_general": buscador_general,
                        "cluster_nucleos": db["cluster_nucleos"],
                    }
                )
                numeric_cols = list(NUMERICAS)
                cat_cols = list(CATEGORICAS)

                def evaluar_lotes(lotes):
                    import numpy as _np
                    import pandas as _pd

                    m = modelos.value
                    p = m["prep"]
                    kp_m = m["kp"]
                    busc = m["buscadores"]
                    busc_gen = m["buscador_general"]
                    clust_nuc = m["cluster_nucleos"]
                    factor_cat = _np.sqrt(gamma / 2)

                    for lote in lotes:
                        lote = lote.copy()
                        X_num = p["escalador"].transform(
                            p["imputador"].transform(lote[numeric_cols])
                        )
                        X_cat = p["codificador"].transform(lote[cat_cols].astype(str)).astype(int)
                        Z = _np.hstack([X_num, p["onehot"].transform(X_cat) * factor_cat])

                        parte_num = ((X_num[:, None, :] - kp_m["centros_num"][None]) ** 2).sum(
                            axis=2
                        )
                        parte_cat = (X_cat[:, None, :] != kp_m["centros_cat"][None]).sum(axis=2)
                        D = parte_num + gamma * parte_cat
                        kp_cluster = D.argmin(axis=1)
                        kp_distancia = D.min(axis=1)
                        kp_puntaje = kp_distancia / kp_m["limites"][kp_cluster]

                        db_cluster = _np.full(len(Z), -1)
                        db_distancia = _np.full(len(Z), _np.inf)
                        claves = _pd.Series(["-".join(map(str, fila)) for fila in X_cat])
                        for clave, filas in claves.groupby(claves).groups.items():
                            if clave in busc:
                                buscador, etiquetas = busc[clave]
                                filas_arr = _np.asarray(filas)
                                d, i = buscador.kneighbors(X_num[filas_arr])
                                cerca = d[:, 0] <= eps
                                db_cluster[filas_arr[cerca]] = etiquetas[i[cerca, 0]]
                                db_distancia[filas_arr[cerca]] = d[cerca, 0]
                        pendientes = _np.where(db_cluster == -1)[0]
                        if len(pendientes) > 0:
                            d, i = busc_gen.kneighbors(Z[pendientes])
                            cerca = d[:, 0] <= eps
                            db_cluster[pendientes[cerca]] = clust_nuc[i[cerca, 0]]
                            db_distancia[pendientes] = d[:, 0]

                        lote["kp_cluster"] = kp_cluster.astype("int32")
                        lote["kp_distancia"] = kp_distancia.astype("float32")
                        lote["kp_puntaje"] = kp_puntaje.astype("float32")
                        lote["kp_atipico"] = (kp_puntaje > 1).astype("bool")
                        lote["db_cluster"] = db_cluster.astype("int32")
                        lote["db_distancia"] = db_distancia.astype("float32")
                        lote["db_ruido"] = (db_cluster == -1).astype("bool")
                        yield lote

                contratos = (
                    spark.read.parquet(str(silver_path))
                    .filter(F.col("motivo_exclusion").isNull())
                    .drop(*["raw_" + c for c in NUMERICAS + CATEGORICAS])
                )

                columnas_nuevas = [
                    StructField("kp_cluster", IntegerType()),
                    StructField("kp_distancia", FloatType()),
                    StructField("kp_puntaje", FloatType()),
                    StructField("kp_atipico", BooleanType()),
                    StructField("db_cluster", IntegerType()),
                    StructField("db_distancia", FloatType()),
                    StructField("db_ruido", BooleanType()),
                ]
                esquema_salida = StructType(contratos.schema.fields + columnas_nuevas)

                contratos.mapInPandas(evaluar_lotes, schema=esquema_salida).write.mode(
                    "overwrite"
                ).parquet(str(gold_path))
            segundos = time.time() - inicio

        if spark is None:
            spark = _iniciar_spark()

        gold = spark.read.parquet(str(gold_path))
        n = gold.count()
        if segundos > 0:
            print(
                f"{n:,} contratos evaluados en {segundos / 60:.1f} minutos "
                f"({n / segundos:,.0f} contratos por segundo)"
            )
        else:
            print(f"{n:,} contratos cargados desde gold existente")

        gold.agg(
            F.round(100 * F.avg(F.col("db_ruido").cast("int")), 2).alias("% ruido DBSCAN"),
            F.round(100 * F.avg(F.col("kp_atipico").cast("int")), 2).alias("% atípicos K-prototypes"),
            F.round(100 * F.avg((F.col("db_ruido") & F.col("kp_atipico")).cast("int")), 2).alias(
                "% atípicos en ambos"
            ),
        ).show()
    finally:
        if spark is not None:
            spark.stop()

    return {"gold": gold_path}
