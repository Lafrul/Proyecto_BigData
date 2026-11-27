"""03 · DBSCAN (Prefect @task)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kneed import KneeLocator
from prefect import task
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from pipeline.config import (
    FIGURAS,
    MIN_SAMPLES,
    MODELOS,
    RESULTADOS,
    asegurar_directorios,
)

import matplotlib.pyplot as plt


def elegir_eps(Z: np.ndarray):
    """Calcula la curva de k-distancias y encuentra el codo con KneeLocator."""
    vecinos = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Z)
    distancias, _ = vecinos.kneighbors(Z[:10_000])
    k_distancias = np.sort(distancias[:, -1])
    k_distancias = k_distancias[: int(len(k_distancias) * 0.995)]
    codo = KneeLocator(
        range(len(k_distancias)),
        k_distancias,
        curve="convex",
        direction="increasing",
    )
    return k_distancias[codo.knee], k_distancias


@task(
    name="03 · DBSCAN",
    retries=1,
    retry_delay_seconds=5,
    tags=["secop", "modelos", "dbscan"],
    log_prints=True,
)
def ejecutar_dbscan(
    prep_cruda_path: Path = MODELOS / "preparacion_cruda.joblib",
    prep_limpia_path: Path = MODELOS / "preparacion_limpia.joblib",
) -> dict[str, Path]:
    """Ejecuta el notebook 03: selección de eps, ajuste de DBSCAN y guardado de núcleos."""
    asegurar_directorios()

    prep = {
        "cruda": joblib.load(prep_cruda_path),
        "limpia": joblib.load(prep_limpia_path),
    }

    # 1. Elegir los parámetros
    eps = {}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for escenario, color in [("cruda", "#c8553d"), ("limpia", "#2a6f97")]:
        eps[escenario], curva = elegir_eps(prep[escenario]["Z"])
        ax.plot(
            np.linspace(0, 99.5, len(curva)),
            curva,
            color=color,
            label=f"{escenario}  (eps = {eps[escenario]:.2f})",
        )
        ax.axhline(eps[escenario], color=color, ls="--", lw=0.8)
    ax.set_xlabel("percentil de los puntos")
    ax.set_ylabel(f"distancia al vecino {MIN_SAMPLES}")
    ax.set_title("Curva de k-distancias y eps elegido (codo)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "dbscan_k_distancias.png", dpi=130)
    plt.close(fig)
    print("min_samples =", MIN_SAMPLES, "| eps =", {k: round(v, 3) for k, v in eps.items()})

    # 2. Entrenar DBSCAN en las dos muestras
    modelos_dbscan = {}
    resumen = []
    for escenario in ["cruda", "limpia"]:
        Z = prep[escenario]["Z"]
        modelo = DBSCAN(eps=eps[escenario], min_samples=MIN_SAMPLES, n_jobs=-1).fit(Z)
        modelos_dbscan[escenario] = modelo
        etiquetas = modelo.labels_
        resumen.append(
            {
                "escenario": escenario,
                "eps": round(eps[escenario], 3),
                "clusters": etiquetas.max() + 1,
                "puntos núcleo": len(modelo.core_sample_indices_),
                "ruido (atípicos)": int((etiquetas == -1).sum()),
                "% ruido": round(100 * (etiquetas == -1).mean(), 2),
            }
        )
    print(pd.DataFrame(resumen))

    for escenario in ["cruda", "limpia"]:
        print(
            escenario,
            pd.Series(modelos_dbscan[escenario].labels_).value_counts().sort_index().to_dict(),
        )

    # 3. ¿Cómo son los contratos que DBSCAN marca como ruido? (muestra limpia)
    datos = prep["limpia"]["datos"].copy()
    datos["db_ruido"] = modelos_dbscan["limpia"].labels_ == -1
    print(
        datos.groupby("db_ruido")[
            ["valor_contrato", "plazo_dias", "adiciones_dias", "dias_firma_cargue"]
        ].median()
    )

    tabla = pd.crosstab(datos["modalidad_de_contratacion"], datos["db_ruido"], normalize="columns") * 100
    tabla.columns = ["% en normales", "% en ruido"]
    print(tabla.sort_values("% en ruido", ascending=False).round(1).head(8))

    # 4. Guardar resultados
    dbscan_paths = {}
    for escenario in ["cruda", "limpia"]:
        etiquetas = modelos_dbscan[escenario].labels_
        ruta_salida = RESULTADOS / f"dbscan_{escenario}.parquet"
        pd.DataFrame(
            {
                "row_id": prep[escenario]["datos"]["row_id"],
                "db_cluster": etiquetas,
                "db_ruido": etiquetas == -1,
            }
        ).to_parquet(ruta_salida)
        dbscan_paths[f"dbscan_{escenario}"] = ruta_salida

    nucleos = modelos_dbscan["limpia"].core_sample_indices_
    modelo_limpio_path = MODELOS / "dbscan_limpio.joblib"
    joblib.dump(
        {
            "eps": eps["limpia"],
            "min_samples": MIN_SAMPLES,
            "Z_nucleos": prep["limpia"]["Z"][nucleos],
            "X_cat_nucleos": prep["limpia"]["X_cat"][nucleos],
            "cluster_nucleos": modelos_dbscan["limpia"].labels_[nucleos],
        },
        modelo_limpio_path,
    )
    print("Guardado")

    return {
        **dbscan_paths,
        "modelo_limpio": modelo_limpio_path,
    }
