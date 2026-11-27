"""02 · Preparación de los datos para los modelos (Prefect @task)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prefect import task
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pipeline.config import (
    CATEGORICAS,
    FIGURAS,
    GAMMA,
    MODELOS,
    MUESTRAS,
    NUMERICAS,
    asegurar_directorios,
)

import matplotlib.pyplot as plt


def preparar(datos: pd.DataFrame) -> dict:
    """Aplica imputación, escalado, codificación ordinal y matriz Z compartida."""
    imputador = SimpleImputer(strategy="median").fit(datos[NUMERICAS])
    escalador = StandardScaler().fit(imputador.transform(datos[NUMERICAS]))
    codificador = OrdinalEncoder(
        min_frequency=0.001,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(datos[CATEGORICAS].astype(str))

    X_num = escalador.transform(imputador.transform(datos[NUMERICAS]))
    X_cat = codificador.transform(datos[CATEGORICAS].astype(str)).astype(int)

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_cat)
    Z = np.hstack([X_num, onehot.transform(X_cat) * np.sqrt(GAMMA / 2)])

    return {
        "datos": datos.reset_index(drop=True),
        "X_num": X_num,
        "X_cat": X_cat,
        "Z": Z,
        "imputador": imputador,
        "escalador": escalador,
        "codificador": codificador,
        "onehot": onehot,
    }


@task(
    name="02 · Preparación de datos",
    retries=1,
    retry_delay_seconds=5,
    tags=["secop", "preparacion"],
    log_prints=True,
)
def preparacion_datos(
    muestra_cruda_path: Path = MUESTRAS / "muestra_cruda.parquet",
    muestra_limpia_path: Path = MUESTRAS / "muestra_limpia.parquet",
) -> dict[str, Path]:
    """Ejecuta el notebook 02: compara crudo vs limpio y prepara matrices para los modelos."""
    asegurar_directorios()

    limpia = pd.read_parquet(muestra_limpia_path)
    cruda = pd.read_parquet(muestra_cruda_path)

    cruda = cruda.drop(columns=NUMERICAS + CATEGORICAS)
    cruda = cruda.rename(columns={"raw_" + c: c for c in NUMERICAS + CATEGORICAS})
    print(cruda.shape, limpia.shape)

    # 1. ¿Qué cambia entre los datos crudos y los limpios?
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    for ax, col in zip(axes.ravel(), NUMERICAS):
        ax.hist(cruda[col], bins=60, alpha=0.6, label="crudo", color="#c8553d")
        ax.hist(limpia[col].dropna(), bins=60, alpha=0.6, label="limpio", color="#2a6f97")
        ax.set_title(col, fontsize=9)
    axes[0, 0].legend()
    plt.suptitle("Distribución de las variables numéricas: crudo vs limpio")
    plt.tight_layout()
    plt.savefig(FIGURAS / "variables_crudo_vs_limpio.png", dpi=130)
    plt.close(fig)

    resumen_variables = pd.DataFrame(
        {
            "mediana cruda": cruda[NUMERICAS].median(),
            "mediana limpia": limpia[NUMERICAS].median(),
            "% ceros crudo": (cruda[NUMERICAS] == 0).mean() * 100,
            "% ceros limpio": (limpia[NUMERICAS] == 0).mean() * 100,
        }
    ).round(2)
    print(resumen_variables)

    # 2 y 3. Transformaciones y distancia compartida entre contratos
    prep_cruda = preparar(cruda)
    prep_limpia = preparar(limpia)
    print("Columnas para DBSCAN (crudo): ", prep_cruda["Z"].shape[1])
    print("Columnas para DBSCAN (limpio):", prep_limpia["Z"].shape[1])

    p = prep_limpia
    for i in range(1, 6):
        a, b = 0, i
        d_kprototypes = (
            ((p["X_num"][a] - p["X_num"][b]) ** 2).sum()
            + GAMMA * (p["X_cat"][a] != p["X_cat"][b]).sum()
        )
        d_dbscan = ((p["Z"][a] - p["Z"][b]) ** 2).sum()
        print(
            f"contrato 0 vs {b}:  K-prototypes = {d_kprototypes:.4f}   "
            f"euclidiana² en Z = {d_dbscan:.4f}"
        )

    prep_cruda_path = MODELOS / "preparacion_cruda.joblib"
    prep_limpia_path = MODELOS / "preparacion_limpia.joblib"
    joblib.dump(prep_cruda, prep_cruda_path)
    joblib.dump(prep_limpia, prep_limpia_path)
    print("Guardado")

    return {
        "preparacion_cruda": prep_cruda_path,
        "preparacion_limpia": prep_limpia_path,
    }
