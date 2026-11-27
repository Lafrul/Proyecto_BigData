"""04 · K-prototypes y distancia al centro del cluster (Prefect @task)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from prefect import task
from sklearn.metrics import silhouette_score

from pipeline.config import (
    CATEGORICAS,
    FIGURAS,
    GAMMA,
    MODELOS,
    N_NUM,
    NUMERICAS,
    RESULTADOS,
    TABLAS,
    asegurar_directorios,
)

import matplotlib.pyplot as plt


def entrenar_kprototypes(X_num: np.ndarray, X_cat: np.ndarray, k: int, n_init: int) -> KPrototypes:
    """Entrena un modelo K-prototypes sobre variables numéricas y categóricas."""
    X = np.hstack([X_num, X_cat])
    modelo = KPrototypes(
        n_clusters=k,
        gamma=GAMMA,
        init="Cao",
        n_init=n_init,
        max_iter=50,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X, categorical=list(range(N_NUM, X.shape[1])))
    return modelo


def distancia_a_prototipos(X_num: np.ndarray, X_cat: np.ndarray, modelo: KPrototypes) -> np.ndarray:
    """Calcula la matriz de distancias de cada contrato a cada prototipo."""
    centros = np.asarray(modelo.cluster_centroids_, dtype=float)
    centros_num, centros_cat = centros[:, :N_NUM], centros[:, N_NUM:]
    parte_numerica = ((X_num[:, None, :] - centros_num[None, :, :]) ** 2).sum(axis=2)
    parte_categorica = (X_cat[:, None, :] != centros_cat[None, :, :]).sum(axis=2)
    return parte_numerica + GAMMA * parte_categorica


def limites_por_cluster(cluster: np.ndarray, distancia: np.ndarray) -> pd.Series:
    """Calcula el umbral de Tukey Q3 + 3·IQR de distancia para cada cluster."""
    tabla = pd.DataFrame({"cluster": cluster, "distancia": distancia})
    q = tabla.groupby("cluster")["distancia"].quantile([0.25, 0.75]).unstack()
    return q[0.75] + 3 * (q[0.75] - q[0.25])


@task(
    name="04 · K-prototypes",
    retries=1,
    retry_delay_seconds=5,
    tags=["secop", "modelos", "kprototypes"],
    log_prints=True,
)
def ejecutar_kprototypes(
    prep_cruda_path: Path = MODELOS / "preparacion_cruda.joblib",
    prep_limpia_path: Path = MODELOS / "preparacion_limpia.joblib",
) -> dict[str, Path]:
    """Ejecuta el notebook 04: selección de k, ajuste de K-prototypes y regla de atipicidad."""
    asegurar_directorios()

    prep = {
        "cruda": joblib.load(prep_cruda_path),
        "limpia": joblib.load(prep_limpia_path),
    }

    # 1. Elegir el número de clusters k
    rng = np.random.default_rng(42)
    evaluacion = []
    for escenario in ["cruda", "limpia"]:
        p = prep[escenario]
        sub = rng.choice(len(p["X_num"]), 10_000, replace=False)
        sub_silueta = sub[:5_000]
        for k in range(2, 9):
            modelo = entrenar_kprototypes(p["X_num"][sub], p["X_cat"][sub], k, n_init=2)
            etiquetas = modelo.labels_[:5_000]
            silueta = silhouette_score(p["Z"][sub_silueta], etiquetas)
            evaluacion.append(
                {"escenario": escenario, "k": k, "costo": modelo.cost_, "silueta": silueta}
            )
            print(f"{escenario:6s} k={k}  costo={modelo.cost_:10,.0f}  silueta={silueta:.3f}")

    evaluacion = pd.DataFrame(evaluacion)
    evaluacion.to_csv(TABLAS / "kprototypes_eleccion_k.csv", index=False)

    k_elegido = (
        evaluacion.loc[evaluacion.groupby("escenario")["silueta"].idxmax()]
        .set_index("escenario")["k"]
        .to_dict()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for escenario, color in [("cruda", "#c8553d"), ("limpia", "#2a6f97")]:
        e = evaluacion[evaluacion.escenario == escenario]
        axes[0].plot(e.k, e.costo / e.costo.max(), "o-", color=color, label=escenario)
        axes[1].plot(
            e.k,
            e.silueta,
            "o-",
            color=color,
            label=f"{escenario} (k = {k_elegido[escenario]})",
        )
        axes[1].axvline(k_elegido[escenario], color=color, ls="--", lw=0.8)
    axes[0].set(title="Costo (relativo)", xlabel="k")
    axes[1].set(title="Silueta (más alto = mejor)", xlabel="k")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "kprototypes_eleccion_k.png", dpi=130)
    plt.close(fig)
    print("k elegido:", k_elegido)

    # 2. Entrenar con la muestra completa (50.000)
    modelos_kp = {}
    for escenario in ["cruda", "limpia"]:
        p = prep[escenario]
        modelos_kp[escenario] = entrenar_kprototypes(
            p["X_num"], p["X_cat"], k_elegido[escenario], n_init=3
        )
        print(escenario, "listo")

    # 3. Distancia al prototipo y regla de atipicidad
    resultados_kp = {}
    limites = {}
    kp_paths = {}
    for escenario in ["cruda", "limpia"]:
        p = prep[escenario]
        D = distancia_a_prototipos(p["X_num"], p["X_cat"], modelos_kp[escenario])
        cluster = D.argmin(axis=1)
        distancia = D.min(axis=1)
        limites[escenario] = limites_por_cluster(cluster, distancia)

        r = pd.DataFrame(
            {"row_id": p["datos"]["row_id"], "kp_cluster": cluster, "kp_distancia": distancia}
        )
        r["kp_puntaje"] = r["kp_distancia"] / r["kp_cluster"].map(limites[escenario])
        r["kp_atipico"] = r["kp_puntaje"] > 1
        resultados_kp[escenario] = r
        ruta_salida = RESULTADOS / f"kprototypes_{escenario}.parquet"
        r.to_parquet(ruta_salida)
        kp_paths[f"kprototypes_{escenario}"] = ruta_salida
        print(
            f"{escenario}: k = {k_elegido[escenario]}, "
            f"atípicos = {r.kp_atipico.sum():,} ({100 * r.kp_atipico.mean():.2f}%)"
        )

    r = resultados_kp["limpia"]
    clusters = sorted(r.kp_cluster.unique())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.boxplot(
        [np.log1p(r.loc[r.kp_cluster == c, "kp_distancia"]) for c in clusters],
        tick_labels=[str(c) for c in clusters],
        showfliers=False,
    )
    ax.scatter(
        np.arange(1, len(clusters) + 1),
        np.log1p(limites["limpia"][clusters]),
        marker="_",
        s=500,
        color="crimson",
        label="límite Q3 + 3·IQR",
    )
    ax.set(
        xlabel="cluster",
        ylabel="log(1 + distancia al prototipo)",
        title="Distancia al prototipo por cluster (escenario limpio)",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "kprototypes_distancias.png", dpi=130)
    plt.close(fig)

    # 4. ¿Cómo es el "contrato típico" de cada cluster? (escenario limpio)
    p = prep["limpia"]
    centros = np.asarray(modelos_kp["limpia"].cluster_centroids_, dtype=float)
    num = pd.DataFrame(p["escalador"].inverse_transform(centros[:, :N_NUM]), columns=NUMERICAS)
    cat = pd.DataFrame(
        p["codificador"].inverse_transform(centros[:, N_NUM:].astype(int)),
        columns=CATEGORICAS,
    )

    prototipos = pd.DataFrame(
        {
            "valor contrato (millones COP)": (np.expm1(num["log_valor_contrato"]) / 1e6).round(1),
            "plazo (días)": np.expm1(num["log_plazo_dias"]).round(0),
            "adiciones / valor": num["ratio_adiciones"].round(2),
            "días firma→publicación": (
                np.sign(num["slog_dias_firma_cargue"])
                * np.expm1(num["slog_dias_firma_cargue"].abs())
            ).round(0),
        }
    )
    prototipos = pd.concat(
        [
            prototipos,
            cat[
                [
                    "modalidad_de_contratacion",
                    "tipo_de_contrato",
                    "orden_entidad",
                    "tipo_identifi_del_contratista",
                ]
            ],
        ],
        axis=1,
    )
    prototipos["% contratos"] = (
        resultados_kp["limpia"].kp_cluster.value_counts(normalize=True).sort_index() * 100
    ).round(1)
    prototipos["% atípicos"] = (
        resultados_kp["limpia"].groupby("kp_cluster").kp_atipico.mean() * 100
    ).round(1)
    prototipos.to_csv(TABLAS / "kprototypes_prototipos.csv", index_label="cluster")
    print(prototipos)

    modelo_limpio_path = MODELOS / "kprototypes_limpio.joblib"
    joblib.dump(
        {
            "centros_num": centros[:, :N_NUM],
            "centros_cat": centros[:, N_NUM:],
            "limites": limites["limpia"].to_numpy(),
            "gamma": GAMMA,
        },
        modelo_limpio_path,
    )
    print("Guardado")

    return {
        **kp_paths,
        "modelo_limpio": modelo_limpio_path,
    }
