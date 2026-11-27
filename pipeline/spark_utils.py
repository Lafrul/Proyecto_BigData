"""Utilidades para crear sesiones locales de PySpark."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pipeline.config import RAIZ


def _limpiar_gateway_caido() -> None:
    """Resetea un gateway Py4J muerto para que getOrCreate() pueda iniciar una JVM nueva."""
    gw = getattr(SparkContext, "_gateway", None)
    if gw is None:
        return
    try:
        gw.jvm.java.lang.System.currentTimeMillis()
    except Exception:
        SparkContext._active_spark_context = None
        SparkContext._gateway = None
        SparkContext._jvm = None


def _configurar_hadoop_windows() -> str | None:
    """Configura HADOOP_HOME y PATH en Windows si existe winutils.exe."""
    if sys.platform != "win32":
        return None

    candidatos = []
    if env_home := os.environ.get("HADOOP_HOME"):
        candidatos.append(Path(env_home))
    candidatos.extend([Path(r"C:\hadoop"), RAIZ / "hadoop"])

    for base in candidatos:
        winutils = base / "bin" / "winutils.exe"
        if winutils.exists():
            home_str = str(base)
            bin_str = str(base / "bin")
            os.environ["HADOOP_HOME"] = home_str
            rutas = os.environ.get("PATH", "").split(os.pathsep)
            if bin_str.lower() not in [r.lower() for r in rutas]:
                os.environ["PATH"] = os.pathsep.join([bin_str, *rutas])
            return home_str
    return None


def crear_spark_session(
    app_name: str,
    extra_configs: dict[str, str] | None = None,
    master: str = "local[*]",
) -> SparkSession:
    """Crea y configura una sesión local de PySpark."""
    if sys.platform == "darwin":
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"]
        ).decode().strip()

    _limpiar_gateway_caido()
    hadoop_home = _configurar_hadoop_windows()
    os.environ["PYSPARK_PYTHON"] = sys.executable

    # Asegura que los workers de PySpark puedan importar el paquete `pipeline`
    raiz_str = str(RAIZ)
    rutas_py = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if raiz_str not in rutas_py:
        os.environ["PYTHONPATH"] = os.pathsep.join([raiz_str, *rutas_py])

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
    )
    if hadoop_home:
        builder = builder.config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={hadoop_home}")
    for clave, valor in (extra_configs or {}).items():
        builder = builder.config(clave, valor)
    return builder.getOrCreate()
