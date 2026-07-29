"""MLflow etrafında ince bir sarmalayıcı: deney takibi + model versiyonlama.

Neden sarmalayıcı: MLflow API'sini doğrudan her modülde tekrarlamak yerine, projeye
özel varsayılanları (deney ismi, tracking URI) tek yerde tanımlıyoruz. Colab'da
`mlflow ui` arka planda çalıştırılıp ngrok/localtunnel ile dışa açılabilir; yerelde
doğrudan `mlflow ui` yeterlidir.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlflow

from src.config import ROOT_DIR

DEFAULT_EXPERIMENT = "tiha-asistan"
TRACKING_URI = f"file:{ROOT_DIR / 'mlruns'}"


def _ensure_experiment(experiment_name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)


@contextmanager
def run(run_name: str, experiment_name: str = DEFAULT_EXPERIMENT, params: dict | None = None):
    """Kullanım:

    with run("qlora-v1", params={"lora_r": 16}) as active_run:
        ... eğitim ...
        log_metrics({"eval_loss": 0.42})
    """
    _ensure_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as active_run:
        if params:
            mlflow.log_params(params)
        yield active_run


def log_metrics(metrics: dict, step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact_dir(path: str, artifact_path: str | None = None) -> None:
    mlflow.log_artifacts(path, artifact_path=artifact_path)


def register_model_version(model_uri: str, name: str) -> str:
    """Model kayıt defterine (registry) yeni bir versiyon ekler; A/B test ve rollback
    için versiyon numarasını döndürür."""
    result = mlflow.register_model(model_uri, name)
    return result.version
