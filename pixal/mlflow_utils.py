"""
Minimal MLflow helper utilities for PIXAL.
- start_run(params): starts an MLflow run with given params
- log_params(params): logs a dict of params
- log_metrics(metrics): logs metrics
- log_artifact(path, artifact_path=None): logs artifact file/dir
- KerasModelLoggerCallback: Keras callback to log epoch metrics

Configuration:
- Set `MLFLOW_TRACKING_URI` env var to point to a remote server (optional)
- By default MLflow will use local `mlruns/` in working directory

This file keeps the integration small and optional. Training scripts import and
call `with mlflow_utils.run_experiment(params):` to record runs.
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Dict, Optional

MLFLOW_URI_ENV = "MLFLOW_TRACKING_URI"

try:
    import mlflow
    import mlflow.keras  # type: ignore
except Exception:
    mlflow = None  # type: ignore

# allow overriding tracking uri via env var
if mlflow is not None:
    # Prefer an explicit env var if set; otherwise default to a repo-local mlruns
    _tracking_uri = os.environ.get(MLFLOW_URI_ENV)
    if not _tracking_uri:
        # Default to a predictable file-backed store inside the repository
        # (use an absolute path so different CWDs still write to the same place)
        _tracking_uri = "file:/home/jacob/work/pixal/mlruns"
        # Export it for subprocesses that may inspect the env
        os.environ[MLFLOW_URI_ENV] = _tracking_uri

    try:
        mlflow.set_tracking_uri(_tracking_uri)
        # If the backend is file-based, ensure the directory exists
        if _tracking_uri.startswith("file:"):
            # strip file: prefix and create the directory if needed
            path = _tracking_uri[len("file:"):]
            try:
                os.makedirs(path, exist_ok=True)
                # create an artifacts subdir for convenience
                os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
            except Exception:
                pass
    except Exception:
        pass


@contextmanager
def run_experiment(params: Optional[Dict] = None, run_name: Optional[str] = None):
    """Context manager that starts and ends an MLflow run and logs params.

    This is best-effort: if MLflow isn't installed or fails, it yields a dummy
    context but does not raise so training can continue.
    """
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            try:
                mlflow.log_params({k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items()})
            except Exception:
                # Best-effort: avoid failing training if mlflow can't serialize
                pass
        yield run


def log_params(params: Dict):
    if mlflow is None:
        return
    try:
        mlflow.log_params({k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items()})
    except Exception:
        pass


def log_metrics(metrics: Dict, step: Optional[int] = None):
    if mlflow is None:
        return
    try:
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)
    except Exception:
        pass


def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    if mlflow is None:
        return
    try:
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path)
    except Exception:
        pass


def log_keras_model(model_or_path, artifact_path: str = "model"):
    """Log a Keras model to MLflow.

    Accepts either a `tf.keras.Model` instance or a filesystem path to a saved model.
    This is best-effort: if MLflow or mlflow.keras are not available the call is a no-op.
    """
    if mlflow is None:
        return
    try:
        # If a model instance is provided, prefer mlflow.keras.log_model
        if hasattr(model_or_path, 'save'):
            try:
                mlflow.keras.log_model(model_or_path, artifact_path=artifact_path)
                return
            except Exception:
                pass

        # Otherwise treat as a path and log the artifact (directory or file)
        path = str(model_or_path)
        if os.path.isdir(path):
            # log directory contents
            mlflow.log_artifacts(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception:
        pass


try:
    import tensorflow as _tf  # noqa: N812
    TF_AVAILABLE = True
except Exception:
    _tf = None  # type: ignore
    TF_AVAILABLE = False


if TF_AVAILABLE:
    class KerasModelLoggerCallback(_tf.keras.callbacks.Callback):
        """Keras callback that logs epoch metrics to MLflow.

        Usage: pass instance in `model.fit(..., callbacks=[KerasModelLoggerCallback()])`
        """
        def on_epoch_end(self, epoch, logs=None):
            if not logs or mlflow is None:
                return
            try:
                for k, v in logs.items():
                    # logs may include arrays; coerce
                    mlflow.log_metric(k, float(v), step=epoch)
            except Exception:
                pass
else:
    class KerasModelLoggerCallback(object):
        def __init__(self, *args, **kwargs):
            pass
