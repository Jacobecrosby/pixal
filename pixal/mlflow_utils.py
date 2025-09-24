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
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict

try:
    import mlflow
except Exception:
    mlflow = None  # best-effort behavior below

MLFLOW_URI_ENV = "MLFLOW_TRACKING_URI"

def _normalize_uri_path(p: Path) -> str:
    # Return absolute path string
    return str(p.resolve())

def _as_tracking_uri(value: str) -> str:
    """
    Convert a value into an MLflow tracking URI.
    - If it looks like a URI scheme (contains '://'), pass through unchanged.
    - Otherwise treat as a local path and prefix 'file://'.
    """
    if "://" in value:
        return value
    # local filesystem path
    abs_path = _normalize_uri_path(Path(value))
    return f"file://{abs_path}"

def _ensure_local_dir_for_uri(uri: str) -> None:
    """
    If the tracking URI is a local 'file://' URI, ensure the folder exists.
    """
    if not uri.startswith("file://"):
        return
    # Strip scheme; on Linux this is straightforward
    local_path = uri[len("file://"):]
    Path(local_path).mkdir(parents=True, exist_ok=True)

# ---- Initialize MLflow tracking URI early ----
if mlflow is not None:
    # Prefer explicit env var if set; otherwise default to repo-local pixal/mlruns
    env_uri = os.environ.get(MLFLOW_URI_ENV)
    if env_uri:
        effective_uri = _as_tracking_uri(env_uri)
    else:
        # Use repo-local `pixal/mlruns` dir (absolute path), e.g. "<repo_root>/pixal/mlruns"
        repo_root = Path(__file__).resolve().parents[2]  # adjust depth if needed
        default_dir = repo_root / "pixal" / "mlruns"
        effective_uri = _as_tracking_uri(_normalize_uri_path(default_dir))
        # Also populate the env var so subprocesses inherit it
        os.environ[MLFLOW_URI_ENV] = effective_uri

    # Ensure local directory exists for file:// stores
    _ensure_local_dir_for_uri(effective_uri)

    # Now set it on mlflow
    mlflow.set_tracking_uri(effective_uri)

    # Optional but helpful for debugging: print once
    try:
        print(f"[mlflow] tracking URI = {effective_uri}")
    except Exception:
        pass

DEFAULT_EXPERIMENT_NAME = "pixal"  # choose something project-specific

@contextmanager
def run_experiment(
    params: Optional[Dict] = None,
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    """
    Context manager that starts and ends an MLflow run and logs params.
    Best-effort: if MLflow isn't installed or fails, yields None and continues.
    """
    if mlflow is None:
        yield None
        return

    try:
        # Ensure an experiment exists & is selected (creates if missing)
        mlflow.set_experiment(experiment_name or DEFAULT_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name) as run:
            if params:
                try:
                    mlflow.log_params({
                        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                        for k, v in params.items()
                    })
                except Exception:
                    pass
            yield run
    except Exception:
        # fall back gracefully
        yield None

def log_params(params: Dict):
    if mlflow is None:
        return
    try:
        mlflow.log_params({
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in params.items()
        })
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
    Best-effort: if MLflow or mlflow.keras are not available the call is a no-op.
    """
    if mlflow is None:
        return
    try:
        # If a model instance is provided, prefer mlflow.keras.log_model
        if hasattr(model_or_path, "save"):
            try:
                mlflow.keras.log_model(model_or_path, artifact_path=artifact_path)
                return
            except Exception:
                pass

        # Otherwise treat as a path and log the artifact (directory or file)
        path = str(model_or_path)
        if os.path.isdir(path):
            mlflow.log_artifacts(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception:
        pass

# --- Keras callback unchanged below ---
try:
    import tensorflow as _tf  # noqa: N812
    TF_AVAILABLE = True
except Exception:
    _tf = None  # type: ignore
    TF_AVAILABLE = False

if TF_AVAILABLE:
    class KerasModelLoggerCallback(_tf.keras.callbacks.Callback):
        """Keras callback that logs epoch metrics to MLflow."""
        def on_epoch_end(self, epoch, logs=None):
            if not logs or mlflow is None:
                return
            try:
                for k, v in logs.items():
                    mlflow.log_metric(k, float(v), step=epoch)
            except Exception:
                pass
else:
    class KerasModelLoggerCallback(object):
        def __init__(self, *args, **kwargs):
            pass
