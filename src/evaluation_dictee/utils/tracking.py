"""Suivi d'expériences avec MLflow.

Enregistre la config, les métriques et les artefacts d'un run. Conçu pour être
optionnel : si MLflow n'est pas configuré, les fonctions deviennent silencieuses
afin de ne pas bloquer un développement local.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False


@contextmanager
def experiment_run(config: ExperimentConfig) -> Iterator[None]:
    """Ouvre un run MLflow nommé d'après la config (no-op si MLflow absent).

    Args:
        config: configuration de l'expérience (sert de nom de run et de params).

    Yields:
        Rien ; le contexte gère le cycle de vie du run.
    """
    if not _MLFLOW_AVAILABLE:
        logger.warning("MLflow indisponible : le run ne sera pas tracé.")
        yield
        return

    with mlflow.start_run(run_name=config.name):
        mlflow.log_params(
            {
                "model": config.model.name,
                "method": config.prompt.method,
                "scheme": config.grid.scheme,
                "n_few_shot": config.prompt.n_few_shot,
                "corpus": config.data.corpus,
            }
        )
        yield


def log_metrics(metrics: dict[str, Any]) -> None:
    """Enregistre des métriques numériques dans le run courant (no-op si absent).

    Args:
        metrics: dictionnaire {nom: valeur} de métriques scalaires.
    """
    if not _MLFLOW_AVAILABLE:
        return
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(numeric)
