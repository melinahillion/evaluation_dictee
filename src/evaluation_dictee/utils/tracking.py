"""Suivi d'expériences avec Langfuse (traces, prompts, scores).

Structure conforme aux recommandations Langfuse pour un benchmark :

    session  (= un lancement)  →  trace  (= une copie)  →  generation  (= appel LLM)

- `experiment_run` ouvre le contexte du run. Il rattache un `session_id` UNIQUE
  PAR LANCEMENT (nom du run + horodatage), un `user_id` (le namespace Kubernetes
  de l'utilisateur, cf. `KUBERNETES_NAMESPACE` sur Onyxia) et les paramètres de
  l'expérience (modèle, méthode, schéma, corpus) en tags/metadata, PROPAGÉS à
  toutes les traces créées pendant le run (`propagate_attributes`). Chaque exécution de l'évaluation
  ouvre ainsi une NOUVELLE session Langfuse : relancer la même config (reprise
  après crash comprise) ne réutilise pas la session précédente, qui est close de
  fait — plus aucune trace ne s'y rattache. Le nom du run reste porté en tag/metadata
  (`run:<nom>`) pour regrouper au besoin les lancements d'une même config.
  Il n'ouvre volontairement PAS de trace englobante :
  chaque copie devient ainsi une trace autonome — inspectable individuellement,
  avec ses propres entrée/sortie et scores — regroupée dans la session du run.
  (Une trace unique de plusieurs milliers de générations serait ingérable dans
  l'UI et ferait perdre l'inspection copie par copie.)
- `copy_trace` ouvre une trace par copie ; les générations LLM émises par
  `VLMScorer` (client `langfuse.openai`) s'y imbriquent automatiquement.
- `log_metrics` enregistre les métriques agrégées du run à la fois en metadata ET
  en Scores Langfuse (onglet « Scores » de l'UI), sur une trace de synthèse
  rattachée à la session.

Tout est optionnel : si Langfuse est indisponible ou mal configuré, les fonctions
deviennent silencieuses (no-op) afin de ne pas bloquer un développement local.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langfuse import get_client, propagate_attributes

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.utils.logging import get_logger

if TYPE_CHECKING:
    from langfuse._client.span import LangfuseSpan

    from evaluation_dictee.data.loaders import Copy

logger = get_logger(__name__)


def _run_metadata(config: ExperimentConfig) -> dict[str, str]:
    """Paramètres de l'expérience, en chaînes US-ASCII (contrainte Langfuse metadata)."""
    return {
        "model": config.model.name,
        "method": config.prompt.method,
        "scheme": config.grid.scheme,
        "n_few_shot": str(config.prompt.n_few_shot),
        "corpus": config.data.corpus,
    }


def _run_tags(config: ExperimentConfig) -> list[str]:
    """Tags de filtrage/regroupement dans l'UI Langfuse."""
    return [
        config.prompt.method,
        config.data.corpus,
        f"scheme:{config.grid.scheme}",
        f"model:{config.model.name}",
        # Le session_id étant unique par lancement, ce tag regroupe tous les
        # lancements d'une même config dans l'UI.
        f"run:{config.name}",
    ]


def _user_id() -> str | None:
    """Identifiant d'utilisateur Langfuse, basé sur le namespace Kubernetes.

    Sur le SSP Cloud (Onyxia), chaque utilisateur dispose d'un namespace dédié
    (variable `KUBERNETES_NAMESPACE`, typiquement « user-<login> »), qui sert
    d'identifiant naturel pour attribuer les traces dans l'UI Langfuse.

    Returns:
        La valeur de `KUBERNETES_NAMESPACE`, ou None hors de ce contexte (le
        `user_id` est alors simplement omis des traces).
    """
    return os.environ.get("KUBERNETES_NAMESPACE") or None


def _launch_session_id(config: ExperimentConfig) -> str:
    """Identifiant de session Langfuse UNIQUE PAR LANCEMENT (nom du run + horodatage).

    Chaque exécution de l'évaluation ouvre ainsi une nouvelle session, même en
    relançant la même config. Format trié chronologiquement : « <nom>-AAAAMMJJ-HHMMSS ».

    Args:
        config: configuration de l'expérience (fournit le nom du run).

    Returns:
        L'identifiant de session du lancement courant.
    """
    return f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"


@contextmanager
def experiment_run(config: ExperimentConfig) -> Iterator[str]:
    """Ouvre le contexte Langfuse d'un run de benchmark, dans une session neuve.

    Génère un `session_id` unique pour CE lancement (voir `_launch_session_id`) et
    le rattache — avec le `user_id` (namespace Kubernetes, voir `_user_id`) et les
    paramètres de l'expérience — en tags/metadata propagés à toutes les traces
    créées dans ce contexte. Chaque copie évaluée devient alors
    une trace autonome regroupée dans cette session, et ses appels LLM s'y
    imbriquent (voir `copy_trace`). Relancer la même config ouvre une nouvelle
    session (l'ancienne est close de fait).

    Args:
        config: configuration de l'expérience (nom du run, paramètres).

    Yields:
        Le `session_id` du lancement courant (utile pour le journaliser).
    """
    session_id = _launch_session_id(config)
    user_id = _user_id()
    logger.info("Langfuse : nouvelle session « %s » (user_id=%s).", session_id, user_id)
    metadata = {**_run_metadata(config), "run": config.name}
    try:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=_run_tags(config),
            metadata=metadata,
        ):
            yield session_id
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        logger.warning("Langfuse indisponible : le run ne sera pas tracé.", exc_info=True)
        yield session_id


@contextmanager
def copy_trace(copy: Copy) -> Iterator[LangfuseSpan | None]:
    """Ouvre une trace Langfuse pour une copie (à utiliser dans `experiment_run`).

    Les générations LLM émises pendant l'évaluation de la copie (client
    `langfuse.openai` de `VLMScorer`, tentatives successives comprises)
    s'imbriquent automatiquement sous cette trace. L'appelant complète l'objet
    renvoyé : `.update(output=..., level=...)`, `.score_trace(...)`.

    Args:
        copy: copie évaluée (fournit l'identifiant et l'entrée de la trace).

    Yields:
        La trace (span racine de la copie), ou None si Langfuse est indisponible.
    """
    langfuse = get_client()
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"copy:{copy.copy_id}",
            input={"copy_id": copy.copy_id, "n_items": len(copy.item_ids)},
        ) as span:
            yield span
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        logger.warning("Trace de la copie %s non créée (Langfuse indisponible).", copy.copy_id)
        yield None


def log_metrics(config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    """Enregistre les métriques agrégées du run dans Langfuse.

    Crée une trace de synthèse (« run-summary:<run> ») rattachée à la session du
    run, portant les métriques à la fois en metadata ET en Scores Langfuse
    (onglet « Scores » de l'UI). No-op si Langfuse est indisponible.

    Args:
        config: configuration de l'expérience (nomme la trace, rattache à la session).
        metrics: dictionnaire {nom: valeur} de métriques scalaires.
    """
    langfuse = get_client()
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    metadata = {**_run_metadata(config), **{k: str(v) for k, v in metrics.items()}}
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"run-summary:{config.name}",
            output=numeric,
            metadata=metadata,
        ):
            for name, value in numeric.items():
                langfuse.score_current_trace(name=name, value=value, data_type="NUMERIC")
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        logger.warning("Métriques non enregistrées dans Langfuse (indisponible).", exc_info=True)
