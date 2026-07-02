"""Orchestration d'un run de benchmark : données → modèle → métriques.

Ce module est le cœur du projet. Il enchaîne le chargement des données, l'appel
au modèle copie par copie, la normalisation des codes selon le schéma choisi, et
le calcul des métriques. Tout est piloté par une `ExperimentConfig`.

Les prédictions détaillées (une ligne JSON par item × copie) sont sauvegardées dans
data/processed/<run_name>_predictions.jsonl pour analyse ultérieure via
evaluation/report.py et le notebook 03_analyse_resultats.ipynb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data import grid
from evaluation_dictee.data.loaders import Copy, load_dataset
from evaluation_dictee.data.reference import load_grid
from evaluation_dictee.evaluation.metrics import ScoringMetrics, compute_scoring_metrics
from evaluation_dictee.models.base import Scorer
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Résultat d'un run : métriques + prédictions et labels alignés.

    Attributes:
        metrics: métriques globales (accord, kappa...).
        y_true: codes experts normalisés, aplatis (toutes copies × tous items).
        y_pred: codes prédits normalisés, alignés sur y_true.
        confidences: confiance par item (None si indisponible).
        item_ids: identifiant de l'item pour chaque position (pour l'analyse/item).
        copy_ids: identifiant de la copie pour chaque position.
        predictions_path: chemin du fichier JSONL sauvegardé (None si non sauvegardé).
    """

    metrics: ScoringMetrics
    y_true: list[str]
    y_pred: list[str]
    confidences: list[float | None]
    item_ids: list[str] = field(default_factory=list)
    copy_ids: list[str] = field(default_factory=list)
    predictions_path: Path | None = None
    non_transcribed: list[str] = field(default_factory=list)


def run_benchmark(
    config: ExperimentConfig,
    scorer: Scorer,
    output_dir: str | Path = "data/processed",
) -> BenchmarkResult:
    """Exécute un benchmark complet pour une config et un modèle donnés.

    Args:
        config: configuration de l'expérience.
        scorer: modèle implémentant l'interface Scorer.
        output_dir: dossier où sauvegarder le fichier de prédictions détaillées.

    Returns:
        Les métriques et les vecteurs alignés (vrais codes, prédictions, confiances).
    """
    logger.info("Labels : %s", config.data.labels_path)
    logger.info("Images : %s", config.data.images_path)

    copies: list[Copy] = load_dataset(
        images_dir=config.data.images_path,
        labels_csv=config.data.labels_path,
        limit=config.data.limit,
    )

    # Échec explicite si aucune copie — évite la boucle silencieuse
    if not copies:
        raise RuntimeError(
            "Aucune copie chargée. Causes possibles :\n"
            f"  1. Le CSV est inaccessible : {config.data.labels_path}\n"
            f"  2. Les images sont introuvables : {config.data.images_path}\n"
            "     (les noms dans le CSV doivent correspondre aux fichiers du dossier)\n"
            "  3. Les identifiants S3 ne sont pas configurés.\n"
            'Diagnostic : python -c "'
            "from evaluation_dictee.data.loaders import load_labels; "
            f"labels=load_labels('{config.data.labels_path}'); "
            "print(len(labels), 'copies dans le CSV')\""
        )

    logger.info("%d copies chargées — début de l'évaluation.", len(copies))
    reference = load_grid(config.data.grid_path).reference_text
    scheme = config.grid.scheme

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float | None] = []
    item_ids: list[str] = []
    copy_ids: list[str] = []
    records: list[dict] = []

    non_transcrites: list[str] = []
    for copy in tqdm(copies, desc=f"Évaluation ({config.name})"):
        prediction = scorer.score_copy(copy, reference)

        # Copie non transcrite (réponse vide même après retries) : on l'écarte des
        # métriques de performance, mais on la consigne pour inspection.
        if not prediction.transcribed:
            non_transcrites.append(copy.copy_id)
            logger.warning(
                "Copie non transcrite après %d tentative(s) : %s (exclue des métriques).",
                prediction.n_attempts,
                copy.copy_id,
            )
            continue

        pred_by_id = {it.item_id: it for it in prediction.items}

        for item_id, expert_code in zip(copy.item_ids, copy.expert_codes, strict=True):
            pred = pred_by_id.get(item_id)
            true_code = grid.normalize(expert_code, scheme)
            pred_code = grid.normalize(pred.code, scheme) if pred else "?"
            conf = pred.confidence if pred else 0.0

            y_true.append(true_code)
            y_pred.append(pred_code)
            confidences.append(conf)
            item_ids.append(item_id)
            copy_ids.append(copy.copy_id)
            records.append(
                {
                    "copy_id": copy.copy_id,
                    "item_id": item_id,
                    "y_true": true_code,
                    "y_pred": pred_code,
                    "confidence": conf,
                    "transcription": pred.transcription if pred else None,
                    "raw_transcription": prediction.raw_transcription,
                }
            )

    # Sauvegarde des prédictions détaillées (une ligne JSON par item × copie)
    out_path = Path(output_dir) / f"{config.name}_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Prédictions sauvegardées : %s", out_path)
    # Garde-fou : modèle et experts doivent coder dans le MÊME jeu de modalités.
    # Après normalisation, tout code hors de l'alphabet attendu signale une
    # incohérence (ex. prétraitement expert oublié, prompt non aligné sur le schéma).
    attendus = grid.allowed_codes(scheme)
    codes_vus = set(y_true) | set(y_pred)
    intrus = codes_vus - attendus - {"?"}  # "?" = réponse modèle non parsée, traité à part
    if intrus:
        logger.warning(
            "Codes hors du schéma '%s' (attendu %s) détectés : %s. "
            "Vérifier le prétraitement des codes experts et le prompt du modèle.",
            scheme,
            sorted(attendus),
            sorted(intrus),
        )

    if non_transcrites:
        logger.warning(
            "%d copie(s) non transcrite(s) exclue(s) des métriques : %s",
            len(non_transcrites),
            non_transcrites,
        )

    metrics = compute_scoring_metrics(y_true, y_pred)
    return BenchmarkResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        item_ids=item_ids,
        copy_ids=copy_ids,
        predictions_path=out_path,
        non_transcribed=non_transcrites,
    )
