"""Orchestration d'un run de benchmark : données -> modèle -> métriques, via `ExperimentConfig`.

Prédictions écrites dans data/processed/<run_name>_predictions.jsonl (une ligne par item x copie).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data import reference
from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.data.loaders import Copy, load_dataset
from evaluation_dictee.evaluation.metrics import ScoringMetrics, compute_scoring_metrics
from evaluation_dictee.models.base import Scorer
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking import copy_trace

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Résultat d'un run : métriques + prédictions/labels alignés (y_true/y_pred aplatis)."""

    metrics: ScoringMetrics
    y_true: list[str]
    y_pred: list[str]
    confidences: list[float | None]
    item_ids: list[str] = field(default_factory=list)
    copy_ids: list[str] = field(default_factory=list)
    predictions_path: Path | None = None
    non_transcribed: list[str] = field(default_factory=list)


def _load_processed_copy_ids(predictions_path: Path) -> set[str]:
    """Renvoie les copy_id déjà présents dans un fichier de prédictions (pour reprendre un run).

    Args:
        predictions_path: chemin du fichier JSONL de prédictions.

    Returns:
        L'ensemble des copy_id déjà traités (vide si le fichier n'existe pas).
    """
    if not predictions_path.exists():
        return set()
    processed: set[str] = set()
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                processed.add(rec["copy_id"])
            except (json.JSONDecodeError, KeyError):
                # Ligne tronquée par un crash : ignorée.
                continue
    return processed


def run_benchmark(
    config: ExperimentConfig,
    scorer: Scorer,
    output_dir: str | Path = "data/processed",
) -> BenchmarkResult:
    """Exécute un benchmark complet pour une config et un modèle donnés.

    Écriture incrémentale (fsync par copie) + reprise auto depuis le JSONL existant ;
    une erreur API sur une copie est loggée dans failed_copies.txt et le run continue.
    Pour repartir de zéro : supprimer le JSONL ou changer `config.name`.

    Args:
        config: configuration de l'expérience (données, grille, nom du run).
        scorer: modèle chargé de coder chaque copie.
        output_dir: dossier où écrire les prédictions et les fichiers d'échec.

    Returns:
        Le résultat du run : métriques agrégées, labels/prédictions alignés et
        chemin du fichier de prédictions.

    Raises:
        RuntimeError: si aucune copie n'a pu être chargée.
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
            'Diagnostic : uv run python -c "'
            "from evaluation_dictee.data.loaders import load_labels; "
            f"labels=load_labels('{config.data.labels_path}'); "
            "print(len(labels), 'copies dans le CSV')\""
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config.name}_predictions.jsonl"
    failed_path = output_dir / f"{config.name}_failed_copies.txt"

    processed = _load_processed_copy_ids(out_path)
    if processed:
        logger.info(
            "Reprise détectée : %d copies déjà traitées dans %s. On saute ces copies.",
            len(processed),
            out_path,
        )
    copies_a_traiter = [c for c in copies if c.copy_id not in processed]
    logger.info(
        "%d copies au total, %d à traiter (%d déjà faites).",
        len(copies),
        len(copies_a_traiter),
        len(processed),
    )

    reference_text = load_grid(config.data.grid_path).reference_text
    scheme = config.grid.scheme

    non_transcrites: list[str] = []
    failed_copies: list[tuple[str, str]] = []  # (copy_id, message d'erreur)

    # Mode APPEND : conserve les copies déjà traitées lors d'une reprise.
    with open(out_path, "a", encoding="utf-8") as f_pred:
        for copy in tqdm(copies_a_traiter, desc=f"Évaluation ({config.name})"):
            # Une trace Langfuse par copie (no-op si Langfuse indisponible, trace vaut None).
            with copy_trace(copy) as trace:
                # try/except par copie : une erreur API sur une copie ne perd pas les précédentes.
                try:
                    prediction = scorer.score_copy(copy, reference_text)
                except Exception as exc:  # noqa: BLE001 — on veut TOUT rattraper ici
                    logger.error(
                        "Échec sur la copie %s : %s. On passe à la suivante.",
                        copy.copy_id,
                        exc,
                    )
                    failed_copies.append((copy.copy_id, str(exc)))
                    if trace is not None:
                        trace.update(level="ERROR", status_message=str(exc))
                    continue

                if not prediction.transcribed:
                    non_transcrites.append(copy.copy_id)
                    logger.warning(
                        "Copie non transcrite après %d tentative(s) : %s (exclue des métriques).",
                        prediction.n_attempts,
                        copy.copy_id,
                    )
                    if trace is not None:
                        trace.update(
                            level="WARNING",
                            status_message="copie non transcrite",
                            output={"transcribed": False, "n_attempts": prediction.n_attempts},
                        )
                    continue

                pred_by_id = {it.item_id: it for it in prediction.items}

                n_items_copie = 0
                n_accord = 0
                for item_id, expert_code in zip(copy.item_ids, copy.expert_codes, strict=True):
                    pred = pred_by_id.get(item_id)
                    true_code = reference.normalize(expert_code, scheme)
                    pred_code = reference.normalize(pred.code, scheme) if pred else "?"
                    conf = pred.confidence if pred else 0.0
                    record = {
                        "copy_id": copy.copy_id,
                        "item_id": item_id,
                        "y_true": true_code,
                        "y_pred": pred_code,
                        "confidence": conf,
                        "transcription": pred.transcription if pred else None,
                        "comparaison": pred.comparaison if pred else None,
                        "raw_transcription": prediction.raw_transcription,
                    }
                    f_pred.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_items_copie += 1
                    if pred_code == true_code:
                        n_accord += 1

                # Flush + fsync : garantit que la copie écrite survit à un crash ultérieur.
                f_pred.flush()
                os.fsync(f_pred.fileno())

                if trace is not None and n_items_copie:
                    accord_copie = n_accord / n_items_copie
                    trace.update(output={"n_items": n_items_copie, "raw_agreement": accord_copie})
                    trace.score_trace(name="raw_agreement", value=accord_copie)

    if failed_copies:
        with open(failed_path, "w", encoding="utf-8") as f:
            for cid, msg in failed_copies:
                f.write(f"{cid}\t{msg}\n")
        logger.warning(
            "%d copie(s) en échec, listées dans %s (le run continuera à les "
            "reprendre au prochain lancement).",
            len(failed_copies),
            failed_path,
        )

    logger.info("Prédictions sauvegardées : %s", out_path)

    # Recharger tout le JSONL (copies de ce run + reprises) pour construire les métriques.
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float | None] = []
    item_ids: list[str] = []
    copy_ids: list[str] = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            y_true.append(rec["y_true"])
            y_pred.append(rec["y_pred"])
            confidences.append(rec.get("confidence"))
            item_ids.append(rec["item_id"])
            copy_ids.append(rec["copy_id"])
    # Garde-fou : après normalisation, un code hors de l'alphabet attendu signale une
    # incohérence (prétraitement expert oublié, prompt non aligné sur le schéma).
    attendus = reference.allowed_codes(scheme)
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
