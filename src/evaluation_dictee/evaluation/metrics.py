"""Métriques de scoring : accord brut, kappa de Cohen, matrice de confusion.

Ces métriques reproduisent celles utilisées dans le test pilote (CLAUDE.md §5).
On s'appuie sur scikit-learn pour le kappa et la matrice de confusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import cohen_kappa_score, confusion_matrix


@dataclass
class ScoringMetrics:
    """Métriques d'accord entre prédictions et codes experts.

    Attributes:
        n_items: nombre d'items comparés.
        raw_agreement: proportion d'items identiques.
        cohen_kappa: kappa de Cohen (accord corrigé du hasard).
        labels: liste ordonnée des codes apparaissant.
        confusion: matrice de confusion (lignes = experts, colonnes = prédictions).
    """

    n_items: int
    raw_agreement: float
    cohen_kappa: float
    labels: list[str]
    confusion: list[list[int]]


def compute_scoring_metrics(y_true: list[str], y_pred: list[str]) -> ScoringMetrics:
    """Calcule les métriques d'accord entre codes experts et codes prédits.

    Args:
        y_true: codes experts (normalisés).
        y_pred: codes prédits (normalisés, même longueur que y_true).

    Returns:
        Les métriques d'accord.

    Raises:
        ValueError: si les deux listes n'ont pas la même longueur ou sont vides.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred doivent avoir la même longueur.")
    if not y_true:
        raise ValueError("Listes vides : aucune métrique calculable.")

    n = len(y_true)
    raw = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b) / n
    kappa = float(cohen_kappa_score(y_true, y_pred))

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return ScoringMetrics(
        n_items=n,
        raw_agreement=raw,
        cohen_kappa=kappa,
        labels=labels,
        confusion=matrix,
    )
