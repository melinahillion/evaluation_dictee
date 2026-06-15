"""Calibration de la confiance et courbe de renvoi humain.

Livrable décisionnel central du projet (CLAUDE.md §4) : pour chaque seuil de
confiance, on mesure la part d'items renvoyés à un correcteur humain et le taux
d'erreur résiduel sur les items conservés (auto-validés par le modèle).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReferralPoint:
    """Un point de la courbe renvoi-humain / erreur-résiduelle.

    Attributes:
        threshold: seuil de confiance ; les items sous ce seuil sont renvoyés.
        human_referral_rate: part d'items renvoyés à l'humain.
        residual_error_rate: taux d'erreur sur les items auto-validés.
        n_auto_validated: nombre d'items auto-validés.
    """

    threshold: float
    human_referral_rate: float
    residual_error_rate: float
    n_auto_validated: int


def referral_curve(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float | None],
    thresholds: list[float] | None = None,
) -> list[ReferralPoint]:
    """Calcule la courbe « taux de renvoi humain vs taux d'erreur résiduel ».

    Les items dont la confiance est strictement inférieure au seuil sont
    « renvoyés à l'humain » et exclus du calcul d'erreur résiduelle. Les items
    sans confiance disponible sont toujours renvoyés.

    Args:
        y_true: codes experts.
        y_pred: codes prédits.
        confidences: confiance par item (None = toujours renvoyé).
        thresholds: seuils à tester (par défaut 0.0, 0.1, ..., 1.0).

    Returns:
        Liste de points de la courbe, un par seuil.
    """
    if thresholds is None:
        thresholds = [i / 10 for i in range(11)]

    n_total = len(y_true)
    points: list[ReferralPoint] = []

    for threshold in thresholds:
        kept_true: list[str] = []
        kept_pred: list[str] = []
        for true_code, pred_code, conf in zip(y_true, y_pred, confidences, strict=True):
            if conf is not None and conf >= threshold:
                kept_true.append(true_code)
                kept_pred.append(pred_code)

        n_kept = len(kept_true)
        n_referred = n_total - n_kept
        if n_kept > 0:
            errors = sum(1 for a, b in zip(kept_true, kept_pred, strict=True) if a != b)
            residual = errors / n_kept
        else:
            residual = 0.0

        points.append(
            ReferralPoint(
                threshold=threshold,
                human_referral_rate=n_referred / n_total,
                residual_error_rate=residual,
                n_auto_validated=n_kept,
            )
        )
    return points


@dataclass
class ReliabilityBin:
    """Une tranche de confiance pour le diagramme de fiabilité.

    Attributes:
        bin_lower: borne inférieure de la tranche de confiance.
        bin_upper: borne supérieure.
        mean_confidence: confiance moyenne des items de la tranche.
        accuracy: taux d'accord observé dans la tranche.
        n: nombre d'items dans la tranche.
    """

    bin_lower: float
    bin_upper: float
    mean_confidence: float
    accuracy: float
    n: int


def reliability_bins(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float | None],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Regroupe les items par tranche de confiance et mesure l'accord réel.

    Un modèle bien calibré a accuracy ≈ mean_confidence dans chaque tranche
    (les points du diagramme de fiabilité sont sur la diagonale).

    Args:
        y_true: codes experts.
        y_pred: codes prédits.
        confidences: confiance par item (None = ignoré).
        n_bins: nombre de tranches entre 0 et 1.

    Returns:
        Une liste de tranches non vides.
    """
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        confs, corrects = [], []
        for t, p, c in zip(y_true, y_pred, confidences, strict=True):
            if c is None:
                continue
            # dernière tranche inclusive à droite pour capter c == 1.0
            if (lo <= c < hi) or (b == n_bins - 1 and c == hi):
                confs.append(c)
                corrects.append(t == p)
        if confs:
            bins.append(
                ReliabilityBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    mean_confidence=sum(confs) / len(confs),
                    accuracy=sum(corrects) / len(corrects),
                    n=len(confs),
                )
            )
    return bins


def expected_calibration_error(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float | None],
    n_bins: int = 10,
) -> float:
    """ECE : écart moyen pondéré entre confiance annoncée et accord observé.

    0 = parfaitement calibré. Au-delà de ~0.1, le score de confiance n'est pas
    fiable tel quel et doit être recalibré avant d'être utilisé pour le seuil
    de renvoi humain.

    Args:
        y_true: codes experts.
        y_pred: codes prédits.
        confidences: confiance par item.
        n_bins: nombre de tranches.

    Returns:
        L'ECE (entre 0 et 1).
    """
    bins = reliability_bins(y_true, y_pred, confidences, n_bins)
    n_total = sum(b.n for b in bins)
    if n_total == 0:
        return float("nan")
    return sum(b.n * abs(b.accuracy - b.mean_confidence) for b in bins) / n_total
