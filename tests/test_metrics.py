"""Tests des métriques de scoring."""

import pytest

from evaluation_dictee.evaluation.metrics import compute_scoring_metrics


def test_accord_parfait() -> None:
    y = ["1", "9", "0", "1"]
    metrics = compute_scoring_metrics(y, y)
    assert metrics.raw_agreement == 1.0
    assert metrics.cohen_kappa == 1.0
    assert metrics.n_items == 4


def test_accord_partiel() -> None:
    y_true = ["1", "1", "9", "0"]
    y_pred = ["1", "9", "9", "0"]
    metrics = compute_scoring_metrics(y_true, y_pred)
    assert metrics.raw_agreement == 0.75


def test_longueurs_differentes_leve_erreur() -> None:
    with pytest.raises(ValueError):
        compute_scoring_metrics(["1"], ["1", "9"])


def test_listes_vides_levent_erreur() -> None:
    with pytest.raises(ValueError):
        compute_scoring_metrics([], [])
