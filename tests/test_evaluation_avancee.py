"""Tests des outils statistiques et du rapport par item."""

import pandas as pd
import pytest

from evaluation_dictee.evaluation.calibration import (
    expected_calibration_error,
    reliability_bins,
)
from evaluation_dictee.evaluation.report import (
    disagreement_decomposition,
    per_copy_metrics,
    per_item_metrics,
)
from evaluation_dictee.evaluation.statistics import cluster_bootstrap, wilson_interval


# ── Wilson ────────────────────────────────────────────────────────────────────
def test_wilson_proportion_mediane() -> None:
    ci = wilson_interval(50, 100)
    assert ci.estimate == 0.5
    assert ci.lower < 0.5 < ci.upper
    assert 0.39 < ci.lower < 0.41  # valeur connue ≈ 0.404


def test_wilson_bornes_dans_0_1() -> None:
    ci = wilson_interval(100, 100)
    assert ci.upper <= 1.0
    ci0 = wilson_interval(0, 100)
    assert ci0.lower >= 0.0


def test_wilson_n_nul_leve_erreur() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


# ── Bootstrap par grappes ─────────────────────────────────────────────────────
def _df_jouet() -> pd.DataFrame:
    rows = []
    for copie in ["c1", "c2", "c3", "c4"]:
        for i in range(10):
            ok = i < 9  # 90 % d'accord
            rows.append(
                {
                    "copy_id": copie,
                    "item_id": f"it{i}",
                    "y_true": "1",
                    "y_pred": "1" if ok else "9",
                    "confidence": 0.9 if ok else 0.4,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_contient_estimation() -> None:
    df = _df_jouet()
    ci = cluster_bootstrap(df, lambda d: float((d["y_true"] == d["y_pred"]).mean()), n_boot=200)
    assert ci.lower <= ci.estimate <= ci.upper
    assert ci.estimate == 0.9


def test_cluster_bootstrap_reproductible() -> None:
    df = _df_jouet()
    fn = lambda d: float((d["y_true"] == d["y_pred"]).mean())  # noqa: E731
    ci1 = cluster_bootstrap(df, fn, n_boot=100, seed=7)
    ci2 = cluster_bootstrap(df, fn, n_boot=100, seed=7)
    assert (ci1.lower, ci1.upper) == (ci2.lower, ci2.upper)


# ── Rapport par item / copie ──────────────────────────────────────────────────
def test_per_item_metrics_colonnes() -> None:
    stats = per_item_metrics(_df_jouet())
    for col in ["accord", "accord_lo", "accord_hi", "rappel_erreur", "n_sur_correction"]:
        assert col in stats.columns
    assert (stats["accord_lo"] <= stats["accord"]).all()
    assert (stats["accord"] <= stats["accord_hi"]).all()


def test_sur_correction_et_sur_detection() -> None:
    df = pd.DataFrame(
        {
            "copy_id": ["c"] * 4,
            "item_id": ["i1", "i1", "i1", "i1"],
            # expert: erreur, modèle: correct → sur-correction (1 cas)
            # expert: correct, modèle: erreur → sur-détection (1 cas)
            "y_true": ["9", "1", "1", "9"],
            "y_pred": ["1", "9", "1", "9"],
            "confidence": [0.8] * 4,
        }
    )
    stats = per_item_metrics(df)
    assert stats.loc["i1", "n_sur_correction"] == 1
    assert stats.loc["i1", "n_sur_detection"] == 1
    # 2 erreurs expert, 1 retrouvée → rappel 0.5 ; 2 erreurs modèle, 1 confirmée → précision 0.5
    assert stats.loc["i1", "rappel_erreur"] == 0.5
    assert stats.loc["i1", "precision_erreur"] == 0.5


def test_per_copy_metrics() -> None:
    stats = per_copy_metrics(_df_jouet())
    assert len(stats) == 4
    assert (stats["accord"] == 0.9).all()


def test_disagreement_decomposition() -> None:
    deco = disagreement_decomposition(_df_jouet())
    assert deco["n"].sum() == 4  # un désaccord par copie
    assert deco["pct_desaccords"].sum() == pytest.approx(100.0)


# ── Calibration ───────────────────────────────────────────────────────────────
def test_ece_modele_parfaitement_calibre() -> None:
    # confiance 1.0 et toujours juste → ECE = 0
    ece = expected_calibration_error(["1"] * 10, ["1"] * 10, [1.0] * 10)
    assert ece == pytest.approx(0.0)


def test_ece_modele_surconfiant() -> None:
    # confiance 0.95 mais 50 % d'accord seulement → ECE ≈ 0.45
    y_true = ["1"] * 5 + ["9"] * 5
    y_pred = ["1"] * 10
    ece = expected_calibration_error(y_true, y_pred, [0.95] * 10)
    assert ece == pytest.approx(0.45, abs=0.01)


def test_reliability_bins_couvrent_les_donnees() -> None:
    bins = reliability_bins(["1", "1"], ["1", "9"], [0.2, 0.99], n_bins=10)
    assert sum(b.n for b in bins) == 2
