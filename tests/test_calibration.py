"""Tests de la courbe de renvoi humain basée sur la confiance."""

from evaluation_dictee.evaluation.calibration import referral_curve


def test_seuil_zero_garde_tout() -> None:
    y_true = ["1", "9", "1"]
    y_pred = ["1", "1", "1"]  # une erreur sur l'item 2
    conf = [0.9, 0.3, 0.8]
    points = {p.threshold: p for p in referral_curve(y_true, y_pred, conf)}
    # Au seuil 0, rien n'est renvoyé : erreur résiduelle = 1/3
    assert points[0.0].human_referral_rate == 0.0
    assert points[0.0].residual_error_rate == 1 / 3


def test_seuil_filtre_item_incertain() -> None:
    y_true = ["1", "9", "1"]
    y_pred = ["1", "1", "1"]
    conf = [0.9, 0.3, 0.8]
    points = {p.threshold: p for p in referral_curve(y_true, y_pred, conf)}
    # Au seuil 0.5, l'item incertain (et faux) est renvoyé : erreur résiduelle nulle
    assert points[0.5].residual_error_rate == 0.0
    assert points[0.5].human_referral_rate > 0.0


def test_confiance_absente_toujours_renvoyee() -> None:
    points = {p.threshold: p for p in referral_curve(["1"], ["1"], [None])}
    assert points[0.0].human_referral_rate == 1.0
