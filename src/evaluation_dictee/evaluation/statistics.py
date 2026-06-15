"""Outils statistiques pour une évaluation rigoureuse.

Deux outils principaux :

1. **Intervalle de Wilson** pour les proportions (accord par item). Plus fiable
   que l'intervalle normal quand n est petit ou la proportion proche de 0/1.

2. **Bootstrap par grappes (cluster bootstrap)** pour les métriques globales.
   Point méthodologique important : les 83 items d'une même copie ne sont PAS
   indépendants (même élève, même écriture, même niveau). Un bootstrap naïf
   par item sous-estimerait l'incertitude. On rééchantillonne donc les COPIES
   entières, pas les items.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfidenceInterval:
    """Un intervalle de confiance avec sa valeur ponctuelle.

    Attributes:
        estimate: valeur observée.
        lower: borne inférieure.
        upper: borne supérieure.
        level: niveau de confiance (ex. 0.95).
    """

    estimate: float
    lower: float
    upper: float
    level: float = 0.95


def wilson_interval(successes: int, n: int, level: float = 0.95) -> ConfidenceInterval:
    """Intervalle de Wilson pour une proportion.

    Args:
        successes: nombre de succès (ex. items en accord).
        n: nombre total d'essais.
        level: niveau de confiance.

    Returns:
        L'intervalle (estimate = proportion observée).

    Raises:
        ValueError: si n == 0.
    """
    if n == 0:
        raise ValueError("n doit être > 0.")
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ConfidenceInterval(
        estimate=p, lower=max(0.0, centre - half), upper=min(1.0, centre + half), level=level
    )


def cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    cluster_col: str = "copy_id",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Bootstrap par grappes : rééchantillonne les copies, pas les items.

    Args:
        df: DataFrame des prédictions (une ligne par item × copie).
        metric_fn: fonction qui calcule la métrique sur un DataFrame
            (ex. lambda d: cohen_kappa_score(d["y_true"], d["y_pred"])).
        cluster_col: colonne identifiant la grappe (la copie).
        n_boot: nombre de rééchantillonnages.
        level: niveau de confiance.
        seed: graine aléatoire pour la reproductibilité.

    Returns:
        Intervalle percentile autour de la métrique observée.
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    groups = {c: g for c, g in df.groupby(cluster_col)}

    estimate = float(metric_fn(df))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat([groups[c] for c in drawn], ignore_index=True)
        samples[b] = metric_fn(boot_df)

    alpha = (1 - level) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return ConfidenceInterval(
        estimate=estimate, lower=float(lower), upper=float(upper), level=level
    )
