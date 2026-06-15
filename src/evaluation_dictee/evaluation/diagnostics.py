"""Diagnostic des écarts entre le modèle et l'annotateur.

Ce module ne calcule pas des métriques agrégées (voir report.py) mais cherche à
EXPLIQUER d'où viennent les désaccords, et à pointer les copies à inspecter
visuellement. Il répond à des questions précises :

- Y a-t-il un effet de POSITION ? (l'accord se dégrade-t-il vers la fin de la
  dictée — signature d'un décalage propagé par un oubli de mot)
- Le modèle est-il SUR-CONFIANT ? (confiance élevée sur des prédictions fausses)
- Quelles copies concentrent les désaccords (cas types à regarder) ?
- Le motif des désaccords ressemble-t-il à un DÉCALAGE (run de codes 9/0
  consécutifs) plutôt qu'à des erreurs isolées ?

Toutes les fonctions prennent le DataFrame de prédictions chargé par
report.load_predictions et renvoient des DataFrames.
"""

from __future__ import annotations

import pandas as pd


def add_position(df: pd.DataFrame, ordered_item_ids: list[str]) -> pd.DataFrame:
    """Ajoute une colonne `position` (rang de l'item dans la dictée, 1..N).

    Args:
        df: prédictions (copy_id, item_id, y_true, y_pred, confidence).
        ordered_item_ids: item_ids dans l'ordre du texte (depuis la grille).

    Returns:
        Copie du DataFrame avec colonnes `position` et `correct` (booléen).
    """
    rang = {iid: i + 1 for i, iid in enumerate(ordered_item_ids)}
    out = df.copy()
    out["position"] = out["item_id"].map(rang)
    out["correct"] = out["y_true"] == out["y_pred"]
    return out


def accuracy_by_position(
    df: pd.DataFrame, ordered_item_ids: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """Accord moyen par tranche de position dans la dictée.

    Un accord qui chute en fin de dictée est la signature d'un décalage propagé
    (oubli de mot non géré, fusion, ligne sautée...).

    Args:
        df: prédictions.
        ordered_item_ids: ordre des items.
        n_bins: nombre de tranches de position.

    Returns:
        DataFrame par tranche : position moyenne, accord, effectif.
    """
    d = add_position(df, ordered_item_ids)
    d["tranche"] = pd.cut(d["position"], bins=n_bins)
    g = d.groupby("tranche", observed=True).agg(
        position_moy=("position", "mean"),
        accord=("correct", "mean"),
        n=("correct", "size"),
    )
    return g.reset_index(drop=True)


def position_accuracy_correlation(df: pd.DataFrame, ordered_item_ids: list[str]) -> float:
    """Corrélation entre position et justesse (négative = dégradation en fin).

    Args:
        df: prédictions.
        ordered_item_ids: ordre des items.

    Returns:
        Coefficient de corrélation de Pearson entre position et `correct`.
    """
    d = add_position(df, ordered_item_ids)
    return float(d["position"].corr(d["correct"].astype(float)))


def per_copy_position_effect(df: pd.DataFrame, ordered_item_ids: list[str]) -> pd.DataFrame:
    """Pour chaque copie, compare l'accord 1re moitié vs 2de moitié de la dictée.

    Une chute marquée en 2de moitié sur une copie donnée suggère un décalage
    déclenché en cours de copie (typiquement un mot oublié par l'élève).

    Args:
        df: prédictions.
        ordered_item_ids: ordre des items.

    Returns:
        DataFrame par copie : accord_moitie1, accord_moitie2, chute (m1 - m2),
        trié par chute décroissante (les plus suspectes en tête).
    """
    d = add_position(df, ordered_item_ids)
    milieu = len(ordered_item_ids) / 2
    rows = []
    for copy_id, grp in d.groupby("copy_id"):
        m1 = grp[grp["position"] <= milieu]["correct"].mean()
        m2 = grp[grp["position"] > milieu]["correct"].mean()
        rows.append(
            {
                "copy_id": copy_id,
                "accord_moitie1": m1,
                "accord_moitie2": m2,
                "chute": m1 - m2,
            }
        )
    return pd.DataFrame(rows).set_index("copy_id").sort_values("chute", ascending=False)


def longest_disagreement_run(df: pd.DataFrame, ordered_item_ids: list[str]) -> pd.DataFrame:
    """Plus longue séquence de désaccords CONSÉCUTIFS par copie.

    Des erreurs isolées dispersées = vraies erreurs de codage.
    Une longue séquence ininterrompue de désaccords = signature d'un décalage
    (tous les items après un oubli sont faux jusqu'à un éventuel recalage).

    Args:
        df: prédictions.
        ordered_item_ids: ordre des items.

    Returns:
        DataFrame par copie : n_desaccords, plus_longue_sequence,
        position_debut_sequence. Trié par plus_longue_sequence décroissante.
    """
    d = add_position(df, ordered_item_ids).sort_values(["copy_id", "position"])
    rows = []
    for copy_id, grp in d.groupby("copy_id"):
        faux = (~grp["correct"]).tolist()
        positions = grp["position"].tolist()
        best_len, best_start = 0, None
        cur_len, cur_start = 0, None
        for f, pos in zip(faux, positions, strict=True):
            if f:
                if cur_len == 0:
                    cur_start = pos
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        rows.append(
            {
                "copy_id": copy_id,
                "n_desaccords": int(sum(faux)),
                "plus_longue_sequence": best_len,
                "position_debut_sequence": best_start,
            }
        )
    return (
        pd.DataFrame(rows).set_index("copy_id").sort_values("plus_longue_sequence", ascending=False)
    )


def overconfident_errors(df: pd.DataFrame, seuil_confiance: float = 0.9) -> pd.DataFrame:
    """Prédictions FAUSSES annoncées avec une confiance élevée.

    Ces cas sont les plus dangereux : le modèle se trompe sans le signaler, donc
    le renvoi humain ne les rattrape pas. Beaucoup de tels cas = confiance
    inexploitable (cohérent avec un ECE élevé).

    Args:
        df: prédictions.
        seuil_confiance: seuil au-dessus duquel la confiance est dite « élevée ».

    Returns:
        DataFrame des prédictions fausses et sur-confiantes (copy_id, item_id,
        y_true, y_pred, confidence), triées par confiance décroissante.
    """
    faux = df[df["y_true"] != df["y_pred"]]
    surconf = faux[faux["confidence"].fillna(0) >= seuil_confiance]
    return surconf.sort_values("confidence", ascending=False).reset_index(drop=True)


def disagreement_hotspots(
    df: pd.DataFrame, ordered_item_ids: list[str], top: int = 5
) -> dict[str, pd.DataFrame]:
    """Sélectionne des copies-types à inspecter visuellement.

    Renvoie plusieurs « palmarès » complémentaires, chacun éclairant un mécanisme
    d'erreur différent :
    - `pire_accord` : copies au plus faible accord global.
    - `plus_longue_sequence` : copies avec le plus long run de désaccords
      (suspicion de décalage).
    - `plus_forte_chute` : copies dont l'accord s'effondre en 2de moitié.
    - `sur_confiance` : copies cumulant le plus d'erreurs sur-confiantes.

    Args:
        df: prédictions.
        ordered_item_ids: ordre des items.
        top: nombre de copies par palmarès.

    Returns:
        Dictionnaire de DataFrames (un par palmarès).
    """
    d = add_position(df, ordered_item_ids)

    accord_copie = (
        d.groupby("copy_id")["correct"].mean().sort_values().head(top).rename("accord").to_frame()
    )

    runs = longest_disagreement_run(df, ordered_item_ids).head(top)

    chute = per_copy_position_effect(df, ordered_item_ids).head(top)

    surconf = overconfident_errors(df)
    surconf_copie = (
        surconf.groupby("copy_id")
        .size()
        .sort_values(ascending=False)
        .head(top)
        .rename("n_erreurs_surconfiantes")
        .to_frame()
    )

    return {
        "pire_accord": accord_copie,
        "plus_longue_sequence": runs,
        "plus_forte_chute": chute,
        "sur_confiance": surconf_copie,
    }


def simulate_word_omission_shift(
    expert_codes: list[str],
    omission_position: int,
) -> dict[str, float]:
    """Teste l'effet d'un décalage dû à un mot oublié, sur des codes synthétiques.

    Outil pédagogique/de test : on part d'un codage parfait (le modèle reproduit
    l'expert), puis on simule un modèle qui, à partir de `omission_position`,
    décale toutes ses prédictions d'un cran (comme s'il n'avait pas vu qu'un mot
    était absent et avait aligné le mot suivant sur la position courante).

    On mesure l'accord résultant : s'il s'effondre, cela démontre qu'un simple
    oubli non géré peut détruire l'évaluation de toute la fin de copie.

    Args:
        expert_codes: codes experts d'une copie (vérité), dans l'ordre.
        omission_position: index (0-based) où l'élève a omis un mot.

    Returns:
        {"accord_sans_decalage": 1.0, "accord_avec_decalage": x} où x illustre
        la chute provoquée par le décalage.
    """
    n = len(expert_codes)
    # Modèle parfait : copie exacte
    parfait = list(expert_codes)
    accord_parfait = sum(a == b for a, b in zip(expert_codes, parfait, strict=True)) / n

    # Modèle décalé : à partir de l'omission, tout glisse d'un cran
    decale = list(expert_codes[:omission_position])
    decale += list(expert_codes[omission_position + 1 :])  # saute la position omise
    decale += ["1"]  # complète à droite (valeur arbitraire)
    decale = decale[:n]
    accord_decale = sum(a == b for a, b in zip(expert_codes, decale, strict=True)) / n

    return {
        "accord_sans_decalage": accord_parfait,
        "accord_avec_decalage": accord_decale,
        "position_omission": omission_position,
        "items_apres_omission": n - omission_position,
    }
