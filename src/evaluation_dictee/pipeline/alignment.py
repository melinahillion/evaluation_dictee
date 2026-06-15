"""Ré-alignement des prédictions du modèle sur les items attendus.

Filet de sécurité contre les décalages : même avec une consigne stricte, le modèle
peut scinder un mot (« re trouver ») ou en coller deux (« nousles »), ce qui décale
toutes les prédictions suivantes. Ce module recolle les prédictions du modèle sur
les items attendus de la grille, en s'appuyant sur les TRANSCRIPTIONS fournies par
le modèle (ce qu'il dit avoir lu) comparées aux mots attendus.

Principe : alignement de séquences (type Needleman-Wunsch simplifié) entre la suite
des mots attendus et la suite des transcriptions du modèle. On maximise la
correspondance mot attendu ↔ transcription, ce qui réabsorbe les insertions
(mot scindé) et les suppressions (mots collés).

Ce ré-alignement n'est appliqué que si un décalage est détecté ; sinon les
prédictions positionnelles du modèle sont conservées telles quelles.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass
class AlignedPrediction:
    """Prédiction d'un item après ré-alignement.

    Attributes:
        code: code attribué à l'item attendu (1/9/0, ou "?" si non aligné).
        transcription: transcription rattachée à cet item (peut être None).
        confidence: confiance associée (peut être None).
        realigned: True si l'item a été déplacé par le ré-alignement.
    """

    code: str
    transcription: str | None
    confidence: float | None
    realigned: bool = False


def _norm(s: str | None) -> str:
    """Normalise une chaîne pour la comparaison (minuscule, sans accents/espaces)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(s.lower().split())


def _similar(attendu: str, lu: str | None) -> float:
    """Score de similarité simple entre mot attendu et transcription (0..1)."""
    a, b = _norm(attendu), _norm(lu)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Similarité par préfixe commun + inclusion (capte « re trouver » vs « retrouver »)
    if a in b or b in a:
        return 0.8
    commun = 0
    for ca, cb in zip(a, b, strict=False):
        if ca == cb:
            commun += 1
        else:
            break
    return commun / max(len(a), len(b))


def needs_realignment(
    expected_words: list[str],
    transcriptions: list[str | None],
    seuil: float = 0.5,
) -> bool:
    """Détecte un décalage : la transcription positionnelle colle-t-elle aux mots ?

    Args:
        expected_words: mots attendus dans l'ordre.
        transcriptions: transcriptions du modèle, alignées positionnellement.
        seuil: en dessous de cette similarité moyenne, on suspecte un décalage.

    Returns:
        True si un ré-alignement est souhaitable.
    """
    paires = [
        _similar(a, t)
        for a, t in zip(expected_words, transcriptions, strict=False)
        if t is not None
    ]
    if not paires:
        return False
    return (sum(paires) / len(paires)) < seuil


def realign(
    expected_words: list[str],
    codes: list[str],
    transcriptions: list[str | None],
    confidences: list[float | None],
    gap_penalty: float = -0.3,
) -> list[AlignedPrediction]:
    """Ré-aligne les prédictions du modèle sur les mots attendus.

    Alignement de séquences entre `expected_words` et `transcriptions` maximisant
    la similarité. Chaque mot attendu reçoit le code/transcription du token modèle
    qui lui correspond le mieux ; les items attendus sans correspondance reçoivent
    le code "0" (absent, considéré non lu) avec une confiance nulle.

    Args:
        expected_words: mots attendus (longueur = nombre d'items de la grille).
        codes: codes prédits par le modèle (longueur = sortie du modèle).
        transcriptions: transcriptions du modèle (même longueur que codes).
        confidences: confiances (même longueur que codes).
        gap_penalty: pénalité d'insertion/suppression dans l'alignement.

    Returns:
        Une prédiction ré-alignée par mot attendu (longueur = len(expected_words)).
    """
    n, m = len(expected_words), len(codes)
    # Programmation dynamique : score[i][j] = meilleur alignement des i premiers
    # mots attendus avec les j premières prédictions.
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + _similar(expected_words[i - 1], transcriptions[j - 1])
            delete = score[i - 1][j] + gap_penalty  # mot attendu sans correspondance
            insert = score[i][j - 1] + gap_penalty  # token modèle en trop
            score[i][j] = max(match, delete, insert)

    # Backtrack
    aligned: list[AlignedPrediction | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        sim = _similar(expected_words[i - 1], transcriptions[j - 1])
        if score[i][j] == score[i - 1][j - 1] + sim:
            aligned[i - 1] = AlignedPrediction(
                code=codes[j - 1],
                transcription=transcriptions[j - 1],
                confidence=confidences[j - 1],
                realigned=(i - 1) != (j - 1),
            )
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] + gap_penalty:
            # mot attendu non lu → absent
            aligned[i - 1] = AlignedPrediction("0", None, 0.0, realigned=True)
            i -= 1
        else:
            j -= 1  # token modèle en trop : ignoré
    while i > 0:
        aligned[i - 1] = AlignedPrediction("0", None, 0.0, realigned=True)
        i -= 1

    return [a if a is not None else AlignedPrediction("?", None, 0.0, True) for a in aligned]


def _alignment_quality(expected_words: list[str], aligned: list[AlignedPrediction]) -> float:
    """Mesure la qualité d'un alignement : similarité moyenne mot attendu ↔ transcription.

    Args:
        expected_words: mots attendus.
        aligned: prédictions alignées (même longueur).

    Returns:
        Similarité moyenne dans [0, 1]. Plus c'est haut, meilleur est l'alignement.
    """
    sims = [_similar(w, a.transcription) for w, a in zip(expected_words, aligned, strict=False)]
    return sum(sims) / len(sims) if sims else 0.0


def realign_anchored(
    expected_words: list[str],
    codes: list[str],
    transcriptions: list[str | None],
    confidences: list[float | None],
) -> list[AlignedPrediction]:
    """Stratégie d'alignement par ancrage sur les correspondances exactes.

    Complémentaire de `realign` (Needleman-Wunsch global). On repère d'abord les
    « ancres » : positions où un mot attendu et une transcription coïncident
    exactement et de façon unique. Entre deux ancres, on répartit les prédictions
    proportionnellement. Cette approche est robuste quand quelques mots très
    distinctifs (ex. « Martine », « téléphoner ») jalonnent la dictée et limitent
    la propagation d'un décalage local.

    Args:
        expected_words: mots attendus.
        codes: codes prédits (ordre modèle).
        transcriptions: transcriptions (ordre modèle).
        confidences: confiances (ordre modèle).

    Returns:
        Une prédiction par mot attendu.
    """
    n = len(expected_words)
    m = len(codes)
    norm_exp = [_norm(w) for w in expected_words]
    norm_tr = [_norm(t) for t in transcriptions]

    # Ancres : mot attendu unique == transcription unique
    from collections import Counter

    cnt_exp, cnt_tr = Counter(norm_exp), Counter(norm_tr)
    anchors: list[tuple[int, int]] = []  # (index attendu, index modèle)
    for i, w in enumerate(norm_exp):
        if not w or cnt_exp[w] != 1 or cnt_tr.get(w, 0) != 1:
            continue
        j = norm_tr.index(w)
        anchors.append((i, j))
    anchors.sort()

    result: list[AlignedPrediction | None] = [None] * n
    # Segments délimités par les ancres ; alignement positionnel à l'intérieur
    bornes = [(-1, -1), *anchors, (n, m)]
    for (i0, j0), (i1, j1) in zip(bornes, bornes[1:], strict=False):
        # placer l'ancre i1 elle-même
        if 0 <= i1 < n and 0 <= j1 < m:
            result[i1] = AlignedPrediction(
                codes[j1], transcriptions[j1], confidences[j1], realigned=(i1 != j1)
            )
        # répartir le segment ouvert (i0, i1) ↔ (j0, j1) positionnellement
        seg_exp = list(range(i0 + 1, i1))
        seg_mod = list(range(j0 + 1, j1))
        for k, ie in enumerate(seg_exp):
            if k < len(seg_mod):
                jm = seg_mod[k]
                result[ie] = AlignedPrediction(
                    codes[jm], transcriptions[jm], confidences[jm], realigned=(ie != jm)
                )
            else:
                result[ie] = AlignedPrediction("0", None, 0.0, realigned=True)

    return [a if a is not None else AlignedPrediction("?", None, 0.0, True) for a in result]


def best_realignment(
    expected_words: list[str],
    codes: list[str],
    transcriptions: list[str | None],
    confidences: list[float | None],
) -> list[AlignedPrediction]:
    """Choisit la meilleure des stratégies d'alignement disponibles.

    Applique plusieurs approches (Needleman-Wunsch global et ancrage) et retient
    celle dont l'alignement colle le mieux aux mots attendus. Combine ainsi les
    forces de chacune pour rattraper un maximum de cas de décalage.

    Args:
        expected_words: mots attendus.
        codes: codes prédits (ordre modèle).
        transcriptions: transcriptions (ordre modèle).
        confidences: confiances (ordre modèle).

    Returns:
        Le meilleur alignement (une prédiction par mot attendu).
    """
    candidats = [
        realign(expected_words, codes, transcriptions, confidences),
        realign_anchored(expected_words, codes, transcriptions, confidences),
    ]
    return max(candidats, key=lambda a: _alignment_quality(expected_words, a))
