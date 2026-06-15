"""Définition de la grille de codage de la dictée et logique de simplification.

Voir CLAUDE.md §3. La grille complète distingue 6 codes ; la grille simplifiée
(cible principale du projet) regroupe les erreurs de mots en un seul code.
"""

from __future__ import annotations

# Codes de la grille complète
CODE_CORRECT = "1"
CODE_ERR_LEXICALE = "3"
CODE_ERR_GRAMMATICALE = "4"
CODE_ERR_LES_DEUX = "5"
CODE_ERR_PONCT = "9"  # documenté pour la ponctuation (et observé sur des mots)
CODE_ABSENT = "0"

# Codes de la grille simplifiée
SIMPLE_CORRECT = "1"
SIMPLE_ERREUR = "9"  # toute erreur, quel qu'en soit le type
SIMPLE_ABSENT = "0"

# Ensemble des codes considérés comme « une erreur » (hors absent/correct)
_CODES_ERREUR = {CODE_ERR_LEXICALE, CODE_ERR_GRAMMATICALE, CODE_ERR_LES_DEUX, CODE_ERR_PONCT}

# Alphabet de codes attendu pour chaque schéma (après normalisation).
# Sert de référence pour vérifier que modèle et experts codent dans le même jeu.
ALLOWED_CODES = {
    "simplifiee": {SIMPLE_CORRECT, SIMPLE_ERREUR, SIMPLE_ABSENT},  # {1, 9, 0}
    "complete": {
        CODE_CORRECT,
        CODE_ERR_LEXICALE,
        CODE_ERR_GRAMMATICALE,
        CODE_ERR_LES_DEUX,
        CODE_ERR_PONCT,
        CODE_ABSENT,  # {1,3,4,5,9,0}
    },
}


def allowed_codes(scheme: str) -> set[str]:
    """Renvoie l'ensemble des codes valides après normalisation pour un schéma.

    Args:
        scheme: "simplifiee" ou "complete".

    Returns:
        L'ensemble des codes attendus (ex. {"1", "9", "0"} pour simplifiee).

    Raises:
        ValueError: si le schéma est inconnu.
    """
    if scheme not in ALLOWED_CODES:
        raise ValueError(f"Schéma inconnu : {scheme!r}. Attendu : {set(ALLOWED_CODES)}.")
    return ALLOWED_CODES[scheme]


def to_simplified(code: str) -> str:
    """Convertit un code de la grille complète vers la grille simplifiée 1/9/0.

    Args:
        code: code de la grille complète ("1", "3", "4", "5", "9" ou "0").

    Returns:
        "1" (correct), "9" (erreur quelconque) ou "0" (absent).
    """
    code = code.strip()
    if code == CODE_ABSENT:
        return SIMPLE_ABSENT
    if code == CODE_CORRECT:
        return SIMPLE_CORRECT
    if code in _CODES_ERREUR:
        return SIMPLE_ERREUR
    # Code inattendu : on le renvoie tel quel pour qu'il soit visible dans l'analyse
    return code


def normalize(code: str, scheme: str) -> str:
    """Normalise un code selon le schéma cible.

    Args:
        code: code brut.
        scheme: "simplifiee" ou "complete".

    Returns:
        Le code normalisé.
    """
    if scheme == "simplifiee":
        return to_simplified(code)
    return code.strip()
