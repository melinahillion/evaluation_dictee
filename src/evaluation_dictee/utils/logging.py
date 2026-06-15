"""Configuration du logging avec rich pour une sortie lisible."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Renvoie un logger configuré avec un affichage enrichi (rich).

    Args:
        name: nom du logger (typiquement __name__).

    Returns:
        Le logger prêt à l'emploi.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
