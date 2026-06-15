"""Chargement des imagettes et des labels experts (local ou S3).

Points d'attention :
- Les fichiers fournis par la DEPP sont des TIFF bi-level 1 bit (extension
  trompeuse ".png"). On les normalise systématiquement en niveaux de gris.
- Les chemins peuvent être locaux OU sur S3 (préfixe "s3://"). L'accès S3 passe
  par fsspec/s3fs, qui lit les identifiants depuis les variables d'environnement
  (AWS_ACCESS_KEY_ID, etc., voir .env). Sur le SSP Cloud, ces variables sont déjà
  injectées par Onyxia.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import fsspec
from PIL import Image


@dataclass
class Copy:
    """Une copie d'élève : son identifiant, son image et les codes experts.

    Attributes:
        copy_id: identifiant de la copie (nom de fichier image).
        image_path: chemin (local ou s3://) de l'image.
        expert_codes: codes de l'annotateur par item (vérité de terrain).
        item_ids: identifiants des items.
    """

    copy_id: str
    image_path: str
    expert_codes: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)


def _join(base: str, name: str) -> str:
    """Concatène un dossier (local ou s3://) et un nom de fichier."""
    return base.rstrip("/") + "/" + name


def load_image(path: str) -> Image.Image:
    """Charge une image (locale ou S3) en niveaux de gris.

    Gère les TIFF 1 bit déguisés en .png. La conversion en "L" (8 bits) évite
    les erreurs de décodage en aval et homogénéise le format d'entrée.

    Args:
        path: chemin de l'image (local ou s3://...).

    Returns:
        L'image en mode niveaux de gris.
    """
    with fsspec.open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
    return img.convert("L")


def load_labels(csv_path: str) -> dict[str, dict[str, str]]:
    """Charge les codes de l'annotateur depuis le CSV de correction (local ou S3).

    Le CSV a une colonne d'index, une colonne nom d'image, puis une colonne par item.

    Args:
        csv_path: chemin du CSV (séparateur ';'). Local ou s3://...

    Returns:
        Dictionnaire {copy_id: {item_id: code}}.
    """
    labels: dict[str, dict[str, str]] = {}
    with fsspec.open(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        item_ids = header[2:]
        for row in reader:
            if not row:
                continue
            copy_id = row[1].strip()
            labels[copy_id] = {item: row[i + 2].strip() for i, item in enumerate(item_ids)}
    return labels


def _exists(path: str) -> bool:
    """Teste l'existence d'un fichier local ou S3."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return bool(paths) and fs.exists(paths[0])


def load_dataset(
    images_dir: str,
    labels_csv: str,
    limit: int | None = None,
) -> list[Copy]:
    """Construit la liste des copies à partir des images et du CSV de labels.

    Args:
        images_dir: dossier des imagettes (local ou s3://...).
        labels_csv: chemin du CSV de codes annotateur (local ou s3://...).
        limit: nombre maximal de copies à charger (None = toutes).

    Returns:
        Liste de Copy, ordonnée par identifiant.
    """
    labels = load_labels(labels_csv)

    copies: list[Copy] = []
    for copy_id in sorted(labels):
        image_path = _join(images_dir, copy_id)
        if not _exists(image_path):
            continue
        item_codes = labels[copy_id]
        copies.append(
            Copy(
                copy_id=copy_id,
                image_path=image_path,
                item_ids=list(item_codes.keys()),
                expert_codes=list(item_codes.values()),
            )
        )
        if limit is not None and len(copies) >= limit:
            break
    return copies
