# Journal des décisions méthodologiques

Ce fichier trace les décisions structurantes du projet et leur justification.
Format inspiré des ADR (Architecture Decision Records).

---

## D1 — Commencer par la dictée

**Décision** : la phase 1 porte sur la dictée, pas sur la production écrite libre.

**Raison** : la dictée a un texte de référence connu, ce qui rend l'évaluation
mesurable (CER/WER exacts, codage item par item). La production libre n'a pas de
référence et dépend davantage de la qualité HTR.

---

## D2 — Grille simplifiée 1/erreur/0 en cible principale

**Décision** : la cible principale du codage regroupe les codes 3/4/5 en un seul
code « erreur » (1 = correct, 9 = erreur, 0 = absent). La grille complète à 6 codes
reste un objectif secondaire.

**Raison** : sur le test pilote (3 copies, 249 items), la simplification fait passer
le kappa de 0,785 à 0,879 et fait disparaître 8 des 17 désaccords — exactement ceux
portant sur la frontière lexical/grammatical, là où les correcteurs humains eux-mêmes
divergent le plus.

**À valider avec la DEPP** : la distinction lexical/grammatical reste-t-elle
indispensable à certaines exploitations (suivi fin des compétences) ? Si oui, la
traiter comme un sur-codage optionnel.

---

## D3 — Règle des ratures : lire l'état final

**Décision** : quand l'élève rature et réécrit, on lit la version finale (corrigée
par l'élève).

**Raison** : tranché par la DEPP. Élimine 2 désaccords du test pilote. Inscrit dans
tous les prompts (`pipeline/prompts.py`).

---

## D4 — Score de confiance par item obligatoire

**Décision** : chaque code prédit doit s'accompagner d'un score de confiance, pour
permettre un renvoi ciblé des items incertains vers un codeur humain.

**Raison** : besoin DEPP confirmé. Le livrable décisionnel central est la courbe
« taux de renvoi humain vs taux d'erreur résiduel » (`evaluation/calibration.py`).

---

## D5 — Modèle principal Qwen2.5-VL, tester le 7B avant le 72B

**Décision** : Qwen2.5-VL est le modèle principal. On évalue d'abord le 7B (1 GPU
H100) et on ne passe au 72B (4 GPU, TP=4) que si nécessaire.

**Raison** : calibrer le coût GPU. Si le 7B atteint déjà κ > 0,75 en un seul appel,
le 72B n'est pas requis pour la phase 1.

---

## D6 — Fine-tuning : scoring oui, HTR conditionné

**Décision** : les ~3 000 copies annotées suffisent pour fine-tuner le scoring
(cible 1/erreur/0). Le fine-tuning HTR est conditionné à la production préalable
d'un ground truth de transcription mot à mot (les annotations actuelles sont des
codes, pas des transcriptions).

---

## Questions ouvertes (à trancher avec la DEPP)

- Localisation des corrections multi-annotateurs (un seul code disponible pour l'instant).
- Sémantique du code 9 sur les items mots (hors grille écrite).
- Existence de scans en niveaux de gris (au lieu du TIFF 1 bit).
- Deadline réelle des copies CEDRE 2026 sur S3.
- Seuil de « performance satisfaisante » (QWK) différencié par critère.
- Répartition précise des tâches avec TEKLIA.
- Validation RGPD du niveau de sécurité SSP Cloud pour des données de mineurs.
