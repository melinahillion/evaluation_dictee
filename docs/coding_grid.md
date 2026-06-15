# Grille de codage de la dictée (CEDRE MDL 2015)

Référence pour le codage. Source : guide de codage DEPP B2 (PDF dans le dossier docs).

## Texte de la dictée

> Le soir tombait. Papa et maman, inquiets, se demandaient pourquoi leurs quatre
> garçons n'étaient pas rentrés. — Les gamins se sont certainement perdus, dit maman.
> S'ils n'ont pas encore retrouvé leur chemin, nous les verrons arriver très fatigués
> à la maison. — Pourquoi ne pas téléphoner à Martine ? Elle les a peut-être vus !
> Aussitôt dit, aussitôt fait ! À ce moment, le chien se mit à aboyer.

Chaque mot et chaque signe de ponctuation = un item (83 items au total).

## Grille complète (6 codes)

### Mots
| Code | Signification |
|------|---------------|
| 1 | Orthographe correcte |
| 3 | Erreur lexicale (n'altère pas la prononciation : absence de trait d'union, accent, lettre muette, mauvais graphème — « soire » pour « soir ») |
| 4 | Erreur grammaticale (accord, conjugaison, confusion de catégorie « on »/« ont », majuscule absente en début de phrase, ou erreur changeant la prononciation) |
| 5 | Erreur lexicale ET grammaticale |
| 0 | Mot absent (oubli) |

### Ponctuation
| Code | Signification |
|------|---------------|
| 1 | Ponctuation correcte |
| 9 | Erreur (virgule à la place d'un point, etc.) |
| 0 | Absence de ponctuation |

## Grille simplifiée (cible principale du projet — voir docs/decisions.md D2)

| Code | Signification |
|------|---------------|
| 1 | Correct |
| 9 | Erreur (regroupe 3, 4, 5 pour les mots ; erreur de ponctuation) |
| 0 | Absent |

## Codes spéciaux

- **9 sur un mot** : observé dans les corrections réelles mais non prévu par la grille
  écrite pour les mots. Sémantique à clarifier avec la DEPP (illisible ? inclassable ?).
- **i** : imagette tronquée / illisible (mentionné dans le guide général).

## Points d'attention pour le codage automatique

- Lire l'**état final** en cas de rature (décision D3).
- Ne **pas corriger silencieusement** l'orthographe (biais de sur-correction des VLMs).
- Les **homophones** (a/à, ce/se) sont des cas-frontières 3 vs 4 où la pratique des
  correcteurs diverge de la doctrine écrite : à arbitrer.
