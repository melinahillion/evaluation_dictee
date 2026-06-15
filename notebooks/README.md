# Notebooks

Espace d'**exploration** : inspection d'images, tests de prompts, analyse de
résultats, visualisations.

## Règles

- Les notebooks servent à explorer, pas à héberger la logique du projet.
- Dès qu'un bout de code devient réutilisable, le déplacer dans `src/evaluation_dictee/`.
- Ne jamais committer de sortie contenant des images d'élèves (données sensibles).
- Vider les sorties avant commit (`jupyter nbconvert --clear-output`).

## Suggestions de premiers notebooks

- `01_exploration_images.ipynb` : charger quelques imagettes, vérifier le format
  TIFF 1 bit, tester la conversion en niveaux de gris et le recadrage par ligne.
- `02_analyse_grille.ipynb` : distribution des codes experts dans le CSV, repérer
  les codes hors grille (9 sur des mots), quantifier la fréquence des erreurs.
- `03_analyse_resultats.ipynb` : charger un run, tracer la matrice de confusion et
  la courbe de renvoi humain.
