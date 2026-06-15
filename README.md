# Évaluation automatique de la dictée des élèves

Évaluer automatiquement la dictée manuscrite d'élèves (CEDRE, école primaire) à
l'aide de modèles multimodaux open weight, et **comparer rigoureusement** le codage
automatique à celui d'un correcteur expert. Collaboration **DEPP × SSP Lab (INSEE)**.

> Contexte complet et décisions : voir **[CLAUDE.md](CLAUDE.md)**, la
> **[note méthodologique](docs/note_methodologique.md)** et les
> **[décisions](docs/decisions.md)** dans [`docs/`](docs/).

---

## Ce que fait le projet, en bref

1. **Charge** les imagettes de dictée (TIFF 1 bit, depuis S3) et les codes de
   l'annotateur expert (gold standard).
2. **Demande à un modèle multimodal** (gemma4 sur llm.lab) de coder chaque mot de
   la dictée — correct / erreur / absent — directement à partir de l'image et du
   texte de référence, sans étape d'OCR séparée.
3. **Compare** les codes du modèle à ceux de l'expert.
4. **Mesure** la fiabilité (kappa, rappel des fautes, sur-correction) et la
   **calibration de la confiance**, pour décider quels items renvoyer à un humain.

La tâche cible est la **grille simplifiée** : `1` correct / `9` erreur / `0` absent
(voir [docs/decisions.md](docs/decisions.md), décision D2).

---

## Démarrage rapide (SSP Cloud / VSCode)

### 1. Installer

```bash
git clone <url-du-depot> evaluation_dictee
cd evaluation_dictee
python -m pip install -e ".[dev]"     # layout src/ : l'installation est obligatoire
```

> Le paquet s'appelle `evaluation_dictee`. Sans `pip install -e .`, les imports
> `from evaluation_dictee...` échouent (le dossier `src/` n'est pas dans le PYTHONPATH).

### 2. Configurer les accès

Sur le SSP Cloud (Onyxia), les identifiants S3 sont généralement déjà dans
l'environnement. Pour le modèle, il faut un token llm.lab :

```bash
cp .env.example .env
# éditer .env :
#   LLM_BASE_URL=https://llm.lab.sspcloud.fr/api/v1   (vérifier le chemin exact)
#   LLM_API_KEY=<ton-token-llm.lab>
```

Vérifier l'accès aux données et au modèle :

```bash
# S3 accessible ?
python -c "from evaluation_dictee.data.loaders import load_labels; \
print(len(load_labels('s3://projet-production-ecrits-depp/resultat_dictee_2015.csv')), 'copies')"

# modèle accessible ?
python -c "from openai import OpenAI; from evaluation_dictee.config import Secrets; \
s=Secrets(); c=OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key); \
print(c.chat.completions.create(model='gemma4-26b-moe', \
messages=[{'role':'user','content':'Dis bonjour'}], max_tokens=10).choices[0].message.content)"
```

### 3. Lancer un benchmark

```bash
python scripts/run_benchmark.py --config configs/dictee_gemma4_zeroshot.yaml
```

Cela produit `data/processed/dictee_gemma4_zeroshot_predictions.jsonl`
(une ligne par item × copie) et journalise les métriques dans MLflow.

### 4. Analyser les résultats

Ouvrir **`notebooks/03_analyse_resultats.ipynb`** et exécuter toutes les cellules.
Le notebook lit le fichier JSONL ci-dessus (aucun besoin de relancer le benchmark)
et produit métriques, figures et exports CSV. Pour analyser un autre run, changer
la seule variable `RUN_NAME` en tête de notebook.

---

## Configurer une expérience

Un fichier YAML dans `configs/` = une expérience reproductible. Exemple commenté :
[`configs/dictee_gemma4_zeroshot.yaml`](configs/dictee_gemma4_zeroshot.yaml).

| Champ | Rôle |
|-------|------|
| `model.name` | nom du modèle servi par llm.lab (ex. `gemma4-26b-moe`) |
| `data.images_path` | dossier des imagettes (local ou `s3://...`) |
| `data.labels_path` | CSV des codes de l'annotateur (local ou `s3://...`) |
| `data.grid_path` | grille de codage JSON (`configs/grille_dictee_2015.json`) |
| `data.limit` | nombre de copies (mettre `null` pour tout le corpus) |
| `grid.scheme` | `simplifiee` (1/9/0) ou `complete` (1/3/4/5/9/0) |
| `prompt.method` | `C` end-to-end (image → code), voir note méthodologique |
| `prompt.read_final_state` | règle des ratures : lire l'état final corrigé |

---

## Structure du dépôt

```
evaluation_dictee/
├── CLAUDE.md                  ← contexte du projet pour humains et IA
├── README.md                  ← ce fichier
├── pyproject.toml             ← dépendances + config ruff/mypy/pytest
├── configs/
│   ├── grille_dictee_2015.json   ← grille de codage (mot attendu + fautes connues)
│   └── *.yaml                     ← une config par expérience
├── src/evaluation_dictee/
│   ├── config.py              ← configs validées (Pydantic) + secrets (.env)
│   ├── data/                  ← chargement images (S3, TIFF 1 bit) + grille + labels
│   ├── models/                ← interface Scorer + client VLM (llm.lab)
│   ├── pipeline/              ← prompts + orchestration du benchmark
│   ├── evaluation/            ← metrics, statistics (bootstrap/Wilson),
│   │                            calibration (ECE), report (par item/copie)
│   └── utils/                 ← logging, suivi MLflow
├── scripts/run_benchmark.py   ← point d'entrée CLI
├── notebooks/03_analyse_resultats.ipynb   ← analyse complète d'un run
├── tests/                     ← 29 tests unitaires (pytest)
└── docs/                      ← note méthodologique, cadrage, décisions, grille
```

---

## Bonnes pratiques

Avant chaque commit :

```bash
ruff format src tests scripts        # formater
ruff check src tests scripts         # linter
pytest                               # tester (29 tests)
```

---

## ⚠️ Données sensibles

Les copies anonymes **ne sont jamais commitées**. Le dossier `/data/` est ignoré par Git
(voir [`.gitignore`](.gitignore)) ; seul le `.env.example` (sans secret) est versionné.