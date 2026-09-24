# Déployer et comprendre AlphaFold2

Ce dépôt permet de réaliser une prédiction de structure avec AlphaFold2, puis d'en analyser les principales sorties :

```
séquence (FASTA) ──► ColabFold 1.5.5 / AlphaFold2 (Docker) ──► structure (PDB) + pLDDT + PAE ──► scripts/analyse.py ──► figures
```

La solution utilise [ColabFold](https://github.com/sokrypton/ColabFold) (Mirdita *et al.*, 2022). ColabFold exécute **les mêmes réseaux et les mêmes poids qu'AlphaFold2**, mais construit le MSA via un serveur MMseqs2 public : il n'y a donc pas 2,6 To de bases de données à télécharger. Le calcul se fait **en local dans Docker**, sur CPU ou sur GPU. Pour une machine sans Linux ni Docker, une alternative Google Colab est décrite [plus bas](#alternative-sans-docker--google-colab).

Le rapport (contexte, choix technique, analyse) se trouve dans [modele_de_rapport/rapport.pdf](modele_de_rapport/rapport.pdf).

---

## Pour les étudiants de l'année prochaine

> Mode d'emploi court. Les détails se trouvent dans les sections suivantes.

1. **À installer ou ouvrir** : Docker (`docker run hello-world` doit fonctionner) et Python ≥ 3.9. Environ 20 Go de disque libre et une connexion Internet.
2. **Préparer la séquence** : créer `data/MA_PROT.fasta` avec **un en-tête court** :
   ```
   >MA_PROT
   MADQLTEEQIAEFKEAFSLFDKDGDG...
   ```
   Une seule séquence protéique, en lettres majuscules, sans `*` final. On la récupère sur UniProt : `curl -s https://rest.uniprot.org/uniprotkb/<ACCESSION>.fasta`.
3. **Lancer la prédiction** :
   ```bash
   scripts/predict.sh data/MA_PROT.fasta
   ```
   Au premier lancement, le script télécharge l'image Docker (~10 Go) et les poids (~5 Go).
4. **Temps à prévoir** : sur CPU (12 cœurs), **TEMPS_TOTAL pour 149 résidus** avec 5 modèles. Le coût croît à peu près comme le carré de la longueur. Sur GPU ou sur Colab, comptez quelques minutes.
5. **Récupérer les résultats** : dans `results/MA_PROT/`. La structure est `*_unrelaxed_rank_001_*.pdb` et les scores sont dans `*_scores_rank_001_*.json`.
6. **Lancer l'analyse pLDDT** :
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt   # une seule fois
   .venv/bin/python scripts/analyse.py results/MA_PROT -o figures/MA_PROT
   ```
   Le script produit `plddt.png`, `plddt_rank_001.csv` et `resume.txt`.
7. **Afficher et interpréter la PAE** : ouvrir `figures/MA_PROT/pae.png`. Un carré vert foncé sur la diagonale correspond à un domaine rigide et bien prédit. Une zone claire entre deux blocs signifie que **leur position relative est incertaine**. L'option `--domains 1-78,82-149` calcule la PAE moyenne entre domaines.
8. **Pièges à éviter** :
   - un en-tête FASTA long (`>sp|P0DP23|CALM1_HUMAN ...`) donne des noms de fichiers illisibles ;
   - sans `--user $(id -u):$(id -g)`, les résultats appartiennent à `root` (le script `predict.sh` s'en charge) ;
   - au-delà d'environ 400 résidus sur CPU, le calcul prend plusieurs heures : utilisez Colab ou un GPU ;
   - pLDDT élevé ≠ domaines bien placés les uns par rapport aux autres : **regardez toujours la PAE** ;
   - AlphaFold2 ne modélise ni ligands, ni ions, ni effet des mutations ponctuelles ;
   - le serveur MMseqs2 est partagé : ne soumettez pas des centaines de séquences d'un coup.

---

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `data/P0DP23.fasta` | Séquence de test (calmoduline humaine, 149 résidus) |
| `scripts/predict.sh` | Lance ColabFold dans Docker (poids + MSA + prédiction) |
| `scripts/analyse.py` | Extrait le pLDDT et la PAE et produit les figures et un résumé |
| `scripts/requirements.txt` | Dépendances Python de l'analyse (numpy, matplotlib) |
| `results/P0DP23/` | Sorties brutes de ColabFold pour la protéine de test |
| `figures/P0DP23/` | Figures et résumé produits par `analyse.py` |
| `logs/` | Log complet de la prédiction (avec la durée) |
| `modele_de_rapport/` | Rapport LaTeX (`rapport.tex` → `rapport.pdf`) |
| `cache/` | Poids AlphaFold2 téléchargés (non versionné, ~5 Go) |

---

## Prérequis

| Élément | Version testée | Remarque |
|---|---|---|
| OS | Debian 12 (Linux 6.12) | Tout Linux avec Docker. Sous macOS ou Windows, Docker Desktop devrait fonctionner mais n'a pas été testé ; préférez Colab. |
| Docker | 29.8.1 | L'utilisateur doit pouvoir lancer `docker` sans `sudo` (groupe `docker`) |
| CPU / RAM | 12 cœurs, 14 Go | ~4 Go utilisés pour 149 résidus |
| GPU | aucun | Facultatif. Si `nvidia-smi` existe, `predict.sh` ajoute `--gpus all` (nécessite `nvidia-container-toolkit`) |
| Disque | ~20 Go | Image Docker ~10 Go + poids 5,3 Go |
| Réseau | requis | Téléchargements initiaux + MSA via `api.colabfold.com` |
| Python | 3.11.2 | Uniquement pour l'analyse (numpy 2.4, matplotlib 3.11) |
| Comptes | aucun | (Un compte Google seulement pour l'alternative Colab) |

## Installation / préparation

À faire une seule fois, dans cet ordre :

```bash
# 1. Vérifier Docker
docker run --rm hello-world

# 2. Télécharger l'image ColabFold (version figée)
docker pull ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2

# 3. Télécharger les poids AlphaFold2 dans ./cache (≈ 5 Go, ~4 min)
#    (predict.sh le fait automatiquement s'ils sont absents)
mkdir -p cache
docker run --rm -v "$PWD/cache:/cache" ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
    python -m colabfold.download

# 4. Environnement Python pour l'analyse
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
```

## Exécution

### 1. Préparer la séquence

```bash
mkdir -p data
curl -s https://rest.uniprot.org/uniprotkb/P0DP23.fasta | sed '1s/.*/>P0DP23/' > data/P0DP23.fasta
```

La commande `sed` remplace l'en-tête UniProt par un identifiant court.

### 2. Lancer la prédiction

```bash
scripts/predict.sh data/P0DP23.fasta          # résultats dans results/P0DP23/
```

Commande Docker équivalente, lancée par le script :

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/cache:/cache" -v "$PWD/data:/work/in:ro" -v "$PWD/results:/work/results" \
  ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
  colabfold_batch --num-models 5 /work/in/P0DP23.fasta /work/results/P0DP23
```

Options utiles de `colabfold_batch` :
- `--num-models 1` : un seul modèle, 5 fois plus rapide, pratique pour un premier essai ;
- `--num-recycle N` : nombre de recyclages (3 par défaut) ;
- `--amber` : relaxation de la structure par champ de force (plus lent) ;
- `--templates` : utilise des gabarits de la PDB.

Déroulement visible dans le log :
1. `Query 1/1: ... (length 149)` suivi de `SUBMIT/COMPLETE` : MSA sur le serveur MMseqs2, environ 15 s ;
2. `recycle=0 ... recycle=3` pour chacun des 5 modèles (`alphafold2_ptm_model_1` à `_5`) ;
3. `reranking models by 'plddt' metric` puis `Done`.

### 3. Analyser

```bash
.venv/bin/python scripts/analyse.py results/P0DP23 -o figures/P0DP23 --domains 1-78,82-149
```

### Alternative sans Docker : Google Colab

1. Ouvrir le notebook officiel : <https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb>
2. *Exécution > Modifier le type d'exécution* > **GPU T4**.
3. Coller la séquence dans `query_sequence`, donner un `jobname` court, puis *Exécution > Tout exécuter*.
4. À la fin, le notebook télécharge un fichier `.zip` : le décompresser dans `results/<jobname>/`.
5. Lancer `scripts/analyse.py` comme ci-dessus : les fichiers sont exactement les mêmes.

Cette voie n'a pas été retenue comme solution principale : les sessions sont limitées dans le temps, le GPU n'est pas garanti et le notebook n'est pas versionné.

## Résultats

Dans `results/<nom>/`, les modèles sont classés par pLDDT moyen décroissant : `rank_001` est le meilleur.

| Fichier | Contenu |
|---|---|
| `<nom>_unrelaxed_rank_001_alphafold2_ptm_model_X_seed_000.pdb` | **Structure prédite** (format PDB). Le **pLDDT** de chaque résidu est stocké dans la colonne *B-factor*. |
| `<nom>_scores_rank_001_alphafold2_ptm_model_X_seed_000.json` | **Scores** : `plddt` (liste de N valeurs), `pae` (matrice N×N en Å), `max_pae`, `ptm` |
| `<nom>_predicted_aligned_error_v1.json` | PAE seule, au format de l'AlphaFold DB |
| `<nom>_plddt.png`, `<nom>_pae.png`, `<nom>_coverage.png` | Figures générées automatiquement par ColabFold (pLDDT des 5 modèles, PAE, profondeur du MSA) |
| `<nom>.a3m` | MSA utilisé (format A3M) |
| `log.txt`, `config.json` | Log et paramètres exacts du calcul (traçabilité) |
| `cite.bibtex` | Références à citer |

Visualiser la structure :
- en ligne : glisser le `.pdb` sur <https://molstar.org/viewer/> (*Color > Uncertainty/Disorder* ou *B-factor* pour colorer par pLDDT) ;
- en local : PyMOL (`spectrum b, red_yellow_green_cyan_blue, minimum=50, maximum=90`) ou ChimeraX (`color bfactor palette alphafold`).

Fichiers produits par `analyse.py` dans `figures/<nom>/` : `plddt.png`, `pae.png`, `plddt_rank_001.csv`, `resume.txt`.

### Interpréter

- **pLDDT** (0–100, par résidu) : confiance *locale*. > 90 : très fiable, y compris les chaînes latérales. 70–90 : squelette fiable. 50–70 : faible. < 50 : à ne pas interpréter, souvent une région désordonnée.
- **PAE** (Å, matrice N×N) : la case (i, j) donne l'erreur attendue sur la position du résidu i si l'on superpose prédiction et vraie structure sur le résidu j. Une valeur faible entre deux domaines signifie que leur **position relative** est fiable ; une valeur élevée signifie qu'elle ne l'est pas, même si chaque domaine a un bon pLDDT.

## Problèmes connus

| Problème | Cause | Solution |
|---|---|---|
| `no GPU detected, will be using CPU` | Pas de GPU NVIDIA | Normal : le calcul marche sur CPU, mais plus lentement |
| Fichiers de `results/` appartenant à `root` | Docker écrit en root par défaut | `--user $(id -u):$(id -g) -e HOME=/tmp` (déjà dans `predict.sh`) |
| Noms de fichiers de 100 caractères | ColabFold reprend l'en-tête FASTA | En-tête court (`>P0DP23`) |
| `Allocation of ... exceeds 10% of free system memory` | Avertissement de TensorFlow | Sans conséquence ; fermer les autres applications si la RAM est faible |
| `permission denied ... docker.sock` | Utilisateur hors du groupe `docker` | `sudo usermod -aG docker $USER` puis se reconnecter |
| Erreur de compilation LaTeX du modèle de rapport | `inputenc[utf8x]` et `\lstset` utilisant des macros non définies | Remplacés par `utf8` ; `\lstset` Python retiré |

## Limites

- **Connexion Internet obligatoire** pour le MSA, et dépendance au serveur public `api.colabfold.com`. S'il est indisponible ou si l'usage est limité, il faut héberger ses propres bases (`colabfold_search`, ~1 To).
- **CPU lent** : au-delà d'environ 400 résidus, il faut un GPU ou Colab.
- **Image Docker pour x86_64** : non testée sur ARM (Mac M1/M2).
- Limites d'AlphaFold2 lui-même : une seule conformation statique ; pas de ligands, d'ions ni de cofacteurs (la calmoduline est prédite sans ses Ca²⁺) ; insensible aux mutations ponctuelles ; mauvais résultats pour les protéines orphelines (MSA peu profond) ; les complexes demandent le mode multimère (`A:B` dans le FASTA).
- Les poids AlphaFold2 sont sous licence CC BY 4.0 et le code sous Apache 2.0 : usage académique sans restriction.

## Références

- Jumper J. *et al.* (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589. <https://doi.org/10.1038/s41586-021-03819-2>
- Mirdita M. *et al.* (2022). ColabFold: making protein folding accessible to all. *Nature Methods* 19, 679–682. <https://doi.org/10.1038/s41592-022-01488-1>
- ColabFold, dépôt officiel : <https://github.com/sokrypton/ColabFold>
- AlphaFold, dépôt officiel : <https://github.com/google-deepmind/alphafold>
- EMBL-EBI Training, *AlphaFold: A practical guide* : <https://www.ebi.ac.uk/training/online/courses/alphafold/> (pages pLDDT et PAE)
- UniProt P0DP23 : <https://www.uniprot.org/uniprotkb/P0DP23>
