#!/usr/bin/env bash
# Lance une prédiction AlphaFold2 (ColabFold 1.5.5) dans Docker, sur CPU ou GPU.
#
# Usage : scripts/predict.sh data/ma_proteine.fasta [nom_du_dossier_resultat]
#
# - Les poids du modèle (~5 Go) sont stockés dans ./cache (téléchargés une seule fois).
# - Le MSA est calculé par le serveur public MMseqs2 de ColabFold (connexion Internet requise).
# - Les résultats sont écrits dans ./results/<nom>/ et le log dans ./logs/<nom>.log
set -euo pipefail

IMAGE="ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2"
FASTA="${1:?Usage: $0 fichier.fasta [nom]}"
NAME="${2:-$(basename "${FASTA%.*}")}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$ROOT/cache" "$ROOT/results" "$ROOT/logs"

# Téléchargement des poids au premier lancement uniquement
if [ ! -f "$ROOT/cache/colabfold/params/download_finished.txt" ]; then
  echo ">> Téléchargement des poids AlphaFold2 (une seule fois, ~5 Go)..."
  docker run --rm -v "$ROOT/cache:/cache" "$IMAGE" python -m colabfold.download
fi

# Utilise le GPU s'il est disponible (nécessite nvidia-container-toolkit)
GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then GPU_FLAG="--gpus all"; fi

echo ">> Prédiction de $FASTA -> results/$NAME (log : logs/$NAME.log)"
time docker run --rm $GPU_FLAG \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT/cache:/cache" \
  -v "$(cd "$(dirname "$FASTA")" && pwd):/work/in:ro" \
  -v "$ROOT/results:/work/results" \
  "$IMAGE" \
  colabfold_batch --num-models 5 "/work/in/$(basename "$FASTA")" "/work/results/$NAME" \
  2>&1 | tee "$ROOT/logs/$NAME.log"
