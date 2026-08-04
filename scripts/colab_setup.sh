#!/usr/bin/env bash
# Colab/Kaggle ortamında projeyi hazırlayan kurulum betiği.
# Kullanım: !bash scripts/colab_setup.sh <repo-url>
set -euo pipefail

REPO_URL="${1:-}"
PROJECT_DIR="/content/baykar-nlp-hazirlik"

if [ -n "$REPO_URL" ] && [ ! -d "$PROJECT_DIR" ]; then
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

if [ -f scripts/colab_bootstrap.py ]; then
    python scripts/colab_bootstrap.py
else
    pip install -q -r requirements.txt
fi

python - <<'PY'
import nltk
nltk.download("punkt")
PY

python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"GPU hazir: {torch.cuda.get_device_name(0)}")
else:
    print("UYARI: GPU bulunamadi. Runtime > Degistir calisma zamani turu > T4 GPU secin.")
PY

echo "Kurulum tamamlandi. notebooks/00_ortam_kurulumu.ipynb ile devam edin."
