#!/usr/bin/env bash
# Yerel geliştirme ortamında (GPU gerektirmeyen) servis + panel + testleri çalıştırır.
set -euo pipefail

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

pip install -q -r requirements.txt

echo "Testler çalıştırılıyor..."
pytest tests/ -q

echo "FastAPI servisi başlatılıyor (http://localhost:8000)..."
uvicorn src.serving.api:app --reload --port 8000
