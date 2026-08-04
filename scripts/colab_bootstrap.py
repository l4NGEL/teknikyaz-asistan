#!/usr/bin/env python3
"""Colab runtime: kaynak yamalari + guvenli paket kurulumu."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "scripts" / "colab_patches"

# Zorunlu pinler (trl/transformers birlikte yuklenmez — qlora_train inspect ile uyum saglar)
PINNED = [
    "pyarrow==17.0.0",
    "datasets==2.21.0",
    "chromadb==0.5.23",
]

OPTIONAL = [
    "peft",
    "bitsandbytes",
    "accelerate",
    "sentence-transformers",
    "rank-bm25",
    "sumy",
    "nltk",
    "mlflow",
]

PATCH_MAP = {
    "nlp_tasks/summarization.py": "src/nlp_tasks/summarization.py",
    "nlp_tasks/qa.py": "src/nlp_tasks/qa.py",
    "nlp_tasks/translation.py": "src/nlp_tasks/translation.py",
    "rag/rag_pipeline.py": "src/rag/rag_pipeline.py",
    "finetune/qlora_train.py": "src/finetune/qlora_train.py",
}


def _pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _fix_opentelemetry() -> None:
    """Colab'da karisik OTEL surumlerini temizler."""
    otel_pkgs = [
        "opentelemetry-sdk",
        "opentelemetry-api",
        "opentelemetry-exporter-otlp-proto-grpc",
        "opentelemetry-exporter-otlp-proto-common",
        "opentelemetry-proto",
        "opentelemetry-instrumentation-fastapi",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *otel_pkgs],
        capture_output=True,
    )
    _pip(
        "--force-reinstall",
        "--no-cache-dir",
        "opentelemetry-api==1.27.0",
        "opentelemetry-sdk==1.27.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.27.0",
        "opentelemetry-exporter-otlp-proto-common==1.27.0",
        "opentelemetry-proto==1.27.0",
    )


def install_packages() -> None:
    _fix_opentelemetry()
    chromadb_pin = "chromadb==0.5.23"
    optional = [p for p in OPTIONAL if p != "chromadb"]
    try:
        _pip(chromadb_pin)
    except subprocess.CalledProcessError:
        _pip("chromadb")
    for pkg in optional:
        try:
            _pip(pkg)
        except subprocess.CalledProcessError:
            print(f"Uyari: {pkg} kurulamadi, atlaniyor.")
    # ONEMLI: transformers/trl pinini EN SONDA (sentence-transformers, accelerate gibi
    # paketlerden SONRA) uyguluyoruz. Nedeni: bu paketlerin bazilari transformers'a
    # daha eski bir ust sinir dayatir; ayni pip komutunda once transformers>=5.0
    # kurulsa bile SONRAKI bagimsiz bir "pip install" cagrisi (yukaridaki optional
    # dongu) transformers'i sessizce eski bir surume dusurebilir (pip, her ayri
    # cagriyi kendi icinde bagimsiz coz er, onceki cagrinin niyetini "hatirlamaz").
    # Bu yuzden "son pinleyen kazanir" stratejisini kullaniyoruz — pyarrow/datatsets
    # icin zaten ayni sebeple asagida da uyguluyoruz.
    _pip("transformers>=5.0", "trl>=0.19.1,<0.24")
    # Colab'da onceden kurulu huggingface-hub, transformers>=5.0'in bile beklemedigi
    # kadar yeni (1.x) olabilir; eski transformers surumune sessizce dusulduyse bu
    # satir hf-hub'i o surumun kabul ettigi araliga (<1.0) geri ceker.
    _pip("--force-reinstall", "--no-cache-dir", "huggingface-hub<1.0")
    # trl/mlflow datasets'i 5.x'e yukseltebilir — pin en sonda
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "pyarrow", "datasets"],
        capture_output=True,
    )
    _pip(
        "--force-reinstall",
        "--no-cache-dir",
        "pyarrow==17.0.0",
        "datasets==2.21.0",
    )
    try:
        import trl
    except ImportError:
        _pip("trl", "peft")
        import trl
    print(f"trl: {trl.__version__} (qlora_train otomatik uyum saglar)")
    print("pyarrow/datasets pinlendi. Colab'da import oncesi runtime restart onerilir.")


def apply_patches() -> None:
    if not PATCHES.exists():
        raise FileNotFoundError(
            f"{PATCHES} bulunamadi. Guncel zip'i Drive'a yukleyin."
        )
    for rel_patch, rel_dst in PATCH_MAP.items():
        src = PATCHES / rel_patch
        dst = ROOT / rel_dst
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Yama: {rel_dst}")

    qlora = ROOT / "src/finetune/qlora_train.py"
    if "_seq_length_config_kwargs" not in qlora.read_text(encoding="utf-8"):
        raise RuntimeError("qlora_train.py yamasi uygulanamadi.")


def main() -> None:
    apply_patches()
    install_packages()
    print("Colab bootstrap tamam.")


if __name__ == "__main__":
    main()
