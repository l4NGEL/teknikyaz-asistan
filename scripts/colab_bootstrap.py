#!/usr/bin/env python3
"""Colab runtime: kaynak yamalari + guvenli paket kurulumu."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "scripts" / "colab_patches"

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
    # transformers.pipelines, hic kullanmadigimiz torchvision'i (goruntu isleme icin)
    # opsiyonel olarak yuklemeye calisiyor; torch'un guncellenmesi torchvision ile ic
    # surum uyumsuzlugu yaratabiliyor (torch._dynamo.config icinde TypeError).
    # Kullanmadigimiz icin kaldiriyoruz.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"],
        capture_output=True,
    )
    # ONEMLI: fragile pin zincirindeki paketleri (transformers/trl/pyarrow/datasets/
    # numpy/scipy) EN SONA, birbirini bozamayacaklari kesin bir sirada koyuyoruz.
    # sentence-transformers/chromadb gibi paketlerin kendi transformers gereksinimleri
    # olabilir; onlarin transformers'i istedigi surume cekmesine izin veriyoruz.
    _pip(
        "chromadb==0.5.23", "sentence-transformers", "rank-bm25",
        "peft", "bitsandbytes", "accelerate", "mlflow",
        "sumy", "nltk",
    )
    # 1) transformers/trl KESIN surumle. trl==0.11.4, transformers 4.46'nin
    # Trainer.get_batch_samples(epoch_iterator, num_batches) imzasiyla UYUMSUZDU
    # (DPOTrainer eski imzayi bekliyor, "generator has no attribute generate" hatasi
    # veriyordu) -- trl==0.12.2 bu uyumu saglayan ilk surum. Aralik degil KESIN surum
    # kullaniyoruz cunku pip'in cok sayida paketi ayni anda aralikla cozmeye calismasi
    # saatler surebilecek bir backtracking aramasina yol acabiliyor.
    _pip("--force-reinstall", "--no-cache-dir", "transformers==4.46.3", "trl==0.12.2")
    # 2) pyarrow/datasets: trl'in KENDI datasets bagimliligi bunlari bozabiliyor,
    # bu yuzden transformers/trl'den SONRA tekrar pinliyoruz.
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
    # 3) numpy+scipy. Yukaridaki adimlarin herhangi biri numpy'i kendi
    # gereksinimlerine gore tekrar degistirebiliyor (ABI uyumsuzlugu -> "cannot import
    # name '_center'").
    _pip("--force-reinstall", "--no-cache-dir", "numpy", "scipy")
    # 4) MUTLAKA GERCEKTEN EN SON: huggingface-hub. datasets'in KENDI hf-hub
    # bagimliligi (ust siniri olmayan) bu pini defalarca bozdu — orijinal
    # ImportError'un gercek kaynagi buydu. Artik hicbir sonraki adim yok, bu yuzden
    # burada kalici olmasi lazim.
    _pip("--force-reinstall", "--no-cache-dir", "huggingface-hub<1.0")
    try:
        import trl
    except ImportError:
        _pip("trl", "peft")
        import trl
    print(f"trl: {trl.__version__} (qlora_train otomatik uyum saglar)")
    print("pyarrow/datasets/huggingface-hub pinlendi. Import oncesi runtime restart onerilir.")


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
