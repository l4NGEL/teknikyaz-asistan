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
    # ONEMLI: transformers/trl'yi ARALIK (>=, <) degil, KESIN surum (==) ile pinliyoruz,
    # ve ayri bir cagriyla kuruyoruz. Aralikli bir kisitlamayi (orn. "trl>=0.19.1,<0.24")
    # baska bircok paketle (chromadb, sentence-transformers, peft, accelerate...) AYNI
    # pip cagrisinda birlikte cozdurmek, pip'in resolver'ini saatlerce surebilecek bir
    # geri izleme (backtracking) aramasina sokabiliyor. Kesin surumlerde pip arama
    # yapmaz, dogrudan indirir — hem cok daha hizli hem de mutual-uyumlulugu garanti eder
    # (trl'in DPOConfig'inin TrainingArguments'ta olmayan bir ozelliye basvurmasi gibi
    # AttributeError'lari onler).
    _pip("--force-reinstall", "--no-cache-dir", "transformers==4.46.3", "trl==0.11.4")
    # transformers.pipelines, hic kullanmadigimiz torchvision'i (goruntu isleme icin)
    # opsiyonel olarak yuklemeye calisiyor; torch'un guncellenmesi torchvision ile ic
    # surum uyumsuzlugu yaratabiliyor (torch._dynamo.config icinde TypeError).
    # Kullanmadigimiz icin kaldiriyoruz.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"],
        capture_output=True,
    )
    # transformers/trl ARTIK pinli oldugu icin digerlerini ayri, "-U" olmadan kuruyoruz.
    _pip(
        "chromadb==0.5.23", "sentence-transformers", "rank-bm25",
        "peft", "bitsandbytes", "accelerate", "mlflow",
        "sumy", "nltk",
    )
    # pyarrow/datasets'i hf-hub'dan ONCE pinliyoruz: datasets'in kendi kurulumu da
    # hf-hub'i (ust siniri olmayan kendi gereksinimine gore) yukseltebilir; hf-hub
    # pinini bundan SONRA uygulamazsak bu adim onu tekrar bozar.
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
    # ASIL KRITIK SATIR, MUTLAKA EN SON: yukaridaki datasets kurulumu bile
    # huggingface-hub'i kendi (ust siniri olmayan) gereksinimine gore en guncel
    # surume (1.x) yukseltebiliyor — orijinal ImportError'un gercek kaynagi buydu.
    # hf-hub'i, hicbir sonraki adim onu tekrar degistiremeyecek sekilde EN SONDA,
    # transformers'in (hangi surume duserse dussun) kabul ettigi araliga (<1.0)
    # geri cekiyoruz.
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
