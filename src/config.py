"""Proje genelinde paylaşılan yollar, model isimleri ve hiperparametreler.

Tek bir yerden değiştirilebilsin diye tüm "hangi modeli / hangi klasörü kullanıyoruz"
kararları burada toplanır. Diğer modüller bu dosyadan import eder.
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Yeni mlflow surumleri duz dosya tabanli (file:./mlruns) takip backend'ini "bakim
# modu"na aldi ve varsayilan olarak MlflowException firlatiyor (veritabani backend'ine
# gecmeyi oneriyor). Bu proje bilerek basit dosya tabanli ./mlruns kullaniyor (Drive'a
# symlink'li, ekstra bir DB dosyasi yonetmeye gerek yok), o yuzden bu projedeki HERHANGI
# bir modul import edildiginde (qlora_train.py/dpo_train.py'nin report_to=["mlflow"]'u
# ve llmops/experiment_tracker.py ikisi de) bayragi en basta aciyoruz.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# HF Trainer'in kendi MLflowCallback'i (report_to=["mlflow"]), ortam degiskeninde bir
# deney adi bulamazsa deney secmeden dogrudan start_run() cagiriyor; bu da (mlruns/
# klasorunde "tiha-asistan" DISINDA hicbir deney hic olusmadigi icin) "Could not find
# experiment with ID 0" hatasina yol aciyor. Deney adini burada, merkezi olarak
# ayarlayip experiment_tracker.py'deki DEFAULT_EXPERIMENT ile ayni deneyde birlestiriyoruz.
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "tiha-asistan")

# Colab'da ONCEDEN KURULU gelen torchao (0.10.0), peft'in LoRA adapter yukleme
# dispatcher'ini (dispatch_torchao -> is_torchao_available) ImportError ile
# patlatiyor -- bu proje torchao hic kullanmiyor (bitsandbytes NF4 ile quantize
# ediyor). colab_bootstrap.py de kaldiriyor ama "Runtime > Restart session" bazen
# ayni VM'i korur (pip kurulumlari kalicidir), bazen Colab tamamen YENI bir VM
# veriyor (torchao 0.10.0 geri gelir) -- hangisi oldugunu onceden bilemeyiz. pip
# durumuna guvenmek yerine, HER taze kernel'de (bu modul importlandiginda) kesin
# olarak kaldiriyoruz; zaten kurulu degilse importlib.metadata hemen firlatir ve
# hicbir subprocess calismaz (sonraki importlarda no-op).
try:
    import importlib.metadata as _importlib_metadata

    _importlib_metadata.version("torchao")
except _importlib_metadata.PackageNotFoundError:
    pass
else:
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        capture_output=True,
    )

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DB_PATH = DATA_DIR / "metrics.sqlite3"

# ONEMLI: chroma_db BILEREK DATA_DIR disinda (Drive'a symlink'lenmeyen, yerel/gecici
# Colab diskinde) tutuluyor. Google Drive'in FUSE dosya sistemi SQLite'in ihtiyac
# duydugu dosya kilitlerini desteklemiyor -> "OperationalError: attempt to write a
# readonly database" hatasi. Chunk'lar (data/processed/chunks.jsonl) zaten Drive'da
# kalici oldugu icin her runtime'da chroma_db'yi yeniden indekslemek ucuz/hizli;
# bu yuzden Drive'a tasimaya gerek yok, sadece yerel diskte kalmasi yeterli.
VECTOR_DB_DIR = ROOT_DIR / ".local_cache" / "chroma_db"

EVAL_DIR = DATA_DIR / "eval"

for _dir in (RAW_DIR, PROCESSED_DIR, VECTOR_DB_DIR, MODELS_DIR, EVAL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    # Çok dilli embedding modeli — Türkçe dahil ~100 dilde iyi çalışır, RAG retrieval için.
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Fine-tuning tabanı: Türkçe'de makul performans gösteren, Colab T4/A100'e QLoRA ile sığan
    # açık kaynak bir instruct model. Daha güçlü GPU'nuz varsa 7B/8B varyantlara geçebilirsiniz.
    base_llm: str = "Qwen/Qwen2.5-3B-Instruct"

    # Hazır Türkçe NLP görev modelleri (sıfırdan eğitim yerine transfer öğrenme ile başlıyoruz,
    # ardından kendi verimizle fine-tune ediyoruz — gerçek dünyada da böyle çalışılır).
    ner_model: str = "savasy/bert-base-turkish-ner-cased"
    qa_model: str = "savasy/bert-base-turkish-squad"
    summarization_model: str = "ozcangundes/mt5-small-turkish-summarization"
    classification_base_model: str = "dbmdz/bert-base-turkish-cased"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    mt_tr_en_model: str = "Helsinki-NLP/opus-mt-tr-en"
    mt_en_tr_model: str = "Helsinki-NLP/opus-mt-en-tr"
    offensive_model: str = "Overfit-GM/distilbert-base-turkish-cased-offensive"


@dataclass(frozen=True)
class RagConfig:
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32
    # Hiyerarşik RAG: child chunk retrieve edilir, üretim için aynı belgenin
    # kardeş parçaları (parent) birleştirilir. Diagnis ilanındaki "chunk stratejisi /
    # hiyerarşik RAG" maddesine karşılık gelir.
    hierarchical_expand: bool = True
    top_k: int = 5
    collection_name: str = "teknik_yazim_asistani"
    eval_gold_path: str = str(EVAL_DIR / "retrieval_gold.jsonl")
    eval_k_values: tuple = (1, 3, 5, 10)


@dataclass(frozen=True)
class AgentConfig:
    # Düşük güven / çelişkili kaynak -> insan onayı. 0.45, NLI groundedness'in
    # "destek zayıf" bölgesine denk gelecek şekilde seçildi; kalibrasyon defteri
    # `src/xai/confidence.py` içinde.
    confidence_threshold: float = 0.45
    use_nli_groundedness: bool = True
    use_langgraph: bool = True


@dataclass(frozen=True)
class VllmConfig:
    base_url: str = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    model: str = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    timeout_s: float = 120.0


@dataclass(frozen=True)
class QloraConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    # SFT ornekleri artik gercek RAG bagsamini (tam bir referans dokuman) da icerdigi
    # icin ortalama dizi uzunlugu onemli olcude artti -- kayip hesaplamasi (logits
    # tensoru) sozluk boyutuyla (vocab_size) carpilan bu uzunlukla olcekleniyor, T4'un
    # 15GB'ini asip OOM veriyordu. batch=2'yi 1'e dusurup grad_accum'i 16'ya cikararak
    # ETKIN batch boyutunu (2*8=16 -> 1*16=16) AYNI tutuyoruz, sadece adim basina
    # bellek yukunu yariya indiriyoruz.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    # batch=1'e ragmen OOM devam etti -- demek ki bazi tekil ornekler (bagsam dokumani +
    # notlar + hedef bir arada) 1024'e yakin/onu asiyor. Asil buyuten sey (tam bagsam
    # dokumani) kaynaginda kisaltildi (bkz. instruction_dataset.py), burasi sadece
    # geriye kalan ucuk durumlar icin bir guvenlik sinirimiz -- 512 medyan orneklerin
    # yarisini (muhtemelen hedef ciktinin sonunu) kirpardi, 768 daha az agresif.
    max_seq_length: int = 768
    output_dir: str = str(MODELS_DIR / "qlora-adapter")


CLASSIFICATION_LABELS = [
    "README",
    "API Dokümantasyonu",
    "Kurulum Kılavuzu",
    "Kullanıcı Kılavuzu",
    "SSS",
    "Mimari Doküman",
    "Sürüm Notları",
    "Diğer",
]

MODEL_CONFIG = ModelConfig()
RAG_CONFIG = RagConfig()
QLORA_CONFIG = QloraConfig()
AGENT_CONFIG = AgentConfig()
VLLM_CONFIG = VllmConfig()
