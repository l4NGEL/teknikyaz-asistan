"""Proje genelinde paylaşılan yollar, model isimleri ve hiperparametreler.

Tek bir yerden değiştirilebilsin diye tüm "hangi modeli / hangi klasörü kullanıyoruz"
kararları burada toplanır. Diğer modüller bu dosyadan import eder.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DB_PATH = DATA_DIR / "metrics.sqlite3"

for _dir in (RAW_DIR, PROCESSED_DIR, VECTOR_DB_DIR, MODELS_DIR):
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
    top_k: int = 5
    collection_name: str = "teknik_yazim_asistani"


@dataclass(frozen=True)
class QloraConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
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
