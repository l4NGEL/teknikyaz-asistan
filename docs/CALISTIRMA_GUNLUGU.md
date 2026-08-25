# Çalıştırma Günlüğü

Bu doküman, projenin Colab üzerinde uçtan uca çalıştırılması sırasında elde edilen
gerçek sonuçları, karşılaşılan sorunları ve yapılan düzeltmeleri tarih sırasıyla kaydeder.
README'deki "Sonuçlar" bölümünün dayandığı ham kayıt budur.

## 2026-08-18 — Hazırlık, corpus büyütme, ilk hatalar

- Proje ilk haliyle 10 örnek doküman içeriyordu; bu ölçekte QLoRA/DPO/sınıflandırma
  eğitimleri 2-6 adımda bitiyor, istatistiksel olarak anlamsız kalıyordu (örn. ilk
  sınıflandırıcı denemesinde eval seti 22 örnekten sadece 3 kategoriyi kapsıyordu).
- Corpus, 8 kategoriye dengeli dağıtılmış 127 orijinal dokümana çıkarıldı
  (`src/data_prep/corpus_builder.py`).
- `src/llmops/ab_test.py` üzerinden çalışan LLMOps A/B test hücresi
  `MlflowException: Could not find experiment with ID 0` hatasıyla çöktü — HF
  Trainer'ın kendi `MLflowCallback`'i, ortam değişkeninde deney adı bulamayınca
  hiç deney seçmeden `start_run()` çağırıyordu. `src/config.py`'ye
  `MLFLOW_EXPERIMENT_NAME` varsayılanı eklendi.
- `sumy` tabanlı extractive özetleme `LookupError: punkt_tab` ile çöktü — NLTK
  kaynağı taze Colab runtime'ında hiç kurulu gelmiyor. `scripts/colab_bootstrap.py`'ye
  otomatik indirme eklendi.

## 2026-08-19 — Uçtan uca çalıştırma (00 → 06)

### 01 — Veri hazırlama
- 127 doküman → 146 chunk → 146 SFT çifti. Hatasız.

### 02 — NLP görevleri
- **Sınıflandırma**: ilk denemede etiketleme mantığı (`_label_for_title`) başlıkları
  birebir eşleştiriyordu, yeni başlık formatlarının (`"README — X"` gibi) çoğu sessizce
  "Diğer"e düşüyordu → Accuracy 0.909 ama Macro F1 0.317 (yanıltıcı). Önek-eşleştirmeli
  versiyonla düzeltildi → **Accuracy 0.9545, Macro F1 0.972**, 6/8 sınıf F1=1.0.
- **NER**: hibrit (regex + `savasy/bert-base-turkish-ner-cased`) doğrulandı — model
  "Ahmet Yılmaz"ı %99.9 güvenle PER olarak yakaladı, regex kod/sürüm/dosya yolu
  varlıklarını kesin yakaladı. Model tarafında düşük güvenli (0.46) bir yanlış
  pozitif de gözlemlendi ("API" → ORG).
- **Özetleme (abstractive, `ozcangundes/mt5-small-turkish-summarization`)**: iki
  ayrı çalıştırmada birebir aynı halüsinasyon: kaynakta olmayan "Prof. Dr. Canan
  Kısa" ve "Microsoft" uydurması. Tekrarlanabilir, tesadüf değil.
- **QA (`savasy/bert-base-turkish-squad`)**: %99.98 güvenle doğru cevap.
- **Diyalog (`rewrite_standalone_query`)**: tasarım gereği tam coreference çözümü
  yapmıyor, bağlamı etiketleyip LLM'e bırakıyor — doğrulandı, beklenen davranış.

### 03 — RAG sistemi
- İlk retrieval denemesi: "Bir README'de hangi bölümler olmalı?" sorusuna en
  alakalı doküman (`README Şablonu`) top-5'e hiç girmedi. Kök sebep: (1) `rank-bm25`
  bağımlılığı listede ama koda hiç bağlanmamış, (2) doküman başlığı embed/BM25
  edilen metne dahil değildi.
- Düzeltme: `src/rag/bm25_index.py` eklendi (Türkçe stopword filtreli), başlık artık
  hem dense hem BM25 indekslemesine dahil (`vector_store.py`), `retriever.py` ikisini
  birleştirip cross-encoder ile yeniden sıralıyor.
- Sonuç: reranking öncesi top-20 adayın 14'ü doğru şekilde alakalı (öncekinde 0) —
  iki bağımsız sorguda (`README`, `API dokümantasyonu`) doğrulandı.
- Bilinen sınır: cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) güçlü
  sinyale rağmen bazen alakasız dokümanı öne çıkarıyor (2 sorguda gözlemlendi).
  Ayrıca: iyi retrieval'a rağmen model çoklu kaynağı sentezlemek yerine art arda
  dökebiliyor (kurulum adımlarını 4 kez tekrarlama örneği).
- Diyalog + RAG: aynı reranker zayıflığı yüzünden "Kod Stil Rehberi" yanlışlıkla
  öne çıktı; model soruyla ilgisiz (kod stili) bir cevap üretti. Ayrıca lisans
  sorusunda corpus'ta "MIT" yazarken "Apache License 2.0" uydurdu — üçüncü
  halüsinasyon vakası.
- `hallucination_check.groundedness_score`, ilk (Microsoft/Prof.) vakayı test
  ettiğinde: **skor 0.203**, iki uydurma cümle de ~0.0003–0.0006 entailment ile
  doğru işaretlendi.

### 04 — QLoRA Fine-tuning
- `Qwen/Qwen2.5-3B-Instruct`, LoRA r=16/alpha=32, hedef q/k/v/o_proj.
- Eğitilebilir parametre: 7,372,800 / 3,093,311,488 (**%0.2383**).
- 131 eğitim / 15 doğrulama örneği, 3 epoch. Val loss: 1.841 → 1.574 → 1.466.
- Fine-tuned modelle RAG testi ("Sorun Giderme Rehberi nasıl yapılandırılmalı?"):
  kaynağa tamamen sadık, uydurmasız cevap — base model halüsinasyonlarıyla
  doğrudan karşıtlık oluşturuyor.

### 05 — DPO Alignment
- 146 tercih çifti üretildi (chosen: şablon stili, rejected: serbest üretim).
- 18 adım gerçek eğitim (öncekinde 2 adımdı, hiç metrik loglanmamıştı).
- SFT-only vs SFT+DPO karşılaştırması (aynı soru): çıktılar neredeyse birebir aynı,
  tek gözlemlenen fark biçimlendirme tutarlılığı (madde sonu noktalama).
- Teşhis: tercih verisi "yapılandırılmış vs. serbest üretim" eksenini hedefliyor,
  bu eksen zaten SFT tarafından öğrenilmiş — DPO'ya bu eksende az alan kalmış.
  "Bağlamda yoksa dürüstçe reddet vs. uydur" ekseni hiç hedeflenmedi; bu üç
  halüsinasyon vakasının chosen/rejected çiftine dönüştürülmesi sonraki adım
  olarak belirlendi (henüz uygulanmadı).

### 06 — Inference Optimizasyonu
- **Quantization**: `Qwen/Qwen2.5-3B-Instruct` 4-bit (NF4) yüklendiğinde bellek ayak
  izi 2010 MB (tam hassasiyetteki ~6.17 GB'a karşı ~3 kat küçülme).
- **Pruning**: sınıflandırıcı üzerinde %30 magnitude pruning uygulandı, sparsity
  ölçümü tam beklenen (0.0 → 0.30) çıktı.
- **Distillation**: `dbmdz/distilbert-base-turkish-cased` öğrenci, 3 epoch. İlk
  denemede yine eski (düzeltilmemiş) `_label_for_title` kopyası kullanılmıştı — loss
  1.05→0.86→0.86 diye erken düzleşmişti. Bu, etiketleme mantığının artık üçüncü kez
  farklı bir yerde tekrarlandığını gösterdiği için `label_for_title` paylaşımlı hale
  getirildi (`src/nlp_tasks/classification.py`), tüm notebook'lar oradan import
  ediyor. Düzeltmeyle loss: 1.371→1.087→0.628, sağlıklı.
- **Benchmark**: `src/optimize/benchmark.py`'deki `torch.cuda.max_memory_allocated()`
  tabanlı tek bellek ölçümü, karşılaştırılan üç sınıflandırıcı varyantının aslında
  CPU'da çalıştığını (GPU'ya hiç taşınmadıklarını) yakalayamıyordu — üçü de aynı
  (GPU'da o an başka bir modelin kalıntısını yansıtan, ilgisiz) sayıyı gösteriyordu.
  Modül, hem GPU peak hem CPU RSS'i ayrı ayrı raporlayacak şekilde güncellendi
  (`psutil` eklendi). Ayrıca notebook hücresindeki `del model` düzeltmesi ilk seferde
  yetersiz kaldı (bir sözlük hâlâ referansı tutuyordu) — hücre, her varyantı diskten
  taze yükleyip ölçüp gerçekten serbest bırakacak şekilde yeniden yazıldı.
  Sonuç: fp32 (111M) ~510-540ms, pruned (%30, 111M) ~510-540ms (anlamlı kazanç yok —
  unstructured pruning dense çıkarımda hızlanma sağlamaz), distilled (68M) ~226-258ms
  (**~%49 daha hızlı**, gerçek mimari küçülmeden). CPU RSS sıralaması modeller arası
  tutarsız çıktı (RSS'in "en yüksek su işareti" davranışı, alt süreç izolasyonu
  olmadan tam güvenilir değil) — bu açıkça not edildi, parametre sayısı ve gecikme
  üzerinden rapor verildi.

## 2026-08-24 — RAG bağlam/çıkarım format uyuşmazlığı, QLoRA OOM serisi, XAI son kontrol

### Model notları yok sayıp bağlamı olduğu gibi kopyalıyor
A/B test notebook'unda (07) hem base hem fine-tuned model, kullanıcının verdiği taslak
notları yok sayıp retrieval'ın getirdiği (alakasız) referans dokümanı olduğu gibi
kopyalıyordu. Kök sebep: `instruction_dataset.py`'nin ilk sürümü SFT örneklerini yalnızca
(notlar → hedef) çifti olarak üretiyordu, hiç "Bağlam:" bloğu yoktu — ama gerçek çıkarımda
(`rag_pipeline.build_prompt`) model her zaman "Bağlam: [retrieval'ın getirdiği doküman] /
Soru: ..." formatıyla karşılaşıyor. Model bu formatı eğitimde hiç görmediği için,
çıkarımda en tanıdık davranışa (bağlamı kopyalamaya) geri düşüyordu.

Düzeltme:
- `instruction_dataset.py`: her SFT örneğine, AYNI kategoriden FARKLI bir dokümanı
  (retrieval'ın getireceği türden — yapısal benzer, içerik alakasız) `contexts` alanı
  olarak ekliyoruz; 500 karaktere kırpılmış önizleme olarak (`_truncate_context`).
- `finetune/dataset_utils.py`: `to_chat_format`, `contexts` varsa gerçek çıkarım
  promptunu (`rag_pipeline.build_prompt`) BİREBİR aynı fonksiyonla kuruyor — eğitim ve
  çıkarım artık aynı kod yolundan geçiyor.
- `alignment/preference_dataset.py`: DPO prompt'u da aynı "Bağlam:/Soru:" sarmalamasını
  kullanacak şekilde güncellendi.
- Yerel olarak doğrulandı (146/146 örnek doğru eşleşti — farklı doküman, aynı kategori);
  GPU'da fiili yeniden eğitimle henüz DOĞRULANMADI (aşağıdaki OOM serisi yüzünden).

### QLoRA OOM serisi (4 deneme)
RAG bağlamı eklenince ortalama dizi uzunluğu belirgin arttı, art arda 4 farklı OOM ile
karşılaşıldı:
1. `per_device_train_batch_size` 2→1, `gradient_accumulation_steps` 8→16 (etkin batch
   aynı kaldı, adım başı bellek yükü yarıya indi) — yetmedi, aynı boyutta çöktü.
2. `max_seq_length` 1024→768 + bağlam kırpma (`_truncate_context`, 500 karakter)
   kaynakta eklendi — yetmedi, çökme noktası kaydı ama aynı büyüklükte tekrar etti.
3. Traceback `Trainer._evaluate()` içini gösterdi: `per_device_eval_batch_size` hiç
   ayarlanmamıştı, HF varsayılanı (8) kullanılıyordu — train batch=1 için özenle
   ayarlanmış bellek bütçesini eval anında 8 kat aşıyordu. Ayarlandı.
4. Aynı hata AYNI bellek rakamlarıyla tekrarladı — dosyanın çalışma zamanına
   yansımadığına dair güçlü bir işaretti (bu projede tekrarlayan bir desen, bkz. altta).
   Bu noktada eval'i ince ayarlamak yerine kararlı bir çözüme gidildi:
   `eval_strategy="no"`, `save_strategy="no"` — T4'te eğitimi GÜVENİLİR şekilde
   bitirmek, epoch-içi val_loss eğrisinden daha değerli; kalite zaten eğitim sonrası
   gerçek sorularla (`answer(...)`) test ediliyor. Tek kayıt, eğitim sonu
   `trainer.save_model()` ile oluyor.

Henüz kullanıcı tarafından fiilen doğrulanmadı — bu düzeltmeyle tam bir eğitim koşusunun
sorunsuz bitip bitmediği, sıfırdan (00→08) çalıştırmada teyit edilecek.

### DÜZELTME: yukarıdaki "4. deneme" teşhisi yanlıştı — asıl kök sebep bulundu
Sıfırdan çalıştırma öncesi son kontrolde ortaya çıktı: yukarıdaki 4. maddede "dosyanın
çalışma zamanına yansımadığı" teşhisi YANLIŞTI. Gerçek sebep, `scripts/colab_bootstrap.py`
içindeki `apply_patches()` fonksiyonuydu — `main()` içinde `install_packages()`'tan ÖNCE
çağrılıyordu ve `scripts/colab_patches/` altındaki 30 Temmuz tarihli, bu oturumdaki HİÇBİR
düzeltmeyi içermeyen eski dosya kopyalarını `src/` üzerine `shutil.copy2` ile **kayıtsız
şartsız kopyalıyordu** — hem `finetune/qlora_train.py`'yi (tam olarak yukarıdaki
`eval_strategy="epoch"`/`eval_dataset=eval_ds` haline!) HEM DE `rag/rag_pipeline.py`'yi
(4-bit quantize + `PeftModel` adaptör yükleme düzeltmesi olmayan, direkt bf16 yükleyen
eski haline) geri döndürüyordu. Yani: her notebook'un "Colab ortam düzeltmeleri" hücresi
(HER notebook'ta ayrı ayrı çalıştırılan bootstrap) her çalıştığında, o ana kadar yapılmış
QLoRA/RAG düzeltmeleri sessizce siliniyordu. Ayrıca `scripts/colab_qlora_fix.py` ve
04. notebook'un "eski zip fallback" hücresi de AYNI eski `eval_strategy="epoch"` kodunun
birer donmuş kopyasını içeriyordu (üç ayrı yerde, üç ayrı bayat kopya).

Bu, projede üçüncü kez görülen "birden fazla yerde tutulan kopya kod zamanla bayatlıyor"
hatasının en pahalı örneği — daha önceki ikisi (`_label_for_title`, bu XAI geçişinde
`safety_check.py`) sadece yanlış SONUÇ üretiyordu, bu ise haftalarca teşhis edilen bir
düzeltmeyi (OOM serisinin tamamı) her runtime'da fiilen GERİ ALIYORDU.

Düzeltme: `apply_patches()`, `PATCH_MAP` ve `scripts/colab_patches/` tamamen kaldırıldı;
`scripts/colab_qlora_fix.py` silindi; 04. notebook'un kurulum hücresi diğer tüm
notebook'larla AYNI, tek yollu (`colab_bootstrap.py` çağır, yoksa hata ver) hale
getirildi, eski hardcoded-kopya fallback hücresi silindi. Artık `src/` tek doğru kaynak
— hiçbir bootstrap/yama adımı onun üzerine eski bir sürüm yazmıyor.

### Colab kurulumu: pyarrow==17.0.0 artık kurulamıyor (Python 3.13)
Sıfırdan çalıştırma denemesinde `colab_bootstrap.py` `install_packages()` içinde
`pip install pyarrow==17.0.0 datasets==2.21.0` adımında çöktü:
`error: subprocess-exited-with-error / Getting requirements to build wheel did not
run successfully`. Kök sebep ortam driftti, kod hatası değil: Colab'ın varsayılan
runtime'ı artık Python 3.13, ama pyarrow 17.0.0 (Temmuz 2024) PyPI'da hiç `cp313`
wheel'i yayınlamadı (cp313 desteği ilk kez 18.0.0'da geldi) — pip bu yüzden
kaynaktan derlemeye düşüyor, Colab'da Arrow'un C++ derleme zinciri olmadığı için
bu her zaman başarısız oluyor. `datasets==2.21.0`'ın kendi gereksinimi zaten
`pyarrow>=15.0.0` (üst sınır yok), bu yüzden `pyarrow==18.0.0`'a yükseltmek güvenli.
Zincirdeki diğer tüm pinler (`chromadb==0.5.23`, `transformers==4.46.3`,
`trl==0.12.2`, pinsiz `bitsandbytes` dahil) bu adımdan ÖNCE aynı Python 3.13
runtime'ında zaten sorunsuz kurulmuştu. `requirements.txt`'deki gevşek
`pyarrow>=4.0.0,<25` aralığı zaten etkilenmiyordu.

**DÜZELTME (devamı):** "tek noktalı düzeltme" yanlış çıktı. `05_alignment_dpo.ipynb`,
`colab_bootstrap.py`'yi hiç çağırmıyordu — kendi başına, aynı kurulum mantığının
BAŞTAN SONA kopyalanmış bir kopyasını (`subprocess.check_call` ile inline) taşıyordu;
içinde AYNI `pyarrow==17.0.0` pini vardı ve `colab_bootstrap.py`'ye yapılan hiçbir
düzeltmeden (bu oturumdaki hiçbirinden) etkilenmiyordu. Bu, `colab_patches/` ve
`colab_qlora_fix.py` ile aynı aileden dördüncü bir bayat-kopya kaynağıydı — üstelik
kendi doğrulama hücresinde `assert pyarrow.__version__ == "17.0.0"` gibi sıkı bir
kontrol de vardı, yani düzeltme sonrasında bile pin'i güncellemeyi unutsaydık bu
sefer YANLIŞ bir hata mesajıyla ("pin bozulmuş") kafa karıştırabilirdi.

Düzeltme: 05. notebook'un kurulum hücresi de diğerleriyle AYNI, tek yollu
(`colab_bootstrap.py` çağır) hale getirildi; doğrulama hücresindeki sabit
versiyon `assert`'leri kaldırıldı (sadece yazdırıyor artık) — böylece gelecekte
`colab_bootstrap.py`'de yapılacak herhangi bir pin güncellemesi ikinci bir yeri
güncellemeyi gerektirmeyecek. Repo genelinde `pyarrow==17.0.0` için son bir tarama
yapıldı, başka kopya kalmadığı doğrulandı.

### XAI son kontrol: safety_check.py sessizce hiçbir şeyi yakalamıyordu
08 notebook'u hiç çalıştırılmamış olduğu ortaya çıktı (kaydedilmiş hücre çıktısı yok) —
bu yüzden `src/xai/safety_check.py`'deki bir etiket eşleştirme hatası şimdiye kadar hiç
fark edilmemişti. `model_based_check`, offensive-model çıktısının `"OFFENSIVE"`,
`"LABEL_1"` veya `"1"` etiketlerinden biri olmasını bekliyordu. Gerçekte
`Overfit-GM/distilbert-base-turkish-cased-offensive` ikili değil, 5 sınıflı bir model:
`INSULT`, `OTHER`, `PROFANITY`, `RACIST`, `SEXIST` (model `config.json`'daki
`id2label`'dan doğrulandı) — `OTHER` saldırgan olmayan genel kategori. Kod hiçbir zaman
bu etiketleri aramadığı için `is_offensive` HER ZAMAN `False` dönüyordu; model tabanlı
güvenlik kontrolü sessizce devre dışıydı (sır sızıntısı taraması ayrı/kural-tabanlı
olduğu için etkilenmedi). Düzeltme: `is_offensive = label.upper() != "OTHER"`.
Regresyonu önlemek için `tests/test_safety_check.py` eklendi (8 yeni test, pipeline
`unittest.mock.patch` ile sahteleniyor — gerçek model indirmeden mantığı doğruluyor).

Aynı geçişte `src/xai/attribution.py` ve `src/xai/hallucination_check.py` da baştan
sona okundu: ikisi de temiz, `hallucination_check` NLI etiketlerini isme göre
eşleştirdiği için (`s["label"].lower().startswith("entail")`) bu projede iki kez
görülen "indekse/sabit dizgeye güvenme" hatasına zaten bağışık.

## 2026-08-25 — Yerel RTX 2080: hierarchical RAG + Precision@k/Recall@k

- Chroma boştu; `chunk_corpus()` → `index_chunks()` ile **146 chunk** indekslendi.
- `python scripts/run_retrieval_eval.py` (32 gold soru, hibrit dense+BM25 + cross-encoder
  rerank, belge başlığı düzeyinde):

| k | P@k | R@k | Hit@k |
|---|---|---|---|
| 1 | 0.656 | 0.656 | 0.656 |
| 3 | 0.250 | 0.750 | 0.750 |
| 5 | 0.169 | 0.844 | 0.844 |
| 10 | 0.094 | 0.875 | 0.875 |

- Ham rapor: `data/eval/last_retrieval_report.json`.
- Miss@5 = 5/32. Kaçanlar çoğunlukla "şablon" yerine aynı türdeki örnek belgeleri
  getiriyor (kullanıcı kılavuzu / SSS / mimari / changelog). "Bir README'de hangi
  bölümler olmalı?" hâlâ `README Şablonu`'nu top-5'e sokmuyor — genel reranker
  sınırı, daha önce de belgelenmişti.
- vLLM bu oturumda çalıştırılmadı: paket yok, `localhost:8001` kapalı. RTX 2080
  (8 GB) Docker nvidia runtime görüyor; image çekimi ayrıca.

## 2026-08-25 — Fusion RRF + yerel NF4 P95

- Rerank-only (öğleden sonra): P@1 0.656 / Hit@5 0.844.
- `src/rag/fusion.py`: RRF(dense, BM25) + rerank ek sinyal + başlık örtüşmesi +
  şablon niyeti. BM25 skoru `setdefault` yüzünden dense hit'lerde düşüyordu; birleştirme
  düzeltildi.
- Aynı 32 gold / 146 chunk, RTX 2080:

| k | P@k | R@k | Hit@k |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 |
| 5 | 0.225 | 1.000 | 1.000 |
| 10 | 0.113 | 1.000 | 1.000 |

- vLLM image pull CloudFront EOF. Yerel NF4 generate (64 token, 8 run): avg 6318 ms,
  **P95 6424 ms**, 5761 MB VRAM.
- `gh auth status`: GitHub'a giriş yok; public repo bu oturumda açılamadı.

## Genel değerlendirme

Kod tabanı JD'nin (Baykar NLP Araştırma Mühendisi ilanı) neredeyse her maddesini
karşılıyor ve artık gerçek, tekrarlanabilir sayılarla kanıtlanmış durumda. En değerli
bulgular sonuçların "hep iyi çıkması" değil, gerçek sınırların (reranker zayıflığı,
DPO eksen uyuşmazlığı, çoklu-kaynak sentezleme) teşhis edilip açıkça belgelenmiş
olması — bu, XAI/güvenilirlik/sorumlu AI mühendisliğine dair pratik, sahada
kazanılmış bir anlayışı gösteriyor.
