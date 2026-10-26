# Elasticsearch arka ucu

`RETRIEVAL_BACKEND=elasticsearch` ile retriever, dense ve BM25 adaylarını Chroma +
`rank_bm25` yerine tek bir Elasticsearch 8.15 indeksinden alır. Füzyon (RRF + başlık
sinyali), ACL süzmesi ve cross-encoder rerank aynı kalır; yalnızca aday kaynağı değişir.

| Dosya | İçerik |
|---|---|
| `src/rag/es_index.py` | İndeks şeması, toplu indeksleme, BM25 ve kNN sorguları |
| `src/rag/retriever.py` | `backend` parametresi |
| `scripts/run_es_eval.py` | İki arka ucu aynı gold sette karşılaştırır |
| `tests/test_es_index.py` | Şema/sorgu birim testleri + canlı analizör testi (`ES_URL` varsa) |

## Çalıştırma

```bash
docker run -d --name teknikyaz-es -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.3
python scripts/run_es_eval.py
```

## İndeks

Tek indekste her chunk için:

- `embedding`: `dense_vector`, 1024 boyut (multilingual-e5-large), kosinüs, HNSW.
  Chroma'yla aynı `"başlık: metin"` pasaj biçimi gömülüyor.
- Üç BM25 alanı, hepsi `başlık + gövde`. Analizörün etkisini aynı motor ve aynı BM25
  parametreleriyle ayrı ölçmek için:
  - `title_text_std`: standard analizör, kök bulma yok.
  - `title_text`: yerleşik `turkish` analizörü (Snowball kök bulucu).
  - `title_text_f5`: Türkçe küçük harf + kesme işareti temizliği + stopword + her
    token'ı ilk 5 harfe kesme. Varsayılan alan.

## Sonuçlar

32 soruluk gold set (`data/eval/retrieval_gold.jsonl`), belge başlığı düzeyinde.
Ham çıktı: `data/eval/es_comparison.json`.

| Yapılandırma | Hit@1 | Hit@3 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| BM25 – rank_bm25 (yerel) | 0.812 | 0.875 | 0.906 | 0.846 |
| BM25 – ES standard | 0.750 | 0.781 | 0.812 | 0.781 |
| BM25 – ES turkish (Snowball) | 0.844 | 0.875 | 0.875 | 0.854 |
| BM25 – ES F5 | 0.781 | 0.844 | 0.906 | 0.831 |
| Dense – Chroma | 0.906 | 0.969 | 1.000 | 0.944 |
| Dense – ES kNN | 0.906 | 0.969 | 0.969 | 0.938 |
| **Hibrit – yerel** (varsayılan) | **1.000** | **1.000** | **1.000** | **1.000** |
| **Hibrit – ES** | 0.969 | 0.969 | 1.000 | 0.977 |

Gecikme: ES sunucusunun kendi ölçtüğü sorgu süresi p50 **10 ms (BM25)** ve **11 ms
(kNN)**. İstemci tarafında ölçülen ~55 ms'nin çoğu Windows'taki Docker Desktop ağ
katmanından geliyor; bu dizüstü ölçümü üretim gecikmesi değildir. Bellekteki
`rank_bm25` 146 chunk'ta 0,4 ms'de kalıyor, korpus büyüdükçe bu avantaj kaybolur.

## Bulgular

1. **Türkçe kök bulucu tutarsız.** İlk ölçümde hibrit ES hattı, "Mimari doküman
   şablonunda hangi bölümler olmalı?" sorusunda doğru belgeyi ilk 5'e bile
   getirmedi. Sebep: Snowball `Şablonu` kelimesini `şablo`, sorudaki `şablonunda`
   kelimesini `şablon` köküne indiriyor; belge kendi adını taşıyan sorguda BM25
   adayı olamıyor (146 chunk içinde 23. sıra). Kelimeleri ilk 5 harfe kesen F5
   alanı ikisini de `şablo` yapıyor; bu alanla hibrit Recall@5 tekrar 1.000 oldu.
   Davranış `tests/test_es_index.py` içinde canlı testle sabitlendi.
2. **Tek başına hiçbir analizör kazanmıyor.** Snowball Hit@1'de en iyi BM25 (0.844),
   F5 Recall@5'te (0.906). Hibrit hatta dense aday listesi bu farkı büyük ölçüde
   kapatıyor.
3. **ES kNN, Chroma ile Hit@1'de aynı**; Recall@5'te bir soru farkı var (yaklaşık
   HNSW araması, `num_candidates=100`).
4. **Varsayılan yerel arka uç kalıyor.** Bu küçük korpusta hibrit ES, yerel hattın
   0.969'unda (32 sorunun 31'i ilk sırada) kalıyor. ES'nin getirisi ölçek, aynı
   indekste vektör + tam metin araması ve sunucu tarafında ~10 ms sorgu süresi.

## Sınırlar

- Gold set küçük (32 soru) ve analizör seçimi de bu sette yapıldı; ayrı bir test
  seti yok. F5 literatürde bilinen bir Türkçe yaklaşımı olduğu için seçildi, ama
  sonuçlar bu sette ayarlanmış sayılmalı.
- ES'nin yerleşik RRF retriever'ı kullanılmadı; füzyon mevcut `fusion.py` ile
  istemci tarafında yapılıyor, böylece iki arka uç aynı füzyon kodunu paylaşıyor.
