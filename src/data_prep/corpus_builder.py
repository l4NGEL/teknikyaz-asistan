"""Türkçe teknik dokümantasyon örnekleri (şablon + stil kılavuzu) korpusu.

Dış kaynaktan (web scraping) veri çekmek yerine, iyi yapılandırılmış Türkçe teknik
dokümantasyon örneklerini burada elle yazıyoruz. Bunun iki nedeni var:
1. Telif/lisans riski yok — tamamen orijinal içerik.
2. RAG'ın amacı burada "genel bilgi" değil, "iyi dokümantasyon nasıl yazılır" stilini/
   yapısını öğretmek; bu yüzden kaynağın kendisi zaten hedef kalitede olmalı (retrieval
   edilen örnek ne kadar iyiyse, üretilen taslak da o kadar iyi olur).

Her şablon, gerçek bir doküman türünü (README, API dokümantasyonu, kurulum kılavuzu vb.)
ve o türe özgü yapısal kalıpları (başlık hiyerarşisi, örnek kod bloğu, adım adım liste)
örnekler.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.config import RAW_DIR

TEMPLATES: dict[str, str] = {
    "README Şablonu": """# Proje Adı

Kısa, tek cümlelik bir açıklama: bu proje ne yapar, kimin için yapılmıştır.

## Özellikler

- Öne çıkan özellik bir
- Öne çıkan özellik iki
- Öne çıkan özellik üç

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from proje import ana_islev

sonuc = ana_islev(girdi="ornek")
print(sonuc)
```

## Katkı Sağlama

Katkılarınızı bekliyoruz. Lütfen değişikliklerinizi göndermeden önce testleri çalıştırın.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "API Dokümantasyonu Şablonu": """# API Referansı: kullanici_olustur

## Açıklama

Sisteme yeni bir kullanıcı kaydı ekler ve oluşturulan kullanıcının kimliğini döndürür.

## İmza

```
kullanici_olustur(ad: str, e_posta: str, rol: str = "standart") -> int
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| ad | str | Evet | Kullanıcının tam adı |
| e_posta | str | Evet | Benzersiz olmalıdır |
| rol | str | Hayır | "standart" veya "yönetici", varsayılan "standart" |

## Dönüş Değeri

Oluşturulan kullanıcının benzersiz kimlik numarası (int).

## Hatalar

- `DuplicateEmailError`: e_posta zaten kayıtlıysa fırlatılır.
- `ValidationError`: ad boş bırakılırsa fırlatılır.

## Örnek

```python
kullanici_id = kullanici_olustur(ad="Ayşe Yılmaz", e_posta="ayse@ornek.com")
```""",

    "Kurulum Kılavuzu Şablonu": """# Kurulum Kılavuzu

## Ön Koşullar

- Python 3.10 veya üzeri
- En az 4 GB RAM
- İnternet bağlantısı (bağımlılıkların indirilmesi için)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd proje-klasoru
```

## Adım 2: Sanal Ortam Oluşturun

```bash
python -m venv .venv
source .venv/bin/activate
```

## Adım 3: Bağımlılıkları Kurun

```bash
pip install -r requirements.txt
```

## Adım 4: Kurulumu Doğrulayın

```bash
python -c "import proje; print(proje.__version__)"
```

Yukarıdaki komut bir hata vermeden sürüm numarasını yazdırıyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Eğer `ModuleNotFoundError` alıyorsanız, sanal ortamın aktif olduğundan emin olun.""",

    "Kullanıcı Kılavuzu Şablonu": """# Kullanıcı Kılavuzu: Rapor Oluşturma

Bu bölüm, sistemde aylık rapor oluşturma adımlarını açıklar.

## 1. Rapor Menüsünü Açın

Sol menüden "Raporlar" sekmesine tıklayın.

## 2. Tarih Aralığı Seçin

Başlangıç ve bitiş tarihlerini seçtikten sonra "Filtrele" butonuna basın.

## 3. Rapor Formatını Belirleyin

PDF, Excel veya CSV formatlarından birini seçebilirsiniz.

## 4. Raporu İndirin

"Oluştur" butonuna bastıktan sonra rapor otomatik olarak indirilir.

> Not: Büyük tarih aralıkları için rapor oluşturma birkaç dakika sürebilir.""",

    "SSS (Sıkça Sorulan Sorular) Şablonu": """# Sıkça Sorulan Sorular

**S: Şifremi unuttum, ne yapmalıyım?**

C: Giriş ekranındaki "Şifremi Unuttum" bağlantısına tıklayarak e-posta adresinize
sıfırlama bağlantısı gönderebilirsiniz.

**S: Verilerim ne kadar süre saklanıyor?**

C: Hesap verileriniz, hesabınızı silene kadar saklanır. Silme talebinden sonra 30 gün
içinde kalıcı olarak temizlenir.

**S: API istek limiti nedir?**

C: Standart planda dakikada 60 istek, kurumsal planda dakikada 600 istek limiti vardır.

**S: Mobil uygulama var mı?**

C: Şu anda yalnızca web arayüzü sunulmaktadır; mobil uygulama yol haritamızda yer
almaktadır.""",

    "Mimari Doküman Şablonu": """# Sistem Mimarisi Genel Bakış

## Amaç

Bu doküman, sistemin ana bileşenlerini ve bunlar arasındaki veri akışını açıklar.

## Bileşenler

- **API Katmanı**: Dış isteklerin karşılandığı, doğrulama ve yetkilendirmenin yapıldığı katman.
- **İş Mantığı Katmanı**: Temel iş kurallarının uygulandığı katman.
- **Veri Katmanı**: Kalıcı depolama (veritabanı) ile etkileşim.
- **Mesaj Kuyruğu**: Asenkron işlerin (örn. bildirim gönderimi) yönetildiği bileşen.

## Veri Akışı

1. İstemci, API katmanına bir HTTP isteği gönderir.
2. API katmanı isteği doğrular ve iş mantığı katmanına iletir.
3. İş mantığı katmanı gerekli veri katmanı çağrılarını yapar.
4. Sonuç istemciye döndürülür; yan etkiler (örn. e-posta gönderimi) mesaj kuyruğuna eklenir.

## Ölçeklenebilirlik Notları

API katmanı yatay olarak ölçeklenebilir; veri katmanı için okuma replikaları kullanılır.""",

    "Sürüm Notları (Changelog) Şablonu": """# Sürüm Notları

## v2.3.0 — 2026-01-15

### Eklenenler
- Kullanıcılar artık raporlarını CSV formatında dışa aktarabilir.

### Değişenler
- Giriş ekranının performansı iyileştirildi (ortalama yükleme süresi %40 azaldı).

### Düzeltilenler
- Belirli tarayıcılarda tarih seçicinin açılmaması sorunu giderildi.

## v2.2.1 — 2025-12-02

### Düzeltilenler
- Kritik bir güvenlik açığı (CVE-2025-XXXXX) kapatıldı. Tüm kullanıcıların güncellemesi
  önerilir.""",

    "Katkı Sağlama Rehberi Şablonu": """# Katkı Sağlama Rehberi

Bu projeye katkıda bulunduğunuz için teşekkür ederiz. Lütfen katkı göndermeden önce
aşağıdaki adımları izleyin.

## Geliştirme Ortamı Kurulumu

1. Depoyu fork'layın ve yerel makinenize klonlayın.
2. `pip install -r requirements-dev.txt` ile geliştirme bağımlılıklarını kurun.
3. `pytest` ile mevcut testlerin geçtiğini doğrulayın.

## Kod Standartları

- Tüm yeni kod için birim testi eklenmelidir.
- Fonksiyon ve değişken isimleri Türkçe veya İngilizce olabilir, ancak dosya içinde
  tutarlı olmalıdır.
- Commit mesajları, değişikliğin "ne" değil "neden" yapıldığını açıklamalıdır.

## Pull Request Süreci

1. Değişikliklerinizi ayrı bir dalda (branch) yapın.
2. Açıklayıcı bir başlık ve açıklama ile pull request açın.
3. En az bir onay aldıktan sonra birleştirilir (merge edilir).""",

    "Sorun Giderme Rehberi Şablonu": """# Sorun Giderme Rehberi

## Belirti: Uygulama başlatılamıyor

**Olası Neden 1:** Ortam değişkenleri eksik.
**Çözüm:** `.env.example` dosyasını `.env` olarak kopyalayıp gerekli değerleri doldurun.

**Olası Neden 2:** Port zaten kullanımda.
**Çözüm:** `lsof -i :8000` ile portu kullanan süreci bulup sonlandırın ya da farklı bir
port belirtin.

## Belirti: Veritabanı bağlantı hatası

**Olası Neden:** Veritabanı servisi çalışmıyor ya da bağlantı bilgileri yanlış.
**Çözüm:** `docker ps` ile veritabanı konteynerinin çalıştığını doğrulayın, `.env`
içindeki `DATABASE_URL` değerini kontrol edin.

## Destek Talebi Açma

Yukarıdaki adımlar sorunu çözmediyse, hata mesajının tam metni ve loglarla birlikte
destek talebi açın.""",

    "Test Planı Dokümanı Şablonu": """# Test Planı: Ödeme Modülü

## Kapsam

Bu test planı, ödeme modülünün kart doğrulama, işlem onayı ve iade akışlarını kapsar.

## Test Senaryoları

| ID | Senaryo | Beklenen Sonuç |
|---|---|---|
| T-01 | Geçerli kart ile ödeme | İşlem onaylanır, makbuz oluşturulur |
| T-02 | Süresi dolmuş kart ile ödeme | İşlem reddedilir, kullanıcıya uygun hata gösterilir |
| T-03 | Yetersiz bakiye | İşlem reddedilir, "yetersiz bakiye" mesajı gösterilir |
| T-04 | Kısmi iade | Belirtilen tutar kadar iade işlenir, orijinal işlem güncellenir |

## Test Ortamı

Testler, gerçek ödeme sağlayıcısı yerine sandbox (test) ortamında çalıştırılmalıdır.""",
}


@dataclass
class DocTemplate:
    title: str
    text: str

    def to_dict(self) -> dict:
        return {"title": self.title, "text": self.text, "url": f"local://templates/{self.title}"}


def _normalize(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def build_corpus() -> list[DocTemplate]:
    return [DocTemplate(title=title, text=_normalize(text)) for title, text in TEMPLATES.items()]


def save_corpus(docs: list[DocTemplate], filename: str = "corpus.jsonl") -> str:
    out_path = RAW_DIR / filename
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    return str(out_path)


if __name__ == "__main__":
    corpus = build_corpus()
    path = save_corpus(corpus)
    print(f"{len(corpus)} şablon doküman kaydedildi -> {path}")
