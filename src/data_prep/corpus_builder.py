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

    # --- README (2. tur) ---
    "README — Randevu ve Rezervasyon Sistemi": """# Randevu ve Rezervasyon Sistemi

Hizmet sağlayıcıları için müsaitlik takvimi, randevu alma ve hatırlatma
bildirimlerini yöneten kütüphane.

## Özellikler

- Zaman dilimi (timezone) farkındalıklı müsaitlik hesaplama
- Çakışan randevuları otomatik reddetme
- Randevudan 24 saat önce otomatik hatırlatma

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from randevu import musaitlik_bul, randevu_al

slotlar = musaitlik_bul(saglayici_id="doktor_12", tarih="2026-03-10")
randevu_al(saglayici_id="doktor_12", slot=slotlar[0], musteri_id="m_5")
```

## Katkı Sağlama

Yeni bir hatırlatma kanalı eklerken mevcut zamanlama testlerini çalıştırın.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Anket ve Form Oluşturucu": """# Anket ve Form Oluşturucu

Sürükle-bırak arayüzle anket/form tasarlamayı ve yanıtları toplamayı sağlayan
kütüphane.

## Özellikler

- Koşullu soru mantığı (bir yanıta göre sonraki soruyu belirleme)
- Yanıtları CSV/JSON olarak dışa aktarma
- Form başına özelleştirilebilir marka/tema

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from anket import Form

form = Form(baslik="Memnuniyet Anketi")
form.soru_ekle("Hizmetimizi tavsiye eder misiniz?", tip="olcek_1_10")
```

## Katkı Sağlama

Yeni bir soru tipi eklerken `sorular/` altındaki temel sınıfı genişletin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Sohbet Widget'ı": """# Sohbet Widget'ı

Web sitelerine gömülebilen, canlı destek ve otomatik yanıt botunu birleştiren
sohbet widget'ı.

## Özellikler

- Mesai saatleri dışında otomatik bot yanıtına geçiş
- Ziyaretçi bilgilerini (sayfa, tarayıcı) temsilciye otomatik iletme
- Sohbet geçmişini dışa aktarma

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```html
<script src="sohbet-widget.js" data-site-id="abc123"></script>
```

## Katkı Sağlama

Yeni bir dil paketi eklerken `diller/` klasörüne çeviri dosyası ekleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Belge İmzalama Servisi": """# Belge İmzalama Servisi

PDF belgeleri için elektronik imza talebi oluşturan ve imza durumunu takip eden
servis.

## Özellikler

- Çoklu imzacı sırası (birinci imzacı imzalamadan ikinciye bildirim gitmez)
- İmza denetim izi (audit trail): kim, ne zaman, hangi IP'den imzaladı
- Süresi dolan imza taleplerini otomatik iptal etme

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from eimza import imza_talebi_olustur

talep = imza_talebi_olustur(belge_yolu="sozlesme.pdf", imzacilar=["a@ornek.com", "b@ornek.com"])
```

## Katkı Sağlama

İmza sağlayıcı entegrasyonu eklerken sandbox kimlik bilgilerini `.env.example`'a ekleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Sadakat ve Puan Programı": """# Sadakat ve Puan Programı

Müşteri işlemlerine karşılık puan biriktiren, seviye ve ödül tanımlayan
sadakat programı kütüphanesi.

## Özellikler

- Yapılandırılabilir puan kazanma kuralları (tutar başına, kampanya bazlı)
- Otomatik seviye yükseltme/düşürme
- Puanların son kullanma tarihi desteği

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from sadakat import puan_ekle

puan_ekle(musteri_id="m_88", tutar=250.0, kaynak="siparis")
```

## Katkı Sağlama

Yeni bir seviye kuralı eklerken `seviyeler.yaml` dosyasını güncelleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Destek Bileti Sistemi": """# Destek Bileti Sistemi

Müşteri destek taleplerini bilet olarak kaydeden, önceliklendiren ve doğru
ekibe yönlendiren sistem.

## Özellikler

- Anahtar kelimeye göre otomatik kategori/öncelik atama
- SLA (hizmet seviyesi anlaşması) süresi aşımında otomatik uyarı
- Müşteriyle bilet üzerinden e-posta senkronize yazışma

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from destek import bilet_olustur

bilet = bilet_olustur(musteri_id="m_3", konu="Fatura hatası", oncelik="yuksek")
```

## Katkı Sağlama

Yeni bir yönlendirme kuralı eklerken `kurallar/` klasörüne test ekleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Görev Planlama Arayüzü": """# Görev Planlama Arayüzü

Saha ekiplerinin görev/rota planlarını oluşturup onay akışından geçirdiği
planlama arayüzü.

## Özellikler

- Harita üzerinde çoklu nokta ile rota taslağı oluşturma
- Onay öncesi çakışma ve kaynak müsaitliği kontrolü
- Planlanan görevi PDF/KML olarak dışa aktarma

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from gorev_planlama import taslak_olustur

taslak = taslak_olustur(ekip_id="ekip_4", noktalar=[(39.9, 32.8), (39.95, 32.9)])
```

## Katkı Sağlama

Yeni bir onay adımı eklerken akış diyagramını `docs/akis.md` içinde güncelleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    # --- API Dokümantasyonu (2. tur) ---
    "API Referansı: randevu_olustur": """# API Referansı: randevu_olustur

## Açıklama

Belirtilen sağlayıcı ve zaman diliminde yeni bir randevu kaydı oluşturur.

## İmza

```
randevu_olustur(saglayici_id: str, musteri_id: str, baslangic: str, sure_dk: int = 30) -> dict
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| saglayici_id | str | Evet | Hizmet sağlayıcısının kimliği |
| musteri_id | str | Evet | Randevu alan müşterinin kimliği |
| baslangic | str | Evet | ISO 8601 tarih-saat |
| sure_dk | int | Hayır | Dakika cinsinden süre, varsayılan 30 |

## Dönüş Değeri

`{"randevu_id": str, "durum": "onaylandi"}` şeklinde bir sözlük.

## Hatalar

- `SlotUnavailableError`: seçilen zaman dilimi doluysa fırlatılır.
- `OutsideWorkingHoursError`: sağlayıcının çalışma saatleri dışındaysa fırlatılır.

## Örnek

```python
randevu = randevu_olustur(saglayici_id="doktor_12", musteri_id="m_5", baslangic="2026-03-10T14:00:00")
```""",

    "API Referansı: form_gonder": """# API Referansı: form_gonder

## Açıklama

Bir forma ait kullanıcı yanıtlarını kaydeder ve doğrulama kurallarını çalıştırır.

## İmza

```
form_gonder(form_id: str, yanitlar: dict, gonderen_id: str | None = None) -> str
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| form_id | str | Evet | Yanıtlanan formun kimliği |
| yanitlar | dict | Evet | Soru kimliği → yanıt eşlemesi |
| gonderen_id | str | Hayır | Anonim gönderimlerde boş bırakılabilir |

## Dönüş Değeri

Kaydedilen yanıtın benzersiz kimliği (str).

## Hatalar

- `RequiredQuestionMissingError`: zorunlu bir soru boş bırakılırsa fırlatılır.
- `FormClosedError`: form kapatılmışsa fırlatılır.

## Örnek

```python
yanit_id = form_gonder(form_id="anket_7", yanitlar={"q1": "Evet", "q2": 9})
```""",

    "API Referansı: imza_talebi_olustur": """# API Referansı: imza_talebi_olustur

## Açıklama

Bir PDF belgesi için bir veya daha fazla imzacıya elektronik imza talebi gönderir.

## İmza

```
imza_talebi_olustur(belge_yolu: str, imzacilar: list[str], sira_onemli: bool = True) -> str
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| belge_yolu | str | Evet | İmzalanacak PDF'nin depodaki yolu |
| imzacilar | list[str] | Evet | İmzacıların e-posta adresleri, sırayla |
| sira_onemli | bool | Hayır | True ise bir önceki imzalamadan sonraki bilgilendirilir |

## Dönüş Değeri

Talebin durumunu izlemek için kullanılan talep kimliği (str).

## Hatalar

- `InvalidDocumentError`: dosya PDF değilse fırlatılır.
- `DuplicateSignerError`: aynı e-posta birden fazla kez listelenirse fırlatılır.

## Örnek

```python
talep_id = imza_talebi_olustur(belge_yolu="sozlesme.pdf", imzacilar=["a@ornek.com", "b@ornek.com"])
```""",

    "API Referansı: puan_ekle": """# API Referansı: puan_ekle

## Açıklama

Bir müşterinin sadakat puan bakiyesine, belirtilen kaynağa dayalı olarak puan ekler.

## İmza

```
puan_ekle(musteri_id: str, tutar: float, kaynak: str) -> int
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| musteri_id | str | Evet | Puanı eklenecek müşteri |
| tutar | float | Evet | İşlem tutarı (puan hesaplamasının tabanı) |
| kaynak | str | Evet | "siparis", "yorum" veya "davet" |

## Dönüş Değeri

Müşterinin güncel toplam puan bakiyesi (int).

## Hatalar

- `UnknownSourceError`: kaynak tanımlı değerlerden biri değilse fırlatılır.
- `CustomerNotFoundError`: müşteri kimliği bulunamazsa fırlatılır.

## Örnek

```python
yeni_bakiye = puan_ekle(musteri_id="m_88", tutar=250.0, kaynak="siparis")
```""",

    "API Referansı: bilet_olustur": """# API Referansı: bilet_olustur

## Açıklama

Yeni bir destek bileti oluşturur ve anahtar kelimelere göre otomatik öncelik atar.

## İmza

```
bilet_olustur(musteri_id: str, konu: str, aciklama: str, oncelik: str | None = None) -> dict
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| musteri_id | str | Evet | Bileti açan müşteri |
| konu | str | Evet | Kısa başlık |
| aciklama | str | Evet | Detaylı açıklama |
| oncelik | str | Hayır | Boşsa içerikten otomatik belirlenir |

## Dönüş Değeri

`{"bilet_id": str, "oncelik": str, "atanan_ekip": str}` şeklinde bir sözlük.

## Hatalar

- `EmptyDescriptionError`: açıklama boşsa fırlatılır.

## Örnek

```python
bilet = bilet_olustur(musteri_id="m_3", konu="Fatura hatası", aciklama="Çift ücretlendirme oldu")
```""",

    "API Referansı: ucus_plani_onayla": """# API Referansı: ucus_plani_onayla

## Açıklama

Bekleyen bir görev planını inceleyip onaylar veya gerekçeyle reddeder.

## İmza

```
ucus_plani_onayla(plan_id: str, onaylayan_id: str, karar: str, gerekce: str | None = None) -> None
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| plan_id | str | Evet | Onaylanacak planın kimliği |
| onaylayan_id | str | Evet | Onay yetkisine sahip kullanıcı |
| karar | str | Evet | "onayla" veya "reddet" |
| gerekce | str | Hayır | "reddet" seçildiğinde zorunlu |

## Dönüş Değeri

Yok; başarılı çağrı HTTP 204 ile sonuçlanır.

## Hatalar

- `MissingReasonError`: red kararında gerekçe boşsa fırlatılır.
- `AlreadyDecidedError`: plan zaten karara bağlanmışsa fırlatılır.

## Örnek

```python
ucus_plani_onayla(plan_id="plan_9", onaylayan_id="yetkili_2", karar="onayla")
```""",

    "API Referansı: parca_stok_sorgula": """# API Referansı: parca_stok_sorgula

## Açıklama

Belirtilen yedek parçanın depolardaki güncel stok miktarını sorgular.

## İmza

```
parca_stok_sorgula(parca_kodu: str, depo_id: str | None = None) -> list[dict]
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| parca_kodu | str | Evet | Parçanın benzersiz kodu |
| depo_id | str | Hayır | Boşsa tüm depolar taranır |

## Dönüş Değeri

Her biri `{"depo_id": str, "miktar": int, "min_seviye": int}` içeren bir liste.

## Hatalar

- `UnknownPartError`: parça kodu tanımlı değilse fırlatılır.

## Örnek

```python
stoklar = parca_stok_sorgula(parca_kodu="RTR-2201")
```""",

    # --- Kurulum Kılavuzu (2. tur) ---
    "Kurulum Kılavuzu — Randevu Sistemi": """# Kurulum Kılavuzu — Randevu Sistemi

## Ön Koşullar

- Python 3.10 veya üzeri
- PostgreSQL 14+
- Saat dilimi veritabanının güncel olması (`tzdata` paketi)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd randevu-sistemi
```

## Adım 2: Veritabanını Hazırlayın

```bash
createdb randevu
python -m randevu.migrate
```

## Adım 3: Servisi Başlatın

```bash
pip install -r requirements.txt
python -m randevu.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl localhost:8010/musaitlik?saglayici_id=test
```

Boş bir liste bile olsa hatasız dönüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Müsaitlik saatleri yanlış görünüyorsa sunucu ve veritabanının aynı saat
dilimini kullandığından emin olun.""",

    "Kurulum Kılavuzu — Form Motoru": """# Kurulum Kılavuzu — Form Motoru

## Ön Koşullar

- Node.js 18 veya üzeri
- MongoDB 6+

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd form-motoru
```

## Adım 2: Ortam Değişkenlerini Ayarlayın

```bash
cp .env.example .env
```

## Adım 3: Bağımlılıkları Kurun ve Başlatın

```bash
npm install
npm run start
```

## Adım 4: Kurulumu Doğrulayın

Tarayıcıda `localhost:3000/ornek-form` açıldığında form görüntüleniyorsa
kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Form kaydedilmiyorsa MongoDB bağlantı dizesinin `.env` içinde doğru
olduğundan emin olun.""",

    "Kurulum Kılavuzu — Video Konferans Sunucusu": """# Kurulum Kılavuzu — Video Konferans Sunucusu

## Ön Koşullar

- Docker ve Docker Compose
- Açık UDP port aralığı (medya trafiği için)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd video-sunucusu
```

## Adım 2: Port Aralığını Yapılandırın

```bash
# docker-compose.yml içinde UDP_PORT_RANGE değerini ağınıza göre ayarlayın
```

## Adım 3: Sunucuyu Başlatın

```bash
docker compose up -d
```

## Adım 4: Kurulumu Doğrulayın

İki farklı tarayıcı sekmesinden test odasına katılıp karşılıklı görüntü/ses
geldiğini doğrulayın.

## Sık Karşılaşılan Sorunlar

Bağlantı kuruluyor ama görüntü gelmiyorsa güvenlik duvarında UDP port
aralığının açık olduğunu kontrol edin.""",

    "Kurulum Kılavuzu — Sadakat Servisi": """# Kurulum Kılavuzu — Sadakat Servisi

## Ön Koşullar

- Python 3.10 veya üzeri
- Redis (puan bakiyesi önbelleği için)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd sadakat-servisi
```

## Adım 2: Seviye Kurallarını Tanımlayın

```bash
cp seviyeler.example.yaml seviyeler.yaml
```

## Adım 3: Servisi Başlatın

```bash
pip install -r requirements.txt
python -m sadakat.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
python -m sadakat.test_ekle --musteri test --tutar 100
```

Bakiye artışı yazdırılıyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Puanlar anlık yansımıyorsa Redis önbelleğinin süresinin (TTL) çok uzun
ayarlanmadığından emin olun.""",

    "Kurulum Kılavuzu — Destek Bileti Sistemi": """# Kurulum Kılavuzu — Destek Bileti Sistemi

## Ön Koşullar

- Python 3.10 veya üzeri
- PostgreSQL 14+
- Gelen e-postaları okumak için bir IMAP hesabı (opsiyonel)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd destek-sistemi
```

## Adım 2: Veritabanını Kurun

```bash
python -m destek.migrate
```

## Adım 3: Sunucuyu Başlatın

```bash
pip install -r requirements.txt
python -m destek.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl -X POST localhost:8020/biletler -d '{"musteri_id":"test","konu":"deneme"}'
```

Bir bilet kimliği dönüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Otomatik öncelik ataması çalışmıyorsa anahtar kelime kurallarının
yüklendiğini (`kurallar/oncelik.yaml`) doğrulayın.""",

    "Kurulum Kılavuzu — Filo Bakım Takip Sistemi": """# Kurulum Kılavuzu — Filo Bakım Takip Sistemi

## Ön Koşullar

- Python 3.10 veya üzeri
- PostgreSQL 14+

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd filo-bakim
```

## Adım 2: Araç/Envanter Verisini İçe Aktarın

```bash
python -m filo.ice_aktar --dosya envanter.csv
```

## Adım 3: Servisi Başlatın

```bash
pip install -r requirements.txt
python -m filo.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl localhost:8030/araclar
```

İçe aktarılan araç listesi dönüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Bakım hatırlatmaları gelmiyorsa zamanlayıcı servisinin (bkz. Görev
Zamanlayıcı) ayrıca çalışır durumda olduğundan emin olun.""",

    "Kurulum Kılavuzu — Görev Planlama Sunucusu": """# Kurulum Kılavuzu — Görev Planlama Sunucusu

## Ön Koşullar

- Python 3.10 veya üzeri
- PostGIS eklentili PostgreSQL

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd gorev-planlama
```

## Adım 2: Coğrafi Veritabanını Hazırlayın

```bash
python -m gorev_planlama.migrate --with-postgis
```

## Adım 3: Sunucuyu Başlatın

```bash
pip install -r requirements.txt
python -m gorev_planlama.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl localhost:8040/saglik
```

`{"durum": "ok"}` dönüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Rota çizimi hata veriyorsa PostGIS eklentisinin veritabanında etkin
olduğunu (`CREATE EXTENSION postgis;`) kontrol edin.""",

    # --- Kullanıcı Kılavuzu (2. tur) ---
    "Kullanıcı Kılavuzu: Randevu Alma": """# Kullanıcı Kılavuzu: Randevu Alma

Bu bölüm, sistem üzerinden bir hizmet sağlayıcısından randevu almayı açıklar.

## 1. Hizmet Sağlayıcısını Seçin

Arama kutusundan istediğiniz hizmet sağlayıcısını bulun.

## 2. Müsait Bir Zaman Seçin

Takvimde yeşil renkle işaretli müsait saatlerden birine tıklayın.

## 3. Bilgilerinizi Onaylayın

Ad, iletişim bilgisi ve varsa notunuzu girip "Randevuyu Onayla" butonuna
basın.

## 4. Hatırlatma Alın

Randevudan 24 saat önce otomatik olarak e-posta veya SMS hatırlatması
alırsınız.

> Not: Randevunuzu iptal etmek için onay e-postasındaki bağlantıyı
> kullanabilirsiniz.""",

    "Kullanıcı Kılavuzu: Anket Oluşturma": """# Kullanıcı Kılavuzu: Anket Oluşturma

Bu bölüm, sıfırdan bir müşteri memnuniyeti anketi oluşturmayı açıklar.

## 1. Yeni Form Oluşturun

"Formlarım" sekmesinden "Yeni Form" butonuna tıklayın.

## 2. Soruları Ekleyin

Sağdaki soru tiplerinden (çoktan seçmeli, ölçek, açık uçlu) sürükleyerek
forma ekleyin.

## 3. Koşullu Mantık Tanımlayın (İsteğe Bağlı)

Bir sorunun yanıtına göre sonraki sorunun değişmesini istiyorsanız "Koşul
Ekle" seçeneğini kullanın.

## 4. Yayınlayın ve Paylaşın

"Yayınla" butonuna bastıktan sonra oluşan bağlantıyı paylaşabilirsiniz.

> Not: Yanıtlar "Sonuçlar" sekmesinden gerçek zamanlı olarak izlenebilir.""",

    "Kullanıcı Kılavuzu: Kupon Kullanma": """# Kullanıcı Kılavuzu: Kupon Kullanma

Bu bölüm, ödeme sırasında bir indirim kuponu uygulamayı açıklar.

## 1. Sepetinizi Tamamlayın

Satın almak istediğiniz ürünleri sepete ekleyip ödeme adımına geçin.

## 2. Kupon Kodunu Girin

"Kupon Kodu" alanına kodu yazıp "Uygula" butonuna basın.

## 3. İndirimi Doğrulayın

Uygulanan indirim tutarı sipariş özetinde görünür.

## 4. Ödemeyi Tamamlayın

Kalan tutar üzerinden normal ödeme adımlarına devam edin.

> Not: Bir siparişte yalnızca bir kupon kodu kullanılabilir; kampanyalar
> otomatik olarak birleştirilmez.""",

    "Kullanıcı Kılavuzu: Destek Bileti Açma": """# Kullanıcı Kılavuzu: Destek Bileti Açma

Bu bölüm, bir sorun yaşadığınızda destek ekibiyle nasıl iletişime
geçeceğinizi açıklar.

## 1. Yardım Merkezini Açın

Sağ alttaki "Yardım" simgesine tıklayın.

## 2. Konuyu ve Açıklamayı Girin

Sorununuzu kısaca özetleyin, mümkünse ekran görüntüsü ekleyin.

## 3. Önceliği Kontrol Edin

Sistem, açıklamanıza göre otomatik bir öncelik seviyesi atar; gerekirse
değiştirebilirsiniz.

## 4. Yanıtı Takip Edin

Bilet durumunu "Biletlerim" sekmesinden takip edebilir, ek bilgi
ekleyebilirsiniz.

> Not: Kritik sorunlar için "Acil" önceliğini seçmeniz yanıt süresini
> kısaltır.""",

    "Kullanıcı Kılavuzu: Ekip Vardiyası Planlama": """# Kullanıcı Kılavuzu: Ekip Vardiyası Planlama

Bu bölüm, haftalık ekip vardiya planı oluşturmayı açıklar.

## 1. Vardiya Takvimini Açın

Sol menüden "Vardiyalar" sekmesine gidin.

## 2. Ekip Üyelerini Atayın

Her güne, müsaitlik durumuna göre ekip üyelerini sürükleyip bırakın.

## 3. Çakışmaları Kontrol Edin

Sistem, bir üyeye aynı anda iki vardiya atanırsa uyarı gösterir.

## 4. Planı Yayınlayın

"Yayınla" butonuna bastığınızda tüm ekip üyelerine bildirim gider.

> Not: Yayınlanmış bir plandaki değişiklikler, etkilenen üyelere ayrıca
> bildirilir.""",

    "Kullanıcı Kılavuzu: Uçuş Planı Onaylama": """# Kullanıcı Kılavuzu: Uçuş Planı Onaylama

Bu bölüm, yetkili bir kullanıcının bekleyen bir görev planını onaylama
adımlarını açıklar.

## 1. Onay Kuyruğunu Açın

Sol menüden "Onay Bekleyenler" sekmesine gidin.

## 2. Plan Detaylarını İnceleyin

Rota, tahmini süre ve kaynak kullanımı bilgilerini gözden geçirin.

## 3. Karar Verin

"Onayla" veya "Reddet" butonuna basın; reddederken bir gerekçe girmeniz
istenir.

## 4. Bildirimi Kontrol Edin

Planı oluşturan ekip, kararınızdan anında bildirimle haberdar edilir.

> Not: Onaylanmış bir plan, yalnızca yetkili bir kullanıcı tarafından
> iptal edilebilir.""",

    "Kullanıcı Kılavuzu: Parça Talebi Oluşturma": """# Kullanıcı Kılavuzu: Parça Talebi Oluşturma

Bu bölüm, bakım için gereken bir yedek parçanın talep edilmesini açıklar.

## 1. Envanter Sayfasını Açın

Sol menüden "Envanter" sekmesine gidin.

## 2. Parçayı Arayın

Parça kodunu veya adını arama kutusuna yazın.

## 3. Talep Oluşturun

Stok yetersizse "Tedarik Talebi Oluştur" butonuna tıklayıp miktarı girin.

## 4. Talep Durumunu Takip Edin

Talebin durumu "Taleplerim" sekmesinden izlenebilir.

> Not: Acil talepler, onaylayan yöneticiye anında bildirim olarak iletilir.""",

    # --- SSS (2. tur) ---
    "SSS — Randevu ve İptal": """# Sıkça Sorulan Sorular — Randevu ve İptal

**S: Randevumu nasıl iptal ederim?**

C: Onay e-postasındaki "İptal Et" bağlantısına tıklayarak veya "Randevularım"
sekmesinden iptal edebilirsiniz.

**S: İptal için son süre var mı?**

C: Randevudan en az 2 saat öncesine kadar ücretsiz iptal edebilirsiniz.

**S: Randevu saatimi değiştirebilir miyim?**

C: Evet, "Randevularım" sekmesinden "Yeniden Planla" seçeneğini
kullanabilirsiniz.

**S: Sağlayıcı randevuyu iptal ederse ne olur?**

C: Otomatik olarak bildirim alırsınız ve alternatif zaman dilimleri
önerilir.""",

    "SSS — Sadakat Puanları": """# Sıkça Sorulan Sorular — Sadakat Puanları

**S: Puanlarım ne zaman hesabıma yansır?**

C: Sipariş tamamlandıktan sonra genellikle 24 saat içinde yansır.

**S: Puanların son kullanma tarihi var mı?**

C: Evet, kazanıldıkları tarihten itibaren 12 ay geçerlidir.

**S: Puanları nasıl kullanırım?**

C: Ödeme adımında "Puan Kullan" seçeneğini işaretleyip kullanmak istediğiniz
miktarı girebilirsiniz.

**S: İade edilen bir siparişin puanı geri alınır mı?**

C: Evet, iade onaylandığında kazanılan puanlar bakiyenizden otomatik
düşülür.""",

    "SSS — Video Konferans": """# Sıkça Sorulan Sorular — Video Konferans

**S: Toplantıya katılmak için hesap açmam gerekiyor mu?**

C: Hayır, davet bağlantısına tıklayarak misafir olarak katılabilirsiniz.

**S: Toplantı kaç kişiye kadar destekliyor?**

C: Standart planda 25, kurumsal planda 250 katılımcıya kadar destekleniyor.

**S: Toplantıları kaydedebilir miyim?**

C: Evet, toplantı sahibi "Kaydı Başlat" seçeneğiyle kaydı başlatabilir;
kayıt bulut depolamada saklanır.

**S: Bağlantım sürekli kopuyor, ne yapmalıyım?**

C: Kablosuz yerine kablolu bağlantı deneyin; sorun devam ederse ekran
paylaşımını kapatarak bant genişliği tasarrufu sağlayabilirsiniz.""",

    "SSS — Destek Bileti Süreci": """# Sıkça Sorulan Sorular — Destek Bileti Süreci

**S: Bir bilete ortalama yanıt süresi nedir?**

C: Standart öncelikte 24 saat, acil öncelikte 2 saat içinde ilk yanıt
verilir.

**S: Kapatılmış bir bileti yeniden açabilir miyim?**

C: Evet, bilet kapatıldıktan sonraki 7 gün içinde yeni bir yorum eklerseniz
otomatik olarak yeniden açılır.

**S: Birden fazla bilet açarsam öncelik değişir mi?**

C: Hayır, her bilet bağımsız değerlendirilir; aynı konu için tekrar bilet
açmak yanıt süresini kısaltmaz.

**S: Bilet geçmişimi nereden görürüm?**

C: "Biletlerim" sekmesinde kapatılmış olanlar da dahil tüm geçmiş listelenir.""",

    "SSS — Vardiya Değişikliği": """# Sıkça Sorulan Sorular — Vardiya Değişikliği

**S: Vardiyamı başka bir ekip üyesiyle değiştirebilir miyim?**

C: Evet, "Değişim Talep Et" seçeneğiyle uygun bir meslektaşınıza talep
gönderebilirsiniz; yönetici onayı gerekir.

**S: Yayınlanmış bir vardiya planı değişebilir mi?**

C: Evet, ancak değişiklik yapıldığında etkilenen tüm üyelere bildirim
gönderilir.

**S: Müsaitlik durumumu nasıl güncellerim?**

C: Profilinizdeki "Müsaitlik" sekmesinden haftalık uygun olduğunuz saatleri
işaretleyebilirsiniz.

**S: Son dakika vardiya boşluklarını kim dolduruyor?**

C: Sistem, müsait ve uygun rolde olan üyelere otomatik bir doldurma çağrısı
gönderir.""",

    "SSS — Veri Gizliliği": """# Sıkça Sorulan Sorular — Veri Gizliliği

**S: Verilerim üçüncü taraflarla paylaşılıyor mu?**

C: Yalnızca hizmeti sağlamak için gerekli alt yükleniciler (örn. ödeme
sağlayıcısı) ile, ve yalnızca gerekli asgari veriyle paylaşılır.

**S: Verilerimin hangi ülkede saklandığını öğrenebilir miyim?**

C: Evet, Ayarlar → Gizlilik bölümünde veri merkezinizin bölgesi belirtilir.

**S: Reklam amaçlı takip yapılıyor mu?**

C: Hayır, ürün içi analitik yalnızca hizmet kalitesini iyileştirmek için
kullanılır, reklam amaçlı üçüncü taraf takip kodu bulunmaz.

**S: Bir veri ihlali durumunda bilgilendirilir miyim?**

C: Evet, yasal olarak gerekli olduğu her durumda etkilenen kullanıcılar
en kısa sürede e-posta ile bilgilendirilir.""",

    "SSS — Filo Bakımı": """# Sıkça Sorulan Sorular — Filo Bakımı

**S: Bir varlığın bakım geçmişini nereden görürüm?**

C: Envanter sayfasında ilgili varlığı seçip "Bakım Geçmişi" sekmesine
bakabilirsiniz.

**S: Bakım hatırlatmaları kime gönderiliyor?**

C: Varlığın atandığı bakım ekibine ve isteğe bağlı olarak saha
koordinatörüne gönderilir.

**S: Bakım eşiği nasıl belirleniyor?**

C: Varlık tipine göre kullanım süresi, kilometre veya çalışma saati
kombinasyonuyla otomatik hesaplanır; manuel olarak da özelleştirilebilir.

**S: Bir varlığı geçici olarak hizmet dışı bırakabilir miyim?**

C: Evet, "Hizmet Dışı Bırak" seçeneğiyle varlığı işaretleyebilirsiniz; bu
süre boyunca bakım hatırlatmaları duraklatılır.""",

    # --- Mimari Doküman (2. tur) ---
    "Mimari Doküman — Randevu ve Rezervasyon Sistemi": """# Mimari Doküman — Randevu ve Rezervasyon Sistemi

## Amaç

Bu doküman, çakışmasız randevu ayırma ve hatırlatma mimarisini açıklar.

## Bileşenler

- **Müsaitlik Servisi**: Sağlayıcı takvimlerini ve mevcut boşlukları hesaplar.
- **Rezervasyon Kilidi**: Aynı slotun eşzamanlı olarak iki kez ayrılmasını
  önleyen kısa ömürlü kilit mekanizması.
- **Hatırlatma Zamanlayıcısı**: Randevu öncesi bildirimleri tetikler.

## Veri Akışı

1. İstemci, Müsaitlik Servisi'nden uygun slotları ister.
2. Bir slot seçildiğinde Rezervasyon Kilidi kısa süreliğine devreye girer.
3. Randevu onaylanırsa kilit kalıcı bir kayda dönüşür; onaylanmazsa süre
   sonunda otomatik serbest kalır.
4. Hatırlatma Zamanlayıcısı, randevu saatine göre bildirim zamanını planlar.

## Ölçeklenebilirlik Notları

Kilit mekanizması Redis üzerinde kısa TTL ile çalışır; bu sayede yüksek
eşzamanlı taleplerde bile çift rezervasyon oluşmaz.""",

    "Mimari Doküman — Anket ve Form Motoru": """# Mimari Doküman — Anket ve Form Motoru

## Amaç

Bu doküman, koşullu soru mantığı destekleyen form motorunun mimarisini
açıklar.

## Bileşenler

- **Form Tanım Deposu**: Soru, seçenek ve koşul kurallarının şema olarak
  saklandığı katman.
- **Sunum Motoru**: İstemci tarafında, kullanıcının önceki yanıtlarına göre
  bir sonraki soruyu belirleyen mantık.
- **Yanıt Toplayıcı**: Gelen yanıtları doğrulayıp kalıcı depoya yazan servis.

## Veri Akışı

1. İstemci form tanımını Form Tanım Deposu'ndan çeker.
2. Sunum Motoru, kullanıcı ilerledikçe koşul kurallarını değerlendirip
   sıradaki soruyu belirler.
3. Form tamamlandığında yanıtlar Yanıt Toplayıcı'ya gönderilir ve doğrulanır.

## Ölçeklenebilirlik Notları

Koşul değerlendirmesi tamamen istemci tarafında yapıldığı için sunucu
yükü, yalnızca nihai gönderim anında oluşur.""",

    "Mimari Doküman — Sadakat ve Puan Sistemi": """# Mimari Doküman — Sadakat ve Puan Sistemi

## Amaç

Bu doküman, puan kazanma/harcama işlemlerinin tutarlılığının nasıl
sağlandığını açıklar.

## Bileşenler

- **İşlem Defteri (Ledger)**: Her puan hareketinin değiştirilemez bir kayıt
  olarak tutulduğu tablo.
- **Bakiye Önbelleği**: Sık okunan güncel bakiyeyi hızlı sunmak için Redis
  önbelleği.
- **Kural Motoru**: Kaynak bazlı puan hesaplama kurallarını uygular.

## Veri Akışı

1. Bir olay (sipariş, yorum vb.) gerçekleştiğinde Kural Motoru kazanılacak
   puanı hesaplar.
2. Sonuç, İşlem Defteri'ne yeni bir satır olarak eklenir (asla güncellenmez,
   yalnızca eklenir).
3. Bakiye Önbelleği, yeni satırı okuyup güncel toplamı yeniden hesaplar.

## Ölçeklenebilirlik Notları

Defterin yalnızca-ekleme (append-only) olması, eşzamanlı yazımlarda
tutarsızlık riskini ortadan kaldırır; bakiye her zaman defterden yeniden
türetilebilir.""",

    "Mimari Doküman — Destek Bileti Yönlendirme": """# Mimari Doküman — Destek Bileti Yönlendirme

## Amaç

Bu doküman, gelen destek taleplerinin doğru ekibe otomatik yönlendirilme
mimarisini açıklar.

## Bileşenler

- **Alım Katmanı**: Web formu, e-posta ve sohbetten gelen talepleri tek
  formata dönüştürür.
- **Sınıflandırma Servisi**: İçeriğe göre kategori ve öncelik önerir.
- **Yönlendirme Motoru**: Kategori, ekip müsaitliği ve iş yüküne göre
  bileti bir temsilciye atar.

## Veri Akışı

1. Talep, hangi kanaldan gelirse gelsin Alım Katmanı'nda standart bilet
   formatına dönüştürülür.
2. Sınıflandırma Servisi kategori/öncelik önerir.
3. Yönlendirme Motoru, uygun ekibin en az yüklü temsilcisine bileti atar.

## Ölçeklenebilirlik Notları

Sınıflandırma ve yönlendirme birbirinden bağımsız servislerdir; sınıflandırma
modeli güncellenirken yönlendirme mantığı etkilenmez.""",

    "Mimari Doküman — Filo Bakım Takibi": """# Mimari Doküman — Filo Bakım Takibi

## Amaç

Bu doküman, araç/ekipman filosunun bakım takviminin nasıl yönetildiğini
açıklar.

## Bileşenler

- **Envanter Kaydı**: Her varlığın (araç, ekipman) temel bilgilerini tutar.
- **Bakım Kural Motoru**: Kullanım süresi veya çalışma saatine göre bir
  sonraki bakım tarihini hesaplar.
- **Uyarı Servisi**: Yaklaşan/gecikmiş bakımlar için bildirim üretir.

## Veri Akışı

1. Bir varlığın kullanım verisi (saat, kilometre vb.) periyodik olarak
   güncellenir.
2. Bakım Kural Motoru bu veriyle bir sonraki bakım eşiğini yeniden hesaplar.
3. Eşiğe yaklaşıldığında Uyarı Servisi ilgili ekibe bildirim gönderir.

## Ölçeklenebilirlik Notları

Bakım kuralları varlık tipine göre eklenebilir (araç, jeneratör, sensör vb.);
yeni bir varlık tipi eklemek mevcut kuralları etkilemez.""",

    "Mimari Doküman — Görev Planlama ve Onay Akışı": """# Mimari Doküman — Görev Planlama ve Onay Akışı

## Amaç

Bu doküman, saha görevi planlarının taslak-onay-yayın aşamalarından nasıl
geçtiğini açıklar.

## Bileşenler

- **Planlama Arayüzü**: Rota ve kaynak seçiminin yapıldığı istemci.
- **Doğrulama Servisi**: Çakışma, kaynak müsaitliği ve kural ihlali
  kontrolü yapar.
- **Onay İş Akışı**: Planın kimden geçeceğini ve karar geçmişini yönetir.

## Veri Akışı

1. Planlama Arayüzü'nde oluşturulan taslak, Doğrulama Servisi'ne gönderilir.
2. Doğrulama başarılıysa plan Onay İş Akışı'na girer.
3. Yetkili onaylayınca plan "aktif" durumuna geçer ve ilgili ekiplere
   bildirilir; reddedilirse gerekçesiyle taslağa geri döner.

## Ölçeklenebilirlik Notları

Onay iş akışı yapılandırılabilir basamaklardan oluşur; kurum büyüdükçe
ek onay basamağı eklemek mevcut planları etkilemez.""",

    "Mimari Doküman — İçerik Moderasyon Hattı": """# Mimari Doküman — İçerik Moderasyon Hattı

## Amaç

Bu doküman, kullanıcı tarafından oluşturulan içeriğin yayınlanmadan önce
nasıl denetlendiğini açıklar.

## Bileşenler

- **Ön Filtre**: Otomatik kural/kelime listesi tabanlı hızlı kontrol.
- **İnceleme Kuyruğu**: Ön filtreden şüpheli işaretlenen içeriklerin insan
  incelemesine düştüğü kuyruk.
- **Karar Kaydı**: Onay/red kararlarının ve gerekçelerinin saklandığı kayıt.

## Veri Akışı

1. Yeni içerik önce Ön Filtre'den geçer.
2. Temiz görünen içerik doğrudan yayınlanır; şüpheli olan İnceleme
   Kuyruğu'na düşer.
3. İnceleyen karar verdiğinde sonuç Karar Kaydı'na yazılır ve içerik
   yayınlanır ya da kaldırılır.

## Ölçeklenebilirlik Notları

Ön Filtre kuralları sık güncellenebilir olacak şekilde ayrı bir
yapılandırma dosyasında tutulur; kod değişikliği gerektirmez.""",

    # --- Sürüm Notları (2. tur) ---
    "Sürüm Notları — Randevu Sistemi v1.3.0": """# Sürüm Notları — Randevu Sistemi

## v1.3.0 — 2026-02-25

### Eklenenler
- Randevu iptal nedeninin isteğe bağlı olarak kaydedilmesi eklendi.
- Sağlayıcılar için haftalık tekrarlayan müsaitlik şablonu eklendi.

### Değişenler
- Hatırlatma bildirimleri artık randevudan 24 saat yerine hem 24 saat hem
  1 saat önce gönderiliyor.

### Düzeltilenler
- Farklı saat dilimlerindeki kullanıcılarda yanlış müsaitlik gösterme
  sorunu giderildi.

## v1.2.5 — 2026-01-14

### Düzeltilenler
- Aynı anda gelen iki rezervasyon talebinde nadir çift ayırma (double
  booking) sorunu giderildi.""",

    "Sürüm Notları — Form Motoru v2.1.0": """# Sürüm Notları — Form Motoru

## v2.1.0 — 2026-02-08

### Eklenenler
- Koşullu soru mantığına "birden fazla koşul" (VE/VEYA) desteği eklendi.
- Yanıtları doğrudan e-tabloya senkronize etme özelliği eklendi.

### Değişenler
- Form yükleme süresi, gereksiz alan render'ları kaldırılarak iyileştirildi.

### Düzeltilenler
- Mobilde çoktan seçmeli sorularda seçimlerin bazen kaybolması sorunu
  giderildi.

## v2.0.3 — 2026-01-02

### Düzeltilenler
- Türkçe karakterli soru başlıklarının dışa aktarımda bozulması sorunu
  giderildi.""",

    "Sürüm Notları — Video Konferans v1.0.0": """# Sürüm Notları — Video Konferans

## v1.0.0 — 2026-01-20

### Eklenenler
- İlk kararlı sürüm: toplantı oluşturma, ekran paylaşımı, kayıt.

### Bilinen Sınırlamalar
- Aynı anda en fazla 250 katılımcı desteklenmektedir.
- Kayıtlar şu an yalnızca MP4 formatında dışa aktarılabiliyor.

## v0.9.0-beta — 2025-12-15

### Eklenenler
- Beta test kullanıcıları için toplantı odası ve temel sohbet özelliği
  eklendi.""",

    "Sürüm Notları — Sadakat Servisi v1.5.0": """# Sürüm Notları — Sadakat Servisi

## v1.5.0 — 2026-03-02

### Eklenenler
- Puanların son kullanma tarihinden 30 gün önce hatırlatma bildirimi
  eklendi.
- Seviye yükseltme olayları için webhook desteği eklendi.

### Değişenler
- Bakiye hesaplama, defterden (ledger) her seferinde yeniden türetilecek
  şekilde değiştirildi (tutarlılık için).

### Düzeltilenler
- İade işlemlerinde puan düşürmenin bazen gecikmesi sorunu giderildi.

## v1.4.2 — 2026-01-28

### Düzeltilenler
- Kampanya bazlı çarpanların (2x puan günleri) yanlış hesaplanması sorunu
  giderildi.""",

    "Sürüm Notları — Destek Bileti Sistemi v3.0.0": """# Sürüm Notları — Destek Bileti Sistemi

## v3.0.0 — 2026-02-17

### Eklenenler
- SLA süresi aşımında otomatik yükseltme (escalation) eklendi.
- E-posta üzerinden bilete yanıt verme desteği eklendi.

### Değişenler
- **Kırıcı değişiklik:** öncelik seviyeleri ("düşük/orta/yüksek") yerine
  sayısal skor (1-5) kullanılıyor; entegrasyonların güncellenmesi gerekir.

### Düzeltilenler
- Kapatılan biletlerin arama sonuçlarında görünmeye devam etmesi sorunu
  giderildi.

## v2.4.1 — 2026-01-10

### Düzeltilenler
- Dosya eki 10 MB'ı aştığında sessizce başarısız olma sorunu giderildi.""",

    "Sürüm Notları — Filo Bakım Takip v1.2.0": """# Sürüm Notları — Filo Bakım Takip

## v1.2.0 — 2026-02-22

### Eklenenler
- Yedek parça stok seviyesi düşükken bakım planlamasını uyaran entegrasyon
  eklendi.
- Varlık başına bakım geçmişi PDF olarak dışa aktarılabiliyor.

### Değişenler
- Bakım eşiği hesaplaması artık kilometre/saat yanında kullanım yoğunluğunu
  da dikkate alıyor.

### Düzeltilenler
- Bazı varlıklarda bakım tarihinin geçmişte gösterilmesi sorunu giderildi.

## v1.1.4 — 2026-01-06

### Düzeltilenler
- CSV içe aktarımında ondalık ayracı (nokta/virgül) karışıklığı sorunu
  giderildi.""",

    "Sürüm Notları — Görev Planlama v2.4.0": """# Sürüm Notları — Görev Planlama

## v2.4.0 — 2026-03-08

### Eklenenler
- Çok basamaklı onay iş akışı desteği eklendi (birden fazla yetkili
  sırayla onaylayabiliyor).
- Plan taslaklarının otomatik kaydedilmesi eklendi.

### Değişenler
- Rota çizim aracının performansı, büyük nokta sayılarında iyileştirildi.

### Düzeltilenler
- Reddedilen bir planın gerekçesinin bazen boş görünmesi sorunu giderildi.

## v2.3.2 — 2026-02-01

### Düzeltilenler
- PostGIS sürüm uyumsuzluğunda rota hesaplamasının sessizce başarısız
  olması sorunu giderildi.""",

    # --- Diğer (2. tur) ---
    "Değişiklik Yönetimi (Change Management) Politikası Şablonu": """# Değişiklik Yönetimi Politikası

## Kapsam

Bu politika, üretim ortamına yapılan değişikliklerin nasıl planlanıp
onaylanacağını tanımlar.

## Değişiklik Kategorileri

| Kategori | Örnek | Onay Gereksinimi |
|---|---|---|
| Standart | Bağımlılık güncellemesi | Otomatik (CI geçerse) |
| Normal | Yeni özellik dağıtımı | Bir onay |
| Acil | Kritik güvenlik yaması | Sonradan gözden geçirme |

## Değişiklik Öncesi Kontrol Listesi

- [ ] Geri alma (rollback) planı tanımlı mı?
- [ ] Değişiklik düşük trafikli bir saatte mi planlandı?
- [ ] İlgili ekipler bilgilendirildi mi?

## Acil Değişiklikler

Acil değişiklikler önce uygulanır, en geç 24 saat içinde geriye dönük
onay ve dokümantasyon tamamlanır.""",

    "Erişilebilirlik (Accessibility) Kontrol Listesi Şablonu": """# Erişilebilirlik Kontrol Listesi

## Klavye Erişimi

- [ ] Tüm etkileşimli öğeler yalnızca klavye ile ulaşılabilir mi?
- [ ] Odak (focus) sırası mantıksal mı?
- [ ] Odaklanılan öğe görsel olarak belirgin mi?

## Ekran Okuyucu Uyumluluğu

- [ ] Görsellerde anlamlı `alt` metni var mı?
- [ ] Form alanlarının etiketleri (label) doğru ilişkilendirilmiş mi?
- [ ] Dinamik içerik değişiklikleri (örn. hata mesajı) duyuruluyor mu?

## Görsel Tasarım

- [ ] Metin/arka plan kontrastı WCAG AA eşiğini karşılıyor mu?
- [ ] Bilgi yalnızca renkle değil, ek bir işaretle de veriliyor mu?

## Test

Her önemli sürümde en az bir ekran okuyucu (NVDA/VoiceOver) ile manuel
test yapılmalıdır.""",

    "Performans Test Planı Şablonu": """# Performans Test Planı

## Amaç

Bu plan, sistemin beklenen yük altında kabul edilebilir gecikme ile
çalıştığını doğrulamayı hedefler.

## Test Senaryoları

| Senaryo | Eşzamanlı Kullanıcı | Hedef p95 Gecikme |
|---|---|---|
| Normal yük | 500 | 300 ms |
| Kampanya yoğunluğu | 5000 | 800 ms |
| Ani sıçrama (spike) | 0 → 3000 (30 sn içinde) | Hata oranı < %1 |

## Test Ortamı

Testler, üretimle aynı donanım profiline sahip ayrı bir ortamda
çalıştırılır; üretim verisiyle değil, sentetik veriyle beslenir.

## Kabul Kriterleri

Belirlenen eşzamanlı kullanıcı sayısında hata oranı %1'i, p95 gecikme
hedef değeri aşmamalıdır. Aşan senaryo, üretime çıkışı engeller.""",

    "Üçüncü Taraf Bağımlılık Politikası Şablonu": """# Üçüncü Taraf Bağımlılık Politikası

## Amaç

Bu politika, dış kütüphane/servis eklerken izlenecek değerlendirme
sürecini tanımlar.

## Yeni Bağımlılık Eklemeden Önce

- [ ] Aktif olarak bakımı yapılıyor mu (son 12 ayda commit var mı)?
- [ ] Lisansı projeyle uyumlu mu (bkz. onaylı lisans listesi)?
- [ ] Bilinen kritik güvenlik açığı var mı?
- [ ] Aynı işlevi zaten mevcut bir bağımlılık karşılıyor mu?

## Sürüm Sabitleme

Üretim bağımlılıkları kesin sürümle sabitlenir; büyük sürüm güncellemeleri
ayrı bir PR'da, değişiklik notları incelenerek yapılır.

## Periyodik Denetim

Tüm bağımlılıklar üç ayda bir otomatik güvenlik taramasından (örn.
`pip-audit`) geçirilir.""",

    "Veri Sınıflandırma Politikası Şablonu": """# Veri Sınıflandırma Politikası

## Amaç

Bu politika, sistemde işlenen verilerin hassasiyet düzeyine göre nasıl
sınıflandırılacağını ve korunacağını tanımlar.

## Sınıflandırma Seviyeleri

| Seviye | Örnek | Gereken Koruma |
|---|---|---|
| Genel | Ürün kataloğu | Ek koruma gerekmez |
| Dahili | İç raporlar | Yalnızca kimlik doğrulamalı erişim |
| Gizli | Kullanıcı kişisel verisi | Şifreleme + erişim günlüğü |
| Kritik | Ödeme bilgisi, kimlik belgesi | Şifreleme + sınırlı erişim + denetim |

## Etiketleme

Yeni bir veri alanı eklenirken şemaya bir sınıflandırma etiketi
eklenmelidir; bu etiket erişim kontrolü kurallarını otomatik belirler.

## İhlal Durumunda

Kritik seviyeli bir veriye yetkisiz erişim şüphesi, güvenlik ekibine
derhal bildirilmelidir.""",

    "API Sürümleme Politikası Şablonu": """# API Sürümleme Politikası

## Amaç

Bu politika, API'de yapılan değişikliklerin istemcileri nasıl etkileyeceğini
ve ne zaman yeni bir sürüm açılacağını tanımlar.

## Kırıcı Olmayan Değişiklikler (sürüm artışı gerekmez)

- Yeni, isteğe bağlı alan eklemek
- Yeni bir uç nokta eklemek
- Hata mesajı metnini iyileştirmek

## Kırıcı Değişiklikler (yeni büyük sürüm gerekir)

- Bir alanı kaldırmak veya yeniden adlandırmak
- Bir alanın tipini değiştirmek
- Varsayılan davranışı değiştirmek

## Sürüm Yaşam Döngüsü

Yeni bir büyük sürüm yayınlandığında önceki sürüm en az 6 ay boyunca
paralel desteklenir; kapatılmadan 90 gün önce tüm aktif istemcilere
e-posta ile bildirim yapılır.""",

    "Saha Operasyonu Güvenlik Talimatı Şablonu": """# Saha Operasyonu Güvenlik Talimatı

## Amaç

Bu doküman, saha ekiplerinin ekipman kurulum/bakım operasyonlarında
uyması gereken temel güvenlik kurallarını özetler.

## Operasyon Öncesi

- [ ] Hava durumu ve saha koşulları operasyona uygun mu?
- [ ] Gerekli kişisel koruyucu ekipman (KKE) tam mı?
- [ ] İletişim kanalı (telsiz/telefon) test edildi mi?

## Operasyon Sırasında

- Ekip, saha sorumlusuyla düzenli aralıklarla durum bildirimi yapar.
- Beklenmeyen bir tehlike durumunda operasyon derhal durdurulur.

## Operasyon Sonrası

- [ ] Kullanılan ekipman kontrol edilip envantere geri kaydedildi mi?
- [ ] Varsa aksaklıklar rapor edildi mi?

## Acil Durum

Bir kaza/yaralanma durumunda önce saha sorumlusuna, ardından acil
durum hattına bildirim yapılır.""",

    "Görev Sonrası Değerlendirme (After-Action Report) Şablonu": """# Görev Sonrası Değerlendirme Raporu

## Görev Özeti

**Görev:** Rutin saha denetimi · **Tarih:** 2026-02-19 · **Süre:** 3 saat 40 dakika

## Planlanan vs. Gerçekleşen

| Metrik | Planlanan | Gerçekleşen |
|---|---|---|
| Kontrol noktası sayısı | 12 | 12 |
| Tahmini süre | 3 saat | 3 saat 40 dakika |
| Ekip sayısı | 2 | 2 |

## İyi Giden Noktalar

- Tüm kontrol noktaları planlanan sırayla tamamlandı.
- Ekipler arası iletişim kesintisiz sürdü.

## İyileştirme Alanları

- 4. kontrol noktasında erişim gecikmesi yaşandı (kapı erişim izni
  eksikliği); süre aşımının ana nedeni bu oldu.

## Aksiyon Maddeleri

- [ ] Erişim izinlerinin operasyon öncesi kontrol listesine eklenmesi
  (Sahip: Saha Koordinasyon, 1 hafta)
- [ ] Süre tahminlerinin geçmiş veriye göre güncellenmesi (Sahip: Planlama, 2 hafta)""",

    # --- README ---
    "README — Bildirim Servisi": """# Bildirim Servisi

E-posta, SMS ve push bildirimlerini tek bir arayüz altında gönderen, sağlayıcı
bağımsız bir bildirim servisi.

## Özellikler

- Sağlayıcı soyutlaması: e-posta için SMTP/SES, SMS için birden fazla operatör arasında geçiş
- Şablon motoru ile çok dilli bildirik içerikleri
- Başarısız gönderimler için otomatik yeniden deneme kuyruğu

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from bildirim import gonder

gonder(kanal="eposta", alici="kullanici@ornek.com", sablon="hosgeldin")
```

## Katkı Sağlama

Yeni bir sağlayıcı eklemeden önce `Saglayici` arayüzünü inceleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Dosya Depolama İstemcisi": """# Dosya Depolama İstemcisi

Nesne depolama servisleri (S3 uyumlu) için tek tip bir Python istemcisi.

## Özellikler

- Bölümlü (multipart) büyük dosya yükleme
- İmzalı, süreli erişim bağlantıları üretme
- Yerel diskte otomatik önbellekleme

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from depolama import Istemci

istemci = Istemci(kova="dokumanlar")
istemci.yukle("rapor.pdf", yol="2026/rapor.pdf")
```

## Katkı Sağlama

Yeni bir depolama sağlayıcısı eklerken mevcut entegrasyon testlerini çalıştırın.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Görev Zamanlayıcı": """# Görev Zamanlayıcı

Belirli aralıklarla ya da tek seferlik olarak arka plan işlerini çalıştıran hafif
bir zamanlayıcı kütüphanesi.

## Özellikler

- Cron ifadesiyle veya sabit aralıkla zamanlama
- Çakışan çalıştırmaları otomatik atlama
- Başarısız işler için uyarı kancası (webhook)

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from zamanlayici import gorev_ekle

@gorev_ekle(cron="0 3 * * *")
def gece_temizligi():
    ...
```

## Katkı Sağlama

Yeni tetikleyici tipleri `tetikleyiciler/` klasörüne eklenmelidir.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Ödeme Ağ Geçidi Entegrasyonu": """# Ödeme Ağ Geçidi Entegrasyonu

Birden fazla ödeme sağlayıcısını (kart, cüzdan, banka transferi) tek bir arayüz
arkasında birleştiren entegrasyon katmanı.

## Özellikler

- Sağlayıcı bazında otomatik yeniden deneme ve devreye alma (failover)
- 3D Secure akışı desteği
- İşlem başına idempotency anahtarı zorunluluğu

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from odeme import tahsil_et

sonuc = tahsil_et(tutar=149.90, para_birimi="TRY", kart_token="tok_abc123")
```

## Katkı Sağlama

Yeni sağlayıcı eklerken sandbox kimlik bilgilerini `.env.example`'a ekleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Arama Motoru İstemcisi": """# Arama Motoru İstemcisi

Elasticsearch/OpenSearch üzerinde çalışan uygulamalar için tip güvenli bir sorgu
oluşturucu ve istemci kütüphanesi.

## Özellikler

- Zincirlenebilir (chainable) sorgu oluşturucu
- Facet ve öneri (autocomplete) sorguları için hazır yardımcılar
- İndeks şeması sürümleme desteği

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from arama import Sorgu

sonuclar = Sorgu(indeks="urunler").metin("kablosuz kulaklık").calistir()
```

## Katkı Sağlama

Yeni bir sorgu tipi eklerken ilgili birim testini `tests/queries/` altına ekleyin.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Telemetri Veri Toplama Servisi": """# Telemetri Veri Toplama Servisi

Uzak cihazlardan (sensör, araç, saha ekipmanı) gelen telemetri akışını toplayıp
zaman serisi veritabanına yazan servis.

## Özellikler

- MQTT ve HTTP üzerinden veri alımı
- Örnekleme hızını cihaz bağlantı kalitesine göre otomatik ayarlama
- Bozuk/aykırı ölçümleri işaretleyen basit anomali filtresi

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from telemetri import Toplayici

toplayici = Toplayici(kaynak="mqtt://filo-agi:1883")
toplayici.baslat()
```

## Katkı Sağlama

Yeni bir veri kaynağı eklerken `kaynaklar/` altındaki arayüzü uygulayın.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    "README — Çok Dilli İçerik Yönetimi Kütüphanesi": """# Çok Dilli İçerik Yönetimi Kütüphanesi

Uygulama arayüzü metinlerini birden fazla dilde yönetmek için basit bir
i18n/çeviri kütüphanesi.

## Özellikler

- Eksik çeviriler için derleme zamanında uyarı
- Çoğul (plural) ve cinsiyet formlarını destekleyen mesaj biçimi
- Çeviri dosyalarını JSON veya YAML olarak dışa/içe aktarma

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```python
from i18n import t

mesaj = t("hosgeldin_mesaji", dil="tr", ad="Ayşe")
```

## Katkı Sağlama

Yeni bir dil eklerken `ceviriler/<dil_kodu>.json` dosyasını oluşturup PR açın.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.""",

    # --- API Dokümantasyonu ---
    "API Referansı: siparis_olustur": """# API Referansı: siparis_olustur

## Açıklama

Sepetteki ürünlerden yeni bir sipariş oluşturur ve ödeme sürecini başlatır.

## İmza

```
siparis_olustur(sepet_id: str, teslimat_adres_id: str, odeme_yontemi: str) -> dict
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| sepet_id | str | Evet | Doldurulmuş sepetin kimliği |
| teslimat_adres_id | str | Evet | Kayıtlı adres kimliği |
| odeme_yontemi | str | Evet | "kart", "havale" veya "kapida_odeme" |

## Dönüş Değeri

`{"siparis_id": str, "durum": str, "toplam_tutar": float}` şeklinde bir sözlük.

## Hatalar

- `EmptyCartError`: sepet boşsa fırlatılır.
- `InvalidAddressError`: adres kimliği bulunamazsa fırlatılır.

## Örnek

```python
siparis = siparis_olustur(sepet_id="sp_1", teslimat_adres_id="adr_9", odeme_yontemi="kart")
```""",

    "API Referansı: dosya_yukle": """# API Referansı: dosya_yukle

## Açıklama

Verilen dosyayı depolama servisine yükler ve erişim bağlantısını döndürür.

## İmza

```
dosya_yukle(dosya: bytes, dosya_adi: str, herkese_acik: bool = False) -> str
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| dosya | bytes | Evet | Ham dosya içeriği |
| dosya_adi | str | Evet | Depoda saklanacak isim |
| herkese_acik | bool | Hayır | True ise bağlantı süresiz, aksi halde 1 saat geçerli |

## Dönüş Değeri

Dosyaya erişim için imzalı URL (str).

## Hatalar

- `FileTooLargeError`: dosya 100 MB'ı aşarsa fırlatılır.
- `UnsupportedFormatError`: uzantı izin verilen listede değilse fırlatılır.

## Örnek

```python
url = dosya_yukle(dosya=icerik, dosya_adi="rapor.pdf")
```""",

    "API Referansı: bildirim_gonder": """# API Referansı: bildirim_gonder

## Açıklama

Belirtilen kullanıcıya seçilen kanaldan bildirim gönderir.

## İmza

```
bildirim_gonder(kullanici_id: str, kanal: str, sablon: str, degiskenler: dict) -> bool
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| kullanici_id | str | Evet | Bildirimi alacak kullanıcı |
| kanal | str | Evet | "eposta", "sms" veya "push" |
| sablon | str | Evet | Önceden tanımlı şablon adı |
| degiskenler | dict | Hayır | Şablon içinde doldurulacak değerler |

## Dönüş Değeri

Gönderim kuyruğa başarıyla eklendiyse `True`.

## Hatalar

- `UserOptedOutError`: kullanıcı bu kanalı kapatmışsa fırlatılır.
- `TemplateNotFoundError`: şablon adı tanımlı değilse fırlatılır.

## Örnek

```python
bildirim_gonder(kullanici_id="u_42", kanal="push", sablon="siparis_kargoda", degiskenler={"kargo_no": "TR123"})
```""",

    "API Referansı: konum_guncelle": """# API Referansı: konum_guncelle

## Açıklama

Bir cihazın veya aracın son bilinen konumunu günceller ve geçmiş konum kaydı oluşturur.

## İmza

```
konum_guncelle(cihaz_id: str, enlem: float, boylam: float, zaman_damgasi: int) -> None
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| cihaz_id | str | Evet | Konum bildiren cihazın kimliği |
| enlem | float | Evet | -90 ile 90 arası |
| boylam | float | Evet | -180 ile 180 arası |
| zaman_damgasi | int | Evet | Unix epoch (saniye) |

## Dönüş Değeri

Yok; başarılı çağrı HTTP 204 ile sonuçlanır.

## Hatalar

- `InvalidCoordinateError`: enlem/boylam geçerli aralığın dışındaysa fırlatılır.
- `StaleTimestampError`: zaman damgası sunucu saatinden 5 dakikadan fazla geride ise fırlatılır.

## Örnek

```python
konum_guncelle(cihaz_id="arac_17", enlem=39.92, boylam=32.85, zaman_damgasi=1755500000)
```""",

    "API Referansı: rapor_olustur": """# API Referansı: rapor_olustur

## Açıklama

Belirtilen tarih aralığı ve filtrelerle asenkron bir rapor oluşturma işi başlatır.

## İmza

```
rapor_olustur(baslangic: str, bitis: str, format: str = "pdf", filtreler: dict | None = None) -> str
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| baslangic | str | Evet | ISO 8601 tarih |
| bitis | str | Evet | ISO 8601 tarih |
| format | str | Hayır | "pdf", "csv" veya "xlsx" |
| filtreler | dict | Hayır | Ek filtre anahtar-değerleri |

## Dönüş Değeri

Rapor işinin durumunu sorgulamak için kullanılacak iş kimliği (str).

## Hatalar

- `InvalidDateRangeError`: bitiş tarihi başlangıçtan önceyse fırlatılır.
- `RangeTooLargeError`: aralık 1 yıldan uzunsa fırlatılır.

## Örnek

```python
is_id = rapor_olustur(baslangic="2026-01-01", bitis="2026-03-31", format="xlsx")
```""",

    "API Referansı: oturum_dogrula": """# API Referansı: oturum_dogrula

## Açıklama

Gelen bir istekteki oturum belirtecinin (token) geçerliliğini kontrol eder ve
kullanıcı bilgisini döndürür.

## İmza

```
oturum_dogrula(token: str) -> dict
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| token | str | Evet | İstemcinin gönderdiği JWT erişim belirteci |

## Dönüş Değeri

`{"kullanici_id": str, "roller": list[str], "gecerlilik_bitis": int}` şeklinde bir sözlük.

## Hatalar

- `TokenExpiredError`: belirtecin süresi dolmuşsa fırlatılır.
- `TokenRevokedError`: belirteç manuel olarak iptal edilmişse fırlatılır.

## Örnek

```python
bilgi = oturum_dogrula(token=istek.headers["Authorization"])
```""",

    "API Referansı: sensor_verisi_al": """# API Referansı: sensor_verisi_al

## Açıklama

Belirtilen sensörün, verilen zaman aralığındaki ölçüm geçmişini sayfalanmış olarak döndürür.

## İmza

```
sensor_verisi_al(sensor_id: str, baslangic: int, bitis: int, sayfa_boyutu: int = 500) -> list[dict]
```

## Parametreler

| Parametre | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| sensor_id | str | Evet | Sensörün benzersiz kimliği |
| baslangic | int | Evet | Unix epoch (saniye) |
| bitis | int | Evet | Unix epoch (saniye) |
| sayfa_boyutu | int | Hayır | Bir çağrıda dönecek en fazla kayıt sayısı |

## Dönüş Değeri

Her biri `{"zaman": int, "deger": float, "birim": str}` içeren bir liste.

## Hatalar

- `SensorNotFoundError`: sensor_id kayıtlı değilse fırlatılır.
- `RangeTooLargeError`: aralık 30 günden uzunsa fırlatılır.

## Örnek

```python
olcumler = sensor_verisi_al(sensor_id="sck_04", baslangic=1755000000, bitis=1755100000)
```""",

    # --- Kurulum Kılavuzu ---
    "Kurulum Kılavuzu — Bildirim Servisi": """# Kurulum Kılavuzu — Bildirim Servisi

## Ön Koşullar

- Python 3.10 veya üzeri
- Çalışan bir Redis örneği (kuyruk için)
- E-posta sağlayıcısı için API anahtarı

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd bildirim-servisi
```

## Adım 2: Ortam Değişkenlerini Ayarlayın

```bash
cp .env.example .env
# .env içine SMTP_API_KEY değerini girin
```

## Adım 3: Bağımlılıkları Kurun ve Servisi Başlatın

```bash
pip install -r requirements.txt
python -m bildirim.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl -X POST localhost:8000/saglik
```

`{"durum": "ok"}` dönüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Redis'e bağlanılamıyorsa `REDIS_URL` değişkeninin doğru olduğundan emin olun.""",

    "Kurulum Kılavuzu — Arama Motoru Kümesi": """# Kurulum Kılavuzu — Arama Motoru Kümesi

## Ön Koşullar

- Docker ve Docker Compose
- En az 8 GB RAM (küme 3 düğümlü çalışır)

## Adım 1: Küme Yapılandırmasını İndirin

```bash
git clone https://ornek-depo-adresi
cd arama-kumesi
```

## Adım 2: Düğümleri Başlatın

```bash
docker compose up -d
```

## Adım 3: İndeksi Oluşturun

```bash
python -m arama.indeksle --kaynak veri/urunler.jsonl
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl localhost:9200/_cluster/health
```

`"status":"green"` görüyorsanız küme sağlıklıdır.

## Sık Karşılaşılan Sorunlar

Küme "yellow" durumda kalıyorsa replika sayısını düğüm sayısına göre azaltın.""",

    "Kurulum Kılavuzu — Mesajlaşma Sunucusu": """# Kurulum Kılavuzu — Mesajlaşma Sunucusu

## Ön Koşullar

- Python 3.10 veya üzeri
- PostgreSQL 14+
- WebSocket destekleyen bir ters vekil (nginx vb.)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd mesajlasma-sunucusu
```

## Adım 2: Veritabanını Hazırlayın

```bash
createdb mesajlasma
python -m mesajlasma.migrate
```

## Adım 3: Sunucuyu Başlatın

```bash
pip install -r requirements.txt
python -m mesajlasma.sunucu --port 8080
```

## Adım 4: Kurulumu Doğrulayın

Bir WebSocket istemcisiyle `ws://localhost:8080/baglan` adresine bağlanıp "ping"
gönderin; "pong" cevabı gelmelidir.

## Sık Karşılaşılan Sorunlar

Bağlantı hemen kapanıyorsa ters vekilinizin WebSocket yükseltmesini (upgrade
header) ilettiğinden emin olun.""",

    "Kurulum Kılavuzu — Görüntü İşleme Servisi": """# Kurulum Kılavuzu — Görüntü İşleme Servisi

## Ön Koşullar

- Python 3.10 veya üzeri
- CUDA destekli GPU (opsiyonel, yoksa CPU modunda çalışır)
- FFmpeg kurulu olmalı

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd goruntu-servisi
```

## Adım 2: Bağımlılıkları Kurun

```bash
pip install -r requirements.txt
```

## Adım 3: Model Ağırlıklarını İndirin

```bash
python -m goruntu.model_indir --hedef modeller/
```

## Adım 4: Kurulumu Doğrulayın

```bash
python -m goruntu.test_calistir ornek.jpg
```

Çıktı görselinde tespit kutuları görünüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

`CUDA out of memory` hatası alıyorsanız `--batch-size` değerini düşürün.""",

    "Kurulum Kılavuzu — Veri Yedekleme Aracı": """# Kurulum Kılavuzu — Veri Yedekleme Aracı

## Ön Koşullar

- Python 3.10 veya üzeri
- Yedeklerin yazılacağı bir nesne depolama kovası

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd yedekleme-araci
```

## Adım 2: Yapılandırma Dosyasını Oluşturun

```bash
cp config.example.yaml config.yaml
# config.yaml içine kova adını ve erişim anahtarlarını girin
```

## Adım 3: İlk Yedeği Alın

```bash
pip install -r requirements.txt
python -m yedekle.calistir --tam
```

## Adım 4: Kurulumu Doğrulayın

```bash
python -m yedekle.listele
```

En az bir yedek kaydı görünüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

Yedekleme yarıda kesiliyorsa disk alanını ve ağ zaman aşımı ayarlarını kontrol edin.""",

    "Kurulum Kılavuzu — Telemetri İstemcisi": """# Kurulum Kılavuzu — Telemetri İstemcisi

## Ön Koşullar

- Python 3.9 veya üzeri (düşük kaynaklı cihazlarda çalışmak üzere tasarlandı)
- Cihazın ağ erişimi (MQTT broker'a ulaşabilmeli)

## Adım 1: Paketi Kurun

```bash
pip install telemetri-istemci
```

## Adım 2: Cihaz Kimliğini Tanımlayın

```bash
export CIHAZ_ID=sensor-042
export BROKER_URL=mqtt://filo-agi:1883
```

## Adım 3: İstemciyi Başlatın

```bash
python -m telemetri.istemci
```

## Adım 4: Kurulumu Doğrulayın

Broker tarafında `telemetri/sensor-042` konusuna abone olun; birkaç saniye içinde
veri akışı görünmelidir.

## Sık Karşılaşılan Sorunlar

Veri akmıyorsa cihazın saat senkronizasyonunu (NTP) kontrol edin; büyük saat
sapmaları mesajların reddedilmesine yol açar.""",

    "Kurulum Kılavuzu — Kimlik Doğrulama Sunucusu": """# Kurulum Kılavuzu — Kimlik Doğrulama Sunucusu

## Ön Koşullar

- Python 3.10 veya üzeri
- PostgreSQL 14+
- Bir imzalama anahtarı çifti (JWT için)

## Adım 1: Depoyu İndirin

```bash
git clone https://ornek-depo-adresi
cd kimlik-sunucusu
```

## Adım 2: İmzalama Anahtarlarını Üretin

```bash
python -m kimlik.anahtar_uret --hedef anahtarlar/
```

## Adım 3: Veritabanını Kurun ve Sunucuyu Başlatın

```bash
python -m kimlik.migrate
pip install -r requirements.txt
python -m kimlik.sunucu
```

## Adım 4: Kurulumu Doğrulayın

```bash
curl -X POST localhost:8001/token -d "kullanici=test&sifre=test"
```

Bir JWT belirteci dönüyorsa kurulum başarılıdır.

## Sık Karşılaşılan Sorunlar

"invalid signature" hatası alıyorsanız istemci ve sunucunun aynı anahtar çiftini
kullandığından emin olun.""",

    # --- Kullanıcı Kılavuzu ---
    "Kullanıcı Kılavuzu: Fatura Oluşturma": """# Kullanıcı Kılavuzu: Fatura Oluşturma

Bu bölüm, bir müşteri için manuel fatura oluşturma adımlarını açıklar.

## 1. Faturalar Sekmesini Açın

Sol menüden "Faturalar" sekmesine, ardından "Yeni Fatura" butonuna tıklayın.

## 2. Müşteri ve Kalemleri Seçin

Kayıtlı müşteri listesinden birini seçin veya yeni müşteri ekleyin, ardından
faturalanacak ürün/hizmet kalemlerini girin.

## 3. Vade ve Ödeme Koşullarını Belirleyin

Vade tarihi ve ödeme yöntemi bilgilerini doldurun.

## 4. Faturayı Gönderin

"Oluştur ve Gönder" butonuna bastığınızda fatura PDF olarak müşteriye e-posta ile iletilir.

> Not: Taslak olarak kaydedilen faturalar "Taslaklar" sekmesinden düzenlenebilir.""",

    "Kullanıcı Kılavuzu: Takım Üyesi Davet Etme": """# Kullanıcı Kılavuzu: Takım Üyesi Davet Etme

Bu bölüm, çalışma alanınıza yeni bir takım üyesi eklemeyi açıklar.

## 1. Ayarlar → Takım Bölümüne Gidin

Sağ üstteki profil menüsünden "Ayarlar", ardından "Takım" sekmesini açın.

## 2. Davet Bağlantısı Oluşturun

"Üye Davet Et" butonuna tıklayıp e-posta adresini ve rolünü ("üye" veya
"yönetici") girin.

## 3. Daveti Gönderin

"Gönder" butonuna bastığınızda davet bağlantısı e-posta ile iletilir.

## 4. Davetin Durumunu Takip Edin

Bekleyen davetler "Beklemede" etiketiyle listede görünür; gerekirse yeniden
gönderilebilir veya iptal edilebilir.

> Not: Davet bağlantıları 7 gün sonra otomatik olarak geçersiz olur.""",

    "Kullanıcı Kılavuzu: Bildirim Tercihlerini Ayarlama": """# Kullanıcı Kılavuzu: Bildirim Tercihlerini Ayarlama

Bu bölüm, hangi olaylar için hangi kanaldan bildirim alacağınızı özelleştirmeyi açıklar.

## 1. Bildirim Ayarlarını Açın

Profil menüsünden "Bildirim Tercihleri" sekmesine gidin.

## 2. Kanalları Seçin

Her olay türü (sipariş güncellemesi, güvenlik uyarısı, haftalık özet vb.) için
e-posta, push veya SMS kanallarından istediklerinizi işaretleyin.

## 3. Sessiz Saatleri Ayarlayın

İsteğe bağlı olarak, push bildirimlerinin gönderilmeyeceği saat aralığını belirleyebilirsiniz.

## 4. Değişiklikleri Kaydedin

"Kaydet" butonuna bastıktan sonra tercihler anında etkinleşir.

> Not: Güvenlikle ilgili kritik uyarılar (örn. şüpheli giriş) tercihlerden
> bağımsız olarak her zaman gönderilir.""",

    "Kullanıcı Kılavuzu: Harita Üzerinde Konum İşaretleme": """# Kullanıcı Kılavuzu: Harita Üzerinde Konum İşaretleme

Bu bölüm, harita görünümünde özel bir konum işaretleme adımlarını açıklar.

## 1. Harita Görünümünü Açın

Sol menüden "Harita" sekmesine tıklayın.

## 2. İşaretleme Aracını Seçin

Araç çubuğundaki iğne simgesine tıklayıp harita üzerinde istediğiniz noktaya
tıklayın.

## 3. Konum Bilgilerini Girin

Açılan panelde konuma bir ad ve isteğe bağlı not ekleyin.

## 4. İşareti Kaydedin

"Kaydet" butonuna bastığınızda işaret haritada kalıcı olarak görünür ve ekip
üyeleriyle paylaşılabilir.

> Not: Bir işareti silmek için üzerine sağ tıklayıp "Kaldır" seçeneğini kullanın.""",

    "Kullanıcı Kılavuzu: Dosya Paylaşımı": """# Kullanıcı Kılavuzu: Dosya Paylaşımı

Bu bölüm, bir dosyayı ekip dışından biriyle güvenli şekilde paylaşmayı açıklar.

## 1. Dosyayı Seçin

Dosya listesinde paylaşmak istediğiniz dosyanın yanındaki "..." menüsünü açın.

## 2. "Bağlantı ile Paylaş" Seçeneğini Kullanın

Açılan pencerede bağlantının süresini (1 gün, 7 gün, süresiz) ve isteğe bağlı
bir şifre belirleyin.

## 3. Bağlantıyı Kopyalayın

"Bağlantıyı Kopyala" butonuna basıp alıcıya iletin.

## 4. Erişimi Yönetin

Paylaşılan bağlantılar "Paylaşımlarım" sekmesinden istediğiniz zaman iptal
edilebilir.

> Not: Süresi dolan bağlantılar otomatik olarak devre dışı kalır, yeniden
> etkinleştirilemez.""",

    "Kullanıcı Kılavuzu: Görev Panosu Kullanımı": """# Kullanıcı Kılavuzu: Görev Panosu Kullanımı

Bu bölüm, sürükle-bırak görev panosunda iş takibini açıklar.

## 1. Pano Görünümünü Açın

Proje menüsünden "Pano" sekmesine tıklayın.

## 2. Yeni Görev Ekleyin

"Yapılacak" sütununun altındaki "+" butonuna tıklayıp görev başlığını yazın.

## 3. Görevi Sürükleyin

Görev üzerinde çalışmaya başladığınızda kartı "Devam Ediyor" sütununa,
tamamlandığında "Tamamlandı" sütununa sürükleyin.

## 4. Görev Detaylarını Doldurun

Karta tıklayarak atanan kişi, son tarih ve etiket bilgilerini ekleyebilirsiniz.

> Not: Sütun isimleri proje ayarlarından özelleştirilebilir.""",

    "Kullanıcı Kılavuzu: Uçuş Günlüğü Dışa Aktarma": """# Kullanıcı Kılavuzu: Uçuş Günlüğü Dışa Aktarma

Bu bölüm, tamamlanmış bir görevin uçuş günlüğünü dışa aktarma adımlarını açıklar.

## 1. Görev Arşivini Açın

Sol menüden "Görev Arşivi" sekmesine gidip ilgili görevi seçin.

## 2. Günlük Sekmesine Geçin

Görev detay sayfasında "Uçuş Günlüğü" sekmesine tıklayın.

## 3. Format ve Aralığı Seçin

CSV veya KML formatını, isteğe bağlı olarak belirli bir zaman aralığını seçin.

## 4. Dışa Aktarın

"Dışa Aktar" butonuna bastığınızda dosya tarayıcınıza indirilir.

> Not: Büyük görevlerde dışa aktarma birkaç dakika sürebilir; işlem
> tamamlandığında bildirim alırsınız.""",

    # --- SSS ---
    "SSS — Faturalandırma": """# Sıkça Sorulan Sorular — Faturalandırma

**S: Faturamı nereden indirebilirim?**

C: "Faturalar" sekmesinden geçmiş tüm faturalarınızı PDF olarak indirebilirsiniz.

**S: Ödeme yöntemimi nasıl değiştiririm?**

C: Ayarlar → Faturalandırma bölümünden yeni bir kart ekleyip varsayılan olarak
işaretleyebilirsiniz.

**S: Plan yükseltmesi ne zaman ücretlendirilir?**

C: Yükseltmeler anında etkinleşir; fark, kalan gün sayısına oranlanarak bir
sonraki faturaya yansıtılır.

**S: Fatura üzerinde vergi numaramı nasıl güncellerim?**

C: Ayarlar → Fatura Bilgileri bölümünden güncelleyebilirsiniz; değişiklik bir
sonraki faturadan itibaren geçerli olur.""",

    "SSS — Hesap Güvenliği": """# Sıkça Sorulan Sorular — Hesap Güvenliği

**S: İki faktörlü doğrulamayı nasıl açarım?**

C: Ayarlar → Güvenlik bölümünden "İki Faktörlü Doğrulama"yı etkinleştirip bir
kimlik doğrulama uygulamasıyla eşleştirebilirsiniz.

**S: Şüpheli bir giriş bildirimi aldım, ne yapmalıyım?**

C: Şifrenizi hemen değiştirin ve Ayarlar → Oturumlar bölümünden tüm aktif
oturumları sonlandırın.

**S: API anahtarımı sızdırdıysam ne yapmalıyım?**

C: Ayarlar → API Anahtarları bölümünden ilgili anahtarı derhal iptal edip yenisini
oluşturun; eski anahtar anında geçersiz olur.

**S: Hesabımı nasıl kalıcı olarak silerim?**

C: Ayarlar → Hesap bölümünden "Hesabı Sil" seçeneğini kullanabilirsiniz; bu
işlem geri alınamaz.""",

    "SSS — Dosya Depolama Limitleri": """# Sıkça Sorulan Sorular — Dosya Depolama Limitleri

**S: Depolama limitim ne kadar?**

C: Standart planda 50 GB, kurumsal planda kullanım bazlı sınırsız depolama
sunulmaktadır.

**S: Limitimi aştığımda ne olur?**

C: Yeni dosya yükleyemezsiniz; mevcut dosyalarınıza erişim kesintisiz devam eder.

**S: Silinen dosyalar depolama alanımı hemen boşaltır mı?**

C: Hayır, silinen dosyalar 30 gün boyunca "Çöp Kutusu"nda tutulur ve bu süre
boyunca alan kullanmaya devam eder.

**S: Tek bir dosyanın boyut limiti nedir?**

C: Standart planda dosya başına 5 GB'a kadar yükleme yapılabilir.""",

    "SSS — Mobil Bildirimler": """# Sıkça Sorulan Sorular — Mobil Bildirimler

**S: Push bildirimleri neden gelmiyor?**

C: Cihaz ayarlarınızda uygulama için bildirim izninin açık olduğundan ve
uygulama içi Bildirim Tercihleri'nde ilgili kanalın etkin olduğundan emin olun.

**S: Bildirim sesini nasıl değiştiririm?**

C: Bu, işletim sisteminin bildirim ayarlarından yönetilir; uygulama kendi
başına özel ses sunmaz.

**S: Birden fazla cihazda oturum açtım, bildirimler hepsine mi gelir?**

C: Evet, bildirim tercihleri hesaba bağlıdır; tüm oturum açılmış cihazlara
aynı anda gönderilir.

**S: Rahatsız Etmeyin modunda bildirimler kaybolur mu?**

C: Hayır, bildirimler sunucuda saklanır ve uygulamayı açtığınızda bildirim
merkezinde görünür.""",

    "SSS — Veri Aktarımı ve Yedekleme": """# Sıkça Sorulan Sorular — Veri Aktarımı ve Yedekleme

**S: Verilerimin tam yedeğini nasıl alırım?**

C: Ayarlar → Veri Yönetimi bölümünden "Tüm Verileri Dışa Aktar" seçeneğiyle
ZIP formatında bir arşiv talep edebilirsiniz; hazır olduğunda e-posta ile
bilgilendirilirsiniz.

**S: Otomatik yedekleme sıklığı nedir?**

C: Kurumsal planda veriler günlük olarak otomatik yedeklenir ve 90 gün saklanır.

**S: Başka bir hesaba veri taşıyabilir miyim?**

C: Evet, destek ekibimizle iletişime geçerek hesaplar arası taşıma talebinde
bulunabilirsiniz.

**S: Yedekten geri yükleme ne kadar sürer?**

C: Veri hacmine bağlı olarak genellikle birkaç saat içinde tamamlanır.""",

    "SSS — API Entegrasyonu": """# Sıkça Sorulan Sorular — API Entegrasyonu

**S: API anahtarımı nereden alırım?**

C: Ayarlar → Geliştirici bölümünden yeni bir API anahtarı oluşturabilirsiniz.

**S: Sandbox (test) ortamı var mı?**

C: Evet, `api-sandbox.ornek.com` adresi üzerinden gerçek işlem yapmadan test
edebilirsiniz.

**S: Webhook denemeleri kaç kez tekrarlanır?**

C: Başarısız webhook çağrıları üstel geri çekilme (exponential backoff) ile
en fazla 5 kez tekrar denenir.

**S: API sürümü ne zaman değişir, eski sürüm ne kadar desteklenir?**

C: Büyük sürüm değişiklikleri en az 6 ay önceden duyurulur; eski sürüm bu süre
boyunca paralel çalışmaya devam eder.""",

    "SSS — Ekip ve Roller": """# Sıkça Sorulan Sorular — Ekip ve Roller

**S: Roller arasındaki fark nedir?**

C: "Görüntüleyici" yalnızca okuma yapabilir, "Üye" içerik oluşturup düzenleyebilir,
"Yönetici" ise faturalandırma ve ekip ayarlarını da yönetebilir.

**S: Bir üyenin rolünü nasıl değiştiririm?**

C: Ayarlar → Takım bölümünden ilgili üyenin yanındaki rol menüsünden yeni rolü
seçebilirsiniz.

**S: Ekipten çıkarılan bir üyenin verileri ne olur?**

C: Üyenin oluşturduğu içerikler ekipte kalır, yalnızca kişisel erişimi
kaldırılır.

**S: Ekipteki maksimum üye sayısı nedir?**

C: Standart planda 10, kurumsal planda sınırsız üye eklenebilir.""",

    # --- Mimari Doküman ---
    "Mimari Doküman — Bildirim Servisi": """# Mimari Doküman — Bildirim Servisi

## Amaç

Bu doküman, çok kanallı bildirim servisinin bileşenlerini ve veri akışını açıklar.

## Bileşenler

- **Kabul Katmanı**: Diğer servislerden gelen bildirim isteklerini alan API.
- **Kuyruk**: Redis tabanlı, kanal başına ayrı kuyruklar.
- **Gönderici Çalışanlar (Workers)**: Her kanal için ayrı ölçeklenen tüketici süreçleri.
- **Sağlayıcı Adaptörleri**: SMTP, SMS operatörleri ve push servisleri için soyutlama katmanı.

## Veri Akışı

1. Kaynak servis, Kabul Katmanı'na bir bildirim isteği gönderir.
2. İstek doğrulanır ve ilgili kanal kuyruğuna eklenir.
3. Gönderici Çalışan, kuyruktan mesajı alıp uygun sağlayıcı adaptörünü çağırır.
4. Sonuç (başarılı/başarısız) durum tablosuna yazılır; başarısızlıkta yeniden
   deneme kuyruğuna aktarılır.

## Ölçeklenebilirlik Notları

Her kanal bağımsız ölçeklenir; SMS gönderimi e-posta gönderiminden çok daha
düşük throughput gerektirdiği için ayrı worker havuzları kullanılır.""",

    "Mimari Doküman — Arama ve İndeksleme Alt Sistemi": """# Mimari Doküman — Arama ve İndeksleme Alt Sistemi

## Amaç

Bu doküman, ürün kataloğu arama alt sisteminin indeksleme ve sorgulama
mimarisini açıklar.

## Bileşenler

- **Değişiklik Yakalayıcı (CDC)**: Ana veritabanındaki değişiklikleri olay
  akışına yayınlar.
- **İndeksleyici**: Olayları tüketip arama motorunda ilgili dokümanı günceller.
- **Sorgu API'si**: İstemcilerin arama isteklerini karşılayan katman.
- **Önbellek**: Sık sorgulanan aramalar için kısa ömürlü sonuç önbelleği.

## Veri Akışı

1. Ürün veritabanında bir değişiklik olur (ekleme/güncelleme/silme).
2. CDC bu değişikliği bir olay olarak yayınlar.
3. İndeksleyici olayı işleyip arama motorundaki dokümanı senkronize eder.
4. Sorgu API'si, önbellekte yoksa arama motorundan sonucu alıp istemciye döner.

## Ölçeklenebilirlik Notları

İndeksleme ile sorgulama birbirinden bağımsız ölçeklenir; yoğun toplu
güncellemeler sorgu gecikmesini etkilemez çünkü ayrı süreçlerde çalışırlar.""",

    "Mimari Doküman — Dosya Depolama Katmanı": """# Mimari Doküman — Dosya Depolama Katmanı

## Amaç

Bu doküman, uygulama genelinde kullanılan dosya depolama katmanının mimarisini
açıklar.

## Bileşenler

- **Yükleme API'si**: İstemcilerden dosya alan, doğrulayan uç nokta.
- **Nesne Deposu**: S3 uyumlu kalıcı depolama.
- **Meta Veri Tablosu**: Dosya sahibi, boyut, MIME tipi gibi bilgileri tutar.
- **CDN**: Sık erişilen dosyalar için kenar (edge) önbellekleme.

## Veri Akışı

1. İstemci dosyayı Yükleme API'sine gönderir.
2. API dosyayı doğrulayıp Nesne Deposu'na yazar, meta veriyi kaydeder.
3. Erişim taleplerinde önce CDN kontrol edilir; yoksa Nesne Deposu'ndan alınıp
   CDN'e yayılır.

## Ölçeklenebilirlik Notları

Nesne Deposu doğası gereği yatay ölçeklenir; asıl darboğaz genellikle Meta Veri
Tablosu'dur, bu yüzden okuma replikaları kullanılır.""",

    "Mimari Doküman — Telemetri Veri Hattı": """# Mimari Doküman — Telemetri Veri Hattı

## Amaç

Bu doküman, uzak cihazlardan gelen telemetri verisinin toplama, işleme ve
saklama hattını açıklar.

## Bileşenler

- **Alım Uç Noktası**: MQTT/HTTP üzerinden ham veriyi kabul eden katman.
- **Doğrulama ve Zenginleştirme**: Şema doğrulama, cihaz meta verisi ekleme.
- **Akış İşleyici**: Anlık toplulaştırma ve anomali tespiti yapan bileşen.
- **Zaman Serisi Veritabanı**: Uzun dönem saklama ve sorgulama.

## Veri Akışı

1. Cihaz, ölçümü Alım Uç Noktası'na gönderir.
2. Veri şemaya göre doğrulanır, cihaz meta verisiyle zenginleştirilir.
3. Akış İşleyici, veriyi hem gerçek zamanlı panolara hem de Zaman Serisi
   Veritabanı'na yönlendirir.
4. Anomali tespit edilirse ayrı bir uyarı konusuna yayınlanır.

## Ölçeklenebilirlik Notları

Alım katmanı yatay ölçeklenir; Zaman Serisi Veritabanı, eski verinin otomatik
özetlenip küçültülmesi (downsampling) ile uzun vadede yönetilebilir kalır.""",

    "Mimari Doküman — Kimlik Doğrulama ve Yetkilendirme": """# Mimari Doküman — Kimlik Doğrulama ve Yetkilendirme

## Amaç

Bu doküman, merkezi kimlik doğrulama (authentication) ve yetkilendirme
(authorization) sisteminin mimarisini açıklar.

## Bileşenler

- **Kimlik Sunucusu**: Kullanıcı adı/şifre veya SSO ile giriş doğrulayan servis.
- **Belirteç (Token) Servisi**: Kısa ömürlü erişim belirteci ve uzun ömürlü
  yenileme belirteci üretir.
- **Yetki Motoru**: Rol tabanlı erişim kontrolü (RBAC) kararlarını verir.
- **Oturum Deposu**: Aktif oturumların ve iptal edilen belirteçlerin listesi.

## Veri Akışı

1. Kullanıcı Kimlik Sunucusu'na giriş bilgilerini gönderir.
2. Doğrulama başarılıysa Belirteç Servisi bir erişim ve yenileme belirteci üretir.
3. Sonraki her istekte, hizmetler Yetki Motoru'na sorup kullanıcının ilgili
   kaynağa erişim iznini kontrol eder.

## Ölçeklenebilirlik Notları

Belirteç doğrulama imza kontrolüyle durumsuz (stateless) yapılır; yalnızca iptal
kontrolü için Oturum Deposu'na gidilir, bu da merkezi veritabanı yükünü azaltır.""",

    "Mimari Doküman — Çok Kiracılı (Multi-tenant) Veri İzolasyonu": """# Mimari Doküman — Çok Kiracılı (Multi-tenant) Veri İzolasyonu

## Amaç

Bu doküman, tek bir uygulama örneğinin birden fazla müşteriye (kiracıya) nasıl
izole biçimde hizmet verdiğini açıklar.

## Bileşenler

- **Kiracı Çözümleyici**: Gelen isteğin alan adı veya başlığından kiracı
  kimliğini belirler.
- **Satır Düzeyi Güvenlik**: Veritabanı sorgularına otomatik kiracı filtresi
  ekleyen katman.
- **Kiracı Bazlı Yapılandırma**: Her kiracının kendi özellik bayrakları ve
  limitleri.

## Veri Akışı

1. İstek geldiğinde Kiracı Çözümleyici kiracı kimliğini belirler ve isteğe iliştirir.
2. Veri katmanı her sorguya bu kimliği otomatik olarak filtre olarak ekler.
3. Yanıt, yalnızca ilgili kiracıya ait verilerle döner.

## Ölçeklenebilirlik Notları

Küçük kiracılar paylaşımlı veritabanı şemasında tutulur; çok büyük kiracılar
gerektiğinde ayrı bir veritabanı örneğine taşınabilir (kiracı başına izolasyon).""",

    "Mimari Doküman — Olay Güdümlü (Event-Driven) İşleme Hattı": """# Mimari Doküman — Olay Güdümlü (Event-Driven) İşleme Hattı

## Amaç

Bu doküman, servisler arası senkron çağrılar yerine olay güdümlü iletişimin
nasıl kurulduğunu açıklar.

## Bileşenler

- **Olay Yayıncıları**: İş olaylarını (örn. "sipariş oluşturuldu") yayınlayan servisler.
- **Olay Akışı (Broker)**: Kafka/RabbitMQ gibi mesajlaşma altyapısı.
- **Aboneler**: İlgili olayları dinleyip kendi sorumluluk alanlarında işlem yapan servisler.
- **Ölü Mektup Kuyruğu (DLQ)**: İşlenemeyen olayların biriktirildiği kuyruk.

## Veri Akışı

1. Bir servis iş olayını Olay Akışı'na yayınlar.
2. İlgili tüm Aboneler olayı bağımsız olarak tüketir (örn. fatura servisi,
   bildirim servisi, analitik servisi).
3. Bir abone olayı işleyemezse, birkaç denemeden sonra olay DLQ'ya taşınır ve
   uyarı tetiklenir.

## Ölçeklenebilirlik Notları

Yayıncı ve aboneler birbirinden tamamen bağımsız ölçeklenir; yeni bir abone
eklemek yayıncı tarafında hiçbir değişiklik gerektirmez.""",

    # --- Sürüm Notları ---
    "Sürüm Notları — Mobil Uygulama v3.1.0": """# Sürüm Notları — Mobil Uygulama

## v3.1.0 — 2026-02-10

### Eklenenler
- Karanlık mod desteği eklendi.
- Çevrimdışı mod: internet bağlantısı olmadan son görüntülenen içerikler
  görüntülenebilir.

### Değişenler
- Uygulama açılış süresi ortalama %25 kısaltıldı.

### Düzeltilenler
- Bildirim rozetinin bazı Android cihazlarda güncellenmeme sorunu giderildi.

## v3.0.2 — 2026-01-22

### Düzeltilenler
- Türkçe karakter içeren dosya adlarının yüklenememesi sorunu giderildi.""",

    "Sürüm Notları — API v1.4.0": """# Sürüm Notları — API

## v1.4.0 — 2026-03-05

### Eklenenler
- `/siparisler` uç noktasına sayfalama (cursor tabanlı) desteği eklendi.
- Webhook olaylarına `idempotency_key` alanı eklendi.

### Değişenler
- Hız sınırı (rate limit) başlıkları artık her yanıtta döndürülüyor
  (`X-RateLimit-Remaining`).

### Kullanımdan Kaldırılanlar
- `/eski-siparisler` uç noktası kullanımdan kaldırıldı; 2026-09-01'de
  tamamen kapatılacak, `/siparisler`'e geçiş yapın.

## v1.3.1 — 2026-02-01

### Düzeltilenler
- Eş zamanlı istek limitini aşan çağrılarda yanlış hata kodu (500 yerine 429)
  dönme sorunu giderildi.""",

    "Sürüm Notları — Depolama Servisi v2.0.0": """# Sürüm Notları — Depolama Servisi

## v2.0.0 — 2026-01-18

### Eklenenler
- Bölümlü (multipart) yükleme artık 5 GB'a kadar dosyaları destekliyor
  (önceki limit 1 GB idi).

### Değişenler
- **Kırıcı değişiklik:** imzalı URL'lerin varsayılan geçerlilik süresi 24
  saatten 1 saate düşürüldü; uzun süreli erişim için `sure_saniye` parametresini
  açıkça belirtin.

### Düzeltilenler
- Eş zamanlı silme isteklerinde oluşan yarış durumu (race condition) giderildi.

## v1.9.4 — 2025-12-10

### Düzeltilenler
- Büyük dosyalarda checksum doğrulamasının zaman aşımına uğraması sorunu
  giderildi.""",

    "Sürüm Notları — Bildirim Servisi v1.2.0": """# Sürüm Notları — Bildirim Servisi

## v1.2.0 — 2026-02-20

### Eklenenler
- Push bildirimleri için sessiz saat (quiet hours) desteği eklendi.
- Şablonlarda değişken doğrulama artık gönderim öncesi yapılıyor.

### Değişenler
- SMS sağlayıcısı devreye alma (failover) süresi 30 saniyeden 5 saniyeye
  indirildi.

### Düzeltilenler
- Aynı bildirimin nadir durumlarda iki kez gönderilmesine yol açan kuyruk
  hatası giderildi.

## v1.1.3 — 2026-01-08

### Düzeltilenler
- E-posta şablonlarında bazı özel karakterlerin kaçışlanmaması (escaping)
  sorunu giderildi.""",

    "Sürüm Notları — Arama Motoru v4.0.0": """# Sürüm Notları — Arama Motoru

## v4.0.0 — 2026-01-30

### Eklenenler
- Yazım hatası toleranslı (fuzzy) arama desteği eklendi.
- Facet sonuçlarına ürün stok durumu eklendi.

### Değişenler
- **Kırıcı değişiklik:** sorgu API'sindeki `q` parametresi `sorgu` olarak
  yeniden adlandırıldı; eski parametre 3 ay boyunca da kabul edilecek.

### Düzeltilenler
- Türkçe büyük/küçük harf dönüşümünde ("İ"/"i", "I"/"ı") hatalı eşleşme sorunu
  giderildi.

## v3.5.1 — 2025-12-19

### Düzeltilenler
- İndeksleme sırasında bazı ürünlerin sessizce atlanması sorunu giderildi.""",

    "Sürüm Notları — Telemetri İstemcisi v1.1.0": """# Sürüm Notları — Telemetri İstemcisi

## v1.1.0 — 2026-02-14

### Eklenenler
- Bağlantı kesintilerinde yerel diske geçici veri yazma (buffering) desteği
  eklendi; bağlantı geri geldiğinde veri otomatik gönderilir.

### Değişenler
- Varsayılan örnekleme aralığı 5 saniyeden 1 saniyeye düşürüldü.

### Düzeltilenler
- Düşük bant genişliğinde mesajların sıraya girmeden kaybolması sorunu
  giderildi.

## v1.0.3 — 2026-01-05

### Düzeltilenler
- Saat dilimi farkı olan cihazlarda yanlış zaman damgası gönderilmesi sorunu
  giderildi.""",

    "Sürüm Notları — Yönetim Paneli v5.2.0": """# Sürüm Notları — Yönetim Paneli

## v5.2.0 — 2026-03-01

### Eklenenler
- Ekip aktivite günlüğüne dışa aktarma (CSV) özelliği eklendi.
- Rol bazlı özel izin (custom permission) tanımlama desteği eklendi.

### Değişenler
- Pano yükleme performansı, gereksiz API çağrıları azaltılarak iyileştirildi.

### Düzeltilenler
- Bazı tarayıcılarda tarih filtresinin sıfırlanmama sorunu giderildi.

## v5.1.2 — 2026-02-05

### Düzeltilenler
- Çok üyeli ekiplerde arama kutusunun yavaş yanıt vermesi sorunu giderildi.""",

    # --- Diğer ---
    "Güvenlik Politikası Şablonu": """# Güvenlik Politikası

## Kapsam

Bu politika, sistemin işlediği kullanıcı verilerinin korunmasına yönelik temel
kuralları tanımlar.

## Veri Şifreleme

- Aktarım sırasında tüm trafik TLS 1.2 veya üzeri ile şifrelenir.
- Durağan veriler (disk üzerinde) AES-256 ile şifrelenir.

## Erişim Kontrolü

- Üretim ortamına erişim, en az ayrıcalık ilkesine göre rol bazlı verilir.
- Tüm yönetici erişimleri için çok faktörlü doğrulama zorunludur.

## Güvenlik Açığı Bildirimi

Bir güvenlik açığı tespit ederseniz `guvenlik@ornek.com` adresine, açığın
kapsamını ve tekrar üretme adımlarını içeren bir e-posta gönderin. Bildirimler
2 iş günü içinde yanıtlanır.

## Düzenli Denetim

Erişim kayıtları üç ayda bir gözden geçirilir; kullanılmayan hesaplar 90 gün
sonra otomatik olarak devre dışı bırakılır.""",

    "Kod Stil Rehberi Şablonu": """# Kod Stil Rehberi

## Genel İlkeler

- Okunabilirlik, kısalıktan önceliklidir.
- Bir fonksiyon tek bir sorumluluğa sahip olmalıdır.
- Yorum, "ne" yapıldığını değil "neden" yapıldığını açıklamalıdır.

## İsimlendirme

- Değişken ve fonksiyon isimleri anlamlı ve açıklayıcı olmalı (`i`, `tmp` gibi
  isimlerden kaçının).
- Boolean değişkenler `is_`, `has_` gibi öneklerle başlamalıdır.

## Biçimlendirme

- Otomatik biçimlendirici (formatter) her commit öncesi çalıştırılmalıdır.
- Satır uzunluğu 100 karakteri geçmemelidir.

## Test Yazımı

- Her yeni özellik için en az bir mutlu yol (happy path) ve bir hata durumu
  testi eklenmelidir.
- Test isimleri, test ettikleri davranışı cümle gibi okunacak şekilde
  yazılmalıdır.""",

    "Terimler Sözlüğü (Glossary) Şablonu": """# Terimler Sözlüğü

**Kiracı (Tenant):** Sistemi kullanan, verileri diğerlerinden izole edilmiş
her bir müşteri organizasyonu.

**Belirteç (Token):** Bir isteğin kimliğini doğrulamak için kullanılan, süreli
şifreli metin.

**Ölü Mektup Kuyruğu (DLQ):** Birden fazla denemeye rağmen işlenemeyen
mesajların biriktirildiği kuyruk.

**İdempotency Anahtarı:** Aynı isteğin yanlışlıkla birden fazla kez
işlenmesini önlemek için istemcinin ürettiği benzersiz anahtar.

**Devreye Alma (Failover):** Birincil bileşen çalışmaz duruma geldiğinde
otomatik olarak yedek bileşene geçiş.

**Örnekleme Hızı (Sampling Rate):** Bir sensörün veya izleme sisteminin
birim zamanda kaç ölçüm/kayıt aldığı.""",

    "Yeni Başlayan (Onboarding) Rehberi Şablonu": """# Yeni Başlayan Rehberi

## İlk Gün

1. Şirket e-posta hesabınız ve VPN erişiminiz BT ekibi tarafından
   sağlanacaktır.
2. Takım liderinizle 30 dakikalık bir tanışma görüşmesi planlayın.
3. Bu depoyu klonlayıp yerel geliştirme ortamınızı kurun (bkz. Kurulum
   Kılavuzu).

## İlk Hafta

- Kod tabanını tanımak için etiketlenmiş "iyi-ilk-görev" (good-first-issue)
  konularından birini seçip çözün.
- Ekip standup toplantılarına katılın ve mevcut sprint panosunu inceleyin.

## İlk Ay

- En az bir üretim değişikliği yapıp code review sürecinden geçirin.
- Nöbetçi (on-call) rotasyonuna gölge (shadow) olarak katılın.

## Faydalı Kaynaklar

Mimari dokümanlar, stil rehberi ve sık sorulan sorular şirket içi bilgi
tabanında bulunabilir.""",

    "Kullanımdan Kaldırma (Deprecation) Duyurusu Şablonu": """# Kullanımdan Kaldırma Duyurusu: v1 Kimlik Doğrulama Uç Noktası

## Özet

`/v1/auth/login` uç noktası kullanımdan kaldırılmıştır ve **2026-12-01**
tarihinde tamamen kapatılacaktır.

## Neden Kaldırılıyor?

Yeni `/v2/auth/token` uç noktası, çok faktörlü doğrulama ve daha kısa ömürlü
belirteçler gibi güvenlik iyileştirmeleri sunmaktadır.

## Geçiş Adımları

1. İstemci kodunuzda `/v1/auth/login` çağrılarını `/v2/auth/token` ile
   değiştirin.
2. Yanıt gövdesindeki `access_token` alanı aynı kalmıştır; ek olarak
   `refresh_token` alanı eklenmiştir.
3. Test ortamında doğrulayıp üretime alın.

## Zaman Çizelgesi

- **Bugün:** v1 hâlâ çalışıyor, ancak yanıtlarda uyarı başlığı (`Deprecation`)
  dönüyor.
- **2026-10-01:** v1 istekleri için hız sınırı düşürülecek.
- **2026-12-01:** v1 tamamen kapatılacak.""",

    "Nöbetçi (On-Call) El Kitabı Şablonu": """# Nöbetçi (On-Call) El Kitabı

## Sorumluluklar

Nöbetçi mühendis, mesai saati dışında tetiklenen tüm kritik (P1/P2) uyarılara
15 dakika içinde yanıt vermekle sorumludur.

## Uyarı Aldığınızda

1. Uyarıyı onaylayın (acknowledge) ki başka birine yönlendirilmesin.
2. İlgili panoyu (dashboard) açıp etkinin kapsamını değerlendirin.
3. Kullanıcı etkisi varsa durum sayfasını güncelleyin.

## Yükseltme (Escalation)

30 dakika içinde sorunu çözemez veya kapsamını belirleyemezseniz, ikincil
nöbetçiyi çağırın; kritik olaylarda takım liderini doğrudan arayabilirsiniz.

## Olay Sonrası

Her P1/P2 olayından sonra 48 saat içinde bir olay sonrası rapor (postmortem)
taslağı oluşturulmalıdır.

## Sık Kullanılan Komutlar

```bash
kubectl get pods -n uretim
kubectl logs -n uretim <pod-adi> --tail=200
```""",

    "Veri Saklama Politikası Şablonu": """# Veri Saklama Politikası

## Amaç

Bu politika, farklı veri türlerinin ne kadar süreyle saklanacağını ve nasıl
imha edileceğini tanımlar.

## Saklama Süreleri

| Veri Türü | Saklama Süresi | İmha Yöntemi |
|---|---|---|
| Kullanıcı hesap verisi | Hesap aktif olduğu sürece + 30 gün | Kalıcı silme |
| Uygulama günlükleri (log) | 90 gün | Otomatik döngüsel silme |
| Faturalandırma kayıtları | 10 yıl (yasal zorunluluk) | Arşivleme, sonra silme |
| Yedekler | 90 gün | Otomatik döngüsel silme |

## Silme Talepleri

Kullanıcılar Ayarlar → Veri Yönetimi bölümünden hesap silme talebinde
bulunabilir; talep sonrası 30 günlük bekleme süresi vardır, bu süre içinde
talep iptal edilebilir.

## Yasal İstisnalar

Devam eden bir yasal işlem veya denetim kapsamında olan veriler, ilgili süreç
sonuçlanana kadar imha edilmez.""",

    "Hız Sınırlama (Rate Limit) Politikası Şablonu": """# Hız Sınırlama (Rate Limit) Politikası

## Amaç

Bu doküman, API'nin aşırı kullanımdan korunması için uygulanan hız sınırlama
kurallarını açıklar.

## Limitler

| Plan | Dakika Başına İstek | Eş Zamanlı Bağlantı |
|---|---|---|
| Ücretsiz | 60 | 5 |
| Standart | 600 | 20 |
| Kurumsal | Özelleştirilebilir | Özelleştirilebilir |

## Limit Aşıldığında

Sunucu `429 Too Many Requests` durum koduyla yanıt verir ve `Retry-After`
başlığında kaç saniye sonra tekrar denenebileceğini belirtir.

## Bildirim Başlıkları

Her yanıt, kalan kotayı gösteren `X-RateLimit-Remaining` ve pencerenin
sıfırlanacağı zamanı gösteren `X-RateLimit-Reset` başlıklarını içerir.

## İstisnalar

Toplu veri aktarım uç noktaları (`/export/*`) ayrı, daha düşük bir limit
havuzuna tabidir; bu uç noktalar için önceden onay gerekir.""",

    "Olay Sonrası Rapor (Postmortem) Şablonu": """# Olay Sonrası Rapor: Kısa Süreli API Kesintisi

## Özet

**Tarih:** 2026-02-11 · **Süre:** 23 dakika · **Etki:** API isteklerinin
%40'ında 500 hatası

## Zaman Çizelgesi

- **14:02** — İzleme sistemi hata oranı artışını tespit etti, uyarı tetiklendi.
- **14:05** — Nöbetçi mühendis olayı onayladı, araştırmaya başladı.
- **14:14** — Kök neden, veritabanı bağlantı havuzunun tükenmesi olarak
  belirlendi.
- **14:20** — Bağlantı havuzu boyutu artırılıp servis yeniden başlatıldı.
- **14:25** — Hata oranı normale döndü, olay kapatıldı.

## Kök Neden

Yakın zamanda eklenen bir arka plan işi, bağlantıları düzgün kapatmadan
havuzdan sürekli yeni bağlantı talep ediyordu.

## Aksiyon Maddeleri

- [ ] Arka plan işine bağlantı zaman aşımı ekle (Sahip: B. Ekip, 1 hafta)
- [ ] Bağlantı havuzu doluluk oranı için uyarı eşiği ekle (Sahip: Platform, 3 gün)
- [ ] Yük testlerine bağlantı sızıntısı senaryosu ekle (Sahip: QA, 2 hafta)""",

    "Kod İnceleme Kontrol Listesi Şablonu": """# Kod İnceleme Kontrol Listesi

## Gönderen İçin

- [ ] Değişiklik tek bir amaca odaklanıyor, ilgisiz değişiklikler ayrı PR'da.
- [ ] Yeni davranış için testler eklendi.
- [ ] PR açıklaması "ne" değişti değil "neden" değişti sorusunu yanıtlıyor.

## İnceleyen İçin

- [ ] Değişiklik, çözmeyi iddia ettiği sorunu gerçekten çözüyor mu?
- [ ] Hata durumları (boş girdi, ağ hatası, eş zamanlılık) ele alınmış mı?
- [ ] Geriye dönük uyumluluğu bozan bir değişiklik varsa açıkça belirtilmiş mi?
- [ ] Güvenlik açısından hassas bir alan (kimlik doğrulama, ödeme, kullanıcı
  verisi) değiştiyse ek dikkatle incelendi mi?

## Birleştirmeden (Merge) Önce

- [ ] CI tüm kontrollerden geçti.
- [ ] En az bir onay alındı.
- [ ] Gerekliyse dokümantasyon güncellendi.""",

    "Felaket Kurtarma Planı Şablonu": """# Felaket Kurtarma Planı

## Amaç

Bu doküman, birincil veri merkezinin tamamen kullanılamaz hale gelmesi
durumunda hizmetin nasıl geri getirileceğini tanımlar.

## Hedefler

- **RTO (Kurtarma Süresi Hedefi):** 4 saat
- **RPO (Kurtarma Noktası Hedefi):** 15 dakika (en fazla kabul edilebilir veri
  kaybı)

## Yedekli Bölge Mimarisi

Veritabanı, ikincil bölgeye 15 dakikada bir eşzamansız olarak çoğaltılır.
Uygulama sunucuları ikincil bölgede pasif (standby) modda hazır tutulur.

## Kurtarma Adımları

1. Olayı doğrulayıp felaket kurtarma prosedürünü resmi olarak başlatın.
2. DNS yönlendirmesini ikincil bölgeye çevirin.
3. İkincil bölgedeki veritabanını yazma moduna alın.
4. Uygulama sunucularını etkinleştirip sağlık kontrollerini doğrulayın.
5. İzleme panellerinden trafiğin ikincil bölgeye geçtiğini teyit edin.

## Test Sıklığı

Bu plan, gerçek bir kesinti yaşanmadan hazır olduğundan emin olmak için üç
ayda bir tatbikatla test edilir.""",
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
