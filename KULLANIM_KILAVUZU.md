# Render Üzerinde Kripto Ticaret Botu Kullanım Kılavuzu

Bu kılavuz, bu depodaki Python botlarını Render platformunda nasıl yayınlayacağınızı ve yapılandıracağınızı adım adım açıklamaktadır.

## 1. Hazırlık
- Bir **GitHub** hesabınızın olduğundan emin olun.
- Bir **Render.com** hesabınızın olduğundan emin olun.
- İlgili borsa (Binance veya KuCoin) için API anahtarlarınızı (Key ve Secret) hazır bulundurun.

## 2. Depoyu Forklama
Bu projeyi kendi GitHub hesabınıza "Fork" edin. Bu sayede Render kendi deponuza erişebilir.

## 3. Render Üzerinde Yayınlama (Deployment)
1. Render Dashboard'a gidin.
2. **"New"** butonuna tıklayın ve **"Blueprint"** seçeneğini seçin.
3. Forkladığınız GitHub deposunu bağlayın.
4. Bir "Service Group" adı verin (örneğin: `ticaret-botlarim`).
5. Render, projedeki `render.yaml` dosyasını otomatik olarak tanıyacak ve bir **"Background Worker"** oluşturacaktır.

## 4. API Anahtarlarını ve Botu Ayarlama
Yayınlama işlemi başladığında veya bittikten sonra Render panelinde **"Environment"** sekmesine gitmeniz gerekir. Burada şu değişkenleri tanımlamalısınız:

### Temel Ayar:
- `BOT_PATH`: Çalıştırmak istediğiniz botun dosya yolu.
  - Örn: `BINANCE/bot_3.py` (Tavsiye edilen Binance botu)
  - Örn: `KUCOIN/bot_1301[kucoin].py` (KuCoin botu)

### Borsa Ayarları:
**Binance kullanıyorsanız:**
- `BINANCE_API_KEY`: Binance API Anahtarınız.
- `BINANCE_API_SECRET`: Binance API Gizli Anahtarınız.

**KuCoin kullanıyorsanız:**
- `KUCOIN_API_KEY`: KuCoin API Anahtarınız.
- `KUCOIN_API_SECRET`: KuCoin API Gizli Anahtarınız.
- `KUCOIN_API_PASSPHRASE`: KuCoin API Parolanız.

## 5. Mevcut Botlar ve Özellikleri

### Binance Botları
1. **bot_1.py**: Basit fiyat referans botu. Belirli bir kar oranına göre al-sat yapar.
2. **bot_2.py**: RSI (Göreceli Güç Endeksi) göstergesini kullanır.
3. **bot_3.py (Gelişmiş)**: RSI + MACD + StopLoss (Zarar Durdur) özelliklerine sahiptir. En güvenli seçenek budur.

### KuCoin Botları
1. **bot_1301[kucoin].py**: Sipariş defteri dengesi (order rate) ve RSI göstergesine göre işlem yapar.

## 6. İzleme ve Günlükler (Logs)
Botunuzun ne yaptığını görmek için Render panelindeki **"Logs"** sekmesini kullanabilirsiniz. Bot, her dakikada bir mevcut durumu (fiyat, RSI, strateji vb.) ekrana yazdıracaktır.

## Önemli Uyarı
Bu botlar finansal risk içerir. API anahtarlarınızı oluştururken sadece "Spot Trading" yetkisi verdiğinizden ve "Withdrawal" (Para Çekme) yetkisini **kapattığınızdan** emin olun.

---
*Bu proje Render üzerinde sorunsuz çalışacak şekilde Jules (AI) tarafından optimize edilmiştir.*
