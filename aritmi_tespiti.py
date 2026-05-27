"""
============================================================
  KALP RİTMİ EKSİKLİK VE ARİTMİ TESPİT SİSTEMİ
  Bitirme Tezi Projesi
  Not: Klinik tanı amacı taşımaz; algoritma çıktısı ön değerlendirmedir.
  
  Kullanılan Yöntem:
    - Sentetik EKG sinyali üretimi (gerçekçi PQRST morfolojisi)
    - Pan-Tompkins tabanlı R-peak tespiti
    - RR aralığı analizi ile ritim bozukluğu ön değerlendirmesi
    - Eksik/erken atım adayı tespiti
    - Gürültülü sinyalde güvenilirlik uyarısı
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# ─── Türkçe font ayarı ───────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'


# ══════════════════════════════════════════════════════════
#  1. EKG SİNYALİ ÜRETİCİ
# ══════════════════════════════════════════════════════════

def gaussian(t, mu, sigma, amp):
    """Gaussian tepe üretici — PQRST dalgaları için"""
    return amp * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))

def tek_atim_olustur(fs: int = 500, kalp_hizi: float = 70.0,
                     amplitud: float = 1.0, tip: str = "normal") -> np.ndarray:
    """
    Tek bir kalp atımı (PQRST kompleksi) üretir.
    
    Parametreler:
        fs       : Örnekleme frekansı (Hz)
        kalp_hizi: Dakikadaki atım sayısı
        amplitud : QRS genliği çarpanı
        tip      : 'normal', 'pvc', 'pac'
    """
    rr_suresi = 60.0 / kalp_hizi          # saniye cinsinden RR
    n = int(rr_suresi * fs)
    t = np.linspace(0, rr_suresi, n)
    
    signal = np.zeros(n)
    
    if tip == "normal":
        # P dalgası
        signal += gaussian(t, 0.16, 0.025, 0.15 * amplitud)
        # Q dalgası
        signal += gaussian(t, 0.26, 0.010, -0.10 * amplitud)
        # R dalgası (ana tepe)
        signal += gaussian(t, 0.28, 0.014, 1.00 * amplitud)
        # S dalgası
        signal += gaussian(t, 0.30, 0.010, -0.15 * amplitud)
        # T dalgası
        signal += gaussian(t, 0.42, 0.040, 0.25 * amplitud)
        
    elif tip == "pvc":
        # P dalgası yok, geniş ve garip QRS
        signal += gaussian(t, 0.28, 0.030, -0.35 * amplitud)
        signal += gaussian(t, 0.32, 0.015, 1.30 * amplitud)
        signal += gaussian(t, 0.38, 0.025, -0.40 * amplitud)
        # Ters T dalgası
        signal += gaussian(t, 0.55, 0.050, -0.30 * amplitud)
        
    elif tip == "pac":
        # Erken P dalgası, dar QRS
        signal += gaussian(t, 0.10, 0.020, 0.12 * amplitud)
        signal += gaussian(t, 0.19, 0.012, 1.00 * amplitud)
        signal += gaussian(t, 0.22, 0.035, 0.20 * amplitud)
    
    return signal


def ekg_sinyali_uret(fs: int = 500, sure: float = 10.0,
                     senaryo: str = "normal") -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Belirtilen senaryoya göre tam EKG sinyali üretir.
    
    Senaryolar:
        'normal'    : Düzenli sinüs ritmi (70 atım/dk)
        'bradikardi': Yavaş ritim (45 atım/dk)
        'takikardi' : Hızlı ritim (120 atım/dk)
        'pvc'       : Erken ventriküler kasılma
        'pac'       : Erken atriyal kasılma
        'eksik_atim': Dropped beat (2. dereceli blok)
        'afib'      : Atriyal fibrilasyon (düzensiz ritim)
    """
    n_toplam = int(sure * fs)
    signal = np.zeros(n_toplam)
    t = np.linspace(0, sure, n_toplam)
    gercek_atimlar = []     # (örnek indeksi, atım tipi)
    
    def atim_ekle(baslangic_idx, tip, hz):
        """Sinyale tek atım ekler, sığmazsa keser"""
        atim = tek_atim_olustur(fs=fs, kalp_hizi=hz, tip=tip)
        bitis = baslangic_idx + len(atim)
        if bitis > n_toplam:
            bitis = n_toplam
            atim = atim[:bitis - baslangic_idx]
        signal[baslangic_idx:bitis] += atim
        # R-peak konumu (atımın ~%28'i)
        r_orn = baslangic_idx + int(0.28 * len(atim))
        gercek_atimlar.append((r_orn, tip))
    
    if senaryo == "normal":
        hz = 70
        rr_ornekler = int(60.0 / hz * fs)
        idx = int(0.1 * fs)
        while idx + rr_ornekler < n_toplam:
            atim_ekle(idx, "normal", hz)
            idx += rr_ornekler + int(np.random.normal(0, 5))
    
    elif senaryo == "bradikardi":
        hz = 42
        rr_ornekler = int(60.0 / hz * fs)
        idx = int(0.1 * fs)
        while idx + rr_ornekler < n_toplam:
            atim_ekle(idx, "normal", hz)
            idx += rr_ornekler + int(np.random.normal(0, 8))
    
    elif senaryo == "takikardi":
        hz = 130
        rr_ornekler = int(60.0 / hz * fs)
        idx = int(0.05 * fs)
        while idx + rr_ornekler < n_toplam:
            atim_ekle(idx, "normal", hz)
            idx += rr_ornekler + int(np.random.normal(0, 4))
    
    elif senaryo == "pvc":
        hz = 72
        rr_ornekler = int(60.0 / hz * fs)
        idx = int(0.1 * fs)
        sayac = 0
        while idx + rr_ornekler < n_toplam:
            if sayac in [4, 9]:          # 5. ve 10. atım PVC olacak
                # Normal atım ekle
                atim_ekle(idx, "normal", hz)
                idx += rr_ornekler + int(np.random.normal(0, 5))
                sayac += 1
                
                # Erken PVC atım ekle (kısa RR)
                if idx + int(rr_ornekler * 0.65) < n_toplam:
                    atim_ekle(idx, "pvc", hz)
                    idx += int(rr_ornekler * 0.65)  # erken atım (kısa RR)
                    sayac += 1
                    
                    # Kompansatuar duraklama (uzun RR)
                    if idx + int(rr_ornekler * 1.4) < n_toplam:
                        atim_ekle(idx, "normal", hz)
                        idx += int(rr_ornekler * 1.4)  # uzun RR
                        sayac += 1
            else:
                atim_ekle(idx, "normal", hz)
                idx += rr_ornekler + int(np.random.normal(0, 5))
                sayac += 1
    
    elif senaryo == "pac":
        hz = 75
        rr_ornekler = int(60.0 / hz * fs)
        idx = int(0.1 * fs)
        sayac = 0
        while idx + rr_ornekler < n_toplam:
            if sayac in [3, 7, 11]:      # PAC olan atımlar
                atim_ekle(idx, "pac", hz)
                idx += int(rr_ornekler * 0.70)
            else:
                atim_ekle(idx, "normal", hz)
                idx += rr_ornekler + int(np.random.normal(0, 5))
            sayac += 1
    
    elif senaryo == "eksik_atim":
        hz = 68
        rr_ornekler = int(60.0 / hz * fs)
        idx = int(0.1 * fs)
        sayac = 0
        while idx + rr_ornekler < n_toplam:
            if sayac in [5, 11]:         # bu pozisyonlarda atım eksik
                gercek_atimlar.append((idx + int(rr_ornekler * 0.28), "eksik"))
                idx += int(rr_ornekler * 2.0)   # çift RR aralığı geçer
            else:
                atim_ekle(idx, "normal", hz)
                idx += rr_ornekler + int(np.random.normal(0, 6))
            sayac += 1
    
    elif senaryo == "afib":
        # Afib: tamamen düzensiz RR, P dalgası yok, ince titreme
        baz_hz = 110
        idx = int(0.05 * fs)
        while idx < n_toplam - int(0.8 * fs):
            # Rastgele RR aralığı (250–900 ms arası)
            rr_ms = np.random.uniform(250, 850)
            rr_ornekler_lokal = int(rr_ms / 1000 * fs)
            hz_lokal = 60000 / rr_ms
            atim_ekle(idx, "normal", min(hz_lokal, 200))
            idx += rr_ornekler_lokal
        # Fibrilasyon dalgacıkları (ince titreme baseline)
        signal += 0.04 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
        signal += 0.03 * np.random.randn(n_toplam)
    
    # Baseline gürültü ekle (gerçekçilik için)
    signal += 0.02 * np.random.randn(n_toplam)
    signal += 0.015 * np.sin(2 * np.pi * 0.3 * t)  # solunum hareketi
    
    return t, signal, gercek_atimlar


# ══════════════════════════════════════════════════════════
#  2. SİNYAL İŞLEME — FİLTRELEME
# ══════════════════════════════════════════════════════════

def bant_geciren_filtre(signal: np.ndarray, fs: int,
                        dusuk: float = 0.5, yuksek: float = 40.0) -> np.ndarray:
    """
    Butterworth bant geçiren filtre.
    EKG için standart: 0.5 – 40 Hz
    """
    nyq = fs / 2.0
    b, a = butter(N=4, Wn=[dusuk/nyq, yuksek/nyq], btype='band')
    return filtfilt(b, a, signal)

def turev_al(signal: np.ndarray, fs: int) -> np.ndarray:
    """Pan-Tompkins türev filtresi"""
    h = np.array([-1, -2, 0, 2, 1]) / 8.0
    return np.convolve(signal, h, mode='same')

def kareye_al(signal: np.ndarray) -> np.ndarray:
    return signal ** 2

def hareketli_ortalama(signal: np.ndarray, pencere: int) -> np.ndarray:
    return np.convolve(signal, np.ones(pencere)/pencere, mode='same')


# ══════════════════════════════════════════════════════════
#  3. R-PEAK TESPİTİ (Pan-Tompkins)
# ══════════════════════════════════════════════════════════

def r_peak_tespit(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Pan-Tompkins algoritması ile R-peak tespiti (adaptif eşik).
    
    Adımlar:
        1. Bant geçiren filtreleme
        2. Türev
        3. Kareye alma
        4. Hareketli ortalama (integral)
        5. Kayan pencere ile adaptif eşik tespiti (uzun kayıtlar için)
    """
    # 1. Filtreleme
    filtrelenmis = bant_geciren_filtre(signal, fs)
    
    # 2. Türev
    turev = turev_al(filtrelenmis, fs)
    
    # 3. Kareye alma
    kare = kareye_al(turev)
    
    # 4. Hareketli ortalama (150 ms pencere)
    pencere = int(0.150 * fs)
    integral = hareketli_ortalama(kare, pencere)
    
    # 5. ADAPTIF Tepe tespiti (kayan pencere ile)
    min_mesafe = int(0.30 * fs)    # en az 300 ms iki R-peak arası
    pencere_suresi = 60  # Her 60 saniyede bir eşik güncelle
    pencere_ornekler = int(pencere_suresi * fs)
    
    # Sinyal çok kısa ise (< 60 sn), eski yöntemi kullan
    if len(integral) < pencere_ornekler:
        esik = np.median(integral) + (np.max(integral) * 0.12)
        tepeler, _ = find_peaks(integral, height=esik, distance=min_mesafe)
    else:
        # Uzun kayıtlar için: her 60 saniyelik pencerede eşik hesapla
        tum_tepeler = []
        for baslangic in range(0, len(integral), pencere_ornekler):
            bitis = min(baslangic + pencere_ornekler, len(integral))
            pencere_veri = integral[baslangic:bitis]
            
            # Bu pencere için eşik hesapla
            esik_lokal = np.median(pencere_veri) + (np.max(pencere_veri) * 0.12)
            
            # Eğer eşik çok düşükse (gürültülü bölge), atla
            if esik_lokal < np.max(pencere_veri) * 0.05:
                continue
            
            # Bu pencerede tepe bul
            tepeler_lokal, _ = find_peaks(pencere_veri, height=esik_lokal, distance=min_mesafe)
            
            # Global indekse çevir
            tepeler_global = tepeler_lokal + baslangic
            tum_tepeler.extend(tepeler_global)
        
        tepeler = np.array(tum_tepeler, dtype=int)
    
    # Her tepeyi orijinal sinyalde hassaslaştır
    hassas_tepeler = []
    pencere_hassas = int(0.040 * fs)
    for p in tepeler:
        baslangic = max(0, p - pencere_hassas)
        bitis = min(len(signal), p + pencere_hassas)
        lokal_max = np.argmax(np.abs(filtrelenmis[baslangic:bitis]))
        hassas_tepeler.append(baslangic + lokal_max)
    
    return np.array(hassas_tepeler, dtype=int)


# ══════════════════════════════════════════════════════════
#  4. RR ARALIĞI ANALİZİ
# ══════════════════════════════════════════════════════════
@dataclass
class AtimAnalizi:
    indeks: int
    rr_ms: float
    tip: str           # 'normal', 'erken', 'geç', 'eksik', 'pvc'
    sapma_yuzde: float
def rr_analiz(r_peaks: np.ndarray, fs: int) -> List[AtimAnalizi]:
    """
    RR aralıklarını hesaplar ve her atımı sınıflandırır.
    (Kompansatuar duraklama kontrolü eklendi)
    """
    if len(r_peaks) < 2:
        return []
    
    rr_ornekler = np.diff(r_peaks)
    rr_ms = (rr_ornekler / fs) * 1000.0
    medyan_rr = np.median(rr_ms)
    
    sonuclar = []
    for i, rr in enumerate(rr_ms):
        oran = rr / medyan_rr
        sapma = ((rr - medyan_rr) / medyan_rr) * 100
        
        # 1. DEĞİŞİKLİK: 1.80'den büyük olsa bile, bir önceki atım "erken" mi diye bakıyoruz.
        if oran >= 1.80:
            if i > 0 and "erken" in sonuclar[-1].tip:
                tip = "geç / kompansatuar"  # PVC sonrası normal duraklama
            else:
                tip = "eksik"               # Gerçek AV Blok / İletim hatası
        elif oran <= 0.75:
            tip = "erken (PAC/PVC)"
        elif oran >= 1.15:
            tip = "geç / kompansatuar"
        else:
            tip = "normal"
        
        sonuclar.append(AtimAnalizi(
            indeks=i,
            rr_ms=round(rr, 1),
            tip=tip,
            sapma_yuzde=round(sapma, 1)
        ))
    
    return sonuclar

def kalp_hizi_hesapla(r_peaks: np.ndarray, fs: int) -> float:
    if len(r_peaks) < 2:
        return 0.0
    rr_ms = np.diff(r_peaks) / fs * 1000.0
    return round(60000.0 / np.median(rr_ms), 1)

def duzensizlik_skoru(rr_analiz_sonuclari: List[AtimAnalizi]) -> float:
    """RMSSD tabanlı düzensizlik skoru (%)"""
    if len(rr_analiz_sonuclari) < 2:
        return 0.0
    rr_deger = [a.rr_ms for a in rr_analiz_sonuclari]
    farklar = np.diff(rr_deger)
    rmssd = np.sqrt(np.mean(farklar**2))
    medyan_rr = np.median(rr_deger)
    return round((rmssd / medyan_rr) * 100, 1)


def _mad(x: np.ndarray) -> float:
    """Robust gürültü tahmini için median absolute deviation."""
    x = np.asarray(x)
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def sinyal_kalitesi_degerlendir(signal: np.ndarray, fs: int, r_peaks: np.ndarray) -> dict:
    """
    Basit sinyal kalitesi kontrolü.

    Amaç:
        Gürültülü/artefaktlı kayıtlar doğrudan 'tanı' gibi sunulmasın.
        R-peak tespit hatası olabilecek durumlarda rapora uyarı eklensin.

    Not:
        Bu bölüm klinik bir sinyal kalite indeksi değildir; bitirme tezi kapsamı için
        R-peak/RR analiz güvenilirliğini kabaca işaretleyen koruyucu bir katmandır.
    """
    uyarilar = []
    score = 100.0

    if signal is None or len(signal) == 0 or not np.all(np.isfinite(signal)):
        return {
            "skor": 0.0,
            "durum": "düşük",
            "guvenilir": False,
            "uyarilar": ["Sinyal boş veya sayısal olarak geçersiz."],
            "snr_benzeri": 0.0,
            "hf_oran": 1.0,
            "rr_cv": 0.0
        }

    # Filtrelenmiş sinyal üstünden R-tepe genliği / gürültü tabanı oranı
    try:
        filtrelenmis = bant_geciren_filtre(signal, fs)
    except Exception:
        filtrelenmis = signal.copy()
        uyarilar.append("Bant geçiren filtre uygulanamadı; kalite hesabı ham sinyal ile yapıldı.")
        score -= 10

    if len(r_peaks) >= 3:
        valid_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(filtrelenmis))]
        qrs_amp = float(np.median(np.abs(filtrelenmis[valid_peaks]))) if len(valid_peaks) else 0.0
        noise_est = 1.4826 * _mad(filtrelenmis)
        snr_benzeri = qrs_amp / (noise_est + 1e-12)
    else:
        snr_benzeri = 0.0
        uyarilar.append("R-peak sayısı çok az; RR analizi güvenilir olmayabilir.")
        score -= 25

    if snr_benzeri < 4.0:
        uyarilar.append("R-tepe/gürültü oranı düşük; sahte veya kaçan R-peak olabilir.")
        score -= 30
    elif snr_benzeri < 6.0:
        uyarilar.append("R-tepe/gürültü oranı orta seviyede; sonuç dikkatli yorumlanmalıdır.")
        score -= 12

    # Yüksek frekanslı gürültü oranı için kaba gösterge
    try:
        nyq = fs / 2.0
        cutoff = min(20.0 / nyq, 0.95)
        b, a = butter(N=3, Wn=cutoff, btype='low')
        low = filtfilt(b, a, signal)
        hf_residual = signal - low
        hf_oran = float(np.std(hf_residual) / (np.std(signal) + 1e-12))
    except Exception:
        hf_oran = 0.0

    if hf_oran > 0.45:
        uyarilar.append("Yüksek frekanslı gürültü oranı yüksek; sınıflandırma güvenilirliği azalır.")
        score -= 25
    elif hf_oran > 0.35:
        uyarilar.append("Yüksek frekanslı gürültü oranı orta-yüksek seviyede.")
        score -= 10

    # R-peak sayısı ve RR tutarlılığı
    rr_cv = 0.0
    if len(r_peaks) >= 2:
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        med_rr = np.median(rr_ms)
        rr_cv = float(np.std(rr_ms) / (med_rr + 1e-12))
        hr = 60000.0 / med_rr if med_rr > 0 else 0.0

        if hr < 30 or hr > 220:
            uyarilar.append(f"Kalp hızı fizyolojik sınırların dışında görünüyor ({hr:.1f} atım/dk); R-peak hatası olabilir.")
            score -= 20

        if np.min(rr_ms) < 240:
            uyarilar.append("Çok kısa RR aralığı tespit edildi; gürültü kaynaklı sahte R-peak olasılığı vardır.")
            score -= 15

        # Bu tek başına hata değildir; AF/aritmi de yapabilir. Bu yüzden sadece uyarı ve küçük ceza.
        if rr_cv > 0.45:
            uyarilar.append("RR aralıkları çok düzensiz; bu gerçek aritmi veya R-peak tespit hatası olabilir.")
            score -= 5
    else:
        uyarilar.append("Yeterli RR aralığı oluşturulamadı.")
        score -= 25

    score = max(0.0, min(100.0, score))
    if score >= 75:
        durum = "iyi"
    elif score >= 60:
        durum = "orta"
    else:
        durum = "düşük"

    if not uyarilar:
        uyarilar = ["Sinyal kalitesi R-peak/RR analizi için kabul edilebilir görünüyor."]

    return {
        "skor": round(score, 1),
        "durum": durum,
        "guvenilir": score >= 60,
        "uyarilar": uyarilar,
        "snr_benzeri": round(float(snr_benzeri), 2),
        "hf_oran": round(float(hf_oran), 3),
        "rr_cv": round(float(rr_cv), 3)
    }


# ══════════════════════════════════════════════════════════
#  5. ARİTMİ SINIFLANDIRICI (YÜZDESEL MANTIKLA GÜNCELLENDİ)
# ══════════════════════════════════════════════════════════

def aritmi_siniflandir(hr: float, analiz: List[AtimAnalizi],
                       duzensizlik: float, senaryo: str,
                       kalite_raporu: Optional[dict] = None) -> dict:
    """
    RR aralığına dayalı kural tabanlı ritim ön değerlendirmesi.

    Önemli not:
        Bu fonksiyon klinik tanı koymaz. Çıktı, R-peak/RR analizine dayalı
        'algoritma yorumu' veya 'ön değerlendirme' olarak raporlanmalıdır.
    """
    bulgular = []
    toplam_atim = len(analiz)
    klinik_uyari = "Bu çıktı klinik tanı değildir; algoritma yorumu/ön değerlendirme niteliğindedir."

    # Sıfıra bölünme hatasını önleme
    if toplam_atim == 0:
        return {
            "tani": "Analiz Edilemedi",
            "ciddiyet": "dikkat",
            "bulgular": ["Yeterli RR aralığı bulunamadı."],
            "oneri": "Sinyal ve R-peak tespiti kontrol edilmelidir.",
            "eksik_atim": 0,
            "erken_atim": 0,
            "kalite": kalite_raporu,
            "klinik_uyari": klinik_uyari
        }

    eksik_sayisi = sum(1 for a in analiz if a.tip == "eksik")
    erken_sayisi = sum(1 for a in analiz if "erken" in a.tip)
    gec_sayisi   = sum(1 for a in analiz if "geç" in a.tip or "kompansatuar" in a.tip)

    eksik_oran = eksik_sayisi / toplam_atim
    erken_oran = erken_sayisi / toplam_atim

    if eksik_sayisi > 0:
        bulgular.append(f"{eksik_sayisi} uzun RR/eksik atım adayı tespit edildi (Toplamın %{eksik_oran*100:.1f}'i)")
    if erken_sayisi > 0:
        bulgular.append(f"{erken_sayisi} erken atım adayı saptandı (Toplamın %{erken_oran*100:.1f}'i)")

    # Gürültülü/kalitesiz sinyalde kesin ritim yorumu verme
    if kalite_raporu is not None and not kalite_raporu.get("guvenilir", True):
        kalite_bulgular = [
            f"Sinyal kalitesi düşük/şüpheli (skor: {kalite_raporu.get('skor', 0)}/100).",
            "Gürültü veya artefakt R-peak tespitini bozabileceğinden RR tabanlı ritim yorumu güvenilir değildir."
        ]
        kalite_bulgular.extend(kalite_raporu.get("uyarilar", [])[:3])
        return {
            "tani": "Sinyal Kalitesi Düşük — Ritim Yorumu Güvenilir Değil",
            "ciddiyet": "dikkat",
            "bulgular": kalite_bulgular,
            "oneri": "Daha temiz kayıt, ek filtreleme veya uzman kontrolü ile yeniden değerlendirme önerilir.",
            "eksik_atim": eksik_sayisi,
            "erken_atim": erken_sayisi,
            "kalite": kalite_raporu,
            "klinik_uyari": klinik_uyari
        }

    # ─── ÖN DEĞERLENDİRME HİYERARŞİSİ ───
    # 1. Uzun RR / eksik atım adayı
    if eksik_oran > 0.02:
        tani = "Uzun RR / Eksik Atım Adayı"
        ciddiyet = "kritik"

    # 2. Yüksek RR düzensizliği: AF kesin tanısı değil, AF adayı/düzensiz ritim yorumu
    elif duzensizlik > 20 and erken_oran < 0.10:
        tani = "Yüksek RR Düzensizliği / Muhtemel AF Adayı"
        ciddiyet = "kritik"
        bulgular.append("RR düzensizliği yüksek; AF benzeri ritim olasılığı ancak P dalgası/morfoloji ile doğrulanmalıdır.")

    # 3. Yoğun erken atım adayı
    elif erken_oran >= 0.05:
        tani = "Yoğun Erken Atım Şüphesi"
        ciddiyet = "kritik"
        bulgular.append("PVC/PAC ayrımı yalnızca RR aralığı ile kesin yapılamaz; QRS morfolojisi gerekir.")

    # 4. İzole erken atımlar
    elif erken_oran >= 0.01:
        tani = "İzole Erken Atım Adayı"
        ciddiyet = "dikkat"
        bulgular.append("Erken atım saptandı; PVC/PAC ayrımı bu sürümde kesin yapılmamaktadır.")

    # 5. Kalp hızı tabanlı yorumlar veya normal ritim adayı
    else:
        if hr < 60:
            tani = "Şiddetli Bradikardi Adayı" if hr < 40 else "Bradikardi Adayı"
            ciddiyet = "kritik" if hr < 40 else "dikkat"
        elif hr > 100:
            tani = "Şiddetli Taşikardi Adayı" if hr > 150 else "Taşikardi Adayı"
            ciddiyet = "kritik" if hr > 150 else "dikkat"
        else:
            tani = "Normal Ritim Adayı"
            ciddiyet = "normal"

    oneri_map = {
        "normal"   : "Sinyal kalitesi uygunsa rutin takip yeterli kabul edilebilir.",
        "dikkat"   : "Sonuç ön değerlendirmedir; referans anotasyon/uzman yorumu ile doğrulanmalıdır.",
        "kritik"   : "Belirgin ritim düzensizliği adayı saptandı; uzman değerlendirmesi önerilir."
    }

    return {
        "tani": tani,
        "ciddiyet": ciddiyet,
        "bulgular": bulgular if bulgular else ["Belirgin anormal RR paterni saptanmadı."],
        "oneri": oneri_map[ciddiyet],
        "eksik_atim": eksik_sayisi,
        "erken_atim": erken_sayisi,
        "kalite": kalite_raporu,
        "klinik_uyari": klinik_uyari
    }


# ══════════════════════════════════════════════════════════
#  6. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════

RENKLER = {
    "normal"          : "#1D9E75",
    "erken (PAC/PVC)" : "#BA7517",
    "geç / kompansatuar": "#185FA5",
    "eksik"           : "#A32D2D"
}

CİDDİYET_RENKLER = {
    "normal" : "#1D9E75",
    "dikkat" : "#BA7517",
    "kritik" : "#A32D2D"
}

def gorsellestir(t, signal, r_peaks, analiz_sonuclari,
                 tani_sonucu, senaryo, fs=500):
    """Analiz sonuçlarını 4 panelde görselleştirir"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#F8F8F6')
    
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.06)
    
    senaryo_etiketleri = {
        "normal"    : "Normal Sinüs Ritmi",
        "bradikardi": "Bradikardi",
        "takikardi" : "Taşikardi",
        "pvc"       : "PVC (Erken Ventriküler Kasılma)",
        "pac"       : "PAC (Erken Atriyal Kasılma)",
        "eksik_atim": "Eksik Atım / 2. Derece AV Blok",
        "afib"      : "Atriyal Fibrilasyon"
    }
    
    fig.suptitle(
        f"Kalp Ritmi Aritmi Tespit Sistemi  —  {senaryo_etiketleri.get(senaryo, senaryo)}",
        fontsize=15, fontweight='bold', color='#2C2C2A', y=0.95
    )
    
    # ── Panel 1: EKG Sinyali ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#FAFAF8')
    ax1.plot(t, signal, color='#2C2C2A', linewidth=0.8, alpha=0.9, label='EKG Sinyali')
    
    # R-peak işaretle
    if len(r_peaks) > 0 and len(analiz_sonuclari) > 0:
        for i, r in enumerate(r_peaks[:-1]):
            if i < len(analiz_sonuclari):
                tip = analiz_sonuclari[i].tip
                renk = RENKLER.get(tip, "#1D9E75")
                ax1.axvline(t[r], color=renk, alpha=0.4, linewidth=1.0, linestyle='--')
                ax1.plot(t[r], signal[r], 'o',
                         color=renk, markersize=5, zorder=5)
        
        # Son R-peak
        ax1.plot(t[r_peaks[-1]], signal[r_peaks[-1]], 'o',
                 color=RENKLER["normal"], markersize=5, zorder=5)
    
    ax1.set_xlabel("Zaman (s)", fontsize=10)
    ax1.set_ylabel("Genlik (mV)", fontsize=10)
    ax1.set_title("EKG Sinyali — R-Peak Tespiti", fontsize=11, pad=6)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.set_xlim([t[0], t[-1]])
    
    # Renk açıklaması
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RENKLER["normal"],          markersize=7, label='Normal'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RENKLER["erken (PAC/PVC)"], markersize=7, label='Erken (PAC/PVC)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RENKLER["geç / kompansatuar"], markersize=7, label='Geç / Kompansatuar'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RENKLER["eksik"],           markersize=7, label='Eksik Atım'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', fontsize=8, framealpha=0.8)
    
    # ── Panel 2: RR Aralıkları ────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#FAFAF8')
    
    if analiz_sonuclari:
        rr_degerler = [a.rr_ms for a in analiz_sonuclari]
        renkler_bar = [RENKLER.get(a.tip, RENKLER["normal"]) for a in analiz_sonuclari]
        medyan_rr = np.median(rr_degerler)
        
        x_pos = np.arange(len(rr_degerler))
        ax2.bar(x_pos, rr_degerler, color=renkler_bar, alpha=0.8, width=0.7, edgecolor='white', linewidth=0.5)
        ax2.axhline(medyan_rr, color='#E24B4A', linewidth=1.5, linestyle='--',
                    label=f'Medyan RR: {medyan_rr:.0f} ms')
        ax2.axhline(medyan_rr * 1.80, color='#A32D2D', linewidth=1.0, linestyle=':',
                    alpha=0.7, label='Eksik atım eşiği (×1.80)')
        ax2.axhline(medyan_rr * 0.75, color='#BA7517', linewidth=1.0, linestyle=':',
                    alpha=0.7, label='Erken atım eşiği (×0.75)')
        
        # Eksik atım etiketi
        for a in analiz_sonuclari:
            if a.tip == "eksik":
                ax2.text(a.indeks, rr_degerler[a.indeks] + 20, "EKSİK!",
                         ha='center', va='bottom', fontsize=7, color='#A32D2D', fontweight='bold')
        
        ax2.set_xlabel("Atım Numarası", fontsize=10)
        ax2.set_ylabel("RR Aralığı (ms)", fontsize=10)
        ax2.set_title("RR Aralığı Analizi", fontsize=11, pad=6)
        ax2.legend(fontsize=7, loc='upper right')
        ax2.grid(True, axis='y', alpha=0.25, linewidth=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([str(i+1) for i in x_pos], fontsize=8)
    
    # ── Panel 3: Sapma Grafiği ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#FAFAF8')
    
    if analiz_sonuclari:
        sapmalar = [a.sapma_yuzde for a in analiz_sonuclari]
        renkler_sap = [RENKLER.get(a.tip, RENKLER["normal"]) for a in analiz_sonuclari]
        x_pos = np.arange(len(sapmalar))
        
        ax3.bar(x_pos, sapmalar, color=renkler_sap, alpha=0.8, width=0.7,
                edgecolor='white', linewidth=0.5)
        ax3.axhline(0,    color='#888780', linewidth=1.0, linestyle='-')
        ax3.axhline(80,   color='#A32D2D', linewidth=1.0, linestyle=':', alpha=0.7)
        ax3.axhline(-25,  color='#BA7517', linewidth=1.0, linestyle=':', alpha=0.7)
        
        ax3.set_xlabel("Atım Numarası", fontsize=10)
        ax3.set_ylabel("Medyandan Sapma (%)", fontsize=10)
        ax3.set_title("RR Sapma Analizi", fontsize=11, pad=6)
        ax3.grid(True, axis='y', alpha=0.25, linewidth=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(i+1) for i in x_pos], fontsize=8)
    
    # ── Panel 4: Ön Değerlendirme Raporu ────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#FAFAF8')
    ax4.axis('off')
    
    if analiz_sonuclari:
        hr = kalp_hizi_hesapla(r_peaks, fs)
        duz = duzensizlik_skoru(analiz_sonuclari)
        tani = tani_sonucu
        
        cid_renk = CİDDİYET_RENKLER[tani["ciddiyet"]]
        
        rapor_metni = (
            f"ALGORİTMA YORUMU: {tani['tani']}    |    "
            f"Kalp Hızı: {hr} atım/dk    |    "
            f"Eksik Atım: {tani['eksik_atim']}    |    "
            f"Erken Atım: {tani['erken_atim']}    |    "
            f"Düzensizlik: %{duz}"
        )
        
        ax4.text(0.5, 0.86, rapor_metni, transform=ax4.transAxes,
                 ha='center', va='top', fontsize=9.5, color=cid_renk, fontweight='bold')
        
        kalite = tani.get("kalite") or {}
        kalite_metni = "Sinyal Kalitesi: " + str(kalite.get("durum", "belirtilmedi")).upper()
        if "skor" in kalite:
            kalite_metni += f"  (Skor: {kalite.get('skor')}/100)"
        ax4.text(0.5, 0.66, kalite_metni, transform=ax4.transAxes,
                 ha='center', va='top', fontsize=9, color='#2C2C2A', fontweight='bold')

        bulgular_metni = "Bulgular: " + "  |  ".join(tani["bulgular"][:3])
        ax4.text(0.5, 0.48, bulgular_metni, transform=ax4.transAxes,
                 ha='center', va='top', fontsize=8.5, color='#2C2C2A', wrap=True)

        ax4.text(0.5, 0.28, f"Öneri: {tani['oneri']}",
                 transform=ax4.transAxes, ha='center', va='top',
                 fontsize=8.5, color=cid_renk, style='italic')

        ax4.text(0.5, 0.13, tani.get("klinik_uyari", "Bu çıktı klinik tanı değildir."),
                 transform=ax4.transAxes, ha='center', va='top',
                 fontsize=8, color='#555555', style='italic')
        
        ciddiyet_kutu_renk = {'normal': '#E1F5EE', 'dikkat': '#FAEEDA', 'kritik': '#FCEBEB'}
        kutu = plt.Rectangle((0.01, 0.05), 0.98, 0.93,
                              transform=ax4.transAxes, fill=True,
                              facecolor=ciddiyet_kutu_renk[tani["ciddiyet"]],
                              edgecolor=cid_renk, linewidth=1.5,
                              clip_on=False, zorder=-1)
        ax4.add_patch(kutu)
        
        ax4.set_title("Otomatik Ön Değerlendirme Raporu", fontsize=11, pad=6)
    
    plt.savefig("aritmi_analiz_raporu.png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("  → Görsel kaydedildi: aritmi_analiz_raporu.png")


# ══════════════════════════════════════════════════════════
#  7. KONSOL RAPORU
# ══════════════════════════════════════════════════════════

def konsol_raporu(r_peaks, analiz_sonuclari, tani_sonucu, fs):
    """Terminal üzerinde detaylı metin raporu yazar"""
    hr = kalp_hizi_hesapla(r_peaks, fs)
    duz = duzensizlik_skoru(analiz_sonuclari)
    
    sep = "═" * 60
    print(f"\n{sep}")
    print("  KALP RİTMİ ANALİZ RAPORU")
    print(sep)
    print(f"  Tespit edilen R-peak sayısı : {len(r_peaks)}")
    print(f"  Analiz edilen RR aralığı    : {len(analiz_sonuclari)}")
    print(f"  Kalp hızı (ort.)            : {hr} atım/dk")
    print(f"  Düzensizlik skoru (RMSSD)   : %{duz}")
    print(f"\n  {'Atım':<6} {'RR (ms)':<10} {'Tip':<25} {'Sapma'}")
    print(f"  {'-'*55}")
    
    for a in analiz_sonuclari:
        isaretler = {
            "eksik"           : "  ⚠ EKSİK ATIM",
            "erken (PAC/PVC)" : "  ◆ erken",
            "geç / kompansatuar": "  ◇ geç",
            "normal"          : ""
        }
        isaret = isaretler.get(a.tip, "")
        print(f"  {a.indeks+1:<6} {a.rr_ms:<10.1f} {a.tip:<25} %{a.sapma_yuzde:+.1f}{isaret}")
    
    print(f"\n{sep}")
    print(f"  ALGORİTMA YORUMU : {tani_sonucu['tani']}")
    print(f"  RAPOR SEVİYESİ   : {tani_sonucu['ciddiyet'].upper()}")
    kalite = tani_sonucu.get("kalite")
    if kalite:
        print(f"  SİNYAL KALİTESİ  : {kalite.get('durum', 'belirsiz').upper()} | Skor: {kalite.get('skor', '-')}/100")
        print(f"  Kalite göstergeleri: SNR-benzeri={kalite.get('snr_benzeri', '-')}, HF oranı={kalite.get('hf_oran', '-')}, RR-CV={kalite.get('rr_cv', '-')}")
        print("  Kalite uyarıları:")
        for u in kalite.get("uyarilar", []):
            print(f"    - {u}")
    print(f"\n  Bulgular:")
    for b in tani_sonucu["bulgular"]:
        print(f"    • {b}")
    print(f"\n  Öneri: {tani_sonucu['oneri']}")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════
#  8. ANA PROGRAM
# ══════════════════════════════════════════════════════════

def analiz_et(senaryo: str = "eksik_atim", fs: int = 500, sure: float = 12.0):
    """
    Belirtilen senaryo için tam analiz döngüsünü çalıştırır.
    
    Parametre:
        senaryo : 'normal', 'bradikardi', 'takikardi',
                  'pvc', 'pac', 'eksik_atim', 'afib'
        fs      : Örnekleme frekansı (Hz)
        sure    : Kayıt süresi (saniye)
    """
    print(f"\n  Senaryo: {senaryo.upper()} | fs={fs} Hz | Süre={sure}s")
    print("  EKG sinyali üretiliyor...", end="", flush=True)
    
    t, signal, _ = ekg_sinyali_uret(fs=fs, sure=sure, senaryo=senaryo)
    print(" ✓")
    
    print("  R-peak tespiti (Pan-Tompkins)...", end="", flush=True)
    r_peaks = r_peak_tespit(signal, fs)
    print(f" ✓  ({len(r_peaks)} tepe)")
    
    print("  RR aralığı analizi...", end="", flush=True)
    analiz = rr_analiz(r_peaks, fs)
    print(" ✓")

    print("  Sinyal kalitesi kontrol ediliyor...", end="", flush=True)
    kalite = sinyal_kalitesi_degerlendir(signal, fs, r_peaks)
    print(f" ✓  ({kalite['durum']} | skor={kalite['skor']}/100)")

    hr   = kalp_hizi_hesapla(r_peaks, fs)
    duz  = duzensizlik_skoru(analiz)
    tani = aritmi_siniflandir(hr, analiz, duz, senaryo, kalite)
    
    konsol_raporu(r_peaks, analiz, tani, fs)
    
    print("  Grafik oluşturuluyor...")
    gorsellestir(t, signal, r_peaks, analiz, tani, senaryo, fs)
    
    return r_peaks, analiz, tani


# ── Çalıştır ──────────────────────────────────────────────
if __name__ == "__main__":
    # İstediğin senaryoyu buradan değiştir:
    # 'normal' | 'bradikardi' | 'takikardi' | 'pvc' | 'pac' | 'eksik_atim' | 'afib'
    SENARYO = "pac"
    
    analiz_et(senaryo=SENARYO, fs=500, sure=12.0)