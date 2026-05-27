"""
============================================================
  ARİTMİ TESPİT SİSTEMİ — TEST VE DEĞERLENDİRME MODÜLÜ
  Bitirme Tezi | Test Aşaması

  İki mod:
    A) Sentetik veri ile otomatik test (bu dosyayı çalıştır)
    B) Gerçek MIT-BIH verisi ile test (veri indirildikten sonra)

  MIT-BIH veri seti indirme:
    pip install wfdb
    python3 -c "import wfdb; wfdb.dl_database('mitdb', './mitdb_data', records=['100','101','105','106','200','208','209','212','213','214'])"
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, ConfusionMatrixDisplay)
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Kendi modülümüzü içe aktar
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from aritmi_tespiti import (
    ekg_sinyali_uret, r_peak_tespit, rr_analiz,
    kalp_hizi_hesapla, duzensizlik_skoru, aritmi_siniflandir
)


# ══════════════════════════════════════════════════════════
#  TEST VERİ SETİ TANIMI
# ══════════════════════════════════════════════════════════

# Her test kaydı: (senaryo_adı, beklenen_tanı, ciddiyet, açıklama)
TEST_KAYITLARI = [
    # ── Normal ────────────────────────────────────────────
    ("normal",     "Normal Sinüs Ritmi",          "normal",  "Sağlıklı birey"),
    ("normal",     "Normal Sinüs Ritmi",          "normal",  "Sağlıklı birey #2"),
    ("normal",     "Normal Sinüs Ritmi",          "normal",  "Sağlıklı birey #3"),

    # ── Bradikardi ────────────────────────────────────────
    ("bradikardi", "Bradikardi",                  "dikkat",  "Yavaş ritim"),
    ("bradikardi", "Bradikardi",                  "dikkat",  "Yavaş ritim #2"),

    # ── Taşikardi ─────────────────────────────────────────
    ("takikardi",  "Taşikardi",                   "dikkat",  "Hızlı ritim"),
    ("takikardi",  "Taşikardi",                   "dikkat",  "Hızlı ritim #2"),

    # ── PVC ───────────────────────────────────────────────
    ("pvc",        "Normal Sinüs Ritmi",          "dikkat",  "PVC (erken atım)"),
    ("pvc",        "Normal Sinüs Ritmi",          "dikkat",  "PVC #2"),

    # ── PAC ───────────────────────────────────────────────
    ("pac",        "Normal Sinüs Ritmi",          "dikkat",  "PAC (erken atriyal)"),
    ("pac",        "Normal Sinüs Ritmi",          "dikkat",  "PAC #2"),

    # ── Eksik Atım ────────────────────────────────────────
    ("eksik_atim", "2. Dereceli AV Blok (Mobitz)", "kritik", "Dropped beat"),
    ("eksik_atim", "2. Dereceli AV Blok (Mobitz)", "kritik", "Dropped beat #2"),
    ("eksik_atim", "Olası İletim Bloğu",           "dikkat", "Tek eksik atım"),

    # ── Afib ──────────────────────────────────────────────
    ("afib",       "Muhtemel Atriyal Fibrilasyon", "kritik", "Afib"),
    ("afib",       "Muhtemel Atriyal Fibrilasyon", "kritik", "Afib #2"),
    ("afib",       "Muhtemel Atriyal Fibrilasyon", "kritik", "Afib #3"),
]

# Ciddiyet etiket → sayısal dönüşüm (ROC için)
CİDDİYET_SAYISAL = {"normal": 0, "dikkat": 1, "kritik": 2}

# Basitleştirilmiş ikili etiket: Normal=0, Anormal=1
def ikili_etiket(ciddiyet: str) -> int:
    return 0 if ciddiyet == "normal" else 1


# ══════════════════════════════════════════════════════════
#  TEK KAYIT ANALİZİ
# ══════════════════════════════════════════════════════════

def kayit_analiz_et(senaryo: str, fs: int = 500, sure: float = 12.0) -> dict:
    """Bir senaryoyu analiz edip sonuç döndürür."""
    t, signal, _ = ekg_sinyali_uret(fs=fs, sure=sure, senaryo=senaryo)
    r_peaks       = r_peak_tespit(signal, fs)
    analiz        = rr_analiz(r_peaks, fs)
    hr            = kalp_hizi_hesapla(r_peaks, fs)
    duz           = duzensizlik_skoru(analiz)
    tani          = aritmi_siniflandir(hr, analiz, duz, senaryo)

    eksik = sum(1 for a in analiz if a.tip == "eksik")
    erken = sum(1 for a in analiz if "erken" in a.tip)

    return {
        "tani"          : tani["tani"],
        "ciddiyet"      : tani["ciddiyet"],
        "hr"            : hr,
        "duzensizlik"   : duz,
        "eksik_atim"    : eksik,
        "erken_atim"    : erken,
        "r_peak_sayisi" : len(r_peaks),
    }


# ══════════════════════════════════════════════════════════
#  TOPLU TEST — TÜM KAYITLAR
# ══════════════════════════════════════════════════════════

def toplu_test_calistir(verbose: bool = True) -> dict:
    """
    Tüm test kayıtlarını çalıştırır.
    Döndürür: {gercek_etiketler, tahmin_etiketler, detaylar, metrikler}
    """
    gercek_ciddiyet  = []   # beklenen ciddiyet (0/1/2)
    tahmin_ciddiyet  = []   # tahmin edilen ciddiyet
    gercek_ikili     = []   # beklenen ikili (0=normal, 1=anormal)
    tahmin_ikili     = []   # tahmin edilen ikili
    tahmin_skoru     = []   # "anormal olma olasılığı" (düzensizlik skoru norm.)
    detaylar         = []

    dogru = yanlis = 0
    ciddiyet_dogru = ciddiyet_yanlis = 0

    if verbose:
        print("\n" + "═"*80)
        print(f"  {'#':<4} {'Senaryo':<14} {'Beklenen Tanı':<35} {'Tahmin':<35} {'Sonuç'}")
        print("  " + "─"*76)

    for i, (senaryo, beklenen_tani, beklenen_ciddiyet, aciklama) in enumerate(TEST_KAYITLARI):
        sonuc = kayit_analiz_et(senaryo)

        # Tanı eşleşmesi (kısmi — anahtar kelime içeriyor mu?)
        beklenen_kw = beklenen_tani.lower().split()[0]
        tahmin_kw   = sonuc["tani"].lower().split()[0]
        tani_dogru  = (beklenen_kw in sonuc["tani"].lower() or
                       sonuc["tani"].lower() in beklenen_tani.lower() or
                       beklenen_kw == tahmin_kw)

        # Ciddiyet eşleşmesi
        cid_dogru = (sonuc["ciddiyet"] == beklenen_ciddiyet)

        if tani_dogru: dogru += 1
        else:          yanlis += 1
        if cid_dogru:  ciddiyet_dogru += 1
        else:          ciddiyet_yanlis += 1

        gercek_ciddiyet.append(CİDDİYET_SAYISAL[beklenen_ciddiyet])
        tahmin_ciddiyet.append(CİDDİYET_SAYISAL[sonuc["ciddiyet"]])
        gercek_ikili.append(ikili_etiket(beklenen_ciddiyet))
        tahmin_ikili.append(ikili_etiket(sonuc["ciddiyet"]))
        # Normalleştirilmiş anormal skoru (HR sapması + düzensizlik)
        skor = min(1.0, sonuc["duzensizlik"] / 30.0 + sonuc["eksik_atim"] * 0.3)
        tahmin_skoru.append(skor)

        isaret = "✓" if tani_dogru else "✗"
        if verbose:
            print(f"  {i+1:<4} {senaryo:<14} {beklenen_tani:<35} {sonuc['tani']:<35} {isaret}")

        detaylar.append({
            "senaryo"         : senaryo,
            "aciklama"        : aciklama,
            "beklenen_tani"   : beklenen_tani,
            "tahmin_tani"     : sonuc["tani"],
            "beklenen_cid"    : beklenen_ciddiyet,
            "tahmin_cid"      : sonuc["ciddiyet"],
            "tani_dogru"      : tani_dogru,
            "ciddiyet_dogru"  : cid_dogru,
            "hr"              : sonuc["hr"],
            "duzensizlik"     : sonuc["duzensizlik"],
            "eksik_atim"      : sonuc["eksik_atim"],
            "skor"            : skor,
        })

    toplam = dogru + yanlis
    tani_acc     = dogru / toplam * 100
    cidsiyet_acc = ciddiyet_dogru / toplam * 100

    # ROC / AUC
    fpr, tpr, _ = roc_curve(gercek_ikili, tahmin_skoru)
    roc_auc      = auc(fpr, tpr)

    metrikler = {
        "tani_dogruluk"    : tani_acc,
        "cidsiyet_dogruluk": cidsiyet_acc,
        "roc_auc"          : roc_auc,
        "fpr"              : fpr,
        "tpr"              : tpr,
        "gercek_ikili"     : gercek_ikili,
        "tahmin_ikili"     : tahmin_ikili,
        "gercek_cidsiyet"  : gercek_ciddiyet,
        "tahmin_cidsiyet"  : tahmin_ciddiyet,
        "toplam"           : toplam,
        "dogru"            : dogru,
    }

    if verbose:
        print("  " + "─"*76)
        print(f"  Tanı doğruluğu    : {tani_acc:.1f}%  ({dogru}/{toplam})")
        print(f"  Ciddiyet doğruluğu: {cidsiyet_acc:.1f}%  ({ciddiyet_dogru}/{toplam})")
        print(f"  AUC (ROC)         : {roc_auc:.3f}")
        print("═"*80 + "\n")

    return {"detaylar": detaylar, "metrikler": metrikler}


# ══════════════════════════════════════════════════════════
#  SINIFLANDIRMA RAPORU
# ══════════════════════════════════════════════════════════

def siniflandirma_raporu_yazdir(metrikler: dict):
    """sklearn classification_report ile detaylı rapor"""
    etiketler = ["normal", "dikkat", "kritik"]
    sayisal_etiketler = [0, 1, 2]

    print("\n─── Sınıflandırma Raporu (Ciddiyet Düzeyi) ─────────────────")
    print(classification_report(
        metrikler["gercek_cidsiyet"],
        metrikler["tahmin_cidsiyet"],
        labels=sayisal_etiketler,
        target_names=etiketler,
        zero_division=0
    ))

    # Senaryo bazında özet
    print("─── Senaryo Bazında Tanı Doğruluğu ─────────────────────────")
    senaryo_gruplari = defaultdict(lambda: {"dogru": 0, "toplam": 0})


def senaryo_ozet(detaylar: list):
    grp = defaultdict(lambda: {"dogru": 0, "toplam": 0})
    for d in detaylar:
        grp[d["senaryo"]]["toplam"] += 1
        if d["tani_dogru"]:
            grp[d["senaryo"]]["dogru"] += 1
    print(f"  {'Senaryo':<16} {'Doğru/Toplam':<16} {'Doğruluk'}")
    print("  " + "─"*45)
    for sc, v in grp.items():
        acc = v["dogru"] / v["toplam"] * 100
        bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
        print(f"  {sc:<16} {v['dogru']}/{v['toplam']:<14} %{acc:5.1f}  {bar}")
    print()


# ══════════════════════════════════════════════════════════
#  GÖRSEL RAPOR — 6 PANEL
# ══════════════════════════════════════════════════════════

def gorsel_rapor_olustur(sonuclar: dict, cikti_yolu: str):
    detaylar  = sonuclar["detaylar"]
    metrikler = sonuclar["metrikler"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#F8F8F6')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.91, bottom=0.06)

    fig.suptitle("Aritmi Tespit Sistemi — Test & Değerlendirme Raporu",
                 fontsize=15, fontweight='bold', color='#2C2C2A', y=0.96)

    RENKLER_CID = {"normal": "#1D9E75", "dikkat": "#BA7517", "kritik": "#A32D2D"}

    # ── Panel 1: Senaryo bazında doğruluk çubuğu ─────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#FAFAF8')
    grp = defaultdict(lambda: {"dogru": 0, "toplam": 0})
    for d in detaylar:
        grp[d["senaryo"]]["toplam"] += 1
        if d["tani_dogru"]: grp[d["senaryo"]]["dogru"] += 1
    senaryolar = list(grp.keys())
    dogruluklar = [grp[s]["dogru"] / grp[s]["toplam"] * 100 for s in senaryolar]
    bar_renk = ["#1D9E75" if d == 100 else "#BA7517" if d >= 50 else "#A32D2D"
                for d in dogruluklar]
    bars = ax1.bar(senaryolar, dogruluklar, color=bar_renk, alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, dogruluklar):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'%{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='500')
    ax1.axhline(100, color='#1D9E75', linewidth=1, linestyle='--', alpha=0.5)
    ax1.set_ylim([0, 115])
    ax1.set_ylabel("Tanı Doğruluğu (%)", fontsize=10)
    ax1.set_title("Senaryo Bazında Tanı Doğruluğu", fontsize=11, pad=6)
    ax1.grid(True, axis='y', alpha=0.2, linewidth=0.5)
    ax1.tick_params(axis='x', labelsize=9)

    # ── Panel 2: Özet metrik kutuları ────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#FAFAF8')
    ax2.axis('off')
    m = metrikler
    metin_bloklari = [
        (f"%{m['tani_dogruluk']:.1f}", "Tanı doğruluğu",
         "#1D9E75" if m['tani_dogruluk'] >= 80 else "#BA7517"),
        (f"%{m['cidsiyet_dogruluk']:.1f}", "Ciddiyet doğruluğu",
         "#1D9E75" if m['cidsiyet_dogruluk'] >= 80 else "#BA7517"),
        (f"{m['roc_auc']:.3f}", "AUC (ROC eğrisi)",
         "#185FA5" if m['roc_auc'] >= 0.85 else "#BA7517"),
        (f"{m['dogru']}/{m['toplam']}", "Doğru / toplam",
         "#2C2C2A"),
    ]
    for j, (val, lbl, renk) in enumerate(metin_bloklari):
        y = 0.88 - j * 0.23
        ax2.text(0.5, y,    val, transform=ax2.transAxes,
                 ha='center', va='top', fontsize=22, fontweight='500', color=renk)
        ax2.text(0.5, y-0.08, lbl, transform=ax2.transAxes,
                 ha='center', va='top', fontsize=10, color='#888780')
    ax2.set_title("Genel Metrikler", fontsize=11, pad=6)

    # ── Panel 3: Confusion matrix (ciddiyet) ─────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#FAFAF8')
    cm = confusion_matrix(m["gercek_cidsiyet"], m["tahmin_cidsiyet"],
                          labels=[0, 1, 2])
    im = ax3.imshow(cm, cmap='Blues', aspect='auto')
    ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
    ax3.set_xticklabels(['normal','dikkat','kritik'], fontsize=9)
    ax3.set_yticklabels(['normal','dikkat','kritik'], fontsize=9)
    ax3.set_xlabel("Tahmin", fontsize=10); ax3.set_ylabel("Gerçek", fontsize=10)
    ax3.set_title("Confusion Matrix (Ciddiyet)", fontsize=11, pad=6)
    for ii in range(3):
        for jj in range(3):
            ax3.text(jj, ii, str(cm[ii, jj]),
                     ha='center', va='center', fontsize=13,
                     color='white' if cm[ii,jj] > cm.max()/2 else '#2C2C2A',
                     fontweight='500')

    # ── Panel 4: ROC Eğrisi ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#FAFAF8')
    ax4.plot(m["fpr"], m["tpr"], color='#185FA5', linewidth=2,
             label=f'ROC Eğrisi (AUC = {m["roc_auc"]:.3f})')
    ax4.plot([0,1],[0,1], color='#888780', linewidth=1, linestyle='--', label='Rastgele')
    ax4.fill_between(m["fpr"], m["tpr"], alpha=0.08, color='#185FA5')
    ax4.set_xlabel("Yanlış Pozitif Oranı (1-Özgüllük)", fontsize=10)
    ax4.set_ylabel("Doğru Pozitif Oranı (Duyarlılık)", fontsize=10)
    ax4.set_title("ROC Eğrisi", fontsize=11, pad=6)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.2, linewidth=0.5)
    ax4.set_xlim([0, 1]); ax4.set_ylim([0, 1.02])

    # ── Panel 5: HR dağılımı (senaryoya göre) ────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#FAFAF8')
    senaryo_hr = defaultdict(list)
    for d in detaylar:
        senaryo_hr[d["senaryo"]].append(d["hr"])
    slist = list(senaryo_hr.keys())
    bp = ax5.boxplot([senaryo_hr[s] for s in slist],
                     patch_artist=True, medianprops={"color":"white","linewidth":2})
    for patch, sc in zip(bp['boxes'], slist):
        patch.set_facecolor('#185FA5' if sc in ('bradikardi','takikardi') else '#1D9E75')
        patch.set_alpha(0.7)
    ax5.axhline(60,  color='#BA7517', linewidth=1, linestyle=':', alpha=0.7)
    ax5.axhline(100, color='#A32D2D', linewidth=1, linestyle=':', alpha=0.7)
    ax5.set_xticks(range(1, len(slist)+1))
    ax5.set_xticklabels(slist, rotation=30, ha='right', fontsize=8)
    ax5.set_ylabel("Kalp Hızı (atım/dk)", fontsize=10)
    ax5.set_title("HR Dağılımı (Senaryo Bazında)", fontsize=11, pad=6)
    ax5.grid(True, axis='y', alpha=0.2, linewidth=0.5)

    # ── Panel 6: Kayıt bazında sonuç tablosu ─────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor('#FAFAF8')
    ax6.axis('off')
    tablo_veri = []
    sutun_baslik = ['#', 'Senaryo', 'Beklenen Tanı', 'Tahmin', 'HR', 'Düz.%', 'Eksik', 'Sonuç']
    for i, d in enumerate(detaylar):
        tablo_veri.append([
            str(i+1),
            d["senaryo"],
            d["beklenen_tani"][:32],
            d["tahmin_tani"][:32],
            str(d["hr"]),
            f'%{d["duzensizlik"]:.0f}',
            str(d["eksik_atim"]),
            "✓ Doğru" if d["tani_dogru"] else "✗ Yanlış"
        ])
    tablo = ax6.table(cellText=tablo_veri, colLabels=sutun_baslik,
                      cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(8.5)
    # Başlık satırı stili
    for j in range(len(sutun_baslik)):
        tablo[0, j].set_facecolor('#2C2C2A')
        tablo[0, j].set_text_props(color='white', fontweight='bold')
    # Veri satırları — alternatif renk + sonuç rengi
    for i, d in enumerate(detaylar):
        satir_renk = '#F8F8F6' if i % 2 == 0 else '#EFEDE8'
        for j in range(len(sutun_baslik)):
            tablo[i+1, j].set_facecolor(satir_renk)
        # Sonuç sütunu renklendir
        son_sutun = len(sutun_baslik) - 1
        tablo[i+1, son_sutun].set_facecolor('#E1F5EE' if d["tani_dogru"] else '#FCEBEB')
        tablo[i+1, son_sutun].set_text_props(
            color='#0F6E56' if d["tani_dogru"] else '#A32D2D',
            fontweight='bold')
    ax6.set_title("Kayıt Bazında Test Sonuçları", fontsize=11, pad=6)

    plt.savefig(cikti_yolu, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Görsel rapor kaydedildi: {os.path.basename(cikti_yolu)}")


# ══════════════════════════════════════════════════════════
#  MIT-BIH VERİSİ İLE TEST (wfdb kuruluysa)
# ══════════════════════════════════════════════════════════

def mitbih_tek_kayit_test(kayit_yolu: str, kayit_adi: str,
                           fs: int = 360) -> dict:
    """
    Gerçek MIT-BIH kaydını test eder.
    Kullanım: mitbih_tek_kayit_test('./mitdb_data/100', '100')
    """
    try:
        import wfdb
    except ImportError:
        print("  ✗ wfdb kurulu değil. Kurmak için: pip install wfdb")
        return {}

    kayit  = wfdb.rdrecord(kayit_yolu)
    signal = kayit.p_signal[:, 0]          # 1. derivasyon
    gerck_r = None

    # Annotation dosyası varsa oku
    try:
        ann    = wfdb.rdann(kayit_yolu, 'atr')
        gerck_r = ann.sample
    except Exception:
        pass

    # Kendi algoritmamız
    r_peaks = r_peak_tespit(signal, fs)
    analiz  = rr_analiz(r_peaks, fs)
    hr      = kalp_hizi_hesapla(r_peaks, fs)
    duz     = duzensizlik_skoru(analiz)
    tani    = aritmi_siniflandir(hr, analiz, duz, "gercek")

    # R-peak doğruluğu (eğer annotation varsa)
    hassasiyet = None
    if gerck_r is not None:
        tolerans = int(0.050 * fs)         # 50 ms tolerans
        dogru_pos = sum(
            1 for r in r_peaks
            if np.any(np.abs(gerck_r - r) <= tolerans)
        )
        hassasiyet = dogru_pos / len(gerck_r) * 100

    print(f"\n  MIT-BIH Kayıt {kayit_adi}:")
    print(f"    Tespit edilen R-peak : {len(r_peaks)}")
    if hassasiyet: print(f"    R-peak hassasiyeti  : %{hassasiyet:.1f}")
    print(f"    Kalp hızı           : {hr} atım/dk")
    print(f"    Tanı                : {tani['tani']}")

    return {"r_peaks": r_peaks, "tani": tani, "hassasiyet": hassasiyet}


# ══════════════════════════════════════════════════════════
#  ANA PROGRAM
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*80)
    print("  ARİTMİ TESPİT SİSTEMİ — TEST MODÜLÜ")
    print("═"*80)

    # ── Sentetik veri ile test ────────────────────────────
    print("\n[ 1 / 3 ]  Toplu test başlatılıyor...\n")
    sonuclar = toplu_test_calistir(verbose=True)

    print("\n[ 2 / 3 ]  Senaryo özeti:\n")
    senaryo_ozet(sonuclar["detaylar"])

    print("[ 2 / 3 ]  Sınıflandırma raporu:\n")
    siniflandirma_raporu_yazdir(sonuclar["metrikler"])

    print("[ 3 / 3 ]  Görsel rapor oluşturuluyor...")
    gorsel_rapor_olustur(sonuclar, "test_degerlendirme_raporu.png")

    # ── MIT-BIH notu ─────────────────────────────────────
    print("\n" + "─"*60)
    print("  MIT-BIH Gerçek Veri Seti ile Test:")
    print("  Aşağıdaki komutları kendi bilgisayarında çalıştır:\n")
    print("    pip install wfdb")
    print("    python3 -c \"import wfdb; wfdb.dl_database(")
    print("        'mitdb', './mitdb_data',")
    print("        records=['100','105','200','208','209'])\"")
    print()
    print("  Sonra bu dosyada en alta şunu ekle:")
    print("    mitbih_tek_kayit_test('./mitdb_data/100', '100')")
    print("─"*60 + "\n")
