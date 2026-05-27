"""
============================================================
  MIT-BIH ARRHYTHMIA DATABASE — GERÇEK VERİ TESTİ
  Bitirme Tezi | Gerçek Veri ile Doğrulama

  Kullanım:
    1. pip install wfdb numpy scipy matplotlib scikit-learn
    2. python -c "import wfdb; wfdb.dl_database('mitdb', './mitdb_data', records=['100','101','105','106','108','200','207','208','209','213'])"
    3. python mitbih_test.py

  Not:
    Bu program klinik tanı koymaz. Çıktılar R-peak/atım tespiti ve
    RR aralığı analizine dayalı algoritma yorumu/ön değerlendirme niteliğindedir.
============================================================
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Kendi algoritmamızı içe aktar ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from aritmi_tespiti import (  # noqa: E402
    r_peak_tespit,
    rr_analiz,
    kalp_hizi_hesapla,
    duzensizlik_skoru,
    aritmi_siniflandir,
    sinyal_kalitesi_degerlendir,
)

# ── wfdb kontrolü ────────────────────────────────────────
try:
    import wfdb
except ImportError:
    print("\n  HATA: wfdb kurulu değil.")
    print("  Çözüm: pip install wfdb\n")
    sys.exit(1)

plt.rcParams["font.family"] = "DejaVu Sans"

# ══════════════════════════════════════════════════════════
#  KAYIT LİSTESİ VE AÇIKLAMALAR
# ══════════════════════════════════════════════════════════

KAYITLAR: Dict[str, Dict[str, str]] = {
    "100": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "normal", "aciklama": "Referans normal ritim"},
    "101": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "normal", "aciklama": "Normal + az PAC"},
    "105": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "PVC içerir"},
    "106": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "Yoğun PVC"},
    "108": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "PVC + diğer anormallik"},
    "200": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "Karışık aritmiler"},
    "207": {"beklenen": "Bradikardi", "ciddiyet": "kritik", "aciklama": "AV blok + yavaş ritim"},
    "208": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "Yoğun PVC"},
    "209": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "PAC içerir"},
    "213": {"beklenen": "Normal Sinüs Ritmi", "ciddiyet": "dikkat", "aciklama": "PVC + PAC"},
}

# MIT-BIH annotation sembolleri → kategori
NORMAL_SEMBOLLER = {"N", "L", "R", "e", "j"}
PVC_SEMBOLLER = {"V", "E"}
PAC_SEMBOLLER = {"A", "a", "S", "J"}
BLOK_SEMBOLLER = {"/", "f", "Q"}
DIGER_BEAT_SEMBOLLER = {"B", "r", "F", "n", "?"}

# R-peak/atım tespiti performansı hesaplanırken yalnızca beat annotation sembolleri kullanılır.
BEAT_SEMBOLLER = NORMAL_SEMBOLLER | PVC_SEMBOLLER | PAC_SEMBOLLER | BLOK_SEMBOLLER | DIGER_BEAT_SEMBOLLER


def beat_annotation_filtrele(ann, sure_ornekler: int) -> Tuple[np.ndarray, List[str]]:
    """MIT-BIH annotation içinden yalnızca beat/QRS kabul edilen sembolleri seçer."""
    secilen_ornekler = []
    secilen_semboller = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if sample < sure_ornekler and symbol in BEAT_SEMBOLLER:
            secilen_ornekler.append(int(sample))
            secilen_semboller.append(symbol)
    return np.array(secilen_ornekler, dtype=int), secilen_semboller


def r_peak_eslesme_metrikleri(tespit_r: np.ndarray, gercek_r: np.ndarray, tolerans: int) -> Tuple[int, int, int]:
    """
    50 ms toleransla one-to-one R-peak eşleştirmesi yapar.

    Eski yaklaşımdaki gibi aynı gerçek atımı birden fazla algoritma tepesine
    saymamak için her gerçek anotasyon en fazla bir kez eşleştirilir.
    """
    tespit = np.sort(np.asarray(tespit_r, dtype=int))
    gercek = np.sort(np.asarray(gercek_r, dtype=int))

    if len(tespit) == 0:
        return 0, 0, len(gercek)
    if len(gercek) == 0:
        return 0, len(tespit), 0

    kullanildi = np.zeros(len(gercek), dtype=bool)
    tp = 0
    fp = 0

    for r in tespit:
        sol = np.searchsorted(gercek, r - tolerans, side="left")
        sag = np.searchsorted(gercek, r + tolerans, side="right")
        adaylar = [idx for idx in range(sol, sag) if not kullanildi[idx]]
        if not adaylar:
            fp += 1
            continue
        en_yakin = min(adaylar, key=lambda idx: abs(int(gercek[idx]) - int(r)))
        kullanildi[en_yakin] = True
        tp += 1

    fn = int(len(gercek) - tp)
    return int(tp), int(fp), int(fn)


def metrik_hesapla(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def yorum_metni_temizle(yorum: str) -> str:
    """Tez çıktısında fazla klinik/iddialı görünen ifadeleri yumuşatır."""
    return (
        yorum
        .replace("Yüksek RR Düzensizliği / Muhtemel AF Adayı", "Yüksek RR Düzensizliği / Düzensiz Ritim Adayı")
        .replace("Muhtemel Atriyal Fibrilasyon", "Yüksek RR Düzensizliği")
    )


def guvenilirlik_belirle(sonuc: dict) -> str:
    """F1, duyarlılık ve sinyal kalite skoruna göre rapor güvenilirliği etiketi üretir."""
    f1 = sonuc.get("f1") or 0.0
    duy = sonuc.get("duyarlilik") or 0.0
    kalite = sonuc.get("kalite") or {}
    kalite_skor = kalite.get("skor", 100.0)
    kalite_durum = kalite.get("durum", "iyi")

    if kalite_durum == "düşük" or kalite_skor < 60 or f1 < 85 or duy < 80:
        return "Dikkatli yorumlanmalı"
    if kalite_durum == "orta" or kalite_skor < 75 or f1 < 92 or duy < 90:
        return "Orta - kontrol önerilir"
    return "Yüksek"


# ══════════════════════════════════════════════════════════
#  TEK KAYIT ANALİZİ
# ══════════════════════════════════════════════════════════

def kayit_analiz_et(kayit_yolu: str, kayit_adi: str) -> Optional[dict]:
    """
    Bir MIT-BIH kaydını okur, algoritmayla analiz eder ve referans
    beat annotation'ları ile R-peak/atım tespit başarısını ölçer.
    """
    try:
        kayit = wfdb.rdrecord(kayit_yolu)
    except Exception as e:
        print(f"  ✗ Kayıt okunamadı ({kayit_adi}): {e}")
        return None

    fs = int(kayit.fs)
    signal = kayit.p_signal[:, 0].copy()  # İlk kanal / MLII

    sure_ornekler = min(len(signal), int(1800 * fs))  # İlk 30 dk / tam kayıt
    signal_kisim = signal[:sure_ornekler]

    try:
        ann = wfdb.rdann(kayit_yolu, "atr")
        gercek_r_kisim, ann_kisim = beat_annotation_filtrele(ann, sure_ornekler)
    except Exception:
        gercek_r_kisim = np.array([], dtype=int)
        ann_kisim = []

    tespit_r = r_peak_tespit(signal_kisim, fs)
    kalite = sinyal_kalitesi_degerlendir(signal_kisim, fs, tespit_r)

    tp = fp = fn = 0
    hassasiyet_pct = duyarlilik_pct = f1_score = None
    if len(gercek_r_kisim) > 0:
        tolerans = int(0.050 * fs)
        tp, fp, fn = r_peak_eslesme_metrikleri(tespit_r, gercek_r_kisim, tolerans)
        hassasiyet_pct, duyarlilik_pct, f1_score = metrik_hesapla(tp, fp, fn)

    analiz = rr_analiz(tespit_r, fs)
    hr = kalp_hizi_hesapla(tespit_r, fs)
    duz = duzensizlik_skoru(analiz)
    tani = aritmi_siniflandir(hr, analiz, duz, "gercek", kalite)
    tani["tani"] = yorum_metni_temizle(tani.get("tani", ""))

    ann_normal = sum(1 for s in ann_kisim if s in NORMAL_SEMBOLLER)
    ann_pvc = sum(1 for s in ann_kisim if s in PVC_SEMBOLLER)
    ann_pac = sum(1 for s in ann_kisim if s in PAC_SEMBOLLER)
    ann_blok = sum(1 for s in ann_kisim if s in BLOK_SEMBOLLER)

    sonuc = {
        "kayit_adi": kayit_adi,
        "fs": fs,
        "signal": signal_kisim,
        "tespit_r": tespit_r,
        "gercek_r": gercek_r_kisim,
        "tani": tani,
        "hr": hr,
        "duzensizlik": duz,
        "hassasiyet": hassasiyet_pct,
        "duyarlilik": duyarlilik_pct,
        "f1": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ann_normal": ann_normal,
        "ann_pvc": ann_pvc,
        "ann_pac": ann_pac,
        "ann_blok": ann_blok,
        "ann_toplam_beat": len(ann_kisim),
        "eksik_atim": tani.get("eksik_atim", 0),
        "erken_atim": tani.get("erken_atim", 0),
        "kalite": kalite,
        "aciklama": KAYITLAR.get(kayit_adi, {}).get("aciklama", ""),
    }
    sonuc["guvenilirlik"] = guvenilirlik_belirle(sonuc)
    return sonuc


# ══════════════════════════════════════════════════════════
#  TOPLU TEST VE RAPORLAMA
# ══════════════════════════════════════════════════════════

def toplu_test(veri_klasoru: str = "./mitdb_data") -> List[dict]:
    """Tüm seçili kayıtları test eder, sonuçları döndürür."""
    sonuclar = []

    print("\n" + "═" * 118)
    print("  MIT-BIH ARRHYTHMIA DATABASE — GERÇEK VERİ TESTİ")
    print("═" * 118)
    print(
        f"\n  {'#':<3} {'Kayıt':<6} {'HR':<6} {'Duy.%':<8} {'PPV.%':<8} {'F1':<7} "
        f"{'Kalite':<9} {'Güvenilirlik':<24} {'Algoritma Yorumu':<42} Açıklama"
    )
    print("  " + "─" * 116)

    for i, (kayit_no, bilgi) in enumerate(KAYITLAR.items(), start=1):
        yol = os.path.join(veri_klasoru, kayit_no)

        if not os.path.exists(yol + ".hea"):
            print(f"  {i:<3} {kayit_no:<6} {'─':<6} {'─':<8} {'─':<8} {'─':<7} Dosya bulunamadı → mitdb_data/{kayit_no}.hea")
            continue

        sonuc = kayit_analiz_et(yol, kayit_no)
        if sonuc is None:
            continue

        duy = f"%{sonuc['duyarlilik']:.1f}" if sonuc["duyarlilik"] is not None else "─"
        has = f"%{sonuc['hassasiyet']:.1f}" if sonuc["hassasiyet"] is not None else "─"
        f1 = f"{sonuc['f1']:.2f}" if sonuc["f1"] is not None else "─"
        kalite = (sonuc.get("kalite") or {}).get("durum", "─")
        yorum = yorum_metni_temizle(sonuc["tani"]["tani"])

        print(
            f"  {i:<3} {kayit_no:<6} {sonuc['hr']:<6} {duy:<8} {has:<8} {f1:<7} "
            f"{kalite:<9} {sonuc['guvenilirlik']:<24} {yorum:<42} {bilgi['aciklama']}"
        )

        sonuclar.append(sonuc)

    return sonuclar


def ozet_yazdir(sonuclar: List[dict]) -> None:
    """Genel performans özetini yazar."""
    f1_ler = [s["f1"] for s in sonuclar if s["f1"] is not None]
    duy_ler = [s["duyarlilik"] for s in sonuclar if s["duyarlilik"] is not None]
    has_ler = [s["hassasiyet"] for s in sonuclar if s["hassasiyet"] is not None]

    print("\n" + "─" * 72)
    print("  GENEL PERFORMANS ÖZETİ")
    print("─" * 72)
    if duy_ler:
        print(f"  Ort. Duyarlılık / Recall / Sensitivity : %{np.mean(duy_ler):.1f}")
        print(f"  Ort. Precision / PPV                  : %{np.mean(has_ler):.1f}")
        print(f"  Ort. F1 Skoru                         : {np.mean(f1_ler):.3f}")
    print(f"  Analiz edilen kayıt sayısı            : {len(sonuclar)}")
    print("─" * 72)

    if duy_ler:
        ort_duy = np.mean(duy_ler)
        if ort_duy >= 90:
            print("  Değerlendirme: R-peak/atım tespit performansı yüksek bulunmuştur.")
        elif ort_duy >= 80:
            print("  Değerlendirme: R-peak/atım tespit performansı kabul edilebilir düzeydedir.")
        else:
            print("  Değerlendirme: R-peak/atım tespiti için filtreleme/eşik parametreleri gözden geçirilmelidir.")

    print("  Not: Bu metrikler aritmi türü sınıflandırma başarısını değil,")
    print("       referans annotation'lara göre R-peak/atım tespit başarısını gösterir.")
    print("       Algoritma yorumları klinik tanı değil, RR analizi tabanlı ön değerlendirmedir.\n")


def csv_kaydet(sonuclar: List[dict], cikti: str = "mitbih_sonuclar.csv") -> None:
    """Tez tablosu için sonuçları CSV dosyasına kaydeder."""
    alanlar = [
        "Kayit", "HR", "Recall_%", "Precision_PPV_%", "F1", "TP", "FP", "FN",
        "Sinyal_Kalitesi", "Kalite_Skoru", "Guvenilirlik", "Normal", "PVC", "PAC", "Blok_Diger",
        "Algoritma_Yorumu", "Aciklama",
    ]
    with open(cikti, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=alanlar)
        writer.writeheader()
        for s in sonuclar:
            kalite = s.get("kalite") or {}
            writer.writerow({
                "Kayit": s["kayit_adi"],
                "HR": s["hr"],
                "Recall_%": round(s["duyarlilik"], 2) if s["duyarlilik"] is not None else "",
                "Precision_PPV_%": round(s["hassasiyet"], 2) if s["hassasiyet"] is not None else "",
                "F1": round(s["f1"], 3) if s["f1"] is not None else "",
                "TP": s.get("tp", ""),
                "FP": s.get("fp", ""),
                "FN": s.get("fn", ""),
                "Sinyal_Kalitesi": kalite.get("durum", ""),
                "Kalite_Skoru": kalite.get("skor", ""),
                "Guvenilirlik": s.get("guvenilirlik", ""),
                "Normal": s.get("ann_normal", ""),
                "PVC": s.get("ann_pvc", ""),
                "PAC": s.get("ann_pac", ""),
                "Blok_Diger": s.get("ann_blok", ""),
                "Algoritma_Yorumu": yorum_metni_temizle(s["tani"]["tani"]),
                "Aciklama": s.get("aciklama", ""),
            })
    print(f"  → CSV sonuç dosyası kaydedildi: {cikti}")


# ══════════════════════════════════════════════════════════
#  GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════

def gorsel_rapor(sonuclar: List[dict], cikti: str = "mitbih_test_raporu.png") -> None:
    """EKG sinyali + R-peak karşılaştırması + metrik özeti."""
    if not sonuclar:
        print("  Görselleştirilecek sonuç yok.")
        return

    n_kayit = min(len(sonuclar), 4)
    fig = plt.figure(figsize=(20, 4 * n_kayit + 4.8))
    fig.patch.set_facecolor("#F8F8F6")

    gs = gridspec.GridSpec(
        n_kayit + 1, 3, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.05, right=0.98,
        top=0.93, bottom=0.06,
    )

    fig.suptitle(
        "MIT-BIH Arrhythmia Database — Gerçek Veri Test Raporu",
        fontsize=14, fontweight="bold", color="#2C2C2A", y=0.97,
    )

    for i, sonuc in enumerate(sonuclar[:n_kayit]):
        signal = sonuc["signal"]
        fs = sonuc["fs"]
        t = np.arange(len(signal)) / fs
        t_max = min(60.0, t[-1])
        maske = t <= t_max

        ax = fig.add_subplot(gs[i, :2])
        ax.set_facecolor("#FAFAF8")
        ax.plot(t[maske], signal[maske], color="#2C2C2A", linewidth=0.6, alpha=0.85)

        for r in sonuc["tespit_r"]:
            if r < len(signal) and t[r] <= t_max:
                ax.axvline(t[r], color="#185FA5", alpha=0.35, linewidth=0.8)
                ax.plot(t[r], signal[r], "o", color="#185FA5", markersize=4, zorder=5)

        for r in sonuc["gercek_r"]:
            if r < len(signal) and t[r] <= t_max:
                ax.plot(t[r], signal[r] + 0.08, "^", color="#1D9E75", markersize=3, zorder=6, alpha=0.7)

        duy_str = f"%{sonuc['duyarlilik']:.1f}" if sonuc["duyarlilik"] is not None else "─"
        f1_str = f"F1={sonuc['f1']:.2f}" if sonuc["f1"] is not None else ""
        yorum = yorum_metni_temizle(sonuc["tani"]["tani"])
        kalite = (sonuc.get("kalite") or {}).get("durum", "─")
        ax.set_title(
            f"Kayıt {sonuc['kayit_adi']} | HR: {sonuc['hr']} atım/dk | "
            f"Duyarlılık: {duy_str} {f1_str} | Kalite: {kalite} | Algoritma Yorumu: {yorum}",
            fontsize=8.5, loc="left", pad=4,
        )
        ax.set_xlim([0, t_max])
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.set_yticks([])
        if i == n_kayit - 1:
            ax.set_xlabel("Zaman (s)", fontsize=9)

        if i == 0:
            from matplotlib.lines import Line2D
            legend_items = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#185FA5", markersize=6, label="Algoritma"),
                Line2D([0], [0], marker="^", color="w", markerfacecolor="#1D9E75", markersize=6, label="Referans (MIT-BIH)"),
            ]
            ax.legend(handles=legend_items, loc="upper right", fontsize=8)

        ax2 = fig.add_subplot(gs[i, 2])
        ax2.set_facecolor("#FAFAF8")
        etiketler = ["Normal", "PVC", "PAC"]
        degerler = [sonuc["ann_normal"], sonuc["ann_pvc"], sonuc["ann_pac"]]
        renkler = ["#1D9E75", "#A32D2D", "#BA7517"]
        bars = ax2.bar(etiketler, degerler, color=renkler, alpha=0.8, edgecolor="white", linewidth=0.5, width=0.5)
        for bar, val in zip(bars, degerler):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val), ha="center", va="bottom", fontsize=9)
        ax2.set_title("Beat annotation dağılımı (30 dk)", fontsize=9, pad=4)
        ax2.grid(True, axis="y", alpha=0.2, linewidth=0.4)
        ax2.set_ylabel("Atım sayısı", fontsize=8)

    ax_son = fig.add_subplot(gs[n_kayit, :])
    ax_son.set_facecolor("#FAFAF8")
    ax_son.axis("off")

    tablo_veri = []
    for s in sonuclar:
        duy = f"%{s['duyarlilik']:.1f}" if s["duyarlilik"] is not None else "─"
        has = f"%{s['hassasiyet']:.1f}" if s["hassasiyet"] is not None else "─"
        f1 = f"{s['f1']:.3f}" if s["f1"] is not None else "─"
        kalite = (s.get("kalite") or {}).get("durum", "─")
        tablo_veri.append([
            s["kayit_adi"], str(s["hr"]), duy, has, f1, kalite,
            s.get("guvenilirlik", ""), str(s["ann_normal"]), str(s["ann_pvc"]), str(s["ann_pac"]),
            yorum_metni_temizle(s["tani"]["tani"]),
        ])

    basliklar = ["Kayıt", "HR", "Duyarlılık", "PPV", "F1", "Kalite", "Güvenilirlik", "Normal", "PVC", "PAC", "Algoritma Yorumu"]
    col_widths = [0.055, 0.055, 0.08, 0.07, 0.07, 0.07, 0.15, 0.06, 0.055, 0.055, 0.28]
    tablo = ax_son.table(
        cellText=tablo_veri,
        colLabels=basliklar,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.10, 1, 0.84],
    )
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(7.6)

    for j in range(len(basliklar)):
        tablo[0, j].set_facecolor("#2C2C2A")
        tablo[0, j].set_text_props(color="white", fontweight="bold")

    for i, _ in enumerate(sonuclar):
        satir_renk = "#F8F8F6" if i % 2 == 0 else "#EFEDE8"
        for j in range(len(basliklar)):
            tablo[i + 1, j].set_facecolor(satir_renk)

    ax_son.set_title("Tüm Kayıtlar — R-peak/Atım Tespit Performansı ve Algoritma Yorumu", fontsize=10, pad=6)
    ax_son.text(
        0.5, 0.02,
        "Not: Duyarlılık, PPV ve F1 değerleri aritmi türü tanısını değil, referans annotation'lara göre R-peak/atım tespit başarısını gösterir.",
        transform=ax_son.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="#555555", style="italic",
    )

    plt.savefig(cikti, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Görsel rapor kaydedildi: {cikti}")


# ══════════════════════════════════════════════════════════
#  ANA PROGRAM
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    VERI_KLASORU = "./mitdb_data"

    if not os.path.isdir(VERI_KLASORU):
        print(f"\n  HATA: '{VERI_KLASORU}' klasörü bulunamadı.")
        print("  Önce veriyi indir:\n")
        print("    python -c \"import wfdb; wfdb.dl_database('mitdb', './mitdb_data',")
        print("         records=['100','101','105','106','108','200','207','208','209','213'])\"")
        print()
        sys.exit(1)

    sonuclar = toplu_test(VERI_KLASORU)

    if not sonuclar:
        print("\n  Hiç kayıt analiz edilemedi. Veri klasörünü kontrol et.")
        sys.exit(1)

    ozet_yazdir(sonuclar)
    csv_kaydet(sonuclar, cikti="mitbih_sonuclar.csv")

    print("  Görsel rapor oluşturuluyor...")
    gorsel_rapor(sonuclar, cikti="mitbih_test_raporu.png")

    print("  Tamamlandı. mitbih_test_raporu.png ve mitbih_sonuclar.csv dosyalarını kontrol et.\n")
