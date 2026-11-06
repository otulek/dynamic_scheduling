#!/usr/bin/env python3
"""
Test pencereleri ozeti: sabitlenmis politikalarla (kurallar.POLITIKA_SURUM) ECT, agirlik, yerel arama,
kesin plan ve geldigi sira (fcfs) karsilastirmasi.

Kullanim:  python test_ozet.py test        (dogrulama.py --out test ... ciktisi)
           python test_ozet.py test --haric 4,134   (+ duyarlilik analizi: bu pencereler cikarilmis)

Genelleme birimi PENCEREDIR. Bir pencerede birden fazla tohum varsa once pencere icinde ortalama
alinir; pencereler arasi ozet (medyan, en kucuk, en buyuk, kac pencerede daha iyi, Wilcoxon
isaretli siralar testi) bu pencere degerleri uzerinden yapilir. Wilcoxon p degeri aciklayici
amaclidir: coklu karsilastirma duzeltmesi yapilmaz, pencere sayisi kucuktur.

Olcutler (kapali dongu, pencerede gelen butun isler uzerinden ortalama):
  akis    = bitis - gelis            (ana olcut; planlayicinin tahmini amaci bunun karsiligidir)
  bekleme = baslama - gelis
  islem   = bitis - baslama          (akis = bekleme + islem)
  bsld    = bounded slowdown = max(1, akis / max(islem, 10 sn))   (ikincil olcut)
Fark: (yontem - referans) / referans * 100; negatif = yontem daha iyi (dort olcutte de kucuk iyidir).

Kesinlik kapsami:
  kesin_geri_dusus_% : kesin planli kosuda, bekleyen is --cap'i astigi icin kesin cozulmeyen karar orani
  kesin_K_%          : kesin cozulen kararlarda K yuva sinirinin baglayici oldugu oran
  kesin_guvenilir    : ikisi de %1'in altinda; kesin plan karsilastirmasi yalniz bu hucrelerde "kesin
                       plan" anlamina gelir, digerlerinde kismidir
  acik_kesin_cozulen_%: acik dongu olcumunde (ECT gidisati) kesin referansi olan karar orani
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kural_ozet import open_loop  # noqa: E402

METHODS = [("greedy", "ECT"), ("agirlik", "agirlik"), ("yerel", "yerel"), ("exact", "kesin"), ("fcfs", "fcfs")]
METRICS = [("ort_akis", "akis"), ("ort_bekleme", "bekleme"), ("ort_islem", "islem"), ("ort_bsld", "bsld")]
COMPARE = [("agirlik", "greedy"), ("yerel", "greedy"), ("exact", "greedy"), ("fcfs", "greedy"),
           ("yerel", "agirlik"), ("exact", "agirlik")]
NAME = dict(METHODS)


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def duyarlilik(S, haric):
    """Tam hassasiyetli pencere farklarindan: butun pencereler ve haric tutulanlar cikarilmis halde,
    her yuk ve yontem icin ECT'ye gore medyan akis farki ve kac pencerede daha iyi (fark < 0).
    Cikarma butun yontemlere ayni bicimde uygulanir."""
    rows = []
    for rho, g in S.groupby("rho"):
        for y in ("agirlik", "yerel", "exact", "fcfs"):
            c = f"fark_{NAME[y]}_vs_ECT_akis_%"
            if c not in g:
                continue
            for alt, gg in (("tum", g), ("haric", g[~g.gun.isin(haric)])):
                v = gg[c].dropna()
                if len(v):
                    rows.append({"rho": rho, "yontem": NAME[y], "pencereler": alt, "n": len(v),
                                 "medyan_%": v.median(), "daha_iyi": int((v < 0).sum()),
                                 "esit": int((v == 0).sum()), "min_%": v.min(), "max_%": v.max()})
    return pd.DataFrame(rows)


def maliyet(out, S):
    """Karar basina hesaplama suresi, acik dongu olcum kayitlarindan (ECT gidisati, >= 2 bekleyen is).
    Kayitli ms_kesin yalniz cozucu adimini olcer; politika maliyeti onu gerektiren planlari da ekler:
      ECT                   = ms_hizli
      agirlik               = ms_agirlik
      yerel (politika)      = ms_hizli + ms_yerel          (ms_yerel agirligi zaten icerir; ECT baslangic karsilastirmasi icin)
      kesin cozucu          = ms_kesin
      kesin (karma politika)= ms_hizli + ms_agirlik + ms_kesin
    Kapsamlar: 'kesin' = kesin cozulen ve K siniri baglayici olmayan kararlar (ayni karar kumesi butun
    yontemler icin); 'K' = K siniri baglayici; 'geri' = bekleyen > cap; 'tum' = hepsi (karma politika)."""
    rows = []
    for gun, rho in S[["gun", "rho"]].itertuples(index=False):
        gun = int(gun)
        dp = os.path.join(out, "kararlar", f"gun{gun}_rho{rho}_greedy_s0.csv")
        if not os.path.exists(dp):
            continue
        D = pd.read_csv(dp)
        if D.empty or not {"ms_hizli", "ms_agirlik", "ms_yerel", "ms_kesin"} <= set(D.columns):
            continue
        D["ECT"] = D.ms_hizli
        D["agirlik"] = D.ms_agirlik
        D["yerel_politika"] = D.ms_hizli + D.ms_yerel
        D["kesin_cozucu"] = D.ms_kesin
        D["kesin_politika"] = D.ms_hizli + D.ms_agirlik + D.ms_kesin
        for kap, sub in (("kesin", D[D.durum == "kesin"]), ("K", D[D.durum == "K"]),
                         ("geri", D[D.durum == "geri"]), ("tum", D)):
            if not len(sub):
                continue
            row = {"gun": gun, "rho": rho, "kapsam": kap, "karar": len(sub),
                   "bekleyen_medyan": sub.bekleyen.median()}
            for c in ("ECT", "agirlik", "yerel_politika", "kesin_cozucu", "kesin_politika"):
                row[f"{c}_ort_ms"] = sub[c].mean()
                row[f"{c}_p95_ms"] = sub[c].quantile(.95)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="test")
    ap.add_argument("--haric", default="", help="duyarlilik analizinde cikarilacak pencereler, ornek 4,134")
    a = ap.parse_args()
    out = a.out
    haric = [int(x) for x in a.haric.split(",") if x.strip()]
    R = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(out, "gorevler", "*.json"))])
    if R.empty:
        sys.exit("gorev sonucu yok")
    if "ort_islem" not in R:
        R["ort_islem"] = np.nan
    R["ort_islem"] = R.ort_islem.fillna(R.ort_akis - R.ort_bekleme)
    if "politika" in R and R.politika.nunique() > 1:
        print(f"UYARI: farkli politika surumleri karisik: {R.politika.unique()}", file=sys.stderr)

    # pencere x yuk x yontem: tohumlar uzerinden ortalama
    W = R.groupby(["gun", "rho", "yontem"])[[m for m, _ in METRICS]].mean().reset_index()
    long_rows, wide = [], []
    for (day, rho), g in W.groupby(["gun", "rho"]):
        g = g.set_index("yontem")
        sub = R[(R.gun == day) & (R.rho == rho)]
        row = {"gun": day, "rho": rho, "vm": int(sub.vm.iloc[0]), "is": int(sub["is"].iloc[0]),
               "tohum": int(sub.tohum.nunique())}
        for y, ny in METHODS:
            if y in g.index:
                for m, nm in METRICS:
                    row[f"{ny}_{nm}"] = g.loc[y, m]
        for y, ref in COMPARE:
            if y in g.index and ref in g.index:
                for m, nm in METRICS:
                    d = 100 * (g.loc[y, m] - g.loc[ref, m]) / g.loc[ref, m]
                    row[f"fark_{NAME[y]}_vs_{NAME[ref]}_{nm}_%"] = d
                    long_rows.append({"gun": day, "rho": rho, "yontem": NAME[y], "referans": NAME[ref],
                                      "olcut": nm, "fark_%": d})
        x = sub[sub.yontem == "exact"]
        if len(x):
            gd = 100 * (x.geri_dusus / (x.kesin + x.geri_dusus).clip(lower=1)).mean()
            kk = 100 * (x.K_siniri / x.kesin.clip(lower=1)).mean()
            row.update({"kesin_geri_dusus_%": gd, "kesin_K_%": kk, "kesin_guvenilir": bool(gd < 1 and kk < 1)})
        for y in ("agirlik", "yerel", "exact", "greedy"):
            z = sub[(sub.yontem == y) & (~sub.olcum)] if "olcum" in sub else sub[sub.yontem == y]
            if len(z):
                row[f"sure_{NAME[y]}_sn"] = z.calisma_sn.mean()
        dp = os.path.join(out, "kararlar", f"gun{day}_rho{rho}_greedy_s0.csv")
        if os.path.exists(dp):
            ol = open_loop(dp)
            row.update({("acik_" + k if not k.startswith("acik_") else k): v for k, v in ol.items()})
        wide.append(row)
    S = pd.DataFrame(wide).sort_values(["rho", "gun"])
    Lg = pd.DataFrame(long_rows)
    S.to_csv(os.path.join(out, "test_pencere_yuk.csv"), index=False)
    Lg.to_csv(os.path.join(out, "test_farklar_uzun.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    L = [__doc__.split("Kullanim")[1].split("\n", 2)[2].strip(), ""]
    summ_rows = []
    for rho, g in S.groupby("rho"):
        L += ["=" * 110, f"YUK rho = {rho}   (pencere sayisi {len(g)})", "=" * 110]
        cols = ["gun", "is", "vm"] + [f"{ny}_akis" for _, ny in METHODS if f"{ny}_akis" in g]
        L += ["1. Ortalama akis suresi (sn)", g[cols].round(0).to_string(index=False), ""]
        for nm in ("akis", "bekleme", "islem", "bsld"):
            cs = [f"fark_{NAME[y]}_vs_ECT_{nm}_%" for y in ("agirlik", "yerel", "exact", "fcfs")]
            cs = [c for c in cs if c in g]
            L += [f"2.{nm}  ECT'ye gore fark (%)", g[["gun"] + cs].round(2).to_string(index=False), ""]
        kc = [c for c in ["gun", "kesin_geri_dusus_%", "kesin_K_%", "kesin_guvenilir", "acik_kesin_cozulen_%",
                          "acik_bekleyen_medyan"] if c in g]
        L += ["3. Kesinlik kapsami", g[kc].round(2).to_string(index=False), ""]
        oc = [c for c in ["gun", "acik_tahmini_pay_kesin_%", "acik_kapatma_agirlik_%", "acik_kapatma_yerel_%",
                          "acik_optimal_agirlik_%", "acik_optimal_yerel_%", "acik_kalan_agirlik_%",
                          "acik_gercek_agirlik_%", "acik_gercek_yerel_%", "acik_gercek_kesin_%",
                          "acik_buyuk_pay_agirlik_%", "acik_ms_hizli_ort", "acik_ms_agirlik_ort",
                          "acik_ms_yerel_ort", "acik_ms_kesin_ort"] if c in g]
        L += ["4. Acik dongu (ECT gidisatindaki ayni karar durumlari)", g[oc].round(3).to_string(index=False), ""]
        L.append("5. Pencereler arasi ozet: fark % = yontem - referans; medyan [en kucuk, en buyuk]; "
                 "daha iyi = farki < 0 olan pencere sayisi; p = Wilcoxon (aciklayici)")
        for y, ref in COMPARE:
            for nm in ("akis", "bekleme", "islem", "bsld"):
                c = f"fark_{NAME[y]}_vs_{NAME[ref]}_{nm}_%"
                if c not in g or g[c].isna().all():
                    continue
                for alt, gg in (("tum", g), ("guvenilir", g[g.get("kesin_guvenilir", False) == True])):  # noqa: E712
                    if alt == "guvenilir" and (y != "exact" or "kesin_guvenilir" not in g):
                        continue
                    v = gg[c].dropna()
                    if not len(v):
                        continue
                    p = wil(v)
                    summ_rows.append({"rho": rho, "yontem": NAME[y], "referans": NAME[ref], "olcut": nm,
                                      "pencereler": alt, "n": len(v), "medyan_%": v.median(), "min_%": v.min(),
                                      "max_%": v.max(), "daha_iyi": int((v < 0).sum()), "p_wilcoxon": p})
                    L.append(f"  {NAME[y]:>7} vs {NAME[ref]:<7} {nm:<7} {alt:<9} n={len(v):>2}  "
                             f"{v.median():+8.3f}  [{v.min():+8.3f}, {v.max():+8.3f}]  "
                             f"daha iyi {int((v < 0).sum())}/{len(v)}  p={p:.3g}")
        L.append("")
    pd.DataFrame(summ_rows).to_csv(os.path.join(out, "test_pencereler_arasi.csv"), index=False)

    # 6. hesaplama maliyeti, ayni karar kapsaminda
    M = maliyet(out, S)
    if len(M):
        M.to_csv(os.path.join(out, "test_maliyet.csv"), index=False)
        cols = [c for c in M.columns if c.endswith("_ort_ms") or c.endswith("_p95_ms")]
        Mz = M.groupby(["rho", "kapsam"]).agg(pencere=("gun", "nunique"), karar=("karar", "sum"),
                                               **{c: (c, "median") for c in cols}).reset_index()
        L += ["=" * 110, "6. Karar basina hesaplama suresi (ms): pencere ortalamalarinin / p95'lerinin pencereler arasi medyani",
              "   kapsam kesin = kesin cozulen, K siniri baglayici olmayan kararlar (butun yontemler icin ayni karar kumesi);",
              "   K = K siniri baglayici; geri = bekleyen > cap; tum = butun olculen kararlar (karma politikanin maliyeti).",
              "   yerel_politika = ECT + yerel (agirlik dahil); kesin_politika = ECT + agirlik + cozucu; kesin_cozucu = yalniz cozucu.",
              "=" * 110]
        for kap in ("kesin", "K", "geri", "tum"):
            sub = Mz[Mz.kapsam == kap]
            if len(sub):
                L += [f"  kapsam = {kap}", sub.drop(columns="kapsam").round(2).to_string(index=False), ""]

    # 7. duyarlilik analizi
    if haric:
        Dz = duyarlilik(S, haric)
        Dz.to_csv(os.path.join(out, "test_duyarlilik.csv"), index=False)
        L += ["=" * 110, f"7. Duyarlilik analizi: {haric} cikarilmis (sonuclar goruldukten sonra eklenen ek analiz)",
              "   ECT'ye gore akis farki (%), tam hassasiyetli pencere degerlerinden; daha_iyi = fark < 0 olan pencere;",
              "   esit = fark tam olarak 0 olan pencere. Cikarma butun yontemlere ayni bicimde uygulanir.", "=" * 110]
        W = Dz.pivot_table(index=["rho", "yontem"], columns="pencereler",
                           values=["medyan_%", "daha_iyi", "esit", "n"], aggfunc="first")
        W.columns = [f"{a}_{b}" for a, b in W.columns]
        order = [c for c in ("n_tum", "medyan_%_tum", "daha_iyi_tum", "esit_tum",
                             "n_haric", "medyan_%_haric", "daha_iyi_haric", "esit_haric") if c in W]
        L += [W[order].round(4).to_string(), ""]
    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(out, "test_raporu.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")


if __name__ == "__main__":
    main()
