#!/usr/bin/env python3
"""
Kural karsilastirmasi ozeti: hizli kural (ECT), agirlik kurali, yerel arama ve kesin plan.

Kullanim:  python kural_ozet.py kurallar        (dogrulama.py --out kurallar ... ciktisi)

A. Acik dongu, ayni karar durumlari (ECT gidisati, tohum 0; yalniz kesin cozulen kararlar)
  kapatma_X_%  = sum(tahmin_hizli - tahmin_X) / sum(tahmin_hizli - tahmin_kesin) * 100
                 -> X, ECT ile kesin plan arasindaki tahmini farkin ne kadarini kapatti
                    (100 = kesin plana esit; ECT'den kotuyse negatif)
  kalan_X_%    = sum(tahmin_X - tahmin_kesin) / sum(tahmin_kesin) * 100   (kesin plana uzaklik)
  optimal_X_%  = tahmini maliyeti kesin planla esit olan karar orani
  gercek_X_%   = sum(gercek_hizli - gercek_X) / sum(gercek_hizli) * 100
                 -> ayni plan gercek surelerle (yeni gelis ve yeniden planlama olmadan) yurutulunce
                    ECT'ye gore kazanc. Kapali dongu performansini garanti etmez.
  ms_X         = karar basina planlama suresi (ortalama / p95). yerel = agirlik + arama suresi.
  Kesin cozulemeyen kararlarda (bekleyen > --cap) kesin referans yoktur; o kararlar icin yalniz
  ECT'ye gore tahmini kazanc verilir: buyuk_pay_X_%.
  Toplamlar "simdiden itibaren kalan akis suresi" cinsindendir; ardisik kararlar bagimsiz degildir.

B. Kapali dongu (eslestirilmis tohumlar; tohum 0 = bozulmamis iz, > 0 = gercek sureler %1 sarsilmis)
  fark_X_akis_% = (X - ECT) / ECT * 100, tohumlar uzerinden ortalama [min, max]; negatif = X iyi
  tutarli       = butun tohumlarda ayni isaret
  ECT_yayilma_% = ECT'nin tohumlar arasi (max - min) / ort: kucuk girdi sarsintisina duyarlilik
  kesin icin yalniz kendi kosusunda geri dusus ve K sinirinin %1'in altinda kaldigi hucreler
  "guvenilir" sayilir; digerlerinde kesin sutunu kismidir.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RULES = [("agirlik", "agirlik"), ("yerel", "yerel")]
CLOSED = [("exact", "kesin"), ("agirlik", "agirlik"), ("yerel", "yerel"), ("fcfs", "fcfs")]


def open_loop(path):
    D = pd.read_csv(path)
    out = {"karar": len(D), "kesin_cozulen_%": 100 * (D.durum == "kesin").mean() if len(D) else np.nan}
    if "tahmin_agirlik" not in D:
        return out
    E = D[D.durum == "kesin"]
    B = D[D.durum != "kesin"]
    gap = (E.tahmin_hizli - E.tahmin_kesin).sum()
    out["bekleyen_medyan"] = float(D.bekleyen.median())
    out["tahmini_pay_kesin_%"] = 100 * gap / E.tahmin_hizli.sum() if len(E) else np.nan
    out["gercek_kesin_%"] = 100 * (E.gercek_hizli - E.gercek_kesin).sum() / E.gercek_hizli.sum() if len(E) else np.nan
    for col, nm in RULES:
        te, tr = E[f"tahmin_{col}"], E[f"gercek_{col}"]
        if len(E):
            out[f"kapatma_{nm}_%"] = 100 * (E.tahmin_hizli - te).sum() / gap if gap > 0 else np.nan
            out[f"kalan_{nm}_%"] = 100 * (te - E.tahmin_kesin).sum() / E.tahmin_kesin.sum()
            out[f"optimal_{nm}_%"] = 100 * ((te - E.tahmin_kesin).abs()
                                            <= 1e-9 * E.tahmin_kesin.abs().clip(lower=1)).mean()
            out[f"gercek_{nm}_%"] = 100 * (E.gercek_hizli - tr).sum() / E.gercek_hizli.sum()
        if len(B):
            out[f"buyuk_pay_{nm}_%"] = 100 * (B.tahmin_hizli - B[f"tahmin_{col}"]).sum() / B.tahmin_hizli.sum()
    for c in ("hizli", "agirlik", "yerel", "kesin"):
        v = D[f"ms_{c}"].dropna()
        if len(v):
            out[f"ms_{c}_ort"] = float(v.mean())
            out[f"ms_{c}_p95"] = float(v.quantile(.95))
    out["yerel_hamle_ort"] = float(D.yerel_hamle.mean())
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "kurallar"
    R = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(out, "gorevler", "*.json"))])
    if R.empty:
        sys.exit("gorev sonucu yok")
    rows = []
    for (day, rho), g in R.groupby(["gun", "rho"]):
        row = {"gun": day, "rho": rho, "vm": int(g.vm.iloc[0]), "is": int(g["is"].iloc[0])}
        P = g.pivot_table(index="tohum", columns="yontem", values=["ort_akis", "ort_bsld"])
        for met, nm in (("ort_akis", "akis"), ("ort_bsld", "bsld")):
            if (met, "greedy") not in P:
                continue
            h = P[(met, "greedy")].dropna()
            row[f"ECT_{nm}"] = h.mean()
            row[f"ECT_yayilma_{nm}_%"] = 100 * (h.max() - h.min()) / h.mean() if len(h) > 1 else np.nan
            for y, ny in CLOSED:
                if (met, y) in P:
                    d = (100 * (P[(met, y)] - P[(met, "greedy")]) / P[(met, "greedy")]).dropna()
                    row[f"fark_{ny}_{nm}_%"] = d.mean()
                    row[f"fark_{ny}_{nm}_min"] = d.min()
                    row[f"fark_{ny}_{nm}_max"] = d.max()
                    row[f"tutarli_{ny}_{nm}"] = bool(len(d) > 1 and ((d > 0).all() or (d < 0).all()))
        x = g[g.yontem == "exact"]
        if len(x):
            gd = 100 * (x.geri_dusus / (x.kesin + x.geri_dusus).clip(lower=1)).mean()
            kk = 100 * (x.K_siniri / x.kesin.clip(lower=1)).mean()
            row["kesin_geri_dusus_%"], row["kesin_K_%"] = gd, kk
            row["kesin_guvenilir"] = bool(gd < 1 and kk < 1)
        for y in ("agirlik", "yerel", "exact"):
            z = g[g.yontem == y]
            if len(z):
                row[f"sure_{y}_sn"] = z.calisma_sn.mean()
        dp = os.path.join(out, "kararlar", f"gun{day}_rho{rho}_greedy_s0.csv")
        if os.path.exists(dp):
            row.update(open_loop(dp))
        rows.append(row)
    S = pd.DataFrame(rows).sort_values(["rho", "gun"])
    S.to_csv(os.path.join(out, "kural_ozet.csv"), index=False)

    A_cols = ["gun", "bekleyen_medyan", "kesin_cozulen_%", "tahmini_pay_kesin_%", "kapatma_agirlik_%",
              "kapatma_yerel_%", "kalan_agirlik_%", "kalan_yerel_%", "optimal_agirlik_%", "optimal_yerel_%",
              "gercek_kesin_%", "gercek_agirlik_%", "gercek_yerel_%", "buyuk_pay_agirlik_%", "buyuk_pay_yerel_%",
              "ms_hizli_ort", "ms_agirlik_ort", "ms_yerel_ort", "ms_yerel_p95", "ms_kesin_ort"]
    B_cols = ["gun", "ECT_akis", "ECT_yayilma_akis_%"]
    for _, ny in CLOSED:
        B_cols += [f"fark_{ny}_akis_%", f"fark_{ny}_akis_min", f"fark_{ny}_akis_max", f"tutarli_{ny}_akis"]
    B_cols += ["kesin_guvenilir", "fark_agirlik_bsld_%", "fark_yerel_bsld_%", "fark_kesin_bsld_%"]
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    L = [__doc__.split("Kullanim")[1].split("\n", 2)[2].strip(), ""]
    for rho, g in S.groupby("rho"):
        L += ["=" * 110, f"YUK rho = {rho}", "=" * 110]
        a = [c for c in A_cols if c in g]
        b = [c for c in B_cols if c in g]
        L += ["A. Acik dongu (ayni karar durumlari)", g[a].round(3).to_string(index=False), ""]
        L += ["B. Kapali dongu (ECT'ye gore, %)", g[b].round(3).to_string(index=False), ""]
        L.append("Pencereler arasi (medyan [min, max]):")
        for c in ["kapatma_agirlik_%", "kapatma_yerel_%", "kalan_yerel_%", "gercek_agirlik_%", "gercek_yerel_%",
                  "fark_agirlik_akis_%", "fark_yerel_akis_%", "fark_kesin_akis_%"]:
            if c in g and g[c].notna().any():
                v = g[c].dropna()
                L.append(f"  {c:<22} {v.median():9.3f}  [{v.min():.3f}, {v.max():.3f}]   n={len(v)}")
        for _, ny in CLOSED:
            c = f"fark_{ny}_akis_%"
            if c in g:
                x = g.dropna(subset=[c])
                tut = x[f"tutarli_{ny}_akis"]
                extra = ""
                if ny == "kesin" and "kesin_guvenilir" in x:
                    gv = x[x.kesin_guvenilir]
                    extra = f"; yalniz guvenilir hucreler: {int((gv[c] < 0).sum())} / {len(gv)}"
                L.append(f"  akis, {ny}: ECT'den iyi {int((x[c] < 0).sum())} / {len(x)} pencere; isareti tohumlarda "
                         f"tutarli ve iyi {int((tut & (x[c] < 0)).sum())}, tutarli ve kotu {int((tut & (x[c] > 0)).sum())}"
                         + extra)
        L.append("")
    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(out, "kural_raporu.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")


if __name__ == "__main__":
    main()
