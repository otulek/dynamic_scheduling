#!/usr/bin/env python3
"""
kayip_analizi.py ciktisindan (is duzeyi dosyalari) iki gidisatin (ECT, agirlik) belirli saatlerdeki
VM doluluğunu ve bu doluluğun hangi gelis dalgalarindan geldigini karsilastirir. Yeni simulasyon
kosmaz; yalniz kayitli is sonuclarini okur.

Kullanim:
  python durum_anlik.py --kok kayip177 --gun 177 --rho 0.8,1.2,2.0 --saatler 10,11,12,14,16,17,18,20

Soru: Sonraki gelis dalgasi geldiginde iki yontem hangi kaynaklari, ne kadar sureyle ve hangi
onceki is gruplariyla mesgul tutuyordu?
  1. VM dolulugu ve kalan gercek is (her saat T, hiz sinifi bazinda): mesgul VM; mesgul VM'lerde
     T'den sonra kalan gercek calisma (VM-saat); kalan suresi 1 / 4 / 12 saati asan mesgul VM sayisi;
     kalan surenin medyani ve 90. yuzdeligi; o anda bekleyen is.
     Kalan sure, o VM'de T'de calisan isin gercek bitisine kadardir (sonraki atamalar sayilmaz).
  2. Dolulugun kaynagi (her saat T): mesgul VM'ler, calisan isin GELIS SAATI grubuna gore; her grup
     icin sinif bazinda VM sayisi ve kalan VM-saat. --min-grup'tan kucuk gruplar "diger" altinda.
  3. Gelis dalgalarinin siniflara dagilimi: her dalga (gelis saati grubu) iki gidisatta hangi hiz
     siniflarinda calisti; sinif bazinda tukettigi VM-saat (islem suresi toplami); ortalama bekleme
     ve islem; ilk ve son baslama; T saatlerinde hala calisan is sayisi.
Bunlar betimleyicidir: bir durum farkinin kaybi aciklayip aciklamadigi ayri bir mudahaleyle
(gecis deneyi gibi) sinanmalidir. Isler tek cekirdekli oldugundan "calisan is" = "mesgul VM".
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HOUR, DAY = 3600.0, 86400.0
AD = (("ECT", "greedy"), ("agirlik", "agirlik"))


def yukle(kok, rho, tohum):
    f = lambda b: os.path.join(kok, "isler", f"r{rho}_{b}_s{tohum}.csv.gz")  # noqa: E731
    J = {ad: pd.read_csv(f(b)) for ad, b in AD}
    meta = json.load(open(os.path.join(kok, "gorevler", f"r{rho}_greedy_s{tohum}.json")))
    return J, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kok", default="kayip177")
    ap.add_argument("--gun", type=int, default=177)
    ap.add_argument("--rho", default="0.8,1.2,2.0")
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--saatler", default="10,11,12,14,16,17,18,20")
    ap.add_argument("--min-grup", type=int, default=300, help="gelis saati grubu en az bu kadar is (yoksa 'diger')")
    a = ap.parse_args()
    t0 = a.gun * DAY
    saat = [float(x) for x in a.saatler.split(",")]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    L = [f"DURUM KARSILASTIRMASI: gun {a.gun}, tohum {a.tohum}", ""]
    for rho in [float(x) for x in a.rho.split(",")]:
        try:
            J, meta = yukle(a.kok, rho, a.tohum)
        except FileNotFoundError as e:
            L.append(f"rho {rho}: dosya yok ({e.filename})")
            continue
        speeds, m_cls = meta["hizlar"], meta["vm_sinif"]
        L += ["#" * 110, f"rho = {rho}   VM {meta['vm']} = sinif " +
              ", ".join(f"hiz {s:g}: {m}" for s, m in zip(speeds, m_cls)), "#" * 110]
        for X in J.values():
            X["gelis_saati"] = np.floor((X.submit - t0) / HOUR).astype(int)
        cnt = J["ECT"].gelis_saati.value_counts()
        dalgalar = sorted(cnt[cnt >= a.min_grup].index)
        for X in J.values():
            X["grup"] = np.where(X.gelis_saati.isin(dalgalar), X.gelis_saati.astype(str), "diger")

        # ---- 1. VM dolulugu ve kalan gercek is
        rows = []
        for h in saat:
            T = t0 + h * HOUR
            for ad, X in J.items():
                run = (X.start <= T) & (X.end > T)
                for s, mc in zip(speeds, m_cls):
                    kal = ((X.end - T) / HOUR)[run & (X.hiz == s)]
                    rows.append({"saat": h, "yontem": ad, "hiz": s, "VM": mc, "mesgul": len(kal),
                                 "mesgul_%": 100 * len(kal) / mc, "kalan_vmsaat": kal.sum(),
                                 "kalan>1sa": int((kal > 1).sum()), "kalan>4sa": int((kal > 4).sum()),
                                 "kalan>12sa": int((kal > 12).sum()),
                                 "kalan_medyan_sa": kal.median() if len(kal) else np.nan,
                                 "kalan_p90_sa": kal.quantile(.9) if len(kal) else np.nan})
                rows.append({"saat": h, "yontem": ad, "hiz": "bekleyen",
                             "mesgul": int(((X.submit <= T) & (X.start > T)).sum())})
        S = pd.DataFrame(rows)
        for c in ("VM", "kalan>1sa", "kalan>4sa", "kalan>12sa"):
            S[c] = S[c].astype("Int64")
        S.to_csv(os.path.join(a.kok, f"doluluk_rho{rho}.csv"), index=False)
        L += ["1. VM dolulugu ve kalan gercek is (kalan: T'de calisan isin gercek bitisine kadar, saat; "
              "'bekleyen' satirinda mesgul sutunu = bekleyen is)"]
        for h in saat:
            sub = S[S.saat == h].drop(columns="saat")
            L += [f"  T = {h:g}. saat", sub.round(1).to_string(index=False), ""]

        # ---- 2. dolulugun kaynagi
        rows = []
        for h in saat:
            T = t0 + h * HOUR
            for ad, X in J.items():
                R = X[(X.start <= T) & (X.end > T)]
                for g, G in R.groupby("grup"):
                    r = {"saat": h, "yontem": ad, "gelis_grubu": g}
                    for s in speeds:
                        k = G.hiz == s
                        r[f"vm_hiz{s:g}"] = int(k.sum())
                        r[f"kalan_vmsaat_hiz{s:g}"] = float(((G.end - T)[k]).sum() / HOUR)
                    rows.append(r)
        O = pd.DataFrame(rows)
        if len(O):
            O["_s"] = O.gelis_grubu.map(lambda g: 1e9 if g == "diger" else int(g))
            O = O.sort_values(["saat", "_s", "yontem"]).drop(columns="_s")
            O.to_csv(os.path.join(a.kok, f"doluluk_kaynagi_rho{rho}.csv"), index=False)
            L += [f"2. Dolulugun kaynagi: T'de mesgul VM'ler, calisan isin gelis saati grubuna gore "
                  f"(>= {a.min_grup} is; digerleri 'diger')"]
            for h in saat:
                sub = O[O.saat == h].drop(columns="saat")
                if len(sub):
                    L += [f"  T = {h:g}. saat", sub.round(1).to_string(index=False), ""]

        # ---- 3. gelis dalgalarinin siniflara dagilimi
        rows = []
        for g in dalgalar:
            for ad, X in J.items():
                G = X[X.gelis_saati == g]
                r = {"dalga_saati": g, "yontem": ad, "is": len(G),
                     "ort_bekleme_sn": G.bekleme.mean(), "ort_islem_sn": G.islem.mean(),
                     "ilk_baslama_sa": (G.start.min() - t0) / HOUR, "son_baslama_sa": (G.start.max() - t0) / HOUR}
                for s in speeds:
                    k = G.hiz == s
                    r[f"pay_hiz{s:g}_%"] = 100 * k.mean()
                    r[f"vmsaat_hiz{s:g}"] = G.islem[k].sum() / HOUR
                for h in saat:
                    T = t0 + h * HOUR
                    if T >= G.submit.min():
                        r[f"calisan@{h:g}"] = int(((G.start <= T) & (G.end > T)).sum())
                rows.append(r)
        W = pd.DataFrame(rows)
        W.to_csv(os.path.join(a.kok, f"dalgalar_rho{rho}.csv"), index=False)
        L += [f"3. Gelis dalgalarinin siniflara dagilimi (>= {a.min_grup} is); vmsaat_hiz: o sinifta tuketilen "
              "VM-saat (islem suresi toplami); calisan@T: o saatte hala calisan is",
              W.round(1).to_string(index=False), ""]

    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(a.kok, "durum_raporu.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")


if __name__ == "__main__":
    main()
