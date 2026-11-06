#!/usr/bin/env python3
"""
Dogrulama pencerelerini onceden belirlenmis, tekrar uretilebilir bir kuralla secer.

Kullanim:
  python pencere_sec.py pencereler_1gun.csv [--per 1] [--seed 2026] [--haric 0,51,364]
                        [--min-is 1000] [--max-is 20000]

Kural: is sayisi uc dilime (tertile) ve uzun is payi iki dilime (medyan) bolunur -> 6 tabaka.
Her tabakadan --per gun rastgele (--seed) secilir. Gun 51 pilotta kullanildigi icin varsayilan
olarak disarida birakilir (gelistirme penceresi test kumesine girmesin). Gun 0 iz baslangici,
gun 364 eksik son hafta nedeniyle disarida. --max-is, kesin planin calisma suresini sinirlar;
cok buyuk gunleri ayrica raporlamak icin bu siniri bilincli olarak degistirin.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--per", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--haric", default="0,51,364")
    ap.add_argument("--min-is", type=int, default=1000)
    ap.add_argument("--max-is", type=int, default=20000)
    ap.add_argument("--cikti", default="secilen_pencereler.txt")
    a = ap.parse_args()
    P = pd.read_csv(a.csv)
    excl = {int(x) for x in a.haric.split(",") if x}
    P = P[~P.pencere.isin(excl) & P.is_sayisi.between(a.min_is, a.max_is)].copy()
    P["is_dilim"] = pd.qcut(P.is_sayisi, 3, labels=["az", "orta", "cok"])
    P["uzun_dilim"] = (P.uzun_is_payi > P.uzun_is_payi.median()).map({False: "dusuk", True: "yuksek"})
    S = pd.concat([g.sample(min(a.per, len(g)), random_state=a.seed)
                   for _, g in P.groupby(["is_dilim", "uzun_dilim"], observed=True)])
    cols = ["pencere", "is_dilim", "uzun_dilim", "is_sayisi", "kapasite", "uzun_is_payi",
            "baskin_kullanici_payi", "en_buyuk_grup"]
    print(S[cols].sort_values("pencere").round(3).to_string(index=False))
    days = ",".join(str(int(x)) for x in sorted(S.pencere))
    print(f"\nAday havuzu: {len(P)} gun   secilen: {len(S)}")
    print(f"--days {days}")
    with open(a.cikti, "w") as fh:
        fh.write(days + "\n")
    S[cols].sort_values("pencere").to_csv(a.cikti.rsplit(".", 1)[0] + "_ozellikler.csv", index=False)
    print(f"yazildi: {a.cikti}")


if __name__ == "__main__":
    main()
