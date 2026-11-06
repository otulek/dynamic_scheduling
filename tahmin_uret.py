#!/usr/bin/env python3
"""
Gonderim aninda uretilebilen calisma suresi tahmini (soft_walltime yerine).

Kullanim:
  python tahmin_uret.py METACENTRUM-2023.swf --out tahmin_gonderim.csv.gz
  python tahmin_uret.py METACENTRUM-2023.swf --out tahmin_gonderim.csv.gz --gunler @test_pencereleri.txt

Neden: Izdeki soft_walltime, isin SON degeridir. PBS tahmini is baslarken yeniden hesaplar ve
tahmin asilirsa iki katina cikarir (Klusacek ve Chlumsky, JSSPP 2024, s. 4-5). Bu deger gonderim
aninda bilinemeyecek bilgi icerebilir. Bu betik ayni ureticiyi (Algorithm 1, s. 5) YALNIZ gonderim
aninda bilinen bilgiyle yeniden uretir:

  H_u(t)  = kullanicinin, gercek sistemde t'den ONCE tamamlanmis (bitis < t) en son 15 isi
  p_hat_i = min( W_i , W_i * max_{h in H_u(r_i)} p_h / W_h + 900 )      W = walltime_limit
  H_u bos ise p_hat_i = W_i

  - Gecmis, izdeki BUTUN isleri kapsar (cok cekirdekli, GPU'lu isler dahil): uretim sistemindeki
    ongorucu da kullanicinin butun tamamlanmis islerini gorur. Tahmin yalniz filtrelenmis
    (tek cekirdek, tek dugum, GPU yok) isler icin yazilir.
  - Tamamlanma zamani gercek sistemdeki zamandir: submit + wait + runtime. Simulasyondaki
    tamamlanma zamani kullanilmaz; boylece tahmin yonteme bagli olmaz ve butun yontemler ayni
    tahmini gorur. Simulasyonun bos baslamasi ongorucunun gecmisini bosaltmaz.
  - Bitis zamani gonderimle AYNI saniyede olan isler gecmise girmez (kesin esitsizlik).
  - PBS'teki baslangicta yeniden hesaplama ve asimda iki katina cikarma UYGULANMAZ; asimdan sonra
    simulator kendi kuralini (kalan sure walltime_limit'e gore) kullanir.
  - Gecmis izin basladigi ana (1 Ocak 2023) kadar uzanir; daha onceki isler izde yoktur. Yilin ilk
    gunlerinde gecmis kisadir ve tahminler uretim sistemindekinden farkli olabilir. Bu bir sizinti
    degil, gecmisin baslangic sinirdir; raporda gecmisi bos / 15'ten kisa islerin payi verilir.
  - Tahmin saniyeye yuvarlanir (np.rint; izdeki alanlar da tam saniyedir). Formulden fark en fazla
    0,5 sn'dir, ancak yuvarlama tahmin esitliklerini etkileyebilir; raporda yuvarlamanin yarattigi
    ek esitlikler sayilir. Bu, ureticinin bir yeniden uygulamasidir; uretim sisteminin ilk
    tahminlerinin birebir yeniden uretimi DEGILDIR (zamanlama, veritabani icerigi ve esitlik
    sirasi farkli olabilir).

Cikti: <out> (job, soft_pred, soft_kayit, walltime, runtime, gecmis) ve <out>_rapor.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

COLS = {0: "job", 1: "submit", 2: "wait", 3: "runtime", 4: "cores", 8: "walltime",
        11: "user", 18: "gpus", 19: "soft", 20: "topology"}
SON = 15          # gecmisteki is sayisi (Algorithm 1)
REZERV = 900.0    # sn (Algorithm 1)
DAY = 86400


def load(path, chunksize=1_000_000):
    parts = []
    for k, ch in enumerate(pd.read_csv(path, sep=r"\s+", comment=";", header=None, usecols=list(COLS),
                                       chunksize=chunksize, dtype={20: "string"})):
        ch = ch.rename(columns=COLS)
        nodes = pd.to_numeric(ch.pop("topology").str.split("x", n=1).str[0], errors="coerce")
        ch["filtre"] = (ch.cores == 1) & (nodes == 1) & (ch.gpus == 0)
        parts.append(ch.drop(columns=["cores", "gpus"]))
        print(f"  iz okunuyor: parca {k + 1}", file=sys.stderr)
    f = pd.concat(parts, ignore_index=True)
    for c in ("job", "submit", "wait", "runtime", "walltime", "user", "soft"):
        f[c] = f[c].astype("int64")
    return f


def uret(f):
    """Her is icin gonderim anindaki tahmin ve gecmisteki is sayisi."""
    n = len(f)
    sub = f.submit.to_numpy(np.float64)
    wall = f.walltime.to_numpy(np.float64)
    end = (f.submit + f.wait + f.runtime).to_numpy(np.float64)
    ratio = np.where(wall > 0, f.runtime.to_numpy(np.float64) / np.where(wall > 0, wall, 1.0), np.nan)
    # gecmise girebilecek isler: gecerli bitis ve oran
    hist_ok = (f.wait.to_numpy() >= 0) & (f.runtime.to_numpy() >= 0) & np.isfinite(ratio)
    pred = wall.copy()
    nh = np.zeros(n, dtype=np.int64)
    groups = f.groupby("user", sort=False).indices
    for k, (u, idx) in enumerate(groups.items()):
        h = idx[hist_ok[idx]]
        if len(h) == 0:
            continue
        o = np.argsort(end[h], kind="stable")
        e, r = end[h][o], ratio[h][o]
        # win[m] = max(r[m-SON+1 .. m])  -> son tamamlanan m+1 isin en buyuk SON'unun maks.
        pad = np.r_[np.full(SON - 1, -np.inf), r]
        win = sliding_window_view(pad, SON).max(axis=1)
        c = np.searchsorted(e, sub[idx], side="left")       # bitis < gonderim olan is sayisi
        has = c > 0
        mx = np.where(has, win[np.maximum(c - 1, 0)], np.nan)
        w = wall[idx]
        pred[idx] = np.where(has, np.minimum(w, w * mx + REZERV), w)
        nh[idx] = np.minimum(c, SON)
        if (k + 1) % 400 == 0:
            print(f"  kullanici {k + 1:,}/{len(groups):,}", file=sys.stderr)
    return pred, nh


def qline(s):
    s = pd.Series(s, dtype=float)
    q = s.quantile([.05, .25, .5, .75, .95])
    return ("  " + "  ".join(f"p{int(k * 100):02d}={v:,.4g}" for k, v in q.items())
            + f"  ort={s.mean():,.4g}")


def dogruluk(name, est, rt, P):
    est = np.asarray(est, float)
    rt = np.asarray(rt, float)
    ok = est > 0
    r = rt[ok] / est[ok]
    P(f"  {name}: oran = gercek / tahmin (n={ok.sum():,})")
    P(qline(r))
    P(f"    eksik tahmin (gercek > tahmin)      : %{100 * (r > 1).mean():.2f}")
    P(f"    tahmin > 10 x gercek               : %{100 * (r < 0.1).mean():.2f}")
    P(f"    gercek > 10 x tahmin               : %{100 * (r > 10).mean():.2f}")
    P(f"    Spearman(gercek, tahmin)           : {pd.Series(rt).corr(pd.Series(est), method='spearman'):.3f}")


def concordance(bucket, est, real, seed=0):
    """profile_windows.py ile ayni ornekleme: her is icin ayni kovadan rastgele bir es.
    Donus: cift sayisi, tahmini esit cift payi, tahminleri farkli ciftlerde dogru siralama payi."""
    rng = np.random.default_rng(seed)
    order = np.argsort(bucket, kind="stable")
    b, e, r = bucket[order], est[order], real[order]
    _, start, cnt = np.unique(b, return_index=True, return_counts=True)
    s = np.repeat(start, cnt)
    c = np.repeat(cnt, cnt)
    pos = np.arange(len(b))
    partner = s + (rng.random(len(b)) * c).astype(np.int64)
    ok = (partner != pos) & (r != r[partner])
    sr = np.sign(r[ok] - r[partner[ok]])
    se = np.sign(e[ok] - e[partner[ok]])
    n = int(ok.sum())
    if n == 0:
        return 0, float("nan"), float("nan")
    d = se != 0
    return n, float((~d).mean()), float((sr[d] == se[d]).mean()) if d.any() else float("nan")


def rapor(F, P, baslik):
    P("=" * 78)
    P(baslik)
    P("=" * 78)
    n = len(F)
    P(f"is: {n:,}")
    P(f"gecmisi olmayan (tahmin = walltime_limit)       : %{100 * (F.gecmis == 0).mean():.2f}")
    P(f"gecmisi 15'ten az                               : %{100 * (F.gecmis < SON).mean():.2f}")
    ilk = F.submit < 30 * DAY
    if ilk.any():
        P(f"  yilin ilk 30 gunu: is {int(ilk.sum()):,}, gecmisi bos %{100 * (F.gecmis[ilk] == 0).mean():.2f}, "
          f"15'ten az %{100 * (F.gecmis[ilk] < SON).mean():.2f}  (gecmis izin basinda baslar)")
    g = F.groupby("submit")
    ek = int((g.soft_pred.transform("nunique") < g.soft_ham.transform("nunique")).sum())
    P(f"yuvarlamanin ayni zaman damgasinda yeni esitlik yarattigi gruplardaki is: {ek:,}")
    P(f"soft_pred = walltime_limit                      : %{100 * (F.soft_pred == F.walltime).mean():.2f}")
    P(f"soft_kayit = walltime_limit                     : %{100 * (F.soft_kayit == F.walltime).mean():.2f}")
    d = F.soft_kayit - F.soft_pred
    P(f"kayitli = uretilen (|fark| <= 1 sn)             : %{100 * (d.abs() <= 1).mean():.2f}")
    P(f"kayitli > uretilen (uzatma ya da baslangicta yeniden hesap): %{100 * (d > 1).mean():.2f}")
    P(f"kayitli < uretilen                              : %{100 * (d < -1).mean():.2f}")
    P("tahmin dogrulugu:")
    dogruluk("soft_pred (gonderim)", F.soft_pred, F.runtime, P)
    dogruluk("soft_kayit (izdeki son deger)", F.soft_kayit, F.runtime, P)
    P("yarisan isler arasinda siralama (profile_windows.py ile ayni ornekleme, tohum 0):")
    sub, rt = F.submit.to_numpy(), F.runtime.to_numpy()
    for bname, bucket in (("ayni saat", sub // 3600), ("ayni zaman damgasi", sub)):
        for ename in ("soft_pred", "soft_kayit", "walltime"):
            k, tie, conc = concordance(bucket, F[ename].to_numpy(), rt)
            P(f"  {bname:<19} {ename:<11}: cift={k:,}  tahmini esit=%{100 * tie:.1f}  "
              f"farkli tahminlerde dogru siralama=%{100 * conc:.1f}")
    P("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("swf")
    ap.add_argument("--out", default="tahmin_gonderim.csv.gz")
    ap.add_argument("--gunler", default="", help="rapor icin ayrica bu gunler (ornek @test_pencereleri.txt)")
    a = ap.parse_args()
    f = load(a.swf)
    print(f"toplam is: {len(f):,}  filtreli: {int(f.filtre.sum()):,}", file=sys.stderr)
    ham, nh = uret(f)
    f["soft_pred"], f["soft_ham"], f["gecmis"] = np.rint(ham).astype(np.int64), ham, nh
    F = f.loc[f.filtre, ["job", "submit", "user", "walltime", "runtime", "soft", "soft_pred", "soft_ham", "gecmis"]]
    F = F.rename(columns={"soft": "soft_kayit"}).reset_index(drop=True)
    # tutarlilik denetimleri (gecersiz kayitlar yazilir ama simulasyonda kullanilirsa hata verir)
    assert F.job.is_unique, "is numaralari tekil olmali"
    assert ((F.soft_pred <= F.walltime) | (F.walltime <= 0)).all(), "tahmin walltime_limit'i asmamali"
    gecersiz = int(((F.walltime <= 0) | (F.soft_pred <= 0)).sum())
    if gecersiz:
        print(f"UYARI: walltime_limit <= 0 ya da tahmin <= 0 olan {gecersiz:,} filtreli is var; "
              f"bu isleri iceren pencereler simulasyonda hata verir", file=sys.stderr)
    F[["job", "soft_pred", "soft_kayit", "walltime", "runtime", "gecmis"]].to_csv(a.out, index=False)
    L = []
    P = L.append
    P(f"Gonderim aninda tahmin (Algorithm 1; son {SON} is, rezerv {REZERV:.0f} sn). Gecmis: butun isler; "
      f"tahmin: filtreli isler.")
    P(f"walltime_limit <= 0 olan filtreli is: {int((F.walltime <= 0).sum()):,}   "
      f"tahmin <= 0 olan: {int((F.soft_pred <= 0).sum()):,}")
    P("Tahmin saniyeye yuvarlanmistir (en fazla 0,5 sn fark). Bu, ureticinin bir yeniden uygulamasidir; "
      "uretimdeki ilk tahminlerin birebir kopyasi degildir.")
    P("")
    rapor(F, P, "A. TAM YIL (filtreli isler)")
    if a.gunler:
        spec = open(a.gunler[1:]).read().strip() if a.gunler.startswith("@") else a.gunler
        days = [int(x) for x in spec.split(",") if x.strip()]
        G = F[(F.submit // DAY).isin(days)]
        rapor(G, P, f"B. SECILEN GUNLER ({len(days)} gun: {', '.join(map(str, days))})")
    rep = "\n".join(L)
    print(rep)
    base = a.out[:-7] if a.out.endswith(".csv.gz") else os.path.splitext(a.out)[0]
    with open(base + "_rapor.txt", "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
    print(f"yazildi: {a.out}, {base}_rapor.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
