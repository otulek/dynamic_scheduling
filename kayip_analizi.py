#!/usr/bin/env python3
"""
Bir pencerede agirlik kuralinin ECT'ye gore kaybini (ornek: 177. gun) is ve karar duzeyinde inceler.
Pencere dislanmaz; amac, ortalama akistaki farkin HANGI isler ve HANGI kararlar uzerinden olustugunu
gostermektir. Politikalar dogrulama.py / kurallar.py ile aynidir (POLITIKA_SURUM); hicbiri degismez.

Kullanim (dogrulama.py ile ayni klasorde; pencere CSV'si daha once test kosusunda cikarilmis olmali):
  python kayip_analizi.py --pencere-kok test --gun 177 --rho 0.8,1.2,2.0 --out kayip177 --workers 4
  -> kayip177/kayip_raporu.txt ve CSV'ler. Yarida kalirsa ayni komutla devam eder.

Uc parca:
  A. Is duzeyi (gozlem). Iki tam kosu (ECT, agirlik) ayni pencerede, ayni tohumla. Her is icin
     fark = akis(agirlik) - akis(ECT) = bekleme farki + islem farki. Kirilimlar: VM hiz sinifi
     gecisi, tahmin dogrulugu (runtime/soft), tahmini sure, gelis saati, kullanici; kaybin kac iste
     yogunlastigi. Isler arasi farklar birbirinden bagimsiz DEGILDIR (bir isin yeri digerlerinin
     beklemesini degistirir); bu kirilimlar kaybin nerede GORUNDUGUNU soyler, nedenini kanitlamaz.
  B. Kapasite kullanimi (gozlem). Kuyruk bosken degil, en az bir is beklerken bos kalan VM-saat,
     hiz sinifina gore; saatlik ortalama bekleyen is sayisi.
  C. Gecis deneyi (mudahale). H_T: T saatine kadar agirlik, sonra ECT. G_T: T'ye kadar ECT, sonra
     agirlik. H_T - ECT, T'den once verilen agirlik kararlarinin (sonraki etkileriyle birlikte)
     yarattigi farktir; T ilerledikce hangi donemdeki kararlarin nihai farki degistirdigi gorulur. Etkiler
     toplanabilir olmak zorunda degildir; H_T + G_T'nin toplamla uyumu ayrica raporlanir.
  D. Karar duzeyi (acik dongu). Her iki tam kosunun gidisatindaki her kararda (>= 2 bekleyen is)
     ECT ve agirlik planlari AYNI durumda hesaplanir: tahmini maliyet, gercek surelerle bir kez
     yurutulen maliyet, simdi baslatilan is sayisi, bos birakilan VM, ayni tahminli is payi.

Tohumlar: --seeds 0,1,2 (0 = izdeki sureler; s > 0 dogrulama.py ile ayni %1 sarsinti). Gecis deneyi
yalniz --gecis-tohum (varsayilan 0) ile kosar.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dogrulama as dg  # noqa: E402
import kurallar  # noqa: E402

AD = {"greedy": "ECT", "agirlik": "agirlik"}
HOUR = 3600.0


# ============================================================================ kurulum
def kur(cfg, rho, seed):
    """dogrulama.run_task ile ayni pencere, VM sayisi ve tohum sarsintisi."""
    J = pd.read_csv(dg.window_path(cfg["pkok"], cfg["gun"], cfg["ndays"]))
    J = J.sort_values(["submit", "job"], kind="stable").reset_index(drop=True)
    J = dg.tahmin_uygula(J, cfg.get("tahmin", ""))
    classes = dg.parse_classes(cfg["classes"])
    Lw = J.runtime.sum() / (cfg["ndays"] * dg.DAY)
    m = max(1, round(Lw / rho / sum(s * w for s, w in classes)))
    if seed > 0:
        g = np.random.default_rng([cfg["gun"], seed])
        r = J.runtime.to_numpy(float) * np.exp(g.normal(0.0, cfg["jitter"], len(J)))
        J["runtime"] = np.clip(r, 1.0, np.maximum(J.walltime.to_numpy(float), J.soft.to_numpy(float)))
    speed, bounds, cls_speed = dg.build_vms(classes, m)
    return J, speed, bounds, cls_speed


def plan_of(name, s):
    if name == "greedy":
        return dg.greedy(s["d"], s["avail"], s["cls_speed"], s["bounds"], s["rk"])
    return kurallar.agirlik_kurali(s["d"], s["avail"], s["cls_speed"], s["bounds"], s["rk"])


def tag_of(rho, base, to, T, seed):
    return f"r{rho}_{base}" + (f"_{to}T{T:g}" if to else "") + f"_s{seed}"


def run(args):
    """base yontemiyle kosar; to verilmisse t >= T kararlarinda 'to' yontemine gecer.
    Gecis, kanca icinde plan dizisinin YERINDE degistirilmesiyle yapilir (simulate bu diziyi kullanir);
    tek bekleyen isli kararlarda butun kurallar aynidir ve kanca cagrilmaz."""
    rho, base, to, T, seed, log, cfg = args
    J, speed, bounds, cls_speed = kur(cfg, rho, seed)
    t_sw = cfg["gun"] * dg.DAY + T * HOUR if to else np.inf
    rows = []
    t0w = cfg["gun"] * dg.DAY

    def hook(s):
        R = s["planlar"]
        vms = R["plan"][base]
        if s["t"] >= t_sw:
            vms[:] = plan_of(to, s)
        if not log:
            return
        d, av, avr, dr, sp = s["d"], s["avail"], s["avail_gercek"], s["d_gercek"], s["speed"]
        PE = R["plan"]["greedy"]
        PA = R["plan"]["agirlik"] if "agirlik" in R["plan"] else plan_of("agirlik", s)
        idle = avr <= 0
        bE, bA = np.unique(PE[idle[PE]]), np.unique(PA[idle[PA]])
        diff, _, _ = kurallar.anlamli_fark(PE, PA, d, av, sp)
        same = np.r_[d[1:] == d[:-1], False] | np.r_[False, d[1:] == d[:-1]]
        rows.append({
            "saat": (s["t"] - t0w) / HOUR, "bekleyen": len(d), "bos_vm": int(idle.sum()),
            "tahmin_ECT": kurallar.plan_cost(PE, d, av, sp), "tahmin_agirlik": kurallar.plan_cost(PA, d, av, sp),
            "gercek_ECT": kurallar.plan_cost(PE, dr, avr, sp), "gercek_agirlik": kurallar.plan_cost(PA, dr, avr, sp),
            "simdi_ECT": len(bE), "simdi_agirlik": len(bA),
            "simdi_hiz_ECT": float(sp[bE].sum()), "simdi_hiz_agirlik": float(sp[bA].sum()),
            "farkli_is": int(diff.sum()), "esit_tahmin_payi": float(same.mean()),
            "uygulanan": AD[to] if s["t"] >= t_sw else AD[base]})

    summ, _, jobs = dg.simulate(J, speed, bounds, cls_speed, base, seed, cfg, hook=hook)
    jobs["user"] = J.user.to_numpy()
    jobs["walltime"] = J.walltime.to_numpy()
    tag = tag_of(rho, base, to, T, seed)
    if log:
        jobs.to_csv(os.path.join(cfg["out"], "isler", tag + ".csv.gz"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(cfg["out"], "kararlar", tag + ".csv.gz"), index=False)
    m_cls = np.diff(bounds).tolist()
    rec = {"rho": rho, "base": base, "to": to, "T": T, "tohum": seed, "vm": int(len(speed)),
           "vm_sinif": m_cls, "hizlar": cls_speed.tolist(), "politika": kurallar.POLITIKA_SURUM,
           **{k: summ[k] for k in ("is", "ort_akis", "ort_bekleme", "ort_islem", "ort_bsld", "karar", "calisma_sn")}}
    with open(os.path.join(cfg["out"], "gorevler", tag + ".json"), "w") as fh:
        json.dump(rec, fh)
    return tag, rec


# ============================================================================ analiz yardimcilari
def kapasite(jobs, speeds, m_cls, t_lo):
    """Parca parca sabit zaman cizelgesi: en az bir is beklerken bos kalan VM-saat (sinif bazinda),
    saatlik ortalama bekleyen is. Bekleme suresi 0 olan isler kuyruga girmis sayilmaz."""
    sub, st, en, hz = (jobs[c].to_numpy(float) for c in ("submit", "start", "end", "hiz"))
    # saat sinirlari da kirilma noktasi: hicbir aralik bir saat sinirini asmaz (saatlik ortalamalar dogru)
    grid = t_lo + HOUR * np.arange(0, np.floor((en.max() - t_lo) / HOUR) + 2)
    tau = np.unique(np.r_[sub, st, en, grid])
    dt = np.diff(tau)
    a = tau[:-1]
    W = np.searchsorted(np.sort(sub), a, "right") - np.searchsorted(np.sort(st), a, "right")
    out, idle_tot = {}, np.zeros(len(a))
    for s, mc in zip(speeds, m_cls):
        k = hz == s
        B = np.searchsorted(np.sort(st[k]), a, "right") - np.searchsorted(np.sort(en[k]), a, "right")
        idle = mc - B
        idle_tot += idle
        out[f"bos_kuyruk_vmsaat_{s:g}"] = float(((W > 0) * idle * dt).sum() / HOUR)
    out["bos_kuyruk_vmsaat"] = float(((W > 0) * idle_tot * dt).sum() / HOUR)
    out["kullanilabilir_kayip_issaat"] = float((np.minimum(W, idle_tot) * dt).sum() / HOUR)
    out["kuyruklu_sure_saat"] = float(((W > 0) * dt).sum() / HOUR)
    # saatlik ortalama bekleyen is
    h = np.floor((a - t_lo) / HOUR).astype(int)
    hourly = pd.Series(W * dt).groupby(h).sum() / HOUR
    return out, hourly


def yogunlasma(delta):
    """Net farkin yuzde 50/80/100'une ulasmak icin, net farkla ayni yondeki en buyuk farklardan kac is
    gerektigi (net fark pozitifse kayiplar, negatifse kazanclar)."""
    delta = np.sign(delta.sum()) * delta
    tot = delta.sum()
    pos = np.sort(delta[delta > 0])[::-1]
    cs = np.cumsum(pos)
    res = {}
    for q in (0.5, 0.8, 1.0):
        k = int(np.searchsorted(cs, q * tot) + 1) if tot > 0 and cs[-1] >= q * tot else np.nan
        res[f"%{int(q * 100)}"] = k
    return res


def tablo(df, by, delta, n_tot, tot):
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({"is": g.size(), "pay_is_%": 100 * g.size() / n_tot,
                      "ort_fark_akis_sn": g[delta].mean(), "toplam_fark_saat": g[delta].sum() / HOUR,
                      "net_farka_pay_%": 100 * g[delta].sum() / tot if tot else np.nan,
                      "ort_fark_bekleme_sn": g["fark_bekleme"].mean(), "ort_fark_islem_sn": g["fark_islem"].mean()})
    return t


# ============================================================================ rapor
def rapor(cfg, rhos, seeds, T_grid):
    out, L = cfg["out"], []
    R = pd.DataFrame([json.load(open(os.path.join(out, "gorevler", f)))
                      for f in os.listdir(os.path.join(out, "gorevler"))])
    R["to"] = R["to"].fillna("")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    t_lo = cfg["gun"] * dg.DAY
    L += [f"KAYIP ANALIZI: gun {cfg['gun']}   politika {kurallar.POLITIKA_SURUM}", ""]
    L += ["Fark = agirlik - ECT; pozitif = agirlik daha kotu. net_farka_pay_%: grubun toplam farki / butun islerin",
          "toplam farki (net fark pozitifken pozitif pay = kayba katki). Is duzeyi kirilimlar gozlemdir: isler arasi",
          "etkilesim nedeniyle bir gruptaki fark o grubun 'neden oldugu' kayip olarak okunmamali.", ""]
    full = R[R.to == ""]

    # ---- 0. tohumlara gore toplam fark
    L += ["=" * 100, "0. Toplam fark, tohumlara gore (ortalama akis, %)", "=" * 100]
    rows = []
    for (rho, s), g in full.groupby(["rho", "tohum"]):
        g = g.set_index("base")
        if {"greedy", "agirlik"} <= set(g.index):
            e, a = g.loc["greedy"], g.loc["agirlik"]
            rows.append({"rho": rho, "tohum": s, "is": int(e["is"]), "vm": int(e.vm),
                         "ECT_akis": e.ort_akis, "agirlik_akis": a.ort_akis,
                         "fark_akis_%": 100 * (a.ort_akis / e.ort_akis - 1),
                         "fark_bekleme_%": 100 * (a.ort_bekleme / e.ort_bekleme - 1),
                         "fark_islem_%": 100 * (a.ort_islem / e.ort_islem - 1),
                         "fark_bsld_%": 100 * (a.ort_bsld / e.ort_bsld - 1)})
    T0 = pd.DataFrame(rows)
    L += [T0.round(2).to_string(index=False), ""]
    T0.to_csv(os.path.join(out, "toplam_fark.csv"), index=False)

    for rho in rhos:
        s0 = seeds[0]
        pe = os.path.join(out, "isler", tag_of(rho, "greedy", None, None, s0) + ".csv.gz")
        pa = os.path.join(out, "isler", tag_of(rho, "agirlik", None, None, s0) + ".csv.gz")
        if not (os.path.exists(pe) and os.path.exists(pa)):
            continue
        E, A = pd.read_csv(pe), pd.read_csv(pa)
        meta = full[(full.rho == rho) & (full.tohum == s0)].iloc[0]
        speeds, m_cls = meta.hizlar, meta.vm_sinif
        D = E.merge(A, on="job", suffixes=("_E", "_A"))
        for c in ("submit", "soft", "runtime", "user", "walltime"):
            D[c] = D[c + "_E"]
        D["fark_akis"] = D.akis_A - D.akis_E
        D["fark_bekleme"] = D.bekleme_A - D.bekleme_E
        D["fark_islem"] = D.islem_A - D.islem_E
        n, tot = len(D), D.fark_akis.sum()
        L += ["", "#" * 100, f"rho = {rho}   (tohum {s0}; is {n}; VM {int(meta.vm)} = sinif {m_cls})", "#" * 100]

        # ---- A1 ozet
        L += ["A1. Is basina ortalama fark (sn) ve bilesenleri",
              f"  akis {D.fark_akis.mean():+.1f} = bekleme {D.fark_bekleme.mean():+.1f} + islem {D.fark_islem.mean():+.1f}"
              f"   (ECT ort. akis {D.akis_E.mean():.0f} sn; fark %{100 * tot / D.akis_E.sum():+.2f})",
              f"  akisi agirlikta daha uzun is: {int((D.fark_akis > 1).sum())}  daha kisa: {int((D.fark_akis < -1).sum())}"
              f"  |fark| <= 1 sn: {int((D.fark_akis.abs() <= 1).sum())}",
              f"  ayni VM hiz sinifinda calisan is: {int((D.hiz_E == D.hiz_A).sum())}/{n}",
              "  yogunlasma: net farkin %50/%80/%100'u icin, ayni yondeki en buyuk farklardan gereken is sayisi: "
              + ", ".join(f"{k}={v}" for k, v in yogunlasma(D.fark_akis.to_numpy()).items()), ""]

        # ---- A2 hiz sinifi gecisi
        D["sinif_gecisi"] = D.hiz_E.map("{:g}".format) + " -> " + D.hiz_A.map("{:g}".format)
        L += ["A2. VM hiz sinifi: ECT'de -> agirlikta", tablo(D, "sinif_gecisi", "fark_akis", n, tot).round(2).to_string(), ""]

        # ---- A3 tahmin dogrulugu
        ratio = D.runtime / D.soft.clip(lower=1)
        D["dogruluk"] = pd.cut(ratio, [0, 0.1, 0.5, 0.9, 1.0 - 1e-12, 1.0 + 1e-12, np.inf],
                               labels=["<0,1", "0,1-0,5", "0,5-0,9", "0,9-1", "=1 (soft'a esit)", ">1 (asim)"],
                               include_lowest=True)
        L += ["A3. Tahmin dogrulugu (runtime / soft_walltime)", tablo(D, "dogruluk", "fark_akis", n, tot).round(2).to_string(), ""]

        # ---- A4 tahmini sure ve gercek sure
        D["soft_grup"] = pd.cut(D.soft, [0, 600, 3600, 4 * 3600, 24 * 3600, 3 * 24 * 3600, np.inf],
                                labels=["<=10dk", "10dk-1sa", "1-4sa", "4-24sa", "1-3gun", ">3gun"])
        D["runtime_grup"] = pd.cut(D.runtime, [0, 60, 600, 3600, 4 * 3600, 24 * 3600, np.inf],
                                   labels=["<=1dk", "1-10dk", "10dk-1sa", "1-4sa", "4-24sa", ">1gun"], include_lowest=True)
        L += ["A4a. Tahmini sure (soft_walltime)", tablo(D, "soft_grup", "fark_akis", n, tot).round(2).to_string(), ""]
        L += ["A4b. Gercek sure (runtime)", tablo(D, "runtime_grup", "fark_akis", n, tot).round(2).to_string(), ""]
        vc = D.soft.value_counts()
        L += [f"  en sik 5 tahmin degeri (sn: is sayisi): " + ", ".join(f"{int(k)}: {v}" for k, v in vc.head(5).items()), ""]

        # ---- A5 gelis saati + B saatlik kuyruk + C gecis egrisi
        D["gelis_saati"] = np.floor((D.submit - t_lo) / HOUR).astype(int)
        H = tablo(D, "gelis_saati", "fark_akis", n, tot)[["is", "ort_fark_akis_sn", "toplam_fark_saat", "net_farka_pay_%"]]
        H["kumulatif_pay_%"] = H["net_farka_pay_%"].cumsum()
        capE, hE = kapasite(E, speeds, m_cls, t_lo)
        capA, hA = kapasite(A, speeds, m_cls, t_lo)
        H["ort_bekleyen_ECT"] = hE.reindex(H.index)
        H["ort_bekleyen_agirlik"] = hA.reindex(H.index)
        L += ["A5. Gelis saatine gore (pencere basindan) is duzeyi fark; B: o saatte ortalama bekleyen is",
              H.round(2).to_string(), ""]

        # ---- A6 kullanici
        U = tablo(D, "user", "fark_akis", n, tot).sort_values("toplam_fark_saat", ascending=False)
        L += ["A6. Kullanici (net kayba katkisi en buyuk 8 ve en kucuk 3)",
              pd.concat([U.head(8), U.tail(3)]).round(2).to_string(), ""]

        # ---- A7 en buyuk farkli isler
        cols = ["job", "user", "submit", "soft", "runtime", "hiz_E", "hiz_A", "bekleme_E", "bekleme_A", "fark_akis"]
        top = D.sort_values("fark_akis", ascending=False)[cols].head(15).copy()
        top["submit"] = (top.submit - t_lo) / HOUR
        L += ["A7. Agirlikta en cok geciken 15 is (submit: saat)", top.round(1).to_string(index=False), ""]
        top = D.sort_values("fark_akis")[cols].head(5).copy()
        top["submit"] = (top.submit - t_lo) / HOUR
        L += ["    Agirlikta en cok kazanan 5 is", top.round(1).to_string(index=False), ""]

        # ---- B kapasite
        Bk = pd.DataFrame([capE, capA], index=["ECT", "agirlik"]).T
        L += ["B. Kuyruk doluyken bos kapasite",
              "   bos_kuyruk_vmsaat: en az bir is beklerken bos VM-saat (sinif bazinda da);",
              "   kullanilabilir_kayip_issaat: integral min(bekleyen, bos VM) -> baslatilabilecekken bekleyen is-saat",
              Bk.round(2).to_string(), ""]

        # ---- C gecis egrisi
        sw = R[(R.rho == rho) & (R.tohum == cfg["gecis_tohum"])]
        fE = full[(full.rho == rho) & (full.tohum == cfg["gecis_tohum"]) & (full.base == "greedy")]
        fA = full[(full.rho == rho) & (full.tohum == cfg["gecis_tohum"]) & (full.base == "agirlik")]
        if len(fE) and len(fA):
            e0, a0 = float(fE.ort_akis.iloc[0]), float(fA.ort_akis.iloc[0])
            rows = []
            for T in T_grid:
                h = sw[(sw.base == "agirlik") & (sw.to == "greedy") & (sw["T"] == T)]
                gg = sw[(sw.base == "greedy") & (sw.to == "agirlik") & (sw["T"] == T)]
                hv = e0 if T == 0 else (float(h.ort_akis.iloc[0]) if len(h) else np.nan)
                gv = a0 if T == 0 else (float(gg.ort_akis.iloc[0]) if len(gg) else np.nan)
                rows.append({"T_saat": T, "H_T_fark_%": 100 * (hv / e0 - 1), "G_T_fark_%": 100 * (gv / e0 - 1),
                             "H_T_toplam_kayba_orani_%": 100 * (hv - e0) / (a0 - e0) if a0 != e0 else np.nan,
                             "toplanabilirlik_%": 100 * ((hv - e0) + (gv - e0)) / (a0 - e0) if a0 != e0 else np.nan})
            rows.append({"T_saat": np.inf, "H_T_fark_%": 100 * (a0 / e0 - 1), "G_T_fark_%": 0.0,
                         "H_T_toplam_kayba_orani_%": 100.0, "toplanabilirlik_%": 100.0})
            C = pd.DataFrame(rows)
            C.to_csv(os.path.join(out, f"gecis_rho{rho}.csv"), index=False)
            L += [f"C. Gecis deneyi (tohum {cfg['gecis_tohum']}). H_T: T'ye kadar agirlik sonra ECT; G_T: T'ye kadar ECT sonra agirlik.",
                  "   H_T_toplam_kayba_orani_%: (H_T - ECT) / (agirlik - ECT). T'de politika degistirilen kosunun nihai",
                  "   farkinin tam agirlik kosusunun farkina orani; donemlere ait bagimsiz bir pay DEGILDIR;",
                  "   toplanabilirlik_% = (H_T + G_T farklari) / toplam fark; 100'den uzaklik etkilesimi gosterir.",
                  C.round(2).to_string(index=False), ""]

        # ---- D karar duzeyi
        for base in ("greedy", "agirlik"):
            pk = os.path.join(out, "kararlar", tag_of(rho, base, None, None, s0) + ".csv.gz")
            if not os.path.exists(pk):
                continue
            K = pd.read_csv(pk)
            if K.empty:
                continue
            K["tahmin_fark"] = K.tahmin_agirlik - K.tahmin_ECT
            K["gercek_fark"] = K.gercek_agirlik - K.gercek_ECT
            K["saat_i"] = np.floor(K.saat).astype(int)
            K["simdi_fark"] = K.simdi_agirlik - K.simdi_ECT
            dif = K[K.farkli_is > 0]
            L += [f"D. Karar duzeyi, {AD[base]} gidisati (acik dongu; >= 2 bekleyen isli {len(K)} karar)",
                  f"   planlarin anlamli farkli oldugu karar: {len(dif)}; bunlarda agirlik tahminde daha iyi: "
                  f"{int((dif.tahmin_fark < 0).sum())}, gercekte daha iyi: {int((dif.gercek_fark < 0).sum())}, "
                  f"gercekte daha kotu: {int((dif.gercek_fark > 0).sum())}",
                  f"   agirlik simdi daha AZ is baslatiyor: {int((K.simdi_fark < 0).sum())} karar; daha cok: "
                  f"{int((K.simdi_fark > 0).sum())}; simdi baslatilan toplam hiz farki "
                  f"{(K.simdi_hiz_agirlik - K.simdi_hiz_ECT).sum():+.1f}",
                  f"   ortalama esit tahminli is payi (bekleyenler arasinda): %{100 * K.esit_tahmin_payi.mean():.1f}"]
            G = K.groupby("saat_i").agg(karar=("saat", "size"), farkli=("farkli_is", lambda x: int((x > 0).sum())),
                                        ort_bekleyen=("bekleyen", "mean"), tahmin_fark_saat=("tahmin_fark", "sum"),
                                        gercek_fark_saat=("gercek_fark", "sum"), simdi_fark=("simdi_fark", "sum"))
            G[["tahmin_fark_saat", "gercek_fark_saat"]] /= HOUR
            L += ["   saatlik (fark = agirlik - ECT, is-saat; acik dongu farklari kararlar arasinda toplanamaz,",
                  "   yalniz hangi saatlerde iki planin gercekte ayristigini gosterir)", G.round(2).to_string(), ""]
            W = K.sort_values("gercek_fark", ascending=False).head(10)[
                ["saat", "bekleyen", "bos_vm", "farkli_is", "simdi_ECT", "simdi_agirlik", "esit_tahmin_payi",
                 "tahmin_fark", "gercek_fark"]]
            L += ["   agirligin gercekte en cok kaybettirdigi 10 karar (fark sn)", W.round(2).to_string(index=False), ""]
            K.to_csv(os.path.join(out, f"kararlar_ozet_{AD[base]}_rho{rho}.csv"), index=False)
        D.to_csv(os.path.join(out, f"is_farklari_rho{rho}.csv.gz"), index=False)

    rep = "\n".join(L)
    print(rep)
    with open(os.path.join(out, "kayip_raporu.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pencere-kok", required=True, help="icinde pencereler/ klasoru olan cikti (ornek: test)")
    ap.add_argument("--gun", type=int, default=177)
    ap.add_argument("--ndays", type=int, default=1)
    ap.add_argument("--rho", default="0.8,1.2,2.0")
    ap.add_argument("--classes", default="1.0:0.5,1.5:0.3,2.0:0.2")
    ap.add_argument("--seeds", default="0,1,2", help="tam kosular icin tohumlar (ilki is/karar analizinde)")
    ap.add_argument("--gecis-tohum", type=int, default=0)
    ap.add_argument("--gecis-saat", default="2,4,6,8,10,12,14,16,18,20,22,24,30,36",
                    help="gecis zamanlari, pencere basindan saat")
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--tahmin", default="", help="tahmin_uret.py ciktisi (dogrulama.py --tahmin ile ayni)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="kayip177")
    ap.add_argument("--yalniz-rapor", action="store_true")
    a = ap.parse_args()
    rhos = [float(x) for x in a.rho.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    T_grid = [0.0] + [float(x) for x in a.gecis_saat.split(",")]
    for sub in ("gorevler", "isler", "kararlar"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    cfg = {"out": a.out, "pkok": a.pencere_kok, "gun": a.gun, "ndays": a.ndays, "classes": a.classes,
           "jitter": a.jitter, "K": 12, "cap": 600, "ls_tur": 50, "ls_ms": 0.0, "oracle_ref": False,
           "gecis_tohum": a.gecis_tohum, "tahmin": a.tahmin}
    if not os.path.exists(dg.window_path(a.pencere_kok, a.gun, a.ndays)):
        sys.exit(f"pencere yok: {dg.window_path(a.pencere_kok, a.gun, a.ndays)}")
    if not a.yalniz_rapor:
        tasks = []
        for rho in rhos:
            for T in T_grid[1:]:   # agir olanlar once
                tasks.append((rho, "agirlik", "greedy", T, a.gecis_tohum, False, cfg))
                tasks.append((rho, "greedy", "agirlik", T, a.gecis_tohum, False, cfg))
            for s in seeds:
                for b in ("agirlik", "greedy"):
                    tasks.append((rho, b, None, None, s, s == seeds[0], cfg))
        todo = [t for t in tasks if not os.path.exists(os.path.join(a.out, "gorevler", tag_of(*t[:5]) + ".json"))]
        print(f"gorev: {len(tasks)}  kalan: {len(todo)}  isci: {a.workers}", file=sys.stderr)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run, t): t for t in todo}
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    tag, rec = f.result()
                except Exception as e:
                    print(f"  [{k}/{len(todo)}] HATA {tag_of(*futs[f][:5])}: {e!r}", file=sys.stderr)
                    continue
                print(f"  [{k}/{len(todo)}] {tag}: ort_akis={rec['ort_akis']:,.0f}  {rec['calisma_sn']:.0f}s "
                      f"(toplam {time.time() - t0:.0f}s)", file=sys.stderr)
    with open(os.path.join(a.out, "ayarlar.json"), "w") as fh:
        json.dump({**vars(a), "politika": kurallar.POLITIKA_SURUM}, fh, indent=1)
    rapor(cfg, rhos, seeds, T_grid)


if __name__ == "__main__":
    main()
