#!/usr/bin/env python3
"""
Coklu pencere ve yuk duzeyinde iki soru:
  (1) Tahmini amacta aramanin iyilestirme payi ne kadar?
      -> her kararda hizli kural (SPT + en erken tahmini bitis) ile kesin en iyi plan arasindaki fark.
  (2) Bu pay gercek surelerle yurutuldugunde ne kadar korunuyor?
      (a) Karar duzeyi, acik dongu: ayni durumda iki plan gercek surelerle, yeniden planlama ve yeni
          gelis olmadan yurutulur. Korunma = gercek kazanc / tahmini kazanc.
      (b) Kapali dongu: butun simulasyon hizli kuralla ve kesin planla ayri ayri kosar.
          Fark, gercek surelerdeki kucuk sarsintilarin (tohumlar) yarattigi oynamayla karsilastirilir.

Yontemler (--planners): fcfs, greedy (ECT, "hizli kural"), agirlik (Horowitz-Sahni turu agirlik
kurali, kurallar.py), yerel (ECT ve agirlik kuralinin iyisinden baslayan yerel arama, kurallar.py),
exact (kesin plan). Olcum kosusu (ECT gidisati, tohum 0) her kararda dort plani da ayni durumda
hesaplar; ozet icin: python kural_ozet.py <out>.
Butun planlama karsilastirmalari karar anina gore goreli zamanlarla ve tek bir esitlik toleransiyla
(kurallar.ESITLIK_TOL) yapilir; esitlik politikasi plan_hesapla() icinde tanimlidir.

Kullanim
  python pencere_sec.py pencereler_1gun.csv          # -> secilen_pencereler.txt
  python dogrulama.py --trace METACENTRUM-2023.swf --days @secilen_pencereler.txt --rho 0.8,1.2,2.0 \
                      --out dogrulama --workers 4
  python dogrulama_ozet.py dogrulama

Pencere CSV'leri --out/pencereler altina bir kez cikarilir; sonraki calistirmalar izi okumaz.
Her gorev bittiginde --out/gorevler/*.json yazilir; yarida kalan calistirma ayni komutla devam eder.

Model pilot_sim.py ile aynidir: tek cekirdek + tek dugum + GPU yok; bos baslangic; pencerede gelen
butun isler bitene kadar simulasyon; Q modeli; ayni zaman damgasindaki olaylar tek karar; tetikleyiciler
gelis / tamamlanma / soft asimi; asimdan sonra kalan sure walltime_limit'e gore; ortak is bag kirma
(soft, submit, is no). Planlama yalniz tahmin kullanir.

Tahmin kaynagi (--tahmin): verilmezse izdeki soft_walltime (isin SON degeri; is sirasinda guncellenmis
olabilir). Verilirse tahmin_uret.py ciktisindaki gonderim anindaki tahmin (soft_pred) 'soft' sutununun
yerine konur; izdeki deger 'soft_kayit' olarak saklanir. Pencereler, VM sayisi ve politikalar degismez.

Tohumlar (kapali dongu gurultu tabani): tohum 0 izdeki gercek sureleri aynen kullanir ve pilot_sim.py /
kesin_plan.py ile birebir ayni sonucu verir. Tohum s > 0'da gercek sureler kucuk bir carpimsal
gurultuyle bozulur: runtime * exp(N(0, --jitter)), walltime_limit ile sinirli. Tahminler, gelisler ve
VM sayisi degismez. Ayni (pencere, tohum) icin butun yontemler AYNI bozulmus ornegi gorur (eslestirilmis
karsilastirma). Boylece "iki yontem arasindaki fark, girdideki kucuk bir sarsintidan daha mi buyuk?"
sorusu yanitlanir. (VM bag kirmasini rastgelelestirmek bu isi gormez: esdeger bos VM'ler arasindaki
secim simulasyonu yalnizca yeniden adlandirir ve sonucu degistirmez.)
"""
import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kurallar  # noqa: E402  agirlik_kurali, yerel_arama

DAY = 86400
TAU = 10.0
COLS = {0: "job", 1: "submit", 3: "runtime", 4: "cores", 8: "walltime", 9: "ram_kb",
        11: "user", 18: "gpus", 19: "soft", 20: "topology"}


# ============================================================================ veri
def window_path(out, day, ndays):
    """out: icinde 'pencereler' klasoru olan kok (varsayilan --out; --pencere-kok ile degisir)."""
    return os.path.join(out, "pencereler", f"pencere_gun{day}_{ndays}g.csv")


def extract_windows(trace, days, ndays, out):
    """Eksik pencereleri izden TEK gecisle cikarir."""
    os.makedirs(os.path.join(out, "pencereler"), exist_ok=True)
    missing = [d for d in days if not os.path.exists(window_path(out, d, ndays))]
    if not missing:
        return
    if not trace:
        sys.exit(f"Su pencereler yok ve --trace verilmedi: {missing}")
    parts = {d: [] for d in missing}
    for k, ch in enumerate(pd.read_csv(trace, sep=r"\s+", comment=";", header=None, usecols=list(COLS),
                                       chunksize=1_000_000, dtype={20: "string"})):
        ch = ch.rename(columns=COLS)
        nodes = pd.to_numeric(ch.pop("topology").str.split("x", n=1).str[0], errors="coerce")
        ch = ch[(ch.cores == 1) & (nodes == 1) & (ch.gpus == 0)].drop(columns=["cores", "gpus"])
        for d in missing:
            sel = ch[(ch.submit >= d * DAY) & (ch.submit < (d + ndays) * DAY)]
            if len(sel):
                parts[d].append(sel)
        print(f"  iz okunuyor: parca {k + 1}", file=sys.stderr)
    for d in missing:
        df = pd.concat(parts[d], ignore_index=True).astype("int64") if parts[d] else pd.DataFrame(columns=list(COLS.values()))
        df.to_csv(window_path(out, d, ndays), index=False)
        print(f"  yazildi: gun {d}, {len(df):,} is", file=sys.stderr)


_TAHMIN = {}


def tahmin_uygula(J, path):
    """J'deki 'soft' sutununu tahmin_uret.py ciktisindaki soft_pred ile degistirir (is numarasina gore).
    Denetimler: dosyada is numaralari tekil; penceredeki her is dosyada var; tahmin sonlu ve > 0;
    walltime_limit > 0; tahmin <= walltime_limit. Degerler oldugu gibi (float) aktarilir; yuvarlama
    yalniz tahmin_uret.py'de yapilir."""
    if not path:
        return J
    if path not in _TAHMIN:
        T = pd.read_csv(path, usecols=["job", "soft_pred"])
        if not T.job.is_unique:
            raise ValueError(f"tahmin dosyasinda tekrarlanan is numarasi var ({path})")
        _TAHMIN[path] = pd.Series(T.soft_pred.to_numpy(np.float64), index=T.job.to_numpy())
    s = _TAHMIN[path].reindex(J.job.to_numpy()).to_numpy()
    if np.isnan(s).any():
        raise ValueError(f"tahmin dosyasinda olmayan {int(np.isnan(s).sum())} is var ({path})")
    w = J.walltime.to_numpy(np.float64)
    bad = ~np.isfinite(s) | (s <= 0) | (w <= 0) | (s > w)
    if bad.any():
        raise ValueError(f"gecersiz tahmin ya da walltime_limit: {int(bad.sum())} is "
                         f"(ornek is no: {J.job.to_numpy()[bad][:5].tolist()})")
    J = J.copy()
    J["soft_kayit"] = J["soft"]
    J["soft"] = s
    return J


def parse_classes(spec):
    out = []
    for item in spec.split(","):
        f = item.split(":")
        out.append((float(f[0]), float(f[1])))
    tot = sum(w for _, w in out)
    return [(s, w / tot) for s, w in out]


def build_vms(classes, m):
    counts = [max(1, round(m * w)) for _, w in classes]
    speed = np.concatenate([np.full(c, s) for c, (s, _) in zip(counts, classes)])
    bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return speed, bounds, np.array([s for s, _ in classes])


# ============================================================================ planlama
plan_cost = kurallar.plan_cost  # tek tanim: kurallar.py


def greedy(d, avail, cls_speed, bounds, rk):
    """d sirasindaki isleri tek tek, tahmini en erken bitirecek VM'e atar (ECT)."""
    heaps = []
    for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        h = list(zip(avail[lo:hi].tolist(), rk[lo:hi].tolist(), range(lo, hi)))
        heapq.heapify(h)
        heaps.append(h)
    sp = cls_speed.tolist()
    vms = np.empty(len(d), dtype=np.int64)
    for p, x in enumerate(d.tolist()):
        best, bk = math.inf, 0
        for k in range(len(sp)):
            c = heaps[k][0][0] + x / sp[k]
            if c < best:
                best, bk = c, k
        _, r, v = heaps[bk][0]
        heapq.heapreplace(heaps[bk], (best, r, v))
        vms[p] = v
    return vms


def exact(d, avail, cls_speed, bounds, rk, K):
    """Toplam tamamlanma zamanini en aza indiren kesin plan (atama problemi).
    Maliyet c(i; j, k) = A_j + k d_i / s_j; k = VM'de sondan sira. Aday VM: her siniftan en erken
    bosalan n VM (kesinligi bozmaz). K: VM basina yuva siniri (baglayici olursa bildirilir)."""
    n = len(d)
    cand = np.concatenate([lo + np.lexsort((rk[lo:hi], avail[lo:hi]))[:n]
                           for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist())])
    speed = np.repeat(cls_speed, np.diff(bounds))[cand]
    kk = min(n, K)
    if n > len(cand) * kk:
        # Yuva sayisi bekleyen isten az: atama butun isleri kapsayamaz. K siniri baglayici sayilir;
        # cagiran taraf (ECT, agirlik)'in iyisini kullanir. Bekleyen <= cap = 600 ve K = 12 iken bu
        # ancak toplam VM sayisi 50'den azsa olabilir; test pencerelerinde olusmaz.
        return None, float("inf"), True
    C = (avail[cand][None, :, None]
         + (d[:, None] / speed[None, :])[:, :, None] * np.arange(1, kk + 1)[None, None, :]).reshape(n, -1)
    r, c = linear_sum_assignment(C)
    vms = np.empty(n, dtype=np.int64)
    vms[r] = cand[c // kk]
    hit = kk < n and bool(((c % kk) == kk - 1).any())
    return vms, float(C[r, c].sum()), hit


def plan_hesapla(names, dd, av, speed, cls_speed, bounds, rk, cfg, stats=None):
    """Istenen kurallarin planlarini ayni karar durumu icin hesaplar. av karar anina gore goreli.

    Esitlik politikasi (gelistirme asamasinda sabitlendi):
      yerel  : agirlik planindan baslar; ECT'nin plani ancak kesin_daha_iyi ise ECT'den baslar.
      exact  : kesin cozuldu ve K siniri baglayici degilse kesin plan; K baglayiciysa kesin plan
               ancak (ECT, agirlik)'in iyisinden kesin_daha_iyi ise; cozulmediyse (bekleyen > cap)
               (ECT, agirlik)'in iyisi. "Iyisi" esitlikte agirlik.
    Donus: {"plan": {ad: vm}, "maliyet": {ad: tahmini goreli maliyet}, "ms": {...}, "durum": ..., "ls": {...}}
    """
    need = set(names)
    if "yerel" in need or "exact" in need:
        need |= {"greedy", "agirlik"}
    P, C, MS = {}, {}, {}
    out = {"plan": P, "maliyet": C, "ms": MS, "durum": None, "ls": {}}
    n = len(dd)
    if n <= 1:  # tek iste en erken bitis en iyisidir; butun kurallar ayni
        t1 = time.perf_counter()
        v = greedy(dd, av, cls_speed, bounds, rk)
        for nm in need:
            P[nm], C[nm], MS[nm] = v, plan_cost(v, dd, av, speed), (time.perf_counter() - t1) * 1e3
        out["durum"] = "kesin"
        return out
    t1 = time.perf_counter()
    P["greedy"] = greedy(dd, av, cls_speed, bounds, rk)
    MS["greedy"] = (time.perf_counter() - t1) * 1e3
    C["greedy"] = plan_cost(P["greedy"], dd, av, speed)
    if "agirlik" in need:
        t1 = time.perf_counter()
        P["agirlik"] = kurallar.agirlik_kurali(dd, av, cls_speed, bounds, rk)
        MS["agirlik"] = (time.perf_counter() - t1) * 1e3
        C["agirlik"] = plan_cost(P["agirlik"], dd, av, speed)
        iyi = "greedy" if kurallar.kesin_daha_iyi(C["greedy"], C["agirlik"]) else "agirlik"
    if "yerel" in need:
        vl, lst = kurallar.yerel_arama(dd, av, cls_speed, bounds, rk, P[iyi],
                                       cfg.get("ls_tur", 50), cfg.get("ls_ms", 0.0))
        P["yerel"], MS["yerel"], out["ls"] = vl, MS["agirlik"] + lst["ms"], lst
        C["yerel"] = plan_cost(vl, dd, av, speed)
    if "exact" in need:
        t1 = time.perf_counter()
        if n > cfg["cap"]:
            out["durum"] = "geri"
            P["exact"] = P[iyi]
            if stats is not None:
                stats["geri_dusus"] += 1
        else:
            vx, cx, hit = exact(dd, av, cls_speed, bounds, rk, cfg["K"])
            out["durum"] = "K" if hit else "kesin"
            out["kesin_deger"] = cx
            P["exact"] = vx if (not hit or kurallar.kesin_daha_iyi(cx, C[iyi])) else P[iyi]
            if stats is not None:
                stats["kesin"] += 1
                stats["K_siniri"] += hit
        MS["exact"] = (time.perf_counter() - t1) * 1e3
        C["exact"] = plan_cost(P["exact"], dd, av, speed)
    return out


# ============================================================================ simulasyon
def simulate(J, speed, bounds, cls_speed, planner, seed, cfg, measure=False, hook=None):
    """hook(durum): her >=2 bekleyen isli kararda, plan uygulanmadan once cagrilir (analiz icin)."""
    n, m = len(J), len(speed)
    submit = J.submit.to_numpy(float)
    soft = J.soft.to_numpy(float)
    wall = np.maximum(J.walltime.to_numpy(float), soft)
    run = J.runtime.to_numpy(float)
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), submit, soft))] = np.arange(n)
    rk = np.arange(m)  # VM bag kirma: numara sirasi (esdeger bos VM'ler arasindaki secim sonucu degistirmez)
    gt, gs = np.unique(submit, return_index=True)
    ge = np.append(gs[1:], n)
    busy = np.zeros(m, bool)
    st, cph, cL, cend = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
    queues = [deque() for _ in range(m)]
    planned, waiting, heap = [], set(), []
    start_t, end_t = np.full(n, np.nan), np.full(n, np.nan)
    on_vm = np.full(n, -1, dtype=np.int64)
    stats = {"karar": 0, "kesin": 0, "geri_dusus": 0, "K_siniri": 0, "plan_sn": 0.0}
    rows = []
    gi = done = 0
    wall0 = time.perf_counter()

    def start(i, v, t):
        s = speed[v]
        busy[v], st[v], cph[v], cL[v] = True, t, soft[i] / s, wall[i] / s
        p = run[i] / s
        cend[v] = t + p
        start_t[i] = t
        on_vm[i] = v
        waiting.discard(i)
        heapq.heappush(heap, (t + p, 0, v, i))
        if p > cph[v]:
            heapq.heappush(heap, (t + cph[v], 1, v, i))

    while gi < len(gt) or heap:
        t = min(gt[gi] if gi < len(gt) else math.inf, heap[0][0] if heap else math.inf)
        trig, freed = [], []
        while heap and heap[0][0] <= t:
            _, kind, v, i = heapq.heappop(heap)
            if kind == 0:
                busy[v] = False
                end_t[i] = t
                freed.append(v)
                done += 1
                if "tamamlanma" not in trig:
                    trig.append("tamamlanma")
            elif "asim" not in trig:
                trig.append("asim")
        if gi < len(gt) and gt[gi] == t:
            waiting.update(range(gs[gi], ge[gi]))
            gi += 1
            trig.insert(0, "gelis")
        if waiting and trig:
            t0 = time.perf_counter()
            W = np.fromiter(waiting, dtype=np.int64, count=len(waiting))
            el = t - st
            # karar anina gore goreli zamanlar (0 = simdi): tahmini ve gercek bosalma
            av = np.where(busy, np.maximum(np.where(el < cph, st + cph, st + cL), t), t) - t
            avr = np.where(busy, cend, t) - t
            order = np.sort(W) if planner == "fcfs" else W[np.argsort(rank[W])]
            dd = soft[order]
            nw = len(order)
            names = ["greedy", "agirlik", "yerel", "exact"] if (measure and nw > 1) else \
                ["greedy" if planner == "fcfs" else planner]
            R = plan_hesapla(names, dd, av, speed, cls_speed, bounds, rk, cfg,
                             stats if (planner == "exact" or measure) else None)
            vms = R["plan"]["greedy" if planner == "fcfs" else planner]
            stats["plan_sn"] += time.perf_counter() - t0
            if hook is not None and nw > 1:
                hook({"t": t, "order": order, "d": dd, "avail": av, "avail_gercek": avr,
                      "d_gercek": run[order], "planlar": R, "planner": planner, "speed": speed,
                      "bounds": bounds, "cls_speed": cls_speed, "rk": rk, "cfg": cfg})
            if measure and nw > 1:
                dr = run[order]
                Cg = R["maliyet"]
                real = {k: plan_cost(R["plan"][k], dr, avr, speed) for k in ("greedy", "agirlik", "yerel", "exact")}
                durum = R["durum"]
                ex = R.get("kesin_deger", Cg["exact"]) if durum != "geri" else Cg["exact"]
                ro = np.nan
                if cfg["oracle_ref"] and nw <= cfg["cap"]:
                    oo = np.argsort(dr, kind="stable")
                    ro = exact(dr[oo], avr, cls_speed, bounds, rk, cfg["K"])[1]
                lst = R["ls"]
                rows.append((t, "+".join(trig), nw, int(m - busy.sum()), durum,
                             Cg["greedy"], ex, real["greedy"], real["exact"], ro,
                             Cg["agirlik"], real["agirlik"], Cg["yerel"], real["yerel"],
                             R["ms"]["greedy"], R["ms"]["agirlik"], R["ms"]["yerel"], R["ms"]["exact"],
                             lst.get("tur", 0), lst.get("tasima", 0) + lst.get("takas", 0)))
            for v in planned:
                queues[v].clear()
            for i, v in zip(order.tolist(), vms.tolist()):
                queues[v].append(i)
            planned = np.unique(vms).tolist()
            cand = planned
            stats["karar"] += 1
        else:
            cand = freed
        for v in cand:
            if not busy[v] and queues[v]:
                start(queues[v].popleft(), v, t)

    assert done == n and not waiting
    flow = end_t - submit
    wt = start_t - submit
    bsld = np.maximum(1.0, flow / np.maximum(end_t - start_t, TAU))
    summ = {"is": n, "ort_akis": float(flow.mean()), "ort_bekleme": float(wt.mean()),
            "ort_islem": float((end_t - start_t).mean()), "p95_bsld": float(np.quantile(bsld, .95)),
            "ort_bsld": float(bsld.mean()), "p95_akis": float(np.quantile(flow, .95)),
            "p99_bekleme": float(np.quantile(wt, .99)), "bekleyen_orani": float((wt > 0).mean()),
            "son_bitis_gun": float((end_t.max() - submit.min()) / DAY),
            "calisma_sn": time.perf_counter() - wall0, **stats}
    D = pd.DataFrame(rows, columns=["t", "tetik", "bekleyen", "bos_vm", "durum", "tahmin_hizli",
                                    "tahmin_kesin", "gercek_hizli", "gercek_kesin", "gercek_kahin",
                                    "tahmin_agirlik", "gercek_agirlik", "tahmin_yerel", "gercek_yerel",
                                    "ms_hizli", "ms_agirlik", "ms_yerel", "ms_kesin", "yerel_tur", "yerel_hamle"])
    jobs = pd.DataFrame({"job": J.job.to_numpy(), "submit": submit, "soft": soft, "runtime": run,
                         "vm": on_vm, "hiz": speed[on_vm], "start": start_t, "end": end_t,
                         "bekleme": wt, "islem": end_t - start_t, "akis": flow, "bsld": bsld})
    return summ, D, jobs


# ============================================================================ gorevler
def task_id(day, rho, planner, seed):
    return f"gun{day}_rho{rho}_{planner}_s{seed}"


def run_task(args):
    day, rho, planner, seed, measure, cfg = args
    J = pd.read_csv(window_path(cfg.get("pkok", cfg["out"]), day, cfg["ndays"]))
    J = J.sort_values(["submit", "job"], kind="stable").reset_index(drop=True)
    J = tahmin_uygula(J, cfg.get("tahmin", ""))
    classes = parse_classes(cfg["classes"])
    Lw = J.runtime.sum() / (cfg["ndays"] * DAY)
    m = max(1, round(Lw / rho / sum(s * w for s, w in classes)))  # VM sayisi bozulmamis izden
    if seed > 0:
        g = np.random.default_rng([day, seed])
        r = J.runtime.to_numpy(float) * np.exp(g.normal(0.0, cfg["jitter"], len(J)))
        J["runtime"] = np.clip(r, 1.0, np.maximum(J.walltime.to_numpy(float), J.soft.to_numpy(float)))
    if cfg.get("bilgi", "tahmin") == "gercek":  # planlayici gercek sureleri bilir (kahin deneyi)
        J["soft"] = J["runtime"]
        J["walltime"] = J["runtime"]
    speed, bounds, cls_speed = build_vms(classes, m)
    summ, D, jobs = simulate(J, speed, bounds, cls_speed, planner, seed, cfg, measure)
    tid = task_id(day, rho, planner, seed)
    if measure:
        D.to_csv(os.path.join(cfg["out"], "kararlar", tid + ".csv"), index=False)
    if cfg.get("save_jobs"):
        os.makedirs(os.path.join(cfg["out"], "isler"), exist_ok=True)
        jobs.to_csv(os.path.join(cfg["out"], "isler", tid + ".csv.gz"), index=False)
    rec = {"gun": day, "rho": rho, "yontem": planner, "tohum": seed, "olcum": measure,
           "politika": kurallar.POLITIKA_SURUM,
           "bilgi": cfg.get("bilgi", "tahmin"),
           "tahmin_kaynagi": os.path.basename(cfg.get("tahmin", "")) or "soft_walltime (iz)",
           "vm": int(len(speed)), "nominal_rho": float(Lw / speed.sum()), **summ}
    with open(os.path.join(cfg["out"], "gorevler", tid + ".json"), "w") as fh:
        json.dump(rec, fh)
    return tid, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="", help="METACENTRUM SWF (pencereler yoksa gerekli)")
    ap.add_argument("--days", required=True,
                    help="virgulle ayrilmis gun numaralari ya da @secilen_pencereler.txt")
    ap.add_argument("--ndays", type=int, default=1, help="gelis penceresi uzunlugu (gun)")
    ap.add_argument("--rho", default="0.8,1.2,2.0")
    ap.add_argument("--classes", default="1.0:0.5,1.5:0.3,2.0:0.2", help="hiz:pay,...")
    ap.add_argument("--seeds", type=int, default=3, help="tohum sayisi (0 = bozulmamis iz dahil)")
    ap.add_argument("--jitter", type=float, default=0.01, help="tohum > 0 icin gercek sure gurultusu (log-sd)")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--cap", type=int, default=600, help="bundan fazla bekleyen iste kesin plan yerine hizli kural")
    ap.add_argument("--oracle-ref", action="store_true",
                    help="karar duzeyinde gercek surelerle kesin plan (ust sinir); sureyi yaklasik ikiye katlar")
    ap.add_argument("--save-jobs", action="store_true", help="is duzeyi sonuclari isler/ altina yaz")
    ap.add_argument("--bilgi", choices=["tahmin", "gercek"], default="tahmin",
                    help="gercek: planlayici gercek sureleri bilir (ayri bir --out kullanin)")
    ap.add_argument("--planners", default="fcfs,greedy,agirlik,yerel,exact",
                    help="kapali dongu yontemleri: fcfs, greedy (ECT), agirlik, yerel, exact")
    ap.add_argument("--ls-tur", type=int, default=50, help="yerel arama: en fazla tur")
    ap.add_argument("--ls-ms", type=float, default=0.0, help="yerel arama: karar basina sure siniri (0 = yok)")
    ap.add_argument("--tahmin", default="",
                    help="tahmin_uret.py ciktisi; verilirse soft_walltime yerine gonderim anindaki tahmin")
    ap.add_argument("--pencere-kok", default="",
                    help="pencereleri baska bir ciktinin altindan oku (ornek: dogrulama); yeniden cikarmaz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="dogrulama")
    a = ap.parse_args()
    spec = open(a.days[1:]).read().strip() if a.days.startswith("@") else a.days
    days = [int(x) for x in spec.split(",")]
    rhos = [float(x) for x in a.rho.split(",")]
    for sub in ("gorevler", "kararlar"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    pkok = a.pencere_kok or a.out
    extract_windows(a.trace, days, a.ndays, pkok)
    empty = [d for d in days if len(pd.read_csv(window_path(pkok, d, a.ndays))) < 2]
    if empty:
        print(f"UYARI: filtreden sonra is kalmayan pencereler atlandi: {empty}", file=sys.stderr)
        days = [d for d in days if d not in empty]
    cfg = {"out": a.out, "ndays": a.ndays, "classes": a.classes, "K": a.K, "cap": a.cap,
           "oracle_ref": a.oracle_ref, "jitter": a.jitter, "save_jobs": a.save_jobs, "bilgi": a.bilgi,
           "ls_tur": a.ls_tur, "ls_ms": a.ls_ms, "pkok": pkok, "tahmin": a.tahmin}
    with open(os.path.join(a.out, "ayarlar.json"), "w") as fh:
        json.dump({**vars(a), "days": days, "rhos": rhos, "politika": kurallar.POLITIKA_SURUM,
                   "esitlik_tol": kurallar.ESITLIK_TOL}, fh, indent=1)

    planners = a.planners.split(",")
    bad = set(planners) - {"fcfs", "greedy", "agirlik", "yerel", "exact"}
    if bad:
        sys.exit(f"bilinmeyen yontem: {bad}")
    tasks = []
    for d in days:
        for r in rhos:
            tasks.append((d, r, "greedy", 0, True, cfg))      # karar duzeyi olcum (agir), ECT gidisati
            for p in ("exact", "yerel", "agirlik", "greedy", "fcfs"):   # agir olanlar once
                if p in planners:
                    tasks += [(d, r, p, s, False, cfg) for s in range(1 if p == "greedy" else 0, a.seeds)]
    todo = [t for t in tasks if not os.path.exists(
        os.path.join(a.out, "gorevler", task_id(*t[:4]) + ".json"))]
    print(f"gorev: {len(tasks)}  tamamlanmis: {len(tasks) - len(todo)}  kalan: {len(todo)}  "
          f"isci: {a.workers}", file=sys.stderr)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_task, t): t for t in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                tid, rec = f.result()
            except Exception as e:  # bir gorevin hatasi butun calismayi durdurmasin
                print(f"  [{k}/{len(todo)}] HATA {task_id(*futs[f][:4])}: {e!r}", file=sys.stderr)
                continue
            print(f"  [{k}/{len(todo)}] {tid}: ort_akis={rec['ort_akis']:,.0f}  karar={rec['karar']:,}  "
                  f"geri_dusus={rec['geri_dusus']:,}  {rec['calisma_sn']:.0f}s  "
                  f"(toplam {time.time() - t0:.0f}s)", file=sys.stderr)
    print(f"Bitti. Ozet icin: python kural_ozet.py {a.out}  (ya da dogrulama_ozet.py)", file=sys.stderr)


if __name__ == "__main__":
    main()
