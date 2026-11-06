"""
Ayni tahmini amac (bekleyen islerin tahmini tamamlanma zamanlari toplami) icin planlama kurallari.

Gosterim (bir karar ani)
  Bekleyen isler 0..n-1, tahmini kisa olan once siralanmis (ortak bag kirma ile). Is indisi bu
  sira oldugu icin "x'ten kisa" = "indisi x'ten kucuk". d[x]: tahmini sure.
  VM j: hiz s_j, tahmini bosalma A_j. VM icinde isler bu sirayla (tahmini kisa once) calisir.
  VM j'nin katkisi:  n_j A_j + (1/s_j) * sum_x d_x * k_x,   k_x = 1 + (ayni VM'de x'ten uzun is sayisi)
  yani k sondan siradir.

Aday VM'ler: her hiz sinifinda (A, VM numarasi) sirasina gore en erken n VM. n is en fazla n VM
kullanir ve ayni hizda daha erken bosalan VM hicbir zaman daha kotu degildir; bu kume kesin plani,
ECT'nin ve buradaki kurallarin kullanabilecegi her VM'i kapsar.

agirlik_kurali
  Isler uzundan kisaya islenir; her is, o anki en kucuk  A_j + (sayac_j + 1) * d / s_j  maliyetli
  yuvaya konur (sayac_j: VM j'ye daha once konmus, yani bu isten uzun is sayisi = isin sondan
  sirasi - 1). Butun A_j esitken bu, Q||sum C_j icin kesin en iyi kuraldir (Horowitz ve Sahni,
  1976). A_j farkliyken bir sezgiseldir; en iyilik garantisi yoktur.

yerel_arama
  Baslangic plani verilir. Komsuluk: (1) bir isi baska bir VM'e tasima, (2) farkli VM'lerdeki iki
  isi takas. VM ici sira her zaman tahmini kisa once kalir (atama sabitken bu sira en iyisidir).
  Her hamlenin amactaki degisimi kesin olarak hesaplanir; aday secimi bu degisime gore yapilir.
  Is basina: once en iyi tasima; iyilestirmiyorsa en iyi takas; iyilestiriyorsa uygulanir.
  Durma: bir turda hic iyilestirme yoksa (yerel optimum), --ls-tur tur dolunca ya da --ls-ms
  sure dolunca. Varsayilan sure siniri yoktur; sonuc deterministiktir.

SABITLENMIS POLITIKALAR (POLITIKA_SURUM; test pencerelerinden once donduruldu)
  Ortak: butun karsilastirmalar karar anina gore goreli zamanlarla ve ESITLIK_TOL ile yapilir.
  Bekleyen islerin sirasi (butun yontemler icin, fcfs haric): tahmini sure artan; esitlikte gelis
  zamani; o da esitse izdeki satir sirasi (is numarasi). VM numaralari hiz sinifina gore (yavas
  sinif once) ve sinif icinde sabit sirayla verilir.
  ECT (greedy)  : isler yukaridaki sirayla tek tek; her is tahmini bitisi en erken VM'e. Esitlikte
                  once yavas hiz sinifi (siniflar 1,0 < 1,5 < 2,0 sirasiyla denenir, ancak kesin
                  olarak daha iyi olan secilir), sinif icinde en erken bosalan, o da esitse en kucuk
                  VM numarasi.
  agirlik       : isler TERS sirayla (uzun once; esit tahminlilerde gelisi gec olan once) islenir;
                  her is en kucuk A_j + k d / s_j maliyetli yuvaya. Esitlikte aday listesindeki ilk
                  yuva: yavas sinif once, sinif icinde en erken bosalan, sonra en kucuk VM numarasi.
  yerel         : agirlik planindan baslar; ECT'nin plani ancak kesin_daha_iyi ise ECT'den. Bir
                  hamle ancak amaci ESITLIK_TOL'un otesinde dusururse kabul edilir; esit hamleler
                  uygulanmaz. Isler uzun once taranir; en iyi tasima/takas adaylarinda esitlikte
                  ilk aday.
  exact (kesin) : bekleyen <= cap ise atama problemi kesin cozulur ve COZUCUNUN sectigi en iyi plan
                  uygulanir (esit en iyi planlar arasindaki secim cozucuye birakilmistir; belirlenim-
                  cidir ama ilkeli degildir). K siniri baglayiciysa kesin plan ancak (ECT, agirlik)'in
                  iyisinden kesin_daha_iyi ise. Bekleyen > cap ise (ECT, agirlik)'in iyisi; esitlikte
                  agirlik. Bu son iki durum "kesinlik kapsami" disidir ve ayrica raporlanir.
"""

POLITIKA_SURUM = "P1-2026-10-13"
import time

import numpy as np

# Ortak esitlik toleransi. Butun karsilastirmalar KARAR ANINA GORE (goreli) zamanlarla yapilir:
# boşalma zamanlari "simdiden kac saniye sonra", maliyetler "simdiden itibaren kalan tamamlanma
# suresi toplami". c, ref'ten ancak  c < ref - ESITLIK_TOL * max(1, |ref|)  ise "daha iyi" sayilir.
ESITLIK_TOL = 1e-9


def kesin_daha_iyi(c, ref):
    return c < ref - ESITLIK_TOL * max(1.0, abs(ref))


def esit(c, ref):
    return abs(c - ref) <= ESITLIK_TOL * max(1.0, abs(ref))


def tamamlanmalar(vm, d, avail, speed):
    """Isler verilen sirada (tahmini kisa once); VM ici sira bu sirayi korur.
    Her isin tamamlanma zamanini verilen sirada dondurur."""
    o = np.argsort(vm, kind="stable")
    v = vm[o]
    t = d[o] / speed[v]
    cs = np.cumsum(t)
    first = np.empty(len(v), dtype=bool)
    first[0] = True
    first[1:] = v[1:] != v[:-1]
    base = np.maximum.accumulate(np.where(first, cs - t, 0.0))
    out = np.empty(len(v))
    out[o] = avail[v] + cs - base
    return out


def plan_cost(vm, d, avail, speed):
    return float(tamamlanmalar(vm, d, avail, speed).sum())


def anlamli_fark(vA, vB, d, avail, speed):
    """Iki plan arasinda VM adindan ote fark: is basina hiz ya da tahmini baslama farkli mi."""
    sA, sB = speed[vA], speed[vB]
    bA = tamamlanmalar(vA, d, avail, speed) - d / sA
    bB = tamamlanmalar(vB, d, avail, speed) - d / sB
    return (sA != sB) | (np.abs(bA - bB) > ESITLIK_TOL * np.maximum(1.0, np.abs(bA))), bA, bB


def cand_set(n, avail, bounds, rk):
    return np.concatenate([lo + np.lexsort((rk[lo:hi], avail[lo:hi]))[:n]
                           for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist())])


def agirlik_kurali(d, avail, cls_speed, bounds, rk):
    n = len(d)
    cand = cand_set(n, avail, bounds, rk)
    A = avail[cand]
    s = np.repeat(cls_speed, np.diff(bounds))[cand]
    cnt = np.zeros(len(cand))
    loc = np.empty(n, dtype=np.int64)
    for x in range(n - 1, -1, -1):
        j = int(np.argmin(A + (cnt + 1.0) * (d[x] / s)))
        loc[x] = j
        cnt[j] += 1.0
    return cand[loc]


def yerel_arama(d, avail, cls_speed, bounds, rk, vm0, max_tur=50, sure_ms=0.0):
    """Donus: (vm dizisi, istatistik sozlugu). avail karar anina gore goreli olmali.
    Bir hamle, amaci ESITLIK_TOL * max(1, baslangic maliyeti) kadardan fazla dusururse kabul edilir."""
    n = len(d)
    t0 = time.perf_counter()
    cand = cand_set(n, avail, bounds, rk)
    M = len(cand)
    pos = {int(c): i for i, c in enumerate(cand.tolist())}
    v = np.array([pos[int(x)] for x in vm0], dtype=np.int64)
    A = avail[cand]
    s = np.repeat(cls_speed, np.diff(bounds))[cand]
    stat = {"tur": 0, "tasima": 0, "takas": 0, "deger": 0, "toplam_kazanc": 0.0, "sure_siniri": False}
    eps = ESITLIK_TOL * max(1.0, plan_cost(cand[v], d, avail, np.repeat(cls_speed, np.diff(bounds))))
    idx = np.arange(n)
    dirty = True
    for _ in range(max_tur):
        stat["tur"] += 1
        improved = False
        for x in range(n - 1, -1, -1):          # uzun isler once
            if sure_ms and (time.perf_counter() - t0) * 1e3 > sure_ms:
                stat["sure_siniri"] = True
                break
            a = v[x]
            L = np.bincount(v[x + 1:], minlength=M)                 # VM'de x'ten uzun is sayisi
            S = np.bincount(v[:x], weights=d[:x], minlength=M)       # VM'de x'ten kisa islerin toplam suresi
            rem = -(A[a] + (d[x] * (1 + L[a]) + S[a]) / s[a])
            ins = A + (d[x] * (1 + L) + S) / s
            ins[a] = np.inf
            j = int(np.argmin(ins))
            stat["deger"] += M
            if rem + ins[j] < -eps:
                v[x] = j
                dirty = True
                stat["tasima"] += 1
                stat["toplam_kazanc"] -= rem + ins[j]
                improved = True
                continue
            # takas: x (VM a) <-> y (VM b != a)
            ys = idx[v != a]
            if len(ys) == 0:
                continue
            if dirty:                                                # yalniz plan degisince yeniden hesapla
                o = np.lexsort((idx, v))                             # VM'ye, sonra indise gore
                vs = v[o]
                first = np.r_[True, vs[1:] != vs[:-1]]
                grp_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
                cnt_v = np.bincount(v, minlength=M)
                rank_in = np.empty(n, dtype=np.int64)
                rank_in[o] = np.arange(n) - grp_start                # VM icinde kac kisa is onunde
                cs = np.cumsum(d[o])
                Sself = np.empty(n)
                Sself[o] = (cs - d[o]) - (cs - d[o])[grp_start]      # ayni VM'de kisa islerin toplami
                Lself = cnt_v[v] - 1 - rank_in                       # ayni VM'de uzun is sayisi
                dirty = False
            b = v[ys]
            dy = d[ys]
            rem_y = -(A[b] + (dy * (1 + Lself[ys]) + Sself[ys]) / s[b])
            gt = ys > x
            ins_x = A[b] + (d[x] * (1 + L[b] - gt) + S[b] - np.where(~gt, dy, 0.0)) / s[b]
            on_a = idx[(v == a) & (idx != x)]
            pa = np.searchsorted(on_a, ys, side="right")
            La = len(on_a) - pa
            csa = np.r_[0.0, np.cumsum(d[on_a])]
            Sa = csa[pa]
            ins_y = A[a] + (dy * (1 + La) + Sa) / s[a]
            tot = rem + rem_y + ins_x + ins_y
            stat["deger"] += len(ys)
            k = int(np.argmin(tot))
            if tot[k] < -eps:
                y = ys[k]
                v[x], v[y] = v[y], a
                dirty = True
                stat["takas"] += 1
                stat["toplam_kazanc"] -= tot[k]
                improved = True
        if stat["sure_siniri"] or not improved:
            break
    stat["ms"] = (time.perf_counter() - t0) * 1e3
    return cand[v], stat
