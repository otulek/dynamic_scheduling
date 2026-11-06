#!/usr/bin/env python3
"""
MetaCentrum-2023 ikinci profil: grup yapisi, yerel siralama hatasi, aday pencereler.

Kullanim:
    python profile_windows.py METACENTRUM-2023.swf [--out profil2]

Filtre: tek cekirdek + tek dugum + used_GPUs = 0. Hicbir kaydi silmez.
Karar anindaki bekleyen is sayisi burada YOK: o, izin degil simulasyonun ciktisidir.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

COLS = {1: "submit", 3: "runtime", 4: "cores", 8: "walltime", 9: "ram_kb",
        11: "user", 18: "gpus", 19: "soft", 20: "topology"}
QS = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
WINDOWS = {"6saat": 6 * 3600, "1gun": 86400, "7gun": 7 * 86400}


def load(path, chunksize=1_000_000):
    parts = []
    for ch in pd.read_csv(path, sep=r"\s+", comment=";", header=None,
                          usecols=list(COLS), chunksize=chunksize, dtype={20: "string"}):
        ch = ch.rename(columns=COLS)
        nodes = pd.to_numeric(ch.pop("topology").str.split("x", n=1).str[0], errors="coerce")
        ch = ch[(ch.cores == 1) & (nodes == 1) & (ch.gpus == 0)]
        parts.append(ch.drop(columns=["cores", "gpus"]).astype("int64"))
        print(f"  okunan parca: {len(parts)}", file=sys.stderr)
    return pd.concat(parts, ignore_index=True)


def qline(s):
    s = pd.Series(s)
    if len(s) == 0:
        return "  (bos)"
    body = "  ".join(f"p{int(k * 100):02d}={v:,.4g}" for k, v in s.quantile(QS).items())
    return f"  min={s.min():,.4g}  {body}  max={s.max():,.4g}  ort={s.mean():,.4g}"


def pct(a, b):
    return f"{a:,} / {b:,} (%{100 * a / b:.2f})" if b else "0"


def concordance(bucket, est, real, rng):
    """Ayni kovadaki rastgele is ciftlerinde tahminin gercek siralamayi tutturma orani."""
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
        return n, float("nan"), float("nan")
    return n, float(((sr == se) + 0.5 * (se == 0)).sum() / n), float((se == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("swf")
    ap.add_argument("--out", default="profil2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    f = load(args.swf)
    n = len(f)
    L = []
    P = L.append

    P("=" * 78)
    P("A. GRUPLAR (ayni submit zaman damgasi, filtreli isler)")
    P("=" * 78)
    f["spec"] = f.groupby(["soft", "walltime", "ram_kb"], sort=False).ngroup()
    g = f.groupby("submit").agg(
        size=("runtime", "size"), users=("user", "nunique"), softs=("soft", "nunique"),
        specs=("spec", "nunique"), rt_mean=("runtime", "mean"), rt_std=("runtime", "std"),
        rt_min=("runtime", "min"), rt_max=("runtime", "max"))
    ng = len(g)
    P(f"Is: {n:,}   Farkli zaman damgasi (grup): {ng:,}")
    P("grup buyuklugu (grup basina):")
    P(qline(g["size"]))
    jw = np.repeat(g["size"].to_numpy(), g["size"].to_numpy())
    P("grup buyuklugu (is basina: tipik bir isin icinde oldugu grup):")
    P(qline(jw))
    for t in (1, 10, 100, 1000):
        P(f"  buyuklugu > {t:>4} olan gruplardaki isler: {pct(int(g.loc[g['size'] > t, 'size'].sum()), n)}")
    m = g[g["size"] >= 2]
    nm = int(m["size"].sum())
    P(f"-- yalniz buyuklugu >= 2 olan gruplar: {len(m):,} grup, {nm:,} is --")
    if nm:
        P(f"tek kullanicili gruplardaki isler            : {pct(int(m.loc[m.users == 1, 'size'].sum()), nm)}")
        P(f"tek soft_walltime degerli gruplardaki isler  : {pct(int(m.loc[m.softs == 1, 'size'].sum()), nm)}")
        P(f"tek (soft, walltime, RAM) ucluklu gruplar    : {pct(int(m.loc[m.specs == 1, 'size'].sum()), nm)}")
        P("grup icindeki farkli soft_walltime sayisi (grup basina):")
        P(qline(m.softs))
        same = m[m.softs == 1]
        if len(same):
            P("tahmini ayni olan gruplarda gercek surenin degiskenligi:")
            P("  maks/min orani:")
            P(qline(same.rt_max / same.rt_min.clip(lower=1)))
            P("  degisim katsayisi (std/ort):")
            P(qline((same.rt_std / same.rt_mean).dropna()))
    g.drop(columns=["rt_mean", "rt_std"]).to_csv(os.path.join(args.out, "gruplar.csv"))

    P("")
    P("=" * 78)
    P("B. YEREL SIRALAMA HATASI (ayni kovadaki rastgele ciftler; 0.5 = rastgele, 1 = kusursuz)")
    P("=" * 78)
    rng = np.random.default_rng(0)
    sub, rt = f.submit.to_numpy(), f.runtime.to_numpy()
    for name, bucket in (("ayni saat", sub // 3600), ("ayni zaman damgasi", sub)):
        for est in ("soft", "walltime"):
            k, conc, ties = concordance(bucket, f[est].to_numpy(), rt, rng)
            P(f"{name:<20} {est:<9}: cift={k:,}  uyum={conc:.3f}  tahmini esit cift=%{100 * ties:.1f}")

    P("")
    P("=" * 78)
    P("C. ADAY PENCERELER")
    P("=" * 78)
    P("kapasite = penceredeki toplam referans is yuku / pencere uzunlugu (hiz = 1 VM cinsinden)")
    gsize = f.groupby("submit")["runtime"].transform("size")
    for name, w in WINDOWS.items():
        win = f.submit // w
        t = pd.DataFrame({
            "is_sayisi": f.groupby(win).size(),
            "is_yuku_sn": f.groupby(win)["runtime"].sum(),
            "uzun_is_yuku_sn": f.runtime.where(f.runtime > w, 0).groupby(win).sum(),
            "en_buyuk_grup": gsize.groupby(win).max(),
            "kullanici": f.groupby(win)["user"].nunique(),
            "baskin_kullanici_is": f.groupby([win, "user"]).size().groupby(level=0).max(),
        })
        t["kapasite"] = t.is_yuku_sn / w
        t["uzun_is_payi"] = t.uzun_is_yuku_sn / t.is_yuku_sn
        t["baskin_kullanici_payi"] = t.baskin_kullanici_is / t.is_sayisi
        t.index.name = "pencere"
        t.to_csv(os.path.join(args.out, f"pencereler_{name}.csv"))
        P(f"-- pencere = {name}: {len(t):,} pencere --")
        P("is sayisi:")
        P(qline(t.is_sayisi))
        P("ortalama referans kapasite gereksinimi:")
        P(qline(t.kapasite))
        P("is yukunde pencereden uzun islerin payi:")
        P(qline(t.uzun_is_payi))
        P("baskin kullanicinin is payi:")
        P(qline(t.baskin_kullanici_payi))
        P("en buyuk grup:")
        P(qline(t.en_buyuk_grup))

    report = "\n".join(L)
    print(report)
    with open(os.path.join(args.out, "profil2_raporu.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
