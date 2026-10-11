#!/usr/bin/env python3
"""
analisar_etapa9.py
==================
Analise do Estudo 9 (seleção adaptativa em ALNS) conforme o PRE-REGISTRO (diario, Entrada 103). So o que la esta
fixado e confirmatorio; o resto e EXPLORATORIO.

  PRIMARIO (alfa 0,05, um teste)        : RP - U nas 51 instancias (X + SYN_VAL)
  SECUNDARIO (Holm m=12)                : {ucb1, pm, hgucb, ts, exp3, eff} - U, em X e em SYN_VAL separadamente
  TERCIARIO (Holm m=4)                  : U-ALNS - VNS-U e U-ALNS - VNS-estatica (mesmo orcamento 1500 n^2), em X e SYN_VAL
  Exploratorio                          : RP - U por conjunto; equivalencia (IC90% em +-0,10pp); uso dos operadores
dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por instancia (media das seeds), IC95% bootstrap
(10.000, semente 0), empate = |dif| < 1e-9.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
SEC = ["ucb1", "pm", "hgucb", "ts", "exp3", "eff"]
DESTR = ["aleatoria", "pior", "relacionada", "rota"]
REPAR = ["guloso", "regret2", "barato"]


def holm(pvals, m):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def contraste(piv, cid, a, b, rng, grupo):
    dif = piv[a].values - piv[b].values
    dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)
    idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
    med = dif[idx].mean(axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    lo90, hi90 = np.percentile(med, [5, 95])
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    return dict(grupo=grupo, id=cid, a=a, b=b, n_inst=len(dif), dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()), p_bruto=p,
                equiv_010=bool(lo90 > -0.1 and hi90 < 0.1), mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)))


def main():
    rng = np.random.default_rng(0)
    e = pd.read_csv(RES / "etapa9_eval.csv")
    print(f"etapa9_eval: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}")
    e = e[e.viavel]
    t = e.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
    vp = RES / "etapa9_vns.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        v = v[v.viavel]
        tv = v.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
        t = t.join(tv, how="left")
    conj = e.drop_duplicates("instancia").set_index("instancia").conjunto
    t = t.dropna()
    t["conj"] = conj.reindex(t.index)
    print(f"{len(t)} instancias completas ({(t.conj == 'X').sum()} X, {(t.conj == 'SYN_VAL').sum()} SYN_VAL)")
    arms = [c for c in t.columns if c.startswith("s9_")]
    print("\nGap medio (%) por braco, X | SYN_VAL | todas:")
    print(pd.DataFrame({"X": t[t.conj == "X"][arms].mean(), "SYN_VAL": t[t.conj == "SYN_VAL"][arms].mean(),
                        "todas": t[arms].mean()}).round(3).sort_values("todas").to_string())
    linhas = []
    linhas.append(contraste(t, "PRIM:RP", "s9_RP", "s9_U", rng, "primario"))
    for c in ("X", "SYN_VAL"):
        sub = t[t.conj == c]
        for a in SEC:
            linhas.append(contraste(sub, f"SEC:{a}@{c}", f"s9_{a}", "s9_U", rng, "secundario"))
        if "s9_vns_U" in sub:
            for b in ("s9_vns_U", "s9_vns_static"):
                linhas.append(contraste(sub, f"TERC:ALNS-U vs {b[3:]}@{c}", "s9_U", b, rng, "terciario"))
        linhas.append(contraste(sub, f"EXPL:RP@{c}", "s9_RP", "s9_U", rng, "exploratorio"))
    r = pd.DataFrame(linhas)
    for grupo, m in (("secundario", 12), ("terciario", 4)):
        s = r.grupo == grupo
        if s.any():
            r.loc[s, "p_holm"] = holm(r.loc[s, "p_bruto"].values, m)
    r["signif_005"] = np.where(r.grupo == "primario", r.p_bruto < 0.05, r.p_holm < 0.05)
    pd.set_option("display.width", 240)
    print("\n=== CONFIRMATORIO (dif = gap(a) - gap(b); negativo = a melhor) ===")
    print(r[r.grupo != "exploratorio"].drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    print("\n=== EXPLORATORIO ===")
    print(r[r.grupo == "exploratorio"].drop(columns=["grupo", "a", "b", "p_holm", "signif_005"], errors="ignore").round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa9_confirmatoria.csv", index=False)
    # hipoteses
    prim = r[r.grupo == "primario"].iloc[0]
    sec = r[r.grupo == "secundario"]
    print(f"\nH34 (RP significativamente melhor que U): dif {prim.dif_pp:+.3f}pp, p={prim.p_bruto:.4f} -> "
          f"{'CONFIRMADA' if prim.p_bruto < 0.05 and prim.dif_pp < 0 else 'nao confirmada'}")
    melhores = sec[(sec.dif_pp < 0) & (sec.p_holm < 0.05)]
    print(f"H35 (alguma secundaria significativamente melhor, Holm): {len(melhores)} -> "
          f"{'CONFIRMADA' if len(melhores) else 'nao confirmada'}")
    ts_eff = sec[sec.a.isin(["s9_ts", "s9_eff"])]
    print(f"H36 (TS e EFF significativamente piores em X e SYN_VAL): {int(((ts_eff.dif_pp > 0) & (ts_eff.p_holm < 0.05)).sum())} de 4")
    terc = r[r.grupo == "terciario"]
    if len(terc):
        ok = int(((terc.dif_pp <= -0.5) & (terc.p_holm < 0.05)).sum())
        print(f"H37 (ALNS-U melhor que as 2 referencias VNS por >= 0,5pp, Holm): {ok} de {len(terc)}")
    outras = sec[~sec.a.isin(["s9_ts", "s9_eff"])]
    print(f"H38 (|dif| <= 0,25pp para as adaptativas exceto TS e EFF): max |dif| = {outras.dif_pp.abs().max():.3f}pp")
    # uso dos operadores
    print("\n[exploratorio] uso medio (%) dos operadores de destruicao e de reparo; tempo medio de CPU (s):")
    for b, g in e.groupby("braco"):
        ud = np.mean([np.array(s.split("|"), float) / np.array(s.split("|"), float).sum() for s in g.uso_d], axis=0) * 100
        ur = np.mean([np.array(s.split("|"), float) / np.array(s.split("|"), float).sum() for s in g.uso_r], axis=0) * 100
        print(f"  {b:10s} D: " + " ".join(f"{n}={v:4.1f}" for n, v in zip(DESTR, ud)) +
              " | R: " + " ".join(f"{n}={v:4.1f}" for n, v in zip(REPAR, ur)) + f" | CPU {g.tempo_s.mean():.0f}s")


if __name__ == "__main__":
    main()
