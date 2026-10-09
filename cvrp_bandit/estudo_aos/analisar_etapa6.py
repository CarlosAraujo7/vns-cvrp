#!/usr/bin/env python3
"""
analisar_etapa6.py
==================
Analise da Etapa 6 conforme o PRE-REGISTRO (diario, Entrada 91). So o que la
esta fixado e confirmatorio; o resto e EXPLORATORIO.

Confirmatorio nas 74 (E1 e E2; Holm m=12 = 6 bracos x 2 niveis):
  W{i}: s6_R5_{exp3, ts, swucb, ap, eff, eff2}  vs  s3_R5_U
  Omnibus: Friedman em {U + 6 bracos} por nivel.
Exploratorio: contrastes contra a mistura estatica ajustada (s4_R5_static) e contra o UCB1 padrao
(s3_R5_ucb1c); equivalencia (IC90% em +-0,10pp); fracao das avaliacoes de distancia por operador;
replicacao em SYN_VAL (etapa6_val.csv, 30 instancias, E1, 5 seeds; sem familia de testes).

dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por instancia (media das
seeds), IC95% bootstrap (10.000, semente 0), empate = |dif| < 1e-9.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
NOVOS = ["s6_R5_exp3", "s6_R5_ts", "s6_R5_swucb", "s6_R5_ap", "s6_R5_eff", "s6_R5_eff2"]
M_FAMILIA = 12
OPS = ["2-Opt", "Move(1)", "Swap(1,1)", "Reloc(1,0)", "Reloc(2,0)", "2-opt*", "Or-opt", "Swap-intra", "CROSS", "SWAP*"]


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
    return dict(grupo=grupo, id=cid, a=a, b=b, dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi, vit_a=int((dif < 0).sum()),
                der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()), p_bruto=p,
                equiv_010=bool(lo90 > -0.1 and hi90 < 0.1), mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)))


def main():
    rng = np.random.default_rng(0)
    e6 = pd.read_csv(RES / "etapa6_full.csv")
    e3 = pd.read_csv(RES / "etapa3_full.csv")
    e4 = pd.read_csv(RES / "etapa4_eval74.csv")
    print(f"etapa6_full: {len(e6)} execucoes, inviaveis {int((~e6.viavel).sum())}")
    cols = ["braco", "nivel", "instancia", "seed", "gap_pct", "viavel"]
    base = pd.concat([e6[cols], e3[e3.braco.isin(["s3_R5_U", "s3_R5_ucb1c"])][cols],
                      e4[e4.braco == "s4_R5_static"][cols]])
    base = base[base.viavel]
    linhas = []
    for nv in (1, 5):
        t = base[base.nivel == nv].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
        n_inst = t.dropna().shape[0]
        print(f"\n=== E{nv}: {n_inst} instancias completas; gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
        cj = ["s3_R5_U"] + NOVOS
        fr = stats.friedmanchisquare(*[t[c].values for c in cj])
        print(f"Friedman (U + 6 bracos): chi2={fr.statistic:.2f} p={fr.pvalue:.3g}")
        for i, a in enumerate(NOVOS, 1):
            l = contraste(t, f"W{i}", a, "s3_R5_U", rng, "confirmatorio")
            l["nivel"] = nv
            linhas.append(l)
        for a in NOVOS:
            for b, rot in (("s4_R5_static", "expl_vs_static"), ("s3_R5_ucb1c", "expl_vs_ucb1")):
                l = contraste(t, f"{rot}", a, b, rng, rot)
                l["nivel"] = nv
                linhas.append(l)
    r = pd.DataFrame(linhas)
    conf = r.grupo == "confirmatorio"
    r.loc[conf, "p_holm"] = holm(r.loc[conf, "p_bruto"].values, M_FAMILIA)
    r["signif_005"] = r.p_holm < 0.05
    pd.set_option("display.width", 230)
    print("\n=== CONFIRMATORIO (dif = gap(a) - gap(b); Holm m=12) ===")
    print(r[conf].drop(columns=["grupo"]).round(4).to_string(index=False))
    print(f"\nSignificativos: {int(r[conf].signif_005.sum())} de {int(conf.sum())}")
    print("\n=== EXPLORATORIO: contra a mistura estatica e contra o UCB1 padrao (sem correcao) ===")
    print(r[~conf].drop(columns=["grupo", "p_holm", "signif_005"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa6_confirmatoria.csv", index=False)
    print("\n[exploratorio] % das avaliacoes de distancia por operador (E1/E2):")
    for (br, nv), g in e6.groupby(["braco", "nivel"]):
        arr = np.array([[int(v) for v in s.split("|")] for s in g.calls_por_operador])
        pct = 100 * arr.sum(0) / arr.sum()
        print(f"  {br:14s} E{nv}: " + " ".join(f"{p:5.1f}" for p in pct))
    vp = RES / "etapa6_val.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        vv = pd.read_csv(RES / "etapa4_val.csv")
        v = pd.concat([v, vv[vv.braco == "s3_R5_U"]])
        v = v[v.viavel]
        tv = v.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
        print(f"\n=== SYN_VAL (exploratorio; {tv.dropna().shape[0]} instancias): gap medio ===")
        print(tv.mean().round(3).sort_values().to_string())
        for a in NOVOS:
            if a in tv:
                l = contraste(tv.dropna(), "VAL", a, "s3_R5_U", rng, "val")
                print(f"  {a} - U: {l['dif_pp']:+.3f}pp [{l['ic_lo']:+.3f}, {l['ic_hi']:+.3f}] p={l['p_bruto']:.3g}")


if __name__ == "__main__":
    main()
