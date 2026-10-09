#!/usr/bin/env python3
"""
analisar_etapa2.py
==================
Analise da Etapa 2 da Direcao A conforme o PRE-REGISTRO (diario, Entrada
71). Implementa SOMENTE o que la esta fixado como confirmatorio; o resto e
rotulado EXPLORATORIO.

Confirmatorio, por orcamento B in {800, 4000} (famil. de 28 testes, Holm m=28):
  2A (6): C1 s2a_U_best_LO   vs s2a_U_best_F40    (termino, aceitacao best)
          C2 s2a_U_shaken_LO vs uniforme           (termino, aceitacao shaken)
          C3 s2a_U_best_LO   vs s2a_U_shaken_LO    (aceitacao, LO)
          C4 s2a_U_best_F40  vs uniforme           (aceitacao, F40)
          C5 gvns_classico   vs s2a_U_best_LO      (ordem fixa vs aleat., best/LO)
          C6 s2a_FO_shaken_LO vs s2a_U_shaken_LO   (ordem, shaken/LO)
  2B (8): D1-D4 s2b_{ucb1,eps_greedy,prob_matching,hgucb_norm} vs s2a_U_best_LO
          D5-D8 s2b_{...} vs gvns_classico
  Omnibus: Friedman nas 6 celulas do 2A e nas 6 politicas do 2B, por B.
`gvns_classico` e `uniforme` vem de results/aos_study/full.csv (Etapa 1,
mesmas instancias/seeds/solucao inicial; reproducao exata verificada no
piloto da Etapa 2, Entrada 72).

dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por
instancia (media das seeds), IC95% bootstrap (10.000, semente 0).
Se faltar orcamento/braco completo, a analise e marcada PARCIAL (Holm
continua com m=28, conservador).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
M_FAMILIA = 28

CONTRASTES = [
    ("C1", "s2a_U_best_LO", "s2a_U_best_F40"),
    ("C2", "s2a_U_shaken_LO", "uniforme"),
    ("C3", "s2a_U_best_LO", "s2a_U_shaken_LO"),
    ("C4", "s2a_U_best_F40", "uniforme"),
    ("C5", "gvns_classico", "s2a_U_best_LO"),
    ("C6", "s2a_FO_shaken_LO", "s2a_U_shaken_LO"),
] + [(f"D{i+1}", f"s2b_{x}", "s2a_U_best_LO")
     for i, x in enumerate(["ucb1", "eps_greedy", "prob_matching", "hgucb_norm"])] \
  + [(f"D{i+5}", f"s2b_{x}", "gvns_classico")
     for i, x in enumerate(["ucb1", "eps_greedy", "prob_matching", "hgucb_norm"])]
CELULAS_2A = ["gvns_classico", "uniforme", "s2a_FO_shaken_LO", "s2a_U_best_LO",
              "s2a_U_best_F40", "s2a_U_shaken_LO"]
POLITICAS_2B = ["gvns_classico", "s2a_U_best_LO", "s2b_ucb1", "s2b_eps_greedy",
                "s2b_prob_matching", "s2b_hgucb_norm"]


def holm(pvals, m=M_FAMILIA):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def carregar():
    e2 = pd.read_csv(RES / "etapa2_full.csv").rename(columns={"braco": "arm"})
    e1 = pd.read_csv(RES / "full.csv").rename(columns={"politica": "arm"})
    e1 = e1[e1.arm.isin(["gvns_classico", "uniforme"]) & e1.B.isin([800, 4000])]
    cols = ["arm", "B", "conjunto", "instancia", "n_customers", "seed", "gap_pct", "viavel"]
    d = pd.concat([e2[cols + ["shakes", "tempo_s", "uso_operadores"]], e1[cols + ["tempo_s", "uso_operadores"]]],
                  ignore_index=True)
    return d


def completo(d, B, arms):
    sub = d[(d.B == B) & d.arm.isin(arms) & d.viavel]
    g = sub.groupby(["arm", "instancia"]).size().unstack(0)
    return g.shape[0] == 74 and set(arms) <= set(g.columns) and (g[arms].min().min() == 10)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    d = carregar()
    print(f"Execucoes lidas: {len(d)}  inviaveis: {int((~d.viavel).sum())}")
    d = d[d.viavel]
    arms_necessarios = sorted({a for _, a, b in CONTRASTES} | {b for _, a, b in CONTRASTES})
    completos = [B for B in (800, 4000) if completo(d, B, arms_necessarios)]
    parcial = len(completos) < 2
    print(f"Orcamentos completos: {completos}{'  -> ANALISE PARCIAL' if parcial else ''}\n")
    rng = np.random.default_rng(0)
    linhas = []
    for B in completos:
        t = d[d.B == B].groupby(["arm", "instancia"]).gap_pct.mean().unstack(0)
        print(f"=== B={B} === gap medio (74 instancias):")
        print(t[arms_necessarios].mean().round(3).to_string())
        for nome, cols in (("2A (6 celulas)", CELULAS_2A), ("2B (6 politicas)", POLITICAS_2B)):
            fr = stats.friedmanchisquare(*[t[c].values for c in cols])
            print(f"Friedman {nome}: chi2={fr.statistic:.2f} p={fr.pvalue:.3g}")
        for cid, a, b in CONTRASTES:
            x, y = t[a].values, t[b].values
            dif = x - y
            dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)   # empates: diferencas de ponto flutuante (~1e-15) nao sao vitorias/derrotas
            idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
            lo, hi = np.percentile(dif[idx].mean(axis=1), [2.5, 97.5])
            linhas.append(dict(B=B, id=cid, a=a, b=b, dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                               vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()),
                               emp=int((dif == 0).sum()), p_bruto=stats.wilcoxon(dif).pvalue))
        print()
    if linhas:
        r = pd.DataFrame(linhas)
        r["p_holm"] = holm(r.p_bruto.values)
        r["signif_005"] = r.p_holm < 0.05
        pd.set_option("display.width", 220)
        print(f"=== CONFIRMATORIO {'(PARCIAL)' if parcial else '(familia completa de 28)'}  dif = gap(a) - gap(b) ===")
        print(r.round(4).to_string(index=False))
        r.to_csv(RES / "analise_etapa2_confirmatoria.csv", index=False)

    print("\n######## EXPLORATORIO (nao confirmatorio) ########")
    for B in sorted(d.B.unique()):
        sub = d[d.B == B]
        print(f"\n-- B={B}: gap medio por conjunto --")
        print(sub.groupby(["arm", "conjunto", "instancia"]).gap_pct.mean().groupby(["arm", "conjunto"]).mean()
              .unstack().round(3).to_string())
    print("\n-- shakes medios e tempo de parede por aplicacao (ms) por braco --")
    for (arm, B), g in d.groupby(["arm", "B"]):
        sh = g.shakes.mean() if "shakes" in g and g.shakes.notna().any() else float("nan")
        print(f"  {arm:20s} B={B:5d}: shakes={sh:7.1f}  tempo/aplic={1000*(g.tempo_s/B).mean():6.2f} ms")
    print("\n-- uso de operadores (%): 2-Opt|Move(1)|Swap(1,1)|Reloc(1,0)|Reloc(2,0) --")
    for (arm, B), g in d.groupby(["arm", "B"]):
        uso = np.array([[int(v) for v in s.split("|")] for s in g.uso_operadores])
        pct = 100 * uso.sum(axis=0) / uso.sum()
        print(f"  {arm:20s} B={B:5d}: " + " ".join(f"{p:5.1f}" for p in pct))
    hgs = pd.read_csv(AQUI.parent / "results" / "hgs_74_n01.csv")
    print(f"\nTeto HGS (n01, 0,1 s/cliente): gap medio {hgs[hgs.viavel].groupby('instancia').gap_pct.mean().mean():.3f}%")


if __name__ == "__main__":
    main()
