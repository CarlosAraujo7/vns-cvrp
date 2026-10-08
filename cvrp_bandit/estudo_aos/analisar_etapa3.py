#!/usr/bin/env python3
"""
analisar_etapa3.py
==================
Analise da Etapa 3 da Direcao A conforme o PRE-REGISTRO (diario, Entrada
77). Implementa SOMENTE o que la esta fixado como confirmatorio; o resto e
rotulado EXPLORATORIO.

Confirmatorio, por nivel de orcamento (E1 = 600 n^2, E2 = 5 E1) -- 17 testes
por nivel, familia de 34, Holm m=34:
  T1 s3_LO_U   vs s3_P5_LO_U   (efeito do conjunto estendido)
  T2 s3_R5_U   vs s3_LO_U      (efeito do termino)
  T3 s3_P5_LO_FO vs s3_P5_LO_U (ordem fixa vs aleatoria, P5, orcamento em dist.)
  para t in {LO, R5}: s3_t_FOC vs s3_t_U;
                      s3_t_{ucb1c,pmc,hgucbc} vs s3_t_U;
                      s3_t_{ucb1c,pmc,hgucbc} vs s3_t_FOC
  Omnibus: Friedman nos 10 bracos principais (P10), por nivel.

dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por
instancia (media das seeds), IC95% bootstrap (10.000, semente 0). Se faltar
nivel/braco completo a analise e marcada PARCIAL (Holm continua com m=34).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
M_FAMILIA = 34
IA = ["ucb1c", "pmc", "hgucbc"]
CONTRASTES = [("T1", "s3_LO_U", "s3_P5_LO_U"), ("T2", "s3_R5_U", "s3_LO_U"),
              ("T3", "s3_P5_LO_FO", "s3_P5_LO_U")]
for t in ("LO", "R5"):
    CONTRASTES.append((f"{t}_FOCvsU", f"s3_{t}_FOC", f"s3_{t}_U"))
    CONTRASTES += [(f"{t}_{x}vsU", f"s3_{t}_{x}", f"s3_{t}_U") for x in IA]
    CONTRASTES += [(f"{t}_{x}vsFOC", f"s3_{t}_{x}", f"s3_{t}_FOC") for x in IA]
PRINCIPAIS = [f"s3_{t}_{p}" for t in ("LO", "R5") for p in ["FOC", "U"] + IA]
NECESSARIOS = sorted({a for _, a, b in CONTRASTES} | {b for _, a, b in CONTRASTES})


def holm(pvals, m=M_FAMILIA):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    d = pd.read_csv(RES / "etapa3_full.csv")
    print(f"Execucoes lidas: {len(d)}  inviaveis: {int((~d.viavel).sum())}")
    d = d[d.viavel]
    assert len(CONTRASTES) == 17, len(CONTRASTES)

    def completo(nv, arms):
        sub = d[(d.nivel == nv) & d.braco.isin(arms)]
        g = sub.groupby(["braco", "instancia"]).size().unstack(0)
        return g.shape[0] == 74 and set(arms) <= set(g.columns) and g[arms].min().min() == 10

    niveis = [nv for nv in (1, 5) if completo(nv, NECESSARIOS)]
    parcial = len(niveis) < 2
    print(f"Niveis completos: {niveis}{'  -> ANALISE PARCIAL' if parcial else ''}\n")
    rng = np.random.default_rng(0)
    linhas = []
    for nv in niveis:
        t = d[d.nivel == nv].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
        print(f"=== E{nv} === gap medio (74 instancias):")
        print(t.mean().round(3).sort_values().to_string())
        fr = stats.friedmanchisquare(*[t[c].values for c in PRINCIPAIS])
        print(f"Friedman (10 bracos principais): chi2={fr.statistic:.2f} p={fr.pvalue:.3g}\n")
        for cid, a, b in CONTRASTES:
            x, y = t[a].values, t[b].values
            dif = x - y
            idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
            lo, hi = np.percentile(dif[idx].mean(axis=1), [2.5, 97.5])
            linhas.append(dict(E=nv, id=cid, a=a, b=b, dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                               vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()),
                               emp=int((dif == 0).sum()), p_bruto=stats.wilcoxon(x, y).pvalue))
    if linhas:
        r = pd.DataFrame(linhas)
        r["p_holm"] = holm(r.p_bruto.values)
        r["signif_005"] = r.p_holm < 0.05
        pd.set_option("display.width", 230)
        print(f"=== CONFIRMATORIO {'(PARCIAL)' if parcial else '(familia completa de 34)'}  dif = gap(a) - gap(b) ===")
        print(r.round(4).to_string(index=False))
        r.to_csv(RES / "analise_etapa3_confirmatoria.csv", index=False)

    print("\n######## EXPLORATORIO (nao confirmatorio) ########")
    for nv in sorted(d.nivel.unique()):
        sub = d[d.nivel == nv]
        print(f"\n-- E{nv}: gap medio por conjunto --")
        print(sub.groupby(["braco", "conjunto", "instancia"]).gap_pct.mean().groupby(["braco", "conjunto"])
              .mean().unstack().round(3).to_string())
        if {"s3_R5_hgucbc", "s3_R5_hgucb_cego"} <= set(sub.braco):
            t = sub.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
            x, y = t["s3_R5_hgucbc"], t["s3_R5_hgucb_cego"]
            print(f"  hgucbc (sensivel a custo) - hgucb_cego: dif {np.mean(x-y):+.3f}pp  p={stats.wilcoxon(x, y).pvalue:.3g}")
    print("\n-- % das avaliacoes de distancia por operador (P10: 2-Opt|Move|Swap11|Reloc10|Reloc20|2opt*|Oropt|SwapIntra|CROSS|SWAP*) --")
    for (br, nv), g in d.groupby(["braco", "nivel"]):
        arr = np.array([[int(v) for v in s.split("|")] for s in g.calls_por_operador])
        pct = 100 * arr.sum(0) / arr.sum()
        print(f"  {br:18s} E{nv}: " + " ".join(f"{p:5.1f}" for p in pct))
    print("\n-- shakes medios, estouro relativo medio do orcamento (%) e tempo medio (s) --")
    for (br, nv), g in d.groupby(["braco", "nivel"]):
        est = 100 * ((g.calls_usadas - g.orcamento) / g.orcamento).mean()
        print(f"  {br:18s} E{nv}: shakes={g.shakes.mean():7.1f}  estouro={est:5.2f}%  tempo={g.tempo_s.mean():6.1f}s")
    hgs = pd.read_csv(AQUI.parent / "results" / "hgs_74_n01.csv")
    print(f"\nTeto HGS (n01): gap medio {hgs[hgs.viavel].groupby('instancia').gap_pct.mean().mean():.3f}%")


if __name__ == "__main__":
    main()
