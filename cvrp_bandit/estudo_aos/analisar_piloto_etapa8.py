#!/usr/bin/env python3
"""Analise do PILOTO do Estudo 8 (SYN_TUNE; exploratorio, nao entra em inferencia): gap medio por pool x politica,
diferenca para a uniforme DENTRO do pool (negativo = melhor que U) e perda de cada pool inflado em relacao ao P10 limpo."""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
d = pd.read_csv(RES / "etapa8_pilot.csv")
d = d[d.viavel]
d["pool"] = d.braco.str.extract(r"s8_(P10[^_]*)_")[0]
d["pol"] = d.braco.str.extract(r"s8_P10[^_]*_(.*)$")[0]
m = d.groupby(["pool", "pol", "instancia"]).gap_pct.mean().reset_index()
print("execucoes:", len(d), "| instancias:", m.instancia.nunique())
t = m.groupby(["pool", "pol"]).gap_pct.mean().unstack("pol")
ordem = [c for c in ["U", "ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff"] if c in t.columns]
print("\nGap medio (%) por pool x politica (media das medias por instancia):")
print(t[ordem].round(3).to_string())
print("\nDiferenca para a uniforme do MESMO pool (pp; negativo = melhor que U) [Wilcoxon p entre parenteses]:")
linhas = {}
for pool in t.index:
    u = m[(m.pool == pool) & (m.pol == "U")].set_index("instancia").gap_pct
    linha = {}
    for pol in ordem[1:]:
        a = m[(m.pool == pool) & (m.pol == pol)].set_index("instancia").gap_pct.reindex(u.index)
        dif = (a - u).dropna()
        p = stats.wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
        linha[pol] = f"{dif.mean():+.3f} ({p:.3f})"
    linhas[pool] = linha
print(pd.DataFrame(linhas).T.to_string())
print("\nPerda do pool inflado em relacao ao P10 limpo com a MESMA politica (pp; positivo = pior):")
for pol in ordem:
    base = m[(m.pool == "P10") & (m.pol == pol)].set_index("instancia").gap_pct
    out = []
    for pool in t.index:
        if pool == "P10":
            continue
        x = m[(m.pool == pool) & (m.pol == pol)].set_index("instancia").gap_pct.reindex(base.index)
        out.append(f"{pool}: {(x - base).mean():+.3f}")
    print(f"  {pol:7s} " + "  ".join(out))
# recuperacao: (perda de U - perda da politica) / perda de U
print("\nRecuperacao = 1 - perda(politica)/perda(U), por pool inflado:")
base_u = m[(m.pool == "P10") & (m.pol == "U")].set_index("instancia").gap_pct
for pool in [p for p in t.index if p != "P10"]:
    perda_u = (m[(m.pool == pool) & (m.pol == "U")].set_index("instancia").gap_pct.reindex(base_u.index) - base_u).mean()
    rec = []
    for pol in ordem[1:]:
        b = m[(m.pool == "P10") & (m.pol == pol)].set_index("instancia").gap_pct
        x = m[(m.pool == pool) & (m.pol == pol)].set_index("instancia").gap_pct.reindex(b.index)
        perda = (x - b).mean()
        rec.append(f"{pol}={1 - perda / perda_u:+.2f}" if abs(perda_u) > 1e-9 else f"{pol}=nan")
    print(f"  {pool:8s} perda(U)={perda_u:+.3f}pp  " + "  ".join(rec))
