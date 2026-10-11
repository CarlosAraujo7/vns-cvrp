#!/usr/bin/env python3
"""Analise do PILOTO do ALNS (Estudo 9; SYN_TUNE; exploratorio): gap medio por politica, diferenca para U, vitorias,
uso dos operadores de destruicao/reparo e escore acumulado."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
arq = sys.argv[1] if len(sys.argv) > 1 else "etapa9_pilot.csv"
d = pd.read_csv(RES / arq)
print("execucoes:", len(d), "inviaveis:", int((~d.viavel).sum()))
d = d[d.viavel]
m = d.groupby(["braco", "instancia"]).gap_pct.mean().reset_index()
u_nome = [b for b in m.braco.unique() if b.endswith("_U")][0]
u = m[m.braco == u_nome].set_index("instancia").gap_pct
print(f"\nreferencia: {u_nome}  gap medio {u.mean():.3f}  (n instancias {len(u)})")
linhas = []
for b in sorted(m.braco.unique()):
    x = m[m.braco == b].set_index("instancia").gap_pct.reindex(u.index)
    dif = (x - u).dropna()
    p = stats.wilcoxon(dif).pvalue if (dif != 0).any() and len(dif) > 5 else float("nan")
    linhas.append((b, x.mean(), dif.mean(), int((dif < 0).sum()), int((dif > 0).sum()), p))
print(pd.DataFrame(linhas, columns=["braco", "gap_medio", "dif_vs_U_pp", "vit", "der", "p_wilcoxon"]).round(3).to_string(index=False))
# desvio entre seeds
sd = d.groupby(["braco", "instancia"]).gap_pct.std().groupby("braco").mean()
print("\ndesvio-padrao medio entre seeds:\n" + sd.round(3).to_string())
# por tamanho
d["faixa"] = pd.cut(d.n_customers, [0, 75, 125, 1000], labels=["<=75", "76-125", ">125"])
print("\ngap por faixa de n:\n" + d.groupby(["braco", "faixa"], observed=True).gap_pct.mean().unstack().round(3).to_string())
# uso e escore
nomes_d = ["aleat", "pior", "relac", "rota"]
nomes_r = ["guloso", "regret2", "barato"]
print("\nuso medio (%) dos operadores de destruicao | reparo, e custo medio por uso (k aval.)")
for b in sorted(d.braco.unique()):
    g = d[d.braco == b]
    ud = np.mean([np.array(s.split("|"), float) / np.array(s.split("|"), float).sum() for s in g.uso_d], axis=0) * 100
    ur = np.mean([np.array(s.split("|"), float) / np.array(s.split("|"), float).sum() for s in g.uso_r], axis=0) * 100
    print(f"  {b:12s} D: " + " ".join(f"{n}={v:4.1f}" for n, v in zip(nomes_d, ud)) +
          " | R: " + " ".join(f"{n}={v:4.1f}" for n, v in zip(nomes_r, ur)))
