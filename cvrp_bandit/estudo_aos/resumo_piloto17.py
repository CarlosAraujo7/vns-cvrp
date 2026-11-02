#!/usr/bin/env python3
"""Resumo do piloto do Estudo 17 (SYN_TUNE): gap medio por braco e nivel de orcamento, iteracoes e uso dos conjuntos. NAO e resultado do estudo."""
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
d = pd.read_csv(RES / (sys.argv[1] if len(sys.argv) > 1 else "etapa17_piloto.csv"))
print(f"{len(d)} execucoes; inviaveis {int((~d.viavel).sum())}")
d = d[d.viavel]
m = d.groupby(["braco", "nivel"]).gap_pct.mean().unstack(1).round(3)
n = d.groupby(["braco", "nivel"]).gap_pct.count().unstack(1)
print("\n=== gap medio (%) por braco e nivel (segundos por cliente) ===")
print(m.to_string())
print("\n(execucoes por celula, min/max)", int(n.min().min()), int(n.max().max()))
it = d.groupby(["braco", "nivel"]).iteracoes.mean().unstack(1).round(0)
print("\n=== iteracoes medias ===")
print(it.to_string())
for nome in ("pil_U", "pil_RP", "pil_hgucb", "pil_ucb1", "pil_eff"):
    g = d[d.braco == nome]
    if len(g):
        u = pd.DataFrame(g.uso.str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
        print(f"uso medio dos conjuntos P1..P7 ({nome}):", (u / u.sum()).round(3).tolist())
d["classe_n"] = pd.cut(d.n_customers, [0, 70, 120, 1000], labels=["n<=70", "n~100", "n~150"])
print("\n=== P6 (padrao do PyVRP): gap medio por tamanho e nivel ===")
print(d[d.braco == "pil_P6"].groupby(["classe_n", "nivel"], observed=True).gap_pct.mean().unstack(1).round(3).to_string())
