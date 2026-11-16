#!/usr/bin/env python3
"""EXPLORATORIO: a vantagem do RP sobre a uniforme em P21 (X e SYN_VAL) depende da folga da instancia (o gap da uniforme) e do tamanho? Correlacoes de Spearman por instancia."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analisar_etapa21 as a21  # noqa: E402
from analisar_etapa16 import pivo  # noqa: E402

d = pd.concat([a21.referencias(), a21.ler_novos("P21")], ignore_index=True)
t = pivo(d)
n = d[d.braco == "U@P21"].groupby("instancia").n_customers.first().reindex(t.index)
for p in ("P7", "P21"):
    dif = t[f"RP@{p}"] - t[f"U@{p}"]
    gu = t[f"U@{p}"]
    r1, p1 = stats.spearmanr(gu, dif)
    r2, p2 = stats.spearmanr(n, dif)
    print(f"{p}: Spearman(RP-U, gap de U) = {r1:+.2f} (p {p1:.4f}); Spearman(RP-U, n) = {r2:+.2f} (p {p2:.3f}); media RP-U {dif.mean():+.3f}")
    q = pd.qcut(gu, 3, labels=["gap baixo", "medio", "alto"])
    print(dif.groupby(q).agg(["mean", "count"]).round(3).to_string())
    print(gu.groupby(q).mean().round(3).to_string())
