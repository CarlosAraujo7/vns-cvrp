#!/usr/bin/env python3
"""Calibracao do ALNS (Estudo 9; SYN_TUNE, politica U, 3 seeds): gap medio, tempo e iteracoes por configuracao."""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
d = pd.read_csv(RES / "etapa9_calib.csv")
print("execucoes:", len(d), "inviaveis:", int((~d.viavel).sum()))
d = d[d.viavel]
m = d.groupby(["config_id", "instancia"]).agg(gap=("gap_pct", "mean"), t=("tempo_s", "mean"), it=("iteracoes", "mean")).reset_index()
t = m.groupby("config_id").agg(gap=("gap", "mean"), gap_med=("gap", "median"), tempo_medio_s=("t", "mean"),
                               tempo_max_s=("t", "max"), iteracoes=("it", "mean"))
print(t.round(3).to_string())
# por tamanho
d["faixa"] = pd.cut(d.n_customers, [0, 75, 125, 1000], labels=["<=75", "76-125", ">125"])
print(d.groupby(["config_id", "faixa"], observed=True).gap_pct.mean().unstack().round(3).to_string())
