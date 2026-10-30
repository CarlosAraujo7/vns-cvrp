"""Resumo do piloto do Estudo 16 (SYN_TUNE_W): gap medio por braco e configuracao, uso do reparo e iteracoes."""
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
f = sys.argv[1] if len(sys.argv) > 1 else "etapa16_piloto_a.csv"
d = pd.read_csv(RES / f)
print("execucoes:", len(d), "| inviaveis (solucao final):", int((~d.viavel).sum()))
g = d[d.viavel].groupby(["config_id", "braco"]).agg(gap=("gap_pct", "mean"), it=("iteracoes", "mean"),
                                                      inv=("inviaveis", "mean"), t=("tempo_s", "mean")).reset_index()
print(g.sort_values(["config_id", "gap"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
r = d[d.braco.str.startswith("pa_U_k") & ~d.braco.str.contains("_w")].copy()
r["uso_r"] = r.uso_r.str.split("|").apply(lambda v: [float(x) for x in v])
for k, h in r.groupby("config_id"):
    m = pd.DataFrame(h.uso_r.tolist()).mean()
    print(k, "uso dos reparos sob U (guloso, regret2, barato):", [round(x, 1) for x in m.tolist()])
rp = d[d.braco.str.startswith("pa_RP_k") & ~d.braco.str.contains("_w")].copy()
rp["uso_r"] = rp.uso_r.str.split("|").apply(lambda v: [float(x) for x in v])
for k, h in rp.groupby("config_id"):
    m = pd.DataFrame(h.uso_r.tolist()).mean()
    print(k, "uso dos reparos sob RP:", [round(x / m.sum(), 3) for x in m.tolist()])
