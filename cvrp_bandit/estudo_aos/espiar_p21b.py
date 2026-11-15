#!/usr/bin/env python3
"""Olhada DESCRITIVA (planejamento): pesos das misturas de P21 e uso medio de operadores de RP e HG na avaliacao de P21."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import alns_ext as ax  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
esc = json.loads((RES / "etapa21_ajuste_escolhidas_P21.json").read_text())
nd = [n for n, _ in ax.DESTRUIR_P21]
nr = [n for n, _ in ax.REPARAR_P21]
for k in ("S33", "S65"):
    print(f"{k}: destruir", {n: round(100 * w, 1) for n, w in zip(nd, esc[k]["pesos_d"])})
    print(f"{k}: reparar ", {n: round(100 * w, 1) for n, w in zip(nr, esc[k]["pesos_r"])})
e = pd.read_csv(RES / "etapa21_eval_P21.csv")
for a in ("RP", "RPt", "HG", "S33", "S65", "U"):
    h = e[e.braco == f"s21_P21_{a}"]
    m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
    ud, ur = 100 * m["uso_d"] / m["uso_d"].sum(), 100 * m["uso_r"] / m["uso_r"].sum()
    print(f"uso {a}: destruir", {n: round(x, 1) for n, x in zip(nd, ud)})
    print(f"uso {a}: reparar ", {n: round(x, 1) for n, x in zip(nr, ur)})
    print(f"   iteracoes medias {h.iteracoes.mean():.0f}; inviaveis {h.inviaveis.sum()}; tempo medio {h.tempo_s.mean():.0f} s")
