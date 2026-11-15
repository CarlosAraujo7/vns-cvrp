#!/usr/bin/env python3
"""Olhada DESCRITIVA nos dados de avaliacao de P21 (antes de P12 existir), so para planejar a continuacao; NAO e a analise pre-registrada (analisar_etapa21.py roda quando P12 e a emenda terminam)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

rng = np.random.default_rng(0)
d = pd.concat([a21.referencias(), a21.ler_novos("P21")], ignore_index=True)
t = pivo(d)
print(len(t), "instancias; gap medio (%) por braco")
print(t.mean().round(3).sort_values().to_string())
t = t.copy()
t["DiD@P21"] = (t["RP@P21"] - t["S33@P21"]) - (t["RP@P7"] - t["S33@P7"])
t["zero"] = 0.0
for cid, a, b in (("H120 RP-S33@P21", "RP@P21", "S33@P21"), ("H121 DiD", "DiD@P21", "zero"), ("H122 RPt-S33@P21", "RPt@P21", "S33@P21"),
                  ("H123 S65-S33@P21", "S65@P21", "S33@P21"), ("H124 RP-U@P21", "RP@P21", "U@P21"), ("H125 S33-U@P21", "S33@P21", "U@P21"),
                  ("H126 HG-U@P21", "HG@P21", "U@P21"), ("RP-S65@P21", "RP@P21", "S65@P21"), ("S65-U@P21", "S65@P21", "U@P21"),
                  ("RPt-RP@P21", "RPt@P21", "RP@P21")):
    r = contraste(t, cid, a, b, rng, "x")
    print(f"{cid:20s} dif {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] W/L {r['vit_a']}/{r['der_a']} p {r['p_bruto']:.4f} mde {r['mde_pp']:.2f}")
