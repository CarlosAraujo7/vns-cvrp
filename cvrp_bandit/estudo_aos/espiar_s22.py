#!/usr/bin/env python3
"""Olhada DESCRITIVA no regime cls (A, B e CMT, 1.500 n^2, recozimento) do Estudo 22 antes de os demais regimes terminarem; NAO e a analise pre-registrada."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa22 as a22  # noqa: E402

rng = np.random.default_rng(0)
d = a22.completar(a22.ler("cls"))
t = pivo(d)
t["S*@cls"] = t["S129@cls"]
print(len(t), "instancias; gap medio (%) em cls")
print(t.mean().round(3).sort_values().to_string())
for cid, a, b in (("RP-U", "RP@cls", "U@cls"), ("RP-S*", "RP@cls", "S*@cls"), ("RP-S33", "RP@cls", "S33@cls"), ("HG-U", "HG@cls", "U@cls"), ("S*-U", "S*@cls", "U@cls"),
                  ("S33-U", "S33@cls", "U@cls"), ("RPt-RP", "RPt@cls", "RP@cls")):
    r = contraste(t, cid, a, b, rng, "x")
    print(f"{cid:8s} dif {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] W/L {r['vit_a']}/{r['der_a']} p {r['p_bruto']:.4f} mde {r['mde_pp']:.2f}")
conj = d[d.braco == "U@cls"].groupby("instancia").conjunto.first().reindex(t.index)
for c in ("A", "B", "CMT"):
    sub = t[(conj == c).values]
    r = contraste(sub, f"RP-U@{c}", "RP@cls", "U@cls", rng, "x")
    print(f"{c:4s} n {len(sub)}: U {sub['U@cls'].mean():.3f} RP {sub['RP@cls'].mean():.3f}  RP-U {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}]")
