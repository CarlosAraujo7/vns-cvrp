#!/usr/bin/env python3
"""Olhada DESCRITIVA no braco SRP (mistura estatica = uso medio do RP nas validacoes) de P21; NAO e a analise pre-registrada."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa23 as a23  # noqa: E402

rng = np.random.default_rng(0)
d = pd.concat([a23.referencias(), a23.ler_novo("p21", a23.NOVOS_P21)], ignore_index=True)
t = pivo(d[d.braco.str.endswith("@P21")])
print(len(t), "instancias; gap medio (%) em P21")
print(t.mean().round(3).sort_values().to_string())
for cid, a, b in (("SRP-RP", "SRP@P21", "RP@P21"), ("SRP-U", "SRP@P21", "U@P21"), ("SRP-S65", "SRP@P21", "S65@P21"), ("SRP-RPh", "SRP@P21", "RPh@P21"), ("SRP-Uq", "SRP@P21", "Uq@P21")):
    r = contraste(t, cid, a, b, rng, "x")
    print(f"{cid:10s} dif {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] W/L {r['vit_a']}/{r['der_a']} p {r['p_bruto']:.4f}")
h = d[d.braco == "SRP@P21"]
print("SRP iteracoes", round(h.iteracoes.mean()))
