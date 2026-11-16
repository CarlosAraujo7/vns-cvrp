#!/usr/bin/env python3
"""Olhada DESCRITIVA nos dados de avaliacao de P21 do Estudo 23 (antes de P7, SRP e S129 existirem); NAO e a analise pre-registrada (analisar_etapa23.py roda quando tudo termina)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa23 as a23  # noqa: E402

rng = np.random.default_rng(0)
ref = a23.referencias()
d = pd.concat([ref, a23.ler_novo("p21", a23.NOVOS_P21)], ignore_index=True)
t = pivo(d[d.braco.str.endswith("@P21")])
print(len(t), "instancias; gap medio (%) em P21")
print(t.mean().round(3).sort_values().to_string())
for cid, a, b in (("RPh-U", "RPh@P21", "U@P21"), ("RPh-S65", "RPh@P21", "S65@P21"), ("RPh-S33", "RPh@P21", "S33@P21"), ("RPh-RP", "RPh@P21", "RP@P21"),
                  ("RPsr-RP", "RPsr@P21", "RP@P21"), ("RPsr-U", "RPsr@P21", "U@P21"), ("Upr-RP", "Upr@P21", "RP@P21"), ("Upr-U", "Upr@P21", "U@P21"),
                  ("Uq-RP", "Uq@P21", "RP@P21"), ("Uq-U", "Uq@P21", "U@P21"), ("RP-U", "RP@P21", "U@P21"), ("RPh-Upr", "RPh@P21", "Upr@P21")):
    r = contraste(t, cid, a, b, rng, "x")
    print(f"{cid:10s} dif {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] W/L {r['vit_a']}/{r['der_a']} p {r['p_bruto']:.4f}")
for a in ("RPh", "RPsr", "Upr", "Uq", "RP"):
    h = d[d.braco == f"{a}@P21"]
    m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
    print(f"{a:5s} iteracoes {h.iteracoes.mean():8.0f}  uso d {np.round(100 * m['uso_d'] / m['uso_d'].sum(), 1).tolist()}  r {np.round(100 * m['uso_r'] / m['uso_r'].sum(), 1).tolist()}")
