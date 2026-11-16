#!/usr/bin/env python3
"""Olhada DESCRITIVA no braco de P7 do Estudo 23 (RPh7: pontuacao fiel no conjunto de 7 operadores); NAO e a analise pre-registrada."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa23 as a23  # noqa: E402

rng = np.random.default_rng(0)
d = pd.concat([a23.a21.referencias().assign(iteracoes=np.nan), a23.ler_novo("p7", ("RPh7",))], ignore_index=True)
t = pivo(d[d.braco.str.endswith("@P7")])
print(len(t), "instancias; gap medio (%) em P7")
print(t.mean().round(3).sort_values().to_string())
for cid, a, b in (("RPh7-RP", "RPh7@P7", "RP@P7"), ("RPh7-U", "RPh7@P7", "U@P7"), ("RPh7-S33", "RPh7@P7", "S33@P7"), ("RPh7-S65", "RPh7@P7", "S65@P7"), ("RP-S33", "RP@P7", "S33@P7")):
    r = contraste(t, cid, a, b, rng, "x")
    print(f"{cid:10s} dif {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] W/L {r['vit_a']}/{r['der_a']} p {r['p_bruto']:.4f}")
h = d[d.braco == "RPh7@P7"]
r_ = d[d.braco == "RP@P7"]
for nome, x in (("RPh7", h), ("RP", r_)):
    m = {c: pd.DataFrame(x[c].str.split("|").apply(lambda v: [float(y) for y in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
    print(f"{nome:5s} uso d {np.round(100 * m['uso_d'] / m['uso_d'].sum(), 1).tolist()}  r {np.round(100 * m['uso_r'] / m['uso_r'].sum(), 1).tolist()}  iteracoes {x.iteracoes.mean():.0f}")
