#!/usr/bin/env python3
"""Olhada DESCRITIVA nos regimes cls e short (A, B e CMT) do Estudo 22; NAO e a analise pre-registrada."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa22 as a22  # noqa: E402

rng = np.random.default_rng(0)
for reg in sys.argv[1:] or ["cls", "short"]:
    d = a22.completar(a22.ler(reg))
    t = pivo(d)
    t["S*@" + reg] = t["S129@" + reg]
    print(f"=== {reg}: {len(t)} instancias; gap medio (%)")
    print("  " + "  ".join(f"{c.split('@')[0]} {t[c].mean():.3f}" for c in t.columns if "*" not in c))
    for cid, a, b in (("RP-U", "RP", "U"), ("RP-S*", "RP", "S*"), ("S*-U", "S*", "U")):
        r = contraste(t, cid, f"{a}@{reg}", f"{b}@{reg}", rng, "x")
        print(f"  {cid:6s} dif {r['dif_pp']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] W/L {r['vit_a']}/{r['der_a']} p {r['p_bruto']:.4f} mde {r['mde_pp']:.2f}")
