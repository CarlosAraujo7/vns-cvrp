#!/usr/bin/env python3
"""Confere que os conjuntos A, B e CMT carregam com solucao inicial fixada em disco e otimo conhecido (insumo do Estudo 22)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402

tot = 0
for c in ("A", "B", "CMT"):
    ns = []
    for n in r4.nomes(c):
        inst, cws = r9._instancia(c, n)
        assert inst.optimum and inst.optimum > 0, (c, n)
        ns.append((inst.n_customers, bool(inst.vehicles_exact), inst.n_vehicles_ref, round(inst.optimum, 1), n))
    print(c, len(ns), "clientes", min(x[0] for x in ns), "a", max(x[0] for x in ns), "| K exato:", sum(1 for x in ns if x[1]), "| exemplo", ns[0])
    tot += len(ns)
print("total", tot)
