#!/usr/bin/env python3
"""
verifica_tsplib_lkh.py
======================
Controle do Estudo 13: o leitor de instancias (src/tsp_inst.py), a regra de distancia (EUC_2D = nint) e o LKH-3 estao
corretos se o LKH-3 reproduz o OTIMO PUBLICADO (TSPLIB, TSP-BEST.html; instances/TSPLIB/optimos_tsplib.csv) nas 32 instancias
baixadas, com o custo recalculado pela NOSSA funcao de custo. Saida: results/aos_study/etapa13_tsplib_lkh_check.csv.
Uso (WSL, venv): python verifica_tsplib_lkh.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tsp_inst import ler_tsplib  # noqa: E402
from solver_lkh_tsp import resolver_tsp_lkh  # noqa: E402

DIR = ROOT / "instances" / "TSPLIB"
RES = ROOT / "results" / "aos_study"


def main():
    linhas = list(csv.DictReader(open(DIR / "optimos_tsplib.csv")))
    out = []
    for r in linhas:
        inst = ler_tsplib(DIR / f"{r['instancia']}.tsp", ref=float(r["otimo"]))
        assert inst.N == int(r["n"]), (r["instancia"], inst.N, r["n"])
        tour, custo, t = resolver_tsp_lkh(inst, runs=2, tempo_limite=0.5 * inst.N, seed=1)
        out.append(dict(instancia=r["instancia"], n=inst.N, otimo=r["otimo"], custo_lkh=int(custo),
                        gap_pct=100 * (custo - float(r["otimo"])) / float(r["otimo"]), tempo_s=round(t, 1)))
        print(out[-1], flush=True)
    with open(RES / "etapa13_tsplib_lkh_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    iguais = sum(1 for o in out if abs(o["gap_pct"]) < 1e-9)
    abaixo = [o["instancia"] for o in out if o["gap_pct"] < -1e-9]
    print(f"LKH-3 reproduz o otimo publicado em {iguais} de {len(out)}; abaixo do otimo (impossivel): {abaixo}")


if __name__ == "__main__":
    main()
