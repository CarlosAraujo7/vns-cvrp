#!/usr/bin/env python3
"""
refina_refs_tsp.py
==================
Onde as duas chamadas do LKH-3 de gerar_instancias_tsp.py DISCORDAM (a referencia pode nao ser o otimo), roda tres chamadas
adicionais mais longas (RUNS=5, MAX_TRIALS=3N, sementes 13-15, tempo 0,6 s por no) e fixa ref = minimo de todas. Acrescenta as
colunas `ref_refinada` (bool) e `lkh_extra` ao ref.csv. Feito ANTES de qualquer execucao dos bracos (diario, Entrada 113).
Uso (WSL, venv): python refina_refs_tsp.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tsp_inst import ler_tsplib  # noqa: E402
from solver_lkh_tsp import resolver_tsp_lkh  # noqa: E402


def main():
    for conj in ("SYN_TUNE_T", "SYN_VAL_T"):
        arq = ROOT / "instances" / conj / "ref.csv"
        linhas = list(csv.DictReader(open(arq)))
        for l in linhas:
            if "ref_refinada" in l and l["ref_refinada"] in ("True", "False"):
                continue
            l["ref_refinada"], l["lkh_extra"] = "False", ""
            if l["lkh1"] != l["lkh2"]:
                inst = ler_tsplib(ROOT / "instances" / conj / f"{l['instancia']}.tsp")
                extra = [int(resolver_tsp_lkh(inst, runs=5, tempo_limite=0.6 * inst.N, seed=s, max_trials=3 * inst.N)[1])
                         for s in (13, 14, 15)]
                l["ref"] = str(min(int(l["ref"]), *extra))
                l["ref_refinada"], l["lkh_extra"] = "True", "|".join(map(str, extra))
                print(l, flush=True)
        with open(arq, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
    print("referencias refinadas")


if __name__ == "__main__":
    main()
