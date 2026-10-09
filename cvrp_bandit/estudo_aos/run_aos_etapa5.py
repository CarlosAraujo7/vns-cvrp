#!/usr/bin/env python3
"""
run_aos_etapa5.py
=================
Runner da Etapa 5 (atribuicao do efeito do conjunto de operadores). PROTOCOLO
PRE-REGISTRADO em diario_pesquisa.md, Entrada 89 -- leia la antes de mexer.

Politica uniforme, aceitacao "best", shake dos 5 operadores originais,
orcamento E1 = 600 n^2 e E2 = 5 E1 em avaliacoes de distancia, 74 instancias,
seeds 1000-1009, solucao inicial FIXADA em disco (Entrada 88).
Subconjuntos do pool de 10 (indices: 0 2-Opt, 1 Move(1), 2 Swap(1,1),
3 Reloc(1,0), 4 Reloc(2,0), 5 2-opt*, 6 Or-opt, 7 Swap-intra, 8 CROSS, 9 SWAP*):
  P5 = [0-4]   P8c = [0-7]   P7e = [0-4, 8, 9]   P7c = [0, 1, 3, 4, 5, 6, 7]

Uso (WSL, venv de direcao1_local):
    python3 -u run_aos_etapa5.py --pilot
    python3 -u run_aos_etapa5.py --full --workers 11
Resumivel por chave (braco, config, nivel, instancia, seed).
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AQUI))
import run_aos_etapa4 as r4  # noqa: E402
from run_aos_study import todas_instancias  # noqa: E402

RES = ROOT / "results" / "aos_study"
POOLS = {"P5": [0, 1, 2, 3, 4], "P8c": [0, 1, 2, 3, 4, 5, 6, 7], "P7e": [0, 1, 2, 3, 4, 8, 9],
         "P7c": [0, 1, 3, 4, 5, 6, 7]}
ARMS = {"s5_LO_P8c": ("P8c", "LO"), "s5_LO_P7e": ("P7e", "LO"), "s5_LO_P7c": ("P7c", "LO"),
        "s5_R5_P5": ("P5", "R"), "s5_R5_P8c": ("P8c", "R"), "s5_R5_P7e": ("P7e", "R"),
        "s5_R5_P7c": ("P7c", "R")}


def specs():
    return {n: r4.spec(n, "U", termino=t, custo=False, params=None, config_id="e5", pool=POOLS[p])
            for n, (p, t) in ARMS.items()}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    sp = specs()
    insts = todas_instancias()
    if a.pilot:
        esc = {"A-n32-k5", "B-n41-k6", "CMT1", "X-n101-k25"}
        ctrl = r4.spec("ctrl_P10_idx", "U", termino="R", custo=False, params=None, config_id="e5",
                       pool=list(range(10)))    # controle: deve reproduzir s3_R5_U exatamente
        tarefas = [(s, 1, c, n, 1000 + sd) for (c, n, _) in insts if n in esc
                   for s in list(sp.values()) + [ctrl] for sd in range(2)]
        out = RES / "etapa5_pilot.csv"
    else:
        tarefas = [(s, nv, c, n, 1000 + sd) for nv in (1, 5)
                   for (c, n, nc) in sorted(insts, key=lambda t: -t[2]) for s in sp.values() for sd in range(10)]
        out = RES / "etapa5_full.csv"
    print(f"{len(tarefas)} tarefas", flush=True)
    with mp.Pool(a.workers) as pool:
        r4.rodar(tarefas, out, pool)
    print("concluido", flush=True)


if __name__ == "__main__":
    main()
